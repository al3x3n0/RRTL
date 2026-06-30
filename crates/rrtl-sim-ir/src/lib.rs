use std::collections::{HashMap, HashSet};

use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

use rrtl_core::{
    CompiledDesign, CompiledExpr, CompiledMemoryWrite, CompiledModule, CompiledRegister,
    PortDirection,
};
use rrtl_ir::{
    BitType, Diagnostic, ErrorReport, Expr, Reset, ResetKind, ResetPolarity, Signal, SignalKind,
    Signedness, Width,
};
use serde::{Deserialize, Serialize};

pub mod bitparallel;
pub mod bitslice;
pub mod policy;
pub mod specialize;
pub mod tabulate;
pub use policy::{
    apply_machine_optimization_policy, apply_packed_optimization_policy, MachineOptimizationResult,
    MemoryInitializer, OptimizationBackend, OptimizationGoal, OptimizationPass, OptimizationPolicy,
    OptimizationPolicyRecommendation, OptimizationReport, OptimizationRequest, OptimizationStep,
    PackedOptimizationResult, PolicyOptimizationResult, RecommendedOptimizationResult,
    optimize_for_goal, optimize_with_policy, optimize_with_request, recommend_optimization_policy,
    recommend_optimization_policy_for_request,
};
pub use specialize::{
    freeze_signals_program, rebalance_mux_chains_program, slot_allocate_program,
    specialize_program, strength_reduce_program, FreezeStats, RebalanceStats, SlotStats,
    SpecializeStats, StrengthStats,
};

pub mod instance_fold;
pub use instance_fold::{analyze_instance_fold, FoldConnection, InstanceFold};

pub mod value_range;
pub use value_range::{block_maxbits, range_reduction_stats, RangeStats};

pub mod activity;
pub mod linearize;
pub mod leap;
pub mod gpu_codegen;
pub use activity::{register_support, RegisterSupport};

#[cfg(feature = "jit")]
pub mod jit;

#[cfg(feature = "aot")]
pub mod aot;

#[cfg(feature = "aot")]
pub mod linear_aot;

#[cfg(feature = "aot")]
pub mod hybrid;

/// Lane-packed simulation IR derived from a compiled RTL module.
///
/// This is the higher-level public packed form: operations still carry expression
/// trees, while signal and memory storage has already been flattened into
/// per-lane word layouts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedProgram {
    pub top: String,
    pub signals: Vec<PackedSignal>,
    pub memories: Vec<PackedMemory>,
    pub streams: PackedStreams,
    pub top_signal_indices: HashMap<Signal, usize>,
    pub total_signal_words: usize,
    pub total_memory_words_per_lane: usize,
    /// Maps a register's destination signal index to its clock's signal index,
    /// for engines that gate register capture by clock edge (multi-clock). Empty
    /// for designs whose registers carry no explicit clock; consulted only by the
    /// `*_clocked` tick variants, so single-clock paths are unaffected.
    pub reg_clocks: HashMap<usize, usize>,
    /// Maps a memory index to its write clock's signal index (one clock per
    /// memory), so the `*_clocked` ticks also gate memory writes by clock edge.
    pub mem_clocks: HashMap<usize, usize>,
}

impl PackedProgram {
    pub fn signal_index(&self, signal: Signal) -> Option<usize> {
        self.top_signal_indices.get(&signal).copied()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedStreams {
    pub async_reset_comb: Vec<PackedPacket>,
    pub comb: Vec<PackedPacket>,
    pub tick_next: Vec<PackedPacket>,
    pub tick_commit: Vec<PackedPacket>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedPacket {
    pub ops: Vec<PackedOp>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackedOp {
    Assign {
        dst: usize,
        expr: PackedExpr,
    },
    CaptureReg {
        dst: usize,
        next: PackedExpr,
        reset: Option<PackedReset>,
    },
    MemoryWrite {
        memory: usize,
        enable: PackedExpr,
        addr: PackedExpr,
        data: PackedExpr,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedSignal {
    pub name: String,
    pub source: Option<Signal>,
    /// Instance-path of the context that owns this signal (e.g. `Top.u_child`).
    /// Unlike `source` (which is only set for the top module and is keyed by
    /// module *definition*), this disambiguates signals across multiple
    /// instances of the same module and is the provenance key used to slice the
    /// program into partition groups.
    pub owner_path: String,
    pub layout: PackedValueLayout,
    pub kind: PackedSignalKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedSignalKind {
    Input,
    Output,
    Wire,
    Reg,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedMemory {
    pub name: String,
    pub source: Signal,
    /// Instance-path of the owning context (see [`PackedSignal::owner_path`]).
    pub owner_path: String,
    pub addr_width: Width,
    pub depth: usize,
    pub data_layout: PackedValueLayout,
    pub offset: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedValueLayout {
    pub width: Width,
    pub ty: BitType,
    pub offset: usize,
    pub limbs: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedReset {
    pub signal: usize,
    pub value: Vec<u32>,
    pub kind: ResetKind,
    pub polarity: ResetPolarity,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PackedExpr {
    pub kind: PackedExprKind,
    pub ty: BitType,
}

impl PackedExpr {
    pub fn width(&self) -> Width {
        self.ty.width
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PackedExprKind {
    Lit(Vec<u32>),
    Signal(usize),
    Not(Box<PackedExpr>),
    And(Box<PackedExpr>, Box<PackedExpr>),
    Or(Box<PackedExpr>, Box<PackedExpr>),
    Xor(Box<PackedExpr>, Box<PackedExpr>),
    Add(Box<PackedExpr>, Box<PackedExpr>),
    Sub(Box<PackedExpr>, Box<PackedExpr>),
    Mul(Box<PackedExpr>, Box<PackedExpr>),
    Eq(Box<PackedExpr>, Box<PackedExpr>),
    Ne(Box<PackedExpr>, Box<PackedExpr>),
    Lt {
        lhs: Box<PackedExpr>,
        rhs: Box<PackedExpr>,
        signed: bool,
    },
    Mux {
        cond: Box<PackedExpr>,
        then_expr: Box<PackedExpr>,
        else_expr: Box<PackedExpr>,
    },
    Slice {
        expr: Box<PackedExpr>,
        lsb: Width,
    },
    Zext(Box<PackedExpr>),
    Sext(Box<PackedExpr>),
    Trunc(Box<PackedExpr>),
    Cast(Box<PackedExpr>),
    Concat(Vec<PackedExpr>),
    MemRead {
        memory: usize,
        addr: Box<PackedExpr>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedValueId(pub usize);

/// Machine-oriented packed simulation program lowered from [`PackedProgram`].
///
/// The streams contain packeted instructions with reusable SSA-like values and
/// explicit side effects. `PackedSimulator` and the experimental GPU emitter use
/// this form as their execution representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedMachineProgram {
    pub source: PackedProgram,
    pub streams: PackedMachineStreams,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedMachineStreams {
    pub async_reset_comb: PackedBlock,
    pub comb: PackedBlock,
    pub tick_next: PackedBlock,
    pub tick_commit: PackedBlock,
}

/// A dependency-scheduled block of packed machine packets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PackedBlock {
    pub packets: Vec<PackedMachinePacket>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PackedMachinePacket {
    pub instrs: Vec<PackedInstr>,
    pub effects: Vec<PackedEffect>,
}

/// One machine instruction that produces a typed packed value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedInstr {
    pub dst: PackedValueId,
    pub ty: BitType,
    pub kind: PackedInstrKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PackedInstrKind {
    Lit(Vec<u32>),
    Signal(usize),
    Not(PackedValueId),
    And(PackedValueId, PackedValueId),
    Or(PackedValueId, PackedValueId),
    Xor(PackedValueId, PackedValueId),
    Add(PackedValueId, PackedValueId),
    Sub(PackedValueId, PackedValueId),
    Mul(PackedValueId, PackedValueId),
    Eq(PackedValueId, PackedValueId),
    Ne(PackedValueId, PackedValueId),
    Lt {
        lhs: PackedValueId,
        rhs: PackedValueId,
        signed: bool,
    },
    Mux {
        cond: PackedValueId,
        then_value: PackedValueId,
        else_value: PackedValueId,
    },
    Slice {
        value: PackedValueId,
        lsb: Width,
    },
    Zext(PackedValueId),
    Sext(PackedValueId),
    Trunc(PackedValueId),
    Cast(PackedValueId),
    Concat(Vec<PackedValueId>),
    MemRead {
        memory: usize,
        addr: PackedValueId,
    },
}

/// Side effects emitted after the instructions in a machine packet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackedEffect {
    StoreSignal {
        dst: usize,
        value: PackedValueId,
    },
    CaptureReg {
        dst: usize,
        value: PackedValueId,
        reset: Option<PackedReset>,
    },
    MemoryWrite {
        memory: usize,
        enable: PackedValueId,
        addr: PackedValueId,
        data: PackedValueId,
    },
}

/// Analysis summary for a packed machine program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedMachineAnalysis {
    pub async_reset_comb: PackedBlockAnalysis,
    pub comb: PackedBlockAnalysis,
    pub tick_next: PackedBlockAnalysis,
    pub tick_commit: PackedBlockAnalysis,
    pub instr_count: usize,
    pub effect_count: usize,
    pub max_packet_width: usize,
    pub max_live_values: usize,
    pub avg_live_values_x100: usize,
    pub max_packet_memory_reads: usize,
}

/// Analysis summary for one packed machine block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedBlockAnalysis {
    pub packets: Vec<PackedPacketAnalysis>,
    pub values: HashMap<PackedValueId, PackedValueAnalysis>,
    pub instr_count: usize,
    pub effect_count: usize,
    pub max_packet_width: usize,
    pub max_live_values: usize,
    pub avg_live_values_x100: usize,
    pub max_packet_memory_reads: usize,
}

/// Analysis summary for one VLIW packet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedPacketAnalysis {
    pub instr_count: usize,
    pub effect_count: usize,
    pub class_counts: PackedInstrClassCounts,
}

/// Definition and use information for one machine value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedValueAnalysis {
    pub ty: BitType,
    pub defined_packet: usize,
    pub uses: Vec<PackedValueUse>,
    pub last_use_packet: Option<usize>,
}

/// One recorded use of a machine value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedValueUse {
    pub packet: usize,
    pub kind: PackedValueUseKind,
}

/// Where a machine value is consumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedValueUseKind {
    InstructionInput,
    EffectInput,
}

/// Coarse instruction class for VLIW packet analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PackedInstrClass {
    Literal,
    SignalLoad,
    Bitwise,
    Arithmetic,
    Compare,
    Select,
    BitMovement,
    MemoryRead,
}

/// Per-packet instruction class counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PackedInstrClassCounts {
    pub literal: usize,
    pub signal_load: usize,
    pub bitwise: usize,
    pub arithmetic: usize,
    pub compare: usize,
    pub select: usize,
    pub bit_movement: usize,
    pub memory_read: usize,
}

impl PackedInstrClassCounts {
    fn increment(&mut self, class: PackedInstrClass) {
        match class {
            PackedInstrClass::Literal => self.literal += 1,
            PackedInstrClass::SignalLoad => self.signal_load += 1,
            PackedInstrClass::Bitwise => self.bitwise += 1,
            PackedInstrClass::Arithmetic => self.arithmetic += 1,
            PackedInstrClass::Compare => self.compare += 1,
            PackedInstrClass::Select => self.select += 1,
            PackedInstrClass::BitMovement => self.bit_movement += 1,
            PackedInstrClass::MemoryRead => self.memory_read += 1,
        }
    }
}

impl SimdFastPathProfile {
    fn record_one_limb_op(&mut self) {
        self.one_limb_ops += 1;
    }

    fn record_two_limb_op(&mut self) {
        self.two_limb_ops += 1;
    }

    fn record_two_limb_mul_op(&mut self) {
        self.two_limb_ops += 1;
        self.two_limb_mul_ops += 1;
    }

    fn record_two_limb_memory_read(&mut self) {
        self.two_limb_memory_reads += 1;
    }

    fn record_two_limb_mux_op(&mut self) {
        self.two_limb_mux_ops += 1;
    }

    fn record_memory_write_effect(&mut self) {
        self.memory_write_effects += 1;
    }

    fn total_profiled_instrs(&self) -> usize {
        self.one_limb_ops + self.two_limb_ops + self.two_limb_memory_reads + self.two_limb_mux_ops
    }

    fn merge(&mut self, other: Self) {
        self.one_limb_ops += other.one_limb_ops;
        self.two_limb_ops += other.two_limb_ops;
        self.two_limb_mul_ops += other.two_limb_mul_ops;
        self.two_limb_memory_reads += other.two_limb_memory_reads;
        self.two_limb_mux_ops += other.two_limb_mux_ops;
        self.memory_write_effects += other.memory_write_effects;
    }
}

/// Static estimate of whether the current SIMD CPU kernels will profitably run a program.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimdSuitabilityReport {
    pub streams: SimdSuitabilityStreamsReport,
    pub total: SimdSuitabilityBlockReport,
    pub recommendation: SimdSuitabilityRecommendation,
    pub score_x100: usize,
    pub estimated_fast_cost: usize,
    pub estimated_fallback_cost: usize,
    pub estimated_materialization_cost: usize,
    pub fallback_ratio_x100: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimdSuitabilityStreamsReport {
    pub async_reset_comb: SimdSuitabilityBlockReport,
    pub comb: SimdSuitabilityBlockReport,
    pub tick_next: SimdSuitabilityBlockReport,
    pub tick_commit: SimdSuitabilityBlockReport,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimdSuitabilityBlockReport {
    pub instr_count: usize,
    pub fast_instrs: usize,
    pub fallback_instrs: usize,
    pub lane_materializations_per_lane: usize,
    #[serde(default)]
    pub fast_path_profile: SimdFastPathProfile,
    pub fallback_reasons: ReplaySimdFallbackStats,
    pub one_limb_instrs: usize,
    pub wide_instrs: usize,
    pub memory_read_instrs: usize,
    pub memory_write_effects: usize,
    pub signed_lt_instrs: usize,
    pub wide_concat_instrs: usize,
    pub max_packet_width: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimdFastPathProfile {
    pub one_limb_ops: usize,
    pub two_limb_ops: usize,
    pub two_limb_mul_ops: usize,
    pub two_limb_memory_reads: usize,
    pub two_limb_mux_ops: usize,
    pub memory_write_effects: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SimdSuitabilityRecommendation {
    ScalarPreferred,
    SimdCandidate,
    MixedCandidate,
    GpuCandidateBlocked,
}

/// Static backend-affinity summary for explaining replay backend choices.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendAffinityReport {
    pub streams: BackendAffinityStreamsReport,
    pub total: BackendAffinityBlockReport,
    pub recommendation: BackendAffinityRecommendation,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendAffinityStreamsReport {
    pub async_reset_comb: BackendAffinityBlockReport,
    pub comb: BackendAffinityBlockReport,
    pub tick_next: BackendAffinityBlockReport,
    pub tick_commit: BackendAffinityBlockReport,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendAffinityBlockReport {
    pub instr_count: usize,
    pub scalar_fast_one_limb_instrs: usize,
    pub simd_fast_instrs: usize,
    pub wide_fallback_instrs: usize,
    #[serde(default)]
    pub simd_fast_path_profile: SimdFastPathProfile,
    pub memory_ops: usize,
    pub concat_pressure_instrs: usize,
    pub signed_ops: usize,
    pub max_packet_width: usize,
    pub estimated_scalar_cost: usize,
    pub estimated_packed_cpu_cost: usize,
    pub estimated_simd_cpu_cost: usize,
    pub gpu_hostile_reasons: ReplaySimdFallbackStats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BackendAffinityRecommendation {
    ScalarPreferred,
    PackedCpuCandidate,
    SimdCpuCandidate,
    MixedScalarSimdCandidate,
    GpuBlocked,
}

/// Options for conservative packed machine scheduling passes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedScheduleOptions {
    pub max_packet_width: Option<usize>,
    pub max_memory_reads_per_packet: Option<usize>,
    pub liveness_priority: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct FlatKey {
    instance: usize,
    signal: Signal,
}

#[derive(Clone, Debug)]
struct InstanceCtx {
    path: String,
    module: CompiledModule,
}

#[derive(Clone, Debug)]
struct FlatAssignment {
    dst: usize,
    expr: PackedExpr,
}

pub fn lower_to_packed_program(
    design: &CompiledDesign,
    top: &str,
) -> Result<PackedProgram, ErrorReport> {
    let mut diagnostics = Vec::new();
    let Some(top_module) = design.find_module(top).cloned() else {
        return Err(ErrorReport::new(vec![Diagnostic::new(
            "E_SIM_IR_MODULE",
            format!("module `{top}` does not exist"),
        )]));
    };
    if top_module.is_external {
        diagnostics.push(Diagnostic::new(
            "E_SIM_IR_EXTERN",
            "packed simulation IR does not support external top modules",
        ));
    }

    let mut lower = Lowering {
        design,
        diagnostics,
        contexts: Vec::new(),
        signals: Vec::new(),
        memories: Vec::new(),
        signal_map: HashMap::new(),
        memory_map: HashMap::new(),
        top_signal_indices: HashMap::new(),
        assignments: Vec::new(),
        registers: Vec::new(),
        memory_writes: Vec::new(),
        mem_clocks: HashMap::new(),
        next_signal_offset: 0,
        next_memory_offset: 0,
    };
    lower.add_context(top_module, top.to_string());
    lower.allocate_contexts();
    lower.lower_contexts();

    if !lower.diagnostics.is_empty() {
        return Err(ErrorReport::new(lower.diagnostics));
    }

    let comb = schedule_assignments(&lower.assignments);
    let async_reset_comb = schedule_async_resets(&lower.registers);
    let tick_next = schedule_tick_next(&lower.registers);
    let tick_commit = schedule_memory_writes(&lower.memory_writes);

    Ok(PackedProgram {
        top: top.to_string(),
        signals: lower.signals,
        memories: lower.memories,
        streams: PackedStreams {
            async_reset_comb,
            comb,
            tick_next,
            tick_commit,
        },
        top_signal_indices: lower.top_signal_indices,
        total_signal_words: lower.next_signal_offset,
        total_memory_words_per_lane: lower.next_memory_offset,
        reg_clocks: lower
            .registers
            .iter()
            .filter_map(|reg| reg.clock.map(|clk| (reg.signal, clk)))
            .collect(),
        mem_clocks: lower.mem_clocks,
    })
}

pub fn lower_to_machine_program(program: &PackedProgram) -> PackedMachineProgram {
    PackedMachineProgram {
        source: program.clone(),
        streams: PackedMachineStreams {
            async_reset_comb: lower_packets_to_block(&program.streams.async_reset_comb),
            comb: lower_packets_to_block(&program.streams.comb),
            tick_next: lower_packets_to_block(&program.streams.tick_next),
            tick_commit: lower_packets_to_block(&program.streams.tick_commit),
        },
    }
}

/// One partition group's owned instance-path contexts, used to slice a
/// [`PackedProgram`] into per-partition sub-programs. A group owns every
/// signal/memory whose [`PackedSignal::owner_path`] is at or below one of its
/// `owned_paths` in the instance hierarchy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedSliceGroup {
    pub id: String,
    /// Instance-path contexts owned by this group (e.g. `["Top.u_a"]`).
    pub owned_paths: Vec<String>,
}

/// True iff `prefix` is a path-prefix of `path` on `.`-separated components
/// (equal paths included). Component-wise so `Top.u_a` is a prefix of
/// `Top.u_a.inner` but not of `Top.u_ab`.
fn is_instance_path_prefix(prefix: &str, path: &str) -> bool {
    let mut prefix_components = prefix.split('.');
    let mut path_components = path.split('.');
    loop {
        match prefix_components.next() {
            None => return true,
            Some(want) => match path_components.next() {
                Some(have) if have == want => continue,
                _ => return false,
            },
        }
    }
}

/// Returns the index into `groups` of the group owning `owner_path` via its
/// most-specific (longest) matching context, or `None` if unclaimed. Ties
/// resolve to the earliest group; well-formed specs whose `owned_paths`
/// partition the hierarchy never tie.
fn instance_path_owner(owner_path: &str, groups: &[PackedSliceGroup]) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None; // (group index, matched component count)
    for (group_index, group) in groups.iter().enumerate() {
        for owned in &group.owned_paths {
            if is_instance_path_prefix(owned, owner_path) {
                let specificity = owned.split('.').count();
                if best.is_none_or(|(_, best_specificity)| specificity > best_specificity) {
                    best = Some((group_index, specificity));
                }
            }
        }
    }
    best.map(|(group_index, _)| group_index)
}

/// Assigns each signal of `program` (by index) to the owning group index, or
/// `None` if no group claims it, using most-specific instance-path ownership.
pub fn classify_signal_owners(
    program: &PackedProgram,
    groups: &[PackedSliceGroup],
) -> Vec<Option<usize>> {
    program
        .signals
        .iter()
        .map(|signal| instance_path_owner(&signal.owner_path, groups))
        .collect()
}

/// Assigns each memory of `program` (by index) to the owning group index, or
/// `None` if no group claims it. See [`classify_signal_owners`].
pub fn classify_memory_owners(
    program: &PackedProgram,
    groups: &[PackedSliceGroup],
) -> Vec<Option<usize>> {
    program
        .memories
        .iter()
        .map(|memory| instance_path_owner(&memory.owner_path, groups))
        .collect()
}

/// One partition group's sliced sub-program plus the maps needed to wire
/// cross-group boundary exchange by original global index (name-independent).
#[derive(Clone, Debug)]
pub struct PackedSlice {
    pub program: PackedProgram,
    /// new signal index -> original global signal index.
    pub signal_origin: Vec<usize>,
    /// new memory index -> original global memory index.
    pub memory_origin: Vec<usize>,
    /// new signal indices that are boundary inputs (a register owned by another
    /// group, consumed here as an `Input` set from that group each cycle).
    pub boundary_inputs: Vec<usize>,
}

fn slice_error(message: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![Diagnostic::new("E_SIM_IR_SLICE", message)])
}

/// Slices `program` into one [`PackedSlice`] per group (same order as `groups`).
///
/// Each slice owns the signals/memories assigned to its group by
/// [`classify_signal_owners`]/[`classify_memory_owners`]. Any signal a group
/// reads but does not own becomes a boundary `Input` (must be register-stable —
/// guaranteed when `groups` come from a legalized plan). A cross-group memory
/// read is rejected, since memory reads are combinational and cannot be exposed
/// as a per-cycle input.
pub fn slice_packed_program(
    program: &PackedProgram,
    groups: &[PackedSliceGroup],
) -> Result<Vec<PackedSlice>, ErrorReport> {
    let signal_owner = classify_signal_owners(program, groups);
    let memory_owner = classify_memory_owners(program, groups);
    // Disjoint single-owner ownership is the no-replication special case of the
    // present-set slicer.
    (0..groups.len())
        .map(|group| {
            let present_signals: Vec<bool> = signal_owner
                .iter()
                .map(|owner| *owner == Some(group))
                .collect();
            let present_memories: Vec<bool> = memory_owner
                .iter()
                .map(|owner| *owner == Some(group))
                .collect();
            slice_present(program, &present_signals, &present_memories)
        })
        .collect()
}

/// Whether a group that computes the given `present` signals/memories emits this
/// op — i.e. the op produces a signal/memory the group is responsible for.
/// (A combinational signal present in several groups is replicated into each.)
fn op_present(op: &PackedOp, present_signals: &[bool], present_memories: &[bool]) -> bool {
    match op {
        PackedOp::Assign { dst, .. } | PackedOp::CaptureReg { dst, .. } => present_signals[*dst],
        PackedOp::MemoryWrite { memory, .. } => present_memories[*memory],
    }
}

fn collect_expr_reads(expr: &PackedExpr, signals: &mut [bool], memories: &mut [bool]) {
    match &expr.kind {
        PackedExprKind::Lit(_) => {}
        PackedExprKind::Signal(index) => signals[*index] = true,
        PackedExprKind::Not(inner)
        | PackedExprKind::Zext(inner)
        | PackedExprKind::Sext(inner)
        | PackedExprKind::Trunc(inner)
        | PackedExprKind::Cast(inner)
        | PackedExprKind::Slice { expr: inner, .. } => {
            collect_expr_reads(inner, signals, memories)
        }
        PackedExprKind::And(lhs, rhs)
        | PackedExprKind::Or(lhs, rhs)
        | PackedExprKind::Xor(lhs, rhs)
        | PackedExprKind::Add(lhs, rhs)
        | PackedExprKind::Sub(lhs, rhs)
        | PackedExprKind::Mul(lhs, rhs)
        | PackedExprKind::Eq(lhs, rhs)
        | PackedExprKind::Ne(lhs, rhs)
        | PackedExprKind::Lt { lhs, rhs, .. } => {
            collect_expr_reads(lhs, signals, memories);
            collect_expr_reads(rhs, signals, memories);
        }
        PackedExprKind::Mux {
            cond,
            then_expr,
            else_expr,
        } => {
            collect_expr_reads(cond, signals, memories);
            collect_expr_reads(then_expr, signals, memories);
            collect_expr_reads(else_expr, signals, memories);
        }
        PackedExprKind::Concat(parts) => {
            for part in parts {
                collect_expr_reads(part, signals, memories);
            }
        }
        PackedExprKind::MemRead { memory, addr } => {
            memories[*memory] = true;
            collect_expr_reads(addr, signals, memories);
        }
    }
}

fn collect_op_reads(op: &PackedOp, signals: &mut [bool], memories: &mut [bool]) {
    match op {
        PackedOp::Assign { expr, .. } => collect_expr_reads(expr, signals, memories),
        PackedOp::CaptureReg { next, reset, .. } => {
            collect_expr_reads(next, signals, memories);
            if let Some(reset) = reset {
                signals[reset.signal] = true;
            }
        }
        PackedOp::MemoryWrite {
            enable, addr, data, ..
        } => {
            collect_expr_reads(enable, signals, memories);
            collect_expr_reads(addr, signals, memories);
            collect_expr_reads(data, signals, memories);
        }
    }
}

/// Backward cone-of-influence from a set of OBSERVED signals/memories: every signal
/// and memory that transitively feeds an observed one — through combinational defs
/// and register next-state cones. Everything outside the cone is dead for those
/// observations and can be pruned (feed the returned masks to [`slice_present`]).
///
/// This is the static "compute only what you observe" optimization: a testbench
/// checks a handful of outputs (a checksum, a trap, a bus trace), so most of the
/// design falls out of the cone. Because the observed set is fixed it is data-
/// oblivious, so the pruned program composes with every backend and multiplies the
/// batch/GPU throughput. Returns `(present_signals, present_memories)` masks.
pub fn cone_of_influence(
    program: &PackedProgram,
    observed_signals: &[usize],
    observed_memories: &[usize],
) -> (Vec<bool>, Vec<bool>) {
    let mut sig = vec![false; program.signals.len()];
    let mut mem = vec![false; program.memories.len()];
    for &s in observed_signals {
        if s < sig.len() {
            sig[s] = true;
        }
    }
    for &m in observed_memories {
        if m < mem.len() {
            mem[m] = true;
        }
    }
    // Fixpoint: an op's reads enter the cone once its destination is in the cone.
    loop {
        let before =
            sig.iter().filter(|x| **x).count() + mem.iter().filter(|x| **x).count();
        for stream in [
            &program.streams.async_reset_comb,
            &program.streams.comb,
            &program.streams.tick_next,
            &program.streams.tick_commit,
        ] {
            for packet in stream {
                for op in &packet.ops {
                    if op_present(op, &sig, &mem) {
                        collect_op_reads(op, &mut sig, &mut mem);
                    }
                }
            }
        }
        let after = sig.iter().filter(|x| **x).count() + mem.iter().filter(|x| **x).count();
        if after == before {
            break;
        }
    }
    (sig, mem)
}

/// The natural byte size that holds a signal of the given width: 1/2/4/8/16.
pub fn state_store_bytes(width: u32) -> usize {
    match width {
        0..=8 => 1,
        9..=16 => 2,
        17..=32 => 4,
        33..=64 => 8,
        _ => 16,
    }
}

/// Compute a **per-width-packed** state layout: each signal gets a slot of
/// [`state_store_bytes`] bytes (vs a uniform 16-byte slot), shrinking the working
/// set ~4–8× on width-heavy designs for far better cache density. Returns
/// `(byte_offset_per_signal, total_bytes)` (total aligned to 16).
///
/// `affinity_order` optionally lists signal indices in the order they should be
/// placed, so co-accessed signals (a register cone's support) land adjacent and
/// share cache lines; remaining signals are appended. `None` packs by size class
/// (largest first), which is naturally aligned with zero inter-slot padding.
///
/// This is the single source of truth for state packing across backends (the AOT
/// uses it; the single-instance JIT's uniform `i*16`/`i*2` layout is a public
/// contract — `JitSimulator::signal_word`/`state_words`, consumed by the
/// partitioned/zero-copy batch harnesses — so adopting this there is a
/// cross-cutting migration, tracked separately).
pub fn packed_signal_layout(widths: &[u32], affinity_order: Option<&[usize]>) -> (Vec<usize>, usize) {
    let align = |x: usize, a: usize| (x + a - 1) & !(a - 1);
    let order: Vec<usize> = match affinity_order {
        Some(o) => {
            let mut seen = vec![false; widths.len()];
            let mut ord = Vec::with_capacity(widths.len());
            for &s in o {
                if s < widths.len() && !seen[s] {
                    seen[s] = true;
                    ord.push(s);
                }
            }
            for s in 0..widths.len() {
                if !seen[s] {
                    ord.push(s);
                }
            }
            ord
        }
        None => {
            // Size class largest-first: every class starts at a multiple of its
            // size, so all slots are naturally aligned with zero padding.
            let mut ord = Vec::with_capacity(widths.len());
            for size in [16usize, 8, 4, 2, 1] {
                for (s, &w) in widths.iter().enumerate() {
                    if state_store_bytes(w) == size {
                        ord.push(s);
                    }
                }
            }
            ord
        }
    };
    let mut off = vec![0usize; widths.len()];
    let mut cur = 0usize;
    for &s in &order {
        let sz = state_store_bytes(widths[s]);
        cur = align(cur, sz);
        off[s] = cur;
        cur += sz;
    }
    (off, align(cur, 16))
}

fn remap_signal(old: usize, signal_map: &[Option<usize>]) -> Result<usize, ErrorReport> {
    signal_map[old].ok_or_else(|| slice_error(format!("unmapped signal index {old} in slice")))
}

fn remap_expr(
    expr: &PackedExpr,
    signal_map: &[Option<usize>],
    memory_map: &[Option<usize>],
) -> Result<PackedExpr, ErrorReport> {
    let kind = match &expr.kind {
        PackedExprKind::Lit(words) => PackedExprKind::Lit(words.clone()),
        PackedExprKind::Signal(index) => {
            PackedExprKind::Signal(remap_signal(*index, signal_map)?)
        }
        PackedExprKind::Not(inner) => {
            PackedExprKind::Not(Box::new(remap_expr(inner, signal_map, memory_map)?))
        }
        PackedExprKind::Zext(inner) => {
            PackedExprKind::Zext(Box::new(remap_expr(inner, signal_map, memory_map)?))
        }
        PackedExprKind::Sext(inner) => {
            PackedExprKind::Sext(Box::new(remap_expr(inner, signal_map, memory_map)?))
        }
        PackedExprKind::Trunc(inner) => {
            PackedExprKind::Trunc(Box::new(remap_expr(inner, signal_map, memory_map)?))
        }
        PackedExprKind::Cast(inner) => {
            PackedExprKind::Cast(Box::new(remap_expr(inner, signal_map, memory_map)?))
        }
        PackedExprKind::Slice { expr: inner, lsb } => PackedExprKind::Slice {
            expr: Box::new(remap_expr(inner, signal_map, memory_map)?),
            lsb: *lsb,
        },
        PackedExprKind::And(lhs, rhs) => PackedExprKind::And(
            Box::new(remap_expr(lhs, signal_map, memory_map)?),
            Box::new(remap_expr(rhs, signal_map, memory_map)?),
        ),
        PackedExprKind::Or(lhs, rhs) => PackedExprKind::Or(
            Box::new(remap_expr(lhs, signal_map, memory_map)?),
            Box::new(remap_expr(rhs, signal_map, memory_map)?),
        ),
        PackedExprKind::Xor(lhs, rhs) => PackedExprKind::Xor(
            Box::new(remap_expr(lhs, signal_map, memory_map)?),
            Box::new(remap_expr(rhs, signal_map, memory_map)?),
        ),
        PackedExprKind::Add(lhs, rhs) => PackedExprKind::Add(
            Box::new(remap_expr(lhs, signal_map, memory_map)?),
            Box::new(remap_expr(rhs, signal_map, memory_map)?),
        ),
        PackedExprKind::Sub(lhs, rhs) => PackedExprKind::Sub(
            Box::new(remap_expr(lhs, signal_map, memory_map)?),
            Box::new(remap_expr(rhs, signal_map, memory_map)?),
        ),
        PackedExprKind::Mul(lhs, rhs) => PackedExprKind::Mul(
            Box::new(remap_expr(lhs, signal_map, memory_map)?),
            Box::new(remap_expr(rhs, signal_map, memory_map)?),
        ),
        PackedExprKind::Eq(lhs, rhs) => PackedExprKind::Eq(
            Box::new(remap_expr(lhs, signal_map, memory_map)?),
            Box::new(remap_expr(rhs, signal_map, memory_map)?),
        ),
        PackedExprKind::Ne(lhs, rhs) => PackedExprKind::Ne(
            Box::new(remap_expr(lhs, signal_map, memory_map)?),
            Box::new(remap_expr(rhs, signal_map, memory_map)?),
        ),
        PackedExprKind::Lt { lhs, rhs, signed } => PackedExprKind::Lt {
            lhs: Box::new(remap_expr(lhs, signal_map, memory_map)?),
            rhs: Box::new(remap_expr(rhs, signal_map, memory_map)?),
            signed: *signed,
        },
        PackedExprKind::Mux {
            cond,
            then_expr,
            else_expr,
        } => PackedExprKind::Mux {
            cond: Box::new(remap_expr(cond, signal_map, memory_map)?),
            then_expr: Box::new(remap_expr(then_expr, signal_map, memory_map)?),
            else_expr: Box::new(remap_expr(else_expr, signal_map, memory_map)?),
        },
        PackedExprKind::Concat(parts) => PackedExprKind::Concat(
            parts
                .iter()
                .map(|part| remap_expr(part, signal_map, memory_map))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        PackedExprKind::MemRead { memory, addr } => PackedExprKind::MemRead {
            memory: memory_map[*memory]
                .ok_or_else(|| slice_error("cross-group memory read cannot be exposed as input"))?,
            addr: Box::new(remap_expr(addr, signal_map, memory_map)?),
        },
    };
    Ok(PackedExpr {
        kind,
        ty: expr.ty,
    })
}

fn remap_op(
    op: &PackedOp,
    signal_map: &[Option<usize>],
    memory_map: &[Option<usize>],
) -> Result<PackedOp, ErrorReport> {
    Ok(match op {
        PackedOp::Assign { dst, expr } => PackedOp::Assign {
            dst: remap_signal(*dst, signal_map)?,
            expr: remap_expr(expr, signal_map, memory_map)?,
        },
        PackedOp::CaptureReg { dst, next, reset } => PackedOp::CaptureReg {
            dst: remap_signal(*dst, signal_map)?,
            next: remap_expr(next, signal_map, memory_map)?,
            reset: reset
                .as_ref()
                .map(|reset| {
                    Ok::<_, ErrorReport>(PackedReset {
                        signal: remap_signal(reset.signal, signal_map)?,
                        value: reset.value.clone(),
                        kind: reset.kind,
                        polarity: reset.polarity,
                    })
                })
                .transpose()?,
        },
        PackedOp::MemoryWrite {
            memory,
            enable,
            addr,
            data,
        } => PackedOp::MemoryWrite {
            memory: memory_map[*memory]
                .ok_or_else(|| slice_error("memory write to unowned memory in slice"))?,
            enable: remap_expr(enable, signal_map, memory_map)?,
            addr: remap_expr(addr, signal_map, memory_map)?,
            data: remap_expr(data, signal_map, memory_map)?,
        },
    })
}

fn slice_stream(
    stream: &[PackedPacket],
    present_signals: &[bool],
    present_memories: &[bool],
    signal_map: &[Option<usize>],
    memory_map: &[Option<usize>],
) -> Result<Vec<PackedPacket>, ErrorReport> {
    let mut packets = Vec::new();
    for packet in stream {
        let mut ops = Vec::new();
        for op in &packet.ops {
            if op_present(op, present_signals, present_memories) {
                ops.push(remap_op(op, signal_map, memory_map)?);
            }
        }
        if !ops.is_empty() {
            packets.push(PackedPacket { ops });
        }
    }
    Ok(packets)
}

/// Builds one group's sub-program given the signals/memories it is responsible
/// for computing (`present_*`). A signal present here is computed locally and
/// keeps its kind; a signal read but not present becomes a boundary `Input`.
/// Combinational signals may be present in several groups at once (replication).
pub fn slice_present(
    program: &PackedProgram,
    present_signals: &[bool],
    present_memories: &[bool],
) -> Result<PackedSlice, ErrorReport> {
    // Pass 1: mark every signal/memory read by an op this group computes.
    let mut read_signals = vec![false; program.signals.len()];
    let mut read_memories = vec![false; program.memories.len()];
    for stream in [
        &program.streams.async_reset_comb,
        &program.streams.comb,
        &program.streams.tick_next,
        &program.streams.tick_commit,
    ] {
        for packet in stream {
            for op in &packet.ops {
                if op_present(op, present_signals, present_memories) {
                    collect_op_reads(op, &mut read_signals, &mut read_memories);
                }
            }
        }
    }

    // A memory read must be of a memory this group holds: memory state cannot be
    // a per-cycle boundary input (it is read combinationally and is mutable).
    for (memory, &read) in read_memories.iter().enumerate() {
        if read && !present_memories[memory] {
            return Err(slice_error(format!(
                "group reads memory `{}` it does not hold",
                program.memories[memory].name
            )));
        }
    }

    // Pass 2: build the compact signal space (present signals keep their kind;
    // read-but-absent signals become boundary `Input`s) with fresh offsets.
    let mut signal_map = vec![None; program.signals.len()];
    let mut signals = Vec::new();
    let mut signal_origin = Vec::new();
    let mut boundary_inputs = Vec::new();
    let mut signal_offset = 0;
    for (old, signal) in program.signals.iter().enumerate() {
        let present = present_signals[old];
        if !present && !read_signals[old] {
            continue;
        }
        let new = signals.len();
        signal_map[old] = Some(new);
        let mut sliced = signal.clone();
        sliced.layout.offset = signal_offset;
        signal_offset += sliced.layout.limbs;
        if !present {
            sliced.kind = PackedSignalKind::Input;
            boundary_inputs.push(new);
        }
        signals.push(sliced);
        signal_origin.push(old);
    }

    // Memories: include every memory this group holds, compactly re-offset.
    let mut memory_map = vec![None; program.memories.len()];
    let mut memories = Vec::new();
    let mut memory_origin = Vec::new();
    let mut memory_offset = 0;
    for (old, memory) in program.memories.iter().enumerate() {
        if !present_memories[old] {
            continue;
        }
        let new = memories.len();
        memory_map[old] = Some(new);
        let mut sliced = memory.clone();
        sliced.offset = memory_offset;
        memory_offset += memory.depth * memory.data_layout.limbs;
        memories.push(sliced);
        memory_origin.push(old);
    }

    let streams = PackedStreams {
        async_reset_comb: slice_stream(
            &program.streams.async_reset_comb,
            present_signals,
            present_memories,
            &signal_map,
            &memory_map,
        )?,
        comb: slice_stream(
            &program.streams.comb,
            present_signals,
            present_memories,
            &signal_map,
            &memory_map,
        )?,
        tick_next: slice_stream(
            &program.streams.tick_next,
            present_signals,
            present_memories,
            &signal_map,
            &memory_map,
        )?,
        tick_commit: slice_stream(
            &program.streams.tick_commit,
            present_signals,
            present_memories,
            &signal_map,
            &memory_map,
        )?,
    };

    // Rebuild the top-signal handle index from surviving sourced signals.
    let top_signal_indices = signals
        .iter()
        .enumerate()
        .filter_map(|(new, signal)| signal.source.map(|handle| (handle, new)))
        .collect();

    let sliced_program = PackedProgram {
        top: program.top.clone(),
        signals,
        memories,
        streams,
        top_signal_indices,
        total_signal_words: signal_offset,
        total_memory_words_per_lane: memory_offset,
        // Slicing (partitioned JIT) is single-clock-oriented; clock gating in a
        // slice would need index remapping, omitted until the two are combined.
        reg_clocks: HashMap::new(),
        mem_clocks: HashMap::new(),
    };

    Ok(PackedSlice {
        program: sliced_program,
        signal_origin,
        memory_origin,
        boundary_inputs,
    })
}

/// Collects the signal indices read by `expr` (memories ignored), appending to
/// `out`; duplicates are allowed (callers dedup).
fn push_signal_reads(expr: &PackedExpr, out: &mut Vec<usize>) {
    match &expr.kind {
        PackedExprKind::Lit(_) => {}
        PackedExprKind::Signal(index) => out.push(*index),
        PackedExprKind::Not(inner)
        | PackedExprKind::Zext(inner)
        | PackedExprKind::Sext(inner)
        | PackedExprKind::Trunc(inner)
        | PackedExprKind::Cast(inner)
        | PackedExprKind::Slice { expr: inner, .. } => push_signal_reads(inner, out),
        PackedExprKind::And(lhs, rhs)
        | PackedExprKind::Or(lhs, rhs)
        | PackedExprKind::Xor(lhs, rhs)
        | PackedExprKind::Add(lhs, rhs)
        | PackedExprKind::Sub(lhs, rhs)
        | PackedExprKind::Mul(lhs, rhs)
        | PackedExprKind::Eq(lhs, rhs)
        | PackedExprKind::Ne(lhs, rhs)
        | PackedExprKind::Lt { lhs, rhs, .. } => {
            push_signal_reads(lhs, out);
            push_signal_reads(rhs, out);
        }
        PackedExprKind::Mux {
            cond,
            then_expr,
            else_expr,
        } => {
            push_signal_reads(cond, out);
            push_signal_reads(then_expr, out);
            push_signal_reads(else_expr, out);
        }
        PackedExprKind::Concat(parts) => {
            for part in parts {
                push_signal_reads(part, out);
            }
        }
        PackedExprKind::MemRead { addr, .. } => push_signal_reads(addr, out),
    }
}

/// Transitive combinational fan-in of `seed` reads: the set of combinational
/// signals (those with a definition in `comb_def`) reachable backwards, stopping
/// at register/input leaves. This is the cone of logic a group must replicate to
/// compute a sink (a register's next value or an output) reading only registers
/// and inputs at its boundary.
fn build_cone(seed: &[usize], comb_def: &HashMap<usize, &PackedExpr>, signals: usize) -> Vec<usize> {
    let mut in_cone = vec![false; signals];
    let mut cone = Vec::new();
    let mut stack = seed.to_vec();
    while let Some(signal) = stack.pop() {
        if in_cone[signal] {
            continue;
        }
        if let Some(expr) = comb_def.get(&signal) {
            in_cone[signal] = true;
            cone.push(signal);
            push_signal_reads(expr, &mut stack);
        }
    }
    cone
}

/// Computes per-group "present" signal sets for a register-cone (RepCut-style)
/// partition of `program` into `target_groups` balanced groups.
///
/// Each register and output is a *sink* assigned to one group; that group also
/// computes (replicates) the sink's entire combinational cone. Because every
/// group computes the full cone of its sinks, all cross-group reads are of
/// registers or inputs (stable), so the groups carry no combinational
/// dependency on each other and execute fully in parallel. Sinks are greedily
/// bin-packed onto the lightest group by cone size. Memories are not yet
/// supported (their state cannot be replicated).
pub fn partition_registers_balanced(
    program: &PackedProgram,
    target_groups: usize,
) -> Result<Vec<Vec<bool>>, ErrorReport> {
    if !program.memories.is_empty() {
        return Err(slice_error(
            "register-cone partitioning does not yet support memories",
        ));
    }
    let groups = target_groups.max(1);
    let n = program.signals.len();

    // Combinational definitions: signal -> its `comb`-stream assign expression.
    let mut comb_def: HashMap<usize, &PackedExpr> = HashMap::new();
    for packet in &program.streams.comb {
        for op in &packet.ops {
            if let PackedOp::Assign { dst, expr } = op {
                comb_def.insert(*dst, expr);
            }
        }
    }

    // Sinks: (owned signal, cone). Registers (from tick_next) and outputs.
    let mut sinks: Vec<(usize, Vec<usize>)> = Vec::new();
    for packet in &program.streams.tick_next {
        for op in &packet.ops {
            if let PackedOp::CaptureReg { dst, next, reset } = op {
                let mut seed = Vec::new();
                push_signal_reads(next, &mut seed);
                if let Some(reset) = reset {
                    seed.push(reset.signal);
                }
                sinks.push((*dst, build_cone(&seed, &comb_def, n)));
            }
        }
    }
    for (index, signal) in program.signals.iter().enumerate() {
        if signal.kind == PackedSignalKind::Output {
            let mut seed = Vec::new();
            if let Some(expr) = comb_def.get(&index) {
                push_signal_reads(expr, &mut seed);
            }
            sinks.push((index, build_cone(&seed, &comb_def, n)));
        }
    }

    // Greedy longest-cone-first bin-packing onto the lightest group.
    sinks.sort_by_key(|(owned, cone)| (std::cmp::Reverse(cone.len()), *owned));

    let mut present = vec![vec![false; n]; groups];
    // Top-level inputs are held by group 0 (the driver target); other groups
    // read them as stable boundary inputs.
    for (index, signal) in program.signals.iter().enumerate() {
        if signal.kind == PackedSignalKind::Input {
            present[0][index] = true;
        }
    }
    let mut group_cost = vec![0usize; groups];
    for (owned, cone) in sinks {
        let group = (0..groups).min_by_key(|&g| (group_cost[g], g)).unwrap();
        present[group][owned] = true;
        for &signal in &cone {
            present[group][signal] = true;
        }
        group_cost[group] += cone.len() + 1;
    }

    Ok(present)
}

/// Deterministic topological *levels* of `n` groups given comb dependency
/// `edges` (`(producer, consumer)`): a group's level is one past the deepest
/// combinational producer feeding it, so all groups within a level are mutually
/// independent and may evaluate concurrently. Errors on a combinational cycle
/// across groups, which v0 cannot schedule without replication.
fn topological_group_levels(
    n: usize,
    edges: &[(usize, usize)],
) -> Result<Vec<Vec<usize>>, ErrorReport> {
    let mut indegree = vec![0usize; n];
    let mut adjacency = vec![Vec::new(); n];
    for &(producer, consumer) in edges {
        adjacency[producer].push(consumer);
        indegree[consumer] += 1;
    }
    let mut ready: Vec<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
    let mut level = vec![0usize; n];
    let mut processed = 0;
    let mut head = 0;
    while head < ready.len() {
        let node = ready[head];
        head += 1;
        processed += 1;
        for &next in &adjacency[node] {
            // `node` is final here (Kahn pops a node only after all its
            // producers), so this relaxation can only raise `next`'s level.
            level[next] = level[next].max(level[node] + 1);
            indegree[next] -= 1;
            if indegree[next] == 0 {
                ready.push(next);
            }
        }
    }
    if processed != n {
        return Err(slice_error(
            "combinational cycle across partition groups (needs replication)",
        ));
    }
    let depth = level.iter().copied().max().unwrap_or(0);
    let mut levels = vec![Vec::new(); depth + 1];
    for (group, &group_level) in level.iter().enumerate() {
        levels[group_level].push(group);
    }
    Ok(levels)
}

#[derive(Clone, Copy)]
struct BoundaryRoute {
    consumer_group: usize,
    consumer_index: usize,
    producer_group: usize,
    producer_index: usize,
    /// Producer is a register or top-level input → value is stable for the
    /// whole cycle, so it is exchanged once before settle and imposes no
    /// scheduling order. Combinational producers (`false`) are exchanged in
    /// topological order during settle.
    stable: bool,
}

struct GroupSim {
    /// Each group runs the lane-vectorized engine (SIMD across stimulus lanes
    /// with preallocated workspaces), so partitioned execution composes
    /// thread-parallelism across groups with SIMD parallelism within a group.
    sim: SimdCpuSimulator,
}

/// Per-tick work (total ops × lanes) below which threading across groups costs
/// more (pool dispatch, boundary exchange, per-group fixed overhead) than it
/// saves. Calibrated for the SIMD per-group engine, which makes each op ~20-60x
/// cheaper than a scalar walk, so the crossover sits high: ~53k ops/tick still
/// regresses under threads while ~800k benefits (~2.4x at 4-8 groups). This is a
/// coarse heuristic; callers with a known workload should override via
/// [`PartitionedSimulator::set_parallel`].
const PARALLEL_WORK_THRESHOLD: usize = 262_144;

fn packed_program_op_count(program: &PackedProgram) -> usize {
    [
        &program.streams.async_reset_comb,
        &program.streams.comb,
        &program.streams.tick_next,
        &program.streams.tick_commit,
    ]
    .iter()
    .flat_map(|stream| stream.iter())
    .map(|packet| packet.ops.len())
    .sum()
}

/// Executes a [`PackedProgram`] split into partition groups, one
/// [`PackedSimulator`] per group, exchanging boundary signal values between
/// them each cycle. Results are bit-identical to running the whole program in a
/// single [`PackedSimulator`] (the differential oracle), for designs whose
/// cross-group combinational boundaries are acyclic.
///
/// Stable boundary inputs (produced by a register or top-level input) are
/// exchanged once at the start of combinational settle; combinational boundary
/// inputs are exchanged as their producer group settles, in topological order.
/// A combinational cycle across groups is rejected.
pub struct PartitionedSimulator {
    groups: Vec<GroupSim>,
    routes: Vec<BoundaryRoute>,
    comb_levels: Vec<Vec<usize>>,
    handle_group: HashMap<Signal, usize>,
    lanes: usize,
    parallel: bool,
    /// Per group, the local signal indices that are combinational-cone *leaves*
    /// (kind `Input` or `Reg`) — the group's activity inputs. A group can be
    /// skipped on any cycle where none of these changed (see `tick_activity`).
    activity_inputs: Vec<Vec<usize>>,
    /// Per group, the activity-input values as of that group's last settle (empty
    /// = never settled, forcing the first tick to evaluate every group).
    activity_snap: Vec<Vec<u32>>,
    activity_skips: u64,
    activity_total: u64,
}

fn exchange_route(groups: &mut [GroupSim], route: &BoundaryRoute, lanes: usize) {
    for lane in 0..lanes {
        let value = groups[route.producer_group]
            .sim
            .inner()
            .signal_limbs_at(route.producer_index, lane);
        groups[route.consumer_group]
            .sim
            .inner_mut()
            .set_signal_limbs_at(route.consumer_index, lane, &value);
    }
}

/// Evaluates the combinational logic of every group in one topological level.
/// The groups are mutually independent (no combinational edge between them) and
/// their boundary inputs are already exchanged, so they run concurrently on
/// rayon's persistent thread pool (no per-cycle thread spawn); results are
/// independent of thread scheduling.
fn eval_level(groups: &mut [GroupSim], level: &[usize], parallel: bool) {
    if !parallel || level.len() == 1 {
        for &group in level {
            groups[group]
                .sim
                .eval_combinational()
                .expect("group eval_combinational");
        }
        return;
    }
    let mut active = vec![false; groups.len()];
    for &group in level {
        active[group] = true;
    }
    groups
        .par_iter_mut()
        .zip(active)
        .for_each(|(group, active)| {
            if active {
                group.sim.eval_combinational().expect("group eval_combinational");
            }
        });
}

impl PartitionedSimulator {
    /// Builds a partitioned simulator by slicing `program` into `groups`.
    /// `groups` are expected to partition the design (disjoint, covering); a
    /// boundary input must be owned by exactly one other group.
    pub fn new(
        program: &PackedProgram,
        groups: &[PackedSliceGroup],
        lanes: usize,
    ) -> Result<Self, ErrorReport> {
        let slices = slice_packed_program(program, groups)?;
        Self::from_slices(program, slices, lanes)
    }

    /// Builds an executor by partitioning `program` into `target_groups`
    /// balanced register-cone groups (RepCut-style replication). All groups are
    /// combinationally independent, so they execute fully in parallel with one
    /// barrier per cycle. Bit-identical to the whole-design simulator.
    pub fn new_register_balanced(
        program: &PackedProgram,
        target_groups: usize,
        lanes: usize,
    ) -> Result<Self, ErrorReport> {
        let present = partition_registers_balanced(program, target_groups)?;
        let no_memories = vec![false; program.memories.len()];
        let slices = present
            .iter()
            .map(|present_signals| slice_present(program, present_signals, &no_memories))
            .collect::<Result<Vec<_>, ErrorReport>>()?;
        Self::from_slices(program, slices, lanes)
    }

    /// Builds the executor from pre-computed slices — the shared core of every
    /// partitioning strategy (instance-path or register-cone).
    pub fn from_slices(
        program: &PackedProgram,
        slices: Vec<PackedSlice>,
        lanes: usize,
    ) -> Result<Self, ErrorReport> {
        // origin global signal index -> (group, local index) for the OWNER
        // (a local index that is not a boundary input).
        let mut owner_of: HashMap<usize, (usize, usize)> = HashMap::new();
        for (group, slice) in slices.iter().enumerate() {
            let boundary: HashSet<usize> = slice.boundary_inputs.iter().copied().collect();
            for (local, &origin) in slice.signal_origin.iter().enumerate() {
                if !boundary.contains(&local) {
                    owner_of.insert(origin, (group, local));
                }
            }
        }

        // Boundary routes + combinational dependency edges.
        let mut routes = Vec::new();
        let mut edges = Vec::new();
        for (group, slice) in slices.iter().enumerate() {
            for &local in &slice.boundary_inputs {
                let origin = slice.signal_origin[local];
                let &(producer_group, producer_index) = owner_of.get(&origin).ok_or_else(|| {
                    slice_error(format!("boundary input (origin {origin}) has no owning group"))
                })?;
                let stable = matches!(
                    program.signals[origin].kind,
                    PackedSignalKind::Reg | PackedSignalKind::Input
                );
                routes.push(BoundaryRoute {
                    consumer_group: group,
                    consumer_index: local,
                    producer_group,
                    producer_index,
                    stable,
                });
                if !stable {
                    edges.push((producer_group, group));
                }
            }
        }

        let comb_levels = topological_group_levels(slices.len(), &edges)?;

        // Signal handle -> owning group (set/get by handle targets the owner; a
        // boundary copy of the same handle in another group is ignored).
        let mut handle_group = HashMap::new();
        for (group, slice) in slices.iter().enumerate() {
            let boundary: HashSet<usize> = slice.boundary_inputs.iter().copied().collect();
            for (local, signal) in slice.program.signals.iter().enumerate() {
                if !boundary.contains(&local) {
                    if let Some(handle) = signal.source {
                        handle_group.insert(handle, group);
                    }
                }
            }
        }

        let groups = slices
            .into_iter()
            .map(|slice| {
                Ok(GroupSim {
                    sim: SimdCpuSimulator::new(slice.program, lanes)?,
                })
            })
            .collect::<Result<Vec<_>, ErrorReport>>()?;

        // Default to parallel only when there is enough work per tick to amortize
        // pool dispatch; tiny designs are faster serial (measured). Callers can
        // override with `set_parallel`.
        let ops_per_tick: usize = groups
            .iter()
            .map(|group| packed_program_op_count(group.sim.program()))
            .sum::<usize>()
            .saturating_mul(lanes);
        let parallel = groups.len() > 1 && ops_per_tick >= PARALLEL_WORK_THRESHOLD;

        // Activity inputs per group: cone leaves = signals of kind Input or Reg.
        let activity_inputs: Vec<Vec<usize>> = groups
            .iter()
            .map(|group| {
                group
                    .sim
                    .program()
                    .signals
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| {
                        matches!(s.kind, PackedSignalKind::Input | PackedSignalKind::Reg)
                    })
                    .map(|(i, _)| i)
                    .collect()
            })
            .collect();
        let activity_snap = vec![Vec::new(); groups.len()];

        Ok(Self {
            groups,
            routes,
            comb_levels,
            handle_group,
            lanes,
            parallel,
            activity_inputs,
            activity_snap,
            activity_skips: 0,
            activity_total: 0,
        })
    }

    /// Enables/disables parallel (multi-threaded) level evaluation. When off,
    /// groups within a level run serially on the calling thread — useful for
    /// isolating the slicing cost from the threading benefit, and faster for
    /// designs whose per-tick work is too small to amortize pool dispatch.
    pub fn set_parallel(&mut self, parallel: bool) {
        self.parallel = parallel;
    }

    /// Whether parallel level evaluation is currently enabled (auto-selected at
    /// construction from per-tick work, unless overridden by `set_parallel`).
    pub fn is_parallel(&self) -> bool {
        self.parallel
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    fn group_for(&self, signal: Signal) -> Result<usize, ErrorReport> {
        self.handle_group
            .get(&signal)
            .copied()
            .ok_or_else(|| slice_error("signal is not owned by any partition group"))
    }

    pub fn set_signal(&mut self, signal: Signal, lane_values: &[u128]) -> Result<(), ErrorReport> {
        let group = self.group_for(signal)?;
        self.groups[group].sim.set_signal(signal, lane_values)
    }

    pub fn get_signal(&self, signal: Signal) -> Result<Vec<u128>, ErrorReport> {
        let group = self.group_for(signal)?;
        self.groups[group].sim.get_signal(signal)
    }

    /// Settles combinational logic across all groups: stable boundary inputs are
    /// exchanged first, then each group evaluates in topological order with its
    /// combinational boundary inputs exchanged just before it runs.
    pub fn eval_combinational(&mut self) {
        let Self {
            groups,
            routes,
            comb_levels,
            lanes,
            parallel,
            ..
        } = self;
        let lanes = *lanes;
        let parallel = *parallel;
        // Stable boundary inputs (register / top-level) are valid for the whole
        // cycle: exchange them once up front.
        for route in routes.iter().filter(|route| route.stable) {
            exchange_route(groups, route, lanes);
        }
        // Then settle level by level: exchange each level's combinational inputs
        // (produced by earlier levels) and evaluate the level concurrently.
        for level in comb_levels.iter() {
            for &group in level {
                for route in routes
                    .iter()
                    .filter(|route| !route.stable && route.consumer_group == group)
                {
                    exchange_route(groups, route, lanes);
                }
            }
            eval_level(groups, level, parallel);
        }
    }

    /// One clock cycle, mirroring [`PackedSimulator::tick`]: settle, capture and
    /// commit each group's registers, then settle again.
    pub fn tick(&mut self) {
        self.eval_combinational();
        for group in &mut self.groups {
            group
                .sim
                .tick_from_evaluated_no_post_eval()
                .expect("group tick");
        }
        self.eval_combinational();
    }

    /// Current activity-input values of group `g`, flattened over (signal, lane,
    /// limb) for change detection.
    fn group_activity_values(&self, g: usize) -> Vec<u32> {
        let mut out = Vec::new();
        for &idx in &self.activity_inputs[g] {
            for lane in 0..self.lanes {
                out.extend(self.groups[g].sim.inner().signal_limbs_at(idx, lane));
            }
        }
        out
    }

    /// Like [`tick`](Self::tick) but **activity-skipped**: a group is evaluated
    /// (and its registers captured/committed) only on cycles where one of its
    /// combinational-cone leaf signals (inputs/registers) changed. A skipped
    /// group's registers hold and its combinational outputs are already correct in
    /// storage, so consumers reading them (via boundary exchange) stay correct.
    /// Bit-identical to `tick` for register-cone partitions (all boundary routes
    /// stable); falls back to evaluating everything if combinational cross-group
    /// boundaries exist. Returns `(skipped_groups, total_groups)` for this tick.
    pub fn tick_activity(&mut self) -> (u64, u64) {
        let n = self.groups.len();
        let has_comb_routes = self.routes.iter().any(|r| !r.stable);

        // Phase 1 (next-state): exchange stable boundaries, then settle + capture
        // + commit any group whose cone leaves changed *since that group's last
        // capture settle*. The leaf snapshot is taken HERE (and only here): a
        // group's next-state is a function of its leaves at the clock edge, so a
        // self-updating register (e.g. a counter, whose leaf is its own register)
        // sees its leaf change every cycle and is correctly never skipped.
        for route in self.routes.iter().filter(|r| r.stable) {
            exchange_route(&mut self.groups, route, self.lanes);
        }
        let mut active1 = vec![false; n];
        for g in 0..n {
            let cur = self.group_activity_values(g);
            let changed =
                has_comb_routes || self.activity_snap[g].is_empty() || self.activity_snap[g] != cur;
            if changed {
                active1[g] = true;
                self.groups[g].sim.eval_combinational().expect("group eval");
                self.activity_snap[g] = cur;
            }
        }
        for g in 0..n {
            if active1[g] {
                self.groups[g]
                    .sim
                    .tick_from_evaluated_no_post_eval()
                    .expect("group tick");
            }
        }

        // Phase 2 (observability re-settle): refresh combinational outputs of any
        // group that committed a register this cycle, or that consumes a committed
        // group's output, so post-tick `get_signal` is bit-exact with the oracle.
        // This does NOT touch the next-state snapshot.
        for route in self.routes.iter().filter(|r| r.stable) {
            exchange_route(&mut self.groups, route, self.lanes);
        }
        let mut resettle = active1.clone();
        for route in &self.routes {
            if active1[route.producer_group] {
                resettle[route.consumer_group] = true;
            }
        }
        for g in 0..n {
            if resettle[g] {
                self.groups[g].sim.eval_combinational().expect("group eval");
            }
        }

        let skipped = (0..n).filter(|&g| !active1[g]).count() as u64;
        self.activity_skips += skipped;
        self.activity_total += n as u64;
        (skipped, n as u64)
    }

    pub fn tick_many_activity(&mut self, steps: usize) {
        for _ in 0..steps {
            self.tick_activity();
        }
    }

    /// Cumulative fraction of group-ticks skipped by `tick_activity`.
    pub fn activity_skip_rate(&self) -> f64 {
        if self.activity_total == 0 {
            0.0
        } else {
            self.activity_skips as f64 / self.activity_total as f64
        }
    }

    pub fn tick_many(&mut self, steps: usize) {
        for _ in 0..steps {
            self.tick();
        }
    }
}

/// Verifies and summarizes all blocks in a packed machine program.
pub fn analyze_machine_program(
    program: &PackedMachineProgram,
) -> Result<PackedMachineAnalysis, ErrorReport> {
    let async_reset_comb =
        analyze_machine_block(&program.streams.async_reset_comb, &program.source)?;
    let comb = analyze_machine_block(&program.streams.comb, &program.source)?;
    let tick_next = analyze_machine_block(&program.streams.tick_next, &program.source)?;
    let tick_commit = analyze_machine_block(&program.streams.tick_commit, &program.source)?;
    Ok(PackedMachineAnalysis {
        instr_count: async_reset_comb.instr_count
            + comb.instr_count
            + tick_next.instr_count
            + tick_commit.instr_count,
        effect_count: async_reset_comb.effect_count
            + comb.effect_count
            + tick_next.effect_count
            + tick_commit.effect_count,
        max_packet_width: async_reset_comb
            .max_packet_width
            .max(comb.max_packet_width)
            .max(tick_next.max_packet_width)
            .max(tick_commit.max_packet_width),
        max_live_values: async_reset_comb
            .max_live_values
            .max(comb.max_live_values)
            .max(tick_next.max_live_values)
            .max(tick_commit.max_live_values),
        avg_live_values_x100: average_live_values_x100([
            &async_reset_comb,
            &comb,
            &tick_next,
            &tick_commit,
        ]),
        max_packet_memory_reads: async_reset_comb
            .max_packet_memory_reads
            .max(comb.max_packet_memory_reads)
            .max(tick_next.max_packet_memory_reads)
            .max(tick_commit.max_packet_memory_reads),
        async_reset_comb,
        comb,
        tick_next,
        tick_commit,
    })
}

/// Statically predicts how well the current SIMD CPU execution path matches a packed program.
pub fn analyze_simd_suitability(
    program: &PackedProgram,
) -> Result<SimdSuitabilityReport, ErrorReport> {
    let machine = lower_to_machine_program(program);
    analyze_machine_program(&machine)?;
    let async_reset_comb = analyze_simd_suitability_block(&machine.streams.async_reset_comb);
    let comb = analyze_simd_suitability_block(&machine.streams.comb);
    let tick_next = analyze_simd_suitability_block(&machine.streams.tick_next);
    let tick_commit = analyze_simd_suitability_block(&machine.streams.tick_commit);
    let total = merge_simd_suitability_blocks([async_reset_comb, comb, tick_next, tick_commit]);
    let cost = simd_suitability_cost(&total);
    let score_x100 = simd_suitability_score_x100(cost);
    let fallback_ratio_x100 = simd_fallback_ratio_x100(&total);
    let recommendation = simd_suitability_recommendation(&total, score_x100, fallback_ratio_x100);
    Ok(SimdSuitabilityReport {
        streams: SimdSuitabilityStreamsReport {
            async_reset_comb,
            comb,
            tick_next,
            tick_commit,
        },
        total,
        recommendation,
        score_x100,
        estimated_fast_cost: cost.fast,
        estimated_fallback_cost: cost.fallback,
        estimated_materialization_cost: cost.materialization,
        fallback_ratio_x100,
    })
}

/// Explains which replay backend family the packed program structurally favors.
pub fn analyze_backend_affinity(
    program: &PackedProgram,
) -> Result<BackendAffinityReport, ErrorReport> {
    let machine = lower_to_machine_program(program);
    analyze_machine_program(&machine)?;
    let async_reset_comb = analyze_backend_affinity_block(&machine.streams.async_reset_comb);
    let comb = analyze_backend_affinity_block(&machine.streams.comb);
    let tick_next = analyze_backend_affinity_block(&machine.streams.tick_next);
    let tick_commit = analyze_backend_affinity_block(&machine.streams.tick_commit);
    let total = merge_backend_affinity_blocks([async_reset_comb, comb, tick_next, tick_commit]);
    let recommendation = backend_affinity_recommendation(&total);
    Ok(BackendAffinityReport {
        streams: BackendAffinityStreamsReport {
            async_reset_comb,
            comb,
            tick_next,
            tick_commit,
        },
        total,
        recommendation,
        reasons: backend_affinity_reasons(&total, recommendation),
    })
}

/// Verifies and summarizes one packed machine block.
pub fn analyze_machine_block(
    block: &PackedBlock,
    source: &PackedProgram,
) -> Result<PackedBlockAnalysis, ErrorReport> {
    let mut diagnostics = Vec::new();
    let mut values: HashMap<PackedValueId, PackedValueAnalysis> = HashMap::new();
    let mut packet_analysis = Vec::new();
    let mut instr_count = 0;
    let mut effect_count = 0;
    let mut max_packet_width = 0;
    let mut max_packet_memory_reads = 0;

    for (packet_index, packet) in block.packets.iter().enumerate() {
        let mut class_counts = PackedInstrClassCounts::default();
        instr_count += packet.instrs.len();
        effect_count += packet.effects.len();
        max_packet_width = max_packet_width.max(packet.instrs.len());

        for instr in &packet.instrs {
            validate_machine_instr_refs(instr, source, &mut diagnostics);
            for value in instr_value_deps(&instr.kind) {
                record_machine_value_use(
                    &mut values,
                    &mut diagnostics,
                    value,
                    packet_index,
                    PackedValueUseKind::InstructionInput,
                    false,
                );
            }
            if values.contains_key(&instr.dst) {
                diagnostics.push(Diagnostic::new(
                    "E_SIM_IR_MACHINE_DUP_VALUE",
                    format!("machine value {:?} is defined more than once", instr.dst),
                ));
            } else {
                values.insert(
                    instr.dst,
                    PackedValueAnalysis {
                        ty: instr.ty,
                        defined_packet: packet_index,
                        uses: Vec::new(),
                        last_use_packet: None,
                    },
                );
            }
            class_counts.increment(machine_instr_class(&instr.kind));
        }

        for effect in &packet.effects {
            validate_machine_effect_refs(effect, source, &mut diagnostics);
            for value in effect_value_deps(effect) {
                record_machine_value_use(
                    &mut values,
                    &mut diagnostics,
                    value,
                    packet_index,
                    PackedValueUseKind::EffectInput,
                    true,
                );
            }
        }

        packet_analysis.push(PackedPacketAnalysis {
            instr_count: packet.instrs.len(),
            effect_count: packet.effects.len(),
            class_counts,
        });
        max_packet_memory_reads = max_packet_memory_reads.max(class_counts.memory_read);
    }

    if !diagnostics.is_empty() {
        return Err(ErrorReport::new(diagnostics));
    }

    let live_counts = live_value_counts(&values, block.packets.len());
    let max_live_values = live_counts.iter().copied().max().unwrap_or(0);
    let avg_live_values_x100 = if live_counts.is_empty() {
        0
    } else {
        live_counts.iter().sum::<usize>() * 100 / live_counts.len()
    };
    Ok(PackedBlockAnalysis {
        packets: packet_analysis,
        values,
        instr_count,
        effect_count,
        max_packet_width,
        max_live_values,
        avg_live_values_x100,
        max_packet_memory_reads,
    })
}

fn analyze_simd_suitability_block(block: &PackedBlock) -> SimdSuitabilityBlockReport {
    let cache = PackedBlockCache::new(block);
    let mut report = SimdSuitabilityBlockReport::default();
    for packet in &block.packets {
        report.max_packet_width = report.max_packet_width.max(packet.instrs.len());
        for instr in &packet.instrs {
            report.instr_count += 1;
            if limbs(instr.ty.width) == 1 {
                report.one_limb_instrs += 1;
            } else {
                report.wide_instrs += 1;
            }
            match simd_instr_fallback_reason(instr, &cache) {
                None => {
                    report.fast_instrs += 1;
                    if let Some(path) = simd_instr_fast_path(instr, &cache) {
                        record_simd_fast_path(&mut report.fast_path_profile, path);
                    }
                }
                Some(reason) => {
                    report.fallback_instrs += 1;
                    report.lane_materializations_per_lane += 1;
                    record_replay_simd_fallback_reason(&mut report.fallback_reasons, reason);
                }
            }
            match &instr.kind {
                PackedInstrKind::MemRead { .. } => report.memory_read_instrs += 1,
                PackedInstrKind::Lt { signed: true, .. } => report.signed_lt_instrs += 1,
                PackedInstrKind::Concat(_)
                    if matches!(
                        simd_instr_fallback_reason(instr, &cache),
                        Some(SimdFallbackReason::WideConcat)
                    ) =>
                {
                    report.wide_concat_instrs += 1
                }
                _ => {}
            }
        }
        for effect in &packet.effects {
            if matches!(effect, PackedEffect::MemoryWrite { .. }) {
                report.memory_write_effects += 1;
                report.fast_path_profile.record_memory_write_effect();
            }
        }
    }
    report
}

fn analyze_backend_affinity_block(block: &PackedBlock) -> BackendAffinityBlockReport {
    let cache = PackedBlockCache::new(block);
    let mut report = BackendAffinityBlockReport::default();
    for packet in &block.packets {
        report.max_packet_width = report.max_packet_width.max(packet.instrs.len());
        for instr in &packet.instrs {
            report.instr_count += 1;
            if limbs(instr.ty.width) == 1 {
                report.scalar_fast_one_limb_instrs += 1;
            }
            match simd_instr_fallback_reason(instr, &cache) {
                None => {
                    report.simd_fast_instrs += 1;
                    if let Some(path) = simd_instr_fast_path(instr, &cache) {
                        record_simd_fast_path(&mut report.simd_fast_path_profile, path);
                    }
                }
                Some(reason) => {
                    report.wide_fallback_instrs += 1;
                    record_replay_simd_fallback_reason(&mut report.gpu_hostile_reasons, reason);
                    match reason {
                        SimdFallbackReason::WideConcat => report.concat_pressure_instrs += 1,
                        SimdFallbackReason::SignedLt => report.signed_ops += 1,
                        _ => {}
                    }
                }
            }
            match &instr.kind {
                PackedInstrKind::MemRead { .. } => report.memory_ops += 1,
                PackedInstrKind::Lt { signed: true, .. } => report.signed_ops += 1,
                PackedInstrKind::Concat(_) => report.concat_pressure_instrs += 1,
                _ => {}
            }
        }
        for effect in &packet.effects {
            if matches!(effect, PackedEffect::MemoryWrite { .. }) {
                report.memory_ops += 1;
                report.simd_fast_path_profile.record_memory_write_effect();
            }
        }
    }
    report.estimated_scalar_cost = report.scalar_fast_one_limb_instrs
        + (report.instr_count - report.scalar_fast_one_limb_instrs) * 3
        + report.memory_ops * 4;
    report.estimated_packed_cpu_cost =
        report.instr_count * 2 + report.wide_fallback_instrs * 3 + report.memory_ops * 4;
    report.estimated_simd_cpu_cost =
        simd_fast_path_profile_cost(report.simd_fast_instrs, report.simd_fast_path_profile)
            + report.wide_fallback_instrs * 64;
    report
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimdFastPathKind {
    OneLimbOp,
    TwoLimbOp,
    TwoLimbMulOp,
    TwoLimbMemoryRead,
    TwoLimbMuxOp,
}

fn simd_instr_fast_path(instr: &PackedInstr, cache: &PackedBlockCache) -> Option<SimdFastPathKind> {
    if simd_instr_fallback_reason(instr, cache).is_some() {
        return None;
    }
    match &instr.kind {
        PackedInstrKind::Lit(_) | PackedInstrKind::Signal(_) => None,
        PackedInstrKind::Not(value)
        | PackedInstrKind::Zext(value)
        | PackedInstrKind::Trunc(value)
        | PackedInstrKind::Cast(value) => {
            if cache.value_width(*value) <= 32 && instr.ty.width <= 32 {
                Some(SimdFastPathKind::OneLimbOp)
            } else {
                Some(SimdFastPathKind::TwoLimbOp)
            }
        }
        PackedInstrKind::And(lhs, rhs)
        | PackedInstrKind::Or(lhs, rhs)
        | PackedInstrKind::Xor(lhs, rhs)
        | PackedInstrKind::Add(lhs, rhs)
        | PackedInstrKind::Sub(lhs, rhs) => {
            if cache.value_width(*lhs) <= 32
                && cache.value_width(*rhs) <= 32
                && instr.ty.width <= 32
            {
                Some(SimdFastPathKind::OneLimbOp)
            } else {
                Some(SimdFastPathKind::TwoLimbOp)
            }
        }
        PackedInstrKind::Mul(lhs, rhs) => {
            if cache.value_width(*lhs) <= 32
                && cache.value_width(*rhs) <= 32
                && instr.ty.width <= 32
            {
                Some(SimdFastPathKind::OneLimbOp)
            } else {
                Some(SimdFastPathKind::TwoLimbMulOp)
            }
        }
        PackedInstrKind::Eq(lhs, rhs) | PackedInstrKind::Ne(lhs, rhs) => {
            if cache.value_width(*lhs) <= 32 && cache.value_width(*rhs) <= 32 {
                Some(SimdFastPathKind::OneLimbOp)
            } else {
                Some(SimdFastPathKind::TwoLimbOp)
            }
        }
        PackedInstrKind::Lt { lhs, rhs, .. } => {
            if cache.value_width(*lhs) <= 32
                && cache.value_width(*rhs) <= 32
                && instr.ty.width <= 32
            {
                Some(SimdFastPathKind::OneLimbOp)
            } else {
                Some(SimdFastPathKind::TwoLimbOp)
            }
        }
        PackedInstrKind::Mux {
            cond,
            then_value,
            else_value,
        } => {
            if cache.value_width(*cond) <= 32
                && cache.value_width(*then_value) <= 32
                && cache.value_width(*else_value) <= 32
                && instr.ty.width <= 32
            {
                Some(SimdFastPathKind::OneLimbOp)
            } else {
                Some(SimdFastPathKind::TwoLimbMuxOp)
            }
        }
        PackedInstrKind::Slice { value, lsb } => {
            if cache.value_width(*value) <= 32
                && instr.ty.width <= 32
                && lsb.checked_add(instr.ty.width).is_some_and(|end| end <= 32)
            {
                Some(SimdFastPathKind::OneLimbOp)
            } else {
                Some(SimdFastPathKind::TwoLimbOp)
            }
        }
        PackedInstrKind::Concat(values) => {
            let total_width = values.iter().try_fold(0u32, |total, value| {
                total.checked_add(cache.value_width(*value))
            });
            if instr.ty.width <= 32
                && values.iter().all(|value| cache.value_width(*value) <= 32)
                && total_width.is_some_and(|width| width <= 32)
            {
                Some(SimdFastPathKind::OneLimbOp)
            } else {
                Some(SimdFastPathKind::TwoLimbOp)
            }
        }
        PackedInstrKind::Sext(value) => {
            if cache.value_width(*value) <= 32 && instr.ty.width <= 32 {
                Some(SimdFastPathKind::OneLimbOp)
            } else {
                Some(SimdFastPathKind::TwoLimbOp)
            }
        }
        PackedInstrKind::MemRead { .. } => Some(SimdFastPathKind::TwoLimbMemoryRead),
    }
}

fn record_simd_fast_path(profile: &mut SimdFastPathProfile, path: SimdFastPathKind) {
    match path {
        SimdFastPathKind::OneLimbOp => profile.record_one_limb_op(),
        SimdFastPathKind::TwoLimbOp => profile.record_two_limb_op(),
        SimdFastPathKind::TwoLimbMulOp => profile.record_two_limb_mul_op(),
        SimdFastPathKind::TwoLimbMemoryRead => profile.record_two_limb_memory_read(),
        SimdFastPathKind::TwoLimbMuxOp => profile.record_two_limb_mux_op(),
    }
}

fn simd_instr_fallback_reason(
    instr: &PackedInstr,
    cache: &PackedBlockCache,
) -> Option<SimdFallbackReason> {
    match &instr.kind {
        PackedInstrKind::Lit(_) | PackedInstrKind::Signal(_) => None,
        PackedInstrKind::Not(value) => (cache.value_width(*value) > 64 || instr.ty.width > 64)
            .then_some(SimdFallbackReason::WideOp),
        PackedInstrKind::And(lhs, rhs)
        | PackedInstrKind::Or(lhs, rhs)
        | PackedInstrKind::Xor(lhs, rhs)
        | PackedInstrKind::Add(lhs, rhs)
        | PackedInstrKind::Sub(lhs, rhs) => {
            (cache.value_width(*lhs) > 64 || cache.value_width(*rhs) > 64 || instr.ty.width > 64)
                .then_some(SimdFallbackReason::WideOp)
        }
        PackedInstrKind::Mul(lhs, rhs) => {
            (cache.value_width(*lhs) > 64 || cache.value_width(*rhs) > 64 || instr.ty.width > 64)
                .then_some(SimdFallbackReason::WideOp)
        }
        PackedInstrKind::Eq(lhs, rhs) | PackedInstrKind::Ne(lhs, rhs) => {
            (cache.value_width(*lhs) > 64 || cache.value_width(*rhs) > 64)
                .then_some(SimdFallbackReason::WideOp)
        }
        PackedInstrKind::Lt { lhs, rhs, signed } => {
            if cache.value_width(*lhs) > 64 || cache.value_width(*rhs) > 64 {
                Some(SimdFallbackReason::WideOp)
            } else if *signed && cache.value_width(*lhs) == 0 {
                Some(SimdFallbackReason::SignedLt)
            } else {
                None
            }
        }
        PackedInstrKind::Mux {
            cond,
            then_value,
            else_value,
        } => (cache.value_width(*cond) > 32
            || cache.value_width(*then_value) > 64
            || cache.value_width(*else_value) > 64
            || instr.ty.width > 64)
            .then_some(SimdFallbackReason::WideOp),
        PackedInstrKind::Slice { value, lsb } => (cache.value_width(*value) > 64
            || instr.ty.width > 64
            || !lsb.checked_add(instr.ty.width).is_some_and(|end| end <= 64))
        .then_some(SimdFallbackReason::WideOp),
        PackedInstrKind::Zext(value)
        | PackedInstrKind::Trunc(value)
        | PackedInstrKind::Cast(value) => (cache.value_width(*value) > 64 || instr.ty.width > 64)
            .then_some(SimdFallbackReason::WideOp),
        PackedInstrKind::Sext(value) => (cache.value_width(*value) > 64 || instr.ty.width > 64)
            .then_some(SimdFallbackReason::Sext),
        PackedInstrKind::Concat(values) => {
            let all_narrow = values.iter().all(|value| cache.value_width(*value) <= 64);
            let total_width = values.iter().try_fold(0u32, |total, value| {
                total.checked_add(cache.value_width(*value))
            });
            (!all_narrow || instr.ty.width > 64 || !total_width.is_some_and(|width| width <= 64))
                .then_some(SimdFallbackReason::WideConcat)
        }
        PackedInstrKind::MemRead { addr, .. } => (cache.value_width(*addr) > 64
            || instr.ty.width > 64)
            .then_some(SimdFallbackReason::MemRead),
    }
}

fn record_replay_simd_fallback_reason(
    stats: &mut ReplaySimdFallbackStats,
    reason: SimdFallbackReason,
) {
    match reason {
        SimdFallbackReason::Sext => stats.sext += 1,
        SimdFallbackReason::MemRead => stats.mem_read += 1,
        SimdFallbackReason::WideOp => stats.wide_op += 1,
        SimdFallbackReason::SignedLt => stats.signed_lt += 1,
        SimdFallbackReason::WideConcat => stats.wide_concat += 1,
    }
}

fn merge_simd_suitability_blocks(
    blocks: [SimdSuitabilityBlockReport; 4],
) -> SimdSuitabilityBlockReport {
    let mut total = SimdSuitabilityBlockReport::default();
    for block in blocks {
        total.instr_count += block.instr_count;
        total.fast_instrs += block.fast_instrs;
        total.fallback_instrs += block.fallback_instrs;
        total.lane_materializations_per_lane += block.lane_materializations_per_lane;
        total.fast_path_profile.merge(block.fast_path_profile);
        total.fallback_reasons.sext += block.fallback_reasons.sext;
        total.fallback_reasons.mem_read += block.fallback_reasons.mem_read;
        total.fallback_reasons.wide_op += block.fallback_reasons.wide_op;
        total.fallback_reasons.signed_lt += block.fallback_reasons.signed_lt;
        total.fallback_reasons.wide_concat += block.fallback_reasons.wide_concat;
        total.fallback_reasons.other += block.fallback_reasons.other;
        total.one_limb_instrs += block.one_limb_instrs;
        total.wide_instrs += block.wide_instrs;
        total.memory_read_instrs += block.memory_read_instrs;
        total.memory_write_effects += block.memory_write_effects;
        total.signed_lt_instrs += block.signed_lt_instrs;
        total.wide_concat_instrs += block.wide_concat_instrs;
        total.max_packet_width = total.max_packet_width.max(block.max_packet_width);
    }
    total
}

fn merge_backend_affinity_blocks(
    blocks: [BackendAffinityBlockReport; 4],
) -> BackendAffinityBlockReport {
    let mut total = BackendAffinityBlockReport::default();
    for block in blocks {
        total.instr_count += block.instr_count;
        total.scalar_fast_one_limb_instrs += block.scalar_fast_one_limb_instrs;
        total.simd_fast_instrs += block.simd_fast_instrs;
        total.wide_fallback_instrs += block.wide_fallback_instrs;
        total
            .simd_fast_path_profile
            .merge(block.simd_fast_path_profile);
        total.memory_ops += block.memory_ops;
        total.concat_pressure_instrs += block.concat_pressure_instrs;
        total.signed_ops += block.signed_ops;
        total.max_packet_width = total.max_packet_width.max(block.max_packet_width);
        total.estimated_scalar_cost += block.estimated_scalar_cost;
        total.estimated_packed_cpu_cost += block.estimated_packed_cpu_cost;
        total.estimated_simd_cpu_cost += block.estimated_simd_cpu_cost;
        total.gpu_hostile_reasons.sext += block.gpu_hostile_reasons.sext;
        total.gpu_hostile_reasons.mem_read += block.gpu_hostile_reasons.mem_read;
        total.gpu_hostile_reasons.wide_op += block.gpu_hostile_reasons.wide_op;
        total.gpu_hostile_reasons.signed_lt += block.gpu_hostile_reasons.signed_lt;
        total.gpu_hostile_reasons.wide_concat += block.gpu_hostile_reasons.wide_concat;
        total.gpu_hostile_reasons.other += block.gpu_hostile_reasons.other;
    }
    total
}

fn backend_affinity_recommendation(
    report: &BackendAffinityBlockReport,
) -> BackendAffinityRecommendation {
    if report.memory_ops > 0 {
        return BackendAffinityRecommendation::GpuBlocked;
    }
    if report.instr_count == 0 {
        return BackendAffinityRecommendation::ScalarPreferred;
    }
    let simd_fast_ratio_x100 = report.simd_fast_instrs * 100 / report.instr_count;
    let scalar_fast_ratio_x100 = report.scalar_fast_one_limb_instrs * 100 / report.instr_count;
    let profile = report.simd_fast_path_profile;
    let costly_fast_paths = profile.two_limb_mul_ops
        + profile.two_limb_memory_reads
        + profile.two_limb_mux_ops
        + profile.memory_write_effects;
    let costly_fast_ratio_x100 = if report.simd_fast_instrs == 0 {
        0
    } else {
        costly_fast_paths * 100 / report.simd_fast_instrs
    };
    if simd_fast_ratio_x100 >= 95
        && report.wide_fallback_instrs == 0
        && costly_fast_ratio_x100 <= 25
    {
        BackendAffinityRecommendation::SimdCpuCandidate
    } else if simd_fast_ratio_x100 >= 50 && report.wide_fallback_instrs <= report.simd_fast_instrs {
        BackendAffinityRecommendation::MixedScalarSimdCandidate
    } else if report.max_packet_width >= 64 && scalar_fast_ratio_x100 < 25 {
        BackendAffinityRecommendation::PackedCpuCandidate
    } else {
        BackendAffinityRecommendation::ScalarPreferred
    }
}

fn backend_affinity_reasons(
    report: &BackendAffinityBlockReport,
    recommendation: BackendAffinityRecommendation,
) -> Vec<String> {
    let mut reasons = Vec::new();
    match recommendation {
        BackendAffinityRecommendation::GpuBlocked => {
            reasons.push("memory operations currently block GPU-oriented replay candidates".into());
        }
        BackendAffinityRecommendation::SimdCpuCandidate => {
            reasons.push("nearly all instructions are low-cost SIMD-fast operations".into());
        }
        BackendAffinityRecommendation::MixedScalarSimdCandidate => {
            reasons.push("SIMD-fast coverage is meaningful but fallback pressure remains".into());
        }
        BackendAffinityRecommendation::PackedCpuCandidate => {
            reasons.push("wide packets dominate while scalar one-limb coverage is low".into());
        }
        BackendAffinityRecommendation::ScalarPreferred => {
            reasons.push(
                "scalar execution is expected to avoid fallback/materialization overhead".into(),
            );
        }
    }
    if report.wide_fallback_instrs > 0 {
        reasons.push(format!(
            "{} instructions require wide or fallback handling",
            report.wide_fallback_instrs
        ));
    }
    let profile = report.simd_fast_path_profile;
    let costly_fast_paths = profile.two_limb_mul_ops
        + profile.two_limb_memory_reads
        + profile.two_limb_mux_ops
        + profile.memory_write_effects;
    if costly_fast_paths > 0 {
        reasons.push(format!(
            "{costly_fast_paths} SIMD-fast operations use higher-cost two-limb or memory paths"
        ));
    }
    if report.memory_ops > 0 {
        reasons.push(format!(
            "{} memory operations are present",
            report.memory_ops
        ));
    }
    if report.concat_pressure_instrs > 0 {
        reasons.push(format!(
            "{} concat-heavy instructions increase packing pressure",
            report.concat_pressure_instrs
        ));
    }
    if report.signed_ops > 0 {
        reasons.push(format!(
            "{} signed comparison/extension-sensitive operations are present",
            report.signed_ops
        ));
    }
    reasons
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SimdSuitabilityCost {
    fast: usize,
    fallback: usize,
    materialization: usize,
}

fn simd_suitability_cost(report: &SimdSuitabilityBlockReport) -> SimdSuitabilityCost {
    let reasons = report.fallback_reasons;
    SimdSuitabilityCost {
        fast: simd_fast_path_profile_cost(report.fast_instrs, report.fast_path_profile),
        fallback: reasons.wide_op * 48
            + reasons.mem_read * 64
            + reasons.wide_concat * 32
            + reasons.signed_lt * 24
            + reasons.sext * 16
            + reasons.other * 32,
        materialization: report.lane_materializations_per_lane * 16,
    }
}

fn simd_fast_path_profile_cost(fast_instrs: usize, profile: SimdFastPathProfile) -> usize {
    let profiled_instrs = profile.total_profiled_instrs();
    let unprofiled_fast_instrs = fast_instrs.saturating_sub(profiled_instrs);
    unprofiled_fast_instrs
        + profile.one_limb_ops
        + profile.two_limb_ops * 3
        + profile.two_limb_mul_ops * 5
        + profile.two_limb_memory_reads * 8
        + profile.two_limb_mux_ops * 4
        + profile.memory_write_effects * 8
}

fn simd_suitability_score_x100(cost: SimdSuitabilityCost) -> usize {
    let total_cost = cost.fast + cost.fallback + cost.materialization;
    if total_cost == 0 {
        return 0;
    }
    cost.fast * 10_000 / total_cost
}

fn simd_fallback_ratio_x100(report: &SimdSuitabilityBlockReport) -> usize {
    if report.instr_count == 0 {
        return 0;
    }
    report.fallback_instrs * 100 / report.instr_count
}

fn simd_suitability_recommendation(
    report: &SimdSuitabilityBlockReport,
    score_x100: usize,
    fallback_ratio_x100: usize,
) -> SimdSuitabilityRecommendation {
    if report.memory_read_instrs > 0 || report.memory_write_effects > 0 {
        return SimdSuitabilityRecommendation::GpuCandidateBlocked;
    }
    let profile = report.fast_path_profile;
    let costly_fast_paths = profile.two_limb_mul_ops
        + profile.two_limb_memory_reads
        + profile.two_limb_mux_ops
        + profile.memory_write_effects;
    let costly_fast_ratio_x100 = if report.fast_instrs == 0 {
        0
    } else {
        costly_fast_paths * 100 / report.fast_instrs
    };
    if score_x100 >= 8_500 && fallback_ratio_x100 <= 5 {
        return if costly_fast_ratio_x100 <= 25 {
            SimdSuitabilityRecommendation::SimdCandidate
        } else {
            SimdSuitabilityRecommendation::MixedCandidate
        };
    }
    if score_x100 >= 6_500 && fallback_ratio_x100 <= 15 {
        return SimdSuitabilityRecommendation::MixedCandidate;
    }
    SimdSuitabilityRecommendation::ScalarPreferred
}

fn average_live_values_x100(blocks: [&PackedBlockAnalysis; 4]) -> usize {
    let total_packets = blocks
        .iter()
        .map(|block| block.packets.len())
        .sum::<usize>();
    if total_packets == 0 {
        return 0;
    }
    blocks
        .iter()
        .map(|block| block.avg_live_values_x100 * block.packets.len())
        .sum::<usize>()
        / total_packets
}

/// Applies conservative scheduling optimizations to a packed machine program.
pub fn optimize_machine_program(
    program: &PackedMachineProgram,
    options: PackedScheduleOptions,
) -> Result<PackedMachineProgram, ErrorReport> {
    analyze_machine_program(program)?;
    let optimized = PackedMachineProgram {
        source: program.source.clone(),
        streams: PackedMachineStreams {
            async_reset_comb: optimize_machine_block(
                &program.streams.async_reset_comb,
                &program.source,
                options,
            )?,
            comb: optimize_machine_block(&program.streams.comb, &program.source, options)?,
            tick_next: optimize_machine_block(
                &program.streams.tick_next,
                &program.source,
                options,
            )?,
            tick_commit: optimize_machine_block(
                &program.streams.tick_commit,
                &program.source,
                options,
            )?,
        },
    };
    analyze_machine_program(&optimized)?;
    Ok(optimized)
}

/// Applies conservative scheduling optimizations to one packed machine block.
pub fn optimize_machine_block(
    block: &PackedBlock,
    source: &PackedProgram,
    options: PackedScheduleOptions,
) -> Result<PackedBlock, ErrorReport> {
    analyze_machine_block(block, source)?;
    if options.max_packet_width.is_none()
        && options.max_memory_reads_per_packet.is_none()
        && !options.liveness_priority
    {
        return Ok(block.clone());
    }
    if options.max_packet_width == Some(0) {
        return Err(ErrorReport::new(vec![Diagnostic::new(
            "E_SIM_IR_SCHEDULE_WIDTH",
            "max packet width must be greater than zero",
        )]));
    }
    if options.max_memory_reads_per_packet == Some(0) {
        return Err(ErrorReport::new(vec![Diagnostic::new(
            "E_SIM_IR_SCHEDULE_MEMORY_READS",
            "max memory reads per packet must be greater than zero",
        )]));
    }

    let mut optimized = PackedBlock::default();
    let mut region_instrs = Vec::new();
    for packet in &block.packets {
        region_instrs.extend(packet.instrs.iter().cloned());
        if !packet.effects.is_empty() {
            optimized.packets.extend(schedule_machine_region(
                &region_instrs,
                &packet.effects,
                options,
            ));
            region_instrs.clear();
        } else if packet.instrs.is_empty() && region_instrs.is_empty() {
            optimized.packets.push(packet.clone());
        }
    }
    if !region_instrs.is_empty() {
        optimized
            .packets
            .extend(schedule_machine_region(&region_instrs, &[], options));
    }

    analyze_machine_block(&optimized, source)?;
    Ok(optimized)
}

fn schedule_machine_region(
    instrs: &[PackedInstr],
    effects: &[PackedEffect],
    options: PackedScheduleOptions,
) -> Vec<PackedMachinePacket> {
    if instrs.is_empty() {
        return if effects.is_empty() {
            Vec::new()
        } else {
            vec![PackedMachinePacket {
                instrs: Vec::new(),
                effects: effects.to_vec(),
            }]
        };
    }

    let instr_by_value = instrs
        .iter()
        .enumerate()
        .map(|(index, instr)| (instr.dst, index))
        .collect::<HashMap<_, _>>();
    let mut dependents = vec![Vec::new(); instrs.len()];
    let mut remaining_deps = vec![0usize; instrs.len()];
    for (index, instr) in instrs.iter().enumerate() {
        for dep in instr_value_deps(&instr.kind) {
            if let Some(dep_index) = instr_by_value.get(&dep).copied() {
                remaining_deps[index] += 1;
                dependents[dep_index].push(index);
            }
        }
    }
    let priorities = schedule_priorities(instrs, &instr_by_value, effects);

    let mut ready = remaining_deps
        .iter()
        .enumerate()
        .filter_map(|(index, deps)| (*deps == 0).then_some(index))
        .collect::<Vec<_>>();
    let mut queued = vec![false; instrs.len()];
    for index in &ready {
        queued[*index] = true;
    }
    let mut scheduled = vec![false; instrs.len()];
    let mut scheduled_count = 0;
    let mut packets = Vec::new();
    let max_packet_width = options.max_packet_width.unwrap_or(usize::MAX);

    while scheduled_count < instrs.len() {
        let mut selected = Vec::new();
        let mut packet_instrs = Vec::new();
        let mut packet_memory_reads = 0usize;
        while packet_instrs.len() < max_packet_width {
            let Some(ready_index) = select_ready_position(
                &ready,
                instrs,
                &scheduled,
                &priorities,
                packet_memory_reads,
                options,
            ) else {
                break;
            };
            let index = ready.remove(ready_index);
            if scheduled[index] {
                continue;
            }
            queued[index] = false;
            scheduled[index] = true;
            scheduled_count += 1;
            if is_memory_read(&instrs[index]) {
                packet_memory_reads += 1;
            }
            selected.push(index);
            packet_instrs.push(instrs[index].clone());
        }

        if packet_instrs.is_empty() {
            break;
        }

        for index in selected {
            for dependent in &dependents[index] {
                remaining_deps[*dependent] -= 1;
            }
        }
        for (index, deps) in remaining_deps.iter().enumerate() {
            if *deps == 0 && !scheduled[index] && !queued[index] {
                ready.push(index);
                queued[index] = true;
            }
        }

        packets.push(PackedMachinePacket {
            instrs: packet_instrs,
            effects: Vec::new(),
        });
    }

    if scheduled_count != instrs.len() {
        let chunk_width = options
            .max_packet_width
            .unwrap_or_else(|| instrs.len().max(1));
        return instrs
            .chunks(chunk_width)
            .enumerate()
            .map(|(chunk_index, chunk)| PackedMachinePacket {
                instrs: chunk.to_vec(),
                effects: if chunk_index + 1 == instrs.len().div_ceil(chunk_width) {
                    effects.to_vec()
                } else {
                    Vec::new()
                },
            })
            .collect();
    }

    packets
        .last_mut()
        .expect("non-empty instruction region creates at least one packet")
        .effects
        .extend_from_slice(effects);
    packets
}

fn memory_read_allowed(
    instr: &PackedInstr,
    packet_memory_reads: usize,
    max_memory_reads_per_packet: Option<usize>,
) -> bool {
    !is_memory_read(instr)
        || max_memory_reads_per_packet.is_none_or(|cap| packet_memory_reads < cap)
}

fn is_memory_read(instr: &PackedInstr) -> bool {
    matches!(instr.kind, PackedInstrKind::MemRead { .. })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SchedulePriority {
    final_use_operands: usize,
    latest_operand_last_use: usize,
}

fn select_ready_position(
    ready: &[usize],
    instrs: &[PackedInstr],
    scheduled: &[bool],
    priorities: &[SchedulePriority],
    packet_memory_reads: usize,
    options: PackedScheduleOptions,
) -> Option<usize> {
    let eligible = |index: usize| {
        !scheduled[index]
            && memory_read_allowed(
                &instrs[index],
                packet_memory_reads,
                options.max_memory_reads_per_packet,
            )
    };
    if !options.liveness_priority {
        return ready.iter().position(|index| eligible(*index));
    }
    ready
        .iter()
        .enumerate()
        .filter(|(_, index)| eligible(**index))
        .max_by(|(_, lhs), (_, rhs)| compare_schedule_priority(**lhs, **rhs, priorities))
        .map(|(position, _)| position)
}

fn compare_schedule_priority(
    lhs: usize,
    rhs: usize,
    priorities: &[SchedulePriority],
) -> std::cmp::Ordering {
    priorities[lhs]
        .final_use_operands
        .cmp(&priorities[rhs].final_use_operands)
        .then_with(|| {
            priorities[rhs]
                .latest_operand_last_use
                .cmp(&priorities[lhs].latest_operand_last_use)
        })
        .then_with(|| rhs.cmp(&lhs))
}

fn schedule_priorities(
    instrs: &[PackedInstr],
    instr_by_value: &HashMap<PackedValueId, usize>,
    effects: &[PackedEffect],
) -> Vec<SchedulePriority> {
    let mut last_use_by_value = HashMap::new();
    for (index, instr) in instrs.iter().enumerate() {
        for dep in instr_value_deps(&instr.kind) {
            if instr_by_value.contains_key(&dep) {
                last_use_by_value
                    .entry(dep)
                    .and_modify(|last: &mut usize| *last = (*last).max(index))
                    .or_insert(index);
            }
        }
    }
    for effect in effects {
        for dep in effect_value_deps(effect) {
            if instr_by_value.contains_key(&dep) {
                last_use_by_value
                    .entry(dep)
                    .and_modify(|last: &mut usize| *last = (*last).max(instrs.len()))
                    .or_insert(instrs.len());
            }
        }
    }

    instrs
        .iter()
        .enumerate()
        .map(|(index, instr)| {
            let deps = instr_value_deps(&instr.kind)
                .into_iter()
                .filter(|dep| instr_by_value.contains_key(dep))
                .collect::<Vec<_>>();
            let final_use_operands = deps
                .iter()
                .filter(|dep| {
                    last_use_by_value
                        .get(dep)
                        .is_some_and(|last| *last == index)
                })
                .count();
            let latest_operand_last_use = deps
                .iter()
                .filter_map(|dep| last_use_by_value.get(dep).copied())
                .max()
                .unwrap_or(usize::MAX);
            SchedulePriority {
                final_use_operands,
                latest_operand_last_use,
            }
        })
        .collect()
}

fn lower_packets_to_block(packets: &[PackedPacket]) -> PackedBlock {
    let mut block = PackedBlock::default();
    let mut next_value = 0;
    for packet in packets {
        let mut lower = MachinePacketLowerer::new(next_value);
        let mut effects = Vec::new();
        for op in &packet.ops {
            match op {
                PackedOp::Assign { dst, expr } => {
                    let value = lower.lower_expr(expr);
                    effects.push(PackedEffect::StoreSignal { dst: *dst, value });
                }
                PackedOp::CaptureReg { dst, next, reset } => {
                    let value = lower.lower_expr(next);
                    effects.push(PackedEffect::CaptureReg {
                        dst: *dst,
                        value,
                        reset: reset.clone(),
                    });
                }
                PackedOp::MemoryWrite {
                    memory,
                    enable,
                    addr,
                    data,
                } => {
                    let enable = lower.lower_expr(enable);
                    let addr = lower.lower_expr(addr);
                    let data = lower.lower_expr(data);
                    effects.push(PackedEffect::MemoryWrite {
                        memory: *memory,
                        enable,
                        addr,
                        data,
                    });
                }
            }
        }
        next_value = lower.next_value;
        let mut lowered_packets = schedule_machine_instrs(&lower.instrs);
        if lowered_packets.is_empty() {
            lowered_packets.push(PackedMachinePacket::default());
        }
        lowered_packets
            .last_mut()
            .expect("created at least one packet")
            .effects
            .extend(effects);
        block.packets.extend(lowered_packets);
    }
    block
}

#[derive(Default)]
struct MachinePacketLowerer {
    next_value: usize,
    memo: HashMap<PackedExpr, PackedValueId>,
    instrs: Vec<PackedInstr>,
}

impl MachinePacketLowerer {
    fn new(next_value: usize) -> Self {
        Self {
            next_value,
            memo: HashMap::new(),
            instrs: Vec::new(),
        }
    }

    fn lower_expr(&mut self, expr: &PackedExpr) -> PackedValueId {
        if let Some(value) = self.memo.get(expr).copied() {
            return value;
        }
        let kind = match &expr.kind {
            PackedExprKind::Lit(value) => PackedInstrKind::Lit(value.clone()),
            PackedExprKind::Signal(signal) => PackedInstrKind::Signal(*signal),
            PackedExprKind::Not(inner) => PackedInstrKind::Not(self.lower_expr(inner)),
            PackedExprKind::And(lhs, rhs) => {
                PackedInstrKind::And(self.lower_expr(lhs), self.lower_expr(rhs))
            }
            PackedExprKind::Or(lhs, rhs) => {
                PackedInstrKind::Or(self.lower_expr(lhs), self.lower_expr(rhs))
            }
            PackedExprKind::Xor(lhs, rhs) => {
                PackedInstrKind::Xor(self.lower_expr(lhs), self.lower_expr(rhs))
            }
            PackedExprKind::Add(lhs, rhs) => {
                PackedInstrKind::Add(self.lower_expr(lhs), self.lower_expr(rhs))
            }
            PackedExprKind::Sub(lhs, rhs) => {
                PackedInstrKind::Sub(self.lower_expr(lhs), self.lower_expr(rhs))
            }
            PackedExprKind::Mul(lhs, rhs) => {
                PackedInstrKind::Mul(self.lower_expr(lhs), self.lower_expr(rhs))
            }
            PackedExprKind::Eq(lhs, rhs) => {
                PackedInstrKind::Eq(self.lower_expr(lhs), self.lower_expr(rhs))
            }
            PackedExprKind::Ne(lhs, rhs) => {
                PackedInstrKind::Ne(self.lower_expr(lhs), self.lower_expr(rhs))
            }
            PackedExprKind::Lt { lhs, rhs, signed } => PackedInstrKind::Lt {
                lhs: self.lower_expr(lhs),
                rhs: self.lower_expr(rhs),
                signed: *signed,
            },
            PackedExprKind::Mux {
                cond,
                then_expr,
                else_expr,
            } => PackedInstrKind::Mux {
                cond: self.lower_expr(cond),
                then_value: self.lower_expr(then_expr),
                else_value: self.lower_expr(else_expr),
            },
            PackedExprKind::Slice { expr, lsb } => PackedInstrKind::Slice {
                value: self.lower_expr(expr),
                lsb: *lsb,
            },
            PackedExprKind::Zext(inner) => PackedInstrKind::Zext(self.lower_expr(inner)),
            PackedExprKind::Sext(inner) => PackedInstrKind::Sext(self.lower_expr(inner)),
            PackedExprKind::Trunc(inner) => PackedInstrKind::Trunc(self.lower_expr(inner)),
            PackedExprKind::Cast(inner) => PackedInstrKind::Cast(self.lower_expr(inner)),
            PackedExprKind::Concat(parts) => {
                PackedInstrKind::Concat(parts.iter().map(|part| self.lower_expr(part)).collect())
            }
            PackedExprKind::MemRead { memory, addr } => PackedInstrKind::MemRead {
                memory: *memory,
                addr: self.lower_expr(addr),
            },
        };
        let value = PackedValueId(self.next_value);
        self.next_value += 1;
        self.instrs.push(PackedInstr {
            dst: value,
            ty: expr.ty,
            kind,
        });
        self.memo.insert(expr.clone(), value);
        value
    }
}

fn schedule_machine_instrs(instrs: &[PackedInstr]) -> Vec<PackedMachinePacket> {
    let instr_by_value = instrs
        .iter()
        .enumerate()
        .map(|(index, instr)| (instr.dst, index))
        .collect::<HashMap<_, _>>();
    let mut memo = HashMap::new();
    let mut packets: Vec<PackedMachinePacket> = Vec::new();
    for (index, instr) in instrs.iter().enumerate() {
        let level = machine_instr_level(index, instrs, &instr_by_value, &mut memo);
        if packets.len() <= level {
            packets.resize_with(level + 1, PackedMachinePacket::default);
        }
        packets[level].instrs.push(instr.clone());
    }
    packets
        .into_iter()
        .filter(|packet| !packet.instrs.is_empty() || !packet.effects.is_empty())
        .collect()
}

fn machine_instr_level(
    index: usize,
    instrs: &[PackedInstr],
    instr_by_value: &HashMap<PackedValueId, usize>,
    memo: &mut HashMap<usize, usize>,
) -> usize {
    if let Some(level) = memo.get(&index).copied() {
        return level;
    }
    let level = instr_value_deps(&instrs[index].kind)
        .into_iter()
        .filter_map(|value| instr_by_value.get(&value).copied())
        .map(|dep| machine_instr_level(dep, instrs, instr_by_value, memo) + 1)
        .max()
        .unwrap_or(0);
    memo.insert(index, level);
    level
}

fn instr_value_deps(kind: &PackedInstrKind) -> Vec<PackedValueId> {
    match kind {
        PackedInstrKind::Lit(_) | PackedInstrKind::Signal(_) => Vec::new(),
        PackedInstrKind::Not(value)
        | PackedInstrKind::Zext(value)
        | PackedInstrKind::Sext(value)
        | PackedInstrKind::Trunc(value)
        | PackedInstrKind::Cast(value)
        | PackedInstrKind::Slice { value, .. } => vec![*value],
        PackedInstrKind::And(lhs, rhs)
        | PackedInstrKind::Or(lhs, rhs)
        | PackedInstrKind::Xor(lhs, rhs)
        | PackedInstrKind::Add(lhs, rhs)
        | PackedInstrKind::Sub(lhs, rhs)
        | PackedInstrKind::Mul(lhs, rhs)
        | PackedInstrKind::Eq(lhs, rhs)
        | PackedInstrKind::Ne(lhs, rhs) => vec![*lhs, *rhs],
        PackedInstrKind::Lt { lhs, rhs, .. } => vec![*lhs, *rhs],
        PackedInstrKind::Mux {
            cond,
            then_value,
            else_value,
        } => vec![*cond, *then_value, *else_value],
        PackedInstrKind::Concat(values) => values.clone(),
        PackedInstrKind::MemRead { addr, .. } => vec![*addr],
    }
}

fn effect_value_deps(effect: &PackedEffect) -> Vec<PackedValueId> {
    match effect {
        PackedEffect::StoreSignal { value, .. } | PackedEffect::CaptureReg { value, .. } => {
            vec![*value]
        }
        PackedEffect::MemoryWrite {
            enable, addr, data, ..
        } => vec![*enable, *addr, *data],
    }
}

fn record_machine_value_use(
    values: &mut HashMap<PackedValueId, PackedValueAnalysis>,
    diagnostics: &mut Vec<Diagnostic>,
    value: PackedValueId,
    packet: usize,
    kind: PackedValueUseKind,
    allow_current_packet: bool,
) {
    let Some(analysis) = values.get_mut(&value) else {
        diagnostics.push(Diagnostic::new(
            "E_SIM_IR_MACHINE_USE_BEFORE_DEF",
            format!("machine value {value:?} is used before it is defined"),
        ));
        return;
    };
    if analysis.defined_packet > packet
        || (!allow_current_packet && analysis.defined_packet == packet)
    {
        diagnostics.push(Diagnostic::new(
            "E_SIM_IR_MACHINE_USE_BEFORE_DEF",
            format!(
                "machine value {value:?} is used in packet {packet} before an earlier packet defines it"
            ),
        ));
        return;
    }
    analysis.uses.push(PackedValueUse { packet, kind });
    analysis.last_use_packet = Some(
        analysis
            .last_use_packet
            .map_or(packet, |last| last.max(packet)),
    );
}

fn validate_machine_instr_refs(
    instr: &PackedInstr,
    source: &PackedProgram,
    diagnostics: &mut Vec<Diagnostic>,
) {
    match &instr.kind {
        PackedInstrKind::Signal(signal) => {
            validate_signal_index(*signal, source, diagnostics, "machine signal load")
        }
        PackedInstrKind::MemRead { memory, .. } => {
            validate_memory_index(*memory, source, diagnostics, "machine memory read")
        }
        _ => {}
    }
}

fn validate_machine_effect_refs(
    effect: &PackedEffect,
    source: &PackedProgram,
    diagnostics: &mut Vec<Diagnostic>,
) {
    match effect {
        PackedEffect::StoreSignal { dst, .. } => {
            validate_signal_index(*dst, source, diagnostics, "machine store")
        }
        PackedEffect::CaptureReg { dst, reset, .. } => {
            validate_signal_index(*dst, source, diagnostics, "machine register capture");
            if let Some(reset) = reset {
                validate_signal_index(reset.signal, source, diagnostics, "machine reset condition");
            }
        }
        PackedEffect::MemoryWrite { memory, .. } => {
            validate_memory_index(*memory, source, diagnostics, "machine memory write")
        }
    }
}

fn validate_signal_index(
    signal: usize,
    source: &PackedProgram,
    diagnostics: &mut Vec<Diagnostic>,
    context: &str,
) {
    if signal >= source.signals.len() {
        diagnostics.push(Diagnostic::new(
            "E_SIM_IR_MACHINE_SIGNAL",
            format!(
                "{context} references signal index {signal}, but packed program has {} signals",
                source.signals.len()
            ),
        ));
    }
}

fn validate_memory_index(
    memory: usize,
    source: &PackedProgram,
    diagnostics: &mut Vec<Diagnostic>,
    context: &str,
) {
    if memory >= source.memories.len() {
        diagnostics.push(Diagnostic::new(
            "E_SIM_IR_MACHINE_MEMORY",
            format!(
                "{context} references memory index {memory}, but packed program has {} memories",
                source.memories.len()
            ),
        ));
    }
}

fn machine_instr_class(kind: &PackedInstrKind) -> PackedInstrClass {
    match kind {
        PackedInstrKind::Lit(_) => PackedInstrClass::Literal,
        PackedInstrKind::Signal(_) => PackedInstrClass::SignalLoad,
        PackedInstrKind::Not(_)
        | PackedInstrKind::And(_, _)
        | PackedInstrKind::Or(_, _)
        | PackedInstrKind::Xor(_, _) => PackedInstrClass::Bitwise,
        PackedInstrKind::Add(_, _) | PackedInstrKind::Sub(_, _) | PackedInstrKind::Mul(_, _) => {
            PackedInstrClass::Arithmetic
        }
        PackedInstrKind::Eq(_, _) | PackedInstrKind::Ne(_, _) | PackedInstrKind::Lt { .. } => {
            PackedInstrClass::Compare
        }
        PackedInstrKind::Mux { .. } => PackedInstrClass::Select,
        PackedInstrKind::Slice { .. }
        | PackedInstrKind::Zext(_)
        | PackedInstrKind::Sext(_)
        | PackedInstrKind::Trunc(_)
        | PackedInstrKind::Cast(_)
        | PackedInstrKind::Concat(_) => PackedInstrClass::BitMovement,
        PackedInstrKind::MemRead { .. } => PackedInstrClass::MemoryRead,
    }
}

fn live_value_counts(
    values: &HashMap<PackedValueId, PackedValueAnalysis>,
    packet_count: usize,
) -> Vec<usize> {
    (0..packet_count)
        .map(|packet| {
            values
                .values()
                .filter(|value| {
                    value
                        .last_use_packet
                        .is_some_and(|last| value.defined_packet <= packet && last > packet)
                })
                .count()
        })
        .collect()
}

/// CPU interpreter for a [`PackedMachineProgram`] across multiple lanes.
///
/// Inputs and outputs are exposed through source `Signal` handles, while the
/// internal storage is lane-packed into 32-bit limbs.
#[derive(Clone, Debug)]
pub struct PackedSimulator {
    program: PackedProgram,
    machine: PackedMachineProgram,
    execution: PackedExecutionCache,
    workspaces: PackedExecutionWorkspaces,
    lanes: usize,
    values: Vec<u32>,
    memories: Vec<u32>,
    register_captures: Vec<PackedRegisterCapture>,
    register_capture_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PackedRegisterCapture {
    signal: usize,
    layout: PackedValueLayout,
    values: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
struct PackedExecutionWorkspaces {
    async_reset_comb: PackedBlockState,
    comb: PackedBlockState,
    tick_next: PackedBlockState,
    tick_commit: PackedBlockState,
}

#[derive(Clone, Debug, Default)]
struct PackedBlockState {
    lanes: usize,
    slots: Vec<SimdValueSlot>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SimBackendKind {
    Scalar,
    PackedCpu,
    SimdCpu,
    JitCpu,
}

impl SimBackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::PackedCpu => "packed-cpu",
            Self::SimdCpu => "simd-cpu",
            Self::JitCpu => "jit-cpu",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimBackendOptions {
    pub kind: SimBackendKind,
    pub lanes: usize,
}

impl SimBackendOptions {
    pub fn scalar() -> Self {
        Self {
            kind: SimBackendKind::Scalar,
            lanes: 1,
        }
    }

    pub fn packed_cpu(lanes: usize) -> Self {
        Self {
            kind: SimBackendKind::PackedCpu,
            lanes,
        }
    }

    pub fn simd_cpu(lanes: usize) -> Self {
        Self {
            kind: SimBackendKind::SimdCpu,
            lanes,
        }
    }

    pub fn jit_cpu(lanes: usize) -> Self {
        Self {
            kind: SimBackendKind::JitCpu,
            lanes,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReplayLaneMode {
    Replicated,
    Independent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReplayCheckMode {
    Lane0Fast,
    AllLanes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayOptions {
    pub lane_mode: ReplayLaneMode,
    pub check_mode: ReplayCheckMode,
    pub max_mismatches: usize,
}

impl Default for ReplayOptions {
    fn default() -> Self {
        Self {
            lane_mode: ReplayLaneMode::Replicated,
            check_mode: ReplayCheckMode::Lane0Fast,
            max_mismatches: 16,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedTraceReplayPlan {
    steps: Vec<EncodedTraceReplayStep>,
    independent_lanes: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodedTraceReplayWorkload {
    pub steps: usize,
    pub lanes: usize,
    pub input_ops: usize,
    pub check_ops: usize,
    pub one_limb_input_ops: usize,
    pub one_limb_check_ops: usize,
    pub one_limb_input_batches: usize,
    pub one_limb_check_batches: usize,
    pub generic_input_ops: usize,
    pub generic_check_ops: usize,
    pub estimated_lane_work_units: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedTraceReplayStep {
    inputs: Vec<EncodedTraceInput>,
    checks: Vec<EncodedTraceCheck>,
    independent_input_ops: Vec<EncodedTraceInputOp>,
    independent_check_ops: Vec<EncodedTraceCheckOp>,
    independent_input_batches: Vec<EncodedTraceOneLimbInputBatch>,
    independent_check_batches: Vec<EncodedTraceOneLimbCheckBatch>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedTraceInput {
    signal: Signal,
    signal_index: usize,
    limbs: Vec<u32>,
    lane_limbs: Option<Vec<Vec<u32>>>,
    lane_limbs_flat: Option<Vec<u32>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedTraceCheck {
    check_index: usize,
    signal: Signal,
    signal_index: usize,
    expected: u128,
    expected_limbs: Vec<u32>,
    lane_expected: Option<Vec<u128>>,
    lane_expected_limbs: Option<Vec<Vec<u32>>>,
    lane_expected_limbs_flat: Option<Vec<u32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EncodedTraceInputOp {
    Generic {
        input_index: usize,
    },
    OneLimb {
        input_index: usize,
        layout: PackedValueLayout,
    },
    OneLimbBatch {
        batch_index: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EncodedTraceCheckOp {
    Generic {
        check_index: usize,
    },
    OneLimb {
        check_index: usize,
        layout: PackedValueLayout,
    },
    OneLimbBatch {
        batch_index: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct EncodedTraceOneLimbInputBatch {
    start_offset: usize,
    values: Vec<u32>,
    signals: Vec<EncodedTraceOneLimbInputBatchSignal>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EncodedTraceOneLimbInputBatchSignal {
    input_index: usize,
    layout: PackedValueLayout,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct EncodedTraceOneLimbCheckBatch {
    start_offset: usize,
    values: Vec<u32>,
    signals: Vec<EncodedTraceOneLimbCheckBatchSignal>,
    all_full_limb: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EncodedTraceOneLimbCheckBatchSignal {
    check_index: usize,
    layout: PackedValueLayout,
}

impl EncodedTraceReplayPlan {
    pub fn new(
        program: &PackedProgram,
        steps: impl IntoIterator<Item = EncodedTraceReplayStep>,
    ) -> Result<Self, ErrorReport> {
        let mut steps = steps.into_iter().collect::<Vec<_>>();
        let mut independent_lanes = None;
        for step in &steps {
            for input in &step.inputs {
                validate_encoded_signal_index(program, input.signal, input.signal_index)?;
                let layout = program.signals[input.signal_index].layout;
                if input.limbs.len() != layout.limbs {
                    return Err(ErrorReport::new(vec![Diagnostic::new(
                        "E_SIM_IR_REPLAY_INPUT_WIDTH",
                        format!(
                            "replay input {:?} has {} limbs, expected {}",
                            input.signal.id,
                            input.limbs.len(),
                            layout.limbs
                        ),
                    )]));
                }
                if let Some(lane_limbs) = &input.lane_limbs {
                    validate_replay_lane_count(
                        &mut independent_lanes,
                        lane_limbs.len(),
                        "input",
                        input.signal,
                    )?;
                    for value in lane_limbs {
                        if value.len() != layout.limbs {
                            return Err(ErrorReport::new(vec![Diagnostic::new(
                                "E_SIM_IR_REPLAY_INPUT_WIDTH",
                                format!(
                                    "independent replay input {:?} has {} limbs, expected {}",
                                    input.signal.id,
                                    value.len(),
                                    layout.limbs
                                ),
                            )]));
                        }
                    }
                }
                if let Some(lane_limbs_flat) = &input.lane_limbs_flat {
                    if lane_limbs_flat.len() % layout.limbs != 0 {
                        return Err(ErrorReport::new(vec![Diagnostic::new(
                            "E_SIM_IR_REPLAY_INPUT_WIDTH",
                            format!(
                                "independent replay input {:?} flat limb count {} is not divisible by {}",
                                input.signal.id,
                                lane_limbs_flat.len(),
                                layout.limbs
                            ),
                        )]));
                    }
                    validate_replay_lane_count(
                        &mut independent_lanes,
                        lane_limbs_flat.len() / layout.limbs,
                        "input",
                        input.signal,
                    )?;
                }
                if input.lane_limbs.is_none() && independent_lanes.is_some() {
                    return Err(ErrorReport::new(vec![Diagnostic::new(
                        "E_SIM_IR_REPLAY_LANES",
                        format!(
                            "independent replay input {:?} is missing lane data",
                            input.signal.id
                        ),
                    )]));
                }
            }
            for check in &step.checks {
                validate_encoded_signal_index(program, check.signal, check.signal_index)?;
                let layout = program.signals[check.signal_index].layout;
                if check.expected_limbs.len() != layout.limbs {
                    return Err(ErrorReport::new(vec![Diagnostic::new(
                        "E_SIM_IR_REPLAY_CHECK_WIDTH",
                        format!(
                            "replay check {:?} has {} limbs, expected {}",
                            check.signal.id,
                            check.expected_limbs.len(),
                            layout.limbs
                        ),
                    )]));
                }
                if let Some(lane_expected) = &check.lane_expected {
                    validate_replay_lane_count(
                        &mut independent_lanes,
                        lane_expected.len(),
                        "check",
                        check.signal,
                    )?;
                }
                if let Some(lane_expected_limbs) = &check.lane_expected_limbs {
                    validate_replay_lane_count(
                        &mut independent_lanes,
                        lane_expected_limbs.len(),
                        "check",
                        check.signal,
                    )?;
                    for expected in lane_expected_limbs {
                        if expected.len() != layout.limbs {
                            return Err(ErrorReport::new(vec![Diagnostic::new(
                                "E_SIM_IR_REPLAY_CHECK_WIDTH",
                                format!(
                                    "independent replay check {:?} has {} limbs, expected {}",
                                    check.signal.id,
                                    expected.len(),
                                    layout.limbs
                                ),
                            )]));
                        }
                    }
                }
                if let Some(lane_expected_limbs_flat) = &check.lane_expected_limbs_flat {
                    if lane_expected_limbs_flat.len() % layout.limbs != 0 {
                        return Err(ErrorReport::new(vec![Diagnostic::new(
                            "E_SIM_IR_REPLAY_CHECK_WIDTH",
                            format!(
                                "independent replay check {:?} flat limb count {} is not divisible by {}",
                                check.signal.id,
                                lane_expected_limbs_flat.len(),
                                layout.limbs
                            ),
                        )]));
                    }
                    validate_replay_lane_count(
                        &mut independent_lanes,
                        lane_expected_limbs_flat.len() / layout.limbs,
                        "check",
                        check.signal,
                    )?;
                }
                if independent_lanes.is_some()
                    && (check.lane_expected.is_none()
                        || check.lane_expected_limbs.is_none()
                        || check.lane_expected_limbs_flat.is_none())
                {
                    return Err(ErrorReport::new(vec![Diagnostic::new(
                        "E_SIM_IR_REPLAY_LANES",
                        format!(
                            "independent replay check {:?} is missing lane data",
                            check.signal.id
                        ),
                    )]));
                }
            }
        }
        for step in &mut steps {
            populate_encoded_replay_ops(program, step);
        }
        Ok(Self {
            steps,
            independent_lanes,
        })
    }

    pub fn from_signal_steps(
        program: &PackedProgram,
        steps: impl IntoIterator<Item = (Vec<(Signal, u128)>, Vec<(usize, Signal, u128)>)>,
    ) -> Result<Self, ErrorReport> {
        let mut encoded_steps = Vec::new();
        for (inputs, checks) in steps {
            let mut encoded_inputs = Vec::with_capacity(inputs.len());
            for (signal, value) in inputs {
                let signal_index = replay_signal_index(program, signal)?;
                let layout = program.signals[signal_index].layout;
                encoded_inputs.push(EncodedTraceInput {
                    signal,
                    signal_index,
                    limbs: encode_u128_limbs(value, layout.ty),
                    lane_limbs: None,
                    lane_limbs_flat: None,
                });
            }

            let mut encoded_checks = Vec::with_capacity(checks.len());
            for (check_index, signal, expected) in checks {
                let signal_index = replay_signal_index(program, signal)?;
                let layout = program.signals[signal_index].layout;
                encoded_checks.push(EncodedTraceCheck {
                    check_index,
                    signal,
                    signal_index,
                    expected: fit_u128(expected, layout.ty),
                    expected_limbs: encode_u128_limbs(expected, layout.ty),
                    lane_expected: None,
                    lane_expected_limbs: None,
                    lane_expected_limbs_flat: None,
                });
            }
            encoded_steps.push(EncodedTraceReplayStep {
                inputs: encoded_inputs,
                checks: encoded_checks,
                independent_input_ops: Vec::new(),
                independent_check_ops: Vec::new(),
                independent_input_batches: Vec::new(),
                independent_check_batches: Vec::new(),
            });
        }
        Self::new(program, encoded_steps)
    }

    pub fn from_independent_lane_steps(
        program: &PackedProgram,
        lanes: usize,
        steps: impl IntoIterator<Item = (Vec<(Signal, Vec<u128>)>, Vec<(usize, Signal, Vec<u128>)>)>,
    ) -> Result<Self, ErrorReport> {
        if lanes == 0 {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_REPLAY_LANES",
                "independent replay requires at least one lane",
            )]));
        }
        let mut encoded_steps = Vec::new();
        for (inputs, checks) in steps {
            let mut encoded_inputs = Vec::with_capacity(inputs.len());
            for (signal, values) in inputs {
                if values.len() != lanes {
                    return Err(ErrorReport::new(vec![Diagnostic::new(
                        "E_SIM_IR_REPLAY_LANES",
                        format!(
                            "independent replay input {:?} has {} lanes, expected {lanes}",
                            signal.id,
                            values.len()
                        ),
                    )]));
                }
                let signal_index = replay_signal_index(program, signal)?;
                let layout = program.signals[signal_index].layout;
                let lane_limbs = values
                    .iter()
                    .copied()
                    .map(|value| encode_u128_limbs(value, layout.ty))
                    .collect::<Vec<_>>();
                let lane_limbs_flat = flatten_replay_lanes(&lane_limbs);
                encoded_inputs.push(EncodedTraceInput {
                    signal,
                    signal_index,
                    limbs: encode_u128_limbs(values[0], layout.ty),
                    lane_limbs: Some(lane_limbs),
                    lane_limbs_flat: Some(lane_limbs_flat),
                });
            }

            let mut encoded_checks = Vec::with_capacity(checks.len());
            for (check_index, signal, expected_values) in checks {
                if expected_values.len() != lanes {
                    return Err(ErrorReport::new(vec![Diagnostic::new(
                        "E_SIM_IR_REPLAY_LANES",
                        format!(
                            "independent replay check {:?} has {} lanes, expected {lanes}",
                            signal.id,
                            expected_values.len()
                        ),
                    )]));
                }
                let signal_index = replay_signal_index(program, signal)?;
                let layout = program.signals[signal_index].layout;
                let lane_expected = expected_values
                    .iter()
                    .copied()
                    .map(|value| fit_u128(value, layout.ty))
                    .collect::<Vec<_>>();
                let lane_expected_limbs = expected_values
                    .iter()
                    .copied()
                    .map(|value| encode_u128_limbs(value, layout.ty))
                    .collect::<Vec<_>>();
                encoded_checks.push(EncodedTraceCheck {
                    check_index,
                    signal,
                    signal_index,
                    expected: lane_expected[0],
                    expected_limbs: lane_expected_limbs[0].clone(),
                    lane_expected: Some(lane_expected),
                    lane_expected_limbs_flat: Some(flatten_replay_lanes(&lane_expected_limbs)),
                    lane_expected_limbs: Some(lane_expected_limbs),
                });
            }
            encoded_steps.push(EncodedTraceReplayStep {
                inputs: encoded_inputs,
                checks: encoded_checks,
                independent_input_ops: Vec::new(),
                independent_check_ops: Vec::new(),
                independent_input_batches: Vec::new(),
                independent_check_batches: Vec::new(),
            });
        }
        let mut plan = Self::new(program, encoded_steps)?;
        plan.independent_lanes = Some(lanes);
        Ok(plan)
    }

    pub fn steps(&self) -> &[EncodedTraceReplayStep] {
        &self.steps
    }

    pub fn independent_lanes(&self) -> Option<usize> {
        self.independent_lanes
    }

    pub fn workload(&self) -> EncodedTraceReplayWorkload {
        let lanes = self.independent_lanes.unwrap_or(1);
        let mut workload = EncodedTraceReplayWorkload {
            steps: self.steps.len(),
            lanes,
            ..EncodedTraceReplayWorkload::default()
        };
        for step in &self.steps {
            workload.input_ops += step.independent_input_ops.len();
            workload.check_ops += step.independent_check_ops.len();
            for op in &step.independent_input_ops {
                match op {
                    EncodedTraceInputOp::Generic { .. } => workload.generic_input_ops += 1,
                    EncodedTraceInputOp::OneLimb { .. } => workload.one_limb_input_ops += 1,
                    EncodedTraceInputOp::OneLimbBatch { .. } => {
                        workload.one_limb_input_batches += 1
                    }
                }
            }
            for op in &step.independent_check_ops {
                match op {
                    EncodedTraceCheckOp::Generic { .. } => workload.generic_check_ops += 1,
                    EncodedTraceCheckOp::OneLimb { .. } => workload.one_limb_check_ops += 1,
                    EncodedTraceCheckOp::OneLimbBatch { .. } => {
                        workload.one_limb_check_batches += 1
                    }
                }
            }
        }
        workload.estimated_lane_work_units = lanes
            * (workload.one_limb_input_ops
                + workload.one_limb_check_ops
                + workload.generic_input_ops * 4
                + workload.generic_check_ops * 4)
            + lanes
                * self
                    .steps
                    .iter()
                    .map(|step| {
                        step.independent_input_batches
                            .iter()
                            .map(|batch| batch.signals.len())
                            .sum::<usize>()
                            + step
                                .independent_check_batches
                                .iter()
                                .map(|batch| batch.signals.len())
                                .sum::<usize>()
                    })
                    .sum::<usize>();
        workload
    }

    pub fn slice_lanes(&self, start: usize, lanes: usize) -> Result<Self, ErrorReport> {
        if lanes == 0 {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_REPLAY_LANES",
                "lane slices must include at least one lane",
            )]));
        }
        if let Some(total_lanes) = self.independent_lanes {
            let end = start.saturating_add(lanes);
            if start > total_lanes || end > total_lanes {
                return Err(ErrorReport::new(vec![Diagnostic::new(
                    "E_SIM_IR_REPLAY_LANES",
                    format!("lane slice [{start}, {end}) is out of range for {total_lanes} lanes"),
                )]));
            }
        }
        let mut steps = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            let inputs = step
                .inputs
                .iter()
                .map(|input| {
                    let lane_limbs = input
                        .lane_limbs
                        .as_ref()
                        .map(|values| slice_replay_lanes(values, start, lanes))
                        .transpose()?;
                    Ok(EncodedTraceInput {
                        signal: input.signal,
                        signal_index: input.signal_index,
                        limbs: lane_limbs
                            .as_ref()
                            .and_then(|values| values.first().cloned())
                            .unwrap_or_else(|| input.limbs.clone()),
                        lane_limbs_flat: lane_limbs
                            .as_ref()
                            .map(|values| flatten_replay_lanes(values)),
                        lane_limbs,
                    })
                })
                .collect::<Result<Vec<_>, ErrorReport>>()?;
            let checks = step
                .checks
                .iter()
                .map(|check| {
                    let lane_expected = check
                        .lane_expected
                        .as_ref()
                        .map(|values| slice_replay_lanes(values, start, lanes))
                        .transpose()?;
                    let lane_expected_limbs = check
                        .lane_expected_limbs
                        .as_ref()
                        .map(|values| slice_replay_lanes(values, start, lanes))
                        .transpose()?;
                    Ok(EncodedTraceCheck {
                        check_index: check.check_index,
                        signal: check.signal,
                        signal_index: check.signal_index,
                        expected: lane_expected
                            .as_ref()
                            .and_then(|values| values.first().copied())
                            .unwrap_or(check.expected),
                        expected_limbs: lane_expected_limbs
                            .as_ref()
                            .and_then(|values| values.first().cloned())
                            .unwrap_or_else(|| check.expected_limbs.clone()),
                        lane_expected,
                        lane_expected_limbs_flat: lane_expected_limbs
                            .as_ref()
                            .map(|values| flatten_replay_lanes(values)),
                        lane_expected_limbs,
                    })
                })
                .collect::<Result<Vec<_>, ErrorReport>>()?;
            let input_layouts = encoded_input_layouts_from_ops(step);
            let check_layouts = encoded_check_layouts_from_ops(step);
            let mut sliced_step = EncodedTraceReplayStep {
                inputs,
                checks,
                independent_input_ops: Vec::new(),
                independent_check_ops: Vec::new(),
                independent_input_batches: Vec::new(),
                independent_check_batches: Vec::new(),
            };
            populate_encoded_replay_ops_from_layouts(
                &mut sliced_step,
                &input_layouts,
                &check_layouts,
            );
            steps.push(sliced_step);
        }
        Ok(Self {
            steps,
            independent_lanes: self.independent_lanes.map(|_| lanes),
        })
    }
}

fn populate_encoded_replay_ops(program: &PackedProgram, step: &mut EncodedTraceReplayStep) {
    let input_layouts = step
        .inputs
        .iter()
        .map(|input| program.signals[input.signal_index].layout)
        .collect::<Vec<_>>();
    let check_layouts = step
        .checks
        .iter()
        .map(|check| program.signals[check.signal_index].layout)
        .collect::<Vec<_>>();
    populate_encoded_replay_ops_from_layouts(step, &input_layouts, &check_layouts);
}

fn populate_encoded_replay_ops_from_layouts(
    step: &mut EncodedTraceReplayStep,
    input_layouts: &[PackedValueLayout],
    check_layouts: &[PackedValueLayout],
) {
    step.independent_input_batches.clear();
    step.independent_check_batches.clear();
    step.independent_input_ops = build_encoded_input_ops(step, input_layouts);
    step.independent_check_ops = build_encoded_check_ops(step, check_layouts);
}

fn build_encoded_input_ops(
    step: &mut EncodedTraceReplayStep,
    layouts: &[PackedValueLayout],
) -> Vec<EncodedTraceInputOp> {
    let mut ops = Vec::new();
    let mut index = 0;
    while index < step.inputs.len() {
        let layout = layouts[index];
        if layout.limbs != 1 || step.inputs[index].lane_limbs_flat.is_none() {
            ops.push(EncodedTraceInputOp::Generic { input_index: index });
            index += 1;
            continue;
        }

        let start = index;
        let mut end = index + 1;
        let mut next_offset = layout.offset + 1;
        while end < step.inputs.len() {
            let candidate = layouts[end];
            if candidate.limbs != 1
                || candidate.offset != next_offset
                || step.inputs[end].lane_limbs_flat.is_none()
            {
                break;
            }
            next_offset += 1;
            end += 1;
        }

        if end - start == 1 {
            ops.push(EncodedTraceInputOp::OneLimb {
                input_index: start,
                layout,
            });
            index = end;
            continue;
        }

        let mut values = Vec::new();
        let mut signals = Vec::with_capacity(end - start);
        for input_index in start..end {
            let values_for_signal = step.inputs[input_index]
                .lane_limbs_flat
                .as_ref()
                .expect("one-limb input batch requires lane data");
            values.extend_from_slice(values_for_signal);
            signals.push(EncodedTraceOneLimbInputBatchSignal {
                input_index,
                layout: layouts[input_index],
            });
        }
        let batch_index = step.independent_input_batches.len();
        step.independent_input_batches
            .push(EncodedTraceOneLimbInputBatch {
                start_offset: layout.offset,
                values,
                signals,
            });
        ops.push(EncodedTraceInputOp::OneLimbBatch { batch_index });
        index = end;
    }
    ops
}

fn build_encoded_check_ops(
    step: &mut EncodedTraceReplayStep,
    layouts: &[PackedValueLayout],
) -> Vec<EncodedTraceCheckOp> {
    let mut ops = Vec::new();
    let mut index = 0;
    while index < step.checks.len() {
        let layout = layouts[index];
        if layout.limbs != 1 || step.checks[index].lane_expected_limbs_flat.is_none() {
            ops.push(EncodedTraceCheckOp::Generic { check_index: index });
            index += 1;
            continue;
        }

        let start = index;
        let mut end = index + 1;
        let mut next_offset = layout.offset + 1;
        while end < step.checks.len() {
            let candidate = layouts[end];
            if candidate.limbs != 1
                || candidate.offset != next_offset
                || step.checks[end].lane_expected_limbs_flat.is_none()
            {
                break;
            }
            next_offset += 1;
            end += 1;
        }

        if end - start == 1 {
            ops.push(EncodedTraceCheckOp::OneLimb {
                check_index: start,
                layout,
            });
            index = end;
            continue;
        }

        let mut values = Vec::new();
        let mut signals = Vec::with_capacity(end - start);
        let mut all_full_limb = true;
        for check_index in start..end {
            let values_for_signal = step.checks[check_index]
                .lane_expected_limbs_flat
                .as_ref()
                .expect("one-limb check batch requires lane data");
            values.extend_from_slice(values_for_signal);
            let layout = layouts[check_index];
            all_full_limb &= layout.width == 32;
            signals.push(EncodedTraceOneLimbCheckBatchSignal {
                check_index,
                layout,
            });
        }
        let batch_index = step.independent_check_batches.len();
        step.independent_check_batches
            .push(EncodedTraceOneLimbCheckBatch {
                start_offset: layout.offset,
                values,
                signals,
                all_full_limb,
            });
        ops.push(EncodedTraceCheckOp::OneLimbBatch { batch_index });
        index = end;
    }
    ops
}

fn encoded_input_layouts_from_ops(step: &EncodedTraceReplayStep) -> Vec<PackedValueLayout> {
    let mut layouts = vec![
        PackedValueLayout {
            width: 0,
            ty: BitType::new(0, Signedness::Unsigned),
            offset: 0,
            limbs: 0,
        };
        step.inputs.len()
    ];
    for op in &step.independent_input_ops {
        match *op {
            EncodedTraceInputOp::Generic { input_index } => {
                layouts[input_index].limbs = 2;
            }
            EncodedTraceInputOp::OneLimb {
                input_index,
                layout,
            } => {
                layouts[input_index] = layout;
            }
            EncodedTraceInputOp::OneLimbBatch { batch_index } => {
                for signal in &step.independent_input_batches[batch_index].signals {
                    layouts[signal.input_index] = signal.layout;
                }
            }
        }
    }
    layouts
}

fn encoded_check_layouts_from_ops(step: &EncodedTraceReplayStep) -> Vec<PackedValueLayout> {
    let mut layouts = vec![
        PackedValueLayout {
            width: 0,
            ty: BitType::new(0, Signedness::Unsigned),
            offset: 0,
            limbs: 0,
        };
        step.checks.len()
    ];
    for op in &step.independent_check_ops {
        match *op {
            EncodedTraceCheckOp::Generic { check_index } => {
                layouts[check_index].limbs = 2;
            }
            EncodedTraceCheckOp::OneLimb {
                check_index,
                layout,
            } => {
                layouts[check_index] = layout;
            }
            EncodedTraceCheckOp::OneLimbBatch { batch_index } => {
                for signal in &step.independent_check_batches[batch_index].signals {
                    layouts[signal.check_index] = signal.layout;
                }
            }
        }
    }
    layouts
}
fn validate_replay_lane_count(
    expected: &mut Option<usize>,
    actual: usize,
    kind: &str,
    signal: Signal,
) -> Result<(), ErrorReport> {
    if actual == 0 {
        return Err(ErrorReport::new(vec![Diagnostic::new(
            "E_SIM_IR_REPLAY_LANES",
            format!("independent replay {kind} {:?} has zero lanes", signal.id),
        )]));
    }
    match *expected {
        Some(expected) if expected != actual => Err(ErrorReport::new(vec![Diagnostic::new(
            "E_SIM_IR_REPLAY_LANES",
            format!(
                "independent replay {kind} {:?} has {actual} lanes, expected {expected}",
                signal.id
            ),
        )])),
        Some(_) => Ok(()),
        None => {
            *expected = Some(actual);
            Ok(())
        }
    }
}

fn flatten_replay_lanes(values: &[Vec<u32>]) -> Vec<u32> {
    let limbs = values.first().map_or(0, Vec::len);
    let mut flat = Vec::with_capacity(values.len() * limbs);
    for value in values {
        flat.extend_from_slice(value);
    }
    flat
}

fn slice_replay_lanes<T: Clone>(
    values: &[T],
    start: usize,
    lanes: usize,
) -> Result<Vec<T>, ErrorReport> {
    let end = start.saturating_add(lanes);
    if start > values.len() || end > values.len() {
        return Err(ErrorReport::new(vec![Diagnostic::new(
            "E_SIM_IR_REPLAY_LANES",
            format!(
                "lane slice [{start}, {end}) is out of range for {} lanes",
                values.len()
            ),
        )]));
    }
    Ok(values[start..end].to_vec())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayMismatch {
    pub step: usize,
    pub check_index: usize,
    pub expected: u128,
    pub actual: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayTimingReport {
    pub input_ns: u128,
    pub eval_ns: u128,
    pub compare_ns: u128,
    pub tick_ns: u128,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplaySimdStats {
    pub fast_instrs: usize,
    pub fallback_instrs: usize,
    pub state_reuses: usize,
    pub lane_materializations: usize,
    #[serde(default)]
    pub fast_paths: ReplaySimdFastPathStats,
    pub fallback_reasons: ReplaySimdFallbackStats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplaySimdFastPathStats {
    pub one_limb_ops: usize,
    pub two_limb_ops: usize,
    #[serde(default)]
    pub native_two_limb_ops: usize,
    pub two_limb_mul_ops: usize,
    pub two_limb_memory_reads: usize,
    pub two_limb_mux_ops: usize,
    pub memory_write_effects: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplaySimdFallbackStats {
    pub sext: usize,
    pub mem_read: usize,
    pub wide_op: usize,
    pub signed_lt: usize,
    pub wide_concat: usize,
    pub other: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayReport {
    pub mismatch_count: usize,
    pub mismatches: Vec<ReplayMismatch>,
    pub timing: ReplayTimingReport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub simd_stats: Option<ReplaySimdStats>,
}

impl ReplayReport {
    fn record_mismatch(
        &mut self,
        options: ReplayOptions,
        step: usize,
        check_index: usize,
        expected: u128,
        actual: u128,
        lane: Option<usize>,
    ) {
        self.mismatch_count += 1;
        if self.mismatches.len() < options.max_mismatches {
            self.mismatches.push(ReplayMismatch {
                step,
                check_index,
                expected,
                actual,
                lane,
            });
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThreadedReplayWorkerOptions {
    pub backend: SimBackendKind,
    pub lanes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThreadedReplayOptions {
    pub workers: Vec<ThreadedReplayWorkerOptions>,
    pub max_mismatches: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThreadedReplayInitialState {
    pub signals: Vec<(Signal, u128)>,
    pub memories: Vec<(Signal, Vec<u128>)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThreadedReplayWorkerReport {
    pub worker_index: usize,
    pub backend: SimBackendKind,
    pub start_lane: usize,
    pub lanes: usize,
    pub replay: ReplayReport,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThreadedReplayReport {
    pub total_lanes: usize,
    pub workers: Vec<ThreadedReplayWorkerReport>,
    pub replay: ReplayReport,
    pub lane_cycles_per_sec: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayAutotuneOptions {
    pub warmup_steps: usize,
    pub max_workers: usize,
    pub candidates: Vec<ThreadedReplayOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub simd_suitability: Option<SimdSuitabilityReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_affinity: Option<BackendAffinityReport>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplayAutotuneCandidateReport {
    pub candidate_index: usize,
    pub available: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub layout: ThreadedReplayOptions,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report: Option<ThreadedReplayReport>,
    pub lane_cycles_per_sec: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayAutotunePrunedCandidateReport {
    pub candidate_index: usize,
    pub layout: ThreadedReplayOptions,
    pub reason: ReplayAutotunePruneReason,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReplayAutotunePruneReason {
    ScalarPreferred,
    MixedOnly,
    GpuCandidateBlocked,
    PackedCpuPreferred,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplayAutotuneReport {
    pub selected_candidate: usize,
    pub candidates: Vec<ReplayAutotuneCandidateReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pruned_candidates: Vec<ReplayAutotunePrunedCandidateReport>,
    pub selected: ThreadedReplayReport,
}

pub fn replay_trace_threaded(
    program: &PackedProgram,
    plan: &EncodedTraceReplayPlan,
    options: &ThreadedReplayOptions,
) -> Result<ThreadedReplayReport, ErrorReport> {
    replay_trace_threaded_with_initial_state(
        program,
        plan,
        options,
        &ThreadedReplayInitialState::default(),
    )
}

pub fn replay_trace_threaded_with_initial_state(
    program: &PackedProgram,
    plan: &EncodedTraceReplayPlan,
    options: &ThreadedReplayOptions,
    initial_state: &ThreadedReplayInitialState,
) -> Result<ThreadedReplayReport, ErrorReport> {
    let mut runner =
        ThreadedReplayRunner::new_with_initial_state(program, plan, options, initial_state)?;
    runner.replay()
}

pub struct ThreadedReplayRunner {
    workers: Vec<ThreadedReplayRunnerWorker>,
    total_lanes: usize,
    steps: usize,
    max_mismatches: usize,
}

enum ThreadedReplayRunnerWorker {
    Scalar {
        worker_index: usize,
        start_lane: usize,
        lanes: usize,
        sim: SingleLaneMachineSimulator,
        snapshot: PackedSimulatorStorage,
        plan: Arc<SingleLaneCompiledReplayPlan>,
    },
    Backend {
        worker_index: usize,
        backend: SimBackendKind,
        start_lane: usize,
        lanes: usize,
        sim: SimBackendInstance,
        snapshot: PackedSimulatorStorage,
        plan: EncodedTraceReplayPlan,
    },
}

impl ThreadedReplayRunner {
    pub fn new(
        program: &PackedProgram,
        plan: &EncodedTraceReplayPlan,
        options: &ThreadedReplayOptions,
    ) -> Result<Self, ErrorReport> {
        Self::new_with_initial_state(
            program,
            plan,
            options,
            &ThreadedReplayInitialState::default(),
        )
    }

    pub fn new_with_initial_state(
        program: &PackedProgram,
        plan: &EncodedTraceReplayPlan,
        options: &ThreadedReplayOptions,
        initial_state: &ThreadedReplayInitialState,
    ) -> Result<Self, ErrorReport> {
        let (total_lanes, ranges) = validate_threaded_replay_layout(plan, options)?;
        let scalar_compiled_plan = if options
            .workers
            .iter()
            .any(|worker| worker.backend == SimBackendKind::Scalar)
        {
            Some(Arc::new(SingleLaneCompiledReplayPlan::new(program, plan)?))
        } else {
            None
        };
        let mut workers = Vec::with_capacity(options.workers.len());
        for (worker_index, (worker, start_lane)) in options
            .workers
            .iter()
            .cloned()
            .zip(ranges.into_iter())
            .enumerate()
        {
            if worker.backend == SimBackendKind::Scalar {
                let mut sim = SingleLaneMachineSimulator::new(program.clone())?;
                initialize_scalar_threaded_replay_backend(&mut sim, initial_state)?;
                let snapshot = sim.snapshot_storage();
                workers.push(ThreadedReplayRunnerWorker::Scalar {
                    worker_index,
                    start_lane,
                    lanes: worker.lanes,
                    sim,
                    snapshot,
                    plan: Arc::clone(
                        scalar_compiled_plan
                            .as_ref()
                            .expect("scalar compiled replay plan is available"),
                    ),
                });
            } else {
                let lane_plan = plan.slice_lanes(start_lane, worker.lanes)?;
                let mut sim = SimBackendInstance::new(
                    program.clone(),
                    SimBackendOptions {
                        kind: worker.backend,
                        lanes: worker.lanes,
                    },
                )?;
                initialize_threaded_replay_backend(&mut sim, initial_state)?;
                let snapshot = sim.snapshot_storage();
                workers.push(ThreadedReplayRunnerWorker::Backend {
                    worker_index,
                    backend: worker.backend,
                    start_lane,
                    lanes: worker.lanes,
                    sim,
                    snapshot,
                    plan: lane_plan,
                });
            }
        }
        Ok(Self {
            workers,
            total_lanes,
            steps: plan.steps.len(),
            max_mismatches: options.max_mismatches,
        })
    }

    pub fn replay(&mut self) -> Result<ThreadedReplayReport, ErrorReport> {
        let started = Instant::now();
        let max_mismatches = self.max_mismatches;
        let worker_results = std::thread::scope(|scope| {
            let handles = self
                .workers
                .iter_mut()
                .map(|worker| scope.spawn(move || worker.replay(max_mismatches)))
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| match handle.join() {
                    Ok(result) => result,
                    Err(_) => Err(ErrorReport::new(vec![Diagnostic::new(
                        "E_SIM_IR_THREADED_REPLAY_PANIC",
                        "threaded replay worker panicked",
                    )])),
                })
                .collect::<Vec<_>>()
        });

        let mut workers = Vec::with_capacity(worker_results.len());
        for result in worker_results {
            workers.push(result?);
        }
        let elapsed_ns = started.elapsed().as_nanos();
        let replay = merge_threaded_replay_reports(&workers, self.max_mismatches);
        Ok(ThreadedReplayReport {
            total_lanes: self.total_lanes,
            workers,
            replay,
            lane_cycles_per_sec: lane_cycles_per_sec(self.total_lanes, self.steps, elapsed_ns),
        })
    }
}

impl ThreadedReplayRunnerWorker {
    fn replay(&mut self, max_mismatches: usize) -> Result<ThreadedReplayWorkerReport, ErrorReport> {
        match self {
            Self::Scalar {
                worker_index,
                start_lane,
                lanes,
                sim,
                snapshot,
                plan,
            } => {
                let mut merged = ReplayReport::default();
                for local_lane in 0..*lanes {
                    sim.restore_storage(snapshot)?;
                    let report = sim.replay_compiled_independent_lane_trace(
                        plan,
                        *start_lane + local_lane,
                        max_mismatches,
                    )?;
                    merge_replay_report_with_lane_offset(&mut merged, report, 0, max_mismatches);
                }
                Ok(ThreadedReplayWorkerReport {
                    worker_index: *worker_index,
                    backend: SimBackendKind::Scalar,
                    start_lane: *start_lane,
                    lanes: *lanes,
                    replay: merged,
                })
            }
            Self::Backend {
                worker_index,
                backend,
                start_lane,
                lanes,
                sim,
                snapshot,
                plan,
            } => {
                sim.restore_storage(snapshot)?;
                let report = sim.replay_trace(
                    plan,
                    ReplayOptions {
                        lane_mode: ReplayLaneMode::Independent,
                        check_mode: ReplayCheckMode::AllLanes,
                        max_mismatches,
                    },
                )?;
                Ok(ThreadedReplayWorkerReport {
                    worker_index: *worker_index,
                    backend: *backend,
                    start_lane: *start_lane,
                    lanes: *lanes,
                    replay: offset_replay_report_lanes(report, *start_lane, max_mismatches),
                })
            }
        }
    }
}

fn validate_threaded_replay_layout(
    plan: &EncodedTraceReplayPlan,
    options: &ThreadedReplayOptions,
) -> Result<(usize, Vec<usize>), ErrorReport> {
    if options.workers.is_empty() {
        return Err(ErrorReport::new(vec![Diagnostic::new(
            "E_SIM_IR_THREADED_REPLAY_WORKERS",
            "threaded replay requires at least one worker",
        )]));
    }
    let total_lanes = options.workers.iter().try_fold(0usize, |total, worker| {
        if worker.lanes == 0 {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_THREADED_REPLAY_WORKERS",
                "threaded replay workers must own at least one lane",
            )]));
        }
        total.checked_add(worker.lanes).ok_or_else(|| {
            ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_THREADED_REPLAY_WORKERS",
                "threaded replay lane count overflowed",
            )])
        })
    })?;
    if plan.independent_lanes() != Some(total_lanes) {
        return Err(ErrorReport::new(vec![Diagnostic::new(
            "E_SIM_IR_THREADED_REPLAY_LANES",
            match plan.independent_lanes() {
                Some(plan_lanes) => format!(
                    "threaded replay workers cover {total_lanes} lanes, but replay plan has {plan_lanes} independent lanes"
                ),
                None => "threaded replay requires an independent-lane replay plan".to_string(),
            },
        )]));
    }
    let mut ranges = Vec::with_capacity(options.workers.len());
    let mut start_lane = 0usize;
    for worker in &options.workers {
        ranges.push(start_lane);
        start_lane += worker.lanes;
    }
    Ok((total_lanes, ranges))
}

fn initialize_threaded_replay_backend(
    sim: &mut SimBackendInstance,
    initial_state: &ThreadedReplayInitialState,
) -> Result<(), ErrorReport> {
    if !initial_state.signals.is_empty() {
        sim.set_signals_replicated(&initial_state.signals)?;
    }
    for (memory, words) in &initial_state.memories {
        sim.set_memory_replicated(*memory, words)?;
    }
    if initial_state.signals.is_empty() && initial_state.memories.is_empty() {
        return Ok(());
    }
    sim.eval_combinational()
}

fn initialize_scalar_threaded_replay_backend(
    sim: &mut SingleLaneMachineSimulator,
    initial_state: &ThreadedReplayInitialState,
) -> Result<(), ErrorReport> {
    if !initial_state.signals.is_empty() {
        sim.set_signals(&initial_state.signals)?;
    }
    for (memory, words) in &initial_state.memories {
        sim.set_memory_replicated(*memory, words)?;
    }
    if initial_state.signals.is_empty() && initial_state.memories.is_empty() {
        return Ok(());
    }
    sim.eval_combinational();
    Ok(())
}

fn merge_threaded_replay_reports(
    workers: &[ThreadedReplayWorkerReport],
    max_mismatches: usize,
) -> ReplayReport {
    let mut merged = ReplayReport::default();
    for worker in workers {
        merge_replay_report_with_lane_offset(&mut merged, worker.replay.clone(), 0, max_mismatches);
    }
    merged
}

fn merge_replay_report_with_lane_offset(
    merged: &mut ReplayReport,
    report: ReplayReport,
    lane_offset: usize,
    max_mismatches: usize,
) {
    merged.mismatch_count += report.mismatch_count;
    merged.timing.input_ns += report.timing.input_ns;
    merged.timing.eval_ns += report.timing.eval_ns;
    merged.timing.compare_ns += report.timing.compare_ns;
    merged.timing.tick_ns += report.timing.tick_ns;
    merge_simd_stats(&mut merged.simd_stats, report.simd_stats);
    for mut mismatch in report.mismatches {
        if merged.mismatches.len() >= max_mismatches {
            break;
        }
        mismatch.lane = Some(mismatch.lane.unwrap_or(0) + lane_offset);
        merged.mismatches.push(mismatch);
    }
}

fn offset_replay_report_lanes(
    report: ReplayReport,
    lane_offset: usize,
    max_mismatches: usize,
) -> ReplayReport {
    let mut out = ReplayReport {
        mismatch_count: report.mismatch_count,
        timing: report.timing,
        simd_stats: report.simd_stats,
        mismatches: Vec::new(),
    };
    for mut mismatch in report.mismatches {
        if out.mismatches.len() >= max_mismatches {
            break;
        }
        mismatch.lane = Some(mismatch.lane.unwrap_or(0) + lane_offset);
        out.mismatches.push(mismatch);
    }
    out
}

fn merge_simd_stats(dst: &mut Option<ReplaySimdStats>, src: Option<ReplaySimdStats>) {
    let Some(src) = src else {
        return;
    };
    let dst = dst.get_or_insert_with(ReplaySimdStats::default);
    dst.fast_instrs += src.fast_instrs;
    dst.fallback_instrs += src.fallback_instrs;
    dst.state_reuses += src.state_reuses;
    dst.lane_materializations += src.lane_materializations;
    dst.fast_paths.one_limb_ops += src.fast_paths.one_limb_ops;
    dst.fast_paths.two_limb_ops += src.fast_paths.two_limb_ops;
    dst.fast_paths.native_two_limb_ops += src.fast_paths.native_two_limb_ops;
    dst.fast_paths.two_limb_mul_ops += src.fast_paths.two_limb_mul_ops;
    dst.fast_paths.two_limb_memory_reads += src.fast_paths.two_limb_memory_reads;
    dst.fast_paths.two_limb_mux_ops += src.fast_paths.two_limb_mux_ops;
    dst.fast_paths.memory_write_effects += src.fast_paths.memory_write_effects;
    dst.fallback_reasons.sext += src.fallback_reasons.sext;
    dst.fallback_reasons.mem_read += src.fallback_reasons.mem_read;
    dst.fallback_reasons.wide_op += src.fallback_reasons.wide_op;
    dst.fallback_reasons.signed_lt += src.fallback_reasons.signed_lt;
    dst.fallback_reasons.wide_concat += src.fallback_reasons.wide_concat;
    dst.fallback_reasons.other += src.fallback_reasons.other;
}

fn lane_cycles_per_sec(lanes: usize, steps: usize, elapsed_ns: u128) -> f64 {
    if elapsed_ns == 0 {
        return 0.0;
    }
    (lanes as f64 * steps as f64) * 1_000_000_000.0 / elapsed_ns as f64
}

pub fn replay_trace_autotune(
    program: &PackedProgram,
    plan: &EncodedTraceReplayPlan,
    total_lanes: usize,
    options: ReplayAutotuneOptions,
) -> Result<ReplayAutotuneReport, ErrorReport> {
    replay_trace_autotune_with_initial_state(
        program,
        plan,
        total_lanes,
        options,
        &ThreadedReplayInitialState::default(),
    )
}

pub fn replay_trace_autotune_with_initial_state(
    program: &PackedProgram,
    plan: &EncodedTraceReplayPlan,
    total_lanes: usize,
    options: ReplayAutotuneOptions,
    initial_state: &ThreadedReplayInitialState,
) -> Result<ReplayAutotuneReport, ErrorReport> {
    if total_lanes == 0 {
        return Err(ErrorReport::new(vec![Diagnostic::new(
            "E_SIM_IR_AUTOTUNE_LANES",
            "autotune replay requires at least one lane",
        )]));
    }
    if plan.independent_lanes() != Some(total_lanes) {
        return Err(ErrorReport::new(vec![Diagnostic::new(
            "E_SIM_IR_AUTOTUNE_LANES",
            match plan.independent_lanes() {
                Some(plan_lanes) => format!(
                    "autotune requested {total_lanes} lanes, but replay plan has {plan_lanes} independent lanes"
                ),
                None => "autotune replay requires an independent-lane replay plan".to_string(),
            },
        )]));
    }
    let candidate_plan = if options.candidates.is_empty() {
        build_replay_autotune_candidate_set_with_affinity(
            total_lanes,
            options.max_workers,
            options.simd_suitability.as_ref(),
            options.backend_affinity.as_ref(),
        )
    } else {
        ReplayAutotuneCandidateSet {
            candidates: options.candidates,
            pruned_candidates: Vec::new(),
        }
    };
    let candidates = candidate_plan.candidates;
    let pruned_candidates = candidate_plan.pruned_candidates;
    if candidates.is_empty() {
        return Err(ErrorReport::new(vec![Diagnostic::new(
            "E_SIM_IR_AUTOTUNE_CANDIDATES",
            "autotune replay requires at least one candidate",
        )]));
    }
    let warmup_steps = options.warmup_steps.max(1).min(plan.steps.len().max(1));
    let warmup_plan = plan.prefix_steps(warmup_steps);
    let mut candidate_reports = Vec::with_capacity(candidates.len());
    let mut selected_candidate = None;
    let mut selected_score = f64::NEG_INFINITY;

    for (candidate_index, candidate) in candidates.into_iter().enumerate() {
        match replay_trace_threaded_with_initial_state(
            program,
            &warmup_plan,
            &candidate,
            initial_state,
        ) {
            Ok(report) if report.replay.mismatch_count == 0 => {
                let score = report.lane_cycles_per_sec;
                if score > selected_score {
                    selected_score = score;
                    selected_candidate = Some((candidate_index, candidate.clone()));
                }
                candidate_reports.push(ReplayAutotuneCandidateReport {
                    candidate_index,
                    available: true,
                    error: None,
                    layout: candidate,
                    lane_cycles_per_sec: score,
                    report: Some(report),
                });
            }
            Ok(report) => {
                candidate_reports.push(ReplayAutotuneCandidateReport {
                    candidate_index,
                    available: false,
                    error: Some(format!(
                        "candidate produced {} mismatches",
                        report.replay.mismatch_count
                    )),
                    layout: candidate,
                    lane_cycles_per_sec: report.lane_cycles_per_sec,
                    report: Some(report),
                });
            }
            Err(err) => {
                candidate_reports.push(ReplayAutotuneCandidateReport {
                    candidate_index,
                    available: false,
                    error: Some(format!("{err}")),
                    layout: candidate,
                    lane_cycles_per_sec: 0.0,
                    report: None,
                });
            }
        }
    }

    let Some((selected_index, selected_layout)) = selected_candidate else {
        return Err(ErrorReport::new(vec![Diagnostic::new(
            "E_SIM_IR_AUTOTUNE_CANDIDATES",
            "all replay autotune candidates failed",
        )]));
    };
    let selected =
        replay_trace_threaded_with_initial_state(program, plan, &selected_layout, initial_state)?;
    Ok(ReplayAutotuneReport {
        selected_candidate: selected_index,
        candidates: candidate_reports,
        pruned_candidates,
        selected,
    })
}

impl EncodedTraceReplayPlan {
    fn prefix_steps(&self, steps: usize) -> Self {
        Self {
            steps: self.steps.iter().take(steps).cloned().collect(),
            independent_lanes: self.independent_lanes,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayAutotuneCandidateSet {
    pub candidates: Vec<ThreadedReplayOptions>,
    pub pruned_candidates: Vec<ReplayAutotunePrunedCandidateReport>,
}

pub fn build_replay_autotune_candidate_set(
    total_lanes: usize,
    max_workers: usize,
    suitability: Option<&SimdSuitabilityReport>,
) -> ReplayAutotuneCandidateSet {
    build_replay_autotune_candidate_set_with_affinity(total_lanes, max_workers, suitability, None)
}

pub fn build_replay_autotune_candidate_set_with_affinity(
    total_lanes: usize,
    max_workers: usize,
    suitability: Option<&SimdSuitabilityReport>,
    backend_affinity: Option<&BackendAffinityReport>,
) -> ReplayAutotuneCandidateSet {
    let workers = max_workers
        .max(1)
        .min(total_lanes)
        .min(std::thread::available_parallelism().map_or(1, usize::from));
    let mut candidates = Vec::new();
    for scalar_workers in scalar_autotune_worker_counts(total_lanes, workers) {
        candidates.push(ThreadedReplayOptions {
            workers: split_worker_lanes(total_lanes, scalar_workers)
                .into_iter()
                .map(|lanes| ThreadedReplayWorkerOptions {
                    backend: SimBackendKind::Scalar,
                    lanes,
                })
                .collect(),
            max_mismatches: 16,
        });
    }
    candidates.push(ThreadedReplayOptions {
        workers: split_worker_lanes(total_lanes, workers)
            .into_iter()
            .map(|lanes| ThreadedReplayWorkerOptions {
                backend: SimBackendKind::PackedCpu,
                lanes,
            })
            .collect(),
        max_mismatches: 16,
    });
    candidates.push(ThreadedReplayOptions {
        workers: split_worker_lanes(total_lanes, workers)
            .into_iter()
            .map(|lanes| ThreadedReplayWorkerOptions {
                backend: SimBackendKind::SimdCpu,
                lanes,
            })
            .collect(),
        max_mismatches: 16,
    });
    if workers > 1 && total_lanes > 1 {
        let scalar_lanes = total_lanes / 2;
        let simd_lanes = total_lanes - scalar_lanes;
        candidates.push(ThreadedReplayOptions {
            workers: vec![
                ThreadedReplayWorkerOptions {
                    backend: SimBackendKind::Scalar,
                    lanes: scalar_lanes,
                },
                ThreadedReplayWorkerOptions {
                    backend: SimBackendKind::SimdCpu,
                    lanes: simd_lanes,
                },
            ],
            max_mismatches: 16,
        });
    }
    if suitability.is_none() && backend_affinity.is_none() {
        return ReplayAutotuneCandidateSet {
            candidates,
            pruned_candidates: Vec::new(),
        };
    };

    let mut kept = Vec::with_capacity(candidates.len());
    let mut pruned_candidates = Vec::new();
    for (candidate_index, candidate) in candidates.into_iter().enumerate() {
        if let Some(reason) =
            replay_autotune_prune_reason(&candidate, suitability, backend_affinity)
        {
            pruned_candidates.push(ReplayAutotunePrunedCandidateReport {
                candidate_index,
                layout: candidate,
                reason,
            });
        } else {
            kept.push(candidate);
        }
    }
    ReplayAutotuneCandidateSet {
        candidates: kept,
        pruned_candidates,
    }
}

fn replay_autotune_prune_reason(
    candidate: &ThreadedReplayOptions,
    suitability: Option<&SimdSuitabilityReport>,
    backend_affinity: Option<&BackendAffinityReport>,
) -> Option<ReplayAutotunePruneReason> {
    let kind = replay_candidate_layout_kind(candidate);
    replay_autotune_prune_reason_for_affinity(kind, backend_affinity)
        .or_else(|| replay_autotune_prune_reason_for_suitability(kind, suitability))
}

fn replay_autotune_prune_reason_for_affinity(
    kind: ReplayCandidateLayoutKind,
    backend_affinity: Option<&BackendAffinityReport>,
) -> Option<ReplayAutotunePruneReason> {
    let recommendation = backend_affinity?.recommendation;
    match recommendation {
        BackendAffinityRecommendation::ScalarPreferred
        | BackendAffinityRecommendation::GpuBlocked => {
            if kind == ReplayCandidateLayoutKind::Scalar {
                None
            } else if recommendation == BackendAffinityRecommendation::GpuBlocked {
                Some(ReplayAutotunePruneReason::GpuCandidateBlocked)
            } else {
                Some(ReplayAutotunePruneReason::ScalarPreferred)
            }
        }
        BackendAffinityRecommendation::MixedScalarSimdCandidate => match kind {
            ReplayCandidateLayoutKind::Scalar | ReplayCandidateLayoutKind::Mixed => None,
            ReplayCandidateLayoutKind::Packed | ReplayCandidateLayoutKind::Simd => {
                Some(ReplayAutotunePruneReason::MixedOnly)
            }
        },
        BackendAffinityRecommendation::SimdCpuCandidate => None,
        BackendAffinityRecommendation::PackedCpuCandidate => match kind {
            ReplayCandidateLayoutKind::Scalar | ReplayCandidateLayoutKind::Packed => None,
            ReplayCandidateLayoutKind::Simd | ReplayCandidateLayoutKind::Mixed => {
                Some(ReplayAutotunePruneReason::PackedCpuPreferred)
            }
        },
    }
}

fn replay_autotune_prune_reason_for_suitability(
    kind: ReplayCandidateLayoutKind,
    suitability: Option<&SimdSuitabilityReport>,
) -> Option<ReplayAutotunePruneReason> {
    let suitability = suitability?;
    match kind {
        ReplayCandidateLayoutKind::Scalar => None,
        ReplayCandidateLayoutKind::Packed | ReplayCandidateLayoutKind::Simd => {
            match suitability.recommendation {
                SimdSuitabilityRecommendation::SimdCandidate => None,
                SimdSuitabilityRecommendation::MixedCandidate => {
                    Some(ReplayAutotunePruneReason::MixedOnly)
                }
                SimdSuitabilityRecommendation::ScalarPreferred => {
                    Some(ReplayAutotunePruneReason::ScalarPreferred)
                }
                SimdSuitabilityRecommendation::GpuCandidateBlocked => {
                    Some(ReplayAutotunePruneReason::GpuCandidateBlocked)
                }
            }
        }
        ReplayCandidateLayoutKind::Mixed => match suitability.recommendation {
            SimdSuitabilityRecommendation::SimdCandidate
            | SimdSuitabilityRecommendation::MixedCandidate => None,
            SimdSuitabilityRecommendation::ScalarPreferred => {
                if suitability.total.instr_count > 0
                    && suitability.total.fast_instrs * 100 / suitability.total.instr_count >= 50
                    && suitability.total.fallback_instrs <= suitability.total.fast_instrs
                {
                    None
                } else {
                    Some(ReplayAutotunePruneReason::ScalarPreferred)
                }
            }
            SimdSuitabilityRecommendation::GpuCandidateBlocked => {
                Some(ReplayAutotunePruneReason::GpuCandidateBlocked)
            }
        },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplayCandidateLayoutKind {
    Scalar,
    Packed,
    Simd,
    Mixed,
}

fn replay_candidate_layout_kind(candidate: &ThreadedReplayOptions) -> ReplayCandidateLayoutKind {
    let has_scalar = candidate
        .workers
        .iter()
        .any(|worker| worker.backend == SimBackendKind::Scalar);
    let has_packed = candidate
        .workers
        .iter()
        .any(|worker| worker.backend == SimBackendKind::PackedCpu);
    let has_simd = candidate
        .workers
        .iter()
        .any(|worker| worker.backend == SimBackendKind::SimdCpu);
    if has_scalar && !has_packed && !has_simd {
        ReplayCandidateLayoutKind::Scalar
    } else if has_packed && !has_scalar && !has_simd {
        ReplayCandidateLayoutKind::Packed
    } else if has_simd && !has_scalar && !has_packed {
        ReplayCandidateLayoutKind::Simd
    } else {
        ReplayCandidateLayoutKind::Mixed
    }
}

fn scalar_autotune_worker_counts(total_lanes: usize, workers: usize) -> Vec<usize> {
    let workers = workers.max(1).min(total_lanes);
    let mut seen = HashSet::new();
    [1, workers.div_ceil(2), workers]
        .into_iter()
        .filter(|count| seen.insert(*count))
        .collect()
}

fn split_worker_lanes(total_lanes: usize, workers: usize) -> Vec<usize> {
    let workers = workers.max(1).min(total_lanes);
    let base = total_lanes / workers;
    let extra = total_lanes % workers;
    (0..workers)
        .map(|index| base + usize::from(index < extra))
        .collect()
}

pub trait SimBackend {
    fn kind(&self) -> SimBackendKind;
    fn lanes(&self) -> usize;
    fn set_signals_replicated(&mut self, values: &[(Signal, u128)]) -> Result<(), ErrorReport>;
    fn set_signals_replicated_raw(&mut self, values: &[(Signal, u128)]) -> Result<(), ErrorReport>;
    fn get_signal_lane(&self, signal: Signal, lane: usize) -> Result<u128, ErrorReport>;
    fn set_memory_replicated(&mut self, memory: Signal, words: &[u128]) -> Result<(), ErrorReport>;
    fn eval_combinational(&mut self) -> Result<(), ErrorReport>;
    fn tick(&mut self) -> Result<(), ErrorReport>;
    fn tick_from_evaluated_no_post_eval(&mut self) -> Result<(), ErrorReport>;
    fn tick_many(&mut self, steps: usize) -> Result<(), ErrorReport>;
    fn replay_trace(
        &mut self,
        plan: &EncodedTraceReplayPlan,
        options: ReplayOptions,
    ) -> Result<ReplayReport, ErrorReport> {
        if options.lane_mode == ReplayLaneMode::Independent && self.lanes() != 1 {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_REPLAY_LANE_MODE",
                "generic independent-lane replay supports only one-lane backends",
            )]));
        }
        let mut report = ReplayReport::default();
        let lanes_to_check = match options.lane_mode {
            ReplayLaneMode::Independent => self.lanes(),
            ReplayLaneMode::Replicated => match options.check_mode {
                ReplayCheckMode::Lane0Fast => 1,
                ReplayCheckMode::AllLanes => self.lanes(),
            },
        };
        for (step_index, step) in plan.steps.iter().enumerate() {
            let start = Instant::now();
            let inputs = step
                .inputs
                .iter()
                .map(|input| {
                    let limbs = match options.lane_mode {
                        ReplayLaneMode::Replicated => &input.limbs,
                        ReplayLaneMode::Independent => input
                            .lane_limbs
                            .as_ref()
                            .and_then(|values| values.first())
                            .unwrap_or(&input.limbs),
                    };
                    (input.signal, decode_u128_limbs(limbs))
                })
                .collect::<Vec<_>>();
            self.set_signals_replicated_raw(&inputs)?;
            report.timing.input_ns += start.elapsed().as_nanos();

            let start = Instant::now();
            self.eval_combinational()?;
            report.timing.eval_ns += start.elapsed().as_nanos();

            let start = Instant::now();
            for check in &step.checks {
                for lane in 0..lanes_to_check {
                    let actual = self.get_signal_lane(check.signal, lane)?;
                    let expected = match options.lane_mode {
                        ReplayLaneMode::Replicated => check.expected,
                        ReplayLaneMode::Independent => check
                            .lane_expected
                            .as_ref()
                            .and_then(|values| values.get(lane))
                            .copied()
                            .unwrap_or(check.expected),
                    };
                    if actual != expected {
                        report.record_mismatch(
                            options,
                            step_index,
                            check.check_index,
                            expected,
                            actual,
                            if self.lanes() == 1 { None } else { Some(lane) },
                        );
                    }
                }
            }
            report.timing.compare_ns += start.elapsed().as_nanos();

            let start = Instant::now();
            self.tick_from_evaluated_no_post_eval()?;
            report.timing.tick_ns += start.elapsed().as_nanos();
        }
        Ok(report)
    }
}

fn replay_signal_index(program: &PackedProgram, signal: Signal) -> Result<usize, ErrorReport> {
    program.signal_index(signal).ok_or_else(|| {
        ErrorReport::new(vec![Diagnostic::new(
            "E_SIM_IR_REPLAY_SIGNAL",
            format!("signal {:?} is not part of this replay program", signal.id),
        )])
    })
}

fn validate_encoded_signal_index(
    program: &PackedProgram,
    signal: Signal,
    signal_index: usize,
) -> Result<(), ErrorReport> {
    if program.signal_index(signal) == Some(signal_index) {
        Ok(())
    } else {
        Err(ErrorReport::new(vec![Diagnostic::new(
            "E_SIM_IR_REPLAY_SIGNAL",
            format!(
                "signal {:?} does not match replay signal index {}",
                signal.id, signal_index
            ),
        )]))
    }
}

#[derive(Clone, Debug)]
pub struct SimdCpuSimulator {
    inner: PackedSimulator,
    workspaces: SimdExecutionWorkspaces,
    last_simd_stats: SimdBlockStats,
    /// During a `tick_clocked`, the set of memory indices to SKIP this step
    /// (their write clock has no rising edge). `None` outside `tick_clocked`.
    skip_mems: Option<std::collections::HashSet<usize>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SimdBlockStats {
    fast_instrs: usize,
    fallback_instrs: usize,
    state_reuses: usize,
    lane_materializations: usize,
    fast_paths: SimdFastPathStats,
    fallback_reasons: SimdFallbackStats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SimdFastPathStats {
    one_limb_ops: usize,
    two_limb_ops: usize,
    native_two_limb_ops: usize,
    two_limb_mul_ops: usize,
    two_limb_memory_reads: usize,
    two_limb_mux_ops: usize,
    memory_write_effects: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SimdFallbackStats {
    sext: usize,
    mem_read: usize,
    wide_op: usize,
    signed_lt: usize,
    wide_concat: usize,
    other: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimdFallbackReason {
    Sext,
    MemRead,
    WideOp,
    SignedLt,
    WideConcat,
}

impl SimdFallbackStats {
    fn record(&mut self, reason: SimdFallbackReason) {
        match reason {
            SimdFallbackReason::Sext => self.sext += 1,
            SimdFallbackReason::MemRead => self.mem_read += 1,
            SimdFallbackReason::WideOp => self.wide_op += 1,
            SimdFallbackReason::SignedLt => self.signed_lt += 1,
            SimdFallbackReason::WideConcat => self.wide_concat += 1,
        }
    }
}

impl SimdBlockStats {
    fn record_fast_instr(&mut self) {
        self.fast_instrs += 1;
    }

    fn record_one_limb_op(&mut self) {
        self.fast_instrs += 1;
        self.fast_paths.one_limb_ops += 1;
    }

    fn record_two_limb_op(&mut self) {
        self.fast_instrs += 1;
        self.fast_paths.two_limb_ops += 1;
    }

    fn record_native_two_limb_op(&mut self) {
        self.fast_paths.native_two_limb_ops += 1;
    }

    fn record_two_limb_mul_op(&mut self) {
        self.fast_instrs += 1;
        self.fast_paths.two_limb_ops += 1;
        self.fast_paths.two_limb_mul_ops += 1;
    }

    fn record_two_limb_memory_read(&mut self) {
        self.fast_instrs += 1;
        self.fast_paths.two_limb_memory_reads += 1;
    }

    fn record_two_limb_mux_op(&mut self) {
        self.fast_instrs += 1;
        self.fast_paths.two_limb_mux_ops += 1;
    }

    fn record_memory_write_effect(&mut self) {
        self.fast_paths.memory_write_effects += 1;
    }
}

impl From<SimdBlockStats> for ReplaySimdStats {
    fn from(value: SimdBlockStats) -> Self {
        Self {
            fast_instrs: value.fast_instrs,
            fallback_instrs: value.fallback_instrs,
            state_reuses: value.state_reuses,
            lane_materializations: value.lane_materializations,
            fast_paths: ReplaySimdFastPathStats {
                one_limb_ops: value.fast_paths.one_limb_ops,
                two_limb_ops: value.fast_paths.two_limb_ops,
                native_two_limb_ops: value.fast_paths.native_two_limb_ops,
                two_limb_mul_ops: value.fast_paths.two_limb_mul_ops,
                two_limb_memory_reads: value.fast_paths.two_limb_memory_reads,
                two_limb_mux_ops: value.fast_paths.two_limb_mux_ops,
                memory_write_effects: value.fast_paths.memory_write_effects,
            },
            fallback_reasons: ReplaySimdFallbackStats {
                sext: value.fallback_reasons.sext,
                mem_read: value.fallback_reasons.mem_read,
                wide_op: value.fallback_reasons.wide_op,
                signed_lt: value.fallback_reasons.signed_lt,
                wide_concat: value.fallback_reasons.wide_concat,
                other: value.fallback_reasons.other,
            },
        }
    }
}

#[derive(Clone, Debug)]
struct SimdValueSlot {
    limbs: usize,
    width: Width,
    words: Vec<u32>,
    valid: bool,
}

#[derive(Clone, Debug, Default)]
struct SimdBlockState {
    lanes: usize,
    slots: Vec<SimdValueSlot>,
    scalar_scratch: Vec<u32>,
    stats: SimdBlockStats,
}

#[derive(Clone, Debug, Default)]
struct SimdExecutionWorkspaces {
    async_reset_comb: SimdBlockState,
    comb: SimdBlockState,
    tick_next: SimdBlockState,
    tick_commit: SimdBlockState,
}

impl SimdCpuSimulator {
    pub fn new(program: PackedProgram, lanes: usize) -> Result<Self, ErrorReport> {
        let inner = PackedSimulator::new(program, lanes)?;
        let workspaces = SimdExecutionWorkspaces::new(&inner.execution, inner.lanes);
        Ok(Self {
            inner,
            workspaces,
            last_simd_stats: SimdBlockStats::default(),
            skip_mems: None,
        })
    }

    pub fn program(&self) -> &PackedProgram {
        self.inner.program()
    }

    pub fn inner(&self) -> &PackedSimulator {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut PackedSimulator {
        &mut self.inner
    }

    pub fn has_native_simd() -> bool {
        cfg!(target_arch = "aarch64") || cfg!(target_arch = "x86_64")
    }

    pub fn snapshot_storage(&self) -> PackedSimulatorStorage {
        self.inner.snapshot_storage()
    }

    pub fn restore_storage(&mut self, storage: &PackedSimulatorStorage) -> Result<(), ErrorReport> {
        self.inner.restore_storage(storage)?;
        self.eval_combinational()
    }

    pub fn set_signal(&mut self, signal: Signal, lane_values: &[u128]) -> Result<(), ErrorReport> {
        let index = self.inner.signal_index(signal)?;
        let ty = self.inner.program.signals[index].layout.ty;
        let values = lane_values
            .iter()
            .map(|value| encode_u128_limbs(*value, ty))
            .collect::<Vec<_>>();
        self.set_signal_limbs(signal, &values)
    }

    pub fn set_signals_replicated(&mut self, values: &[(Signal, u128)]) -> Result<(), ErrorReport> {
        self.set_signals_replicated_raw(values)?;
        self.eval_combinational()
    }

    pub fn set_signals_replicated_raw(
        &mut self,
        values: &[(Signal, u128)],
    ) -> Result<(), ErrorReport> {
        let mut encoded = Vec::with_capacity(values.len());
        for (signal, value) in values {
            let index = self.inner.signal_index(*signal)?;
            let ty = self.inner.program.signals[index].layout.ty;
            encoded.push((index, encode_u128_limbs(*value, ty)));
        }
        for (index, limbs) in encoded {
            let layout = self.inner.program.signals[index].layout;
            for lane in 0..self.inner.lanes {
                self.inner.store_layout(layout, lane, &limbs);
            }
        }
        Ok(())
    }

    pub fn set_signal_limbs(
        &mut self,
        signal: Signal,
        lane_values: &[Vec<u32>],
    ) -> Result<(), ErrorReport> {
        if lane_values.len() != self.inner.lanes {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_LANE_VALUES",
                format!(
                    "expected {} lane values, got {}",
                    self.inner.lanes,
                    lane_values.len()
                ),
            )]));
        }
        let index = self.inner.signal_index(signal)?;
        let layout = self.inner.program.signals[index].layout;
        for (lane, value) in lane_values.iter().enumerate() {
            if value.len() != layout.limbs {
                return Err(ErrorReport::new(vec![Diagnostic::new(
                    "E_SIM_IR_LANE_VALUES",
                    format!("expected {} limbs, got {}", layout.limbs, value.len()),
                )]));
            }
            self.inner.store_layout(layout, lane, value);
        }
        self.eval_combinational()
    }

    pub fn get_signal(&self, signal: Signal) -> Result<Vec<u128>, ErrorReport> {
        self.inner.get_signal(signal)
    }

    pub fn get_signal_limbs(&self, signal: Signal) -> Result<Vec<Vec<u32>>, ErrorReport> {
        self.inner.get_signal_limbs(signal)
    }

    pub fn set_memory_limbs(
        &mut self,
        memory: Signal,
        lane_words: &[Vec<Vec<u32>>],
    ) -> Result<(), ErrorReport> {
        let index = self.inner.memory_signal_index(memory)?;
        if lane_words.len() != self.inner.lanes {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_MEMORY_VALUES",
                format!(
                    "expected {} lanes of memory values, got {}",
                    self.inner.lanes,
                    lane_words.len()
                ),
            )]));
        }
        let shape = self.inner.program.memories[index].clone();
        for (lane, words) in lane_words.iter().enumerate() {
            if words.len() != shape.depth {
                return Err(ErrorReport::new(vec![Diagnostic::new(
                    "E_SIM_IR_MEMORY_VALUES",
                    format!("expected {} memory words, got {}", shape.depth, words.len()),
                )]));
            }
            for (addr, word) in words.iter().enumerate() {
                if word.len() != shape.data_layout.limbs {
                    return Err(ErrorReport::new(vec![Diagnostic::new(
                        "E_SIM_IR_MEMORY_VALUES",
                        format!(
                            "expected {} limbs per memory word, got {}",
                            shape.data_layout.limbs,
                            word.len()
                        ),
                    )]));
                }
                self.inner.store_memory(index, lane, addr, word);
            }
        }
        self.eval_combinational()
    }

    pub fn set_memory(
        &mut self,
        memory: Signal,
        lane_words: &[Vec<u128>],
    ) -> Result<(), ErrorReport> {
        let index = self.inner.memory_signal_index(memory)?;
        let ty = self.inner.program.memories[index].data_layout.ty;
        if ty.width > 128 {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_WIDE_MEMORY",
                "use set_memory_limbs for memories wider than 128 bits",
            )]));
        }
        let lane_limbs = lane_words
            .iter()
            .map(|words| {
                words
                    .iter()
                    .copied()
                    .map(|word| encode_u128_limbs(word, ty))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        self.set_memory_limbs(memory, &lane_limbs)
    }

    pub fn set_memory_replicated(
        &mut self,
        memory: Signal,
        words: &[u128],
    ) -> Result<(), ErrorReport> {
        let index = self.inner.memory_signal_index(memory)?;
        let ty = self.inner.program.memories[index].data_layout.ty;
        if ty.width > 128 {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_WIDE_MEMORY",
                "use set_memory_limbs for memories wider than 128 bits",
            )]));
        }
        let lane_words = vec![
            words
                .iter()
                .copied()
                .map(|word| encode_u128_limbs(word, ty))
                .collect::<Vec<_>>();
            self.inner.lanes
        ];
        self.set_memory_limbs(memory, &lane_words)
    }

    pub fn get_memory_limbs(&self, memory: Signal) -> Result<Vec<Vec<Vec<u32>>>, ErrorReport> {
        self.inner.get_memory_limbs(memory)
    }

    pub fn get_memory(&self, memory: Signal) -> Result<Vec<Vec<u128>>, ErrorReport> {
        self.inner.get_memory(memory)
    }

    pub fn eval_combinational(&mut self) -> Result<(), ErrorReport> {
        self.execute_stream(PackedStreamKind::Comb);
        self.execute_stream(PackedStreamKind::AsyncResetComb);
        Ok(())
    }

    pub fn tick(&mut self) -> Result<(), ErrorReport> {
        self.eval_combinational()?;
        self.tick_from_evaluated_no_post_eval()?;
        self.eval_combinational()?;
        Ok(())
    }

    pub fn tick_from_evaluated_no_post_eval(&mut self) -> Result<(), ErrorReport> {
        self.capture_register_stream();
        self.execute_stream(PackedStreamKind::TickCommit);
        self.inner.commit_register_captures();
        Ok(())
    }

    /// Advance one step under an explicit set of clocks with a rising edge this
    /// step (the multi-clock primitive; mirrors `rrtl_core::Simulator::tick_clocked`).
    /// Registers whose clock is not in `active_clocks` hold. With every clock
    /// listed this is identical to [`Self::tick`]. (Memory writes are not yet
    /// clock-gated — register domains only.)
    pub fn tick_clocked(&mut self, active_clocks: &[Signal]) -> Result<(), ErrorReport> {
        let active: std::collections::HashSet<usize> = active_clocks
            .iter()
            .filter_map(|signal| self.inner.program.signal_index(*signal))
            .collect();
        // Memories whose write clock is inactive this step are skipped.
        let skip: std::collections::HashSet<usize> = self
            .inner
            .program
            .mem_clocks
            .iter()
            .filter(|(_, clk)| !active.contains(clk))
            .map(|(&mem, _)| mem)
            .collect();
        self.eval_combinational()?;
        self.capture_register_stream();
        self.skip_mems = Some(skip);
        self.execute_stream(PackedStreamKind::TickCommit);
        self.skip_mems = None;
        self.inner.commit_register_captures_clocked(&active);
        self.eval_combinational()?;
        Ok(())
    }

    pub fn tick_many(&mut self, steps: usize) -> Result<(), ErrorReport> {
        // Comb-fusion: `tick()` is comb→capture/commit→comb, so a naive loop runs
        // the combinational stream TWICE per cycle. Across cycles the trailing comb
        // (after commit) settles exactly the state the next cycle's leading comb
        // would — so within `tick_many` (inputs fixed for the whole call) it is
        // redundant. Evaluate comb once up front, then per cycle do capture/commit
        // and a single comb (which serves as both the next cycle's leading settle
        // and, on the last iteration, the final observable settle). This is
        // bit-identical to the per-`tick()` loop, at ~half the comb work — the bulk
        // of an RTL tick. (The GPU interpreter uses the same fusion.)
        if steps == 0 {
            return Ok(());
        }
        self.eval_combinational()?;
        for _ in 0..steps {
            self.tick_from_evaluated_no_post_eval()?;
            self.eval_combinational()?;
        }
        Ok(())
    }

    pub fn replay_trace(
        &mut self,
        plan: &EncodedTraceReplayPlan,
        options: ReplayOptions,
    ) -> Result<ReplayReport, ErrorReport> {
        self.last_simd_stats = SimdBlockStats::default();
        let mut report = ReplayReport::default();
        let lanes_to_check = match options.lane_mode {
            ReplayLaneMode::Independent => self.inner.lanes,
            ReplayLaneMode::Replicated => match options.check_mode {
                ReplayCheckMode::Lane0Fast => 1,
                ReplayCheckMode::AllLanes => self.inner.lanes,
            },
        };
        for (step_index, step) in plan.steps.iter().enumerate() {
            let start = Instant::now();
            match options.lane_mode {
                ReplayLaneMode::Replicated => {
                    for input in &step.inputs {
                        self.inner
                            .set_packed_signal_replicated_limbs(input.signal_index, &input.limbs)
                    }
                }
                ReplayLaneMode::Independent => self.inner.replay_independent_inputs(step)?,
            }
            report.timing.input_ns += start.elapsed().as_nanos();

            let start = Instant::now();
            self.eval_combinational()?;
            report.timing.eval_ns += start.elapsed().as_nanos();

            let start = Instant::now();
            match options.lane_mode {
                ReplayLaneMode::Replicated => {
                    for check in &step.checks {
                        let layout = self.inner.program.signals[check.signal_index].layout;
                        for lane in 0..lanes_to_check {
                            if !self.inner.layout_matches_encoded(
                                layout,
                                lane,
                                check.expected_limbs.as_slice(),
                            ) {
                                let actual_limbs = self.inner.load_layout(layout, lane);
                                report.record_mismatch(
                                    options,
                                    step_index,
                                    check.check_index,
                                    check.expected,
                                    decode_u128_limbs(&actual_limbs),
                                    if self.inner.lanes == 1 {
                                        None
                                    } else {
                                        Some(lane)
                                    },
                                );
                            }
                        }
                    }
                }
                ReplayLaneMode::Independent => self.inner.check_independent_replay_outputs(
                    step,
                    step_index,
                    options,
                    &mut report,
                )?,
            }
            report.timing.compare_ns += start.elapsed().as_nanos();

            let start = Instant::now();
            self.tick_from_evaluated_no_post_eval()?;
            report.timing.tick_ns += start.elapsed().as_nanos();
        }
        report.simd_stats = Some(self.last_simd_stats.into());
        Ok(report)
    }

    #[cfg(test)]
    fn reset_simd_stats(&mut self) {
        self.last_simd_stats = SimdBlockStats::default();
    }

    #[cfg(test)]
    fn simd_stats(&self) -> SimdBlockStats {
        self.last_simd_stats
    }

    fn execute_stream(&mut self, stream: PackedStreamKind) {
        match stream {
            PackedStreamKind::AsyncResetComb => {
                if self
                    .inner
                    .machine
                    .streams
                    .async_reset_comb
                    .packets
                    .is_empty()
                {
                    return;
                }
                let block = std::mem::take(&mut self.inner.machine.streams.async_reset_comb);
                let cache = std::mem::take(&mut self.inner.execution.async_reset_comb);
                let mut state = std::mem::take(&mut self.workspaces.async_reset_comb);
                self.execute_machine_block(&block, &cache, &mut state);
                self.workspaces.async_reset_comb = state;
                self.inner.machine.streams.async_reset_comb = block;
                self.inner.execution.async_reset_comb = cache;
            }
            PackedStreamKind::Comb => {
                if self.inner.machine.streams.comb.packets.is_empty() {
                    return;
                }
                let block = std::mem::take(&mut self.inner.machine.streams.comb);
                let cache = std::mem::take(&mut self.inner.execution.comb);
                let mut state = std::mem::take(&mut self.workspaces.comb);
                self.execute_machine_block(&block, &cache, &mut state);
                self.workspaces.comb = state;
                self.inner.machine.streams.comb = block;
                self.inner.execution.comb = cache;
            }
            PackedStreamKind::TickCommit => {
                if self.inner.machine.streams.tick_commit.packets.is_empty() {
                    return;
                }
                let block = std::mem::take(&mut self.inner.machine.streams.tick_commit);
                let cache = std::mem::take(&mut self.inner.execution.tick_commit);
                let mut state = std::mem::take(&mut self.workspaces.tick_commit);
                self.execute_machine_block(&block, &cache, &mut state);
                self.workspaces.tick_commit = state;
                self.inner.machine.streams.tick_commit = block;
                self.inner.execution.tick_commit = cache;
            }
        }
    }

    fn capture_register_stream(&mut self) {
        self.inner.register_capture_count = 0;
        if self.inner.machine.streams.tick_next.packets.is_empty() {
            return;
        }
        let block = std::mem::take(&mut self.inner.machine.streams.tick_next);
        let cache = std::mem::take(&mut self.inner.execution.tick_next);
        let mut state = std::mem::take(&mut self.workspaces.tick_next);
        self.capture_machine_registers(&block, &cache, &mut state);
        self.workspaces.tick_next = state;
        self.inner.machine.streams.tick_next = block;
        self.inner.execution.tick_next = cache;
    }

    fn execute_machine_block(
        &mut self,
        block: &PackedBlock,
        cache: &PackedBlockCache,
        state: &mut SimdBlockState,
    ) {
        state.reset(cache, self.inner.lanes);
        for packet in &block.packets {
            for instr in &packet.instrs {
                self.eval_simd_instr_into(instr, state, cache);
            }
            for effect in &packet.effects {
                self.execute_simd_effect(effect, state);
            }
        }
        self.record_simd_stats(state.stats);
    }

    fn capture_machine_registers(
        &mut self,
        block: &PackedBlock,
        cache: &PackedBlockCache,
        state: &mut SimdBlockState,
    ) {
        state.reset(cache, self.inner.lanes);
        for packet in &block.packets {
            for instr in &packet.instrs {
                self.eval_simd_instr_into(instr, state, cache);
            }
            for effect in &packet.effects {
                if let PackedEffect::CaptureReg { dst, value, reset } = effect {
                    self.capture_simd_register_value(*dst, state.value(*value), reset.as_ref());
                }
            }
        }
        self.record_simd_stats(state.stats);
    }

    fn capture_simd_register_value(
        &mut self,
        signal: usize,
        value: &SimdValueSlot,
        reset: Option<&PackedReset>,
    ) {
        let layout = self.inner.program.signals[signal].layout;
        if self.inner.register_capture_count == self.inner.register_captures.len() {
            self.inner.register_captures.push(PackedRegisterCapture {
                signal,
                layout,
                values: Vec::new(),
            });
        }
        let capture_index = self.inner.register_capture_count;
        self.inner.register_capture_count += 1;

        let mut flat_values = Vec::new();
        std::mem::swap(
            &mut flat_values,
            &mut self.inner.register_captures[capture_index].values,
        );
        flat_values.resize(layout.limbs * self.inner.lanes, 0);
        for lane in 0..self.inner.lanes {
            if reset.is_some_and(|reset| self.inner.reset_asserted(reset, lane)) {
                let reset = reset.unwrap();
                let reset_value = fit_limbs(reset.value.clone(), layout.ty);
                for limb in 0..layout.limbs {
                    flat_values[limb * self.inner.lanes + lane] = reset_value[limb];
                }
            } else {
                for limb in 0..layout.limbs {
                    let mut word = value.word(limb, lane, self.inner.lanes);
                    if limb + 1 == layout.limbs {
                        word &= final_limb_mask(layout.width);
                    }
                    flat_values[limb * self.inner.lanes + lane] = word;
                }
            }
        }

        let capture = &mut self.inner.register_captures[capture_index];
        capture.signal = signal;
        capture.layout = layout;
        capture.values = flat_values;
    }

    fn record_simd_stats(&mut self, stats: SimdBlockStats) {
        self.last_simd_stats.fast_instrs += stats.fast_instrs;
        self.last_simd_stats.fallback_instrs += stats.fallback_instrs;
        self.last_simd_stats.state_reuses += stats.state_reuses;
        self.last_simd_stats.lane_materializations += stats.lane_materializations;
        self.last_simd_stats.fast_paths.one_limb_ops += stats.fast_paths.one_limb_ops;
        self.last_simd_stats.fast_paths.two_limb_ops += stats.fast_paths.two_limb_ops;
        self.last_simd_stats.fast_paths.native_two_limb_ops += stats.fast_paths.native_two_limb_ops;
        self.last_simd_stats.fast_paths.two_limb_mul_ops += stats.fast_paths.two_limb_mul_ops;
        self.last_simd_stats.fast_paths.two_limb_memory_reads +=
            stats.fast_paths.two_limb_memory_reads;
        self.last_simd_stats.fast_paths.two_limb_mux_ops += stats.fast_paths.two_limb_mux_ops;
        self.last_simd_stats.fast_paths.memory_write_effects +=
            stats.fast_paths.memory_write_effects;
        self.last_simd_stats.fallback_reasons.sext += stats.fallback_reasons.sext;
        self.last_simd_stats.fallback_reasons.mem_read += stats.fallback_reasons.mem_read;
        self.last_simd_stats.fallback_reasons.wide_op += stats.fallback_reasons.wide_op;
        self.last_simd_stats.fallback_reasons.signed_lt += stats.fallback_reasons.signed_lt;
        self.last_simd_stats.fallback_reasons.wide_concat += stats.fallback_reasons.wide_concat;
        self.last_simd_stats.fallback_reasons.other += stats.fallback_reasons.other;
    }

    fn execute_simd_effect(&mut self, effect: &PackedEffect, state: &mut SimdBlockState) {
        match effect {
            PackedEffect::StoreSignal { dst, value } => {
                let value = state.value(*value);
                self.store_signal_value(*dst, &value);
            }
            PackedEffect::CaptureReg { dst, value, reset } => {
                let Some(reset) = reset else {
                    return;
                };
                if reset.kind != ResetKind::Async {
                    return;
                }
                let value = state.value(*value);
                for lane in 0..self.inner.lanes {
                    if self.inner.reset_asserted(reset, lane) {
                        self.store_signal_slot_lane(*dst, lane, value);
                    }
                }
            }
            PackedEffect::MemoryWrite {
                memory,
                enable,
                addr,
                data,
            } => {
                // Multi-clock: skip writes whose memory's clock is inactive this step.
                if let Some(skip) = &self.skip_mems {
                    if skip.contains(memory) {
                        return;
                    }
                }
                state.stats.record_memory_write_effect();
                let enable = state.value(*enable);
                let addr = state.value(*addr);
                let data = state.value(*data);
                for lane in 0..self.inner.lanes {
                    if enable.decode_bool_lane(lane) {
                        let addr = addr.decode_usize_lane(lane);
                        let mem = &self.inner.program.memories[*memory];
                        if addr < mem.depth {
                            self.store_memory_slot_lane(*memory, lane, addr, data);
                        }
                    }
                }
            }
        }
    }

    fn eval_simd_instr_into(
        &self,
        instr: &PackedInstr,
        state: &mut SimdBlockState,
        cache: &PackedBlockCache,
    ) {
        match &instr.kind {
            PackedInstrKind::Lit(value) => {
                let lanes = state.lanes;
                state
                    .slot_mut(instr.dst)
                    .fill_lit(value, instr.ty.width, lanes);
                state.stats.record_fast_instr();
            }
            PackedInstrKind::Signal(signal) => {
                let lanes = state.lanes;
                let layout = self.inner.program.signals[*signal].layout;
                state
                    .slot_mut(instr.dst)
                    .load_signal(&self.inner, layout, lanes);
                state.stats.record_fast_instr();
            }
            PackedInstrKind::Not(value) => {
                if state.value(*value).is_one_limb() && instr.ty.width <= 32 {
                    let lanes = state.lanes;
                    let mask = final_limb_mask(instr.ty.width);
                    let (value, dst) = state.slot_ref_mut(*value, instr.dst);
                    dst.prepare(instr.ty.width, 1, lanes);
                    simd_words::not_into(value.one_limb_words(lanes), mask, &mut dst.words);
                    dst.mask_final_limb(lanes);
                    state.stats.record_one_limb_op();
                } else if state.value(*value).is_at_most_two_limb()
                    && cache.value_width(*value) <= 64
                    && instr.ty.width <= 64
                {
                    self.eval_u64_not_into(instr, state, *value);
                } else {
                    self.eval_simd_instr_scalar_into(
                        instr,
                        state,
                        cache,
                        SimdFallbackReason::WideOp,
                    );
                }
            }
            PackedInstrKind::And(lhs, rhs) => self.eval_u64_binop_into(
                instr,
                state,
                *lhs,
                *rhs,
                cache,
                simd_words::and_into,
                |lhs, rhs| lhs & rhs,
                true,
                false,
                Some(simd_words::TwoLimbBinOp::And),
            ),
            PackedInstrKind::Or(lhs, rhs) => self.eval_u64_binop_into(
                instr,
                state,
                *lhs,
                *rhs,
                cache,
                simd_words::or_into,
                |lhs, rhs| lhs | rhs,
                true,
                false,
                Some(simd_words::TwoLimbBinOp::Or),
            ),
            PackedInstrKind::Xor(lhs, rhs) => self.eval_u64_binop_into(
                instr,
                state,
                *lhs,
                *rhs,
                cache,
                simd_words::xor_into,
                |lhs, rhs| lhs ^ rhs,
                true,
                false,
                Some(simd_words::TwoLimbBinOp::Xor),
            ),
            PackedInstrKind::Add(lhs, rhs) => self.eval_u64_binop_into(
                instr,
                state,
                *lhs,
                *rhs,
                cache,
                simd_words::add_into,
                u64::wrapping_add,
                true,
                false,
                Some(simd_words::TwoLimbBinOp::Add),
            ),
            PackedInstrKind::Sub(lhs, rhs) => self.eval_u64_binop_into(
                instr,
                state,
                *lhs,
                *rhs,
                cache,
                simd_words::sub_into,
                u64::wrapping_sub,
                true,
                false,
                Some(simd_words::TwoLimbBinOp::Sub),
            ),
            PackedInstrKind::Mul(lhs, rhs) => self.eval_u64_binop_into(
                instr,
                state,
                *lhs,
                *rhs,
                cache,
                simd_words::mul_into,
                |lhs, rhs| ((lhs as u128).wrapping_mul(rhs as u128)) as u64,
                true,
                true,
                None,
            ),
            PackedInstrKind::Eq(lhs, rhs) => {
                let lhs_id = *lhs;
                let rhs_id = *rhs;
                if state.value(lhs_id).is_one_limb() && state.value(rhs_id).is_one_limb() {
                    let lanes = state.lanes;
                    let mask =
                        final_limb_mask(cache.value_width(lhs_id).max(cache.value_width(rhs_id)));
                    let (lhs, rhs, dst) = state.slot_refs_mut(lhs_id, rhs_id, instr.dst);
                    dst.prepare(instr.ty.width, 1, lanes);
                    simd_words::eq_into(
                        lhs.one_limb_words(lanes),
                        rhs.one_limb_words(lanes),
                        mask,
                        &mut dst.words,
                    );
                    dst.mask_final_limb(lanes);
                    state.stats.record_one_limb_op();
                } else if state.value(lhs_id).is_at_most_two_limb()
                    && state.value(rhs_id).is_at_most_two_limb()
                    && cache.value_width(lhs_id) <= 64
                    && cache.value_width(rhs_id) <= 64
                {
                    self.eval_u64_compare_into(
                        instr,
                        state,
                        lhs_id,
                        rhs_id,
                        cache,
                        simd_words::TwoLimbCompareOp::Eq,
                    );
                } else {
                    self.eval_simd_instr_scalar_into(
                        instr,
                        state,
                        cache,
                        SimdFallbackReason::WideOp,
                    );
                }
            }
            PackedInstrKind::Ne(lhs, rhs) => {
                let lhs_id = *lhs;
                let rhs_id = *rhs;
                if state.value(lhs_id).is_one_limb() && state.value(rhs_id).is_one_limb() {
                    let lanes = state.lanes;
                    let mask =
                        final_limb_mask(cache.value_width(lhs_id).max(cache.value_width(rhs_id)));
                    let (lhs, rhs, dst) = state.slot_refs_mut(lhs_id, rhs_id, instr.dst);
                    dst.prepare(instr.ty.width, 1, lanes);
                    simd_words::ne_into(
                        lhs.one_limb_words(lanes),
                        rhs.one_limb_words(lanes),
                        mask,
                        &mut dst.words,
                    );
                    dst.mask_final_limb(lanes);
                    state.stats.record_one_limb_op();
                } else if state.value(lhs_id).is_at_most_two_limb()
                    && state.value(rhs_id).is_at_most_two_limb()
                    && cache.value_width(lhs_id) <= 64
                    && cache.value_width(rhs_id) <= 64
                {
                    self.eval_u64_compare_into(
                        instr,
                        state,
                        lhs_id,
                        rhs_id,
                        cache,
                        simd_words::TwoLimbCompareOp::Ne,
                    );
                } else {
                    self.eval_simd_instr_scalar_into(
                        instr,
                        state,
                        cache,
                        SimdFallbackReason::WideOp,
                    );
                }
            }
            PackedInstrKind::Mux {
                cond,
                then_value,
                else_value,
            } => {
                if state.value(*cond).is_one_limb()
                    && state.value(*then_value).is_one_limb()
                    && state.value(*else_value).is_one_limb()
                    && instr.ty.width <= 32
                {
                    let lanes = state.lanes;
                    let (cond, then_value, else_value, dst) =
                        state.slot3_refs_mut(*cond, *then_value, *else_value, instr.dst);
                    dst.prepare(instr.ty.width, 1, lanes);
                    simd_words::mux_into(
                        cond.one_limb_words(lanes),
                        then_value.one_limb_words(lanes),
                        else_value.one_limb_words(lanes),
                        &mut dst.words,
                    );
                    dst.mask_final_limb(lanes);
                    state.stats.record_one_limb_op();
                } else if state.value(*cond).is_one_limb()
                    && state.value(*then_value).is_at_most_two_limb()
                    && state.value(*else_value).is_at_most_two_limb()
                    && cache.value_width(*cond) <= 32
                    && cache.value_width(*then_value) <= 64
                    && cache.value_width(*else_value) <= 64
                    && instr.ty.width <= 64
                {
                    self.eval_u64_mux_into(instr, state, *cond, *then_value, *else_value);
                } else {
                    self.eval_simd_instr_scalar_into(
                        instr,
                        state,
                        cache,
                        SimdFallbackReason::WideOp,
                    );
                }
            }
            PackedInstrKind::Zext(value)
            | PackedInstrKind::Trunc(value)
            | PackedInstrKind::Cast(value) => {
                if state.value(*value).is_one_limb() && instr.ty.width <= 32 {
                    let lanes = state.lanes;
                    let mask = final_limb_mask(instr.ty.width);
                    let (value, dst) = state.slot_ref_mut(*value, instr.dst);
                    dst.prepare(instr.ty.width, 1, lanes);
                    simd_words::mask_into(value.one_limb_words(lanes), mask, &mut dst.words);
                    dst.mask_final_limb(lanes);
                    state.stats.record_one_limb_op();
                } else if state.value(*value).is_at_most_two_limb()
                    && cache.value_width(*value) <= 64
                    && instr.ty.width <= 64
                {
                    self.eval_u64_pass_into(instr, state, *value);
                } else {
                    self.eval_simd_instr_scalar_into(
                        instr,
                        state,
                        cache,
                        SimdFallbackReason::WideOp,
                    );
                }
            }
            PackedInstrKind::Lt { lhs, rhs, signed } => {
                if state.value(*lhs).is_one_limb()
                    && state.value(*rhs).is_one_limb()
                    && cache.value_width(*lhs) <= 32
                    && cache.value_width(*rhs) <= 32
                    && (!*signed || cache.value_width(*lhs) > 0)
                {
                    let lanes = state.lanes;
                    let width = if *signed {
                        cache.value_width(*lhs)
                    } else {
                        cache.value_width(*lhs).max(cache.value_width(*rhs))
                    };
                    let mask = final_limb_mask(width);
                    let (lhs, rhs, dst) = state.slot_refs_mut(*lhs, *rhs, instr.dst);
                    dst.prepare(instr.ty.width, 1, lanes);
                    if *signed {
                        simd_words::lt_s_into(
                            lhs.one_limb_words(lanes),
                            rhs.one_limb_words(lanes),
                            width,
                            mask,
                            &mut dst.words,
                        );
                    } else {
                        simd_words::lt_u_into(
                            lhs.one_limb_words(lanes),
                            rhs.one_limb_words(lanes),
                            mask,
                            &mut dst.words,
                        );
                    }
                    dst.mask_final_limb(lanes);
                    state.stats.record_one_limb_op();
                } else if state.value(*lhs).is_at_most_two_limb()
                    && state.value(*rhs).is_at_most_two_limb()
                    && cache.value_width(*lhs) <= 64
                    && cache.value_width(*rhs) <= 64
                    && (!*signed || cache.value_width(*lhs) > 0)
                {
                    self.eval_u64_lt_into(instr, state, *lhs, *rhs, cache, *signed);
                } else {
                    let reason = if *signed {
                        SimdFallbackReason::SignedLt
                    } else {
                        SimdFallbackReason::WideOp
                    };
                    self.eval_simd_instr_scalar_into(instr, state, cache, reason);
                }
            }
            PackedInstrKind::Slice { value, lsb } => {
                if state.value(*value).is_one_limb()
                    && instr.ty.width <= 32
                    && lsb.checked_add(instr.ty.width).is_some_and(|end| end <= 32)
                {
                    let lanes = state.lanes;
                    let mask = final_limb_mask(instr.ty.width);
                    let (value, dst) = state.slot_ref_mut(*value, instr.dst);
                    dst.prepare(instr.ty.width, 1, lanes);
                    simd_words::slice_into(value.one_limb_words(lanes), *lsb, mask, &mut dst.words);
                    dst.mask_final_limb(lanes);
                    state.stats.record_one_limb_op();
                } else if state.value(*value).is_at_most_two_limb()
                    && instr.ty.width <= 64
                    && lsb.checked_add(instr.ty.width).is_some_and(|end| end <= 64)
                {
                    self.eval_u64_slice_into(instr, state, *value, *lsb);
                } else {
                    self.eval_simd_instr_scalar_into(
                        instr,
                        state,
                        cache,
                        SimdFallbackReason::WideOp,
                    );
                }
            }
            PackedInstrKind::Concat(values) => {
                if instr.ty.width <= 32
                    && values.iter().all(|value| state.value(*value).is_one_limb())
                    && values
                        .iter()
                        .try_fold(0u32, |total, value| {
                            total.checked_add(cache.value_width(*value))
                        })
                        .is_some_and(|width| width <= 32)
                {
                    state.concat_one_limb_into(values, instr.dst, instr.ty.width, cache);
                    state.stats.record_one_limb_op();
                } else if instr.ty.width <= 64
                    && values
                        .iter()
                        .all(|value| state.value(*value).is_at_most_two_limb())
                    && values
                        .iter()
                        .try_fold(0u32, |total, value| {
                            total.checked_add(cache.value_width(*value))
                        })
                        .is_some_and(|width| width <= 64)
                {
                    state.concat_u64_into(values, instr.dst, instr.ty.width, cache);
                    state.stats.record_two_limb_op();
                } else {
                    self.eval_simd_instr_scalar_into(
                        instr,
                        state,
                        cache,
                        SimdFallbackReason::WideConcat,
                    );
                }
            }
            PackedInstrKind::Sext(value) => {
                let src_width = cache.value_width(*value);
                if src_width <= 32 && instr.ty.width <= 32 && state.value(*value).is_one_limb() {
                    let lanes = state.lanes;
                    let mask = final_limb_mask(instr.ty.width);
                    let (value, dst) = state.slot_ref_mut(*value, instr.dst);
                    dst.prepare(instr.ty.width, 1, lanes);
                    simd_words::sext_into(
                        value.one_limb_words(lanes),
                        src_width,
                        mask,
                        &mut dst.words,
                    );
                    dst.mask_final_limb(lanes);
                    state.stats.record_one_limb_op();
                } else if src_width <= 64
                    && instr.ty.width <= 64
                    && state.value(*value).is_at_most_two_limb()
                    && src_width > 0
                {
                    self.eval_u64_sext_into(instr, state, *value, src_width);
                } else {
                    self.eval_simd_instr_scalar_into(instr, state, cache, SimdFallbackReason::Sext);
                }
            }
            PackedInstrKind::MemRead { memory, addr } => {
                let mem = &self.inner.program.memories[*memory];
                if state.value(*addr).is_at_most_two_limb()
                    && cache.value_width(*addr) <= 64
                    && mem.data_layout.limbs <= 2
                    && instr.ty.width <= 64
                {
                    self.eval_simd_mem_read_u64_into(instr, state, *memory, *addr);
                } else {
                    self.eval_simd_instr_scalar_into(
                        instr,
                        state,
                        cache,
                        SimdFallbackReason::MemRead,
                    );
                }
            }
        }
    }

    fn eval_simd_mem_read_u64_into(
        &self,
        instr: &PackedInstr,
        state: &mut SimdBlockState,
        memory: usize,
        addr: PackedValueId,
    ) {
        let lanes = state.lanes;
        let mem = &self.inner.program.memories[memory];
        let (addr, dst) = state.slot_ref_mut(addr, instr.dst);
        dst.prepare(instr.ty.width, mem.data_layout.limbs, lanes);
        for lane in 0..lanes {
            let read_addr = addr.decode_usize_lane(lane);
            if read_addr < mem.depth {
                for limb in 0..mem.data_layout.limbs {
                    dst.words[limb * lanes + lane] =
                        self.inner.memories[self.inner.memory_index(mem, lane, read_addr, limb)];
                }
            } else {
                for limb in 0..mem.data_layout.limbs {
                    dst.words[limb * lanes + lane] = 0;
                }
            }
        }
        dst.mask_final_limb(lanes);
        state.stats.record_two_limb_memory_read();
    }

    fn eval_u64_mux_into(
        &self,
        instr: &PackedInstr,
        state: &mut SimdBlockState,
        cond: PackedValueId,
        then_value: PackedValueId,
        else_value: PackedValueId,
    ) {
        let lanes = state.lanes;
        let mask = mask_u64(instr.ty.width);
        let (cond, then_value, else_value, dst) =
            state.slot3_refs_mut(cond, then_value, else_value, instr.dst);
        dst.prepare(instr.ty.width, limbs(instr.ty.width), lanes);
        let mut native = false;
        if dst.limbs == 2 {
            native = simd_words::two_limb_mux_into(
                cond.one_limb_words(lanes),
                &then_value.words,
                &else_value.words,
                lanes,
                then_value.limbs,
                else_value.limbs,
                final_limb_mask(instr.ty.width - 32),
                &mut dst.words,
            );
        } else {
            for lane in 0..lanes {
                let value = if cond.word(0, lane, lanes) & 1 != 0 {
                    then_value.lane_u64(lane, lanes)
                } else {
                    else_value.lane_u64(lane, lanes)
                };
                dst.set_lane_u64(lane, value & mask);
            }
        }
        dst.mask_final_limb(lanes);
        state.stats.record_two_limb_mux_op();
        if native {
            state.stats.record_native_two_limb_op();
        }
    }

    fn eval_u64_not_into(
        &self,
        instr: &PackedInstr,
        state: &mut SimdBlockState,
        value: PackedValueId,
    ) {
        let lanes = state.lanes;
        let mask = mask_u64(instr.ty.width);
        let (value, dst) = state.slot_ref_mut(value, instr.dst);
        dst.prepare(instr.ty.width, limbs(instr.ty.width), lanes);
        let mut native = false;
        if dst.limbs == 2 {
            native = simd_words::two_limb_not_into(
                &value.words,
                lanes,
                value.limbs,
                final_limb_mask(instr.ty.width - 32),
                &mut dst.words,
            );
        } else {
            for lane in 0..lanes {
                dst.set_lane_u64(lane, !value.lane_u64(lane, lanes) & mask);
            }
        }
        dst.mask_final_limb(lanes);
        state.stats.record_two_limb_op();
        if native {
            state.stats.record_native_two_limb_op();
        }
    }

    fn eval_u64_binop_into(
        &self,
        instr: &PackedInstr,
        state: &mut SimdBlockState,
        lhs: PackedValueId,
        rhs: PackedValueId,
        cache: &PackedBlockCache,
        one_limb_op: fn(&[u32], &[u32], u32, &mut [u32]),
        op: impl Fn(u64, u64) -> u64,
        allow_two_limb: bool,
        is_mul: bool,
        two_limb_op: Option<simd_words::TwoLimbBinOp>,
    ) {
        if state.value(lhs).is_one_limb() && state.value(rhs).is_one_limb() && instr.ty.width <= 32
        {
            let lanes = state.lanes;
            let mask = final_limb_mask(instr.ty.width);
            let (lhs, rhs, dst) = state.slot_refs_mut(lhs, rhs, instr.dst);
            dst.prepare(instr.ty.width, 1, lanes);
            one_limb_op(
                lhs.one_limb_words(lanes),
                rhs.one_limb_words(lanes),
                mask,
                &mut dst.words,
            );
            dst.mask_final_limb(lanes);
            state.stats.record_one_limb_op();
        } else if allow_two_limb
            && state.value(lhs).is_at_most_two_limb()
            && state.value(rhs).is_at_most_two_limb()
            && cache.value_width(lhs) <= 64
            && cache.value_width(rhs) <= 64
            && instr.ty.width <= 64
        {
            let lanes = state.lanes;
            let mask = mask_u64(instr.ty.width);
            let (lhs, rhs, dst) = state.slot_refs_mut(lhs, rhs, instr.dst);
            dst.prepare(instr.ty.width, limbs(instr.ty.width), lanes);
            let mut native = false;
            if dst.limbs == 2 {
                if let Some(two_limb_op) = two_limb_op {
                    native = simd_words::two_limb_binop_into(
                        &lhs.words,
                        &rhs.words,
                        lanes,
                        lhs.limbs,
                        rhs.limbs,
                        final_limb_mask(instr.ty.width - 32),
                        &mut dst.words,
                        two_limb_op,
                    );
                }
            }
            if !native {
                for lane in 0..lanes {
                    dst.set_lane_u64(
                        lane,
                        op(lhs.lane_u64(lane, lanes), rhs.lane_u64(lane, lanes)) & mask,
                    );
                }
            }
            dst.mask_final_limb(lanes);
            if is_mul {
                state.stats.record_two_limb_mul_op();
            } else {
                state.stats.record_two_limb_op();
                if native {
                    state.stats.record_native_two_limb_op();
                }
            }
        } else {
            self.eval_simd_instr_scalar_into(instr, state, cache, SimdFallbackReason::WideOp);
        }
    }

    fn eval_u64_mask_into(
        &self,
        instr: &PackedInstr,
        state: &mut SimdBlockState,
        value: PackedValueId,
    ) {
        let lanes = state.lanes;
        let mask = mask_u64(instr.ty.width);
        let (value, dst) = state.slot_ref_mut(value, instr.dst);
        dst.prepare(instr.ty.width, limbs(instr.ty.width), lanes);
        let mut native = false;
        if dst.limbs == 2 {
            native = simd_words::two_limb_mask_into(
                &value.words,
                lanes,
                value.limbs,
                final_limb_mask(instr.ty.width - 32),
                &mut dst.words,
            );
        } else {
            for lane in 0..lanes {
                dst.set_lane_u64(lane, value.lane_u64(lane, lanes) & mask);
            }
        }
        dst.mask_final_limb(lanes);
        state.stats.record_two_limb_op();
        if native {
            state.stats.record_native_two_limb_op();
        }
    }

    fn eval_u64_compare_into(
        &self,
        instr: &PackedInstr,
        state: &mut SimdBlockState,
        lhs: PackedValueId,
        rhs: PackedValueId,
        cache: &PackedBlockCache,
        op: simd_words::TwoLimbCompareOp,
    ) {
        let lanes = state.lanes;
        let width = cache.value_width(lhs).max(cache.value_width(rhs));
        let low_mask = if width >= 32 {
            u32::MAX
        } else {
            final_limb_mask(width)
        };
        let high_mask = if width > 32 {
            final_limb_mask(width - 32)
        } else {
            0
        };
        let (lhs, rhs, dst) = state.slot_refs_mut(lhs, rhs, instr.dst);
        dst.prepare(instr.ty.width, 1, lanes);
        let native = simd_words::two_limb_compare_into(
            &lhs.words,
            &rhs.words,
            lanes,
            lhs.limbs,
            rhs.limbs,
            low_mask,
            high_mask,
            &mut dst.words,
            op,
        );
        state.stats.record_two_limb_op();
        if native {
            state.stats.record_native_two_limb_op();
        }
    }

    fn eval_u64_lt_into(
        &self,
        instr: &PackedInstr,
        state: &mut SimdBlockState,
        lhs: PackedValueId,
        rhs: PackedValueId,
        cache: &PackedBlockCache,
        signed: bool,
    ) {
        let lanes = state.lanes;
        let width = if signed {
            cache.value_width(lhs)
        } else {
            cache.value_width(lhs).max(cache.value_width(rhs))
        };
        let low_mask = if width >= 32 {
            u32::MAX
        } else {
            final_limb_mask(width)
        };
        let high_mask = if width > 32 {
            final_limb_mask(width - 32)
        } else {
            0
        };
        let (lhs, rhs, dst) = state.slot_refs_mut(lhs, rhs, instr.dst);
        dst.prepare(instr.ty.width, 1, lanes);
        let native = simd_words::two_limb_lt_into(
            &lhs.words,
            &rhs.words,
            lanes,
            lhs.limbs,
            rhs.limbs,
            low_mask,
            high_mask,
            signed.then_some(width),
            &mut dst.words,
        );
        state.stats.record_two_limb_op();
        if native {
            state.stats.record_native_two_limb_op();
        }
    }

    fn eval_u64_pass_into(
        &self,
        instr: &PackedInstr,
        state: &mut SimdBlockState,
        value: PackedValueId,
    ) {
        self.eval_u64_mask_into(instr, state, value);
    }

    fn eval_u64_slice_into(
        &self,
        instr: &PackedInstr,
        state: &mut SimdBlockState,
        value: PackedValueId,
        lsb: Width,
    ) {
        let lanes = state.lanes;
        let (value, dst) = state.slot_ref_mut(value, instr.dst);
        dst.prepare(instr.ty.width, limbs(instr.ty.width), lanes);
        let high_mask = if dst.limbs >= 2 {
            final_limb_mask(instr.ty.width - 32)
        } else {
            0
        };
        let native = simd_words::two_limb_slice_into(
            &value.words,
            lanes,
            value.limbs,
            lsb,
            dst.limbs,
            high_mask,
            &mut dst.words,
        );
        dst.mask_final_limb(lanes);
        state.stats.record_two_limb_op();
        if native {
            state.stats.record_native_two_limb_op();
        }
    }

    fn eval_u64_sext_into(
        &self,
        instr: &PackedInstr,
        state: &mut SimdBlockState,
        value: PackedValueId,
        src_width: Width,
    ) {
        let lanes = state.lanes;
        let (value, dst) = state.slot_ref_mut(value, instr.dst);
        dst.prepare(instr.ty.width, limbs(instr.ty.width), lanes);
        let high_mask = if dst.limbs >= 2 {
            final_limb_mask(instr.ty.width - 32)
        } else {
            0
        };
        let native = simd_words::two_limb_sext_into(
            &value.words,
            lanes,
            value.limbs,
            src_width,
            dst.limbs,
            high_mask,
            &mut dst.words,
        );
        dst.mask_final_limb(lanes);
        state.stats.record_two_limb_op();
        if native {
            state.stats.record_native_two_limb_op();
        }
    }

    fn eval_simd_instr_scalar_into(
        &self,
        instr: &PackedInstr,
        state: &mut SimdBlockState,
        cache: &PackedBlockCache,
        reason: SimdFallbackReason,
    ) {
        let lanes = self.inner.lanes;
        state
            .slot_mut(instr.dst)
            .prepare(instr.ty.width, limbs(instr.ty.width), lanes);
        let mut scratch = std::mem::take(&mut state.scalar_scratch);
        for lane in 0..lanes {
            self.eval_simd_instr_scalar_lane_into(instr, state, cache, lane, &mut scratch);
            state.slot_mut(instr.dst).set_lane(lane, &scratch);
            state.stats.lane_materializations += 1;
        }
        state.scalar_scratch = scratch;
        state.stats.fallback_instrs += 1;
        state.stats.fallback_reasons.record(reason);
    }

    fn eval_simd_instr_scalar_lane_into(
        &self,
        instr: &PackedInstr,
        state: &SimdBlockState,
        cache: &PackedBlockCache,
        lane: usize,
        out: &mut Vec<u32>,
    ) {
        out.clear();
        out.extend(self.eval_simd_instr_scalar_lane(instr, state, cache, lane));
    }

    fn eval_simd_instr_scalar_lane(
        &self,
        instr: &PackedInstr,
        state: &SimdBlockState,
        cache: &PackedBlockCache,
        lane: usize,
    ) -> Vec<u32> {
        let value = match &instr.kind {
            PackedInstrKind::Lit(value) => value.clone(),
            PackedInstrKind::Signal(signal) => self
                .inner
                .load_layout(self.inner.program.signals[*signal].layout, lane),
            PackedInstrKind::Not(value) => {
                bit_not(self.simd_lane_value(state, *value, lane), instr.ty)
            }
            PackedInstrKind::And(lhs, rhs) => bit_binop(
                self.simd_lane_value(state, *lhs, lane),
                self.simd_lane_value(state, *rhs, lane),
                instr.ty,
                |lhs, rhs| lhs & rhs,
            ),
            PackedInstrKind::Or(lhs, rhs) => bit_binop(
                self.simd_lane_value(state, *lhs, lane),
                self.simd_lane_value(state, *rhs, lane),
                instr.ty,
                |lhs, rhs| lhs | rhs,
            ),
            PackedInstrKind::Xor(lhs, rhs) => bit_binop(
                self.simd_lane_value(state, *lhs, lane),
                self.simd_lane_value(state, *rhs, lane),
                instr.ty,
                |lhs, rhs| lhs ^ rhs,
            ),
            PackedInstrKind::Add(lhs, rhs) => add_limbs(
                self.simd_lane_value(state, *lhs, lane),
                self.simd_lane_value(state, *rhs, lane),
                instr.ty,
            ),
            PackedInstrKind::Sub(lhs, rhs) => sub_limbs(
                self.simd_lane_value(state, *lhs, lane),
                self.simd_lane_value(state, *rhs, lane),
                instr.ty,
            ),
            PackedInstrKind::Mul(lhs, rhs) => mul_limbs(
                self.simd_lane_value(state, *lhs, lane),
                self.simd_lane_value(state, *rhs, lane),
                instr.ty,
            ),
            PackedInstrKind::Eq(lhs, rhs) => encode_bool(eq_values(
                self.simd_lane_value(state, *lhs, lane),
                self.simd_lane_value(state, *rhs, lane),
                cache.value_width(*lhs).max(cache.value_width(*rhs)),
            )),
            PackedInstrKind::Ne(lhs, rhs) => encode_bool(!eq_values(
                self.simd_lane_value(state, *lhs, lane),
                self.simd_lane_value(state, *rhs, lane),
                cache.value_width(*lhs).max(cache.value_width(*rhs)),
            )),
            PackedInstrKind::Lt { lhs, rhs, signed } => encode_bool(lt_values(
                self.simd_lane_value(state, *lhs, lane),
                self.simd_lane_value(state, *rhs, lane),
                cache.value_width(*lhs),
                *signed,
            )),
            PackedInstrKind::Mux {
                cond,
                then_value,
                else_value,
            } => {
                if decode_bool(&self.simd_lane_value(state, *cond, lane)) {
                    self.simd_lane_value(state, *then_value, lane)
                } else {
                    self.simd_lane_value(state, *else_value, lane)
                }
            }
            PackedInstrKind::Slice { value, lsb } => {
                slice_limbs(self.simd_lane_value(state, *value, lane), *lsb, instr.ty)
            }
            PackedInstrKind::Zext(value)
            | PackedInstrKind::Trunc(value)
            | PackedInstrKind::Cast(value) => {
                fit_limbs(self.simd_lane_value(state, *value, lane), instr.ty)
            }
            PackedInstrKind::Sext(value) => sext_limbs(
                self.simd_lane_value(state, *value, lane),
                cache.value_width(*value),
                instr.ty,
            ),
            PackedInstrKind::Concat(values) => concat_limbs(
                values
                    .iter()
                    .map(|value| {
                        (
                            self.simd_lane_value(state, *value, lane),
                            cache.value_width(*value),
                        )
                    })
                    .collect(),
                instr.ty,
            ),
            PackedInstrKind::MemRead { memory, addr } => {
                let addr = decode_usize(&self.simd_lane_value(state, *addr, lane));
                let mem = &self.inner.program.memories[*memory];
                if addr < mem.depth {
                    self.inner.load_memory(mem, lane, addr)
                } else {
                    vec![0; limbs(instr.ty.width)]
                }
            }
        };
        fit_limbs(value, instr.ty)
    }

    fn simd_lane_value(
        &self,
        state: &SimdBlockState,
        value: PackedValueId,
        lane: usize,
    ) -> Vec<u32> {
        state.value(value).lane(lane)
    }

    fn store_signal_value(&mut self, signal: usize, value: &SimdValueSlot) {
        let layout = self.inner.program.signals[signal].layout;
        for lane in 0..self.inner.lanes {
            self.store_layout_slot_lane(layout, lane, value);
        }
    }

    fn store_signal_slot_lane(&mut self, signal: usize, lane: usize, value: &SimdValueSlot) {
        let layout = self.inner.program.signals[signal].layout;
        self.store_layout_slot_lane(layout, lane, value);
    }

    fn store_layout_slot_lane(
        &mut self,
        layout: PackedValueLayout,
        lane: usize,
        value: &SimdValueSlot,
    ) {
        let lanes = self.inner.lanes;
        let final_limb = layout.limbs.saturating_sub(1);
        for limb in 0..layout.limbs {
            let mut word = value.word(limb, lane, lanes);
            if limb == final_limb {
                word &= final_limb_mask(layout.width);
            }
            let index = self.inner.value_index(layout.offset, limb, lane);
            self.inner.values[index] = word;
        }
    }

    fn store_memory_slot_lane(
        &mut self,
        memory: usize,
        lane: usize,
        addr: usize,
        value: &SimdValueSlot,
    ) {
        let memory = self.inner.program.memories[memory].clone();
        let layout = memory.data_layout;
        let lanes = self.inner.lanes;
        let final_limb = layout.limbs.saturating_sub(1);
        for limb in 0..layout.limbs {
            let mut word = value.word(limb, lane, lanes);
            if limb == final_limb {
                word &= final_limb_mask(layout.width);
            }
            let index = self.inner.memory_index(&memory, lane, addr, limb);
            self.inner.memories[index] = word;
        }
    }
}

impl SimdExecutionWorkspaces {
    fn new(execution: &PackedExecutionCache, lanes: usize) -> Self {
        Self {
            async_reset_comb: SimdBlockState::new(&execution.async_reset_comb, lanes),
            comb: SimdBlockState::new(&execution.comb, lanes),
            tick_next: SimdBlockState::new(&execution.tick_next, lanes),
            tick_commit: SimdBlockState::new(&execution.tick_commit, lanes),
        }
    }
}

impl PackedExecutionWorkspaces {
    fn new(execution: &PackedExecutionCache, lanes: usize) -> Self {
        Self {
            async_reset_comb: PackedBlockState::new(&execution.async_reset_comb, lanes),
            comb: PackedBlockState::new(&execution.comb, lanes),
            tick_next: PackedBlockState::new(&execution.tick_next, lanes),
            tick_commit: PackedBlockState::new(&execution.tick_commit, lanes),
        }
    }
}

impl PackedBlockState {
    fn new(cache: &PackedBlockCache, lanes: usize) -> Self {
        let mut state = Self::default();
        state.reset(cache, lanes);
        state
    }

    fn reset(&mut self, cache: &PackedBlockCache, lanes: usize) {
        self.lanes = lanes;
        if self.slots.len() != cache.value_types.len() {
            self.slots.clear();
            self.slots
                .resize_with(cache.value_types.len(), || SimdValueSlot {
                    limbs: 1,
                    width: 1,
                    words: vec![0; lanes],
                    valid: false,
                });
        }
        for (slot, ty) in self.slots.iter_mut().zip(&cache.value_types) {
            let ty = ty.unwrap_or_else(|| BitType::new(1, Signedness::Unsigned));
            let slot_limbs = limbs(ty.width);
            slot.width = ty.width;
            slot.limbs = slot_limbs;
            slot.valid = false;
            let len = slot_limbs * lanes;
            if slot.words.len() != len {
                slot.words.resize(len, 0);
            }
        }
    }

    fn value(&self, value: PackedValueId) -> &SimdValueSlot {
        &self.slots[value.0]
    }

    fn slot_mut(&mut self, value: PackedValueId) -> &mut SimdValueSlot {
        &mut self.slots[value.0]
    }
}

impl SimdBlockState {
    fn new(cache: &PackedBlockCache, lanes: usize) -> Self {
        let mut state = Self::default();
        state.reset(cache, lanes);
        state.stats = SimdBlockStats::default();
        state
    }

    fn reset(&mut self, cache: &PackedBlockCache, lanes: usize) {
        let reused = self.lanes == lanes && self.slots.len() == cache.value_types.len();
        self.lanes = lanes;
        self.stats = SimdBlockStats::default();
        if reused {
            self.stats.state_reuses += 1;
        }
        if self.slots.len() != cache.value_types.len() {
            self.slots.clear();
            self.slots
                .resize_with(cache.value_types.len(), || SimdValueSlot {
                    limbs: 1,
                    width: 1,
                    words: vec![0; lanes],
                    valid: false,
                });
        }
        for (slot, ty) in self.slots.iter_mut().zip(&cache.value_types) {
            let ty = ty.unwrap_or_else(|| BitType::new(1, Signedness::Unsigned));
            let slot_limbs = limbs(ty.width);
            slot.width = ty.width;
            slot.limbs = slot_limbs;
            slot.valid = false;
            let len = slot_limbs * lanes;
            if slot.words.len() != len {
                slot.words.resize(len, 0);
            }
        }
    }

    fn value(&self, value: PackedValueId) -> &SimdValueSlot {
        &self.slots[value.0]
    }

    fn slot_mut(&mut self, value: PackedValueId) -> &mut SimdValueSlot {
        &mut self.slots[value.0]
    }

    fn slot_ref_mut(
        &mut self,
        value: PackedValueId,
        dst: PackedValueId,
    ) -> (&SimdValueSlot, &mut SimdValueSlot) {
        debug_assert_ne!(value.0, dst.0);
        let slots = self.slots.as_mut_ptr();
        // Machine values are SSA temporaries, so an instruction never writes to one of its inputs.
        unsafe { (&*slots.add(value.0), &mut *slots.add(dst.0)) }
    }

    fn slot_refs_mut(
        &mut self,
        lhs: PackedValueId,
        rhs: PackedValueId,
        dst: PackedValueId,
    ) -> (&SimdValueSlot, &SimdValueSlot, &mut SimdValueSlot) {
        debug_assert_ne!(lhs.0, dst.0);
        debug_assert_ne!(rhs.0, dst.0);
        let slots = self.slots.as_mut_ptr();
        // Source/source aliasing is allowed; only the destination must be distinct.
        unsafe {
            (
                &*slots.add(lhs.0),
                &*slots.add(rhs.0),
                &mut *slots.add(dst.0),
            )
        }
    }

    fn slot3_refs_mut(
        &mut self,
        a: PackedValueId,
        b: PackedValueId,
        c: PackedValueId,
        dst: PackedValueId,
    ) -> (
        &SimdValueSlot,
        &SimdValueSlot,
        &SimdValueSlot,
        &mut SimdValueSlot,
    ) {
        debug_assert_ne!(a.0, dst.0);
        debug_assert_ne!(b.0, dst.0);
        debug_assert_ne!(c.0, dst.0);
        let slots = self.slots.as_mut_ptr();
        // Source/source aliasing is allowed; only the destination must be distinct.
        unsafe {
            (
                &*slots.add(a.0),
                &*slots.add(b.0),
                &*slots.add(c.0),
                &mut *slots.add(dst.0),
            )
        }
    }

    fn concat_one_limb_into(
        &mut self,
        values: &[PackedValueId],
        dst: PackedValueId,
        width: Width,
        cache: &PackedBlockCache,
    ) {
        debug_assert!(values.iter().all(|value| value.0 != dst.0));
        let lanes = self.lanes;
        let mask = final_limb_mask(width);
        let slots = self.slots.as_mut_ptr();
        unsafe {
            let dst_slot = &mut *slots.add(dst.0);
            dst_slot.prepare(width, 1, lanes);
            for lane in 0..lanes {
                let mut word = 0u32;
                let mut offset = 0;
                for value in values.iter().rev() {
                    let part_width = cache.value_width(*value);
                    let part =
                        (*slots.add(value.0)).word(0, lane, lanes) & final_limb_mask(part_width);
                    if offset < 32 {
                        word |= part << offset;
                    }
                    offset += part_width;
                }
                dst_slot.words[lane] = word & mask;
            }
        }
    }

    fn concat_u64_into(
        &mut self,
        values: &[PackedValueId],
        dst: PackedValueId,
        width: Width,
        cache: &PackedBlockCache,
    ) {
        debug_assert!(values.iter().all(|value| value.0 != dst.0));
        let lanes = self.lanes;
        let mask = mask_u64(width);
        let slots = self.slots.as_mut_ptr();
        unsafe {
            let dst_slot = &mut *slots.add(dst.0);
            dst_slot.prepare(width, limbs(width), lanes);
            for lane in 0..lanes {
                let mut word = 0u64;
                let mut offset = 0;
                for value in values.iter().rev() {
                    let part_width = cache.value_width(*value);
                    let part = (*slots.add(value.0)).lane_u64(lane, lanes) & mask_u64(part_width);
                    if offset < 64 {
                        word |= part << offset;
                    }
                    offset += part_width;
                }
                dst_slot.set_lane_u64(lane, word & mask);
            }
            dst_slot.mask_final_limb(lanes);
        }
    }
}

impl SimdValueSlot {
    fn prepare(&mut self, width: Width, limbs: usize, lanes: usize) {
        self.width = width;
        self.limbs = limbs;
        self.valid = true;
        let len = limbs * lanes;
        if self.words.len() != len {
            self.words.resize(len, 0);
        }
    }

    fn fill_lit(&mut self, value: &[u32], width: Width, lanes: usize) {
        let limbs = limbs(width);
        self.prepare(width, limbs, lanes);
        let mut value = value.to_vec();
        mask_to_width(&mut value, width);
        for limb in 0..limbs {
            let word = value.get(limb).copied().unwrap_or(0);
            for lane in 0..lanes {
                self.words[limb * lanes + lane] = word;
            }
        }
    }

    fn load_signal(&mut self, sim: &PackedSimulator, layout: PackedValueLayout, lanes: usize) {
        self.prepare(layout.width, layout.limbs, lanes);
        for limb in 0..layout.limbs {
            for lane in 0..lanes {
                self.words[limb * lanes + lane] =
                    sim.values[sim.value_index(layout.offset, limb, lane)];
            }
        }
        self.mask_final_limb(lanes);
    }

    fn is_one_limb(&self) -> bool {
        self.valid && self.limbs == 1
    }

    fn is_at_most_two_limb(&self) -> bool {
        self.valid && self.limbs <= 2
    }

    fn one_limb_words(&self, lanes: usize) -> &[u32] {
        &self.words[..lanes]
    }

    fn word(&self, limb: usize, lane: usize, lanes: usize) -> u32 {
        if !self.valid || limb >= self.limbs {
            0
        } else {
            self.words[limb * lanes + lane]
        }
    }

    fn lane_u64(&self, lane: usize, lanes: usize) -> u64 {
        self.word(0, lane, lanes) as u64 | ((self.word(1, lane, lanes) as u64) << 32)
    }

    fn decode_bool_lane(&self, lane: usize) -> bool {
        self.word(0, lane, self.lanes()) & 1 != 0
    }

    fn decode_usize_lane(&self, lane: usize) -> usize {
        let lanes = self.lanes();
        let take_limbs = (usize::BITS.div_ceil(32) as usize).min(self.limbs);
        let mut out = 0usize;
        for limb in 0..take_limbs {
            out |= (self.word(limb, lane, lanes) as usize) << (limb * 32);
        }
        if (take_limbs..self.limbs).any(|limb| self.word(limb, lane, lanes) != 0) {
            usize::MAX
        } else {
            out
        }
    }

    fn copy_lane_into(&self, lane: usize, out: &mut Vec<u32>) {
        out.clear();
        if !self.valid {
            out.push(0);
            return;
        }
        let lanes = self.lanes();
        out.extend((0..self.limbs).map(|limb| self.words[limb * lanes + lane]));
    }

    fn lane(&self, lane: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.limbs.max(1));
        self.copy_lane_into(lane, &mut out);
        out
    }

    fn set_lane(&mut self, lane: usize, value: &[u32]) {
        let lanes = self.lanes();
        let final_limb = self.limbs.saturating_sub(1);
        for limb in 0..self.limbs {
            let mut word = value.get(limb).copied().unwrap_or(0);
            if limb == final_limb {
                word &= final_limb_mask(self.width);
            }
            self.words[limb * lanes + lane] = word;
        }
        self.valid = true;
    }

    fn set_lane_u64(&mut self, lane: usize, value: u64) {
        let lanes = self.lanes();
        if self.limbs >= 1 {
            self.words[lane] = value as u32;
        }
        if self.limbs >= 2 {
            self.words[lanes + lane] = (value >> 32) as u32;
        }
    }

    fn mask_final_limb(&mut self, lanes: usize) {
        if self.limbs == 0 {
            return;
        }
        let mask = final_limb_mask(self.width);
        let start = (self.limbs - 1) * lanes;
        for lane in 0..lanes {
            self.words[start + lane] &= mask;
        }
    }

    fn lanes(&self) -> usize {
        if self.limbs == 0 {
            0
        } else {
            self.words.len() / self.limbs
        }
    }
}

mod simd_words {
    #[derive(Clone, Copy)]
    pub enum TwoLimbBinOp {
        And,
        Or,
        Xor,
        Add,
        Sub,
    }

    #[derive(Clone, Copy)]
    pub enum TwoLimbCompareOp {
        Eq,
        Ne,
    }

    fn optional_high_limb(input: &[u32], limbs: usize, lanes: usize) -> Option<&[u32]> {
        (limbs >= 2 && input.len() >= lanes * 2).then(|| &input[lanes..lanes * 2])
    }

    pub fn mask_into(input: &[u32], mask: u32, out: &mut [u32]) {
        for (out, input) in out.iter_mut().zip(input) {
            *out = *input & mask;
        }
    }

    pub fn not_into(input: &[u32], mask: u32, out: &mut [u32]) {
        for (out, input) in out.iter_mut().zip(input) {
            *out = !*input & mask;
        }
    }

    pub fn and_into(lhs: &[u32], rhs: &[u32], mask: u32, out: &mut [u32]) {
        #[cfg(target_arch = "aarch64")]
        {
            aarch64_binop_into(lhs, rhs, mask, out, Aarch64BinOp::And);
        }
        #[cfg(target_arch = "x86_64")]
        {
            x86_binop_into(lhs, rhs, mask, out, X86BinOp::And);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            scalar_binop_into(lhs, rhs, mask, out, |lhs, rhs| lhs & rhs);
        }
    }

    pub fn or_into(lhs: &[u32], rhs: &[u32], mask: u32, out: &mut [u32]) {
        #[cfg(target_arch = "aarch64")]
        {
            aarch64_binop_into(lhs, rhs, mask, out, Aarch64BinOp::Or);
        }
        #[cfg(target_arch = "x86_64")]
        {
            x86_binop_into(lhs, rhs, mask, out, X86BinOp::Or);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            scalar_binop_into(lhs, rhs, mask, out, |lhs, rhs| lhs | rhs);
        }
    }

    pub fn xor_into(lhs: &[u32], rhs: &[u32], mask: u32, out: &mut [u32]) {
        #[cfg(target_arch = "aarch64")]
        {
            aarch64_binop_into(lhs, rhs, mask, out, Aarch64BinOp::Xor);
        }
        #[cfg(target_arch = "x86_64")]
        {
            x86_binop_into(lhs, rhs, mask, out, X86BinOp::Xor);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            scalar_binop_into(lhs, rhs, mask, out, |lhs, rhs| lhs ^ rhs);
        }
    }

    pub fn add_into(lhs: &[u32], rhs: &[u32], mask: u32, out: &mut [u32]) {
        #[cfg(target_arch = "aarch64")]
        {
            aarch64_binop_into(lhs, rhs, mask, out, Aarch64BinOp::Add);
        }
        #[cfg(target_arch = "x86_64")]
        {
            x86_binop_into(lhs, rhs, mask, out, X86BinOp::Add);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            scalar_binop_into(lhs, rhs, mask, out, |lhs, rhs| lhs.wrapping_add(rhs));
        }
    }

    pub fn sub_into(lhs: &[u32], rhs: &[u32], mask: u32, out: &mut [u32]) {
        #[cfg(target_arch = "aarch64")]
        {
            aarch64_binop_into(lhs, rhs, mask, out, Aarch64BinOp::Sub);
        }
        #[cfg(target_arch = "x86_64")]
        {
            x86_binop_into(lhs, rhs, mask, out, X86BinOp::Sub);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            scalar_binop_into(lhs, rhs, mask, out, |lhs, rhs| lhs.wrapping_sub(rhs));
        }
    }

    pub fn mul_into(lhs: &[u32], rhs: &[u32], mask: u32, out: &mut [u32]) {
        #[cfg(target_arch = "aarch64")]
        {
            aarch64_binop_into(lhs, rhs, mask, out, Aarch64BinOp::Mul);
        }
        #[cfg(target_arch = "x86_64")]
        {
            x86_binop_into(lhs, rhs, mask, out, X86BinOp::Mul);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            scalar_binop_into(lhs, rhs, mask, out, |lhs, rhs| lhs.wrapping_mul(rhs));
        }
    }

    pub fn eq_into(lhs: &[u32], rhs: &[u32], mask: u32, out: &mut [u32]) {
        for ((out, lhs), rhs) in out.iter_mut().zip(lhs).zip(rhs) {
            *out = u32::from((*lhs & mask) == (*rhs & mask));
        }
    }

    pub fn ne_into(lhs: &[u32], rhs: &[u32], mask: u32, out: &mut [u32]) {
        for ((out, lhs), rhs) in out.iter_mut().zip(lhs).zip(rhs) {
            *out = u32::from((*lhs & mask) != (*rhs & mask));
        }
    }

    pub fn lt_u_into(lhs: &[u32], rhs: &[u32], mask: u32, out: &mut [u32]) {
        for ((out, lhs), rhs) in out.iter_mut().zip(lhs).zip(rhs) {
            *out = u32::from((*lhs & mask) < (*rhs & mask));
        }
    }

    pub fn lt_s_into(lhs: &[u32], rhs: &[u32], width: u32, mask: u32, out: &mut [u32]) {
        let sign_bit = 1u32 << (width - 1);
        for ((out, lhs), rhs) in out.iter_mut().zip(lhs).zip(rhs) {
            let lhs = *lhs & mask;
            let rhs = *rhs & mask;
            let lhs_sign = lhs & sign_bit != 0;
            let rhs_sign = rhs & sign_bit != 0;
            *out = u32::from(if lhs_sign != rhs_sign {
                lhs_sign
            } else {
                lhs < rhs
            });
        }
    }

    pub fn mux_into(cond: &[u32], then_value: &[u32], else_value: &[u32], out: &mut [u32]) {
        for (((out, cond), then_value), else_value) in
            out.iter_mut().zip(cond).zip(then_value).zip(else_value)
        {
            *out = if *cond & 1 != 0 {
                *then_value
            } else {
                *else_value
            };
        }
    }

    pub fn slice_into(input: &[u32], lsb: u32, mask: u32, out: &mut [u32]) {
        for (out, input) in out.iter_mut().zip(input) {
            *out = (*input >> lsb) & mask;
        }
    }

    pub fn sext_into(input: &[u32], src_width: u32, dst_mask: u32, out: &mut [u32]) {
        let src_mask = super::final_limb_mask(src_width);
        let sign_bit = 1u32 << (src_width - 1);
        for (out, input) in out.iter_mut().zip(input) {
            let value = *input & src_mask;
            let extended = if value & sign_bit != 0 {
                value | !src_mask
            } else {
                value
            };
            *out = extended & dst_mask;
        }
    }

    pub fn two_limb_not_into(
        input: &[u32],
        lanes: usize,
        input_limbs: usize,
        high_mask: u32,
        out: &mut [u32],
    ) -> bool {
        let (out_low, out_high) = out.split_at_mut(lanes);
        let input_low = &input[..lanes];
        let input_high = optional_high_limb(input, input_limbs, lanes);
        let native =
            aarch64_two_limb_not_into(input_low, input_high, lanes, high_mask, out_low, out_high);
        if !native {
            scalar_two_limb_not_into(input_low, input_high, lanes, high_mask, out_low, out_high);
        }
        native
    }

    pub fn two_limb_mask_into(
        input: &[u32],
        lanes: usize,
        input_limbs: usize,
        high_mask: u32,
        out: &mut [u32],
    ) -> bool {
        let (out_low, out_high) = out.split_at_mut(lanes);
        let input_low = &input[..lanes];
        let input_high = optional_high_limb(input, input_limbs, lanes);
        let native =
            aarch64_two_limb_mask_into(input_low, input_high, lanes, high_mask, out_low, out_high);
        if !native {
            scalar_two_limb_mask_into(input_low, input_high, lanes, high_mask, out_low, out_high);
        }
        native
    }

    pub fn two_limb_binop_into(
        lhs: &[u32],
        rhs: &[u32],
        lanes: usize,
        lhs_limbs: usize,
        rhs_limbs: usize,
        high_mask: u32,
        out: &mut [u32],
        op: TwoLimbBinOp,
    ) -> bool {
        let (out_low, out_high) = out.split_at_mut(lanes);
        let lhs_low = &lhs[..lanes];
        let rhs_low = &rhs[..lanes];
        let lhs_high = optional_high_limb(lhs, lhs_limbs, lanes);
        let rhs_high = optional_high_limb(rhs, rhs_limbs, lanes);
        let native = aarch64_two_limb_binop_into(
            lhs_low, lhs_high, rhs_low, rhs_high, lanes, high_mask, out_low, out_high, op,
        );
        if !native {
            scalar_two_limb_binop_into(
                lhs_low, lhs_high, rhs_low, rhs_high, lanes, high_mask, out_low, out_high, op,
            );
        }
        native
    }

    pub fn two_limb_mux_into(
        cond: &[u32],
        then_value: &[u32],
        else_value: &[u32],
        lanes: usize,
        then_limbs: usize,
        else_limbs: usize,
        high_mask: u32,
        out: &mut [u32],
    ) -> bool {
        let (out_low, out_high) = out.split_at_mut(lanes);
        let then_low = &then_value[..lanes];
        let else_low = &else_value[..lanes];
        let then_high = optional_high_limb(then_value, then_limbs, lanes);
        let else_high = optional_high_limb(else_value, else_limbs, lanes);
        let native = aarch64_two_limb_mux_into(
            cond, then_low, then_high, else_low, else_high, lanes, high_mask, out_low, out_high,
        );
        if !native {
            scalar_two_limb_mux_into(
                cond, then_low, then_high, else_low, else_high, lanes, high_mask, out_low, out_high,
            );
        }
        native
    }

    pub fn two_limb_compare_into(
        lhs: &[u32],
        rhs: &[u32],
        lanes: usize,
        lhs_limbs: usize,
        rhs_limbs: usize,
        low_mask: u32,
        high_mask: u32,
        out: &mut [u32],
        op: TwoLimbCompareOp,
    ) -> bool {
        let lhs_low = &lhs[..lanes];
        let rhs_low = &rhs[..lanes];
        let lhs_high = optional_high_limb(lhs, lhs_limbs, lanes);
        let rhs_high = optional_high_limb(rhs, rhs_limbs, lanes);
        let native = aarch64_two_limb_compare_into(
            lhs_low, lhs_high, rhs_low, rhs_high, lanes, low_mask, high_mask, out, op,
        );
        if !native {
            scalar_two_limb_compare_into(
                lhs_low, lhs_high, rhs_low, rhs_high, lanes, low_mask, high_mask, out, op,
            );
        }
        native
    }

    pub fn two_limb_lt_into(
        lhs: &[u32],
        rhs: &[u32],
        lanes: usize,
        lhs_limbs: usize,
        rhs_limbs: usize,
        low_mask: u32,
        high_mask: u32,
        signed_width: Option<u32>,
        out: &mut [u32],
    ) -> bool {
        let lhs_low = &lhs[..lanes];
        let rhs_low = &rhs[..lanes];
        let lhs_high = optional_high_limb(lhs, lhs_limbs, lanes);
        let rhs_high = optional_high_limb(rhs, rhs_limbs, lanes);
        let native = aarch64_two_limb_lt_into(
            lhs_low,
            lhs_high,
            rhs_low,
            rhs_high,
            lanes,
            low_mask,
            high_mask,
            signed_width,
            out,
        );
        if !native {
            scalar_two_limb_lt_into(
                lhs_low,
                lhs_high,
                rhs_low,
                rhs_high,
                lanes,
                low_mask,
                high_mask,
                signed_width,
                out,
            );
        }
        native
    }

    pub fn two_limb_slice_into(
        input: &[u32],
        lanes: usize,
        input_limbs: usize,
        lsb: u32,
        out_limbs: usize,
        high_mask: u32,
        out: &mut [u32],
    ) -> bool {
        let (out_low, out_high) = out.split_at_mut(lanes);
        let input_low = &input[..lanes];
        let input_high = optional_high_limb(input, input_limbs, lanes);
        let native = aarch64_two_limb_slice_into(
            input_low, input_high, lanes, lsb, out_limbs, high_mask, out_low, out_high,
        );
        if !native {
            scalar_two_limb_slice_into(
                input_low, input_high, lanes, lsb, out_limbs, high_mask, out_low, out_high,
            );
        }
        native
    }

    pub fn two_limb_sext_into(
        input: &[u32],
        lanes: usize,
        input_limbs: usize,
        src_width: u32,
        out_limbs: usize,
        high_mask: u32,
        out: &mut [u32],
    ) -> bool {
        let (out_low, out_high) = out.split_at_mut(lanes);
        let input_low = &input[..lanes];
        let input_high = optional_high_limb(input, input_limbs, lanes);
        let native = aarch64_two_limb_sext_into(
            input_low, input_high, lanes, src_width, out_limbs, high_mask, out_low, out_high,
        );
        if !native {
            scalar_two_limb_sext_into(
                input_low, input_high, lanes, src_width, out_limbs, high_mask, out_low, out_high,
            );
        }
        native
    }

    fn scalar_two_limb_not_into(
        input_low: &[u32],
        input_high: Option<&[u32]>,
        lanes: usize,
        high_mask: u32,
        out_low: &mut [u32],
        out_high: &mut [u32],
    ) {
        for lane in 0..lanes {
            out_low[lane] = !input_low[lane];
            out_high[lane] = !input_high.map_or(0, |high| high[lane]) & high_mask;
        }
    }

    fn scalar_two_limb_mask_into(
        input_low: &[u32],
        input_high: Option<&[u32]>,
        lanes: usize,
        high_mask: u32,
        out_low: &mut [u32],
        out_high: &mut [u32],
    ) {
        for lane in 0..lanes {
            out_low[lane] = input_low[lane];
            out_high[lane] = input_high.map_or(0, |high| high[lane]) & high_mask;
        }
    }

    fn scalar_two_limb_binop_into(
        lhs_low: &[u32],
        lhs_high: Option<&[u32]>,
        rhs_low: &[u32],
        rhs_high: Option<&[u32]>,
        lanes: usize,
        high_mask: u32,
        out_low: &mut [u32],
        out_high: &mut [u32],
        op: TwoLimbBinOp,
    ) {
        for lane in 0..lanes {
            let lhs_low_word = lhs_low[lane];
            let rhs_low_word = rhs_low[lane];
            let lhs_high_word = lhs_high.map_or(0, |high| high[lane]);
            let rhs_high_word = rhs_high.map_or(0, |high| high[lane]);
            let (low, high) = match op {
                TwoLimbBinOp::And => (lhs_low_word & rhs_low_word, lhs_high_word & rhs_high_word),
                TwoLimbBinOp::Or => (lhs_low_word | rhs_low_word, lhs_high_word | rhs_high_word),
                TwoLimbBinOp::Xor => (lhs_low_word ^ rhs_low_word, lhs_high_word ^ rhs_high_word),
                TwoLimbBinOp::Add => {
                    let (low, carry) = lhs_low_word.overflowing_add(rhs_low_word);
                    (
                        low,
                        lhs_high_word
                            .wrapping_add(rhs_high_word)
                            .wrapping_add(u32::from(carry)),
                    )
                }
                TwoLimbBinOp::Sub => {
                    let (low, borrow) = lhs_low_word.overflowing_sub(rhs_low_word);
                    (
                        low,
                        lhs_high_word
                            .wrapping_sub(rhs_high_word)
                            .wrapping_sub(u32::from(borrow)),
                    )
                }
            };
            out_low[lane] = low;
            out_high[lane] = high & high_mask;
        }
    }

    fn scalar_two_limb_mux_into(
        cond: &[u32],
        then_low: &[u32],
        then_high: Option<&[u32]>,
        else_low: &[u32],
        else_high: Option<&[u32]>,
        lanes: usize,
        high_mask: u32,
        out_low: &mut [u32],
        out_high: &mut [u32],
    ) {
        for lane in 0..lanes {
            if cond[lane] & 1 != 0 {
                out_low[lane] = then_low[lane];
                out_high[lane] = then_high.map_or(0, |high| high[lane]) & high_mask;
            } else {
                out_low[lane] = else_low[lane];
                out_high[lane] = else_high.map_or(0, |high| high[lane]) & high_mask;
            }
        }
    }

    fn scalar_two_limb_compare_into(
        lhs_low: &[u32],
        lhs_high: Option<&[u32]>,
        rhs_low: &[u32],
        rhs_high: Option<&[u32]>,
        lanes: usize,
        low_mask: u32,
        high_mask: u32,
        out: &mut [u32],
        op: TwoLimbCompareOp,
    ) {
        for lane in 0..lanes {
            let equal = (lhs_low[lane] & low_mask) == (rhs_low[lane] & low_mask)
                && (lhs_high.map_or(0, |high| high[lane]) & high_mask)
                    == (rhs_high.map_or(0, |high| high[lane]) & high_mask);
            out[lane] = match op {
                TwoLimbCompareOp::Eq => u32::from(equal),
                TwoLimbCompareOp::Ne => u32::from(!equal),
            };
        }
    }

    fn scalar_two_limb_lt_into(
        lhs_low: &[u32],
        lhs_high: Option<&[u32]>,
        rhs_low: &[u32],
        rhs_high: Option<&[u32]>,
        lanes: usize,
        low_mask: u32,
        high_mask: u32,
        signed_width: Option<u32>,
        out: &mut [u32],
    ) {
        for lane in 0..lanes {
            let lhs_low = lhs_low[lane] & low_mask;
            let rhs_low = rhs_low[lane] & low_mask;
            let lhs_high = lhs_high.map_or(0, |high| high[lane]) & high_mask;
            let rhs_high = rhs_high.map_or(0, |high| high[lane]) & high_mask;
            let lt = if let Some(width) = signed_width {
                let (lhs_sign, rhs_sign) = if width <= 32 {
                    let sign_bit = 1u32 << (width - 1);
                    (lhs_low & sign_bit != 0, rhs_low & sign_bit != 0)
                } else {
                    let sign_bit = 1u32 << (width - 33);
                    (lhs_high & sign_bit != 0, rhs_high & sign_bit != 0)
                };
                if lhs_sign != rhs_sign {
                    lhs_sign
                } else {
                    lhs_high < rhs_high || (lhs_high == rhs_high && lhs_low < rhs_low)
                }
            } else {
                lhs_high < rhs_high || (lhs_high == rhs_high && lhs_low < rhs_low)
            };
            out[lane] = u32::from(lt);
        }
    }

    fn scalar_two_limb_slice_into(
        input_low: &[u32],
        input_high: Option<&[u32]>,
        lanes: usize,
        lsb: u32,
        out_limbs: usize,
        high_mask: u32,
        out_low: &mut [u32],
        out_high: &mut [u32],
    ) {
        for lane in 0..lanes {
            let value =
                input_low[lane] as u64 | ((input_high.map_or(0, |high| high[lane]) as u64) << 32);
            let sliced = value >> lsb;
            out_low[lane] = sliced as u32;
            if out_limbs >= 2 {
                out_high[lane] = ((sliced >> 32) as u32) & high_mask;
            }
        }
    }

    fn scalar_two_limb_sext_into(
        input_low: &[u32],
        input_high: Option<&[u32]>,
        lanes: usize,
        src_width: u32,
        out_limbs: usize,
        high_mask: u32,
        out_low: &mut [u32],
        out_high: &mut [u32],
    ) {
        let src_mask = super::mask_u64(src_width);
        let sign_bit = 1u64 << (src_width - 1);
        for lane in 0..lanes {
            let value = (input_low[lane] as u64
                | ((input_high.map_or(0, |high| high[lane]) as u64) << 32))
                & src_mask;
            let extended = if value & sign_bit != 0 {
                value | !src_mask
            } else {
                value
            };
            out_low[lane] = extended as u32;
            if out_limbs >= 2 {
                out_high[lane] = ((extended >> 32) as u32) & high_mask;
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn aarch64_two_limb_not_into(
        _input_low: &[u32],
        _input_high: Option<&[u32]>,
        _lanes: usize,
        _high_mask: u32,
        _out_low: &mut [u32],
        _out_high: &mut [u32],
    ) -> bool {
        false
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn aarch64_two_limb_mask_into(
        _input_low: &[u32],
        _input_high: Option<&[u32]>,
        _lanes: usize,
        _high_mask: u32,
        _out_low: &mut [u32],
        _out_high: &mut [u32],
    ) -> bool {
        false
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn aarch64_two_limb_binop_into(
        _lhs_low: &[u32],
        _lhs_high: Option<&[u32]>,
        _rhs_low: &[u32],
        _rhs_high: Option<&[u32]>,
        _lanes: usize,
        _high_mask: u32,
        _out_low: &mut [u32],
        _out_high: &mut [u32],
        _op: TwoLimbBinOp,
    ) -> bool {
        false
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn aarch64_two_limb_mux_into(
        _cond: &[u32],
        _then_low: &[u32],
        _then_high: Option<&[u32]>,
        _else_low: &[u32],
        _else_high: Option<&[u32]>,
        _lanes: usize,
        _high_mask: u32,
        _out_low: &mut [u32],
        _out_high: &mut [u32],
    ) -> bool {
        false
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn aarch64_two_limb_compare_into(
        _lhs_low: &[u32],
        _lhs_high: Option<&[u32]>,
        _rhs_low: &[u32],
        _rhs_high: Option<&[u32]>,
        _lanes: usize,
        _low_mask: u32,
        _high_mask: u32,
        _out: &mut [u32],
        _op: TwoLimbCompareOp,
    ) -> bool {
        false
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn aarch64_two_limb_lt_into(
        _lhs_low: &[u32],
        _lhs_high: Option<&[u32]>,
        _rhs_low: &[u32],
        _rhs_high: Option<&[u32]>,
        _lanes: usize,
        _low_mask: u32,
        _high_mask: u32,
        _signed_width: Option<u32>,
        _out: &mut [u32],
    ) -> bool {
        false
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn aarch64_two_limb_slice_into(
        _input_low: &[u32],
        _input_high: Option<&[u32]>,
        _lanes: usize,
        _lsb: u32,
        _out_limbs: usize,
        _high_mask: u32,
        _out_low: &mut [u32],
        _out_high: &mut [u32],
    ) -> bool {
        false
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn aarch64_two_limb_sext_into(
        _input_low: &[u32],
        _input_high: Option<&[u32]>,
        _lanes: usize,
        _src_width: u32,
        _out_limbs: usize,
        _high_mask: u32,
        _out_low: &mut [u32],
        _out_high: &mut [u32],
    ) -> bool {
        false
    }

    #[cfg(target_arch = "aarch64")]
    fn optional_load_u32x4(input: Option<&[u32]>, offset: usize) -> std::arch::aarch64::uint32x4_t {
        use std::arch::aarch64::*;
        unsafe {
            match input {
                Some(input) => vld1q_u32(input.as_ptr().add(offset)),
                None => vdupq_n_u32(0),
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn aarch64_two_limb_not_into(
        input_low: &[u32],
        input_high: Option<&[u32]>,
        lanes: usize,
        high_mask: u32,
        out_low: &mut [u32],
        out_high: &mut [u32],
    ) -> bool {
        use std::arch::aarch64::*;

        let chunks = lanes / 4;
        if chunks == 0 {
            return false;
        }
        unsafe {
            let high_mask = vdupq_n_u32(high_mask);
            for chunk in 0..chunks {
                let offset = chunk * 4;
                let low = vld1q_u32(input_low.as_ptr().add(offset));
                let high = optional_load_u32x4(input_high, offset);
                vst1q_u32(out_low.as_mut_ptr().add(offset), vmvnq_u32(low));
                vst1q_u32(
                    out_high.as_mut_ptr().add(offset),
                    vandq_u32(vmvnq_u32(high), high_mask),
                );
            }
        }
        for lane in chunks * 4..lanes {
            out_low[lane] = !input_low[lane];
            out_high[lane] = !input_high.map_or(0, |high| high[lane]) & high_mask;
        }
        true
    }

    #[cfg(target_arch = "aarch64")]
    fn aarch64_two_limb_mask_into(
        input_low: &[u32],
        input_high: Option<&[u32]>,
        lanes: usize,
        high_mask: u32,
        out_low: &mut [u32],
        out_high: &mut [u32],
    ) -> bool {
        use std::arch::aarch64::*;

        let chunks = lanes / 4;
        if chunks == 0 {
            return false;
        }
        unsafe {
            let high_mask = vdupq_n_u32(high_mask);
            for chunk in 0..chunks {
                let offset = chunk * 4;
                let low = vld1q_u32(input_low.as_ptr().add(offset));
                let high = optional_load_u32x4(input_high, offset);
                vst1q_u32(out_low.as_mut_ptr().add(offset), low);
                vst1q_u32(
                    out_high.as_mut_ptr().add(offset),
                    vandq_u32(high, high_mask),
                );
            }
        }
        for lane in chunks * 4..lanes {
            out_low[lane] = input_low[lane];
            out_high[lane] = input_high.map_or(0, |high| high[lane]) & high_mask;
        }
        true
    }

    #[cfg(target_arch = "aarch64")]
    fn aarch64_two_limb_binop_into(
        lhs_low: &[u32],
        lhs_high: Option<&[u32]>,
        rhs_low: &[u32],
        rhs_high: Option<&[u32]>,
        lanes: usize,
        high_mask: u32,
        out_low: &mut [u32],
        out_high: &mut [u32],
        op: TwoLimbBinOp,
    ) -> bool {
        use std::arch::aarch64::*;

        let chunks = lanes / 4;
        if chunks == 0 {
            return false;
        }
        unsafe {
            let high_mask_vec = vdupq_n_u32(high_mask);
            let one = vdupq_n_u32(1);
            for chunk in 0..chunks {
                let offset = chunk * 4;
                let lhs_low_vec = vld1q_u32(lhs_low.as_ptr().add(offset));
                let rhs_low_vec = vld1q_u32(rhs_low.as_ptr().add(offset));
                let lhs_high_vec = optional_load_u32x4(lhs_high, offset);
                let rhs_high_vec = optional_load_u32x4(rhs_high, offset);
                let (low, high) = match op {
                    TwoLimbBinOp::And => (
                        vandq_u32(lhs_low_vec, rhs_low_vec),
                        vandq_u32(lhs_high_vec, rhs_high_vec),
                    ),
                    TwoLimbBinOp::Or => (
                        vorrq_u32(lhs_low_vec, rhs_low_vec),
                        vorrq_u32(lhs_high_vec, rhs_high_vec),
                    ),
                    TwoLimbBinOp::Xor => (
                        veorq_u32(lhs_low_vec, rhs_low_vec),
                        veorq_u32(lhs_high_vec, rhs_high_vec),
                    ),
                    TwoLimbBinOp::Add => {
                        let low = vaddq_u32(lhs_low_vec, rhs_low_vec);
                        let carry = vandq_u32(vcgtq_u32(lhs_low_vec, low), one);
                        let high = vaddq_u32(vaddq_u32(lhs_high_vec, rhs_high_vec), carry);
                        (low, high)
                    }
                    TwoLimbBinOp::Sub => {
                        let low = vsubq_u32(lhs_low_vec, rhs_low_vec);
                        let borrow = vandq_u32(vcgtq_u32(rhs_low_vec, lhs_low_vec), one);
                        let high = vsubq_u32(vsubq_u32(lhs_high_vec, rhs_high_vec), borrow);
                        (low, high)
                    }
                };
                vst1q_u32(out_low.as_mut_ptr().add(offset), low);
                vst1q_u32(
                    out_high.as_mut_ptr().add(offset),
                    vandq_u32(high, high_mask_vec),
                );
            }
        }
        for lane in chunks * 4..lanes {
            let lhs_low_word = lhs_low[lane];
            let rhs_low_word = rhs_low[lane];
            let lhs_high_word = lhs_high.map_or(0, |high| high[lane]);
            let rhs_high_word = rhs_high.map_or(0, |high| high[lane]);
            let (low, high) = match op {
                TwoLimbBinOp::And => (lhs_low_word & rhs_low_word, lhs_high_word & rhs_high_word),
                TwoLimbBinOp::Or => (lhs_low_word | rhs_low_word, lhs_high_word | rhs_high_word),
                TwoLimbBinOp::Xor => (lhs_low_word ^ rhs_low_word, lhs_high_word ^ rhs_high_word),
                TwoLimbBinOp::Add => {
                    let (low, carry) = lhs_low_word.overflowing_add(rhs_low_word);
                    (
                        low,
                        lhs_high_word
                            .wrapping_add(rhs_high_word)
                            .wrapping_add(u32::from(carry)),
                    )
                }
                TwoLimbBinOp::Sub => {
                    let (low, borrow) = lhs_low_word.overflowing_sub(rhs_low_word);
                    (
                        low,
                        lhs_high_word
                            .wrapping_sub(rhs_high_word)
                            .wrapping_sub(u32::from(borrow)),
                    )
                }
            };
            out_low[lane] = low;
            out_high[lane] = high & high_mask;
        }
        true
    }

    #[cfg(target_arch = "aarch64")]
    fn aarch64_two_limb_mux_into(
        cond: &[u32],
        then_low: &[u32],
        then_high: Option<&[u32]>,
        else_low: &[u32],
        else_high: Option<&[u32]>,
        lanes: usize,
        high_mask: u32,
        out_low: &mut [u32],
        out_high: &mut [u32],
    ) -> bool {
        use std::arch::aarch64::*;

        let chunks = lanes / 4;
        if chunks == 0 {
            return false;
        }
        unsafe {
            let high_mask_vec = vdupq_n_u32(high_mask);
            let one = vdupq_n_u32(1);
            let zero = vdupq_n_u32(0);
            for chunk in 0..chunks {
                let offset = chunk * 4;
                let cond_vec = vld1q_u32(cond.as_ptr().add(offset));
                let select = vcgtq_u32(vandq_u32(cond_vec, one), zero);
                let then_low_vec = vld1q_u32(then_low.as_ptr().add(offset));
                let else_low_vec = vld1q_u32(else_low.as_ptr().add(offset));
                let then_high_vec = optional_load_u32x4(then_high, offset);
                let else_high_vec = optional_load_u32x4(else_high, offset);
                let low = vbslq_u32(select, then_low_vec, else_low_vec);
                let high = vbslq_u32(select, then_high_vec, else_high_vec);
                vst1q_u32(out_low.as_mut_ptr().add(offset), low);
                vst1q_u32(
                    out_high.as_mut_ptr().add(offset),
                    vandq_u32(high, high_mask_vec),
                );
            }
        }
        for lane in chunks * 4..lanes {
            if cond[lane] & 1 != 0 {
                out_low[lane] = then_low[lane];
                out_high[lane] = then_high.map_or(0, |high| high[lane]) & high_mask;
            } else {
                out_low[lane] = else_low[lane];
                out_high[lane] = else_high.map_or(0, |high| high[lane]) & high_mask;
            }
        }
        true
    }

    #[cfg(target_arch = "aarch64")]
    fn aarch64_two_limb_compare_into(
        lhs_low: &[u32],
        lhs_high: Option<&[u32]>,
        rhs_low: &[u32],
        rhs_high: Option<&[u32]>,
        lanes: usize,
        low_mask: u32,
        high_mask: u32,
        out: &mut [u32],
        op: TwoLimbCompareOp,
    ) -> bool {
        use std::arch::aarch64::*;

        let chunks = lanes / 4;
        if chunks == 0 {
            return false;
        }
        unsafe {
            let low_mask_vec = vdupq_n_u32(low_mask);
            let high_mask_vec = vdupq_n_u32(high_mask);
            let one = vdupq_n_u32(1);
            for chunk in 0..chunks {
                let offset = chunk * 4;
                let lhs_low_vec = vandq_u32(vld1q_u32(lhs_low.as_ptr().add(offset)), low_mask_vec);
                let rhs_low_vec = vandq_u32(vld1q_u32(rhs_low.as_ptr().add(offset)), low_mask_vec);
                let lhs_high_vec = vandq_u32(optional_load_u32x4(lhs_high, offset), high_mask_vec);
                let rhs_high_vec = vandq_u32(optional_load_u32x4(rhs_high, offset), high_mask_vec);
                let equal = vandq_u32(
                    vceqq_u32(lhs_low_vec, rhs_low_vec),
                    vceqq_u32(lhs_high_vec, rhs_high_vec),
                );
                let result = match op {
                    TwoLimbCompareOp::Eq => vandq_u32(equal, one),
                    TwoLimbCompareOp::Ne => vandq_u32(vmvnq_u32(equal), one),
                };
                vst1q_u32(out.as_mut_ptr().add(offset), result);
            }
        }
        for lane in chunks * 4..lanes {
            let equal = (lhs_low[lane] & low_mask) == (rhs_low[lane] & low_mask)
                && (lhs_high.map_or(0, |high| high[lane]) & high_mask)
                    == (rhs_high.map_or(0, |high| high[lane]) & high_mask);
            out[lane] = match op {
                TwoLimbCompareOp::Eq => u32::from(equal),
                TwoLimbCompareOp::Ne => u32::from(!equal),
            };
        }
        true
    }

    #[cfg(target_arch = "aarch64")]
    fn aarch64_two_limb_lt_into(
        lhs_low: &[u32],
        lhs_high: Option<&[u32]>,
        rhs_low: &[u32],
        rhs_high: Option<&[u32]>,
        lanes: usize,
        low_mask: u32,
        high_mask: u32,
        signed_width: Option<u32>,
        out: &mut [u32],
    ) -> bool {
        use std::arch::aarch64::*;

        let chunks = lanes / 4;
        if chunks == 0 || signed_width.is_some_and(|width| width <= 32) {
            return false;
        }
        unsafe {
            let low_mask_vec = vdupq_n_u32(low_mask);
            let high_mask_vec = vdupq_n_u32(high_mask);
            let one = vdupq_n_u32(1);
            let sign_bit = signed_width
                .and_then(|width| (width > 32).then(|| vdupq_n_u32(1u32 << (width - 33))));
            for chunk in 0..chunks {
                let offset = chunk * 4;
                let lhs_low_vec = vandq_u32(vld1q_u32(lhs_low.as_ptr().add(offset)), low_mask_vec);
                let rhs_low_vec = vandq_u32(vld1q_u32(rhs_low.as_ptr().add(offset)), low_mask_vec);
                let lhs_high_vec = vandq_u32(optional_load_u32x4(lhs_high, offset), high_mask_vec);
                let rhs_high_vec = vandq_u32(optional_load_u32x4(rhs_high, offset), high_mask_vec);
                let high_lt = vcgtq_u32(rhs_high_vec, lhs_high_vec);
                let high_eq = vceqq_u32(lhs_high_vec, rhs_high_vec);
                let low_lt = vcgtq_u32(rhs_low_vec, lhs_low_vec);
                let unsigned_lt = vorrq_u32(high_lt, vandq_u32(high_eq, low_lt));
                let result_mask = if let Some(sign_bit) = sign_bit {
                    let lhs_sign = vcgtq_u32(vandq_u32(lhs_high_vec, sign_bit), vdupq_n_u32(0));
                    let rhs_sign = vcgtq_u32(vandq_u32(rhs_high_vec, sign_bit), vdupq_n_u32(0));
                    let sign_diff = veorq_u32(lhs_sign, rhs_sign);
                    vbslq_u32(sign_diff, lhs_sign, unsigned_lt)
                } else {
                    unsigned_lt
                };
                let result = vandq_u32(result_mask, one);
                vst1q_u32(out.as_mut_ptr().add(offset), result);
            }
        }
        for lane in chunks * 4..lanes {
            let lhs_low = lhs_low[lane] & low_mask;
            let rhs_low = rhs_low[lane] & low_mask;
            let lhs_high = lhs_high.map_or(0, |high| high[lane]) & high_mask;
            let rhs_high = rhs_high.map_or(0, |high| high[lane]) & high_mask;
            let lt = if let Some(width) = signed_width {
                let (lhs_sign, rhs_sign) = if width <= 32 {
                    let sign_bit = 1u32 << (width - 1);
                    (lhs_low & sign_bit != 0, rhs_low & sign_bit != 0)
                } else {
                    let sign_bit = 1u32 << (width - 33);
                    (lhs_high & sign_bit != 0, rhs_high & sign_bit != 0)
                };
                if lhs_sign != rhs_sign {
                    lhs_sign
                } else {
                    lhs_high < rhs_high || (lhs_high == rhs_high && lhs_low < rhs_low)
                }
            } else {
                lhs_high < rhs_high || (lhs_high == rhs_high && lhs_low < rhs_low)
            };
            out[lane] = u32::from(lt);
        }
        true
    }

    #[cfg(target_arch = "aarch64")]
    fn aarch64_two_limb_slice_into(
        input_low: &[u32],
        input_high: Option<&[u32]>,
        lanes: usize,
        lsb: u32,
        out_limbs: usize,
        high_mask: u32,
        out_low: &mut [u32],
        out_high: &mut [u32],
    ) -> bool {
        use std::arch::aarch64::*;

        let chunks = lanes / 4;
        if chunks == 0 || lsb > 32 {
            return false;
        }
        unsafe {
            let high_mask_vec = vdupq_n_u32(high_mask);
            for chunk in 0..chunks {
                let offset = chunk * 4;
                let low = vld1q_u32(input_low.as_ptr().add(offset));
                let high = optional_load_u32x4(input_high, offset);
                let out_low_vec = match lsb {
                    0 => low,
                    1 => vorrq_u32(vshrq_n_u32::<1>(low), vshlq_n_u32::<31>(high)),
                    2..=31 => match lsb {
                        2 => vorrq_u32(vshrq_n_u32::<2>(low), vshlq_n_u32::<30>(high)),
                        3 => vorrq_u32(vshrq_n_u32::<3>(low), vshlq_n_u32::<29>(high)),
                        4 => vorrq_u32(vshrq_n_u32::<4>(low), vshlq_n_u32::<28>(high)),
                        5 => vorrq_u32(vshrq_n_u32::<5>(low), vshlq_n_u32::<27>(high)),
                        6 => vorrq_u32(vshrq_n_u32::<6>(low), vshlq_n_u32::<26>(high)),
                        7 => vorrq_u32(vshrq_n_u32::<7>(low), vshlq_n_u32::<25>(high)),
                        8 => vorrq_u32(vshrq_n_u32::<8>(low), vshlq_n_u32::<24>(high)),
                        9 => vorrq_u32(vshrq_n_u32::<9>(low), vshlq_n_u32::<23>(high)),
                        10 => vorrq_u32(vshrq_n_u32::<10>(low), vshlq_n_u32::<22>(high)),
                        11 => vorrq_u32(vshrq_n_u32::<11>(low), vshlq_n_u32::<21>(high)),
                        12 => vorrq_u32(vshrq_n_u32::<12>(low), vshlq_n_u32::<20>(high)),
                        13 => vorrq_u32(vshrq_n_u32::<13>(low), vshlq_n_u32::<19>(high)),
                        14 => vorrq_u32(vshrq_n_u32::<14>(low), vshlq_n_u32::<18>(high)),
                        15 => vorrq_u32(vshrq_n_u32::<15>(low), vshlq_n_u32::<17>(high)),
                        16 => vorrq_u32(vshrq_n_u32::<16>(low), vshlq_n_u32::<16>(high)),
                        17 => vorrq_u32(vshrq_n_u32::<17>(low), vshlq_n_u32::<15>(high)),
                        18 => vorrq_u32(vshrq_n_u32::<18>(low), vshlq_n_u32::<14>(high)),
                        19 => vorrq_u32(vshrq_n_u32::<19>(low), vshlq_n_u32::<13>(high)),
                        20 => vorrq_u32(vshrq_n_u32::<20>(low), vshlq_n_u32::<12>(high)),
                        21 => vorrq_u32(vshrq_n_u32::<21>(low), vshlq_n_u32::<11>(high)),
                        22 => vorrq_u32(vshrq_n_u32::<22>(low), vshlq_n_u32::<10>(high)),
                        23 => vorrq_u32(vshrq_n_u32::<23>(low), vshlq_n_u32::<9>(high)),
                        24 => vorrq_u32(vshrq_n_u32::<24>(low), vshlq_n_u32::<8>(high)),
                        25 => vorrq_u32(vshrq_n_u32::<25>(low), vshlq_n_u32::<7>(high)),
                        26 => vorrq_u32(vshrq_n_u32::<26>(low), vshlq_n_u32::<6>(high)),
                        27 => vorrq_u32(vshrq_n_u32::<27>(low), vshlq_n_u32::<5>(high)),
                        28 => vorrq_u32(vshrq_n_u32::<28>(low), vshlq_n_u32::<4>(high)),
                        29 => vorrq_u32(vshrq_n_u32::<29>(low), vshlq_n_u32::<3>(high)),
                        30 => vorrq_u32(vshrq_n_u32::<30>(low), vshlq_n_u32::<2>(high)),
                        31 => vorrq_u32(vshrq_n_u32::<31>(low), vshlq_n_u32::<1>(high)),
                        _ => unreachable!(),
                    },
                    32 => high,
                    _ => unreachable!(),
                };
                vst1q_u32(out_low.as_mut_ptr().add(offset), out_low_vec);
                if out_limbs >= 2 {
                    let out_high_vec = if lsb == 0 {
                        high
                    } else if lsb < 32 {
                        match lsb {
                            1 => vshrq_n_u32::<1>(high),
                            2 => vshrq_n_u32::<2>(high),
                            3 => vshrq_n_u32::<3>(high),
                            4 => vshrq_n_u32::<4>(high),
                            5 => vshrq_n_u32::<5>(high),
                            6 => vshrq_n_u32::<6>(high),
                            7 => vshrq_n_u32::<7>(high),
                            8 => vshrq_n_u32::<8>(high),
                            9 => vshrq_n_u32::<9>(high),
                            10 => vshrq_n_u32::<10>(high),
                            11 => vshrq_n_u32::<11>(high),
                            12 => vshrq_n_u32::<12>(high),
                            13 => vshrq_n_u32::<13>(high),
                            14 => vshrq_n_u32::<14>(high),
                            15 => vshrq_n_u32::<15>(high),
                            16 => vshrq_n_u32::<16>(high),
                            17 => vshrq_n_u32::<17>(high),
                            18 => vshrq_n_u32::<18>(high),
                            19 => vshrq_n_u32::<19>(high),
                            20 => vshrq_n_u32::<20>(high),
                            21 => vshrq_n_u32::<21>(high),
                            22 => vshrq_n_u32::<22>(high),
                            23 => vshrq_n_u32::<23>(high),
                            24 => vshrq_n_u32::<24>(high),
                            25 => vshrq_n_u32::<25>(high),
                            26 => vshrq_n_u32::<26>(high),
                            27 => vshrq_n_u32::<27>(high),
                            28 => vshrq_n_u32::<28>(high),
                            29 => vshrq_n_u32::<29>(high),
                            30 => vshrq_n_u32::<30>(high),
                            31 => vshrq_n_u32::<31>(high),
                            _ => high,
                        }
                    } else {
                        vdupq_n_u32(0)
                    };
                    vst1q_u32(
                        out_high.as_mut_ptr().add(offset),
                        vandq_u32(out_high_vec, high_mask_vec),
                    );
                }
            }
        }
        for lane in chunks * 4..lanes {
            let value =
                input_low[lane] as u64 | ((input_high.map_or(0, |high| high[lane]) as u64) << 32);
            let sliced = value >> lsb;
            out_low[lane] = sliced as u32;
            if out_limbs >= 2 {
                out_high[lane] = ((sliced >> 32) as u32) & high_mask;
            }
        }
        true
    }

    #[cfg(target_arch = "aarch64")]
    fn aarch64_two_limb_sext_into(
        input_low: &[u32],
        input_high: Option<&[u32]>,
        lanes: usize,
        src_width: u32,
        out_limbs: usize,
        high_mask: u32,
        out_low: &mut [u32],
        out_high: &mut [u32],
    ) -> bool {
        use std::arch::aarch64::*;

        let chunks = lanes / 4;
        if chunks == 0 || src_width == 0 || src_width > 64 {
            return false;
        }
        unsafe {
            let zero = vdupq_n_u32(0);
            let high_mask_vec = vdupq_n_u32(high_mask);
            for chunk in 0..chunks {
                let offset = chunk * 4;
                let low = vld1q_u32(input_low.as_ptr().add(offset));
                let high = optional_load_u32x4(input_high, offset);
                let (low_ext, high_ext) = if src_width <= 32 {
                    let src_mask = super::final_limb_mask(src_width);
                    let src_mask_vec = vdupq_n_u32(src_mask);
                    let sign_bit_vec = vdupq_n_u32(1u32 << (src_width - 1));
                    let low_masked = vandq_u32(low, src_mask_vec);
                    let sign = vcgtq_u32(vandq_u32(low_masked, sign_bit_vec), zero);
                    let low_fill = vandq_u32(sign, vmvnq_u32(src_mask_vec));
                    let high_fill = sign;
                    (vorrq_u32(low_masked, low_fill), high_fill)
                } else {
                    let src_high_mask = super::final_limb_mask(src_width - 32);
                    let src_high_mask_vec = vdupq_n_u32(src_high_mask);
                    let sign_bit_vec = vdupq_n_u32(1u32 << (src_width - 33));
                    let high_masked = vandq_u32(high, src_high_mask_vec);
                    let sign = vcgtq_u32(vandq_u32(high_masked, sign_bit_vec), zero);
                    let high_fill = vandq_u32(sign, vmvnq_u32(src_high_mask_vec));
                    (low, vorrq_u32(high_masked, high_fill))
                };
                vst1q_u32(out_low.as_mut_ptr().add(offset), low_ext);
                if out_limbs >= 2 {
                    vst1q_u32(
                        out_high.as_mut_ptr().add(offset),
                        vandq_u32(high_ext, high_mask_vec),
                    );
                }
            }
        }
        for lane in chunks * 4..lanes {
            let src_mask = super::mask_u64(src_width);
            let sign_bit = 1u64 << (src_width - 1);
            let value = (input_low[lane] as u64
                | ((input_high.map_or(0, |high| high[lane]) as u64) << 32))
                & src_mask;
            let extended = if value & sign_bit != 0 {
                value | !src_mask
            } else {
                value
            };
            out_low[lane] = extended as u32;
            if out_limbs >= 2 {
                out_high[lane] = ((extended >> 32) as u32) & high_mask;
            }
        }
        true
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn scalar_binop_into(
        lhs: &[u32],
        rhs: &[u32],
        mask: u32,
        out: &mut [u32],
        op: impl Fn(u32, u32) -> u32,
    ) {
        for ((out, lhs), rhs) in out.iter_mut().zip(lhs).zip(rhs) {
            *out = op(*lhs, *rhs) & mask;
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[derive(Clone, Copy)]
    enum Aarch64BinOp {
        And,
        Or,
        Xor,
        Add,
        Sub,
        Mul,
    }

    #[cfg(target_arch = "aarch64")]
    fn aarch64_binop_into(lhs: &[u32], rhs: &[u32], mask: u32, out: &mut [u32], op: Aarch64BinOp) {
        use std::arch::aarch64::*;

        let len = out.len().min(lhs.len()).min(rhs.len());
        let chunks = len / 4;
        unsafe {
            let mask_vec = vdupq_n_u32(mask);
            for chunk in 0..chunks {
                let offset = chunk * 4;
                let lhs_vec = vld1q_u32(lhs.as_ptr().add(offset));
                let rhs_vec = vld1q_u32(rhs.as_ptr().add(offset));
                let value = match op {
                    Aarch64BinOp::And => vandq_u32(lhs_vec, rhs_vec),
                    Aarch64BinOp::Or => vorrq_u32(lhs_vec, rhs_vec),
                    Aarch64BinOp::Xor => veorq_u32(lhs_vec, rhs_vec),
                    Aarch64BinOp::Add => vaddq_u32(lhs_vec, rhs_vec),
                    Aarch64BinOp::Sub => vsubq_u32(lhs_vec, rhs_vec),
                    Aarch64BinOp::Mul => vmulq_u32(lhs_vec, rhs_vec),
                };
                let value = vandq_u32(value, mask_vec);
                vst1q_u32(out.as_mut_ptr().add(offset), value);
            }
        }
        for index in chunks * 4..len {
            out[index] = match op {
                Aarch64BinOp::And => lhs[index] & rhs[index],
                Aarch64BinOp::Or => lhs[index] | rhs[index],
                Aarch64BinOp::Xor => lhs[index] ^ rhs[index],
                Aarch64BinOp::Add => lhs[index].wrapping_add(rhs[index]),
                Aarch64BinOp::Sub => lhs[index].wrapping_sub(rhs[index]),
                Aarch64BinOp::Mul => lhs[index].wrapping_mul(rhs[index]),
            } & mask;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[derive(Clone, Copy)]
    enum X86BinOp {
        And,
        Or,
        Xor,
        Add,
        Sub,
        Mul,
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn x86_binop_scalar_one(lhs: u32, rhs: u32, op: X86BinOp) -> u32 {
        match op {
            X86BinOp::And => lhs & rhs,
            X86BinOp::Or => lhs | rhs,
            X86BinOp::Xor => lhs ^ rhs,
            X86BinOp::Add => lhs.wrapping_add(rhs),
            X86BinOp::Sub => lhs.wrapping_sub(rhs),
            X86BinOp::Mul => lhs.wrapping_mul(rhs),
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn x86_binop_scalar_into(lhs: &[u32], rhs: &[u32], mask: u32, out: &mut [u32], op: X86BinOp) {
        let len = out.len().min(lhs.len()).min(rhs.len());
        for index in 0..len {
            out[index] = x86_binop_scalar_one(lhs[index], rhs[index], op) & mask;
        }
    }

    /// AVX2 path: processes 8 lanes (u32) per vector. Falls back to the scalar
    /// kernel at runtime when AVX2 is unavailable, mirroring the NEON dispatch
    /// on aarch64 (where NEON is part of the baseline so no detection is needed).
    #[cfg(target_arch = "x86_64")]
    fn x86_binop_into(lhs: &[u32], rhs: &[u32], mask: u32, out: &mut [u32], op: X86BinOp) {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by the runtime AVX2 feature check above.
            unsafe { x86_binop_into_avx2(lhs, rhs, mask, out, op) }
        } else {
            x86_binop_scalar_into(lhs, rhs, mask, out, op);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn x86_binop_into_avx2(
        lhs: &[u32],
        rhs: &[u32],
        mask: u32,
        out: &mut [u32],
        op: X86BinOp,
    ) {
        use std::arch::x86_64::*;

        let len = out.len().min(lhs.len()).min(rhs.len());
        let chunks = len / 8;
        let mask_vec = _mm256_set1_epi32(mask as i32);
        for chunk in 0..chunks {
            let offset = chunk * 8;
            // Unaligned loads/stores: word slices carry no 32-byte alignment guarantee.
            let lhs_vec = _mm256_loadu_si256(lhs.as_ptr().add(offset) as *const __m256i);
            let rhs_vec = _mm256_loadu_si256(rhs.as_ptr().add(offset) as *const __m256i);
            let value = match op {
                X86BinOp::And => _mm256_and_si256(lhs_vec, rhs_vec),
                X86BinOp::Or => _mm256_or_si256(lhs_vec, rhs_vec),
                X86BinOp::Xor => _mm256_xor_si256(lhs_vec, rhs_vec),
                X86BinOp::Add => _mm256_add_epi32(lhs_vec, rhs_vec),
                X86BinOp::Sub => _mm256_sub_epi32(lhs_vec, rhs_vec),
                X86BinOp::Mul => _mm256_mullo_epi32(lhs_vec, rhs_vec),
            };
            let value = _mm256_and_si256(value, mask_vec);
            _mm256_storeu_si256(out.as_mut_ptr().add(offset) as *mut __m256i, value);
        }
        for index in chunks * 8..len {
            out[index] = x86_binop_scalar_one(lhs[index], rhs[index], op) & mask;
        }
    }
}

#[derive(Clone, Debug)]
pub struct JitCpuSimulator {
    inner: SimdCpuSimulator,
}

impl JitCpuSimulator {
    pub fn new(_program: PackedProgram, _lanes: usize) -> Result<Self, ErrorReport> {
        if !cfg!(feature = "jit") {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_JIT_UNAVAILABLE",
                "JIT CPU backend requires building rrtl-sim-ir with the `jit` feature",
            )]));
        }
        Err(ErrorReport::new(vec![Diagnostic::new(
            "E_SIM_IR_JIT_UNIMPLEMENTED",
            "JIT CPU backend feature is enabled, but Cranelift codegen is not implemented yet",
        )]))
    }

    pub fn snapshot_storage(&self) -> PackedSimulatorStorage {
        self.inner.snapshot_storage()
    }

    pub fn restore_storage(&mut self, storage: &PackedSimulatorStorage) -> Result<(), ErrorReport> {
        self.inner.restore_storage(storage)
    }

    pub fn set_signal(&mut self, signal: Signal, lane_values: &[u128]) -> Result<(), ErrorReport> {
        self.inner.set_signal(signal, lane_values)
    }

    pub fn set_signal_limbs(
        &mut self,
        signal: Signal,
        lane_values: &[Vec<u32>],
    ) -> Result<(), ErrorReport> {
        self.inner.set_signal_limbs(signal, lane_values)
    }

    pub fn get_signal(&self, signal: Signal) -> Result<Vec<u128>, ErrorReport> {
        self.inner.get_signal(signal)
    }

    pub fn get_signal_limbs(&self, signal: Signal) -> Result<Vec<Vec<u32>>, ErrorReport> {
        self.inner.get_signal_limbs(signal)
    }

    pub fn set_memory_limbs(
        &mut self,
        memory: Signal,
        lane_words: &[Vec<Vec<u32>>],
    ) -> Result<(), ErrorReport> {
        self.inner.set_memory_limbs(memory, lane_words)
    }

    pub fn get_memory_limbs(&self, memory: Signal) -> Result<Vec<Vec<Vec<u32>>>, ErrorReport> {
        self.inner.get_memory_limbs(memory)
    }

    pub fn eval_combinational(&mut self) -> Result<(), ErrorReport> {
        self.inner.eval_combinational()
    }

    pub fn tick(&mut self) -> Result<(), ErrorReport> {
        self.inner.tick()
    }

    pub fn tick_many(&mut self, steps: usize) -> Result<(), ErrorReport> {
        self.inner.tick_many(steps)
    }
}

#[derive(Clone, Debug)]
pub enum SimBackendInstance {
    Scalar(SingleLaneMachineSimulator),
    PackedCpu(PackedSimulator),
    SimdCpu(SimdCpuSimulator),
    JitCpu(JitCpuSimulator),
}

impl SimBackendInstance {
    pub fn new(program: PackedProgram, options: SimBackendOptions) -> Result<Self, ErrorReport> {
        match options.kind {
            SimBackendKind::Scalar => {
                if options.lanes > 1 {
                    return Err(ErrorReport::new(vec![Diagnostic::new(
                        "E_SIM_IR_BACKEND_LANES",
                        "scalar backend supports exactly one lane",
                    )]));
                }
                Ok(Self::Scalar(SingleLaneMachineSimulator::new(program)?))
            }
            SimBackendKind::PackedCpu => Ok(Self::PackedCpu(PackedSimulator::new(
                program,
                options.lanes,
            )?)),
            SimBackendKind::SimdCpu => Ok(Self::SimdCpu(SimdCpuSimulator::new(
                program,
                options.lanes,
            )?)),
            SimBackendKind::JitCpu => {
                Ok(Self::JitCpu(JitCpuSimulator::new(program, options.lanes)?))
            }
        }
    }

    pub fn snapshot_storage(&self) -> PackedSimulatorStorage {
        match self {
            Self::Scalar(sim) => sim.snapshot_storage(),
            Self::PackedCpu(sim) => sim.snapshot_storage(),
            Self::SimdCpu(sim) => sim.snapshot_storage(),
            Self::JitCpu(sim) => sim.snapshot_storage(),
        }
    }

    pub fn restore_storage(&mut self, storage: &PackedSimulatorStorage) -> Result<(), ErrorReport> {
        match self {
            Self::Scalar(sim) => sim.restore_storage(storage),
            Self::PackedCpu(sim) => sim.restore_storage(storage),
            Self::SimdCpu(sim) => sim.restore_storage(storage),
            Self::JitCpu(sim) => sim.restore_storage(storage),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct PackedExecutionCache {
    async_reset_comb: PackedBlockCache,
    comb: PackedBlockCache,
    tick_next: PackedBlockCache,
    tick_commit: PackedBlockCache,
}

#[derive(Clone, Debug, Default)]
struct PackedBlockCache {
    value_types: Vec<Option<BitType>>,
}

impl PackedExecutionCache {
    fn new(machine: &PackedMachineProgram) -> Self {
        Self {
            async_reset_comb: PackedBlockCache::new(&machine.streams.async_reset_comb),
            comb: PackedBlockCache::new(&machine.streams.comb),
            tick_next: PackedBlockCache::new(&machine.streams.tick_next),
            tick_commit: PackedBlockCache::new(&machine.streams.tick_commit),
        }
    }
}

impl PackedBlockCache {
    fn new(block: &PackedBlock) -> Self {
        let max_value = block
            .packets
            .iter()
            .flat_map(|packet| packet.instrs.iter().map(|instr| instr.dst.0))
            .max()
            .map(|value| value + 1)
            .unwrap_or(0);
        let mut value_types = vec![None; max_value];
        for packet in &block.packets {
            for instr in &packet.instrs {
                value_types[instr.dst.0] = Some(instr.ty);
            }
        }
        Self { value_types }
    }

    fn value_width(&self, value: PackedValueId) -> Width {
        self.value_types
            .get(value.0)
            .and_then(|ty| *ty)
            .map(|ty| ty.width)
            .unwrap_or(1)
    }
}

#[derive(Clone, Copy, Debug)]
enum PackedStreamKind {
    AsyncResetComb,
    Comb,
    TickCommit,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedSimulatorStorage {
    pub values: Vec<u32>,
    pub memories: Vec<u32>,
}

impl PackedSimulator {
    pub fn new(program: PackedProgram, lanes: usize) -> Result<Self, ErrorReport> {
        if lanes == 0 {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_LANES",
                "packed simulator requires at least one lane",
            )]));
        }
        let machine = lower_to_machine_program(&program);
        analyze_machine_program(&machine)?;
        let execution = PackedExecutionCache::new(&machine);
        let workspaces = PackedExecutionWorkspaces::new(&execution, lanes);
        let values = vec![0; program.total_signal_words * lanes];
        let memories = vec![0; program.total_memory_words_per_lane * lanes];
        let mut sim = Self {
            program,
            machine,
            execution,
            workspaces,
            lanes,
            values,
            memories,
            register_captures: Vec::new(),
            register_capture_count: 0,
        };
        sim.eval_combinational();
        Ok(sim)
    }

    pub fn program(&self) -> &PackedProgram {
        &self.program
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    pub fn snapshot_storage(&self) -> PackedSimulatorStorage {
        PackedSimulatorStorage {
            values: self.values.clone(),
            memories: self.memories.clone(),
        }
    }

    pub fn restore_storage(&mut self, storage: &PackedSimulatorStorage) -> Result<(), ErrorReport> {
        let expected_values = self.program.total_signal_words * self.lanes;
        if storage.values.len() != expected_values {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_STORAGE_VALUES",
                format!(
                    "expected {expected_values} packed signal words, got {}",
                    storage.values.len()
                ),
            )]));
        }
        let expected_memories = self.program.total_memory_words_per_lane * self.lanes;
        if storage.memories.len() != expected_memories {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_STORAGE_MEMORIES",
                format!(
                    "expected {expected_memories} packed memory words, got {}",
                    storage.memories.len()
                ),
            )]));
        }

        self.values.clone_from(&storage.values);
        self.memories.clone_from(&storage.memories);
        Ok(())
    }

    pub fn set_signal(&mut self, signal: Signal, lane_values: &[u128]) -> Result<(), ErrorReport> {
        let ty = self.signal_ty(signal)?;
        let values = lane_values
            .iter()
            .map(|value| encode_u128_limbs(*value, ty))
            .collect::<Vec<_>>();
        self.set_signal_limbs(signal, &values)
    }

    pub fn set_signal_replicated(
        &mut self,
        signal: Signal,
        value: u128,
    ) -> Result<(), ErrorReport> {
        let index = self.signal_index(signal)?;
        self.set_packed_signal_replicated(index, value)?;
        self.eval_combinational();
        Ok(())
    }

    pub fn set_signals_replicated(&mut self, values: &[(Signal, u128)]) -> Result<(), ErrorReport> {
        self.set_signals_replicated_raw(values)?;
        self.eval_combinational();
        Ok(())
    }

    pub fn set_signals_replicated_raw(
        &mut self,
        values: &[(Signal, u128)],
    ) -> Result<(), ErrorReport> {
        let mut encoded = Vec::with_capacity(values.len());
        for (signal, value) in values {
            let index = self.signal_index(*signal)?;
            let ty = self.program.signals[index].layout.ty;
            encoded.push((index, encode_u128_limbs(*value, ty)));
        }
        for (index, limbs) in encoded {
            self.set_packed_signal_replicated_limbs(index, &limbs);
        }
        Ok(())
    }

    pub fn set_memory_replicated(
        &mut self,
        memory: Signal,
        words: &[u128],
    ) -> Result<(), ErrorReport> {
        self.set_memory(memory, &vec![words.to_vec(); self.lanes])
    }

    pub fn set_signal_limbs(
        &mut self,
        signal: Signal,
        lane_values: &[Vec<u32>],
    ) -> Result<(), ErrorReport> {
        let index = self.signal_index(signal)?;
        self.set_packed_signal_limbs(index, lane_values)
    }

    pub fn get_signal(&self, signal: Signal) -> Result<Vec<u128>, ErrorReport> {
        Ok(self
            .get_signal_limbs(signal)?
            .iter()
            .map(|limbs| decode_u128_limbs(limbs))
            .collect())
    }

    pub fn get_signal_lane(&self, signal: Signal, lane: usize) -> Result<u128, ErrorReport> {
        Ok(decode_u128_limbs(
            &self.get_signal_lane_limbs(signal, lane)?,
        ))
    }

    pub fn get_signal_lane_limbs(
        &self,
        signal: Signal,
        lane: usize,
    ) -> Result<Vec<u32>, ErrorReport> {
        if lane >= self.lanes {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_LANE",
                format!("lane {lane} is out of range for {} lanes", self.lanes),
            )]));
        }
        let index = self.signal_index(signal)?;
        Ok(self.load_layout(self.program.signals[index].layout, lane))
    }

    pub fn get_signal_limbs(&self, signal: Signal) -> Result<Vec<Vec<u32>>, ErrorReport> {
        let index = self.signal_index(signal)?;
        let layout = self.program.signals[index].layout;
        Ok((0..self.lanes)
            .map(|lane| self.load_layout(layout, lane))
            .collect())
    }

    pub fn peek_memory_lanes(&self, memory: usize, addr: usize) -> Option<Vec<Vec<u32>>> {
        let mem = self.program.memories.get(memory)?;
        if addr >= mem.depth {
            return None;
        }
        Some(
            (0..self.lanes)
                .map(|lane| self.load_memory(mem, lane, addr))
                .collect(),
        )
    }

    pub fn set_memory(
        &mut self,
        memory: Signal,
        lane_words: &[Vec<u128>],
    ) -> Result<(), ErrorReport> {
        let index = self.memory_signal_index(memory)?;
        let ty = self.program.memories[index].data_layout.ty;
        if ty.width > 128 {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_WIDE_MEMORY",
                "use set_memory_limbs for memories wider than 128 bits",
            )]));
        }
        let lane_limbs = lane_words
            .iter()
            .map(|words| {
                words
                    .iter()
                    .copied()
                    .map(|word| encode_u128_limbs(word, ty))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        self.set_memory_limbs(memory, &lane_limbs)
    }

    pub fn set_memory_limbs(
        &mut self,
        memory: Signal,
        lane_words: &[Vec<Vec<u32>>],
    ) -> Result<(), ErrorReport> {
        let index = self.memory_signal_index(memory)?;
        self.set_memory_index_limbs(index, lane_words)
    }

    pub fn get_memory(&self, memory: Signal) -> Result<Vec<Vec<u128>>, ErrorReport> {
        let index = self.memory_signal_index(memory)?;
        if self.program.memories[index].data_layout.width > 128 {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_WIDE_MEMORY",
                "use get_memory_limbs for memories wider than 128 bits",
            )]));
        }
        Ok(self
            .get_memory_limbs(memory)?
            .iter()
            .map(|lane| {
                lane.iter()
                    .map(|word| decode_u128_limbs(word))
                    .collect::<Vec<_>>()
            })
            .collect())
    }

    pub fn get_memory_limbs(&self, memory: Signal) -> Result<Vec<Vec<Vec<u32>>>, ErrorReport> {
        let index = self.memory_signal_index(memory)?;
        let memory = self.program.memories[index].clone();
        Ok((0..self.lanes)
            .map(|lane| {
                (0..memory.depth)
                    .map(|addr| self.load_memory(&memory, lane, addr))
                    .collect::<Vec<_>>()
            })
            .collect())
    }

    pub fn eval_combinational(&mut self) {
        self.execute_stream(PackedStreamKind::Comb);
        self.execute_stream(PackedStreamKind::AsyncResetComb);
    }

    pub fn tick(&mut self) {
        self.eval_combinational();
        self.tick_from_evaluated_no_post_eval();
        self.eval_combinational();
    }

    pub fn tick_from_evaluated_no_post_eval(&mut self) {
        self.capture_register_stream();
        self.execute_stream(PackedStreamKind::TickCommit);
        self.commit_register_captures();
    }

    pub fn tick_many(&mut self, steps: usize) {
        // Comb-fusion (see SimdCpuSimulator::tick_many): one comb up front, then
        // per cycle capture/commit + a single comb that doubles as the next leading
        // settle. Bit-identical to looping `tick()`, ~half the comb work.
        if steps == 0 {
            return;
        }
        self.eval_combinational();
        for _ in 0..steps {
            self.tick_from_evaluated_no_post_eval();
            self.eval_combinational();
        }
    }

    fn replay_independent_inputs(
        &mut self,
        step: &EncodedTraceReplayStep,
    ) -> Result<(), ErrorReport> {
        for op in &step.independent_input_ops {
            match *op {
                EncodedTraceInputOp::OneLimb {
                    input_index,
                    layout,
                } => {
                    let values = step.inputs[input_index]
                        .lane_limbs_flat
                        .as_ref()
                        .ok_or_else(|| {
                            ErrorReport::new(vec![Diagnostic::new(
                                "E_SIM_IR_REPLAY_LANE_MODE",
                                "independent replay input is missing lane values",
                            )])
                        })?;
                    if values.len() != self.lanes {
                        return Err(ErrorReport::new(vec![Diagnostic::new(
                            "E_SIM_IR_REPLAY_LANES",
                            format!(
                                "independent replay input has {} limbs, expected {}",
                                values.len(),
                                self.lanes
                            ),
                        )]));
                    }
                    self.store_one_limb_lanes(layout, values);
                }
                EncodedTraceInputOp::OneLimbBatch { batch_index } => {
                    let batch = &step.independent_input_batches[batch_index];
                    if batch.values.len() != self.lanes * batch.signals.len() {
                        return Err(ErrorReport::new(vec![Diagnostic::new(
                            "E_SIM_IR_REPLAY_LANES",
                            format!(
                                "independent replay input batch has {} limbs, expected {}",
                                batch.values.len(),
                                self.lanes * batch.signals.len()
                            ),
                        )]));
                    }
                    self.store_one_limb_batch(batch);
                }
                EncodedTraceInputOp::Generic { input_index } => {
                    let input = &step.inputs[input_index];
                    let values = input.lane_limbs_flat.as_ref().ok_or_else(|| {
                        ErrorReport::new(vec![Diagnostic::new(
                            "E_SIM_IR_REPLAY_LANE_MODE",
                            "independent replay input is missing lane values",
                        )])
                    })?;
                    let layout = self.program.signals[input.signal_index].layout;
                    if values.len() != self.lanes * layout.limbs {
                        return Err(ErrorReport::new(vec![Diagnostic::new(
                            "E_SIM_IR_REPLAY_LANES",
                            format!(
                                "independent replay input has {} limbs, expected {}",
                                values.len(),
                                self.lanes * layout.limbs
                            ),
                        )]));
                    }
                    for lane in 0..self.lanes {
                        let start = lane * layout.limbs;
                        self.store_layout_encoded(
                            layout,
                            lane,
                            &values[start..start + layout.limbs],
                        );
                    }
                }
            }
        }
        Ok(())
    }

    fn check_independent_replay_outputs(
        &self,
        step: &EncodedTraceReplayStep,
        step_index: usize,
        options: ReplayOptions,
        report: &mut ReplayReport,
    ) -> Result<(), ErrorReport> {
        for op in &step.independent_check_ops {
            match *op {
                EncodedTraceCheckOp::OneLimb {
                    check_index,
                    layout,
                } => {
                    let check = &step.checks[check_index];
                    let values = check.lane_expected_limbs_flat.as_ref().ok_or_else(|| {
                        ErrorReport::new(vec![Diagnostic::new(
                            "E_SIM_IR_REPLAY_LANE_MODE",
                            "independent replay check is missing lane values",
                        )])
                    })?;
                    if values.len() != self.lanes {
                        return Err(ErrorReport::new(vec![Diagnostic::new(
                            "E_SIM_IR_REPLAY_LANES",
                            format!(
                                "independent replay check has {} limbs, expected {}",
                                values.len(),
                                self.lanes
                            ),
                        )]));
                    }
                    self.for_each_one_limb_mismatch(layout, values, |lane| {
                        let expected = check
                            .lane_expected
                            .as_ref()
                            .and_then(|values| values.get(lane))
                            .copied()
                            .unwrap_or(check.expected);
                        report.record_mismatch(
                            options,
                            step_index,
                            check.check_index,
                            expected,
                            self.load_one_limb_lane(layout, lane),
                            if self.lanes == 1 { None } else { Some(lane) },
                        );
                    });
                }
                EncodedTraceCheckOp::OneLimbBatch { batch_index } => {
                    let batch = &step.independent_check_batches[batch_index];
                    if batch.values.len() != self.lanes * batch.signals.len() {
                        return Err(ErrorReport::new(vec![Diagnostic::new(
                            "E_SIM_IR_REPLAY_LANES",
                            format!(
                                "independent replay check batch has {} limbs, expected {}",
                                batch.values.len(),
                                self.lanes * batch.signals.len()
                            ),
                        )]));
                    }
                    self.for_each_one_limb_batch_mismatch(batch, |signal_index, lane| {
                        let signal = batch.signals[signal_index];
                        let check = &step.checks[signal.check_index];
                        let expected = check
                            .lane_expected
                            .as_ref()
                            .and_then(|values| values.get(lane))
                            .copied()
                            .unwrap_or(check.expected);
                        report.record_mismatch(
                            options,
                            step_index,
                            check.check_index,
                            expected,
                            self.load_one_limb_lane(signal.layout, lane),
                            if self.lanes == 1 { None } else { Some(lane) },
                        );
                    });
                }
                EncodedTraceCheckOp::Generic { check_index } => {
                    let check = &step.checks[check_index];
                    let layout = self.program.signals[check.signal_index].layout;
                    for lane in 0..self.lanes {
                        let values = check.lane_expected_limbs_flat.as_ref().ok_or_else(|| {
                            ErrorReport::new(vec![Diagnostic::new(
                                "E_SIM_IR_REPLAY_LANE_MODE",
                                "independent replay check is missing lane values",
                            )])
                        })?;
                        let start = lane * layout.limbs;
                        let expected_limbs =
                            values.get(start..start + layout.limbs).ok_or_else(|| {
                                ErrorReport::new(vec![Diagnostic::new(
                                    "E_SIM_IR_REPLAY_LANES",
                                    "independent replay check lane slice is out of range",
                                )])
                            })?;
                        if !self.layout_matches_encoded(layout, lane, expected_limbs) {
                            let actual_limbs = self.load_layout(layout, lane);
                            let expected = check
                                .lane_expected
                                .as_ref()
                                .and_then(|values| values.get(lane))
                                .copied()
                                .unwrap_or(check.expected);
                            report.record_mismatch(
                                options,
                                step_index,
                                check.check_index,
                                expected,
                                decode_u128_limbs(&actual_limbs),
                                if self.lanes == 1 { None } else { Some(lane) },
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn replay_trace(
        &mut self,
        plan: &EncodedTraceReplayPlan,
        options: ReplayOptions,
    ) -> Result<ReplayReport, ErrorReport> {
        let mut report = ReplayReport::default();
        let lanes_to_check = match options.lane_mode {
            ReplayLaneMode::Independent => self.lanes,
            ReplayLaneMode::Replicated => match options.check_mode {
                ReplayCheckMode::Lane0Fast => 1,
                ReplayCheckMode::AllLanes => self.lanes,
            },
        };
        for (step_index, step) in plan.steps.iter().enumerate() {
            let start = Instant::now();
            match options.lane_mode {
                ReplayLaneMode::Replicated => {
                    for input in &step.inputs {
                        self.set_packed_signal_replicated_limbs(input.signal_index, &input.limbs)
                    }
                }
                ReplayLaneMode::Independent => self.replay_independent_inputs(step)?,
            }
            report.timing.input_ns += start.elapsed().as_nanos();

            let start = Instant::now();
            self.eval_combinational();
            report.timing.eval_ns += start.elapsed().as_nanos();

            let start = Instant::now();
            match options.lane_mode {
                ReplayLaneMode::Replicated => {
                    for check in &step.checks {
                        let layout = self.program.signals[check.signal_index].layout;
                        for lane in 0..lanes_to_check {
                            if !self.layout_matches_encoded(
                                layout,
                                lane,
                                check.expected_limbs.as_slice(),
                            ) {
                                let actual_limbs = self.load_layout(layout, lane);
                                report.record_mismatch(
                                    options,
                                    step_index,
                                    check.check_index,
                                    check.expected,
                                    decode_u128_limbs(&actual_limbs),
                                    if self.lanes == 1 { None } else { Some(lane) },
                                );
                            }
                        }
                    }
                }
                ReplayLaneMode::Independent => {
                    self.check_independent_replay_outputs(step, step_index, options, &mut report)?
                }
            }
            report.timing.compare_ns += start.elapsed().as_nanos();

            let start = Instant::now();
            self.tick_from_evaluated_no_post_eval();
            report.timing.tick_ns += start.elapsed().as_nanos();
        }
        Ok(report)
    }

    fn execute_stream(&mut self, stream: PackedStreamKind) {
        match stream {
            PackedStreamKind::AsyncResetComb => {
                if self.machine.streams.async_reset_comb.packets.is_empty() {
                    return;
                }
                let block = std::mem::take(&mut self.machine.streams.async_reset_comb);
                let cache = std::mem::take(&mut self.execution.async_reset_comb);
                let mut state = std::mem::take(&mut self.workspaces.async_reset_comb);
                self.execute_machine_block(&block, &cache, &mut state);
                self.workspaces.async_reset_comb = state;
                self.machine.streams.async_reset_comb = block;
                self.execution.async_reset_comb = cache;
            }
            PackedStreamKind::Comb => {
                if self.machine.streams.comb.packets.is_empty() {
                    return;
                }
                let block = std::mem::take(&mut self.machine.streams.comb);
                let cache = std::mem::take(&mut self.execution.comb);
                let mut state = std::mem::take(&mut self.workspaces.comb);
                self.execute_machine_block(&block, &cache, &mut state);
                self.workspaces.comb = state;
                self.machine.streams.comb = block;
                self.execution.comb = cache;
            }
            PackedStreamKind::TickCommit => {
                if self.machine.streams.tick_commit.packets.is_empty() {
                    return;
                }
                let block = std::mem::take(&mut self.machine.streams.tick_commit);
                let cache = std::mem::take(&mut self.execution.tick_commit);
                let mut state = std::mem::take(&mut self.workspaces.tick_commit);
                self.execute_machine_block(&block, &cache, &mut state);
                self.workspaces.tick_commit = state;
                self.machine.streams.tick_commit = block;
                self.execution.tick_commit = cache;
            }
        }
    }

    fn capture_register_stream(&mut self) {
        self.register_capture_count = 0;
        if self.machine.streams.tick_next.packets.is_empty() {
            return;
        }
        let block = std::mem::take(&mut self.machine.streams.tick_next);
        let cache = std::mem::take(&mut self.execution.tick_next);
        let mut state = std::mem::take(&mut self.workspaces.tick_next);
        self.capture_machine_registers(&block, &cache, &mut state);
        self.workspaces.tick_next = state;
        self.machine.streams.tick_next = block;
        self.execution.tick_next = cache;
    }

    fn capture_machine_registers(
        &mut self,
        block: &PackedBlock,
        cache: &PackedBlockCache,
        state: &mut PackedBlockState,
    ) {
        state.reset(cache, self.lanes);
        for packet in &block.packets {
            for instr in &packet.instrs {
                self.eval_machine_instr_into(instr, state, cache);
            }
            for effect in &packet.effects {
                if let PackedEffect::CaptureReg { dst, value, reset } = effect {
                    self.capture_register_value(*dst, *value, reset.as_ref(), state);
                }
            }
        }
    }

    fn capture_register_value(
        &mut self,
        signal: usize,
        value: PackedValueId,
        reset: Option<&PackedReset>,
        state: &PackedBlockState,
    ) {
        let layout = self.program.signals[signal].layout;
        if self.register_capture_count == self.register_captures.len() {
            self.register_captures.push(PackedRegisterCapture {
                signal,
                layout,
                values: Vec::new(),
            });
        }
        let capture_index = self.register_capture_count;
        self.register_capture_count += 1;

        let mut flat_values = Vec::new();
        std::mem::swap(
            &mut flat_values,
            &mut self.register_captures[capture_index].values,
        );
        flat_values.resize(layout.limbs * self.lanes, 0);
        for lane in 0..self.lanes {
            let lane_value = if reset.is_some_and(|reset| self.reset_asserted(reset, lane)) {
                fit_limbs(reset.unwrap().value.clone(), layout.ty)
            } else {
                self.machine_value(state, value, lane)
            };
            for limb in 0..layout.limbs {
                flat_values[limb * self.lanes + lane] = lane_value[limb];
            }
        }

        let capture = &mut self.register_captures[capture_index];
        capture.signal = signal;
        capture.layout = layout;
        capture.values = flat_values;
    }

    fn commit_register_captures(&mut self) {
        for capture_index in 0..self.register_capture_count {
            let capture = &self.register_captures[capture_index];
            debug_assert_eq!(capture.values.len(), capture.layout.limbs * self.lanes);
            for limb in 0..capture.layout.limbs {
                let start = (capture.layout.offset + limb) * self.lanes;
                let source_start = limb * self.lanes;
                self.values[start..start + self.lanes]
                    .copy_from_slice(&capture.values[source_start..source_start + self.lanes]);
            }
        }
    }

    /// Like [`Self::commit_register_captures`] but commits only registers whose
    /// clock is in `active` (a rising edge this step); a register with no entry
    /// in `reg_clocks` (unclocked) always commits. Registers whose clock is
    /// inactive keep their current value (the capture is computed but discarded).
    fn commit_register_captures_clocked(&mut self, active: &std::collections::HashSet<usize>) {
        for capture_index in 0..self.register_capture_count {
            let capture = &self.register_captures[capture_index];
            if let Some(clock) = self.program.reg_clocks.get(&capture.signal) {
                if !active.contains(clock) {
                    continue;
                }
            }
            debug_assert_eq!(capture.values.len(), capture.layout.limbs * self.lanes);
            for limb in 0..capture.layout.limbs {
                let start = (capture.layout.offset + limb) * self.lanes;
                let source_start = limb * self.lanes;
                self.values[start..start + self.lanes]
                    .copy_from_slice(&capture.values[source_start..source_start + self.lanes]);
            }
        }
    }

    fn execute_machine_block(
        &mut self,
        block: &PackedBlock,
        cache: &PackedBlockCache,
        state: &mut PackedBlockState,
    ) {
        state.reset(cache, self.lanes);
        for packet in &block.packets {
            for instr in &packet.instrs {
                self.eval_machine_instr_into(instr, state, cache);
            }
            for effect in &packet.effects {
                self.execute_machine_effect(effect, state);
            }
        }
    }

    fn execute_machine_effect(&mut self, effect: &PackedEffect, state: &PackedBlockState) {
        match effect {
            PackedEffect::StoreSignal { dst, value } => {
                self.store_signal_slot(*dst, state.value(*value));
            }
            PackedEffect::CaptureReg { dst, value, reset } => {
                let Some(reset) = reset else {
                    return;
                };
                if reset.kind != ResetKind::Async {
                    return;
                }
                for lane in 0..self.lanes {
                    if self.reset_asserted(reset, lane) {
                        let lane_value = self.machine_value(state, *value, lane);
                        self.store_signal_lane(*dst, lane, &lane_value);
                    }
                }
            }
            PackedEffect::MemoryWrite {
                memory,
                enable,
                addr,
                data,
            } => {
                for lane in 0..self.lanes {
                    if decode_bool(&self.machine_value(state, *enable, lane)) {
                        let addr = decode_usize(&self.machine_value(state, *addr, lane));
                        let mem = &self.program.memories[*memory];
                        if addr < mem.depth {
                            let value = self.machine_value(state, *data, lane);
                            self.store_memory(*memory, lane, addr, &value);
                        }
                    }
                }
            }
        }
    }

    fn eval_machine_instr_into(
        &self,
        instr: &PackedInstr,
        state: &mut PackedBlockState,
        cache: &PackedBlockCache,
    ) {
        let lanes = self.lanes;
        let limbs = limbs(instr.ty.width);
        state
            .slot_mut(instr.dst)
            .prepare(instr.ty.width, limbs, lanes);
        for lane in 0..lanes {
            let value = self.eval_machine_instr_lane(instr, lane, state, cache);
            state.slot_mut(instr.dst).set_lane(lane, &value);
        }
    }

    fn eval_machine_instr_lane(
        &self,
        instr: &PackedInstr,
        lane: usize,
        state: &PackedBlockState,
        cache: &PackedBlockCache,
    ) -> Vec<u32> {
        let value = match &instr.kind {
            PackedInstrKind::Lit(value) => value.clone(),
            PackedInstrKind::Signal(signal) => {
                self.load_layout(self.program.signals[*signal].layout, lane)
            }
            PackedInstrKind::Not(value) => {
                bit_not(self.machine_value(state, *value, lane), instr.ty)
            }
            PackedInstrKind::And(lhs, rhs) => bit_binop(
                self.machine_value(state, *lhs, lane),
                self.machine_value(state, *rhs, lane),
                instr.ty,
                |lhs, rhs| lhs & rhs,
            ),
            PackedInstrKind::Or(lhs, rhs) => bit_binop(
                self.machine_value(state, *lhs, lane),
                self.machine_value(state, *rhs, lane),
                instr.ty,
                |lhs, rhs| lhs | rhs,
            ),
            PackedInstrKind::Xor(lhs, rhs) => bit_binop(
                self.machine_value(state, *lhs, lane),
                self.machine_value(state, *rhs, lane),
                instr.ty,
                |lhs, rhs| lhs ^ rhs,
            ),
            PackedInstrKind::Add(lhs, rhs) => add_limbs(
                self.machine_value(state, *lhs, lane),
                self.machine_value(state, *rhs, lane),
                instr.ty,
            ),
            PackedInstrKind::Sub(lhs, rhs) => sub_limbs(
                self.machine_value(state, *lhs, lane),
                self.machine_value(state, *rhs, lane),
                instr.ty,
            ),
            PackedInstrKind::Mul(lhs, rhs) => mul_limbs(
                self.machine_value(state, *lhs, lane),
                self.machine_value(state, *rhs, lane),
                instr.ty,
            ),
            PackedInstrKind::Eq(lhs, rhs) => encode_bool(eq_values(
                self.machine_value(state, *lhs, lane),
                self.machine_value(state, *rhs, lane),
                cache.value_width(*lhs).max(cache.value_width(*rhs)),
            )),
            PackedInstrKind::Ne(lhs, rhs) => encode_bool(!eq_values(
                self.machine_value(state, *lhs, lane),
                self.machine_value(state, *rhs, lane),
                cache.value_width(*lhs).max(cache.value_width(*rhs)),
            )),
            PackedInstrKind::Lt { lhs, rhs, signed } => encode_bool(lt_values(
                self.machine_value(state, *lhs, lane),
                self.machine_value(state, *rhs, lane),
                cache.value_width(*lhs),
                *signed,
            )),
            PackedInstrKind::Mux {
                cond,
                then_value,
                else_value,
            } => {
                if decode_bool(&self.machine_value(state, *cond, lane)) {
                    self.machine_value(state, *then_value, lane)
                } else {
                    self.machine_value(state, *else_value, lane)
                }
            }
            PackedInstrKind::Slice { value, lsb } => {
                slice_limbs(self.machine_value(state, *value, lane), *lsb, instr.ty)
            }
            PackedInstrKind::Zext(value)
            | PackedInstrKind::Trunc(value)
            | PackedInstrKind::Cast(value) => {
                fit_limbs(self.machine_value(state, *value, lane), instr.ty)
            }
            PackedInstrKind::Sext(value) => sext_limbs(
                self.machine_value(state, *value, lane),
                cache.value_width(*value),
                instr.ty,
            ),
            PackedInstrKind::Concat(values) => concat_limbs(
                values
                    .iter()
                    .map(|value| {
                        (
                            self.machine_value(state, *value, lane),
                            cache.value_width(*value),
                        )
                    })
                    .collect(),
                instr.ty,
            ),
            PackedInstrKind::MemRead { memory, addr } => {
                let addr = decode_usize(&self.machine_value(state, *addr, lane));
                let mem = &self.program.memories[*memory];
                if addr < mem.depth {
                    self.load_memory(mem, lane, addr)
                } else {
                    vec![0; limbs(instr.ty.width)]
                }
            }
        };
        fit_limbs(value, instr.ty)
    }

    fn machine_value(
        &self,
        state: &PackedBlockState,
        value: PackedValueId,
        lane: usize,
    ) -> Vec<u32> {
        state.value(value).lane(lane)
    }

    fn set_packed_signal_limbs(
        &mut self,
        index: usize,
        lane_values: &[Vec<u32>],
    ) -> Result<(), ErrorReport> {
        if lane_values.len() != self.lanes {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_LANE_VALUES",
                format!(
                    "expected {} lane values, got {}",
                    self.lanes,
                    lane_values.len()
                ),
            )]));
        }
        let layout = self.program.signals[index].layout;
        for (lane, value) in lane_values.iter().enumerate() {
            if value.len() != layout.limbs {
                return Err(ErrorReport::new(vec![Diagnostic::new(
                    "E_SIM_IR_LANE_VALUES",
                    format!("expected {} limbs, got {}", layout.limbs, value.len()),
                )]));
            }
            self.store_layout(layout, lane, value);
        }
        self.eval_combinational();
        Ok(())
    }

    fn set_packed_signal_replicated(
        &mut self,
        index: usize,
        value: u128,
    ) -> Result<(), ErrorReport> {
        let ty = self.program.signals[index].layout.ty;
        let limbs = encode_u128_limbs(value, ty);
        self.set_packed_signal_replicated_limbs(index, &limbs);
        Ok(())
    }

    fn set_packed_signal_replicated_limbs(&mut self, index: usize, value: &[u32]) {
        let layout = self.program.signals[index].layout;
        for lane in 0..self.lanes {
            self.store_layout(layout, lane, value);
        }
    }

    fn reset_asserted(&self, reset: &PackedReset, lane: usize) -> bool {
        let value = self.load_layout(self.program.signals[reset.signal].layout, lane);
        match reset.polarity {
            ResetPolarity::ActiveHigh => decode_bool(&value),
            ResetPolarity::ActiveLow => !decode_bool(&value),
        }
    }

    fn signal_index(&self, signal: Signal) -> Result<usize, ErrorReport> {
        self.program.signal_index(signal).ok_or_else(|| {
            ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_SIGNAL",
                format!("signal {:?} is not part of this packed program", signal.id),
            )])
        })
    }

    fn signal_ty(&self, signal: Signal) -> Result<BitType, ErrorReport> {
        let index = self.signal_index(signal)?;
        Ok(self.program.signals[index].layout.ty)
    }

    fn memory_signal_index(&self, memory: Signal) -> Result<usize, ErrorReport> {
        self.program
            .memories
            .iter()
            .position(|packed| packed.source == memory)
            .ok_or_else(|| {
                ErrorReport::new(vec![Diagnostic::new(
                    "E_SIM_IR_MEMORY",
                    format!("memory {:?} is not part of this packed program", memory.id),
                )])
            })
    }

    fn set_memory_index_limbs(
        &mut self,
        index: usize,
        lane_words: &[Vec<Vec<u32>>],
    ) -> Result<(), ErrorReport> {
        if lane_words.len() != self.lanes {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_MEMORY_VALUES",
                format!(
                    "expected {} lanes of memory values, got {}",
                    self.lanes,
                    lane_words.len()
                ),
            )]));
        }
        let memory = self.program.memories[index].clone();
        for (lane, words) in lane_words.iter().enumerate() {
            if words.len() != memory.depth {
                return Err(ErrorReport::new(vec![Diagnostic::new(
                    "E_SIM_IR_MEMORY_VALUES",
                    format!(
                        "expected {} memory words, got {}",
                        memory.depth,
                        words.len()
                    ),
                )]));
            }
            for (addr, word) in words.iter().enumerate() {
                if word.len() != memory.data_layout.limbs {
                    return Err(ErrorReport::new(vec![Diagnostic::new(
                        "E_SIM_IR_MEMORY_VALUES",
                        format!(
                            "expected {} limbs per memory word, got {}",
                            memory.data_layout.limbs,
                            word.len()
                        ),
                    )]));
                }
                self.store_memory(index, lane, addr, word);
            }
        }
        self.eval_combinational();
        Ok(())
    }

    fn value_index(&self, offset: usize, limb: usize, lane: usize) -> usize {
        (offset + limb) * self.lanes + lane
    }

    fn memory_index(&self, memory: &PackedMemory, lane: usize, addr: usize, limb: usize) -> usize {
        lane * self.program.total_memory_words_per_lane
            + memory.offset
            + addr * memory.data_layout.limbs
            + limb
    }

    fn load_layout(&self, layout: PackedValueLayout, lane: usize) -> Vec<u32> {
        let mut out = (0..layout.limbs)
            .map(|limb| self.values[self.value_index(layout.offset, limb, lane)])
            .collect::<Vec<_>>();
        mask_to_width(&mut out, layout.width);
        out
    }

    fn store_layout(&mut self, layout: PackedValueLayout, lane: usize, value: &[u32]) {
        let stored = fit_limbs(value.to_vec(), layout.ty);
        for limb in 0..layout.limbs {
            let index = self.value_index(layout.offset, limb, lane);
            self.values[index] = stored[limb];
        }
    }

    fn store_layout_encoded(&mut self, layout: PackedValueLayout, lane: usize, value: &[u32]) {
        for limb in 0..layout.limbs {
            let index = self.value_index(layout.offset, limb, lane);
            self.values[index] = value[limb];
        }
    }

    /// Reads a signal's limbs for one lane by packed signal index. Unlike the
    /// `Signal`-handle API this reaches every signal, including nested-instance
    /// signals with no source handle — used to move boundary values between
    /// partition slices in [`PartitionedSimulator`].
    pub fn signal_limbs_at(&self, index: usize, lane: usize) -> Vec<u32> {
        self.load_layout(self.program.signals[index].layout, lane)
    }

    /// Writes a signal's limbs for one lane by packed signal index.
    pub fn set_signal_limbs_at(&mut self, index: usize, lane: usize, limbs: &[u32]) {
        let layout = self.program.signals[index].layout;
        self.store_layout(layout, lane, limbs);
    }

    fn layout_matches_encoded(
        &self,
        layout: PackedValueLayout,
        lane: usize,
        expected: &[u32],
    ) -> bool {
        if layout.limbs == 0 {
            return true;
        }
        for limb in 0..layout.limbs {
            let mut actual = self.values[self.value_index(layout.offset, limb, lane)];
            if limb + 1 == layout.limbs {
                actual &= final_limb_mask(layout.width);
            }
            if actual != expected[limb] {
                return false;
            }
        }
        true
    }

    fn store_one_limb_lanes(&mut self, layout: PackedValueLayout, values: &[u32]) {
        debug_assert_eq!(layout.limbs, 1);
        debug_assert_eq!(values.len(), self.lanes);
        let range = self.one_limb_lane_range(layout);
        self.values[range].copy_from_slice(values);
    }

    fn store_one_limb_batch(&mut self, batch: &EncodedTraceOneLimbInputBatch) {
        debug_assert_eq!(batch.values.len(), batch.signals.len() * self.lanes);
        let range = self.one_limb_batch_range(batch.start_offset, batch.signals.len());
        self.values[range].copy_from_slice(&batch.values);
    }

    fn for_each_one_limb_mismatch(
        &self,
        layout: PackedValueLayout,
        expected: &[u32],
        mut visit: impl FnMut(usize),
    ) {
        debug_assert_eq!(layout.limbs, 1);
        debug_assert_eq!(expected.len(), self.lanes);
        let actual = &self.values[self.one_limb_lane_range(layout)];
        if layout.width == 32 {
            if actual == expected {
                return;
            }
            for (lane, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual != expected {
                    visit(lane);
                }
            }
            return;
        }
        let mask = final_limb_mask(layout.width);
        for (lane, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            if (*actual & mask) != *expected {
                visit(lane);
            }
        }
    }

    fn for_each_one_limb_batch_mismatch(
        &self,
        batch: &EncodedTraceOneLimbCheckBatch,
        mut visit: impl FnMut(usize, usize),
    ) {
        debug_assert_eq!(batch.values.len(), batch.signals.len() * self.lanes);
        let actual =
            &self.values[self.one_limb_batch_range(batch.start_offset, batch.signals.len())];
        if batch.all_full_limb {
            if actual == batch.values.as_slice() {
                return;
            }
            for (index, (actual, expected)) in actual.iter().zip(&batch.values).enumerate() {
                if actual != expected {
                    visit(index / self.lanes, index % self.lanes);
                }
            }
            return;
        }
        for (signal_index, signal) in batch.signals.iter().enumerate() {
            let start = signal_index * self.lanes;
            let actual = &actual[start..start + self.lanes];
            let expected = &batch.values[start..start + self.lanes];
            if signal.layout.width == 32 {
                if actual == expected {
                    continue;
                }
                for (lane, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    if actual != expected {
                        visit(signal_index, lane);
                    }
                }
                continue;
            }
            let mask = final_limb_mask(signal.layout.width);
            for (lane, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if (*actual & mask) != *expected {
                    visit(signal_index, lane);
                }
            }
        }
    }

    fn one_limb_lane_range(&self, layout: PackedValueLayout) -> std::ops::Range<usize> {
        debug_assert_eq!(layout.limbs, 1);
        let start = layout.offset * self.lanes;
        start..start + self.lanes
    }

    fn one_limb_batch_range(
        &self,
        start_offset: usize,
        signal_count: usize,
    ) -> std::ops::Range<usize> {
        let start = start_offset * self.lanes;
        start..start + signal_count * self.lanes
    }

    fn load_one_limb_lane(&self, layout: PackedValueLayout, lane: usize) -> u128 {
        debug_assert_eq!(layout.limbs, 1);
        let index = self.value_index(layout.offset, 0, lane);
        u128::from(self.values[index] & final_limb_mask(layout.width))
    }

    fn store_signal_slot(&mut self, signal: usize, value: &SimdValueSlot) {
        let layout = self.program.signals[signal].layout;
        debug_assert_eq!(value.limbs, layout.limbs);
        debug_assert_eq!(value.words.len(), layout.limbs * self.lanes);
        for limb in 0..layout.limbs {
            let start = (layout.offset + limb) * self.lanes;
            let source_start = limb * self.lanes;
            self.values[start..start + self.lanes]
                .copy_from_slice(&value.words[source_start..source_start + self.lanes]);
        }
    }

    fn store_signal_lane(&mut self, signal: usize, lane: usize, value: &[u32]) {
        let layout = self.program.signals[signal].layout;
        self.store_layout(layout, lane, value);
    }

    fn load_memory(&self, memory: &PackedMemory, lane: usize, addr: usize) -> Vec<u32> {
        let mut out = (0..memory.data_layout.limbs)
            .map(|limb| self.memories[self.memory_index(memory, lane, addr, limb)])
            .collect::<Vec<_>>();
        mask_to_width(&mut out, memory.data_layout.width);
        out
    }

    fn store_memory(&mut self, memory: usize, lane: usize, addr: usize, value: &[u32]) {
        let memory = self.program.memories[memory].clone();
        let stored = fit_limbs(value.to_vec(), memory.data_layout.ty);
        for (limb, limb_value) in stored.iter().copied().enumerate() {
            let index = self.memory_index(&memory, lane, addr, limb);
            self.memories[index] = limb_value;
        }
    }
}

/// Single-lane machine interpreter for packed programs with values up to 128 bits.
#[derive(Clone, Debug)]
pub struct SingleLaneMachineSimulator {
    program: PackedProgram,
    scalar_execution: SingleLaneCompiledExecution,
    workspaces: SingleLaneExecutionWorkspaces,
    values: Vec<u128>,
    memories: Vec<Vec<u128>>,
}

#[derive(Clone, Debug, Default)]
struct SingleLaneCompiledExecution {
    async_reset_comb: SingleLaneCompiledBlock,
    comb: SingleLaneCompiledBlock,
    tick_next: SingleLaneCompiledBlock,
    tick_commit: SingleLaneCompiledBlock,
}

#[derive(Clone, Debug, Default)]
struct SingleLaneCompiledBlock {
    packets: Vec<SingleLaneCompiledPacket>,
}

#[derive(Clone, Debug, Default)]
struct SingleLaneCompiledPacket {
    ops: Vec<SingleLaneCompiledOp>,
    effects: Vec<SingleLaneEffect>,
}

#[derive(Clone, Debug)]
enum SingleLaneCompiledOp {
    Fast(SingleLaneFastOp),
    Wide(SingleLaneOp),
}

#[derive(Clone, Debug)]
enum SingleLaneFastOp {
    Lit {
        dst: usize,
        value: u32,
    },
    Signal {
        dst: usize,
        signal: usize,
        mask: u32,
    },
    Not {
        dst: usize,
        value: usize,
        mask: u32,
    },
    Binary {
        dst: usize,
        lhs: usize,
        rhs: usize,
        kind: SingleLaneBinaryOp,
        mask: u32,
    },
    Compare {
        dst: usize,
        lhs: usize,
        rhs: usize,
        kind: SingleLaneCompareOp,
        mask: u32,
    },
    Mux {
        dst: usize,
        cond: usize,
        then_value: usize,
        else_value: usize,
        mask: u32,
    },
    Slice {
        dst: usize,
        value: usize,
        lsb: Width,
        mask: u32,
    },
    Pass {
        dst: usize,
        value: usize,
        mask: u32,
    },
    Concat {
        dst: usize,
        values: Vec<(usize, Width)>,
        mask: u32,
    },
}

#[derive(Clone, Debug)]
enum SingleLaneOp {
    Lit {
        dst: usize,
        value: u128,
        mask: u128,
    },
    Signal {
        dst: usize,
        signal: usize,
        mask: u128,
    },
    Not {
        dst: usize,
        value: usize,
        mask: u128,
    },
    Binary {
        dst: usize,
        lhs: usize,
        rhs: usize,
        kind: SingleLaneBinaryOp,
        mask: u128,
    },
    Compare {
        dst: usize,
        lhs: usize,
        rhs: usize,
        kind: SingleLaneCompareOp,
        width: Width,
    },
    Mux {
        dst: usize,
        cond: usize,
        then_value: usize,
        else_value: usize,
        mask: u128,
    },
    Slice {
        dst: usize,
        value: usize,
        lsb: Width,
        mask: u128,
    },
    Pass {
        dst: usize,
        value: usize,
        mask: u128,
    },
    Sext {
        dst: usize,
        value: usize,
        from_width: Width,
        to_width: Width,
        mask: u128,
    },
    Concat {
        dst: usize,
        values: Vec<(usize, Width)>,
        mask: u128,
    },
    MemRead {
        dst: usize,
        memory: usize,
        addr: usize,
        mask: u128,
    },
}

#[derive(Clone, Copy, Debug)]
enum SingleLaneBinaryOp {
    And,
    Or,
    Xor,
    Add,
    Sub,
    Mul,
}

#[derive(Clone, Copy, Debug)]
enum SingleLaneCompareOp {
    Eq,
    Ne,
    Lt { signed: bool },
}

#[derive(Clone, Debug)]
enum SingleLaneEffect {
    StoreSignal {
        dst: usize,
        value: usize,
        mask: u128,
    },
    CaptureReg {
        dst: usize,
        value: usize,
        reset: Option<SingleLaneReset>,
        mask: u128,
    },
    MemoryWrite {
        memory: usize,
        enable: usize,
        addr: usize,
        data: usize,
        mask: u128,
    },
}

#[derive(Clone, Copy, Debug)]
struct SingleLaneReset {
    signal: usize,
    value: u128,
    kind: ResetKind,
    polarity: ResetPolarity,
}

#[derive(Clone, Debug)]
struct SingleLaneCompiledReplayPlan {
    total_lanes: usize,
    steps: Vec<SingleLaneCompiledReplayStep>,
}

#[derive(Clone, Debug, Default)]
struct SingleLaneCompiledReplayStep {
    inputs: Vec<SingleLaneCompiledReplayInput>,
    checks: Vec<SingleLaneCompiledReplayCheck>,
}

#[derive(Clone, Debug)]
enum SingleLaneCompiledReplayInput {
    OneLimb {
        signal: usize,
        mask: u128,
        values: Vec<u32>,
    },
    OneLimbBatch {
        signals: Vec<SingleLaneCompiledReplayBatchSignal>,
        values: Vec<u32>,
    },
    Generic {
        signal: usize,
        mask: u128,
        limbs: usize,
        values: Vec<u32>,
    },
}

#[derive(Clone, Debug)]
enum SingleLaneCompiledReplayCheck {
    OneLimb {
        signal: usize,
        check_index: usize,
        mask: u128,
        values: Vec<u32>,
    },
    OneLimbBatch {
        signals: Vec<SingleLaneCompiledReplayBatchCheck>,
        values: Vec<u32>,
    },
    Generic {
        signal: usize,
        check_index: usize,
        mask: u128,
        values: Vec<u128>,
    },
}

#[derive(Clone, Copy, Debug)]
struct SingleLaneCompiledReplayBatchSignal {
    signal: usize,
    mask: u128,
    value_base: usize,
}

#[derive(Clone, Copy, Debug)]
struct SingleLaneCompiledReplayBatchCheck {
    signal: usize,
    check_index: usize,
    mask: u128,
    value_base: usize,
}

#[derive(Clone, Debug, Default)]
struct SingleLaneExecutionWorkspaces {
    async_reset_comb: Vec<u128>,
    comb: Vec<u128>,
    tick_next: Vec<u128>,
    tick_commit: Vec<u128>,
    register_captures: Vec<(usize, u128)>,
}

impl SingleLaneExecutionWorkspaces {
    fn new(execution: &PackedExecutionCache) -> Self {
        Self {
            async_reset_comb: vec![0; execution.async_reset_comb.value_types.len()],
            comb: vec![0; execution.comb.value_types.len()],
            tick_next: vec![0; execution.tick_next.value_types.len()],
            tick_commit: vec![0; execution.tick_commit.value_types.len()],
            register_captures: Vec::new(),
        }
    }
}

impl SingleLaneCompiledExecution {
    fn new(machine: &PackedMachineProgram, execution: &PackedExecutionCache) -> Self {
        Self {
            async_reset_comb: SingleLaneCompiledBlock::new(
                &machine.streams.async_reset_comb,
                &execution.async_reset_comb,
                &machine.source,
            ),
            comb: SingleLaneCompiledBlock::new(
                &machine.streams.comb,
                &execution.comb,
                &machine.source,
            ),
            tick_next: SingleLaneCompiledBlock::new(
                &machine.streams.tick_next,
                &execution.tick_next,
                &machine.source,
            ),
            tick_commit: SingleLaneCompiledBlock::new(
                &machine.streams.tick_commit,
                &execution.tick_commit,
                &machine.source,
            ),
        }
    }
}

impl SingleLaneCompiledBlock {
    fn new(block: &PackedBlock, cache: &PackedBlockCache, program: &PackedProgram) -> Self {
        Self {
            packets: block
                .packets
                .iter()
                .map(|packet| SingleLaneCompiledPacket {
                    ops: packet
                        .instrs
                        .iter()
                        .map(|instr| compile_single_lane_compiled_op(instr, cache))
                        .collect(),
                    effects: packet
                        .effects
                        .iter()
                        .map(|effect| compile_single_lane_effect(effect, program))
                        .collect(),
                })
                .collect(),
        }
    }
}

fn compile_single_lane_compiled_op(
    instr: &PackedInstr,
    cache: &PackedBlockCache,
) -> SingleLaneCompiledOp {
    compile_single_lane_fast_op(instr, cache)
        .map(SingleLaneCompiledOp::Fast)
        .unwrap_or_else(|| SingleLaneCompiledOp::Wide(compile_single_lane_op(instr, cache)))
}

fn compile_single_lane_fast_op(
    instr: &PackedInstr,
    cache: &PackedBlockCache,
) -> Option<SingleLaneFastOp> {
    if instr.ty.width > 32 {
        return None;
    }
    let dst = instr.dst.0;
    let mask = mask_u32(instr.ty.width);
    match &instr.kind {
        PackedInstrKind::Lit(value) => Some(SingleLaneFastOp::Lit {
            dst,
            value: decode_u128_limbs(value) as u32 & mask,
        }),
        PackedInstrKind::Signal(signal) => Some(SingleLaneFastOp::Signal {
            dst,
            signal: *signal,
            mask,
        }),
        PackedInstrKind::Not(value) if cache.value_width(*value) <= 32 => {
            Some(SingleLaneFastOp::Not {
                dst,
                value: value.0,
                mask,
            })
        }
        PackedInstrKind::And(lhs, rhs)
        | PackedInstrKind::Or(lhs, rhs)
        | PackedInstrKind::Xor(lhs, rhs)
        | PackedInstrKind::Add(lhs, rhs)
        | PackedInstrKind::Sub(lhs, rhs)
        | PackedInstrKind::Mul(lhs, rhs)
            if cache.value_width(*lhs) <= 32 && cache.value_width(*rhs) <= 32 =>
        {
            let kind = match &instr.kind {
                PackedInstrKind::And(_, _) => SingleLaneBinaryOp::And,
                PackedInstrKind::Or(_, _) => SingleLaneBinaryOp::Or,
                PackedInstrKind::Xor(_, _) => SingleLaneBinaryOp::Xor,
                PackedInstrKind::Add(_, _) => SingleLaneBinaryOp::Add,
                PackedInstrKind::Sub(_, _) => SingleLaneBinaryOp::Sub,
                PackedInstrKind::Mul(_, _) => SingleLaneBinaryOp::Mul,
                _ => unreachable!(),
            };
            Some(SingleLaneFastOp::Binary {
                dst,
                lhs: lhs.0,
                rhs: rhs.0,
                kind,
                mask,
            })
        }
        PackedInstrKind::Eq(lhs, rhs) | PackedInstrKind::Ne(lhs, rhs)
            if cache.value_width(*lhs) <= 32 && cache.value_width(*rhs) <= 32 =>
        {
            let width = cache.value_width(*lhs);
            Some(SingleLaneFastOp::Compare {
                dst,
                lhs: lhs.0,
                rhs: rhs.0,
                kind: match &instr.kind {
                    PackedInstrKind::Eq(_, _) => SingleLaneCompareOp::Eq,
                    PackedInstrKind::Ne(_, _) => SingleLaneCompareOp::Ne,
                    _ => unreachable!(),
                },
                mask: mask_u32(width),
            })
        }
        PackedInstrKind::Lt {
            lhs,
            rhs,
            signed: false,
        } if cache.value_width(*lhs) <= 32 && cache.value_width(*rhs) <= 32 => {
            Some(SingleLaneFastOp::Compare {
                dst,
                lhs: lhs.0,
                rhs: rhs.0,
                kind: SingleLaneCompareOp::Lt { signed: false },
                mask: mask_u32(cache.value_width(*lhs)),
            })
        }
        PackedInstrKind::Mux {
            cond,
            then_value,
            else_value,
        } if cache.value_width(*cond) <= 32
            && cache.value_width(*then_value) <= 32
            && cache.value_width(*else_value) <= 32 =>
        {
            Some(SingleLaneFastOp::Mux {
                dst,
                cond: cond.0,
                then_value: then_value.0,
                else_value: else_value.0,
                mask,
            })
        }
        PackedInstrKind::Slice { value, lsb } if cache.value_width(*value) <= 32 => {
            Some(SingleLaneFastOp::Slice {
                dst,
                value: value.0,
                lsb: *lsb,
                mask,
            })
        }
        PackedInstrKind::Zext(value)
        | PackedInstrKind::Trunc(value)
        | PackedInstrKind::Cast(value)
            if cache.value_width(*value) <= 32 =>
        {
            Some(SingleLaneFastOp::Pass {
                dst,
                value: value.0,
                mask,
            })
        }
        PackedInstrKind::Concat(values)
            if values.iter().all(|value| cache.value_width(*value) <= 32)
                && values
                    .iter()
                    .map(|value| cache.value_width(*value) as usize)
                    .sum::<usize>()
                    <= 32 =>
        {
            Some(SingleLaneFastOp::Concat {
                dst,
                values: values
                    .iter()
                    .map(|value| (value.0, cache.value_width(*value)))
                    .collect(),
                mask,
            })
        }
        _ => None,
    }
}

fn compile_single_lane_op(instr: &PackedInstr, cache: &PackedBlockCache) -> SingleLaneOp {
    let dst = instr.dst.0;
    let mask = mask_u128(instr.ty.width);
    match &instr.kind {
        PackedInstrKind::Lit(value) => SingleLaneOp::Lit {
            dst,
            value: decode_u128_limbs(value),
            mask,
        },
        PackedInstrKind::Signal(signal) => SingleLaneOp::Signal {
            dst,
            signal: *signal,
            mask,
        },
        PackedInstrKind::Not(value) => SingleLaneOp::Not {
            dst,
            value: value.0,
            mask,
        },
        PackedInstrKind::And(lhs, rhs) => {
            compile_single_lane_binary_op(dst, *lhs, *rhs, SingleLaneBinaryOp::And, mask)
        }
        PackedInstrKind::Or(lhs, rhs) => {
            compile_single_lane_binary_op(dst, *lhs, *rhs, SingleLaneBinaryOp::Or, mask)
        }
        PackedInstrKind::Xor(lhs, rhs) => {
            compile_single_lane_binary_op(dst, *lhs, *rhs, SingleLaneBinaryOp::Xor, mask)
        }
        PackedInstrKind::Add(lhs, rhs) => {
            compile_single_lane_binary_op(dst, *lhs, *rhs, SingleLaneBinaryOp::Add, mask)
        }
        PackedInstrKind::Sub(lhs, rhs) => {
            compile_single_lane_binary_op(dst, *lhs, *rhs, SingleLaneBinaryOp::Sub, mask)
        }
        PackedInstrKind::Mul(lhs, rhs) => {
            compile_single_lane_binary_op(dst, *lhs, *rhs, SingleLaneBinaryOp::Mul, mask)
        }
        PackedInstrKind::Eq(lhs, rhs) => SingleLaneOp::Compare {
            dst,
            lhs: lhs.0,
            rhs: rhs.0,
            kind: SingleLaneCompareOp::Eq,
            width: cache.value_width(*lhs),
        },
        PackedInstrKind::Ne(lhs, rhs) => SingleLaneOp::Compare {
            dst,
            lhs: lhs.0,
            rhs: rhs.0,
            kind: SingleLaneCompareOp::Ne,
            width: cache.value_width(*lhs),
        },
        PackedInstrKind::Lt { lhs, rhs, signed } => SingleLaneOp::Compare {
            dst,
            lhs: lhs.0,
            rhs: rhs.0,
            kind: SingleLaneCompareOp::Lt { signed: *signed },
            width: cache.value_width(*lhs),
        },
        PackedInstrKind::Mux {
            cond,
            then_value,
            else_value,
        } => SingleLaneOp::Mux {
            dst,
            cond: cond.0,
            then_value: then_value.0,
            else_value: else_value.0,
            mask,
        },
        PackedInstrKind::Slice { value, lsb } => SingleLaneOp::Slice {
            dst,
            value: value.0,
            lsb: *lsb,
            mask,
        },
        PackedInstrKind::Zext(value)
        | PackedInstrKind::Trunc(value)
        | PackedInstrKind::Cast(value) => SingleLaneOp::Pass {
            dst,
            value: value.0,
            mask,
        },
        PackedInstrKind::Sext(value) => SingleLaneOp::Sext {
            dst,
            value: value.0,
            from_width: cache.value_width(*value),
            to_width: instr.ty.width,
            mask,
        },
        PackedInstrKind::Concat(values) => SingleLaneOp::Concat {
            dst,
            values: values
                .iter()
                .map(|value| (value.0, cache.value_width(*value)))
                .collect(),
            mask,
        },
        PackedInstrKind::MemRead { memory, addr } => SingleLaneOp::MemRead {
            dst,
            memory: *memory,
            addr: addr.0,
            mask,
        },
    }
}

fn compile_single_lane_binary_op(
    dst: usize,
    lhs: PackedValueId,
    rhs: PackedValueId,
    kind: SingleLaneBinaryOp,
    mask: u128,
) -> SingleLaneOp {
    SingleLaneOp::Binary {
        dst,
        lhs: lhs.0,
        rhs: rhs.0,
        kind,
        mask,
    }
}

fn compile_single_lane_effect(effect: &PackedEffect, program: &PackedProgram) -> SingleLaneEffect {
    match effect {
        PackedEffect::StoreSignal { dst, value } => SingleLaneEffect::StoreSignal {
            dst: *dst,
            value: value.0,
            mask: mask_u128(program.signals[*dst].layout.width),
        },
        PackedEffect::CaptureReg { dst, value, reset } => SingleLaneEffect::CaptureReg {
            dst: *dst,
            value: value.0,
            reset: reset.as_ref().map(|reset| SingleLaneReset {
                signal: reset.signal,
                value: decode_u128_limbs(&reset.value),
                kind: reset.kind,
                polarity: reset.polarity,
            }),
            mask: mask_u128(program.signals[*dst].layout.width),
        },
        PackedEffect::MemoryWrite {
            memory,
            enable,
            addr,
            data,
        } => SingleLaneEffect::MemoryWrite {
            memory: *memory,
            enable: enable.0,
            addr: addr.0,
            data: data.0,
            mask: mask_u128(program.memories[*memory].data_layout.width),
        },
    }
}

impl SingleLaneCompiledReplayPlan {
    fn new(program: &PackedProgram, plan: &EncodedTraceReplayPlan) -> Result<Self, ErrorReport> {
        let total_lanes = plan.independent_lanes().ok_or_else(|| {
            ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_REPLAY_LANES",
                "compiled scalar replay requires an independent-lane replay plan",
            )])
        })?;
        if total_lanes == 0 {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_REPLAY_LANES",
                "compiled scalar replay requires at least one lane",
            )]));
        }
        let steps = plan
            .steps
            .iter()
            .map(|step| SingleLaneCompiledReplayStep::new(program, step, total_lanes))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { total_lanes, steps })
    }
}

impl SingleLaneCompiledReplayStep {
    fn new(
        program: &PackedProgram,
        step: &EncodedTraceReplayStep,
        total_lanes: usize,
    ) -> Result<Self, ErrorReport> {
        let inputs = step
            .independent_input_ops
            .iter()
            .map(|op| compile_single_lane_replay_input(program, step, *op, total_lanes))
            .collect::<Result<Vec<_>, _>>()?;
        let checks = step
            .independent_check_ops
            .iter()
            .map(|op| compile_single_lane_replay_check(program, step, *op, total_lanes))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { inputs, checks })
    }
}

fn compile_single_lane_replay_input(
    program: &PackedProgram,
    step: &EncodedTraceReplayStep,
    op: EncodedTraceInputOp,
    total_lanes: usize,
) -> Result<SingleLaneCompiledReplayInput, ErrorReport> {
    match op {
        EncodedTraceInputOp::Generic { input_index } => {
            let input = &step.inputs[input_index];
            let layout = program.signals[input.signal_index].layout;
            let values = input.lane_limbs_flat.clone().ok_or_else(|| {
                missing_compiled_lane_data("input", input.signal, "flat limb values")
            })?;
            validate_compiled_replay_value_count(
                values.len(),
                total_lanes * layout.limbs,
                "input",
                input.signal,
            )?;
            Ok(SingleLaneCompiledReplayInput::Generic {
                signal: input.signal_index,
                mask: mask_u128(layout.width),
                limbs: layout.limbs,
                values,
            })
        }
        EncodedTraceInputOp::OneLimb { input_index, .. } => {
            let input = &step.inputs[input_index];
            let layout = program.signals[input.signal_index].layout;
            let values = input.lane_limbs_flat.clone().ok_or_else(|| {
                missing_compiled_lane_data("input", input.signal, "one-limb values")
            })?;
            validate_compiled_replay_value_count(values.len(), total_lanes, "input", input.signal)?;
            Ok(SingleLaneCompiledReplayInput::OneLimb {
                signal: input.signal_index,
                mask: mask_u128(layout.width),
                values,
            })
        }
        EncodedTraceInputOp::OneLimbBatch { batch_index } => {
            let batch = &step.independent_input_batches[batch_index];
            if let Some(first) = batch.signals.first() {
                validate_compiled_replay_value_count(
                    batch.values.len(),
                    total_lanes * batch.signals.len(),
                    "input batch",
                    step.inputs[first.input_index].signal,
                )?;
            }
            let signals = batch
                .signals
                .iter()
                .enumerate()
                .map(|(signal_ordinal, signal)| {
                    let input = &step.inputs[signal.input_index];
                    SingleLaneCompiledReplayBatchSignal {
                        signal: input.signal_index,
                        mask: mask_u128(signal.layout.width),
                        value_base: signal_ordinal * total_lanes,
                    }
                })
                .collect();
            Ok(SingleLaneCompiledReplayInput::OneLimbBatch {
                signals,
                values: batch.values.clone(),
            })
        }
    }
}

fn compile_single_lane_replay_check(
    program: &PackedProgram,
    step: &EncodedTraceReplayStep,
    op: EncodedTraceCheckOp,
    total_lanes: usize,
) -> Result<SingleLaneCompiledReplayCheck, ErrorReport> {
    match op {
        EncodedTraceCheckOp::Generic { check_index } => {
            let check = &step.checks[check_index];
            let layout = program.signals[check.signal_index].layout;
            let values = check.lane_expected.clone().ok_or_else(|| {
                missing_compiled_lane_data("check", check.signal, "expected values")
            })?;
            validate_compiled_replay_value_count(values.len(), total_lanes, "check", check.signal)?;
            Ok(SingleLaneCompiledReplayCheck::Generic {
                signal: check.signal_index,
                check_index: check.check_index,
                mask: mask_u128(layout.width),
                values,
            })
        }
        EncodedTraceCheckOp::OneLimb { check_index, .. } => {
            let check = &step.checks[check_index];
            let layout = program.signals[check.signal_index].layout;
            let values = check.lane_expected_limbs_flat.clone().ok_or_else(|| {
                missing_compiled_lane_data("check", check.signal, "one-limb expected values")
            })?;
            validate_compiled_replay_value_count(values.len(), total_lanes, "check", check.signal)?;
            Ok(SingleLaneCompiledReplayCheck::OneLimb {
                signal: check.signal_index,
                check_index: check.check_index,
                mask: mask_u128(layout.width),
                values,
            })
        }
        EncodedTraceCheckOp::OneLimbBatch { batch_index } => {
            let batch = &step.independent_check_batches[batch_index];
            if let Some(first) = batch.signals.first() {
                validate_compiled_replay_value_count(
                    batch.values.len(),
                    total_lanes * batch.signals.len(),
                    "check batch",
                    step.checks[first.check_index].signal,
                )?;
            }
            let signals = batch
                .signals
                .iter()
                .enumerate()
                .map(|(signal_ordinal, signal)| {
                    let check = &step.checks[signal.check_index];
                    SingleLaneCompiledReplayBatchCheck {
                        signal: check.signal_index,
                        check_index: check.check_index,
                        mask: mask_u128(signal.layout.width),
                        value_base: signal_ordinal * total_lanes,
                    }
                })
                .collect();
            Ok(SingleLaneCompiledReplayCheck::OneLimbBatch {
                signals,
                values: batch.values.clone(),
            })
        }
    }
}

fn missing_compiled_lane_data(kind: &str, signal: Signal, data: &str) -> ErrorReport {
    ErrorReport::new(vec![Diagnostic::new(
        "E_SIM_IR_REPLAY_LANES",
        format!(
            "compiled scalar replay {kind} {:?} is missing {data}",
            signal.id
        ),
    )])
}

fn validate_compiled_replay_value_count(
    actual: usize,
    expected: usize,
    kind: &str,
    signal: Signal,
) -> Result<(), ErrorReport> {
    if actual == expected {
        Ok(())
    } else {
        Err(ErrorReport::new(vec![Diagnostic::new(
            "E_SIM_IR_REPLAY_LANES",
            format!(
                "compiled scalar replay {kind} {:?} has {actual} values, expected {expected}",
                signal.id
            ),
        )]))
    }
}

impl SingleLaneMachineSimulator {
    pub fn new(program: PackedProgram) -> Result<Self, ErrorReport> {
        let machine = lower_to_machine_program(&program);
        analyze_machine_program(&machine)?;
        validate_single_lane_widths(&program, &machine)?;
        let execution = PackedExecutionCache::new(&machine);
        let scalar_execution = SingleLaneCompiledExecution::new(&machine, &execution);
        let workspaces = SingleLaneExecutionWorkspaces::new(&execution);
        let values = vec![0; program.signals.len()];
        let memories = program
            .memories
            .iter()
            .map(|memory| vec![0; memory.depth])
            .collect::<Vec<_>>();
        let mut sim = Self {
            program,
            scalar_execution,
            workspaces,
            values,
            memories,
        };
        sim.eval_combinational();
        Ok(sim)
    }

    pub fn program(&self) -> &PackedProgram {
        &self.program
    }

    pub fn lanes(&self) -> usize {
        1
    }

    pub fn snapshot_storage(&self) -> PackedSimulatorStorage {
        let mut values = vec![0; self.program.total_signal_words];
        for (signal_index, signal) in self.program.signals.iter().enumerate() {
            let limbs = encode_u128_limbs(self.values[signal_index], signal.layout.ty);
            for (limb, value) in limbs.into_iter().enumerate() {
                values[signal.layout.offset + limb] = value;
            }
        }

        let mut memories = vec![0; self.program.total_memory_words_per_lane];
        for (memory_index, memory) in self.program.memories.iter().enumerate() {
            for addr in 0..memory.depth {
                let limbs =
                    encode_u128_limbs(self.memories[memory_index][addr], memory.data_layout.ty);
                for (limb, value) in limbs.into_iter().enumerate() {
                    memories[memory.offset + addr * memory.data_layout.limbs + limb] = value;
                }
            }
        }

        PackedSimulatorStorage { values, memories }
    }

    pub fn restore_storage(&mut self, storage: &PackedSimulatorStorage) -> Result<(), ErrorReport> {
        if storage.values.len() != self.program.total_signal_words {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_STORAGE_VALUES",
                format!(
                    "expected {} scalar signal words, got {}",
                    self.program.total_signal_words,
                    storage.values.len()
                ),
            )]));
        }
        if storage.memories.len() != self.program.total_memory_words_per_lane {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_STORAGE_MEMORIES",
                format!(
                    "expected {} scalar memory words, got {}",
                    self.program.total_memory_words_per_lane,
                    storage.memories.len()
                ),
            )]));
        }

        for (signal_index, signal) in self.program.signals.iter().enumerate() {
            let start = signal.layout.offset;
            let end = start + signal.layout.limbs;
            self.values[signal_index] = fit_u128(
                decode_u128_limbs(&storage.values[start..end]),
                signal.layout.ty,
            );
        }
        for (memory_index, memory) in self.program.memories.iter().enumerate() {
            for addr in 0..memory.depth {
                let start = memory.offset + addr * memory.data_layout.limbs;
                let end = start + memory.data_layout.limbs;
                self.memories[memory_index][addr] = fit_u128(
                    decode_u128_limbs(&storage.memories[start..end]),
                    memory.data_layout.ty,
                );
            }
        }
        self.eval_combinational();
        Ok(())
    }

    pub fn set_signal(&mut self, signal: Signal, value: u128) -> Result<(), ErrorReport> {
        let index = self.signal_index(signal)?;
        self.values[index] = self.fit_signal(index, value);
        self.eval_combinational();
        Ok(())
    }

    pub fn set_signal_limbs(
        &mut self,
        signal: Signal,
        lane_values: &[Vec<u32>],
    ) -> Result<(), ErrorReport> {
        if lane_values.len() != 1 {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_LANE_VALUES",
                format!("expected one scalar lane value, got {}", lane_values.len()),
            )]));
        }
        self.set_signal(signal, decode_u128_limbs(&lane_values[0]))
    }

    pub fn get_signal_limbs(&self, signal: Signal) -> Result<Vec<Vec<u32>>, ErrorReport> {
        let index = self.signal_index(signal)?;
        Ok(vec![encode_u128_limbs(
            self.values[index],
            self.program.signals[index].layout.ty,
        )])
    }

    pub fn set_signals(&mut self, values: &[(Signal, u128)]) -> Result<(), ErrorReport> {
        self.set_signals_raw(values)?;
        self.eval_combinational();
        Ok(())
    }

    pub fn set_signals_raw(&mut self, values: &[(Signal, u128)]) -> Result<(), ErrorReport> {
        let mut indexed = Vec::with_capacity(values.len());
        for (signal, value) in values {
            let index = self.signal_index(*signal)?;
            indexed.push((index, self.fit_signal(index, *value)));
        }
        for (index, value) in indexed {
            self.values[index] = value;
        }
        Ok(())
    }

    pub fn get_signal(&self, signal: Signal) -> Result<u128, ErrorReport> {
        let index = self.signal_index(signal)?;
        Ok(self.fit_signal(index, self.values[index]))
    }

    pub fn set_memory(&mut self, memory: Signal, words: &[u128]) -> Result<(), ErrorReport> {
        let index = self.memory_signal_index(memory)?;
        let shape = &self.program.memories[index];
        if words.len() != shape.depth {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_MEMORY_VALUES",
                format!("expected {} memory words, got {}", shape.depth, words.len()),
            )]));
        }
        let ty = shape.data_layout.ty;
        self.memories[index] = words
            .iter()
            .copied()
            .map(|word| fit_u128(word, ty))
            .collect();
        self.eval_combinational();
        Ok(())
    }

    pub fn set_memory_replicated(
        &mut self,
        memory: Signal,
        words: &[u128],
    ) -> Result<(), ErrorReport> {
        self.set_memory(memory, words)
    }

    pub fn set_memory_limbs(
        &mut self,
        memory: Signal,
        lane_words: &[Vec<Vec<u32>>],
    ) -> Result<(), ErrorReport> {
        if lane_words.len() != 1 {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_MEMORY_VALUES",
                format!(
                    "expected one scalar lane of memory values, got {}",
                    lane_words.len()
                ),
            )]));
        }
        let words = lane_words[0]
            .iter()
            .map(|word| decode_u128_limbs(word))
            .collect::<Vec<_>>();
        self.set_memory(memory, &words)
    }

    pub fn get_memory_limbs(&self, memory: Signal) -> Result<Vec<Vec<Vec<u32>>>, ErrorReport> {
        let index = self.memory_signal_index(memory)?;
        let memory = &self.program.memories[index];
        let words = (0..memory.depth)
            .map(|addr| encode_u128_limbs(self.memories[index][addr], memory.data_layout.ty))
            .collect::<Vec<_>>();
        Ok(vec![words])
    }

    pub fn tick(&mut self) {
        self.eval_combinational();
        self.tick_from_evaluated_no_post_eval();
        self.eval_combinational();
    }

    pub fn tick_from_evaluated_no_post_eval(&mut self) {
        self.capture_register_stream();
        self.execute_stream(SingleLaneStreamKind::TickCommit);
        for index in 0..self.workspaces.register_captures.len() {
            let (signal, value) = self.workspaces.register_captures[index];
            self.values[signal] = self.fit_signal(signal, value);
        }
    }

    pub fn tick_many(&mut self, steps: usize) {
        for _ in 0..steps {
            self.tick();
        }
    }

    pub fn eval_combinational(&mut self) {
        self.execute_stream(SingleLaneStreamKind::Comb);
        self.execute_stream(SingleLaneStreamKind::AsyncResetComb);
    }

    pub fn replay_independent_lane_trace(
        &mut self,
        plan: &EncodedTraceReplayPlan,
        lane: usize,
        max_mismatches: usize,
    ) -> Result<ReplayReport, ErrorReport> {
        let compiled = SingleLaneCompiledReplayPlan::new(&self.program, plan)?;
        self.replay_compiled_independent_lane_trace(&compiled, lane, max_mismatches)
    }

    fn replay_compiled_independent_lane_trace(
        &mut self,
        plan: &SingleLaneCompiledReplayPlan,
        lane: usize,
        max_mismatches: usize,
    ) -> Result<ReplayReport, ErrorReport> {
        if lane >= plan.total_lanes {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_REPLAY_LANES",
                format!("lane {lane} is out of range for {} lanes", plan.total_lanes),
            )]));
        }
        let options = ReplayOptions {
            lane_mode: ReplayLaneMode::Independent,
            check_mode: ReplayCheckMode::AllLanes,
            max_mismatches,
        };
        let mut report = ReplayReport::default();
        for (step_index, step) in plan.steps.iter().enumerate() {
            let start = Instant::now();
            for input in &step.inputs {
                match input {
                    SingleLaneCompiledReplayInput::OneLimb {
                        signal,
                        mask,
                        values,
                    } => {
                        self.values[*signal] = u128::from(values[lane]) & *mask;
                    }
                    SingleLaneCompiledReplayInput::OneLimbBatch { signals, values } => {
                        for signal in signals {
                            self.values[signal.signal] =
                                u128::from(values[signal.value_base + lane]) & signal.mask;
                        }
                    }
                    SingleLaneCompiledReplayInput::Generic {
                        signal,
                        mask,
                        limbs,
                        values,
                    } => {
                        let start = lane * *limbs;
                        self.values[*signal] =
                            decode_u128_limbs(&values[start..start + *limbs]) & *mask;
                    }
                }
            }
            report.timing.input_ns += start.elapsed().as_nanos();

            let start = Instant::now();
            self.eval_combinational();
            report.timing.eval_ns += start.elapsed().as_nanos();

            let start = Instant::now();
            for check in &step.checks {
                match check {
                    SingleLaneCompiledReplayCheck::OneLimb {
                        signal,
                        check_index,
                        mask,
                        values,
                    } => {
                        let expected = u128::from(values[lane]) & *mask;
                        let actual = self.values[*signal] & *mask;
                        if actual != expected {
                            report.record_mismatch(
                                options,
                                step_index,
                                *check_index,
                                expected,
                                actual,
                                Some(lane),
                            );
                        }
                    }
                    SingleLaneCompiledReplayCheck::OneLimbBatch { signals, values } => {
                        for signal in signals {
                            let expected =
                                u128::from(values[signal.value_base + lane]) & signal.mask;
                            let actual = self.values[signal.signal] & signal.mask;
                            if actual != expected {
                                report.record_mismatch(
                                    options,
                                    step_index,
                                    signal.check_index,
                                    expected,
                                    actual,
                                    Some(lane),
                                );
                            }
                        }
                    }
                    SingleLaneCompiledReplayCheck::Generic {
                        signal,
                        check_index,
                        mask,
                        values,
                    } => {
                        let expected = values[lane] & *mask;
                        let actual = self.values[*signal] & *mask;
                        if actual != expected {
                            report.record_mismatch(
                                options,
                                step_index,
                                *check_index,
                                expected,
                                actual,
                                Some(lane),
                            );
                        }
                    }
                }
            }
            report.timing.compare_ns += start.elapsed().as_nanos();

            let start = Instant::now();
            self.tick_from_evaluated_no_post_eval();
            report.timing.tick_ns += start.elapsed().as_nanos();
        }
        Ok(report)
    }

    fn execute_stream(&mut self, stream: SingleLaneStreamKind) {
        match stream {
            SingleLaneStreamKind::AsyncResetComb => {
                if self.scalar_execution.async_reset_comb.packets.is_empty() {
                    return;
                }
                let block = std::mem::take(&mut self.scalar_execution.async_reset_comb);
                let mut env = std::mem::take(&mut self.workspaces.async_reset_comb);
                self.execute_compiled_block(&block, &mut env);
                self.workspaces.async_reset_comb = env;
                self.scalar_execution.async_reset_comb = block;
            }
            SingleLaneStreamKind::Comb => {
                if self.scalar_execution.comb.packets.is_empty() {
                    return;
                }
                let block = std::mem::take(&mut self.scalar_execution.comb);
                let mut env = std::mem::take(&mut self.workspaces.comb);
                self.execute_compiled_block(&block, &mut env);
                self.workspaces.comb = env;
                self.scalar_execution.comb = block;
            }
            SingleLaneStreamKind::TickCommit => {
                if self.scalar_execution.tick_commit.packets.is_empty() {
                    return;
                }
                let block = std::mem::take(&mut self.scalar_execution.tick_commit);
                let mut env = std::mem::take(&mut self.workspaces.tick_commit);
                self.execute_compiled_block(&block, &mut env);
                self.workspaces.tick_commit = env;
                self.scalar_execution.tick_commit = block;
            }
        }
    }

    fn capture_register_stream(&mut self) {
        self.workspaces.register_captures.clear();
        if self.scalar_execution.tick_next.packets.is_empty() {
            return;
        }
        let block = std::mem::take(&mut self.scalar_execution.tick_next);
        let mut env = std::mem::take(&mut self.workspaces.tick_next);
        let mut captures = std::mem::take(&mut self.workspaces.register_captures);
        self.capture_compiled_registers(&block, &mut env, &mut captures);
        self.workspaces.tick_next = env;
        self.workspaces.register_captures = captures;
        self.scalar_execution.tick_next = block;
    }

    fn capture_compiled_registers(
        &self,
        block: &SingleLaneCompiledBlock,
        env: &mut [u128],
        captured: &mut Vec<(usize, u128)>,
    ) {
        for packet in &block.packets {
            for op in &packet.ops {
                self.eval_compiled_single_lane_op(op, env);
            }
            for effect in &packet.effects {
                if let SingleLaneEffect::CaptureReg {
                    dst, value, reset, ..
                } = effect
                {
                    let next = if reset
                        .as_ref()
                        .is_some_and(|reset| self.compiled_reset_asserted(*reset))
                    {
                        reset.as_ref().unwrap().value
                    } else {
                        self.machine_value(env, *value)
                    };
                    captured.push((*dst, next));
                }
            }
        }
    }

    fn execute_compiled_block(&mut self, block: &SingleLaneCompiledBlock, env: &mut [u128]) {
        for packet in &block.packets {
            for op in &packet.ops {
                self.eval_compiled_single_lane_op(op, env);
            }
            for effect in &packet.effects {
                self.execute_single_lane_effect(effect, env);
            }
        }
    }

    fn execute_single_lane_effect(&mut self, effect: &SingleLaneEffect, env: &[u128]) {
        match effect {
            SingleLaneEffect::StoreSignal { dst, value, mask } => {
                self.values[*dst] = self.machine_value(env, *value) & *mask;
            }
            SingleLaneEffect::CaptureReg {
                dst,
                value,
                reset,
                mask,
            } => {
                let Some(reset) = reset else {
                    return;
                };
                if reset.kind != ResetKind::Async || !self.compiled_reset_asserted(*reset) {
                    return;
                }
                self.values[*dst] = self.machine_value(env, *value) & *mask;
            }
            SingleLaneEffect::MemoryWrite {
                memory,
                enable,
                addr,
                data,
                mask,
            } => {
                if self.machine_value(env, *enable) & 1 == 0 {
                    return;
                }
                let addr = self.machine_value(env, *addr) as usize;
                if addr < self.memories[*memory].len() {
                    self.memories[*memory][addr] = self.machine_value(env, *data) & *mask;
                }
            }
        }
    }

    fn eval_compiled_single_lane_op(&self, op: &SingleLaneCompiledOp, env: &mut [u128]) {
        match op {
            SingleLaneCompiledOp::Fast(op) => self.eval_single_lane_fast_op(op, env),
            SingleLaneCompiledOp::Wide(op) => self.eval_single_lane_op(op, env),
        }
    }

    fn eval_single_lane_fast_op(&self, op: &SingleLaneFastOp, env: &mut [u128]) {
        match op {
            SingleLaneFastOp::Lit { dst, value } => env[*dst] = u128::from(*value),
            SingleLaneFastOp::Signal { dst, signal, mask } => {
                env[*dst] = u128::from(self.values[*signal] as u32 & *mask);
            }
            SingleLaneFastOp::Not { dst, value, mask } => {
                env[*dst] = u128::from(!(env[*value] as u32) & *mask);
            }
            SingleLaneFastOp::Binary {
                dst,
                lhs,
                rhs,
                kind,
                mask,
            } => {
                let lhs = env[*lhs] as u32;
                let rhs = env[*rhs] as u32;
                let value = match kind {
                    SingleLaneBinaryOp::And => lhs & rhs,
                    SingleLaneBinaryOp::Or => lhs | rhs,
                    SingleLaneBinaryOp::Xor => lhs ^ rhs,
                    SingleLaneBinaryOp::Add => lhs.wrapping_add(rhs),
                    SingleLaneBinaryOp::Sub => lhs.wrapping_sub(rhs),
                    SingleLaneBinaryOp::Mul => lhs.wrapping_mul(rhs),
                };
                env[*dst] = u128::from(value & *mask);
            }
            SingleLaneFastOp::Compare {
                dst,
                lhs,
                rhs,
                kind,
                mask,
            } => {
                let lhs = env[*lhs] as u32 & *mask;
                let rhs = env[*rhs] as u32 & *mask;
                env[*dst] = u128::from(match kind {
                    SingleLaneCompareOp::Eq => lhs == rhs,
                    SingleLaneCompareOp::Ne => lhs != rhs,
                    SingleLaneCompareOp::Lt { signed: false } => lhs < rhs,
                    SingleLaneCompareOp::Lt { signed: true } => unreachable!(),
                });
            }
            SingleLaneFastOp::Mux {
                dst,
                cond,
                then_value,
                else_value,
                mask,
            } => {
                let value = if env[*cond] as u32 & 1 != 0 {
                    env[*then_value] as u32
                } else {
                    env[*else_value] as u32
                };
                env[*dst] = u128::from(value & *mask);
            }
            SingleLaneFastOp::Slice {
                dst,
                value,
                lsb,
                mask,
            } => {
                let value = if *lsb >= 32 {
                    0
                } else {
                    (env[*value] as u32) >> *lsb
                };
                env[*dst] = u128::from(value & *mask);
            }
            SingleLaneFastOp::Pass { dst, value, mask } => {
                env[*dst] = u128::from(env[*value] as u32 & *mask);
            }
            SingleLaneFastOp::Concat { dst, values, mask } => {
                let mut out = 0u32;
                let mut offset = 0u32;
                for (value, width) in values.iter().rev() {
                    if offset < 32 {
                        out |= ((env[*value] as u32) & mask_u32(*width)) << offset;
                    }
                    offset = offset.saturating_add(*width);
                }
                env[*dst] = u128::from(out & *mask);
            }
        }
    }

    fn eval_single_lane_op(&self, op: &SingleLaneOp, env: &mut [u128]) {
        match op {
            SingleLaneOp::Lit { dst, value, mask } => env[*dst] = *value & *mask,
            SingleLaneOp::Signal { dst, signal, mask } => env[*dst] = self.values[*signal] & *mask,
            SingleLaneOp::Not { dst, value, mask } => {
                env[*dst] = !self.machine_value(env, *value) & *mask
            }
            SingleLaneOp::Binary {
                dst,
                lhs,
                rhs,
                kind,
                mask,
            } => {
                let lhs = self.machine_value(env, *lhs);
                let rhs = self.machine_value(env, *rhs);
                let value = match kind {
                    SingleLaneBinaryOp::And => lhs & rhs,
                    SingleLaneBinaryOp::Or => lhs | rhs,
                    SingleLaneBinaryOp::Xor => lhs ^ rhs,
                    SingleLaneBinaryOp::Add => lhs.wrapping_add(rhs),
                    SingleLaneBinaryOp::Sub => lhs.wrapping_sub(rhs),
                    SingleLaneBinaryOp::Mul => lhs.wrapping_mul(rhs),
                };
                env[*dst] = value & *mask;
            }
            SingleLaneOp::Compare {
                dst,
                lhs,
                rhs,
                kind,
                width,
            } => {
                let lhs = self.machine_value(env, *lhs);
                let rhs = self.machine_value(env, *rhs);
                env[*dst] = u128::from(match kind {
                    SingleLaneCompareOp::Eq => {
                        fit_width_u128(lhs, *width) == fit_width_u128(rhs, *width)
                    }
                    SingleLaneCompareOp::Ne => {
                        fit_width_u128(lhs, *width) != fit_width_u128(rhs, *width)
                    }
                    SingleLaneCompareOp::Lt { signed } => lt_u128(lhs, rhs, *width, *signed),
                });
            }
            SingleLaneOp::Mux {
                dst,
                cond,
                then_value,
                else_value,
                mask,
            } => {
                let value = if self.machine_value(env, *cond) & 1 != 0 {
                    self.machine_value(env, *then_value)
                } else {
                    self.machine_value(env, *else_value)
                };
                env[*dst] = value & *mask;
            }
            SingleLaneOp::Slice {
                dst,
                value,
                lsb,
                mask,
            } => env[*dst] = shift_right_u128(self.machine_value(env, *value), *lsb) & *mask,
            SingleLaneOp::Pass { dst, value, mask } => {
                env[*dst] = self.machine_value(env, *value) & *mask
            }
            SingleLaneOp::Sext {
                dst,
                value,
                from_width,
                to_width,
                mask,
            } => {
                env[*dst] =
                    sext_u128(self.machine_value(env, *value), *from_width, *to_width) & *mask
            }
            SingleLaneOp::Concat { dst, values, mask } => {
                let mut out = 0u128;
                let mut offset = 0u32;
                for (value, width) in values.iter().rev() {
                    if offset < 128 {
                        out |= fit_width_u128(self.machine_value(env, *value), *width) << offset;
                    }
                    offset = offset.saturating_add(*width);
                }
                env[*dst] = out & *mask;
            }
            SingleLaneOp::MemRead {
                dst,
                memory,
                addr,
                mask,
            } => {
                let addr = self.machine_value(env, *addr) as usize;
                env[*dst] = self
                    .memories
                    .get(*memory)
                    .and_then(|memory| memory.get(addr))
                    .copied()
                    .unwrap_or(0)
                    & *mask;
            }
        }
    }

    fn machine_value(&self, env: &[u128], value: usize) -> u128 {
        env.get(value).copied().unwrap_or(0)
    }

    fn compiled_reset_asserted(&self, reset: SingleLaneReset) -> bool {
        let value = self.fit_signal(reset.signal, self.values[reset.signal]);
        match reset.polarity {
            ResetPolarity::ActiveHigh => value & 1 != 0,
            ResetPolarity::ActiveLow => value & 1 == 0,
        }
    }

    fn signal_index(&self, signal: Signal) -> Result<usize, ErrorReport> {
        self.program.signal_index(signal).ok_or_else(|| {
            ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_SIGNAL",
                format!("signal {:?} is not part of this packed program", signal.id),
            )])
        })
    }

    fn memory_signal_index(&self, memory: Signal) -> Result<usize, ErrorReport> {
        self.program
            .memories
            .iter()
            .position(|packed| packed.source == memory)
            .ok_or_else(|| {
                ErrorReport::new(vec![Diagnostic::new(
                    "E_SIM_IR_MEMORY",
                    format!("memory {:?} is not part of this packed program", memory.id),
                )])
            })
    }

    fn fit_signal(&self, signal: usize, value: u128) -> u128 {
        fit_u128(value, self.program.signals[signal].layout.ty)
    }
}

impl SimBackend for SingleLaneMachineSimulator {
    fn kind(&self) -> SimBackendKind {
        SimBackendKind::Scalar
    }

    fn lanes(&self) -> usize {
        1
    }

    fn set_signals_replicated(&mut self, values: &[(Signal, u128)]) -> Result<(), ErrorReport> {
        self.set_signals(values)
    }

    fn set_signals_replicated_raw(&mut self, values: &[(Signal, u128)]) -> Result<(), ErrorReport> {
        self.set_signals_raw(values)
    }

    fn get_signal_lane(&self, signal: Signal, lane: usize) -> Result<u128, ErrorReport> {
        if lane != 0 {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_IR_LANE",
                format!("lane {lane} is out of range for scalar backend"),
            )]));
        }
        self.get_signal(signal)
    }

    fn set_memory_replicated(&mut self, memory: Signal, words: &[u128]) -> Result<(), ErrorReport> {
        self.set_memory(memory, words)
    }

    fn eval_combinational(&mut self) -> Result<(), ErrorReport> {
        SingleLaneMachineSimulator::eval_combinational(self);
        Ok(())
    }

    fn tick(&mut self) -> Result<(), ErrorReport> {
        SingleLaneMachineSimulator::tick(self);
        Ok(())
    }

    fn tick_from_evaluated_no_post_eval(&mut self) -> Result<(), ErrorReport> {
        SingleLaneMachineSimulator::tick_from_evaluated_no_post_eval(self);
        Ok(())
    }

    fn tick_many(&mut self, steps: usize) -> Result<(), ErrorReport> {
        SingleLaneMachineSimulator::tick_many(self, steps);
        Ok(())
    }
}

impl SimBackend for PackedSimulator {
    fn kind(&self) -> SimBackendKind {
        SimBackendKind::PackedCpu
    }

    fn lanes(&self) -> usize {
        self.lanes
    }

    fn set_signals_replicated(&mut self, values: &[(Signal, u128)]) -> Result<(), ErrorReport> {
        PackedSimulator::set_signals_replicated(self, values)
    }

    fn set_signals_replicated_raw(&mut self, values: &[(Signal, u128)]) -> Result<(), ErrorReport> {
        PackedSimulator::set_signals_replicated_raw(self, values)
    }

    fn get_signal_lane(&self, signal: Signal, lane: usize) -> Result<u128, ErrorReport> {
        PackedSimulator::get_signal_lane(self, signal, lane)
    }

    fn set_memory_replicated(&mut self, memory: Signal, words: &[u128]) -> Result<(), ErrorReport> {
        PackedSimulator::set_memory_replicated(self, memory, words)
    }

    fn eval_combinational(&mut self) -> Result<(), ErrorReport> {
        PackedSimulator::eval_combinational(self);
        Ok(())
    }

    fn tick(&mut self) -> Result<(), ErrorReport> {
        PackedSimulator::tick(self);
        Ok(())
    }

    fn tick_from_evaluated_no_post_eval(&mut self) -> Result<(), ErrorReport> {
        PackedSimulator::tick_from_evaluated_no_post_eval(self);
        Ok(())
    }

    fn tick_many(&mut self, steps: usize) -> Result<(), ErrorReport> {
        PackedSimulator::tick_many(self, steps);
        Ok(())
    }

    fn replay_trace(
        &mut self,
        plan: &EncodedTraceReplayPlan,
        options: ReplayOptions,
    ) -> Result<ReplayReport, ErrorReport> {
        PackedSimulator::replay_trace(self, plan, options)
    }
}

impl SimBackend for SimdCpuSimulator {
    fn kind(&self) -> SimBackendKind {
        SimBackendKind::SimdCpu
    }

    fn lanes(&self) -> usize {
        self.inner.lanes()
    }

    fn set_signals_replicated(&mut self, values: &[(Signal, u128)]) -> Result<(), ErrorReport> {
        SimdCpuSimulator::set_signals_replicated(self, values)
    }

    fn set_signals_replicated_raw(&mut self, values: &[(Signal, u128)]) -> Result<(), ErrorReport> {
        SimdCpuSimulator::set_signals_replicated_raw(self, values)
    }

    fn get_signal_lane(&self, signal: Signal, lane: usize) -> Result<u128, ErrorReport> {
        self.inner.get_signal_lane(signal, lane)
    }

    fn set_memory_replicated(&mut self, memory: Signal, words: &[u128]) -> Result<(), ErrorReport> {
        SimdCpuSimulator::set_memory_replicated(self, memory, words)
    }

    fn eval_combinational(&mut self) -> Result<(), ErrorReport> {
        SimdCpuSimulator::eval_combinational(self)
    }

    fn tick(&mut self) -> Result<(), ErrorReport> {
        SimdCpuSimulator::tick(self)
    }

    fn tick_from_evaluated_no_post_eval(&mut self) -> Result<(), ErrorReport> {
        SimdCpuSimulator::tick_from_evaluated_no_post_eval(self)
    }

    fn tick_many(&mut self, steps: usize) -> Result<(), ErrorReport> {
        SimdCpuSimulator::tick_many(self, steps)
    }

    fn replay_trace(
        &mut self,
        plan: &EncodedTraceReplayPlan,
        options: ReplayOptions,
    ) -> Result<ReplayReport, ErrorReport> {
        SimdCpuSimulator::replay_trace(self, plan, options)
    }
}

impl SimBackend for JitCpuSimulator {
    fn kind(&self) -> SimBackendKind {
        SimBackendKind::JitCpu
    }

    fn lanes(&self) -> usize {
        self.inner.lanes()
    }

    fn set_signals_replicated(&mut self, values: &[(Signal, u128)]) -> Result<(), ErrorReport> {
        self.inner.set_signals_replicated(values)
    }

    fn set_signals_replicated_raw(&mut self, values: &[(Signal, u128)]) -> Result<(), ErrorReport> {
        self.inner.set_signals_replicated_raw(values)
    }

    fn get_signal_lane(&self, signal: Signal, lane: usize) -> Result<u128, ErrorReport> {
        self.inner.get_signal_lane(signal, lane)
    }

    fn set_memory_replicated(&mut self, memory: Signal, words: &[u128]) -> Result<(), ErrorReport> {
        self.inner.set_memory_replicated(memory, words)
    }

    fn eval_combinational(&mut self) -> Result<(), ErrorReport> {
        self.inner.eval_combinational()
    }

    fn tick(&mut self) -> Result<(), ErrorReport> {
        self.inner.tick()
    }

    fn tick_from_evaluated_no_post_eval(&mut self) -> Result<(), ErrorReport> {
        self.inner.tick_from_evaluated_no_post_eval()
    }

    fn tick_many(&mut self, steps: usize) -> Result<(), ErrorReport> {
        self.inner.tick_many(steps)
    }

    fn replay_trace(
        &mut self,
        plan: &EncodedTraceReplayPlan,
        options: ReplayOptions,
    ) -> Result<ReplayReport, ErrorReport> {
        self.inner.replay_trace(plan, options)
    }
}

impl SimBackend for SimBackendInstance {
    fn kind(&self) -> SimBackendKind {
        match self {
            Self::Scalar(sim) => sim.kind(),
            Self::PackedCpu(sim) => sim.kind(),
            Self::SimdCpu(sim) => sim.kind(),
            Self::JitCpu(sim) => sim.kind(),
        }
    }

    fn lanes(&self) -> usize {
        match self {
            Self::Scalar(sim) => SimBackend::lanes(sim),
            Self::PackedCpu(sim) => SimBackend::lanes(sim),
            Self::SimdCpu(sim) => SimBackend::lanes(sim),
            Self::JitCpu(sim) => SimBackend::lanes(sim),
        }
    }

    fn set_signals_replicated(&mut self, values: &[(Signal, u128)]) -> Result<(), ErrorReport> {
        match self {
            Self::Scalar(sim) => sim.set_signals_replicated(values),
            Self::PackedCpu(sim) => sim.set_signals_replicated(values),
            Self::SimdCpu(sim) => sim.set_signals_replicated(values),
            Self::JitCpu(sim) => sim.set_signals_replicated(values),
        }
    }

    fn set_signals_replicated_raw(&mut self, values: &[(Signal, u128)]) -> Result<(), ErrorReport> {
        match self {
            Self::Scalar(sim) => sim.set_signals_replicated_raw(values),
            Self::PackedCpu(sim) => sim.set_signals_replicated_raw(values),
            Self::SimdCpu(sim) => sim.set_signals_replicated_raw(values),
            Self::JitCpu(sim) => sim.set_signals_replicated_raw(values),
        }
    }

    fn get_signal_lane(&self, signal: Signal, lane: usize) -> Result<u128, ErrorReport> {
        match self {
            Self::Scalar(sim) => sim.get_signal_lane(signal, lane),
            Self::PackedCpu(sim) => sim.get_signal_lane(signal, lane),
            Self::SimdCpu(sim) => sim.get_signal_lane(signal, lane),
            Self::JitCpu(sim) => sim.get_signal_lane(signal, lane),
        }
    }

    fn set_memory_replicated(&mut self, memory: Signal, words: &[u128]) -> Result<(), ErrorReport> {
        match self {
            Self::Scalar(sim) => sim.set_memory_replicated(memory, words),
            Self::PackedCpu(sim) => sim.set_memory_replicated(memory, words),
            Self::SimdCpu(sim) => sim.set_memory_replicated(memory, words),
            Self::JitCpu(sim) => sim.set_memory_replicated(memory, words),
        }
    }

    fn eval_combinational(&mut self) -> Result<(), ErrorReport> {
        match self {
            Self::Scalar(sim) => SimBackend::eval_combinational(sim),
            Self::PackedCpu(sim) => SimBackend::eval_combinational(sim),
            Self::SimdCpu(sim) => SimBackend::eval_combinational(sim),
            Self::JitCpu(sim) => SimBackend::eval_combinational(sim),
        }
    }

    fn tick(&mut self) -> Result<(), ErrorReport> {
        match self {
            Self::Scalar(sim) => SimBackend::tick(sim),
            Self::PackedCpu(sim) => SimBackend::tick(sim),
            Self::SimdCpu(sim) => SimBackend::tick(sim),
            Self::JitCpu(sim) => SimBackend::tick(sim),
        }
    }

    fn tick_from_evaluated_no_post_eval(&mut self) -> Result<(), ErrorReport> {
        match self {
            Self::Scalar(sim) => SimBackend::tick_from_evaluated_no_post_eval(sim),
            Self::PackedCpu(sim) => SimBackend::tick_from_evaluated_no_post_eval(sim),
            Self::SimdCpu(sim) => SimBackend::tick_from_evaluated_no_post_eval(sim),
            Self::JitCpu(sim) => SimBackend::tick_from_evaluated_no_post_eval(sim),
        }
    }

    fn tick_many(&mut self, steps: usize) -> Result<(), ErrorReport> {
        match self {
            Self::Scalar(sim) => SimBackend::tick_many(sim, steps),
            Self::PackedCpu(sim) => SimBackend::tick_many(sim, steps),
            Self::SimdCpu(sim) => SimBackend::tick_many(sim, steps),
            Self::JitCpu(sim) => SimBackend::tick_many(sim, steps),
        }
    }

    fn replay_trace(
        &mut self,
        plan: &EncodedTraceReplayPlan,
        options: ReplayOptions,
    ) -> Result<ReplayReport, ErrorReport> {
        match self {
            Self::Scalar(sim) => SimBackend::replay_trace(sim, plan, options),
            Self::PackedCpu(sim) => SimBackend::replay_trace(sim, plan, options),
            Self::SimdCpu(sim) => SimBackend::replay_trace(sim, plan, options),
            Self::JitCpu(sim) => SimBackend::replay_trace(sim, plan, options),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum SingleLaneStreamKind {
    AsyncResetComb,
    Comb,
    TickCommit,
}

fn validate_single_lane_widths(
    program: &PackedProgram,
    machine: &PackedMachineProgram,
) -> Result<(), ErrorReport> {
    let mut diagnostics = Vec::new();
    for signal in &program.signals {
        if signal.layout.width > 128 {
            diagnostics.push(Diagnostic::new(
                "E_SIM_IR_WIDE_VALUE",
                format!(
                    "single-lane machine simulator does not support signal `{}` width {}",
                    signal.name, signal.layout.width
                ),
            ));
        }
    }
    for memory in &program.memories {
        if memory.data_layout.width > 128 {
            diagnostics.push(Diagnostic::new(
                "E_SIM_IR_WIDE_VALUE",
                format!(
                    "single-lane machine simulator does not support memory `{}` width {}",
                    memory.name, memory.data_layout.width
                ),
            ));
        }
    }
    for block in [
        &machine.streams.async_reset_comb,
        &machine.streams.comb,
        &machine.streams.tick_next,
        &machine.streams.tick_commit,
    ] {
        for packet in &block.packets {
            for instr in &packet.instrs {
                if instr.ty.width > 128 {
                    diagnostics.push(Diagnostic::new(
                        "E_SIM_IR_WIDE_VALUE",
                        format!(
                            "single-lane machine simulator does not support value width {}",
                            instr.ty.width
                        ),
                    ));
                }
            }
        }
    }
    if diagnostics.is_empty() {
        Ok(())
    } else {
        Err(ErrorReport::new(diagnostics))
    }
}

fn fit_u128(value: u128, ty: BitType) -> u128 {
    fit_width_u128(value, ty.width)
}

fn fit_width_u128(value: u128, width: Width) -> u128 {
    value & mask_u128(width)
}

fn mask_u128(width: Width) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

fn mask_u64(width: Width) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

fn mask_u32(width: Width) -> u32 {
    if width >= 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    }
}

fn shift_right_u128(value: u128, shift: Width) -> u128 {
    if shift >= 128 {
        0
    } else {
        value >> shift
    }
}

fn sext_u128(value: u128, input_width: Width, output_width: Width) -> u128 {
    let value = fit_width_u128(value, input_width);
    if input_width == 0 || input_width >= output_width {
        return fit_width_u128(value, output_width);
    }
    let sign = (value >> (input_width - 1)) & 1 != 0;
    if sign {
        fit_width_u128(value | (!mask_u128(input_width)), output_width)
    } else {
        fit_width_u128(value, output_width)
    }
}

fn lt_u128(lhs: u128, rhs: u128, width: Width, signed: bool) -> bool {
    let lhs = fit_width_u128(lhs, width);
    let rhs = fit_width_u128(rhs, width);
    if signed && width > 0 {
        let sign_bit = 1u128 << (width - 1);
        let lhs_sign = lhs & sign_bit != 0;
        let rhs_sign = rhs & sign_bit != 0;
        if lhs_sign != rhs_sign {
            return lhs_sign;
        }
    }
    lhs < rhs
}

struct Lowering<'a> {
    design: &'a CompiledDesign,
    diagnostics: Vec<Diagnostic>,
    contexts: Vec<InstanceCtx>,
    signals: Vec<PackedSignal>,
    memories: Vec<PackedMemory>,
    signal_map: HashMap<FlatKey, usize>,
    memory_map: HashMap<FlatKey, usize>,
    top_signal_indices: HashMap<Signal, usize>,
    assignments: Vec<FlatAssignment>,
    registers: Vec<CompiledRegRef>,
    memory_writes: Vec<PackedOp>,
    /// Memory index → its write clock's signal index (one clock per memory).
    mem_clocks: HashMap<usize, usize>,
    next_signal_offset: usize,
    next_memory_offset: usize,
}

#[derive(Clone, Debug)]
struct CompiledRegRef {
    signal: usize,
    next: PackedExpr,
    reset: Option<PackedReset>,
    /// Packed signal index of this register's clock, if one is associated.
    clock: Option<usize>,
}

impl<'a> Lowering<'a> {
    fn add_context(&mut self, module: CompiledModule, path: String) -> usize {
        let index = self.contexts.len();
        for instance in &module.instances {
            let Some(target) = self.design.find_module(&instance.module).cloned() else {
                self.diagnostics.push(
                    Diagnostic::new(
                        "E_SIM_IR_INSTANCE",
                        format!("instance target `{}` does not exist", instance.module),
                    )
                    .with_module(module.name.clone()),
                );
                continue;
            };
            if target.is_external {
                self.diagnostics.push(
                    Diagnostic::new(
                        "E_SIM_IR_EXTERN",
                        format!(
                            "instance `{}` targets external module `{}`",
                            instance.name, target.name
                        ),
                    )
                    .with_module(module.name.clone()),
                );
                continue;
            }
            self.add_context(target, format!("{path}.{}", instance.name));
        }
        self.contexts.insert(index, InstanceCtx { path, module });
        index
    }

    fn allocate_contexts(&mut self) {
        for instance in 0..self.contexts.len() {
            let ctx = self.contexts[instance].clone();
            if !ctx.module.assertions.is_empty() || !ctx.module.cover_points.is_empty() {
                self.diagnostics.push(
                    Diagnostic::new(
                        "E_SIM_IR_ASSERT_COVER",
                        "packed simulation IR does not support assertions or cover points",
                    )
                    .with_module(ctx.module.name.clone()),
                );
            }
            for signal in &ctx.module.signals {
                if matches!(signal.kind, SignalKind::Inout) {
                    self.diagnostics.push(
                        Diagnostic::new(
                            "E_SIM_IR_INOUT",
                            "packed simulation IR does not support inout",
                        )
                        .with_module(ctx.module.name.clone())
                        .with_signal(signal.name.clone()),
                    );
                    continue;
                }
                let key = FlatKey {
                    instance,
                    signal: signal.handle,
                };
                if let SignalKind::Mem {
                    addr_width,
                    data_width: _,
                    depth,
                } = signal.kind
                {
                    let layout = self.alloc_memory_layout(signal.ty);
                    let index = self.memories.len();
                    self.memory_map.insert(key, index);
                    self.memories.push(PackedMemory {
                        name: format!("{}.{}", ctx.path, signal.name),
                        source: signal.handle,
                        owner_path: ctx.path.clone(),
                        addr_width,
                        depth,
                        data_layout: layout,
                        offset: self.next_memory_offset,
                    });
                    self.next_memory_offset += depth * layout.limbs;
                } else {
                    let layout = self.alloc_signal_layout(signal.ty);
                    let index = self.signals.len();
                    self.signal_map.insert(key, index);
                    if instance == 0 {
                        self.top_signal_indices.insert(signal.handle, index);
                    }
                    self.signals.push(PackedSignal {
                        name: format!("{}.{}", ctx.path, signal.name),
                        source: if instance == 0 {
                            Some(signal.handle)
                        } else {
                            None
                        },
                        owner_path: ctx.path.clone(),
                        layout,
                        kind: packed_signal_kind(&signal.kind),
                    });
                }
            }
        }
    }

    fn lower_contexts(&mut self) {
        for instance in 0..self.contexts.len() {
            let module = self.contexts[instance].module.clone();
            for assignment in &module.assignments {
                let Some(dst) = self.signal_index(instance, assignment.dst, &module) else {
                    continue;
                };
                let expr = self.lower_expr(instance, &module, &assignment.expr);
                self.assignments.push(FlatAssignment { dst, expr });
            }
            for child in &module.instances {
                let Some(child_instance) = self.find_child_context(instance, &child.name) else {
                    continue;
                };
                let Some(child_module) = self.design.find_module(&child.module).cloned() else {
                    continue;
                };
                for connection in &child.connections {
                    let Some(port) = child_module
                        .signals
                        .iter()
                        .find(|signal| signal.name == connection.port)
                    else {
                        continue;
                    };
                    match connection.direction {
                        PortDirection::Input => {
                            if let (Some(dst), Some(src)) = (
                                self.signal_index(child_instance, port.handle, &child_module),
                                self.signal_index(instance, connection.signal, &module),
                            ) {
                                self.assignments.push(FlatAssignment {
                                    dst,
                                    expr: self.signal_expr(src),
                                });
                            }
                        }
                        PortDirection::Output => {
                            if let (Some(dst), Some(src)) = (
                                self.signal_index(instance, connection.signal, &module),
                                self.signal_index(child_instance, port.handle, &child_module),
                            ) {
                                self.assignments.push(FlatAssignment {
                                    dst,
                                    expr: self.signal_expr(src),
                                });
                            }
                        }
                        PortDirection::Inout => {}
                    }
                }
            }
            for reg in &module.registers {
                let Some(signal) = self.signal_index(instance, reg.signal, &module) else {
                    continue;
                };
                let next = self.lower_expr(instance, &module, &reg.next);
                let reset = reg
                    .reset
                    .as_ref()
                    .and_then(|reset| self.lower_reset(instance, &module, reg, reset));
                let clock = self.signal_index(instance, reg.clock, &module);
                self.registers.push(CompiledRegRef {
                    signal,
                    next,
                    reset,
                    clock,
                });
            }
            for write in &module.memory_writes {
                if let Some(op) = self.lower_memory_write(instance, &module, write) {
                    if let PackedOp::MemoryWrite { memory, .. } = &op {
                        if let Some(clk) = self.signal_index(instance, write.clock, &module) {
                            // One clock per memory (the common case); first wins.
                            self.mem_clocks.entry(*memory).or_insert(clk);
                        }
                    }
                    self.memory_writes.push(op);
                }
            }
        }
    }

    fn lower_expr(
        &mut self,
        instance: usize,
        module: &CompiledModule,
        expr: &CompiledExpr,
    ) -> PackedExpr {
        self.lower_raw_expr(instance, module, &expr.expr, expr.ty)
    }

    fn lower_raw_expr(
        &mut self,
        instance: usize,
        module: &CompiledModule,
        expr: &Expr,
        ty: BitType,
    ) -> PackedExpr {
        let kind = match expr {
            Expr::Lit { value, ty } => PackedExprKind::Lit(encode_limbs(*value, *ty)),
            Expr::Signal(signal) => match self.signal_index(instance, *signal, module) {
                Some(index) => PackedExprKind::Signal(index),
                None => PackedExprKind::Lit(vec![0; limbs(ty.width)]),
            },
            Expr::Not(inner) => {
                PackedExprKind::Not(Box::new(self.lower_raw_expr(instance, module, inner, ty)))
            }
            Expr::And(lhs, rhs) => {
                self.lower_binary(instance, module, lhs, rhs, ty, PackedExprKind::And)
            }
            Expr::Or(lhs, rhs) => {
                self.lower_binary(instance, module, lhs, rhs, ty, PackedExprKind::Or)
            }
            Expr::Xor(lhs, rhs) => {
                self.lower_binary(instance, module, lhs, rhs, ty, PackedExprKind::Xor)
            }
            Expr::Add(lhs, rhs) => {
                self.lower_binary(instance, module, lhs, rhs, ty, PackedExprKind::Add)
            }
            Expr::Sub(lhs, rhs) => {
                self.lower_binary(instance, module, lhs, rhs, ty, PackedExprKind::Sub)
            }
            Expr::Mul(lhs, rhs) => {
                self.lower_binary(instance, module, lhs, rhs, ty, PackedExprKind::Mul)
            }
            Expr::Eq(lhs, rhs) => {
                self.lower_binary(instance, module, lhs, rhs, ty, PackedExprKind::Eq)
            }
            Expr::Ne(lhs, rhs) => {
                self.lower_binary(instance, module, lhs, rhs, ty, PackedExprKind::Ne)
            }
            Expr::Lt(lhs, rhs) => {
                let lhs_ty = expr_type(module, lhs).unwrap_or(ty);
                PackedExprKind::Lt {
                    lhs: Box::new(self.lower_raw_expr(instance, module, lhs, lhs_ty)),
                    rhs: Box::new(self.lower_raw_expr(instance, module, rhs, lhs_ty)),
                    signed: lhs_ty.signedness == Signedness::Signed,
                }
            }
            Expr::Mux {
                cond,
                then_expr,
                else_expr,
            } => PackedExprKind::Mux {
                cond: Box::new(self.lower_raw_expr(instance, module, cond, rrtl_ir::uint(1))),
                then_expr: Box::new(self.lower_raw_expr(instance, module, then_expr, ty)),
                else_expr: Box::new(self.lower_raw_expr(instance, module, else_expr, ty)),
            },
            Expr::Slice { expr, lsb, .. } => {
                let inner_ty = expr_type(module, expr).unwrap_or(ty);
                PackedExprKind::Slice {
                    expr: Box::new(self.lower_raw_expr(instance, module, expr, inner_ty)),
                    lsb: *lsb,
                }
            }
            Expr::Zext { expr, .. } => {
                let inner_ty = expr_type(module, expr).unwrap_or(ty);
                PackedExprKind::Zext(Box::new(
                    self.lower_raw_expr(instance, module, expr, inner_ty),
                ))
            }
            Expr::Sext { expr, .. } => {
                let inner_ty = expr_type(module, expr).unwrap_or(ty);
                PackedExprKind::Sext(Box::new(
                    self.lower_raw_expr(instance, module, expr, inner_ty),
                ))
            }
            Expr::Trunc { expr, .. } => {
                let inner_ty = expr_type(module, expr).unwrap_or(ty);
                PackedExprKind::Trunc(Box::new(
                    self.lower_raw_expr(instance, module, expr, inner_ty),
                ))
            }
            Expr::Cast { expr, .. } => {
                let inner_ty = expr_type(module, expr).unwrap_or(ty);
                PackedExprKind::Cast(Box::new(
                    self.lower_raw_expr(instance, module, expr, inner_ty),
                ))
            }
            Expr::Concat(parts) => {
                let packed = parts
                    .iter()
                    .map(|part| {
                        let part_ty = expr_type(module, part).unwrap_or(rrtl_ir::uint(1));
                        self.lower_raw_expr(instance, module, part, part_ty)
                    })
                    .collect();
                PackedExprKind::Concat(packed)
            }
            Expr::MemRead { mem, addr } => {
                let memory = self.memory_index(instance, *mem, module).unwrap_or(0);
                let addr_ty = expr_type(module, addr).unwrap_or(rrtl_ir::uint(32));
                PackedExprKind::MemRead {
                    memory,
                    addr: Box::new(self.lower_raw_expr(instance, module, addr, addr_ty)),
                }
            }
        };
        PackedExpr { kind, ty }
    }

    fn lower_binary(
        &mut self,
        instance: usize,
        module: &CompiledModule,
        lhs: &Expr,
        rhs: &Expr,
        ty: BitType,
        ctor: fn(Box<PackedExpr>, Box<PackedExpr>) -> PackedExprKind,
    ) -> PackedExprKind {
        let lhs_ty = expr_type(module, lhs).unwrap_or(ty);
        ctor(
            Box::new(self.lower_raw_expr(instance, module, lhs, lhs_ty)),
            Box::new(self.lower_raw_expr(instance, module, rhs, lhs_ty)),
        )
    }

    fn lower_reset(
        &mut self,
        instance: usize,
        module: &CompiledModule,
        reg: &CompiledRegister,
        reset: &Reset,
    ) -> Option<PackedReset> {
        let signal = self.signal_index(instance, reset.signal, module)?;
        Some(PackedReset {
            signal,
            value: encode_limbs(reset.value, reg.next.ty),
            kind: reset.kind,
            polarity: reset.polarity,
        })
    }

    fn lower_memory_write(
        &mut self,
        instance: usize,
        module: &CompiledModule,
        write: &CompiledMemoryWrite,
    ) -> Option<PackedOp> {
        Some(PackedOp::MemoryWrite {
            memory: self.memory_index(instance, write.mem, module)?,
            enable: self.lower_expr(instance, module, &write.enable),
            addr: self.lower_expr(instance, module, &write.addr),
            data: self.lower_expr(instance, module, &write.data),
        })
    }

    fn signal_expr(&self, index: usize) -> PackedExpr {
        PackedExpr {
            kind: PackedExprKind::Signal(index),
            ty: self.signals[index].layout.ty,
        }
    }

    fn signal_index(
        &mut self,
        instance: usize,
        signal: Signal,
        module: &CompiledModule,
    ) -> Option<usize> {
        let key = FlatKey { instance, signal };
        match self.signal_map.get(&key).copied() {
            Some(index) => Some(index),
            None => {
                self.diagnostics.push(
                    Diagnostic::new(
                        "E_SIM_IR_SIGNAL",
                        format!("signal {:?} is not a packed scalar signal", signal.id),
                    )
                    .with_module(module.name.clone()),
                );
                None
            }
        }
    }

    fn memory_index(
        &mut self,
        instance: usize,
        signal: Signal,
        module: &CompiledModule,
    ) -> Option<usize> {
        let key = FlatKey { instance, signal };
        match self.memory_map.get(&key).copied() {
            Some(index) => Some(index),
            None => {
                self.diagnostics.push(
                    Diagnostic::new(
                        "E_SIM_IR_MEMORY",
                        format!("signal {:?} is not a packed memory", signal.id),
                    )
                    .with_module(module.name.clone()),
                );
                None
            }
        }
    }

    fn find_child_context(&self, parent: usize, instance_name: &str) -> Option<usize> {
        let prefix = format!("{}.{instance_name}", self.contexts[parent].path);
        self.contexts.iter().position(|ctx| ctx.path == prefix)
    }

    fn alloc_signal_layout(&mut self, ty: BitType) -> PackedValueLayout {
        let layout = PackedValueLayout {
            width: ty.width,
            ty,
            offset: self.next_signal_offset,
            limbs: limbs(ty.width),
        };
        self.next_signal_offset += layout.limbs;
        layout
    }

    fn alloc_memory_layout(&self, ty: BitType) -> PackedValueLayout {
        PackedValueLayout {
            width: ty.width,
            ty,
            offset: 0,
            limbs: limbs(ty.width),
        }
    }
}

fn schedule_assignments(assignments: &[FlatAssignment]) -> Vec<PackedPacket> {
    let dst_to_assignment = assignments
        .iter()
        .enumerate()
        .map(|(index, assignment)| (assignment.dst, index))
        .collect::<HashMap<_, _>>();
    let mut memo = HashMap::new();
    let mut levels: Vec<Vec<PackedOp>> = Vec::new();
    for (index, assignment) in assignments.iter().enumerate() {
        let mut visiting = HashSet::new();
        let level = assignment_level(
            index,
            assignments,
            &dst_to_assignment,
            &mut memo,
            &mut visiting,
        );
        if levels.len() <= level {
            levels.resize_with(level + 1, Vec::new);
        }
        levels[level].push(PackedOp::Assign {
            dst: assignment.dst,
            expr: assignment.expr.clone(),
        });
    }
    levels
        .into_iter()
        .filter(|ops| !ops.is_empty())
        .map(|ops| PackedPacket { ops })
        .collect()
}

fn assignment_level(
    index: usize,
    assignments: &[FlatAssignment],
    dst_to_assignment: &HashMap<usize, usize>,
    memo: &mut HashMap<usize, usize>,
    visiting: &mut HashSet<usize>,
) -> usize {
    if let Some(level) = memo.get(&index).copied() {
        return level;
    }
    if !visiting.insert(index) {
        return 0;
    }
    let level = expr_signal_deps(&assignments[index].expr)
        .into_iter()
        .filter_map(|signal| dst_to_assignment.get(&signal).copied())
        .filter(|dep| *dep != index)
        .map(|dep| assignment_level(dep, assignments, dst_to_assignment, memo, visiting) + 1)
        .max()
        .unwrap_or(0);
    visiting.remove(&index);
    memo.insert(index, level);
    level
}

fn schedule_async_resets(registers: &[CompiledRegRef]) -> Vec<PackedPacket> {
    let ops = registers
        .iter()
        .filter_map(|reg| {
            let reset = reg.reset.as_ref()?;
            if reset.kind != ResetKind::Async {
                return None;
            }
            Some(PackedOp::CaptureReg {
                dst: reg.signal,
                next: PackedExpr {
                    kind: PackedExprKind::Lit(reset.value.clone()),
                    ty: reg.next.ty,
                },
                reset: Some(reset.clone()),
            })
        })
        .collect::<Vec<_>>();
    if ops.is_empty() {
        Vec::new()
    } else {
        vec![PackedPacket { ops }]
    }
}

fn schedule_tick_next(registers: &[CompiledRegRef]) -> Vec<PackedPacket> {
    let ops = registers
        .iter()
        .map(|reg| PackedOp::CaptureReg {
            dst: reg.signal,
            next: reg.next.clone(),
            reset: reg.reset.clone(),
        })
        .collect::<Vec<_>>();
    if ops.is_empty() {
        Vec::new()
    } else {
        vec![PackedPacket { ops }]
    }
}

fn schedule_memory_writes(writes: &[PackedOp]) -> Vec<PackedPacket> {
    if writes.is_empty() {
        Vec::new()
    } else {
        vec![PackedPacket {
            ops: writes.to_vec(),
        }]
    }
}

fn expr_signal_deps(expr: &PackedExpr) -> HashSet<usize> {
    let mut deps = HashSet::new();
    collect_expr_signal_deps(expr, &mut deps);
    deps
}

fn collect_expr_signal_deps(expr: &PackedExpr, deps: &mut HashSet<usize>) {
    match &expr.kind {
        PackedExprKind::Signal(signal) => {
            deps.insert(*signal);
        }
        PackedExprKind::Lit(_) => {}
        PackedExprKind::Not(inner)
        | PackedExprKind::Zext(inner)
        | PackedExprKind::Sext(inner)
        | PackedExprKind::Trunc(inner)
        | PackedExprKind::Cast(inner)
        | PackedExprKind::Slice { expr: inner, .. } => collect_expr_signal_deps(inner, deps),
        PackedExprKind::And(lhs, rhs)
        | PackedExprKind::Or(lhs, rhs)
        | PackedExprKind::Xor(lhs, rhs)
        | PackedExprKind::Add(lhs, rhs)
        | PackedExprKind::Sub(lhs, rhs)
        | PackedExprKind::Mul(lhs, rhs)
        | PackedExprKind::Eq(lhs, rhs)
        | PackedExprKind::Ne(lhs, rhs) => {
            collect_expr_signal_deps(lhs, deps);
            collect_expr_signal_deps(rhs, deps);
        }
        PackedExprKind::Lt { lhs, rhs, .. } => {
            collect_expr_signal_deps(lhs, deps);
            collect_expr_signal_deps(rhs, deps);
        }
        PackedExprKind::Mux {
            cond,
            then_expr,
            else_expr,
        } => {
            collect_expr_signal_deps(cond, deps);
            collect_expr_signal_deps(then_expr, deps);
            collect_expr_signal_deps(else_expr, deps);
        }
        PackedExprKind::Concat(parts) => {
            for part in parts {
                collect_expr_signal_deps(part, deps);
            }
        }
        PackedExprKind::MemRead { addr, .. } => collect_expr_signal_deps(addr, deps),
    }
}

fn packed_signal_kind(kind: &SignalKind) -> PackedSignalKind {
    match kind {
        SignalKind::Input => PackedSignalKind::Input,
        SignalKind::Output => PackedSignalKind::Output,
        SignalKind::Reg { .. } => PackedSignalKind::Reg,
        SignalKind::Wire => PackedSignalKind::Wire,
        SignalKind::Inout | SignalKind::Mem { .. } => PackedSignalKind::Wire,
    }
}

fn expr_type(module: &CompiledModule, expr: &Expr) -> Option<BitType> {
    match expr {
        Expr::Lit { ty, .. } => Some(*ty),
        Expr::Signal(signal) => module.signal(*signal).map(|signal| signal.ty),
        Expr::Not(inner)
        | Expr::And(inner, _)
        | Expr::Or(inner, _)
        | Expr::Xor(inner, _)
        | Expr::Add(inner, _)
        | Expr::Sub(inner, _)
        | Expr::Mul(inner, _) => expr_type(module, inner),
        Expr::Eq(_, _) | Expr::Ne(_, _) | Expr::Lt(_, _) => Some(rrtl_ir::uint(1)),
        Expr::Mux {
            then_expr,
            else_expr: _,
            ..
        } => expr_type(module, then_expr),
        Expr::Slice { width, .. } => Some(rrtl_ir::uint(*width)),
        Expr::Zext { width, .. } => Some(rrtl_ir::uint(*width)),
        Expr::Sext { width, .. } => Some(rrtl_ir::sint(*width)),
        Expr::Trunc { expr, width } => {
            let inner = expr_type(module, expr)?;
            Some(BitType::new(*width, inner.signedness))
        }
        Expr::Cast { expr, signedness } => {
            let inner = expr_type(module, expr)?;
            Some(BitType::new(inner.width, *signedness))
        }
        Expr::Concat(parts) => {
            let mut width = 0;
            for part in parts {
                width += expr_type(module, part)?.width;
            }
            Some(rrtl_ir::uint(width))
        }
        Expr::MemRead { mem, .. } => module.signal(*mem).map(|signal| signal.ty),
    }
}

pub fn limbs(width: Width) -> usize {
    width.div_ceil(32).max(1) as usize
}

pub fn final_limb_mask(width: Width) -> u32 {
    let rem = width % 32;
    if rem == 0 {
        u32::MAX
    } else {
        (1u32 << rem) - 1
    }
}

pub fn encode_limbs(value: i128, ty: BitType) -> Vec<u32> {
    let mut encoded = value as u128;
    let count = limbs(ty.width);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(encoded as u32);
        encoded >>= 32;
    }
    if let Some(last) = out.last_mut() {
        *last &= final_limb_mask(ty.width);
    }
    out
}

fn encode_u128_limbs(value: u128, ty: BitType) -> Vec<u32> {
    let mut encoded = value;
    let count = limbs(ty.width);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(encoded as u32);
        encoded >>= 32;
    }
    mask_to_width(&mut out, ty.width);
    out
}

fn decode_u128_limbs(value: &[u32]) -> u128 {
    let mut out = 0u128;
    for (index, limb) in value.iter().take(4).copied().enumerate() {
        out |= (limb as u128) << (index * 32);
    }
    out
}

fn decode_usize(value: &[u32]) -> usize {
    let mut out = 0usize;
    for (index, limb) in value
        .iter()
        .take(usize::BITS.div_ceil(32) as usize)
        .copied()
        .enumerate()
    {
        out |= (limb as usize) << (index * 32);
    }
    if value
        .iter()
        .skip(usize::BITS.div_ceil(32) as usize)
        .any(|limb| *limb != 0)
    {
        usize::MAX
    } else {
        out
    }
}

fn decode_bool(value: &[u32]) -> bool {
    value.first().copied().unwrap_or(0) & 1 != 0
}

fn encode_bool(value: bool) -> Vec<u32> {
    vec![u32::from(value)]
}

fn fit_limbs(mut value: Vec<u32>, ty: BitType) -> Vec<u32> {
    value.resize(limbs(ty.width), 0);
    mask_to_width(&mut value, ty.width);
    value
}

fn mask_to_width(value: &mut Vec<u32>, width: Width) {
    value.truncate(limbs(width));
    value.resize(limbs(width), 0);
    if let Some(last) = value.last_mut() {
        *last &= final_limb_mask(width);
    }
}

fn bit_not(mut value: Vec<u32>, ty: BitType) -> Vec<u32> {
    value.resize(limbs(ty.width), 0);
    for limb in &mut value {
        *limb = !*limb;
    }
    fit_limbs(value, ty)
}

fn bit_binop(
    mut lhs: Vec<u32>,
    mut rhs: Vec<u32>,
    ty: BitType,
    op: impl Fn(u32, u32) -> u32,
) -> Vec<u32> {
    let count = limbs(ty.width);
    lhs.resize(count, 0);
    rhs.resize(count, 0);
    let mut out = (0..count)
        .map(|index| op(lhs[index], rhs[index]))
        .collect::<Vec<_>>();
    mask_to_width(&mut out, ty.width);
    out
}

fn add_limbs(mut lhs: Vec<u32>, mut rhs: Vec<u32>, ty: BitType) -> Vec<u32> {
    let count = limbs(ty.width);
    lhs.resize(count, 0);
    rhs.resize(count, 0);
    let mut out = Vec::with_capacity(count);
    let mut carry = 0u64;
    for index in 0..count {
        let sum = lhs[index] as u64 + rhs[index] as u64 + carry;
        out.push(sum as u32);
        carry = sum >> 32;
    }
    mask_to_width(&mut out, ty.width);
    out
}

fn sub_limbs(mut lhs: Vec<u32>, mut rhs: Vec<u32>, ty: BitType) -> Vec<u32> {
    let count = limbs(ty.width);
    lhs.resize(count, 0);
    rhs.resize(count, 0);
    let mut out = Vec::with_capacity(count);
    let mut borrow = 0u64;
    for index in 0..count {
        let subtrahend = rhs[index] as u64 + borrow;
        let lhs_word = lhs[index] as u64;
        out.push(lhs_word.wrapping_sub(subtrahend) as u32);
        borrow = u64::from(lhs_word < subtrahend);
    }
    mask_to_width(&mut out, ty.width);
    out
}

fn mul_limbs(mut lhs: Vec<u32>, mut rhs: Vec<u32>, ty: BitType) -> Vec<u32> {
    let count = limbs(ty.width);
    lhs.resize(count, 0);
    rhs.resize(count, 0);
    let mut accum = vec![0u128; count * 2];
    for lhs_index in 0..count {
        for rhs_index in 0..count {
            let out_index = lhs_index + rhs_index;
            if out_index < accum.len() {
                accum[out_index] += lhs[lhs_index] as u128 * rhs[rhs_index] as u128;
            }
        }
    }

    let mut out = Vec::with_capacity(count);
    let mut carry = 0u128;
    for index in 0..count {
        let word = accum[index] + carry;
        out.push(word as u32);
        carry = word >> 32;
    }
    mask_to_width(&mut out, ty.width);
    out
}

fn eq_values(mut lhs: Vec<u32>, mut rhs: Vec<u32>, width: Width) -> bool {
    let count = limbs(width);
    lhs.resize(count, 0);
    rhs.resize(count, 0);
    mask_to_width(&mut lhs, width);
    mask_to_width(&mut rhs, width);
    lhs == rhs
}

fn lt_values(mut lhs: Vec<u32>, mut rhs: Vec<u32>, width: Width, signed: bool) -> bool {
    let count = limbs(width);
    lhs.resize(count, 0);
    rhs.resize(count, 0);
    mask_to_width(&mut lhs, width);
    mask_to_width(&mut rhs, width);
    if signed {
        let lhs_sign = get_bit(&lhs, width - 1);
        let rhs_sign = get_bit(&rhs, width - 1);
        if lhs_sign != rhs_sign {
            return lhs_sign;
        }
    }
    for index in (0..count).rev() {
        if lhs[index] != rhs[index] {
            return lhs[index] < rhs[index];
        }
    }
    false
}

fn slice_limbs(value: Vec<u32>, lsb: Width, ty: BitType) -> Vec<u32> {
    let mut out = vec![0; limbs(ty.width)];
    for bit in 0..ty.width {
        if get_bit(&value, lsb + bit) {
            set_bit(&mut out, bit);
        }
    }
    mask_to_width(&mut out, ty.width);
    out
}

fn sext_limbs(value: Vec<u32>, input_width: Width, ty: BitType) -> Vec<u32> {
    let sign = get_bit(&value, input_width - 1);
    let mut out = fit_limbs(value, BitType::new(input_width, Signedness::Unsigned));
    out.resize(limbs(ty.width), if sign { u32::MAX } else { 0 });
    if sign {
        for bit in input_width..ty.width {
            set_bit(&mut out, bit);
        }
    }
    mask_to_width(&mut out, ty.width);
    out
}

fn concat_limbs(parts: Vec<(Vec<u32>, Width)>, ty: BitType) -> Vec<u32> {
    let mut out = vec![0; limbs(ty.width)];
    let mut offset = 0;
    for (value, width) in parts.into_iter().rev() {
        for bit in 0..width {
            if get_bit(&value, bit) {
                set_bit(&mut out, offset + bit);
            }
        }
        offset += width;
    }
    mask_to_width(&mut out, ty.width);
    out
}

fn get_bit(value: &[u32], bit: Width) -> bool {
    let limb = (bit / 32) as usize;
    let offset = bit % 32;
    value
        .get(limb)
        .is_some_and(|word| (word & (1u32 << offset)) != 0)
}

fn set_bit(value: &mut Vec<u32>, bit: Width) {
    let limb = (bit / 32) as usize;
    let offset = bit % 32;
    if value.len() <= limb {
        value.resize(limb + 1, 0);
    }
    value[limb] |= 1u32 << offset;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rrtl_core::{compile, concat, lit_s, lit_u, mux, sext, sint, uint, Design, Simulator};

    // The shared per-width packer must place every signal in a non-overlapping,
    // naturally-aligned slot, pack far below the uniform 16-byte layout, and honor
    // an affinity order (placing listed signals first).
    #[test]
    fn packed_signal_layout_is_dense_aligned_and_ordered() {
        let widths = [1u32, 32, 8, 64, 128, 16, 32];
        let (off, total) = packed_signal_layout(&widths, None);
        // each slot fits its store-bytes, is aligned, and none overlap
        let mut spans: Vec<(usize, usize)> =
            off.iter().zip(&widths).map(|(&o, &w)| (o, o + state_store_bytes(w))).collect();
        for (i, &(o, _)) in spans.iter().enumerate() {
            assert_eq!(o % state_store_bytes(widths[i]), 0, "signal {i} misaligned");
            assert!(spans[i].1 <= total);
        }
        spans.sort_unstable();
        for w in spans.windows(2) {
            assert!(w[0].1 <= w[1].0, "overlap {:?} {:?}", w[0], w[1]);
        }
        // far denser than uniform 16-byte slots
        assert!(total < widths.len() * 16);
        // affinity order places the listed signals first (signal 6 then 2)
        let (aoff, _) = packed_signal_layout(&widths, Some(&[6, 2]));
        assert!(aoff[6] < aoff[2], "affinity order not honored");
    }

    // The batch SIMD engine's clock-gated tick must match the gold oracle on an
    // independent multi-clock design (two counters on distinct clocks driven at
    // different rates) — the engine-side of multi-clock support.
    #[test]
    fn simd_cpu_tick_clocked_matches_oracle() {
        let mut design = Design::new();
        let (clka, clkb, a, b);
        {
            let mut m = design.module("TwoClock");
            clka = m.input("clka", uint(1));
            clkb = m.input("clkb", uint(1));
            let ca = m.reg("countA", uint(8));
            let cb = m.reg("countB", uint(8));
            m.clock(ca, clka);
            m.clock(cb, clkb);
            m.next(ca, ca + lit_u(1, 8));
            m.next(cb, cb + lit_u(1, 8));
            a = m.output("a", uint(8));
            b = m.output("b", uint(8));
            m.assign(a, ca);
            m.assign(b, cb);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "TwoClock").unwrap();
        // The clock association survived lowering into the side-table.
        assert_eq!(program.reg_clocks.len(), 2);

        let mut sim = SimdCpuSimulator::new(program, 1).unwrap();
        let mut gold = Simulator::new(&design, "TwoClock").unwrap();
        for step in 0..12 {
            // clkA every step; clkB every 3rd step — an irregular multi-rate schedule.
            let active: Vec<_> = if step % 3 == 0 { vec![clka, clkb] } else { vec![clka] };
            sim.tick_clocked(&active).unwrap();
            gold.tick_clocked(&active).unwrap();
            assert_eq!(sim.get_signal(a).unwrap()[0], gold.get(a), "a@{step}");
            assert_eq!(sim.get_signal(b).unwrap()[0], gold.get(b), "b@{step}");
        }
        // The slow domain really lagged the fast one.
        assert!(sim.get_signal(b).unwrap()[0] < sim.get_signal(a).unwrap()[0]);
    }

    // A memory written on a slow clock must capture only on that clock's edges —
    // the engine's clock-gated memory write vs the gold oracle.
    #[test]
    fn simd_cpu_tick_clocked_gates_memory_writes() {
        let mut design = Design::new();
        let (clka, clkb, out);
        {
            let mut m = design.module("MemClk");
            clka = m.input("clka", uint(1));
            clkb = m.input("clkb", uint(1));
            let cnt = m.reg("cnt", uint(8));
            m.clock(cnt, clka);
            m.next(cnt, cnt + lit_u(1, 8));
            // mem[0] <= cnt, clocked by the SLOW clock clkb (enable always on).
            let mem = m.mem("mem", 2, uint(8), 4);
            m.mem_write(mem, clkb, lit_u(1, 1), lit_u(0, 2), cnt);
            let rd = m.mem_read(mem, lit_u(0, 2));
            out = m.output("out", uint(8));
            m.assign(out, rd);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "MemClk").unwrap();
        assert_eq!(program.mem_clocks.len(), 1, "memory's write clock survived lowering");

        let mut sim = SimdCpuSimulator::new(program, 1).unwrap();
        let mut gold = Simulator::new(&design, "MemClk").unwrap();
        let mut updates = 0u128;
        let mut prev = 0u128;
        for step in 0..12 {
            let active: Vec<_> = if step % 3 == 0 { vec![clka, clkb] } else { vec![clka] };
            sim.tick_clocked(&active).unwrap();
            gold.tick_clocked(&active).unwrap();
            let o = sim.get_signal(out).unwrap()[0];
            assert_eq!(o, gold.get(out), "out@{step}");
            if o != prev {
                updates += 1;
                prev = o;
            }
        }
        // The memory output changed only on clkB edges, not every (clkA) step.
        assert!(updates <= 5, "memory write was not clock-gated (updates={updates})");
    }

    // Exercises the one-limb binop kernels against an independent scalar
    // reference. On x86_64 this drives the AVX2 vector body plus its scalar
    // tail (lengths chosen to be non-multiples of 8 and 4); on aarch64 it
    // drives the NEON path; elsewhere the portable scalar path. The dispatch
    // and final masking are validated on every architecture.
    #[test]
    fn simd_words_binops_match_scalar_reference() {
        fn reference(op: &str, lhs: u32, rhs: u32, mask: u32) -> u32 {
            let raw = match op {
                "and" => lhs & rhs,
                "or" => lhs | rhs,
                "xor" => lhs ^ rhs,
                "add" => lhs.wrapping_add(rhs),
                "sub" => lhs.wrapping_sub(rhs),
                "mul" => lhs.wrapping_mul(rhs),
                _ => unreachable!(),
            };
            raw & mask
        }

        // Lengths spanning below, at, and across the 8-wide (AVX2) and 4-wide
        // (NEON) chunk boundaries so the scalar tail handling is covered.
        let lengths = [0usize, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 33];
        let masks = [u32::MAX, 0xFFFF, 0xFF, 0x1, 0x8000_0000, 0];
        let kernels: &[(&str, fn(&[u32], &[u32], u32, &mut [u32]))] = &[
            ("and", simd_words::and_into),
            ("or", simd_words::or_into),
            ("xor", simd_words::xor_into),
            ("add", simd_words::add_into),
            ("sub", simd_words::sub_into),
            ("mul", simd_words::mul_into),
        ];

        for &len in &lengths {
            // Deterministic but varied operands; include high-bit patterns so
            // add/sub carry and mul overflow wrap-around are exercised.
            let lhs: Vec<u32> = (0..len)
                .map(|i| (i as u32).wrapping_mul(2_654_435_761) ^ 0xDEAD_0000)
                .collect();
            let rhs: Vec<u32> = (0..len)
                .map(|i| (i as u32).wrapping_mul(40_503).wrapping_add(0x9E37_79B9))
                .collect();
            for &mask in &masks {
                for &(op, kernel) in kernels {
                    let mut out = vec![0u32; len];
                    kernel(&lhs, &rhs, mask, &mut out);
                    for i in 0..len {
                        assert_eq!(
                            out[i],
                            reference(op, lhs[i], rhs[i], mask),
                            "op={op} len={len} mask={mask:#010x} idx={i}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn lowers_wide_signed_concat_and_packets_assignments_by_dependency() {
        let mut design = Design::new();
        let (a, b, y, z);
        {
            let mut m = design.module("Top");
            a = m.input("a", uint(40));
            b = m.input("b", uint(40));
            y = m.wire("y", uint(40));
            z = m.output("z", uint(80));
            m.assign(y, a + b);
            m.assign(z, concat([y.value(), lit_u(0xff, 40)]));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();
        assert_eq!(
            program.signals[program.signal_index(a).unwrap()]
                .layout
                .limbs,
            2
        );
        assert_eq!(
            program.signals[program.signal_index(z).unwrap()]
                .layout
                .limbs,
            3
        );
        assert_eq!(program.streams.comb.len(), 2);
    }

    #[test]
    fn cone_of_influence_prunes_unobserved_logic() {
        // Two independent accumulators; observing one must exclude the other.
        let mut design = Design::new();
        let (clk, a, b, ra, rb);
        {
            let mut m = design.module("Two");
            clk = m.input("clk", uint(1));
            a = m.input("a", uint(32));
            b = m.input("b", uint(32));
            ra = m.reg("ra", uint(32));
            rb = m.reg("rb", uint(32));
            m.clock(ra, clk);
            m.clock(rb, clk);
            m.next(ra, ra.value() + a.value());
            m.next(rb, rb.value() + b.value());
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Two").unwrap();
        let si = |s| program.signal_index(s).unwrap();
        let (ra_i, rb_i, a_i, b_i) = (si(ra), si(rb), si(a), si(b));

        let (sig, _) = cone_of_influence(&program, &[ra_i], &[]);
        assert!(sig[ra_i] && sig[a_i], "observed register and its input are in the cone");
        assert!(!sig[rb_i] && !sig[b_i], "the independent register and its input are pruned");

        // and the slice keeps strictly fewer signals than the full program.
        let slice = slice_present(&program, &sig, &vec![false; program.memories.len()]).unwrap();
        assert!(slice.program.signals.len() < program.signals.len());
    }

    #[test]
    fn flattens_instances_and_keeps_top_signal_mapping() {
        let mut design = Design::new();
        {
            let mut m = design.module("Child");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.assign(y, a + lit_u(1, 8));
        }
        let (a, y);
        {
            let mut m = design.module("Top");
            a = m.input("a", uint(8));
            y = m.output("y", uint(8));
            m.instance("u0", "Child", [("a", a), ("y", y)]);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();
        assert!(program.signal_index(a).is_some());
        assert!(program.signal_index(y).is_some());
        assert!(program
            .signals
            .iter()
            .any(|signal| signal.name == "Top.u0.y"));
    }

    #[test]
    fn preserves_register_reset_and_memory_write_streams() {
        let mut design = Design::new();
        {
            let mut m = design.module("Top");
            let clk = m.input("clk", uint(1));
            let rst_n = m.input("rst_n", uint(1));
            let we = m.input("we", uint(1));
            let addr = m.input("addr", uint(2));
            let data = m.input("data", sint(8));
            let q = m.reg("q", sint(8));
            let mem = m.mem("mem", 2, sint(8), 4);
            m.clock(q, clk);
            m.reset_low(q, rst_n, -1);
            m.next(q, mux(we, data, lit_s(0, 8)));
            m.mem_write(mem, clk, we, addr, q);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();
        assert_eq!(program.memories.len(), 1);
        assert_eq!(program.streams.tick_next[0].ops.len(), 1);
        assert_eq!(program.streams.tick_commit[0].ops.len(), 1);
    }

    #[test]
    fn machine_lowering_reuses_repeated_expression_dependencies() {
        let mut design = Design::new();
        let (a, b, y);
        {
            let mut m = design.module("Reuse");
            a = m.input("a", uint(8));
            b = m.input("b", uint(8));
            y = m.output("y", uint(8));
            let sum = a + b;
            m.assign(y, sum.clone() + sum);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Reuse").unwrap();
        let machine = lower_to_machine_program(&program);
        let add_count = machine
            .streams
            .comb
            .packets
            .iter()
            .flat_map(|packet| &packet.instrs)
            .filter(|instr| matches!(instr.kind, PackedInstrKind::Add(_, _)))
            .count();
        assert_eq!(add_count, 2);
        assert!(machine.streams.comb.packets.len() >= 3);
        assert!(machine
            .streams
            .comb
            .packets
            .last()
            .unwrap()
            .effects
            .iter()
            .any(|effect| matches!(effect, PackedEffect::StoreSignal { dst, .. } if *dst == program.signal_index(y).unwrap())));
    }

    #[test]
    fn machine_lowering_preserves_tick_capture_and_memory_effects() {
        let mut design = Design::new();
        {
            let mut m = design.module("Effects");
            let clk = m.input("clk", uint(1));
            let we = m.input("we", uint(1));
            let addr = m.input("addr", uint(2));
            let data = m.input("data", uint(8));
            let q = m.reg("q", uint(8));
            let mem = m.mem("mem", 2, uint(8), 4);
            m.clock(q, clk);
            m.next(q, data);
            m.mem_write(mem, clk, we, addr, q);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Effects").unwrap();
        let machine = lower_to_machine_program(&program);
        assert!(machine
            .streams
            .tick_next
            .packets
            .iter()
            .flat_map(|packet| &packet.effects)
            .any(|effect| matches!(effect, PackedEffect::CaptureReg { .. })));
        assert!(machine
            .streams
            .tick_commit
            .packets
            .iter()
            .flat_map(|packet| &packet.effects)
            .any(|effect| matches!(effect, PackedEffect::MemoryWrite { .. })));
    }

    #[test]
    fn machine_analysis_reports_vliw_stats_and_value_liveness() {
        let mut design = Design::new();
        {
            let mut m = design.module("Reuse");
            let a = m.input("a", uint(8));
            let b = m.input("b", uint(8));
            let y = m.output("y", uint(8));
            let sum = a + b;
            m.assign(y, sum.clone() + sum);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Reuse").unwrap();
        let machine = lower_to_machine_program(&program);
        let analysis = analyze_machine_program(&machine).unwrap();

        assert_eq!(analysis.comb.instr_count, 4);
        assert_eq!(analysis.comb.effect_count, 1);
        assert_eq!(analysis.comb.max_packet_width, 2);
        assert_eq!(analysis.comb.max_live_values, 2);
        assert_eq!(analysis.comb.avg_live_values_x100, 100);
        assert!(analysis.avg_live_values_x100 > 0);
        assert_eq!(analysis.comb.max_packet_memory_reads, 0);
        assert_eq!(analysis.comb.packets[0].class_counts.signal_load, 2);
        assert_eq!(analysis.comb.packets[1].class_counts.arithmetic, 1);
        assert_eq!(analysis.comb.packets[2].class_counts.arithmetic, 1);
        assert_eq!(
            analysis
                .comb
                .values
                .get(&PackedValueId(2))
                .unwrap()
                .last_use_packet,
            Some(2)
        );
    }

    #[test]
    fn machine_analysis_rejects_duplicate_value_definitions() {
        let mut design = Design::new();
        {
            let mut m = design.module("Dup");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.assign(y, a);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Dup").unwrap();
        let machine = lower_to_machine_program(&program);
        let mut block = machine.streams.comb.clone();
        let duplicate = block.packets[0].instrs[0].clone();
        block.packets[0].instrs.push(duplicate);

        let err = analyze_machine_block(&block, &program).unwrap_err();
        assert!(diagnostic_codes(&err).contains(&"E_SIM_IR_MACHINE_DUP_VALUE"));
    }

    #[test]
    fn machine_analysis_rejects_instruction_uses_before_definitions() {
        let mut design = Design::new();
        {
            let mut m = design.module("UseBeforeDef");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.assign(y, a);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "UseBeforeDef").unwrap();
        let block = PackedBlock {
            packets: vec![PackedMachinePacket {
                instrs: vec![PackedInstr {
                    dst: PackedValueId(0),
                    ty: uint(8),
                    kind: PackedInstrKind::Add(PackedValueId(1), PackedValueId(2)),
                }],
                effects: Vec::new(),
            }],
        };

        let err = analyze_machine_block(&block, &program).unwrap_err();
        assert!(diagnostic_codes(&err).contains(&"E_SIM_IR_MACHINE_USE_BEFORE_DEF"));
    }

    #[test]
    fn machine_analysis_rejects_invalid_signal_and_memory_indices() {
        let mut design = Design::new();
        {
            let mut m = design.module("Refs");
            let addr = m.input("addr", uint(2));
            let y = m.output("y", uint(8));
            let mem = m.mem("mem", 2, uint(8), 4);
            let read = m.mem_read(mem, addr);
            m.assign(y, read);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Refs").unwrap();
        let block = PackedBlock {
            packets: vec![
                PackedMachinePacket {
                    instrs: vec![PackedInstr {
                        dst: PackedValueId(0),
                        ty: uint(2),
                        kind: PackedInstrKind::Signal(program.signals.len()),
                    }],
                    effects: Vec::new(),
                },
                PackedMachinePacket {
                    instrs: vec![PackedInstr {
                        dst: PackedValueId(1),
                        ty: uint(8),
                        kind: PackedInstrKind::MemRead {
                            memory: program.memories.len(),
                            addr: PackedValueId(0),
                        },
                    }],
                    effects: Vec::new(),
                },
            ],
        };

        let err = analyze_machine_block(&block, &program).unwrap_err();
        let codes = diagnostic_codes(&err);
        assert!(codes.contains(&"E_SIM_IR_MACHINE_SIGNAL"));
        assert!(codes.contains(&"E_SIM_IR_MACHINE_MEMORY"));
    }

    #[test]
    fn machine_analysis_rejects_memory_write_effect_undefined_values() {
        let mut design = Design::new();
        {
            let mut m = design.module("Write");
            let clk = m.input("clk", uint(1));
            let we = m.input("we", uint(1));
            let addr = m.input("addr", uint(2));
            let data = m.input("data", uint(8));
            let mem = m.mem("mem", 2, uint(8), 4);
            m.mem_write(mem, clk, we, addr, data);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Write").unwrap();
        let block = PackedBlock {
            packets: vec![PackedMachinePacket {
                instrs: Vec::new(),
                effects: vec![PackedEffect::MemoryWrite {
                    memory: 0,
                    enable: PackedValueId(0),
                    addr: PackedValueId(1),
                    data: PackedValueId(2),
                }],
            }],
        };

        let err = analyze_machine_block(&block, &program).unwrap_err();
        assert_eq!(
            diagnostic_codes(&err)
                .iter()
                .filter(|code| **code == "E_SIM_IR_MACHINE_USE_BEFORE_DEF")
                .count(),
            3
        );
    }

    #[test]
    fn machine_optimizer_caps_packet_width_without_changing_counts() {
        let mut design = Design::new();
        {
            let mut m = design.module("Wide");
            let a0 = m.input("a0", uint(4));
            let a1 = m.input("a1", uint(4));
            let a2 = m.input("a2", uint(4));
            let a3 = m.input("a3", uint(4));
            let a4 = m.input("a4", uint(4));
            let y = m.output("y", uint(20));
            m.assign(
                y,
                concat([a0.value(), a1.value(), a2.value(), a3.value(), a4.value()]),
            );
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Wide").unwrap();
        let machine = lower_to_machine_program(&program);
        let before = analyze_machine_program(&machine).unwrap();
        assert!(before.max_packet_width > 2);

        let optimized = optimize_machine_program(
            &machine,
            PackedScheduleOptions {
                max_packet_width: Some(2),
                max_memory_reads_per_packet: None,
                liveness_priority: false,
            },
        )
        .unwrap();
        let after = analyze_machine_program(&optimized).unwrap();

        assert_eq!(after.instr_count, before.instr_count);
        assert_eq!(after.effect_count, before.effect_count);
        assert!(after.max_packet_width <= 2);
        assert!(after.comb.packets.len() > before.comb.packets.len());
    }

    #[test]
    fn machine_optimizer_repacks_ready_instrs_across_effect_region() {
        let mut design = Design::new();
        let (a, b, y);
        {
            let mut m = design.module("Dense");
            a = m.input("a", uint(8));
            b = m.input("b", uint(8));
            y = m.output("y", uint(8));
            m.assign(y, a + b);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Dense").unwrap();
        let a_index = program.signal_index(a).unwrap();
        let b_index = program.signal_index(b).unwrap();
        let y_index = program.signal_index(y).unwrap();
        let block = PackedBlock {
            packets: vec![
                PackedMachinePacket {
                    instrs: vec![PackedInstr {
                        dst: PackedValueId(0),
                        ty: uint(8),
                        kind: PackedInstrKind::Signal(a_index),
                    }],
                    effects: Vec::new(),
                },
                PackedMachinePacket {
                    instrs: vec![PackedInstr {
                        dst: PackedValueId(1),
                        ty: uint(8),
                        kind: PackedInstrKind::Signal(b_index),
                    }],
                    effects: Vec::new(),
                },
                PackedMachinePacket {
                    instrs: vec![PackedInstr {
                        dst: PackedValueId(2),
                        ty: uint(8),
                        kind: PackedInstrKind::Add(PackedValueId(0), PackedValueId(1)),
                    }],
                    effects: vec![PackedEffect::StoreSignal {
                        dst: y_index,
                        value: PackedValueId(2),
                    }],
                },
            ],
        };

        let optimized = optimize_machine_block(
            &block,
            &program,
            PackedScheduleOptions {
                max_packet_width: Some(2),
                max_memory_reads_per_packet: None,
                liveness_priority: false,
            },
        )
        .unwrap();

        assert_eq!(block.packets.len(), 3);
        assert_eq!(optimized.packets.len(), 2);
        assert_eq!(optimized.packets[0].instrs.len(), 2);
        assert!(optimized.packets[0].effects.is_empty());
        assert!(matches!(
            optimized.packets[1].instrs[0].kind,
            PackedInstrKind::Add(PackedValueId(0), PackedValueId(1))
        ));
        assert!(matches!(
            optimized.packets[1].effects[0],
            PackedEffect::StoreSignal { dst, value }
                if dst == y_index && value == PackedValueId(2)
        ));
        assert!(analyze_machine_block(&optimized, &program).is_ok());
    }

    #[test]
    fn machine_optimizer_keeps_effects_on_final_split_packet() {
        let mut design = Design::new();
        {
            let mut m = design.module("Write");
            let clk = m.input("clk", uint(1));
            let we = m.input("we", uint(1));
            let addr = m.input("addr", uint(2));
            let data = m.input("data", uint(8));
            let mem = m.mem("mem", 2, uint(8), 4);
            m.mem_write(mem, clk, we, addr, data);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Write").unwrap();
        let machine = lower_to_machine_program(&program);
        let optimized = optimize_machine_program(
            &machine,
            PackedScheduleOptions {
                max_packet_width: Some(2),
                max_memory_reads_per_packet: None,
                liveness_priority: false,
            },
        )
        .unwrap();

        let packets = &optimized.streams.tick_commit.packets;
        assert!(packets.len() >= 2);
        assert!(packets[..packets.len() - 1]
            .iter()
            .all(|packet| packet.effects.is_empty()));
        assert!(packets
            .last()
            .unwrap()
            .effects
            .iter()
            .any(|effect| matches!(effect, PackedEffect::MemoryWrite { .. })));
        assert!(analyze_machine_program(&optimized).is_ok());
    }

    #[test]
    fn machine_optimizer_rejects_zero_packet_width() {
        let mut design = Design::new();
        {
            let mut m = design.module("Zero");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.assign(y, a);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Zero").unwrap();
        let machine = lower_to_machine_program(&program);

        let err = optimize_machine_program(
            &machine,
            PackedScheduleOptions {
                max_packet_width: Some(0),
                max_memory_reads_per_packet: None,
                liveness_priority: false,
            },
        )
        .unwrap_err();
        assert!(diagnostic_codes(&err).contains(&"E_SIM_IR_SCHEDULE_WIDTH"));
    }

    #[test]
    fn machine_optimizer_rejects_zero_memory_read_cap() {
        let mut design = Design::new();
        {
            let mut m = design.module("ZeroMemCap");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.assign(y, a);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "ZeroMemCap").unwrap();
        let machine = lower_to_machine_program(&program);

        let err = optimize_machine_program(
            &machine,
            PackedScheduleOptions {
                max_packet_width: None,
                max_memory_reads_per_packet: Some(0),
                liveness_priority: false,
            },
        )
        .unwrap_err();
        assert!(diagnostic_codes(&err).contains(&"E_SIM_IR_SCHEDULE_MEMORY_READS"));
    }

    #[test]
    fn machine_optimizer_caps_memory_reads_and_fills_with_non_memory_ready_instrs() {
        let mut design = Design::new();
        let (addr, data, y);
        {
            let mut m = design.module("MemCap");
            addr = m.input("addr", uint(2));
            data = m.input("data", uint(8));
            y = m.output("y", uint(8));
            let mem0 = m.mem("mem0", 2, uint(8), 4);
            let mem1 = m.mem("mem1", 2, uint(8), 4);
            let read0 = m.mem_read(mem0, addr);
            let read1 = m.mem_read(mem1, addr);
            m.assign(y, read0 ^ read1 ^ data);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "MemCap").unwrap();
        let addr_index = program.signal_index(addr).unwrap();
        let data_index = program.signal_index(data).unwrap();
        let y_index = program.signal_index(y).unwrap();
        let block = PackedBlock {
            packets: vec![
                PackedMachinePacket {
                    instrs: vec![
                        PackedInstr {
                            dst: PackedValueId(0),
                            ty: uint(2),
                            kind: PackedInstrKind::Signal(addr_index),
                        },
                        PackedInstr {
                            dst: PackedValueId(1),
                            ty: uint(8),
                            kind: PackedInstrKind::Signal(data_index),
                        },
                    ],
                    effects: Vec::new(),
                },
                PackedMachinePacket {
                    instrs: vec![
                        PackedInstr {
                            dst: PackedValueId(2),
                            ty: uint(8),
                            kind: PackedInstrKind::MemRead {
                                memory: 0,
                                addr: PackedValueId(0),
                            },
                        },
                        PackedInstr {
                            dst: PackedValueId(3),
                            ty: uint(8),
                            kind: PackedInstrKind::MemRead {
                                memory: 1,
                                addr: PackedValueId(0),
                            },
                        },
                        PackedInstr {
                            dst: PackedValueId(4),
                            ty: uint(8),
                            kind: PackedInstrKind::Xor(PackedValueId(1), PackedValueId(1)),
                        },
                    ],
                    effects: Vec::new(),
                },
                PackedMachinePacket {
                    instrs: vec![PackedInstr {
                        dst: PackedValueId(5),
                        ty: uint(8),
                        kind: PackedInstrKind::Xor(PackedValueId(2), PackedValueId(3)),
                    }],
                    effects: vec![PackedEffect::StoreSignal {
                        dst: y_index,
                        value: PackedValueId(5),
                    }],
                },
            ],
        };

        let optimized = optimize_machine_block(
            &block,
            &program,
            PackedScheduleOptions {
                max_packet_width: Some(2),
                max_memory_reads_per_packet: Some(1),
                liveness_priority: false,
            },
        )
        .unwrap();
        let analysis = analyze_machine_block(&optimized, &program).unwrap();

        assert!(analysis.max_packet_width <= 2);
        assert_eq!(analysis.max_packet_memory_reads, 1);
        assert!(optimized.packets.iter().any(|packet| packet
            .instrs
            .iter()
            .any(|instr| matches!(instr.kind, PackedInstrKind::MemRead { memory: 0, .. }))
            && packet
                .instrs
                .iter()
                .any(|instr| matches!(instr.kind, PackedInstrKind::Xor(_, _)))));
        assert!(optimized.packets.last().unwrap().effects.iter().any(
            |effect| matches!(effect, PackedEffect::StoreSignal { dst, .. } if *dst == y_index)
        ));
    }

    #[test]
    fn machine_optimizer_uses_fifo_order_without_liveness_priority() {
        let (program, block, expected_first_kind) = liveness_priority_fixture();

        let optimized = optimize_machine_block(
            &block,
            &program,
            PackedScheduleOptions {
                max_packet_width: Some(3),
                max_memory_reads_per_packet: None,
                liveness_priority: false,
            },
        )
        .unwrap();

        assert!(matches!(
            optimized.packets[1].instrs[0].kind,
            PackedInstrKind::Not(PackedValueId(2))
        ));
        assert_eq!(optimized.packets[1].instrs[0].kind, expected_first_kind);
    }

    #[test]
    fn machine_optimizer_liveness_priority_closes_more_operands_first() {
        let (program, block, _) = liveness_priority_fixture();

        let optimized = optimize_machine_block(
            &block,
            &program,
            PackedScheduleOptions {
                max_packet_width: Some(3),
                max_memory_reads_per_packet: None,
                liveness_priority: true,
            },
        )
        .unwrap();

        assert!(matches!(
            optimized.packets[1].instrs[0].kind,
            PackedInstrKind::Add(PackedValueId(0), PackedValueId(1))
        ));
        assert!(matches!(
            optimized.packets[1].instrs[1].kind,
            PackedInstrKind::Not(PackedValueId(2))
        ));
    }

    #[test]
    fn optimized_machine_matches_unoptimized_packed_simulation() {
        let mut design = Design::new();
        let (a0, a1, a2, a3, a4, y);
        {
            let mut m = design.module("Wide");
            a0 = m.input("a0", uint(4));
            a1 = m.input("a1", uint(4));
            a2 = m.input("a2", uint(4));
            a3 = m.input("a3", uint(4));
            a4 = m.input("a4", uint(4));
            y = m.output("y", uint(20));
            m.assign(
                y,
                concat([a0.value(), a1.value(), a2.value(), a3.value(), a4.value()]),
            );
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Wide").unwrap();
        let machine = lower_to_machine_program(&program);
        let optimized = optimize_machine_program(
            &machine,
            PackedScheduleOptions {
                max_packet_width: Some(2),
                max_memory_reads_per_packet: None,
                liveness_priority: false,
            },
        )
        .unwrap();

        let mut reference = PackedSimulator::new(program.clone(), 3).unwrap();
        let mut optimized_sim = packed_sim_with_machine(program, optimized, 3);
        for (signal, values) in [
            (a0, vec![1, 2, 3]),
            (a1, vec![4, 5, 6]),
            (a2, vec![7, 8, 9]),
            (a3, vec![10, 11, 12]),
            (a4, vec![13, 14, 15]),
        ] {
            reference.set_signal(signal, &values).unwrap();
            optimized_sim.set_signal(signal, &values).unwrap();
        }

        assert_eq!(
            optimized_sim.get_signal(y).unwrap(),
            reference.get_signal(y).unwrap()
        );
    }

    #[test]
    fn memory_read_capped_machine_matches_unoptimized_packed_simulation() {
        let mut design = Design::new();
        let (we, addr, data, read);
        {
            let mut m = design.module("MemHeavy");
            let clk = m.input("clk", uint(1));
            we = m.input("we", uint(1));
            addr = m.input("addr", uint(2));
            data = m.input("data", uint(8));
            let mem0 = m.mem("mem0", 2, uint(8), 4);
            let mem1 = m.mem("mem1", 2, uint(8), 4);
            let read0 = m.mem_read(mem0, addr);
            let read1 = m.mem_read(mem1, addr);
            read = m.output("read", uint(8));
            m.assign(read, read0 ^ read1);
            m.mem_write(mem0, clk, we, addr, data);
            m.mem_write(mem1, clk, we, addr, data + lit_u(1, 8));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "MemHeavy").unwrap();
        let machine = lower_to_machine_program(&program);
        let optimized = optimize_machine_program(
            &machine,
            PackedScheduleOptions {
                max_packet_width: Some(4),
                max_memory_reads_per_packet: Some(1),
                liveness_priority: false,
            },
        )
        .unwrap();

        let mut reference = PackedSimulator::new(program.clone(), 3).unwrap();
        let mut optimized_sim = packed_sim_with_machine(program, optimized, 3);
        for sim in [&mut reference, &mut optimized_sim] {
            sim.set_signal(we, &[1, 1, 1]).unwrap();
            sim.set_signal(addr, &[0, 1, 2]).unwrap();
            sim.set_signal(data, &[9, 11, 13]).unwrap();
            sim.tick();
            sim.set_signal(we, &[0, 0, 0]).unwrap();
            sim.eval_combinational();
        }

        assert_eq!(
            optimized_sim.get_signal(read).unwrap(),
            reference.get_signal(read).unwrap()
        );
    }

    #[test]
    fn liveness_priority_machine_matches_unoptimized_packed_simulation() {
        let mut design = Design::new();
        let (a, b, c, y);
        {
            let mut m = design.module("FanoutHeavy");
            a = m.input("a", uint(8));
            b = m.input("b", uint(8));
            c = m.input("c", uint(8));
            y = m.output("y", uint(8));
            let wide = (a + b) ^ (a + c) ^ (b + c);
            m.assign(y, wide + (a ^ b ^ c));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "FanoutHeavy").unwrap();
        let machine = lower_to_machine_program(&program);
        let optimized = optimize_machine_program(
            &machine,
            PackedScheduleOptions {
                max_packet_width: Some(3),
                max_memory_reads_per_packet: None,
                liveness_priority: true,
            },
        )
        .unwrap();

        let mut reference = PackedSimulator::new(program.clone(), 4).unwrap();
        let mut optimized_sim = packed_sim_with_machine(program, optimized, 4);
        for (signal, values) in [
            (a, vec![1, 2, 3, 4]),
            (b, vec![5, 6, 7, 8]),
            (c, vec![9, 10, 11, 12]),
        ] {
            reference.set_signal(signal, &values).unwrap();
            optimized_sim.set_signal(signal, &values).unwrap();
        }

        assert_eq!(
            optimized_sim.get_signal(y).unwrap(),
            reference.get_signal(y).unwrap()
        );
    }

    #[test]
    fn packed_sim_matches_cpu_for_scalar_expressions() {
        let mut design = Design::new();
        let (a, b, ua, ub, sum, diff, lt_s, lt_u, bits, mixed);
        {
            let mut m = design.module("Exprs");
            a = m.input("a", sint(8));
            b = m.input("b", sint(8));
            ua = m.input("ua", uint(8));
            ub = m.input("ub", uint(8));
            sum = m.output("sum", sint(8));
            diff = m.output("diff", uint(8));
            lt_s = m.output("lt_s", uint(1));
            lt_u = m.output("lt_u", uint(1));
            bits = m.output("bits", uint(16));
            mixed = m.output("mixed", sint(16));
            m.assign(sum, a + b);
            m.assign(diff, ua - ub);
            m.assign(lt_s, a.value().lt_expr(b));
            m.assign(lt_u, ua.value().lt_expr(ub));
            m.assign(
                bits,
                concat([ua.value().slice(0, 4), ub.value(), lit_u(0xf, 4)]),
            );
            m.assign(mixed, sext(a, 16));
        }

        assert_packed_matches_cpu(
            &design,
            "Exprs",
            &[
                (a, vec![0xfe, 3, 0x80]),
                (b, vec![1, 0xff, 0x7f]),
                (ua, vec![1, 255, 16]),
                (ub, vec![2, 1, 16]),
            ],
            &[sum, diff, lt_s, lt_u, bits, mixed],
        );
    }

    #[test]
    fn packed_sim_matches_cpu_for_register_resets_and_memory() {
        let mut design = Design::new();
        let (rst, rst_n, we, addr, data, q_sync, q_async, read);
        {
            let mut m = design.module("Seq");
            let clk = m.input("clk", uint(1));
            rst = m.input("rst", uint(1));
            rst_n = m.input("rst_n", uint(1));
            we = m.input("we", uint(1));
            addr = m.input("addr", uint(2));
            data = m.input("data", uint(8));
            q_sync = m.reg("q_sync", uint(8));
            q_async = m.reg("q_async", uint(8));
            let mem = m.mem("mem", 2, uint(8), 2);
            read = m.output("read", uint(8));
            m.clock(q_sync, clk);
            m.clock(q_async, clk);
            m.reset(q_sync, rst, 3);
            m.async_reset_low(q_async, rst_n, 7);
            m.next(q_sync, mux(we, data, q_sync));
            m.next(q_async, q_sync + lit_u(1, 8));
            let read_expr = m.mem_read(mem, addr);
            m.assign(read, read_expr);
            m.mem_write(mem, clk, we, addr, data);
        }

        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Seq").unwrap();
        let mut packed = PackedSimulator::new(program, 3).unwrap();
        let mut cpus = (0..3)
            .map(|_| Simulator::new(&design, "Seq").unwrap())
            .collect::<Vec<_>>();

        set_all(&mut packed, &mut cpus, rst, &[1, 0, 0]);
        set_all(&mut packed, &mut cpus, rst_n, &[1, 0, 1]);
        set_all(&mut packed, &mut cpus, we, &[1, 1, 1]);
        set_all(&mut packed, &mut cpus, addr, &[0, 1, 3]);
        set_all(&mut packed, &mut cpus, data, &[9, 11, 13]);
        packed.tick();
        for cpu in &mut cpus {
            cpu.tick();
        }

        set_all(&mut packed, &mut cpus, rst, &[0, 0, 0]);
        set_all(&mut packed, &mut cpus, rst_n, &[1, 1, 1]);
        set_all(&mut packed, &mut cpus, we, &[0, 0, 0]);
        packed.eval_combinational();
        for cpu in &mut cpus {
            cpu.check_assertions().unwrap();
        }

        assert_outputs_match(&packed, &cpus, &[q_sync, q_async, read]);
        assert_eq!(
            packed.peek_memory_lanes(0, 0).unwrap(),
            vec![vec![9], vec![0], vec![0]]
        );
        assert_eq!(
            packed.peek_memory_lanes(0, 1).unwrap(),
            vec![vec![0], vec![11], vec![0]]
        );
    }

    #[test]
    fn packed_simulator_snapshot_restores_signals_and_memories() {
        let mut design = Design::new();
        let (en, we, addr, data, count, mem, read);
        {
            let mut m = design.module("PackedSnapshot");
            let clk = m.input("clk", uint(1));
            en = m.input("en", uint(1));
            we = m.input("we", uint(1));
            addr = m.input("addr", uint(2));
            data = m.input("data", uint(8));
            count = m.reg("count", uint(8));
            mem = m.mem("mem", 2, uint(8), 4);
            read = m.output("read", uint(8));
            m.clock(count, clk);
            m.next(count, mux(en, count + lit_u(1, 8), count));
            let read_expr = m.mem_read(mem, addr);
            m.assign(read, read_expr);
            m.mem_write(mem, clk, we, addr, data);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "PackedSnapshot").unwrap();
        let mut original = PackedSimulator::new(program.clone(), 2).unwrap();
        original.set_signal(en, &[1, 0]).unwrap();
        original.set_signal(we, &[1, 1]).unwrap();
        original.set_signal(addr, &[2, 1]).unwrap();
        original.set_signal(data, &[9, 11]).unwrap();
        original.tick();
        original.set_signal(we, &[0, 0]).unwrap();
        original.eval_combinational();
        let snapshot = original.snapshot_storage();

        original.set_signal(en, &[1, 1]).unwrap();
        original.tick();
        original
            .set_memory(mem, &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]])
            .unwrap();

        let mut restored = PackedSimulator::new(program, 2).unwrap();
        restored.restore_storage(&snapshot).unwrap();
        assert_eq!(restored.get_signal(count).unwrap(), vec![1, 0]);
        assert_eq!(restored.get_signal(read).unwrap(), vec![9, 11]);
        assert_eq!(
            restored.get_memory(mem).unwrap(),
            vec![vec![0, 0, 9, 0], vec![0, 11, 0, 0]]
        );

        let mut bad = snapshot.clone();
        bad.values.pop();
        let err = restored.restore_storage(&bad).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SIM_IR_STORAGE_VALUES");
    }

    #[test]
    fn packed_tick_many_matches_repeated_tick() {
        let mut design = Design::new();
        let (en, count);
        {
            let mut m = design.module("TickMany");
            let clk = m.input("clk", uint(1));
            en = m.input("en", uint(1));
            count = m.reg("count", uint(8));
            m.clock(count, clk);
            m.next(count, mux(en, count + lit_u(1, 8), count));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "TickMany").unwrap();
        let mut repeated = PackedSimulator::new(program.clone(), 3).unwrap();
        let mut many = PackedSimulator::new(program, 3).unwrap();
        repeated.set_signal(en, &[1, 0, 1]).unwrap();
        many.set_signal(en, &[1, 0, 1]).unwrap();

        for _ in 0..4 {
            repeated.tick();
        }
        many.tick_many(4);

        assert_eq!(
            many.get_signal(count).unwrap(),
            repeated.get_signal(count).unwrap()
        );
    }

    #[test]
    fn packed_execution_workspaces_reuse_wide_slots() {
        let mut design = Design::new();
        let (a, b, y, q);
        {
            let mut m = design.module("PackedWorkspaceReuse");
            let clk = m.input("clk", uint(1));
            a = m.input("a", uint(40));
            b = m.input("b", uint(40));
            y = m.output("y", uint(40));
            q = m.reg("q", uint(40));
            let sum = a + b;
            m.assign(y, sum.clone() ^ lit_u(0x55, 40));
            m.clock(q, clk);
            m.next(q, sum);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "PackedWorkspaceReuse").unwrap();
        let mut sim = PackedSimulator::new(program, 4).unwrap();
        sim.set_signal(a, &[1, 2, 3, 4]).unwrap();
        sim.set_signal(b, &[(1u128 << 32) + 5, 6, 7, 8]).unwrap();
        let comb_ptrs = sim
            .workspaces
            .comb
            .slots
            .iter()
            .map(|slot| slot.words.as_ptr())
            .collect::<Vec<_>>();
        sim.tick();
        let tick_next_ptrs = sim
            .workspaces
            .tick_next
            .slots
            .iter()
            .map(|slot| slot.words.as_ptr())
            .collect::<Vec<_>>();
        assert_eq!(
            sim.get_signal(q).unwrap(),
            vec![(1u128 << 32) + 6, 8, 10, 12]
        );

        sim.set_signal(a, &[9, 10, 11, 12]).unwrap();
        sim.set_signal(b, &[13, 14, 15, 16]).unwrap();
        assert_eq!(
            comb_ptrs,
            sim.workspaces
                .comb
                .slots
                .iter()
                .map(|slot| slot.words.as_ptr())
                .collect::<Vec<_>>()
        );
        sim.tick();
        assert_eq!(
            tick_next_ptrs,
            sim.workspaces
                .tick_next
                .slots
                .iter()
                .map(|slot| slot.words.as_ptr())
                .collect::<Vec<_>>()
        );
        assert_eq!(sim.get_signal(q).unwrap(), vec![22, 24, 26, 28]);
        assert_eq!(sim.get_signal(y).unwrap(), vec![67, 77, 79, 73]);
    }

    #[test]
    fn packed_register_capture_buffer_reuses_wide_storage() {
        let mut design = Design::new();
        let (a, q8, q40);
        {
            let mut m = design.module("CaptureReuse");
            let clk = m.input("clk", uint(1));
            a = m.input("a", uint(40));
            q8 = m.reg("q8", uint(8));
            q40 = m.reg("q40", uint(40));
            m.clock(q8, clk);
            m.clock(q40, clk);
            m.next(q8, a.value().slice(0, 8));
            m.next(q40, a + lit_u(1, 40));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "CaptureReuse").unwrap();
        let mut sim = PackedSimulator::new(program, 3).unwrap();
        sim.set_signal(
            a,
            &[(1u128 << 32) + 7, (1u128 << 32) + 9, (1u128 << 32) + 11],
        )
        .unwrap();
        sim.tick();
        let first_ptrs = sim
            .register_captures
            .iter()
            .map(|capture| capture.values.as_ptr())
            .collect::<Vec<_>>();
        assert_eq!(sim.register_capture_count, 2);
        assert_eq!(sim.get_signal(q8).unwrap(), vec![7, 9, 11]);
        assert_eq!(
            sim.get_signal(q40).unwrap(),
            vec![(1u128 << 32) + 8, (1u128 << 32) + 10, (1u128 << 32) + 12]
        );

        sim.set_signal(
            a,
            &[(1u128 << 33) + 3, (1u128 << 33) + 5, (1u128 << 33) + 13],
        )
        .unwrap();
        sim.tick();
        let second_ptrs = sim
            .register_captures
            .iter()
            .map(|capture| capture.values.as_ptr())
            .collect::<Vec<_>>();
        assert_eq!(first_ptrs, second_ptrs);
        assert_eq!(sim.register_capture_count, 2);
        assert_eq!(sim.get_signal(q8).unwrap(), vec![3, 5, 13]);
        assert_eq!(
            sim.get_signal(q40).unwrap(),
            vec![(1u128 << 33) + 4, (1u128 << 33) + 6, (1u128 << 33) + 14]
        );
    }

    #[test]
    fn packed_replicated_signal_and_lane_read_apis_work() {
        let mut design = Design::new();
        let (a, b, y);
        {
            let mut m = design.module("ReplicatedApi");
            a = m.input("a", uint(8));
            b = m.input("b", uint(8));
            y = m.output("y", uint(8));
            m.assign(y, a + b);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "ReplicatedApi").unwrap();
        let mut packed = PackedSimulator::new(program, 3).unwrap();

        packed.set_signal_replicated(a, 5).unwrap();
        packed.set_signal(b, &[1, 2, 3]).unwrap();
        assert_eq!(packed.get_signal(y).unwrap(), vec![6, 7, 8]);
        assert_eq!(packed.get_signal_lane(y, 2).unwrap(), 8);

        packed.set_signals_replicated(&[(a, 10), (b, 4)]).unwrap();
        assert_eq!(packed.get_signal(y).unwrap(), vec![14, 14, 14]);

        let err = packed.get_signal_lane(y, 3).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SIM_IR_LANE");
    }

    #[test]
    fn simd_cpu_matches_packed_for_one_limb_fast_path() {
        let mut design = Design::new();
        let (a, b, sel, en, y, eq, lt, slice_out, concat_out, q);
        {
            let mut m = design.module("SimdFast");
            let clk = m.input("clk", uint(1));
            a = m.input("a", uint(16));
            b = m.input("b", uint(16));
            sel = m.input("sel", uint(1));
            en = m.input("en", uint(1));
            y = m.output("y", uint(16));
            eq = m.output("eq", uint(1));
            lt = m.output("lt", uint(1));
            slice_out = m.output("slice_out", uint(8));
            concat_out = m.output("concat_out", uint(16));
            q = m.reg("q", uint(16));
            let arithmetic = ((a + b) ^ (a * b) ^ (a - b)).trunc(16);
            m.assign(y, mux(sel, arithmetic, (a | b) & !a.value()));
            m.assign(eq, a.value().eq_expr(b));
            m.assign(lt, a.value().lt_expr(b));
            m.assign(slice_out, a.value().slice(4, 8));
            m.assign(
                concat_out,
                concat([a.value().slice(0, 8), b.value().slice(0, 8)]),
            );
            m.clock(q, clk);
            m.next(q, mux(en, q + lit_u(1, 16), q));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "SimdFast").unwrap();
        let suitability = analyze_simd_suitability(&program).unwrap();
        assert!(suitability.total.fast_instrs > 0);
        assert_eq!(suitability.total.fallback_instrs, 0);
        assert!(suitability.total.fast_path_profile.one_limb_ops > 0);
        assert_eq!(suitability.fallback_ratio_x100, 0);
        assert!(suitability.score_x100 >= 8_500);
        assert_eq!(
            suitability.recommendation,
            SimdSuitabilityRecommendation::SimdCandidate
        );
        let affinity = analyze_backend_affinity(&program).unwrap();
        assert_eq!(
            affinity.recommendation,
            BackendAffinityRecommendation::SimdCpuCandidate
        );
        assert_eq!(affinity.total.wide_fallback_instrs, 0);
        assert!(affinity.total.scalar_fast_one_limb_instrs > 0);
        let mut packed = PackedSimulator::new(program.clone(), 5).unwrap();
        let mut simd = SimdCpuSimulator::new(program, 5).unwrap();

        for (signal, values) in [
            (a, vec![0, 1, 15, 255, 1024]),
            (b, vec![0, 2, 15, 17, 2048]),
            (sel, vec![0, 1, 0, 1, 1]),
            (en, vec![1, 0, 1, 1, 0]),
        ] {
            packed.set_signal(signal, &values).unwrap();
            simd.set_signal(signal, &values).unwrap();
        }
        simd.reset_simd_stats();
        simd.eval_combinational().unwrap();
        let stats = simd.simd_stats();
        assert!(stats.fast_instrs > 0);
        assert_eq!(stats.fallback_instrs, 0);
        assert!(stats.fast_paths.one_limb_ops > 0);
        assert!(stats.state_reuses > 0);
        assert_eq!(stats.lane_materializations, 0);
        assert_eq!(simd.get_signal(y).unwrap(), packed.get_signal(y).unwrap());
        assert_eq!(simd.get_signal(eq).unwrap(), packed.get_signal(eq).unwrap());
        assert_eq!(simd.get_signal(lt).unwrap(), packed.get_signal(lt).unwrap());
        assert_eq!(
            simd.get_signal(slice_out).unwrap(),
            packed.get_signal(slice_out).unwrap()
        );
        assert_eq!(
            simd.get_signal(concat_out).unwrap(),
            packed.get_signal(concat_out).unwrap()
        );

        simd.tick_many(4).unwrap();
        packed.tick_many(4);
        assert_eq!(simd.get_signal(q).unwrap(), packed.get_signal(q).unwrap());
        assert_eq!(simd.get_signal(y).unwrap(), packed.get_signal(y).unwrap());
    }

    /// The comb-fused `tick_many` must be bit-identical to the per-`tick()` loop
    /// (`tick()` is unchanged — still comb→capture/commit→comb — so it is the naive
    /// reference). Uses registers that evolve and a comb output reading them, over
    /// many cycles, across multiple lanes.
    #[test]
    fn comb_fusion_tick_many_matches_naive_tick_loop() {
        let (din, q, r, y, oc);
        let mut design = Design::new();
        {
            let mut m = design.module("Fuse");
            let clk = m.input("clk", uint(1));
            din = m.input("din", uint(16));
            q = m.reg("q", uint(16));
            r = m.reg("r", uint(16));
            m.clock(q, clk);
            m.clock(r, clk);
            // registers evolve and r depends combinationally on q
            m.next(q, (q + din).trunc(16));
            m.next(r, (r + (q.value() ^ din.value())).trunc(16));
            y = m.output("y", uint(16));
            oc = m.output("oc", uint(16));
            m.assign(y, (q + r).trunc(16));
            m.assign(oc, (q.value() ^ (r * din)).trunc(16));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Fuse").unwrap();

        let lanes = 5;
        let dvals: Vec<u128> = vec![1, 7, 255, 4096, 65535];
        for n in [0usize, 1, 2, 3, 17, 64, 257] {
            // fresh engines so each `n` is an independent run from reset
            let mut a = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
            let mut b = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
            a.set_signal(din, &dvals).unwrap();
            b.set_signal(din, &dvals).unwrap();
            for _ in 0..n {
                a.tick().unwrap(); // naive double-comb reference
            }
            b.tick_many(n).unwrap(); // fused
            for sig in [q, r, y, oc] {
                assert_eq!(
                    a.get_signal(sig).unwrap(),
                    b.get_signal(sig).unwrap(),
                    "comb-fusion diverged at n={n}"
                );
            }
        }
    }

    #[test]
    fn simd_cpu_sign_extends_one_limb_values_without_fallback() {
        let mut design = Design::new();
        let (small, bit, medium, small_ext, bit_ext, medium_ext);
        {
            let mut m = design.module("SimdSext");
            small = m.input("small", sint(4));
            bit = m.input("bit", sint(1));
            medium = m.input("medium", sint(16));
            small_ext = m.output("small_ext", sint(8));
            bit_ext = m.output("bit_ext", sint(8));
            medium_ext = m.output("medium_ext", sint(32));
            m.assign(small_ext, sext(small, 8));
            m.assign(bit_ext, sext(bit, 8));
            m.assign(medium_ext, sext(medium, 32));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "SimdSext").unwrap();
        let suitability = analyze_simd_suitability(&program).unwrap();
        assert_eq!(suitability.total.fallback_reasons.sext, 0);
        assert_eq!(suitability.total.fallback_instrs, 0);
        assert_eq!(suitability.estimated_fallback_cost, 0);
        let mut packed = PackedSimulator::new(program.clone(), 4).unwrap();
        let mut simd = SimdCpuSimulator::new(program, 4).unwrap();

        for (signal, values) in [
            (small, vec![0, 7, 8, 15]),
            (bit, vec![0, 1, 1, 0]),
            (medium, vec![0x7fff, 0x8000, 0xffff, 1]),
        ] {
            packed.set_signal(signal, &values).unwrap();
            simd.set_signal(signal, &values).unwrap();
        }
        simd.reset_simd_stats();
        simd.eval_combinational().unwrap();
        let stats = simd.simd_stats();
        assert!(stats.fast_instrs > 0);
        assert_eq!(stats.fallback_instrs, 0);
        assert_eq!(stats.fallback_reasons.sext, 0);
        assert_eq!(stats.lane_materializations, 0);
        assert_eq!(
            simd.get_signal(small_ext).unwrap(),
            packed.get_signal(small_ext).unwrap()
        );
        assert_eq!(
            simd.get_signal(bit_ext).unwrap(),
            packed.get_signal(bit_ext).unwrap()
        );
        assert_eq!(
            simd.get_signal(medium_ext).unwrap(),
            packed.get_signal(medium_ext).unwrap()
        );
    }

    #[test]
    fn simd_cpu_signed_lt_one_limb_values_without_fallback() {
        let mut design = Design::new();
        let (lhs4, rhs4, lhs32, rhs32, lt4, lt32);
        {
            let mut m = design.module("SimdSignedLt");
            lhs4 = m.input("lhs4", sint(4));
            rhs4 = m.input("rhs4", sint(4));
            lhs32 = m.input("lhs32", sint(32));
            rhs32 = m.input("rhs32", sint(32));
            lt4 = m.output("lt4", uint(1));
            lt32 = m.output("lt32", uint(1));
            m.assign(lt4, lhs4.value().lt_expr(rhs4));
            m.assign(lt32, lhs32.value().lt_expr(rhs32));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "SimdSignedLt").unwrap();
        let suitability = analyze_simd_suitability(&program).unwrap();
        assert_eq!(suitability.total.fallback_reasons.signed_lt, 0);
        assert_eq!(suitability.total.fallback_instrs, 0);
        assert_eq!(suitability.estimated_fallback_cost, 0);
        let mut packed = PackedSimulator::new(program.clone(), 5).unwrap();
        let mut simd = SimdCpuSimulator::new(program, 5).unwrap();

        for (signal, values) in [
            (lhs4, vec![0, 7, 8, 15, 8]),
            (rhs4, vec![1, 8, 7, 0, 15]),
            (lhs32, vec![0, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 3]),
            (rhs32, vec![1, 0x8000_0000, 0x7fff_ffff, 0, 0xffff_fffe]),
        ] {
            packed.set_signal(signal, &values).unwrap();
            simd.set_signal(signal, &values).unwrap();
        }
        simd.reset_simd_stats();
        simd.eval_combinational().unwrap();
        let stats = simd.simd_stats();
        assert!(stats.fast_instrs > 0);
        assert_eq!(stats.fallback_instrs, 0);
        assert_eq!(stats.fallback_reasons.signed_lt, 0);
        assert_eq!(stats.lane_materializations, 0);
        assert_eq!(
            simd.get_signal(lt4).unwrap(),
            packed.get_signal(lt4).unwrap()
        );
        assert_eq!(
            simd.get_signal(lt32).unwrap(),
            packed.get_signal(lt32).unwrap()
        );
    }

    #[test]
    fn simd_cpu_handles_two_limb_fast_paths_without_fallback() {
        let mut design = Design::new();
        let (
            a,
            b,
            sa,
            sb,
            small,
            bitwise,
            sum,
            diff,
            product,
            eq,
            ne,
            ult,
            slt,
            slice_out,
            concat_out,
            sext_out,
        );
        {
            let mut m = design.module("SimdTwoLimbFast");
            a = m.input("a", uint(40));
            b = m.input("b", uint(40));
            sa = m.input("sa", sint(40));
            sb = m.input("sb", sint(40));
            small = m.input("small", sint(16));
            bitwise = m.output("bitwise", uint(40));
            sum = m.output("sum", uint(40));
            diff = m.output("diff", uint(40));
            product = m.output("product", uint(40));
            eq = m.output("eq", uint(1));
            ne = m.output("ne", uint(1));
            ult = m.output("ult", uint(1));
            slt = m.output("slt", uint(1));
            slice_out = m.output("slice_out", uint(33));
            concat_out = m.output("concat_out", uint(56));
            sext_out = m.output("sext_out", sint(48));
            m.assign(bitwise, ((a.value() ^ b.value()) | !a.value()).trunc(40));
            m.assign(sum, (a + b).trunc(40));
            m.assign(diff, (a - b).trunc(40));
            m.assign(product, (a * b).trunc(40));
            m.assign(eq, a.value().eq_expr(b));
            m.assign(ne, a.value().ne_expr(b));
            m.assign(ult, a.value().lt_expr(b));
            m.assign(slt, sa.value().lt_expr(sb));
            m.assign(slice_out, a.value().slice(7, 33));
            m.assign(
                concat_out,
                concat([a.value().slice(0, 40), b.value().slice(0, 16)]),
            );
            m.assign(sext_out, sext(small, 48));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "SimdTwoLimbFast").unwrap();
        let suitability = analyze_simd_suitability(&program).unwrap();
        assert_eq!(suitability.total.fallback_instrs, 0);
        assert_eq!(suitability.total.fallback_reasons.wide_op, 0);
        assert_eq!(suitability.total.fallback_reasons.wide_concat, 0);
        assert_eq!(suitability.total.fallback_reasons.signed_lt, 0);
        assert_eq!(suitability.total.fallback_reasons.sext, 0);
        assert!(suitability.total.fast_path_profile.two_limb_ops > 0);
        assert!(suitability.total.fast_path_profile.two_limb_mul_ops > 0);
        let mut packed = PackedSimulator::new(program.clone(), 5).unwrap();
        let mut simd = SimdCpuSimulator::new(program, 5).unwrap();

        let mask40 = (1u128 << 40) - 1;
        for (signal, values) in [
            (a, vec![0, 1, 0xffff_ffff, 0xffff_ffff_ff, 0x8000_0000_00]),
            (b, vec![1, 0xffff_ffff, 1, 0x10, 0x7fff_ffff_ff]),
            (sa, vec![0, 7, 1u128 << 39, mask40, 1u128 << 39]),
            (sb, vec![1, 1u128 << 39, 7, 0, mask40]),
            (small, vec![0, 0x7fff, 0x8000, 0xffff, 1]),
        ] {
            packed.set_signal(signal, &values).unwrap();
            simd.set_signal(signal, &values).unwrap();
        }
        simd.reset_simd_stats();
        simd.eval_combinational().unwrap();
        let stats = simd.simd_stats();
        assert!(stats.fast_instrs > 0);
        assert_eq!(stats.fallback_instrs, 0);
        assert!(stats.fast_paths.two_limb_ops > 0);
        assert!(stats.fast_paths.two_limb_mul_ops > 0);
        #[cfg(target_arch = "aarch64")]
        assert!(stats.fast_paths.native_two_limb_ops > 0);
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(stats.fast_paths.native_two_limb_ops, 0);
        assert_eq!(stats.lane_materializations, 0);
        assert_eq!(stats.fallback_reasons.wide_op, 0);
        assert_eq!(stats.fallback_reasons.wide_concat, 0);
        assert_eq!(stats.fallback_reasons.signed_lt, 0);
        assert_eq!(stats.fallback_reasons.sext, 0);
        for signal in [
            bitwise, sum, diff, product, eq, ne, ult, slt, slice_out, concat_out, sext_out,
        ] {
            assert_eq!(
                simd.get_signal(signal).unwrap(),
                packed.get_signal(signal).unwrap()
            );
        }
    }

    #[test]
    fn two_limb_simd_helper_adds_with_carry() {
        let lanes = 5;
        let lhs = vec![0xffff_ffff, 0, 5, 0xffff_fffe, 10, 0, 1, 2, 0xff, 0x12];
        let rhs = vec![1, 1, 7, 3, 20, 0, 2, 3, 1, 0x34];
        let mut out = vec![0; lanes * 2];
        let native = simd_words::two_limb_binop_into(
            &lhs,
            &rhs,
            lanes,
            2,
            2,
            0xff,
            &mut out,
            simd_words::TwoLimbBinOp::Add,
        );
        #[cfg(target_arch = "aarch64")]
        assert!(native);
        #[cfg(not(target_arch = "aarch64"))]
        assert!(!native);
        assert_eq!(&out[..lanes], &[0, 1, 12, 1, 30]);
        assert_eq!(&out[lanes..], &[1, 3, 5, 1, 0x46]);
    }

    #[test]
    fn two_limb_simd_helper_subtracts_with_borrow() {
        let lanes = 5;
        let lhs = vec![0, 1, 10, 0, 0xffff_ffff, 1, 5, 8, 0x10, 0];
        let rhs = vec![1, 2, 3, 0, 0xffff_fffe, 0, 1, 2, 1, 0];
        let mut out = vec![0; lanes * 2];
        let native = simd_words::two_limb_binop_into(
            &lhs,
            &rhs,
            lanes,
            2,
            2,
            0xff,
            &mut out,
            simd_words::TwoLimbBinOp::Sub,
        );
        #[cfg(target_arch = "aarch64")]
        assert!(native);
        #[cfg(not(target_arch = "aarch64"))]
        assert!(!native);
        assert_eq!(&out[..lanes], &[0xffff_ffff, 0xffff_ffff, 7, 0, 1]);
        assert_eq!(&out[lanes..], &[0, 3, 6, 0x0f, 0]);
    }

    #[test]
    fn two_limb_simd_helper_compares_high_limb() {
        let lanes = 5;
        let lhs = vec![1, 2, 3, 4, 5, 0, 0x80, 0x7f, 0xff, 0x12];
        let rhs = vec![1, 2, 4, 4, 5, 0, 0x80, 0x7f, 0x7f, 0x12];
        let mut out = vec![0; lanes];
        let native = simd_words::two_limb_compare_into(
            &lhs,
            &rhs,
            lanes,
            2,
            2,
            u32::MAX,
            0xff,
            &mut out,
            simd_words::TwoLimbCompareOp::Eq,
        );
        #[cfg(target_arch = "aarch64")]
        assert!(native);
        #[cfg(not(target_arch = "aarch64"))]
        assert!(!native);
        assert_eq!(out, vec![1, 1, 0, 0, 1]);

        let native = simd_words::two_limb_compare_into(
            &lhs,
            &rhs,
            lanes,
            2,
            2,
            u32::MAX,
            0xff,
            &mut out,
            simd_words::TwoLimbCompareOp::Ne,
        );
        #[cfg(target_arch = "aarch64")]
        assert!(native);
        #[cfg(not(target_arch = "aarch64"))]
        assert!(!native);
        assert_eq!(out, vec![0, 0, 1, 1, 0]);
    }

    #[test]
    fn two_limb_simd_helper_compares_unsigned_and_signed_lt() {
        let lanes = 5;
        let lhs = vec![1, 0, 0, 0xffff_ffff, 0, 0, 1, 0x80, 0x7f, 0xff];
        let rhs = vec![2, 0xffff_ffff, 0xffff_ffff, 0, 0, 0, 0, 0x7f, 0x80, 0];
        let mut out = vec![0; lanes];
        let native =
            simd_words::two_limb_lt_into(&lhs, &rhs, lanes, 2, 2, u32::MAX, 0xff, None, &mut out);
        #[cfg(target_arch = "aarch64")]
        assert!(native);
        #[cfg(not(target_arch = "aarch64"))]
        assert!(!native);
        assert_eq!(out, vec![1, 0, 0, 1, 0]);

        let native = simd_words::two_limb_lt_into(
            &lhs,
            &rhs,
            lanes,
            2,
            2,
            u32::MAX,
            0xff,
            Some(40),
            &mut out,
        );
        #[cfg(target_arch = "aarch64")]
        assert!(native);
        #[cfg(not(target_arch = "aarch64"))]
        assert!(!native);
        assert_eq!(out, vec![1, 0, 1, 0, 1]);
    }

    #[test]
    fn two_limb_simd_helper_slices_cross_limb() {
        let lanes = 5;
        let values = [
            0x12_3456_789a_u64,
            0xff_0000_0001,
            0x80_ffff_ffff,
            0x01_0000_0000,
            0xab_cdef_0123,
        ];
        let mut input = values.iter().map(|value| *value as u32).collect::<Vec<_>>();
        input.extend(values.iter().map(|value| (*value >> 32) as u32));
        let mut out = vec![0; lanes * 2];
        let native = simd_words::two_limb_slice_into(&input, lanes, 2, 7, 2, 0xff, &mut out);
        #[cfg(target_arch = "aarch64")]
        assert!(native);
        #[cfg(not(target_arch = "aarch64"))]
        assert!(!native);

        let expected = values
            .iter()
            .map(|value| (value >> 7) & ((1u64 << 40) - 1))
            .collect::<Vec<_>>();
        assert_eq!(
            &out[..lanes],
            expected
                .iter()
                .map(|value| *value as u32)
                .collect::<Vec<_>>()
                .as_slice()
        );
        assert_eq!(
            &out[lanes..],
            expected
                .iter()
                .map(|value| ((*value >> 32) as u32) & 0xff)
                .collect::<Vec<_>>()
                .as_slice()
        );
    }

    #[test]
    fn two_limb_simd_helper_sign_extends() {
        let lanes = 5;
        let values = [
            0x00_0000_0001_u64,
            0x7f_ffff_ffff,
            0x80_0000_0000,
            0xff_ffff_ffff,
            0x80_0000_0001,
        ];
        let mut input = values.iter().map(|value| *value as u32).collect::<Vec<_>>();
        input.extend(values.iter().map(|value| (*value >> 32) as u32));
        let mut out = vec![0; lanes * 2];
        let native = simd_words::two_limb_sext_into(&input, lanes, 2, 40, 2, 0xffff, &mut out);
        #[cfg(target_arch = "aarch64")]
        assert!(native);
        #[cfg(not(target_arch = "aarch64"))]
        assert!(!native);

        let src_mask = (1u64 << 40) - 1;
        let dst_mask = (1u64 << 48) - 1;
        let expected = values
            .iter()
            .map(|value| {
                let value = value & src_mask;
                if value & (1u64 << 39) != 0 {
                    (value | !src_mask) & dst_mask
                } else {
                    value
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(
            &out[..lanes],
            expected
                .iter()
                .map(|value| *value as u32)
                .collect::<Vec<_>>()
                .as_slice()
        );
        assert_eq!(
            &out[lanes..],
            expected
                .iter()
                .map(|value| ((*value >> 32) as u32) & 0xffff)
                .collect::<Vec<_>>()
                .as_slice()
        );
    }

    #[test]
    fn simd_cpu_keeps_wide_multiply_on_fallback_path() {
        let mut design = Design::new();
        let (a, b, product);
        {
            let mut m = design.module("SimdWideMulFallback");
            a = m.input("a", uint(72));
            b = m.input("b", uint(72));
            product = m.output("product", uint(72));
            m.assign(product, (a * b).trunc(72));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "SimdWideMulFallback").unwrap();
        let suitability = analyze_simd_suitability(&program).unwrap();
        assert!(suitability.total.fallback_reasons.wide_op > 0);
        let mut packed = PackedSimulator::new(program.clone(), 3).unwrap();
        let mut simd = SimdCpuSimulator::new(program, 3).unwrap();

        for (signal, values) in [
            (a, vec![0, (1u128 << 71) | 3, (1u128 << 64) + 5]),
            (b, vec![7, (1u128 << 65) + 11, (1u128 << 63) + 13]),
        ] {
            let limbs = values
                .into_iter()
                .map(|value| encode_u128_limbs(value, uint(72)))
                .collect::<Vec<_>>();
            packed.set_signal_limbs(signal, &limbs).unwrap();
            simd.set_signal_limbs(signal, &limbs).unwrap();
        }
        simd.reset_simd_stats();
        simd.eval_combinational().unwrap();
        let stats = simd.simd_stats();
        assert!(stats.fallback_instrs > 0);
        assert!(stats.lane_materializations > 0);
        assert!(stats.fallback_reasons.wide_op > 0);
        assert_eq!(
            simd.get_signal_limbs(product).unwrap(),
            packed.get_signal_limbs(product).unwrap()
        );
    }

    #[test]
    fn simd_cpu_reads_two_limb_memory_without_fallback() {
        let mut design = Design::new();
        let (addr, read);
        let mem;
        {
            let mut m = design.module("SimdTwoLimbMemory");
            addr = m.input("addr", uint(3));
            mem = m.mem("mem", 3, uint(40), 4);
            read = m.output("read", uint(40));
            let read_expr = m.mem_read(mem, addr);
            m.assign(read, read_expr);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "SimdTwoLimbMemory").unwrap();
        let suitability = analyze_simd_suitability(&program).unwrap();
        assert_eq!(suitability.total.fallback_reasons.mem_read, 0);
        assert_eq!(suitability.total.fallback_instrs, 0);
        assert!(suitability.total.fast_path_profile.two_limb_memory_reads > 0);
        let mut packed = PackedSimulator::new(program.clone(), 4).unwrap();
        let mut simd = SimdCpuSimulator::new(program, 4).unwrap();

        let memory = vec![
            vec![0, 1, 0xffff_ffff, 0xffff_ffff_ff],
            vec![5, 6, 7, 8],
            vec![
                0x1000_0000_01,
                0x1000_0000_02,
                0x1000_0000_03,
                0x1000_0000_04,
            ],
            vec![
                0xffff_ffff_ff,
                0x8000_0000_00,
                0x7fff_ffff_ff,
                0x1234_5678_9a,
            ],
        ];
        packed.set_memory(mem, &memory).unwrap();
        simd.set_memory(mem, &memory).unwrap();
        packed.set_signal(addr, &[0, 3, 6, 2]).unwrap();
        simd.set_signal(addr, &[0, 3, 6, 2]).unwrap();

        simd.reset_simd_stats();
        simd.eval_combinational().unwrap();
        let stats = simd.simd_stats();
        assert!(stats.fast_instrs > 0);
        assert_eq!(stats.fallback_instrs, 0);
        assert!(stats.fast_paths.two_limb_memory_reads > 0);
        assert_eq!(stats.lane_materializations, 0);
        assert_eq!(stats.fallback_reasons.mem_read, 0);
        assert_eq!(
            simd.get_signal(read).unwrap(),
            packed.get_signal(read).unwrap()
        );
        assert_eq!(simd.get_signal(read).unwrap()[2], 0);
    }

    #[test]
    fn simd_cpu_muxes_two_limb_values_without_fallback() {
        let mut design = Design::new();
        let (sel, then_in, else_in, out);
        {
            let mut m = design.module("SimdTwoLimbMux");
            sel = m.input("sel", uint(1));
            then_in = m.input("then_in", uint(40));
            else_in = m.input("else_in", uint(40));
            out = m.output("out", uint(40));
            m.assign(out, mux(sel, then_in, else_in));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "SimdTwoLimbMux").unwrap();
        let suitability = analyze_simd_suitability(&program).unwrap();
        assert_eq!(suitability.total.fallback_reasons.wide_op, 0);
        assert_eq!(suitability.total.fallback_instrs, 0);
        assert!(suitability.total.fast_path_profile.two_limb_mux_ops > 0);
        let mut packed = PackedSimulator::new(program.clone(), 4).unwrap();
        let mut simd = SimdCpuSimulator::new(program, 4).unwrap();

        for (signal, values) in [
            (sel, vec![0, 1, 0, 1]),
            (
                then_in,
                vec![0xffff_ffff_ff, 0x8000_0000_00, 7, 0x1234_5678_9a],
            ),
            (else_in, vec![1, 2, 0x7fff_ffff_ff, 0xffff_ffff_ff]),
        ] {
            packed.set_signal(signal, &values).unwrap();
            simd.set_signal(signal, &values).unwrap();
        }
        simd.reset_simd_stats();
        simd.eval_combinational().unwrap();
        let stats = simd.simd_stats();
        assert!(stats.fast_instrs > 0);
        assert_eq!(stats.fallback_instrs, 0);
        assert!(stats.fast_paths.two_limb_mux_ops > 0);
        #[cfg(target_arch = "aarch64")]
        assert!(stats.fast_paths.native_two_limb_ops > 0);
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(stats.fast_paths.native_two_limb_ops, 0);
        assert_eq!(stats.lane_materializations, 0);
        assert_eq!(stats.fallback_reasons.wide_op, 0);
        assert_eq!(
            simd.get_signal(out).unwrap(),
            packed.get_signal(out).unwrap()
        );
    }

    #[test]
    fn simd_suitability_reports_mux_heavy_two_limb_design_as_mixed() {
        let mut design = Design::new();
        {
            let mut m = design.module("SimdMuxHeavy");
            let sel = m.input("sel", uint(1));
            let a = m.input("a", uint(40));
            let b = m.input("b", uint(40));
            let out0 = m.output("out0", uint(40));
            let out1 = m.output("out1", uint(40));
            let out2 = m.output("out2", uint(40));
            let out3 = m.output("out3", uint(40));
            m.assign(out0, mux(sel, a, b));
            m.assign(out1, mux(sel, b, a));
            m.assign(out2, mux(sel, a.value() ^ b.value(), b));
            m.assign(out3, mux(sel, b.value() ^ a.value(), a));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "SimdMuxHeavy").unwrap();
        let suitability = analyze_simd_suitability(&program).unwrap();
        assert_eq!(suitability.total.fallback_instrs, 0);
        assert!(suitability.total.fast_path_profile.two_limb_mux_ops >= 4);
        assert_eq!(
            suitability.recommendation,
            SimdSuitabilityRecommendation::MixedCandidate
        );
        let affinity = analyze_backend_affinity(&program).unwrap();
        assert_eq!(
            affinity.recommendation,
            BackendAffinityRecommendation::MixedScalarSimdCandidate
        );
        assert!(affinity
            .reasons
            .iter()
            .any(|reason| reason.contains("higher-cost two-limb")));
    }

    #[test]
    fn simd_cpu_writes_two_limb_memory_without_fallback() {
        let mut design = Design::new();
        let (we, addr, data, read);
        let mem;
        {
            let mut m = design.module("SimdTwoLimbMemoryWrite");
            let clk = m.input("clk", uint(1));
            we = m.input("we", uint(1));
            addr = m.input("addr", uint(2));
            data = m.input("data", uint(40));
            mem = m.mem("mem", 2, uint(40), 4);
            read = m.output("read", uint(40));
            let read_expr = m.mem_read(mem, addr);
            m.assign(read, read_expr);
            m.mem_write(mem, clk, we, addr, data);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "SimdTwoLimbMemoryWrite").unwrap();
        let suitability = analyze_simd_suitability(&program).unwrap();
        assert!(suitability.total.fast_path_profile.two_limb_memory_reads > 0);
        assert!(suitability.total.fast_path_profile.memory_write_effects > 0);
        let mut packed = PackedSimulator::new(program.clone(), 4).unwrap();
        let mut simd = SimdCpuSimulator::new(program, 4).unwrap();

        let initial = vec![vec![10, 11, 12, 13]; 4];
        packed.set_memory(mem, &initial).unwrap();
        simd.set_memory(mem, &initial).unwrap();
        for (signal, values) in [
            (we, vec![1, 0, 1, 1]),
            (addr, vec![0, 1, 2, 3]),
            (
                data,
                vec![
                    0xffff_ffff_ff,
                    0x8000_0000_00,
                    0x1234_5678_9a,
                    0x7fff_ffff_ff,
                ],
            ),
        ] {
            packed.set_signal(signal, &values).unwrap();
            simd.set_signal(signal, &values).unwrap();
        }
        simd.reset_simd_stats();
        simd.eval_combinational().unwrap();
        let stats = simd.simd_stats();
        assert_eq!(stats.fallback_instrs, 0);
        assert!(stats.fast_paths.two_limb_memory_reads > 0);
        assert_eq!(stats.lane_materializations, 0);

        packed.tick();
        simd.tick().unwrap();
        let stats = simd.simd_stats();
        assert!(stats.fast_paths.memory_write_effects > 0);
        assert_eq!(
            simd.get_memory(mem).unwrap(),
            packed.get_memory(mem).unwrap()
        );
        assert_eq!(simd.get_memory(mem).unwrap()[1][1], 11);
        assert_eq!(
            simd.get_signal(read).unwrap(),
            packed.get_signal(read).unwrap()
        );
    }

    #[test]
    fn simd_cpu_keeps_wide_memory_read_on_fallback_path() {
        let mut design = Design::new();
        let (addr, read);
        let mem;
        {
            let mut m = design.module("SimdWideMemoryFallback");
            addr = m.input("addr", uint(2));
            mem = m.mem("mem", 2, uint(72), 4);
            read = m.output("read", uint(72));
            let read_expr = m.mem_read(mem, addr);
            m.assign(read, read_expr);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "SimdWideMemoryFallback").unwrap();
        let suitability = analyze_simd_suitability(&program).unwrap();
        assert!(suitability.total.fallback_reasons.mem_read > 0);
        let mut packed = PackedSimulator::new(program.clone(), 3).unwrap();
        let mut simd = SimdCpuSimulator::new(program, 3).unwrap();

        let memory_values = [
            vec![0, (1u128 << 71) | 5, (1u128 << 64) + 7, 9],
            vec![11, 13, (1u128 << 70) | 17, 19],
            vec![(1u128 << 69) | 23, 29, 31, 37],
        ];
        let memory_limbs = memory_values
            .iter()
            .map(|lane| {
                lane.iter()
                    .copied()
                    .map(|value| encode_u128_limbs(value, uint(72)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        packed.set_memory_limbs(mem, &memory_limbs).unwrap();
        simd.set_memory_limbs(mem, &memory_limbs).unwrap();
        packed.set_signal(addr, &[1, 2, 3]).unwrap();
        simd.set_signal(addr, &[1, 2, 3]).unwrap();

        simd.reset_simd_stats();
        simd.eval_combinational().unwrap();
        let stats = simd.simd_stats();
        assert!(stats.fallback_instrs > 0);
        assert!(stats.lane_materializations > 0);
        assert!(stats.fallback_reasons.mem_read > 0);
        assert_eq!(
            simd.get_signal_limbs(read).unwrap(),
            packed.get_signal_limbs(read).unwrap()
        );
    }

    #[test]
    fn simd_cpu_matches_packed_for_memory_and_wide_fallbacks() {
        let mut design = Design::new();
        let (we, addr, data, wide_in, read, wide_out);
        let mem;
        {
            let mut m = design.module("SimdFallback");
            let clk = m.input("clk", uint(1));
            we = m.input("we", uint(1));
            addr = m.input("addr", uint(2));
            data = m.input("data", uint(8));
            wide_in = m.input("wide_in", uint(40));
            mem = m.mem("mem", 2, uint(8), 4);
            read = m.output("read", uint(8));
            wide_out = m.output("wide_out", uint(40));
            let read_expr = m.mem_read(mem, addr);
            m.assign(read, read_expr);
            m.assign(
                wide_out,
                concat([wide_in.value().slice(0, 32), data.value()]),
            );
            m.mem_write(mem, clk, we, addr, data);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "SimdFallback").unwrap();
        let suitability = analyze_simd_suitability(&program).unwrap();
        assert_eq!(suitability.total.fallback_reasons.mem_read, 0);
        assert_eq!(suitability.total.fallback_reasons.wide_concat, 0);
        assert_eq!(suitability.total.fallback_reasons.wide_op, 0);
        assert_eq!(suitability.estimated_fallback_cost, 0);
        assert_eq!(
            suitability.recommendation,
            SimdSuitabilityRecommendation::GpuCandidateBlocked
        );
        let affinity = analyze_backend_affinity(&program).unwrap();
        assert_eq!(
            affinity.recommendation,
            BackendAffinityRecommendation::GpuBlocked
        );
        assert!(affinity.total.memory_ops > 0);
        assert_eq!(affinity.total.gpu_hostile_reasons.mem_read, 0);
        let mut packed = PackedSimulator::new(program.clone(), 3).unwrap();
        let mut simd = SimdCpuSimulator::new(program, 3).unwrap();

        let initial = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]];
        packed.set_memory(mem, &initial).unwrap();
        simd.set_memory_replicated(mem, &[1, 2, 3, 4]).unwrap();
        simd.set_memory(mem, &initial).unwrap();
        for (signal, values) in [
            (we, vec![1, 0, 1]),
            (addr, vec![2, 1, 3]),
            (data, vec![21, 22, 23]),
            (wide_in, vec![0x1_0000_ffff, 0x2_0000_1234, 0x3_0000_abcd]),
        ] {
            packed.set_signal(signal, &values).unwrap();
            simd.set_signal(signal, &values).unwrap();
        }
        simd.reset_simd_stats();
        simd.eval_combinational().unwrap();
        let stats = simd.simd_stats();
        assert!(stats.fast_instrs > 0);
        assert_eq!(stats.fallback_instrs, 0);
        assert!(stats.state_reuses > 0);
        assert_eq!(stats.lane_materializations, 0);
        assert_eq!(stats.fallback_reasons.mem_read, 0);
        assert_eq!(stats.fallback_reasons.wide_op, 0);
        assert_eq!(stats.fallback_reasons.wide_concat, 0);
        assert_eq!(
            simd.get_signal(read).unwrap(),
            packed.get_signal(read).unwrap()
        );
        assert_eq!(
            simd.get_signal(wide_out).unwrap(),
            packed.get_signal(wide_out).unwrap()
        );

        packed.tick();
        simd.tick().unwrap();
        packed.set_signal(we, &[0, 0, 0]).unwrap();
        simd.set_signal(we, &[0, 0, 0]).unwrap();
        assert_eq!(
            simd.get_memory(mem).unwrap(),
            packed.get_memory(mem).unwrap()
        );
        assert_eq!(
            simd.get_signal(read).unwrap(),
            packed.get_signal(read).unwrap()
        );
    }

    #[test]
    fn simd_suitability_reports_two_limb_arithmetic_as_fast() {
        let mut design = Design::new();
        {
            let mut m = design.module("WideArithmetic");
            let a = m.input("a", uint(40));
            let b = m.input("b", uint(40));
            let y = m.output("y", uint(40));
            m.assign(y, ((a + b) ^ (a * b) ^ (a - b)).trunc(40));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "WideArithmetic").unwrap();
        let suitability = analyze_simd_suitability(&program).unwrap();
        assert_eq!(suitability.total.fallback_reasons.wide_op, 0);
        assert_eq!(suitability.total.fallback_instrs, 0);
        assert!(suitability.total.wide_instrs > 0);
        assert_eq!(suitability.fallback_ratio_x100, 0);
        assert_eq!(
            suitability.recommendation,
            SimdSuitabilityRecommendation::SimdCandidate
        );
        let affinity = analyze_backend_affinity(&program).unwrap();
        assert_eq!(
            affinity.recommendation,
            BackendAffinityRecommendation::SimdCpuCandidate
        );
        assert_eq!(affinity.total.wide_fallback_instrs, 0);
        assert!(affinity.total.simd_fast_path_profile.two_limb_ops > 0);
        assert!(affinity.total.simd_fast_path_profile.two_limb_mul_ops > 0);
        assert!(affinity.total.estimated_simd_cpu_cost > affinity.total.simd_fast_instrs);
    }

    #[test]
    fn simd_suitability_cost_model_penalizes_systolic_wide_op_profile() {
        let total = SimdSuitabilityBlockReport {
            instr_count: 2688,
            fast_instrs: 2432,
            fallback_instrs: 256,
            lane_materializations_per_lane: 256,
            fallback_reasons: ReplaySimdFallbackStats {
                wide_op: 256,
                ..ReplaySimdFallbackStats::default()
            },
            one_limb_instrs: 2432,
            wide_instrs: 256,
            max_packet_width: 256,
            ..SimdSuitabilityBlockReport::default()
        };
        let cost = simd_suitability_cost(&total);
        let score = simd_suitability_score_x100(cost);
        let fallback_ratio = simd_fallback_ratio_x100(&total);
        let recommendation = simd_suitability_recommendation(&total, score, fallback_ratio);
        assert_eq!(fallback_ratio, 9);
        assert_eq!(cost.fast, 2432);
        assert_eq!(cost.fallback, 12_288);
        assert_eq!(cost.materialization, 4_096);
        assert_ne!(recommendation, SimdSuitabilityRecommendation::SimdCandidate);
        assert_eq!(
            recommendation,
            SimdSuitabilityRecommendation::ScalarPreferred
        );
    }

    #[test]
    fn simd_cpu_snapshot_restore_matches_packed_storage() {
        let mut design = Design::new();
        let (en, count);
        {
            let mut m = design.module("SimdSnapshot");
            let clk = m.input("clk", uint(1));
            en = m.input("en", uint(1));
            count = m.reg("count", uint(8));
            m.clock(count, clk);
            m.next(count, mux(en, count + lit_u(1, 8), count));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "SimdSnapshot").unwrap();
        let mut original = SimdCpuSimulator::new(program.clone(), 4).unwrap();
        original.set_signal(en, &[1, 0, 1, 1]).unwrap();
        original.tick_many(3).unwrap();
        let snapshot = original.snapshot_storage();

        original.set_signal(en, &[1, 1, 1, 1]).unwrap();
        original.tick().unwrap();

        let mut restored = SimdCpuSimulator::new(program.clone(), 4).unwrap();
        restored.restore_storage(&snapshot).unwrap();
        let mut packed = PackedSimulator::new(program, 4).unwrap();
        packed.restore_storage(&snapshot).unwrap();
        assert_eq!(
            restored.get_signal(count).unwrap(),
            packed.get_signal(count).unwrap()
        );
    }

    #[test]
    fn backend_batched_replay_primitives_match_eager_stepping() {
        let mut design = Design::new();
        let (rst_n, we, addr, data, a, y, read, q_out);
        {
            let mut m = design.module("BatchedReplay");
            let clk = m.input("clk", uint(1));
            rst_n = m.input("rst_n", uint(1));
            we = m.input("we", uint(1));
            addr = m.input("addr", uint(2));
            data = m.input("data", uint(8));
            a = m.input("a", uint(8));
            let q = m.reg("q", uint(8));
            let mem = m.mem("mem", 2, uint(8), 4);
            y = m.output("y", uint(8));
            read = m.output("read", uint(8));
            q_out = m.output("q_out", uint(8));
            m.clock(q, clk);
            m.async_reset_low(q, rst_n, 7);
            m.next(q, mux(we, data, q + a));
            let read_expr = m.mem_read(mem, addr);
            m.assign(read, read_expr.clone());
            m.assign(y, q + read_expr + a);
            m.assign(q_out, q);
            m.mem_write(mem, clk, we, addr, data);
        }

        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "BatchedReplay").unwrap();
        let steps = [
            vec![(rst_n, 0), (we, 0), (addr, 0), (data, 11), (a, 1)],
            vec![(rst_n, 1), (we, 1), (addr, 2), (data, 21), (a, 3)],
            vec![(rst_n, 1), (we, 0), (addr, 2), (data, 33), (a, 5)],
            vec![(rst_n, 1), (we, 1), (addr, 1), (data, 44), (a, 7)],
        ];

        for options in [
            SimBackendOptions::scalar(),
            SimBackendOptions::packed_cpu(3),
            SimBackendOptions::simd_cpu(3),
        ] {
            let mut eager = SimBackendInstance::new(program.clone(), options).unwrap();
            let mut batched = SimBackendInstance::new(program.clone(), options).unwrap();
            for inputs in &steps {
                eager.set_signals_replicated(inputs).unwrap();
                batched.set_signals_replicated_raw(inputs).unwrap();
                batched.eval_combinational().unwrap();
                assert_backend_signals_match(&eager, &batched, &[y, read, q_out]);

                eager.tick().unwrap();
                batched.tick_from_evaluated_no_post_eval().unwrap();
            }
            batched.eval_combinational().unwrap();
            eager.eval_combinational().unwrap();
            assert_backend_signals_match(&eager, &batched, &[y, read, q_out]);
        }

        let mut reference =
            SimBackendInstance::new(program.clone(), SimBackendOptions::scalar()).unwrap();
        let mut replay_steps = Vec::new();
        for inputs in &steps {
            reference.set_signals_replicated(inputs).unwrap();
            replay_steps.push((
                inputs.clone(),
                vec![
                    (0, y, reference.get_signal_lane(y, 0).unwrap()),
                    (1, read, reference.get_signal_lane(read, 0).unwrap()),
                    (2, q_out, reference.get_signal_lane(q_out, 0).unwrap()),
                ],
            ));
            reference.tick().unwrap();
        }
        let replay_plan =
            EncodedTraceReplayPlan::from_signal_steps(&program, replay_steps).unwrap();
        for options in [
            SimBackendOptions::scalar(),
            SimBackendOptions::packed_cpu(3),
            SimBackendOptions::simd_cpu(3),
        ] {
            let mut sim = SimBackendInstance::new(program.clone(), options).unwrap();
            let report = sim
                .replay_trace(
                    &replay_plan,
                    ReplayOptions {
                        lane_mode: ReplayLaneMode::Replicated,
                        check_mode: ReplayCheckMode::AllLanes,
                        max_mismatches: 16,
                    },
                )
                .unwrap();
            assert_eq!(report.mismatch_count, 0, "{:?}", options.kind);
            assert!(report.mismatches.is_empty());
            if options.kind == SimBackendKind::SimdCpu {
                assert!(report.simd_stats.unwrap().fast_instrs > 0);
            }
        }
    }

    #[test]
    fn bulk_replay_lane0_fast_skips_replicated_lane_validation() {
        let mut design = Design::new();
        let (q, q_out);
        {
            let mut m = design.module("Lane0Replay");
            let clk = m.input("clk", uint(1));
            q = m.reg("q", uint(8));
            q_out = m.output("q_out", uint(8));
            m.clock(q, clk);
            m.next(q, q + lit_u(1, 8));
            m.assign(q_out, q);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Lane0Replay").unwrap();
        let replay_plan = EncodedTraceReplayPlan::from_signal_steps(
            &program,
            vec![(Vec::new(), vec![(0, q_out, 0)])],
        )
        .unwrap();

        let mut lane0_fast = PackedSimulator::new(program.clone(), 2).unwrap();
        lane0_fast.set_signal(q, &[0, 99]).unwrap();
        let fast_report = lane0_fast
            .replay_trace(&replay_plan, ReplayOptions::default())
            .unwrap();
        assert_eq!(fast_report.mismatch_count, 0);

        let mut all_lanes = PackedSimulator::new(program, 2).unwrap();
        all_lanes.set_signal(q, &[0, 99]).unwrap();
        let all_report = all_lanes
            .replay_trace(
                &replay_plan,
                ReplayOptions {
                    lane_mode: ReplayLaneMode::Replicated,
                    check_mode: ReplayCheckMode::AllLanes,
                    max_mismatches: 16,
                },
            )
            .unwrap();
        assert_eq!(all_report.mismatch_count, 1);
        assert_eq!(all_report.mismatches[0].lane, Some(1));
        assert_eq!(all_report.mismatches[0].actual, 99);
    }

    #[test]
    fn independent_lane_replay_matches_distinct_lane_reference() {
        let (program, a, y, steps) = independent_replay_fixture(4);
        let plan = EncodedTraceReplayPlan::from_independent_lane_steps(&program, 4, steps).unwrap();
        assert_eq!(plan.independent_lanes(), Some(4));
        for options in [
            SimBackendOptions::packed_cpu(4),
            SimBackendOptions::simd_cpu(4),
        ] {
            let mut sim = SimBackendInstance::new(program.clone(), options).unwrap();
            let report = sim
                .replay_trace(
                    &plan,
                    ReplayOptions {
                        lane_mode: ReplayLaneMode::Independent,
                        check_mode: ReplayCheckMode::AllLanes,
                        max_mismatches: 16,
                    },
                )
                .unwrap();
            assert_eq!(report.mismatch_count, 0, "{:?}", options.kind);
        }

        let err = EncodedTraceReplayPlan::from_independent_lane_steps(
            &program,
            4,
            vec![(vec![(a, vec![1, 2])], vec![(0, y, vec![1, 2, 3, 4])])],
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SIM_IR_REPLAY_LANES");
    }

    #[test]
    fn scalar_lane_window_replay_matches_sliced_scalar_replay() {
        let (program, steps) = mixed_width_independent_replay_fixture(4);
        let plan = EncodedTraceReplayPlan::from_independent_lane_steps(&program, 4, steps).unwrap();
        let mut windowed = SingleLaneMachineSimulator::new(program.clone()).unwrap();
        let snapshot = windowed.snapshot_storage();
        for lane in 0..4 {
            let sliced_plan = plan.slice_lanes(lane, 1).unwrap();
            let mut sliced = SingleLaneMachineSimulator::new(program.clone()).unwrap();
            let sliced_report = sliced
                .replay_trace(
                    &sliced_plan,
                    ReplayOptions {
                        lane_mode: ReplayLaneMode::Independent,
                        check_mode: ReplayCheckMode::AllLanes,
                        max_mismatches: 16,
                    },
                )
                .unwrap();
            windowed.restore_storage(&snapshot).unwrap();
            let windowed_report = windowed
                .replay_independent_lane_trace(&plan, lane, 16)
                .unwrap();
            assert_eq!(windowed_report.mismatch_count, sliced_report.mismatch_count);
            assert!(windowed_report
                .mismatches
                .iter()
                .all(|mismatch| mismatch.lane == Some(lane)));
        }
    }

    #[test]
    fn scalar_lane_window_replay_reports_one_limb_and_wide_mismatches() {
        let (program, mut steps) = mixed_width_independent_replay_fixture(4);
        steps[0].1[0].2[2] ^= 1;
        steps[0].1[1].2[3] += 1;
        let plan = EncodedTraceReplayPlan::from_independent_lane_steps(&program, 4, steps).unwrap();
        let mut sim = SingleLaneMachineSimulator::new(program).unwrap();

        let one_limb = sim.replay_independent_lane_trace(&plan, 2, 16).unwrap();
        assert_eq!(one_limb.mismatch_count, 1);
        assert_eq!(one_limb.mismatches[0].lane, Some(2));
        assert_eq!(one_limb.mismatches[0].check_index, 0);

        let wide = sim.replay_independent_lane_trace(&plan, 3, 16).unwrap();
        assert_eq!(wide.mismatch_count, 1);
        assert_eq!(wide.mismatches[0].lane, Some(3));
        assert_eq!(wide.mismatches[0].check_index, 1);
    }

    #[test]
    fn independent_lane_replay_mixes_one_limb_and_wide_signals() {
        let (program, steps) = mixed_width_independent_replay_fixture(4);
        let plan = EncodedTraceReplayPlan::from_independent_lane_steps(&program, 4, steps).unwrap();
        for options in [
            SimBackendOptions::packed_cpu(4),
            SimBackendOptions::simd_cpu(4),
        ] {
            let mut sim = SimBackendInstance::new(program.clone(), options).unwrap();
            let report = sim
                .replay_trace(
                    &plan,
                    ReplayOptions {
                        lane_mode: ReplayLaneMode::Independent,
                        check_mode: ReplayCheckMode::AllLanes,
                        max_mismatches: 16,
                    },
                )
                .unwrap();
            assert_eq!(report.mismatch_count, 0, "{:?}", options.kind);
        }
    }

    #[test]
    fn independent_one_limb_replay_handles_full_limb_slice_equality() {
        let (program, steps) = full_limb_independent_replay_fixture(4);
        let plan = EncodedTraceReplayPlan::from_independent_lane_steps(&program, 4, steps).unwrap();
        for options in [
            SimBackendOptions::packed_cpu(4),
            SimBackendOptions::simd_cpu(4),
        ] {
            let mut sim = SimBackendInstance::new(program.clone(), options).unwrap();
            let report = sim
                .replay_trace(
                    &plan,
                    ReplayOptions {
                        lane_mode: ReplayLaneMode::Independent,
                        check_mode: ReplayCheckMode::AllLanes,
                        max_mismatches: 16,
                    },
                )
                .unwrap();
            assert_eq!(report.mismatch_count, 0, "{:?}", options.kind);
        }
    }

    #[test]
    fn independent_one_limb_replay_groups_adjacent_signals() {
        let (program, steps) = grouped_one_limb_independent_replay_fixture(4, 8);
        let plan = EncodedTraceReplayPlan::from_independent_lane_steps(&program, 4, steps).unwrap();
        for options in [
            SimBackendOptions::packed_cpu(4),
            SimBackendOptions::simd_cpu(4),
        ] {
            let mut sim = SimBackendInstance::new(program.clone(), options).unwrap();
            let report = sim
                .replay_trace(
                    &plan,
                    ReplayOptions {
                        lane_mode: ReplayLaneMode::Independent,
                        check_mode: ReplayCheckMode::AllLanes,
                        max_mismatches: 16,
                    },
                )
                .unwrap();
            assert_eq!(report.mismatch_count, 0, "{:?}", options.kind);
        }
    }

    #[test]
    fn independent_one_limb_replay_groups_full_limb_signals() {
        let (program, steps) = grouped_one_limb_independent_replay_fixture(4, 32);
        let plan = EncodedTraceReplayPlan::from_independent_lane_steps(&program, 4, steps).unwrap();
        let mut sim = PackedSimulator::new(program, 4).unwrap();
        let report = sim
            .replay_trace(
                &plan,
                ReplayOptions {
                    lane_mode: ReplayLaneMode::Independent,
                    check_mode: ReplayCheckMode::AllLanes,
                    max_mismatches: 16,
                },
            )
            .unwrap();
        assert_eq!(report.mismatch_count, 0);
    }

    #[test]
    fn independent_one_limb_grouped_replay_preserves_duplicate_input_order() {
        let (program, mut steps) = grouped_one_limb_independent_replay_fixture(4, 8);
        let first_a = vec![1, 1, 1, 1];
        let second_a = vec![9, 10, 11, 12];
        let b_values = steps[0].0[1].1.clone();
        let expected_y = second_a
            .iter()
            .zip(&b_values)
            .map(|(a, b)| (a + b) & 0xff)
            .collect::<Vec<_>>();
        steps[0].0 = vec![
            (steps[0].0[0].0, first_a),
            (steps[0].0[0].0, second_a),
            (steps[0].0[1].0, b_values),
        ];
        steps[0].1[0].2 = expected_y;

        let plan = EncodedTraceReplayPlan::from_independent_lane_steps(&program, 4, steps).unwrap();
        let mut sim = PackedSimulator::new(program, 4).unwrap();
        let report = sim
            .replay_trace(
                &plan,
                ReplayOptions {
                    lane_mode: ReplayLaneMode::Independent,
                    check_mode: ReplayCheckMode::AllLanes,
                    max_mismatches: 16,
                },
            )
            .unwrap();
        assert_eq!(report.mismatch_count, 0);
    }

    #[test]
    fn independent_one_limb_grouped_replay_reports_mismatch_metadata() {
        let (program, mut steps) = grouped_one_limb_independent_replay_fixture(4, 8);
        steps[0].1[1].2[3] = steps[0].1[1].2[3].wrapping_add(1);
        let expected = steps[0].1[1].2[3];
        let plan = EncodedTraceReplayPlan::from_independent_lane_steps(&program, 4, steps).unwrap();
        let mut sim = PackedSimulator::new(program, 4).unwrap();
        let report = sim
            .replay_trace(
                &plan,
                ReplayOptions {
                    lane_mode: ReplayLaneMode::Independent,
                    check_mode: ReplayCheckMode::AllLanes,
                    max_mismatches: 16,
                },
            )
            .unwrap();

        assert_eq!(report.mismatch_count, 1);
        assert_eq!(report.mismatches[0].step, 0);
        assert_eq!(report.mismatches[0].check_index, 1);
        assert_eq!(report.mismatches[0].lane, Some(3));
        assert_eq!(report.mismatches[0].expected, expected);
        assert_eq!(report.mismatches[0].actual, 23);
    }

    #[test]
    fn independent_one_limb_replay_reports_stable_mismatch_metadata() {
        let (program, mut steps) = mixed_width_independent_replay_fixture(4);
        steps[0].1[0].2[2] = steps[0].1[0].2[2].wrapping_add(1);
        let expected = steps[0].1[0].2[2];
        let plan = EncodedTraceReplayPlan::from_independent_lane_steps(&program, 4, steps).unwrap();
        let mut sim = PackedSimulator::new(program, 4).unwrap();
        let report = sim
            .replay_trace(
                &plan,
                ReplayOptions {
                    lane_mode: ReplayLaneMode::Independent,
                    check_mode: ReplayCheckMode::AllLanes,
                    max_mismatches: 16,
                },
            )
            .unwrap();

        assert_eq!(report.mismatch_count, 1);
        assert_eq!(report.mismatches[0].step, 0);
        assert_eq!(report.mismatches[0].check_index, 0);
        assert_eq!(report.mismatches[0].lane, Some(2));
        assert_eq!(report.mismatches[0].expected, expected);
        assert_eq!(report.mismatches[0].actual, 13);
    }

    #[test]
    fn threaded_replay_mixes_scalar_and_simd_workers() {
        let (program, _a, _y, steps) = independent_replay_fixture(6);
        let plan = EncodedTraceReplayPlan::from_independent_lane_steps(&program, 6, steps).unwrap();
        let report = replay_trace_threaded(
            &program,
            &plan,
            &ThreadedReplayOptions {
                workers: vec![
                    ThreadedReplayWorkerOptions {
                        backend: SimBackendKind::Scalar,
                        lanes: 2,
                    },
                    ThreadedReplayWorkerOptions {
                        backend: SimBackendKind::SimdCpu,
                        lanes: 4,
                    },
                ],
                max_mismatches: 16,
            },
        )
        .unwrap();
        assert_eq!(report.total_lanes, 6);
        assert_eq!(report.workers.len(), 2);
        assert_eq!(report.workers[0].start_lane, 0);
        assert_eq!(report.workers[1].start_lane, 2);
        assert_eq!(report.replay.mismatch_count, 0);
        assert!(report.lane_cycles_per_sec >= 0.0);
        assert!(report.replay.simd_stats.is_some());
    }

    #[test]
    fn threaded_replay_reports_global_mismatch_lanes_deterministically() {
        let (program, _a, y, mut steps) = independent_replay_fixture(4);
        steps[1].1[0].2[3] = steps[1].1[0].2[3].wrapping_add(1);
        let plan = EncodedTraceReplayPlan::from_independent_lane_steps(&program, 4, steps).unwrap();
        let report = replay_trace_threaded(
            &program,
            &plan,
            &ThreadedReplayOptions {
                workers: vec![
                    ThreadedReplayWorkerOptions {
                        backend: SimBackendKind::PackedCpu,
                        lanes: 2,
                    },
                    ThreadedReplayWorkerOptions {
                        backend: SimBackendKind::PackedCpu,
                        lanes: 2,
                    },
                ],
                max_mismatches: 16,
            },
        )
        .unwrap();
        assert_eq!(report.replay.mismatch_count, 1);
        assert_eq!(report.replay.mismatches[0].step, 1);
        assert_eq!(report.replay.mismatches[0].check_index, 0);
        assert_eq!(report.replay.mismatches[0].lane, Some(3));
        assert_eq!(report.workers[1].replay.mismatches[0].lane, Some(3));

        let bad = EncodedTraceReplayPlan::from_independent_lane_steps(
            &program,
            4,
            vec![(Vec::new(), vec![(0, y, vec![0, 0, 0, 0])])],
        )
        .unwrap();
        assert!(replay_trace_threaded(
            &program,
            &bad,
            &ThreadedReplayOptions {
                workers: Vec::new(),
                max_mismatches: 16,
            },
        )
        .is_err());

        let err = replay_trace_threaded(
            &program,
            &plan,
            &ThreadedReplayOptions {
                workers: vec![ThreadedReplayWorkerOptions {
                    backend: SimBackendKind::PackedCpu,
                    lanes: 3,
                }],
                max_mismatches: 16,
            },
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SIM_IR_THREADED_REPLAY_LANES");

        let replicated =
            EncodedTraceReplayPlan::from_signal_steps(&program, vec![(vec![], vec![(0, y, 0)])])
                .unwrap();
        let err = replay_trace_threaded(
            &program,
            &replicated,
            &ThreadedReplayOptions {
                workers: vec![ThreadedReplayWorkerOptions {
                    backend: SimBackendKind::PackedCpu,
                    lanes: 1,
                }],
                max_mismatches: 16,
            },
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SIM_IR_THREADED_REPLAY_LANES");
    }

    #[test]
    fn replay_autotune_selects_successful_candidate() {
        let (program, _a, _y, steps) = independent_replay_fixture(4);
        let plan = EncodedTraceReplayPlan::from_independent_lane_steps(&program, 4, steps).unwrap();
        let report = replay_trace_autotune(
            &program,
            &plan,
            4,
            ReplayAutotuneOptions {
                warmup_steps: 2,
                max_workers: 2,
                candidates: vec![
                    ThreadedReplayOptions {
                        workers: vec![ThreadedReplayWorkerOptions {
                            backend: SimBackendKind::Scalar,
                            lanes: 4,
                        }],
                        max_mismatches: 16,
                    },
                    ThreadedReplayOptions {
                        workers: vec![ThreadedReplayWorkerOptions {
                            backend: SimBackendKind::PackedCpu,
                            lanes: 4,
                        }],
                        max_mismatches: 16,
                    },
                ],
                simd_suitability: None,
                backend_affinity: None,
            },
        )
        .unwrap();
        assert_eq!(report.candidates.len(), 2);
        assert!(report
            .candidates
            .iter()
            .all(|candidate| candidate.available));
        assert_eq!(report.selected.replay.mismatch_count, 0);
    }

    #[test]
    fn replay_autotune_candidate_set_prunes_for_scalar_preferred() {
        let suitability =
            test_simd_suitability_report(SimdSuitabilityRecommendation::ScalarPreferred);
        let set = build_replay_autotune_candidate_set(8, 8, Some(&suitability));
        assert_eq!(set.candidates.len(), 3);
        assert!(set.candidates.iter().all(|candidate| candidate
            .workers
            .iter()
            .all(|worker| worker.backend == SimBackendKind::Scalar)));
        assert_eq!(
            set.candidates
                .iter()
                .map(|candidate| candidate.workers.len())
                .collect::<Vec<_>>(),
            vec![1, 4, 8]
        );
        assert_eq!(set.pruned_candidates.len(), 3);
        assert!(set
            .pruned_candidates
            .iter()
            .all(|candidate| { candidate.reason == ReplayAutotunePruneReason::ScalarPreferred }));
    }

    #[test]
    fn replay_autotune_candidate_set_prunes_with_backend_affinity() {
        let affinity = test_backend_affinity_report(BackendAffinityRecommendation::ScalarPreferred);
        let set = build_replay_autotune_candidate_set_with_affinity(8, 8, None, Some(&affinity));
        assert_eq!(set.candidates.len(), 3);
        assert!(set.candidates.iter().all(|candidate| candidate
            .workers
            .iter()
            .all(|worker| worker.backend == SimBackendKind::Scalar)));
        assert_eq!(set.pruned_candidates.len(), 3);
        assert!(set
            .pruned_candidates
            .iter()
            .all(|candidate| candidate.reason == ReplayAutotunePruneReason::ScalarPreferred));
    }

    #[test]
    fn replay_autotune_candidate_set_keeps_mixed_with_backend_affinity() {
        let affinity =
            test_backend_affinity_report(BackendAffinityRecommendation::MixedScalarSimdCandidate);
        let set = build_replay_autotune_candidate_set_with_affinity(8, 8, None, Some(&affinity));
        assert_eq!(set.candidates.len(), 4);
        assert!(set.candidates.iter().any(|candidate| {
            candidate
                .workers
                .iter()
                .all(|worker| worker.backend == SimBackendKind::Scalar)
        }));
        assert!(set.candidates.iter().any(|candidate| {
            candidate
                .workers
                .iter()
                .any(|worker| worker.backend == SimBackendKind::Scalar)
                && candidate
                    .workers
                    .iter()
                    .any(|worker| worker.backend == SimBackendKind::SimdCpu)
        }));
        assert_eq!(set.pruned_candidates.len(), 2);
        assert!(set
            .pruned_candidates
            .iter()
            .all(|candidate| candidate.reason == ReplayAutotunePruneReason::MixedOnly));
    }

    #[test]
    fn replay_autotune_candidate_set_keeps_packed_with_backend_affinity() {
        let affinity =
            test_backend_affinity_report(BackendAffinityRecommendation::PackedCpuCandidate);
        let set = build_replay_autotune_candidate_set_with_affinity(8, 8, None, Some(&affinity));
        assert_eq!(set.candidates.len(), 4);
        assert!(set.candidates.iter().any(|candidate| {
            candidate
                .workers
                .iter()
                .all(|worker| worker.backend == SimBackendKind::PackedCpu)
        }));
        assert_eq!(set.pruned_candidates.len(), 2);
        assert!(set
            .pruned_candidates
            .iter()
            .all(|candidate| candidate.reason == ReplayAutotunePruneReason::PackedCpuPreferred));
    }

    #[test]
    fn replay_autotune_candidate_set_keeps_simd_candidates() {
        let suitability =
            test_simd_suitability_report(SimdSuitabilityRecommendation::SimdCandidate);
        let set = build_replay_autotune_candidate_set(8, 8, Some(&suitability));
        assert_eq!(set.candidates.len(), 6);
        assert!(set.pruned_candidates.is_empty());
    }

    #[test]
    fn replay_autotune_candidate_set_keeps_only_scalar_and_mixed_for_mixed_candidate() {
        let suitability =
            test_simd_suitability_report(SimdSuitabilityRecommendation::MixedCandidate);
        let set = build_replay_autotune_candidate_set(8, 8, Some(&suitability));
        assert_eq!(set.candidates.len(), 4);
        assert!(set.candidates.iter().any(|candidate| {
            candidate
                .workers
                .iter()
                .all(|worker| worker.backend == SimBackendKind::Scalar)
        }));
        assert!(set.candidates.iter().any(|candidate| {
            candidate
                .workers
                .iter()
                .any(|worker| worker.backend == SimBackendKind::Scalar)
                && candidate
                    .workers
                    .iter()
                    .any(|worker| worker.backend == SimBackendKind::SimdCpu)
        }));
        assert_eq!(set.pruned_candidates.len(), 2);
        assert!(set
            .pruned_candidates
            .iter()
            .all(|candidate| candidate.reason == ReplayAutotunePruneReason::MixedOnly));
    }

    #[test]
    fn replay_autotune_candidate_set_prunes_for_gpu_blocked() {
        let suitability =
            test_simd_suitability_report(SimdSuitabilityRecommendation::GpuCandidateBlocked);
        let set = build_replay_autotune_candidate_set(8, 8, Some(&suitability));
        assert_eq!(set.candidates.len(), 3);
        assert!(set.candidates.iter().all(|candidate| candidate
            .workers
            .iter()
            .all(|worker| worker.backend == SimBackendKind::Scalar)));
        assert_eq!(set.pruned_candidates.len(), 3);
        assert!(set.pruned_candidates.iter().all(|candidate| {
            candidate.reason == ReplayAutotunePruneReason::GpuCandidateBlocked
        }));
    }

    fn test_simd_suitability_report(
        recommendation: SimdSuitabilityRecommendation,
    ) -> SimdSuitabilityReport {
        SimdSuitabilityReport {
            streams: SimdSuitabilityStreamsReport {
                async_reset_comb: SimdSuitabilityBlockReport::default(),
                comb: SimdSuitabilityBlockReport::default(),
                tick_next: SimdSuitabilityBlockReport::default(),
                tick_commit: SimdSuitabilityBlockReport::default(),
            },
            total: SimdSuitabilityBlockReport::default(),
            recommendation,
            score_x100: 0,
            estimated_fast_cost: 0,
            estimated_fallback_cost: 0,
            estimated_materialization_cost: 0,
            fallback_ratio_x100: 0,
        }
    }

    fn test_backend_affinity_report(
        recommendation: BackendAffinityRecommendation,
    ) -> BackendAffinityReport {
        BackendAffinityReport {
            streams: BackendAffinityStreamsReport {
                async_reset_comb: BackendAffinityBlockReport::default(),
                comb: BackendAffinityBlockReport::default(),
                tick_next: BackendAffinityBlockReport::default(),
                tick_commit: BackendAffinityBlockReport::default(),
            },
            total: BackendAffinityBlockReport::default(),
            recommendation,
            reasons: Vec::new(),
        }
    }

    type IndependentReplayFixture = (
        PackedProgram,
        Signal,
        Signal,
        Vec<(Vec<(Signal, Vec<u128>)>, Vec<(usize, Signal, Vec<u128>)>)>,
    );

    fn independent_replay_fixture(lanes: usize) -> IndependentReplayFixture {
        let mut design = Design::new();
        let (a, y);
        {
            let mut m = design.module("IndependentReplay");
            let clk = m.input("clk", uint(1));
            a = m.input("a", uint(8));
            let q = m.reg("q", uint(8));
            y = m.output("y", uint(8));
            m.clock(q, clk);
            m.next(q, q + a);
            m.assign(y, q + a);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "IndependentReplay").unwrap();
        let mut reference = PackedSimulator::new(program.clone(), lanes).unwrap();
        let mut steps = Vec::new();
        for step in 0..4 {
            let values = (0..lanes)
                .map(|lane| (step as u128 * 3 + lane as u128 + 1) & 0xff)
                .collect::<Vec<_>>();
            reference.set_signal(a, &values).unwrap();
            let expected = reference.get_signal(y).unwrap();
            steps.push((vec![(a, values)], vec![(0, y, expected)]));
            reference.tick();
        }
        (program, a, y, steps)
    }

    fn mixed_width_independent_replay_fixture(
        lanes: usize,
    ) -> (
        PackedProgram,
        Vec<(Vec<(Signal, Vec<u128>)>, Vec<(usize, Signal, Vec<u128>)>)>,
    ) {
        let mut design = Design::new();
        let (a, wide, y, wide_y);
        {
            let mut m = design.module("MixedWidthIndependentReplay");
            a = m.input("a", uint(8));
            wide = m.input("wide", uint(40));
            y = m.output("y", uint(8));
            wide_y = m.output("wide_y", uint(40));
            m.assign(y, a);
            m.assign(wide_y, wide + lit_u(5, 40));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "MixedWidthIndependentReplay").unwrap();
        let a_values = (0..lanes).map(|lane| 11 + lane as u128).collect::<Vec<_>>();
        let wide_values = (0..lanes)
            .map(|lane| (1u128 << 32) + 17 + lane as u128)
            .collect::<Vec<_>>();
        let wide_expected = wide_values
            .iter()
            .copied()
            .map(|value| value + 5)
            .collect::<Vec<_>>();
        (
            program,
            vec![(
                vec![(a, a_values.clone()), (wide, wide_values)],
                vec![(0, y, a_values), (1, wide_y, wide_expected)],
            )],
        )
    }

    fn full_limb_independent_replay_fixture(
        lanes: usize,
    ) -> (
        PackedProgram,
        Vec<(Vec<(Signal, Vec<u128>)>, Vec<(usize, Signal, Vec<u128>)>)>,
    ) {
        let mut design = Design::new();
        let (a, y);
        {
            let mut m = design.module("FullLimbIndependentReplay");
            a = m.input("a", uint(32));
            y = m.output("y", uint(32));
            m.assign(y, a ^ lit_u(0xa5a5_5a5a, 32));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "FullLimbIndependentReplay").unwrap();
        let values = (0..lanes)
            .map(|lane| 0xffff_ff00u128 + lane as u128)
            .collect::<Vec<_>>();
        let expected = values
            .iter()
            .copied()
            .map(|value| value ^ 0xa5a5_5a5a)
            .collect::<Vec<_>>();
        (program, vec![(vec![(a, values)], vec![(0, y, expected)])])
    }

    fn grouped_one_limb_independent_replay_fixture(
        lanes: usize,
        width: u32,
    ) -> (
        PackedProgram,
        Vec<(Vec<(Signal, Vec<u128>)>, Vec<(usize, Signal, Vec<u128>)>)>,
    ) {
        let mut design = Design::new();
        let (a, b, y, z);
        {
            let mut m = design.module("GroupedOneLimbIndependentReplay");
            a = m.input("a", uint(width));
            b = m.input("b", uint(width));
            y = m.output("y", uint(width));
            z = m.output("z", uint(width));
            m.assign(y, a + b);
            m.assign(z, b ^ lit_u(3, width));
        }
        let compiled = compile(&design).unwrap();
        let program =
            lower_to_packed_program(&compiled, "GroupedOneLimbIndependentReplay").unwrap();
        let mask = if width == 32 {
            u128::from(u32::MAX)
        } else {
            (1u128 << width) - 1
        };
        let a_values = (0..lanes)
            .map(|lane| (11 + lane as u128) & mask)
            .collect::<Vec<_>>();
        let b_values = (0..lanes)
            .map(|lane| (17 + lane as u128) & mask)
            .collect::<Vec<_>>();
        let y_expected = a_values
            .iter()
            .zip(&b_values)
            .map(|(a, b)| (a + b) & mask)
            .collect::<Vec<_>>();
        let z_expected = b_values
            .iter()
            .copied()
            .map(|value| (value ^ 3) & mask)
            .collect::<Vec<_>>();
        (
            program,
            vec![(
                vec![(a, a_values), (b, b_values)],
                vec![(0, y, y_expected), (1, z, z_expected)],
            )],
        )
    }

    fn assert_backend_signals_match(
        expected: &SimBackendInstance,
        actual: &SimBackendInstance,
        signals: &[Signal],
    ) {
        assert_eq!(expected.lanes(), actual.lanes());
        for signal in signals {
            for lane in 0..expected.lanes() {
                assert_eq!(
                    actual.get_signal_lane(*signal, lane).unwrap(),
                    expected.get_signal_lane(*signal, lane).unwrap()
                );
            }
        }
    }

    #[test]
    fn single_lane_machine_matches_scalar_for_arithmetic_and_registers() {
        let mut design = Design::new();
        let (a, b, en, y, count);
        {
            let mut m = design.module("SingleLane");
            let clk = m.input("clk", uint(1));
            a = m.input("a", uint(8));
            b = m.input("b", uint(8));
            en = m.input("en", uint(1));
            y = m.output("y", uint(8));
            count = m.reg("count", uint(8));
            m.assign(y, ((a + b) ^ (a * b)).trunc(8));
            m.clock(count, clk);
            m.next(count, mux(en, count + lit_u(1, 8), count));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "SingleLane").unwrap();
        let mut single = SingleLaneMachineSimulator::new(program).unwrap();
        let mut scalar = Simulator::new(&design, "SingleLane").unwrap();

        for (signal, value) in [(a, 7), (b, 13), (en, 1)] {
            single.set_signal(signal, value).unwrap();
            scalar.set(signal, value);
        }
        assert_eq!(single.get_signal(y).unwrap(), scalar.get(y));

        single.tick_many(3);
        for _ in 0..3 {
            scalar.tick();
        }
        assert_eq!(single.get_signal(count).unwrap(), scalar.get(count));
    }

    #[test]
    fn single_lane_machine_reuses_workspaces_across_eval_and_ticks() {
        let mut design = Design::new();
        let (a, b, en, y, count);
        {
            let mut m = design.module("SingleLaneWorkspaceReuse");
            let clk = m.input("clk", uint(1));
            a = m.input("a", uint(8));
            b = m.input("b", uint(8));
            en = m.input("en", uint(1));
            y = m.output("y", uint(8));
            count = m.reg("count", uint(8));
            m.clock(count, clk);
            m.next(count, mux(en, count + a + b, count));
            m.assign(y, ((count ^ a) + (b * lit_u(3, 8))).trunc(8));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "SingleLaneWorkspaceReuse").unwrap();
        let mut single = SingleLaneMachineSimulator::new(program).unwrap();
        let mut scalar = Simulator::new(&design, "SingleLaneWorkspaceReuse").unwrap();

        for step in 0..8u128 {
            let values = [
                (a, (step * 7 + 3) & 0xff),
                (b, (step * 5 + 11) & 0xff),
                (en, u128::from(step % 3 != 0)),
            ];
            for (signal, value) in values {
                single.set_signal(signal, value).unwrap();
                scalar.set(signal, value);
            }
            for _ in 0..3 {
                single.eval_combinational();
            }
            assert_eq!(single.get_signal(y).unwrap(), scalar.get(y));
            if step % 2 == 0 {
                single.tick();
                scalar.tick();
            } else {
                single.tick_many(1);
                scalar.tick();
            }
            assert_eq!(single.get_signal(count).unwrap(), scalar.get(count));
        }
    }

    #[test]
    fn single_lane_compiled_ops_match_reference_simulator() {
        let mut design = Design::new();
        let (a, b, s, sel, y_not, y_and, y_or, y_xor, y_add, y_sub, y_mul, y_eq, y_ne, y_lt);
        let (y_slt, y_mux, y_slice, y_concat, y_sext, y_zext, y_trunc, y_cast);
        {
            let mut m = design.module("SingleLaneCompiledOps");
            a = m.input("a", uint(8));
            b = m.input("b", uint(8));
            s = m.input("s", sint(4));
            sel = m.input("sel", uint(1));
            y_not = m.output("y_not", uint(8));
            y_and = m.output("y_and", uint(8));
            y_or = m.output("y_or", uint(8));
            y_xor = m.output("y_xor", uint(8));
            y_add = m.output("y_add", uint(8));
            y_sub = m.output("y_sub", uint(8));
            y_mul = m.output("y_mul", uint(8));
            y_eq = m.output("y_eq", uint(1));
            y_ne = m.output("y_ne", uint(1));
            y_lt = m.output("y_lt", uint(1));
            y_slt = m.output("y_slt", uint(1));
            y_mux = m.output("y_mux", uint(8));
            y_slice = m.output("y_slice", uint(3));
            y_concat = m.output("y_concat", uint(16));
            y_sext = m.output("y_sext", sint(8));
            y_zext = m.output("y_zext", uint(12));
            y_trunc = m.output("y_trunc", uint(4));
            y_cast = m.output("y_cast", uint(8));

            m.assign(y_not, !a.value());
            m.assign(y_and, a & b);
            m.assign(y_or, a | b);
            m.assign(y_xor, a ^ b);
            m.assign(y_add, a + b);
            m.assign(y_sub, a - b);
            m.assign(y_mul, a * b);
            m.assign(y_eq, a.value().eq_expr(b));
            m.assign(y_ne, a.value().ne_expr(b));
            m.assign(y_lt, a.value().lt_expr(b));
            m.assign(y_slt, s.value().lt_expr(lit_s(-1, 4)));
            m.assign(y_mux, mux(sel, a, b));
            m.assign(y_slice, a.value().slice(2, 3));
            m.assign(y_concat, concat([a.value(), b.value()]));
            m.assign(y_sext, sext(s, 8));
            m.assign(y_zext, a.value().zext(12));
            m.assign(y_trunc, a.value().trunc(4));
            m.assign(y_cast, s.value().as_uint().zext(8));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "SingleLaneCompiledOps").unwrap();
        let mut single = SingleLaneMachineSimulator::new(program).unwrap();
        let mut reference = Simulator::new(&design, "SingleLaneCompiledOps").unwrap();
        let outputs = [
            y_not, y_and, y_or, y_xor, y_add, y_sub, y_mul, y_eq, y_ne, y_lt, y_slt, y_mux,
            y_slice, y_concat, y_sext, y_zext, y_trunc, y_cast,
        ];

        for (a_value, b_value, s_value, sel_value) in [
            (0x12, 0x34, 0x7, 0),
            (0xfe, 0x03, 0x8, 1),
            (0x55, 0x55, 0xf, 1),
        ] {
            for (signal, value) in [(a, a_value), (b, b_value), (s, s_value), (sel, sel_value)] {
                single.set_signal(signal, value).unwrap();
                reference.set(signal, value);
            }
            single.eval_combinational();
            for output in outputs {
                assert_eq!(
                    single.get_signal(output).unwrap(),
                    reference.get(output),
                    "output {:?}",
                    output.id
                );
            }
        }
    }

    #[test]
    fn single_lane_machine_handles_memory_and_tick_many() {
        let mut design = Design::new();
        let (we, addr, data, read);
        let mem;
        {
            let mut m = design.module("SingleLaneMem");
            let clk = m.input("clk", uint(1));
            we = m.input("we", uint(1));
            addr = m.input("addr", uint(2));
            data = m.input("data", uint(8));
            mem = m.mem("mem", 2, uint(8), 4);
            read = m.output("read", uint(8));
            let read_expr = m.mem_read(mem, addr);
            m.assign(read, read_expr);
            m.mem_write(mem, clk, we, addr, data);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "SingleLaneMem").unwrap();
        let mut repeated = SingleLaneMachineSimulator::new(program.clone()).unwrap();
        let mut many = SingleLaneMachineSimulator::new(program).unwrap();

        repeated.set_memory(mem, &[1, 2, 3, 4]).unwrap();
        many.set_memory(mem, &[1, 2, 3, 4]).unwrap();
        for sim in [&mut repeated, &mut many] {
            sim.set_signals(&[(we, 1), (addr, 2), (data, 9)]).unwrap();
        }
        repeated.tick();
        many.tick_many(1);
        for sim in [&mut repeated, &mut many] {
            sim.set_signals(&[(we, 0), (addr, 2)]).unwrap();
        }
        assert_eq!(
            many.get_signal(read).unwrap(),
            repeated.get_signal(read).unwrap()
        );
        assert_eq!(many.get_signal(read).unwrap(), 9);
    }

    #[test]
    fn single_lane_machine_rejects_values_wider_than_u128() {
        let mut design = Design::new();
        {
            let mut m = design.module("TooWide");
            let a = m.input("a", uint(129));
            let y = m.output("y", uint(129));
            m.assign(y, a);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "TooWide").unwrap();
        let err = SingleLaneMachineSimulator::new(program).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SIM_IR_WIDE_VALUE");
    }

    #[test]
    fn packed_memory_set_and_get_apis_work() {
        let mut design = Design::new();
        let mem;
        {
            let mut m = design.module("MemApi");
            mem = m.mem("mem", 2, uint(8), 4);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "MemApi").unwrap();
        let mut packed = PackedSimulator::new(program, 2).unwrap();

        packed
            .set_memory(mem, &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]])
            .unwrap();
        assert_eq!(
            packed.get_memory(mem).unwrap(),
            vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]]
        );
        assert_eq!(
            packed.get_memory_limbs(mem).unwrap(),
            vec![
                vec![vec![1], vec![2], vec![3], vec![4]],
                vec![vec![5], vec![6], vec![7], vec![8]]
            ]
        );
    }

    #[test]
    fn packed_memory_limb_apis_support_wide_words() {
        let mut design = Design::new();
        let mem;
        {
            let mut m = design.module("WideMem");
            mem = m.mem("mem", 1, uint(160), 2);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "WideMem").unwrap();
        let mut packed = PackedSimulator::new(program, 1).unwrap();

        let values = vec![vec![vec![1, 2, 3, 4, 0xffff_ffff], vec![5, 6, 7, 8, 9]]];
        packed.set_memory_limbs(mem, &values).unwrap();
        assert_eq!(packed.get_memory_limbs(mem).unwrap(), values);

        let err = packed.set_memory(mem, &[vec![0, 1]]).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SIM_IR_WIDE_MEMORY");
        let err = packed.get_memory(mem).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SIM_IR_WIDE_MEMORY");
    }

    #[test]
    fn packed_memory_apis_validate_shape_and_memory_signal() {
        let mut design = Design::new();
        let (not_memory, mem);
        {
            let mut m = design.module("MemShape");
            not_memory = m.input("not_memory", uint(1));
            mem = m.mem("mem", 2, uint(8), 4);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "MemShape").unwrap();
        let mut packed = PackedSimulator::new(program, 2).unwrap();

        let err = packed.set_memory(mem, &[vec![1, 2, 3, 4]]).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SIM_IR_MEMORY_VALUES");

        let err = packed
            .set_memory(mem, &[vec![1, 2, 3], vec![4, 5, 6]])
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SIM_IR_MEMORY_VALUES");

        let err = packed
            .set_memory_limbs(
                mem,
                &[
                    vec![vec![1, 0], vec![2, 0], vec![3, 0], vec![4, 0]],
                    vec![vec![5, 0], vec![6, 0], vec![7, 0], vec![8, 0]],
                ],
            )
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SIM_IR_MEMORY_VALUES");

        let err = packed.get_memory(not_memory).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SIM_IR_MEMORY");
    }

    #[test]
    fn packed_sim_matches_cpu_for_flattened_hierarchy() {
        let mut design = Design::new();
        {
            let mut m = design.module("Child");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.assign(y, a + lit_u(1, 8));
        }
        let (a, y);
        {
            let mut m = design.module("Top");
            a = m.input("a", uint(8));
            let mid = m.wire("mid", uint(8));
            y = m.output("y", uint(8));
            m.instance("u0", "Child", [("a", a), ("y", mid)]);
            m.instance("u1", "Child", [("a", mid), ("y", y)]);
        }

        assert_packed_matches_cpu(&design, "Top", &[(a, vec![0, 1, 254, 255])], &[y]);
    }

    #[test]
    fn slice_ownership_assigns_signals_by_instance_path() {
        let mut design = Design::new();
        {
            let mut m = design.module("Child");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.assign(y, a + lit_u(1, 8));
        }
        {
            let mut m = design.module("Top");
            let a = m.input("a", uint(8));
            let mid = m.wire("mid", uint(8));
            let y = m.output("y", uint(8));
            m.instance("u0", "Child", [("a", a), ("y", mid)]);
            m.instance("u1", "Child", [("a", mid), ("y", y)]);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();

        let groups = vec![
            PackedSliceGroup {
                id: "top".into(),
                owned_paths: vec!["Top".into()],
            },
            PackedSliceGroup {
                id: "g0".into(),
                owned_paths: vec!["Top.u0".into()],
            },
            PackedSliceGroup {
                id: "g1".into(),
                owned_paths: vec!["Top.u1".into()],
            },
        ];
        let owners = classify_signal_owners(&program, &groups);

        // Every signal is claimed by the group matching its owner_path, with the
        // most-specific (longest) context winning over the "Top" group.
        for (signal, owner) in program.signals.iter().zip(&owners) {
            let expected = match signal.owner_path.as_str() {
                "Top" => 0,
                "Top.u0" => 1,
                "Top.u1" => 2,
                other => panic!("unexpected owner_path {other} for {}", signal.name),
            };
            assert_eq!(
                *owner,
                Some(expected),
                "signal {} (owner_path {})",
                signal.name,
                signal.owner_path
            );
        }
        assert!(owners.iter().any(|owner| *owner == Some(1)));
        assert!(owners.iter().any(|owner| *owner == Some(2)));
    }

    #[test]
    fn slice_ownership_uses_component_wise_prefix() {
        let mut design = Design::new();
        {
            let mut m = design.module("Child");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.assign(y, a + lit_u(1, 8));
        }
        {
            let mut m = design.module("Top");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.instance("u0", "Child", [("a", a), ("y", y)]);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();

        // "Top.u" is not a component-prefix of "Top.u0" (nor of "Top"), so this
        // group must claim nothing — guards against substring-style matching.
        let groups = vec![PackedSliceGroup {
            id: "x".into(),
            owned_paths: vec!["Top.u".into()],
        }];
        let owners = classify_signal_owners(&program, &groups);
        assert!(
            owners.iter().all(|owner| owner.is_none()),
            "Top.u must not falsely prefix-match Top.u0",
        );
    }

    fn program_op_count(program: &PackedProgram) -> usize {
        [
            &program.streams.async_reset_comb,
            &program.streams.comb,
            &program.streams.tick_next,
            &program.streams.tick_commit,
        ]
        .iter()
        .flat_map(|stream| stream.iter())
        .map(|packet| packet.ops.len())
        .sum()
    }

    fn hierarchy_top_design() -> (Design, Signal, Signal) {
        let mut design = Design::new();
        {
            let mut m = design.module("Child");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.assign(y, a + lit_u(1, 8));
        }
        let (a, y);
        {
            let mut m = design.module("Top");
            a = m.input("a", uint(8));
            let mid = m.wire("mid", uint(8));
            y = m.output("y", uint(8));
            m.instance("u0", "Child", [("a", a), ("y", mid)]);
            m.instance("u1", "Child", [("a", mid), ("y", y)]);
        }
        (design, a, y)
    }

    #[test]
    fn slice_single_covering_group_simulates_identically() {
        let (design, a, y) = hierarchy_top_design();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();

        let groups = vec![PackedSliceGroup {
            id: "all".into(),
            owned_paths: vec!["Top".into()],
        }];
        let slices = slice_packed_program(&program, &groups).unwrap();
        assert_eq!(slices.len(), 1);
        let slice = &slices[0];
        // A group that owns the whole hierarchy reads nothing from elsewhere.
        assert!(slice.boundary_inputs.is_empty());
        assert_eq!(
            slice.signal_origin,
            (0..program.signals.len()).collect::<Vec<_>>()
        );
        assert_eq!(program_op_count(&slice.program), program_op_count(&program));

        // Behavioral identity across all lanes.
        let inputs: Vec<u128> = vec![0, 1, 254, 255];
        let mut original = PackedSimulator::new(program.clone(), inputs.len()).unwrap();
        let mut sliced = PackedSimulator::new(slice.program.clone(), inputs.len()).unwrap();
        original.set_signal(a, &inputs).unwrap();
        sliced.set_signal(a, &inputs).unwrap();
        original.eval_combinational();
        sliced.eval_combinational();
        assert_eq!(
            original.get_signal(y).unwrap(),
            sliced.get_signal(y).unwrap()
        );
    }

    #[test]
    fn slice_three_disjoint_groups_preserve_all_ops_and_find_boundaries() {
        let (design, _, _) = hierarchy_top_design();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();

        let groups = vec![
            PackedSliceGroup {
                id: "top".into(),
                owned_paths: vec!["Top".into()],
            },
            PackedSliceGroup {
                id: "u0".into(),
                owned_paths: vec!["Top.u0".into()],
            },
            PackedSliceGroup {
                id: "u1".into(),
                owned_paths: vec!["Top.u1".into()],
            },
        ];
        let slices = slice_packed_program(&program, &groups).unwrap();
        assert_eq!(slices.len(), 3);

        // Disjoint covering groups: every op lands in exactly one slice.
        let total: usize = slices.iter().map(|s| program_op_count(&s.program)).sum();
        assert_eq!(total, program_op_count(&program));

        // The child group consumes signals owned elsewhere, surfaced as Inputs.
        let u0 = &slices[1];
        assert!(!u0.boundary_inputs.is_empty());
        for &boundary in &u0.boundary_inputs {
            assert_eq!(u0.program.signals[boundary].kind, PackedSignalKind::Input);
            assert!(u0.signal_origin[boundary] < program.signals.len());
        }
    }

    #[test]
    fn activity_skip_matches_oracle_and_skips_idle() {
        // A free-running counter (always active) plus an 8-deep gated shift chain
        // (idle whenever the enable is low and data is held). Activity-skipped
        // ticks must be bit-identical to the whole-design oracle, and must skip a
        // large fraction of group-ticks while the chain is idle.
        let mut design = Design::new();
        let (clk, en, din, octr, o);
        {
            let mut m = design.module("Chain");
            clk = m.input("clk", uint(1));
            en = m.input("en", uint(1));
            din = m.input("din", uint(8));
            let ctr = m.reg("ctr", uint(8));
            m.clock(ctr, clk);
            m.next(ctr, ctr.value() + lit_u(1, 8));
            octr = m.output("octr", uint(8));
            m.assign(octr, ctr);
            let mut prev = din;
            for i in 0..8 {
                let s = m.reg(format!("s{i}"), uint(8));
                m.clock(s, clk);
                m.next(s, mux(en, prev.value(), s.value()));
                prev = s;
            }
            o = m.output("o", uint(8));
            m.assign(o, prev);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Chain").unwrap();

        let lanes = 4;
        let mut oracle = PackedSimulator::new(program.clone(), lanes).unwrap();
        let mut act = PartitionedSimulator::new_register_balanced(&program, 4, lanes).unwrap();
        assert!(act.group_count() >= 2);

        let clk_v = vec![1u128; lanes];
        oracle.set_signal(clk, &clk_v).unwrap();
        act.set_signal(clk, &clk_v).unwrap();

        let mut held = vec![0u128; lanes];
        for cycle in 0..60u32 {
            // enable pulses 1-of-8; data only changes while enabled (else held).
            let enable = (cycle % 8 == 0) as u128;
            let en_v = vec![enable; lanes];
            if enable != 0 {
                held = (0..lanes as u128).map(|l| (l * 17 + cycle as u128) & 0xff).collect();
            }
            for s in [&mut oracle] {
                s.set_signal(en, &en_v).unwrap();
                s.set_signal(din, &held).unwrap();
            }
            act.set_signal(en, &en_v).unwrap();
            act.set_signal(din, &held).unwrap();

            oracle.tick();
            act.tick_activity();

            assert_eq!(oracle.get_signal(octr).unwrap(), act.get_signal(octr).unwrap(), "octr cycle {cycle}");
            assert_eq!(oracle.get_signal(o).unwrap(), act.get_signal(o).unwrap(), "o cycle {cycle}");
        }
        // The counter group is always active; the 8 chain groups idle ~7/8 of the
        // time, so overall skip rate should be substantial.
        assert!(
            act.activity_skip_rate() > 0.3,
            "expected substantial skipping, got {:.3}",
            act.activity_skip_rate()
        );
    }

    #[test]
    fn partitioned_simulator_matches_whole_design_oracle() {
        // Sequential hierarchy: din -> u0(+1) -> register r -> u1(+1) -> dout.
        // Cross-group edges: u0 reads din (top input, stable); top reads u0.y
        // (comb); u1 reads r (register, stable); top reads u1.y (comb). The comb
        // edges (top depends on u0 and u1) are acyclic; the register edge breaks
        // what would otherwise be a top<->u1 cycle.
        let mut design = Design::new();
        {
            let mut m = design.module("Inc");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.assign(y, a + lit_u(1, 8));
        }
        let (clk, din, dout);
        {
            let mut m = design.module("Top");
            clk = m.input("clk", uint(1));
            din = m.input("din", uint(8));
            let w0 = m.wire("w0", uint(8));
            let r = m.reg("r", uint(8));
            dout = m.output("dout", uint(8));
            m.clock(r, clk);
            m.next(r, w0);
            m.instance("u0", "Inc", [("a", din), ("y", w0)]);
            m.instance("u1", "Inc", [("a", r), ("y", dout)]);
        }

        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();

        let lanes = 4;
        let groups = vec![
            PackedSliceGroup {
                id: "top".into(),
                owned_paths: vec!["Top".into()],
            },
            PackedSliceGroup {
                id: "u0".into(),
                owned_paths: vec!["Top.u0".into()],
            },
            PackedSliceGroup {
                id: "u1".into(),
                owned_paths: vec!["Top.u1".into()],
            },
        ];

        let mut oracle = PackedSimulator::new(program.clone(), lanes).unwrap();
        let mut partitioned = PartitionedSimulator::new(&program, &groups, lanes).unwrap();
        // Three groups, so the slicer found a real multi-way partition.
        assert_eq!(partitioned.group_count(), 3);

        let clk_lanes = vec![1u128; lanes];
        oracle.set_signal(clk, &clk_lanes).unwrap();
        partitioned.set_signal(clk, &clk_lanes).unwrap();

        // Drive distinct per-lane stimulus for several cycles; compare the
        // registered output bit-exactly after each tick.
        for cycle in 0..6u128 {
            let din_lanes: Vec<u128> = (0..lanes as u128)
                .map(|lane| (cycle.wrapping_mul(37).wrapping_add(lane * 11)) & 0xff)
                .collect();
            oracle.set_signal(din, &din_lanes).unwrap();
            partitioned.set_signal(din, &din_lanes).unwrap();

            oracle.tick();
            partitioned.tick();

            assert_eq!(
                partitioned.get_signal(dout).unwrap(),
                oracle.get_signal(dout).unwrap(),
                "dout mismatch at cycle {cycle}",
            );
        }
    }

    #[test]
    fn register_balanced_partition_matches_oracle_with_shared_comb() {
        // Two registers share the combinational wire `shared = din + 5`. A
        // register-cone partition into 2 groups replicates `shared` into both,
        // so the groups are combinationally independent.
        let mut design = Design::new();
        let (clk, din, o0, o1);
        {
            let mut m = design.module("Top");
            clk = m.input("clk", uint(1));
            din = m.input("din", uint(8));
            let shared = m.wire("shared", uint(8));
            let r0 = m.reg("r0", uint(8));
            let r1 = m.reg("r1", uint(8));
            o0 = m.output("o0", uint(8));
            o1 = m.output("o1", uint(8));
            m.assign(shared, din + lit_u(5, 8));
            m.clock(r0, clk);
            m.clock(r1, clk);
            m.next(r0, shared + lit_u(1, 8));
            m.next(r1, shared * lit_u(3, 8));
            m.assign(o0, r0);
            m.assign(o1, r1);
        }

        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();

        let lanes = 4;
        let mut oracle = PackedSimulator::new(program.clone(), lanes).unwrap();
        let mut partitioned =
            PartitionedSimulator::new_register_balanced(&program, 2, lanes).unwrap();
        assert_eq!(partitioned.group_count(), 2);

        let clk_lanes = vec![1u128; lanes];
        oracle.set_signal(clk, &clk_lanes).unwrap();
        partitioned.set_signal(clk, &clk_lanes).unwrap();

        for cycle in 0..6u128 {
            let din_lanes: Vec<u128> = (0..lanes as u128)
                .map(|lane| (cycle.wrapping_mul(53).wrapping_add(lane * 17)) & 0xff)
                .collect();
            oracle.set_signal(din, &din_lanes).unwrap();
            partitioned.set_signal(din, &din_lanes).unwrap();

            oracle.tick();
            partitioned.tick();

            assert_eq!(
                partitioned.get_signal(o0).unwrap(),
                oracle.get_signal(o0).unwrap(),
                "o0 mismatch at cycle {cycle}",
            );
            assert_eq!(
                partitioned.get_signal(o1).unwrap(),
                oracle.get_signal(o1).unwrap(),
                "o1 mismatch at cycle {cycle}",
            );
        }
    }

    fn assert_packed_matches_cpu(
        design: &Design,
        top: &str,
        inputs: &[(Signal, Vec<u128>)],
        outputs: &[Signal],
    ) {
        let lanes = inputs.first().map(|(_, values)| values.len()).unwrap_or(1);
        let compiled = compile(design).unwrap();
        let program = lower_to_packed_program(&compiled, top).unwrap();
        let mut packed = PackedSimulator::new(program, lanes).unwrap();
        let mut cpus = (0..lanes)
            .map(|_| Simulator::new(design, top).unwrap())
            .collect::<Vec<_>>();

        for (signal, values) in inputs {
            set_all(&mut packed, &mut cpus, *signal, values);
        }
        packed.eval_combinational();
        assert_outputs_match(&packed, &cpus, outputs);
    }

    fn set_all(
        packed: &mut PackedSimulator,
        cpus: &mut [Simulator<'_>],
        signal: Signal,
        values: &[u128],
    ) {
        packed.set_signal(signal, values).unwrap();
        for (cpu, value) in cpus.iter_mut().zip(values) {
            cpu.set(signal, *value);
        }
    }

    fn assert_outputs_match(packed: &PackedSimulator, cpus: &[Simulator<'_>], outputs: &[Signal]) {
        for output in outputs {
            let packed_values = packed.get_signal(*output).unwrap();
            let cpu_values = cpus.iter().map(|cpu| cpu.get(*output)).collect::<Vec<_>>();
            assert_eq!(
                packed_values, cpu_values,
                "mismatch for signal {:?}",
                output.id
            );
        }
    }

    fn packed_sim_with_machine(
        program: PackedProgram,
        machine: PackedMachineProgram,
        lanes: usize,
    ) -> PackedSimulator {
        analyze_machine_program(&machine).unwrap();
        let execution = PackedExecutionCache::new(&machine);
        let workspaces = PackedExecutionWorkspaces::new(&execution, lanes);
        let values = vec![0; program.total_signal_words * lanes];
        let memories = vec![0; program.total_memory_words_per_lane * lanes];
        let mut sim = PackedSimulator {
            program,
            machine,
            execution,
            workspaces,
            lanes,
            values,
            memories,
            register_captures: Vec::new(),
            register_capture_count: 0,
        };
        sim.eval_combinational();
        sim
    }

    fn liveness_priority_fixture() -> (PackedProgram, PackedBlock, PackedInstrKind) {
        let mut design = Design::new();
        let (a, b, c, y);
        {
            let mut m = design.module("LiveTie");
            a = m.input("a", uint(8));
            b = m.input("b", uint(8));
            c = m.input("c", uint(8));
            y = m.output("y", uint(8));
            m.assign(y, a + b + c);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "LiveTie").unwrap();
        let a_index = program.signal_index(a).unwrap();
        let b_index = program.signal_index(b).unwrap();
        let c_index = program.signal_index(c).unwrap();
        let y_index = program.signal_index(y).unwrap();
        let fifo_first = PackedInstrKind::Not(PackedValueId(2));
        let block = PackedBlock {
            packets: vec![
                PackedMachinePacket {
                    instrs: vec![
                        PackedInstr {
                            dst: PackedValueId(0),
                            ty: uint(8),
                            kind: PackedInstrKind::Signal(a_index),
                        },
                        PackedInstr {
                            dst: PackedValueId(1),
                            ty: uint(8),
                            kind: PackedInstrKind::Signal(b_index),
                        },
                        PackedInstr {
                            dst: PackedValueId(2),
                            ty: uint(8),
                            kind: PackedInstrKind::Signal(c_index),
                        },
                    ],
                    effects: Vec::new(),
                },
                PackedMachinePacket {
                    instrs: vec![
                        PackedInstr {
                            dst: PackedValueId(3),
                            ty: uint(8),
                            kind: fifo_first.clone(),
                        },
                        PackedInstr {
                            dst: PackedValueId(4),
                            ty: uint(8),
                            kind: PackedInstrKind::Add(PackedValueId(0), PackedValueId(1)),
                        },
                    ],
                    effects: vec![PackedEffect::StoreSignal {
                        dst: y_index,
                        value: PackedValueId(4),
                    }],
                },
            ],
        };
        (program, block, fifo_first)
    }

    fn diagnostic_codes(report: &ErrorReport) -> Vec<&str> {
        report
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.code.as_str())
            .collect()
    }
}
