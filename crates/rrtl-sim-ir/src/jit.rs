//! Cranelift JIT backend: compiles a [`PackedMachineProgram`] to a native
//! single-lane `tick` function, trading the interpreter's per-op dispatch for
//! straight-line machine code — the path that contests Verilator on
//! single-instance latency (and where activity skipping would pay).
//!
//! Scope: designs whose signals and memory data are all ≤ 128 bits. Supports the
//! full combinational op set, synchronous and asynchronous resets, and memories
//! (register files / RAMs, branch-free clamp+select addressing). Uniform 16-byte
//! (I128-sized) state slot per signal and per memory entry: narrow values use the
//! low 8 bytes (loaded as `I64`), wider values use the full 16 (loaded as `I128`).
//! `compile` returns an error for anything out of scope (e.g. > 128-bit signals)
//! so callers can fall back to the interpreter.

use std::collections::HashMap;
use std::mem;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types::{Type, I128, I16, I16X8, I32, I32X4, I64, I64X2, I8, I8X16};
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use rayon::prelude::*;
use rrtl_ir::{Diagnostic, ErrorReport};

use crate::activity::expr_direct_reads;
use crate::{
    PackedBlock, PackedEffect, PackedExpr, PackedExprKind, PackedInstrKind, PackedMachineProgram,
    PackedOp, PackedProgram, PackedValueId,
};

fn jit_err(msg: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![Diagnostic::new("E_SIM_IR_JIT", msg)])
}

fn width_mask(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

fn mask128(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

fn lit_u128(words: &[u32]) -> u128 {
    let mut v = 0u128;
    for (i, &w) in words.iter().take(4).enumerate() {
        v |= (w as u128) << (32 * i);
    }
    v
}

/// Cranelift value type for a signal/value of the given width (≤64 → I64).
fn val_ty(width: u32) -> Type {
    if width <= 64 {
        I64
    } else {
        I128
    }
}

/// An I128 constant from its low/high 64-bit halves.
fn const128(b: &mut FunctionBuilder, lo: u64, hi: u64) -> Value {
    let l = b.ins().iconst(I64, lo as i64);
    let h = b.ins().iconst(I64, hi as i64);
    b.ins().iconcat(l, h)
}

/// Promote/reduce a value between the I64 and I128 representations to match the
/// target width's type (zero-extending; callers mask afterward).
fn promote(b: &mut FunctionBuilder, v: Value, from_w: u32, to_w: u32) -> Value {
    match (val_ty(from_w), val_ty(to_w)) {
        (a, t) if a == t => v,
        (_, t) if t == I128 => b.ins().uextend(I128, v),
        _ => b.ins().ireduce(I64, v),
    }
}

/// Build a width-typed constant value.
fn emit_const(b: &mut FunctionBuilder, value: u128, width: u32) -> Value {
    if width <= 64 {
        b.ins().iconst(I64, value as u64 as i64)
    } else {
        const128(b, value as u64, (value >> 64) as u64)
    }
}

/// A native-code single-lane simulator.
pub struct JitSimulator {
    _module: JITModule, // owns the executable code; must outlive the fn pointers
    /// `tick_many(state, n)` — runs `n` cycles in a native loop (no per-cycle Rust).
    tick_many_fn: extern "C" fn(*mut i64, i64),
    /// `run_trace(state, in_ptr, out_ptr, n)` — the baked harness: each cycle loads
    /// the input ports from `in_ptr` and stores the output ports to `out_ptr`, all
    /// in one native call.
    run_fn: extern "C" fn(*mut i64, *const i64, *mut i64, i64),
    /// `settle(state)` — the deferred post-commit combinational pass (no-op in
    /// activity mode, which settles inside its own tick).
    settle_fn: extern "C" fn(*mut i64),
    /// `tick_db(cur, nxt)` — the double-buffered (zero-copy) cycle body: register
    /// reads from `cur`, comb/inputs and all writes to `nxt`. For shared-state
    /// partitioned execution (see [`Self::tick_db`]).
    tick_db_fn: extern "C" fn(*const i64, *mut i64),
    /// `tick_clocked(state, mask)` — one cycle gating register commits by clock
    /// (multi-clock); bit i of `mask` = clock i has a rising edge this step.
    tick_clocked_fn: extern "C" fn(*mut i64, i64),
    /// Clock signal index → its bit in the `tick_clocked` mask.
    clock_bit: HashMap<usize, u32>,
    /// Whether the in-memory combinational signals are stale (a tick committed
    /// registers without re-settling). Cleared by [`Self::settle`].
    comb_dirty: bool,
    /// Per-signal: true if the signal is combinationally driven, so reading it
    /// requires a settle when `comb_dirty`. Registers/inputs are false (their
    /// stored value is already correct after a tick).
    needs_settle: Vec<bool>,
    state: Vec<i64>,
    /// Byte offset and depth of each memory in `state` (for [`Self::set_memory`]).
    mem_base: Vec<usize>,
    mem_depth: Vec<usize>,
    widths: Vec<u32>,
    input_idx: Vec<usize>,
    output_idx: Vec<usize>,
    input_names: Vec<String>,
    output_names: Vec<String>,
    /// Activity-skipping mode: when set, `(skips_idx, total_idx)` are the `i64`
    /// state slots that accumulate skipped / total guarded-cone evaluations.
    activity: Option<(usize, usize)>,
}

impl JitSimulator {
    /// Compile a machine program, or return an error if it is out of scope
    /// (any signal or memory data wider than 128 bits).
    pub fn compile(machine: &PackedMachineProgram) -> Result<Self, ErrorReport> {
        Self::compile_inner(machine, false, false, None)
    }

    /// Compile for double-buffered ([`tick_db`](Self::tick_db)) execution with an
    /// explicit global register mask: in a partitioned/zero-copy runner each
    /// partition is one cone of a larger design, so *boundary* registers it reads
    /// (owned by other partitions) must also read from `cur`. `is_reg` must cover
    /// all of the global design's registers (length ≥ this machine's signal count).
    pub fn compile_db(machine: &PackedMachineProgram, is_reg: &[bool]) -> Result<Self, ErrorReport> {
        Self::compile_inner(machine, false, false, Some(is_reg))
    }

    /// Compile with activity ("event-driven") skipping: each combinational signal
    /// and register becomes a guarded cone that re-evaluates only when one of its
    /// direct fan-in signals changed since its last evaluation (value-compared
    /// against a per-cone snapshot, in topological stream order). Bit-exact with
    /// [`compile`]; targets the single-instance latency regime where native
    /// branches can make the skip pay (unlike the data-parallel SIMD path).
    pub fn compile_activity(machine: &PackedMachineProgram) -> Result<Self, ErrorReport> {
        Self::compile_inner(machine, true, false, None)
    }

    /// Like [`compile_activity`] but with native skip/total counters wired in so
    /// [`activity_skip_rate`](Self::activity_skip_rate) reports the realized skip
    /// fraction. The counters add per-cone overhead, so use [`compile_activity`]
    /// for timing and this only to measure skip rate.
    pub fn compile_activity_instrumented(machine: &PackedMachineProgram) -> Result<Self, ErrorReport> {
        Self::compile_inner(machine, true, true, None)
    }

    fn compile_inner(
        machine: &PackedMachineProgram,
        activity: bool,
        counters: bool,
        is_reg_override: Option<&[bool]>,
    ) -> Result<Self, ErrorReport> {
        let source = &machine.source;
        for s in &source.signals {
            if s.layout.width > 128 {
                return Err(jit_err(format!(
                    "JIT supports signals ≤ 128 bits; `{}` is {} bits",
                    s.name, s.layout.width
                )));
            }
        }
        for mem in &source.memories {
            if mem.data_layout.width > 128 {
                return Err(jit_err(format!(
                    "JIT supports memory data ≤ 128 bits; `{}` is {} bits",
                    mem.name, mem.data_layout.width
                )));
            }
        }
        let widths: Vec<u32> = source.signals.iter().map(|s| s.layout.width).collect();
        let num_signals = source.signals.len();

        // Which signals are clocked registers (CaptureReg destinations) — read from
        // the `cur` buffer in the double-buffered (zero-copy partitioned) tick.
        let mut is_reg = vec![false; num_signals];
        for block in [&machine.streams.async_reset_comb, &machine.streams.tick_next] {
            for packet in &block.packets {
                for eff in &packet.effects {
                    if let PackedEffect::CaptureReg { dst, .. } = eff {
                        if *dst < num_signals {
                            is_reg[*dst] = true;
                        }
                    }
                }
            }
        }
        // A partitioned/zero-copy compile supplies the GLOBAL register mask so that
        // boundary registers (this partition reads but does not own) also read `cur`.
        if let Some(o) = is_reg_override {
            is_reg = (0..num_signals).map(|i| o.get(i).copied().unwrap_or(false)).collect();
        }

        // Uniform 16-byte (I128-sized) state slot per signal and per memory entry,
        // so the byte offset of signal `i` is `i*16` regardless of width — narrow
        // signals use only the low 8 bytes. `mem.base[m]` is the BYTE offset of
        // memory m's entry 0.
        let mut mem = MemLayout::default();
        let mut total_mem = 0usize;
        for m in &source.memories {
            mem.base.push((num_signals + total_mem) * 16);
            mem.depth.push(m.depth);
            mem.data_width.push(m.data_layout.width);
            total_mem += m.depth;
        }
        let (saved_mem_base, saved_mem_depth) = (mem.base.clone(), mem.depth.clone());
        let mut state_len = (num_signals + total_mem) * 2; // i64 count (2 per slot)

        // Activity mode reserves a control + snapshot region after signals+memory:
        //   [seen flag][skips counter][total counter][per-cone leaf snapshots...]
        // each a 16-byte slot (control words use the low 8 bytes).
        let act_layout = if activity {
            let ctrl_base = (num_signals + total_mem) * 16; // bytes
            let snap_start = ctrl_base + 48;
            let snap_reads = activity_snap_reads(source);
            state_len = (snap_start) / 8 + snap_reads * 2;
            Some(ActivityLayout {
                seen_off: ctrl_base as i32,
                skips_off: (ctrl_base + 16) as i32,
                total_off: (ctrl_base + 32) as i32,
                snap_start,
            })
        } else {
            None
        };
        let activity_idx = if counters {
            act_layout
                .as_ref()
                .map(|l| (l.skips_off as usize / 8, l.total_off as usize / 8))
        } else {
            None
        };

        // Input/output port indices (for the baked-harness trace function).
        let mut input_idx = Vec::new();
        let mut output_idx = Vec::new();
        let mut input_names = Vec::new();
        let mut output_names = Vec::new();
        for (i, s) in source.signals.iter().enumerate() {
            match s.kind {
                crate::PackedSignalKind::Input => {
                    input_idx.push(i);
                    input_names.push(s.name.clone());
                }
                crate::PackedSignalKind::Output => {
                    output_idx.push(i);
                    output_names.push(s.name.clone());
                }
                _ => {}
            }
        }
        let nin = input_idx.len();
        let nout = output_idx.len();

        // --- Cranelift module setup ---
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").map_err(|e| jit_err(e.to_string()))?;
        let isa_builder = cranelift_native::builder().map_err(|e| jit_err(e.to_string()))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| jit_err(e.to_string()))?;
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        // Multi-clock: assign each distinct register clock a bit in the
        // `tick_clocked` mask, and map each clocked register's dst to that bit
        // (for the per-register commit select). ≤64 clocks (one i64 mask).
        let mut clock_ids: Vec<usize> = machine
            .source
            .reg_clocks
            .values()
            .chain(machine.source.mem_clocks.values())
            .copied()
            .collect();
        clock_ids.sort_unstable();
        clock_ids.dedup();
        let clock_bit: HashMap<usize, u32> =
            clock_ids.iter().take(64).enumerate().map(|(i, &c)| (c, i as u32)).collect();
        let reg_clock_bit: HashMap<usize, u32> = machine
            .source
            .reg_clocks
            .iter()
            .filter_map(|(&dst, &clk)| clock_bit.get(&clk).map(|&b| (dst, b)))
            .collect();
        let mem_clock_bit: HashMap<usize, u32> = machine
            .source
            .mem_clocks
            .iter()
            .filter_map(|(&memory, &clk)| clock_bit.get(&clk).map(|&b| (memory, b)))
            .collect();

        // tick_many(state, n): native loop over the cycle body.
        let act_ref = act_layout.as_ref();
        let tick_many_id = define_fn(&mut module, "tick_many", 2, |b, p| {
            let (state_ptr, n) = (p[0], p[1]);
            emit_loop(b, n, |b, _i| match act_ref {
                Some(lay) => emit_activity_tick(b, state_ptr, source, &widths, &mem, lay, counters),
                None => emit_tick_body(b, state_ptr, state_ptr, &is_reg, machine, &widths, &mem, false, None),
            })?;
            b.ins().return_(&[]);
            Ok(())
        })?;

        // tick_clocked(state, mask): one cycle gating register commits by clock —
        // each clocked register commits only if its clock's bit in `mask` is set.
        let reg_clock_bit_ref = &reg_clock_bit;
        let mem_clock_bit_ref = &mem_clock_bit;
        let tick_clocked_id = define_fn(&mut module, "tick_clocked", 2, |b, p| {
            emit_tick_body(
                b, p[0], p[0], &is_reg, machine, &widths, &mem, false,
                Some((p[1], reg_clock_bit_ref, mem_clock_bit_ref)),
            )?;
            b.ins().return_(&[]);
            Ok(())
        })?;

        // tick_db(cur, nxt): the double-buffered (zero-copy) cycle body — register
        // reads come from `cur`, comb/inputs and all writes go to `nxt`. Lets many
        // partition functions share one global state with no boundary copy: each
        // reads every other partition's last-cycle registers from `cur` and commits
        // its own to `nxt`; the runner swaps cur/nxt each cycle. No post-settle
        // (observed outputs are registers, committed into `nxt`).
        let tick_db_id = define_fn(&mut module, "tick_db", 2, |b, p| {
            let (cur_ptr, nxt_ptr) = (p[0], p[1]);
            emit_tick_body(b, nxt_ptr, cur_ptr, &is_reg, machine, &widths, &mem, false, None)?;
            b.ins().return_(&[]);
            Ok(())
        })?;

        // settle(state): the deferred post-commit combinational pass, run lazily
        // before reading a combinational signal after `tick_many`.
        let settle_id = define_fn(&mut module, "settle", 1, |b, p| {
            let state_ptr = p[0];
            if act_ref.is_none() {
                emit_post_settle(b, state_ptr, state_ptr, &is_reg, machine, &widths, &mem)?;
            }
            b.ins().return_(&[]);
            Ok(())
        })?;

        // run_trace(state, in_ptr, out_ptr, n): baked harness — load input ports
        // from in_ptr[i*nin..], run the cycle, store output ports to out_ptr[i*nout..].
        let run_id = define_fn(&mut module, "run_trace", 4, |b, p| {
            let (state_ptr, in_ptr, out_ptr, n) = (p[0], p[1], p[2], p[3]);
            let in_idx = input_idx.clone();
            let out_idx = output_idx.clone();
            let widths_ref = &widths;
            emit_loop(b, n, move |b, i| {
                // load inputs: state[in_idx[k]] = in_ptr[i*nin + k]
                let base = b.ins().imul_imm(i, nin as i64);
                for (k, &sidx) in in_idx.iter().enumerate() {
                    let off = b.ins().iadd_imm(base, k as i64);
                    let byte = b.ins().ishl_imm(off, 3);
                    let ptr = b.ins().iadd(in_ptr, byte);
                    let v = b.ins().load(I64, MemFlags::trusted(), ptr, 0);
                    let m = mask_to(b, v, widths_ref[sidx]);
                    b.ins().store(MemFlags::trusted(), m, state_ptr, (sidx * 16) as i32);
                }
                match act_ref {
                    Some(lay) => emit_activity_tick(b, state_ptr, source, widths_ref, &mem, lay, counters)?,
                    None => emit_tick_body(b, state_ptr, state_ptr, &is_reg, machine, widths_ref, &mem, true, None)?,
                }
                // store outputs: out_ptr[i*nout + k] = state[out_idx[k]]
                let obase = b.ins().imul_imm(i, nout as i64);
                for (k, &sidx) in out_idx.iter().enumerate() {
                    let v = b.ins().load(I64, MemFlags::trusted(), state_ptr, (sidx * 16) as i32);
                    let off = b.ins().iadd_imm(obase, k as i64);
                    let byte = b.ins().ishl_imm(off, 3);
                    let ptr = b.ins().iadd(out_ptr, byte);
                    b.ins().store(MemFlags::trusted(), v, ptr, 0);
                }
                Ok(())
            })?;
            b.ins().return_(&[]);
            Ok(())
        })?;

        module.finalize_definitions().map_err(|e| jit_err(e.to_string()))?;
        let tick_many_fn: extern "C" fn(*mut i64, i64) =
            unsafe { mem::transmute(module.get_finalized_function(tick_many_id)) };
        let run_fn: extern "C" fn(*mut i64, *const i64, *mut i64, i64) =
            unsafe { mem::transmute(module.get_finalized_function(run_id)) };
        let settle_fn: extern "C" fn(*mut i64) =
            unsafe { mem::transmute(module.get_finalized_function(settle_id)) };
        let tick_db_fn: extern "C" fn(*const i64, *mut i64) =
            unsafe { mem::transmute(module.get_finalized_function(tick_db_id)) };
        let tick_clocked_fn: extern "C" fn(*mut i64, i64) =
            unsafe { mem::transmute(module.get_finalized_function(tick_clocked_id)) };

        // A signal needs a settle before observation iff it is combinationally
        // driven (written in the comb / async-reset-comb streams). Registers
        // (clocked-only) and inputs already hold their correct value after tick.
        let mut needs_settle = vec![false; num_signals];
        for block in [&machine.streams.async_reset_comb, &machine.streams.comb] {
            for packet in &block.packets {
                for effect in &packet.effects {
                    match effect {
                        PackedEffect::StoreSignal { dst, .. }
                        | PackedEffect::CaptureReg { dst, .. } => {
                            if *dst < num_signals {
                                needs_settle[*dst] = true;
                            }
                        }
                        PackedEffect::MemoryWrite { .. } => {}
                    }
                }
            }
        }
        // Trivial register/input-aliased outputs are refreshed at commit (see
        // emit_tick_body), so they need no settle to observe.
        for (dst, _) in trivial_output_aliases(machine) {
            if dst < num_signals {
                needs_settle[dst] = false;
            }
        }

        Ok(Self {
            _module: module,
            tick_many_fn,
            run_fn,
            settle_fn,
            tick_db_fn,
            tick_clocked_fn,
            clock_bit,
            comb_dirty: false,
            needs_settle,
            state: vec![0i64; state_len],
            mem_base: saved_mem_base,
            mem_depth: saved_mem_depth,
            widths,
            input_idx,
            output_idx,
            input_names,
            output_names,
            activity: activity_idx,
        })
    }

    /// Fraction of guarded cone-evaluations skipped so far (activity mode only).
    pub fn activity_skip_rate(&self) -> Option<f64> {
        let (skips_idx, total_idx) = self.activity?;
        let total = self.state[total_idx] as u64;
        let skips = self.state[skips_idx] as u64;
        Some(if total == 0 { 0.0 } else { skips as f64 / total as f64 })
    }

    pub fn set_signal(&mut self, index: usize, value: u64) {
        self.set_signal_u128(index, value as u128);
    }
    pub fn get_signal(&mut self, index: usize) -> u64 {
        self.get_signal_u128(index) as u64
    }

    /// Set/get a signal up to 128 bits wide.
    pub fn set_signal_u128(&mut self, index: usize, value: u128) {
        let v = value & mask128(self.widths[index]);
        self.state[index * 2] = v as u64 as i64;
        self.state[index * 2 + 1] = (v >> 64) as u64 as i64;
    }
    pub fn get_signal_u128(&mut self, index: usize) -> u128 {
        // A combinational signal may be stale after a `tick_many` (whose
        // post-commit settle is deferred); settle on demand before reading it.
        if self.comb_dirty && self.needs_settle[index] {
            self.settle();
        }
        let lo = self.state[index * 2] as u64 as u128;
        let hi = self.state[index * 2 + 1] as u64 as u128;
        (lo | (hi << 64)) & mask128(self.widths[index])
    }

    /// Whether signal `index` is combinationally driven (reading it forces a
    /// settle after `tick_many`). Diagnostic for the lazy-settle path.
    pub fn is_comb_driven(&self, index: usize) -> bool {
        self.needs_settle[index]
    }

    /// Raw access to the `i64` state buffer: signal `i` occupies words
    /// `[i*2, i*2+2)` (low, high). This is the fast path for cross-engine boundary
    /// exchange in a partitioned simulator — a register's slot already holds its
    /// correctly-masked committed value, so exchange is a 16-byte word copy with
    /// none of `get_signal`'s per-access masking / settle / bounds machinery.
    /// (Use only for register-stable signals; combinational reads still need
    /// `get_signal_u128`'s settle path.)
    pub fn state_words(&self) -> &[i64] {
        &self.state
    }
    pub fn state_words_mut(&mut self) -> &mut [i64] {
        &mut self.state
    }
    /// Load a memory's contents (entry `k` ← `words[k]`), e.g. to preload a CPU's
    /// program RAM. `mem_index` is the memory's position in the program's memory
    /// list. Each entry is a 16-byte slot; only the low data-width bits are read.
    pub fn set_memory(&mut self, mem_index: usize, words: &[u128]) {
        let base = self.mem_base[mem_index]; // byte offset of entry 0
        let depth = self.mem_depth[mem_index];
        let w0 = base / 8; // i64 index of entry 0 (16-byte slots ⇒ 2 i64 each)
        for (k, &word) in words.iter().enumerate().take(depth) {
            self.state[w0 + k * 2] = word as i64;
            self.state[w0 + k * 2 + 1] = (word >> 64) as i64;
        }
    }
    /// i64 index in [`state_words`] of memory `mem_index` entry 0 (entries are 2
    /// i64 apart), for writing program RAM directly into a batch state buffer.
    pub fn memory_word_base(&self, mem_index: usize) -> usize {
        self.mem_base[mem_index] / 8
    }
    /// Word offset of signal `index`'s slot in [`state_words`].
    #[inline]
    pub fn signal_word(index: usize) -> usize {
        index * 2
    }

    /// Run one double-buffered cycle on EXTERNAL shared buffers: register reads come
    /// from `cur`, comb/inputs and all commits go to `nxt`. For a zero-copy
    /// partitioned/parallel runner — every partition compiled over the same global
    /// signal layout shares `cur`/`nxt`, reads each other's last-cycle registers
    /// directly from `cur` (no boundary copy), commits its own to `nxt`; the runner
    /// swaps `cur`/`nxt` after all partitions. `cur`/`nxt` must each be
    /// `state_words().len()`-long; the JIT's own `state` is unused in this mode.
    pub fn tick_db(&self, cur: &[i64], nxt: &mut [i64]) {
        (self.tick_db_fn)(cur.as_ptr(), nxt.as_mut_ptr());
    }

    /// Raw-pointer form of [`tick_db`] for PARALLEL partitioned execution: many
    /// partitions can run concurrently on the same shared `cur`/`nxt` buffers.
    /// `cur` is read-only (shared freely); writes go to disjoint register slots of
    /// `nxt` so the JIT'd stores never collide.
    ///
    /// # Safety
    /// The caller must guarantee no two concurrent calls write overlapping `nxt`
    /// slots — true when partitions own disjoint registers and the design has no
    /// replicated combinational state written to shared slots.
    pub unsafe fn tick_db_raw(&self, cur: *const i64, nxt: *mut i64) {
        (self.tick_db_fn)(cur, nxt);
    }

    /// The raw `tick_db` function pointer. Function pointers are `Send`/`Sync` (the
    /// owning `JitSimulator` is not), so a parallel runner can collect these and
    /// dispatch partitions across a thread pool — the JIT keeps the code alive.
    pub fn tick_db_fn_ptr(&self) -> extern "C" fn(*const i64, *mut i64) {
        self.tick_db_fn
    }

    /// The `tick_many(state, n)` function pointer. With one compiled design and many
    /// independent per-instance state buffers (one per stimulus/seed), a *bulk-
    /// synchronous* batch runner can advance each buffer N cycles on its own core
    /// with NO per-cycle synchronization — the reentrant code only touches the state
    /// pointer it's given. The buffer length is [`state_words`]`().len()`.
    pub fn tick_many_fn_ptr(&self) -> extern "C" fn(*mut i64, i64) {
        self.tick_many_fn
    }

    /// Run the deferred post-commit combinational settle if pending.
    fn settle(&mut self) {
        if self.comb_dirty {
            (self.settle_fn)(self.state.as_mut_ptr());
            self.comb_dirty = false;
        }
    }

    pub fn tick(&mut self) {
        (self.tick_many_fn)(self.state.as_mut_ptr(), 1);
        self.comb_dirty = true;
    }

    /// Run `n` cycles in a single native call (no per-cycle Rust). Inputs must be
    /// set beforehand (or held).
    pub fn tick_many(&mut self, n: usize) {
        (self.tick_many_fn)(self.state.as_mut_ptr(), n as i64);
        self.comb_dirty = true;
    }

    /// Advance one cycle, gating register commits by clock (multi-clock).
    /// `active_clocks` lists the clock SIGNAL INDICES with a rising edge this
    /// step; registers on other clocks hold. Single-step (the active set varies
    /// per cycle), so call once per cycle with the schedule. With every clock
    /// listed it is identical to [`Self::tick`]. Unclocked registers always
    /// commit. (Memory writes are not yet clock-gated — register domains only.)
    pub fn tick_clocked(&mut self, active_clocks: &[usize]) {
        let mut mask = 0i64;
        for &clk in active_clocks {
            if let Some(&bit) = self.clock_bit.get(&clk) {
                mask |= 1i64 << bit;
            }
        }
        (self.tick_clocked_fn)(self.state.as_mut_ptr(), mask);
        self.comb_dirty = true;
    }

    /// Port names in the order [`run_trace`] expects/produces them.
    pub fn input_ports(&self) -> &[String] {
        &self.input_names
    }
    pub fn output_ports(&self) -> &[String] {
        &self.output_names
    }
    /// Signal indices for the input/output ports, in `run_trace` buffer order.
    pub fn input_indices(&self) -> &[usize] {
        &self.input_idx
    }
    pub fn output_indices(&self) -> &[usize] {
        &self.output_idx
    }

    /// The baked harness: drive `cycles` cycles of stimulus and capture every
    /// output, in ONE native call. `inputs` is row-major `[cycle][input_port]`
    /// (length `cycles * input_ports().len()`); returns row-major
    /// `[cycle][output_port]`.
    pub fn run_trace(&mut self, inputs: &[u64], cycles: usize) -> Vec<u64> {
        assert_eq!(inputs.len(), cycles * self.input_idx.len(), "input buffer size");
        let in_i64: Vec<i64> = inputs.iter().map(|&v| v as i64).collect();
        let mut out = vec![0i64; cycles * self.output_idx.len()];
        (self.run_fn)(
            self.state.as_mut_ptr(),
            in_i64.as_ptr(),
            out.as_mut_ptr(),
            cycles as i64,
        );
        self.comb_dirty = false; // run_trace settles each cycle internally
        out.iter().map(|&v| v as u64).collect()
    }
}

/// Define a function of `nparams` `i64` params whose body is built by `build`
/// (which receives the parameter values and must emit a terminator/return).
fn define_fn<F>(
    module: &mut JITModule,
    name: &str,
    nparams: usize,
    build: F,
) -> Result<cranelift_module::FuncId, ErrorReport>
where
    F: FnOnce(&mut FunctionBuilder, &[Value]) -> Result<(), ErrorReport>,
{
    let mut sig = module.make_signature();
    for _ in 0..nparams {
        sig.params.push(AbiParam::new(I64));
    }
    let id = module
        .declare_function(name, Linkage::Export, &sig)
        .map_err(|e| jit_err(e.to_string()))?;
    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fb_ctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        let params: Vec<Value> = b.block_params(entry).to_vec();
        build(&mut b, &params)?;
        b.seal_all_blocks();
        b.finalize();
    }
    module.define_function(id, &mut ctx).map_err(|e| jit_err(e.to_string()))?;
    module.clear_context(&mut ctx);
    Ok(id)
}

/// Emit `for i in 0..n { body(i) }` and leave the builder positioned in the exit
/// block (so the caller emits the function's return).
fn emit_loop<F>(b: &mut FunctionBuilder, n: Value, mut body: F) -> Result<(), ErrorReport>
where
    F: FnMut(&mut FunctionBuilder, Value) -> Result<(), ErrorReport>,
{
    let header = b.create_block();
    b.append_block_param(header, I64);
    let body_blk = b.create_block();
    let exit = b.create_block();
    let zero = b.ins().iconst(I64, 0);
    b.ins().jump(header, &[zero]);
    b.switch_to_block(header);
    let i = b.block_params(header)[0];
    let cond = b.ins().icmp(IntCC::UnsignedLessThan, i, n);
    b.ins().brif(cond, body_blk, &[], exit, &[]);
    b.switch_to_block(body_blk);
    body(b, i)?;
    let i1 = b.ins().iadd_imm(i, 1);
    b.ins().jump(header, &[i1]);
    b.switch_to_block(exit);
    Ok(())
}

/// Emit one cycle: settle comb, capture register next-state, write memories,
/// commit registers, then (if `post_settle`) settle comb again so observable
/// combinational outputs reflect the just-committed registers. When
/// `post_settle` is false the post-commit settle is deferred — the caller must
/// run [`emit_settle`] before observing any combinational signal (the lazy-
/// settle path used by `tick_many`, since the next tick's pre-settle re-settles
/// anyway and most harnesses read only registered outputs).
fn emit_tick_body(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    cur: Value,
    is_reg: &[bool],
    machine: &PackedMachineProgram,
    widths: &[u32],
    mem: &MemLayout,
    post_settle: bool,
    // Multi-clock gating: `(mask, dst→clock-bit)`. A clocked register commits only
    // when its clock's bit in `mask` is set, else holds (a branch-free select on
    // its pre-commit value). `None` = every register commits (single-clock).
    // The tuple is `(mask, reg dst→clock-bit, memory idx→clock-bit)`.
    clock_gate: Option<(Value, &HashMap<usize, u32>, &HashMap<usize, u32>)>,
) -> Result<(), ErrorReport> {
    emit_comb(b, state_ptr, cur, is_reg, &machine.streams.async_reset_comb, widths, mem)?;
    emit_comb(b, state_ptr, cur, is_reg, &machine.streams.comb, widths, mem)?;
    let captures = emit_capture(b, state_ptr, cur, is_reg, &machine.streams.tick_next, widths, mem)?;
    let mem_gate = clock_gate.map(|(mask, _, mem_bit)| (mask, mem_bit));
    emit_memwrites(b, state_ptr, cur, is_reg, &machine.streams.tick_commit, widths, mem, mem_gate)?;
    for (dst, value) in captures {
        let masked = mask_to(b, value, widths[dst]);
        let to_store = match clock_gate {
            Some((mask, bit_of, _)) if bit_of.contains_key(&dst) => {
                // committed = (mask >> clock_bit) & 1 ? next : current
                let current = b.ins().load(val_ty(widths[dst]), MemFlags::trusted(), state_ptr, (dst * 16) as i32);
                let shifted = b.ins().ushr_imm(mask, bit_of[&dst] as i64);
                let active = b.ins().band_imm(shifted, 1);
                let cond = b.ins().icmp_imm(IntCC::NotEqual, active, 0);
                b.ins().select(cond, masked, current)
            }
            _ => masked,
        };
        b.ins().store(MemFlags::trusted(), to_store, state_ptr, (dst * 16) as i32);
    }
    if post_settle {
        emit_post_settle(b, state_ptr, cur, is_reg, machine, widths, mem)?;
    } else {
        // Refresh trivial register/input-aliased output ports at commit time so
        // observing them needs no full settle (they're marked needs_settle=false).
        for (dst, src) in trivial_output_aliases(machine) {
            let v = b.ins().load(val_ty(widths[src]), MemFlags::trusted(), state_ptr, (src * 16) as i32);
            let v = promote(b, v, widths[src], widths[dst]);
            let masked = mask_to(b, v, widths[dst]);
            b.ins().store(MemFlags::trusted(), masked, state_ptr, (dst * 16) as i32);
        }
    }
    Ok(())
}

/// Output ports whose only combinational driver is a trivial copy of a register
/// or input (`out = reg`/`out = in`). Such a port can be refreshed at register-
/// commit time instead of via a full combinational settle, so it needs no settle
/// to observe after `tick_many`. Returns `(dst_signal, src_signal)` pairs.
fn trivial_output_aliases(machine: &PackedMachineProgram) -> Vec<(usize, usize)> {
    let source = &machine.source;
    let kind = |idx: usize| source.signals.get(idx).map(|s| s.kind);
    let is_stable = |idx: usize| {
        matches!(kind(idx), Some(crate::PackedSignalKind::Reg) | Some(crate::PackedSignalKind::Input))
    };
    let mut store_count: HashMap<usize, usize> = HashMap::new();
    let mut alias: HashMap<usize, usize> = HashMap::new();
    for block in [&machine.streams.async_reset_comb, &machine.streams.comb] {
        // value-ids flow across packets within a block, so map copies block-wide.
        let mut copy_src: HashMap<usize, usize> = HashMap::new();
        for packet in &block.packets {
            for instr in &packet.instrs {
                if let PackedInstrKind::Signal(src) = instr.kind {
                    copy_src.insert(instr.dst.0, src);
                }
            }
        }
        for packet in &block.packets {
            for effect in &packet.effects {
                if let PackedEffect::StoreSignal { dst, value } = effect {
                    *store_count.entry(*dst).or_default() += 1;
                    if kind(*dst) == Some(crate::PackedSignalKind::Output) {
                        if let Some(&src) = copy_src.get(&value.0) {
                            if is_stable(src) {
                                alias.insert(*dst, src);
                            }
                        }
                    }
                }
            }
        }
    }
    // Only single-driver outputs are safe to relocate.
    alias
        .into_iter()
        .filter(|(dst, _)| store_count.get(dst) == Some(&1))
        .collect()
}

/// Settle the combinational streams (the post-commit observability pass).
fn emit_post_settle(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    cur: Value,
    is_reg: &[bool],
    machine: &PackedMachineProgram,
    widths: &[u32],
    mem: &MemLayout,
) -> Result<(), ErrorReport> {
    emit_comb(b, state_ptr, cur, is_reg, &machine.streams.async_reset_comb, widths, mem)?;
    emit_comb(b, state_ptr, cur, is_reg, &machine.streams.comb, widths, mem)?;
    Ok(())
}

/// Mask a value (whose Cranelift type matches `width`) to its low `width` bits.
fn mask_to(b: &mut FunctionBuilder, v: Value, width: u32) -> Value {
    if width <= 64 {
        if width == 64 {
            v
        } else {
            b.ins().band_imm(v, width_mask(width) as i64)
        }
    } else if width >= 128 {
        v
    } else {
        let mask = const128(b, u64::MAX, (1u64 << (width - 64)) - 1);
        b.ins().band(v, mask)
    }
}

/// Where each memory's state lives in the flat `i64` state array.
#[derive(Default)]
struct MemLayout {
    base: Vec<usize>,
    depth: Vec<usize>,
    data_width: Vec<u32>,
}

/// Computed pointer to memory entry `clamp(addr, depth-1)`. `base` is the BYTE
/// offset of entry 0; entries are 16 bytes apart. Clamping keeps the access
/// in-bounds (the caller `select`s the result against `addr < depth`), so no
/// branch is needed.
fn mem_elem_ptr(b: &mut FunctionBuilder, state_ptr: Value, base: usize, depth: usize, addr: Value) -> Value {
    let dm1 = b.ins().iconst(I64, (depth.saturating_sub(1)) as i64);
    let clamped = b.ins().umin(addr, dm1);
    let byte = b.ins().imul_imm(clamped, 16); // 16 bytes per entry
    let off = b.ins().iadd_imm(byte, base as i64);
    b.ins().iadd(state_ptr, off)
}

/// Boolean (I8) value: is this reset asserted? (reads the reset signal's bit 0,
/// honoring polarity).
/// Pick the buffer a Signal load reads from: registers read the `cur` buffer (their
/// stable value this cycle), everything else (comb, inputs) the working buffer. With
/// `cur == work` (single-buffer mode) this is a no-op; distinct buffers double-buffer
/// the registers for a zero-copy partitioned/parallel tick.
#[inline]
fn read_ptr(work: Value, cur: Value, is_reg: &[bool], idx: usize) -> Value {
    if is_reg.get(idx).copied().unwrap_or(false) {
        cur
    } else {
        work
    }
}

fn reset_asserted(b: &mut FunctionBuilder, state_ptr: Value, cur: Value, is_reg: &[bool], reset: &crate::PackedReset) -> Value {
    let p = read_ptr(state_ptr, cur, is_reg, reset.signal);
    let rsig = b.ins().load(I64, MemFlags::trusted(), p, (reset.signal * 16) as i32);
    let bit = b.ins().band_imm(rsig, 1);
    match reset.polarity {
        rrtl_ir::ResetPolarity::ActiveHigh => b.ins().icmp_imm(IntCC::NotEqual, bit, 0),
        rrtl_ir::ResetPolarity::ActiveLow => b.ins().icmp_imm(IntCC::Equal, bit, 0),
    }
}

/// Emit a combinational block: evaluate instructions, apply `StoreSignal` effects.
fn emit_comb(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    cur: Value,
    is_reg: &[bool],
    block: &PackedBlock,
    widths: &[u32],
    mem: &MemLayout,
) -> Result<(), ErrorReport> {
    // Value ids are block-local and flow across packets, so the map spans the block.
    let mut vals: HashMap<usize, Value> = HashMap::new();
    let mut vw: HashMap<usize, u32> = HashMap::new();
    let mut clean: HashMap<usize, bool> = HashMap::new();
    for packet in &block.packets {
        for instr in &packet.instrs {
            let (v, c) = emit_instr(b, state_ptr, cur, is_reg, &vals, &vw, &clean, instr, mem)?;
            vals.insert(instr.dst.0, v);
            vw.insert(instr.dst.0, instr.ty.width);
            clean.insert(instr.dst.0, c);
        }
        for effect in &packet.effects {
            match effect {
                PackedEffect::StoreSignal { dst, value } => {
                    let m = store_value(b, &vals, &vw, &clean, *value, widths[*dst])?;
                    b.ins().store(MemFlags::trusted(), m, state_ptr, (*dst * 16) as i32);
                }
                PackedEffect::CaptureReg { dst, reset, .. } => {
                    // Async-reset immediate conditional store: while the reset is
                    // asserted, force the register to its reset value (visible
                    // combinationally this cycle); the clocked capture is in tick_next.
                    let Some(reset) = reset else {
                        return Err(jit_err("reset-less capture in combinational stream"));
                    };
                    let dw = widths[*dst];
                    let asserted = reset_asserted(b, state_ptr, cur, is_reg, reset);
                    let rval = emit_const(b, lit_u128(&reset.value), dw);
                    let current = b.ins().load(val_ty(dw), MemFlags::trusted(), read_ptr(state_ptr, cur, is_reg, *dst), (*dst * 16) as i32);
                    let newv = b.ins().select(asserted, rval, current);
                    let masked = mask_to(b, newv, dw);
                    b.ins().store(MemFlags::trusted(), masked, state_ptr, (*dst * 16) as i32);
                }
                PackedEffect::MemoryWrite { .. } => {
                    return Err(jit_err("memory write in combinational stream"))
                }
            }
        }
    }
    Ok(())
}

/// Emit a capture block (tick_next): evaluate instructions, returning the
/// next-state value for each captured register (commit is done by the caller).
fn emit_capture(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    cur: Value,
    is_reg: &[bool],
    block: &PackedBlock,
    widths: &[u32],
    mem: &MemLayout,
) -> Result<Vec<(usize, Value)>, ErrorReport> {
    let mut out = Vec::new();
    let mut vals: HashMap<usize, Value> = HashMap::new();
    let mut vw: HashMap<usize, u32> = HashMap::new();
    let mut clean: HashMap<usize, bool> = HashMap::new();
    for packet in &block.packets {
        for instr in &packet.instrs {
            let (v, c) = emit_instr(b, state_ptr, cur, is_reg, &vals, &vw, &clean, instr, mem)?;
            vals.insert(instr.dst.0, v);
            vw.insert(instr.dst.0, instr.ty.width);
            clean.insert(instr.dst.0, c);
        }
        for effect in &packet.effects {
            match effect {
                PackedEffect::CaptureReg { dst, value, reset } => {
                    let dw = widths[*dst];
                    // commit masks unconditionally (the caller stores this), so the
                    // captured value need only be correct in its low `dw` bits.
                    let next = store_value(b, &vals, &vw, &clean, *value, dw)?;
                    let next = if let Some(reset) = reset {
                        // sync reset: reset_asserted ? reset_value : next
                        let rsig = b.ins().load(I64, MemFlags::trusted(), read_ptr(state_ptr, cur, is_reg, reset.signal), (reset.signal * 16) as i32);
                        let bit = b.ins().band_imm(rsig, 1);
                        let asserted = match reset.polarity {
                            rrtl_ir::ResetPolarity::ActiveHigh => b.ins().icmp_imm(IntCC::NotEqual, bit, 0),
                            rrtl_ir::ResetPolarity::ActiveLow => b.ins().icmp_imm(IntCC::Equal, bit, 0),
                        };
                        let rval = emit_const(b, lit_u128(&reset.value), dw);
                        b.ins().select(asserted, rval, next)
                    } else {
                        next
                    };
                    out.push((*dst, next));
                }
                PackedEffect::StoreSignal { dst, value } => {
                    // a combinational store occurring in tick_next (rare); apply it.
                    let m = store_value(b, &vals, &vw, &clean, *value, widths[*dst])?;
                    b.ins().store(MemFlags::trusted(), m, state_ptr, (*dst * 16) as i32);
                }
                PackedEffect::MemoryWrite { .. } => {
                    return Err(jit_err("JIT v1 does not support memory writes"))
                }
            }
        }
    }
    Ok(out)
}

/// Emit the tick_commit block: combinational instructions feeding memory writes,
/// then the writes themselves (branch-free: write to the clamped address the
/// value selected between the new data and the current contents).
fn emit_memwrites(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    cur: Value,
    is_reg: &[bool],
    block: &PackedBlock,
    widths: &[u32],
    mem: &MemLayout,
    // Multi-clock: `(mask, memory idx→clock-bit)`. A gated memory write happens
    // only when its clock's bit in `mask` is set. `None` = always write.
    clock_gate: Option<(Value, &HashMap<usize, u32>)>,
) -> Result<(), ErrorReport> {
    let mut vals: HashMap<usize, Value> = HashMap::new();
    let mut vw: HashMap<usize, u32> = HashMap::new();
    let mut clean: HashMap<usize, bool> = HashMap::new();
    for packet in &block.packets {
        for instr in &packet.instrs {
            let (v, c) = emit_instr(b, state_ptr, cur, is_reg, &vals, &vw, &clean, instr, mem)?;
            vals.insert(instr.dst.0, v);
            vw.insert(instr.dst.0, instr.ty.width);
            clean.insert(instr.dst.0, c);
        }
        for effect in &packet.effects {
            match effect {
                PackedEffect::MemoryWrite {
                    memory,
                    enable,
                    addr,
                    data,
                } => {
                    let depth = mem.depth[*memory];
                    if depth == 0 {
                        continue;
                    }
                    let en = get(&vals, *enable)?; // only bit 0 → no mask needed
                    let aw = vw.get(&addr.0).copied().unwrap_or(64);
                    let a_raw = get(&vals, *addr)?;
                    let a = if clean.get(&addr.0).copied().unwrap_or(false) {
                        a_raw
                    } else {
                        mask_to(b, a_raw, aw) // address must be clean for clamp/compare
                    };
                    let dw = mem.data_width[*memory];
                    let md = store_value(b, &vals, &vw, &clean, *data, dw)?;
                    let ptr = mem_elem_ptr(b, state_ptr, mem.base[*memory], depth, a);
                    let current = b.ins().load(val_ty(dw), MemFlags::trusted(), ptr, 0);
                    let en_bit = b.ins().band_imm(en, 1);
                    let en_b = b.ins().icmp_imm(IntCC::NotEqual, en_bit, 0);
                    let inb = b.ins().icmp_imm(IntCC::UnsignedLessThan, a, depth as i64);
                    let mut do_write = b.ins().band(en_b, inb);
                    // Clock-gate the write: AND in (mask >> clock_bit) & 1.
                    if let Some((mask, bit_of)) = clock_gate {
                        if let Some(&bit) = bit_of.get(memory) {
                            let shifted = b.ins().ushr_imm(mask, bit as i64);
                            let active = b.ins().band_imm(shifted, 1);
                            let cond = b.ins().icmp_imm(IntCC::NotEqual, active, 0);
                            do_write = b.ins().band(do_write, cond);
                        }
                    }
                    let newv = b.ins().select(do_write, md, current);
                    b.ins().store(MemFlags::trusted(), newv, ptr, 0);
                }
                PackedEffect::StoreSignal { dst, value } => {
                    let m = store_value(b, &vals, &vw, &clean, *value, widths[*dst])?;
                    b.ins().store(MemFlags::trusted(), m, state_ptr, (*dst * 16) as i32);
                }
                PackedEffect::CaptureReg { .. } => {
                    return Err(jit_err("unexpected register capture in tick_commit"))
                }
            }
        }
    }
    Ok(())
}

fn get(vals: &HashMap<usize, Value>, id: PackedValueId) -> Result<Value, ErrorReport> {
    vals.get(&id.0)
        .copied()
        .ok_or_else(|| jit_err(format!("value %{} used before def", id.0)))
}

/// Emit one instruction, returning its (width-masked) result value.
/// Emit one instruction. Returns `(value, clean)` where `clean` means the result
/// has no garbage above its width (upper bits are zero). Lazy masking: most ops
/// leave their result dirty and the mask is inserted only where a consumer needs
/// clean bits (compares, zext, addresses) or at a store — eliminating the bulk of
/// width-masks on arithmetic datapaths.
fn emit_instr(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    cur: Value,
    is_reg: &[bool],
    vals: &HashMap<usize, Value>,
    vw: &HashMap<usize, u32>,
    clean: &HashMap<usize, bool>,
    instr: &crate::PackedInstr,
    mem: &MemLayout,
) -> Result<(Value, bool), ErrorReport> {
    let w = instr.ty.width; // result width; result type = val_ty(w)
    let opw = |id: PackedValueId| vw.get(&id.0).copied().unwrap_or(64);
    let is_clean = |id: PackedValueId| clean.get(&id.0).copied().unwrap_or(false);
    // An operand is "clean to width w" (zero above w) iff it is clean and no
    // wider than w (a wider operand may hold valid bits in [w, opw)).
    let clean_to = |id: PackedValueId, tw: u32| is_clean(id) && opw(id) <= tw;
    // Fetch an operand promoted to `tw`'s type, masking only when the dirty upper
    // bits could corrupt the result. `strict` consumers (compare/zext/address)
    // always need clean inputs; arithmetic only when the operand is narrower than
    // the result (so its garbage lands inside the result range).
    let operand = |b: &mut FunctionBuilder, id: PackedValueId, tw: u32, strict: bool| -> Result<Value, ErrorReport> {
        let v = get(vals, id)?;
        let ow = opw(id);
        let need_mask = !is_clean(id) && (strict || ow < tw);
        let v = if need_mask { mask_to(b, v, ow) } else { v };
        Ok(promote(b, v, ow, tw))
    };
    let (v, c) = match &instr.kind {
        PackedInstrKind::Lit(words) => (emit_const(b, lit_u128(words) & mask128(w), w), true),
        PackedInstrKind::Signal(idx) => (
            b.ins().load(val_ty(w), MemFlags::trusted(), read_ptr(state_ptr, cur, is_reg, *idx), (*idx * 16) as i32),
            true,
        ),
        PackedInstrKind::Not(a) => {
            let x = operand(b, *a, w, false)?;
            (b.ins().bnot(x), false)
        }
        PackedInstrKind::And(a, c) => {
            let (x, y) = (operand(b, *a, w, false)?, operand(b, *c, w, false)?);
            (b.ins().band(x, y), clean_to(*a, w) && clean_to(*c, w))
        }
        PackedInstrKind::Or(a, c) => {
            let (x, y) = (operand(b, *a, w, false)?, operand(b, *c, w, false)?);
            (b.ins().bor(x, y), clean_to(*a, w) && clean_to(*c, w))
        }
        PackedInstrKind::Xor(a, c) => {
            let (x, y) = (operand(b, *a, w, false)?, operand(b, *c, w, false)?);
            (b.ins().bxor(x, y), clean_to(*a, w) && clean_to(*c, w))
        }
        PackedInstrKind::Add(a, c) => {
            let (x, y) = (operand(b, *a, w, false)?, operand(b, *c, w, false)?);
            (b.ins().iadd(x, y), false)
        }
        PackedInstrKind::Sub(a, c) => {
            let (x, y) = (operand(b, *a, w, false)?, operand(b, *c, w, false)?);
            (b.ins().isub(x, y), false)
        }
        PackedInstrKind::Mul(a, c) => {
            let (x, y) = (operand(b, *a, w, false)?, operand(b, *c, w, false)?);
            (b.ins().imul(x, y), false)
        }
        PackedInstrKind::Eq(a, c) => {
            let cw = opw(*a).max(opw(*c));
            let (x, y) = (operand(b, *a, cw, true)?, operand(b, *c, cw, true)?);
            let cmp = b.ins().icmp(IntCC::Equal, x, y);
            (b.ins().uextend(I64, cmp), true)
        }
        PackedInstrKind::Ne(a, c) => {
            let cw = opw(*a).max(opw(*c));
            let (x, y) = (operand(b, *a, cw, true)?, operand(b, *c, cw, true)?);
            let cmp = b.ins().icmp(IntCC::NotEqual, x, y);
            (b.ins().uextend(I64, cmp), true)
        }
        PackedInstrKind::Lt { lhs, rhs, signed } => {
            let cw = opw(*lhs).max(opw(*rhs));
            let cmp = if *signed {
                // sign_extend_to extracts the low `width` bits, so dirty input is fine.
                let x = sign_extend_to(b, get(vals, *lhs)?, opw(*lhs), cw);
                let y = sign_extend_to(b, get(vals, *rhs)?, opw(*rhs), cw);
                b.ins().icmp(IntCC::SignedLessThan, x, y)
            } else {
                let (x, y) = (operand(b, *lhs, cw, true)?, operand(b, *rhs, cw, true)?);
                b.ins().icmp(IntCC::UnsignedLessThan, x, y)
            };
            (b.ins().uextend(I64, cmp), true)
        }
        PackedInstrKind::Mux { cond, then_value, else_value } => {
            let c = get(vals, *cond)?; // only bit 0 matters → tolerant
            let bit = b.ins().band_imm(c, 1);
            let sel = b.ins().icmp_imm(IntCC::NotEqual, bit, 0);
            let (t, e) = (operand(b, *then_value, w, false)?, operand(b, *else_value, w, false)?);
            (
                b.ins().select(sel, t, e),
                clean_to(*then_value, w) && clean_to(*else_value, w),
            )
        }
        PackedInstrKind::Slice { value, lsb } => {
            // Right shift brings bits down; reading within the operand width, so a
            // dirty input only dirties the result above `w`. Clean iff the slice
            // reaches the operand's top (no valid bits left above the result).
            let ow = opw(*value);
            let x = get(vals, *value)?;
            let shifted = if *lsb == 0 { x } else { b.ins().ushr_imm(x, *lsb as i64) };
            (promote(b, shifted, ow, w), is_clean(*value) && *lsb + w >= ow)
        }
        PackedInstrKind::Zext(a) => {
            // Zero-extend must see clean input so the high bits are truly zero.
            (operand(b, *a, w, true)?, true)
        }
        PackedInstrKind::Trunc(a) => (operand(b, *a, w, false)?, false),
        PackedInstrKind::Cast(a) => (operand(b, *a, w, false)?, is_clean(*a) && opw(*a) <= w),
        PackedInstrKind::Sext(a) => {
            (sign_extend_to(b, get(vals, *a)?, opw(*a), w), false)
        }
        PackedInstrKind::Concat(parts) => {
            // MSB-first: last part is the LSB. Parts are masked, so result is clean.
            let mut acc = emit_const(b, 0, w);
            let mut offset = 0u32;
            for part in parts.iter().rev() {
                let pw = opw(*part);
                let pm = mask_to(b, get(vals, *part)?, pw);
                let pp = promote(b, pm, pw, w);
                let shifted = if offset == 0 { pp } else { b.ins().ishl_imm(pp, offset as i64) };
                acc = b.ins().bor(acc, shifted);
                offset += pw;
            }
            (acc, true)
        }
        PackedInstrKind::MemRead { memory, addr } => {
            let depth = mem.depth[*memory];
            if depth == 0 {
                (emit_const(b, 0, w), true)
            } else {
                let a = operand(b, *addr, opw(*addr), true)?; // address must be clean
                let ptr = mem_elem_ptr(b, state_ptr, mem.base[*memory], depth, a);
                let loaded = b.ins().load(val_ty(w), MemFlags::trusted(), ptr, 0);
                let inb = b.ins().icmp_imm(IntCC::UnsignedLessThan, a, depth as i64);
                let zero = emit_const(b, 0, w);
                (b.ins().select(inb, loaded, zero), true)
            }
        }
    };
    // A/B toggle (default true): when false, mask every result eagerly and mark
    // it clean — reproducing the pre-optimization behavior for measurement.
    if LAZY_MASK {
        Ok((v, c))
    } else {
        Ok((mask_to(b, v, w), true))
    }
}

/// When false, the JIT masks after every op (old behavior). True = lazy masking
/// (mask only at clean-requiring consumers and stores).
const LAZY_MASK: bool = std::option_env!("JIT_EAGER_MASK").is_none();

/// Fetch `value` promoted to `dst_w`'s type and masked to `dst_w` only if it is
/// dirty or wider than `dst_w` — for store/commit sites that keep state clean.
fn store_value(
    b: &mut FunctionBuilder,
    vals: &HashMap<usize, Value>,
    vw: &HashMap<usize, u32>,
    clean: &HashMap<usize, bool>,
    id: PackedValueId,
    dst_w: u32,
) -> Result<Value, ErrorReport> {
    let v = get(vals, id)?;
    let vwd = vw.get(&id.0).copied().unwrap_or(64);
    let is_clean = clean.get(&id.0).copied().unwrap_or(false);
    let p = promote(b, v, vwd, dst_w);
    Ok(if is_clean && vwd <= dst_w { p } else { mask_to(b, p, dst_w) })
}

/// Sign-extend the low `from_w` bits of `v` to width `to_w`'s type. The caller
/// masks afterward; the sign bits above `to_w` (if any) are cleared by that mask.
fn sign_extend_to(b: &mut FunctionBuilder, v: Value, from_w: u32, to_w: u32) -> Value {
    let p = promote(b, v, from_w, to_w);
    let bits = if to_w <= 64 { 64 } else { 128 };
    if from_w >= bits {
        return p;
    }
    let shift = (bits - from_w) as i64;
    let up = b.ins().ishl_imm(p, shift);
    b.ins().sshr_imm(up, shift)
}

// ===========================================================================
// Activity ("event-driven") skipping path.
//
// Instead of one straight-line settle of every signal each cycle, each
// combinational signal and register becomes a *cone*: its defining expression
// is re-emitted under a guard that recomputes it only when one of its direct
// fan-in signals changed value since the cone last ran (compared against a
// per-cone snapshot). Cones are emitted in topological stream order, so a
// dirty input has already been refreshed in state before its consumers read it;
// a skipped cone leaves its (unchanged) output and snapshot in place. Lagging
// snapshots (updated to the *pre-evaluation* value only when the cone runs)
// make a self-feeding register — a counter — re-evaluate every cycle, as it
// should, while genuinely stable cones are skipped. Bit-exact with the oblivious
// path; the question it answers is whether native branches make the skip pay at
// single-instance latency (where the SIMD path's per-lane overhead sank it).
// ===========================================================================

/// Control + snapshot region offsets (bytes), all after signals + memory.
struct ActivityLayout {
    seen_off: i32,
    skips_off: i32,
    total_off: i32,
    snap_start: usize,
}

/// The direct fan-in (signals) a guarded cone watches, plus whether it reads a
/// memory (memory-reading cones can't be value-guarded, so they always run).
fn op_guard_reads(op: &PackedOp) -> Option<(Vec<usize>, bool)> {
    match op {
        PackedOp::Assign { expr, .. } => Some(expr_direct_reads(expr)),
        PackedOp::CaptureReg { next, reset, .. } => {
            let (mut reads, m) = expr_direct_reads(next);
            if let Some(r) = reset {
                reads.push(r.signal);
            }
            reads.sort_unstable();
            reads.dedup();
            Some((reads, m))
        }
        PackedOp::MemoryWrite { .. } => None,
    }
}

/// Total snapshot slots needed: one per (guarded comb Assign, guarded register
/// CaptureReg) fan-in signal. Must match the emission walk exactly.
fn activity_snap_reads(source: &PackedProgram) -> usize {
    let mut n = 0;
    for packet in &source.streams.comb {
        for op in &packet.ops {
            if let PackedOp::Assign { .. } = op {
                if let Some((reads, false)) = op_guard_reads(op) {
                    n += reads.len();
                }
            }
        }
    }
    for packet in &source.streams.tick_next {
        for op in &packet.ops {
            if let PackedOp::CaptureReg { .. } = op {
                if let Some((reads, false)) = op_guard_reads(op) {
                    n += reads.len();
                }
            }
        }
    }
    n
}

/// `state[off] += 1` (an `i64` counter).
fn bump(b: &mut FunctionBuilder, state_ptr: Value, off: i32) {
    let v = b.ins().load(I64, MemFlags::trusted(), state_ptr, off);
    let v1 = b.ins().iadd_imm(v, 1);
    b.ins().store(MemFlags::trusted(), v1, state_ptr, off);
}

/// Emit the dirty guard for a cone: returns `(active, current_reads)` where
/// `active` is an I8 bool (`force` OR any read != its snapshot) and
/// `current_reads` are the loaded current values (to store into the snapshot
/// inside the active block).
fn emit_guard(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    reads: &[usize],
    snap_base: usize,
    force: Value,
    widths: &[u32],
) -> (Value, Vec<Value>) {
    let mut active = force;
    let mut curs = Vec::with_capacity(reads.len());
    for (k, &sig) in reads.iter().enumerate() {
        let ty = val_ty(widths[sig]);
        let cur = b.ins().load(ty, MemFlags::trusted(), state_ptr, (sig * 16) as i32);
        let snap = b.ins().load(ty, MemFlags::trusted(), state_ptr, (snap_base + k * 16) as i32);
        let ne = b.ins().icmp(IntCC::NotEqual, cur, snap);
        active = b.ins().bor(active, ne);
        curs.push(cur);
    }
    (active, curs)
}

/// Store the freshly-loaded current read values into the cone's snapshot slots.
fn store_snapshot(b: &mut FunctionBuilder, state_ptr: Value, reads: &[usize], snap_base: usize, curs: &[Value]) {
    for (k, &cur) in curs.iter().enumerate() {
        let _ = reads;
        b.ins().store(MemFlags::trusted(), cur, state_ptr, (snap_base + k * 16) as i32);
    }
}

/// Lower a tree-IR expression to a Cranelift value of `expr.width()`'s type,
/// reading fan-in signals from state. Type-aware (I64 / I128), masked to width.
fn emit_expr(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    expr: &PackedExpr,
    widths: &[u32],
    mem: &MemLayout,
) -> Result<Value, ErrorReport> {
    let w = expr.ty.width;
    // Lower a sub-expression and promote it to width `tw`'s type.
    fn operand(
        b: &mut FunctionBuilder,
        state_ptr: Value,
        e: &PackedExpr,
        tw: u32,
        widths: &[u32],
        mem: &MemLayout,
    ) -> Result<Value, ErrorReport> {
        let v = emit_expr(b, state_ptr, e, widths, mem)?;
        Ok(promote(b, v, e.ty.width, tw))
    }
    let v = match &expr.kind {
        PackedExprKind::Lit(words) => emit_const(b, lit_u128(words), w),
        PackedExprKind::Signal(idx) => {
            b.ins().load(val_ty(w), MemFlags::trusted(), state_ptr, (*idx * 16) as i32)
        }
        PackedExprKind::Not(a) => {
            let x = operand(b, state_ptr, a, w, widths, mem)?;
            b.ins().bnot(x)
        }
        PackedExprKind::And(a, c) => {
            let (x, y) = (operand(b, state_ptr, a, w, widths, mem)?, operand(b, state_ptr, c, w, widths, mem)?);
            b.ins().band(x, y)
        }
        PackedExprKind::Or(a, c) => {
            let (x, y) = (operand(b, state_ptr, a, w, widths, mem)?, operand(b, state_ptr, c, w, widths, mem)?);
            b.ins().bor(x, y)
        }
        PackedExprKind::Xor(a, c) => {
            let (x, y) = (operand(b, state_ptr, a, w, widths, mem)?, operand(b, state_ptr, c, w, widths, mem)?);
            b.ins().bxor(x, y)
        }
        PackedExprKind::Add(a, c) => {
            let (x, y) = (operand(b, state_ptr, a, w, widths, mem)?, operand(b, state_ptr, c, w, widths, mem)?);
            b.ins().iadd(x, y)
        }
        PackedExprKind::Sub(a, c) => {
            let (x, y) = (operand(b, state_ptr, a, w, widths, mem)?, operand(b, state_ptr, c, w, widths, mem)?);
            b.ins().isub(x, y)
        }
        PackedExprKind::Mul(a, c) => {
            let (x, y) = (operand(b, state_ptr, a, w, widths, mem)?, operand(b, state_ptr, c, w, widths, mem)?);
            b.ins().imul(x, y)
        }
        PackedExprKind::Eq(a, c) => {
            let cw = a.ty.width.max(c.ty.width);
            let (x, y) = (operand(b, state_ptr, a, cw, widths, mem)?, operand(b, state_ptr, c, cw, widths, mem)?);
            let cmp = b.ins().icmp(IntCC::Equal, x, y);
            b.ins().uextend(I64, cmp)
        }
        PackedExprKind::Ne(a, c) => {
            let cw = a.ty.width.max(c.ty.width);
            let (x, y) = (operand(b, state_ptr, a, cw, widths, mem)?, operand(b, state_ptr, c, cw, widths, mem)?);
            let cmp = b.ins().icmp(IntCC::NotEqual, x, y);
            b.ins().uextend(I64, cmp)
        }
        PackedExprKind::Lt { lhs, rhs, signed } => {
            let cw = lhs.ty.width.max(rhs.ty.width);
            let cmp = if *signed {
                let x = emit_expr(b, state_ptr, lhs, widths, mem)?;
                let x = sign_extend_to(b, x, lhs.ty.width, cw);
                let y = emit_expr(b, state_ptr, rhs, widths, mem)?;
                let y = sign_extend_to(b, y, rhs.ty.width, cw);
                b.ins().icmp(IntCC::SignedLessThan, x, y)
            } else {
                let (x, y) = (operand(b, state_ptr, lhs, cw, widths, mem)?, operand(b, state_ptr, rhs, cw, widths, mem)?);
                b.ins().icmp(IntCC::UnsignedLessThan, x, y)
            };
            b.ins().uextend(I64, cmp)
        }
        PackedExprKind::Mux { cond, then_expr, else_expr } => {
            let c = emit_expr(b, state_ptr, cond, widths, mem)?;
            let bit = b.ins().band_imm(c, 1);
            let sel = b.ins().icmp_imm(IntCC::NotEqual, bit, 0);
            let t = operand(b, state_ptr, then_expr, w, widths, mem)?;
            let e = operand(b, state_ptr, else_expr, w, widths, mem)?;
            b.ins().select(sel, t, e)
        }
        PackedExprKind::Slice { expr: a, lsb } => {
            let ow = a.ty.width;
            let x = emit_expr(b, state_ptr, a, widths, mem)?;
            let shifted = if *lsb == 0 { x } else { b.ins().ushr_imm(x, *lsb as i64) };
            promote(b, shifted, ow, w)
        }
        PackedExprKind::Zext(a) | PackedExprKind::Trunc(a) | PackedExprKind::Cast(a) => {
            operand(b, state_ptr, a, w, widths, mem)?
        }
        PackedExprKind::Sext(a) => {
            let x = emit_expr(b, state_ptr, a, widths, mem)?;
            sign_extend_to(b, x, a.ty.width, w)
        }
        PackedExprKind::Concat(parts) => {
            let mut acc = emit_const(b, 0, w);
            let mut offset = 0u32;
            for part in parts.iter().rev() {
                let pw = part.ty.width;
                let pv = emit_expr(b, state_ptr, part, widths, mem)?;
                let pm = mask_to(b, pv, pw);
                let pp = promote(b, pm, pw, w);
                let shifted = if offset == 0 { pp } else { b.ins().ishl_imm(pp, offset as i64) };
                acc = b.ins().bor(acc, shifted);
                offset += pw;
            }
            acc
        }
        PackedExprKind::MemRead { memory, addr } => {
            let depth = mem.depth[*memory];
            if depth == 0 {
                emit_const(b, 0, w)
            } else {
                let a0 = emit_expr(b, state_ptr, addr, widths, mem)?;
                let a = promote(b, a0, addr.ty.width, 64);
                let ptr = mem_elem_ptr(b, state_ptr, mem.base[*memory], depth, a);
                let loaded = b.ins().load(val_ty(w), MemFlags::trusted(), ptr, 0);
                let inb = b.ins().icmp_imm(IntCC::UnsignedLessThan, a, depth as i64);
                let zero = emit_const(b, 0, w);
                b.ins().select(inb, loaded, zero)
            }
        }
    };
    Ok(mask_to(b, v, w))
}

/// Store `expr` masked to signal `dst`'s width into its state slot.
fn emit_assign_store(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    dst: usize,
    expr: &PackedExpr,
    widths: &[u32],
    mem: &MemLayout,
) -> Result<(), ErrorReport> {
    let v = emit_expr(b, state_ptr, expr, widths, mem)?;
    let dw = widths[dst];
    let p = promote(b, v, expr.ty.width, dw);
    let m = mask_to(b, p, dw);
    b.ins().store(MemFlags::trusted(), m, state_ptr, (dst * 16) as i32);
    Ok(())
}

/// The async-reset immediate conditional store (settle streams): while reset is
/// asserted, force the register to its reset value this cycle.
fn emit_async_store(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    dst: usize,
    reset: &crate::PackedReset,
    widths: &[u32],
) -> Result<(), ErrorReport> {
    let dw = widths[dst];
    let asserted = reset_asserted(b, state_ptr, state_ptr, &[], reset);
    let rval = emit_const(b, lit_u128(&reset.value), dw);
    let current = b.ins().load(val_ty(dw), MemFlags::trusted(), state_ptr, (dst * 16) as i32);
    let newv = b.ins().select(asserted, rval, current);
    let masked = mask_to(b, newv, dw);
    b.ins().store(MemFlags::trusted(), masked, state_ptr, (dst * 16) as i32);
    Ok(())
}

/// Compute a register's next-state value (with the synchronous reset mux), of
/// the register's width's type. The caller masks/commits.
fn emit_next_value(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    next: &PackedExpr,
    reset: &Option<crate::PackedReset>,
    dw: u32,
    widths: &[u32],
    mem: &MemLayout,
) -> Result<Value, ErrorReport> {
    let raw = emit_expr(b, state_ptr, next, widths, mem)?;
    let nx = promote(b, raw, next.ty.width, dw);
    Ok(match reset {
        Some(reset) => {
            let asserted = reset_asserted(b, state_ptr, state_ptr, &[], reset);
            let rval = emit_const(b, lit_u128(&reset.value), dw);
            b.ins().select(asserted, rval, nx)
        }
        None => nx,
    })
}

/// A memory write (tick_commit): branch-free clamp+select to the addressed entry.
fn emit_mem_write(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    memory: usize,
    enable: &PackedExpr,
    addr: &PackedExpr,
    data: &PackedExpr,
    mem: &MemLayout,
    widths: &[u32],
) -> Result<(), ErrorReport> {
    let depth = mem.depth[memory];
    if depth == 0 {
        return Ok(());
    }
    let en = emit_expr(b, state_ptr, enable, widths, mem)?;
    let a0 = emit_expr(b, state_ptr, addr, widths, mem)?;
    let a = promote(b, a0, addr.ty.width, 64);
    let d = emit_expr(b, state_ptr, data, widths, mem)?;
    let dw = mem.data_width[memory];
    let ptr = mem_elem_ptr(b, state_ptr, mem.base[memory], depth, a);
    let current = b.ins().load(val_ty(dw), MemFlags::trusted(), ptr, 0);
    let en_bit = b.ins().band_imm(en, 1);
    let en_b = b.ins().icmp_imm(IntCC::NotEqual, en_bit, 0);
    let inb = b.ins().icmp_imm(IntCC::UnsignedLessThan, a, depth as i64);
    let do_write = b.ins().band(en_b, inb);
    let pd = promote(b, d, data.ty.width, dw);
    let md = mask_to(b, pd, dw);
    let newv = b.ins().select(do_write, md, current);
    b.ins().store(MemFlags::trusted(), newv, ptr, 0);
    Ok(())
}

/// Emit a combinational settle with activity skipping: async-reset stores
/// (always run), then guarded comb cones. `snap` is advanced over the comb
/// cones' snapshot slots; the caller resets it to `lay.snap_start` before the
/// post-commit re-settle so both settles share the comb snapshots.
fn emit_settle(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    source: &PackedProgram,
    widths: &[u32],
    mem: &MemLayout,
    lay: &ActivityLayout,
    force: Value,
    snap: &mut usize,
    counters: bool,
) -> Result<(), ErrorReport> {
    for packet in &source.streams.async_reset_comb {
        for op in &packet.ops {
            match op {
                PackedOp::Assign { dst, expr } => emit_assign_store(b, state_ptr, *dst, expr, widths, mem)?,
                PackedOp::CaptureReg { dst, reset, .. } => {
                    let Some(reset) = reset else {
                        return Err(jit_err("reset-less capture in async_reset_comb"));
                    };
                    emit_async_store(b, state_ptr, *dst, reset, widths)?;
                }
                PackedOp::MemoryWrite { .. } => return Err(jit_err("memory write in async_reset_comb")),
            }
        }
    }
    for packet in &source.streams.comb {
        for op in &packet.ops {
            match op {
                PackedOp::Assign { dst, expr } => {
                    let (reads, reads_mem) = op_guard_reads(op).unwrap();
                    if reads_mem {
                        emit_assign_store(b, state_ptr, *dst, expr, widths, mem)?;
                        continue;
                    }
                    let snap_base = *snap;
                    *snap += reads.len() * 16;
                    if counters {
                        bump(b, state_ptr, lay.total_off);
                    }
                    let (active, curs) = emit_guard(b, state_ptr, &reads, snap_base, force, widths);
                    let abk = b.create_block();
                    let sk = b.create_block();
                    let cont = b.create_block();
                    b.ins().brif(active, abk, &[], sk, &[]);
                    b.switch_to_block(abk);
                    store_snapshot(b, state_ptr, &reads, snap_base, &curs);
                    emit_assign_store(b, state_ptr, *dst, expr, widths, mem)?;
                    b.ins().jump(cont, &[]);
                    b.switch_to_block(sk);
                    if counters {
                        bump(b, state_ptr, lay.skips_off);
                    }
                    b.ins().jump(cont, &[]);
                    b.switch_to_block(cont);
                }
                PackedOp::CaptureReg { dst, reset, .. } => {
                    let Some(reset) = reset else {
                        return Err(jit_err("reset-less capture in comb"));
                    };
                    emit_async_store(b, state_ptr, *dst, reset, widths)?;
                }
                PackedOp::MemoryWrite { .. } => return Err(jit_err("memory write in comb")),
            }
        }
    }
    Ok(())
}

/// Emit one cycle with activity skipping: settle → capture → memwrites →
/// commit → settle again (the second settle refreshes combinational outputs of
/// the registers just committed, matching the oblivious tick body).
fn emit_activity_tick(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    source: &PackedProgram,
    widths: &[u32],
    mem: &MemLayout,
    lay: &ActivityLayout,
    counters: bool,
) -> Result<(), ErrorReport> {
    // First-ever tick forces every cone to evaluate (snapshots/state are zero).
    let seen = b.ins().load(I64, MemFlags::trusted(), state_ptr, lay.seen_off);
    let force = b.ins().icmp_imm(IntCC::Equal, seen, 0);
    let mut snap = lay.snap_start;

    emit_settle(b, state_ptr, source, widths, mem, lay, force, &mut snap, counters)?;

    // --- capture: guarded register cones, committed after all are computed ---
    let mut caps: Vec<(usize, Value)> = Vec::new();
    for packet in &source.streams.tick_next {
        for op in &packet.ops {
            match op {
                PackedOp::CaptureReg { dst, next, reset } => {
                    let dw = widths[*dst];
                    let (reads, reads_mem) = op_guard_reads(op).unwrap();
                    if reads_mem {
                        let v = emit_next_value(b, state_ptr, next, reset, dw, widths, mem)?;
                        caps.push((*dst, v));
                        continue;
                    }
                    let snap_base = snap;
                    snap += reads.len() * 16;
                    if counters {
                        bump(b, state_ptr, lay.total_off);
                    }
                    let (active, curs) = emit_guard(b, state_ptr, &reads, snap_base, force, widths);
                    let abk = b.create_block();
                    let sk = b.create_block();
                    let cont = b.create_block();
                    b.append_block_param(cont, val_ty(dw));
                    b.ins().brif(active, abk, &[], sk, &[]);
                    b.switch_to_block(abk);
                    store_snapshot(b, state_ptr, &reads, snap_base, &curs);
                    let nv = emit_next_value(b, state_ptr, next, reset, dw, widths, mem)?;
                    b.ins().jump(cont, &[nv]);
                    b.switch_to_block(sk);
                    if counters {
                        bump(b, state_ptr, lay.skips_off);
                    }
                    let cur = b.ins().load(val_ty(dw), MemFlags::trusted(), state_ptr, (*dst * 16) as i32);
                    b.ins().jump(cont, &[cur]);
                    b.switch_to_block(cont);
                    let merged = b.block_params(cont)[0];
                    caps.push((*dst, merged));
                }
                PackedOp::Assign { dst, expr } => emit_assign_store(b, state_ptr, *dst, expr, widths, mem)?,
                PackedOp::MemoryWrite { .. } => return Err(jit_err("memory write in tick_next")),
            }
        }
    }

    // --- memory writes (always run) ---
    for packet in &source.streams.tick_commit {
        for op in &packet.ops {
            match op {
                PackedOp::MemoryWrite { memory, enable, addr, data } => {
                    emit_mem_write(b, state_ptr, *memory, enable, addr, data, mem, widths)?;
                }
                PackedOp::Assign { dst, expr } => emit_assign_store(b, state_ptr, *dst, expr, widths, mem)?,
                PackedOp::CaptureReg { .. } => return Err(jit_err("capture in tick_commit")),
            }
        }
    }

    // --- commit register next-states ---
    for (dst, v) in caps {
        let masked = mask_to(b, v, widths[dst]);
        b.ins().store(MemFlags::trusted(), masked, state_ptr, (dst * 16) as i32);
    }

    // --- re-settle: refresh combinational outputs of the committed registers.
    // Reuses the comb cones' snapshots (cones reading a changed register see the
    // post-commit value differ from the snapshot the first settle just wrote).
    let mut snap2 = lay.snap_start;
    emit_settle(b, state_ptr, source, widths, mem, lay, force, &mut snap2, counters)?;

    // Mark initialized.
    let one = b.ins().iconst(I64, 1);
    b.ins().store(MemFlags::trusted(), one, state_ptr, lay.seen_off);
    Ok(())
}

// ===========================================================================
// Vectorized (SIMD) JIT: native code + lane parallelism on the same axis.
//
// The scalar `JitSimulator` compiles one design instance to straight-line code.
// This compiles the design *once* but every value is a 128-bit vector, so one
// native instruction stream advances several independent instances (distinct
// stimulus) per pass — unioning the JIT's no-dispatch codegen with the SIMD
// engine's lane amortization. State is structure-of-arrays per signal: each
// 16-byte slot holds that signal's lanes, so a whole signal is one aligned
// vector load. MIXED-WIDTH LANE PACKING: the element width (and thus lane count)
// is chosen per design by its widest signal — all-≤8-bit → I8X16 (16 lanes),
// ≤16-bit → I16X8 (8 lanes), ≤32-bit → I32X4 (4 lanes). Scope: signals/mem-data
// ≤ 32 bits, synchronous reset.
// ===========================================================================

/// Vector packing for a design: element type, scalar element, element bit width,
/// and lanes per 128-bit vector (= instances advanced per pass).
#[derive(Clone, Copy)]
struct VecCfg {
    ty: Type,
    elem: Type,
    bits: u32,
    lanes: usize,
}

impl VecCfg {
    /// Pick the widest packing (most lanes) that fits the design's max width.
    fn for_max_width(maxw: u32) -> Option<Self> {
        Some(match maxw {
            0..=8 => VecCfg { ty: I8X16, elem: I8, bits: 8, lanes: 16 },
            9..=16 => VecCfg { ty: I16X8, elem: I16, bits: 16, lanes: 8 },
            17..=32 => VecCfg { ty: I32X4, elem: I32, bits: 32, lanes: 4 },
            _ => return None,
        })
    }
    fn elem_bytes(&self) -> usize {
        (self.bits / 8) as usize
    }
}

/// One signal's lanes, 16-byte aligned (one 128-bit vector) so loads/stores align.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct VSlot([u8; 16]);

fn width_mask32(width: u32) -> u32 {
    if width >= 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    }
}

/// Per-lane memory layout for the vector JIT: each memory entry is a 4-lane
/// `Lane4` slot. `base[m]` is the slot index of memory m's entry 0.
#[derive(Default)]
struct VecMemLayout {
    base: Vec<usize>,
    depth: Vec<usize>,
    data_width: Vec<u32>,
}

/// Pointer to the single element cell `mem[base + clamp(addr, depth-1)][lane]`.
/// `addr` is an `i64` lane address; clamping keeps it in-bounds (the caller
/// `select`s against `addr < depth`). Slot stride is 16 bytes (one vector); lane
/// `l` is at `+l*elem_bytes`.
fn vec_mem_lane_ptr(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    base: usize,
    depth: usize,
    addr: Value,
    lane: usize,
    elem_bytes: usize,
) -> Value {
    let dm1 = b.ins().iconst(I64, (depth.saturating_sub(1)) as i64);
    let clamped = b.ins().umin(addr, dm1);
    let byte = b.ins().imul_imm(clamped, 16);
    let off = b.ins().iadd_imm(byte, (base * 16 + lane * elem_bytes) as i64);
    b.ins().iadd(state_ptr, off)
}

/// A multi-lane native simulator: native code (like [`JitSimulator`]) *and* lane
/// parallelism (like the SIMD interpreter), on one backend. The lane count is
/// chosen per design by its widest signal (mixed-width packing).
pub struct SimdJitSimulator {
    _module: JITModule,
    tick_fn: extern "C" fn(*mut u8, i64, i64),
    state: Vec<VSlot>,
    widths: Vec<u32>,
    num_signals: usize,
    cfg: VecCfg,
    /// Number of independent vector groups (so `lanes = groups * cfg.lanes`).
    groups: usize,
    /// `VSlot`s per group (signals + memory entries).
    slots_per_group: usize,
    /// Spread the (independent) groups across cores with rayon.
    parallel: bool,
}

impl SimdJitSimulator {
    /// Compile a design into a single-group vector tick.
    pub fn compile(machine: &PackedMachineProgram) -> Result<Self, ErrorReport> {
        Self::compile_lanes(machine, 1)
    }

    /// Compile an N-lane batch tick: `ceil(lanes/cfg.lanes)` independent vector
    /// groups advanced in a native group×cycle loop, so one call drives `lanes`
    /// instances of distinct stimulus. The native code complement to the SIMD
    /// interpreter as RRTL's CPU batch backend. Errors if out of scope (signals
    /// or memory data > 32 bits, or asynchronous resets).
    pub fn compile_lanes(machine: &PackedMachineProgram, lanes: usize) -> Result<Self, ErrorReport> {
        let source = &machine.source;
        let mut maxw = 1u32;
        for s in &source.signals {
            maxw = maxw.max(s.layout.width);
        }
        for m in &source.memories {
            maxw = maxw.max(m.data_layout.width).max(m.addr_width);
        }
        let cfg = VecCfg::for_max_width(maxw).ok_or_else(|| {
            jit_err(format!("SIMD JIT supports signals/memory ≤ 32 bits; design max is {maxw} bits"))
        })?;
        if source.streams.async_reset_comb.iter().any(|p| !p.ops.is_empty()) {
            return Err(jit_err("SIMD JIT does not support asynchronous resets yet"));
        }
        let groups = lanes.max(1).div_ceil(cfg.lanes);
        let widths: Vec<u32> = source.signals.iter().map(|s| s.layout.width).collect();
        let num_signals = source.signals.len();

        // Per-lane memory: each memory entry is its own `VSlot`, appended after
        // the signals. `base[m]` is the slot index of entry 0.
        let mut mem = VecMemLayout::default();
        let mut total_mem = 0usize;
        for m in &source.memories {
            mem.base.push(num_signals + total_mem);
            mem.depth.push(m.depth);
            mem.data_width.push(m.data_layout.width);
            total_mem += m.depth;
        }
        let slots_per_group = num_signals + total_mem;
        let state_len = slots_per_group * groups;

        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").map_err(|e| jit_err(e.to_string()))?;
        let isa_builder = cranelift_native::builder().map_err(|e| jit_err(e.to_string()))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| jit_err(e.to_string()))?;
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        // tick(state, n_groups, n_cycles): outer cycle loop, inner group loop.
        // `n_groups` is a runtime arg so a worker thread can process a sub-range
        // of the (group-major) state; each group runs the 4-lane kernel rooted at
        // its own base pointer (stride = slots_per_group * 16 bytes).
        let stride = (slots_per_group * 16) as i64;
        let tick_id = define_fn(&mut module, "tick_simd", 3, |b, p| {
            let (state_ptr, n_groups, n_cycles) = (p[0], p[1], p[2]);
            emit_loop(b, n_cycles, |b, _c| {
                emit_loop(b, n_groups, |b, g| {
                    let goff = b.ins().imul_imm(g, stride);
                    let base = b.ins().iadd(state_ptr, goff);
                    emit_vec_tick(b, base, source, &widths, &mem, cfg)
                })
            })?;
            b.ins().return_(&[]);
            Ok(())
        })?;

        module.finalize_definitions().map_err(|e| jit_err(e.to_string()))?;
        let tick_fn: extern "C" fn(*mut u8, i64, i64) =
            unsafe { mem::transmute(module.get_finalized_function(tick_id)) };

        Ok(Self {
            _module: module,
            tick_fn,
            state: vec![VSlot([0; 16]); state_len],
            widths,
            num_signals,
            cfg,
            groups,
            slots_per_group,
            parallel: groups > 1,
        })
    }

    /// Total lane capacity (`groups * cfg.lanes`).
    pub fn lanes(&self) -> usize {
        self.groups * self.cfg.lanes
    }

    pub fn set_signal(&mut self, lane: usize, index: usize, value: u32) {
        let (g, sub) = (lane / self.cfg.lanes, lane % self.cfg.lanes);
        let eb = self.cfg.elem_bytes();
        let v = (value & width_mask32(self.widths[index])) as u64;
        let bytes = v.to_le_bytes();
        let off = sub * eb;
        self.state[g * self.slots_per_group + index].0[off..off + eb].copy_from_slice(&bytes[..eb]);
    }

    pub fn get_signal(&self, lane: usize, index: usize) -> u32 {
        let (g, sub) = (lane / self.cfg.lanes, lane % self.cfg.lanes);
        let eb = self.cfg.elem_bytes();
        let off = sub * eb;
        let mut bytes = [0u8; 8];
        bytes[..eb].copy_from_slice(&self.state[g * self.slots_per_group + index].0[off..off + eb]);
        (u64::from_le_bytes(bytes) as u32) & width_mask32(self.widths[index])
    }

    /// Enable/disable spreading groups across cores (default: on when >1 group).
    pub fn set_parallel(&mut self, parallel: bool) {
        self.parallel = parallel;
    }

    pub fn tick(&mut self) {
        self.run(1);
    }

    /// Advance all lanes `n` cycles. With >1 group and `parallel`, the groups are
    /// split across cores (they are fully independent — no synchronization).
    pub fn tick_many(&mut self, n: usize) {
        self.run(n);
    }

    fn run(&mut self, cycles: usize) {
        let f = self.tick_fn;
        let spg = self.slots_per_group;
        let cy = cycles as i64;
        if self.parallel && self.groups > 1 {
            // ~4 tasks per core; each task drives a contiguous run of groups.
            let gpt = (self.groups / (rayon::current_num_threads() * 4)).max(1);
            self.state.par_chunks_mut(spg * gpt).for_each(|chunk| {
                let ng = (chunk.len() / spg) as i64;
                f(chunk.as_mut_ptr() as *mut u8, ng, cy);
            });
        } else {
            f(self.state.as_mut_ptr() as *mut u8, self.groups as i64, cy);
        }
    }

    pub fn num_signals(&self) -> usize {
        self.num_signals
    }
}

/// Splat a scalar element constant across all lanes.
fn vsplat(b: &mut FunctionBuilder, cfg: VecCfg, value: u32) -> Value {
    let s = b.ins().iconst(cfg.elem, value as i64);
    b.ins().splat(cfg.ty, s)
}

/// Mask each lane of `v` to its low `width` bits.
fn vmask_to(b: &mut FunctionBuilder, cfg: VecCfg, v: Value, width: u32) -> Value {
    if width >= cfg.bits {
        v
    } else {
        let m = vsplat(b, cfg, width_mask32(width));
        b.ins().band(v, m)
    }
}

/// Per-lane "is this reset asserted?" mask (all-ones / all-zero lanes).
fn reset_asserted_vec(b: &mut FunctionBuilder, state_ptr: Value, cfg: VecCfg, reset: &crate::PackedReset) -> Value {
    let rsig = b.ins().load(cfg.ty, MemFlags::trusted(), state_ptr, (reset.signal * 16) as i32);
    let one = vsplat(b, cfg, 1);
    let bit = b.ins().band(rsig, one);
    let zero = vsplat(b, cfg, 0);
    match reset.polarity {
        rrtl_ir::ResetPolarity::ActiveHigh => b.ins().icmp(IntCC::NotEqual, bit, zero),
        rrtl_ir::ResetPolarity::ActiveLow => b.ins().icmp(IntCC::Equal, bit, zero),
    }
}

/// Lower a tree-IR expression to a 4-lane `I32X4` value, masked to its width.
/// Fetch an arithmetic operand: mask only if it is dirty and *narrower* than the
/// result width `w` (so its garbage would land inside the result). Returns the
/// value and whether it is clean to `w` (zero above `w`). In the 32-bit `I32X4`
/// container any value of width ≥ 32 is automatically clean.
fn varith(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    e: &PackedExpr,
    w: u32,
    widths: &[u32],
    mem: &VecMemLayout,
    cfg: VecCfg,
) -> Result<(Value, bool), ErrorReport> {
    let (v, c) = emit_vec_expr(b, state_ptr, e, widths, mem, cfg)?;
    let ew = e.ty.width;
    let v = if !c && ew < w { vmask_to(b, cfg, v, ew) } else { v };
    Ok((v, w >= cfg.bits || ew < w || (ew == w && c)))
}

/// Fetch a strict operand (compare / zext / address): mask if dirty.
fn vstrict(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    e: &PackedExpr,
    widths: &[u32],
    mem: &VecMemLayout,
    cfg: VecCfg,
) -> Result<Value, ErrorReport> {
    let (v, c) = emit_vec_expr(b, state_ptr, e, widths, mem, cfg)?;
    Ok(if c { v } else { vmask_to(b, cfg, v, e.ty.width) })
}

/// Lower a tree-IR expression to a 4-lane `cfg.ty` value. Returns `(value, clean)`
/// — lazy masking: most ops leave garbage above their width and the mask is
/// inserted only at strict consumers and stores (cf. the scalar path).
fn emit_vec_expr(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    expr: &PackedExpr,
    widths: &[u32],
    mem: &VecMemLayout,
    cfg: VecCfg,
) -> Result<(Value, bool), ErrorReport> {
    let w = expr.ty.width;
    let (v, c) = match &expr.kind {
        PackedExprKind::Lit(words) => {
            (vsplat(b, cfg, words.first().copied().unwrap_or(0) & width_mask32(w)), true)
        }
        PackedExprKind::Signal(idx) => (
            b.ins().load(cfg.ty, MemFlags::trusted(), state_ptr, (*idx * 16) as i32),
            true,
        ),
        PackedExprKind::Not(a) => {
            let (x, _) = varith(b, state_ptr, a, w, widths, mem, cfg)?;
            (b.ins().bnot(x), w >= cfg.bits)
        }
        PackedExprKind::And(a, c) => {
            let (x, xc) = varith(b, state_ptr, a, w, widths, mem, cfg)?;
            let (y, yc) = varith(b, state_ptr, c, w, widths, mem, cfg)?;
            (b.ins().band(x, y), w >= cfg.bits || xc || yc)
        }
        PackedExprKind::Or(a, c) => {
            let (x, xc) = varith(b, state_ptr, a, w, widths, mem, cfg)?;
            let (y, yc) = varith(b, state_ptr, c, w, widths, mem, cfg)?;
            (b.ins().bor(x, y), w >= cfg.bits || (xc && yc))
        }
        PackedExprKind::Xor(a, c) => {
            let (x, xc) = varith(b, state_ptr, a, w, widths, mem, cfg)?;
            let (y, yc) = varith(b, state_ptr, c, w, widths, mem, cfg)?;
            (b.ins().bxor(x, y), w >= cfg.bits || (xc && yc))
        }
        PackedExprKind::Add(a, c) => {
            let (x, _) = varith(b, state_ptr, a, w, widths, mem, cfg)?;
            let (y, _) = varith(b, state_ptr, c, w, widths, mem, cfg)?;
            (b.ins().iadd(x, y), w >= cfg.bits)
        }
        PackedExprKind::Sub(a, c) => {
            let (x, _) = varith(b, state_ptr, a, w, widths, mem, cfg)?;
            let (y, _) = varith(b, state_ptr, c, w, widths, mem, cfg)?;
            (b.ins().isub(x, y), w >= cfg.bits)
        }
        PackedExprKind::Mul(a, c) => {
            let (x, _) = varith(b, state_ptr, a, w, widths, mem, cfg)?;
            let (y, _) = varith(b, state_ptr, c, w, widths, mem, cfg)?;
            (b.ins().imul(x, y), w >= cfg.bits)
        }
        PackedExprKind::Eq(a, c) => {
            let (x, y) = (vstrict(b, state_ptr, a, widths, mem, cfg)?, vstrict(b, state_ptr, c, widths, mem, cfg)?);
            let mask = b.ins().icmp(IntCC::Equal, x, y);
            let one = vsplat(b, cfg, 1);
            (b.ins().band(mask, one), true)
        }
        PackedExprKind::Ne(a, c) => {
            let (x, y) = (vstrict(b, state_ptr, a, widths, mem, cfg)?, vstrict(b, state_ptr, c, widths, mem, cfg)?);
            let mask = b.ins().icmp(IntCC::NotEqual, x, y);
            let one = vsplat(b, cfg, 1);
            (b.ins().band(mask, one), true)
        }
        PackedExprKind::Lt { lhs, rhs, signed } => {
            let mask = if *signed {
                // vsign_extend extracts the low `width` bits → dirty input is fine.
                let (xv, _) = emit_vec_expr(b, state_ptr, lhs, widths, mem, cfg)?;
                let x = vsign_extend(b, cfg, xv, lhs.ty.width);
                let (yv, _) = emit_vec_expr(b, state_ptr, rhs, widths, mem, cfg)?;
                let y = vsign_extend(b, cfg, yv, rhs.ty.width);
                b.ins().icmp(IntCC::SignedLessThan, x, y)
            } else {
                let (x, y) = (vstrict(b, state_ptr, lhs, widths, mem, cfg)?, vstrict(b, state_ptr, rhs, widths, mem, cfg)?);
                b.ins().icmp(IntCC::UnsignedLessThan, x, y)
            };
            let one = vsplat(b, cfg, 1);
            (b.ins().band(mask, one), true)
        }
        PackedExprKind::Mux { cond, then_expr, else_expr } => {
            let c = vstrict(b, state_ptr, cond, widths, mem, cfg)?; // garbage would skew the != 0 test
            let zero = vsplat(b, cfg, 0);
            let sel = b.ins().icmp(IntCC::NotEqual, c, zero);
            let (t, tc) = varith(b, state_ptr, then_expr, w, widths, mem, cfg)?;
            let (e, ec) = varith(b, state_ptr, else_expr, w, widths, mem, cfg)?;
            (b.ins().bitselect(sel, t, e), w >= cfg.bits || (tc && ec))
        }
        PackedExprKind::Slice { expr: a, lsb } => {
            let aw = a.ty.width;
            let (x, c) = emit_vec_expr(b, state_ptr, a, widths, mem, cfg)?;
            let shifted = if *lsb == 0 { x } else { b.ins().ushr_imm(x, *lsb as i64) };
            (shifted, w >= cfg.bits || (c && *lsb + w >= aw))
        }
        PackedExprKind::Zext(a) => (vstrict(b, state_ptr, a, widths, mem, cfg)?, true),
        PackedExprKind::Trunc(a) => {
            let (v, _) = emit_vec_expr(b, state_ptr, a, widths, mem, cfg)?;
            (v, w >= cfg.bits)
        }
        PackedExprKind::Cast(a) => emit_vec_expr(b, state_ptr, a, widths, mem, cfg)?,
        PackedExprKind::Sext(a) => {
            let (x, _) = emit_vec_expr(b, state_ptr, a, widths, mem, cfg)?;
            (vsign_extend(b, cfg, x, a.ty.width), w >= cfg.bits)
        }
        PackedExprKind::Concat(parts) => {
            let mut acc = vsplat(b, cfg, 0);
            let mut offset = 0u32;
            for part in parts.iter().rev() {
                let pw = part.ty.width;
                let (pv, _) = emit_vec_expr(b, state_ptr, part, widths, mem, cfg)?;
                let pm = vmask_to(b, cfg, pv, pw);
                let shifted = if offset == 0 { pm } else { b.ins().ishl_imm(pm, offset as i64) };
                acc = b.ins().bor(acc, shifted);
                offset += pw;
            }
            (acc, true)
        }
        PackedExprKind::MemRead { memory, addr } => {
            // Per-lane gather: each lane reads its own entry at its own address.
            // Cranelift has no portable vector gather, so loop the 4 lanes.
            let depth = mem.depth[*memory];
            if depth == 0 {
                (vsplat(b, cfg, 0), true)
            } else {
                let av = vstrict(b, state_ptr, addr, widths, mem, cfg)?; // address must be clean
                let base = mem.base[*memory];
                let mut result = vsplat(b, cfg, 0);
                for lane in 0..cfg.lanes {
                    let al = b.ins().extractlane(av, lane as u8);
                    let au = b.ins().uextend(I64, al);
                    let ptr = vec_mem_lane_ptr(b, state_ptr, base, depth, au, lane, cfg.elem_bytes());
                    let loaded = b.ins().load(cfg.elem, MemFlags::trusted(), ptr, 0);
                    let inb = b.ins().icmp_imm(IntCC::UnsignedLessThan, au, depth as i64);
                    let zero = b.ins().iconst(cfg.elem, 0);
                    let val = b.ins().select(inb, loaded, zero);
                    result = b.ins().insertlane(result, val, lane as u8);
                }
                (result, true)
            }
        }
    };
    // A/B toggle: when false, mask every result eagerly (pre-optimization behavior).
    if LAZY_MASK {
        Ok((v, c))
    } else {
        Ok((vmask_to(b, cfg, v, w), true))
    }
}

/// Evaluate `expr` and mask it to `dst_w` only when needed (dirty, or wider than
/// `dst_w`) — for store sites that must keep state slots clean.
fn vstore_value(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    expr: &PackedExpr,
    dst_w: u32,
    widths: &[u32],
    mem: &VecMemLayout,
    cfg: VecCfg,
) -> Result<Value, ErrorReport> {
    let (v, c) = emit_vec_expr(b, state_ptr, expr, widths, mem, cfg)?;
    let ew = expr.ty.width;
    Ok(if ew > dst_w {
        vmask_to(b, cfg, v, dst_w)
    } else if !c {
        vmask_to(b, cfg, v, ew) // ew < cfg.bits (32-bit values are always clean)
    } else {
        v
    })
}

/// Sign-extend each lane's low `width` bits across the full 32-bit lane.
fn vsign_extend(b: &mut FunctionBuilder, cfg: VecCfg, v: Value, width: u32) -> Value {
    if width >= cfg.bits {
        return v;
    }
    let shift = (cfg.bits - width) as i64;
    let up = b.ins().ishl_imm(v, shift);
    b.ins().sshr_imm(up, shift)
}

/// Emit one 4-lane cycle: settle comb, capture, write memories, commit, settle.
fn emit_vec_tick(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    source: &PackedProgram,
    widths: &[u32],
    mem: &VecMemLayout,
    cfg: VecCfg,
) -> Result<(), ErrorReport> {
    emit_vec_settle(b, state_ptr, source, widths, mem, cfg)?;

    let mut caps: Vec<(usize, Value)> = Vec::new();
    for packet in &source.streams.tick_next {
        for op in &packet.ops {
            match op {
                PackedOp::CaptureReg { dst, next, reset } => {
                    let dw = widths[*dst];
                    let nv = vstore_value(b, state_ptr, next, dw, widths, mem, cfg)?; // clean to dw
                    let nv = match reset {
                        Some(reset) => {
                            let asserted = reset_asserted_vec(b, state_ptr, cfg, reset);
                            let rval = vsplat(b, cfg, reset.value.first().copied().unwrap_or(0) & width_mask32(dw));
                            b.ins().bitselect(asserted, rval, nv)
                        }
                        None => nv,
                    };
                    caps.push((*dst, nv)); // already clean → committed directly
                }
                PackedOp::Assign { dst, expr } => {
                    let m = vstore_value(b, state_ptr, expr, widths[*dst], widths, mem, cfg)?;
                    b.ins().store(MemFlags::trusted(), m, state_ptr, (*dst * 16) as i32);
                }
                PackedOp::MemoryWrite { .. } => return Err(jit_err("memory write in tick_next")),
            }
        }
    }

    // tick_commit: comb feeding the writes, then the per-lane scatter writes.
    for packet in &source.streams.tick_commit {
        for op in &packet.ops {
            match op {
                PackedOp::Assign { dst, expr } => {
                    let m = vstore_value(b, state_ptr, expr, widths[*dst], widths, mem, cfg)?;
                    b.ins().store(MemFlags::trusted(), m, state_ptr, (*dst * 16) as i32);
                }
                PackedOp::MemoryWrite { memory, enable, addr, data } => {
                    emit_vec_mem_write(b, state_ptr, *memory, enable, addr, data, widths, mem, cfg)?;
                }
                PackedOp::CaptureReg { .. } => return Err(jit_err("capture in tick_commit")),
            }
        }
    }

    for (dst, v) in caps {
        b.ins().store(MemFlags::trusted(), v, state_ptr, (dst * 16) as i32);
    }

    emit_vec_settle(b, state_ptr, source, widths, mem, cfg)?;
    Ok(())
}

/// Emit the combinational settle (comb `Assign`s only; v1 has no async stores).
fn emit_vec_settle(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    source: &PackedProgram,
    widths: &[u32],
    mem: &VecMemLayout,
    cfg: VecCfg,
) -> Result<(), ErrorReport> {
    for packet in &source.streams.comb {
        for op in &packet.ops {
            match op {
                PackedOp::Assign { dst, expr } => {
                    let m = vstore_value(b, state_ptr, expr, widths[*dst], widths, mem, cfg)?;
                    b.ins().store(MemFlags::trusted(), m, state_ptr, (*dst * 16) as i32);
                }
                PackedOp::CaptureReg { .. } => {
                    return Err(jit_err("SIMD JIT does not support asynchronous resets yet"))
                }
                PackedOp::MemoryWrite { .. } => return Err(jit_err("memory write in comb")),
            }
        }
    }
    Ok(())
}

/// Per-lane scatter write: for each lane, `if enable & in-bounds` store its data
/// to its own entry. Branch-free clamp+select (mirrors the scalar JIT).
fn emit_vec_mem_write(
    b: &mut FunctionBuilder,
    state_ptr: Value,
    memory: usize,
    enable: &PackedExpr,
    addr: &PackedExpr,
    data: &PackedExpr,
    widths: &[u32],
    mem: &VecMemLayout,
    cfg: VecCfg,
) -> Result<(), ErrorReport> {
    let depth = mem.depth[memory];
    if depth == 0 {
        return Ok(());
    }
    let (env, _) = emit_vec_expr(b, state_ptr, enable, widths, mem, cfg)?; // only bit 0 per lane
    let av = vstrict(b, state_ptr, addr, widths, mem, cfg)?; // address must be clean
    let (dv, _) = emit_vec_expr(b, state_ptr, data, widths, mem, cfg)?; // masked per-lane below
    let dw = mem.data_width[memory];
    let base = mem.base[memory];
    for lane in 0..cfg.lanes {
        let al = b.ins().extractlane(av, lane as u8);
        let au = b.ins().uextend(I64, al);
        let enl = b.ins().extractlane(env, lane as u8);
        let mut dl = b.ins().extractlane(dv, lane as u8);
        if dw < cfg.bits {
            dl = b.ins().band_imm(dl, width_mask32(dw) as i64);
        }
        let ptr = vec_mem_lane_ptr(b, state_ptr, base, depth, au, lane, cfg.elem_bytes());
        let current = b.ins().load(cfg.elem, MemFlags::trusted(), ptr, 0);
        let en_bit = b.ins().band_imm(enl, 1);
        let en_b = b.ins().icmp_imm(IntCC::NotEqual, en_bit, 0);
        let inb = b.ins().icmp_imm(IntCC::UnsignedLessThan, au, depth as i64);
        let do_write = b.ins().band(en_b, inb);
        let newv = b.ins().select(do_write, dl, current);
        b.ins().store(MemFlags::trusted(), newv, ptr, 0);
    }
    Ok(())
}

/// Bit-parallel gate-level JIT: compiles an all-1-bit / all-bitwise design to a
/// native group×cycle loop where each group is one **`I64X2` vector = 128 lanes**,
/// and every gate is a 128-bit-wide bitwise op (`band`/`bor`/`bxor`/`bnot`). The
/// hand-vectorized complement to [`crate::bitparallel::BitParallelSimulator`] —
/// 128 lanes per op at native speed (vs the vector JIT's 16), matching the AOT's
/// clang-auto-vectorized throughput without the C compiler. Errors out unless the
/// design is gate-level (all 1-bit signals; only And/Or/Xor/Not/Mux/Signal/Lit;
/// no memory; no async reset) so the caller can fall back.
pub struct BitParallelJitSimulator {
    _module: JITModule,
    tick_fn: extern "C" fn(*mut u8, i64, i64),
    /// `num_signals * groups` 16-byte slots; signal `s` of group `g` (= an
    /// `I64X2` holding 128 lanes) at slot `g*num_signals + s`.
    state: Vec<VSlot>,
    num_signals: usize,
    groups: usize,
    lanes: usize,
}

impl BitParallelJitSimulator {
    pub fn compile(machine: &PackedMachineProgram) -> Result<Self, ErrorReport> {
        Self::compile_lanes(machine, 64)
    }

    /// Compile a `lanes`-lane bit-parallel batch tick (`ceil(lanes/64)` groups).
    pub fn compile_lanes(machine: &PackedMachineProgram, lanes: usize) -> Result<Self, ErrorReport> {
        let source = &machine.source;
        for s in &source.signals {
            if s.layout.width != 1 {
                return Err(jit_err(format!(
                    "bit-parallel JIT needs all 1-bit signals; `{}` is {} bits",
                    s.name, s.layout.width
                )));
            }
        }
        if !source.memories.is_empty() {
            return Err(jit_err("bit-parallel JIT does not support memories"));
        }
        if machine.streams.async_reset_comb.packets.iter().any(|p| !p.effects.is_empty()) {
            return Err(jit_err("bit-parallel JIT does not support asynchronous resets yet"));
        }
        for blk in [&machine.streams.comb, &machine.streams.tick_next] {
            for pkt in &blk.packets {
                for instr in &pkt.instrs {
                    use PackedInstrKind::*;
                    if !matches!(
                        instr.kind,
                        And(..) | Or(..) | Xor(..) | Not(..) | Mux { .. } | Signal(_) | Lit(_)
                    ) {
                        return Err(jit_err("bit-parallel JIT: design has a non-gate-level op"));
                    }
                }
            }
        }
        let num_signals = source.signals.len();
        let groups = lanes.max(1).div_ceil(128);
        let stride = (num_signals * 16) as i64; // one I64X2 (16 bytes) per signal/group

        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").map_err(|e| jit_err(e.to_string()))?;
        let isa = cranelift_native::builder()
            .map_err(|e| jit_err(e.to_string()))?
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| jit_err(e.to_string()))?;
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        // tick(state, n_groups, n_cycles): outer cycle, inner group; each group is
        // one I64X2 (128 lanes) rooted at base = state + g*stride.
        let tick_id = define_fn(&mut module, "tick_bp", 3, |b, p| {
            let (state_ptr, n_groups, n_cycles) = (p[0], p[1], p[2]);
            emit_loop(b, n_cycles, |b, _c| {
                emit_loop(b, n_groups, |b, g| {
                    let goff = b.ins().imul_imm(g, stride);
                    let base = b.ins().iadd(state_ptr, goff);
                    emit_bp_tick(b, base, machine)
                })
            })?;
            b.ins().return_(&[]);
            Ok(())
        })?;
        module.finalize_definitions().map_err(|e| jit_err(e.to_string()))?;
        let tick_fn: extern "C" fn(*mut u8, i64, i64) =
            unsafe { mem::transmute(module.get_finalized_function(tick_id)) };

        Ok(Self {
            _module: module,
            tick_fn,
            state: vec![VSlot([0; 16]); num_signals * groups],
            num_signals,
            groups,
            lanes: groups * 128,
        })
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }
    pub fn signal_count(&self) -> usize {
        self.num_signals
    }

    pub fn set_signal(&mut self, lane: usize, idx: usize, bit: bool) {
        // group of 128 lanes = one I64X2 (two u64 words); word w, bit b.
        let (g, sub) = (lane / 128, lane % 128);
        let (w, b) = (sub / 64, sub % 64);
        let slot = &mut self.state[g * self.num_signals + idx];
        let mut word = u64::from_le_bytes(slot.0[w * 8..w * 8 + 8].try_into().unwrap());
        if bit {
            word |= 1u64 << b;
        } else {
            word &= !(1u64 << b);
        }
        slot.0[w * 8..w * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    pub fn get_signal(&self, lane: usize, idx: usize) -> bool {
        let (g, sub) = (lane / 128, lane % 128);
        let (w, b) = (sub / 64, sub % 64);
        let slot = &self.state[g * self.num_signals + idx];
        let word = u64::from_le_bytes(slot.0[w * 8..w * 8 + 8].try_into().unwrap());
        (word >> b) & 1 == 1
    }

    pub fn tick(&mut self) {
        self.tick_many(1);
    }
    pub fn tick_many(&mut self, n: usize) {
        (self.tick_fn)(self.state.as_mut_ptr() as *mut u8, self.groups as i64, n as i64);
    }
}

/// All-ones / all-zeros `I64X2` constant for bit-parallel codegen.
fn bp_vconst(b: &mut FunctionBuilder, all_ones: bool) -> Value {
    let s = b.ins().iconst(I64, if all_ones { -1 } else { 0 });
    b.ins().splat(I64X2, s)
}

/// One bit-parallel cycle: comb settle, capture register next-states, commit,
/// settle again (so observed comb outputs reflect the committed registers).
fn emit_bp_tick(b: &mut FunctionBuilder, base: Value, machine: &PackedMachineProgram) -> Result<(), ErrorReport> {
    emit_bp_comb(b, base, &machine.streams.comb)?;
    let caps = emit_bp_capture(b, base, &machine.streams.tick_next)?;
    for (dst, v) in caps {
        b.ins().store(MemFlags::trusted(), v, base, (dst * 16) as i32);
    }
    emit_bp_comb(b, base, &machine.streams.comb)?;
    Ok(())
}

fn emit_bp_comb(b: &mut FunctionBuilder, base: Value, block: &PackedBlock) -> Result<(), ErrorReport> {
    let mut vals: HashMap<usize, Value> = HashMap::new();
    for pkt in &block.packets {
        for instr in &pkt.instrs {
            let v = emit_bp_instr(b, base, &vals, &instr.kind)?;
            vals.insert(instr.dst.0, v);
        }
        for eff in &pkt.effects {
            if let PackedEffect::StoreSignal { dst, value } = eff {
                let v = *vals.get(&value.0).ok_or_else(|| jit_err("bp: undefined comb value"))?;
                b.ins().store(MemFlags::trusted(), v, base, (dst * 16) as i32);
            }
        }
    }
    Ok(())
}

fn emit_bp_capture(
    b: &mut FunctionBuilder,
    base: Value,
    block: &PackedBlock,
) -> Result<Vec<(usize, Value)>, ErrorReport> {
    let mut vals: HashMap<usize, Value> = HashMap::new();
    let mut caps = Vec::new();
    for pkt in &block.packets {
        for instr in &pkt.instrs {
            let v = emit_bp_instr(b, base, &vals, &instr.kind)?;
            vals.insert(instr.dst.0, v);
        }
        for eff in &pkt.effects {
            if let PackedEffect::CaptureReg { dst, value, reset } = eff {
                let mut nv = *vals.get(&value.0).ok_or_else(|| jit_err("bp: undefined capture value"))?;
                if let Some(r) = reset {
                    // sync reset: nv = asserted ? reset_value : nv (bit-parallel select)
                    let mut asserted = b.ins().load(I64X2, MemFlags::trusted(), base, (r.signal * 16) as i32);
                    if matches!(r.polarity, rrtl_ir::ResetPolarity::ActiveLow) {
                        asserted = b.ins().bnot(asserted);
                    }
                    let bit = r.value.first().copied().unwrap_or(0) & 1;
                    let rv = bp_vconst(b, bit == 1);
                    let na = b.ins().bnot(asserted);
                    let l = b.ins().band(asserted, rv);
                    let rr = b.ins().band(na, nv);
                    nv = b.ins().bor(l, rr);
                }
                caps.push((*dst, nv));
            }
        }
    }
    Ok(caps)
}

fn emit_bp_instr(
    b: &mut FunctionBuilder,
    base: Value,
    vals: &HashMap<usize, Value>,
    kind: &PackedInstrKind,
) -> Result<Value, ErrorReport> {
    use PackedInstrKind::*;
    let get = |v: &PackedValueId| vals.get(&v.0).copied().ok_or_else(|| jit_err("bp: undefined operand"));
    Ok(match kind {
        Signal(s) => b.ins().load(I64X2, MemFlags::trusted(), base, (s * 16) as i32),
        Lit(w) => {
            let bit = w.first().copied().unwrap_or(0) & 1;
            bp_vconst(b, bit == 1)
        }
        Not(a) => {
            let a = get(a)?;
            b.ins().bnot(a)
        }
        And(a, c) => {
            let (a, c) = (get(a)?, get(c)?);
            b.ins().band(a, c)
        }
        Or(a, c) => {
            let (a, c) = (get(a)?, get(c)?);
            b.ins().bor(a, c)
        }
        Xor(a, c) => {
            let (a, c) = (get(a)?, get(c)?);
            b.ins().bxor(a, c)
        }
        Mux { cond, then_value, else_value } => {
            let (c, t, e) = (get(cond)?, get(then_value)?, get(else_value)?);
            let nc = b.ins().bnot(c);
            let l = b.ins().band(c, t);
            let r = b.ins().band(nc, e);
            b.ins().bor(l, r)
        }
        _ => return Err(jit_err("bp: non-gate-level op")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lower_to_machine_program, lower_to_packed_program, SimdCpuSimulator};
    use rrtl_core::{compile, lit_u, mux, uint, Design, Signal, Simulator};

    // The bit-parallel JIT (64 lanes/i64, bitwise) must match the SIMD CPU engine
    // on a gate-level netlist, every output of every lane.
    #[test]
    fn bitparallel_jit_matches_simd_cpu() {
        let mut design = Design::new();
        {
            let mut m = design.module("G");
            let clk = m.input("clk", uint(1));
            let a = m.input("a", uint(1));
            let b = m.input("b", uint(1));
            for c in 0..5 {
                let r = m.reg(format!("r{c}"), uint(1));
                m.clock(r, clk);
                let x = match c % 3 {
                    0 => !(r.value() & a.value()),
                    1 => !(r.value() | b.value()),
                    _ => !(r.value() ^ a.value()),
                };
                m.next(r, x);
                let o = m.output(format!("o{c}"), uint(1));
                m.assign(o, r);
            }
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "G").unwrap();
        let machine = lower_to_machine_program(&program);
        let h = |n: &str| compiled.find_module("G").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
        let lanes = 200; // > 3 words

        let mut jit = BitParallelJitSimulator::compile_lanes(&machine, lanes).unwrap();
        let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        cpu.set_signal(h("clk"), &vec![1u128; lanes]).unwrap();
        let (a_i, b_i) = (program.signal_index(h("a")).unwrap(), program.signal_index(h("b")).unwrap());
        for lane in 0..lanes {
            jit.set_signal(lane, a_i, lane % 2 == 0);
            jit.set_signal(lane, b_i, lane % 3 == 0);
        }
        cpu.set_signal(h("a"), &(0..lanes).map(|l| (l % 2 == 0) as u128).collect::<Vec<_>>()).unwrap();
        cpu.set_signal(h("b"), &(0..lanes).map(|l| (l % 3 == 0) as u128).collect::<Vec<_>>()).unwrap();

        for _ in 0..10 {
            jit.tick();
            cpu.tick().unwrap();
        }
        for c in 0..5 {
            let oi = program.signal_index(h(&format!("o{c}"))).unwrap();
            let cv = cpu.get_signal(h(&format!("o{c}"))).unwrap();
            for lane in 0..lanes {
                assert_eq!(jit.get_signal(lane, oi), cv[lane] & 1 == 1, "o{c}@lane{lane}");
            }
        }
    }

    // The compiled (Cranelift) clock-gated tick must match the gold oracle on an
    // independent multi-clock design — the hard case (the active-clock set is a
    // runtime mask argument to the compiled tick).
    #[test]
    fn jit_tick_clocked_matches_oracle() {
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
        let machine = lower_to_machine_program(&program);
        let (clka_i, clkb_i) = (program.signal_index(clka).unwrap(), program.signal_index(clkb).unwrap());
        let (a_i, b_i) = (program.signal_index(a).unwrap(), program.signal_index(b).unwrap());

        let mut jit = JitSimulator::compile(&machine).unwrap();
        let mut gold = Simulator::new(&design, "TwoClock").unwrap();
        for step in 0..12 {
            let edge = step % 3 == 0; // clkB every 3rd step
            let active_i: Vec<usize> = if edge { vec![clka_i, clkb_i] } else { vec![clka_i] };
            let active_s: Vec<Signal> = if edge { vec![clka, clkb] } else { vec![clka] };
            jit.tick_clocked(&active_i);
            gold.tick_clocked(&active_s).unwrap();
            assert_eq!(jit.get_signal(a_i) as u128, gold.get(a), "a@{step}");
            assert_eq!(jit.get_signal(b_i) as u128, gold.get(b), "b@{step}");
        }
        assert!(jit.get_signal(b_i) < jit.get_signal(a_i));
    }

    // A memory written on a slow clock must capture only on its edges (the JIT's
    // clock-gated memory write) vs the gold oracle.
    #[test]
    fn jit_tick_clocked_gates_memory_writes() {
        let mut design = Design::new();
        let (clka, clkb, out);
        {
            let mut m = design.module("MemClk");
            clka = m.input("clka", uint(1));
            clkb = m.input("clkb", uint(1));
            let cnt = m.reg("cnt", uint(8));
            m.clock(cnt, clka);
            m.next(cnt, cnt + lit_u(1, 8));
            let mem = m.mem("mem", 2, uint(8), 4);
            m.mem_write(mem, clkb, lit_u(1, 1), lit_u(0, 2), cnt);
            let rd = m.mem_read(mem, lit_u(0, 2));
            out = m.output("out", uint(8));
            m.assign(out, rd);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "MemClk").unwrap();
        let machine = lower_to_machine_program(&program);
        let (clka_i, clkb_i) = (program.signal_index(clka).unwrap(), program.signal_index(clkb).unwrap());
        let out_i = program.signal_index(out).unwrap();

        let mut jit = JitSimulator::compile(&machine).unwrap();
        let mut gold = Simulator::new(&design, "MemClk").unwrap();
        for step in 0..12 {
            let edge = step % 3 == 0;
            let active_i: Vec<usize> = if edge { vec![clka_i, clkb_i] } else { vec![clka_i] };
            let active_s: Vec<Signal> = if edge { vec![clka, clkb] } else { vec![clka] };
            jit.tick_clocked(&active_i);
            gold.tick_clocked(&active_s).unwrap();
            assert_eq!(jit.get_signal(out_i) as u128, gold.get(out), "out@{step}");
        }
    }

    fn dut() -> Design {
        let mut design = Design::new();
        {
            let mut m = design.module("Dut");
            let clk = m.input("clk", uint(1));
            let a = m.input("a", uint(16));
            let b = m.input("b", uint(16));
            let sel = m.input("sel", uint(1));
            let rst = m.input("rst", uint(1));
            // combinational variety
            let sum = m.wire("sum", uint(16));
            m.assign(sum, a + b);
            let prod = m.wire("prod", uint(16));
            m.assign(prod, a * b);
            let lt = m.output("lt", uint(1));
            m.assign(lt, a.value().lt_expr(b.value()));
            let mx = m.output("mx", uint(16));
            m.assign(mx, mux(sel, a.value(), b.value()));
            let packed = m.output("packed", uint(16));
            m.assign(packed, rrtl_core::concat([a.value().slice(0, 8), b.value().slice(0, 8)]));
            // register with synchronous reset + zext-accumulate
            let acc = m.reg("acc", uint(32));
            m.clock(acc, clk);
            m.reset(acc, rst, 0);
            m.next(acc, acc.value() + sum.value().zext(32));
            let oacc = m.output("oacc", uint(32));
            m.assign(oacc, acc);
        }
        design
    }

    #[test]
    fn jit_memory_matches_simd_cpu() {
        let mut design = Design::new();
        {
            let mut m = design.module("RegFile");
            let clk = m.input("clk", uint(1));
            let we = m.input("we", uint(1));
            let waddr = m.input("waddr", uint(4));
            let wdata = m.input("wdata", uint(8));
            let raddr = m.input("raddr", uint(4));
            let regs = m.mem("regs", 4, uint(8), 16);
            m.mem_write(regs, clk, we.value(), waddr.value(), wdata.value());
            let rdata = m.output("rdata", uint(8));
            let rd = m.mem_read(regs, raddr.value());
            m.assign(rdata, rd);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "RegFile").unwrap();
        let machine = lower_to_machine_program(&program);
        let mut jit = JitSimulator::compile(&machine).expect("compile");

        let handle = |name: &str| -> Signal {
            compiled.find_module("RegFile").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle
        };
        let idx = |name: &str| program.signal_index(handle(name)).unwrap();
        let mut cpu = SimdCpuSimulator::new(program.clone(), 1).unwrap();

        let mut lcg: u64 = 0xabcd;
        let mut rng = || {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            lcg >> 33
        };
        for cycle in 0..80u32 {
            let we = (rng() & 1) as u64;
            let waddr = (rng() & 0xf) as u64;
            let wdata = (rng() & 0xff) as u64;
            let raddr = (rng() & 0xf) as u64;
            for (n, v) in [("clk", 1u64), ("we", we), ("waddr", waddr), ("wdata", wdata), ("raddr", raddr)] {
                jit.set_signal(idx(n), v);
                cpu.set_signal(handle(n), &[v as u128]).unwrap();
            }
            jit.tick();
            cpu.tick();
            assert_eq!(
                jit.get_signal(idx("rdata")),
                cpu.get_signal(handle("rdata")).unwrap()[0] as u64,
                "rdata mismatch at cycle {cycle}"
            );
        }
    }

    #[test]
    fn jit_wide_memory_matches_simd_cpu() {
        // 128-bit-wide memory data exercises the I128 memory load/store path
        // (16-byte entry slots, val_ty(dw) typed mem access).
        let mut design = Design::new();
        {
            let mut m = design.module("WideRegFile");
            let clk = m.input("clk", uint(1));
            let we = m.input("we", uint(1));
            let waddr = m.input("waddr", uint(4));
            let wdata = m.input("wdata", uint(128));
            let raddr = m.input("raddr", uint(4));
            let regs = m.mem("regs", 4, uint(128), 16);
            m.mem_write(regs, clk, we.value(), waddr.value(), wdata.value());
            let rdata = m.output("rdata", uint(128));
            let rd = m.mem_read(regs, raddr.value());
            m.assign(rdata, rd);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "WideRegFile").unwrap();
        let machine = lower_to_machine_program(&program);
        let mut jit = JitSimulator::compile(&machine).expect("compile");
        let handle = |name: &str| -> Signal {
            compiled.find_module("WideRegFile").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle
        };
        let idx = |name: &str| program.signal_index(handle(name)).unwrap();
        let mut cpu = SimdCpuSimulator::new(program.clone(), 1).unwrap();

        let mut lcg: u64 = 0xc0ffee;
        let mut rng = || {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            lcg
        };
        for cycle in 0..80u32 {
            let we = (rng() & 1) as u128;
            let waddr = (rng() & 0xf) as u128;
            let wdata = (rng() as u128) | ((rng() as u128) << 64);
            let raddr = (rng() & 0xf) as u128;
            for (n, v) in [("clk", 1u128), ("we", we), ("waddr", waddr), ("wdata", wdata), ("raddr", raddr)] {
                jit.set_signal_u128(idx(n), v);
                cpu.set_signal(handle(n), &[v]).unwrap();
            }
            jit.tick();
            cpu.tick();
            assert_eq!(
                jit.get_signal_u128(idx("rdata")),
                cpu.get_signal(handle("rdata")).unwrap()[0],
                "rdata mismatch at cycle {cycle}"
            );
        }
    }

    #[test]
    fn jit_async_reset_matches_simd_cpu() {
        let mut design = Design::new();
        {
            let mut m = design.module("Cnt");
            let clk = m.input("clk", uint(1));
            let rst_n = m.input("rst_n", uint(1));
            let en = m.input("en", uint(1));
            let ctr = m.reg("ctr", uint(8));
            m.clock(ctr, clk);
            m.async_reset_low(ctr, rst_n, 0);
            m.next(ctr, mux(en, ctr.value() + lit_u(1, 8), ctr.value()));
            let out = m.output("out", uint(8));
            m.assign(out, ctr);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Cnt").unwrap();
        let machine = lower_to_machine_program(&program);
        let mut jit = JitSimulator::compile(&machine).expect("compile");
        let handle = |name: &str| -> Signal {
            compiled.find_module("Cnt").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle
        };
        let idx = |name: &str| program.signal_index(handle(name)).unwrap();
        let mut cpu = SimdCpuSimulator::new(program.clone(), 1).unwrap();

        let mut lcg: u64 = 0x5555;
        let mut rng = || { lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1); lcg >> 40 };
        for cycle in 0..60u32 {
            // assert async reset (active-low: 0) on a few cycles.
            let rst_n = if cycle % 13 == 0 { 0u64 } else { 1 };
            let en = (rng() & 1) as u64;
            for (n, v) in [("clk", 1u64), ("rst_n", rst_n), ("en", en)] {
                jit.set_signal(idx(n), v);
                cpu.set_signal(handle(n), &[v as u128]).unwrap();
            }
            jit.tick();
            cpu.tick();
            assert_eq!(
                jit.get_signal(idx("out")),
                cpu.get_signal(handle("out")).unwrap()[0] as u64,
                "out mismatch at cycle {cycle} (rst_n={rst_n})"
            );
        }
    }

    #[test]
    fn jit_run_trace_matches_cycle_by_cycle() {
        let design = dut();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Dut").unwrap();
        let machine = lower_to_machine_program(&program);
        let mut traced = JitSimulator::compile(&machine).unwrap();
        let mut step = JitSimulator::compile(&machine).unwrap();

        let in_idx = traced.input_indices().to_vec();
        let out_idx = traced.output_indices().to_vec();
        let names: Vec<String> = traced.input_ports().to_vec();
        let nin = in_idx.len();
        let nout = out_idx.len();
        let cycles = 40usize;

        // Build a stimulus buffer in port order.
        let mut lcg: u64 = 0x99;
        let mut rng = || { lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1); lcg >> 33 };
        let mut inbuf = vec![0u64; cycles * nin];
        for c in 0..cycles {
            for (k, name) in names.iter().enumerate() {
                let local = name.rsplit('.').next().unwrap();
                inbuf[c * nin + k] = match local {
                    "clk" => 1,
                    "rst" => (c == 0) as u64,
                    "sel" => rng() & 1,
                    _ => rng() & 0xffff,
                };
            }
        }

        // Baked harness: one native call.
        let outbuf = traced.run_trace(&inbuf, cycles);

        // Reference: cycle-by-cycle on a fresh instance.
        for c in 0..cycles {
            for k in 0..nin {
                step.set_signal(in_idx[k], inbuf[c * nin + k]);
            }
            step.tick();
            for k in 0..nout {
                assert_eq!(
                    outbuf[c * nout + k],
                    step.get_signal(out_idx[k]),
                    "trace vs step mismatch at cycle {c}, output {k}"
                );
            }
        }
    }

    #[test]
    fn jit_matches_simd_cpu() {
        let design = dut();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Dut").unwrap();
        let machine = lower_to_machine_program(&program);
        let mut jit = JitSimulator::compile(&machine).expect("compile");

        let handle = |name: &str| -> Signal {
            compiled.find_module("Dut").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle
        };
        let idx = |name: &str| program.signal_index(handle(name)).unwrap();
        let mut cpu = SimdCpuSimulator::new(program.clone(), 1).unwrap();

        let mut lcg: u64 = 0x1234_5678;
        let mut rng = || {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            lcg >> 33
        };
        for cycle in 0..50u32 {
            let a = (rng() & 0xffff) as u64;
            let bv = (rng() & 0xffff) as u64;
            let sel = (rng() & 1) as u64;
            let rst = (cycle == 0) as u64;
            for (n, v) in [("a", a), ("b", bv), ("sel", sel), ("rst", rst), ("clk", 1)] {
                jit.set_signal(idx(n), v);
                cpu.set_signal(handle(n), &[v as u128]).unwrap();
            }
            jit.tick();
            cpu.tick();
            for o in ["lt", "mx", "packed", "oacc"] {
                let j = jit.get_signal(idx(o));
                let c = cpu.get_signal(handle(o)).unwrap()[0] as u64;
                assert_eq!(j, c, "signal `{o}` mismatch at cycle {cycle}: jit={j} cpu={c}");
            }
        }
    }

    /// A bank of `n` independent accumulators, each gated by its own enable bit,
    /// summing a shared held-stable data input. Idle accumulators (enable low,
    /// data stable) should let their cones skip.
    fn gated_bank(n: usize) -> Design {
        let mut design = Design::new();
        {
            let mut m = design.module("Bank");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let x = m.input("x", uint(16));
            let en = m.input("en", uint(n as u32));
            for i in 0..n {
                let acc = m.reg(&format!("acc{i}"), uint(16));
                m.clock(acc, clk);
                m.reset(acc, rst, 0);
                let eni = en.value().slice(i as u32, 1);
                m.next(acc, mux(eni, acc.value() + x.value(), acc.value()));
                let o = m.output(&format!("o{i}"), uint(16));
                m.assign(o, acc);
            }
        }
        design
    }

    #[test]
    fn jit_activity_matches_simd_cpu() {
        // Bit-exact: the activity-skipping path must match the oracle on the full
        // dut (mul-add datapath, mux, slice, concat, lt, sync-reset accumulator).
        let design = dut();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Dut").unwrap();
        let machine = lower_to_machine_program(&program);
        let mut jit = JitSimulator::compile_activity(&machine).expect("compile");
        let handle = |name: &str| -> Signal {
            compiled.find_module("Dut").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle
        };
        let idx = |name: &str| program.signal_index(handle(name)).unwrap();
        let mut cpu = SimdCpuSimulator::new(program.clone(), 1).unwrap();

        let mut lcg: u64 = 0x1234_5678;
        let mut rng = || {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            lcg >> 33
        };
        for cycle in 0..50u32 {
            let a = (rng() & 0xffff) as u64;
            let bv = (rng() & 0xffff) as u64;
            let sel = (rng() & 1) as u64;
            let rst = (cycle == 0) as u64;
            for (n, v) in [("a", a), ("b", bv), ("sel", sel), ("rst", rst), ("clk", 1)] {
                jit.set_signal(idx(n), v);
                cpu.set_signal(handle(n), &[v as u128]).unwrap();
            }
            jit.tick();
            cpu.tick();
            for o in ["lt", "mx", "packed", "oacc"] {
                assert_eq!(
                    jit.get_signal(idx(o)),
                    cpu.get_signal(handle(o)).unwrap()[0] as u64,
                    "signal `{o}` mismatch at cycle {cycle}"
                );
            }
        }
    }

    #[test]
    fn jit_activity_skips_idle() {
        // Mostly-idle gated bank: bit-exact vs oracle AND a substantial skip rate.
        let design = gated_bank(8);
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Bank").unwrap();
        let machine = lower_to_machine_program(&program);
        let mut jit = JitSimulator::compile_activity_instrumented(&machine).expect("compile");
        let handle = |name: &str| -> Signal {
            compiled.find_module("Bank").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle
        };
        let idx = |name: &str| program.signal_index(handle(name)).unwrap();
        let mut cpu = SimdCpuSimulator::new(program.clone(), 1).unwrap();

        let mut lcg: u64 = 0x2024;
        let mut rng = || {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            lcg >> 40
        };
        for cycle in 0..120u32 {
            let rst = (cycle == 0) as u64;
            // x held stable; at most one accumulator enabled, and only rarely.
            let x = 7u64;
            let en = if cycle % 10 == 3 { 1u64 << (rng() % 8) } else { 0 };
            for (n, v) in [("clk", 1u64), ("rst", rst), ("x", x), ("en", en)] {
                jit.set_signal(idx(n), v);
                cpu.set_signal(handle(n), &[v as u128]).unwrap();
            }
            jit.tick();
            cpu.tick();
            for i in 0..8 {
                let o = format!("o{i}");
                assert_eq!(
                    jit.get_signal(idx(&o)),
                    cpu.get_signal(handle(&o)).unwrap()[0] as u64,
                    "signal `{o}` mismatch at cycle {cycle}"
                );
            }
        }
        let rate = jit.activity_skip_rate().unwrap();
        assert!(rate > 0.6, "expected a high skip rate on an idle bank, got {rate:.3}");
    }

    #[test]
    fn jit_activity_async_reset_matches_simd_cpu() {
        // Activity path with an asynchronous (active-low) reset counter.
        let mut design = Design::new();
        {
            let mut m = design.module("Cnt");
            let clk = m.input("clk", uint(1));
            let rst_n = m.input("rst_n", uint(1));
            let en = m.input("en", uint(1));
            let ctr = m.reg("ctr", uint(8));
            m.clock(ctr, clk);
            m.async_reset_low(ctr, rst_n, 0);
            m.next(ctr, mux(en, ctr.value() + lit_u(1, 8), ctr.value()));
            let out = m.output("out", uint(8));
            m.assign(out, ctr);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Cnt").unwrap();
        let machine = lower_to_machine_program(&program);
        let mut jit = JitSimulator::compile_activity(&machine).expect("compile");
        let handle = |name: &str| -> Signal {
            compiled.find_module("Cnt").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle
        };
        let idx = |name: &str| program.signal_index(handle(name)).unwrap();
        let mut cpu = SimdCpuSimulator::new(program.clone(), 1).unwrap();
        let mut lcg: u64 = 0x5555;
        let mut rng = || { lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1); lcg >> 40 };
        for cycle in 0..60u32 {
            let rst_n = if cycle % 13 == 0 { 0u64 } else { 1 };
            let en = (rng() & 1) as u64;
            for (n, v) in [("clk", 1u64), ("rst_n", rst_n), ("en", en)] {
                jit.set_signal(idx(n), v);
                cpu.set_signal(handle(n), &[v as u128]).unwrap();
            }
            jit.tick();
            cpu.tick();
            assert_eq!(
                jit.get_signal(idx("out")),
                cpu.get_signal(handle("out")).unwrap()[0] as u64,
                "out mismatch at cycle {cycle} (rst_n={rst_n})"
            );
        }
    }

    #[test]
    fn simd_jit_matches_simd_cpu_4lane() {
        // 4-lane vector JIT, bit-exact vs the SIMD interpreter at 4 lanes, with a
        // distinct stimulus per lane (mul-add datapath, mux, slice, concat, lt,
        // sync-reset accumulator — all ≤32-bit).
        let design = dut();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Dut").unwrap();
        let machine = lower_to_machine_program(&program);
        let mut jit = SimdJitSimulator::compile(&machine).expect("compile");
        assert_eq!(jit.lanes(), 4);
        let handle = |name: &str| -> Signal {
            compiled.find_module("Dut").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle
        };
        let idx = |name: &str| program.signal_index(handle(name)).unwrap();
        let mut cpu = SimdCpuSimulator::new(program.clone(), 4).unwrap();

        let mut lcg: u64 = 0xd1ce;
        let mut rng = || { lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1); lcg >> 33 };
        for cycle in 0..50u32 {
            let rst = (cycle == 0) as u64;
            for (n, w) in [("a", 16u32), ("b", 16), ("sel", 1), ("rst", 1), ("clk", 1)] {
                let mut lanes = [0u128; 4];
                for (l, slot) in lanes.iter_mut().enumerate() {
                    let v = match n {
                        "clk" => 1,
                        "rst" => rst,
                        _ => rng() & ((1u64 << w) - 1),
                    };
                    *slot = v as u128;
                    jit.set_signal(l, idx(n), v as u32);
                }
                cpu.set_signal(handle(n), &lanes).unwrap();
            }
            jit.tick();
            cpu.tick().unwrap();
            for o in ["lt", "mx", "packed", "oacc"] {
                let cv = cpu.get_signal(handle(o)).unwrap();
                for l in 0..4 {
                    assert_eq!(
                        jit.get_signal(l, idx(o)) as u128,
                        cv[l],
                        "signal `{o}` lane {l} mismatch at cycle {cycle}"
                    );
                }
            }
        }
    }

    #[test]
    fn simd_jit_memory_matches_simd_cpu_4lane() {
        // 4-lane register file: per-lane gather (read) + scatter (write), each
        // lane an independent memory. Bit-exact vs the SIMD interpreter at 4 lanes.
        let mut design = Design::new();
        {
            let mut m = design.module("RegFile");
            let clk = m.input("clk", uint(1));
            let we = m.input("we", uint(1));
            let waddr = m.input("waddr", uint(4));
            let wdata = m.input("wdata", uint(8));
            let raddr = m.input("raddr", uint(4));
            let regs = m.mem("regs", 4, uint(8), 16);
            m.mem_write(regs, clk, we.value(), waddr.value(), wdata.value());
            let rdata = m.output("rdata", uint(8));
            let rd = m.mem_read(regs, raddr.value());
            m.assign(rdata, rd);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "RegFile").unwrap();
        let machine = lower_to_machine_program(&program);
        let mut jit = SimdJitSimulator::compile(&machine).expect("compile");
        let handle = |name: &str| -> Signal {
            compiled.find_module("RegFile").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle
        };
        let idx = |name: &str| program.signal_index(handle(name)).unwrap();
        let mut cpu = SimdCpuSimulator::new(program.clone(), 4).unwrap();

        let mut lcg: u64 = 0xbead;
        let mut rng = || { lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1); lcg >> 33 };
        for cycle in 0..120u32 {
            for (n, w) in [("clk", 1u32), ("we", 1), ("waddr", 4), ("wdata", 8), ("raddr", 4)] {
                let mut lanes = [0u128; 4];
                for (l, slot) in lanes.iter_mut().enumerate() {
                    let v = if n == "clk" { 1 } else { rng() & ((1u64 << w) - 1) };
                    *slot = v as u128;
                    jit.set_signal(l, idx(n), v as u32);
                }
                cpu.set_signal(handle(n), &lanes).unwrap();
            }
            jit.tick();
            cpu.tick().unwrap();
            let cv = cpu.get_signal(handle("rdata")).unwrap();
            for l in 0..4 {
                assert_eq!(
                    jit.get_signal(l, idx("rdata")) as u128,
                    cv[l],
                    "rdata lane {l} mismatch at cycle {cycle}"
                );
            }
        }
    }

    #[test]
    fn simd_jit_batch_16lane_matches_simd_cpu() {
        // N-lane batch (4 groups × 4 lanes), with memory, bit-exact vs the SIMD
        // interpreter at 16 lanes — exercises group-major state + per-group memory.
        let mut design = Design::new();
        {
            let mut m = design.module("RegFile");
            let clk = m.input("clk", uint(1));
            let we = m.input("we", uint(1));
            let waddr = m.input("waddr", uint(4));
            let wdata = m.input("wdata", uint(16));
            let raddr = m.input("raddr", uint(4));
            let regs = m.mem("regs", 4, uint(16), 16);
            m.mem_write(regs, clk, we.value(), waddr.value(), wdata.value());
            // a little logic on the read path too
            let rdata = m.output("rdata", uint(16));
            let rd = m.mem_read(regs, raddr.value());
            m.assign(rdata, rd + wdata.value());
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "RegFile").unwrap();
        let machine = lower_to_machine_program(&program);
        let mut jit = SimdJitSimulator::compile_lanes(&machine, 16).expect("compile");
        assert_eq!(jit.lanes(), 16);
        let handle = |name: &str| -> Signal {
            compiled.find_module("RegFile").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle
        };
        let idx = |name: &str| program.signal_index(handle(name)).unwrap();
        let mut cpu = SimdCpuSimulator::new(program.clone(), 16).unwrap();

        let mut lcg: u64 = 0x7e57;
        let mut rng = || { lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1); lcg >> 33 };
        for cycle in 0..150u32 {
            for (n, w) in [("clk", 1u32), ("we", 1), ("waddr", 4), ("wdata", 16), ("raddr", 4)] {
                let mut lanes = [0u128; 16];
                for (l, slot) in lanes.iter_mut().enumerate() {
                    let v = if n == "clk" { 1 } else { rng() & ((1u64 << w) - 1) };
                    *slot = v as u128;
                    jit.set_signal(l, idx(n), v as u32);
                }
                cpu.set_signal(handle(n), &lanes).unwrap();
            }
            jit.tick();
            cpu.tick().unwrap();
            let cv = cpu.get_signal(handle("rdata")).unwrap();
            for l in 0..16 {
                assert_eq!(
                    jit.get_signal(l, idx("rdata")) as u128,
                    cv[l],
                    "rdata lane {l} mismatch at cycle {cycle}"
                );
            }
        }
    }

    #[test]
    fn jit_wide_128_matches_simd_cpu() {
        // Exercises the >64-bit (I128) paths: add/mul/xor at 128 bits, an I128
        // register with sync reset, a 128→64 slice (ireduce), an unsigned 128-bit
        // compare, and a 64+64 → 128 concat.
        let mut design = Design::new();
        {
            let mut m = design.module("Wide");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let sel = m.input("sel", uint(1));
            let a = m.input("a", uint(128));
            let b = m.input("b", uint(128));
            let s = m.wire("s", uint(128));
            m.assign(s, a.value() + b.value());
            let p = m.wire("p", uint(128));
            m.assign(p, a.value() * b.value());
            // 128-bit register, sync reset, next = acc ^ (sel ? s : p)
            let acc = m.reg("acc", uint(128));
            m.clock(acc, clk);
            m.reset(acc, rst, 0);
            m.next(acc, acc.value() ^ mux(sel, s.value(), p.value()));
            let oacc = m.output("oacc", uint(128));
            m.assign(oacc, acc);
            let lo = m.output("lo", uint(64));
            m.assign(lo, acc.value().slice(0, 64));
            let ltw = m.output("ltw", uint(1));
            m.assign(ltw, a.value().lt_expr(b.value()));
            let packed = m.output("packed", uint(128));
            m.assign(packed, rrtl_core::concat([a.value().slice(0, 64), b.value().slice(0, 64)]));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Wide").unwrap();
        let machine = lower_to_machine_program(&program);
        let mut jit = JitSimulator::compile(&machine).expect("compile");
        let handle = |name: &str| -> Signal {
            compiled.find_module("Wide").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle
        };
        let idx = |name: &str| program.signal_index(handle(name)).unwrap();
        let mut cpu = SimdCpuSimulator::new(program.clone(), 1).unwrap();

        let mut lcg: u64 = 0xfeed_face;
        let mut rng = || {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            lcg
        };
        let mut rng128 = || {
            let lo = rng() as u128;
            let hi = rng() as u128;
            lo | (hi << 64)
        };
        for cycle in 0..50u32 {
            let a = rng128();
            let bv = rng128();
            let sel = (rng128() & 1) as u128;
            let rst = (cycle == 0) as u128;
            for (n, v) in [("a", a), ("b", bv), ("sel", sel), ("rst", rst), ("clk", 1u128)] {
                jit.set_signal_u128(idx(n), v);
                cpu.set_signal(handle(n), &[v]).unwrap();
            }
            jit.tick();
            cpu.tick();
            for o in ["oacc", "lo", "ltw", "packed"] {
                let j = jit.get_signal_u128(idx(o));
                let c = cpu.get_signal(handle(o)).unwrap()[0];
                assert_eq!(j, c, "signal `{o}` mismatch at cycle {cycle}: jit={j:#x} cpu={c:#x}");
            }
        }
    }
}
