//! Auto-partitioning hybrid simulator. Given a packed program it (1) detects the
//! GF(2)-linear register cones and compiles them to the matrix XOR-AOT
//! ([`LinearAot`]), (2) observability-slices the rest (so the linear cones are
//! pruned from the general engine), and (3) runs the non-linear remainder on a
//! general [`SimdCpuSimulator`]. The two partitions have disjoint register state,
//! so `tick_many` advances them concurrently (linear engine on a worker thread).
//!
//! Presents one handle-keyed `set_signal` / `get_signal` / `tick` interface and
//! routes internally: inputs go to whichever partition reads them (shared inputs
//! to both); a linear register (or an output port aliasing one) is read from the
//! matrix AOT, everything else from the general engine.
//!
//! Scope/limitation: output ports served by the linear side must be simple
//! aliases of a linear register (`out <= reg`, possibly through a width cast);
//! a purely-combinational *function* of linear registers is not served (the
//! matrix AOT computes register cones, not arbitrary comb outputs).

use std::collections::{HashMap, HashSet};

use rrtl_ir::{Diagnostic, ErrorReport, Signal};

use crate::leap::LinearLeap;
use crate::linear_aot::LinearAot;
use crate::{
    cone_of_influence, slice_present, PackedExpr, PackedExprKind, PackedOp, PackedProgram,
    PackedSignalKind, SimdCpuSimulator,
};

fn hyb_err(msg: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![Diagnostic::new("E_HYBRID", msg)])
}

/// A general-purpose engine for the non-linear partition, behind a handle-keyed
/// interface. The best available backend is auto-picked at construction: the
/// vector JIT when the slice fits it (≤32-bit, no async reset), else the SIMD CPU
/// interpreter (which handles any width/op). Both consume the same sliced program.
enum GeneralEngine {
    Cpu(SimdCpuSimulator),
    #[cfg(feature = "jit")]
    Jit {
        jit: crate::jit::SimdJitSimulator,
        idx: HashMap<Signal, usize>, // handle -> machine signal index
        lanes: usize,
    },
}

impl GeneralEngine {
    /// Pick the fastest backend that can run `program` (vector JIT preferred).
    fn build(program: PackedProgram, lanes: usize) -> Result<(Self, &'static str), ErrorReport> {
        #[cfg(feature = "jit")]
        {
            let machine = crate::lower_to_machine_program(&program);
            if let Ok(jit) = crate::jit::SimdJitSimulator::compile_lanes(&machine, lanes) {
                let idx: HashMap<Signal, usize> = program
                    .signals
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| s.source.map(|h| (h, i)))
                    .collect();
                return Ok((GeneralEngine::Jit { jit, idx, lanes }, "vector-JIT"));
            }
        }
        Ok((GeneralEngine::Cpu(SimdCpuSimulator::new(program, lanes)?), "SIMD-CPU"))
    }

    fn set_signal(&mut self, signal: Signal, vals: &[u128]) -> Result<(), ErrorReport> {
        match self {
            GeneralEngine::Cpu(c) => c.set_signal(signal, vals),
            #[cfg(feature = "jit")]
            GeneralEngine::Jit { jit, idx, .. } => {
                if let Some(&i) = idx.get(&signal) {
                    for (l, v) in vals.iter().enumerate() {
                        jit.set_signal(l, i, *v as u32);
                    }
                }
                Ok(())
            }
        }
    }

    fn get_signal(&self, signal: Signal) -> Result<Vec<u128>, ErrorReport> {
        match self {
            GeneralEngine::Cpu(c) => c.get_signal(signal),
            #[cfg(feature = "jit")]
            GeneralEngine::Jit { jit, idx, lanes } => {
                let i = *idx.get(&signal).ok_or_else(|| hyb_err("signal not in JIT slice"))?;
                Ok((0..*lanes).map(|l| jit.get_signal(l, i) as u128).collect())
            }
        }
    }

    fn tick(&mut self) -> Result<(), ErrorReport> {
        match self {
            GeneralEngine::Cpu(c) => c.tick(),
            #[cfg(feature = "jit")]
            GeneralEngine::Jit { jit, .. } => {
                jit.tick();
                Ok(())
            }
        }
    }

    fn tick_many(&mut self, n: usize) -> Result<(), ErrorReport> {
        match self {
            GeneralEngine::Cpu(c) => c.tick_many(n),
            #[cfg(feature = "jit")]
            GeneralEngine::Jit { jit, .. } => {
                jit.tick_many(n);
                Ok(())
            }
        }
    }

    fn is_jit(&self) -> bool {
        #[cfg(feature = "jit")]
        {
            matches!(self, GeneralEngine::Jit { .. })
        }
        #[cfg(not(feature = "jit"))]
        {
            false
        }
    }

    /// Read/write a signal by its LOCAL (sliced-program) index — used for
    /// boundary exchange of signals that have no top-level handle (internal
    /// registers exposed as boundary inputs). Only the JIT backend supports it
    /// (it is index-addressed); the CPU is handle-addressed, so coupled hybrids
    /// require the JIT general backend.
    fn get_local(&self, _local: usize, _lane: usize) -> Result<u128, ErrorReport> {
        match self {
            #[cfg(feature = "jit")]
            GeneralEngine::Jit { jit, .. } => Ok(jit.get_signal(_lane, _local) as u128),
            _ => Err(hyb_err("get_local requires the JIT backend")),
        }
    }
    fn set_local(&mut self, _local: usize, _lane: usize, _value: u128) -> Result<(), ErrorReport> {
        match self {
            #[cfg(feature = "jit")]
            GeneralEngine::Jit { jit, .. } => {
                jit.set_signal(_lane, _local, _value as u32);
                Ok(())
            }
            _ => Err(hyb_err("set_local requires the JIT backend")),
        }
    }
}

/// Unwrap an output's defining expression to the single register it aliases
/// (`Signal(reg)`, possibly through a width cast/extension), else `None`.
fn alias_target(expr: &PackedExpr) -> Option<usize> {
    match &expr.kind {
        PackedExprKind::Signal(s) => Some(*s),
        PackedExprKind::Cast(a) | PackedExprKind::Zext(a) | PackedExprKind::Trunc(a) => alias_target(a),
        _ => None,
    }
}

pub struct HybridSimulator {
    program: PackedProgram,
    lin: LinearAot,
    nl: GeneralEngine,
    general_backend: &'static str,
    lin_caller_inputs: HashSet<usize>,  // PRIMARY inputs the caller drives into the AOT
    lin_regs: HashSet<usize>,           // linear register dsts (read from the AOT)
    output_alias: HashMap<usize, usize>, // output signal idx -> linear register idx it aliases
    nl_handles: HashSet<Signal>,        // handles the general engine serves
    // Boundary exchange (coupled partitions). Indices: lin uses GLOBAL signal
    // indices; the general engine uses LOCAL sliced-program indices.
    gen_from_lin: Vec<(usize, usize)>,  // (general local boundary input, linear reg global)
    lin_from_general: Vec<(usize, usize)>, // (linear-leaf global, general local owning it)
    coupled: bool,
    leap: LinearLeap, // temporal-leap operator for the linear partition
    lanes: usize,
}

impl HybridSimulator {
    pub fn new(program: &PackedProgram, lanes: usize) -> Result<Self, ErrorReport> {
        let lin = LinearAot::compile(program, lanes)?;
        let lin_regs: HashSet<usize> = lin.linear_signals().iter().copied().collect();

        // Output ports that are simple aliases of a linear register are served by
        // the AOT; all other outputs (and the non-linear registers) are observed
        // so the general slice computes them.
        let mut output_alias: HashMap<usize, usize> = HashMap::new();
        for stream in [&program.streams.comb, &program.streams.async_reset_comb] {
            for packet in stream {
                for op in &packet.ops {
                    if let PackedOp::Assign { dst, expr } = op {
                        if program.signals[*dst].kind == PackedSignalKind::Output {
                            if let Some(src) = alias_target(expr) {
                                if lin_regs.contains(&src) {
                                    output_alias.insert(*dst, src);
                                }
                            }
                        }
                    }
                }
            }
        }

        // observe = non-linear registers ∪ outputs not served by the linear side.
        let mut observe: Vec<usize> = Vec::new();
        for (i, sig) in program.signals.iter().enumerate() {
            let nonlinear_reg = sig.kind == PackedSignalKind::Reg && !lin_regs.contains(&i);
            let nonlinear_output = sig.kind == PackedSignalKind::Output && !output_alias.contains_key(&i);
            if nonlinear_reg || nonlinear_output {
                observe.push(i);
            }
        }
        let (present_sig, present_mem) = cone_of_influence(program, &observe, &[]);
        let slice = slice_present(program, &present_sig, &present_mem)?;
        let nl_handles: HashSet<Signal> = slice.program.signals.iter().filter_map(|s| s.source).collect();

        // Boundary exchange routes (coupling). A non-linear register is one the
        // general engine owns; a linear leaf that is such a register must be fed
        // to the AOT each cycle, and a general boundary input that is a linear
        // register must be fed from the AOT.
        let nonlinear_reg: HashSet<usize> = program
            .signals
            .iter()
            .enumerate()
            .filter(|(i, s)| s.kind == PackedSignalKind::Reg && !lin_regs.contains(i))
            .map(|(i, _)| i)
            .collect();
        let global_to_local: HashMap<usize, usize> =
            slice.signal_origin.iter().enumerate().map(|(local, &g)| (g, local)).collect();
        // general boundary inputs that are linear registers → fed from the AOT.
        let gen_from_lin: Vec<(usize, usize)> = slice
            .boundary_inputs
            .iter()
            .filter_map(|&b| {
                let g = slice.signal_origin[b];
                lin_regs.contains(&g).then_some((b, g))
            })
            .collect();
        // linear leaves that are non-linear registers → fed from the general engine.
        let lin_from_general: Vec<(usize, usize)> = lin
            .input_leaves()
            .iter()
            .filter_map(|&g| {
                (nonlinear_reg.contains(&g)).then(|| global_to_local.get(&g).map(|&local| (g, local)))?
            })
            .collect();
        let coupled = !gen_from_lin.is_empty() || !lin_from_general.is_empty();

        // Primary inputs the caller drives into the AOT = linear leaves that are
        // NOT boundary registers (those come from exchange).
        let lin_caller_inputs: HashSet<usize> =
            lin.input_leaves().iter().copied().filter(|g| !nonlinear_reg.contains(g)).collect();

        let (nl, general_backend) = GeneralEngine::build(slice.program, lanes)?;
        if coupled && !nl.is_jit() {
            return Err(hyb_err(
                "coupled partitions need the vector-JIT general backend (boundary inputs are index-addressed); design did not fit the JIT",
            ));
        }

        let leap = LinearLeap::build(program)
            .ok_or_else(|| hyb_err("no linear register cones to build a leap operator"))?;
        Ok(Self {
            program: program.clone(),
            lin,
            nl,
            general_backend,
            lin_caller_inputs,
            lin_regs,
            output_alias,
            nl_handles,
            gen_from_lin,
            lin_from_general,
            coupled,
            leap,
            lanes,
        })
    }

    /// Exchange register-stable boundary values between the partitions. Uses the
    /// values committed by the previous cycle (the "old" register values a
    /// synchronous tick reads), so it must run before both partitions tick.
    fn exchange(&mut self) -> Result<(), ErrorReport> {
        for &(local, g) in &self.gen_from_lin {
            for lane in 0..self.lanes {
                let v = self.lin.get_signal(g, lane);
                self.nl.set_local(local, lane, v)?;
            }
        }
        for &(g, local) in &self.lin_from_general {
            for lane in 0..self.lanes {
                let v = self.nl.get_local(local, lane)?;
                self.lin.set_signal(g, lane, v);
            }
        }
        Ok(())
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    /// Which general backend was auto-picked for the non-linear partition
    /// (`"vector-JIT"` or `"SIMD-CPU"`). The linear partition always uses the
    /// GF(2) matrix AOT.
    pub fn general_backend(&self) -> &'static str {
        self.general_backend
    }

    /// Drive an input across all lanes. Routed to whichever partition reads it
    /// (a shared input such as a clock/reset goes to both).
    pub fn set_signal(&mut self, signal: Signal, lane_values: &[u128]) -> Result<(), ErrorReport> {
        if lane_values.len() != self.lanes {
            return Err(hyb_err("lane_values length != lanes"));
        }
        if let Some(fi) = self.program.signal_index(signal) {
            if self.lin_caller_inputs.contains(&fi) {
                for (l, v) in lane_values.iter().enumerate() {
                    self.lin.set_signal(fi, l, *v);
                }
            }
        }
        if self.nl_handles.contains(&signal) {
            self.nl.set_signal(signal, lane_values)?;
        }
        Ok(())
    }

    /// Read a signal across all lanes. Linear registers (and output ports that
    /// alias one) come from the matrix AOT; everything else from the general engine.
    pub fn get_signal(&self, signal: Signal) -> Result<Vec<u128>, ErrorReport> {
        if let Some(fi) = self.program.signal_index(signal) {
            if let Some(&reg) = self.output_alias.get(&fi) {
                return Ok((0..self.lanes).map(|l| self.lin.get_signal(reg, l)).collect());
            }
            if self.lin_regs.contains(&fi) {
                return Ok((0..self.lanes).map(|l| self.lin.get_signal(fi, l)).collect());
            }
        }
        if self.nl_handles.contains(&signal) {
            return self.nl.get_signal(signal);
        }
        Err(hyb_err("signal not served by either partition"))
    }

    pub fn tick(&mut self) -> Result<(), ErrorReport> {
        if self.coupled {
            self.exchange()?;
        }
        self.lin.tick();
        self.nl.tick()
    }

    /// Advance `n` cycles. For INDEPENDENT partitions the two engines run
    /// CONCURRENTLY (linear matrix AOT on a worker thread — it is `Send` — and the
    /// general engine on the current thread; disjoint state, no synchronization).
    /// For COUPLED partitions each cycle exchanges boundary registers, so the
    /// engines run sequentially per cycle behind that barrier.
    pub fn tick_many(&mut self, n: usize) -> Result<(), ErrorReport> {
        if self.coupled {
            for _ in 0..n {
                self.tick()?;
            }
            return Ok(());
        }
        let lin = &mut self.lin;
        let nl = &mut self.nl;
        let mut nl_res = Ok(());
        std::thread::scope(|sc| {
            sc.spawn(|| lin.tick_many(n));
            nl_res = nl.tick_many(n);
        });
        nl_res
    }

    /// Whether the partitions are coupled (require per-cycle boundary exchange).
    pub fn is_coupled(&self) -> bool {
        self.coupled
    }

    /// Advance `n` cycles by TEMPORAL LEAP: the (independent) linear partition is
    /// jumped `n` cycles via `M^n` in O(log n) matrix mults instead of n steps,
    /// while the non-linear partition steps normally. Bit-identical to
    /// `tick_many(n)` **when the linear partition's inputs are held idle (zero)
    /// over the n cycles** — its natural domain (free-running LFSR/scrambler/
    /// counter/CRC-between-feeds). Falls back to `tick_many` when the partitions
    /// are coupled (the linear inputs then vary with the non-linear state).
    pub fn leap(&mut self, n: u64) -> Result<(), ErrorReport> {
        if self.coupled {
            return self.tick_many(n as usize);
        }
        // Build the n-cycle operator once; apply M^n to every lane's linear state.
        let op = self.leap.idle_operator(n);
        let regs: Vec<(usize, u32)> = self.leap.registers().to_vec();
        for lane in 0..self.lanes {
            let mut s = 0u128;
            for &(dst, _) in &regs {
                let base = self.leap.bit_base(dst).unwrap();
                s |= self.lin.get_signal(dst, lane) << base;
            }
            let s2 = self.leap.apply_idle(&op, s);
            for &(dst, w) in &regs {
                let base = self.leap.bit_base(dst).unwrap();
                let mask = if w >= 128 { u128::MAX } else { (1u128 << w) - 1 };
                self.lin.set_signal(dst, lane, (s2 >> base) & mask);
            }
        }
        self.nl.tick_many(n as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower_to_packed_program;
    use rrtl_sv_frontend::import_sv;

    // mixed.sv: linear CRC offloaded to the AOT, non-linear acc/count on the
    // general engine; the hybrid must match the full SIMD CPU on every output.
    #[test]
    fn hybrid_matches_full_simd_cpu_mixed() {
        let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv/mixed.sv")).unwrap();
        let imported = import_sv(&src, Some("mixed")).unwrap();
        let compiled = rrtl_core::compile(&imported.design).unwrap();
        let program = lower_to_packed_program(&compiled, "mixed").unwrap();
        let h = |n: &str| compiled.find_module("mixed").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
        let lanes = 64;

        let mut hyb = HybridSimulator::new(&program, lanes).unwrap();
        let mut full = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        let clk = vec![1u128; lanes];
        full.set_signal(h("clk"), &clk).unwrap();
        hyb.set_signal(h("clk"), &clk).unwrap();

        for cyc in 0..40u64 {
            for (name, m) in [("rst", 1u128), ("din", 0xff), ("a", 0xffff), ("b", 0xffff)] {
                let vals: Vec<u128> = (0..lanes)
                    .map(|l| if name == "rst" { (cyc < 1) as u128 } else { (cyc.wrapping_mul(2654435761).wrapping_add(l as u64) as u128) & m })
                    .collect();
                full.set_signal(h(name), &vals).unwrap();
                hyb.set_signal(h(name), &vals).unwrap();
            }
            full.tick().unwrap();
            hyb.tick().unwrap();
            for out in ["crc", "acc", "count"] {
                let fv = full.get_signal(h(out)).unwrap();
                let hv = hyb.get_signal(h(out)).unwrap();
                assert_eq!(hv, fv, "{out} @ cyc{cyc}");
            }
        }
    }

    // mixed_coupled.sv: crc reads acc and acc reads crc across the cut, so the
    // hybrid must EXCHANGE boundary registers every cycle. Coupling needs the
    // JIT general backend (boundary inputs are index-addressed).
    #[cfg(feature = "jit")]
    #[test]
    fn hybrid_matches_full_simd_cpu_coupled() {
        let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv/mixed_coupled.sv")).unwrap();
        let imported = import_sv(&src, Some("mixed_coupled")).unwrap();
        let compiled = rrtl_core::compile(&imported.design).unwrap();
        let program = lower_to_packed_program(&compiled, "mixed_coupled").unwrap();
        let h = |n: &str| compiled.find_module("mixed_coupled").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
        let lanes = 64;

        let mut hyb = HybridSimulator::new(&program, lanes).unwrap();
        assert!(hyb.is_coupled(), "design should be detected as coupled");
        let mut full = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        let clk = vec![1u128; lanes];
        full.set_signal(h("clk"), &clk).unwrap();
        hyb.set_signal(h("clk"), &clk).unwrap();

        for cyc in 0..40u64 {
            for (name, m) in [("rst", 1u128), ("din", 0xff), ("a", 0xffff), ("b", 0xffff)] {
                let vals: Vec<u128> = (0..lanes)
                    .map(|l| if name == "rst" { (cyc < 1) as u128 } else { (cyc.wrapping_mul(2654435761).wrapping_add(l as u64) as u128) & m })
                    .collect();
                full.set_signal(h(name), &vals).unwrap();
                hyb.set_signal(h(name), &vals).unwrap();
            }
            full.tick().unwrap();
            hyb.tick().unwrap();
            for out in ["crc", "acc", "count"] {
                let fv = full.get_signal(h(out)).unwrap();
                let hv = hyb.get_signal(h(out)).unwrap();
                assert_eq!(hv, fv, "{out} @ cyc{cyc}");
            }
        }
    }

    // leap(n) must equal tick_many(n) when the linear partition (crc) is idle
    // (din held 0): crc jumps via M^n, acc/count step. mixed is independent.
    #[test]
    fn hybrid_leap_matches_tick_many_idle() {
        let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv/mixed.sv")).unwrap();
        let imported = import_sv(&src, Some("mixed")).unwrap();
        let compiled = rrtl_core::compile(&imported.design).unwrap();
        let program = lower_to_packed_program(&compiled, "mixed").unwrap();
        let h = |n: &str| compiled.find_module("mixed").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
        let lanes = 64;

        let mut leaped = HybridSimulator::new(&program, lanes).unwrap();
        let mut stepped = HybridSimulator::new(&program, lanes).unwrap();
        assert!(!leaped.is_coupled());
        let clk = vec![1u128; lanes];
        let a: Vec<u128> = (0..lanes).map(|l| (l as u128 * 13 + 1) & 0xffff).collect();
        let b: Vec<u128> = (0..lanes).map(|l| (l as u128 * 7 + 5) & 0xffff).collect();
        for s in [&mut leaped, &mut stepped] {
            s.set_signal(h("clk"), &clk).unwrap();
            // reset once to seed crc=FFFFFFFF, then idle (din=0), a/b held.
            s.set_signal(h("rst"), &vec![1u128; lanes]).unwrap();
            s.set_signal(h("din"), &vec![0u128; lanes]).unwrap();
            s.set_signal(h("a"), &a).unwrap();
            s.set_signal(h("b"), &b).unwrap();
            s.tick().unwrap();
            s.set_signal(h("rst"), &vec![0u128; lanes]).unwrap();
        }
        let n = 100_000u64;
        stepped.tick_many(n as usize).unwrap();
        leaped.leap(n).unwrap();
        for out in ["crc", "acc", "count"] {
            assert_eq!(leaped.get_signal(h(out)).unwrap(), stepped.get_signal(h(out)).unwrap(), "{out} after leap({n})");
        }
    }
}
