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

use crate::linear_aot::LinearAot;
use crate::{
    cone_of_influence, slice_present, PackedExpr, PackedExprKind, PackedOp, PackedProgram,
    PackedSignalKind, SimdCpuSimulator,
};

fn hyb_err(msg: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![Diagnostic::new("E_HYBRID", msg)])
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
    nl: SimdCpuSimulator,
    lin_inputs: HashSet<usize>,         // program.signals indices the linear AOT drives
    lin_regs: HashSet<usize>,           // linear register dsts (read from the AOT)
    output_alias: HashMap<usize, usize>, // output signal idx -> linear register idx it aliases
    nl_handles: HashSet<Signal>,        // handles the general engine serves
    lanes: usize,
}

impl HybridSimulator {
    pub fn new(program: &PackedProgram, lanes: usize) -> Result<Self, ErrorReport> {
        let lin = LinearAot::compile(program, lanes)?;
        let lin_regs: HashSet<usize> = lin.linear_signals().iter().copied().collect();
        let lin_inputs: HashSet<usize> = lin.input_leaves().iter().copied().collect();

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
        let nl_program = slice_present(program, &present_sig, &present_mem)?.program;
        let nl_handles: HashSet<Signal> = nl_program.signals.iter().filter_map(|s| s.source).collect();
        let nl = SimdCpuSimulator::new(nl_program, lanes)?;

        Ok(Self {
            program: program.clone(),
            lin,
            nl,
            lin_inputs,
            lin_regs,
            output_alias,
            nl_handles,
            lanes,
        })
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    /// Drive an input across all lanes. Routed to whichever partition reads it
    /// (a shared input such as a clock/reset goes to both).
    pub fn set_signal(&mut self, signal: Signal, lane_values: &[u128]) -> Result<(), ErrorReport> {
        if lane_values.len() != self.lanes {
            return Err(hyb_err("lane_values length != lanes"));
        }
        if let Some(fi) = self.program.signal_index(signal) {
            if self.lin_inputs.contains(&fi) {
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
        self.lin.tick();
        self.nl.tick()
    }

    /// Advance `n` cycles, running the two partitions CONCURRENTLY: the linear
    /// matrix AOT on a worker thread (it is `Send`) and the general engine on the
    /// current thread. Disjoint state — no synchronization.
    pub fn tick_many(&mut self, n: usize) -> Result<(), ErrorReport> {
        let lin = &mut self.lin;
        let nl = &mut self.nl;
        let mut nl_res = Ok(());
        std::thread::scope(|sc| {
            sc.spawn(|| lin.tick_many(n));
            nl_res = nl.tick_many(n);
        });
        nl_res
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
}
