//! Instance-level data parallelism for the GPU interpreter: when a design is N
//! independent instances of one child module (see
//! [`rrtl_sim_ir::analyze_instance_fold`]), simulate the child *once* at
//! `N × base_lanes` lanes instead of N flattened copies at `base_lanes`. Instance
//! `i` occupies lane block `[i·base_lanes, (i+1)·base_lanes)`. This converts
//! otherwise-idle GPU parallelism into useful work (a large win when the base lane
//! count is below GPU saturation) and shrinks the program N-fold.

use rrtl_core::CompiledDesign;
use rrtl_ir::{Diagnostic, ErrorReport};
use rrtl_sim_ir::{
    analyze_instance_fold, lower_to_packed_program, InstanceFold, PackedProgram, PackedSignalKind,
};

use crate::interp::{InterpGpuSimulator, InterpProgram};

fn fold_error(msg: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![Diagnostic::new("E_INSTANCE_FOLD", msg)])
}

/// Backend-agnostic fold plan: the detected fold plus the child module lowered
/// standalone (its port offsets are shared by every instance/lane block).
pub struct InstanceFoldPlan {
    pub fold: InstanceFold,
    pub child: PackedProgram,
}

impl InstanceFoldPlan {
    pub fn new(design: &CompiledDesign, top: &str) -> Result<Self, ErrorReport> {
        let fold = analyze_instance_fold(design, top).ok_or_else(|| {
            fold_error(format!(
                "`{top}` is not a flat wrapper of independent identical instances"
            ))
        })?;
        let child = lower_to_packed_program(design, &fold.child_module)?;
        Ok(Self { fold, child })
    }

    pub fn num_instances(&self) -> usize {
        self.fold.num_instances
    }

    /// Storage offset of an input/output port of the child module. Signal names
    /// are owner-path-prefixed (e.g. `M.din`), so match the local (last) segment.
    pub fn port_offset(&self, port: &str) -> Option<usize> {
        self.child
            .signals
            .iter()
            .find(|s| {
                matches!(s.kind, PackedSignalKind::Input | PackedSignalKind::Output)
                    && s.name.rsplit('.').next() == Some(port)
            })
            .map(|s| s.layout.offset)
    }
}

/// GPU simulator that runs the folded child at `num_instances × base_lanes` lanes.
pub struct InstanceFoldSimulator {
    plan: InstanceFoldPlan,
    gpu: InterpGpuSimulator,
    base_lanes: usize,
}

impl InstanceFoldSimulator {
    pub fn new(design: &CompiledDesign, top: &str, base_lanes: usize) -> Result<Self, ErrorReport> {
        let plan = InstanceFoldPlan::new(design, top)?;
        let total = plan.num_instances() * base_lanes;
        let gpu = InterpGpuSimulator::new(&InterpProgram::encode_design(&plan.child)?, total)?;
        Ok(Self {
            plan,
            gpu,
            base_lanes,
        })
    }

    pub fn num_instances(&self) -> usize {
        self.plan.num_instances()
    }
    pub fn base_lanes(&self) -> usize {
        self.base_lanes
    }
    pub fn total_lanes(&self) -> usize {
        self.num_instances() * self.base_lanes
    }
    /// Lane index of stimulus `local` within instance `inst`.
    pub fn instance_lane(&self, inst: usize, local: usize) -> usize {
        inst * self.base_lanes + local
    }
    pub fn port_offset(&self, port: &str) -> Option<usize> {
        self.plan.port_offset(port)
    }

    /// Set a port limb across all `total_lanes` (caller arranges per-instance, with
    /// lane = `instance_lane(inst, local)`).
    pub fn set_port(&self, port: &str, limb: usize, lane_values: &[u32]) -> Result<(), ErrorReport> {
        let off = self
            .port_offset(port)
            .ok_or_else(|| fold_error(format!("no such port `{port}`")))?;
        self.gpu.set_signal(off + limb, lane_values);
        Ok(())
    }
    pub fn get_port(&self, port: &str, limb: usize) -> Result<Vec<u32>, ErrorReport> {
        let off = self
            .port_offset(port)
            .ok_or_else(|| fold_error(format!("no such port `{port}`")))?;
        Ok(self.gpu.get_signal(off + limb))
    }
    pub fn tick_many(&self, steps: usize) {
        self.gpu.tick_many(steps);
    }
    pub fn synchronize(&self) {
        self.gpu.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::InterpRunner;
    use rrtl_core::{compile, lit_u, uint, Design};

    /// Folding N instances into the lane axis must give bit-identical per-instance
    /// results to the naively flattened design. Validated on the deterministic CPU
    /// reference (no GPU dependence).
    #[test]
    fn fold_matches_naive_cpu() {
        let n = 4usize;
        let base = 3usize;
        let total = n * base;

        let mut design = Design::new();
        {
            let mut m = design.module("M");
            let clk = m.input("clk", uint(1));
            let din = m.input("din", uint(32));
            let acc = m.reg("acc", uint(32));
            m.clock(acc, clk);
            m.next(acc, acc * lit_u(0x9e37_79b9, 32) + din);
            let o = m.output("o", uint(32));
            m.assign(o, acc);
        }
        {
            let mut m = design.module("Top");
            let clk = m.input("clk", uint(1));
            for i in 0..n {
                let din = m.input(format!("din{i}"), uint(32));
                let o = m.output(format!("o{i}"), uint(32));
                m.instance(format!("u{i}"), "M", [("clk", clk), ("din", din), ("o", o)]);
            }
        }
        let compiled = compile(&design).unwrap();
        let top = lower_to_packed_program(&compiled, "Top").unwrap();
        let plan = InstanceFoldPlan::new(&compiled, "Top").unwrap();

        let top_off = |name: &str| {
            top.signals
                .iter()
                .find(|s| s.name.rsplit('.').next() == Some(name))
                .unwrap()
                .layout
                .offset
        };
        let mut naive = InterpRunner::new(InterpProgram::encode_design(&top).unwrap(), base);
        let mut folded = InterpRunner::new(InterpProgram::encode_design(&plan.child).unwrap(), total);

        naive.set_signal(top_off("clk"), &vec![1u32; base]);
        folded.set_signal(plan.port_offset("clk").unwrap(), &vec![1u32; total]);

        let din_at = |i: usize, l: usize, cyc: u32| -> u32 {
            ((i as u32) * 1_000 + (l as u32) * 7 + cyc).wrapping_mul(2654435761)
        };
        for cyc in 0..6u32 {
            // naive: per-instance input signals.
            for i in 0..n {
                let v: Vec<u32> = (0..base).map(|l| din_at(i, l, cyc)).collect();
                naive.set_signal(top_off(&format!("din{i}")), &v);
            }
            // folded: one input port, arranged per instance across the lane blocks.
            let v: Vec<u32> = (0..total).map(|lane| din_at(lane / base, lane % base, cyc)).collect();
            folded.set_signal(plan.port_offset("din").unwrap(), &v);

            naive.tick();
            folded.tick();

            let fo = folded.get_signal(plan.port_offset("o").unwrap());
            for i in 0..n {
                let no = naive.get_signal(top_off(&format!("o{i}")));
                assert_eq!(no, &fo[i * base..(i + 1) * base], "instance {i} cycle {cyc}");
            }
        }
    }
}
