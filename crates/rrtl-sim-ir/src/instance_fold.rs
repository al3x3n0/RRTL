//! Instance-level data parallelism: detecting when a design is N *independent*
//! instances of one child module, so they can be folded into the lane dimension
//! (simulate the child once at N×L lanes instead of N copies at L lanes).
//!
//! This is value-oblivious structural analysis only; the actual folding (lower
//! the child standalone, run it at N×L lanes, demux I/O by lane block) is done by
//! the GPU backend's instance-fold simulator. The win is concentrated in the
//! under-saturated regime (few stimulus lanes, large repetitive design), where it
//! converts otherwise-idle GPU parallelism into useful work; it also shrinks the
//! program N-fold (a capacity/compile win) in all regimes.

use std::collections::HashSet;

use rrtl_core::{CompiledDesign, PortDirection};
use rrtl_ir::Signal;

/// A connection from a parent (top) signal to an instance port.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldConnection {
    pub top_signal: Signal,
    pub instance: usize,
    pub port: String,
    pub direction: PortDirection,
}

/// A foldable design: `num_instances` independent instances of `child_module`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstanceFold {
    pub child_module: String,
    pub num_instances: usize,
    pub instance_names: Vec<String>,
    pub connections: Vec<FoldConnection>,
}

/// Detect whether `top` is a flat wrapper around N≥2 independent instances of a
/// single child module, returning the fold plan if so. Conservative: requires the
/// top to have no logic of its own (registers/assignments/memory), all instances
/// to be the same non-external module, no two instances to drive the same signal,
/// and no instance output to feed another instance's input (true independence).
/// Shared inputs (e.g. a common clock) are allowed.
pub fn analyze_instance_fold(design: &CompiledDesign, top: &str) -> Option<InstanceFold> {
    let m = design.find_module(top)?;
    if m.instances.len() < 2 {
        return None;
    }
    // The top must be a pure structural wrapper.
    if !m.registers.is_empty()
        || !m.memory_writes.is_empty()
        || !m.assignments.is_empty()
        || !m.memory_writes.is_empty()
    {
        return None;
    }
    let child = m.instances[0].module.clone();
    if !m.instances.iter().all(|i| i.module == child) {
        return None;
    }
    let child_mod = design.find_module(&child)?;
    if child_mod.is_external {
        return None;
    }

    // Independence: a signal driven by an instance output must not feed any
    // instance input, and no signal may be driven by two instance outputs.
    let mut in_signals: HashSet<Signal> = HashSet::new();
    let mut out_signals: HashSet<Signal> = HashSet::new();
    let mut connections = Vec::new();
    for (idx, inst) in m.instances.iter().enumerate() {
        for c in &inst.connections {
            connections.push(FoldConnection {
                top_signal: c.signal,
                instance: idx,
                port: c.port.clone(),
                direction: c.direction,
            });
            match c.direction {
                PortDirection::Input => {
                    in_signals.insert(c.signal);
                }
                PortDirection::Output => {
                    if !out_signals.insert(c.signal) {
                        return None; // two instances drive the same signal
                    }
                }
                PortDirection::Inout => return None, // unsupported
            }
        }
    }
    if in_signals.intersection(&out_signals).next().is_some() {
        return None; // inter-instance dataflow
    }

    Some(InstanceFold {
        child_module: child,
        num_instances: m.instances.len(),
        instance_names: m.instances.iter().map(|i| i.name.clone()).collect(),
        connections,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rrtl_core::{compile, lit_u, uint, Design};

    fn child_design() -> Design {
        let mut design = Design::new();
        {
            let mut m = design.module("M");
            let clk = m.input("clk", uint(1));
            let din = m.input("din", uint(32));
            let acc = m.reg("acc", uint(32));
            m.clock(acc, clk);
            m.next(acc, acc * lit_u(3, 32) + din);
            let o = m.output("o", uint(32));
            m.assign(o, acc);
        }
        design
    }

    #[test]
    fn detects_independent_instances() {
        let mut design = child_design();
        {
            let mut m = design.module("Top");
            let clk = m.input("clk", uint(1));
            // 4 independent instances, each with its own din/o, sharing clk.
            for i in 0..4 {
                let din = m.input(format!("din{i}"), uint(32));
                let o = m.output(format!("o{i}"), uint(32));
                m.instance(format!("u{i}"), "M", [("clk", clk), ("din", din), ("o", o)]);
            }
        }
        let compiled = compile(&design).unwrap();
        let fold = analyze_instance_fold(&compiled, "Top").expect("should fold");
        assert_eq!(fold.child_module, "M");
        assert_eq!(fold.num_instances, 4);
    }

    #[test]
    fn rejects_inter_instance_dataflow() {
        let mut design = child_design();
        {
            let mut m = design.module("Top");
            let clk = m.input("clk", uint(1));
            let din = m.input("din", uint(32));
            // chain: u0.o feeds u1.din -> NOT independent.
            let mid = m.wire("mid", uint(32));
            let o = m.output("o", uint(32));
            m.instance("u0", "M", [("clk", clk), ("din", din), ("o", mid)]);
            m.instance("u1", "M", [("clk", clk), ("din", mid), ("o", o)]);
        }
        let compiled = compile(&design).unwrap();
        assert!(analyze_instance_fold(&compiled, "Top").is_none());
    }
}
