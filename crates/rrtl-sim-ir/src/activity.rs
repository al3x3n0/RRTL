//! Activity analysis for activity-based ("event-driven") skipping: for each
//! register, the set of *leaf* signals (inputs, registers, top-level signals)
//! that its next-state combinational cone transitively reads. A register's
//! next-state is unchanged on any cycle where none of its support signals changed
//! value, so its cone (and commit) can be skipped.
//!
//! This is the reusable substrate; a profiler (see the `activity_probe` example)
//! uses it to measure the skip *potential* per design and per stimulus regime
//! (per-lane vs all-lanes-in-a-tile) before the skipping machinery is wired into
//! a simulator — activity skipping is regime-dependent (it pays on idle /
//! correlated workloads, not on busy / decorrelated ones).

use std::collections::{HashMap, HashSet};

use crate::{PackedExpr, PackedExprKind, PackedOp, PackedProgram};

/// The combinational fan-in support of one register's next-state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisterSupport {
    /// Signal index of the register.
    pub reg: usize,
    /// Leaf signal indices the next-state cone reads (inputs/registers/others
    /// with no combinational definition in this program).
    pub support: Vec<usize>,
    /// Whether the cone reads a memory (treated as always-active: a register that
    /// reads memory can never be skipped on a memory-only change).
    pub reads_memory: bool,
}

/// Collect the direct signal reads and memory-read flag of an expression.
fn expr_reads(expr: &PackedExpr, signals: &mut Vec<usize>, reads_mem: &mut bool) {
    match &expr.kind {
        PackedExprKind::Lit(_) => {}
        PackedExprKind::Signal(idx) => signals.push(*idx),
        PackedExprKind::Not(a)
        | PackedExprKind::Zext(a)
        | PackedExprKind::Sext(a)
        | PackedExprKind::Trunc(a)
        | PackedExprKind::Cast(a)
        | PackedExprKind::Slice { expr: a, .. } => expr_reads(a, signals, reads_mem),
        PackedExprKind::And(a, b)
        | PackedExprKind::Or(a, b)
        | PackedExprKind::Xor(a, b)
        | PackedExprKind::Add(a, b)
        | PackedExprKind::Sub(a, b)
        | PackedExprKind::Mul(a, b)
        | PackedExprKind::Eq(a, b)
        | PackedExprKind::Ne(a, b) => {
            expr_reads(a, signals, reads_mem);
            expr_reads(b, signals, reads_mem);
        }
        PackedExprKind::Lt { lhs, rhs, .. } => {
            expr_reads(lhs, signals, reads_mem);
            expr_reads(rhs, signals, reads_mem);
        }
        PackedExprKind::Mux {
            cond,
            then_expr,
            else_expr,
        } => {
            expr_reads(cond, signals, reads_mem);
            expr_reads(then_expr, signals, reads_mem);
            expr_reads(else_expr, signals, reads_mem);
        }
        PackedExprKind::Concat(parts) => {
            for p in parts {
                expr_reads(p, signals, reads_mem);
            }
        }
        PackedExprKind::MemRead { addr, .. } => {
            *reads_mem = true;
            expr_reads(addr, signals, reads_mem);
        }
    }
}

/// The signals an expression *directly* reads (its immediate combinational
/// fan-in, not transitively expanded) plus whether it reads a memory. Used by
/// the activity-skipping JIT to build per-cone dirty guards in topological order.
pub fn expr_direct_reads(expr: &PackedExpr) -> (Vec<usize>, bool) {
    let mut signals = Vec::new();
    let mut reads_mem = false;
    expr_reads(expr, &mut signals, &mut reads_mem);
    signals.sort_unstable();
    signals.dedup();
    (signals, reads_mem)
}

/// Build, for each register, the leaf support of its next-state cone.
pub fn register_support(program: &PackedProgram) -> Vec<RegisterSupport> {
    // signal index -> its combinational definition expression (from comb streams).
    let mut comb_def: HashMap<usize, &PackedExpr> = HashMap::new();
    for stream in [&program.streams.async_reset_comb, &program.streams.comb] {
        for packet in stream {
            for op in &packet.ops {
                if let PackedOp::Assign { dst, expr } = op {
                    comb_def.insert(*dst, expr);
                }
            }
        }
    }

    // Memoized leaf set per signal: a signal with no comb definition is a leaf.
    let mut memo: HashMap<usize, (HashSet<usize>, bool)> = HashMap::new();

    fn leaves_of_signal(
        sig: usize,
        comb_def: &HashMap<usize, &PackedExpr>,
        memo: &mut HashMap<usize, (HashSet<usize>, bool)>,
        on_stack: &mut HashSet<usize>,
    ) -> (HashSet<usize>, bool) {
        if let Some(cached) = memo.get(&sig) {
            return cached.clone();
        }
        let Some(expr) = comb_def.get(&sig).copied() else {
            // Leaf: input/register/undriven.
            let mut set = HashSet::new();
            set.insert(sig);
            return (set, false);
        };
        if !on_stack.insert(sig) {
            // Combinational cycle guard: treat as a leaf.
            let mut set = HashSet::new();
            set.insert(sig);
            return (set, false);
        }
        let mut reads = Vec::new();
        let mut reads_mem = false;
        expr_reads(expr, &mut reads, &mut reads_mem);
        let mut leaves = HashSet::new();
        for r in reads {
            let (sub, m) = leaves_of_signal(r, comb_def, memo, on_stack);
            leaves.extend(sub);
            reads_mem |= m;
        }
        on_stack.remove(&sig);
        memo.insert(sig, (leaves.clone(), reads_mem));
        (leaves, reads_mem)
    }

    let mut out = Vec::new();
    for packet in &program.streams.tick_next {
        for op in &packet.ops {
            if let PackedOp::CaptureReg { dst, next, .. } = op {
                let mut reads = Vec::new();
                let mut reads_mem = false;
                expr_reads(next, &mut reads, &mut reads_mem);
                let mut support: HashSet<usize> = HashSet::new();
                let mut on_stack = HashSet::new();
                for r in reads {
                    let (sub, m) = leaves_of_signal(r, &comb_def, &mut memo, &mut on_stack);
                    support.extend(sub);
                    reads_mem |= m;
                }
                let mut support: Vec<usize> = support.into_iter().collect();
                support.sort_unstable();
                out.push(RegisterSupport {
                    reg: *dst,
                    support,
                    reads_memory: reads_mem,
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower_to_packed_program;
    use rrtl_core::{compile, lit_u, uint, Design};

    #[test]
    fn support_of_enabled_counter() {
        // q' = en ? q + 1 : q  -> support = {en, q}; a free input `unused` is not in it.
        let mut design = Design::new();
        {
            let mut m = design.module("M");
            let clk = m.input("clk", uint(1));
            let en = m.input("en", uint(1));
            let _unused = m.input("unused", uint(8));
            let q = m.reg("q", uint(8));
            m.clock(q, clk);
            m.next(q, rrtl_core::mux(en, q.value() + lit_u(1, 8), q.value()));
            let o = m.output("o", uint(8));
            m.assign(o, q);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "M").unwrap();
        let supports = register_support(&program);
        let q_idx = program.signals.iter().position(|s| s.name.ends_with(".q")).unwrap();
        let en_idx = program.signals.iter().position(|s| s.name.ends_with(".en")).unwrap();
        let unused_idx = program.signals.iter().position(|s| s.name.ends_with(".unused")).unwrap();
        let rs = supports.iter().find(|r| r.reg == q_idx).unwrap();
        assert!(rs.support.contains(&en_idx), "support should include en");
        assert!(rs.support.contains(&q_idx), "support should include q itself");
        assert!(!rs.support.contains(&unused_idx), "support must exclude unused input");
    }
}
