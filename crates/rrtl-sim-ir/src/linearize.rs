//! GF(2)-linearity analysis — the front end of a tensor-core specialization.
//!
//! A combinational cone is *affine-linear over GF(2)* in the leaf signals
//! (registers / primary inputs) iff every operation is XOR / NOT, a constant
//! mask or offset (`AND`/`OR` with a literal), or pure bit-routing
//! (slice / concat / zext / sext / trunc / cast). Such a cone computes
//! `out = M · in ⊕ c` over GF(2) — a binary matrix-vector product — which maps
//! to a 1-bit tensor-core (XOR/AND-popcount BMMA) or bit-sliced XOR-popcount
//! GEMM in the batch regime (many stimuli fill the matrix's batch dimension).
//!
//! `AND`/`OR` with a non-constant operand, `MUX`, arithmetic, comparisons and
//! memory reads are non-linear. Sync/async reset lives in `CaptureReg.reset`, so
//! a reset register's next-state cone can still be linear.

use std::collections::{HashMap, HashSet};

use crate::{PackedExpr, PackedExprKind, PackedOp, PackedProgram};

/// Prevalence of GF(2)-linear logic in a program (the tensor-core prize size).
#[derive(Default, Debug, Clone)]
pub struct LinearityReport {
    pub comb_signals: usize,
    pub linear_signals: usize,
    /// expr-node cost summed over combinational definitions, and the linear share.
    pub comb_cost: usize,
    pub linear_cost: usize,
    /// register next-state cones (the per-cycle sequential update logic).
    pub reg_cones: usize,
    pub linear_reg_cones: usize,
    pub reg_cone_bits: usize,
    pub linear_reg_cone_bits: usize,
}

impl LinearityReport {
    pub fn linear_cost_frac(&self) -> f64 {
        self.linear_cost as f64 / self.comb_cost.max(1) as f64
    }
    pub fn linear_reg_bit_frac(&self) -> f64 {
        self.linear_reg_cone_bits as f64 / self.reg_cone_bits.max(1) as f64
    }
}

fn is_const(e: &PackedExpr) -> bool {
    matches!(e.kind, PackedExprKind::Lit(_))
}

/// Recursive expr-node count (a crude cost proxy).
fn node_cost(e: &PackedExpr) -> usize {
    use PackedExprKind::*;
    1 + match &e.kind {
        Lit(_) | Signal(_) => 0,
        Not(a) | Slice { expr: a, .. } | Zext(a) | Sext(a) | Trunc(a) | Cast(a)
        | MemRead { addr: a, .. } => node_cost(a),
        And(a, b) | Or(a, b) | Xor(a, b) | Add(a, b) | Sub(a, b) | Mul(a, b) | Eq(a, b)
        | Ne(a, b) => node_cost(a) + node_cost(b),
        Lt { lhs, rhs, .. } => node_cost(lhs) + node_cost(rhs),
        Mux { cond, then_expr, else_expr } => {
            node_cost(cond) + node_cost(then_expr) + node_cost(else_expr)
        }
        Concat(parts) => parts.iter().map(node_cost).sum(),
    }
}

struct Cls<'a> {
    comb_def: HashMap<usize, &'a PackedExpr>,
    memo: HashMap<usize, bool>,
}

impl<'a> Cls<'a> {
    fn sig_linear(&mut self, idx: usize, on: &mut HashSet<usize>) -> bool {
        if let Some(&b) = self.memo.get(&idx) {
            return b;
        }
        let Some(def) = self.comb_def.get(&idx).copied() else {
            return true; // leaf: register / input — a linear variable
        };
        if !on.insert(idx) {
            return false; // combinational cycle → treat as non-linear (conservative)
        }
        let r = self.expr_linear(def, on);
        on.remove(&idx);
        self.memo.insert(idx, r);
        r
    }

    fn expr_linear(&mut self, e: &'a PackedExpr, on: &mut HashSet<usize>) -> bool {
        use PackedExprKind::*;
        match &e.kind {
            Lit(_) => true,
            Signal(s) => self.sig_linear(*s, on),
            Not(a) | Slice { expr: a, .. } | Zext(a) | Sext(a) | Trunc(a) | Cast(a) => {
                self.expr_linear(a, on)
            }
            Xor(a, b) => self.expr_linear(a, on) && self.expr_linear(b, on),
            // AND/OR with a constant is a per-bit mask/offset → affine-linear.
            And(a, b) | Or(a, b) => {
                (is_const(a) && self.expr_linear(b, on)) || (is_const(b) && self.expr_linear(a, on))
            }
            Concat(parts) => parts.iter().all(|p| self.expr_linear(p, on)),
            // Mux / arithmetic / compares / memory reads are non-linear.
            Add(..) | Sub(..) | Mul(..) | Eq(..) | Ne(..) | Lt { .. } | Mux { .. }
            | MemRead { .. } => false,
        }
    }
}

/// Whether a single expression's cone is GF(2)-affine-linear in the leaves.
pub fn is_linear(program: &PackedProgram, expr: &PackedExpr) -> bool {
    let mut cls = Cls { comb_def: comb_defs(program), memo: HashMap::new() };
    cls.expr_linear(expr, &mut HashSet::new())
}

/// A detected one-hot select chain `mux(sel==L0, D0, mux(sel==L1, D1, ...))` —
/// a `case`/LUT/crossbar over `selector`. Maps to `out = onehot(selector)·D` (an
/// int8 / int-tensor matmul, exact int32 accumulate), or a gather for wide tables.
#[derive(Clone, Debug)]
pub struct SelectInfo {
    pub selector: usize,
    pub sel_width: u32,
    pub cases: usize,
    pub out_width: u32,
    /// op-cost of the select chain's mux+compare nodes (the routing overhead the
    /// matmul replaces; arm-value computation is separate).
    pub chain_cost: usize,
    /// arms (incl default) whose value is a CONSTANT. When all arms are constant
    /// the select is a LUT/ROM → `onehot·table` with a SHARED table → a batched
    /// lookup is one int8 matmul / gather. With data arms it is a crossbar (per-
    /// lane data, no shared matrix) → stays on SIMT.
    pub const_arms: usize,
    pub total_arms: usize,
}

/// True if `e` evaluates to a constant (no signal/memory reads in its cone).
fn is_const_expr(e: &PackedExpr) -> bool {
    use PackedExprKind::*;
    match &e.kind {
        Lit(_) => true,
        Signal(_) | MemRead { .. } => false,
        Not(a) | Slice { expr: a, .. } | Zext(a) | Sext(a) | Trunc(a) | Cast(a) => is_const_expr(a),
        And(a, b) | Or(a, b) | Xor(a, b) | Add(a, b) | Sub(a, b) | Mul(a, b) | Eq(a, b)
        | Ne(a, b) => is_const_expr(a) && is_const_expr(b),
        Lt { lhs, rhs, .. } => is_const_expr(lhs) && is_const_expr(rhs),
        Mux { cond, then_expr, else_expr } => {
            is_const_expr(cond) && is_const_expr(then_expr) && is_const_expr(else_expr)
        }
        Concat(parts) => parts.iter().all(is_const_expr),
    }
}

/// The selector signal a case-condition tests against constants: `sel == const`,
/// or an `Or`-tree of such over the SAME selector (a multi-label case arm).
fn cond_selector(cond: &PackedExpr) -> Option<usize> {
    use PackedExprKind::*;
    match &cond.kind {
        Eq(a, b) => match (&a.kind, &b.kind) {
            (Signal(s), Lit(_)) | (Lit(_), Signal(s)) => Some(*s),
            _ => None,
        },
        Or(a, b) => {
            let (sa, sb) = (cond_selector(a)?, cond_selector(b)?);
            (sa == sb).then_some(sa)
        }
        _ => None,
    }
}

fn find_selects(e: &PackedExpr, sigw: &[u32], out: &mut Vec<SelectInfo>) {
    use PackedExprKind::*;
    // Try to peel a maximal selector chain rooted here.
    if let Mux { cond, then_expr, else_expr } = &e.kind {
        if let Some(sel) = cond_selector(cond) {
            let mut cases = 0usize;
            let mut chain_cost = 0usize;
            let mut cur = e;
            let mut arms: Vec<&PackedExpr> = Vec::new();
            while let Mux { cond, then_expr, else_expr } = &cur.kind {
                match cond_selector(cond) {
                    Some(s) if s == sel => {
                        cases += 1;
                        chain_cost += 2; // the mux + its compare
                        arms.push(then_expr);
                        cur = else_expr;
                    }
                    _ => break,
                }
            }
            if cases >= 2 {
                let const_arms = arms.iter().filter(|a| is_const_expr(a)).count()
                    + is_const_expr(cur) as usize;
                out.push(SelectInfo {
                    selector: sel,
                    sel_width: sigw.get(sel).copied().unwrap_or(0),
                    cases,
                    out_width: e.ty.width,
                    chain_cost,
                    const_arms,
                    total_arms: arms.len() + 1,
                });
                // recurse into the arm values + final default for nested selects
                for arm in arms {
                    find_selects(arm, sigw, out);
                }
                find_selects(cur, sigw, out);
                return;
            }
        }
        // not a select chain → fall through to generic recursion
        find_selects(cond, sigw, out);
        find_selects(then_expr, sigw, out);
        find_selects(else_expr, sigw, out);
        return;
    }
    match &e.kind {
        Lit(_) | Signal(_) => {}
        Not(a) | Slice { expr: a, .. } | Zext(a) | Sext(a) | Trunc(a) | Cast(a)
        | MemRead { addr: a, .. } => find_selects(a, sigw, out),
        And(a, b) | Or(a, b) | Xor(a, b) | Add(a, b) | Sub(a, b) | Mul(a, b) | Eq(a, b)
        | Ne(a, b) => {
            find_selects(a, sigw, out);
            find_selects(b, sigw, out);
        }
        Lt { lhs, rhs, .. } => {
            find_selects(lhs, sigw, out);
            find_selects(rhs, sigw, out);
        }
        Mux { .. } => unreachable!(),
        Concat(parts) => parts.iter().for_each(|p| find_selects(p, sigw, out)),
    }
}

/// Detect all one-hot select chains (`case`/LUT/crossbar) across the per-cycle
/// cones — the int8-tensor-mappable structural class.
pub fn detect_selects(program: &PackedProgram) -> Vec<SelectInfo> {
    let sigw: Vec<u32> = program.signals.iter().map(|s| s.layout.width).collect();
    let mut out = Vec::new();
    for stream in [&program.streams.async_reset_comb, &program.streams.comb] {
        for packet in stream {
            for op in &packet.ops {
                if let PackedOp::Assign { expr, .. } = op {
                    find_selects(expr, &sigw, &mut out);
                }
            }
        }
    }
    for packet in &program.streams.tick_next {
        for op in &packet.ops {
            if let PackedOp::CaptureReg { next, .. } = op {
                find_selects(next, &sigw, &mut out);
            }
        }
    }
    out
}

/// Per-design profile of how the per-cycle op-cost partitions across the EXACT
/// AI-hardware primitive classes (the fp-matmul majority is unusable for bit-
/// exact RTL, so only these map). Disjoint by cone: a fully-GF(2)-linear cone's
/// ops all go to int1 (BMMA); within non-linear cones, MemRead→gather, Mul→int8
/// MAC, the rest→general (SIMT/SIMD only).
#[derive(Default, Debug, Clone)]
pub struct AccelProfile {
    pub total: usize,
    pub linear_int1: usize,
    pub gather: usize,
    pub mul_mac: usize,
    pub general: usize,
}

fn count_by_class(e: &PackedExpr, gather: &mut usize, mac: &mut usize, general: &mut usize) {
    use PackedExprKind::*;
    match &e.kind {
        // constants / signal loads are data movement → general (SIMT/SIMD).
        Lit(_) | Signal(_) => *general += 1,
        MemRead { addr, .. } => {
            *gather += 1;
            count_by_class(addr, gather, mac, general);
        }
        Mul(a, b) => {
            *mac += 1;
            count_by_class(a, gather, mac, general);
            count_by_class(b, gather, mac, general);
        }
        Not(a) | Slice { expr: a, .. } | Zext(a) | Sext(a) | Trunc(a) | Cast(a) => {
            *general += 1;
            count_by_class(a, gather, mac, general);
        }
        And(a, b) | Or(a, b) | Xor(a, b) | Add(a, b) | Sub(a, b) | Eq(a, b) | Ne(a, b) => {
            *general += 1;
            count_by_class(a, gather, mac, general);
            count_by_class(b, gather, mac, general);
        }
        Lt { lhs, rhs, .. } => {
            *general += 1;
            count_by_class(lhs, gather, mac, general);
            count_by_class(rhs, gather, mac, general);
        }
        Mux { cond, then_expr, else_expr } => {
            *general += 1;
            count_by_class(cond, gather, mac, general);
            count_by_class(then_expr, gather, mac, general);
            count_by_class(else_expr, gather, mac, general);
        }
        Concat(parts) => {
            *general += 1;
            parts.iter().for_each(|p| count_by_class(p, gather, mac, general));
        }
    }
}

pub fn accel_profile(program: &PackedProgram) -> AccelProfile {
    // every per-cycle cone: combinational defs + register next-states.
    let comb_def = comb_defs(program);
    let mut cones: Vec<&PackedExpr> = comb_def.values().copied().collect();
    for packet in &program.streams.tick_next {
        for op in &packet.ops {
            if let PackedOp::CaptureReg { next, .. } = op {
                cones.push(next);
            }
        }
    }
    let mut cls = Cls { comb_def, memo: HashMap::new() };
    let mut p = AccelProfile::default();
    for expr in cones {
        let cost = node_cost(expr);
        p.total += cost;
        if cls.expr_linear(expr, &mut HashSet::new()) {
            p.linear_int1 += cost;
        } else {
            count_by_class(expr, &mut p.gather, &mut p.mul_mac, &mut p.general);
        }
    }
    p
}

/// Classify every combinational signal and register next-state cone of `program`.
pub fn classify(program: &PackedProgram) -> LinearityReport {
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

    let mut cls = Cls { comb_def: comb_def.clone(), memo: HashMap::new() };
    let mut rep = LinearityReport::default();

    for (&idx, &def) in &comb_def {
        let cost = node_cost(def);
        rep.comb_signals += 1;
        rep.comb_cost += cost;
        let mut on = HashSet::new();
        if cls.sig_linear(idx, &mut on) {
            rep.linear_signals += 1;
            rep.linear_cost += cost;
        }
    }

    for packet in &program.streams.tick_next {
        for op in &packet.ops {
            if let PackedOp::CaptureReg { next, .. } = op {
                let bits = next.ty.width as usize;
                rep.reg_cones += 1;
                rep.reg_cone_bits += bits;
                let mut on = HashSet::new();
                if cls.expr_linear(next, &mut on) {
                    rep.linear_reg_cones += 1;
                    rep.linear_reg_cone_bits += bits;
                }
            }
        }
    }
    rep
}

fn mask128(w: u32) -> u128 {
    if w >= 128 {
        u128::MAX
    } else {
        (1u128 << w) - 1
    }
}

fn lit128(words: &[u32]) -> u128 {
    let mut v = 0u128;
    for (i, w) in words.iter().enumerate().take(4) {
        v |= (*w as u128) << (32 * i);
    }
    v
}

/// Map of combinational signal definitions for evaluation/extraction.
pub fn comb_defs(program: &PackedProgram) -> HashMap<usize, &PackedExpr> {
    let mut m = HashMap::new();
    for stream in [&program.streams.async_reset_comb, &program.streams.comb] {
        for packet in stream {
            for op in &packet.ops {
                if let PackedOp::Assign { dst, expr } = op {
                    m.insert(*dst, expr);
                }
            }
        }
    }
    m
}

/// Reference evaluator: the integer value of `e` (masked to its width) given leaf
/// signal values; combinational signals are resolved through `comb_def`. Handles
/// the full op set so it doubles as a per-expression interpreter.
pub fn eval_expr(
    e: &PackedExpr,
    comb_def: &HashMap<usize, &PackedExpr>,
    leaves: &HashMap<usize, u128>,
) -> u128 {
    use PackedExprKind::*;
    let w = e.ty.width;
    let v = match &e.kind {
        Lit(words) => lit128(words),
        Signal(s) => {
            if let Some(v) = leaves.get(s) {
                *v
            } else if let Some(def) = comb_def.get(s) {
                eval_expr(def, comb_def, leaves)
            } else {
                0
            }
        }
        Not(a) => !eval_expr(a, comb_def, leaves),
        And(a, b) => eval_expr(a, comb_def, leaves) & eval_expr(b, comb_def, leaves),
        Or(a, b) => eval_expr(a, comb_def, leaves) | eval_expr(b, comb_def, leaves),
        Xor(a, b) => eval_expr(a, comb_def, leaves) ^ eval_expr(b, comb_def, leaves),
        Add(a, b) => eval_expr(a, comb_def, leaves).wrapping_add(eval_expr(b, comb_def, leaves)),
        Sub(a, b) => eval_expr(a, comb_def, leaves).wrapping_sub(eval_expr(b, comb_def, leaves)),
        Mul(a, b) => eval_expr(a, comb_def, leaves).wrapping_mul(eval_expr(b, comb_def, leaves)),
        Eq(a, b) => (eval_expr(a, comb_def, leaves) == eval_expr(b, comb_def, leaves)) as u128,
        Ne(a, b) => (eval_expr(a, comb_def, leaves) != eval_expr(b, comb_def, leaves)) as u128,
        Lt { lhs, rhs, signed } => {
            let (l, r) = (eval_expr(lhs, comb_def, leaves), eval_expr(rhs, comb_def, leaves));
            let res = if *signed {
                let lw = lhs.ty.width;
                let rw = rhs.ty.width;
                let sext = |x: u128, bw: u32| -> i128 {
                    let s = 128 - bw;
                    ((x << s) as i128) >> s
                };
                sext(l, lw) < sext(r, rw)
            } else {
                l < r
            };
            res as u128
        }
        Mux { cond, then_expr, else_expr } => {
            if eval_expr(cond, comb_def, leaves) & 1 != 0 {
                eval_expr(then_expr, comb_def, leaves)
            } else {
                eval_expr(else_expr, comb_def, leaves)
            }
        }
        Slice { expr, lsb } => eval_expr(expr, comb_def, leaves) >> lsb,
        Zext(a) | Trunc(a) | Cast(a) => eval_expr(a, comb_def, leaves),
        Sext(a) => {
            let aw = a.ty.width;
            let x = eval_expr(a, comb_def, leaves) & mask128(aw);
            let s = 128 - aw;
            (((x << s) as i128) >> s) as u128
        }
        Concat(parts) => {
            let mut acc = 0u128;
            let mut off = 0u32;
            for p in parts.iter().rev() {
                acc |= (eval_expr(p, comb_def, leaves) & mask128(p.ty.width)) << off;
                off += p.ty.width;
            }
            acc
        }
        MemRead { .. } => 0,
    };
    v & mask128(w)
}

/// The extracted GF(2) affine form of a linear cone: `out = constant ⊕ ⊕_b in[b]·columns[b]`.
#[derive(Clone, Debug)]
pub struct LinearForm {
    /// leaf signals (the variables), in input-bit order, with widths.
    pub leaves: Vec<(usize, u32)>,
    pub total_in_bits: usize,
    pub out_width: u32,
    /// `columns[b]` = the output-bit vector toggled by input bit `b` (the GF(2) matrix, column-major).
    pub columns: Vec<u128>,
    pub constant: u128,
}

fn collect_leaves(
    e: &PackedExpr,
    comb_def: &HashMap<usize, &PackedExpr>,
    seen: &mut HashSet<usize>,
    out: &mut Vec<usize>,
) {
    use PackedExprKind::*;
    match &e.kind {
        Lit(_) => {}
        Signal(s) => {
            if let Some(def) = comb_def.get(s) {
                if seen.insert(*s) {
                    collect_leaves(def, comb_def, seen, out);
                }
            } else if !out.contains(s) {
                out.push(*s);
            }
        }
        Not(a) | Slice { expr: a, .. } | Zext(a) | Sext(a) | Trunc(a) | Cast(a)
        | MemRead { addr: a, .. } => collect_leaves(a, comb_def, seen, out),
        And(a, b) | Or(a, b) | Xor(a, b) | Add(a, b) | Sub(a, b) | Mul(a, b) | Eq(a, b)
        | Ne(a, b) => {
            collect_leaves(a, comb_def, seen, out);
            collect_leaves(b, comb_def, seen, out);
        }
        Lt { lhs, rhs, .. } => {
            collect_leaves(lhs, comb_def, seen, out);
            collect_leaves(rhs, comb_def, seen, out);
        }
        Mux { cond, then_expr, else_expr } => {
            collect_leaves(cond, comb_def, seen, out);
            collect_leaves(then_expr, comb_def, seen, out);
            collect_leaves(else_expr, comb_def, seen, out);
        }
        Concat(parts) => parts.iter().for_each(|p| collect_leaves(p, comb_def, seen, out)),
    }
}

/// Extract the GF(2) affine transfer matrix of a (presumed-linear) cone by
/// probing the reference evaluator with the zero vector (constant) and each
/// basis input bit (its column). Output width must be ≤128.
pub fn extract_linear_form(
    program: &PackedProgram,
    next: &PackedExpr,
) -> LinearForm {
    let comb_def = comb_defs(program);
    let mut leaf_sigs = Vec::new();
    collect_leaves(next, &comb_def, &mut HashSet::new(), &mut leaf_sigs);
    let widths: Vec<u32> = leaf_sigs.iter().map(|s| program.signals[*s].layout.width).collect();
    let leaves: Vec<(usize, u32)> = leaf_sigs.iter().copied().zip(widths.iter().copied()).collect();
    let total_in_bits: usize = widths.iter().map(|w| *w as usize).sum();

    let zero: HashMap<usize, u128> = leaf_sigs.iter().map(|&s| (s, 0u128)).collect();
    let constant = eval_expr(next, &comb_def, &zero);

    let mut columns = Vec::with_capacity(total_in_bits);
    for (li, &sig) in leaf_sigs.iter().enumerate() {
        for bit in 0..widths[li] {
            let mut lv = zero.clone();
            lv.insert(sig, 1u128 << bit);
            columns.push(eval_expr(next, &comb_def, &lv) ^ constant);
        }
    }
    LinearForm {
        leaves,
        total_in_bits,
        out_width: next.ty.width,
        columns,
        constant,
    }
}

impl LinearForm {
    /// Evaluate via the matrix: out = constant ⊕ ⊕(set input bits' columns).
    pub fn eval(&self, leaf_vals: &HashMap<usize, u128>) -> u128 {
        let mut out = self.constant;
        let mut b = 0usize;
        for &(sig, w) in &self.leaves {
            let v = leaf_vals.get(&sig).copied().unwrap_or(0);
            for bit in 0..w {
                if v >> bit & 1 != 0 {
                    out ^= self.columns[b];
                }
                b += 1;
            }
        }
        out & mask128(self.out_width)
    }
}
