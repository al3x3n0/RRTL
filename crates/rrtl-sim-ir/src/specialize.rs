//! Design-specific AOT specialization of a [`PackedMachineProgram`].
//!
//! This is the front of RRTL's "design specializer" pipeline: a set of static,
//! value-oblivious rewrites over the packed SSA machine IR. Because every pass
//! here runs once at build time and produces a plain (still data-oblivious)
//! program, the result composes perfectly with the data-parallel lane engines —
//! there is no runtime control-flow divergence, unlike activity-based skipping.
//!
//! Pass 1 (this module): constant folding + algebraic identities + copy
//! propagation + dead-code elimination + value-id compaction. Each runs per
//! block (the four streams), since SSA value ids are block-local and ordered
//! defs-before-uses. Within-block common-subexpression elimination already
//! happens during lowering (`MachinePacketLowerer::memo`), so we focus on
//! folding constants and pruning the resulting dead work — both of which shrink
//! the interpreter's instruction count (the measured GPU bottleneck) and, via
//! compaction, the value workspace (the measured bandwidth bottleneck).

use std::collections::{HashMap, HashSet};

use rrtl_ir::{uint, BitType};

use crate::{
    PackedEffect, PackedExpr, PackedExprKind, PackedInstr, PackedInstrKind, PackedMachineProgram,
    PackedOp, PackedProgram, PackedValueId,
};

/// Counters describing what specialization changed, for measurement/reporting.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpecializeStats {
    pub instrs_before: usize,
    pub instrs_after: usize,
    /// Instructions fully evaluated to a literal (all operands constant).
    pub folded: usize,
    /// Instructions reduced to a copy of an operand by an algebraic identity.
    pub copies: usize,
    /// Instructions removed because their result became unused.
    pub dead: usize,
}

impl SpecializeStats {
    pub fn instrs_removed(&self) -> usize {
        self.instrs_before.saturating_sub(self.instrs_after)
    }
}

/// Run the specializer pipeline over a machine program, returning the rewritten
/// program and aggregate statistics. The result is observationally equivalent to
/// the input for every signal and memory.
pub fn specialize_program(prog: &PackedMachineProgram) -> (PackedMachineProgram, SpecializeStats) {
    let mut out = prog.clone();
    let mut stats = SpecializeStats::default();
    let blocks = [
        &mut out.streams.async_reset_comb,
        &mut out.streams.comb,
        &mut out.streams.tick_next,
        &mut out.streams.tick_commit,
    ];
    for block in blocks {
        specialize_block(block, &mut stats);
    }
    (out, stats)
}

/// Counters for register-freezing (dynamic value) specialization.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FreezeStats {
    /// Number of signals frozen to a constant.
    pub frozen_signals: usize,
    /// What the downstream static specializer then folded/pruned.
    pub specialize: SpecializeStats,
}

/// Specialize `prog` under the assumption that the given signals hold the given
/// constant values — e.g. configuration / mode registers that a profiler has
/// observed to be stable over a long run. Every read of a frozen signal is
/// rewritten to its literal value, then the ordinary static specializer
/// ([`specialize_program`]) folds the now-constant control logic and DCEs the
/// datapaths that control used to gate.
///
/// This is the bridge from *dynamic* profiling to *static* specialization: the
/// dynamism is only in **what** we treat as constant (discovered at runtime,
/// beyond what static analysis can prove); the emitted program is still fully
/// data-oblivious, so it composes with the lane engines exactly like any other
/// specializer output — no control-flow divergence. The result is observationally
/// equivalent to `prog` **only while the frozen signals actually hold those
/// values**; a runtime guard must check that each tick and deoptimize otherwise.
pub fn freeze_signals_program(
    prog: &PackedMachineProgram,
    frozen: &HashMap<usize, u128>,
) -> (PackedMachineProgram, FreezeStats) {
    let mut out = prog.clone();
    for block in [
        &mut out.streams.async_reset_comb,
        &mut out.streams.comb,
        &mut out.streams.tick_next,
        &mut out.streams.tick_commit,
    ] {
        for packet in &mut block.packets {
            for instr in &mut packet.instrs {
                if let PackedInstrKind::Signal(s) = &instr.kind {
                    if let Some(&v) = frozen.get(s) {
                        let w = instr.ty.width;
                        instr.kind = PackedInstrKind::Lit(from_u128(v & mask_u128(w), w));
                    }
                }
            }
        }
    }
    let (folded, sstats) = specialize_program(&out);
    (
        folded,
        FreezeStats {
            frozen_signals: frozen.len(),
            specialize: sstats,
        },
    )
}

// =============================================================================
// Pass: priority mux-chain rebalancing (depth N -> log N for superscalar IPC).
// =============================================================================
//
// A Verilog priority `if/else` (or `case (1'b1)`) lowers to a right-nested mux
// chain `c0 ? t0 : (c1 ? t1 : (... : base))`. Each mux's `else` is the next mux,
// so the chain is a *serial* dependency of length N even though every condition
// and value feeds in independently — a latency bottleneck a wide out-of-order CPU
// cannot hide. (Measured on picorv32: a 48-deep comb mux spine, 96% of the comb
// critical path, all conditions ready immediately as 1-bit signal reads.)
//
// Priority-select is ASSOCIATIVE — `combine((h1,v1),(h2,v2)) = (h1|h2, h1?v1:v2)`
// with the left operand higher priority — so the chain can be rebuilt as a
// BALANCED tree of `combine` nodes: same mux count, plus one OR per node for the
// "hit" bit, depth N -> ceil(log2 N). The result is bit-identical (exact priority
// preserved; no one-hot assumption needed), so it is always safe. The C compiler
// cannot do this itself: the C ternary chain encodes the serial dependency and the
// compiler has no licence to reorder a priority select.
//
// This trades a few extra OR ops for much shorter dependency chains, so it helps
// only when the eval is latency-bound (it can mildly hurt a purely op-count-bound
// design); it is therefore a standalone, opt-in pass for the single-instance
// latency backends (JIT/AOT), not part of the default pipeline.

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RebalanceStats {
    /// Mux chains rebalanced (length >= the threshold).
    pub chains: usize,
    /// Total chain muxes that were in those chains (depth removed).
    pub chain_muxes: usize,
    /// Deepest chain rebalanced (its original length).
    pub deepest: usize,
    /// Extra OR (hit-bit) instructions added across all rebalanced chains.
    pub ors_added: usize,
}

/// Minimum chain length to bother rebalancing (short chains aren't worth the ORs).
const REBALANCE_MIN_CHAIN: usize = 8;

/// Rebalance priority mux chains in every block of `prog`. Bit-identical to the
/// input. Runs the standard specializer afterwards to DCE the now-dead chain
/// muxes and recompact value ids.
pub fn rebalance_mux_chains_program(
    prog: &PackedMachineProgram,
) -> (PackedMachineProgram, RebalanceStats) {
    let mut out = prog.clone();
    let mut stats = RebalanceStats::default();
    for block in [
        &mut out.streams.async_reset_comb,
        &mut out.streams.comb,
        &mut out.streams.tick_next,
        &mut out.streams.tick_commit,
    ] {
        rebalance_block(block, &mut stats);
    }
    // The rebalanced head muxes alias the original ids, so the dead link muxes and
    // the recompaction are handled by the existing specializer.
    let (cleaned, _) = specialize_program(&out);
    (cleaned, stats)
}

fn rebalance_block(block: &mut crate::PackedBlock, stats: &mut RebalanceStats) {
    // Flatten to a single instruction list (defs-before-uses preserved) plus the
    // effects in order; none of the machine-program engines use packet boundaries
    // for parallelism, so re-packetizing is free.
    let mut instrs: Vec<PackedInstr> = Vec::new();
    let mut effects: Vec<PackedEffect> = Vec::new();
    for packet in &block.packets {
        instrs.extend(packet.instrs.iter().cloned());
        effects.extend(packet.effects.iter().cloned());
    }

    // Use counts and "is this value used exactly once, as some mux's else?" — a
    // chain LINK. The chain head is a mux whose else is a link, but which is not
    // itself a link.
    let mut use_count: HashMap<usize, usize> = HashMap::new();
    let mut else_uses: HashMap<usize, usize> = HashMap::new(); // value -> #times used as a mux else
    let mut kind_of: HashMap<usize, PackedInstrKind> = HashMap::new();
    let mut ty_of: HashMap<usize, BitType> = HashMap::new();
    let mut max_id = 0usize;
    for instr in &instrs {
        for op in instr_operands(&instr.kind) {
            *use_count.entry(op.0).or_default() += 1;
        }
        if let PackedInstrKind::Mux { else_value, .. } = &instr.kind {
            *else_uses.entry(else_value.0).or_default() += 1;
        }
        kind_of.insert(instr.dst.0, instr.kind.clone());
        ty_of.insert(instr.dst.0, instr.ty);
        max_id = max_id.max(instr.dst.0);
    }
    for effect in &effects {
        for op in effect_operands(effect) {
            *use_count.entry(op.0).or_default() += 1;
        }
    }
    let is_link = |v: usize| {
        matches!(kind_of.get(&v), Some(PackedInstrKind::Mux { .. }))
            && use_count.get(&v).copied().unwrap_or(0) == 1
            && else_uses.get(&v).copied().unwrap_or(0) == 1
    };

    // Discover maximal chains: heads = muxes whose else is a link and which are not
    // themselves a link. Walk the else spine collecting links.
    let mut next_id = max_id + 1;
    let mut new_instrs_before: HashMap<usize, Vec<PackedInstr>> = HashMap::new(); // head id -> tree instrs
    let mut head_replacement: HashMap<usize, PackedInstrKind> = HashMap::new();
    let mut dead: HashSet<usize> = HashSet::new();

    for instr in &instrs {
        let head = instr.dst.0;
        let PackedInstrKind::Mux { else_value, .. } = &instr.kind else { continue };
        if is_link(head) {
            continue; // not a head; handled by its parent
        }
        if !is_link(else_value.0) {
            continue; // chain of length 1
        }
        // Collect the chain: (cond, then) pairs down the else spine.
        let mut conds: Vec<usize> = Vec::new();
        let mut thens: Vec<usize> = Vec::new();
        let mut cur = head;
        let base;
        loop {
            let Some(PackedInstrKind::Mux { cond, then_value, else_value }) = kind_of.get(&cur) else {
                base = cur;
                break;
            };
            conds.push(cond.0);
            thens.push(then_value.0);
            if is_link(else_value.0) {
                cur = else_value.0;
            } else {
                base = else_value.0;
                break;
            }
        }
        let k = conds.len();
        if k < REBALANCE_MIN_CHAIN {
            continue;
        }
        let wty = instr.ty; // chain value width
        let bty = uint(1); // hit bits

        // Build a balanced combine tree. leaves: (hit_id, val_id) = (cond_i, then_i).
        let mut tree: Vec<PackedInstr> = Vec::new();
        let mut alloc = || {
            let id = next_id;
            next_id += 1;
            id
        };
        let mut level: Vec<(usize, usize)> = conds.iter().zip(&thens).map(|(c, t)| (*c, *t)).collect();
        let mut ors = 0usize;
        while level.len() > 1 {
            let mut nxt: Vec<(usize, usize)> = Vec::new();
            let mut i = 0;
            while i < level.len() {
                if i + 1 == level.len() {
                    nxt.push(level[i]); // odd one carries up
                    i += 1;
                    continue;
                }
                let (hl, vl) = level[i];
                let (hr, vr) = level[i + 1];
                let hid = alloc();
                tree.push(PackedInstr {
                    dst: PackedValueId(hid),
                    ty: bty,
                    kind: PackedInstrKind::Or(PackedValueId(hl), PackedValueId(hr)),
                });
                ors += 1;
                let vid = alloc();
                tree.push(PackedInstr {
                    dst: PackedValueId(vid),
                    ty: wty,
                    kind: PackedInstrKind::Mux {
                        cond: PackedValueId(hl),
                        then_value: PackedValueId(vl),
                        else_value: PackedValueId(vr),
                    },
                });
                nxt.push((hid, vid));
                i += 2;
            }
            level = nxt;
        }
        let (root_hit, root_val) = level[0];
        // head becomes: any-hit ? tree-value : base
        head_replacement.insert(
            head,
            PackedInstrKind::Mux {
                cond: PackedValueId(root_hit),
                then_value: PackedValueId(root_val),
                else_value: PackedValueId(base),
            },
        );
        new_instrs_before.insert(head, tree);
        // the interior link muxes are now dead
        let mut c = else_value.0;
        while is_link(c) {
            dead.insert(c);
            if let Some(PackedInstrKind::Mux { else_value, .. }) = kind_of.get(&c) {
                c = else_value.0;
            } else {
                break;
            }
        }
        stats.chains += 1;
        stats.chain_muxes += k;
        stats.deepest = stats.deepest.max(k);
        stats.ors_added += ors;
    }

    if new_instrs_before.is_empty() {
        return; // nothing changed; leave the block byte-identical
    }

    // Rebuild IN PLACE per packet: drop dead link muxes and, immediately before each
    // head, splice in its balanced tree (then the head with its replacement kind).
    // Effects stay in their original packets — comb signals stored by an earlier
    // packet are read by later instrs via Signal(idx), so effect order must be
    // preserved. The tree's operands (the chain's conditions/values) are defined
    // before the head, and engines execute a packet's instrs sequentially, so the
    // spliced-in tree sees its operands and feeds the head.
    for packet in &mut block.packets {
        let mut rebuilt: Vec<PackedInstr> = Vec::with_capacity(packet.instrs.len());
        for instr in packet.instrs.drain(..) {
            if dead.contains(&instr.dst.0) {
                continue;
            }
            if let Some(tree) = new_instrs_before.remove(&instr.dst.0) {
                rebuilt.extend(tree);
                let mut head = instr;
                head.kind = head_replacement.remove(&head.dst.0).unwrap();
                rebuilt.push(head);
            } else {
                rebuilt.push(instr);
            }
        }
        packet.instrs = rebuilt;
    }
}

// =============================================================================
// Pass 2: constant-multiply strength reduction (expr-tree level).
// =============================================================================
//
// Rewrites `e * C` (C a constant, e not) into a sum/difference of shifted copies
// of `e`, `Σ ± (e << k_i)`, using a non-adjacent form (NAF / canonical
// signed-digit) decomposition of C to minimise the number of terms. A left shift
// `e << k` is expressed without any new opcode as `Trunc_w(Concat([e, 0_k]))`
// (Concat is MSB-first, so the zero literal lands in the low k bits).
//
// This runs at the PackedExpr level so it benefits every backend (CPU SIMD and
// the GPU interpreter), and it is gated to the regime where it actually pays:
// multi-limb multiplies (where the half-product multiply is expensive) by
// constants with few signed digits. On 1-limb (≤32-bit) designs the native
// multiply is a single cheap op and the kernel is bandwidth-bound, so we leave
// those alone — adding shift/concat/add ops there would only add traffic.

/// Max NAF terms to still prefer strength reduction over a multi-limb multiply.
const STRENGTH_MAX_TERMS: usize = 4;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StrengthStats {
    /// Constant multiplies replaced by shift/add chains.
    pub reduced: usize,
    /// `e * 0`-equivalent multiplies folded to zero.
    pub zeroed: usize,
}

/// Run constant-multiply strength reduction over every expression in a program.
/// The result is observationally equivalent to the input.
pub fn strength_reduce_program(prog: &PackedProgram) -> (PackedProgram, StrengthStats) {
    let mut out = prog.clone();
    let mut stats = StrengthStats::default();
    for stream in [
        &mut out.streams.async_reset_comb,
        &mut out.streams.comb,
        &mut out.streams.tick_next,
        &mut out.streams.tick_commit,
    ] {
        for packet in stream.iter_mut() {
            for op in &mut packet.ops {
                match op {
                    PackedOp::Assign { expr, .. } => reduce_expr(expr, &mut stats),
                    PackedOp::CaptureReg { next, .. } => reduce_expr(next, &mut stats),
                    PackedOp::MemoryWrite {
                        enable, addr, data, ..
                    } => {
                        reduce_expr(enable, &mut stats);
                        reduce_expr(addr, &mut stats);
                        reduce_expr(data, &mut stats);
                    }
                }
            }
        }
    }
    (out, stats)
}

/// Bottom-up rewrite of one expression in place.
fn reduce_expr(expr: &mut PackedExpr, stats: &mut StrengthStats) {
    for child in expr_children_mut(&mut expr.kind) {
        reduce_expr(child, stats);
    }
    if let PackedExprKind::Mul(a, b) = &expr.kind {
        let w = expr.ty.width;
        // Pick the constant operand (if exactly one) and the variable operand.
        let (var, konst) = match (lit_value(a), lit_value(b)) {
            (None, Some(c)) => (a.as_ref(), c),
            (Some(c), None) => (b.as_ref(), c),
            _ => return, // both const (let const-fold handle) or neither
        };
        if let Some(c) = to_u128(&konst) {
            let c = c & mask_u128(w);
            let terms = naf(c, w);
            if c == 0 {
                *expr = lit_expr(0, w);
                stats.zeroed += 1;
            } else if w > 32 && !terms.is_empty() && terms.len() <= STRENGTH_MAX_TERMS {
                *expr = build_shift_add(var, &terms, w);
                stats.reduced += 1;
            }
        }
    }
}

/// Non-adjacent form of `c` within `width` bits: a minimal list of `(shift, sign)`
/// with `c ≡ Σ sign·2^shift (mod 2^width)`. Terms at or above `width` are dropped
/// (they vanish mod 2^width).
fn naf(mut c: u128, width: u32) -> Vec<(u32, i32)> {
    let mut out = Vec::new();
    let mut k = 0u32;
    while c != 0 && k <= width + 1 {
        if c & 1 == 1 {
            let d: i32 = if c & 3 == 3 { -1 } else { 1 };
            if d == -1 {
                c += 1;
            } else {
                c -= 1;
            }
            out.push((k, d));
        }
        c >>= 1;
        k += 1;
    }
    out.retain(|&(k, _)| k < width);
    out
}

/// Build `Σ sign·(var << shift)` as an expression of width `w`.
fn build_shift_add(var: &PackedExpr, terms: &[(u32, i32)], w: u32) -> PackedExpr {
    let mut acc: Option<PackedExpr> = None;
    for &(k, sign) in terms {
        let shifted = shift_left(var.clone(), k, w);
        acc = Some(match acc {
            None => {
                if sign > 0 {
                    shifted
                } else {
                    sub_expr(lit_expr(0, w), shifted, w)
                }
            }
            Some(a) => {
                if sign > 0 {
                    add_expr(a, shifted, w)
                } else {
                    sub_expr(a, shifted, w)
                }
            }
        });
    }
    acc.unwrap_or_else(|| lit_expr(0, w))
}

/// `var << k`, truncated to width `w`. For `k == 0` this is just `var`.
fn shift_left(var: PackedExpr, k: u32, w: u32) -> PackedExpr {
    if k == 0 {
        return var;
    }
    // Concat is MSB-first: {var, 0_k} places the zero literal in the low k bits.
    let concat = PackedExpr {
        ty: uint(w + k),
        kind: PackedExprKind::Concat(vec![var, lit_expr(0, k)]),
    };
    PackedExpr {
        ty: uint(w),
        kind: PackedExprKind::Trunc(Box::new(concat)),
    }
}

fn add_expr(a: PackedExpr, b: PackedExpr, w: u32) -> PackedExpr {
    PackedExpr {
        ty: uint(w),
        kind: PackedExprKind::Add(Box::new(a), Box::new(b)),
    }
}

fn sub_expr(a: PackedExpr, b: PackedExpr, w: u32) -> PackedExpr {
    PackedExpr {
        ty: uint(w),
        kind: PackedExprKind::Sub(Box::new(a), Box::new(b)),
    }
}

fn lit_expr(v: u128, w: u32) -> PackedExpr {
    PackedExpr {
        ty: uint(w),
        kind: PackedExprKind::Lit(from_u128(v, w)),
    }
}

fn lit_value(e: &PackedExpr) -> Option<Vec<u32>> {
    match &e.kind {
        PackedExprKind::Lit(v) => Some(v.clone()),
        _ => None,
    }
}

fn expr_children_mut(kind: &mut PackedExprKind) -> Vec<&mut PackedExpr> {
    use PackedExprKind::*;
    match kind {
        Lit(_) | Signal(_) => vec![],
        Not(a) | Zext(a) | Sext(a) | Trunc(a) | Cast(a) | Slice { expr: a, .. }
        | MemRead { addr: a, .. } => vec![a.as_mut()],
        And(a, b) | Or(a, b) | Xor(a, b) | Add(a, b) | Sub(a, b) | Mul(a, b) | Eq(a, b)
        | Ne(a, b) => vec![a.as_mut(), b.as_mut()],
        Lt { lhs, rhs, .. } => vec![lhs.as_mut(), rhs.as_mut()],
        Mux {
            cond,
            then_expr,
            else_expr,
        } => vec![cond.as_mut(), then_expr.as_mut(), else_expr.as_mut()],
        Concat(parts) => parts.iter_mut().collect(),
    }
}

// =============================================================================
// Pass 4: liveness-based value-slot allocation.
// =============================================================================
//
// Renumbers each block's SSA value ids with slot REUSE: two values whose live
// ranges do not overlap share the same slot id, shrinking the value workspace
// from O(number of values) to O(max simultaneously-live values). Because the GPU
// interpreter indexes the value buffer through a per-id offset table (`voff`),
// this needs no kernel change — fewer distinct ids simply means a smaller voff
// table and a smaller, more cache-resident value buffer (the measured wall at
// high lane counts).
//
// Safety: a slot is freed only one packet AFTER a value's last use (we expire
// values whose last_use < current_packet). Within a packet the kernel runs all
// instructions and then all effects, so holding a slot through its last-use
// packet prevents a reused slot from being overwritten before a same-packet
// effect reads it.

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SlotStats {
    /// Distinct value slots before allocation (max over blocks).
    pub slots_before: usize,
    /// Distinct value slots after allocation (max over blocks).
    pub slots_after: usize,
}

/// Run liveness-based slot allocation over every block of a machine program.
/// Observationally equivalent to the input; only value-id numbering changes.
pub fn slot_allocate_program(prog: &PackedMachineProgram) -> (PackedMachineProgram, SlotStats) {
    let mut out = prog.clone();
    let mut stats = SlotStats::default();
    for block in [
        &mut out.streams.async_reset_comb,
        &mut out.streams.comb,
        &mut out.streams.tick_next,
        &mut out.streams.tick_commit,
    ] {
        let before = block_value_count(block);
        let after = allocate_block_slots(block);
        stats.slots_before = stats.slots_before.max(before);
        stats.slots_after = stats.slots_after.max(after);
    }
    (out, stats)
}

fn block_value_count(block: &crate::PackedBlock) -> usize {
    block
        .packets
        .iter()
        .flat_map(|p| p.instrs.iter())
        .map(|i| i.dst.0 + 1)
        .max()
        .unwrap_or(0)
}

/// Allocate (and rewrite) slot ids for one block; returns the slot count.
fn allocate_block_slots(block: &mut crate::PackedBlock) -> usize {
    let n = block_value_count(block);
    if n == 0 {
        return 0;
    }

    // Last packet at which each value is read (as an instr or effect operand).
    let mut last_use = vec![0usize; n];
    let mut def_pkt = vec![usize::MAX; n];
    for (p, packet) in block.packets.iter().enumerate() {
        for instr in &packet.instrs {
            def_pkt[instr.dst.0] = p;
        }
        for instr in &packet.instrs {
            for id in instr_operands(&instr.kind) {
                last_use[id.0] = last_use[id.0].max(p);
            }
        }
        for effect in &packet.effects {
            for id in effect_operands(effect) {
                last_use[id.0] = last_use[id.0].max(p);
            }
        }
    }
    for v in 0..n {
        if def_pkt[v] != usize::MAX {
            last_use[v] = last_use[v].max(def_pkt[v]);
        }
    }

    // Linear scan: expire dead values, then assign a (reused or fresh) slot to
    // each value defined in this packet.
    let mut remap = vec![usize::MAX; n];
    let mut free: Vec<usize> = Vec::new();
    let mut active: Vec<(usize, usize)> = Vec::new(); // (last_use_packet, slot)
    let mut next_slot = 0usize;
    for (p, packet) in block.packets.iter().enumerate() {
        let mut i = 0;
        while i < active.len() {
            if active[i].0 < p {
                free.push(active[i].1);
                active.swap_remove(i);
            } else {
                i += 1;
            }
        }
        for instr in &packet.instrs {
            let slot = free.pop().unwrap_or_else(|| {
                let s = next_slot;
                next_slot += 1;
                s
            });
            remap[instr.dst.0] = slot;
            active.push((last_use[instr.dst.0], slot));
        }
    }

    for packet in &mut block.packets {
        for instr in &mut packet.instrs {
            instr.dst = PackedValueId(remap[instr.dst.0]);
            for id in instr_operands_mut(&mut instr.kind) {
                *id = PackedValueId(remap[id.0]);
            }
        }
        for effect in &mut packet.effects {
            for id in effect_operands_mut(effect) {
                *id = PackedValueId(remap[id.0]);
            }
        }
    }
    next_slot
}

/// What a value id has been proven equivalent to.
enum Repl {
    /// A compile-time constant (width-masked limbs).
    Const(Vec<u32>),
    /// Equivalent to another (earlier, same-width) value.
    Copy(PackedValueId),
}

fn specialize_block(block: &mut crate::PackedBlock, stats: &mut SpecializeStats) {
    stats.instrs_before += block.packets.iter().map(|p| p.instrs.len()).sum::<usize>();

    // --- Forward pass: constant propagation + algebraic identities + copy-prop.
    let mut repl: HashMap<usize, Repl> = HashMap::new();
    let mut ty_of: HashMap<usize, BitType> = HashMap::new();
    for packet in &mut block.packets {
        for instr in &mut packet.instrs {
            ty_of.insert(instr.dst.0, instr.ty);
            // Canonicalize operands through known copies (constants keep pointing
            // at their defining Lit instr, which survives DCE iff still used).
            for id in instr_operands_mut(&mut instr.kind) {
                *id = canon(&repl, *id);
            }
            if let PackedInstrKind::Lit(v) = &instr.kind {
                // Record literals so their consumers can fold, but don't count
                // them as rewrites.
                repl.insert(instr.dst.0, Repl::Const(v.clone()));
                continue;
            }
            if let Some(r) = try_fold(&instr.kind, instr.ty, &repl, &ty_of) {
                match &r {
                    Repl::Const(v) => {
                        instr.kind = PackedInstrKind::Lit(v.clone());
                        stats.folded += 1;
                    }
                    Repl::Copy(_) => {
                        stats.copies += 1;
                    }
                }
                repl.insert(instr.dst.0, r);
            }
        }
        for effect in &mut packet.effects {
            for id in effect_operands_mut(effect) {
                *id = canon(&repl, *id);
            }
        }
    }

    // --- Dead-code elimination (backward liveness over topologically ordered
    // packets: operands are always defined in earlier packets).
    let mut live: HashSet<usize> = HashSet::new();
    for packet in &block.packets {
        for effect in &packet.effects {
            for id in effect_operands(effect) {
                live.insert(id.0);
            }
        }
    }
    for packet in block.packets.iter().rev() {
        for instr in packet.instrs.iter().rev() {
            if live.contains(&instr.dst.0) {
                for id in instr_operands(&instr.kind) {
                    live.insert(id.0);
                }
            }
        }
    }
    let before_dce: usize = block.packets.iter().map(|p| p.instrs.len()).sum();
    for packet in &mut block.packets {
        packet.instrs.retain(|instr| live.contains(&instr.dst.0));
    }
    let after_dce: usize = block.packets.iter().map(|p| p.instrs.len()).sum();
    stats.dead += before_dce - after_dce;

    // --- Compact value ids to 0..N in definition order, shrinking the value
    // workspace (a footprint win and a preview of liveness slot allocation).
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut next = 0usize;
    for packet in &block.packets {
        for instr in &packet.instrs {
            remap.insert(instr.dst.0, next);
            next += 1;
        }
    }
    for packet in &mut block.packets {
        for instr in &mut packet.instrs {
            instr.dst = PackedValueId(remap[&instr.dst.0]);
            for id in instr_operands_mut(&mut instr.kind) {
                *id = PackedValueId(remap[&id.0]);
            }
        }
        for effect in &mut packet.effects {
            for id in effect_operands_mut(effect) {
                *id = PackedValueId(remap[&id.0]);
            }
        }
    }

    stats.instrs_after += after_dce;
}

/// Follow copy chains to a canonical value id.
fn canon(repl: &HashMap<usize, Repl>, mut id: PackedValueId) -> PackedValueId {
    while let Some(Repl::Copy(next)) = repl.get(&id.0) {
        id = *next;
    }
    id
}

/// Constant limbs for a value, if it resolves to one.
fn const_of<'a>(repl: &'a HashMap<usize, Repl>, id: PackedValueId) -> Option<&'a [u32]> {
    match repl.get(&canon(repl, id).0) {
        Some(Repl::Const(v)) => Some(v),
        _ => None,
    }
}

// ---- constant arithmetic on <=128-bit values -------------------------------

fn to_u128(limbs: &[u32]) -> Option<u128> {
    if limbs.len() > 4 && limbs[4..].iter().any(|&x| x != 0) {
        return None; // wider than 128 bits and not trivially representable
    }
    let mut v = 0u128;
    for (i, &l) in limbs.iter().take(4).enumerate() {
        v |= (l as u128) << (32 * i);
    }
    Some(v)
}

fn mask_u128(width: u32) -> u128 {
    if width == 0 {
        0
    } else if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

fn from_u128(v: u128, width: u32) -> Vec<u32> {
    let v = v & mask_u128(width);
    let limbs = (((width + 31) / 32).max(1)) as usize;
    (0..limbs).map(|i| (v >> (32 * i)) as u32).collect()
}

/// Sign-extend the low `width` bits of `v` to a full i128.
fn sext_i128(v: u128, width: u32) -> i128 {
    if width == 0 || width >= 128 {
        return v as i128;
    }
    if (v >> (width - 1)) & 1 == 1 {
        (v | !mask_u128(width)) as i128
    } else {
        v as i128
    }
}

/// Attempt to fold or simplify one instruction. Returns the replacement for its
/// destination, or `None` to leave it unchanged.
fn try_fold(
    kind: &PackedInstrKind,
    ty: BitType,
    repl: &HashMap<usize, Repl>,
    ty_of: &HashMap<usize, BitType>,
) -> Option<Repl> {
    use PackedInstrKind::*;
    let w = ty.width;
    let cst = |id: PackedValueId| const_of(repl, id).and_then(to_u128);
    let width_of = |id: PackedValueId| ty_of.get(&id.0).map(|t| t.width).unwrap_or(w);
    let all_ones = mask_u128(w);
    let mk = |v: u128| Some(Repl::Const(from_u128(v, w)));

    match kind {
        Lit(_) | Signal(_) | MemRead { .. } | Concat(_) => None,
        Not(a) => cst(*a).and_then(|x| mk(!x)),
        And(a, b) => {
            if let (Some(x), Some(y)) = (cst(*a), cst(*b)) {
                return mk(x & y);
            }
            if cst(*a) == Some(all_ones) {
                return Some(Repl::Copy(*b));
            }
            if cst(*b) == Some(all_ones) {
                return Some(Repl::Copy(*a));
            }
            if cst(*a) == Some(0) || cst(*b) == Some(0) {
                return mk(0);
            }
            None
        }
        Or(a, b) => {
            if let (Some(x), Some(y)) = (cst(*a), cst(*b)) {
                return mk(x | y);
            }
            if cst(*a) == Some(0) {
                return Some(Repl::Copy(*b));
            }
            if cst(*b) == Some(0) {
                return Some(Repl::Copy(*a));
            }
            if cst(*a) == Some(all_ones) || cst(*b) == Some(all_ones) {
                return mk(all_ones);
            }
            None
        }
        Xor(a, b) => {
            if let (Some(x), Some(y)) = (cst(*a), cst(*b)) {
                return mk(x ^ y);
            }
            if cst(*a) == Some(0) {
                return Some(Repl::Copy(*b));
            }
            if cst(*b) == Some(0) {
                return Some(Repl::Copy(*a));
            }
            None
        }
        Add(a, b) => {
            if let (Some(x), Some(y)) = (cst(*a), cst(*b)) {
                return mk(x.wrapping_add(y));
            }
            if cst(*a) == Some(0) {
                return Some(Repl::Copy(*b));
            }
            if cst(*b) == Some(0) {
                return Some(Repl::Copy(*a));
            }
            None
        }
        Sub(a, b) => {
            if let (Some(x), Some(y)) = (cst(*a), cst(*b)) {
                return mk(x.wrapping_sub(y));
            }
            if cst(*b) == Some(0) {
                return Some(Repl::Copy(*a));
            }
            if canon(repl, *a) == canon(repl, *b) {
                return mk(0);
            }
            None
        }
        Mul(a, b) => {
            if let (Some(x), Some(y)) = (cst(*a), cst(*b)) {
                return mk(x.wrapping_mul(y));
            }
            if cst(*a) == Some(0) || cst(*b) == Some(0) {
                return mk(0);
            }
            if cst(*a) == Some(1) {
                return Some(Repl::Copy(*b));
            }
            if cst(*b) == Some(1) {
                return Some(Repl::Copy(*a));
            }
            None
        }
        Eq(a, b) => {
            if let (Some(x), Some(y)) = (cst(*a), cst(*b)) {
                return mk((x == y) as u128);
            }
            if canon(repl, *a) == canon(repl, *b) {
                return mk(1);
            }
            None
        }
        Ne(a, b) => {
            if let (Some(x), Some(y)) = (cst(*a), cst(*b)) {
                return mk((x != y) as u128);
            }
            if canon(repl, *a) == canon(repl, *b) {
                return mk(0);
            }
            None
        }
        Lt { lhs, rhs, signed } => {
            if let (Some(x), Some(y)) = (cst(*lhs), cst(*rhs)) {
                let res = if *signed {
                    sext_i128(x, width_of(*lhs)) < sext_i128(y, width_of(*rhs))
                } else {
                    x < y
                };
                return mk(res as u128);
            }
            if canon(repl, *lhs) == canon(repl, *rhs) {
                return mk(0);
            }
            None
        }
        Mux {
            cond,
            then_value,
            else_value,
        } => {
            if let Some(c) = cst(*cond) {
                return Some(Repl::Copy(if c & 1 != 0 { *then_value } else { *else_value }));
            }
            if canon(repl, *then_value) == canon(repl, *else_value) {
                return Some(Repl::Copy(*then_value));
            }
            None
        }
        Slice { value, lsb } => cst(*value).and_then(|x| mk(x >> *lsb)),
        Zext(a) | Trunc(a) | Cast(a) => cst(*a).and_then(mk),
        Sext(a) => cst(*a).and_then(|x| mk(sext_i128(x, width_of(*a)) as u128)),
    }
}

// ---- operand accessors ------------------------------------------------------

fn instr_operands_mut(kind: &mut PackedInstrKind) -> Vec<&mut PackedValueId> {
    use PackedInstrKind::*;
    match kind {
        Lit(_) | Signal(_) => vec![],
        Not(a) | Zext(a) | Sext(a) | Trunc(a) | Cast(a) | Slice { value: a, .. }
        | MemRead { addr: a, .. } => vec![a],
        And(a, b) | Or(a, b) | Xor(a, b) | Add(a, b) | Sub(a, b) | Mul(a, b) | Eq(a, b)
        | Ne(a, b) => vec![a, b],
        Lt { lhs, rhs, .. } => vec![lhs, rhs],
        Mux {
            cond,
            then_value,
            else_value,
        } => vec![cond, then_value, else_value],
        Concat(v) => v.iter_mut().collect(),
    }
}

fn instr_operands(kind: &PackedInstrKind) -> Vec<PackedValueId> {
    use PackedInstrKind::*;
    match kind {
        Lit(_) | Signal(_) => vec![],
        Not(a) | Zext(a) | Sext(a) | Trunc(a) | Cast(a) | Slice { value: a, .. }
        | MemRead { addr: a, .. } => vec![*a],
        And(a, b) | Or(a, b) | Xor(a, b) | Add(a, b) | Sub(a, b) | Mul(a, b) | Eq(a, b)
        | Ne(a, b) => vec![*a, *b],
        Lt { lhs, rhs, .. } => vec![*lhs, *rhs],
        Mux {
            cond,
            then_value,
            else_value,
        } => vec![*cond, *then_value, *else_value],
        Concat(v) => v.clone(),
    }
}

fn effect_operands_mut(effect: &mut PackedEffect) -> Vec<&mut PackedValueId> {
    match effect {
        PackedEffect::StoreSignal { value, .. } => vec![value],
        PackedEffect::CaptureReg { value, .. } => vec![value],
        PackedEffect::MemoryWrite {
            enable, addr, data, ..
        } => vec![enable, addr, data],
    }
}

fn effect_operands(effect: &PackedEffect) -> Vec<PackedValueId> {
    match effect {
        PackedEffect::StoreSignal { value, .. } => vec![*value],
        PackedEffect::CaptureReg { value, .. } => vec![*value],
        PackedEffect::MemoryWrite {
            enable, addr, data, ..
        } => vec![*enable, *addr, *data],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lower_to_machine_program, lower_to_packed_program};
    use rrtl_core::{compile, lit_u, uint, Design};

    /// A design rich in foldable constants and identities.
    fn foldable_machine() -> crate::PackedMachineProgram {
        let mut design = Design::new();
        {
            let mut m = design.module("Spec");
            let clk = m.input("clk", uint(1));
            let din = m.input("din", uint(32));
            // 7 + 9 -> 16 (full const fold)
            let k = m.wire("k", uint(32));
            m.assign(k, lit_u(7, 32) + lit_u(9, 32));
            // din & 0xffff_ffff -> din (identity copy)
            let masked = m.wire("masked", uint(32));
            m.assign(masked, din & lit_u(0xffff_ffff, 32));
            // din * 0 -> 0 (annihilator)
            let zilch = m.wire("zilch", uint(32));
            m.assign(zilch, din * lit_u(0, 32));
            let acc = m.reg("acc", uint(32));
            m.clock(acc, clk);
            m.next(acc, acc + k + masked + zilch);
            let o = m.output("o", uint(32));
            m.assign(o, acc);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Spec").unwrap();
        lower_to_machine_program(&program)
    }

    fn block_value_ids_contiguous(block: &crate::PackedBlock) -> bool {
        let mut next = 0usize;
        for packet in &block.packets {
            for instr in &packet.instrs {
                if instr.dst.0 != next {
                    return false;
                }
                next += 1;
            }
        }
        true
    }

    /// A mode register selecting among four multiply-heavy datapaths into `acc`.
    /// Freezing `mode` should fold the select and DCE the three unused arms.
    fn mode_gated_design() -> Design {
        use rrtl_ir::mux;
        let mut design = Design::new();
        {
            let mut m = design.module("Cfg");
            let clk = m.input("clk", uint(1));
            let cfg_we = m.input("cfg_we", uint(1));
            let cfg_in = m.input("cfg_in", uint(2));
            let a_i = m.input("a", uint(32));
            let b_i = m.input("b", uint(32));
            let mode = m.reg("mode", uint(2));
            let acc = m.reg("acc", uint(32));
            m.clock(mode, clk);
            m.clock(acc, clk);
            // mode latches cfg_in when cfg_we, else holds.
            m.next(mode, mux(cfg_we.value(), cfg_in.value(), mode.value()));
            let (a, b) = (a_i.value(), b_i.value());
            let arm0 = a.clone() * b.clone() + b.clone() * a.clone();
            let arm1 = a.clone() * a.clone() + b.clone() * b.clone() + a.clone() * b.clone();
            let arm2 = (a.clone() * b.clone()) ^ (a.clone() * a.clone()) ^ (b.clone() * b.clone());
            let arm3 = a.clone() * b.clone() + a.clone() * a.clone() + b.clone() * b.clone() + a.clone();
            let sel = mux(
                mode.value().eq_expr(lit_u(0, 2)),
                arm0,
                mux(
                    mode.value().eq_expr(lit_u(1, 2)),
                    arm1,
                    mux(mode.value().eq_expr(lit_u(2, 2)), arm2, arm3),
                ),
            );
            m.next(acc, acc.value() + sel);
            let o = m.output("o", uint(32));
            m.assign(o, acc.value());
        }
        design
    }

    #[test]
    fn freeze_prunes_gated_datapaths() {
        let design = mode_gated_design();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Cfg").unwrap();
        let machine = lower_to_machine_program(&program);
        let mode_idx = program
            .signals
            .iter()
            .position(|s| s.name.ends_with(".mode"))
            .unwrap();

        // Freeze mode := 1: the select collapses to arm1; arms 0/2/3 are DCE'd.
        let frozen: HashMap<usize, u128> = [(mode_idx, 1u128)].into_iter().collect();
        let (_spec, fstats) = freeze_signals_program(&machine, &frozen);
        assert_eq!(fstats.frozen_signals, 1);
        assert!(
            fstats.specialize.instrs_removed() > 0,
            "freezing a mode register should prune gated datapaths: {fstats:?}"
        );
    }

    /// Bit-exact against the original WHILE mode actually holds its frozen value
    /// (the JIT runs the machine IR directly, so it can execute the frozen one).
    #[cfg(feature = "jit")]
    #[test]
    fn freeze_preserves_semantics_while_held() {
        use crate::jit::JitSimulator;
        let design = mode_gated_design();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Cfg").unwrap();
        let machine = lower_to_machine_program(&program);
        let idx = |n: &str| program.signals.iter().position(|s| s.name.ends_with(n)).unwrap();
        let (clk, we, a, b, mode, o) =
            (idx(".clk"), idx(".cfg_we"), idx(".a"), idx(".b"), idx(".mode"), idx(".o"));

        let frozen: HashMap<usize, u128> = [(mode, 1u128)].into_iter().collect();
        let (spec, _) = freeze_signals_program(&machine, &frozen);

        let mut orig = JitSimulator::compile(&machine).unwrap();
        let mut fast = JitSimulator::compile(&spec).unwrap();
        for s in [&mut orig, &mut fast] {
            s.set_signal(mode, 1); // hold mode := 1, cfg_we stays 0
        }
        for cycle in 0..32u64 {
            for s in [&mut orig, &mut fast] {
                s.set_signal(clk, 1);
                s.set_signal(we, 0);
                s.set_signal(a, cycle.wrapping_mul(0x1234) + 7);
                s.set_signal(b, cycle.wrapping_mul(0x99) + 3);
                s.tick();
            }
            assert_eq!(
                orig.get_signal(o),
                fast.get_signal(o),
                "frozen-mode specialization diverged at cycle {cycle}"
            );
        }
    }

    #[test]
    fn specialize_folds_and_prunes() {
        let machine = foldable_machine();
        let (spec, stats) = specialize_program(&machine);

        assert!(stats.folded > 0, "expected constant folds, got {stats:?}");
        assert!(stats.copies > 0, "expected identity copies, got {stats:?}");
        assert!(
            stats.instrs_after < stats.instrs_before,
            "specialization should remove instructions: {stats:?}"
        );

        // Value ids are compacted to 0..N per block.
        for block in [
            &spec.streams.async_reset_comb,
            &spec.streams.comb,
            &spec.streams.tick_next,
            &spec.streams.tick_commit,
        ] {
            assert!(block_value_ids_contiguous(block));
        }
    }

    /// Strength reduction must preserve simulation results for a range of
    /// multiplier constants (sparse and dense, power-of-two and odd).
    #[test]
    fn strength_reduction_preserves_semantics() {
        for &cmul in &[2u128, 3, 4, 5, 8, 10, 640, 0x9e37_79b9_7f4a_7c15, 0xffff_ffff_ffff_ffff] {
            let mut design = Design::new();
            let (clk, din);
            {
                let mut m = design.module("Mul");
                clk = m.input("clk", uint(1));
                din = m.input("din", uint(64));
                let acc = m.reg("acc", uint(64));
                m.clock(acc, clk);
                m.next(acc, acc * lit_u(cmul, 64) + din);
                let o = m.output("o", uint(64));
                m.assign(o, acc);
            }
            let compiled = compile(&design).unwrap();
            let program = lower_to_packed_program(&compiled, "Mul").unwrap();
            let (reduced, _) = strength_reduce_program(&program);

            let lanes = 4;
            let mut a = crate::SimdCpuSimulator::new(program.clone(), lanes).unwrap();
            let mut b = crate::SimdCpuSimulator::new(reduced, lanes).unwrap();
            let din_v: Vec<u128> = (0..lanes as u128).map(|l| l.wrapping_mul(0x9e37_79b9) + 1).collect();
            for s in [&mut a, &mut b] {
                s.set_signal(clk, &vec![1u128; lanes]).unwrap();
                s.set_signal(din, &din_v).unwrap();
            }
            let o = compiled
                .find_module("Mul")
                .and_then(|mm| mm.signals.iter().find(|s| s.name == "o"))
                .map(|s| s.handle)
                .unwrap();
            for cycle in 0..16 {
                a.tick();
                b.tick();
                assert_eq!(
                    a.get_signal(o).unwrap(),
                    b.get_signal(o).unwrap(),
                    "mismatch for *{cmul:#x} at cycle {cycle}"
                );
            }
        }
    }

    /// A deep comb chain has many short-lived intermediates, so slot allocation
    /// should reuse aggressively and leave a valid (contiguous, in-range) program.
    #[test]
    fn slot_allocation_reuses_and_stays_valid() {
        let mut design = Design::new();
        {
            let mut m = design.module("Chain");
            let clk = m.input("clk", uint(1));
            let din = m.input("din", uint(32));
            let acc = m.reg("acc", uint(32));
            m.clock(acc, clk);
            // Long dependent chain: each wire dies as soon as the next consumes it.
            let mut prev = acc.value() + din.value();
            for i in 0..32 {
                let w = m.wire(format!("w{i}"), uint(32));
                m.assign(w, prev + lit_u(i as u128 + 1, 32));
                prev = w.value();
            }
            m.next(acc, prev);
            let o = m.output("o", uint(32));
            m.assign(o, acc);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Chain").unwrap();
        let machine = lower_to_machine_program(&program);
        let (alloc, stats) = slot_allocate_program(&machine);

        assert!(
            stats.slots_after < stats.slots_before,
            "expected slot reuse: {stats:?}"
        );
        // Every operand and dst must be within the allocated slot range per block.
        for block in [
            &alloc.streams.async_reset_comb,
            &alloc.streams.comb,
            &alloc.streams.tick_next,
            &alloc.streams.tick_commit,
        ] {
            let n = block_value_count(block);
            for packet in &block.packets {
                for instr in &packet.instrs {
                    assert!(instr.dst.0 < n);
                    for id in instr_operands(&instr.kind) {
                        assert!(id.0 < n, "operand out of range after alloc");
                    }
                }
            }
        }
    }

    #[test]
    fn specialize_is_idempotent() {
        let machine = foldable_machine();
        let (once, _) = specialize_program(&machine);
        let (twice, stats2) = specialize_program(&once);
        assert_eq!(once, twice, "second pass should be a no-op");
        assert_eq!(stats2.folded, 0);
        assert_eq!(stats2.copies, 0);
        assert_eq!(stats2.dead, 0);
    }

    /// A `k`-deep priority mux chain `mux(c0,v0, mux(c1,v1, ... base))` latched into
    /// a register — the structure rebalancing targets.
    fn priority_chain_design(k: usize) -> Design {
        use rrtl_ir::mux;
        let mut design = Design::new();
        {
            let mut m = design.module("Chain");
            let clk = m.input("clk", uint(1));
            let mut conds = Vec::new();
            let mut vals = Vec::new();
            for i in 0..k {
                conds.push(m.input(&format!("c{i}"), uint(1)));
                vals.push(m.input(&format!("v{i}"), uint(8)));
            }
            let base = m.input("base", uint(8));
            let mut expr = base.value();
            for i in (0..k).rev() {
                expr = mux(conds[i].value(), vals[i].value(), expr);
            }
            let out = m.reg("out", uint(8));
            m.clock(out, clk);
            m.next(out, expr);
            let o = m.output("o", uint(8));
            m.assign(o, out.value());
        }
        design
    }

    /// Longest chain of mux->mux dependencies in a block (the serial depth a
    /// rebalance shortens).
    fn mux_chain_depth(block: &crate::PackedBlock) -> usize {
        let mut muxd: HashMap<usize, usize> = HashMap::new();
        let mut best = 0;
        for packet in &block.packets {
            for instr in &packet.instrs {
                let ops = instr_operands(&instr.kind);
                let child = ops.iter().map(|o| *muxd.get(&o.0).unwrap_or(&0)).max().unwrap_or(0);
                let d = if matches!(instr.kind, PackedInstrKind::Mux { .. }) { child + 1 } else { child };
                muxd.insert(instr.dst.0, d);
                best = best.max(d);
            }
        }
        best
    }

    #[test]
    fn rebalance_reduces_mux_chain_depth() {
        let design = priority_chain_design(12);
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Chain").unwrap();
        let machine = lower_to_machine_program(&program);
        let before = mux_chain_depth(&machine.streams.tick_next);
        let (rb, stats) = rebalance_mux_chains_program(&machine);
        let after = mux_chain_depth(&rb.streams.tick_next);
        assert!(stats.chains >= 1, "expected a chain to be rebalanced");
        assert_eq!(stats.deepest, 12, "the full 12-deep chain");
        assert!(after < before, "depth must drop: {before} -> {after}");
        assert!(after <= 5, "12-deep chain should rebalance to ~ceil(log2 12)=4: got {after}");
    }

    #[cfg(feature = "jit")]
    #[test]
    fn rebalance_preserves_semantics() {
        use crate::jit::JitSimulator;
        let k = 12usize;
        let design = priority_chain_design(k);
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Chain").unwrap();
        let machine = lower_to_machine_program(&program);
        let (rb, stats) = rebalance_mux_chains_program(&machine);
        assert!(stats.chains >= 1);

        let idx = |n: &str| {
            let h = compiled.find_module("Chain").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
            program.signal_index(h).unwrap()
        };
        let clk = idx("clk");
        let conds: Vec<usize> = (0..k).map(|i| idx(&format!("c{i}"))).collect();
        let vals: Vec<usize> = (0..k).map(|i| idx(&format!("v{i}"))).collect();
        let base = idx("base");
        let o = idx("o");

        let mut a = JitSimulator::compile(&machine).unwrap();
        let mut b = JitSimulator::compile(&rb).unwrap();
        // Drive a deterministic pseudo-random stream over conditions/values so many
        // priority winners (and the none-hit default) are exercised. The SAME inputs
        // go to both engines each cycle.
        let mut st = 0x9e3779b9u32;
        let mut rng = || { st ^= st << 13; st ^= st >> 17; st ^= st << 5; st };
        for _ in 0..400 {
            let base_v = (rng() & 0xff) as u64;
            let cv: Vec<(u64, u64)> = (0..k).map(|_| ((rng() & 1) as u64, (rng() & 0xff) as u64)).collect();
            for sim in [&mut a, &mut b] {
                sim.set_signal(clk, 1);
                sim.set_signal(base, base_v);
                for i in 0..k {
                    sim.set_signal(conds[i], cv[i].0);
                    sim.set_signal(vals[i], cv[i].1);
                }
            }
            a.tick();
            b.tick();
            assert_eq!(a.get_signal(o), b.get_signal(o), "rebalanced output diverged");
        }
    }
}
