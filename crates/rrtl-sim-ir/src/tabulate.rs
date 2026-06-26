//! Decoder tabulation: replace a deep combinational decode cone — a pure
//! function of a *narrow* instruction key — with a single precomputed ROM
//! lookup. This is the lever §4h.1 attributes to Verilator's maturity
//! ("table-izing decoders"), which `clang -O3` cannot reconstruct from the
//! flattened C and which the value-level specializer (`specialize.rs`) does not
//! perform.
//!
//! Targeted at the Cranelift JIT, where it collapses picorv32's ~48-deep decode
//! mux chain (96 % of the comb critical path) into one load — the JIT's weaker
//! instruction scheduler benefits from depth reduction (cf. the mux-rebalance
//! result, a JIT win / AOT loss). It is a *latency* lever: a ROM read is a
//! data-dependent gather, so it is opt-in and not applied on the data-parallel
//! batch/GPU backends.
//!
//! ## What it recognizes
//! picorv32's decode-flag registers lower to a one-level *hold-mux*:
//! `flag <= trigger ? decode(instr_bits) : flag`. The `then` (decode) branch of
//! a group of such flags that are pure functions of a *single* instruction
//! register (e.g. `mem_rdata_latched`) shares a narrow bit key. We:
//!   1. find that group and its joint key bits (greedily dropping flags whose
//!      bits would push the key past `key_max`),
//!   2. build a ROM by evaluating the decode cone over all `2^key` keys, packing
//!      one output bit per flag,
//!   3. add the ROM as a [`PackedMemory`], emit `addr = pack(key bits)` and one
//!      `MemRead`, and rewrite each flag's `then` to a 1-bit slice of the row,
//!   4. run [`specialize_program`] to DCE the now-dead decode cone.
//!
//! The big datapath cones (`reg_out`, `cpu_state`, the immediate path) have wide
//! inputs and are correctly left untouched — tabulation is a *decoder* lever.

use std::collections::{HashMap, HashSet};

use rrtl_ir::{BitType, Signedness, Width};

use crate::specialize::specialize_program;
use crate::{
    limbs, PackedEffect, PackedInstr, PackedInstrKind, PackedMachinePacket, PackedMachineProgram,
    PackedMemory, PackedValueId, PackedValueLayout,
};

/// Result of a successful tabulation, for reporting and ROM initialization.
#[derive(Clone, Debug)]
pub struct TabulateStats {
    /// Number of decode-flag registers folded into the ROM.
    pub tabulated_flags: usize,
    /// The instruction signal the ROM is keyed on.
    pub key_signal: usize,
    /// The key bit positions of `key_signal`, in packed (LSB-first) order.
    pub key_bits: Vec<u32>,
    /// Destination signal indices of the tabulated flags, in ROM-bit order.
    pub flag_regs: Vec<usize>,
    /// `2^key_bits.len()`: number of ROM entries.
    pub rom_depth: usize,
    /// Index of the ROM in the program's memory list (for `set_memory`).
    pub rom_mem_index: usize,
    /// ROM contents: one packed-flag word per key (length `rom_depth`).
    pub rom_data: Vec<u128>,
    pub instrs_before: usize,
    pub instrs_after: usize,
}

fn lit_u128(words: &[u32]) -> u128 {
    let mut v = 0u128;
    for (i, &w) in words.iter().enumerate().take(4) {
        v |= (w as u128) << (32 * i);
    }
    v
}

fn mask_u128(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

/// Value-id operands of an instruction kind.
fn operands(kind: &PackedInstrKind) -> Vec<PackedValueId> {
    use PackedInstrKind::*;
    match kind {
        Lit(_) | Signal(_) => vec![],
        Not(a) | Zext(a) | Sext(a) | Trunc(a) | Cast(a) | Slice { value: a, .. } => vec![*a],
        And(a, b) | Or(a, b) | Xor(a, b) | Add(a, b) | Sub(a, b) | Mul(a, b) | Eq(a, b)
        | Ne(a, b) => vec![*a, *b],
        Lt { lhs, rhs, .. } => vec![*lhs, *rhs],
        Mux { cond, then_value, else_value } => vec![*cond, *then_value, *else_value],
        Concat(vs) => vs.clone(),
        MemRead { addr, .. } => vec![*addr],
    }
}

/// Bit-level input support of a value's cone, tracking the `Slice(Signal,lsb)`
/// field-extraction pattern precisely. Returns `None` if the cone reads memory
/// (not a pure combinational function).
fn cone_support(
    root: PackedValueId,
    def: &HashMap<PackedValueId, PackedInstr>,
) -> Option<(HashSet<(usize, u32)>, usize)> {
    let mut seen: HashSet<PackedValueId> = HashSet::new();
    let mut stack = vec![root];
    let mut support: HashSet<(usize, u32)> = HashSet::new();
    let mut ops = 0usize;
    while let Some(v) = stack.pop() {
        if !seen.insert(v) {
            continue;
        }
        let Some(instr) = def.get(&v) else { continue };
        ops += 1;
        match &instr.kind {
            PackedInstrKind::Lit(_) => {}
            PackedInstrKind::Signal(idx) => {
                for b in 0..instr.ty.width as u32 {
                    support.insert((*idx, b));
                }
            }
            PackedInstrKind::Slice { value, lsb } => {
                if let Some(inner) = def.get(value) {
                    if let PackedInstrKind::Signal(idx) = &inner.kind {
                        for b in 0..instr.ty.width as u32 {
                            support.insert((*idx, *lsb as u32 + b));
                        }
                        continue;
                    }
                }
                stack.push(*value);
            }
            PackedInstrKind::MemRead { .. } => return None,
            other => {
                for op in operands(other) {
                    stack.push(op);
                }
            }
        }
    }
    Some((support, ops))
}

/// A compact, signal-parameterized evaluation op for fast ROM construction.
enum E {
    Lit(u128),
    Key,  // the keyed signal — returns the supplied key value
    Zero, // any other signal — 0 (should not occur for a pure-key cone)
    Not(usize),
    And(usize, usize),
    Or(usize, usize),
    Xor(usize, usize),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Eq(usize, usize),
    Ne(usize, usize),
    Lt { l: usize, r: usize, signed: bool },
    Mux { c: usize, t: usize, e: usize },
    Slice { v: usize, lsb: u32 },
    Zext(usize),
    Sext { v: usize, src_w: u32 },
    Trunc(usize),
    Cast(usize),
    Concat(Vec<(usize, u32)>), // (operand, width), MSB-first
}

/// A flattened evaluator over one cone: ops in dependency order, each masked to
/// its width, with the keyed signal substituted at run time.
struct Evaluator {
    ops: Vec<(E, u32)>,             // (op, result width)
    index_of: HashMap<PackedValueId, usize>,
}

impl Evaluator {
    /// Build from the union cone reachable from `roots`, keyed on `key_signal`.
    fn build(roots: &[PackedValueId], def: &HashMap<PackedValueId, PackedInstr>, key_signal: usize) -> Self {
        let mut order: Vec<PackedValueId> = Vec::new();
        let mut visiting: HashSet<PackedValueId> = HashSet::new();
        let mut done: HashSet<PackedValueId> = HashSet::new();
        // Iterative post-order DFS (dependencies before dependents).
        let mut stack: Vec<(PackedValueId, bool)> = roots.iter().rev().map(|r| (*r, false)).collect();
        while let Some((v, expanded)) = stack.pop() {
            if done.contains(&v) {
                continue;
            }
            if expanded {
                done.insert(v);
                order.push(v);
                continue;
            }
            if !def.contains_key(&v) {
                done.insert(v);
                order.push(v); // a value with no def (shouldn't happen) → treated as Zero
                continue;
            }
            if !visiting.insert(v) {
                continue;
            }
            stack.push((v, true));
            for op in operands(&def[&v].kind) {
                if !done.contains(&op) {
                    stack.push((op, false));
                }
            }
        }
        let mut index_of: HashMap<PackedValueId, usize> = HashMap::new();
        for (i, v) in order.iter().enumerate() {
            index_of.insert(*v, i);
        }
        let idx = |v: &PackedValueId| index_of[v];
        let mut ops: Vec<(E, u32)> = Vec::with_capacity(order.len());
        for v in &order {
            let Some(instr) = def.get(v) else {
                ops.push((E::Zero, 1));
                continue;
            };
            let w = instr.ty.width as u32;
            let e = match &instr.kind {
                PackedInstrKind::Lit(words) => E::Lit(lit_u128(words) & mask_u128(w)),
                PackedInstrKind::Signal(s) => {
                    if *s == key_signal {
                        E::Key
                    } else {
                        E::Zero
                    }
                }
                PackedInstrKind::Not(a) => E::Not(idx(a)),
                PackedInstrKind::And(a, b) => E::And(idx(a), idx(b)),
                PackedInstrKind::Or(a, b) => E::Or(idx(a), idx(b)),
                PackedInstrKind::Xor(a, b) => E::Xor(idx(a), idx(b)),
                PackedInstrKind::Add(a, b) => E::Add(idx(a), idx(b)),
                PackedInstrKind::Sub(a, b) => E::Sub(idx(a), idx(b)),
                PackedInstrKind::Mul(a, b) => E::Mul(idx(a), idx(b)),
                PackedInstrKind::Eq(a, b) => E::Eq(idx(a), idx(b)),
                PackedInstrKind::Ne(a, b) => E::Ne(idx(a), idx(b)),
                PackedInstrKind::Lt { lhs, rhs, signed } => {
                    E::Lt { l: idx(lhs), r: idx(rhs), signed: *signed }
                }
                PackedInstrKind::Mux { cond, then_value, else_value } => {
                    E::Mux { c: idx(cond), t: idx(then_value), e: idx(else_value) }
                }
                PackedInstrKind::Slice { value, lsb } => E::Slice { v: idx(value), lsb: *lsb as u32 },
                PackedInstrKind::Zext(a) => E::Zext(idx(a)),
                PackedInstrKind::Sext(a) => {
                    let src_w = def.get(a).map(|i| i.ty.width as u32).unwrap_or(w);
                    E::Sext { v: idx(a), src_w }
                }
                PackedInstrKind::Trunc(a) => E::Trunc(idx(a)),
                PackedInstrKind::Cast(a) => E::Cast(idx(a)),
                PackedInstrKind::Concat(parts) => E::Concat(
                    parts.iter().map(|p| (idx(p), def.get(p).map(|i| i.ty.width as u32).unwrap_or(0))).collect(),
                ),
                PackedInstrKind::MemRead { .. } => E::Zero, // pure cone — unreachable
            };
            ops.push((e, w));
        }
        Evaluator { ops, index_of }
    }

    /// Evaluate the cone with the keyed signal set to `key`.
    fn eval(&self, key: u128, vals: &mut [u128]) {
        for (i, (op, w)) in self.ops.iter().enumerate() {
            let raw = match op {
                E::Lit(v) => *v,
                E::Key => key,
                E::Zero => 0,
                E::Not(a) => !vals[*a],
                E::And(a, b) => vals[*a] & vals[*b],
                E::Or(a, b) => vals[*a] | vals[*b],
                E::Xor(a, b) => vals[*a] ^ vals[*b],
                E::Add(a, b) => vals[*a].wrapping_add(vals[*b]),
                E::Sub(a, b) => vals[*a].wrapping_sub(vals[*b]),
                E::Mul(a, b) => vals[*a].wrapping_mul(vals[*b]),
                E::Eq(a, b) => (vals[*a] == vals[*b]) as u128,
                E::Ne(a, b) => (vals[*a] != vals[*b]) as u128,
                E::Lt { l, r, signed } => {
                    if *signed {
                        // operands already masked to their width; sign per the
                        // wider of the two (they share width in practice).
                        (vals[*l] as i128) < (vals[*r] as i128)
                    } else {
                        vals[*l] < vals[*r]
                    }
                    .then_some(1u128)
                    .unwrap_or(0)
                }
                E::Mux { c, t, e } => {
                    if vals[*c] != 0 {
                        vals[*t]
                    } else {
                        vals[*e]
                    }
                }
                E::Slice { v, lsb } => vals[*v] >> *lsb,
                E::Zext(a) => vals[*a],
                E::Sext { v, src_w } => {
                    let x = vals[*v];
                    if *src_w > 0 && (x >> (*src_w - 1)) & 1 == 1 {
                        x | !mask_u128(*src_w)
                    } else {
                        x
                    }
                }
                E::Trunc(a) => vals[*a],
                E::Cast(a) => vals[*a],
                E::Concat(parts) => {
                    let mut acc = 0u128;
                    let mut off = 0u32;
                    for (p, pw) in parts.iter().rev() {
                        acc |= (vals[*p] & mask_u128(*pw)) << off;
                        off += *pw;
                    }
                    acc
                }
            };
            vals[i] = raw & mask_u128(*w);
        }
    }
}

/// Try to tabulate picorv32-style decode flags into a ROM keyed on a single
/// instruction register. Returns the rewritten program and stats, or the
/// program unchanged with `None` if no profitable group is found.
///
/// `key_max` bounds the ROM key width (entries = `2^key_max`); larger covers
/// more flags but builds a bigger ROM.
pub fn tabulate_decode_program(
    machine: &PackedMachineProgram,
    key_max: u32,
) -> (PackedMachineProgram, Option<TabulateStats>) {
    // Def-map over the whole tick_next block (value ids are block-local but flow
    // across packets).
    let mut def: HashMap<PackedValueId, PackedInstr> = HashMap::new();
    let mut max_id = 0usize;
    for p in &machine.streams.tick_next.packets {
        for instr in &p.instrs {
            max_id = max_id.max(instr.dst.0);
            def.insert(instr.dst, instr.clone());
        }
    }

    // Candidate decode flags: CaptureReg whose next-state is the hold-mux
    // `Mux{cond, then, else=Signal(self)}`, with `then` a pure function of a
    // single instruction signal. Group by that signal.
    struct Flag {
        reg: usize,                // destination signal index (the register)
        mux: PackedValueId,        // the next-state Mux value id
        then_v: PackedValueId,     // the decode (then) branch
        then_w: u32,               // width of the decoded value (1 for flags, 5 for rd/rs fields, …)
        support: HashSet<u32>,     // bits of the key signal it reads
    }
    let mut groups: HashMap<usize, Vec<Flag>> = HashMap::new();
    for p in &machine.streams.tick_next.packets {
        for e in &p.effects {
            let PackedEffect::CaptureReg { dst, value, .. } = e else { continue };
            let Some(mux_instr) = def.get(value) else { continue };
            let PackedInstrKind::Mux { then_value, else_value, .. } = &mux_instr.kind else { continue };
            let is_hold = def
                .get(else_value)
                .map(|i| matches!(&i.kind, PackedInstrKind::Signal(s) if *s == *dst))
                .unwrap_or(false);
            if !is_hold {
                continue;
            }
            let Some((support, _ops)) = cone_support(*then_value, &def) else { continue };
            if support.is_empty() {
                continue;
            }
            let sigs: HashSet<usize> = support.iter().map(|(s, _)| *s).collect();
            if sigs.len() != 1 {
                continue; // not a pure function of a single instruction register
            }
            let key_signal = *sigs.iter().next().unwrap();
            let then_w = def.get(then_value).map(|i| i.ty.width as u32).unwrap_or(1);
            groups.entry(key_signal).or_default().push(Flag {
                reg: *dst,
                mux: *value,
                then_v: *then_value,
                then_w,
                support: support.iter().map(|(_, b)| *b).collect(),
            });
        }
    }

    // Pick the group (key signal) with the most flags; within it, greedily drop
    // the flag contributing the most unique key bits until the joint key fits
    // `key_max`. Need ≥2 flags to be worthwhile.
    let Some((&key_signal, _)) = groups.iter().max_by_key(|(_, fs)| fs.len()) else {
        return (machine.clone(), None);
    };
    let mut flags = groups.remove(&key_signal).unwrap();
    let joint_bits = |flags: &[Flag]| -> HashSet<u32> {
        let mut s = HashSet::new();
        for f in flags {
            s.extend(f.support.iter().copied());
        }
        s
    };
    while joint_bits(&flags).len() as u32 > key_max && flags.len() > 2 {
        // bits contributed only by one flag are the cheapest to drop with it.
        let key = joint_bits(&flags);
        // count, per flag, how many of its bits no OTHER flag needs.
        let mut worst = 0usize;
        let mut worst_unique = 0usize;
        for (i, f) in flags.iter().enumerate() {
            let unique = f
                .support
                .iter()
                .filter(|b| !flags.iter().enumerate().any(|(j, g)| j != i && g.support.contains(b)))
                .count();
            // prefer dropping the flag whose removal shrinks the key the most.
            let _ = key;
            if unique > worst_unique {
                worst_unique = unique;
                worst = i;
            }
        }
        if worst_unique == 0 {
            break; // dropping any single flag won't shrink the key
        }
        flags.remove(worst);
    }
    let key_set = joint_bits(&flags);
    if flags.len() < 2 || key_set.is_empty() || key_set.len() as u32 > key_max {
        return (machine.clone(), None);
    }

    // Canonical packed key order (LSB-first by bit position).
    let mut key_bits: Vec<u32> = key_set.into_iter().collect();
    key_bits.sort_unstable();
    let k = key_bits.len() as u32;
    let rom_depth = 1usize << k;
    let nflags = flags.len();

    // Lay out each decoded value in the ROM row at a cumulative bit offset (a
    // flag is 1 bit; decoded_rd/rs are 5 bits, etc.). The whole row must fit one
    // ≤128-bit memory slot.
    let mut row_offsets: Vec<u32> = Vec::with_capacity(nflags);
    let mut row_width = 0u32;
    for f in &flags {
        row_offsets.push(row_width);
        row_width += f.then_w;
    }
    if row_width > 128 {
        return (machine.clone(), None);
    }

    // Build the ROM by evaluating the union decode cone over all keys.
    let roots: Vec<PackedValueId> = flags.iter().map(|f| f.then_v).collect();
    let evaluator = Evaluator::build(&roots, &def, key_signal);
    let flag_slots: Vec<usize> = roots.iter().map(|r| evaluator.index_of[r]).collect();
    let key_signal_width = machine
        .source
        .signals
        .get(key_signal)
        .map(|s| s.layout.width as u32)
        .unwrap_or(32);
    let key_mask = mask_u128(key_signal_width);

    let mut rom_data = vec![0u128; rom_depth];
    let mut vals = vec![0u128; evaluator.ops.len()];
    for addr in 0..rom_depth {
        // Scatter the packed key bits into the instruction-register value.
        let mut key_val = 0u128;
        for (i, &pos) in key_bits.iter().enumerate() {
            if (addr >> i) & 1 == 1 {
                key_val |= 1u128 << pos;
            }
        }
        key_val &= key_mask;
        evaluator.eval(key_val, &mut vals);
        let mut row = 0u128;
        for (j, &slot) in flag_slots.iter().enumerate() {
            let v = vals[slot] & mask_u128(flags[j].then_w);
            row |= v << row_offsets[j];
        }
        rom_data[addr] = row;
    }

    // ---- Rewrite the program ----
    let mut out = machine.clone();

    // Add the ROM as a new memory. The JIT lays it out from `source.memories`
    // and reads it via `MemRead`; `source` (a Signal) is required by the struct
    // but unused by the JIT, so borrow any existing signal handle.
    let borrow_sig = out
        .source
        .signals
        .iter()
        .find_map(|s| s.source)
        .expect("program has at least one sourced signal for the ROM handle");
    let rom_mem_index = out.source.memories.len();
    let row_ty = BitType::new(row_width as Width, Signedness::Unsigned);
    let rom_offset = out
        .source
        .memories
        .iter()
        .map(|m| m.offset + m.depth * m.data_layout.limbs)
        .max()
        .unwrap_or(0);
    out.source.memories.push(PackedMemory {
        name: format!("{}.__decode_rom", out.source.top),
        source: borrow_sig,
        owner_path: out.source.top.clone(),
        addr_width: k as Width,
        depth: rom_depth,
        data_layout: PackedValueLayout {
            width: row_width as Width,
            ty: row_ty,
            offset: 0,
            limbs: limbs(row_width as Width),
        },
        offset: rom_offset,
    });

    // Allocate fresh value ids and build the address-pack + MemRead + per-flag
    // bit-slice instructions in a packet prepended to tick_next.
    let mut next_id = max_id + 1;
    let mut alloc = || {
        let v = PackedValueId(next_id);
        next_id += 1;
        v
    };
    let bit1 = BitType::new(1, Signedness::Unsigned);
    let key_signal_ty = BitType::new(key_signal_width as Width, Signedness::Unsigned);

    let mut prelude: Vec<PackedInstr> = Vec::new();
    // Signal(key) read.
    let s_val = alloc();
    prelude.push(PackedInstr { dst: s_val, ty: key_signal_ty, kind: PackedInstrKind::Signal(key_signal) });
    // One 1-bit slice per key bit.
    let mut bit_vals: Vec<PackedValueId> = Vec::with_capacity(key_bits.len());
    for &pos in &key_bits {
        let bv = alloc();
        prelude.push(PackedInstr {
            dst: bv,
            ty: bit1,
            kind: PackedInstrKind::Slice { value: s_val, lsb: pos as Width },
        });
        bit_vals.push(bv);
    }
    // Concat is MSB-first: place the highest packed bit first so the LSB key bit
    // lands at offset 0 (matching the ROM-construction `addr >> i` order).
    let addr_v = alloc();
    let mut concat_parts: Vec<PackedValueId> = bit_vals.clone();
    concat_parts.reverse();
    prelude.push(PackedInstr {
        dst: addr_v,
        ty: BitType::new(k as Width, Signedness::Unsigned),
        kind: PackedInstrKind::Concat(concat_parts),
    });
    // The ROM read.
    let row_v = alloc();
    prelude.push(PackedInstr {
        dst: row_v,
        ty: row_ty,
        kind: PackedInstrKind::MemRead { memory: rom_mem_index, addr: addr_v },
    });
    // One slice per tabulated value (at its packed offset/width), and the
    // mux→slice rewrite map.
    let mut rewrite: HashMap<PackedValueId, PackedValueId> = HashMap::new();
    for (j, f) in flags.iter().enumerate() {
        let fb = alloc();
        prelude.push(PackedInstr {
            dst: fb,
            ty: BitType::new(f.then_w as Width, Signedness::Unsigned),
            kind: PackedInstrKind::Slice { value: row_v, lsb: row_offsets[j] as Width },
        });
        rewrite.insert(f.mux, fb);
    }

    // Prepend the prelude packet, then rewrite each flag's hold-mux `then`.
    out.streams
        .tick_next
        .packets
        .insert(0, PackedMachinePacket { instrs: prelude, effects: Vec::new() });
    for packet in &mut out.streams.tick_next.packets {
        for instr in &mut packet.instrs {
            if let Some(&fb) = rewrite.get(&instr.dst) {
                if let PackedInstrKind::Mux { then_value, .. } = &mut instr.kind {
                    *then_value = fb;
                }
            }
        }
    }

    let instrs_before = count_instrs(machine);
    // DCE the now-dead decode cone and recompact value ids. `TAB_NODCE=1` skips
    // it (keeps the dead cone) to isolate the rewrite from the DCE pass.
    let out = if std::env::var("TAB_NODCE").is_ok() {
        out
    } else {
        specialize_program(&out).0
    };
    let instrs_after = count_instrs(&out);

    (
        out,
        Some(TabulateStats {
            tabulated_flags: nflags,
            key_signal,
            key_bits,
            flag_regs: flags.iter().map(|f| f.reg).collect(),
            rom_depth,
            rom_mem_index,
            rom_data,
            instrs_before,
            instrs_after,
        }),
    )
}

fn count_instrs(m: &PackedMachineProgram) -> usize {
    [
        &m.streams.async_reset_comb,
        &m.streams.comb,
        &m.streams.tick_next,
        &m.streams.tick_commit,
    ]
    .iter()
    .map(|b| b.packets.iter().map(|p| p.instrs.len()).sum::<usize>())
    .sum()
}

#[cfg(all(test, feature = "jit"))]
mod tests {
    use super::*;
    use crate::jit::JitSimulator;
    use crate::{lower_to_machine_program, lower_to_packed_program};
    use rrtl_core::{compile, Design};
    use rrtl_ir::{lit_u, mux, uint};

    /// A synthetic decoder: three registers latch a function of the low 4 bits
    /// of an 8-bit instruction via the hold-mux pattern (two 1-bit opcode flags
    /// and one 4-bit `rd`-style field, exercising multi-bit tabulation), feeding
    /// an accumulator so the output observably depends on the decode.
    #[test]
    fn tabulated_decode_matches_plain_jit() {
        let mut design = Design::new();
        {
            let mut m = design.module("Tab");
            let clk = m.input("clk", uint(1));
            let trig = m.input("trig", uint(1));
            let instr = m.input("instr", uint(8));

            let r1 = m.reg("r1", uint(1));
            let r2 = m.reg("r2", uint(1));
            let rd = m.reg("rd", uint(4));
            let acc = m.reg("acc", uint(8));
            for r in [r1, r2, rd, acc] {
                m.clock(r, clk);
            }
            // flag <= trig ? decode(instr[3:0]) : flag  (the hold-mux)
            m.next(r1, mux(trig.value(), instr.value().slice(0, 4).eq_expr(lit_u(1, 4)), r1.value()));
            m.next(r2, mux(trig.value(), instr.value().slice(0, 4).eq_expr(lit_u(2, 4)), r2.value()));
            m.next(rd, mux(trig.value(), instr.value().slice(0, 4), rd.value()));
            m.next(
                acc,
                acc.value() + r1.value().zext(8) + r2.value().zext(8) + rd.value().zext(8),
            );
            let o = m.output("o", uint(8));
            m.assign(o, acc.value());
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Tab").unwrap();
        let machine = lower_to_machine_program(&program);
        let module = compiled.find_module("Tab").unwrap();
        let idx = |n: &str| {
            let h = module.signals.iter().find(|s| s.name == n).unwrap().handle;
            program.signal_index(h).unwrap()
        };
        let (i_clk, i_trig, i_instr, i_o) = (idx("clk"), idx("trig"), idx("instr"), idx("o"));

        let (tab, stats) = tabulate_decode_program(&machine, 8);
        let stats = stats.expect("expected a tabulation group");
        assert!(stats.tabulated_flags >= 2, "should tabulate the decode group");
        // (On a tiny synthetic decode the prelude can exceed the removed cone;
        // the instruction-count win is validated on picorv32 — see the example.)

        let mut plain = JitSimulator::compile(&machine).unwrap();
        let mut tabj = JitSimulator::compile(&tab).unwrap();
        tabj.set_memory(stats.rom_mem_index, &stats.rom_data);

        // Deterministic pseudo-random stimulus; outputs must match every cycle.
        let mut lcg = 0x1234_5678u32;
        for _ in 0..400 {
            lcg = lcg.wrapping_mul(1664525).wrapping_add(1013904223);
            let instr_v = ((lcg >> 8) & 0xFF) as u64;
            let trig_v = ((lcg >> 3) & 1) as u64;
            plain.set_signal(i_clk, 1);
            plain.set_signal(i_trig, trig_v);
            plain.set_signal(i_instr, instr_v);
            plain.tick();
            tabj.set_signal(i_clk, 1);
            tabj.set_signal(i_trig, trig_v);
            tabj.set_signal(i_instr, instr_v);
            tabj.tick();
            assert_eq!(plain.get_signal(i_o), tabj.get_signal(i_o), "tabulated output diverged");
        }
    }
}
