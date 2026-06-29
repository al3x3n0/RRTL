//! AOT C backend: lower a `PackedMachineProgram` to C, compile it with the
//! system C compiler at `-O3`, load the shared object, and run it — the *same*
//! optimizing toolchain Verilator uses, driven from RRTL's portable packed IR.
//! This isolates "codegen quality" as the variable when comparing against both
//! the Cranelift JIT and Verilator (research-artifact value), at the cost of an
//! up-front per-design compile (AOT, amortized over a long run).
//!
//! Scope mirrors the JIT: signals/memory ≤128 bits, the four machine streams,
//! a uniform 16-byte state slot per signal (byte offset `i*16`). Eager settle
//! (post-commit re-settle every tick) keeps the wrapper trivial; observation is
//! always correct. Bit-exact with [`crate::jit::JitSimulator`].

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use crate::{
    PackedBlock, PackedEffect, PackedInstr, PackedInstrKind, PackedMachineProgram, PackedReset,
    PackedSignalKind,
};
use rrtl_ir::{ErrorReport, ResetPolarity};

fn aot_err(msg: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![rrtl_ir::Diagnostic::new("E_AOT", msg)])
}

/// Compute type for a value of width `w` (the type intermediate values use).
fn ctype(w: u32) -> &'static str {
    if w <= 64 {
        "u64"
    } else {
        "u128"
    }
}

/// Storage size in bytes for a signal/memory entry of width `w` — the smallest
/// natural type that holds it (1/2/4/8/16). Packing to this instead of a uniform
/// 16-byte slot shrinks the working set ~4x for 32-bit-heavy designs → far better
/// cache locality (the dominant single-instance gap vs Verilator's packed structs).
fn store_bytes(w: u32) -> usize {
    match w {
        0..=8 => 1,
        9..=16 => 2,
        17..=32 => 4,
        33..=64 => 8,
        _ => 16,
    }
}

/// C storage type matching [`store_bytes`].
fn stype(w: u32) -> &'static str {
    match store_bytes(w) {
        1 => "uint8_t",
        2 => "uint16_t",
        4 => "uint32_t",
        8 => "u64",
        _ => "u128",
    }
}

/// Width-packed state layout: byte offset per signal and per memory, grouped by
/// storage size (largest first) so every slot is naturally aligned with zero
/// inter-slot padding.
pub struct Layout {
    pub off: Vec<usize>,        // byte offset of each signal
    pub mem_base: Vec<usize>,   // byte offset of each memory's entry 0
    pub mem_entry: Vec<usize>,  // storage bytes per memory entry
    pub total: usize,           // total state size in bytes
}

fn build_layout(machine: &PackedMachineProgram) -> Layout {
    let widths: Vec<u32> = machine.source.signals.iter().map(|s| s.layout.width).collect();
    // AOT_FAT=1 reverts to uniform 16-byte slots (to A/B the packing win).
    if std::env::var("AOT_FAT").is_ok() {
        let off: Vec<usize> = (0..widths.len()).map(|i| i * 16).collect();
        let mut cur = widths.len() * 16;
        let mut mem_base = Vec::new();
        let mut mem_entry = Vec::new();
        for m in &machine.source.memories {
            mem_base.push(cur);
            mem_entry.push(16);
            cur += 16 * m.depth;
        }
        return Layout { off, mem_base, mem_entry, total: cur };
    }
    let align = |x: usize, a: usize| (x + a - 1) & !(a - 1);
    // AOT_AFFINITY=1: order signals so a register cone's support (+ the register)
    // is contiguous → co-accessed signals share cache lines. Structurally -40%
    // cache-lines-touched/cone on picorv32 at +15% bytes (state_layout_probe);
    // bit-exact by construction (offset permutation). Opt-in until the throughput
    // win is confirmed on a non-throttled machine.
    if std::env::var("AOT_AFFINITY").is_ok() {
        let supports = crate::register_support(&machine.source);
        let mut order: Vec<usize> = Vec::with_capacity(widths.len());
        let mut seen = vec![false; widths.len()];
        for rs in &supports {
            for &s in rs.support.iter().chain(std::iter::once(&rs.reg)) {
                if s < seen.len() && !seen[s] {
                    seen[s] = true;
                    order.push(s);
                }
            }
        }
        for s in 0..widths.len() {
            if !seen[s] {
                order.push(s);
            }
        }
        let mut off = vec![0usize; widths.len()];
        let mut cur = 0usize;
        for &s in &order {
            let sz = store_bytes(widths[s]);
            cur = align(cur, sz);
            off[s] = cur;
            cur += sz;
        }
        cur = align(cur, 16);
        let mut mem_base = Vec::new();
        let mut mem_entry = Vec::new();
        for m in &machine.source.memories {
            let eb = store_bytes(m.data_layout.width);
            cur = align(cur, eb);
            mem_base.push(cur);
            mem_entry.push(eb);
            cur += eb * m.depth;
        }
        return Layout { off, mem_base, mem_entry, total: align(cur, 16) };
    }
    let mut off = vec![0usize; widths.len()];
    let mut cur = 0usize;
    // Place 16-byte slots first (offset 0 = 16-aligned), then 8/4/2/1 — each
    // group starts at a multiple of its size, so all accesses are aligned.
    for size in [16usize, 8, 4, 2, 1] {
        for (i, w) in widths.iter().enumerate() {
            if store_bytes(*w) == size {
                off[i] = cur;
                cur += size;
            }
        }
    }
    cur = align(cur, 16);
    let mut mem_base = Vec::new();
    let mut mem_entry = Vec::new();
    for m in &machine.source.memories {
        let eb = store_bytes(m.data_layout.width);
        cur = align(cur, eb);
        mem_base.push(cur);
        mem_entry.push(eb);
        cur += eb * m.depth;
    }
    Layout { off, mem_base, mem_entry, total: align(cur, 16) }
}

fn lit_u128(words: &[u32]) -> u128 {
    let mut v = 0u128;
    for (i, w) in words.iter().enumerate().take(4) {
        v |= (*w as u128) << (32 * i);
    }
    v
}

fn mask128(w: u32) -> u128 {
    if w >= 128 {
        u128::MAX
    } else {
        (1u128 << w) - 1
    }
}

/// C `& <mask>` suffix to clear bits above `w` (empty when the type is exact).
fn mask_suffix(w: u32) -> String {
    if w >= 128 || w == 64 {
        String::new()
    } else if w > 64 {
        let m = mask128(w);
        format!(" & (((u128)0x{:016x}ULL<<64)|0x{:016x}ULL)", (m >> 64) as u64, m as u64)
    } else {
        format!(" & 0x{:x}ULL", (1u64 << w) - 1)
    }
}

/// A u128/u64 C literal.
fn fmt_const(v: u128, w: u32) -> String {
    if w <= 64 {
        format!("0x{:x}ULL", v as u64)
    } else {
        format!("(((u128)0x{:016x}ULL<<64)|0x{:016x}ULL)", (v >> 64) as u64, v as u64)
    }
}

struct Gen<'a> {
    widths: &'a [u32],
    layout: &'a Layout,
    mem_depth: Vec<usize>,
    mem_width: Vec<u32>,
}

/// Control-flow recovery plan for one block (see `Gen::cflow_analyze`).
struct Cflow<'b> {
    flat: Vec<&'b PackedInstr>,
    def_idx: HashMap<usize, usize>,
    /// value-ids emitted inside some branch (skip at top level).
    sunk: HashSet<usize>,
    /// mux value-id → ordered member value-ids of its then / else arm.
    then_mem: HashMap<usize, Vec<usize>>,
    else_mem: HashMap<usize, Vec<usize>>,
    /// mux value-ids that are emitted as if/else (have a sunk arm cone).
    branched: HashSet<usize>,
}

/// The value-id operands an instruction reads.
fn value_operands(kind: &PackedInstrKind) -> Vec<usize> {
    use PackedInstrKind::*;
    match kind {
        Lit(_) | Signal(_) => vec![],
        Not(a) | Zext(a) | Sext(a) | Trunc(a) | Cast(a) | Slice { value: a, .. } => vec![a.0],
        And(a, b) | Or(a, b) | Xor(a, b) | Add(a, b) | Sub(a, b) | Mul(a, b) | Eq(a, b) | Ne(a, b) => {
            vec![a.0, b.0]
        }
        Lt { lhs, rhs, .. } => vec![lhs.0, rhs.0],
        Mux { cond, then_value, else_value } => vec![cond.0, then_value.0, else_value.0],
        Concat(parts) => parts.iter().map(|p| p.0).collect(),
        MemRead { addr, .. } => vec![addr.0],
    }
}

fn effect_value_operands(effect: &PackedEffect) -> Vec<usize> {
    match effect {
        PackedEffect::StoreSignal { value, .. } => vec![value.0],
        PackedEffect::CaptureReg { value, .. } => vec![value.0],
        PackedEffect::MemoryWrite { enable, addr, data, .. } => vec![enable.0, addr.0, data.0],
    }
}

/// An arm is worth sinking only if its root is real compute (a mux or arithmetic/
/// logic), not a bare copy/leaf — branching a trivial arm only adds a mispredict.
fn arm_worth_sinking(
    v: usize,
    def_idx: &HashMap<usize, usize>,
    use_count: &HashMap<usize, usize>,
    sunk: &HashSet<usize>,
    flat: &[&PackedInstr],
) -> bool {
    if use_count.get(&v) != Some(&1) || sunk.contains(&v) {
        return false;
    }
    let Some(&idx) = def_idx.get(&v) else { return false };
    matches!(
        flat[idx].kind,
        PackedInstrKind::Mux { .. }
            | PackedInstrKind::Mul(..)
            | PackedInstrKind::Add(..)
            | PackedInstrKind::Sub(..)
            | PackedInstrKind::And(..)
            | PackedInstrKind::Or(..)
            | PackedInstrKind::Xor(..)
            | PackedInstrKind::Eq(..)
            | PackedInstrKind::Ne(..)
            | PackedInstrKind::Lt { .. }
            | PackedInstrKind::Concat(..)
            | PackedInstrKind::MemRead { .. }
    )
}

/// Claim the single-use cone rooted at `v` into `members` (topological: operands
/// before `v`). A nested mux claims only its condition cone here — its arms are its
/// own scopes, already populated when it was processed.
fn claim(
    v: usize,
    members: &mut Vec<usize>,
    sunk: &mut HashSet<usize>,
    flat: &[&PackedInstr],
    def_idx: &HashMap<usize, usize>,
    use_count: &HashMap<usize, usize>,
) {
    if use_count.get(&v) != Some(&1) || sunk.contains(&v) {
        return;
    }
    let Some(&idx) = def_idx.get(&v) else { return };
    sunk.insert(v);
    match &flat[idx].kind {
        PackedInstrKind::Mux { cond, .. } => {
            claim(cond.0, members, sunk, flat, def_idx, use_count);
            members.push(v);
        }
        other => {
            for o in value_operands(other) {
                claim(o, members, sunk, flat, def_idx, use_count);
            }
            members.push(v);
        }
    }
}

impl<'a> Gen<'a> {
    /// Read signal `idx` from its packed slot, value-extended to the compute type.
    fn sig_read(&self, idx: usize) -> String {
        format!(
            "(({})*({}*)(st+{}))",
            ctype(self.widths[idx]), stype(self.widths[idx]), self.layout.off[idx]
        )
    }
    /// LHS lvalue for storing to signal `idx`'s packed slot.
    fn sig_lval(&self, idx: usize) -> String {
        format!("*({}*)(st+{})", stype(self.widths[idx]), self.layout.off[idx])
    }

    /// The C right-hand-side expression for one instruction (no decl, no mask).
    /// Shared by the flat and control-flow emitters. `vw` must hold the widths of
    /// all operands (defined earlier); `sig_local` provides fused comb-signal locals.
    fn instr_rhs(
        &self,
        kind: &PackedInstrKind,
        w: u32,
        prefix: &str,
        vw: &HashMap<usize, u32>,
        sig_local: &HashMap<usize, String>,
        fuse: bool,
    ) -> String {
        let t = ctype(w);
        let pv = |a: &crate::PackedValueId| format!("{}{}", prefix, a.0);
        let op = |a: &crate::PackedValueId| format!("(({}){})", t, pv(a));
        match kind {
            PackedInstrKind::Lit(words) => fmt_const(lit_u128(words) & mask128(w), w),
            PackedInstrKind::Signal(s) => {
                let src = if fuse { sig_local.get(s).cloned() } else { None }
                    .unwrap_or_else(|| self.sig_read(*s));
                format!("({})({})", t, src)
            }
            PackedInstrKind::Not(a) => format!("~{}", op(a)),
            PackedInstrKind::And(a, b) => format!("{} & {}", op(a), op(b)),
            PackedInstrKind::Or(a, b) => format!("{} | {}", op(a), op(b)),
            PackedInstrKind::Xor(a, b) => format!("{} ^ {}", op(a), op(b)),
            PackedInstrKind::Add(a, b) => format!("{} + {}", op(a), op(b)),
            PackedInstrKind::Sub(a, b) => format!("{} - {}", op(a), op(b)),
            PackedInstrKind::Mul(a, b) => format!("{} * {}", op(a), op(b)),
            PackedInstrKind::Eq(a, b) => {
                let tc = ctype(vw[&a.0].max(vw[&b.0]));
                format!("(({}){} == ({}){}) ? 1 : 0", tc, pv(a), tc, pv(b))
            }
            PackedInstrKind::Ne(a, b) => {
                let tc = ctype(vw[&a.0].max(vw[&b.0]));
                format!("(({}){} != ({}){}) ? 1 : 0", tc, pv(a), tc, pv(b))
            }
            PackedInstrKind::Lt { lhs, rhs, signed } => {
                let cw = vw[&lhs.0].max(vw[&rhs.0]);
                if *signed {
                    if cw <= 64 {
                        format!(
                            "((int64_t)sext64({},{}) < (int64_t)sext64({},{})) ? 1 : 0",
                            pv(lhs), vw[&lhs.0], pv(rhs), vw[&rhs.0]
                        )
                    } else {
                        format!(
                            "((i128)sext128((u128){},{}) < (i128)sext128((u128){},{})) ? 1 : 0",
                            pv(lhs), vw[&lhs.0], pv(rhs), vw[&rhs.0]
                        )
                    }
                } else {
                    let tc = ctype(cw);
                    format!("(({}){} < ({}){}) ? 1 : 0", tc, pv(lhs), tc, pv(rhs))
                }
            }
            PackedInstrKind::Mux { cond, then_value, else_value } => {
                format!("({} & 1) ? {} : {}", pv(cond), op(then_value), op(else_value))
            }
            PackedInstrKind::Slice { value, lsb } => {
                format!("({})({} >> {})", t, pv(value), lsb)
            }
            PackedInstrKind::Zext(a) | PackedInstrKind::Trunc(a) | PackedInstrKind::Cast(a) => op(a),
            PackedInstrKind::Sext(a) => {
                if w <= 64 {
                    format!("sext64((u64){},{})", pv(a), vw[&a.0])
                } else {
                    format!("sext128((u128){},{})", pv(a), vw[&a.0])
                }
            }
            PackedInstrKind::Concat(parts) => {
                let mut terms = Vec::new();
                let mut offset = 0u32;
                for part in parts.iter().rev() {
                    let pw = vw[&part.0];
                    terms.push(format!("((({}){}{}) << {})", t, pv(part), mask_suffix(pw), offset));
                    offset += pw;
                }
                if terms.is_empty() { "0".to_string() } else { terms.join(" | ") }
            }
            PackedInstrKind::MemRead { memory, addr } => {
                let depth = self.mem_depth[*memory];
                if depth == 0 {
                    "0".to_string()
                } else {
                    let mt = stype(self.mem_width[*memory]);
                    let eb = self.layout.mem_entry[*memory];
                    format!(
                        "((u128){} < {}) ? (*({}*)(st+{}+(size_t)({})*{})) : 0",
                        pv(addr), depth, mt, self.layout.mem_base[*memory], pv(addr), eb
                    )
                }
            }
        }
    }

    /// Emit one block's instructions and effects into `code`. Value-ids are named
    /// `{prefix}{id}` so several fused blocks share one C scope without collision.
    /// A `Signal` read resolves to a live local in `sig_local` when present — this
    /// is the settle→capture FUSION: comb values stay in registers instead of
    /// round-tripping the state buffer. When `store_comb` is false a plain comb
    /// StoreSignal updates `sig_local` but is NOT written to state (capture reads
    /// it via the local; observability is recomputed by `settle()`), removing the
    /// comb store traffic from the hot tick. `capture_to_temp` routes CaptureReg
    /// to `next_<dst>` temps (tick_next) vs a conditional immediate store (async).
    fn emit_block(
        &self,
        code: &mut String,
        block: &PackedBlock,
        prefix: &str,
        sig_local: &mut HashMap<usize, String>,
        store_comb: bool,
        capture_to_temp: bool,
        fuse: bool,
        store_set: Option<&std::collections::HashSet<usize>>,
        cflow: bool,
        // Multi-clock: `memory idx → clock-bit`. A gated memory write is guarded
        // by `(mask >> bit) & 1` (the `mask` param of `tick_clocked`). `None` =
        // always write (single-clock / non-tick_clocked emission).
        mem_gate: Option<&HashMap<usize, u32>>,
    ) -> Result<(), ErrorReport> {
        let mut vw: HashMap<usize, u32> = HashMap::new();
        let pv = |a: &crate::PackedValueId| format!("{}{}", prefix, a.0);
        // Control-flow recovery: sink single-use mux-arm cones into real if/else so
        // the untaken arm's work is skipped (eval-taken). Latency-only (it trades the
        // dataflow IR's divergence-freedom). clang cannot do this from the flattened
        // ternary — it has no licence to not-compute a pure sub-expression.
        let cf = if cflow { Some(self.cflow_analyze(block)) } else { None };
        for packet in &block.packets {
            for instr in &packet.instrs {
                let w = instr.ty.width;
                let id = instr.dst.0;
                if let Some(cf) = &cf {
                    if cf.sunk.contains(&id) {
                        continue; // emitted inside its owning branch
                    }
                    if cf.branched.contains(&id) {
                        self.emit_branch(code, instr, prefix, &mut vw, sig_local, fuse, cf);
                        continue;
                    }
                }
                let rhs = self.instr_rhs(&instr.kind, w, prefix, &vw, sig_local, fuse);
                writeln!(code, "    {} {}{} = ({}){};", ctype(w), prefix, id, rhs, mask_suffix(w)).ok();
                vw.insert(id, w);
            }
            for effect in &packet.effects {
                match effect {
                    PackedEffect::StoreSignal { dst, value } => {
                        let dw = self.widths[*dst];
                        sig_local.insert(*dst, pv(value));
                        // store_set=Some restricts comb stores to the observable cone
                        // leaves (outputs); the value still flows to consumers via
                        // sig_local, so internal comb signals need no state write.
                        let stored = store_comb && store_set.map_or(true, |s| s.contains(dst));
                        if stored {
                            writeln!(code, "    {} = ({})({}{});", self.sig_lval(*dst), stype(dw), pv(value), mask_suffix(dw)).ok();
                        }
                    }
                    PackedEffect::CaptureReg { dst, value, reset } => {
                        let dw = self.widths[*dst];
                        if capture_to_temp {
                            let nv = match reset {
                                Some(r) => format!("({}) ? {} : {}", self.reset_asserted(r), self.reset_val(r, dw), pv(value)),
                                None => pv(value),
                            };
                            writeln!(code, "    next_{} = (({}){}){};", dst, ctype(dw), nv, mask_suffix(dw)).ok();
                        } else {
                            // async-reset comb: conditional immediate store; a later
                            // read of this register loads the stored value, so drop
                            // any alias.
                            let r = reset.as_ref().ok_or_else(|| aot_err("reset-less capture in comb"))?;
                            writeln!(
                                code,
                                "    {} = ({})((({}) ? {} : {}){});",
                                self.sig_lval(*dst), stype(dw), self.reset_asserted(r),
                                self.reset_val(r, dw), self.sig_read(*dst), mask_suffix(dw)
                            ).ok();
                            sig_local.remove(dst);
                        }
                    }
                    PackedEffect::MemoryWrite { memory, enable, addr, data } => {
                        let depth = self.mem_depth[*memory];
                        if depth == 0 {
                            continue;
                        }
                        let mt = stype(self.mem_width[*memory]);
                        let eb = self.layout.mem_entry[*memory];
                        // Clock-gate the write when in tick_clocked.
                        let gate = match mem_gate.and_then(|m| m.get(memory)) {
                            Some(&bit) => format!("((mask >> {bit}) & 1ull) && "),
                            None => String::new(),
                        };
                        writeln!(
                            code,
                            "    if ({gate}({} & 1) && (u128){} < {}) *({}*)(st+{}+(size_t)({})*{}) = (({})({}{}));",
                            pv(enable), pv(addr), depth, mt, self.layout.mem_base[*memory], pv(addr), eb, mt, pv(data),
                            mask_suffix(self.mem_width[*memory])
                        ).ok();
                    }
                }
            }
        }
        Ok(())
    }

    /// Build the control-flow plan for a block: which single-use mux-arm cones are
    /// sunk into which branch, in topological order.
    fn cflow_analyze<'b>(&self, block: &'b PackedBlock) -> Cflow<'b> {
        let mut flat: Vec<&'b PackedInstr> = Vec::new();
        for p in &block.packets {
            for i in &p.instrs {
                flat.push(i);
            }
        }
        let mut def_idx: HashMap<usize, usize> = HashMap::new();
        for (idx, i) in flat.iter().enumerate() {
            def_idx.insert(i.dst.0, idx);
        }
        let mut use_count: HashMap<usize, usize> = HashMap::new();
        for i in &flat {
            for o in value_operands(&i.kind) {
                *use_count.entry(o).or_default() += 1;
            }
        }
        for p in &block.packets {
            for e in &p.effects {
                for o in effect_value_operands(e) {
                    *use_count.entry(o).or_default() += 1;
                }
            }
        }
        let mut sunk: HashSet<usize> = HashSet::new();
        let mut then_mem: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut else_mem: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut branched: HashSet<usize> = HashSet::new();
        // process muxes in def order (inner before outer) so nested arms claim first
        for i in &flat {
            let PackedInstrKind::Mux { then_value, else_value, .. } = &i.kind else { continue };
            let mut tm = Vec::new();
            let mut em = Vec::new();
            // only sink an arm whose root is real compute (not a bare copy/leaf) —
            // branching a trivial arm only adds a mispredictable test.
            if arm_worth_sinking(then_value.0, &def_idx, &use_count, &sunk, &flat) {
                claim(then_value.0, &mut tm, &mut sunk, &flat, &def_idx, &use_count);
            }
            if arm_worth_sinking(else_value.0, &def_idx, &use_count, &sunk, &flat) {
                claim(else_value.0, &mut em, &mut sunk, &flat, &def_idx, &use_count);
            }
            if !tm.is_empty() || !em.is_empty() {
                branched.insert(i.dst.0);
            }
            then_mem.insert(i.dst.0, tm);
            else_mem.insert(i.dst.0, em);
        }
        Cflow { flat, def_idx, sunk, then_mem, else_mem, branched }
    }

    /// Emit a mux as `T dst; if (cond) { <then-cone>; dst = then; } else { ... }`,
    /// recursively emitting any nested branched muxes in the arm cones.
    fn emit_branch(
        &self,
        code: &mut String,
        instr: &PackedInstr,
        prefix: &str,
        vw: &mut HashMap<usize, u32>,
        sig_local: &HashMap<usize, String>,
        fuse: bool,
        cf: &Cflow,
    ) {
        let PackedInstrKind::Mux { cond, then_value, else_value } = &instr.kind else { return };
        let w = instr.ty.width;
        let t = ctype(w);
        let v = instr.dst.0;
        writeln!(code, "    {t} {prefix}{v};").ok();
        writeln!(code, "    if ({prefix}{} & 1) {{", cond.0).ok();
        for &m in cf.then_mem.get(&v).map(|x| x.as_slice()).unwrap_or(&[]) {
            self.emit_member(code, m, prefix, vw, sig_local, fuse, cf);
        }
        writeln!(code, "    {prefix}{v} = ({t}){prefix}{}{};", then_value.0, mask_suffix(w)).ok();
        writeln!(code, "    }} else {{").ok();
        for &m in cf.else_mem.get(&v).map(|x| x.as_slice()).unwrap_or(&[]) {
            self.emit_member(code, m, prefix, vw, sig_local, fuse, cf);
        }
        writeln!(code, "    {prefix}{v} = ({t}){prefix}{}{};", else_value.0, mask_suffix(w)).ok();
        writeln!(code, "    }}").ok();
        vw.insert(v, w);
    }

    /// Emit one sunk member: a nested branch if it is a branched mux, else flat.
    fn emit_member(
        &self,
        code: &mut String,
        m: usize,
        prefix: &str,
        vw: &mut HashMap<usize, u32>,
        sig_local: &HashMap<usize, String>,
        fuse: bool,
        cf: &Cflow,
    ) {
        let instr = cf.flat[cf.def_idx[&m]];
        if cf.branched.contains(&m) {
            self.emit_branch(code, instr, prefix, vw, sig_local, fuse, cf);
        } else {
            let w = instr.ty.width;
            let rhs = self.instr_rhs(&instr.kind, w, prefix, vw, sig_local, fuse);
            writeln!(code, "    {} {}{} = ({}){};", ctype(w), prefix, m, rhs, mask_suffix(w)).ok();
            vw.insert(m, w);
        }
    }

    fn reset_asserted(&self, r: &PackedReset) -> String {
        let bit = format!("({} & 1)", self.sig_read(r.signal));
        match r.polarity {
            ResetPolarity::ActiveHigh => format!("({} != 0)", bit),
            ResetPolarity::ActiveLow => format!("({} == 0)", bit),
        }
    }

    fn reset_val(&self, r: &PackedReset, dw: u32) -> String {
        fmt_const(lit_u128(&r.value) & mask128(dw), dw)
    }
}

/// Output ports whose sole comb driver is a trivial copy of a register/input
/// (see jit.rs); these are refreshed at commit so reading them needs no settle.
fn trivial_output_aliases(machine: &PackedMachineProgram) -> Vec<(usize, usize)> {
    let source = &machine.source;
    let kind = |i: usize| source.signals.get(i).map(|s| s.kind);
    let stable = |i: usize| matches!(kind(i), Some(PackedSignalKind::Reg) | Some(PackedSignalKind::Input));
    let mut store_count: HashMap<usize, usize> = HashMap::new();
    let mut alias: HashMap<usize, usize> = HashMap::new();
    for block in [&machine.streams.async_reset_comb, &machine.streams.comb] {
        let mut copy_src: HashMap<usize, usize> = HashMap::new();
        for packet in &block.packets {
            for instr in &packet.instrs {
                if let PackedInstrKind::Signal(s) = instr.kind {
                    copy_src.insert(instr.dst.0, s);
                }
            }
        }
        for packet in &block.packets {
            for effect in &packet.effects {
                if let PackedEffect::StoreSignal { dst, value } = effect {
                    *store_count.entry(*dst).or_default() += 1;
                    if kind(*dst) == Some(PackedSignalKind::Output) {
                        if let Some(&s) = copy_src.get(&value.0) {
                            if stable(s) {
                                alias.insert(*dst, s);
                            }
                        }
                    }
                }
            }
        }
    }
    alias.into_iter().filter(|(d, _)| store_count.get(d) == Some(&1)).collect()
}

/// Generate the full C source for `machine`.
pub fn generate_c(machine: &PackedMachineProgram) -> Result<String, ErrorReport> {
    let source = &machine.source;
    for s in &source.signals {
        if s.layout.width > 128 {
            return Err(aot_err(format!("AOT supports signals ≤128 bits; `{}` is {}", s.name, s.layout.width)));
        }
    }
    let widths: Vec<u32> = source.signals.iter().map(|s| s.layout.width).collect();
    let mut mem_depth = Vec::new();
    let mut mem_width = Vec::new();
    for m in &source.memories {
        if m.data_layout.width > 128 {
            return Err(aot_err(format!("AOT supports memory ≤128 bits; `{}`", m.name)));
        }
        mem_depth.push(m.depth);
        mem_width.push(m.data_layout.width);
    }
    let layout = build_layout(machine);
    let g = Gen { widths: &widths, layout: &layout, mem_depth, mem_width };

    let mut code = String::new();
    code.push_str(
        "#include <stdint.h>\n#include <stddef.h>\n\
         typedef uint64_t u64; typedef unsigned __int128 u128; typedef __int128 i128;\n\
         static inline u64 sext64(u64 x, unsigned f){ unsigned s=64-f; return (u64)(((int64_t)(x<<s))>>s); }\n\
         static inline u128 sext128(u128 x, unsigned f){ unsigned s=128-f; return (u128)(((i128)(x<<s))>>s); }\n\n",
    );

    // tick_many(state, n): the comb settle, register capture, memory writes and
    // commit all FUSED into one C scope sharing a `sig_local` map, so combinational
    // values feed register next-states directly from C locals (registers) instead
    // of round-tripping the state buffer. Comb stores are skipped in the hot tick
    // (store_comb=false) — observability is recomputed lazily by settle(). The
    // post-commit settle is deferred (lazy), matching the JIT.
    // AOT_NOFUSE=1 disables the fusion (comb round-trips state) to A/B the win.
    let fuse = std::env::var("AOT_NOFUSE").is_err();
    let sc = !fuse; // when not fusing, comb must store to state for capture to read
    // AOT_CFLOW=1 enables control-flow recovery (mux-arm sinking) on the mux-heavy
    // comb + tick_next blocks — eval-taken instead of eval-all. Opt-in (latency).
    let cflow = std::env::var("AOT_CFLOW").is_ok();
    code.push_str("void tick_many(unsigned char* st, long n){\n for(long _c=0;_c<n;_c++){\n");
    let mut sl: HashMap<usize, String> = HashMap::new();
    g.emit_block(&mut code, &machine.streams.async_reset_comb, "a", &mut sl, sc, false, fuse, None, false, None)?;
    g.emit_block(&mut code, &machine.streams.comb, "c", &mut sl, sc, false, fuse, None, cflow, None)?;
    let mut captures: Vec<(usize, u32)> = Vec::new();
    for packet in &machine.streams.tick_next.packets {
        for effect in &packet.effects {
            if let PackedEffect::CaptureReg { dst, .. } = effect {
                captures.push((*dst, widths[*dst]));
            }
        }
    }
    for (dst, w) in &captures {
        writeln!(code, "  {} next_{} = 0;", ctype(*w), dst).ok();
    }
    g.emit_block(&mut code, &machine.streams.tick_next, "n", &mut sl, true, true, fuse, None, cflow, None)?;
    g.emit_block(&mut code, &machine.streams.tick_commit, "m", &mut sl, true, false, fuse, None, false, None)?;
    for (dst, w) in &captures {
        writeln!(code, "  {} = ({})(next_{}{});", g.sig_lval(*dst), stype(*w), dst, mask_suffix(*w)).ok();
    }
    for (dst, src) in trivial_output_aliases(machine) {
        let wd = widths[dst];
        writeln!(code, "  {} = ({})({}{});", g.sig_lval(dst), stype(wd), g.sig_read(src), mask_suffix(wd)).ok();
    }
    code.push_str(" }\n}\n");

    // settle(state): the deferred post-commit combinational pass — must write comb
    // to state (store_comb=true) so observed signals are correct.
    code.push_str("void settle(unsigned char* st){\n");
    let mut sl2: HashMap<usize, String> = HashMap::new();
    g.emit_block(&mut code, &machine.streams.async_reset_comb, "a", &mut sl2, true, false, fuse, None, false, None)?;
    g.emit_block(&mut code, &machine.streams.comb, "c", &mut sl2, true, false, fuse, None, false, None)?;
    code.push_str("}\n");

    // settle_obs(state): same comb compute, but stores ONLY output-port signals to
    // state — the testbench reads ports, not internal comb. Internal values still
    // flow through C locals, so this is bit-exact for observing any output, at a
    // fraction of the store traffic (observability slicing the settle). get_signal
    // of a non-output comb signal falls back to the full settle.
    let outputs: std::collections::HashSet<usize> = source.signals.iter().enumerate()
        .filter(|(_, s)| s.kind == PackedSignalKind::Output)
        .map(|(i, _)| i).collect();
    code.push_str("void settle_obs(unsigned char* st){\n");
    let mut sl3: HashMap<usize, String> = HashMap::new();
    g.emit_block(&mut code, &machine.streams.async_reset_comb, "a", &mut sl3, true, false, fuse, Some(&outputs), false, None)?;
    g.emit_block(&mut code, &machine.streams.comb, "c", &mut sl3, true, false, fuse, Some(&outputs), false, None)?;
    code.push_str("}\n");

    // tick_clocked(state, mask): one cycle gating register commits by clock — a
    // clocked register commits only if its clock's bit in `mask` is set, else
    // holds (the compiled-engine multi-clock primitive; mirrors the JIT). Memory
    // writes are not yet clock-gated (register domains only).
    let clock_bit = clock_bit_assignment(machine);
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
    code.push_str("void tick_clocked(unsigned char* st, unsigned long long mask){\n");
    let mut slc: HashMap<usize, String> = HashMap::new();
    g.emit_block(&mut code, &machine.streams.async_reset_comb, "a", &mut slc, sc, false, fuse, None, false, None)?;
    g.emit_block(&mut code, &machine.streams.comb, "c", &mut slc, sc, false, fuse, None, cflow, None)?;
    for (dst, w) in &captures {
        writeln!(code, "  {} next_{} = 0;", ctype(*w), dst).ok();
    }
    g.emit_block(&mut code, &machine.streams.tick_next, "n", &mut slc, true, true, fuse, None, cflow, None)?;
    g.emit_block(&mut code, &machine.streams.tick_commit, "m", &mut slc, true, false, fuse, None, false, Some(&mem_clock_bit))?;
    for (dst, w) in &captures {
        let store = format!("{} = ({})(next_{}{});", g.sig_lval(*dst), stype(*w), dst, mask_suffix(*w));
        match reg_clock_bit.get(dst) {
            Some(&bit) => writeln!(code, "  if ((mask >> {bit}) & 1ull) {{ {store} }}").ok(),
            None => writeln!(code, "  {store}").ok(),
        };
    }
    for (dst, src) in trivial_output_aliases(machine) {
        let wd = widths[dst];
        writeln!(code, "  {} = ({})({}{});", g.sig_lval(dst), stype(wd), g.sig_read(src), mask_suffix(wd)).ok();
    }
    code.push_str("}\n");
    Ok(code)
}

/// Deterministic clock-signal-index → mask-bit assignment (sorted, ≤64 clocks),
/// shared by codegen and the `AotSimulator` wrapper so the bits agree.
fn clock_bit_assignment(machine: &PackedMachineProgram) -> HashMap<usize, u32> {
    let mut ids: Vec<usize> = machine
        .source
        .reg_clocks
        .values()
        .chain(machine.source.mem_clocks.values())
        .copied()
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids.iter().take(64).enumerate().map(|(i, &c)| (c, i as u32)).collect()
}

/// An AOT-compiled single-lane simulator: same state layout and API as the JIT.
pub struct AotSimulator {
    _lib: libloading::Library,
    tick_many_fn: extern "C" fn(*mut u8, i64),
    /// `tick_clocked(state, mask)` — one cycle gating register commits by clock.
    tick_clocked_fn: extern "C" fn(*mut u8, u64),
    /// Clock signal index → its bit in the `tick_clocked` mask.
    clock_bit: HashMap<usize, u32>,
    settle_fn: extern "C" fn(*mut u8),
    /// Cheap settle that stores only output-port signals (observability-sliced).
    settle_obs_fn: extern "C" fn(*mut u8),
    /// Comb (incl. internal signals) needs a full settle before observation.
    full_dirty: bool,
    /// Output ports need at least the cheap obs-settle before observation.
    obs_dirty: bool,
    needs_settle: Vec<bool>,
    /// Signal is an output port → covered by the cheap `settle_obs`.
    is_output: Vec<bool>,
    /// 16-byte-aligned packed state (backed by u128 so __int128 accesses are
    /// aligned); byte offsets come from the width-packed `off` table.
    state: Vec<u128>,
    off: Vec<usize>,
    widths: Vec<u32>,
    num_signals: usize,
}

impl AotSimulator {
    /// Generate C, compile it with `$CC` (default `clang`) at `-O3`, load it.
    pub fn compile(machine: &PackedMachineProgram) -> Result<Self, ErrorReport> {
        let c = generate_c(machine)?;
        let num_signals = machine.source.signals.len();
        let widths: Vec<u32> = machine.source.signals.iter().map(|s| s.layout.width).collect();
        let layout = build_layout(machine);

        // unique-ish temp paths from the program's structure (no Date/random in tests)
        let stamp = format!("{:x}", fxhash(c.as_bytes()));
        let dir = std::env::temp_dir();
        let cpath = dir.join(format!("rrtl_aot_{stamp}.c"));
        let libname = format!("librrtl_aot_{stamp}.{}", dylib_ext());
        let libpath = dir.join(&libname);
        std::fs::write(&cpath, &c).map_err(|e| aot_err(format!("write C: {e}")))?;

        let cc = std::env::var("CC").unwrap_or_else(|_| "clang".into());
        let out = std::process::Command::new(&cc)
            .args(["-O3", "-shared", "-fPIC", "-o"])
            .arg(&libpath)
            .arg(&cpath)
            .output()
            .map_err(|e| aot_err(format!("spawn {cc}: {e}")))?;
        if !out.status.success() {
            return Err(aot_err(format!(
                "{cc} -O3 failed: {}",
                String::from_utf8_lossy(&out.stderr)
            )));
        }
        let lib = unsafe { libloading::Library::new(&libpath) }
            .map_err(|e| aot_err(format!("dlopen: {e}")))?;
        let tick_many_fn = unsafe {
            let sym: libloading::Symbol<extern "C" fn(*mut u8, i64)> =
                lib.get(b"tick_many").map_err(|e| aot_err(format!("sym tick_many: {e}")))?;
            *sym
        };
        let tick_clocked_fn = unsafe {
            let sym: libloading::Symbol<extern "C" fn(*mut u8, u64)> =
                lib.get(b"tick_clocked").map_err(|e| aot_err(format!("sym tick_clocked: {e}")))?;
            *sym
        };
        let settle_fn = unsafe {
            let sym: libloading::Symbol<extern "C" fn(*mut u8)> =
                lib.get(b"settle").map_err(|e| aot_err(format!("sym settle: {e}")))?;
            *sym
        };
        // AOT_NOOBS=1 points the obs-settle at the full settle (A/B the obs-slice win).
        let obs_sym: &[u8] = if std::env::var("AOT_NOOBS").is_ok() { b"settle" } else { b"settle_obs" };
        let settle_obs_fn = unsafe {
            let sym: libloading::Symbol<extern "C" fn(*mut u8)> =
                lib.get(obs_sym).map_err(|e| aot_err(format!("sym settle_obs: {e}")))?;
            *sym
        };

        // Combinationally-driven signals need a settle to observe; trivial
        // register/input-aliased outputs are refreshed at commit, so they don't.
        let mut needs_settle = vec![false; num_signals];
        for block in [&machine.streams.async_reset_comb, &machine.streams.comb] {
            for packet in &block.packets {
                for effect in &packet.effects {
                    match effect {
                        PackedEffect::StoreSignal { dst, .. } | PackedEffect::CaptureReg { dst, .. } => {
                            if *dst < num_signals {
                                needs_settle[*dst] = true;
                            }
                        }
                        PackedEffect::MemoryWrite { .. } => {}
                    }
                }
            }
        }
        for (dst, _) in trivial_output_aliases(machine) {
            if dst < num_signals {
                needs_settle[dst] = false;
            }
        }
        let is_output: Vec<bool> = machine.source.signals.iter()
            .map(|s| s.kind == PackedSignalKind::Output).collect();

        Ok(Self {
            _lib: lib,
            tick_many_fn,
            tick_clocked_fn,
            clock_bit: clock_bit_assignment(machine),
            settle_fn,
            settle_obs_fn,
            full_dirty: false,
            obs_dirty: false,
            needs_settle,
            is_output,
            state: vec![0u128; layout.total.div_ceil(16).max(1)],
            off: layout.off,
            widths,
            num_signals,
        })
    }

    fn state_ptr(&mut self) -> *mut u8 {
        self.state.as_mut_ptr() as *mut u8
    }

    pub fn set_signal(&mut self, index: usize, value: u64) {
        self.set_signal_u128(index, value as u128);
    }
    pub fn get_signal(&mut self, index: usize) -> u64 {
        self.get_signal_u128(index) as u64
    }
    pub fn set_signal_u128(&mut self, index: usize, value: u128) {
        let v = (value & mask128(self.widths[index])).to_le_bytes();
        let (o, n) = (self.off[index], store_bytes(self.widths[index]));
        let p = self.state_ptr();
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), p.add(o), n) };
    }
    pub fn get_signal_u128(&mut self, index: usize) -> u128 {
        if self.needs_settle[index] {
            if self.is_output[index] {
                // an output port → the cheap obs-settle suffices (it stores ports)
                if self.obs_dirty {
                    (self.settle_obs_fn)(self.state_ptr());
                    self.obs_dirty = false;
                }
            } else if self.full_dirty {
                // an internal comb signal → needs the full settle
                (self.settle_fn)(self.state_ptr());
                self.full_dirty = false;
                self.obs_dirty = false;
            }
        }
        let (o, n) = (self.off[index], store_bytes(self.widths[index]));
        let mut b = [0u8; 16];
        let p = self.state.as_ptr() as *const u8;
        unsafe { std::ptr::copy_nonoverlapping(p.add(o), b.as_mut_ptr(), n) };
        u128::from_le_bytes(b) & mask128(self.widths[index])
    }

    pub fn tick(&mut self) {
        let p = self.state_ptr();
        (self.tick_many_fn)(p, 1);
        self.full_dirty = true;
        self.obs_dirty = true;
    }
    pub fn tick_many(&mut self, n: usize) {
        let p = self.state_ptr();
        (self.tick_many_fn)(p, n as i64);
        self.full_dirty = true;
        self.obs_dirty = true;
    }

    /// Advance one cycle, gating register commits by clock (multi-clock).
    /// `active_clocks` lists the clock SIGNAL INDICES with a rising edge this
    /// step; registers on other clocks hold. Single-step (the active set varies
    /// per cycle). With every clock listed it is identical to one `tick`.
    /// Unclocked registers always commit. (Memory writes are not yet gated.)
    pub fn tick_clocked(&mut self, active_clocks: &[usize]) {
        let mut mask = 0u64;
        for &clk in active_clocks {
            if let Some(&bit) = self.clock_bit.get(&clk) {
                mask |= 1u64 << bit;
            }
        }
        let p = self.state_ptr();
        (self.tick_clocked_fn)(p, mask);
        self.full_dirty = true;
        self.obs_dirty = true;
    }

    /// Total packed state size in bytes.
    pub fn state_bytes(&self) -> usize {
        self.state.len() * 16
    }

    pub fn signal_count(&self) -> usize {
        self.num_signals
    }
}

fn dylib_ext() -> &'static str {
    if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    }
}

/// Tiny FNV-1a so temp filenames are stable per generated program (no rng).
fn fxhash(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::AotSimulator;
    use crate::{lower_to_machine_program, lower_to_packed_program};
    use rrtl_core::{compile, Design, Signal, Simulator};
    use rrtl_ir::{lit_u, uint};

    // The AOT (clang -O3) clock-gated tick must match the gold oracle on an
    // independent multi-clock design — the compiled C mirror of the JIT select.
    #[test]
    fn aot_tick_clocked_matches_oracle() {
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

        let mut aot = AotSimulator::compile(&machine).unwrap();
        let mut gold = Simulator::new(&design, "TwoClock").unwrap();
        for step in 0..12 {
            let edge = step % 3 == 0; // clkB every 3rd step
            let active_i: Vec<usize> = if edge { vec![clka_i, clkb_i] } else { vec![clka_i] };
            let active_s: Vec<Signal> = if edge { vec![clka, clkb] } else { vec![clka] };
            aot.tick_clocked(&active_i);
            gold.tick_clocked(&active_s).unwrap();
            assert_eq!(aot.get_signal(a_i) as u128, gold.get(a), "a@{step}");
            assert_eq!(aot.get_signal(b_i) as u128, gold.get(b), "b@{step}");
        }
        assert!(aot.get_signal(b_i) < aot.get_signal(a_i));
    }

    // A memory written on a slow clock must capture only on its edges (the AOT's
    // clock-gated memory write) vs the gold oracle.
    #[test]
    fn aot_tick_clocked_gates_memory_writes() {
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

        let mut aot = AotSimulator::compile(&machine).unwrap();
        let mut gold = Simulator::new(&design, "MemClk").unwrap();
        for step in 0..12 {
            let edge = step % 3 == 0;
            let active_i: Vec<usize> = if edge { vec![clka_i, clkb_i] } else { vec![clka_i] };
            let active_s: Vec<Signal> = if edge { vec![clka, clkb] } else { vec![clka] };
            aot.tick_clocked(&active_i);
            gold.tick_clocked(&active_s).unwrap();
            assert_eq!(aot.get_signal(out_i) as u128, gold.get(out), "out@{step}");
        }
    }
}
