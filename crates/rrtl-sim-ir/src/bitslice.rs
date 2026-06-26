//! Bit-sliced multi-bit batch engine. A W-bit signal across L lanes is stored as
//! W **bit-planes** — plane `p` holds lane-bit `p` of every lane, packed
//! `ceil(L/64)` u64 words (one lane-bit per bit). So *every* width runs at the
//! full 64-lane-per-word density (vs the SIMD vector engine, which must pick one
//! width-class by the design's max width and runs even 1-bit control signals at
//! that poor density). Bitwise/mux ops are per-plane; add/sub are ripple-carry
//! across planes; eq/ne are xor-planes + OR-reduce; slice/concat/ext are plane
//! reindexing (free). The 1-bit case is exactly the [`crate::bitparallel`] engine.
//!
//! Scope: And/Or/Xor/Not/Mux/Eq/Ne/Add/Sub + Slice/Concat/Zext/Sext/Trunc/Cast +
//! Signal/Lit + sync/async reset; ≤128-bit; no memory, no multiply/Lt yet
//! (rejected at `new`, so the caller falls back to another engine).

use std::collections::HashMap;

use rrtl_ir::{Diagnostic, ErrorReport, ResetPolarity};

use crate::{
    PackedBlock, PackedEffect, PackedInstrKind, PackedMachineProgram, PackedReset, PackedValueId,
};

fn bs_err(msg: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![Diagnostic::new("E_BITSLICE", msg)])
}

pub struct BitSliceSimulator {
    machine: PackedMachineProgram,
    lanes: usize,
    words: usize,           // u64 words per plane = ceil(lanes/64)
    widths: Vec<u32>,       // per signal
    sig_base: Vec<usize>,   // first plane index of each signal (cumulative widths)
    state: Vec<u64>,        // (Σ widths) * words ; signal s plane p word k at (sig_base[s]+p)*words+k
    num_signals: usize,
}

impl BitSliceSimulator {
    pub fn new(machine: &PackedMachineProgram, lanes: usize) -> Result<Self, ErrorReport> {
        if !machine.source.memories.is_empty() {
            return Err(bs_err("bit-slice does not support memories"));
        }
        let widths: Vec<u32> = machine.source.signals.iter().map(|s| s.layout.width).collect();
        for &w in &widths {
            if w > 128 {
                return Err(bs_err("bit-slice scope is ≤128-bit signals"));
            }
        }
        // Validate the op set across all streams.
        for blk in [
            &machine.streams.async_reset_comb,
            &machine.streams.comb,
            &machine.streams.tick_next,
            &machine.streams.tick_commit,
        ] {
            for p in &blk.packets {
                for i in &p.instrs {
                    use PackedInstrKind::*;
                    match &i.kind {
                        And(..) | Or(..) | Xor(..) | Not(..) | Mux { .. } | Eq(..) | Ne(..)
                        | Add(..) | Sub(..) | Slice { .. } | Concat(_) | Zext(_) | Sext(_)
                        | Trunc(_) | Cast(_) | Signal(_) | Lit(_) => {}
                        other => return Err(bs_err(format!("op {other:?} not in bit-slice scope"))),
                    }
                }
                for e in &p.effects {
                    if let PackedEffect::MemoryWrite { .. } = e {
                        return Err(bs_err("bit-slice does not support memory writes"));
                    }
                }
            }
        }
        let mut sig_base = Vec::with_capacity(widths.len());
        let mut cur = 0usize;
        for &w in &widths {
            sig_base.push(cur);
            cur += w as usize;
        }
        let words = lanes.div_ceil(64).max(1);
        Ok(Self {
            machine: machine.clone(),
            lanes,
            words,
            widths,
            sig_base,
            state: vec![0u64; cur * words],
            num_signals: machine.source.signals.len(),
        })
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }
    pub fn signal_count(&self) -> usize {
        self.num_signals
    }

    pub fn set_signal(&mut self, idx: usize, lane: usize, value: u128) {
        let (w, b) = (lane / 64, lane % 64);
        for p in 0..self.widths[idx] as usize {
            let slot = (self.sig_base[idx] + p) * self.words + w;
            if (value >> p) & 1 == 1 {
                self.state[slot] |= 1u64 << b;
            } else {
                self.state[slot] &= !(1u64 << b);
            }
        }
    }
    pub fn get_signal(&self, idx: usize, lane: usize) -> u128 {
        let (w, b) = (lane / 64, lane % 64);
        let mut v = 0u128;
        for p in 0..self.widths[idx] as usize {
            let slot = (self.sig_base[idx] + p) * self.words + w;
            if (self.state[slot] >> b) & 1 == 1 {
                v |= 1u128 << p;
            }
        }
        v
    }

    /// Evaluate a block into a fresh plane workspace; returns (workspace,
    /// per-value plane base, per-value width) so effects can read the results.
    fn eval(&self, block: &PackedBlock) -> (Vec<u64>, HashMap<PackedValueId, (usize, u32)>) {
        // Value plane layout.
        let mut layout: HashMap<PackedValueId, (usize, u32)> = HashMap::new();
        let mut total = 0usize;
        for pkt in &block.packets {
            for i in &pkt.instrs {
                layout.insert(i.dst, (total, i.ty.width));
                total += i.ty.width as usize;
            }
        }
        let mut ws = vec![0u64; total * self.words];
        let words = self.words;
        // operand plane-word reader: a value-id's plane p, word k.
        // (resolves to the workspace; Signal leaves are pulled from state inside
        // the Signal arm.)
        for pkt in &block.packets {
            for instr in &pkt.instrs {
                let (dbase, dw) = layout[&instr.dst];
                use PackedInstrKind::*;
                // helper closures capture ws immutably for reads; we stage all
                // writes through a small temp to avoid aliasing the &mut.
                let rd = |ws: &[u64], v: PackedValueId, p: usize, k: usize| -> u64 {
                    let (b, w) = layout[&v];
                    if p < w as usize {
                        ws[(b + p) * words + k]
                    } else {
                        0
                    }
                };
                match &instr.kind {
                    Signal(s) => {
                        let sb = self.sig_base[*s];
                        let sw = self.widths[*s] as usize;
                        for p in 0..dw as usize {
                            for k in 0..words {
                                ws[(dbase + p) * words + k] =
                                    if p < sw { self.state[(sb + p) * words + k] } else { 0 };
                            }
                        }
                    }
                    Lit(wd) => {
                        for p in 0..dw as usize {
                            let bit = (wd.get(p / 32).copied().unwrap_or(0) >> (p % 32)) & 1;
                            let fill = if bit == 1 { u64::MAX } else { 0 };
                            for k in 0..words {
                                ws[(dbase + p) * words + k] = fill;
                            }
                        }
                    }
                    Not(a) => {
                        for p in 0..dw as usize {
                            for k in 0..words {
                                ws[(dbase + p) * words + k] = !rd(&ws, *a, p, k);
                            }
                        }
                    }
                    And(a, b) => self.binop(&mut ws, &layout, dbase, dw, *a, *b, |x, y| x & y),
                    Or(a, b) => self.binop(&mut ws, &layout, dbase, dw, *a, *b, |x, y| x | y),
                    Xor(a, b) => self.binop(&mut ws, &layout, dbase, dw, *a, *b, |x, y| x ^ y),
                    Mux { cond, then_value, else_value } => {
                        for k in 0..words {
                            let c = rd(&ws, *cond, 0, k);
                            for p in 0..dw as usize {
                                let t = rd(&ws, *then_value, p, k);
                                let e = rd(&ws, *else_value, p, k);
                                ws[(dbase + p) * words + k] = (c & t) | (!c & e);
                            }
                        }
                    }
                    Eq(a, b) | Ne(a, b) => {
                        let wmax = layout[a].1.max(layout[b].1) as usize;
                        for k in 0..words {
                            let mut diff = 0u64;
                            for p in 0..wmax {
                                diff |= rd(&ws, *a, p, k) ^ rd(&ws, *b, p, k);
                            }
                            let eq = !diff;
                            ws[dbase * words + k] = if matches!(instr.kind, Eq(..)) { eq } else { diff };
                            for p in 1..dw as usize {
                                ws[(dbase + p) * words + k] = 0;
                            }
                        }
                    }
                    Add(a, b) | Sub(a, b) => {
                        let sub = matches!(instr.kind, Sub(..));
                        for k in 0..words {
                            let mut carry = if sub { u64::MAX } else { 0 }; // sub: a + ~b + 1
                            for p in 0..dw as usize {
                                let av = rd(&ws, *a, p, k);
                                let bv = if sub { !rd(&ws, *b, p, k) } else { rd(&ws, *b, p, k) };
                                let axb = av ^ bv;
                                ws[(dbase + p) * words + k] = axb ^ carry;
                                carry = (av & bv) | (carry & axb);
                            }
                        }
                    }
                    Slice { value, lsb } => {
                        let lsb = *lsb as usize;
                        for p in 0..dw as usize {
                            for k in 0..words {
                                ws[(dbase + p) * words + k] = rd(&ws, *value, p + lsb, k);
                            }
                        }
                    }
                    Zext(a) | Trunc(a) | Cast(a) => {
                        for p in 0..dw as usize {
                            for k in 0..words {
                                ws[(dbase + p) * words + k] = rd(&ws, *a, p, k);
                            }
                        }
                    }
                    Sext(a) => {
                        let sw = layout[a].1 as usize;
                        for p in 0..dw as usize {
                            let src_p = if p < sw { p } else { sw - 1 }; // replicate sign plane
                            for k in 0..words {
                                ws[(dbase + p) * words + k] = rd(&ws, *a, src_p, k);
                            }
                        }
                    }
                    Concat(parts) => {
                        // MSB-first: parts[0] is the high part. Place low parts first.
                        let mut off = 0usize;
                        for part in parts.iter().rev() {
                            let pw = layout[part].1 as usize;
                            for p in 0..pw {
                                for k in 0..words {
                                    ws[(dbase + off + p) * words + k] = rd(&ws, *part, p, k);
                                }
                            }
                            off += pw;
                        }
                    }
                    _ => unreachable!("validated in new()"),
                }
            }
        }
        (ws, layout)
    }

    #[inline]
    fn binop(
        &self,
        ws: &mut [u64],
        layout: &HashMap<PackedValueId, (usize, u32)>,
        dbase: usize,
        dw: u32,
        a: PackedValueId,
        b: PackedValueId,
        f: impl Fn(u64, u64) -> u64,
    ) {
        let words = self.words;
        let (ab, aw) = layout[&a];
        let (bb, bw) = layout[&b];
        for p in 0..dw as usize {
            for k in 0..words {
                let av = if p < aw as usize { ws[(ab + p) * words + k] } else { 0 };
                let bv = if p < bw as usize { ws[(bb + p) * words + k] } else { 0 };
                ws[(dbase + p) * words + k] = f(av, bv);
            }
        }
    }

    fn reset_asserted(&self, reset: &PackedReset, out: &mut [u64]) {
        let base = self.sig_base[reset.signal];
        for k in 0..self.words {
            let bit = self.state[base * self.words + k]; // reset is 1-bit → plane 0
            out[k] = match reset.polarity {
                ResetPolarity::ActiveLow => !bit,
                _ => bit,
            };
        }
    }

    fn settle(&mut self) {
        let blocks = [
            self.machine.streams.async_reset_comb.clone(),
            self.machine.streams.comb.clone(),
        ];
        for block in &blocks {
            let (ws, layout) = self.eval(block);
            for pkt in &block.packets {
                for eff in &pkt.effects {
                    match eff {
                        PackedEffect::StoreSignal { dst, value } => self.store(*dst, *value, &ws, &layout),
                        PackedEffect::CaptureReg { dst, value, reset } => {
                            // async-reset-comb: immediate conditional store.
                            if let Some(r) = reset {
                                self.store_reset(*dst, *value, r, &ws, &layout);
                            }
                        }
                        PackedEffect::MemoryWrite { .. } => {}
                    }
                }
            }
        }
    }

    fn store(&mut self, dst: usize, value: PackedValueId, ws: &[u64], layout: &HashMap<PackedValueId, (usize, u32)>) {
        let (vb, vw) = layout[&value];
        let db = self.sig_base[dst];
        let dw = self.widths[dst] as usize;
        for p in 0..dw {
            for k in 0..self.words {
                self.state[(db + p) * self.words + k] =
                    if p < vw as usize { ws[(vb + p) * self.words + k] } else { 0 };
            }
        }
    }

    fn store_reset(&mut self, dst: usize, value: PackedValueId, r: &PackedReset, ws: &[u64], layout: &HashMap<PackedValueId, (usize, u32)>) {
        let mut asserted = vec![0u64; self.words];
        self.reset_asserted(r, &mut asserted);
        let (vb, vw) = layout[&value];
        let db = self.sig_base[dst];
        let dw = self.widths[dst] as usize;
        for p in 0..dw {
            let rbit = (r.value.get(p / 32).copied().unwrap_or(0) >> (p % 32)) & 1;
            let rfill = if rbit == 1 { u64::MAX } else { 0 };
            for k in 0..self.words {
                let nv = if p < vw as usize { ws[(vb + p) * self.words + k] } else { 0 };
                self.state[(db + p) * self.words + k] = (asserted[k] & rfill) | (!asserted[k] & nv);
            }
        }
    }

    pub fn tick(&mut self) {
        self.settle();
        // capture tick_next next-states into a buffer, then commit.
        let block = self.machine.streams.tick_next.clone();
        let (ws, layout) = self.eval(&block);
        let mut next: Vec<(usize, Vec<u64>)> = Vec::new();
        for pkt in &block.packets {
            for eff in &pkt.effects {
                if let PackedEffect::CaptureReg { dst, value, reset } = eff {
                    let (vb, vw) = layout[value];
                    let dw = self.widths[*dst] as usize;
                    let mut planes = vec![0u64; dw * self.words];
                    let mut asserted = vec![0u64; self.words];
                    if let Some(r) = reset {
                        self.reset_asserted(r, &mut asserted);
                    }
                    for p in 0..dw {
                        let rbit = reset.as_ref().map_or(0, |r| (r.value.get(p / 32).copied().unwrap_or(0) >> (p % 32)) & 1);
                        let rfill = if rbit == 1 { u64::MAX } else { 0 };
                        for k in 0..self.words {
                            let nv = if p < vw as usize { ws[(vb + p) * self.words + k] } else { 0 };
                            planes[p * self.words + k] = if reset.is_some() {
                                (asserted[k] & rfill) | (!asserted[k] & nv)
                            } else {
                                nv
                            };
                        }
                    }
                    next.push((*dst, planes));
                }
            }
        }
        for (dst, planes) in next {
            let db = self.sig_base[dst];
            let dw = self.widths[dst] as usize;
            for p in 0..dw {
                for k in 0..self.words {
                    self.state[(db + p) * self.words + k] = planes[p * self.words + k];
                }
            }
        }
        self.settle();
    }

    pub fn tick_many(&mut self, steps: usize) {
        for _ in 0..steps {
            self.tick();
        }
    }

    /// Plane bases / widths so the AOT can share the layout validation.
    #[cfg(feature = "aot")]
    fn layout_meta(&self) -> (Vec<usize>, Vec<u32>, usize) {
        (self.sig_base.clone(), self.widths.clone(), self.sig_base.last().copied().unwrap_or(0) + *self.widths.last().unwrap_or(&0) as usize)
    }
}

/// C codegen for the bit-slice AOT. State is **word-major**: word `k` holds all
/// `total_planes` planes contiguously (`st + k*total_planes`), so the emitted
/// `tick_bs` runs an inner loop over words (each = 64 lanes) that clang -O3
/// auto-vectorizes — exactly where the bit-parallel AOT's win came from.
#[cfg(feature = "aot")]
mod aot {
    use super::*;
    use std::fmt::Write as _;

    fn block_widths(block: &PackedBlock) -> HashMap<PackedValueId, u32> {
        let mut w = HashMap::new();
        for pkt in &block.packets {
            for i in &pkt.instrs {
                w.insert(i.dst, i.ty.width);
            }
        }
        w
    }

    /// Reference value `v`'s plane `p` as a C expression (0 above its width).
    fn vref(prefix: &str, v: PackedValueId, p: usize, w: &HashMap<PackedValueId, u32>) -> String {
        if p < *w.get(&v).unwrap_or(&0) as usize {
            format!("{prefix}{}_{p}", v.0)
        } else {
            "0ull".into()
        }
    }

    /// Emit a comb-ish block's instructions as plane locals, then its StoreSignal
    /// effects. Rejects async-reset captures (caller falls back to the interp).
    fn emit_comb(code: &mut String, block: &PackedBlock, prefix: &str, sig_base: &[usize]) -> Result<(), ErrorReport> {
        let w = block_widths(block);
        // All instrs first (Signal leaves read pre-store state — matches the interp).
        for pkt in &block.packets {
            for instr in &pkt.instrs {
                emit_instr(code, instr, prefix, &w, sig_base);
            }
        }
        for pkt in &block.packets {
            for eff in &pkt.effects {
                match eff {
                    PackedEffect::StoreSignal { dst, value } => {
                        let vw = *w.get(value).unwrap_or(&0) as usize;
                        // dst signal width inferred from the stored value's width.
                        for p in 0..vw {
                            writeln!(code, "   s[{}] = {};", sig_base[*dst] + p, vref(prefix, *value, p, &w)).ok();
                        }
                    }
                    PackedEffect::CaptureReg { reset: Some(_), .. } => {
                        return Err(bs_err("bit-slice AOT does not support async reset"));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Emit one instruction's destination planes as `u64 {prefix}{id}_{p}` locals.
    fn emit_instr(code: &mut String, instr: &crate::PackedInstr, prefix: &str, w: &HashMap<PackedValueId, u32>, sig_base: &[usize]) {
        use PackedInstrKind::*;
        let id = instr.dst.0;
        let dw = instr.ty.width as usize;
        let r = |v: PackedValueId, p: usize| vref(prefix, v, p, w);
        match &instr.kind {
            Signal(s) => {
                for p in 0..dw {
                    writeln!(code, "   u64 {prefix}{id}_{p} = s[{}];", sig_base[*s] + p).ok();
                }
            }
            Lit(wd) => {
                for p in 0..dw {
                    let bit = (wd.get(p / 32).copied().unwrap_or(0) >> (p % 32)) & 1;
                    writeln!(code, "   u64 {prefix}{id}_{p} = {};", if bit == 1 { "~0ull" } else { "0ull" }).ok();
                }
            }
            Not(a) => {
                for p in 0..dw {
                    writeln!(code, "   u64 {prefix}{id}_{p} = ~{};", r(*a, p)).ok();
                }
            }
            And(a, b) => emit_bitwise(code, prefix, id, dw, *a, *b, "&", w),
            Or(a, b) => emit_bitwise(code, prefix, id, dw, *a, *b, "|", w),
            Xor(a, b) => emit_bitwise(code, prefix, id, dw, *a, *b, "^", w),
            Mux { cond, then_value, else_value } => {
                let c = r(*cond, 0);
                for p in 0..dw {
                    writeln!(code, "   u64 {prefix}{id}_{p} = ({c} & {}) | (~{c} & {});", r(*then_value, p), r(*else_value, p)).ok();
                }
            }
            Eq(a, b) | Ne(a, b) => {
                let wmax = (*w.get(a).unwrap_or(&0)).max(*w.get(b).unwrap_or(&0)) as usize;
                let terms: Vec<String> = (0..wmax).map(|p| format!("({} ^ {})", r(*a, p), r(*b, p))).collect();
                let diff = if terms.is_empty() { "0ull".into() } else { terms.join(" | ") };
                let neg = if matches!(instr.kind, Eq(..)) { "~" } else { "" };
                writeln!(code, "   u64 {prefix}{id}_0 = {neg}({diff});").ok();
                for p in 1..dw {
                    writeln!(code, "   u64 {prefix}{id}_{p} = 0ull;").ok();
                }
            }
            Add(a, b) | Sub(a, b) => {
                let sub = matches!(instr.kind, Sub(..));
                writeln!(code, "   u64 {prefix}{id}_cy = {};", if sub { "~0ull" } else { "0ull" }).ok();
                for p in 0..dw {
                    let av = r(*a, p);
                    let bv = if sub { format!("(~{})", r(*b, p)) } else { r(*b, p) };
                    writeln!(code, "   u64 {prefix}{id}_axb{p} = {av} ^ {bv};").ok();
                    writeln!(code, "   u64 {prefix}{id}_{p} = {prefix}{id}_axb{p} ^ {prefix}{id}_cy;").ok();
                    writeln!(code, "   {prefix}{id}_cy = ({av} & {bv}) | ({prefix}{id}_cy & {prefix}{id}_axb{p});").ok();
                }
            }
            Slice { value, lsb } => {
                for p in 0..dw {
                    writeln!(code, "   u64 {prefix}{id}_{p} = {};", r(*value, p + *lsb as usize)).ok();
                }
            }
            Zext(a) | Trunc(a) | Cast(a) => {
                for p in 0..dw {
                    writeln!(code, "   u64 {prefix}{id}_{p} = {};", r(*a, p)).ok();
                }
            }
            Sext(a) => {
                let sw = *w.get(a).unwrap_or(&1) as usize;
                for p in 0..dw {
                    writeln!(code, "   u64 {prefix}{id}_{p} = {};", r(*a, p.min(sw.saturating_sub(1)))).ok();
                }
            }
            Concat(parts) => {
                let mut off = 0usize;
                for part in parts.iter().rev() {
                    let pw = *w.get(part).unwrap_or(&0) as usize;
                    for p in 0..pw {
                        writeln!(code, "   u64 {prefix}{id}_{} = {};", off + p, r(*part, p)).ok();
                    }
                    off += pw;
                }
            }
            _ => {}
        }
    }

    fn emit_bitwise(code: &mut String, prefix: &str, id: usize, dw: usize, a: PackedValueId, b: PackedValueId, op: &str, w: &HashMap<PackedValueId, u32>) {
        for p in 0..dw {
            writeln!(code, "   u64 {prefix}{id}_{p} = {} {op} {};", vref(prefix, a, p, w), vref(prefix, b, p, w)).ok();
        }
    }

    pub fn generate_c(machine: &PackedMachineProgram, sig_base: &[usize], widths: &[u32], total_planes: usize) -> Result<String, ErrorReport> {
        let mut c = String::from("typedef unsigned long long u64;\n");
        writeln!(c, "void tick_bs(u64* restrict st, long nw, long nc){{").ok();
        c.push_str(" for(long _c=0;_c<nc;_c++){\n");
        c.push_str("  for(long k=0;k<nw;k++){\n");
        writeln!(c, "   u64* restrict s = st + k*{total_planes};").ok();
        // settle: async_reset_comb then comb.
        emit_comb(&mut c, &machine.streams.async_reset_comb, "a", sig_base)?;
        emit_comb(&mut c, &machine.streams.comb, "c", sig_base)?;
        // tick_next: capture into nxt_ locals, then commit.
        let tn = &machine.streams.tick_next;
        let tw = block_widths(tn);
        for pkt in &tn.packets {
            for instr in &pkt.instrs {
                emit_instr(&mut c, instr, "n", &tw, sig_base);
            }
        }
        let mut caps: Vec<usize> = Vec::new();
        for pkt in &tn.packets {
            for eff in &pkt.effects {
                if let PackedEffect::CaptureReg { dst, value, reset } = eff {
                    let dw = widths[*dst] as usize;
                    for p in 0..dw {
                        let nv = vref("n", *value, p, &tw);
                        let rhs = match reset {
                            Some(r) => {
                                let asserted = match r.polarity {
                                    ResetPolarity::ActiveLow => format!("(~s[{}])", sig_base[r.signal]),
                                    _ => format!("s[{}]", sig_base[r.signal]),
                                };
                                let rbit = (r.value.get(p / 32).copied().unwrap_or(0) >> (p % 32)) & 1;
                                let rfill = if rbit == 1 { "~0ull" } else { "0ull" };
                                format!("(({asserted} & {rfill}) | (~({asserted}) & {nv}))")
                            }
                            None => nv,
                        };
                        writeln!(c, "   u64 nxt{}_{p} = {rhs};", dst).ok();
                    }
                    caps.push(*dst);
                }
            }
        }
        for dst in &caps {
            for p in 0..widths[*dst] as usize {
                writeln!(c, "   s[{}] = nxt{dst}_{p};", sig_base[*dst] + p).ok();
            }
        }
        // settle again after commit.
        emit_comb(&mut c, &machine.streams.async_reset_comb, "b", sig_base)?;
        emit_comb(&mut c, &machine.streams.comb, "d", sig_base)?;
        c.push_str("  }\n }\n}\n");
        Ok(c)
    }
}

/// AOT-compiled bit-slice simulator: emits per-plane `uint64_t` bitwise/ripple C
/// with an inner word loop (clang -O3 auto-vectorizes it, NEON i64x2 = 128
/// lanes/op), and dlopen-loads it. Word-major state. Same scope as
/// [`BitSliceSimulator`].
#[cfg(feature = "aot")]
pub struct BitSliceAot {
    _lib: libloading::Library,
    tick_fn: extern "C" fn(*mut u64, i64, i64),
    state: Vec<u64>,
    sig_base: Vec<usize>,
    widths: Vec<u32>,
    total_planes: usize,
    words: usize,
    lanes: usize,
}

#[cfg(feature = "aot")]
impl BitSliceAot {
    pub fn compile_lanes(machine: &PackedMachineProgram, lanes: usize) -> Result<Self, ErrorReport> {
        // Reuse the interpreter's scope validation (≤128-bit, no mem, op set).
        let interp = BitSliceSimulator::new(machine, 64)?;
        let (sig_base, widths, total_planes) = interp.layout_meta();
        let words = lanes.max(1).div_ceil(64);
        let c = aot::generate_c(machine, &sig_base, &widths, total_planes)?;

        let stamp = {
            let mut h = 0xcbf29ce484222325u64;
            for byte in c.as_bytes() {
                h = (h ^ *byte as u64).wrapping_mul(0x100000001b3);
            }
            format!("{h:x}")
        };
        let ext = if cfg!(target_os = "macos") { "dylib" } else { "so" };
        let dir = std::env::temp_dir();
        let cpath = dir.join(format!("rrtl_bs_{stamp}.c"));
        let libpath = dir.join(format!("librrtl_bs_{stamp}.{ext}"));
        std::fs::write(&cpath, &c).map_err(|e| bs_err(format!("write C: {e}")))?;
        let cc = std::env::var("CC").unwrap_or_else(|_| "clang".into());
        let out = std::process::Command::new(&cc)
            .args(["-O3", "-shared", "-fPIC", "-o"])
            .arg(&libpath)
            .arg(&cpath)
            .output()
            .map_err(|e| bs_err(format!("spawn {cc}: {e}")))?;
        if !out.status.success() {
            return Err(bs_err(format!("{cc} -O3 failed: {}", String::from_utf8_lossy(&out.stderr))));
        }
        let lib = unsafe { libloading::Library::new(&libpath) }.map_err(|e| bs_err(format!("dlopen: {e}")))?;
        let tick_fn = unsafe {
            let sym: libloading::Symbol<extern "C" fn(*mut u64, i64, i64)> =
                lib.get(b"tick_bs").map_err(|e| bs_err(format!("sym tick_bs: {e}")))?;
            *sym
        };
        Ok(Self {
            _lib: lib,
            tick_fn,
            state: vec![0u64; total_planes * words],
            sig_base,
            widths,
            total_planes,
            words,
            lanes: words * 64,
        })
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }
    pub fn set_signal(&mut self, idx: usize, lane: usize, value: u128) {
        let (k, b) = (lane / 64, lane % 64);
        for p in 0..self.widths[idx] as usize {
            let slot = k * self.total_planes + self.sig_base[idx] + p;
            if (value >> p) & 1 == 1 {
                self.state[slot] |= 1u64 << b;
            } else {
                self.state[slot] &= !(1u64 << b);
            }
        }
    }
    pub fn get_signal(&self, idx: usize, lane: usize) -> u128 {
        let (k, b) = (lane / 64, lane % 64);
        let mut v = 0u128;
        for p in 0..self.widths[idx] as usize {
            let slot = k * self.total_planes + self.sig_base[idx] + p;
            if (self.state[slot] >> b) & 1 == 1 {
                v |= 1u128 << p;
            }
        }
        v
    }
    pub fn tick(&mut self) {
        self.tick_many(1);
    }
    pub fn tick_many(&mut self, n: usize) {
        (self.tick_fn)(self.state.as_mut_ptr(), self.words as i64, n as i64);
    }

    /// Multicore `tick_many`: word-major state splits into disjoint word-ranges
    /// (words are fully independent lanes — no shared state), each advanced `n`
    /// cycles on its own core via rayon.
    pub fn tick_many_parallel(&mut self, n: usize) {
        use rayon::prelude::*;
        let f = self.tick_fn;
        let tp = self.total_planes;
        let wpt = (self.words / (rayon::current_num_threads() * 4)).max(1);
        self.state.par_chunks_mut(tp * wpt).for_each(|chunk| {
            let w = (chunk.len() / tp) as i64;
            f(chunk.as_mut_ptr(), w, n as i64);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lower_to_machine_program, lower_to_packed_program, SimdCpuSimulator};
    use rrtl_core::{compile, uint, Design};

    // A multi-bit control design (mux/add/xor/eq/slice + 1-bit control, sync reset)
    // must match the SIMD CPU engine on every output of every lane.
    #[test]
    fn bitslice_matches_simd_cpu() {
        let mut design = Design::new();
        {
            let mut m = design.module("Ctl");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let sel = m.input("sel", uint(1));
            let a = m.input("a", uint(8));
            let b = m.input("b", uint(8));
            let acc = m.reg("acc", uint(8));
            m.clock(acc, clk);
            m.reset(acc, rst, 0);
            m.next(acc, rrtl_ir::mux(sel.value(), acc.value() + a.value(), acc.value() ^ b.value()));
            let o = m.output("o", uint(8));
            m.assign(o, acc.value());
            let o2 = m.output("o2", uint(1));
            m.assign(o2, acc.value().eq_expr(a.value())); // 1-bit eq path
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Ctl").unwrap();
        let machine = lower_to_machine_program(&program);
        let h = |n: &str| compiled.find_module("Ctl").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
        let idx = |n: &str| program.signal_index(h(n)).unwrap();
        let lanes = 100;

        let mut bs = BitSliceSimulator::new(&machine, lanes).unwrap();
        let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        cpu.set_signal(h("clk"), &vec![1u128; lanes]).unwrap();
        for c in 0..30u64 {
            for (name, mask) in [("rst", 1u128), ("sel", 1), ("a", 0xff), ("b", 0xff)] {
                let vals: Vec<u128> = (0..lanes)
                    .map(|l| {
                        let s = c.wrapping_mul(2654435761).wrapping_add(l as u64).wrapping_add(name.len() as u64 * 7);
                        // assert reset only for the first 2 cycles
                        if name == "rst" { (c < 2) as u128 } else { (s as u128) & mask }
                    })
                    .collect();
                for (l, v) in vals.iter().enumerate() {
                    bs.set_signal(idx(name), l, *v);
                }
                cpu.set_signal(h(name), &vals).unwrap();
            }
            bs.tick();
            cpu.tick().unwrap();
            for name in ["o", "o2"] {
                let cv = cpu.get_signal(h(name)).unwrap();
                for l in 0..lanes {
                    assert_eq!(bs.get_signal(idx(name), l), cv[l], "{name}@lane{l} cyc{c}");
                }
            }
        }
    }

    // The bit-slice AOT (clang -O3 per-plane C) must match the SIMD CPU engine.
    #[cfg(feature = "aot")]
    #[test]
    fn bitslice_aot_matches_simd_cpu() {
        let mut design = Design::new();
        {
            let mut m = design.module("Ctl");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let sel = m.input("sel", uint(1));
            let a = m.input("a", uint(8));
            let b = m.input("b", uint(8));
            let acc = m.reg("acc", uint(8));
            m.clock(acc, clk);
            m.reset(acc, rst, 0);
            m.next(acc, rrtl_ir::mux(sel.value(), acc.value() + a.value(), acc.value() - b.value()));
            let o = m.output("o", uint(8));
            m.assign(o, acc.value());
            let o2 = m.output("o2", uint(1));
            m.assign(o2, acc.value().eq_expr(a.value()));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Ctl").unwrap();
        let machine = lower_to_machine_program(&program);
        let h = |n: &str| compiled.find_module("Ctl").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
        let idx = |n: &str| program.signal_index(h(n)).unwrap();
        let lanes = 130; // spans 3 words

        let mut bs = BitSliceAot::compile_lanes(&machine, lanes).unwrap();
        let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        cpu.set_signal(h("clk"), &vec![1u128; lanes]).unwrap();
        for c in 0..30u64 {
            for (name, mask) in [("rst", 1u128), ("sel", 1), ("a", 0xff), ("b", 0xff)] {
                let vals: Vec<u128> = (0..lanes)
                    .map(|l| {
                        let s = c.wrapping_mul(2654435761).wrapping_add(l as u64).wrapping_add(name.len() as u64 * 7);
                        if name == "rst" { (c < 2) as u128 } else { (s as u128) & mask }
                    })
                    .collect();
                for (l, v) in vals.iter().enumerate() {
                    bs.set_signal(idx(name), l, *v);
                }
                cpu.set_signal(h(name), &vals).unwrap();
            }
            bs.tick();
            cpu.tick().unwrap();
            for name in ["o", "o2"] {
                let cv = cpu.get_signal(h(name)).unwrap();
                for l in 0..lanes {
                    assert_eq!(bs.get_signal(idx(name), l), cv[l], "{name}@lane{l} cyc{c}");
                }
            }
        }
    }
}
