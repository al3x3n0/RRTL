//! Lane-parallel GPU kernel codegen: emit a per-design simulation kernel (one
//! GPU thread = one lane/instance) from the packed IR, in CUDA or OpenCL. This
//! is the GPU analog of the AOT C backend — a compiled per-design kernel (no
//! interpreter dispatch) for the batch-throughput moat, and the substrate a
//! tensor-core (BMMA) linear-block offload would later plug into.
//!
//! State is lane-major (`st[slot*n_lanes + lane]`) so consecutive threads touch
//! consecutive addresses (coalesced). Eager tick (settle→capture→commit→settle).
//! v1 targets ≤64-bit signals and memory-free designs (OpenCL C lacks __int128).

use crate::{PackedBlock, PackedEffect, PackedInstr, PackedInstrKind, PackedMachineProgram, PackedReset};
use rrtl_ir::{ErrorReport, ResetPolarity};
use std::collections::HashMap;
use std::fmt::Write as _;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Flavor {
    Cuda,
    OpenCl,
}

fn gerr(m: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![rrtl_ir::Diagnostic::new("E_GPU", m)])
}

fn mask_suffix(w: u32) -> String {
    if w >= 64 {
        String::new()
    } else {
        format!(" & 0x{:x}UL", (1u64 << w) - 1)
    }
}

fn lit64(words: &[u32]) -> u64 {
    let mut v = 0u64;
    for (i, w) in words.iter().enumerate().take(2) {
        v |= (*w as u64) << (32 * i);
    }
    v
}

struct Gen {
    widths: Vec<u32>,
    /// memory base SLOT (= num_signals + cumulative depth); entry `e` of memory
    /// `m` lives at slot `mem_base[m] + e`, i.e. `st[(mem_base[m]+e)*nl + lane]`.
    mem_base: Vec<usize>,
    mem_depth: Vec<usize>,
    mem_width: Vec<u32>,
}

impl Gen {
    fn sig(&self, idx: usize) -> String {
        format!("st[{}*nl + lane]", idx)
    }

    fn emit_block(&self, code: &mut String, block: &PackedBlock, prefix: &str, capture_to_temp: bool) -> Result<(), ErrorReport> {
        let mut vw: HashMap<usize, u32> = HashMap::new();
        for packet in &block.packets {
            for instr in &packet.instrs {
                self.emit_instr(code, instr, prefix, &mut vw)?;
            }
            for effect in &packet.effects {
                self.emit_effect(code, effect, prefix, capture_to_temp)?;
            }
        }
        Ok(())
    }

    /// Emit one machine instruction as `ulong <prefix><id> = (...) & mask;`,
    /// recording its width in `vw` (used by sign/concat of later instrs).
    fn emit_instr(&self, code: &mut String, instr: &PackedInstr, prefix: &str, vw: &mut HashMap<usize, u32>) -> Result<(), ErrorReport> {
        {
                let w = instr.ty.width;
                if w > 64 {
                    return Err(gerr("GPU v1 supports ≤64-bit signals"));
                }
                let id = instr.dst.0;
                let v = |a: &crate::PackedValueId| format!("{}{}", prefix, a.0);
                let opw = |a: &crate::PackedValueId| vw.get(&a.0).copied().unwrap_or(64);
                let rhs = match &instr.kind {
                    PackedInstrKind::Lit(words) => format!("0x{:x}UL", lit64(words) & if w >= 64 { u64::MAX } else { (1 << w) - 1 }),
                    PackedInstrKind::Signal(s) => self.sig(*s),
                    PackedInstrKind::Not(a) => format!("~{}", v(a)),
                    PackedInstrKind::And(a, b) => format!("{} & {}", v(a), v(b)),
                    PackedInstrKind::Or(a, b) => format!("{} | {}", v(a), v(b)),
                    PackedInstrKind::Xor(a, b) => format!("{} ^ {}", v(a), v(b)),
                    PackedInstrKind::Add(a, b) => format!("{} + {}", v(a), v(b)),
                    PackedInstrKind::Sub(a, b) => format!("{} - {}", v(a), v(b)),
                    PackedInstrKind::Mul(a, b) => format!("{} * {}", v(a), v(b)),
                    PackedInstrKind::Eq(a, b) => format!("({} == {}) ? 1UL : 0UL", v(a), v(b)),
                    PackedInstrKind::Ne(a, b) => format!("({} != {}) ? 1UL : 0UL", v(a), v(b)),
                    PackedInstrKind::Lt { lhs, rhs, signed } => {
                        if *signed {
                            let sx = |x: &crate::PackedValueId, bw: u32| format!("((long)({} << {}) >> {})", v(x), 64 - bw, 64 - bw);
                            format!("({} < {}) ? 1UL : 0UL", sx(lhs, opw(lhs)), sx(rhs, opw(rhs)))
                        } else {
                            format!("({} < {}) ? 1UL : 0UL", v(lhs), v(rhs))
                        }
                    }
                    PackedInstrKind::Mux { cond, then_value, else_value } => {
                        format!("({} & 1) ? {} : {}", v(cond), v(then_value), v(else_value))
                    }
                    PackedInstrKind::Slice { value, lsb } => format!("{} >> {}", v(value), lsb),
                    PackedInstrKind::Zext(a) | PackedInstrKind::Trunc(a) | PackedInstrKind::Cast(a) => v(a),
                    PackedInstrKind::Sext(a) => {
                        let aw = opw(a);
                        format!("(ulong)((long)({} << {}) >> {})", v(a), 64 - aw, 64 - aw)
                    }
                    PackedInstrKind::Concat(parts) => {
                        let mut terms = Vec::new();
                        let mut off = 0u32;
                        for p in parts.iter().rev() {
                            let pw = opw(p);
                            terms.push(format!("(({}{}) << {})", v(p), mask_suffix(pw), off));
                            off += pw;
                        }
                        if terms.is_empty() { "0UL".into() } else { terms.join(" | ") }
                    }
                    PackedInstrKind::MemRead { memory, addr } => {
                        let depth = self.mem_depth[*memory];
                        if depth == 0 {
                            "0UL".into()
                        } else {
                            // entry slot = mem_base + addr; out-of-range reads 0.
                            format!(
                                "({} < {}UL) ? st[({} + {})*nl + lane] : 0UL",
                                v(addr), depth, self.mem_base[*memory], v(addr)
                            )
                        }
                    }
                };
                writeln!(code, "    ulong {}{} = ({}){};", prefix, id, rhs, mask_suffix(w)).ok();
                vw.insert(id, w);
        }
        Ok(())
    }

    fn emit_effect(&self, code: &mut String, e: &PackedEffect, prefix: &str, capture_to_temp: bool) -> Result<(), ErrorReport> {
        match e {
            PackedEffect::StoreSignal { dst, value } => {
                writeln!(code, "    {} = {}{}{};", self.sig(*dst), prefix, value.0, mask_suffix(self.widths[*dst])).ok();
            }
            PackedEffect::CaptureReg { dst, value, reset } => {
                let dw = self.widths[*dst];
                if capture_to_temp {
                    let nv = match reset {
                        Some(r) => format!("({}) ? {} : {}{}", self.reset_asserted(r), self.reset_val(r, dw), prefix, value.0),
                        None => format!("{}{}", prefix, value.0),
                    };
                    writeln!(code, "    ulong next_{} = ({}){};", dst, nv, mask_suffix(dw)).ok();
                } else {
                    let r = reset.as_ref().ok_or_else(|| gerr("reset-less capture in comb"))?;
                    writeln!(code, "    {} = (({}) ? {} : {}){};", self.sig(*dst), self.reset_asserted(r), self.reset_val(r, dw), self.sig(*dst), mask_suffix(dw)).ok();
                }
            }
            PackedEffect::MemoryWrite { memory, enable, addr, data } => {
                let depth = self.mem_depth[*memory];
                if depth == 0 {
                    return Ok(());
                }
                writeln!(
                    code,
                    "    if (({p}{e} & 1) && {p}{a} < {d}UL) st[({base} + {p}{a})*nl + lane] = {p}{dat}{mask};",
                    p = prefix, e = enable.0, a = addr.0, d = depth, base = self.mem_base[*memory],
                    dat = data.0, mask = mask_suffix(self.mem_width[*memory])
                ).ok();
            }
        }
        Ok(())
    }

    fn reset_asserted(&self, r: &PackedReset) -> String {
        let bit = format!("({} & 1)", self.sig(r.signal));
        match r.polarity {
            ResetPolarity::ActiveHigh => format!("({} != 0)", bit),
            ResetPolarity::ActiveLow => format!("({} == 0)", bit),
        }
    }
    fn reset_val(&self, r: &PackedReset, dw: u32) -> String {
        format!("0x{:x}UL", lit64(&r.value) & if dw >= 64 { u64::MAX } else { (1 << dw) - 1 })
    }
}

/// Number of u64 slots per lane (signals + all memory entries) — the host
/// allocates `state_slots(machine) * n_lanes` u64 for the state buffer.
pub fn state_slots(machine: &PackedMachineProgram) -> usize {
    machine.source.signals.len() + machine.source.memories.iter().map(|m| m.depth).sum::<usize>()
}

/// Emit a lane-parallel kernel for a FULLY GF(2)-LINEAR design (every register's
/// next-state cone is linear — CRC/FEC/scrambler/LFSR class) where each register
/// update is a binary matrix product `out = M·in ⊕ c`. Rather than the gate tree
/// we emit, per output bit, `out[i] = parity(row_i & in) ⊕ c[i]` =
/// `popcount(row_i & in) & 1` — the SAME 1-bit AND-popcount primitive a
/// tensor-core BMMA executes in bulk (on the M3 it runs as a branchless
/// `popcount`; the CUDA emit uses `__popcll`, and a wmma::bmma path can replace
/// the popcount loop with one matrix op on Ampere+). Errors if any register cone
/// is non-linear, or in/out exceeds 128 bits.
pub fn emit_kernel_linear(
    program: &crate::PackedProgram,
    flavor: Flavor,
) -> Result<String, ErrorReport> {
    use crate::{PackedExprKind, PackedOp};
    use std::fmt::Write as _;

    // (dst, reset, LinearForm) for every captured register; all must be linear.
    let mut forms: Vec<(usize, Option<crate::PackedReset>, crate::linearize::LinearForm)> = Vec::new();
    for packet in &program.streams.tick_next {
        for op in &packet.ops {
            if let PackedOp::CaptureReg { dst, next, reset } = op {
                if !crate::linearize::is_linear(program, next) {
                    return Err(gerr("emit_kernel_linear: design has a non-linear register cone"));
                }
                let lf = crate::linearize::extract_linear_form(program, next);
                if lf.total_in_bits > 128 || lf.out_width > 64 {
                    return Err(gerr("emit_kernel_linear v1: ≤128 input bits, ≤64 output bits"));
                }
                forms.push((*dst, reset.clone(), lf));
            }
        }
    }
    if forms.is_empty() {
        return Err(gerr("emit_kernel_linear: no registers"));
    }
    // Reject combinational state we wouldn't compute (keep the contract honest).
    for packet in &program.streams.tick_next {
        for op in &packet.ops {
            if let PackedOp::Assign { expr, .. } = op {
                if !matches!(expr.kind, PackedExprKind::Signal(_)) {
                    return Err(gerr("emit_kernel_linear: non-trivial combinational op (use the gate kernel)"));
                }
            }
        }
    }

    let popcnt = match flavor {
        Flavor::OpenCl => "popcount",
        Flavor::Cuda => "__popcll",
    };
    let mut c = String::new();
    match flavor {
        Flavor::OpenCl => {
            c.push_str("// OpenCL GF(2)-linear (BMMA-shaped) sim kernel — RRTL tensor-core offload\n");
            c.push_str("__kernel void tick(__global ulong* st, long nl, long ncyc) {\n");
            c.push_str("  long lane = get_global_id(0);\n  if (lane >= nl) return;\n");
        }
        Flavor::Cuda => {
            c.push_str("// CUDA GF(2)-linear (BMMA-shaped) sim kernel — RRTL tensor-core offload\n");
            c.push_str("typedef unsigned long long ulong;\n");
            c.push_str("extern \"C\" __global__ void tick(ulong* st, long nl, long ncyc) {\n");
            c.push_str("  long lane = (long)blockIdx.x * blockDim.x + threadIdx.x;\n  if (lane >= nl) return;\n");
        }
    }
    c.push_str("  for (long _c = 0; _c < ncyc; _c++) {\n");

    for (dst, reset, lf) in &forms {
        linear_reg_code(&mut c, program, *dst, reset, lf, popcnt);
    }
    // commit
    for (dst, _, lf) in &forms {
        writeln!(c, "    st[{d}*nl + lane] = next_{d}{};", mask_suffix(lf.out_width), d = dst).ok();
    }
    c.push_str("  }\n}\n");
    Ok(c)
}

/// Emit the BMMA matrix update for ONE linear register: pack its leaves into a
/// lo/hi input vector, then `out[i] = parity(row_i & in) ⊕ c[i]` per output bit,
/// reset-muxed into `next_<dst>`. Shared by the pure-linear and hybrid kernels.
fn linear_reg_code(
    c: &mut String,
    program: &crate::PackedProgram,
    dst: usize,
    reset: &Option<crate::PackedReset>,
    lf: &crate::linearize::LinearForm,
    popcnt: &str,
) {
    use std::fmt::Write as _;
    let wmask = |w: u32| if w >= 64 { String::new() } else { format!(" & 0x{:x}UL", (1u64 << w) - 1) };
    // --- pack the input vector (leaves, in input-bit order) into lo/hi ---
    writeln!(c, "    ulong ilo{d} = 0, ihi{d} = 0;", d = dst).ok();
    let mut off = 0u32;
    for &(sig, w) in &lf.leaves {
        let v = format!("(st[{}*nl + lane]{})", sig, wmask(w));
        if off < 64 {
            writeln!(c, "    ilo{d} |= {v} << {off};", d = dst).ok();
            if off + w > 64 {
                writeln!(c, "    ihi{d} |= {v} >> {sh};", d = dst, sh = 64 - off).ok();
            }
        } else {
            writeln!(c, "    ihi{d} |= {v} << {sh};", d = dst, sh = off - 64).ok();
        }
        off += w;
    }
    // --- transpose columns → rows, emit one AND-popcount-parity per out bit ---
    let rows: Vec<u128> = (0..lf.out_width)
        .map(|i| {
            let mut r = 0u128;
            for (b, col) in lf.columns.iter().enumerate() {
                if (col >> i) & 1 == 1 {
                    r |= 1u128 << b;
                }
            }
            r
        })
        .collect();
    writeln!(c, "    ulong m{d} = 0x{:x}UL;", lf.constant as u64, d = dst).ok();
    for (i, row) in rows.iter().enumerate() {
        let (rlo, rhi) = (*row as u64, (*row >> 64) as u64);
        if rlo == 0 && rhi == 0 {
            continue; // out bit i is purely constant
        }
        let term = if rhi == 0 {
            format!("{popcnt}(ilo{d} & 0x{rlo:x}UL)", d = dst)
        } else {
            format!("({popcnt}(ilo{d} & 0x{rlo:x}UL) + {popcnt}(ihi{d} & 0x{rhi:x}UL))", d = dst)
        };
        writeln!(c, "    m{d} ^= ((ulong)({term} & 1)) << {i};", d = dst).ok();
    }
    // --- reset mux into the next-temp ---
    let dw = program.signals[dst].layout.width;
    match reset {
        Some(r) => {
            let bit = format!("(st[{}*nl + lane] & 1)", r.signal);
            let asserted = match r.polarity {
                ResetPolarity::ActiveHigh => format!("({bit} != 0)"),
                ResetPolarity::ActiveLow => format!("({bit} == 0)"),
            };
            let rv = lit64(&r.value) & if dw >= 64 { u64::MAX } else { (1 << dw) - 1 };
            writeln!(c, "    ulong next_{d} = {asserted} ? 0x{rv:x}UL : m{d};", d = dst).ok();
        }
        None => {
            writeln!(c, "    ulong next_{d} = m{d};", d = dst).ok();
        }
    }
}

/// The operand value-ids an instruction reads (for liveness pruning).
fn instr_reads(kind: &PackedInstrKind) -> Vec<usize> {
    use PackedInstrKind::*;
    match kind {
        Lit(_) | Signal(_) => vec![],
        Not(a) | Slice { value: a, .. } | Zext(a) | Sext(a) | Trunc(a) | Cast(a)
        | MemRead { addr: a, .. } => vec![a.0],
        And(a, b) | Or(a, b) | Xor(a, b) | Add(a, b) | Sub(a, b) | Mul(a, b) | Eq(a, b)
        | Ne(a, b) => vec![a.0, b.0],
        Lt { lhs, rhs, .. } => vec![lhs.0, rhs.0],
        Mux { cond, then_value, else_value } => vec![cond.0, then_value.0, else_value.0],
        Concat(parts) => parts.iter().map(|p| p.0).collect(),
    }
}

/// SELECTIVE per-cone offload: emit a HYBRID kernel for a MIXED design where SOME
/// register cones are GF(2)-linear and some are not. Linear registers get the
/// BMMA matrix form ([[tensor-core-linear]]); everything else stays gate-tree.
/// The gate instructions that feed ONLY linear registers are pruned by liveness
/// (so the offloaded cones cost nothing on the SIMT side) — the actual point of
/// "offload", vs emitting both and hoping the GPU compiler DCEs.
pub fn emit_kernel_hybrid(
    machine: &PackedMachineProgram,
    program: &crate::PackedProgram,
    flavor: Flavor,
) -> Result<String, ErrorReport> {
    use crate::PackedOp;
    let widths: Vec<u32> = machine.source.signals.iter().map(|s| s.layout.width).collect();
    if widths.iter().any(|w| *w > 64) {
        return Err(gerr("GPU v1 supports ≤64-bit signals"));
    }
    // Which register dsts are linear? (signal index → extracted matrix form)
    let mut linear: HashMap<usize, crate::linearize::LinearForm> = HashMap::new();
    for packet in &program.streams.tick_next {
        for op in &packet.ops {
            if let PackedOp::CaptureReg { dst, next, .. } = op {
                if crate::linearize::is_linear(program, next) {
                    let lf = crate::linearize::extract_linear_form(program, next);
                    if lf.total_in_bits <= 128 && lf.out_width <= 64 {
                        linear.insert(*dst, lf);
                    }
                }
            }
        }
    }
    if linear.is_empty() {
        return Err(gerr("emit_kernel_hybrid: no linear register cones (use emit_kernel)"));
    }
    let n_linear = linear.len();

    // memory layout (same scheme as emit_kernel)
    let num_signals = widths.len();
    let (mut mem_base, mut mem_depth, mut mem_width) = (Vec::new(), Vec::new(), Vec::new());
    let mut acc = 0usize;
    for m in &machine.source.memories {
        if m.data_layout.width > 64 {
            return Err(gerr("GPU v1 supports ≤64-bit memory data"));
        }
        mem_base.push(num_signals + acc);
        mem_depth.push(m.depth);
        mem_width.push(m.data_layout.width);
        acc += m.depth;
    }
    let g = Gen { widths: widths.clone(), mem_base, mem_depth, mem_width };
    let popcnt = match flavor {
        Flavor::OpenCl => "popcount",
        Flavor::Cuda => "__popcll",
    };

    // --- liveness over tick_next: a value-id is LIVE if any non-linear capture,
    //     store, or memory-write needs it. Instrs feeding only linear caps die. ---
    let tn = &machine.streams.tick_next;
    let mut live: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for packet in &tn.packets {
        for eff in &packet.effects {
            match eff {
                PackedEffect::StoreSignal { value, .. } => {
                    live.insert(value.0);
                }
                PackedEffect::CaptureReg { dst, value, .. } => {
                    if !linear.contains_key(dst) {
                        live.insert(value.0);
                    }
                }
                PackedEffect::MemoryWrite { enable, addr, data, .. } => {
                    live.insert(enable.0);
                    live.insert(addr.0);
                    live.insert(data.0);
                }
            }
        }
    }
    // backward propagate (single reverse pass: instrs are in topological order)
    for packet in tn.packets.iter().rev() {
        for instr in packet.instrs.iter().rev() {
            if live.contains(&instr.dst.0) {
                for r in instr_reads(&instr.kind) {
                    live.insert(r);
                }
            }
        }
    }
    let n_instrs_total: usize = tn.packets.iter().map(|p| p.instrs.len()).sum();
    let n_instrs_live = tn.packets.iter().flat_map(|p| &p.instrs).filter(|i| live.contains(&i.dst.0)).count();

    // --- emit ---
    let mut c = String::new();
    match flavor {
        Flavor::OpenCl => {
            writeln!(c, "// OpenCL HYBRID sim kernel — {n_linear} linear cone(s) on BMMA, rest gate-tree; RRTL").ok();
            writeln!(c, "// tick_next gate instrs pruned by liveness: {n_instrs_live}/{n_instrs_total} kept").ok();
            c.push_str("__kernel void tick(__global ulong* st, long nl, long ncyc) {\n");
            c.push_str("  long lane = get_global_id(0);\n  if (lane >= nl) return;\n");
        }
        Flavor::Cuda => {
            writeln!(c, "// CUDA HYBRID sim kernel — {n_linear} linear cone(s) on BMMA, rest gate-tree; RRTL").ok();
            writeln!(c, "// tick_next gate instrs pruned by liveness: {n_instrs_live}/{n_instrs_total} kept").ok();
            c.push_str("typedef unsigned long long ulong;\n");
            c.push_str("extern \"C\" __global__ void tick(ulong* st, long nl, long ncyc) {\n");
            c.push_str("  long lane = (long)blockIdx.x * blockDim.x + threadIdx.x;\n  if (lane >= nl) return;\n");
        }
    }
    c.push_str("  for (long _c = 0; _c < ncyc; _c++) {\n");
    g.emit_block(&mut c, &machine.streams.async_reset_comb, "a", false)?;
    g.emit_block(&mut c, &machine.streams.comb, "c", false)?;

    // tick_next: live gate instrs, then per-capture next-temps (gate or BMMA)
    let mut caps: Vec<(usize, u32)> = Vec::new();
    let mut vw: HashMap<usize, u32> = HashMap::new();
    for packet in &tn.packets {
        for instr in &packet.instrs {
            if !live.contains(&instr.dst.0) {
                continue; // pruned: feeds only an offloaded linear cone
            }
            g.emit_instr(&mut c, instr, "n", &mut vw)?;
        }
        for eff in &packet.effects {
            if let PackedEffect::CaptureReg { dst, value, reset } = eff {
                caps.push((*dst, g.widths[*dst]));
                if let Some(lf) = linear.get(dst) {
                    linear_reg_code(&mut c, program, *dst, reset, lf, popcnt);
                } else {
                    let dw = g.widths[*dst];
                    let nv = match reset {
                        Some(r) => format!("({}) ? {} : n{}", g.reset_asserted(r), g.reset_val(r, dw), value.0),
                        None => format!("n{}", value.0),
                    };
                    writeln!(c, "    ulong next_{} = ({}){};", dst, nv, mask_suffix(dw)).ok();
                }
            } else {
                g.emit_effect(&mut c, eff, "n", true)?;
            }
        }
    }
    g.emit_block(&mut c, &machine.streams.tick_commit, "m", false)?;
    for (dst, w) in &caps {
        writeln!(c, "    {} = next_{}{};", g.sig(*dst), dst, mask_suffix(*w)).ok();
    }
    g.emit_block(&mut c, &machine.streams.async_reset_comb, "A", false)?;
    g.emit_block(&mut c, &machine.streams.comb, "C", false)?;
    c.push_str("  }\n}\n");
    Ok(c)
}

/// Emit the lane-parallel kernel for `machine` in the given flavor.
pub fn emit_kernel(machine: &PackedMachineProgram, flavor: Flavor) -> Result<String, ErrorReport> {
    let widths: Vec<u32> = machine.source.signals.iter().map(|s| s.layout.width).collect();
    if widths.iter().any(|w| *w > 64) {
        return Err(gerr("GPU v1 supports ≤64-bit signals"));
    }
    // Lay memories out as extra slots after the signals (lane-major).
    let num_signals = widths.len();
    let (mut mem_base, mut mem_depth, mut mem_width) = (Vec::new(), Vec::new(), Vec::new());
    let mut acc = 0usize;
    for m in &machine.source.memories {
        if m.data_layout.width > 64 {
            return Err(gerr("GPU v1 supports ≤64-bit memory data"));
        }
        mem_base.push(num_signals + acc);
        mem_depth.push(m.depth);
        mem_width.push(m.data_layout.width);
        acc += m.depth;
    }
    let g = Gen { widths, mem_base, mem_depth, mem_width };

    let mut c = String::new();
    match flavor {
        Flavor::OpenCl => {
            c.push_str("// OpenCL lane-parallel sim kernel (auto-generated by RRTL)\n");
            c.push_str("__kernel void tick(__global ulong* st, long nl, long ncyc) {\n");
            c.push_str("  long lane = get_global_id(0);\n  if (lane >= nl) return;\n");
        }
        Flavor::Cuda => {
            c.push_str("// CUDA lane-parallel sim kernel (auto-generated by RRTL)\n");
            c.push_str("typedef unsigned long long ulong;\n");
            c.push_str("extern \"C\" __global__ void tick(ulong* st, long nl, long ncyc) {\n");
            c.push_str("  long lane = (long)blockIdx.x * blockDim.x + threadIdx.x;\n  if (lane >= nl) return;\n");
        }
    }
    c.push_str("  for (long _c = 0; _c < ncyc; _c++) {\n");
    g.emit_block(&mut c, &machine.streams.async_reset_comb, "a", false)?;
    g.emit_block(&mut c, &machine.streams.comb, "c", false)?;
    let mut caps: Vec<(usize, u32)> = Vec::new();
    for packet in &machine.streams.tick_next.packets {
        for op in &packet.effects {
            if let PackedEffect::CaptureReg { dst, .. } = op {
                caps.push((*dst, g.widths[*dst]));
            }
        }
    }
    g.emit_block(&mut c, &machine.streams.tick_next, "n", true)?;
    g.emit_block(&mut c, &machine.streams.tick_commit, "m", false)?;
    for (dst, w) in &caps {
        writeln!(c, "    {} = next_{}{};", g.sig(*dst), dst, mask_suffix(*w)).ok();
    }
    g.emit_block(&mut c, &machine.streams.async_reset_comb, "A", false)?;
    g.emit_block(&mut c, &machine.streams.comb, "C", false)?;
    c.push_str("  }\n}\n");
    Ok(c)
}
