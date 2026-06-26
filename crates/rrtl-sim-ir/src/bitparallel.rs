//! Bit-parallel gate-level batch engine.
//!
//! For a gate-level / boolean design (every signal 1 bit, every op bitwise), the
//! state of one signal across `L` lanes is `ceil(L/64)` u64 words — one bit per
//! lane. A gate then evaluates as a plain scalar bitwise op over those words, so
//! **one u64 op advances 64 independent lanes**. Compared with the SIMD vector
//! JIT's I8X16 layout (16 lanes per 128-bit register, each lane wasting 7/8 of a
//! byte on a 1-bit signal), this is ~8x the lane density with cheaper ops — the
//! classic logic-simulation bitwise-parallelism technique, applied to RRTL's
//! batch model. It targets the gate-level netlist regime (the huge-design /
//! Verilator-competition target).
//!
//! Scope: all signals/values are 1 bit; ops And/Or/Xor/Not/Mux/Signal/Lit;
//! registers with optional sync/async reset. Arithmetic, width ops, multi-bit
//! signals, and memories are rejected (use another engine, or a future
//! bit-sliced extension). [`BitParallelSimulator::new`] errors out on anything
//! out of scope so the caller can fall back.

use std::collections::HashMap;

use rrtl_ir::{Diagnostic, ErrorReport, ResetPolarity};

use crate::{
    PackedBlock, PackedEffect, PackedInstrKind, PackedMachineProgram, PackedReset, PackedValueId,
};

fn bp_err(msg: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![Diagnostic::new("E_BITPARALLEL", msg)])
}

/// A bit-parallel batch simulator: 64 lanes per u64 word, gates as bitwise ops.
pub struct BitParallelSimulator {
    machine: PackedMachineProgram,
    lanes: usize,
    /// Words (u64) per signal = `ceil(lanes/64)`.
    words: usize,
    /// `num_signals * words` — signal `s` lane-words at `[s*words ..]`.
    state: Vec<u64>,
    num_signals: usize,
}

impl BitParallelSimulator {
    /// Build for `lanes` independent instances, or error if the design is not a
    /// pure gate-level (all-1-bit, bitwise) design.
    pub fn new(machine: &PackedMachineProgram, lanes: usize) -> Result<Self, ErrorReport> {
        let num_signals = machine.source.signals.len();
        for s in &machine.source.signals {
            if s.layout.width != 1 {
                return Err(bp_err(format!(
                    "signal `{}` is {} bits — bit-parallel needs all 1-bit (gate-level) signals",
                    s.name, s.layout.width
                )));
            }
        }
        if !machine.source.memories.is_empty() {
            return Err(bp_err("memories are not supported by the bit-parallel engine"));
        }
        // Validate the op set across every stream.
        for blk in Self::streams(machine) {
            for pkt in &blk.packets {
                for instr in &pkt.instrs {
                    use PackedInstrKind::*;
                    match &instr.kind {
                        And(..) | Or(..) | Xor(..) | Not(..) | Mux { .. } | Signal(_) | Lit(_) => {}
                        other => {
                            return Err(bp_err(format!(
                                "op {:?} is not a gate-level bitwise op (bit-parallel scope)",
                                std::mem::discriminant(other)
                            )))
                        }
                    }
                }
                for eff in &pkt.effects {
                    if let PackedEffect::MemoryWrite { .. } = eff {
                        return Err(bp_err("memory writes are not supported by the bit-parallel engine"));
                    }
                }
            }
        }
        let words = lanes.div_ceil(64).max(1);
        Ok(Self {
            machine: machine.clone(),
            lanes,
            words,
            state: vec![0u64; num_signals * words],
            num_signals,
        })
    }

    fn streams(machine: &PackedMachineProgram) -> [&PackedBlock; 4] {
        [
            &machine.streams.async_reset_comb,
            &machine.streams.comb,
            &machine.streams.tick_next,
            &machine.streams.tick_commit,
        ]
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    /// Set signal `idx`'s bit for one `lane`.
    pub fn set_signal(&mut self, idx: usize, lane: usize, bit: bool) {
        let (w, b) = (lane / 64, lane % 64);
        let slot = idx * self.words + w;
        if bit {
            self.state[slot] |= 1u64 << b;
        } else {
            self.state[slot] &= !(1u64 << b);
        }
    }

    /// Set signal `idx` to the same `bit` on every lane (broadcast).
    pub fn set_signal_all(&mut self, idx: usize, bit: bool) {
        let fill = if bit { u64::MAX } else { 0 };
        for w in 0..self.words {
            self.state[idx * self.words + w] = fill;
        }
    }

    /// Read signal `idx`'s bit on one `lane`.
    pub fn get_signal(&self, idx: usize, lane: usize) -> bool {
        let (w, b) = (lane / 64, lane % 64);
        (self.state[idx * self.words + w] >> b) & 1 == 1
    }

    /// Evaluate a block into a fresh value workspace; returns the per-value-id
    /// word arrays so effects can read them.
    fn eval_block(&self, block: &PackedBlock, work: &mut Vec<u64>, nvals: usize) {
        // `work` is nvals*words; value `v` at `[v*words ..]`.
        for pkt in &block.packets {
            for instr in &pkt.instrs {
                let d = instr.dst.0 * self.words;
                use PackedInstrKind::*;
                match &instr.kind {
                    Signal(s) => {
                        let base = s * self.words;
                        work[d..d + self.words].copy_from_slice(&self.state[base..base + self.words]);
                    }
                    Lit(w) => {
                        // 1-bit literal: all-ones if bit 0 set, else all-zeros.
                        let fill = if w.first().copied().unwrap_or(0) & 1 == 1 { u64::MAX } else { 0 };
                        for k in 0..self.words {
                            work[d + k] = fill;
                        }
                    }
                    Not(a) => {
                        let a = a.0 * self.words;
                        for k in 0..self.words {
                            work[d + k] = !work[a + k];
                        }
                    }
                    And(a, b) => self.binop(work, d, a, b, |x, y| x & y),
                    Or(a, b) => self.binop(work, d, a, b, |x, y| x | y),
                    Xor(a, b) => self.binop(work, d, a, b, |x, y| x ^ y),
                    Mux { cond, then_value, else_value } => {
                        let (c, t, e) = (cond.0 * self.words, then_value.0 * self.words, else_value.0 * self.words);
                        for k in 0..self.words {
                            let cw = work[c + k];
                            work[d + k] = (cw & work[t + k]) | (!cw & work[e + k]);
                        }
                    }
                    _ => unreachable!("validated in new()"),
                }
            }
            let _ = nvals;
        }
    }

    #[inline]
    fn binop(
        &self,
        work: &mut [u64],
        d: usize,
        a: &PackedValueId,
        b: &PackedValueId,
        f: impl Fn(u64, u64) -> u64,
    ) {
        let (a, b) = (a.0 * self.words, b.0 * self.words);
        for k in 0..self.words {
            work[d + k] = f(work[a + k], work[b + k]);
        }
    }

    /// The per-lane "reset asserted" word for a reset (active-high = the signal;
    /// active-low = its complement).
    fn reset_asserted(&self, reset: &PackedReset, out: &mut [u64]) {
        let base = reset.signal * self.words;
        match reset.polarity {
            ResetPolarity::ActiveHigh => out.copy_from_slice(&self.state[base..base + self.words]),
            ResetPolarity::ActiveLow => {
                for k in 0..self.words {
                    out[k] = !self.state[base + k];
                }
            }
        }
    }

    fn nvals(block: &PackedBlock) -> usize {
        block
            .packets
            .iter()
            .flat_map(|p| p.instrs.iter())
            .map(|i| i.dst.0 + 1)
            .max()
            .unwrap_or(0)
    }

    /// Run combinational settle (async-reset-comb then comb), storing comb signals.
    fn settle(&mut self) {
        for blk in [&self.machine.streams.async_reset_comb, &self.machine.streams.comb] {
            let nvals = Self::nvals(blk);
            let mut work = vec![0u64; nvals * self.words];
            // SAFETY-free: eval reads self.state, writes work; then apply effects.
            let block = blk.clone();
            self.eval_block(&block, &mut work, nvals);
            for pkt in &block.packets {
                for eff in &pkt.effects {
                    match eff {
                        PackedEffect::StoreSignal { dst, value } => {
                            let (d, v) = (dst * self.words, value.0 * self.words);
                            self.state[d..d + self.words].copy_from_slice(&work[v..v + self.words]);
                        }
                        PackedEffect::CaptureReg { dst, value, reset } => {
                            // async-reset-comb: immediate conditional store.
                            let v = value.0 * self.words;
                            if let Some(r) = reset {
                                let mut asserted = vec![0u64; self.words];
                                self.reset_asserted(r, &mut asserted);
                                let rv = if r.value.first().copied().unwrap_or(0) & 1 == 1 { u64::MAX } else { 0 };
                                let d = dst * self.words;
                                for k in 0..self.words {
                                    self.state[d + k] = (asserted[k] & rv) | (!asserted[k] & work[v + k]);
                                }
                            }
                        }
                        PackedEffect::MemoryWrite { .. } => {}
                    }
                }
            }
        }
    }

    /// One clock tick: settle, capture register next-states, commit, settle.
    pub fn tick(&mut self) {
        self.settle();
        // Capture tick_next register next-states into a buffer.
        let block = self.machine.streams.tick_next.clone();
        let nvals = Self::nvals(&block);
        let mut work = vec![0u64; nvals * self.words];
        self.eval_block(&block, &mut work, nvals);
        let mut next: HashMap<usize, Vec<u64>> = HashMap::new();
        for pkt in &block.packets {
            for eff in &pkt.effects {
                if let PackedEffect::CaptureReg { dst, value, reset } = eff {
                    let v = value.0 * self.words;
                    let mut nv = work[v..v + self.words].to_vec();
                    if let Some(r) = reset {
                        let mut asserted = vec![0u64; self.words];
                        self.reset_asserted(r, &mut asserted);
                        let rv = if r.value.first().copied().unwrap_or(0) & 1 == 1 { u64::MAX } else { 0 };
                        for k in 0..self.words {
                            nv[k] = (asserted[k] & rv) | (!asserted[k] & nv[k]);
                        }
                    }
                    next.insert(*dst, nv);
                }
            }
        }
        for (dst, nv) in next {
            let d = dst * self.words;
            self.state[d..d + self.words].copy_from_slice(&nv);
        }
        self.settle();
    }

    pub fn tick_many(&mut self, steps: usize) {
        for _ in 0..steps {
            self.tick();
        }
    }

    pub fn signal_count(&self) -> usize {
        self.num_signals
    }
}

// ---------------------------------------------------------------------------
// Bit-parallel AOT: emit `uint64_t` bitwise C over a group loop and let clang
// -O3 auto-vectorize across groups (the SIMD win without hand-coding it).
// ---------------------------------------------------------------------------
#[cfg(feature = "aot")]
mod aot {
    use super::*;
    use std::fmt::Write as _;

    /// C operand reference for an instruction's value-id under `prefix`.
    fn opref(prefix: &str, v: PackedValueId) -> String {
        format!("{prefix}{}", v.0)
    }

    /// Emit one block's instructions as C locals and its effects (comb stores or
    /// register captures). `capture` selects register-capture mode.
    fn emit_block_c(code: &mut String, block: &PackedBlock, prefix: &str, capture: bool, caps: &mut Vec<usize>) {
        for pkt in &block.packets {
            for instr in &pkt.instrs {
                use PackedInstrKind::*;
                let p = prefix;
                let rhs = match &instr.kind {
                    Signal(s) => format!("s[{s}]"),
                    Lit(w) => if w.first().copied().unwrap_or(0) & 1 == 1 { "~0ull".into() } else { "0ull".into() },
                    Not(a) => format!("~{}", opref(p, *a)),
                    And(a, b) => format!("({} & {})", opref(p, *a), opref(p, *b)),
                    Or(a, b) => format!("({} | {})", opref(p, *a), opref(p, *b)),
                    Xor(a, b) => format!("({} ^ {})", opref(p, *a), opref(p, *b)),
                    Mux { cond, then_value, else_value } => {
                        let (c, t, e) = (opref(p, *cond), opref(p, *then_value), opref(p, *else_value));
                        format!("(({c} & {t}) | (~{c} & {e}))")
                    }
                    _ => "0ull".into(), // validated out in new()
                };
                writeln!(code, "  u64 {p}{} = {rhs};", instr.dst.0).ok();
            }
            for eff in &pkt.effects {
                match eff {
                    PackedEffect::StoreSignal { dst, value } if !capture => {
                        writeln!(code, "  s[{dst}] = {};", opref(prefix, *value)).ok();
                    }
                    PackedEffect::CaptureReg { dst, value, reset } if capture => {
                        let v = opref(prefix, *value);
                        let nv = match reset {
                            Some(r) => {
                                let asserted = match r.polarity {
                                    ResetPolarity::ActiveLow => format!("(~s[{}])", r.signal),
                                    _ => format!("s[{}]", r.signal),
                                };
                                let rv = if r.value.first().copied().unwrap_or(0) & 1 == 1 { "~0ull" } else { "0ull" };
                                format!("(({asserted} & {rv}) | (~{asserted} & {v}))")
                            }
                            None => v,
                        };
                        writeln!(code, "  u64 next_{dst} = {nv};").ok();
                        caps.push(*dst);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Generate the bit-parallel C: `tick_bp(st, ng, nc)` with an inner group
    /// loop clang can vectorize (each group = one u64 word = 64 lanes).
    pub fn generate_c(machine: &PackedMachineProgram, num_signals: usize) -> String {
        let mut c = String::from("typedef unsigned long long u64;\n");
        writeln!(c, "void tick_bp(u64* restrict st, long ng, long nc){{").ok();
        c.push_str(" for(long _c=0;_c<nc;_c++){\n");
        c.push_str("  for(long g=0;g<ng;g++){\n");
        writeln!(c, "   u64* restrict s = st + g*{num_signals};").ok();
        let mut caps = Vec::new();
        emit_block_c(&mut c, &machine.streams.comb, "c", false, &mut caps);
        caps.clear();
        emit_block_c(&mut c, &machine.streams.tick_next, "n", true, &mut caps);
        for dst in &caps {
            writeln!(c, "   s[{dst}] = next_{dst};").ok();
        }
        emit_block_c(&mut c, &machine.streams.comb, "d", false, &mut Vec::new());
        c.push_str("  }\n }\n}\n");
        c
    }
}

/// An AOT-compiled bit-parallel gate-level simulator: emits `uint64_t` bitwise C
/// (one u64 word = 64 lanes), compiles with clang -O3 (auto-vectorizing the
/// group loop), and loads it. Same scope as [`BitParallelSimulator`].
#[cfg(feature = "aot")]
pub struct BitParallelAot {
    _lib: libloading::Library,
    tick_fn: extern "C" fn(*mut u64, i64, i64),
    state: Vec<u64>,
    num_signals: usize,
    groups: usize,
    lanes: usize,
}

#[cfg(feature = "aot")]
impl BitParallelAot {
    pub fn compile_lanes(machine: &PackedMachineProgram, lanes: usize) -> Result<Self, ErrorReport> {
        // Reuse the interpreter's scope validation (all 1-bit, gate-level, no mem).
        let _ = BitParallelSimulator::new(machine, 64)?;
        let num_signals = machine.source.signals.len();
        let groups = lanes.max(1).div_ceil(64);
        let c = aot::generate_c(machine, num_signals);

        let stamp = {
            let mut h = 0xcbf29ce484222325u64;
            for byte in c.as_bytes() {
                h = (h ^ *byte as u64).wrapping_mul(0x100000001b3);
            }
            format!("{h:x}")
        };
        let ext = if cfg!(target_os = "macos") { "dylib" } else { "so" };
        let dir = std::env::temp_dir();
        let cpath = dir.join(format!("rrtl_bp_{stamp}.c"));
        let libpath = dir.join(format!("librrtl_bp_{stamp}.{ext}"));
        std::fs::write(&cpath, &c).map_err(|e| bp_err(format!("write C: {e}")))?;
        let cc = std::env::var("CC").unwrap_or_else(|_| "clang".into());
        let out = std::process::Command::new(&cc)
            .args(["-O3", "-shared", "-fPIC", "-o"])
            .arg(&libpath)
            .arg(&cpath)
            .output()
            .map_err(|e| bp_err(format!("spawn {cc}: {e}")))?;
        if !out.status.success() {
            return Err(bp_err(format!("{cc} -O3 failed: {}", String::from_utf8_lossy(&out.stderr))));
        }
        let lib = unsafe { libloading::Library::new(&libpath) }.map_err(|e| bp_err(format!("dlopen: {e}")))?;
        let tick_fn = unsafe {
            let sym: libloading::Symbol<extern "C" fn(*mut u64, i64, i64)> =
                lib.get(b"tick_bp").map_err(|e| bp_err(format!("sym tick_bp: {e}")))?;
            *sym
        };
        Ok(Self { _lib: lib, tick_fn, state: vec![0u64; num_signals * groups], num_signals, groups, lanes: groups * 64 })
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }
    pub fn set_signal(&mut self, lane: usize, idx: usize, bit: bool) {
        let (g, sub) = (lane / 64, lane % 64);
        let slot = g * self.num_signals + idx;
        if bit {
            self.state[slot] |= 1u64 << sub;
        } else {
            self.state[slot] &= !(1u64 << sub);
        }
    }
    pub fn get_signal(&self, lane: usize, idx: usize) -> bool {
        let (g, sub) = (lane / 64, lane % 64);
        (self.state[g * self.num_signals + idx] >> sub) & 1 == 1
    }
    pub fn tick(&mut self) {
        self.tick_many(1);
    }
    pub fn tick_many(&mut self, n: usize) {
        (self.tick_fn)(self.state.as_mut_ptr(), self.groups as i64, n as i64);
    }

    /// Multicore `tick_many`: the group-major state splits into disjoint
    /// group-ranges (groups are fully independent — no shared state, no sync),
    /// each advanced `n` cycles on its own core via rayon. The `tick_bp` fn is
    /// reentrant over its state pointer. ~linear core scaling.
    pub fn tick_many_parallel(&mut self, n: usize) {
        use rayon::prelude::*;
        let f = self.tick_fn; // fn pointers are Send + Sync
        let ns = self.num_signals;
        // ~4 group-chunks per core for load balance; chunk on signal-slot boundary.
        let gpt = (self.groups / (rayon::current_num_threads() * 4)).max(1);
        self.state.par_chunks_mut(ns * gpt).for_each(|chunk| {
            let g = (chunk.len() / ns) as i64;
            f(chunk.as_mut_ptr(), g, n as i64);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lower_to_machine_program, lower_to_packed_program, SimdCpuSimulator};
    use rrtl_core::{compile, uint, Design};

    // A small gate-level netlist (all 1-bit, NAND/NOR/XNOR) must match the SIMD
    // CPU engine on every output of every lane.
    #[test]
    fn bitparallel_matches_simd_cpu() {
        let mut design = Design::new();
        {
            let mut m = design.module("G");
            let clk = m.input("clk", uint(1));
            let a = m.input("a", uint(1));
            let b = m.input("b", uint(1));
            for c in 0..4 {
                let r = m.reg(format!("r{c}"), uint(1));
                m.clock(r, clk);
                let x = match c % 3 {
                    0 => !(r.value() & a.value()),
                    1 => !(r.value() | b.value()),
                    _ => !(r.value() ^ a.value()),
                };
                m.next(r, x);
                let o = m.output(format!("o{c}"), uint(1));
                m.assign(o, r);
            }
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "G").unwrap();
        let machine = lower_to_machine_program(&program);
        let h = |n: &str| compiled.find_module("G").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
        let lanes = 100; // spans two 64-bit words

        let mut bp = BitParallelSimulator::new(&machine, lanes).unwrap();
        let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        cpu.set_signal(h("clk"), &vec![1u128; lanes]).unwrap();
        for lane in 0..lanes {
            bp.set_signal(program.signal_index(h("a")).unwrap(), lane, lane % 2 == 0);
            bp.set_signal(program.signal_index(h("b")).unwrap(), lane, lane % 3 == 0);
        }
        cpu.set_signal(h("a"), &(0..lanes).map(|l| (l % 2 == 0) as u128).collect::<Vec<_>>()).unwrap();
        cpu.set_signal(h("b"), &(0..lanes).map(|l| (l % 3 == 0) as u128).collect::<Vec<_>>()).unwrap();

        for _ in 0..10 {
            bp.tick();
            cpu.tick().unwrap();
        }
        for c in 0..4 {
            let oi = program.signal_index(h(&format!("o{c}"))).unwrap();
            let cv = cpu.get_signal(h(&format!("o{c}"))).unwrap();
            for lane in 0..lanes {
                assert_eq!(bp.get_signal(oi, lane), cv[lane] & 1 == 1, "o{c}@lane{lane}");
            }
        }
    }

    // The bit-parallel AOT (clang -O3 uint64_t bitwise) must match the interpreter.
    #[cfg(feature = "aot")]
    #[test]
    fn bitparallel_aot_matches_interp() {
        let mut design = Design::new();
        {
            let mut m = design.module("G");
            let clk = m.input("clk", uint(1));
            let a = m.input("a", uint(1));
            let b = m.input("b", uint(1));
            for c in 0..5 {
                let r = m.reg(format!("r{c}"), uint(1));
                m.clock(r, clk);
                let x = match c % 3 {
                    0 => !(r.value() & a.value()),
                    1 => !(r.value() | b.value()),
                    _ => !(r.value() ^ a.value()),
                };
                m.next(r, x);
                let o = m.output(format!("o{c}"), uint(1));
                m.assign(o, r);
            }
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "G").unwrap();
        let machine = lower_to_machine_program(&program);
        let h = |n: &str| compiled.find_module("G").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
        let idx = |n: &str| program.signal_index(h(n)).unwrap();
        let lanes = 130;
        let mut interp = BitParallelSimulator::new(&machine, lanes).unwrap();
        let mut aotsim = super::BitParallelAot::compile_lanes(&machine, lanes).unwrap();
        for lane in 0..lanes {
            let (av, bv) = (lane % 2 == 0, lane % 3 == 0);
            interp.set_signal(idx("a"), lane, av);
            interp.set_signal(idx("b"), lane, bv);
            aotsim.set_signal(lane, idx("a"), av);
            aotsim.set_signal(lane, idx("b"), bv);
        }
        for _ in 0..10 {
            interp.tick();
            aotsim.tick();
        }
        for c in 0..5 {
            let oi = idx(&format!("o{c}"));
            for lane in 0..lanes {
                assert_eq!(interp.get_signal(oi, lane), aotsim.get_signal(lane, oi), "o{c}@{lane}");
            }
        }
    }
}
