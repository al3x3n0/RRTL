//! Design-as-data interpreter encoding for the GPU batch simulator.
//!
//! The existing GPU path codegens a straight-line WGSL kernel per design, so
//! shader size scales with (worse than) design size — a 642-signal design
//! generates a 115 MB shader that hangs the Metal compiler. This module encodes
//! the packed machine program into compact data buffers that a single
//! fixed-size interpreter kernel executes, so shader size is O(1) regardless of
//! design size (and because every lane runs the identical instruction stream,
//! control flow is uniform across the workgroup — zero divergence).
//!
//! [`InterpProgram::encode`] produces the buffers; [`InterpRunner`] is a CPU
//! reference interpreter of those buffers used to validate the encoding and pin
//! the exact opcode semantics before they are transliterated into WGSL.
//!
//! v1 scope: signals up to 32 bits (one limb), no memories, no resets, no
//! `Concat`/`MemRead`. Anything else is rejected by `encode`.

use rrtl_ir::{Diagnostic, ErrorReport};
use rrtl_sim_ir::{
    lower_to_machine_program, PackedBlock, PackedEffect, PackedInstrKind, PackedMachineProgram,
    PackedProgram,
};

pub const OP_LIT: u32 = 0;
pub const OP_SIGNAL: u32 = 1;
pub const OP_NOT: u32 = 2;
pub const OP_AND: u32 = 3;
pub const OP_OR: u32 = 4;
pub const OP_XOR: u32 = 5;
pub const OP_ADD: u32 = 6;
pub const OP_SUB: u32 = 7;
pub const OP_MUL: u32 = 8;
pub const OP_EQ: u32 = 9;
pub const OP_NE: u32 = 10;
pub const OP_LT_U: u32 = 11;
pub const OP_LT_S: u32 = 12;
pub const OP_MUX: u32 = 13;
pub const OP_SLICE: u32 = 14;
pub const OP_ZEXT: u32 = 15;
pub const OP_SEXT: u32 = 16;
pub const OP_TRUNC: u32 = 17;
pub const OP_CAST: u32 = 18;
/// Effect: write `value` (field `a`) to signal storage at word-offset `field1`.
pub const OP_STORE_SIGNAL: u32 = 19;
/// Effect: capture `value` (field `a`) as register at storage offset `field1`
/// into the shadow next-state (committed at end of cycle).
pub const OP_CAPTURE_REG: u32 = 20;

/// Words per encoded record: `[op, dst_or_offset, width, a, b, c]`.
pub const RECORD_WORDS: usize = 6;

/// Streams, indexed `[async_reset_comb, comb, tick_next, tick_commit]`.
#[derive(Clone, Copy, Debug)]
pub enum InterpStream {
    AsyncResetComb = 0,
    Comb = 1,
    TickNext = 2,
    TickCommit = 3,
}

/// Encoded program: one flat record array per stream block plus the per-lane
/// value-workspace size. Signal storage is WordMajor (`offset * lanes + lane`).
#[derive(Clone, Debug, Default)]
pub struct InterpProgram {
    pub num_values: usize,
    pub total_signal_words: usize,
    pub blocks: [Vec<u32>; 4],
}

fn interp_error(message: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![Diagnostic::new("E_GPU_INTERP", message)])
}

impl InterpProgram {
    pub fn encode_design(program: &PackedProgram) -> Result<Self, ErrorReport> {
        Self::encode(&lower_to_machine_program(program))
    }

    pub fn encode(machine: &PackedMachineProgram) -> Result<Self, ErrorReport> {
        let source = &machine.source;
        if !source.memories.is_empty() {
            return Err(interp_error("interpreter kernel v1 does not support memories"));
        }
        for signal in &source.signals {
            if signal.layout.limbs != 1 {
                return Err(interp_error(format!(
                    "interpreter kernel v1 supports signals up to 32 bits; `{}` is {} bits",
                    signal.name, signal.layout.width
                )));
            }
        }

        let streams = [
            &machine.streams.async_reset_comb,
            &machine.streams.comb,
            &machine.streams.tick_next,
            &machine.streams.tick_commit,
        ];
        let mut blocks: [Vec<u32>; 4] = Default::default();
        for (slot, block) in blocks.iter_mut().zip(streams) {
            *slot = encode_block(block, source)?;
        }
        let num_values = streams
            .iter()
            .map(|block| block_value_count(block))
            .max()
            .unwrap_or(0);

        Ok(Self {
            num_values,
            total_signal_words: source.total_signal_words,
            blocks,
        })
    }

    pub fn block(&self, stream: InterpStream) -> &[u32] {
        &self.blocks[stream as usize]
    }

    fn captured_offsets(&self) -> Vec<usize> {
        let recs = self.block(InterpStream::TickNext);
        (0..recs.len() / RECORD_WORDS)
            .filter(|&r| recs[r * RECORD_WORDS] == OP_CAPTURE_REG)
            .map(|r| recs[r * RECORD_WORDS + 1] as usize)
            .collect()
    }
}

fn block_value_count(block: &PackedBlock) -> usize {
    block
        .packets
        .iter()
        .flat_map(|packet| packet.instrs.iter())
        .map(|instr| instr.dst.0 + 1)
        .max()
        .unwrap_or(0)
}

fn encode_block(block: &PackedBlock, source: &PackedProgram) -> Result<Vec<u32>, ErrorReport> {
    // Pre-pass: width of every value produced in this block, for ops that need a
    // source operand's width (Sext, signed Lt).
    let mut value_width = vec![0u32; block_value_count(block)];
    for instr in block.packets.iter().flat_map(|p| p.instrs.iter()) {
        value_width[instr.dst.0] = instr.ty.width;
    }

    let mut out = Vec::new();
    for packet in &block.packets {
        for instr in &packet.instrs {
            let dst = instr.dst.0 as u32;
            let width = instr.ty.width;
            let mut rec = |op, a, b, c| push_record(&mut out, op, dst, width, a, b, c);
            match &instr.kind {
                PackedInstrKind::Lit(words) => rec(OP_LIT, words.first().copied().unwrap_or(0), 0, 0),
                PackedInstrKind::Signal(index) => {
                    rec(OP_SIGNAL, source.signals[*index].layout.offset as u32, 0, 0)
                }
                PackedInstrKind::Not(v) => rec(OP_NOT, v.0 as u32, 0, 0),
                PackedInstrKind::Zext(v) => rec(OP_ZEXT, v.0 as u32, 0, 0),
                PackedInstrKind::Trunc(v) => rec(OP_TRUNC, v.0 as u32, 0, 0),
                PackedInstrKind::Cast(v) => rec(OP_CAST, v.0 as u32, 0, 0),
                PackedInstrKind::Sext(v) => rec(OP_SEXT, v.0 as u32, 0, value_width[v.0]),
                PackedInstrKind::And(l, r) => rec(OP_AND, l.0 as u32, r.0 as u32, 0),
                PackedInstrKind::Or(l, r) => rec(OP_OR, l.0 as u32, r.0 as u32, 0),
                PackedInstrKind::Xor(l, r) => rec(OP_XOR, l.0 as u32, r.0 as u32, 0),
                PackedInstrKind::Add(l, r) => rec(OP_ADD, l.0 as u32, r.0 as u32, 0),
                PackedInstrKind::Sub(l, r) => rec(OP_SUB, l.0 as u32, r.0 as u32, 0),
                PackedInstrKind::Mul(l, r) => rec(OP_MUL, l.0 as u32, r.0 as u32, 0),
                PackedInstrKind::Eq(l, r) => rec(OP_EQ, l.0 as u32, r.0 as u32, 0),
                PackedInstrKind::Ne(l, r) => rec(OP_NE, l.0 as u32, r.0 as u32, 0),
                PackedInstrKind::Mux {
                    cond,
                    then_value,
                    else_value,
                } => rec(OP_MUX, cond.0 as u32, then_value.0 as u32, else_value.0 as u32),
                PackedInstrKind::Slice { value, lsb } => rec(OP_SLICE, value.0 as u32, *lsb, 0),
                PackedInstrKind::Lt { lhs, rhs, signed } => rec(
                    if *signed { OP_LT_S } else { OP_LT_U },
                    lhs.0 as u32,
                    rhs.0 as u32,
                    value_width[lhs.0],
                ),
                PackedInstrKind::Concat(_) => {
                    return Err(interp_error("interpreter kernel v1 does not support Concat"))
                }
                PackedInstrKind::MemRead { .. } => {
                    return Err(interp_error("interpreter kernel v1 does not support MemRead"))
                }
            }
        }
        for effect in &packet.effects {
            match effect {
                PackedEffect::StoreSignal { dst, value } => {
                    let layout = source.signals[*dst].layout;
                    push_record(
                        &mut out,
                        OP_STORE_SIGNAL,
                        layout.offset as u32,
                        layout.width,
                        value.0 as u32,
                        0,
                        0,
                    );
                }
                PackedEffect::CaptureReg { dst, value, reset } => {
                    if reset.is_some() {
                        return Err(interp_error(
                            "interpreter kernel v1 does not support register resets",
                        ));
                    }
                    let layout = source.signals[*dst].layout;
                    push_record(
                        &mut out,
                        OP_CAPTURE_REG,
                        layout.offset as u32,
                        layout.width,
                        value.0 as u32,
                        0,
                        0,
                    );
                }
                PackedEffect::MemoryWrite { .. } => {
                    return Err(interp_error(
                        "interpreter kernel v1 does not support memory writes",
                    ))
                }
            }
        }
    }
    Ok(out)
}

fn push_record(out: &mut Vec<u32>, op: u32, field1: u32, width: u32, a: u32, b: u32, c: u32) {
    out.extend_from_slice(&[op, field1, width, a, b, c]);
}

fn mask_of(width: u32) -> u32 {
    if width == 0 {
        0
    } else if width >= 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    }
}

/// CPU reference interpreter of an [`InterpProgram`]. Mirrors exactly what the
/// WGSL kernel will do (one thread per lane), so it both validates the encoding
/// and serves as the executable spec for the kernel.
pub struct InterpRunner {
    program: InterpProgram,
    lanes: usize,
    storage: Vec<u32>,
    values: Vec<u32>,
    reg_next: Vec<u32>,
    captured_offsets: Vec<usize>,
}

impl InterpRunner {
    pub fn new(program: InterpProgram, lanes: usize) -> Self {
        let storage = vec![0u32; program.total_signal_words * lanes];
        let values = vec![0u32; program.num_values * lanes];
        let reg_next = storage.clone();
        let captured_offsets = program.captured_offsets();
        Self {
            program,
            lanes,
            storage,
            values,
            reg_next,
            captured_offsets,
        }
    }

    pub fn set_signal(&mut self, offset: usize, lane_values: &[u32]) {
        for (lane, &value) in lane_values.iter().enumerate() {
            self.storage[offset * self.lanes + lane] = value;
        }
    }

    pub fn get_signal(&self, offset: usize) -> Vec<u32> {
        (0..self.lanes)
            .map(|lane| self.storage[offset * self.lanes + lane])
            .collect()
    }

    pub fn eval_combinational(&mut self) {
        self.eval_block(InterpStream::AsyncResetComb);
        self.eval_block(InterpStream::Comb);
    }

    pub fn tick(&mut self) {
        self.eval_combinational();
        self.eval_block(InterpStream::TickNext);
        self.commit();
        self.eval_combinational();
    }

    pub fn tick_many(&mut self, steps: usize) {
        for _ in 0..steps {
            self.tick();
        }
    }

    fn commit(&mut self) {
        for &offset in &self.captured_offsets {
            for lane in 0..self.lanes {
                let i = offset * self.lanes + lane;
                self.storage[i] = self.reg_next[i];
            }
        }
    }

    fn eval_block(&mut self, stream: InterpStream) {
        let Self {
            program,
            lanes,
            storage,
            values,
            reg_next,
            ..
        } = self;
        let lanes = *lanes;
        let recs = program.block(stream);
        for record in recs.chunks_exact(RECORD_WORDS) {
            let (op, field1, width) = (record[0], record[1] as usize, record[2]);
            let (a, b, c) = (record[3], record[4], record[5]);
            let mask = mask_of(width);
            for lane in 0..lanes {
                let val = |id: u32| values[id as usize * lanes + lane];
                let result = match op {
                    OP_LIT => a & mask,
                    OP_SIGNAL => storage[a as usize * lanes + lane] & mask,
                    OP_NOT => !val(a) & mask,
                    OP_AND => val(a) & val(b) & mask,
                    OP_OR => (val(a) | val(b)) & mask,
                    OP_XOR => (val(a) ^ val(b)) & mask,
                    OP_ADD => val(a).wrapping_add(val(b)) & mask,
                    OP_SUB => val(a).wrapping_sub(val(b)) & mask,
                    OP_MUL => val(a).wrapping_mul(val(b)) & mask,
                    OP_EQ => u32::from(val(a) == val(b)),
                    OP_NE => u32::from(val(a) != val(b)),
                    OP_LT_U => u32::from(val(a) < val(b)),
                    OP_LT_S => {
                        let sign = 1u32 << (c - 1);
                        let (l, r) = (val(a), val(b));
                        let (ls, rs) = (l & sign != 0, r & sign != 0);
                        u32::from(if ls != rs { ls } else { l < r })
                    }
                    OP_MUX => {
                        if val(a) & 1 != 0 {
                            val(b)
                        } else {
                            val(c)
                        }
                    }
                    OP_SLICE => (val(a) >> b) & mask,
                    OP_ZEXT | OP_TRUNC | OP_CAST => val(a) & mask,
                    OP_SEXT => {
                        let src_mask = mask_of(c);
                        let sign = 1u32 << (c - 1);
                        let v = val(a) & src_mask;
                        (if v & sign != 0 { v | !src_mask } else { v }) & mask
                    }
                    OP_STORE_SIGNAL => {
                        storage[field1 * lanes + lane] = val(a) & mask;
                        continue;
                    }
                    OP_CAPTURE_REG => {
                        reg_next[field1 * lanes + lane] = val(a) & mask;
                        continue;
                    }
                    _ => unreachable!("unknown interp opcode {op}"),
                };
                values[field1 * lanes + lane] = result;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rrtl_core::{compile, lit_u, mux, uint, Design};
    use rrtl_sim_ir::{lower_to_packed_program, SimdCpuSimulator};

    /// Builds a design exercising the v1 op set (mul/add registers + a small
    /// mux/eq FSM-ish output), encodes it, and asserts the reference interpreter
    /// is bit-identical to SimdCpuSimulator over several cycles and lanes.
    #[test]
    fn interp_runner_matches_simd_cpu() {
        let mut design = Design::new();
        let (clk, din, acc_out, flag_out);
        {
            let mut m = design.module("Top");
            clk = m.input("clk", uint(1));
            din = m.input("din", uint(16));
            let acc = m.reg("acc", uint(16));
            let big = m.reg("big", uint(1));
            m.clock(acc, clk);
            m.clock(big, clk);
            // acc' = acc * 3 + din ; big' = (acc == din) ? 1 : (acc' bit...)
            let nxt = m.wire("nxt", uint(16));
            m.assign(nxt, acc * lit_u(3, 16) + din);
            m.next(acc, nxt);
            m.next(big, mux(acc.value().eq_expr(din), lit_u(1, 1), big));
            acc_out = m.output("acc_out", uint(16));
            flag_out = m.output("flag_out", uint(1));
            m.assign(acc_out, acc);
            m.assign(flag_out, big);
        }

        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();
        let offset = |sig: rrtl_ir::Signal| {
            program.signals[program.signal_index(sig).unwrap()]
                .layout
                .offset
        };

        let lanes = 5;
        let encoded = InterpProgram::encode_design(&program).unwrap();
        let mut runner = InterpRunner::new(encoded, lanes);
        let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();

        let clk_u: Vec<u128> = vec![1; lanes];
        runner.set_signal(offset(clk), &vec![1u32; lanes]);
        cpu.set_signal(clk, &clk_u).unwrap();

        for cycle in 0..8u32 {
            let din_u32: Vec<u32> = (0..lanes as u32)
                .map(|l| (cycle.wrapping_mul(101).wrapping_add(l * 7)) & 0xffff)
                .collect();
            let din_u128: Vec<u128> = din_u32.iter().map(|&v| v as u128).collect();
            runner.set_signal(offset(din), &din_u32);
            cpu.set_signal(din, &din_u128).unwrap();

            runner.tick();
            cpu.tick();

            let r_acc = runner.get_signal(offset(acc_out));
            let c_acc: Vec<u32> = cpu.get_signal(acc_out).unwrap().iter().map(|&v| v as u32).collect();
            assert_eq!(r_acc, c_acc, "acc mismatch at cycle {cycle}");
            let r_flag = runner.get_signal(offset(flag_out));
            let c_flag: Vec<u32> = cpu.get_signal(flag_out).unwrap().iter().map(|&v| v as u32).collect();
            assert_eq!(r_flag, c_flag, "flag mismatch at cycle {cycle}");
        }
    }

    /// The design that makes the straight-line codegen emit a 115 MB WGSL
    /// shader (642 signals) encodes to a small buffer that grows linearly with
    /// op count — the whole point of design-as-data.
    #[test]
    fn interp_encoding_is_compact_for_large_design() {
        let mut design = Design::new();
        {
            let mut m = design.module("Wide");
            let clk = m.input("clk", uint(1));
            let din = m.input("din", uint(32));
            for lane in 0..64 {
                let acc = m.reg(format!("acc{lane}"), uint(32));
                m.clock(acc, clk);
                let mut prev = acc;
                for stage in 0..8 {
                    let w = m.wire(format!("w{lane}_{stage}"), uint(32));
                    m.assign(w, prev * lit_u(0x9e37_79b9, 32) + din);
                    prev = w;
                }
                m.next(acc, prev + acc);
                let o = m.output(format!("o{lane}"), uint(32));
                m.assign(o, acc);
            }
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Wide").unwrap();
        assert!(program.signals.len() > 600);
        let encoded = InterpProgram::encode_design(&program).unwrap();
        let total_words: usize = encoded.blocks.iter().map(|b| b.len()).sum();
        // ~115 MB of WGSL for the codegen path; the interpreter buffer is KBs.
        assert!(
            total_words * 4 < 1_000_000,
            "encoded buffer unexpectedly large: {} bytes",
            total_words * 4
        );
    }
}
