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
//! Scope: signals and memory data up to 32 bits (one limb), no resets, no
//! `Concat`. Anything else is rejected by `encode`. Memories (RAM / register
//! files) are supported.

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
/// Instr: read memory word `(mem_offset + addr) * lanes + lane`. Record fields:
/// `a` = addr value id, `b` = mem_offset (word base), `c` = depth (bound).
pub const OP_MEM_READ: u32 = 21;
/// Effect: conditional memory write. Record fields: `field1` = mem_offset,
/// `width` = depth, `a` = enable value, `b` = addr value, `c` = data value.
pub const OP_MEM_WRITE: u32 = 22;
/// Instr: bit-concatenate. `a` = aux offset of the operand list, `b` = operand
/// count. Each aux operand is `[value_id, width]`; operands are joined with the
/// last operand in the least-significant bits.
pub const OP_CONCAT: u32 = 23;
/// Effect (async-reset stream): if the reset at aux offset `b` is asserted, set
/// signal `field1` to that reset's value. Used for asynchronous resets.
pub const OP_ASYNC_RESET: u32 = 24;

/// Sentinel reset id (no reset) in an [`OP_CAPTURE_REG`] record's `b` field.
pub const NO_RESET: u32 = u32::MAX;

/// Words per encoded record: `[op, dst_or_offset, width, a, b, c]`.
pub const RECORD_WORDS: usize = 6;

/// Default compute workgroup size for the interpreter kernel. Larger groups
/// (256) measured ~32% faster than 64 on Apple GPU for this memory-bound kernel.
pub const INTERP_DEFAULT_WORKGROUP: u32 = 256;

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
    /// Max limbs (32-bit words) of any value; the per-lane value workspace
    /// stores every value zero-extended to this width.
    pub max_limbs: usize,
    pub total_signal_words: usize,
    pub total_memory_words: usize,
    pub blocks: [Vec<u32>; 4],
    /// Side data for variable-size operands: Concat operand lists
    /// (`[value_id, width]` pairs) and reset entries
    /// (`[reset_signal_offset, reset_value, polarity]`, polarity 0=high/1=low).
    pub aux: Vec<u32>,
}

fn interp_error(message: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![Diagnostic::new("E_GPU_INTERP", message)])
}

/// Number of 32-bit limbs for a bit width (min 1).
fn limbs_of(width: u32) -> usize {
    ((width as usize).div_ceil(32)).max(1)
}

impl InterpProgram {
    pub fn encode_design(program: &PackedProgram) -> Result<Self, ErrorReport> {
        Self::encode(&lower_to_machine_program(program))
    }

    pub fn encode(machine: &PackedMachineProgram) -> Result<Self, ErrorReport> {
        let source = &machine.source;
        for memory in &source.memories {
            if memory.data_layout.limbs != 1 {
                return Err(interp_error(format!(
                    "interpreter supports memories up to 32-bit data; `{}` is {} bits",
                    memory.name, memory.data_layout.width
                )));
            }
        }

        let streams = [
            &machine.streams.async_reset_comb,
            &machine.streams.comb,
            &machine.streams.tick_next,
            &machine.streams.tick_commit,
        ];
        let mut aux = Vec::new();
        let mut blocks: [Vec<u32>; 4] = Default::default();
        for (index, (slot, block)) in blocks.iter_mut().zip(streams).enumerate() {
            // streams[0] is async_reset_comb, whose register captures are
            // immediate conditional stores rather than next-state captures.
            *slot = encode_block(block, source, &mut aux, index == 0)?;
        }
        let num_values = streams
            .iter()
            .map(|block| block_value_count(block))
            .max()
            .unwrap_or(0);
        let max_limbs = streams
            .iter()
            .flat_map(|block| block.packets.iter())
            .flat_map(|packet| packet.instrs.iter())
            .map(|instr| limbs_of(instr.ty.width))
            .max()
            .unwrap_or(1)
            .max(1);

        Ok(Self {
            num_values,
            max_limbs,
            total_signal_words: source.total_signal_words,
            total_memory_words: source.total_memory_words_per_lane,
            blocks,
            aux,
        })
    }

    pub fn block(&self, stream: InterpStream) -> &[u32] {
        &self.blocks[stream as usize]
    }

    /// Register `(storage_offset, limbs)` captured in tick_next, for commit.
    fn captured_offsets(&self) -> Vec<(usize, usize)> {
        let recs = self.block(InterpStream::TickNext);
        (0..recs.len() / RECORD_WORDS)
            .filter(|&r| recs[r * RECORD_WORDS] == OP_CAPTURE_REG)
            .map(|r| {
                (
                    recs[r * RECORD_WORDS + 1] as usize,
                    limbs_of(recs[r * RECORD_WORDS + 2]),
                )
            })
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

/// Appends a reset entry `[reset_signal_offset, polarity, value_limb0..]` to
/// `aux` and returns its offset. polarity: 0 = active-high, 1 = active-low.
/// `limbs` reset-value words are stored.
fn push_reset(
    aux: &mut Vec<u32>,
    source: &PackedProgram,
    reset: &rrtl_sim_ir::PackedReset,
    limbs: usize,
) -> u32 {
    let offset = aux.len() as u32;
    let polarity = match reset.polarity {
        rrtl_ir::ResetPolarity::ActiveHigh => 0,
        rrtl_ir::ResetPolarity::ActiveLow => 1,
    };
    aux.push(source.signals[reset.signal].layout.offset as u32);
    aux.push(polarity);
    for limb in 0..limbs {
        aux.push(reset.value.get(limb).copied().unwrap_or(0));
    }
    offset
}

fn encode_block(
    block: &PackedBlock,
    source: &PackedProgram,
    aux: &mut Vec<u32>,
    is_async_reset: bool,
) -> Result<Vec<u32>, ErrorReport> {
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
                PackedInstrKind::Lit(words) => {
                    // Immediate limbs live in aux (a = offset); the value's width
                    // determines how many limbs are read back.
                    let aux_offset = aux.len() as u32;
                    for limb in 0..limbs_of(width) {
                        aux.push(words.get(limb).copied().unwrap_or(0));
                    }
                    rec(OP_LIT, aux_offset, 0, 0);
                }
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
                PackedInstrKind::MemRead { memory, addr } => {
                    let mem = &source.memories[*memory];
                    rec(OP_MEM_READ, addr.0 as u32, mem.offset as u32, mem.depth as u32)
                }
                PackedInstrKind::Concat(parts) => {
                    let aux_offset = aux.len() as u32;
                    for part in parts {
                        aux.push(part.0 as u32);
                        aux.push(value_width[part.0]);
                    }
                    rec(OP_CONCAT, aux_offset, parts.len() as u32, 0);
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
                    let layout = source.signals[*dst].layout;
                    if is_async_reset {
                        // Immediate conditional store while reset is asserted.
                        let reset = reset.as_ref().expect("async-reset capture has a reset");
                        let reset_off = push_reset(aux, source, reset, limbs_of(layout.width));
                        push_record(
                            &mut out,
                            OP_ASYNC_RESET,
                            layout.offset as u32,
                            layout.width,
                            0,
                            reset_off,
                            0,
                        );
                    } else {
                        let reset_id = reset
                            .as_ref()
                            .map(|reset| push_reset(aux, source, reset, limbs_of(layout.width)))
                            .unwrap_or(NO_RESET);
                        push_record(
                            &mut out,
                            OP_CAPTURE_REG,
                            layout.offset as u32,
                            layout.width,
                            value.0 as u32,
                            reset_id,
                            0,
                        );
                    }
                }
                PackedEffect::MemoryWrite {
                    memory,
                    enable,
                    addr,
                    data,
                } => {
                    let mem = &source.memories[*memory];
                    push_record(
                        &mut out,
                        OP_MEM_WRITE,
                        mem.offset as u32,
                        mem.depth as u32,
                        enable.0 as u32,
                        addr.0 as u32,
                        data.0 as u32,
                    );
                }
            }
        }
    }
    Ok(out)
}

fn push_record(out: &mut Vec<u32>, op: u32, field1: u32, width: u32, a: u32, b: u32, c: u32) {
    out.extend_from_slice(&[op, field1, width, a, b, c]);
}

fn mask128(width: u32) -> u128 {
    if width == 0 {
        0
    } else if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
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
    memories: Vec<u32>,
    captured_offsets: Vec<(usize, usize)>,
}

impl InterpRunner {
    pub fn new(program: InterpProgram, lanes: usize) -> Self {
        let storage = vec![0u32; program.total_signal_words * lanes];
        let values = vec![0u32; program.num_values * program.max_limbs * lanes];
        let reg_next = storage.clone();
        let memories = vec![0u32; program.total_memory_words * lanes];
        let captured_offsets = program.captured_offsets();
        Self {
            program,
            lanes,
            storage,
            values,
            reg_next,
            memories,
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

    /// Sets a multi-limb signal (`limbs` words at `offset`) from per-lane values.
    pub fn set_signal_wide(&mut self, offset: usize, limbs: usize, lane_values: &[u128]) {
        for (lane, &value) in lane_values.iter().enumerate() {
            for l in 0..limbs {
                self.storage[(offset + l) * self.lanes + lane] = (value >> (32 * l)) as u32;
            }
        }
    }

    /// Reads a multi-limb signal (`limbs` words at `offset`) as per-lane values.
    pub fn get_signal_wide(&self, offset: usize, limbs: usize) -> Vec<u128> {
        (0..self.lanes)
            .map(|lane| {
                let mut v = 0u128;
                for l in 0..limbs {
                    v |= (self.storage[(offset + l) * self.lanes + lane] as u128) << (32 * l);
                }
                v
            })
            .collect()
    }

    pub fn eval_combinational(&mut self) {
        self.eval_block(InterpStream::AsyncResetComb);
        self.eval_block(InterpStream::Comb);
    }

    pub fn tick(&mut self) {
        self.eval_combinational();
        self.eval_block(InterpStream::TickNext);
        self.eval_block(InterpStream::TickCommit); // memory writes
        self.commit();
        self.eval_combinational();
    }

    pub fn tick_many(&mut self, steps: usize) {
        if steps == 0 {
            return;
        }
        // Fused: the trailing comb of one cycle and the leading comb of the next
        // settle from the same register state (redundant), so settle once up
        // front and once after each commit — halving combinational work while
        // leaving the final state identical to `steps` separate `tick`s.
        self.eval_combinational();
        for _ in 0..steps {
            self.eval_block(InterpStream::TickNext);
            self.eval_block(InterpStream::TickCommit); // memory writes
            self.commit();
            self.eval_combinational();
        }
    }

    fn commit(&mut self) {
        for &(offset, limbs) in &self.captured_offsets {
            for l in 0..limbs {
                for lane in 0..self.lanes {
                    let i = (offset + l) * self.lanes + lane;
                    self.storage[i] = self.reg_next[i];
                }
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
            memories,
            ..
        } = self;
        let lanes = *lanes;
        let aux = &program.aux;
        let recs = program.block(stream);
        for record in recs.chunks_exact(RECORD_WORDS) {
            let (op, field1, width) = (record[0], record[1] as usize, record[2]);
            let (a, b, c) = (record[3], record[4], record[5]);
            let ml = program.max_limbs;
            let mask = mask128(width);
            let limbs = limbs_of(width);
            for lane in 0..lanes {
                // Every value is stored zero-extended to `ml` limbs, so reading
                // `ml` limbs yields the correct u128 regardless of the value's
                // own width.
                let read = |id: u32| -> u128 {
                    let base = id as usize * ml;
                    let mut v = 0u128;
                    for l in 0..ml {
                        v |= (values[(base + l) * lanes + lane] as u128) << (32 * l);
                    }
                    v
                };
                let reset_asserted = |reset_id: u32| -> bool {
                    let rs = aux[reset_id as usize] as usize;
                    let bit = storage[rs * lanes + lane] & 1;
                    if aux[reset_id as usize + 1] == 0 {
                        bit != 0
                    } else {
                        bit == 0
                    }
                };
                let reset_value = |reset_id: u32| -> u128 {
                    let base = reset_id as usize + 2;
                    let mut v = 0u128;
                    for l in 0..limbs {
                        v |= (aux[base + l] as u128) << (32 * l);
                    }
                    v & mask
                };
                let result: u128 = match op {
                    OP_LIT => {
                        let mut v = 0u128;
                        for l in 0..limbs {
                            v |= (aux[a as usize + l] as u128) << (32 * l);
                        }
                        v & mask
                    }
                    OP_SIGNAL => {
                        let mut v = 0u128;
                        for l in 0..limbs {
                            v |= (storage[(a as usize + l) * lanes + lane] as u128) << (32 * l);
                        }
                        v & mask
                    }
                    OP_NOT => !read(a) & mask,
                    OP_AND => read(a) & read(b) & mask,
                    OP_OR => (read(a) | read(b)) & mask,
                    OP_XOR => (read(a) ^ read(b)) & mask,
                    OP_ADD => read(a).wrapping_add(read(b)) & mask,
                    OP_SUB => read(a).wrapping_sub(read(b)) & mask,
                    OP_MUL => read(a).wrapping_mul(read(b)) & mask,
                    OP_EQ => u128::from(read(a) == read(b)),
                    OP_NE => u128::from(read(a) != read(b)),
                    OP_LT_U => u128::from(read(a) < read(b)),
                    OP_LT_S => {
                        let sign = 1u128 << (c - 1);
                        let (l, r) = (read(a), read(b));
                        let (ls, rs) = (l & sign != 0, r & sign != 0);
                        u128::from(if ls != rs { ls } else { l < r })
                    }
                    OP_MUX => {
                        if read(a) & 1 != 0 {
                            read(b)
                        } else {
                            read(c)
                        }
                    }
                    OP_SLICE => (read(a) >> b) & mask,
                    OP_ZEXT | OP_TRUNC | OP_CAST => read(a) & mask,
                    OP_SEXT => {
                        let src_mask = mask128(c);
                        let sign = 1u128 << (c - 1);
                        let v = read(a) & src_mask;
                        (if v & sign != 0 { v | !src_mask } else { v }) & mask
                    }
                    OP_CONCAT => {
                        let mut result = 0u128;
                        let mut offset = 0u32;
                        for k in (0..b).rev() {
                            let vid = aux[(a + k * 2) as usize];
                            let w = aux[(a + k * 2 + 1) as usize];
                            let part = read(vid) & mask128(w);
                            if offset < 128 {
                                result |= part << offset;
                            }
                            offset += w;
                        }
                        result & mask
                    }
                    OP_MEM_READ => {
                        // a = addr value, b = mem_offset, c = depth (1-limb data).
                        let addr = read(a) as usize;
                        if addr < c as usize {
                            (memories[(b as usize + addr) * lanes + lane] as u128) & mask
                        } else {
                            0
                        }
                    }
                    OP_STORE_SIGNAL => {
                        let v = read(a) & mask;
                        for l in 0..limbs {
                            storage[(field1 + l) * lanes + lane] = (v >> (32 * l)) as u32;
                        }
                        continue;
                    }
                    OP_CAPTURE_REG => {
                        // b = reset id (NO_RESET if none).
                        let v = if b != NO_RESET && reset_asserted(b) {
                            reset_value(b)
                        } else {
                            read(a) & mask
                        };
                        for l in 0..limbs {
                            reg_next[(field1 + l) * lanes + lane] = (v >> (32 * l)) as u32;
                        }
                        continue;
                    }
                    OP_ASYNC_RESET => {
                        if reset_asserted(b) {
                            let v = reset_value(b);
                            for l in 0..limbs {
                                storage[(field1 + l) * lanes + lane] = (v >> (32 * l)) as u32;
                            }
                        }
                        continue;
                    }
                    OP_MEM_WRITE => {
                        // field1 = mem_offset, width = depth, a = enable, b = addr,
                        // c = data (1-limb data).
                        let addr = read(b) as usize;
                        if read(a) & 1 != 0 && addr < width as usize {
                            memories[(field1 + addr) * lanes + lane] = read(c) as u32;
                        }
                        continue;
                    }
                    _ => unreachable!("unknown interp opcode {op}"),
                };
                let base = field1 * ml;
                for l in 0..ml {
                    values[(base + l) * lanes + lane] = (result >> (32 * l)) as u32;
                }
            }
        }
    }
}

/// Fixed-size WGSL interpreter kernel. The design is uploaded as data (the
/// `code` buffer), so this shader never changes with design size — one thread
/// per lane, all lanes running the identical instruction stream (uniform PC,
/// zero divergence). Transliterates [`InterpRunner`]. `{WG}` is the workgroup size.
const INTERP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> sig: array<u32>;
@group(0) @binding(1) var<storage, read_write> reg_next: array<u32>;
@group(0) @binding(2) var<storage, read_write> values: array<u32>;
@group(0) @binding(3) var<storage, read> code: array<u32>;
@group(0) @binding(4) var<storage, read> captured: array<u32>;
@group(0) @binding(5) var<storage, read> params: array<u32>;
@group(0) @binding(6) var<storage, read_write> mem: array<u32>;
@group(0) @binding(7) var<storage, read> aux: array<u32>;

fn mask_of(width: u32) -> u32 {
  if (width == 0u) { return 0u; }
  if (width >= 32u) { return 0xffffffffu; }
  return (1u << width) - 1u;
}

fn vget(id: u32, lanes: u32, lane: u32) -> u32 { return values[id * lanes + lane]; }

fn reset_on(reset_id: u32, lanes: u32, lane: u32) -> bool {
  let bit = sig[aux[reset_id] * lanes + lane] & 1u;
  if (aux[reset_id + 1u] == 0u) { return bit != 0u; }
  return bit == 0u;
}

fn limb_mask(width: u32, limb: u32) -> u32 {
  let lo = limb * 32u;
  if (width <= lo) { return 0u; }
  if (width >= lo + 32u) { return 0xffffffffu; }
  return (1u << (width - lo)) - 1u;
}

fn run_block(begin: u32, end: u32, lanes: u32, lane: u32, ml: u32) {
  if (ml == 1u) { run_block1(begin, end, lanes, lane); }
  else { run_block_ml(begin, end, lanes, lane, ml); }
}

fn run_block1(begin: u32, end: u32, lanes: u32, lane: u32) {
  var r = begin;
  loop {
    if (r >= end) { break; }
    let base = r * 6u;
    let op = code[base];
    let f1 = code[base + 1u];
    let width = code[base + 2u];
    let a = code[base + 3u];
    let b = code[base + 4u];
    let c = code[base + 5u];
    let mask = mask_of(width);
    var res = 0u;
    var is_effect = false;
    switch op {
      case 0u: { res = aux[a] & mask; }
      case 1u: { res = sig[a * lanes + lane] & mask; }
      case 2u: { res = (~vget(a, lanes, lane)) & mask; }
      case 3u: { res = (vget(a, lanes, lane) & vget(b, lanes, lane)) & mask; }
      case 4u: { res = (vget(a, lanes, lane) | vget(b, lanes, lane)) & mask; }
      case 5u: { res = (vget(a, lanes, lane) ^ vget(b, lanes, lane)) & mask; }
      case 6u: { res = (vget(a, lanes, lane) + vget(b, lanes, lane)) & mask; }
      case 7u: { res = (vget(a, lanes, lane) - vget(b, lanes, lane)) & mask; }
      case 8u: { res = (vget(a, lanes, lane) * vget(b, lanes, lane)) & mask; }
      case 9u: { res = select(0u, 1u, vget(a, lanes, lane) == vget(b, lanes, lane)); }
      case 10u: { res = select(0u, 1u, vget(a, lanes, lane) != vget(b, lanes, lane)); }
      case 11u: { res = select(0u, 1u, vget(a, lanes, lane) < vget(b, lanes, lane)); }
      case 12u: {
        let sign = 1u << (c - 1u);
        let l = vget(a, lanes, lane);
        let rr = vget(b, lanes, lane);
        let ls = (l & sign) != 0u;
        let rs = (rr & sign) != 0u;
        if (ls != rs) { res = select(0u, 1u, ls); } else { res = select(0u, 1u, l < rr); }
      }
      case 13u: {
        if ((vget(a, lanes, lane) & 1u) != 0u) { res = vget(b, lanes, lane); }
        else { res = vget(c, lanes, lane); }
      }
      case 14u: { res = (vget(a, lanes, lane) >> b) & mask; }
      case 15u: { res = vget(a, lanes, lane) & mask; }
      case 16u: {
        let src_mask = mask_of(c);
        let sign = 1u << (c - 1u);
        let v = vget(a, lanes, lane) & src_mask;
        if ((v & sign) != 0u) { res = (v | (~src_mask)) & mask; } else { res = v & mask; }
      }
      case 17u: { res = vget(a, lanes, lane) & mask; }
      case 18u: { res = vget(a, lanes, lane) & mask; }
      case 19u: { sig[f1 * lanes + lane] = vget(a, lanes, lane) & mask; is_effect = true; }
      case 20u: {
        var next = vget(a, lanes, lane) & mask;
        if (b != 0xffffffffu && reset_on(b, lanes, lane)) { next = aux[b + 2u] & mask; }
        reg_next[f1 * lanes + lane] = next;
        is_effect = true;
      }
      case 23u: {
        var result = 0u;
        var ofs = 0u;
        var k = b;
        loop {
          if (k == 0u) { break; }
          k = k - 1u;
          let part = vget(aux[a + k * 2u], lanes, lane) & mask_of(aux[a + k * 2u + 1u]);
          if (ofs < 32u) { result = result | (part << ofs); }
          ofs = ofs + aux[a + k * 2u + 1u];
        }
        res = result & mask;
      }
      case 24u: {
        if (reset_on(b, lanes, lane)) { sig[f1 * lanes + lane] = aux[b + 2u] & mask; }
        is_effect = true;
      }
      case 21u: {
        let addr = vget(a, lanes, lane);
        if (addr < c) { res = mem[(b + addr) * lanes + lane] & mask; } else { res = 0u; }
      }
      case 22u: {
        let addr = vget(b, lanes, lane);
        if (((vget(a, lanes, lane) & 1u) != 0u) && (addr < width)) {
          mem[(f1 + addr) * lanes + lane] = vget(c, lanes, lane);
        }
        is_effect = true;
      }
      default: {}
    }
    if (!is_effect) { values[f1 * lanes + lane] = res; }
    r = r + 1u;
  }
}

// Multi-limb path (ml limbs of 32 bits each, ml in 2..=4). Operands are loaded
// into fixed local arrays; multiply uses 16-bit half-products (16x16 fits u32).
fn vload_ml(id: u32, ml: u32, lanes: u32, lane: u32) -> array<u32, 4> {
  var v = array<u32, 4>(0u, 0u, 0u, 0u);
  for (var l = 0u; l < ml; l = l + 1u) { v[l] = values[(id * ml + l) * lanes + lane]; }
  return v;
}

fn run_block_ml(begin: u32, end: u32, lanes: u32, lane: u32, ml: u32) {
  var r = begin;
  loop {
    if (r >= end) { break; }
    let base = r * 6u;
    let op = code[base];
    let f1 = code[base + 1u];
    let width = code[base + 2u];
    let a = code[base + 3u];
    let b = code[base + 4u];
    let c = code[base + 5u];
    let nl = (width + 31u) / 32u;
    var res = array<u32, 4>(0u, 0u, 0u, 0u);
    var is_effect = false;

    switch op {
      case 0u: { for (var l = 0u; l < nl; l = l + 1u) { res[l] = aux[a + l]; } }
      case 1u: { for (var l = 0u; l < nl; l = l + 1u) { res[l] = sig[(a + l) * lanes + lane]; } }
      case 2u: { var x = vload_ml(a, ml, lanes, lane); for (var l = 0u; l < ml; l = l + 1u) { res[l] = ~x[l]; } }
      case 3u: { var x = vload_ml(a, ml, lanes, lane); var y = vload_ml(b, ml, lanes, lane); for (var l = 0u; l < ml; l = l + 1u) { res[l] = x[l] & y[l]; } }
      case 4u: { var x = vload_ml(a, ml, lanes, lane); var y = vload_ml(b, ml, lanes, lane); for (var l = 0u; l < ml; l = l + 1u) { res[l] = x[l] | y[l]; } }
      case 5u: { var x = vload_ml(a, ml, lanes, lane); var y = vload_ml(b, ml, lanes, lane); for (var l = 0u; l < ml; l = l + 1u) { res[l] = x[l] ^ y[l]; } }
      case 6u: {
        var x = vload_ml(a, ml, lanes, lane); var y = vload_ml(b, ml, lanes, lane);
        var carry = 0u;
        for (var l = 0u; l < ml; l = l + 1u) {
          let s1 = x[l] + y[l]; let c1 = select(0u, 1u, s1 < x[l]);
          let s2 = s1 + carry; let c2 = select(0u, 1u, s2 < s1);
          res[l] = s2; carry = c1 + c2;
        }
      }
      case 7u: {
        var x = vload_ml(a, ml, lanes, lane); var y = vload_ml(b, ml, lanes, lane);
        var borrow = 0u;
        for (var l = 0u; l < ml; l = l + 1u) {
          let d1 = x[l] - y[l]; let b1 = select(0u, 1u, x[l] < y[l]);
          let d2 = d1 - borrow; let b2 = select(0u, 1u, d1 < borrow);
          res[l] = d2; borrow = b1 + b2;
        }
      }
      case 8u: {
        var x = vload_ml(a, ml, lanes, lane); var y = vload_ml(b, ml, lanes, lane);
        let halves = ml * 2u;
        var acc = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
        for (var i = 0u; i < halves; i = i + 1u) {
          let ai = (x[i / 2u] >> ((i & 1u) * 16u)) & 0xffffu;
          var carry = 0u;
          for (var j = 0u; j < halves; j = j + 1u) {
            let k = i + j;
            if (k >= halves) { break; }
            let bj = (y[j / 2u] >> ((j & 1u) * 16u)) & 0xffffu;
            let p = ai * bj + acc[k] + carry;
            acc[k] = p & 0xffffu;
            carry = p >> 16u;
          }
        }
        for (var l = 0u; l < ml; l = l + 1u) { res[l] = acc[2u * l] | (acc[2u * l + 1u] << 16u); }
      }
      case 9u: { var x = vload_ml(a, ml, lanes, lane); var y = vload_ml(b, ml, lanes, lane); var eq = true; for (var l = 0u; l < ml; l = l + 1u) { if (x[l] != y[l]) { eq = false; } } res[0] = select(0u, 1u, eq); }
      case 10u: { var x = vload_ml(a, ml, lanes, lane); var y = vload_ml(b, ml, lanes, lane); var eq = true; for (var l = 0u; l < ml; l = l + 1u) { if (x[l] != y[l]) { eq = false; } } res[0] = select(0u, 1u, !eq); }
      case 11u: {
        var x = vload_ml(a, ml, lanes, lane); var y = vload_ml(b, ml, lanes, lane);
        var lt = 0u; var done = false; var l = ml;
        loop { if (l == 0u) { break; } l = l - 1u; if (!done && x[l] != y[l]) { lt = select(0u, 1u, x[l] < y[l]); done = true; } }
        res[0] = lt;
      }
      case 12u: {
        var x = vload_ml(a, ml, lanes, lane); var y = vload_ml(b, ml, lanes, lane);
        let sl = (c - 1u) / 32u; let sb = 1u << ((c - 1u) & 31u);
        let xs = (x[sl] & sb) != 0u; let ys = (y[sl] & sb) != 0u;
        if (xs != ys) { res[0] = select(0u, 1u, xs); }
        else {
          var lt = 0u; var done = false; var l = ml;
          loop { if (l == 0u) { break; } l = l - 1u; if (!done && x[l] != y[l]) { lt = select(0u, 1u, x[l] < y[l]); done = true; } }
          res[0] = lt;
        }
      }
      case 13u: {
        let cond = (values[(a * ml) * lanes + lane] & 1u) != 0u;
        var xt = vload_ml(b, ml, lanes, lane); var xf = vload_ml(c, ml, lanes, lane);
        for (var l = 0u; l < ml; l = l + 1u) { res[l] = select(xf[l], xt[l], cond); }
      }
      case 14u: {
        var x = vload_ml(a, ml, lanes, lane);
        let ls = b; let lsh = ls / 32u; let bsh = ls & 31u;
        for (var l = 0u; l < ml; l = l + 1u) {
          let s0 = l + lsh;
          var v = 0u;
          if (s0 < ml) { v = x[s0] >> bsh; }
          if (bsh != 0u && s0 + 1u < ml) { v = v | (x[s0 + 1u] << (32u - bsh)); }
          res[l] = v;
        }
      }
      case 15u, 17u, 18u: { var x = vload_ml(a, ml, lanes, lane); for (var l = 0u; l < ml; l = l + 1u) { res[l] = x[l]; } }
      case 16u: {
        var x = vload_ml(a, ml, lanes, lane);
        let sw = c; let sl = (sw - 1u) / 32u; let sb = 1u << ((sw - 1u) & 31u);
        let neg = (x[sl] & sb) != 0u;
        for (var l = 0u; l < ml; l = l + 1u) {
          let lo = l * 32u;
          var v = x[l];
          if (neg) {
            if (lo >= sw) { v = 0xffffffffu; }
            else if (lo + 32u > sw) { let m = (1u << (sw - lo)) - 1u; v = (v & m) | (~m); }
          }
          res[l] = v;
        }
      }
      case 19u: {
        var x = vload_ml(a, ml, lanes, lane);
        for (var l = 0u; l < nl; l = l + 1u) { sig[(f1 + l) * lanes + lane] = x[l] & limb_mask(width, l); }
        is_effect = true;
      }
      case 20u: {
        var src = vload_ml(a, ml, lanes, lane);
        if (b != 0xffffffffu && reset_on(b, lanes, lane)) { for (var l = 0u; l < nl; l = l + 1u) { src[l] = aux[b + 2u + l]; } }
        for (var l = 0u; l < nl; l = l + 1u) { reg_next[(f1 + l) * lanes + lane] = src[l] & limb_mask(width, l); }
        is_effect = true;
      }
      case 21u: {
        let addr = values[(a * ml) * lanes + lane];
        if (addr < c) { res[0] = mem[(b + addr) * lanes + lane]; }
        is_effect = false;
      }
      case 22u: {
        let en = values[(a * ml) * lanes + lane] & 1u;
        let addr = values[(b * ml) * lanes + lane];
        if (en != 0u && addr < width) { mem[(f1 + addr) * lanes + lane] = values[(c * ml) * lanes + lane]; }
        is_effect = true;
      }
      case 23u: {
        var bitofs = 0u;
        var k = b;
        loop {
          if (k == 0u) { break; }
          k = k - 1u;
          let vid = aux[a + k * 2u]; let w = aux[a + k * 2u + 1u];
          var part = vload_ml(vid, ml, lanes, lane);
          let dl = bitofs / 32u; let dsh = bitofs & 31u;
          let wl = (w + 31u) / 32u;
          for (var ol = 0u; ol < wl; ol = ol + 1u) {
            let pm = part[ol] & limb_mask(w, ol);
            if (dl + ol < ml) { res[dl + ol] = res[dl + ol] | (pm << dsh); }
            if (dsh != 0u && dl + ol + 1u < ml) { res[dl + ol + 1u] = res[dl + ol + 1u] | (pm >> (32u - dsh)); }
          }
          bitofs = bitofs + w;
        }
      }
      case 24u: {
        if (reset_on(b, lanes, lane)) { for (var l = 0u; l < nl; l = l + 1u) { sig[(f1 + l) * lanes + lane] = aux[b + 2u + l] & limb_mask(width, l); } }
        is_effect = true;
      }
      default: {}
    }
    if (!is_effect) { for (var l = 0u; l < ml; l = l + 1u) { values[(f1 * ml + l) * lanes + lane] = res[l] & limb_mask(width, l); } }
    r = r + 1u;
  }
}

@compute @workgroup_size({WG})
fn interp_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lanes = params[0];
  let lane = gid.x;
  if (lane >= lanes) { return; }
  let steps = params[3];
  let async_b = params[8]; let async_e = params[9];
  let comb_b = params[4]; let comb_e = params[5];
  let tnext_b = params[6]; let tnext_e = params[7];
  let commit_b = params[10]; let commit_e = params[11];
  let cap_count = params[12];
  let ml = params[13];
  // Settle combinational logic once; each cycle then captures, commits, and
  // re-settles. This fuses the otherwise-redundant trailing/leading comb passes
  // (they settle from the same register state) — half the combinational work.
  run_block(async_b, async_e, lanes, lane, ml);
  run_block(comb_b, comb_e, lanes, lane, ml);
  var s = 0u;
  loop {
    if (s >= steps) { break; }
    run_block(tnext_b, tnext_e, lanes, lane, ml);
    run_block(commit_b, commit_e, lanes, lane, ml); // tick_commit: memory writes
    var i = 0u;
    loop {
      if (i >= cap_count) { break; }
      let off = captured[2u * i];
      let lm = captured[2u * i + 1u];
      for (var l = 0u; l < lm; l = l + 1u) {
        sig[(off + l) * lanes + lane] = reg_next[(off + l) * lanes + lane];
      }
      i = i + 1u;
    }
    run_block(async_b, async_e, lanes, lane, ml);
    run_block(comb_b, comb_e, lanes, lane, ml);
    s = s + 1u;
  }
}
"#;

/// GPU batch simulator that executes an [`InterpProgram`] with the fixed-size
/// interpreter kernel (design-as-data). Shader is O(1) in design size.
pub struct InterpGpuSimulator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    sig_buffer: wgpu::Buffer,
    sig_readback: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    mem_buffer: wgpu::Buffer,
    lanes: usize,
    total_signal_words: usize,
    total_memory_words: usize,
    workgroup_size: u32,
}

fn storage_words(words: usize) -> u64 {
    (words.max(1) * 4) as u64
}

impl InterpGpuSimulator {
    pub fn new(program: &InterpProgram, lanes: usize) -> Result<Self, ErrorReport> {
        Self::new_with_workgroup(program, lanes, INTERP_DEFAULT_WORKGROUP)
    }

    /// Builds with an explicit compute workgroup size (threads per group).
    pub fn new_with_workgroup(
        program: &InterpProgram,
        lanes: usize,
        workgroup_size: u32,
    ) -> Result<Self, ErrorReport> {
        pollster::block_on(Self::new_async(program, lanes, workgroup_size))
    }

    async fn new_async(
        program: &InterpProgram,
        lanes: usize,
        workgroup_size: u32,
    ) -> Result<Self, ErrorReport> {
        if program.max_limbs > 4 {
            return Err(interp_error(
                "GPU interpreter kernel supports values up to 128 bits (4 limbs)",
            ));
        }
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| interp_error("no suitable GPU adapter found"))?;
        // The interpreter needs 6 storage buffers; downlevel defaults cap at 4.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rrtl-interp-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                },
                None,
            )
            .await
            .map_err(|err| interp_error(format!("failed to create GPU device: {err}")))?;

        // Concatenate the four blocks; record ranges (in records) for params.
        let mut code: Vec<u32> = Vec::new();
        let mut ranges = [0u32; 8]; // [async_b,async_e, comb_b,comb_e, tnext_b,tnext_e, commit_b,commit_e]
        for (i, block) in program.blocks.iter().enumerate() {
            let begin = (code.len() / RECORD_WORDS) as u32;
            code.extend_from_slice(block);
            let end = (code.len() / RECORD_WORDS) as u32;
            ranges[i * 2] = begin;
            ranges[i * 2 + 1] = end;
        }
        // Flat [offset, limbs] pairs so commit copies all limbs of each register.
        let captured_pairs = program.captured_offsets();
        let captured: Vec<u32> = captured_pairs
            .iter()
            .flat_map(|&(offset, limbs)| [offset as u32, limbs as u32])
            .collect();

        let params: Vec<u32> = vec![
            lanes as u32,
            program.num_values as u32,
            program.total_signal_words as u32,
            0, // steps, set per tick_many
            ranges[2], // comb_b
            ranges[3], // comb_e
            ranges[4], // tnext_b
            ranges[5], // tnext_e
            ranges[0], // async_b
            ranges[1], // async_e
            ranges[6], // commit_b (unused)
            ranges[7], // commit_e (unused)
            captured_pairs.len() as u32, // register count
            program.max_limbs as u32,
        ];

        let storage = |label, words: usize, data: Option<&[u32]>| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: storage_words(words),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            if let Some(data) = data {
                if !data.is_empty() {
                    queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
                }
            }
            buf
        };

        let sig_words = program.total_signal_words * lanes;
        let value_words = program.num_values * program.max_limbs * lanes;
        let zeros_sig = vec![0u32; sig_words];
        let sig_buffer = storage("interp-sig", sig_words, Some(&zeros_sig));
        let reg_next_buffer = storage("interp-reg-next", sig_words, Some(&zeros_sig));
        let values_buffer = storage("interp-values", value_words, Some(&vec![0u32; value_words]));
        let code_buffer = storage("interp-code", code.len(), Some(&code));
        let captured_buffer = storage("interp-captured", captured.len(), Some(&captured));
        let params_buffer = storage("interp-params", params.len(), Some(&params));
        let mem_words = program.total_memory_words * lanes;
        let mem_buffer = storage("interp-mem", mem_words, Some(&vec![0u32; mem_words]));
        let aux_buffer = storage("interp-aux", program.aux.len(), Some(&program.aux));

        let sig_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interp-sig-readback"),
            size: storage_words(sig_words),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bindings 3,4,5,7 (code, captured, params, aux) are read-only;
        // 0,1,2,6 (sig, reg_next, values, mem) are read-write.
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..8)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: (3..=5).contains(&binding) || binding == 7,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("interp-layout"),
            entries: &entries,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("interp-bind"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sig_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: reg_next_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: values_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: code_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: captured_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: mem_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: aux_buffer.as_entire_binding() },
            ],
        });

        let source = INTERP_WGSL.replace("{WG}", &workgroup_size.to_string());
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("interp-kernel"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("interp-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("interp-pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "interp_main",
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group,
            sig_buffer,
            sig_readback,
            params_buffer,
            mem_buffer,
            lanes,
            total_signal_words: program.total_signal_words,
            total_memory_words: program.total_memory_words,
            workgroup_size,
        })
    }

    /// Resets all signal/register state to zero. Only the persistent signal
    /// buffer needs clearing — the value workspace and `reg_next` shadow are
    /// fully rewritten before being read each cycle. Used to reuse one simulator
    /// across independent lane tiles.
    pub fn reset(&self) {
        let sig = vec![0u32; self.total_signal_words * self.lanes];
        if !sig.is_empty() {
            self.queue
                .write_buffer(&self.sig_buffer, 0, bytemuck::cast_slice(&sig));
        }
        let mem = vec![0u32; self.total_memory_words * self.lanes];
        if !mem.is_empty() {
            self.queue
                .write_buffer(&self.mem_buffer, 0, bytemuck::cast_slice(&mem));
        }
    }

    pub fn set_signal(&self, offset: usize, lane_values: &[u32]) {
        // Storage is WordMajor, so a signal's lanes are contiguous at offset*lanes.
        self.queue.write_buffer(
            &self.sig_buffer,
            (offset * self.lanes * 4) as u64,
            bytemuck::cast_slice(lane_values),
        );
    }

    pub fn tick_many(&self, steps: usize) {
        self.queue
            .write_buffer(&self.params_buffer, 3 * 4, bytemuck::cast_slice(&[steps as u32]));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("interp-tick") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("interp-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups = (self.lanes as u32).div_ceil(self.workgroup_size);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        // No blocking wait here: the GPU runs asynchronously so the caller can
        // do other work (e.g. simulate its own lanes on the CPU) concurrently.
        // Submissions are ordered, so a later read drains this work. Call
        // `synchronize` to wait explicitly.
    }

    /// Blocks until all submitted GPU work has completed.
    pub fn synchronize(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    pub fn get_signal(&self, offset: usize) -> Vec<u32> {
        let words = self.total_signal_words * self.lanes;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("interp-readback"),
            });
        encoder.copy_buffer_to_buffer(&self.sig_buffer, 0, &self.sig_readback, 0, storage_words(words));
        self.queue.submit(Some(encoder.finish()));
        let slice = self.sig_readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let all = bytemuck::cast_slice::<u8, u32>(&data);
        let result = (0..self.lanes)
            .map(|lane| all[offset * self.lanes + lane])
            .collect();
        drop(data);
        self.sig_readback.unmap();
        result
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

    /// Fused `tick_many` (one comb settle per cycle) must reach the same final
    /// state as `steps` separate ticks / SimdCpuSimulator.
    #[test]
    fn interp_runner_tick_many_matches_simd_cpu() {
        let mut design = Design::new();
        let (clk, din, acc_out);
        {
            let mut m = design.module("Top");
            clk = m.input("clk", uint(1));
            din = m.input("din", uint(16));
            let acc = m.reg("acc", uint(16));
            let w = m.wire("w", uint(16));
            m.clock(acc, clk);
            m.assign(w, acc * lit_u(5, 16) + din);
            m.next(acc, w);
            acc_out = m.output("acc_out", uint(16));
            m.assign(acc_out, acc);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();
        let offset = |sig: rrtl_ir::Signal| {
            program.signals[program.signal_index(sig).unwrap()].layout.offset
        };
        let lanes = 4;
        let mut runner = InterpRunner::new(InterpProgram::encode_design(&program).unwrap(), lanes);
        let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        let din_v: Vec<u32> = (0..lanes as u32).map(|l| 7 + l * 3).collect();
        runner.set_signal(offset(clk), &vec![1u32; lanes]);
        runner.set_signal(offset(din), &din_v);
        cpu.set_signal(clk, &vec![1u128; lanes]).unwrap();
        cpu.set_signal(din, &din_v.iter().map(|&v| v as u128).collect::<Vec<_>>())
            .unwrap();
        runner.tick_many(20);
        cpu.tick_many(20).unwrap();
        let r: Vec<u32> = runner.get_signal(offset(acc_out));
        let c: Vec<u32> = cpu.get_signal(acc_out).unwrap().iter().map(|&v| v as u32).collect();
        assert_eq!(r, c);
    }

    #[test]
    fn interp_runner_matches_simd_cpu_with_concat() {
        let mut design = Design::new();
        let (a, b, o);
        {
            let mut m = design.module("Top");
            a = m.input("a", uint(8));
            b = m.input("b", uint(4));
            o = m.output("o", uint(12));
            m.assign(o, rrtl_core::concat([a.value(), b.value()]));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();
        let off = |s: rrtl_ir::Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;
        let lanes = 4;
        let mut runner = InterpRunner::new(InterpProgram::encode_design(&program).unwrap(), lanes);
        let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        let av: Vec<u32> = (0..lanes as u32).map(|l| (l * 37 + 1) & 0xff).collect();
        let bv: Vec<u32> = (0..lanes as u32).map(|l| (l * 5 + 2) & 0xf).collect();
        runner.set_signal(off(a), &av);
        runner.set_signal(off(b), &bv);
        cpu.set_signal(a, &av.iter().map(|&x| x as u128).collect::<Vec<_>>()).unwrap();
        cpu.set_signal(b, &bv.iter().map(|&x| x as u128).collect::<Vec<_>>()).unwrap();
        runner.tick();
        cpu.tick();
        let r = runner.get_signal(off(o));
        let c: Vec<u32> = cpu.get_signal(o).unwrap().iter().map(|&v| v as u32).collect();
        assert_eq!(r, c);
    }

    #[test]
    fn interp_runner_matches_simd_cpu_with_resets() {
        let mut design = Design::new();
        let (clk, rst, rstn, din, oqs, oqa);
        {
            let mut m = design.module("Top");
            clk = m.input("clk", uint(1));
            rst = m.input("rst", uint(1));
            rstn = m.input("rstn", uint(1));
            din = m.input("din", uint(8));
            let qs = m.reg("qs", uint(8));
            let qa = m.reg("qa", uint(8));
            m.clock(qs, clk);
            m.clock(qa, clk);
            m.reset(qs, rst, 5);
            m.async_reset_low(qa, rstn, 9);
            m.next(qs, din);
            m.next(qa, din);
            oqs = m.output("oqs", uint(8));
            oqa = m.output("oqa", uint(8));
            m.assign(oqs, qs);
            m.assign(oqa, qa);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();
        let off = |s: rrtl_ir::Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;
        let lanes = 3;
        let mut runner = InterpRunner::new(InterpProgram::encode_design(&program).unwrap(), lanes);
        let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        runner.set_signal(off(clk), &vec![1u32; lanes]);
        cpu.set_signal(clk, &vec![1u128; lanes]).unwrap();

        let mut set = |runner: &mut InterpRunner, cpu: &mut SimdCpuSimulator, s: rrtl_ir::Signal, vals: [u32; 3]| {
            runner.set_signal(off(s), &vals);
            cpu.set_signal(s, &vals.iter().map(|&x| x as u128).collect::<Vec<_>>()).unwrap();
        };
        // (rst, rstn, din) per cycle, varied across the 3 lanes via per-cycle arrays.
        let seq: [([u32; 3], [u32; 3], [u32; 3]); 5] = [
            ([1, 0, 1], [1, 1, 0], [10, 20, 30]),
            ([0, 1, 0], [1, 0, 1], [40, 50, 60]),
            ([0, 0, 1], [0, 1, 1], [70, 80, 90]),
            ([1, 1, 0], [1, 1, 1], [11, 22, 33]),
            ([0, 0, 0], [1, 1, 1], [44, 55, 66]),
        ];
        for (cycle, (r, rn, d)) in seq.iter().enumerate() {
            set(&mut runner, &mut cpu, rst, *r);
            set(&mut runner, &mut cpu, rstn, *rn);
            set(&mut runner, &mut cpu, din, *d);
            runner.tick();
            cpu.tick();
            for (sig, name) in [(oqs, "qs"), (oqa, "qa")] {
                let rr = runner.get_signal(off(sig));
                let cc: Vec<u32> = cpu.get_signal(sig).unwrap().iter().map(|&v| v as u32).collect();
                assert_eq!(rr, cc, "{name} mismatch at cycle {cycle}");
            }
        }
    }

    /// 64-bit (2-limb) datapath: multi-limb mul + add registers must match
    /// SimdCpuSimulator, validating the u128-based multi-limb interpreter.
    #[test]
    fn interp_runner_matches_simd_cpu_64bit() {
        let mut design = Design::new();
        let (clk, din, o);
        {
            let mut m = design.module("Top");
            clk = m.input("clk", uint(1));
            din = m.input("din", uint(64));
            let acc = m.reg("acc", uint(64));
            m.clock(acc, clk);
            let mixed = m.wire("mixed", uint(64));
            m.assign(mixed, acc * lit_u(0x9e37_79b9_7f4a_7c15, 64) + din);
            m.next(acc, mixed);
            o = m.output("o", uint(64));
            m.assign(o, acc);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();
        let encoded = InterpProgram::encode_design(&program).unwrap();
        assert!(encoded.max_limbs >= 2, "expected multi-limb, got {}", encoded.max_limbs);
        let off = |s: rrtl_ir::Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;
        let lanes = 4;
        let mut runner = InterpRunner::new(encoded, lanes);
        let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        runner.set_signal(off(clk), &vec![1u32; lanes]);
        cpu.set_signal(clk, &vec![1u128; lanes]).unwrap();
        for cycle in 0..6u128 {
            let din_v: Vec<u128> = (0..lanes as u128)
                .map(|l| {
                    (cycle.wrapping_mul(0x1234_5678_9abc).wrapping_add(l.wrapping_mul(0x9999_8888_7777)))
                        & 0xffff_ffff_ffff_ffff
                })
                .collect();
            runner.set_signal_wide(off(din), 2, &din_v);
            cpu.set_signal(din, &din_v).unwrap();
            runner.tick();
            cpu.tick();
            let r = runner.get_signal_wide(off(o), 2);
            let c = cpu.get_signal(o).unwrap();
            assert_eq!(r, c, "mismatch at cycle {cycle}");
        }
    }

    /// A design with a memory (write + read port) must match SimdCpuSimulator
    /// cycle by cycle and across lanes.
    #[test]
    fn interp_runner_matches_simd_cpu_with_memory() {
        let mut design = Design::new();
        let (clk, we, waddr, wdata, raddr, rdata);
        {
            let mut m = design.module("Top");
            clk = m.input("clk", uint(1));
            we = m.input("we", uint(1));
            waddr = m.input("waddr", uint(2));
            wdata = m.input("wdata", uint(16));
            raddr = m.input("raddr", uint(2));
            let mem = m.mem("mem", 2, uint(16), 4);
            rdata = m.output("rdata", uint(16));
            m.mem_write(mem, clk, we, waddr, wdata);
            let rd = m.mem_read(mem, raddr);
            m.assign(rdata, rd);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();
        let off = |s: rrtl_ir::Signal| {
            program.signals[program.signal_index(s).unwrap()].layout.offset
        };
        let lanes = 3;
        let mut runner = InterpRunner::new(InterpProgram::encode_design(&program).unwrap(), lanes);
        let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        runner.set_signal(off(clk), &vec![1u32; lanes]);
        cpu.set_signal(clk, &vec![1u128; lanes]).unwrap();

        let mut set = |runner: &mut InterpRunner, cpu: &mut SimdCpuSimulator, s: rrtl_ir::Signal, base: u32| {
            let v: Vec<u32> = (0..lanes as u32).map(|l| base.wrapping_add(l * 5) & 0xffff).collect();
            runner.set_signal(off(s), &v);
            cpu.set_signal(s, &v.iter().map(|&x| x as u128).collect::<Vec<_>>()).unwrap();
        };

        // (we, waddr, wdata, raddr) per cycle.
        let seq = [
            (1, 0, 100, 0),
            (1, 1, 200, 0),
            (1, 2, 300, 1),
            (0, 0, 0, 2),
            (1, 3, 400, 3),
            (0, 0, 0, 0),
            (0, 0, 0, 3),
        ];
        for (cycle, &(w, wa, wd, ra)) in seq.iter().enumerate() {
            set(&mut runner, &mut cpu, we, w);
            set(&mut runner, &mut cpu, waddr, wa);
            set(&mut runner, &mut cpu, wdata, wd);
            set(&mut runner, &mut cpu, raddr, ra);
            runner.tick();
            cpu.tick();
            let r = runner.get_signal(off(rdata));
            let c: Vec<u32> = cpu.get_signal(rdata).unwrap().iter().map(|&v| v as u32).collect();
            assert_eq!(r, c, "rdata mismatch at cycle {cycle}");
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
