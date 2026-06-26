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
/// Fused multiply-add: `dst = (a*b + c) & mask`. Produced by encoder fusion of an
/// `Add(Mul(a,b), c)` whose multiply result is used exactly once, so the product
/// never round-trips the value buffer. All three operands share the result width.
pub const OP_MULADD: u32 = 25;
/// Fused three-input add: `dst = (a + b + c) & mask`. From `Add(Add(a,b), c)`.
pub const OP_ADD3: u32 = 26;
/// Fused and-or (AOI): `dst = ((a & b) | c) & mask`. From `Or(And(a,b), c)`.
pub const OP_ANDOR: u32 = 27;
/// Fused NAND: `dst = ~(a & b) & mask`. From `Not(And(a,b))` (gate-level).
pub const OP_NAND: u32 = 28;
/// Fused NOR: `dst = ~(a | b) & mask`. From `Not(Or(a,b))`.
pub const OP_NOR: u32 = 29;
/// Fused XNOR: `dst = ~(a ^ b) & mask`. From `Not(Xor(a,b))`.
pub const OP_XNOR: u32 = 30;
/// Fused multiply-subtract: `dst = (a*b - c) & mask`. From `Sub(Mul(a,b), c)`
/// (the multiply must be the minuend; Sub is not commutative).
pub const OP_MULSUB: u32 = 31;
/// Fused three-input xor: `dst = (a ^ b ^ c) & mask`. From `Xor(Xor(a,b), c)`.
pub const OP_XOR3: u32 = 32;
/// Fused or-and (OAI): `dst = ((a | b) & c) & mask`. From `And(Or(a,b), c)`.
pub const OP_ORAND: u32 = 33;
/// Fused and-or-invert (AOI21 standard cell): `dst = ~((a & b) | c) & mask`. From
/// the two-level `Not(Or(And(a,b), c))`.
pub const OP_AOI21: u32 = 34;
/// Fused or-and-invert (OAI21 standard cell): `dst = ~((a | b) & c) & mask`. From
/// `Not(And(Or(a,b), c))`.
pub const OP_OAI21: u32 = 35;
/// Fused and-or-invert (AOI22 standard cell): `dst = ~((a & b) | (c & d)) & mask`.
/// From `Not(Or(And(a,b), And(c,d)))`. Operands c and d live in `aux` at the
/// offset stored in the record's `c` field (the 6-word record holds only a, b).
pub const OP_AOI22: u32 = 36;
/// Fused or-and-invert (OAI22 standard cell): `dst = ~((a | b) & (c | d)) & mask`.
/// From `Not(And(Or(a,b), Or(c,d)))`. c and d live in `aux` (see [`OP_AOI22`]).
pub const OP_OAI22: u32 = 37;

/// Fused-cone macro-op (the automatic superoptimizer's output). Evaluates a
/// sub-DAG of simple ops in local registers, writing only the root to the value
/// buffer — so single-use intermediates never round-trip the (global) value
/// buffer. Record: `[OP_MACRO, dst, root_width, aux_off, n_sub, 0]`. `aux` holds
/// `n_sub` 6-word sub-ops `[sub_op, sub_width, ra, rb, rc, imm]`, in topo order;
/// sub-op `i` writes local reg `i`; the root is reg `n_sub-1`. An operand
/// `ra/rb/rc` with bit 31 set is a LOCAL register `(r & MACRO_LOCAL_MASK)`, else
/// a value-buffer id; `imm` carries Slice-lsb / Sext-src-width / Lt-sign-width.
pub const OP_MACRO: u32 = 38;
/// Bit 31 of a macro operand ref: set = local register, clear = value-buffer id.
pub const MACRO_LOCAL: u32 = 0x8000_0000;
pub const MACRO_LOCAL_MASK: u32 = 0x7fff_ffff;
/// Max sub-ops per macro (= the interpreter's local register-file size). Larger
/// cones split across multiple macros.
pub const MACRO_MAX_SUB: usize = 256;

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
    /// Max limbs (32-bit words) of any value (1-limb fast path uses ==1).
    pub max_limbs: usize,
    /// Total packed value-workspace words per lane (sum of each value's limbs).
    pub total_value_words: usize,
    /// Offset into `aux` of the per-value-id packed word-offset table.
    pub value_offsets_base: usize,
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

    /// Like [`encode_design`](Self::encode_design) but first runs the AOT design
    /// specializer (constant folding, algebraic identities, dead-code
    /// elimination, value-id compaction). Returns the specialization stats so
    /// callers can report how much was eliminated.
    pub fn encode_design_specialized(
        program: &PackedProgram,
    ) -> Result<(Self, rrtl_sim_ir::SpecializeStats), ErrorReport> {
        let machine = lower_to_machine_program(program);
        let (specialized, stats) = rrtl_sim_ir::specialize_program(&machine);
        Ok((Self::encode(&specialized)?, stats))
    }

    pub fn encode(machine: &PackedMachineProgram) -> Result<Self, ErrorReport> {
        Self::encode_opts(machine, true)
    }

    /// Encode with explicit control over multiply-add fusion. Fusion assumes SSA
    /// value ids, so it must be disabled when encoding a slot-allocated program
    /// (where ids are reused across non-overlapping lifetimes).
    pub fn encode_opts(machine: &PackedMachineProgram, fuse: bool) -> Result<Self, ErrorReport> {
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
            *slot = encode_block(block, source, &mut aux, index == 0, fuse)?;
        }
        let num_values = streams
            .iter()
            .map(|block| block_value_count(block))
            .max()
            .unwrap_or(0);
        // Per-value-id limb count = max over blocks (a slot is shared across
        // blocks). Pack values tightly: voff[v] = cumulative limbs.
        let mut value_limbs = vec![1usize; num_values];
        for instr in streams
            .iter()
            .flat_map(|block| block.packets.iter())
            .flat_map(|packet| packet.instrs.iter())
        {
            let l = limbs_of(instr.ty.width);
            if l > value_limbs[instr.dst.0] {
                value_limbs[instr.dst.0] = l;
            }
        }
        let max_limbs = value_limbs.iter().copied().max().unwrap_or(1).max(1);
        let mut acc = 0u32;
        let value_offsets_base = aux.len();
        for &l in &value_limbs {
            aux.push(acc);
            acc += l as u32;
        }
        let total_value_words = acc as usize;

        Ok(Self {
            num_values,
            max_limbs,
            total_value_words,
            value_offsets_base,
            total_signal_words: source.total_signal_words,
            total_memory_words: source.total_memory_words_per_lane,
            blocks,
            aux,
        })
    }

    pub fn block(&self, stream: InterpStream) -> &[u32] {
        &self.blocks[stream as usize]
    }

    /// Total encoded instruction words across all four streams (a proxy for code
    /// size / shader-independent program size).
    pub fn total_code_words(&self) -> usize {
        self.blocks.iter().map(|b| b.len()).sum()
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

/// Value ids read by an instruction (for fusion use-count analysis).
fn instr_operand_ids(kind: &PackedInstrKind) -> Vec<usize> {
    use PackedInstrKind::*;
    match kind {
        Lit(_) | Signal(_) => vec![],
        Not(a) | Zext(a) | Sext(a) | Trunc(a) | Cast(a) | Slice { value: a, .. }
        | MemRead { addr: a, .. } => vec![a.0],
        And(a, b) | Or(a, b) | Xor(a, b) | Add(a, b) | Sub(a, b) | Mul(a, b) | Eq(a, b)
        | Ne(a, b) => vec![a.0, b.0],
        Lt { lhs, rhs, .. } => vec![lhs.0, rhs.0],
        Mux {
            cond,
            then_value,
            else_value,
        } => vec![cond.0, then_value.0, else_value.0],
        Concat(parts) => parts.iter().map(|p| p.0).collect(),
    }
}

fn effect_operand_ids(effect: &PackedEffect) -> Vec<usize> {
    match effect {
        PackedEffect::StoreSignal { value, .. } => vec![value.0],
        PackedEffect::CaptureReg { value, .. } => vec![value.0],
        PackedEffect::MemoryWrite {
            enable, addr, data, ..
        } => vec![enable.0, addr.0, data.0],
    }
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
    fuse: bool,
) -> Result<Vec<u32>, ErrorReport> {
    // Pre-pass: width of every value produced in this block, for ops that need a
    // source operand's width (Sext, signed Lt).
    let n = block_value_count(block);
    let mut value_width = vec![0u32; n];
    for instr in block.packets.iter().flat_map(|p| p.instrs.iter()) {
        value_width[instr.dst.0] = instr.ty.width;
    }

    // Fusion pre-pass: fold a two-operand op whose inner operand is a matching
    // single-use op into one fused superoperator, so the inner result never
    // round-trips the value buffer. Patterns (outer op, inner op) -> fused op:
    //   Add(Mul(a,b), c) -> MULADD,  Add(Add(a,b), c) -> ADD3,  Or(And(a,b), c) -> ANDOR.
    // `skip` holds fused-away inner ops; `fused` maps the outer's dst to
    // (fused_op, a, b, c). MULADD is preferred over ADD3 (the multiply is the
    // expensive op worth eliminating).
    let mut use_count = vec![0u32; n];
    for packet in &block.packets {
        for instr in &packet.instrs {
            for id in instr_operand_ids(&instr.kind) {
                use_count[id] += 1;
            }
        }
        for effect in &packet.effects {
            for id in effect_operand_ids(effect) {
                use_count[id] += 1;
            }
        }
    }
    // inner-op definitions: dst -> (kind_tag, a, b). tag: 0=Mul 1=Add 2=And 3=Or 4=Xor.
    let mut inner_def: std::collections::HashMap<usize, (u8, u32, u32)> =
        std::collections::HashMap::new();
    for instr in block.packets.iter().flat_map(|p| p.instrs.iter()) {
        let entry = match &instr.kind {
            PackedInstrKind::Mul(l, r) => Some((0u8, l.0 as u32, r.0 as u32)),
            PackedInstrKind::Add(l, r) => Some((1u8, l.0 as u32, r.0 as u32)),
            PackedInstrKind::And(l, r) => Some((2u8, l.0 as u32, r.0 as u32)),
            PackedInstrKind::Or(l, r) => Some((3u8, l.0 as u32, r.0 as u32)),
            PackedInstrKind::Xor(l, r) => Some((4u8, l.0 as u32, r.0 as u32)),
            _ => None,
        };
        if let Some(e) = entry {
            inner_def.insert(instr.dst.0, e);
        }
    }
    let mut skip: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut outer: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut fused: std::collections::HashMap<usize, (u32, u32, u32, u32)> =
        std::collections::HashMap::new();
    // Four-operand fused ops (AOI22/OAI22): (fused_op, a, b, c, d). c and d are
    // emitted into `aux` since the 6-word record holds only a and b.
    let mut fused4: std::collections::HashMap<usize, (u32, u32, u32, u32, u32)> =
        std::collections::HashMap::new();
    if fuse {
        // Program order: an inner op already chosen as a fused *outer* (it is in
        // `outer`) must not also be skipped as an inner — each instruction takes
        // part in at most one fusion.
        // A value is fusable as an inner only if it is produced exactly once here
        // and not already claimed by another fusion.
        let free_inner = |m: usize, skip: &std::collections::HashSet<usize>, outer: &std::collections::HashSet<usize>| {
            use_count[m] == 1 && !skip.contains(&m) && !outer.contains(&m)
        };
        for instr in block.packets.iter().flat_map(|p| p.instrs.iter()) {
            // `Not` is handled specially: it admits two-level standard-cell fusions
            // (AOI21 = ~((a&b)|c), OAI21 = ~((a|b)&c)) as well as the one-level
            // NAND/NOR/XNOR. Try the deeper (more specific) cells first.
            if let PackedInstrKind::Not(inner) = &instr.kind {
                let m = inner.0;
                if !free_inner(m, &skip, &outer) {
                    continue;
                }
                let Some(&(mtag, mx, my)) = inner_def.get(&m) else {
                    continue;
                };
                // Inner Or (mtag 3) -> AOI family; inner And (mtag 2) -> OAI family.
                //   both operands single-use And/Or -> AOI22/OAI22 ~((a&b)|(c&d))
                //   one operand single-use And/Or    -> AOI21/OAI21 ~((a&b)|c)
                //   neither                          -> NOR/NAND
                // Inner Xor (mtag 4) -> XNOR. Inner Mul/Add (0/1) -> not fusable here.
                match mtag {
                    2 | 3 => {
                        let want = if mtag == 3 { 2u8 } else { 3u8 }; // Or wants And; And wants Or
                        let (op22, op21, op1) = if mtag == 3 {
                            (OP_AOI22, OP_AOI21, OP_NOR)
                        } else {
                            (OP_OAI22, OP_OAI21, OP_NAND)
                        };
                        let gx = if free_inner(mx as usize, &skip, &outer) {
                            inner_def.get(&(mx as usize)).filter(|d| d.0 == want).copied()
                        } else {
                            None
                        };
                        let gy = if free_inner(my as usize, &skip, &outer) {
                            inner_def.get(&(my as usize)).filter(|d| d.0 == want).copied()
                        } else {
                            None
                        };
                        if let (Some((_, xa, xb)), Some((_, ya, yb))) = (gx, gy) {
                            fused4.insert(instr.dst.0, (op22, xa, xb, ya, yb));
                            skip.insert(mx as usize);
                            skip.insert(my as usize);
                        } else if let Some((_, xa, xb)) = gx {
                            fused.insert(instr.dst.0, (op21, xa, xb, my));
                            skip.insert(mx as usize);
                        } else if let Some((_, ya, yb)) = gy {
                            fused.insert(instr.dst.0, (op21, ya, yb, mx));
                            skip.insert(my as usize);
                        } else {
                            fused.insert(instr.dst.0, (op1, mx, my, 0));
                        }
                        skip.insert(m);
                        outer.insert(instr.dst.0);
                    }
                    4 => {
                        fused.insert(instr.dst.0, (OP_XNOR, mx, my, 0));
                        skip.insert(m);
                        outer.insert(instr.dst.0);
                    }
                    _ => {} // Not(Mul)/Not(Add): leave unfused
                }
            }
        }
        // Pass B: two-input arithmetic/logic fusions, run AFTER Not-rooted cells
        // have claimed their inners, so AOI21/OAI21 take priority over the
        // one-level ANDOR/ORAND that would otherwise grab the same inner Or/And.
        for instr in block.packets.iter().flat_map(|p| p.instrs.iter()) {
            // For each outer op: candidate (inner-operand, other-operand) pairs and
            // a preference-ordered plan of (inner tag, fused opcode). Commutative
            // outers try both operands as the inner; Sub (non-commutative) only
            // fuses a multiply that is the minuend.
            let (cands, plan): (Vec<(usize, usize)>, &[(u8, u32)]) = match &instr.kind {
                PackedInstrKind::Add(l, r) => {
                    (vec![(l.0, r.0), (r.0, l.0)], &[(0, OP_MULADD), (1, OP_ADD3)])
                }
                PackedInstrKind::Sub(l, r) => (vec![(l.0, r.0)], &[(0, OP_MULSUB)]),
                PackedInstrKind::Or(l, r) => (vec![(l.0, r.0), (r.0, l.0)], &[(2, OP_ANDOR)]),
                PackedInstrKind::And(l, r) => (vec![(l.0, r.0), (r.0, l.0)], &[(3, OP_ORAND)]),
                PackedInstrKind::Xor(l, r) => (vec![(l.0, r.0), (r.0, l.0)], &[(4, OP_XOR3)]),
                _ => continue,
            };
            'outer: for &(want_tag, fop) in plan {
                for &(m, other) in &cands {
                    if use_count[m] == 1 && !skip.contains(&m) && !outer.contains(&m) {
                        if let Some(&(tag, ia, ib)) = inner_def.get(&m) {
                            if tag == want_tag {
                                fused.insert(instr.dst.0, (fop, ia, ib, other as u32));
                                skip.insert(m);
                                outer.insert(instr.dst.0);
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut out = Vec::new();
    for packet in &block.packets {
        for instr in &packet.instrs {
            let dst = instr.dst.0 as u32;
            let width = instr.ty.width;
            if skip.contains(&instr.dst.0) {
                continue; // inner op fused into a following outer op
            }
            if let Some(&(fop, fa, fb, fc, fd)) = fused4.get(&instr.dst.0) {
                let aux_off = aux.len() as u32;
                aux.push(fc);
                aux.push(fd);
                push_record(&mut out, fop, dst, width, fa, fb, aux_off);
                continue;
            }
            if let Some(&(fop, a, b, c)) = fused.get(&instr.dst.0) {
                push_record(&mut out, fop, dst, width, a, b, c);
                continue;
            }
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
                // c carries the operand width where the result width differs
                // from it (so the multi-limb kernel reads only the operand's limbs).
                PackedInstrKind::Zext(v) => rec(OP_ZEXT, v.0 as u32, 0, value_width[v.0]),
                PackedInstrKind::Trunc(v) => rec(OP_TRUNC, v.0 as u32, 0, 0),
                PackedInstrKind::Cast(v) => rec(OP_CAST, v.0 as u32, 0, 0),
                PackedInstrKind::Sext(v) => rec(OP_SEXT, v.0 as u32, 0, value_width[v.0]),
                PackedInstrKind::And(l, r) => rec(OP_AND, l.0 as u32, r.0 as u32, 0),
                PackedInstrKind::Or(l, r) => rec(OP_OR, l.0 as u32, r.0 as u32, 0),
                PackedInstrKind::Xor(l, r) => rec(OP_XOR, l.0 as u32, r.0 as u32, 0),
                PackedInstrKind::Add(l, r) => rec(OP_ADD, l.0 as u32, r.0 as u32, 0),
                PackedInstrKind::Sub(l, r) => rec(OP_SUB, l.0 as u32, r.0 as u32, 0),
                PackedInstrKind::Mul(l, r) => rec(OP_MUL, l.0 as u32, r.0 as u32, 0),
                PackedInstrKind::Eq(l, r) => rec(OP_EQ, l.0 as u32, r.0 as u32, value_width[l.0]),
                PackedInstrKind::Ne(l, r) => rec(OP_NE, l.0 as u32, r.0 as u32, value_width[l.0]),
                PackedInstrKind::Mux {
                    cond,
                    then_value,
                    else_value,
                } => rec(OP_MUX, cond.0 as u32, then_value.0 as u32, else_value.0 as u32),
                PackedInstrKind::Slice { value, lsb } => {
                    rec(OP_SLICE, value.0 as u32, *lsb, value_width[value.0])
                }
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

/// Simple ops a macro-op can absorb (operands are value refs / per-op immediate;
/// no `aux`-list, no memory/effect). The automatic fused-superoptimizer's vocabulary.
fn macro_simple(op: u32) -> bool {
    matches!(
        op,
        OP_NOT | OP_AND | OP_OR | OP_XOR | OP_ADD | OP_SUB | OP_MUL | OP_EQ | OP_NE
            | OP_LT_U | OP_LT_S | OP_MUX | OP_SLICE | OP_ZEXT | OP_SEXT | OP_TRUNC | OP_CAST
    )
}

/// Does this op produce a value (write the value buffer at `field1`)?
fn produces_value(op: u32) -> bool {
    !matches!(op, OP_STORE_SIGNAL | OP_CAPTURE_REG | OP_ASYNC_RESET | OP_MEM_WRITE)
}

/// For a simple op record `[op, dst, w, a, b, c]`: its operand VALUE-IDS (refs
/// that read the value buffer) and its per-op immediate (Slice lsb / Sext src-w /
/// Lt sign-w). Padded operand slots are `None`.
fn simple_operands(op: u32, a: u32, b: u32, c: u32) -> ([Option<u32>; 3], u32) {
    match op {
        OP_NOT | OP_ZEXT | OP_TRUNC | OP_CAST => ([Some(a), None, None], 0),
        OP_SLICE => ([Some(a), None, None], b),  // b = lsb
        OP_SEXT => ([Some(a), None, None], c),   // c = src width
        OP_MUX => ([Some(a), Some(b), Some(c)], 0),
        OP_LT_S => ([Some(a), Some(b), None], c), // c = sign width
        _ => ([Some(a), Some(b), None], 0),       // AND/OR/XOR/ADD/SUB/MUL/EQ/NE/LT_U
    }
}

/// Every value-id READ by a record (for use-counting, so a value read by any
/// record — incl. effects and `aux`-list ops like Concat — is never inlined).
fn record_reads(op: u32, a: u32, b: u32, c: u32, aux: &[u32]) -> Vec<u32> {
    match op {
        OP_LIT | OP_SIGNAL | OP_ASYNC_RESET => vec![],
        OP_MEM_READ | OP_STORE_SIGNAL | OP_CAPTURE_REG => vec![a],
        OP_MEM_WRITE => vec![a, b, c],
        OP_CONCAT => (0..b).map(|k| aux[(a + k * 2) as usize]).collect(),
        _ if macro_simple(op) => {
            let (ops, _) = simple_operands(op, a, b, c);
            ops.iter().flatten().copied().collect()
        }
        // menu-fused / 4-operand cells (only if fusing a menu-encoded stream):
        _ => vec![a, b, c],
    }
}

/// Post-pass that fuses maximal cones of single-use simple ops into `OP_MACRO`
/// records, so single-use intermediates are kept in macro-local registers
/// instead of round-tripping the (global) value buffer — the automatic
/// fused-superoperator. Run on a `fuse=false` (menu-off) encoding. Bit-identical;
/// reduces the value-buffer-writing record count (the GPU/interp traffic bound).
pub fn fuse_macros(program: &InterpProgram) -> InterpProgram {
    use std::collections::HashMap;
    let mut out = program.clone();
    let mut aux = program.aux.clone();
    for bi in 0..4 {
        let recs = program.blocks[bi].clone();
        let nrec = recs.len() / RECORD_WORDS;
        let get = |r: usize, f: usize| recs[r * RECORD_WORDS + f];

        // value-id -> producing record (simple ops only; for inlining/recursion).
        let mut producer: HashMap<u32, usize> = HashMap::new();
        for r in 0..nrec {
            let op = get(r, 0);
            if macro_simple(op) && produces_value(op) {
                producer.insert(get(r, 1), r);
            }
        }
        // use-count over every value read by any record, plus the op of the sole
        // consumer (valid when use_count==1).
        let mut use_count: HashMap<u32, u32> = HashMap::new();
        let mut consumer_op: HashMap<u32, u32> = HashMap::new();
        for r in 0..nrec {
            let op = get(r, 0);
            for id in record_reads(op, get(r, 3), get(r, 4), get(r, 5), &aux) {
                *use_count.entry(id).or_default() += 1;
                consumer_op.insert(id, op);
            }
        }
        // inlinable: a simple op whose value is used exactly once AND whose sole
        // consumer is itself a simple op (so it can absorb it). Multi-use values,
        // and single-use values feeding an effect / aux-list op, stay as records.
        let inlinable = |dst: u32| -> bool {
            producer.contains_key(&dst)
                && use_count.get(&dst).copied().unwrap_or(0) == 1
                && consumer_op.get(&dst).copied().map(macro_simple).unwrap_or(false)
        };

        let mut new_recs: Vec<u32> = Vec::with_capacity(recs.len());
        // root record index -> (dst, width, aux_off, n_sub) of its emitted macro.
        let mut macros: HashMap<usize, (u32, u32, u32, u32)> = HashMap::new();
        // record indices inlined into some macro (skip when copying through).
        let mut inlined: std::collections::HashSet<usize> = Default::default();

        // First pass: decide macro roots = simple ops that are NOT inlinable, and
        // build each macro by inlining its single-use simple-op operand cones.
        for r in 0..nrec {
            let op = get(r, 0);
            if !(macro_simple(op) && produces_value(op)) || inlinable(get(r, 1)) {
                continue;
            }
            // Build the macro cone rooted at record r.
            let mut sub: Vec<[u32; 6]> = Vec::new();
            let mut reg_of: HashMap<u32, u32> = HashMap::new();
            let mut local_inlined: Vec<usize> = Vec::new();
            // returns an operand ref for value-id `v` (LOCAL reg if inlined, else value-id).
            fn inline_val(
                v: u32,
                producer: &HashMap<u32, usize>,
                inlinable: &dyn Fn(u32) -> bool,
                recs: &[u32],
                sub: &mut Vec<[u32; 6]>,
                reg_of: &mut HashMap<u32, u32>,
                local_inlined: &mut Vec<usize>,
            ) -> u32 {
                if let Some(&reg) = reg_of.get(&v) {
                    return MACRO_LOCAL | reg;
                }
                // Cap macro size (the interp uses a fixed local register file); a
                // larger cone splits — the un-inlined operand stays a normal record
                // (safe: use_count==1, so it is produced exactly once).
                if !inlinable(v) || sub.len() >= MACRO_MAX_SUB - 1 {
                    return v; // external leaf
                }
                let rr = producer[&v];
                let g = |f: usize| recs[rr * RECORD_WORDS + f];
                let (op, w, a, b, c) = (g(0), g(2), g(3), g(4), g(5));
                let (ops, imm) = simple_operands(op, a, b, c);
                let mut refs = [0u32; 3];
                for (i, o) in ops.iter().enumerate() {
                    refs[i] = match o {
                        Some(id) => inline_val(*id, producer, inlinable, recs, sub, reg_of, local_inlined),
                        None => 0,
                    };
                }
                sub.push([op, w, refs[0], refs[1], refs[2], imm]);
                let reg = (sub.len() - 1) as u32;
                reg_of.insert(v, reg);
                local_inlined.push(rr);
                MACRO_LOCAL | reg
            }
            // Inline the root's operands, then push the root sub-op last.
            let (op, w, a, b, c) = (get(r, 0), get(r, 2), get(r, 3), get(r, 4), get(r, 5));
            let (ops, imm) = simple_operands(op, a, b, c);
            let mut refs = [0u32; 3];
            for (i, o) in ops.iter().enumerate() {
                refs[i] = match o {
                    Some(id) => inline_val(*id, &producer, &inlinable, &recs, &mut sub, &mut reg_of, &mut local_inlined),
                    None => 0,
                };
            }
            sub.push([op, w, refs[0], refs[1], refs[2], imm]);
            // Only worth a macro if it actually absorbed ≥1 intermediate.
            if sub.len() < 2 {
                continue;
            }
            let aux_off = aux.len() as u32;
            for s in &sub {
                aux.extend_from_slice(s);
            }
            macros.insert(r, (get(r, 1), w, aux_off, sub.len() as u32));
            for &ri in &local_inlined {
                inlined.insert(ri);
            }
        }

        // Second pass: emit records in order, replacing macro roots and skipping
        // inlined intermediates.
        for r in 0..nrec {
            if let Some(&(dst, w, aux_off, n)) = macros.get(&r) {
                push_record(&mut new_recs, OP_MACRO, dst, w, aux_off, n, 0);
            } else if !inlined.contains(&r) {
                let s = r * RECORD_WORDS;
                new_recs.extend_from_slice(&recs[s..s + RECORD_WORDS]);
            }
        }
        out.blocks[bi] = new_recs;
    }
    out.aux = aux;
    out
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

    /// Loads `words` into a 1-limb memory at `mem_offset` (its `PackedMemory.offset`),
    /// replicated identically across every lane — e.g. a program image into each
    /// lane's instruction RAM.
    pub fn set_memory_replicated(&mut self, mem_offset: usize, words: &[u32]) {
        for (w, &value) in words.iter().enumerate() {
            let base = (mem_offset + w) * self.lanes;
            for lane in 0..self.lanes {
                self.memories[base + lane] = value;
            }
        }
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
        // One reusable macro register file (dense, no per-macro allocation/zeroing —
        // topo order writes reg `i` before any sub-op reads it).
        let mut macro_regs = vec![0u128; MACRO_MAX_SUB];
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
                    OP_MULADD => read(a).wrapping_mul(read(b)).wrapping_add(read(c)) & mask,
                    OP_ADD3 => read(a).wrapping_add(read(b)).wrapping_add(read(c)) & mask,
                    OP_ANDOR => ((read(a) & read(b)) | read(c)) & mask,
                    OP_NAND => !(read(a) & read(b)) & mask,
                    OP_NOR => !(read(a) | read(b)) & mask,
                    OP_XNOR => !(read(a) ^ read(b)) & mask,
                    OP_MULSUB => read(a).wrapping_mul(read(b)).wrapping_sub(read(c)) & mask,
                    OP_XOR3 => (read(a) ^ read(b) ^ read(c)) & mask,
                    OP_ORAND => ((read(a) | read(b)) & read(c)) & mask,
                    OP_AOI21 => !((read(a) & read(b)) | read(c)) & mask,
                    OP_OAI21 => !((read(a) | read(b)) & read(c)) & mask,
                    OP_AOI22 => {
                        let (ci, di) = (aux[c as usize], aux[c as usize + 1]);
                        !((read(a) & read(b)) | (read(ci) & read(di))) & mask
                    }
                    OP_OAI22 => {
                        let (ci, di) = (aux[c as usize], aux[c as usize + 1]);
                        !((read(a) | read(b)) & (read(ci) | read(di))) & mask
                    }
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
                    OP_MACRO => {
                        // a = aux offset, b = #sub-ops; evaluate the sub-DAG into
                        // local regs (no value-buffer round-trip), return the root.
                        let aux_off = a as usize;
                        let n = b as usize;
                        for i in 0..n {
                            let so = aux_off + i * 6;
                            let (sop, sw) = (aux[so], aux[so + 1]);
                            let (ra, rb, rc, imm) = (aux[so + 2], aux[so + 3], aux[so + 4], aux[so + 5]);
                            let sm = mask128(sw);
                            // inline reads (no closure → no persistent borrow of macro_regs).
                            let va = if ra & MACRO_LOCAL != 0 { macro_regs[(ra & MACRO_LOCAL_MASK) as usize] } else { read(ra) };
                            let vb = if rb & MACRO_LOCAL != 0 { macro_regs[(rb & MACRO_LOCAL_MASK) as usize] } else { read(rb) };
                            let vc = if rc & MACRO_LOCAL != 0 { macro_regs[(rc & MACRO_LOCAL_MASK) as usize] } else { read(rc) };
                            macro_regs[i] = match sop {
                                OP_NOT => !va & sm,
                                OP_AND => va & vb & sm,
                                OP_OR => (va | vb) & sm,
                                OP_XOR => (va ^ vb) & sm,
                                OP_ADD => va.wrapping_add(vb) & sm,
                                OP_SUB => va.wrapping_sub(vb) & sm,
                                OP_MUL => va.wrapping_mul(vb) & sm,
                                OP_EQ => u128::from(va == vb),
                                OP_NE => u128::from(va != vb),
                                OP_LT_U => u128::from(va < vb),
                                OP_LT_S => {
                                    let sign = 1u128 << (imm - 1);
                                    let (ls, rs) = (va & sign != 0, vb & sign != 0);
                                    u128::from(if ls != rs { ls } else { va < vb })
                                }
                                OP_MUX => {
                                    if va & 1 != 0 {
                                        vb
                                    } else {
                                        vc
                                    }
                                }
                                OP_SLICE => (va >> imm) & sm,
                                OP_ZEXT | OP_TRUNC | OP_CAST => va & sm,
                                OP_SEXT => {
                                    let srcm = mask128(imm);
                                    let sign = 1u128 << (imm - 1);
                                    let x = va & srcm;
                                    (if x & sign != 0 { x | !srcm } else { x }) & sm
                                }
                                _ => 0,
                            };
                        }
                        macro_regs[n.saturating_sub(1)]
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
  var regs: array<u32, 256>;  // macro-op local register file (reused per macro)
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
      case 25u: { res = (vget(a, lanes, lane) * vget(b, lanes, lane) + vget(c, lanes, lane)) & mask; }
      case 26u: { res = (vget(a, lanes, lane) + vget(b, lanes, lane) + vget(c, lanes, lane)) & mask; }
      case 27u: { res = ((vget(a, lanes, lane) & vget(b, lanes, lane)) | vget(c, lanes, lane)) & mask; }
      case 28u: { res = (~(vget(a, lanes, lane) & vget(b, lanes, lane))) & mask; }
      case 29u: { res = (~(vget(a, lanes, lane) | vget(b, lanes, lane))) & mask; }
      case 30u: { res = (~(vget(a, lanes, lane) ^ vget(b, lanes, lane))) & mask; }
      case 31u: { res = (vget(a, lanes, lane) * vget(b, lanes, lane) - vget(c, lanes, lane)) & mask; }
      case 32u: { res = (vget(a, lanes, lane) ^ vget(b, lanes, lane) ^ vget(c, lanes, lane)) & mask; }
      case 33u: { res = ((vget(a, lanes, lane) | vget(b, lanes, lane)) & vget(c, lanes, lane)) & mask; }
      case 34u: { res = (~((vget(a, lanes, lane) & vget(b, lanes, lane)) | vget(c, lanes, lane))) & mask; }
      case 35u: { res = (~((vget(a, lanes, lane) | vget(b, lanes, lane)) & vget(c, lanes, lane))) & mask; }
      case 36u: { let ci = aux[c]; let di = aux[c + 1u]; res = (~((vget(a, lanes, lane) & vget(b, lanes, lane)) | (vget(ci, lanes, lane) & vget(di, lanes, lane)))) & mask; }
      case 37u: { let ci = aux[c]; let di = aux[c + 1u]; res = (~((vget(a, lanes, lane) | vget(b, lanes, lane)) & (vget(ci, lanes, lane) | vget(di, lanes, lane)))) & mask; }
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
      case 38u: {
        // OP_MACRO: a = aux offset, b = #sub-ops. Evaluate the sub-DAG into local
        // regs (no value-buffer round-trip), result = root reg.
        let n = b;
        for (var i = 0u; i < n; i = i + 1u) {
          let so = a + i * 6u;
          let sop = aux[so];
          let sm = mask_of(aux[so + 1u]);
          let ra = aux[so + 2u]; let rb = aux[so + 3u]; let rc = aux[so + 4u]; let imm = aux[so + 5u];
          let va = select(vget(ra, lanes, lane), regs[ra & 0x7fffffffu], (ra & 0x80000000u) != 0u);
          let vb = select(vget(rb, lanes, lane), regs[rb & 0x7fffffffu], (rb & 0x80000000u) != 0u);
          let vc = select(vget(rc, lanes, lane), regs[rc & 0x7fffffffu], (rc & 0x80000000u) != 0u);
          var rv = 0u;
          switch sop {
            case 2u: { rv = (~va) & sm; }
            case 3u: { rv = (va & vb) & sm; }
            case 4u: { rv = (va | vb) & sm; }
            case 5u: { rv = (va ^ vb) & sm; }
            case 6u: { rv = (va + vb) & sm; }
            case 7u: { rv = (va - vb) & sm; }
            case 8u: { rv = (va * vb) & sm; }
            case 9u: { rv = select(0u, 1u, va == vb); }
            case 10u: { rv = select(0u, 1u, va != vb); }
            case 11u: { rv = select(0u, 1u, va < vb); }
            case 12u: {
              let sgn = 1u << (imm - 1u);
              let ls = (va & sgn) != 0u; let rs = (vb & sgn) != 0u;
              if (ls != rs) { rv = select(0u, 1u, ls); } else { rv = select(0u, 1u, va < vb); }
            }
            case 13u: { rv = select(vc, vb, (va & 1u) != 0u); }
            case 14u: { rv = (va >> imm) & sm; }
            case 15u, 17u, 18u: { rv = va & sm; }
            case 16u: {
              let srcm = mask_of(imm); let sgn = 1u << (imm - 1u); let x = va & srcm;
              if ((x & sgn) != 0u) { rv = (x | ~srcm) & sm; } else { rv = x & sm; }
            }
            default: {}
          }
          regs[i] = rv;
        }
        res = regs[n - 1u];
      }
      default: {}
    }
    if (!is_effect) { values[f1 * lanes + lane] = res; }
    r = r + 1u;
  }
}

// Multi-limb path. `ml` is the uniform value-slot stride; each op processes only
// its own limb count (nl = result limbs, ol = operand limbs from `c`) so narrow
// ops in a wide-max-limbs design pay only their own width. Multiply uses 16-bit
// half-products (16x16 fits u32). Slots high limbs stay valid because no op reads
// past a value's own width.
fn vload_ml(base: u32, count: u32, lanes: u32, lane: u32) -> array<u32, 4> {
  var v = array<u32, 4>(0u, 0u, 0u, 0u);
  for (var l = 0u; l < count; l = l + 1u) { v[l] = values[(base + l) * lanes + lane]; }
  return v;
}

// Packed scalar load: value-id `id` lives at voff[id] = aux[vb + id].
fn vget_p(id: u32, vb: u32, lanes: u32, lane: u32) -> u32 { return values[aux[vb + id] * lanes + lane]; }

fn run_block_ml(begin: u32, end: u32, lanes: u32, lane: u32, ml: u32) {
  let vb = params[14u];
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
    let ol = (c + 31u) / 32u; // operand limbs (ops that set c)

    // 1-limb fast path. Result is 1 limb when nl<=1; operands match result width
    // for most ops, so nl<=1 suffices. Width-changing ops (eq/ne/slt/sge/slice/
    // zext/sext) encode operand width in c, so they also require ol<=1.
    var narrow = nl <= 1u;
    if (op == 9u || op == 10u || op == 11u || op == 12u || op == 14u || op == 15u || op == 16u) {
      narrow = narrow && (ol <= 1u);
    }
    if (narrow) {
      let mask = mask_of(width);
      var sres = 0u;
      var seff = false;
      switch op {
        case 0u: { sres = aux[a] & mask; }
        case 1u: { sres = sig[a * lanes + lane] & mask; }
        case 2u: { sres = (~vget_p(a, vb, lanes, lane)) & mask; }
        case 3u: { sres = (vget_p(a, vb, lanes, lane) & vget_p(b, vb, lanes, lane)) & mask; }
        case 4u: { sres = (vget_p(a, vb, lanes, lane) | vget_p(b, vb, lanes, lane)) & mask; }
        case 5u: { sres = (vget_p(a, vb, lanes, lane) ^ vget_p(b, vb, lanes, lane)) & mask; }
        case 6u: { sres = (vget_p(a, vb, lanes, lane) + vget_p(b, vb, lanes, lane)) & mask; }
        case 7u: { sres = (vget_p(a, vb, lanes, lane) - vget_p(b, vb, lanes, lane)) & mask; }
        case 8u: { sres = (vget_p(a, vb, lanes, lane) * vget_p(b, vb, lanes, lane)) & mask; }
        case 25u: { sres = (vget_p(a, vb, lanes, lane) * vget_p(b, vb, lanes, lane) + vget_p(c, vb, lanes, lane)) & mask; }
        case 26u: { sres = (vget_p(a, vb, lanes, lane) + vget_p(b, vb, lanes, lane) + vget_p(c, vb, lanes, lane)) & mask; }
        case 27u: { sres = ((vget_p(a, vb, lanes, lane) & vget_p(b, vb, lanes, lane)) | vget_p(c, vb, lanes, lane)) & mask; }
        case 28u: { sres = (~(vget_p(a, vb, lanes, lane) & vget_p(b, vb, lanes, lane))) & mask; }
        case 29u: { sres = (~(vget_p(a, vb, lanes, lane) | vget_p(b, vb, lanes, lane))) & mask; }
        case 30u: { sres = (~(vget_p(a, vb, lanes, lane) ^ vget_p(b, vb, lanes, lane))) & mask; }
        case 31u: { sres = (vget_p(a, vb, lanes, lane) * vget_p(b, vb, lanes, lane) - vget_p(c, vb, lanes, lane)) & mask; }
        case 32u: { sres = (vget_p(a, vb, lanes, lane) ^ vget_p(b, vb, lanes, lane) ^ vget_p(c, vb, lanes, lane)) & mask; }
        case 33u: { sres = ((vget_p(a, vb, lanes, lane) | vget_p(b, vb, lanes, lane)) & vget_p(c, vb, lanes, lane)) & mask; }
        case 34u: { sres = (~((vget_p(a, vb, lanes, lane) & vget_p(b, vb, lanes, lane)) | vget_p(c, vb, lanes, lane))) & mask; }
        case 35u: { sres = (~((vget_p(a, vb, lanes, lane) | vget_p(b, vb, lanes, lane)) & vget_p(c, vb, lanes, lane))) & mask; }
        case 36u: { let ci = aux[c]; let di = aux[c + 1u]; sres = (~((vget_p(a, vb, lanes, lane) & vget_p(b, vb, lanes, lane)) | (vget_p(ci, vb, lanes, lane) & vget_p(di, vb, lanes, lane)))) & mask; }
        case 37u: { let ci = aux[c]; let di = aux[c + 1u]; sres = (~((vget_p(a, vb, lanes, lane) | vget_p(b, vb, lanes, lane)) & (vget_p(ci, vb, lanes, lane) | vget_p(di, vb, lanes, lane)))) & mask; }
        case 9u: { sres = select(0u, 1u, vget_p(a, vb, lanes, lane) == vget_p(b, vb, lanes, lane)); }
        case 10u: { sres = select(0u, 1u, vget_p(a, vb, lanes, lane) != vget_p(b, vb, lanes, lane)); }
        case 11u: { sres = select(0u, 1u, vget_p(a, vb, lanes, lane) < vget_p(b, vb, lanes, lane)); }
        case 12u: {
          let sign = 1u << (c - 1u);
          let l = vget_p(a, vb, lanes, lane);
          let rr = vget_p(b, vb, lanes, lane);
          let ls = (l & sign) != 0u; let rs = (rr & sign) != 0u;
          if (ls != rs) { sres = select(0u, 1u, ls); } else { sres = select(0u, 1u, l < rr); }
        }
        case 13u: {
          if ((vget_p(a, vb, lanes, lane) & 1u) != 0u) { sres = vget_p(b, vb, lanes, lane); }
          else { sres = vget_p(c, vb, lanes, lane); }
        }
        case 14u: { sres = (vget_p(a, vb, lanes, lane) >> b) & mask; }
        case 15u, 17u, 18u: { sres = vget_p(a, vb, lanes, lane) & mask; }
        case 16u: {
          let src_mask = mask_of(c);
          let sign = 1u << (c - 1u);
          let v = vget_p(a, vb, lanes, lane) & src_mask;
          if ((v & sign) != 0u) { sres = (v | (~src_mask)) & mask; } else { sres = v & mask; }
        }
        case 19u: { sig[f1 * lanes + lane] = vget_p(a, vb, lanes, lane) & mask; seff = true; }
        case 20u: {
          var next = vget_p(a, vb, lanes, lane) & mask;
          if (b != 0xffffffffu && reset_on(b, lanes, lane)) { next = aux[b + 2u] & mask; }
          reg_next[f1 * lanes + lane] = next; seff = true;
        }
        case 21u: {
          let addr = vget_p(a, vb, lanes, lane);
          if (addr < c) { sres = mem[(b + addr) * lanes + lane] & mask; } else { sres = 0u; }
        }
        case 22u: {
          let addr = vget_p(b, vb, lanes, lane);
          if (((vget_p(a, vb, lanes, lane) & 1u) != 0u) && (addr < width)) {
            mem[(f1 + addr) * lanes + lane] = vget_p(c, vb, lanes, lane);
          }
          seff = true;
        }
        case 23u: {
          var result = 0u; var ofs = 0u; var k = b;
          loop {
            if (k == 0u) { break; }
            k = k - 1u;
            let part = vget_p(aux[a + k * 2u], vb, lanes, lane) & mask_of(aux[a + k * 2u + 1u]);
            if (ofs < 32u) { result = result | (part << ofs); }
            ofs = ofs + aux[a + k * 2u + 1u];
          }
          sres = result & mask;
        }
        case 24u: {
          if (reset_on(b, lanes, lane)) { sig[f1 * lanes + lane] = aux[b + 2u] & mask; }
          seff = true;
        }
        default: {}
      }
      if (!seff) { values[aux[vb + f1] * lanes + lane] = sres; }
      r = r + 1u; continue;
    }

    var res = array<u32, 4>(0u, 0u, 0u, 0u);
    var is_effect = false;

    switch op {
      case 0u: { for (var l = 0u; l < nl; l = l + 1u) { res[l] = aux[a + l]; } }
      case 1u: { for (var l = 0u; l < nl; l = l + 1u) { res[l] = sig[(a + l) * lanes + lane]; } }
      case 2u: { var x = vload_ml(aux[vb + a], nl, lanes, lane); for (var l = 0u; l < nl; l = l + 1u) { res[l] = ~x[l]; } }
      case 3u: { var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane); for (var l = 0u; l < nl; l = l + 1u) { res[l] = x[l] & y[l]; } }
      case 4u: { var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane); for (var l = 0u; l < nl; l = l + 1u) { res[l] = x[l] | y[l]; } }
      case 5u: { var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane); for (var l = 0u; l < nl; l = l + 1u) { res[l] = x[l] ^ y[l]; } }
      case 6u: {
        var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane);
        var carry = 0u;
        for (var l = 0u; l < nl; l = l + 1u) {
          let s1 = x[l] + y[l]; let c1 = select(0u, 1u, s1 < x[l]);
          let s2 = s1 + carry; let c2 = select(0u, 1u, s2 < s1);
          res[l] = s2; carry = c1 + c2;
        }
      }
      case 7u: {
        var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane);
        var borrow = 0u;
        for (var l = 0u; l < nl; l = l + 1u) {
          let d1 = x[l] - y[l]; let b1 = select(0u, 1u, x[l] < y[l]);
          let d2 = d1 - borrow; let b2 = select(0u, 1u, d1 < borrow);
          res[l] = d2; borrow = b1 + b2;
        }
      }
      case 8u: {
        var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane);
        let halves = nl * 2u;
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
        for (var l = 0u; l < nl; l = l + 1u) { res[l] = acc[2u * l] | (acc[2u * l + 1u] << 16u); }
      }
      case 25u: {
        var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane);
        let halves = nl * 2u;
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
        var z = vload_ml(aux[vb + c], nl, lanes, lane);
        var carry2 = 0u;
        for (var l = 0u; l < nl; l = l + 1u) {
          let prod = acc[2u * l] | (acc[2u * l + 1u] << 16u);
          let s1 = prod + z[l]; let c1 = select(0u, 1u, s1 < prod);
          let s2 = s1 + carry2; let c2 = select(0u, 1u, s2 < s1);
          res[l] = s2; carry2 = c1 + c2;
        }
      }
      case 26u: {
        var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane); var z = vload_ml(aux[vb + c], nl, lanes, lane);
        var carry = 0u;
        for (var l = 0u; l < nl; l = l + 1u) {
          let s1 = x[l] + y[l]; let c1 = select(0u, 1u, s1 < x[l]);
          let s2 = s1 + z[l]; let c2 = select(0u, 1u, s2 < s1);
          let s3 = s2 + carry; let c3 = select(0u, 1u, s3 < s2);
          res[l] = s3; carry = c1 + c2 + c3;
        }
      }
      case 27u: {
        var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane); var z = vload_ml(aux[vb + c], nl, lanes, lane);
        for (var l = 0u; l < nl; l = l + 1u) { res[l] = (x[l] & y[l]) | z[l]; }
      }
      case 28u: { var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane); for (var l = 0u; l < nl; l = l + 1u) { res[l] = ~(x[l] & y[l]); } }
      case 29u: { var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane); for (var l = 0u; l < nl; l = l + 1u) { res[l] = ~(x[l] | y[l]); } }
      case 30u: { var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane); for (var l = 0u; l < nl; l = l + 1u) { res[l] = ~(x[l] ^ y[l]); } }
      case 31u: {
        var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane);
        let halves = nl * 2u;
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
        var z = vload_ml(aux[vb + c], nl, lanes, lane);
        var borrow = 0u;
        for (var l = 0u; l < nl; l = l + 1u) {
          let prod = acc[2u * l] | (acc[2u * l + 1u] << 16u);
          let d1 = prod - z[l]; let b1 = select(0u, 1u, prod < z[l]);
          let d2 = d1 - borrow; let b2 = select(0u, 1u, d1 < borrow);
          res[l] = d2; borrow = b1 + b2;
        }
      }
      case 32u: { var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane); var z = vload_ml(aux[vb + c], nl, lanes, lane); for (var l = 0u; l < nl; l = l + 1u) { res[l] = x[l] ^ y[l] ^ z[l]; } }
      case 33u: { var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane); var z = vload_ml(aux[vb + c], nl, lanes, lane); for (var l = 0u; l < nl; l = l + 1u) { res[l] = (x[l] | y[l]) & z[l]; } }
      case 34u: { var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane); var z = vload_ml(aux[vb + c], nl, lanes, lane); for (var l = 0u; l < nl; l = l + 1u) { res[l] = ~((x[l] & y[l]) | z[l]); } }
      case 35u: { var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane); var z = vload_ml(aux[vb + c], nl, lanes, lane); for (var l = 0u; l < nl; l = l + 1u) { res[l] = ~((x[l] | y[l]) & z[l]); } }
      case 36u: {
        var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane);
        var z = vload_ml(aux[vb + aux[c]], nl, lanes, lane); var w2 = vload_ml(aux[vb + aux[c + 1u]], nl, lanes, lane);
        for (var l = 0u; l < nl; l = l + 1u) { res[l] = ~((x[l] & y[l]) | (z[l] & w2[l])); }
      }
      case 37u: {
        var x = vload_ml(aux[vb + a], nl, lanes, lane); var y = vload_ml(aux[vb + b], nl, lanes, lane);
        var z = vload_ml(aux[vb + aux[c]], nl, lanes, lane); var w2 = vload_ml(aux[vb + aux[c + 1u]], nl, lanes, lane);
        for (var l = 0u; l < nl; l = l + 1u) { res[l] = ~((x[l] | y[l]) & (z[l] | w2[l])); }
      }
      case 9u: { var x = vload_ml(aux[vb + a], ol, lanes, lane); var y = vload_ml(aux[vb + b], ol, lanes, lane); var eq = true; for (var l = 0u; l < ol; l = l + 1u) { if (x[l] != y[l]) { eq = false; } } res[0] = select(0u, 1u, eq); }
      case 10u: { var x = vload_ml(aux[vb + a], ol, lanes, lane); var y = vload_ml(aux[vb + b], ol, lanes, lane); var eq = true; for (var l = 0u; l < ol; l = l + 1u) { if (x[l] != y[l]) { eq = false; } } res[0] = select(0u, 1u, !eq); }
      case 11u: {
        var x = vload_ml(aux[vb + a], ol, lanes, lane); var y = vload_ml(aux[vb + b], ol, lanes, lane);
        var lt = 0u; var done = false; var l = ol;
        loop { if (l == 0u) { break; } l = l - 1u; if (!done && x[l] != y[l]) { lt = select(0u, 1u, x[l] < y[l]); done = true; } }
        res[0] = lt;
      }
      case 12u: {
        var x = vload_ml(aux[vb + a], ol, lanes, lane); var y = vload_ml(aux[vb + b], ol, lanes, lane);
        let sl = (c - 1u) / 32u; let sb = 1u << ((c - 1u) & 31u);
        let xs = (x[sl] & sb) != 0u; let ys = (y[sl] & sb) != 0u;
        if (xs != ys) { res[0] = select(0u, 1u, xs); }
        else {
          var lt = 0u; var done = false; var l = ol;
          loop { if (l == 0u) { break; } l = l - 1u; if (!done && x[l] != y[l]) { lt = select(0u, 1u, x[l] < y[l]); done = true; } }
          res[0] = lt;
        }
      }
      case 13u: {
        let cond = (values[aux[vb + a] * lanes + lane] & 1u) != 0u;
        var xt = vload_ml(aux[vb + b], nl, lanes, lane); var xf = vload_ml(aux[vb + c], nl, lanes, lane);
        for (var l = 0u; l < nl; l = l + 1u) { res[l] = select(xf[l], xt[l], cond); }
      }
      case 14u: {
        var x = vload_ml(aux[vb + a], ol, lanes, lane); // c = operand width
        let ls = b; let lsh = ls / 32u; let bsh = ls & 31u;
        for (var l = 0u; l < nl; l = l + 1u) {
          let s0 = l + lsh;
          var v = 0u;
          if (s0 < ol) { v = x[s0] >> bsh; }
          if (bsh != 0u && s0 + 1u < ol) { v = v | (x[s0 + 1u] << (32u - bsh)); }
          res[l] = v;
        }
      }
      case 15u: { var x = vload_ml(aux[vb + a], ol, lanes, lane); for (var l = 0u; l < nl; l = l + 1u) { res[l] = x[l]; } }
      case 17u, 18u: { var x = vload_ml(aux[vb + a], nl, lanes, lane); for (var l = 0u; l < nl; l = l + 1u) { res[l] = x[l]; } }
      case 16u: {
        var x = vload_ml(aux[vb + a], ol, lanes, lane);
        let sw = c; let sl = (sw - 1u) / 32u; let sb = 1u << ((sw - 1u) & 31u);
        let neg = (x[sl] & sb) != 0u;
        for (var l = 0u; l < nl; l = l + 1u) {
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
        var x = vload_ml(aux[vb + a], nl, lanes, lane);
        for (var l = 0u; l < nl; l = l + 1u) { sig[(f1 + l) * lanes + lane] = x[l] & limb_mask(width, l); }
        is_effect = true;
      }
      case 20u: {
        var src = vload_ml(aux[vb + a], nl, lanes, lane);
        if (b != 0xffffffffu && reset_on(b, lanes, lane)) { for (var l = 0u; l < nl; l = l + 1u) { src[l] = aux[b + 2u + l]; } }
        for (var l = 0u; l < nl; l = l + 1u) { reg_next[(f1 + l) * lanes + lane] = src[l] & limb_mask(width, l); }
        is_effect = true;
      }
      case 21u: {
        let addr = values[aux[vb + a] * lanes + lane];
        if (addr < c) { res[0] = mem[(b + addr) * lanes + lane]; }
        is_effect = false;
      }
      case 22u: {
        let en = values[aux[vb + a] * lanes + lane] & 1u;
        let addr = values[aux[vb + b] * lanes + lane];
        if (en != 0u && addr < width) { mem[(f1 + addr) * lanes + lane] = values[aux[vb + c] * lanes + lane]; }
        is_effect = true;
      }
      case 23u: {
        var bitofs = 0u;
        var k = b;
        loop {
          if (k == 0u) { break; }
          k = k - 1u;
          let vid = aux[a + k * 2u]; let w = aux[a + k * 2u + 1u];
          let wl = (w + 31u) / 32u;
          var part = vload_ml(aux[vb + vid], wl, lanes, lane);
          let dl = bitofs / 32u; let dsh = bitofs & 31u;
          for (var pl = 0u; pl < wl; pl = pl + 1u) {
            let pm = part[pl] & limb_mask(w, pl);
            if (dl + pl < nl) { res[dl + pl] = res[dl + pl] | (pm << dsh); }
            if (dsh != 0u && dl + pl + 1u < nl) { res[dl + pl + 1u] = res[dl + pl + 1u] | (pm >> (32u - dsh)); }
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
    if (!is_effect) { for (var l = 0u; l < nl; l = l + 1u) { values[(aux[vb + f1] + l) * lanes + lane] = res[l] & limb_mask(width, l); } }
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
            program.value_offsets_base as u32, // params[14]: aux base of voff table
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
        let value_words = program.total_value_words * lanes;
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

    /// Loads `words` into a 1-limb memory at `mem_offset` (its `PackedMemory.offset`),
    /// replicated identically across every lane (a program image into each lane's RAM).
    pub fn set_memory_replicated(&self, mem_offset: usize, words: &[u32]) {
        // Memory is word-major like signals: word w of lane l is at (mem_offset+w)*lanes+l.
        let mut region = vec![0u32; words.len() * self.lanes];
        for (w, &value) in words.iter().enumerate() {
            for lane in 0..self.lanes {
                region[w * self.lanes + lane] = value;
            }
        }
        self.queue.write_buffer(
            &self.mem_buffer,
            (mem_offset * self.lanes * 4) as u64,
            bytemuck::cast_slice(&region),
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

    /// Specialization (const-fold + identities + DCE + compaction) must produce
    /// a program that is bit-for-bit equivalent to the original, and must reduce
    /// the instruction count on a design rich in foldable constants.
    #[test]
    fn specialized_program_matches_original() {
        use rrtl_sim_ir::specialize_program;

        let mut design = Design::new();
        let (clk, din, din64, o32, o64);
        {
            let mut m = design.module("Spec");
            clk = m.input("clk", uint(1));
            din = m.input("din", uint(32));
            din64 = m.input("din64", uint(64));
            // (100 + 23) * 1 -> 123  (full fold + mul-by-one copy)
            let k = m.wire("k", uint(32));
            m.assign(k, (lit_u(100, 32) + lit_u(23, 32)) * lit_u(1, 32));
            // din | 0 -> din  (identity copy)
            let passthru = m.wire("passthru", uint(32));
            m.assign(passthru, din | lit_u(0, 32));
            // surviving signal-dependent work: acc' = acc*C + k + passthru
            let acc = m.reg("acc", uint(32));
            m.clock(acc, clk);
            m.next(acc, acc * lit_u(0x9e37_79b9, 32) + k + passthru);
            o32 = m.output("o32", uint(32));
            m.assign(o32, acc);
            // 64-bit constant fold: 0x1_0000_0000 + 5 -> 0x1_0000_0005 (2 limbs)
            let big = m.reg("big", uint(64));
            m.clock(big, clk);
            m.next(big, big + (lit_u(0x1_0000_0000, 64) + lit_u(5, 64)) + din64);
            o64 = m.output("o64", uint(64));
            m.assign(o64, big);
        }

        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Spec").unwrap();
        let machine = lower_to_machine_program(&program);
        let (spec, stats) = specialize_program(&machine);
        assert!(
            stats.instrs_after < stats.instrs_before,
            "expected fewer instructions after specialization: {stats:?}"
        );
        assert!(stats.folded > 0 && stats.copies > 0, "{stats:?}");

        let offset = |sig: rrtl_ir::Signal| {
            program.signals[program.signal_index(sig).unwrap()].layout.offset
        };
        let lanes = 6;
        let mut orig = InterpRunner::new(InterpProgram::encode(&machine).unwrap(), lanes);
        let mut opt = InterpRunner::new(InterpProgram::encode(&spec).unwrap(), lanes);
        for r in [&mut orig, &mut opt] {
            r.set_signal(offset(clk), &vec![1u32; lanes]);
        }

        for cycle in 0..12u32 {
            let din_v: Vec<u32> = (0..lanes as u32)
                .map(|l| cycle.wrapping_mul(2654435761).wrapping_add(l * 13))
                .collect();
            let din64_v: Vec<u128> = (0..lanes as u128)
                .map(|l| l.wrapping_mul(0x1234_5678_9abc).wrapping_add(cycle as u128))
                .collect();
            for r in [&mut orig, &mut opt] {
                r.set_signal(offset(din), &din_v);
                r.set_signal_wide(offset(din64), 2, &din64_v);
                r.tick();
            }
            assert_eq!(
                orig.get_signal(offset(o32)),
                opt.get_signal(offset(o32)),
                "o32 mismatch at cycle {cycle}"
            );
            assert_eq!(
                orig.get_signal_wide(offset(o64), 2),
                opt.get_signal_wide(offset(o64), 2),
                "o64 mismatch at cycle {cycle}"
            );
        }
    }

    /// Fused superoperators (Add3, AndOr) plus a wide 64-bit path must match the
    /// independent SimdCpuSimulator engine.
    #[test]
    fn fused_ops_match_simd_cpu() {
        let mut design = Design::new();
        let (clk, din, e, f, g, sum_o, aoi_o, wide_o);
        {
            let mut m = design.module("Top");
            clk = m.input("clk", uint(1));
            din = m.input("din", uint(32));
            e = m.input("e", uint(32));
            f = m.input("f", uint(32));
            g = m.input("g", uint(32));
            // adder chain -> Add3 fusion
            let sum = m.reg("sum", uint(32));
            m.clock(sum, clk);
            m.next(sum, sum + din + e + f);
            sum_o = m.output("sum_o", uint(32));
            m.assign(sum_o, sum);
            // (din & e) | (f & g) -> AndOr fusion
            let aoi = m.reg("aoi", uint(32));
            m.clock(aoi, clk);
            m.next(aoi, (din & e) | (f & g));
            aoi_o = m.output("aoi_o", uint(32));
            m.assign(aoi_o, aoi);
            // 64-bit adder chain -> wide Add3 path
            let wide = m.reg("wide", uint(64));
            m.clock(wide, clk);
            let din64 = m.input("din64", uint(64));
            m.next(wide, wide + din64 + lit_u(0x1_0000_0001, 64));
            wide_o = m.output("wide_o", uint(64));
            m.assign(wide_o, wide);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();
        let off = |s: rrtl_ir::Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;
        let din64 = compiled
            .find_module("Top")
            .and_then(|mm| mm.signals.iter().find(|s| s.name == "din64"))
            .map(|s| s.handle)
            .unwrap();

        let lanes = 5;
        let mut runner = InterpRunner::new(InterpProgram::encode_design(&program).unwrap(), lanes);
        let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        runner.set_signal(off(clk), &vec![1u32; lanes]);
        cpu.set_signal(clk, &vec![1u128; lanes]).unwrap();

        for cycle in 0..10u32 {
            let v = |base: u32| -> Vec<u32> {
                (0..lanes as u32).map(|l| base.wrapping_mul(2654435761).wrapping_add(l * 7 + cycle)).collect()
            };
            for (s, base) in [(din, 1u32), (e, 2), (f, 3), (g, 4)] {
                let val = v(base);
                runner.set_signal(off(s), &val);
                cpu.set_signal(s, &val.iter().map(|&x| x as u128).collect::<Vec<_>>()).unwrap();
            }
            let w64: Vec<u128> = (0..lanes as u128).map(|l| l.wrapping_mul(0x1_0000_0007).wrapping_add(cycle as u128)).collect();
            runner.set_signal_wide(off(din64), 2, &w64);
            cpu.set_signal(din64, &w64).unwrap();

            runner.tick();
            cpu.tick();

            for s in [sum_o, aoi_o] {
                let r: Vec<u32> = runner.get_signal(off(s));
                let c: Vec<u32> = cpu.get_signal(s).unwrap().iter().map(|&x| x as u32).collect();
                assert_eq!(r, c, "{s:?} mismatch at cycle {cycle}");
            }
            let rw = runner.get_signal_wide(off(wide_o), 2);
            let cw = cpu.get_signal(wide_o).unwrap();
            assert_eq!(rw, cw, "wide_o mismatch at cycle {cycle}");
        }
    }

    /// The extended fused menu (NAND/NOR/XNOR, MulSub, Xor3, OrAnd), including
    /// wide 64-bit paths, must match the independent SimdCpuSimulator.
    #[test]
    fn extended_fused_ops_match_simd_cpu() {
        let mut design = Design::new();
        let (clk, a, b, c, a64, b64, c64);
        {
            let mut m = design.module("Top");
            clk = m.input("clk", uint(1));
            a = m.input("a", uint(32));
            b = m.input("b", uint(32));
            c = m.input("c", uint(32));
            a64 = m.input("a64", uint(64));
            b64 = m.input("b64", uint(64));
            c64 = m.input("c64", uint(64));
            let mut reg_eq = |name: &str, expr: rrtl_core::Expr| {
                let r = m.reg(name, uint(32));
                m.clock(r, clk);
                m.next(r, expr);
                let o = m.output(format!("{name}_o"), uint(32));
                m.assign(o, r);
            };
            reg_eq("nand", !(a & b));
            reg_eq("nor", !(a | b));
            reg_eq("xnor", !(a ^ b));
            reg_eq("msub", a * b - c);
            reg_eq("xor3", a ^ b ^ c);
            reg_eq("orand", (a | b) & c);
            // wide 64-bit MulSub + Xor3
            let mut wreg = |name: &str, expr: rrtl_core::Expr| {
                let r = m.reg(name, uint(64));
                m.clock(r, clk);
                m.next(r, expr);
                let o = m.output(format!("{name}_o"), uint(64));
                m.assign(o, r);
            };
            wreg("wmsub", a64 * b64 - c64);
            wreg("wxor3", a64 ^ b64 ^ c64);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();
        let off = |s: rrtl_ir::Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;
        let handle = |name: &str| compiled.find_module("Top").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle;

        let lanes = 5;
        let mut runner = InterpRunner::new(InterpProgram::encode_design(&program).unwrap(), lanes);
        let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        runner.set_signal(off(clk), &vec![1u32; lanes]);
        cpu.set_signal(clk, &vec![1u128; lanes]).unwrap();

        for cycle in 0..8u32 {
            for (s, base) in [(a, 1u32), (b, 2), (c, 3)] {
                let v: Vec<u32> = (0..lanes as u32).map(|l| base.wrapping_mul(2654435761).wrapping_add(l * 5 + cycle)).collect();
                runner.set_signal(off(s), &v);
                cpu.set_signal(s, &v.iter().map(|&x| x as u128).collect::<Vec<_>>()).unwrap();
            }
            for (s, base) in [(a64, 7u128), (b64, 11), (c64, 13)] {
                let v: Vec<u128> = (0..lanes as u128).map(|l| l.wrapping_mul(base.wrapping_mul(0x1_0000_0007)).wrapping_add(cycle as u128)).collect();
                runner.set_signal_wide(off(s), 2, &v);
                cpu.set_signal(s, &v).unwrap();
            }
            runner.tick();
            cpu.tick();
            for name in ["nand", "nor", "xnor", "msub", "xor3", "orand"] {
                let h = handle(&format!("{name}_o"));
                let r = runner.get_signal(off(h));
                let cc: Vec<u32> = cpu.get_signal(h).unwrap().iter().map(|&x| x as u32).collect();
                assert_eq!(r, cc, "{name} mismatch at cycle {cycle}");
            }
            for name in ["wmsub", "wxor3"] {
                let h = handle(&format!("{name}_o"));
                assert_eq!(runner.get_signal_wide(off(h), 2), cpu.get_signal(h).unwrap(), "{name} mismatch at cycle {cycle}");
            }
        }
    }

    /// Two-level standard-cell fusions (AOI21 = ~((a&b)|c), OAI21 = ~((a|b)&c))
    /// must match SimdCpuSimulator, and must not corrupt nearby one-level gates.
    #[test]
    fn aoi_oai_match_simd_cpu() {
        let mut design = Design::new();
        let (clk, a, b, c);
        {
            let mut m = design.module("Top");
            clk = m.input("clk", uint(1));
            a = m.input("a", uint(32));
            b = m.input("b", uint(32));
            c = m.input("c", uint(32));
            let mut reg_eq = |name: &str, expr: rrtl_core::Expr| {
                let r = m.reg(name, uint(32));
                m.clock(r, clk);
                m.next(r, expr);
                let o = m.output(format!("{name}_o"), uint(32));
                m.assign(o, r);
            };
            reg_eq("aoi", !((a & b) | c)); // AOI21
            reg_eq("oai", !((a | b) & c)); // OAI21
            // one-level gate adjacent, to confirm the deeper match doesn't steal it
            reg_eq("nand", !(a & b));
            reg_eq("plainor", (a & b) | c); // ANDOR (no invert)
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();
        let off = |s: rrtl_ir::Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;
        let handle = |name: &str| compiled.find_module("Top").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle;

        let lanes = 5;
        let mut runner = InterpRunner::new(InterpProgram::encode_design(&program).unwrap(), lanes);
        let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        runner.set_signal(off(clk), &vec![1u32; lanes]);
        cpu.set_signal(clk, &vec![1u128; lanes]).unwrap();
        for cycle in 0..8u32 {
            for (s, base) in [(a, 1u32), (b, 2), (c, 3)] {
                let v: Vec<u32> = (0..lanes as u32).map(|l| base.wrapping_mul(2654435761).wrapping_add(l * 5 + cycle)).collect();
                runner.set_signal(off(s), &v);
                cpu.set_signal(s, &v.iter().map(|&x| x as u128).collect::<Vec<_>>()).unwrap();
            }
            runner.tick();
            cpu.tick();
            for name in ["aoi", "oai", "nand", "plainor"] {
                let h = handle(&format!("{name}_o"));
                let r = runner.get_signal(off(h));
                let cc: Vec<u32> = cpu.get_signal(h).unwrap().iter().map(|&x| x as u32).collect();
                assert_eq!(r, cc, "{name} mismatch at cycle {cycle}");
            }
        }
    }

    /// Four-operand standard cells AOI22 = ~((a&b)|(c&d)) and OAI22 = ~((a|b)&(c|d)),
    /// whose 3rd/4th operands live in aux, must match SimdCpuSimulator (incl. a wide
    /// 64-bit instance), and must not disturb adjacent 2- and 3-operand cells.
    #[test]
    fn aoi22_oai22_match_simd_cpu() {
        let mut design = Design::new();
        let (clk, a, b, c, d, a64, b64, c64, d64);
        {
            let mut m = design.module("Top");
            clk = m.input("clk", uint(1));
            a = m.input("a", uint(32));
            b = m.input("b", uint(32));
            c = m.input("c", uint(32));
            d = m.input("d", uint(32));
            a64 = m.input("a64", uint(64));
            b64 = m.input("b64", uint(64));
            c64 = m.input("c64", uint(64));
            d64 = m.input("d64", uint(64));
            let mut r32 = |name: &str, e: rrtl_core::Expr| {
                let r = m.reg(name, uint(32));
                m.clock(r, clk);
                m.next(r, e);
                let o = m.output(format!("{name}_o"), uint(32));
                m.assign(o, r);
            };
            r32("aoi22", !((a & b) | (c & d)));
            r32("oai22", !((a | b) & (c | d)));
            r32("aoi21", !((a & b) | c)); // 3-op neighbour
            r32("nand", !(a & b)); // 2-op neighbour
            let mut r64 = |name: &str, e: rrtl_core::Expr| {
                let r = m.reg(name, uint(64));
                m.clock(r, clk);
                m.next(r, e);
                let o = m.output(format!("{name}_o"), uint(64));
                m.assign(o, r);
            };
            r64("waoi22", !((a64 & b64) | (c64 & d64)));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();
        let off = |s: rrtl_ir::Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;
        let h = |name: &str| compiled.find_module("Top").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle;
        let lanes = 5;
        let mut runner = InterpRunner::new(InterpProgram::encode_design(&program).unwrap(), lanes);
        let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        runner.set_signal(off(clk), &vec![1u32; lanes]);
        cpu.set_signal(clk, &vec![1u128; lanes]).unwrap();
        for cycle in 0..8u32 {
            for (s, base) in [(a, 1u32), (b, 2), (c, 3), (d, 4)] {
                let v: Vec<u32> = (0..lanes as u32).map(|l| base.wrapping_mul(2654435761).wrapping_add(l * 5 + cycle)).collect();
                runner.set_signal(off(s), &v);
                cpu.set_signal(s, &v.iter().map(|&x| x as u128).collect::<Vec<_>>()).unwrap();
            }
            for (s, base) in [(a64, 7u128), (b64, 11), (c64, 13), (d64, 17)] {
                let v: Vec<u128> = (0..lanes as u128).map(|l| l.wrapping_mul(base.wrapping_mul(0x1_0000_0007)).wrapping_add(cycle as u128)).collect();
                runner.set_signal_wide(off(s), 2, &v);
                cpu.set_signal(s, &v).unwrap();
            }
            runner.tick();
            cpu.tick();
            for name in ["aoi22", "oai22", "aoi21", "nand"] {
                let hh = h(&format!("{name}_o"));
                let r = runner.get_signal(off(hh));
                let cc: Vec<u32> = cpu.get_signal(hh).unwrap().iter().map(|&x| x as u32).collect();
                assert_eq!(r, cc, "{name} mismatch at cycle {cycle}");
            }
            let hw = h("waoi22_o");
            assert_eq!(runner.get_signal_wide(off(hw), 2), cpu.get_signal(hw).unwrap(), "waoi22 mismatch at cycle {cycle}");
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
