//! Range (known-zero-high-bit) analysis over the packed machine IR, the basis for
//! limb reduction: a value declared wide (e.g. 64-bit) but provably narrow (its
//! high limbs are always zero) can be computed and stored in fewer limbs, cutting
//! the multi-limb tax.
//!
//! `maxbits(v)` is a conservative upper bound on the number of low bits of `v`
//! that can ever be nonzero; bits `[maxbits, width)` are provably zero. The
//! effective limb count is then `ceil(maxbits/32)`. This module computes the
//! analysis and an opportunity estimate; the actual kernel-level narrowing is a
//! separate (and more invasive) step, gated on this estimate being worthwhile.

use std::collections::HashMap;

use rrtl_ir::BitType;

use crate::{PackedBlock, PackedInstrKind, PackedMachineProgram};

/// Upper bound on nonzero low bits of a value.
fn lit_maxbits(words: &[u32]) -> u32 {
    for (i, &w) in words.iter().enumerate().rev() {
        if w != 0 {
            return i as u32 * 32 + (32 - w.leading_zeros());
        }
    }
    0
}

fn limbs_of_bits(bits: u32) -> usize {
    (((bits + 31) / 32) as usize).max(1)
}

/// Per-value `maxbits` for one block (value ids are block-local, defs before uses).
pub fn block_maxbits(block: &PackedBlock) -> Vec<u32> {
    let n = block
        .packets
        .iter()
        .flat_map(|p| p.instrs.iter())
        .map(|i| i.dst.0 + 1)
        .max()
        .unwrap_or(0);
    let mut mb = vec![0u32; n];
    let mut ty: HashMap<usize, BitType> = HashMap::new();
    let get = |mb: &[u32], id: crate::PackedValueId| mb[id.0];
    for instr in block.packets.iter().flat_map(|p| p.instrs.iter()) {
        ty.insert(instr.dst.0, instr.ty);
        let w = instr.ty.width;
        let bits = match &instr.kind {
            PackedInstrKind::Lit(words) => lit_maxbits(words).min(w),
            // Inputs/registers/memory can hold any value of their width.
            PackedInstrKind::Signal(_) | PackedInstrKind::MemRead { .. } => w,
            // Bitwise: AND zeroes a bit if either side is zero; OR/XOR keep the max.
            PackedInstrKind::And(a, b) => get(&mb, *a).min(get(&mb, *b)),
            PackedInstrKind::Or(a, b) | PackedInstrKind::Xor(a, b) => {
                get(&mb, *a).max(get(&mb, *b))
            }
            // NOT sets the high bits within the width.
            PackedInstrKind::Not(_) => w,
            // Add can carry one bit; Sub can borrow into the top; Mul adds widths.
            PackedInstrKind::Add(a, b) => (get(&mb, *a).max(get(&mb, *b)) + 1).min(w),
            PackedInstrKind::Sub(_, _) => w,
            PackedInstrKind::Mul(a, b) => (get(&mb, *a) + get(&mb, *b)).min(w),
            // Comparisons are one bit.
            PackedInstrKind::Eq(_, _) | PackedInstrKind::Ne(_, _) | PackedInstrKind::Lt { .. } => 1,
            PackedInstrKind::Mux {
                then_value,
                else_value,
                ..
            } => get(&mb, *then_value).max(get(&mb, *else_value)),
            PackedInstrKind::Slice { value, lsb } => {
                get(&mb, *value).saturating_sub(*lsb).min(w)
            }
            // Zero-extend keeps the high bits zero — the main narrowing source.
            PackedInstrKind::Zext(a) | PackedInstrKind::Cast(a) | PackedInstrKind::Trunc(a) => {
                get(&mb, *a).min(w)
            }
            // Sign-extend is narrow only if the source's sign bit is provably zero.
            PackedInstrKind::Sext(a) => {
                let src_w = ty.get(&a.0).map(|t| t.width).unwrap_or(w);
                if get(&mb, *a) < src_w {
                    get(&mb, *a).min(w)
                } else {
                    w
                }
            }
            // Concat: conservative (full width).
            PackedInstrKind::Concat(_) => w,
        };
        mb[instr.dst.0] = bits.min(w);
    }
    mb
}

/// Aggregate opportunity estimate for limb reduction across a program.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RangeStats {
    pub total_values: usize,
    pub narrowable_values: usize,
    /// Sum of declared limb counts over all produced values.
    pub declared_words: usize,
    /// Sum of provable effective limb counts over all produced values.
    pub effective_words: usize,
}

impl RangeStats {
    pub fn words_saved(&self) -> usize {
        self.declared_words.saturating_sub(self.effective_words)
    }
}

pub fn range_reduction_stats(machine: &PackedMachineProgram) -> RangeStats {
    let mut s = RangeStats::default();
    for block in [
        &machine.streams.async_reset_comb,
        &machine.streams.comb,
        &machine.streams.tick_next,
        &machine.streams.tick_commit,
    ] {
        let mb = block_maxbits(block);
        for instr in block.packets.iter().flat_map(|p| p.instrs.iter()) {
            let declared = limbs_of_bits(instr.ty.width);
            let effective = limbs_of_bits(mb[instr.dst.0]);
            s.total_values += 1;
            s.declared_words += declared;
            s.effective_words += effective;
            if effective < declared {
                s.narrowable_values += 1;
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lower_to_machine_program, lower_to_packed_program};
    use rrtl_core::{compile, lit_u, uint, Design};

    #[test]
    fn narrows_zero_extended_values() {
        let mut design = Design::new();
        {
            let mut m = design.module("M");
            let clk = m.input("clk", uint(1));
            let din = m.input("din", uint(8)); // narrow input
            // acc is 64-bit but only ever holds (8-bit & 8-bit) sums -> provably ~<=9 bits
            let acc = m.reg("acc", uint(64));
            m.clock(acc, clk);
            // zero-extend an 8-bit AND into 64-bit
            let masked = m.wire("masked", uint(8));
            m.assign(masked, din & lit_u(0x0f, 8));
            let wide = m.wire("wide", uint(64));
            m.assign(wide, masked.value().zext(64));
            m.next(acc, wide);
            let o = m.output("o", uint(64));
            m.assign(o, acc);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "M").unwrap();
        let machine = lower_to_machine_program(&program);
        let stats = range_reduction_stats(&machine);
        assert!(
            stats.narrowable_values > 0 && stats.words_saved() > 0,
            "expected narrowing from the zext path: {stats:?}"
        );
    }

    #[test]
    fn genuine_wide_not_narrowed() {
        let mut design = Design::new();
        {
            let mut m = design.module("M");
            let clk = m.input("clk", uint(1));
            let din = m.input("din", uint(64)); // genuinely 64-bit
            let acc = m.reg("acc", uint(64));
            m.clock(acc, clk);
            m.next(acc, acc * lit_u(0x9e37_79b9_7f4a_7c15, 64) + din);
            let o = m.output("o", uint(64));
            m.assign(o, acc);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "M").unwrap();
        let machine = lower_to_machine_program(&program);
        let stats = range_reduction_stats(&machine);
        // The 64-bit register and mul/add chain must stay 2-limb.
        assert_eq!(stats.words_saved(), 0, "genuine 64-bit must not narrow: {stats:?}");
    }
}
