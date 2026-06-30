//! Temporal leap for GF(2)-linear register cones — the closed-form cycle jump the
//! linear extraction ([`crate::linearize`]) unlocks. A linear block evolves
//! `s(t+1) = M·s(t) ⊕ c`, so:
//!   - IDLE / constant input: `s(t+N) = M^N·s(t) ⊕ (Σ M^k)·c` in O(log N) matrix
//!     mults via repeated squaring (`leap_idle`).
//!   - STREAMING: a length-N input stream splits into P segments processed from
//!     zero state (independent → P-way parallel) and stitched with `M^N`
//!     (`fold`, the generalized zlib `crc32_combine`).
//! Both are bit-exact and algorithmic (op-count, not constant factor).

use std::collections::HashMap;

use crate::{linearize, PackedOp, PackedProgram};

/// A GF(2) matrix in row form: row `i` is a bitmask of the input bits XOR'd into
/// output bit `i`. Supports up to 128 bits. `(A·B)[i] = ⊕_{k∈A[i]} B[k]`.
#[derive(Clone, Debug)]
pub struct Gf2Mat {
    pub rows: Vec<u128>,
    pub n: usize,
}

impl Gf2Mat {
    pub fn identity(n: usize) -> Self {
        Gf2Mat { rows: (0..n).map(|i| 1u128 << i).collect(), n }
    }

    /// `self · other` (both n×n).
    pub fn matmul(&self, other: &Gf2Mat) -> Gf2Mat {
        let rows = self
            .rows
            .iter()
            .map(|&row| {
                let (mut acc, mut bits) = (0u128, row);
                while bits != 0 {
                    acc ^= other.rows[bits.trailing_zeros() as usize];
                    bits &= bits - 1;
                }
                acc
            })
            .collect();
        Gf2Mat { rows, n: self.n }
    }

    /// `self^e` by repeated squaring (O(log e) matmuls).
    pub fn pow(&self, mut e: u64) -> Gf2Mat {
        let mut r = Gf2Mat::identity(self.n);
        let mut b = self.clone();
        while e > 0 {
            if e & 1 == 1 {
                r = r.matmul(&b);
            }
            e >>= 1;
            if e > 0 {
                b = b.matmul(&b);
            }
        }
        r
    }

    /// Matrix-vector product: `out[i] = parity(row_i & v)`.
    pub fn apply(&self, v: u128) -> u128 {
        let mut out = 0u128;
        for (i, &row) in self.rows.iter().enumerate() {
            if (row & v).count_ones() & 1 == 1 {
                out |= 1u128 << i;
            }
        }
        out
    }
}

/// The combined GF(2) state-transition operator of a design's linear register
/// cones, with the per-cycle constant — the substrate for temporal leaps.
#[derive(Clone, Debug)]
pub struct LinearLeap {
    /// (register signal index, width), in combined-state-layout order.
    regs: Vec<(usize, u32)>,
    base: HashMap<usize, usize>,
    state_bits: usize,
    m: Gf2Mat, // homogeneous transition (din=0, constant excluded)
    c: u128,   // per-idle-cycle constant (state bits)
}

impl LinearLeap {
    /// Build from a program's GF(2)-linear register cones. `None` if there are no
    /// linear register cones, or the combined state exceeds 128 bits.
    pub fn build(program: &PackedProgram) -> Option<LinearLeap> {
        let mut regs: Vec<(usize, u32, linearize::LinearForm)> = Vec::new();
        for packet in &program.streams.tick_next {
            for op in &packet.ops {
                if let PackedOp::CaptureReg { dst, next, .. } = op {
                    if linearize::is_linear(program, next) {
                        regs.push((*dst, next.ty.width, linearize::extract_linear_form(program, next)));
                    }
                }
            }
        }
        if regs.is_empty() {
            return None;
        }
        let mut base = HashMap::new();
        let mut state_bits = 0usize;
        for (dst, w, _) in &regs {
            base.insert(*dst, state_bits);
            state_bits += *w as usize;
        }
        if state_bits > 128 {
            return None;
        }
        let mut rows = vec![0u128; state_bits];
        let mut c = 0u128;
        for (dst, w, form) in &regs {
            let mut leaf_col = HashMap::new();
            let mut col = 0usize;
            for &(sig, lw) in &form.leaves {
                leaf_col.insert(sig, col);
                col += lw as usize;
            }
            for i in 0..*w as usize {
                let mut row = 0u128;
                for (d2, w2, _) in &regs {
                    if let Some(&lc) = leaf_col.get(d2) {
                        for j in 0..*w2 as usize {
                            if (form.columns[lc + j] >> i) & 1 == 1 {
                                row |= 1u128 << (base[d2] + j);
                            }
                        }
                    }
                }
                rows[base[dst] + i] = row;
            }
            let mask = if *w >= 128 { u128::MAX } else { (1u128 << w) - 1 };
            c |= (form.constant & mask) << base[dst];
        }
        Some(LinearLeap {
            regs: regs.iter().map(|(d, w, _)| (*d, *w)).collect(),
            base,
            state_bits,
            m: Gf2Mat { rows, n: state_bits },
            c,
        })
    }

    pub fn state_bits(&self) -> usize {
        self.state_bits
    }
    pub fn registers(&self) -> &[(usize, u32)] {
        &self.regs
    }
    /// Bit offset of register `sig` in the combined state word.
    pub fn bit_base(&self, sig: usize) -> Option<usize> {
        self.base.get(&sig).copied()
    }
    pub fn transition(&self) -> &Gf2Mat {
        &self.m
    }

    /// State after `n` idle cycles (inputs held at zero / the cone's constant):
    /// `M^N·s0 ⊕ (Σ_{k<N} M^k)·c`, via an augmented matrix exponentiated by
    /// squaring — O(log N). Requires `state_bits ≤ 127` (one augmented bit).
    pub fn leap_idle(&self, s0: u128, n: u64) -> u128 {
        if self.state_bits >= 128 {
            // can't augment; fall back to homogeneous (valid only when c == 0)
            debug_assert_eq!(self.c, 0);
            return self.m.pow(n).apply(s0);
        }
        let aug = self.state_bits;
        let mut rows = self.m.rows.clone();
        for (i, r) in rows.iter_mut().enumerate() {
            if (self.c >> i) & 1 == 1 {
                *r |= 1u128 << aug;
            }
        }
        rows.push(1u128 << aug);
        let a = Gf2Mat { rows, n: aug + 1 };
        let v = a.pow(n).apply(s0 | (1u128 << aug));
        v & ((1u128 << aug) - 1)
    }

    /// Combine `P` equal-length segment results (each the segment processed from
    /// ZERO state) into the full stream result, starting from `s0`:
    /// `total = M^{P·L}·s0 ⊕ Σ_i M^{(P-1-i)·L}·V_i`. The `seg_results` are
    /// independent (P-way parallelizable). `seg_len` = L cycles per segment.
    pub fn fold(&self, s0: u128, seg_len: u64, seg_results: &[u128]) -> u128 {
        let p = seg_results.len() as u64;
        let ml = self.m.pow(seg_len);
        let mn = self.m.pow(seg_len * p);
        let mut acc = 0u128;
        for &v in seg_results {
            acc = ml.apply(acc) ^ v;
        }
        mn.apply(s0) ^ acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower_to_packed_program;
    use rrtl_sv_frontend::import_sv;

    // Reference stepper over crc32's linear cones (din feed), and the leap/fold
    // built by LinearLeap, must agree bit-exactly.
    #[test]
    fn leap_and_fold_match_stepping_crc32() {
        let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv/crc32.sv")).unwrap();
        let imported = import_sv(&src, Some("crc32")).unwrap();
        let compiled = rrtl_core::compile(&imported.design).unwrap();
        let program = lower_to_packed_program(&compiled, "crc32").unwrap();
        let comb = linearize::comb_defs(&program);
        let din = program.signals.iter().position(|s| s.name.ends_with(".din")).unwrap();

        // collect reg next exprs for the reference stepper
        let mut nexts: Vec<(usize, u32, &crate::PackedExpr)> = Vec::new();
        for packet in &program.streams.tick_next {
            for op in &packet.ops {
                if let PackedOp::CaptureReg { dst, next, .. } = op {
                    if linearize::is_linear(&program, next) {
                        nexts.push((*dst, next.ty.width, next));
                    }
                }
            }
        }
        let leap = LinearLeap::build(&program).unwrap();
        let pack = |st: &HashMap<usize, u128>| -> u128 {
            let mut v = 0u128;
            for &(d, w) in leap.registers() {
                v |= (st[&d] & ((1u128 << w) - 1)) << leap.bit_base(d).unwrap();
            }
            v
        };
        let step = |st: &HashMap<usize, u128>, d: u128| -> HashMap<usize, u128> {
            let mut leaves = st.clone();
            leaves.insert(din, d);
            nexts.iter().map(|(dst, w, n)| (*dst, linearize::eval_expr(n, &comb, &leaves) & ((1u128 << w) - 1))).collect()
        };
        let zero = || leap.registers().iter().map(|&(d, _)| (d, 0u128)).collect::<HashMap<_, _>>();

        // idle leap (din=0)
        let mut s0 = zero();
        for &(d, _) in leap.registers() {
            if program.signals[d].name.contains("crc") {
                s0.insert(d, 0xFFFF_FFFF);
            }
        }
        for &n in &[1u64, 5, 100, 9999] {
            let mut st = s0.clone();
            for _ in 0..n {
                st = step(&st, 0);
            }
            assert_eq!(leap.leap_idle(pack(&s0), n), pack(&st) & ((1u128 << leap.state_bits()) - 1), "idle leap N={n}");
        }

        // streaming fold (varying din)
        let stream: Vec<u128> = (0..1024).map(|k| (k as u128 * 7 + 3) & 0xff).collect();
        let mut st = s0.clone();
        for &d in &stream {
            st = step(&st, d);
        }
        let seq = pack(&st);
        for &p in &[2usize, 4, 8] {
            let l = stream.len() / p;
            let vs: Vec<u128> = (0..p)
                .map(|i| {
                    let mut s = zero();
                    for &d in &stream[i * l..(i + 1) * l] {
                        s = step(&s, d);
                    }
                    pack(&s)
                })
                .collect();
            assert_eq!(leap.fold(pack(&s0), l as u64, &vs), seq, "fold P={p}");
        }
    }
}
