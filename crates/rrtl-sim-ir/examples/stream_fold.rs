//! Stream-folding for GF(2)-linear blocks: process a length-N input stream
//! through a linear block by SEGMENTS computed independently, then stitch with
//! the M^N leap operator (the generalized zlib `crc32_combine`). Processing a
//! segment of length L from state x gives `M^L·x ⊕ V`, where `V` is the same
//! segment processed from ZERO state — so the V_i are independent (P-way
//! parallel) and combine as `total = M^N·s0 ⊕ Σ_i M^{suffix_i}·V_i`. This turns
//! one sequential linear stream into P-way parallelism, bit-exact. (The win is
//! the parallelism + the homogeneous M^L stitch; validatable by correctness, so
//! throttle-robust.)
//! Build: cargo run --release -p rrtl-sim-ir --example stream_fold
use std::collections::HashMap;

use rrtl_sim_ir::{linearize, lower_to_packed_program, PackedOp};
use rrtl_sv_frontend::import_sv;

fn matmul(a: &[u128], b: &[u128]) -> Vec<u128> {
    a.iter()
        .map(|&row| {
            let (mut acc, mut bits) = (0u128, row);
            while bits != 0 {
                acc ^= b[bits.trailing_zeros() as usize];
                bits &= bits - 1;
            }
            acc
        })
        .collect()
}
fn identity(n: usize) -> Vec<u128> {
    (0..n).map(|i| 1u128 << i).collect()
}
fn matpow(m: &[u128], mut e: u64) -> Vec<u128> {
    let mut r = identity(m.len());
    let mut b = m.to_vec();
    while e > 0 {
        if e & 1 == 1 {
            r = matmul(&r, &b);
        }
        e >>= 1;
        if e > 0 {
            b = matmul(&b, &b);
        }
    }
    r
}
fn apply(m: &[u128], v: u128) -> u128 {
    let mut out = 0u128;
    for (i, &row) in m.iter().enumerate() {
        if (row & v).count_ones() & 1 == 1 {
            out |= 1u128 << i;
        }
    }
    out
}

fn main() {
    let src = std::fs::read_to_string("bench/sv/crc32.sv").expect("read crc32.sv");
    let imported = import_sv(&src, Some("crc32")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "crc32").expect("lower");
    let comb = linearize::comb_defs(&program);
    let din = program.signals.iter().position(|s| s.name.ends_with(".din")).expect("din");

    struct Reg<'a> {
        dst: usize,
        width: u32,
        next: &'a rrtl_sim_ir::PackedExpr,
        form: linearize::LinearForm,
    }
    let mut regs: Vec<Reg> = Vec::new();
    for packet in &program.streams.tick_next {
        for op in &packet.ops {
            if let PackedOp::CaptureReg { dst, next, .. } = op {
                if linearize::is_linear(&program, next) {
                    regs.push(Reg { dst: *dst, width: next.ty.width, next, form: linearize::extract_linear_form(&program, next) });
                }
            }
        }
    }
    let mut base = HashMap::new();
    let mut sb = 0usize;
    for r in &regs {
        base.insert(r.dst, sb);
        sb += r.width as usize;
    }

    // Homogeneous state-transition matrix M (din=0, constant excluded): row i =
    // the state bits whose column toggles output bit i. The per-cycle constant
    // and input contribution live in each segment's V (stepped from zero), so the
    // stitch uses only the homogeneous M.
    let mut m = vec![0u128; sb];
    for r in &regs {
        let mut leaf_col = HashMap::new();
        let mut col = 0usize;
        for &(sig, w) in &r.form.leaves {
            leaf_col.insert(sig, col);
            col += w as usize;
        }
        for i in 0..r.width as usize {
            let mut row = 0u128;
            for r2 in &regs {
                if let Some(&lc) = leaf_col.get(&r2.dst) {
                    for j in 0..r2.width as usize {
                        if (r.form.columns[lc + j] >> i) & 1 == 1 {
                            row |= 1u128 << (base[&r2.dst] + j);
                        }
                    }
                }
            }
            m[base[&r.dst] + i] = row;
        }
    }

    let pack = |st: &HashMap<usize, u128>| -> u128 {
        let mut v = 0u128;
        for r in &regs {
            v |= (st[&r.dst] & ((1u128 << r.width) - 1)) << base[&r.dst];
        }
        v
    };
    let zero_state = || regs.iter().map(|r| (r.dst, 0u128)).collect::<HashMap<_, _>>();
    // step the block one cycle with a given din
    let step = |st: &HashMap<usize, u128>, d: u128| -> HashMap<usize, u128> {
        let mut leaves = st.clone();
        leaves.insert(din, d);
        for r in &regs {
            for &(sig, _) in &r.form.leaves {
                leaves.entry(sig).or_insert(0);
            }
        }
        regs.iter()
            .map(|r| (r.dst, linearize::eval_expr(r.next, &comb, &leaves) & ((1u128 << r.width) - 1)))
            .collect()
    };
    let run = |s0: &HashMap<usize, u128>, stream: &[u128]| -> HashMap<usize, u128> {
        let mut st = s0.clone();
        for &d in stream {
            st = step(&st, d);
        }
        st
    };

    let crc_reg = regs.iter().find(|r| program.signals[r.dst].name.contains("crc")).unwrap().dst;
    let crc_of = |v: u128| (v >> base[&crc_reg]) & 0xFFFF_FFFF;

    // A buffer (varying input) and a non-trivial initial state.
    let n = 4096usize;
    let stream: Vec<u128> = (0..n).map(|k| ((k as u128).wrapping_mul(2654435761) >> 13) & 0xff).collect();
    let mut s0 = zero_state();
    s0.insert(crc_reg, 0xFFFF_FFFF);
    let seq = pack(&run(&s0, &stream));

    println!("crc32 stream-fold: N={n} byte stream, state {sb} bits, sequential crc=0x{:08x}", crc_of(seq));
    for &p in &[2usize, 4, 8, 16] {
        assert!(n % p == 0);
        let l = n / p;
        let ml = matpow(&m, l as u64); // M^L (segment length), reused for every segment
        let mn = matpow(&m, n as u64);
        // V_i = segment i processed from ZERO state (independent → P-way parallel)
        let vs: Vec<u128> = (0..p).map(|i| pack(&run(&zero_state(), &stream[i * l..(i + 1) * l]))).collect();
        // stitch: acc = Σ_i M^{(P-1-i)L} V_i  (Horner with M^L); total = M^N s0 ⊕ acc
        let mut acc = 0u128;
        for &v in &vs {
            acc = apply(&ml, acc) ^ v;
        }
        let total = apply(&mn, pack(&s0)) ^ acc;
        let ok = crc_of(total) == crc_of(seq);
        println!("  P={p:>2} segments (len {l}, computed from zero independently) -> crc=0x{:08x}  {}",
            crc_of(total), if ok { "== sequential [OK, P-way parallelizable]" } else { "!= sequential [MISMATCH]" });
        assert!(ok, "stream-fold diverged at P={p}");
    }
    println!("  result: BIT-EXACT — one linear stream folds into P independent segments + an M^N stitch");
}
