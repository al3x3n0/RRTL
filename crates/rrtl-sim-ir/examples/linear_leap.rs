//! Temporal leap for GF(2)-linear blocks: a linear register cone evolves as
//! `s(t+1) = M·s(t) ⊕ c` over GF(2) (inputs held constant — the idle / free-
//! running regime: LFSRs, scramblers, counters, a CRC between feeds). So
//! `s(t+N) = M^N·s(t) ⊕ (accumulated const)`, and `M^N` by repeated squaring is
//! **O(log N) matrix mults** instead of N cycle-steps — an algorithmic leap, not
//! a constant factor. We assemble M over the design's *combined* linear-register
//! state (crc32 is two coupled regs: `crc` and the blocking temp `c`), augment it
//! with the constant, exponentiate over GF(2), and check bit-exact vs stepping.
//! Build: cargo run --release -p rrtl-sim-ir --example linear_leap
use std::collections::HashMap;

use rrtl_sim_ir::{linearize, lower_to_packed_program, PackedOp};
use rrtl_sv_frontend::import_sv;

// GF(2) matrices in row form: row i is a bitmask of the input bits XOR'd into
// output bit i. (A·B)[i] = XOR over set bits k of A[i] of B[k].
fn matmul(a: &[u128], b: &[u128]) -> Vec<u128> {
    a.iter()
        .map(|&row| {
            let mut acc = 0u128;
            let mut bits = row;
            while bits != 0 {
                let k = bits.trailing_zeros() as usize;
                acc ^= b[k];
                bits &= bits - 1;
            }
            acc
        })
        .collect()
}

fn identity(n: usize) -> Vec<u128> {
    (0..n).map(|i| 1u128 << i).collect()
}

// M^e by repeated squaring (returns the number of matmuls used).
fn matpow(m: &[u128], mut e: u64) -> (Vec<u128>, u32) {
    let n = m.len();
    let mut result = identity(n);
    let mut base = m.to_vec();
    let mut mults = 0u32;
    while e > 0 {
        if e & 1 == 1 {
            result = matmul(&result, &base);
            mults += 1;
        }
        e >>= 1;
        if e > 0 {
            base = matmul(&base, &base);
            mults += 1;
        }
    }
    (result, mults)
}

// Apply a row-form matrix to a state vector: out bit i = parity(row_i & v).
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

    // Collect the linear register cones + GF(2) forms.
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
    // Combined state layout: concatenate register bits.
    let mut base = HashMap::new();
    let mut s_bits = 0usize;
    for r in &regs {
        base.insert(r.dst, s_bits);
        s_bits += r.width as usize;
    }
    let aug = s_bits + 1; // augmented bit carries the constant
    assert!(aug <= 128, "state {s_bits} bits > 127; widen the matrix repr");
    println!("crc32: {} linear regs, combined state {s_bits} bits ({})",
        regs.len(), regs.iter().map(|r| format!("{}={}b", program.signals[r.dst].name.rsplit('.').next().unwrap(), r.width)).collect::<Vec<_>>().join(", "));

    // Build M (augmented, row form) for the IDLE recurrence (din = 0). For each
    // register r, output bit i: M_row[base_r+i] = the state bits j whose column
    // toggles output bit i; plus the constant in the augmented column.
    let mut m = vec![0u128; aug];
    for r in &regs {
        // column base of each leaf in r.form
        let mut leaf_col = HashMap::new();
        let mut col = 0usize;
        for &(sig, w) in &r.form.leaves {
            leaf_col.insert(sig, col);
            col += w as usize;
        }
        for i in 0..r.width as usize {
            let mut row = 0u128;
            // contribution of each combined-state bit j (only state regs are leaves)
            for r2 in &regs {
                if let Some(&lc) = leaf_col.get(&r2.dst) {
                    for j in 0..r2.width as usize {
                        if (r.form.columns[lc + j] >> i) & 1 == 1 {
                            row |= 1u128 << (base[&r2.dst] + j);
                        }
                    }
                }
            }
            if (r.form.constant >> i) & 1 == 1 {
                row |= 1u128 << s_bits; // augmented constant
            }
            m[base[&r.dst] + i] = row;
        }
    }
    m[s_bits] = 1u128 << s_bits; // constant row holds itself

    // Reference: step all register forms N times from s0 with din=0.
    let pack = |state: &HashMap<usize, u128>| -> u128 {
        let mut v = 0u128;
        for r in &regs {
            v |= (state[&r.dst] & ((1u128 << r.width) - 1)) << base[&r.dst];
        }
        v | (1u128 << s_bits)
    };
    let step_n = |s0: &HashMap<usize, u128>, n: u64| -> HashMap<usize, u128> {
        let mut st = s0.clone();
        for _ in 0..n {
            let mut leaves: HashMap<usize, u128> = st.clone();
            // din (and any non-reg leaf) held 0
            for r in &regs {
                for &(sig, _) in &r.form.leaves {
                    leaves.entry(sig).or_insert(0);
                }
            }
            let mut next = HashMap::new();
            for r in &regs {
                next.insert(r.dst, linearize::eval_expr(r.next, &comb, &leaves) & ((1u128 << r.width) - 1));
            }
            st = next;
        }
        st
    };

    // s0 = post-reset-ish state.
    let mut s0 = HashMap::new();
    for r in &regs {
        s0.insert(r.dst, if program.signals[r.dst].name.contains("crc") { 0xFFFF_FFFFu128 } else { 0 });
    }
    let v0 = pack(&s0);

    println!("  validating M^N leap vs N step-by-step (din=0, idle):");
    let mut all_ok = true;
    for &n in &[1u64, 2, 7, 64, 1000, 100_000] {
        let (mn, mults) = matpow(&m, n);
        let leap = apply(&mn, v0);
        let stepped = pack(&step_n(&s0, n));
        let ok = (leap & ((1u128 << s_bits) - 1)) == (stepped & ((1u128 << s_bits) - 1));
        all_ok &= ok;
        let crc_leap = (leap >> base[&regs.iter().find(|r| program.signals[r.dst].name.contains("crc")).unwrap().dst]) & 0xFFFF_FFFF;
        println!("    N={n:>7}: leap crc=0x{crc_leap:08x}  {}  ({mults} matmuls vs {n} steps)", if ok { "== stepped [OK]" } else { "!= stepped [MISMATCH]" });
    }
    // The headline: a leap no stepping loop would ever finish.
    let huge = 1u64 << 40;
    let (mn, mults) = matpow(&m, huge);
    let crc_huge = (apply(&mn, v0) >> base[&regs.iter().find(|r| program.signals[r.dst].name.contains("crc")).unwrap().dst]) & 0xFFFF_FFFF;
    println!("    N=2^40 ({huge}): leap crc=0x{crc_huge:08x} in {mults} matmuls (stepping would take ~{huge} cycles)");
    println!("  result: {}", if all_ok { "BIT-EXACT — O(log N) temporal leap validated" } else { "MISMATCH" });
    assert!(all_ok, "leap diverged from stepping");
}
