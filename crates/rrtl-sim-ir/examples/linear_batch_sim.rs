//! End-to-end proof of the tensor-core (GF(2)-linear) specialization: detect the
//! linear register cones of a design, extract their GF(2) matrices, and run a
//! BATCHED bit-sliced simulation (K lanes at once) entirely via the matrices —
//! then check it bit-exact against the gate-level reference and measure
//! throughput. On a CPU the bit-sliced form is `nnz(M)` word-XORs/cycle; the same
//! matrices feed a 1-bit tensor-core (BMMA) dense GEMM in the dense-batch layout.
//! Build: cargo run --release -p rrtl-sim-ir --example linear_batch_sim
use std::collections::HashMap;

use rrtl_sim_ir::{linearize, lower_to_packed_program, PackedOp};
use rrtl_sv_frontend::import_sv;

const LANES: usize = 64; // one u64 word of lanes per bit

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv");
    let src = std::fs::read_to_string(format!("{base}/crc32.sv")).expect("read crc32.sv");
    let imported = import_sv(&src, Some("crc32")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "crc32").expect("lower packed");
    let comb_def = linearize::comb_defs(&program);

    // Collect linear register cones with their GF(2) forms + reset values.
    struct Reg<'a> {
        dst: usize,
        next: &'a rrtl_sim_ir::PackedExpr,
        form: linearize::LinearForm,
        reset: u128,
    }
    let mut regs: Vec<Reg> = Vec::new();
    for packet in &program.streams.tick_next {
        for op in &packet.ops {
            let PackedOp::CaptureReg { dst, next, reset } = op else { continue };
            if next.ty.width > 64 || !linearize::is_linear(&program, next) {
                continue;
            }
            let mut rv = 0u128;
            if let Some(r) = reset {
                for (i, w) in r.value.iter().enumerate().take(4) {
                    rv |= (*w as u128) << (32 * i);
                }
            }
            regs.push(Reg { dst: *dst, next, form: linearize::extract_linear_form(&program, next), reset: rv });
        }
    }
    let state_sigs: std::collections::HashSet<usize> = regs.iter().map(|r| r.dst).collect();
    // input leaves = any form leaf that isn't a (linear) register
    let mut input_sigs: Vec<(usize, u32)> = Vec::new();
    for r in &regs {
        for &(s, w) in &r.form.leaves {
            if !state_sigs.contains(&s) && !input_sigs.iter().any(|(x, _)| *x == s) {
                input_sigs.push((s, w));
            }
        }
    }
    println!(
        "crc32: {} linear register cones, inputs {:?}",
        regs.len(),
        input_sigs.iter().map(|(s, w)| (program.signals[*s].name.clone(), *w)).collect::<Vec<_>>()
    );

    // Bit-sliced state: signal -> Vec<u64> (one word/bit, LANES lanes).
    let mut words: HashMap<usize, Vec<u64>> = HashMap::new();
    for r in &regs {
        let w = r.next.ty.width;
        words.insert(r.dst, (0..w).map(|b| if (r.reset >> b) & 1 != 0 { u64::MAX } else { 0 }).collect());
    }
    for &(s, w) in &input_sigs {
        words.insert(s, vec![0u64; w as usize]);
    }

    // Precompute, per register, the GF(2) rows (which input bits feed each output bit).
    struct Plan {
        dst: usize,
        out_w: u32,
        // (out_bit) -> list of (leaf_sig, leaf_local_bit) that XOR into it
        rows: Vec<Vec<(usize, u32)>>,
        cbits: u128,
    }
    let plans: Vec<Plan> = regs
        .iter()
        .map(|r| {
            // flat leaf-bit -> (sig, local bit)
            let mut leaf_src = Vec::new();
            for &(s, w) in &r.form.leaves {
                for b in 0..w {
                    leaf_src.push((s, b));
                }
            }
            let mut rows = vec![Vec::new(); r.form.out_width as usize];
            for (b, col) in r.form.columns.iter().enumerate() {
                for i in 0..r.form.out_width {
                    if (col >> i) & 1 != 0 {
                        rows[i as usize].push(leaf_src[b]);
                    }
                }
            }
            Plan { dst: r.dst, out_w: r.form.out_width, rows, cbits: r.form.constant }
        })
        .collect();
    let nnz: usize = plans.iter().map(|p| p.rows.iter().map(|r| r.len()).sum::<usize>()).sum();

    // ---- bit-exact check vs the gate reference over the first cycles ----
    let mut lcg: u64 = 0xC0FFEE;
    let mut rng = || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        lcg
    };
    // scalar reference state for 3 sample lanes
    let sample = [0usize, 17, 63];
    let mut ref_state: Vec<HashMap<usize, u128>> = sample
        .iter()
        .map(|_| regs.iter().map(|r| (r.dst, r.reset & ((1u128 << r.next.ty.width) - 1))).collect())
        .collect();

    let mut ok = true;
    for _ in 0..200 {
        // feed random inputs (bit-sliced)
        for &(s, w) in &input_sigs {
            let v = words.get_mut(&s).unwrap();
            for b in 0..w as usize {
                v[b] = rng();
            }
        }
        // batched next-state via matrices (read old, write new)
        let next: Vec<(usize, Vec<u64>)> = plans
            .iter()
            .map(|p| {
                let out: Vec<u64> = (0..p.out_w as usize)
                    .map(|i| {
                        let mut acc = if (p.cbits >> i) & 1 != 0 { u64::MAX } else { 0 };
                        for &(s, b) in &p.rows[i] {
                            acc ^= words[&s][b as usize];
                        }
                        acc
                    })
                    .collect();
                (p.dst, out)
            })
            .collect();
        // gate reference for the sample lanes (same inputs)
        for (li, &lane) in sample.iter().enumerate() {
            let mut leaves: HashMap<usize, u128> = HashMap::new();
            for (s, _) in &input_sigs {
                let v = &words[s];
                let mut x = 0u128;
                for b in 0..v.len() {
                    x |= (((v[b] >> lane) & 1) as u128) << b;
                }
                leaves.insert(*s, x);
            }
            for (s, val) in ref_state[li].iter() {
                leaves.insert(*s, *val);
            }
            let new: Vec<(usize, u128)> =
                regs.iter().map(|r| (r.dst, linearize::eval_expr(r.next, &comb_def, &leaves))).collect();
            for (s, v) in new {
                ref_state[li].insert(s, v);
            }
        }
        for (dst, ws) in next {
            *words.get_mut(&dst).unwrap() = ws;
        }
        // compare sample lanes
        for (li, &lane) in sample.iter().enumerate() {
            for r in &regs {
                let bs: u128 = (0..r.next.ty.width)
                    .map(|b| (((words[&r.dst][b as usize] >> lane) & 1) as u128) << b)
                    .fold(0, |a, x| a | x);
                if bs != ref_state[li][&r.dst] {
                    ok = false;
                }
            }
        }
    }
    println!("  batched-GEMM sim vs gate reference (3 lanes, 200 cycles): [{}]", if ok { "BIT-EXACT" } else { "FAIL" });

    // ---- throughput: flatten state to a single Vec<u64> (no HashMap in the hot
    // loop), then keep ticking the bit-sliced matrices. ----
    let mut flat: Vec<u64> = Vec::new();
    let mut idx: HashMap<(usize, u32), usize> = HashMap::new();
    for (s, ws) in &words {
        for b in 0..ws.len() {
            idx.insert((*s, b as u32), flat.len());
            flat.push(ws[b]);
        }
    }
    // per output bit: (dst flat index, constant-bit, source flat indices)
    struct FRow {
        dst: usize,
        cbit: bool,
        src: Vec<usize>,
    }
    let mut frows: Vec<FRow> = Vec::new();
    for p in &plans {
        for i in 0..p.out_w as usize {
            frows.push(FRow {
                dst: idx[&(p.dst, i as u32)],
                cbit: (p.cbits >> i) & 1 != 0,
                src: p.rows[i].iter().map(|&(s, b)| idx[&(s, b)]).collect(),
            });
        }
    }
    let mut tmp = vec![0u64; frows.len()];

    let n = 2_000_000usize;
    let t = std::time::Instant::now();
    for _ in 0..n {
        for (k, fr) in frows.iter().enumerate() {
            let mut acc = if fr.cbit { u64::MAX } else { 0 };
            for &s in &fr.src {
                acc ^= flat[s];
            }
            tmp[k] = acc;
        }
        for (k, fr) in frows.iter().enumerate() {
            flat[fr.dst] = tmp[k];
        }
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "  bit-sliced batch: {} lanes x {} cyc, {} GF(2) word-XORs/cyc, {:.1} G lane-cyc/s",
        LANES, n, nnz, (LANES * n) as f64 / dt / 1e9
    );
    println!(
        "  (tensor-core projection: dense BMMA does the same M as 1 binary GEMM/cyc — the {} nnz become free; throughput scales with batch K, not nnz)",
        nnz
    );
}
