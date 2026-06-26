//! Size the tensor-core (GF(2)-linear) specialization prize: report how much of
//! each design's per-cycle logic is affine-linear over GF(2) — register
//! next-state cones and combinational signals — i.e. directly expressible as a
//! binary matrix product (1-bit tensor-core / bit-sliced XOR-popcount GEMM).
//! Build: cargo run --release -p rrtl-sim-ir --example linearity_check
use std::collections::HashMap;

use rrtl_sim_ir::{linearize, lower_to_packed_program, PackedOp};
use rrtl_sv_frontend::import_sv;

fn report(path: &str, top: &str) {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => {
            println!("{path}: (not found, skipped)");
            return;
        }
    };
    let imported = import_sv(&src, Some(top)).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, top).expect("lower packed");
    let r = linearize::classify(&program);
    println!("{top} ({path}):");
    println!(
        "  register next-state cones: {}/{} linear ({:.0}% of cones), {}/{} bits ({:.0}% of next-state width)",
        r.linear_reg_cones, r.reg_cones,
        r.linear_reg_cones as f64 / r.reg_cones.max(1) as f64 * 100.0,
        r.linear_reg_cone_bits, r.reg_cone_bits, r.linear_reg_bit_frac() * 100.0,
    );
    println!(
        "  combinational signals    : {}/{} linear, {}/{} op-cost ({:.0}% of comb logic)",
        r.linear_signals, r.comb_signals,
        r.linear_cost, r.comb_cost, r.linear_cost_frac() * 100.0,
    );
}

/// Partition a design's per-cycle register update into GF(2)-linear cones
/// (extractable as binary matrices → tensor-core/GEMM offloadable) vs the
/// non-linear remainder (stays on the lane/JIT path). Validates each extracted
/// matrix bit-exact against the reference gate eval over random stimuli.
fn partition_report(path: &str, top: &str) {
    let Ok(src) = std::fs::read_to_string(path) else { return };
    let imported = import_sv(&src, Some(top)).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, top).expect("lower packed");
    let comb_def = linearize::comb_defs(&program);

    println!("{top}: register-update partition (linear cones → GEMM, else lane/JIT)");
    let mut lcg: u64 = 0x1234_5678;
    let mut rng = || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        lcg
    };
    let mask = |w: u32| if w >= 128 { u128::MAX } else { (1u128 << w) - 1 };
    let (mut lin_bits, mut tot_bits, mut lin_regs, mut tot_regs) = (0usize, 0usize, 0usize, 0usize);
    for packet in &program.streams.tick_next {
        for op in &packet.ops {
            let PackedOp::CaptureReg { dst, next, .. } = op else { continue };
            tot_regs += 1;
            tot_bits += next.ty.width as usize;
            // only extract cones the classifier confirms linear and ≤128-bit out
            if next.ty.width > 128 || !linearize::is_linear(&program, next) {
                continue;
            }
            let form = linearize::extract_linear_form(&program, next);
            let mut ok = true;
            for _ in 0..2000 {
                let leaf_vals: HashMap<usize, u128> = form
                    .leaves
                    .iter()
                    .map(|&(s, w)| (s, ((rng() as u128) | ((rng() as u128) << 64)) & mask(w)))
                    .collect();
                if form.eval(&leaf_vals) != linearize::eval_expr(next, &comb_def, &leaf_vals) {
                    ok = false;
                    break;
                }
            }
            if ok {
                lin_regs += 1;
                lin_bits += next.ty.width as usize;
                let nnz: usize = form.columns.iter().map(|c| c.count_ones() as usize).sum();
                println!(
                    "  LINEAR  reg `{}`: GEMM {}x{} ({} GF(2) nnz)  [BIT-EXACT]",
                    program.signals[*dst].name, form.out_width, form.total_in_bits, nnz
                );
            }
        }
    }
    println!(
        "  => {}/{} reg cones, {}/{} next-state bits are GEMM-offloadable ({:.0}% of sequential logic)",
        lin_regs, tot_regs, lin_bits, tot_bits, lin_bits as f64 / tot_bits.max(1) as f64 * 100.0
    );
}

/// Per-design AI-HW mappability profile: the share of per-cycle op-cost reachable
/// by each EXACT AI-hardware primitive (the lossy fp-matmul majority can't touch
/// bit-exact RTL) vs the general SIMT/SIMD-only remainder.
fn accel_profile(path: &str, top: &str) {
    let Ok(src) = std::fs::read_to_string(path) else { return };
    let imported = import_sv(&src, Some(top)).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, top).expect("lower packed");
    let p = linearize::accel_profile(&program);
    let pc = |x: usize| x as f64 / p.total.max(1) as f64 * 100.0;
    println!(
        "{top:10}: int1-linear {:>4.0}% | gather {:>4.0}% | int-MAC {:>4.0}% | general(SIMT) {:>4.0}%",
        pc(p.linear_int1), pc(p.gather), pc(p.mul_mac), pc(p.general)
    );
}

/// Detect one-hot select chains (case/LUT/crossbar → int8 onehot·data matmul) —
/// the int8-tensor structural class that the GF(2)-linear detector misses.
fn select_report(path: &str, top: &str) {
    let Ok(src) = std::fs::read_to_string(path) else { return };
    let imported = import_sv(&src, Some(top)).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, top).expect("lower packed");
    let sels = linearize::detect_selects(&program);
    let luts = sels.iter().filter(|s| s.const_arms == s.total_arms).count();
    println!(
        "{top}: {} one-hot select chains ({} LUT/ROM → int8 matmul·gather, {} crossbar → SIMT)",
        sels.len(), luts, sels.len() - luts
    );
    for s in sels.iter().take(5) {
        let kind = if s.const_arms == s.total_arms { "LUT  (onehot·table, shared → int8 GEMM)" } else { "xbar (data arms → per-lane mux)" };
        println!(
            "  `{}` ({}-bit) {}-way → {}b out, {}/{} const arms  [{kind}]",
            program.signals[s.selector].name, s.sel_width, s.cases, s.out_width, s.const_arms, s.total_arms
        );
    }
}

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv");
    for (f, t) in [("crc32.sv", "crc32"), ("cpu.sv", "cpu"), ("picorv32.v", "picorv32")] {
        report(&format!("{base}/{f}"), t);
    }
    println!();
    for (f, t) in [("crc32.sv", "crc32"), ("mixed.sv", "mixed"), ("picorv32.v", "picorv32")] {
        partition_report(&format!("{base}/{f}"), t);
    }
    println!("\nAI-HW mappability profile (share of per-cycle op-cost, EXACT primitives only):");
    for (f, t) in [("crc32.sv", "crc32"), ("mixed.sv", "mixed"), ("cpu.sv", "cpu"), ("picorv32.v", "picorv32")] {
        accel_profile(&format!("{base}/{f}"), t);
    }
    println!("\nint8 one-hot-select detection (case/LUT/crossbar → onehot matmul):");
    for (f, t) in [("cpu.sv", "cpu"), ("picorv32.v", "picorv32")] {
        select_report(&format!("{base}/{f}"), t);
    }
}
