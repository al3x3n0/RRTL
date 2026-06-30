//! Temporal leap demo: compute the state of an idle GF(2)-linear block N cycles
//! ahead in O(log N) matrix mults (vs N steps), bit-exact. Uses the library
//! [`rrtl_sim_ir::leap::LinearLeap`]. See also `stream_fold` (varying input).
//! Build: cargo run --release -p rrtl-sim-ir --example linear_leap
use std::collections::HashMap;

use rrtl_sim_ir::leap::LinearLeap;
use rrtl_sim_ir::{linearize, lower_to_packed_program, PackedOp};
use rrtl_sv_frontend::import_sv;

fn main() {
    let src = std::fs::read_to_string("bench/sv/crc32.sv").expect("read crc32.sv");
    let imported = import_sv(&src, Some("crc32")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "crc32").expect("lower");
    let comb = linearize::comb_defs(&program);
    let leap = LinearLeap::build(&program).expect("linear register cones");

    // Reference stepper over the linear reg cones (din held 0 = idle).
    let mut nexts: Vec<(usize, u32, &rrtl_sim_ir::PackedExpr)> = Vec::new();
    for packet in &program.streams.tick_next {
        for op in &packet.ops {
            if let PackedOp::CaptureReg { dst, next, .. } = op {
                if linearize::is_linear(&program, next) {
                    nexts.push((*dst, next.ty.width, next));
                }
            }
        }
    }
    let pack = |st: &HashMap<usize, u128>| -> u128 {
        let mut v = 0u128;
        for &(d, w) in leap.registers() {
            v |= (st[&d] & ((1u128 << w) - 1)) << leap.bit_base(d).unwrap();
        }
        v
    };
    let step = |st: &HashMap<usize, u128>| -> HashMap<usize, u128> {
        nexts.iter().map(|(d, w, n)| (*d, linearize::eval_expr(n, &comb, st) & ((1u128 << w) - 1))).collect()
    };
    let crc = leap.registers().iter().map(|&(d, _)| d).find(|&d| program.signals[d].name.contains("crc")).unwrap();
    let crc_of = |v: u128| (v >> leap.bit_base(crc).unwrap()) & 0xFFFF_FFFF;

    let mut s0: HashMap<usize, u128> = leap.registers().iter().map(|&(d, _)| (d, 0u128)).collect();
    s0.insert(crc, 0xFFFF_FFFF);
    let v0 = pack(&s0);
    println!("crc32 idle leap: {}-bit combined state ({})", leap.state_bits(),
        leap.registers().iter().map(|&(d, w)| format!("{}={w}b", program.signals[d].name.rsplit('.').next().unwrap())).collect::<Vec<_>>().join("+"));

    let mut all_ok = true;
    for &n in &[1u64, 2, 7, 64, 1000, 100_000] {
        let leaped = crc_of(leap.leap_idle(v0, n));
        let mut st = s0.clone();
        for _ in 0..n {
            st = step(&st);
        }
        let ok = crc_of(pack(&st)) == leaped;
        all_ok &= ok;
        println!("  N={n:>7}: leap crc=0x{leaped:08x}  {} (~{} matmuls vs {n} steps)",
            if ok { "== stepped [OK]" } else { "!= stepped [MISMATCH]" }, 2 * (64 - (n.max(1)).leading_zeros()));
    }
    let huge = 1u64 << 40;
    println!("  N=2^40 ({huge}): leap crc=0x{:08x} in ~{} matmuls (stepping would never finish)",
        crc_of(leap.leap_idle(v0, huge)), 2 * (64 - huge.leading_zeros()));
    println!("  result: {}", if all_ok { "BIT-EXACT — O(log N) temporal leap" } else { "MISMATCH" });
    assert!(all_ok);
}
