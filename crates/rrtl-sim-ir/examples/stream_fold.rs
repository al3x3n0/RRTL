//! Stream-folding demo: process a length-N input stream through a GF(2)-linear
//! block by P segments computed independently (P-way parallel) and stitched with
//! the M^N operator (the generalized zlib `crc32_combine`), bit-exact. Uses the
//! library [`rrtl_sim_ir::leap::LinearLeap::fold`]. See also `linear_leap`.
//! Build: cargo run --release -p rrtl-sim-ir --example stream_fold
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
    let din = program.signals.iter().position(|s| s.name.ends_with(".din")).expect("din");
    let leap = LinearLeap::build(&program).expect("linear register cones");

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
    let zero = || leap.registers().iter().map(|&(d, _)| (d, 0u128)).collect::<HashMap<_, _>>();
    let run = |s0: &HashMap<usize, u128>, stream: &[u128]| -> HashMap<usize, u128> {
        let mut st = s0.clone();
        for &d in stream {
            let mut leaves = st.clone();
            leaves.insert(din, d);
            st = nexts.iter().map(|(dd, w, n)| (*dd, linearize::eval_expr(n, &comb, &leaves) & ((1u128 << w) - 1))).collect();
        }
        st
    };
    let crc = leap.registers().iter().map(|&(d, _)| d).find(|&d| program.signals[d].name.contains("crc")).unwrap();
    let crc_of = |v: u128| (v >> leap.bit_base(crc).unwrap()) & 0xFFFF_FFFF;

    let n = 4096usize;
    let stream: Vec<u128> = (0..n).map(|k| ((k as u128).wrapping_mul(2654435761) >> 13) & 0xff).collect();
    let mut s0 = zero();
    s0.insert(crc, 0xFFFF_FFFF);
    let seq = pack(&run(&s0, &stream));
    println!("crc32 stream-fold: N={n} byte stream, sequential crc=0x{:08x}", crc_of(seq));

    for &p in &[2usize, 4, 8, 16] {
        let l = n / p;
        // V_i = segment i from ZERO state — independent, P-way parallelizable.
        let vs: Vec<u128> = (0..p).map(|i| pack(&run(&zero(), &stream[i * l..(i + 1) * l]))).collect();
        let total = leap.fold(pack(&s0), l as u64, &vs);
        let ok = crc_of(total) == crc_of(seq);
        println!("  P={p:>2} (seg len {l}, from-zero independent) -> crc=0x{:08x}  {}",
            crc_of(total), if ok { "== sequential [OK, P-way parallel]" } else { "!= [MISMATCH]" });
        assert!(ok);
    }
    println!("  result: BIT-EXACT — one linear stream folds into P independent segments + M^N stitch");
}
