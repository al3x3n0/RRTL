//! A representative AI datapath: a W-wide INT8 MAC array (the GEMM/conv primitive)
//! — 8-bit a/b inputs, 16-bit products, 32-bit accumulators. Measures (1) RRTL's
//! vector-JIT batch throughput (the moat for AI datapaths) and (2) the LOW-PRECISION
//! PACKING opportunity: how much of the design is sub-32-bit, i.e. lane bits the
//! current I32X4 vector JIT wastes on INT8/INT16 values.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example ai_mac_bench -- [W cycles lanes]
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_core::{compile, uint, Design};
    use rrtl_sim_ir::jit::SimdJitSimulator;
    use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
    use std::time::Instant;

    let a: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| a.get(i).and_then(|x| x.parse().ok()).unwrap_or(d);
    let (w, cycles, lanes) = (p(1, 64), p(2, 200_000), p(3, 1024));
    // out_mode: 0 = one output per PE (W comb stores/cycle); 1 = single XOR-reduced
    // output (1 comb store/cycle) — to isolate per-PE output-store bandwidth.
    let out_mode = p(4, 0);

    // W parallel INT8 MAC lanes: acc(32) += a(8) * b(8). The product is 16-bit
    // (255*255 < 2^16); the accumulator 32-bit. This is the GEMM/conv inner kernel.
    let mut design = Design::new();
    {
        use rrtl_core::lit_u;
        let mut m = design.module("aimac");
        let clk = m.input("clk", uint(1));
        let mut accs = Vec::new();
        for i in 0..w {
            let ai = m.input(format!("a{i}"), uint(8));
            let bi = m.input(format!("b{i}"), uint(8));
            let acc = m.reg(format!("acc{i}"), uint(32));
            m.clock(acc, clk);
            let prod = (ai.value().zext(16) * bi.value().zext(16)).trunc(16);
            m.next(acc, (acc.value() + prod.zext(32)).trunc(32));
            accs.push(acc);
            if out_mode == 0 {
                let o = m.output(format!("o{i}"), uint(32));
                m.assign(o, acc.value());
            }
        }
        if out_mode != 0 {
            // single XOR-reduced output (1 comb store/cycle, like the W×D datapath)
            let mut red = lit_u(0, 32);
            for acc in &accs {
                red = red ^ acc.value();
            }
            let o = m.output("o", uint(32));
            m.assign(o, red);
        }
    }
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "aimac").unwrap();
    let machine = lower_to_machine_program(&program);

    // ---- low-precision packing opportunity: signal-bits by width bucket ----
    let (mut le8, mut le16, mut wide, mut total) = (0u64, 0u64, 0u64, 0u64);
    for s in &program.signals {
        let bits = s.layout.width as u64;
        total += bits;
        if bits <= 8 {
            le8 += bits;
        } else if bits <= 16 {
            le16 += bits;
        } else {
            wide += bits;
        }
    }
    println!("AI INT8 MAC array: W={w} PEs ({} signals)", program.signals.len());
    println!("  signal-bit width profile (the packing opportunity):");
    println!("    <=8-bit : {:.0}%   <=16-bit (incl 8): {:.0}%   >16-bit: {:.0}%",
        100.0 * le8 as f64 / total as f64,
        100.0 * (le8 + le16) as f64 / total as f64,
        100.0 * wide as f64 / total as f64);
    println!("    => I32X4 vector JIT carries every value in a 32-bit lane; an INT8/INT16");
    println!("       pack (I8X16/I16X8) could run up to ~4x more lanes on the {:.0}% sub-32-bit bits",
        100.0 * (le8 + le16) as f64 / total as f64);

    // ---- op mix: how much is width-conversion overhead (zext/trunc/cast)? ----
    use rrtl_sim_ir::PackedInstrKind as K;
    let (mut widthops, mut mul, mut add, mut other, mut totops) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for blk in [&machine.streams.comb, &machine.streams.tick_next] {
        for pk in &blk.packets {
            for ins in &pk.instrs {
                totops += 1;
                match ins.kind {
                    K::Zext(_) | K::Trunc(_) | K::Cast(_) | K::Sext(_) | K::Slice { .. } | K::Concat(_) => widthops += 1,
                    K::Mul(..) => mul += 1,
                    K::Add(..) => add += 1,
                    _ => other += 1,
                }
            }
        }
    }
    println!("  op mix ({totops} comb+next instrs): width-conv (zext/trunc/cast/slice/concat) {:.0}%, mul {:.0}%, add {:.0}%, other {:.0}%",
        100.0 * widthops as f64 / totops as f64, 100.0 * mul as f64 / totops as f64,
        100.0 * add as f64 / totops as f64, 100.0 * other as f64 / totops as f64);
    println!("    => {:.0}% width-conversion ops are no-ops in a 32-bit lane (a mixed-precision AI tax to attack)",
        100.0 * widthops as f64 / totops as f64);

    // ---- the moat: vector-JIT batch throughput ----
    if let Ok(mut v) = SimdJitSimulator::compile_lanes(&machine, lanes) {
        let l = v.lanes();
        v.tick_many(2000);
        let t = Instant::now();
        v.tick_many(cycles);
        let secs = t.elapsed().as_secs_f64();
        let mlc = (l * cycles) as f64 / secs / 1e6;
        println!("  RRTL vector-JIT batch: lanes={l}  {mlc:.1} M-lane-cyc/s  (per-instance {:.2} Mcyc/s)",
            mlc / l as f64);
        println!("    (each lane = one independent MAC-array instance; vs N Verilator processes)");
    } else {
        println!("  (vector JIT unavailable for this design)");
    }
}
