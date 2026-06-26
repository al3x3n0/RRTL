//! The vector JIT as RRTL's CPU *batch* backend: N independent instances of one
//! design, distinct stimulus, advanced in a native group×cycle loop. Compares
//! lane-cycles/s against the SIMD interpreter (`SimdCpuSimulator`) at the same
//! lane count — the native-code-vs-interpreter batch throughput question.
//!
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example simd_jit_batch_bench -- [lanes width depth cycles]
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_core::{compile, lit_u, uint, Design, Signal};
    use rrtl_sim_ir::{jit::SimdJitSimulator, lower_to_machine_program, lower_to_packed_program, SimdCpuSimulator};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (lanes, width, depth, cycles, bits) = (p(1, 256), p(2, 8), p(3, 8), p(4, 100_000), p(5, 32) as u32);
    println!("lanes={lanes} width={width} depth={depth} cycles={cycles} bits={bits}");
    let cmask = if bits >= 32 { u128::MAX } else { (1u128 << bits) - 1 };

    let mut design = Design::new();
    {
        let mut m = design.module("Wide");
        let clk = m.input("clk", uint(1));
        let din = m.input("din", uint(bits));
        for lane in 0..width {
            let acc = m.reg(format!("acc{lane}"), uint(bits));
            m.clock(acc, clk);
            let mut prev = acc.value();
            for s in 0..depth {
                let w = m.wire(format!("w{lane}_{s}"), uint(bits));
                m.assign(w, prev * lit_u(0x9e37_79b9 & cmask, bits) + din.value());
                prev = w.value();
            }
            m.next(acc, prev + acc.value());
            let o = m.output(format!("o{lane}"), uint(bits));
            m.assign(o, acc);
        }
    }
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Wide").unwrap();
    let machine = lower_to_machine_program(&program);
    let h = |n: &str| -> Signal { compiled.find_module("Wide").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle };
    let idx = |n: &str| program.signal_index(h(n)).unwrap();

    let mut jit = SimdJitSimulator::compile_lanes(&machine, lanes).expect("compile");
    let real_lanes = jit.lanes();
    let mut cpu = SimdCpuSimulator::new(program.clone(), real_lanes).unwrap();
    println!("  vector JIT → {real_lanes} lanes (packing chosen by design max width)");

    // distinct per-lane stimulus
    let din_of = |l: usize| (l as u32).wrapping_mul(2654435761) | 1;
    for l in 0..real_lanes { jit.set_signal(l, idx("clk"), 1); jit.set_signal(l, idx("din"), din_of(l)); }
    let din_lanes: Vec<u128> = (0..real_lanes).map(|l| din_of(l) as u128).collect();
    cpu.set_signal(h("clk"), &vec![1u128; real_lanes]).unwrap();
    cpu.set_signal(h("din"), &din_lanes).unwrap();

    // correctness: a few cycles, JIT vs interpreter, all lanes
    for _ in 0..5 { jit.tick(); cpu.tick().unwrap(); }
    let mut ok = true;
    let cv = cpu.get_signal(h("o0")).unwrap();
    for l in 0..real_lanes { if jit.get_signal(l, idx("o0")) as u128 != cv[l] { ok = false; } }
    println!("  correctness (JIT vs SIMD-interp, lane o0): [{}]", if ok { "OK" } else { "MISMATCH" });

    // throughput
    let lane_cyc = (real_lanes * cycles) as f64;
    let timed = |j: &mut SimdJitSimulator| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..3 { let t = Instant::now(); j.tick_many(cycles); best = best.min(t.elapsed().as_secs_f64()); }
        lane_cyc / best / 1e6
    };

    jit.set_parallel(false);
    let js = timed(&mut jit);
    jit.set_parallel(true);
    let jp = timed(&mut jit);

    let mut bc = f64::INFINITY;
    for _ in 0..3 { let t = Instant::now(); for _ in 0..cycles { cpu.tick().unwrap(); } bc = bc.min(t.elapsed().as_secs_f64()); }
    let clane = lane_cyc / bc / 1e6;

    println!("  --- batch throughput (lane-cycles/s), {} cores ---", rayon::current_num_threads());
    println!("  vector JIT (serial)    {:>8.1} M-lane-cyc/s", js);
    println!("  vector JIT (rayon)     {:>8.1} M-lane-cyc/s   ({:.1}x multicore)", jp, jp / js);
    println!("  SIMD interpreter       {:>8.1} M-lane-cyc/s", clane);
    println!("  rayon vector JIT is {:.0}x the interpreter at {real_lanes} lanes", jp / clane);
}
