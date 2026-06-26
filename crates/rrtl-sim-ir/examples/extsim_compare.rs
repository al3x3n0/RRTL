//! RRTL side of the cross-simulator benchmark (vs Verilator and PyRTL
//! FastSimulation). Same design + same stimulus as bench/extsim/{dut.v,
//! dut_pyrtl.py}; prints `out` for the cross-check and cycles/s.
//!   single-instance (apples-to-apples vs Verilator/PyRTL): the scalar JIT.
//!   batch (the axis they lack): the vector JIT × lanes × cores.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example extsim_compare -- [W D N lanes]
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_core::{compile, lit_u, uint, Design, Expr, Signal};
    use rrtl_sim_ir::{jit::{JitSimulator, SimdJitSimulator}, lower_to_machine_program, lower_to_packed_program};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (w, depth, n, lanes) = (p(1, 16), p(2, 8), p(3, 500_000), p(4, 1024));

    const C: u64 = 0x9e37_79b9;
    let mut design = Design::new();
    {
        let mut m = design.module("dut");
        let clk = m.input("clk", uint(1));
        let din = m.input("din", uint(32));
        let mut accs = Vec::new();
        for i in 0..w {
            let acc = m.reg(format!("acc{i}"), uint(32));
            m.clock(acc, clk);
            let mut t: Expr = acc.value();
            for _ in 0..depth {
                t = t * lit_u(C as u128, 32) + din.value();
            }
            // per-channel salt so channels diverge (XOR-reduce is a real checksum)
            m.next(acc, t + acc.value() + lit_u(i as u128, 32));
            accs.push(acc);
        }
        let mut x: Expr = accs[0].value();
        for a in &accs[1..] {
            x = x ^ a.value();
        }
        let out = m.output("out", uint(32));
        m.assign(out, x);
    }
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "dut").unwrap();
    let machine = lower_to_machine_program(&program);
    let h = |nm: &str| -> Signal { compiled.find_module("dut").unwrap().signals.iter().find(|s| s.name == nm).unwrap().handle };
    let idx = |nm: &str| program.signal_index(h(nm)).unwrap();
    let din_of = |c: usize| ((c as u64).wrapping_mul(2654435761)) as u32;

    println!("W={w} D={depth} N={n} lanes={lanes}");

    // --- single instance: scalar JIT, per-cycle stimulus (matches Verilator/PyRTL) ---
    let mut jit = JitSimulator::compile(&machine).unwrap();
    jit.set_signal(idx("clk"), 1);
    let run = |jit: &mut JitSimulator| {
        for c in 0..n {
            jit.set_signal(idx("din"), din_of(c) as u64);
            jit.tick();
        }
    };
    run(&mut jit); // warm
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t0 = Instant::now();
        run(&mut jit);
        best = best.min(t0.elapsed().as_secs_f64());
    }
    let out = jit.get_signal(idx("out"));
    println!("rrtl-scalar-jit out={out} cycles={n} {:.2} Mcyc/s {:.1} ms", n as f64 / best / 1e6, best * 1e3);

    // --- batch axis: vector JIT × lanes × cores (held stimulus = raw throughput) ---
    if let Ok(mut v) = SimdJitSimulator::compile_lanes(&machine, lanes) {
        let l = v.lanes();
        for lane in 0..l { v.set_signal(lane, idx("clk"), 1); v.set_signal(lane, idx("din"), din_of(lane)); }
        for _ in 0..2000 { v.tick(); }
        let mut best = f64::INFINITY;
        for _ in 0..3 { let t = Instant::now(); v.tick_many(n); best = best.min(t.elapsed().as_secs_f64()); }
        let lane_cyc = (l * n) as f64 / best / 1e6;
        println!("rrtl-vector-jit-batch lanes={l} {:.1} M-lane-cyc/s (per-instance {:.2} Mcyc/s)", lane_cyc, lane_cyc / l as f64);
    }
}
