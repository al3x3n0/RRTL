//! Activity-skipping JIT vs the oblivious JIT, single-instance. A bank of gated
//! accumulators under two stimulus regimes (IDLE: enables low + data held →
//! cones skip; BUSY: all enabled + data churning → nothing skips), built at two
//! cone *granularities*:
//!   - FINE: each datapath step is its own combinational wire ⇒ one cheap cone
//!     (a single mul-add) per step. The per-cone guard (load fan-in + snapshot,
//!     compare, branch) costs as much as the eval, so skipping can't pay.
//!   - COARSE: each accumulate is one deep inline expression ⇒ one expensive
//!     cone (many ops) with few leaves, so one guard amortizes a lot of work.
//! Tests whether native-branch event skipping pays at single-instance latency,
//! and how that hinges on cone coarseness.
//!
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example jit_activity_bench -- [lanes depth ticks]
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_core::{compile, lit_u, mux, uint, Design, Expr, Signal};
    use rrtl_sim_ir::{jit::JitSimulator, lower_to_machine_program, lower_to_packed_program};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (lanes, depth, ticks) = (p(1, 64), p(2, 6), p(3, 1_000_000));
    println!("lanes={lanes} depth={depth} ticks={ticks}");

    let build = |coarse: bool| -> Design {
        let mut design = Design::new();
        {
            let mut m = design.module("Bank");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let x = m.input("x", uint(32));
            let en = m.input("en", uint(lanes as u32));
            for lane in 0..lanes {
                let acc = m.reg(format!("acc{lane}"), uint(32));
                m.clock(acc, clk);
                m.reset(acc, rst, 0);
                let eni = en.value().slice(lane as u32, 1);
                let path: Expr = if coarse {
                    // one deep inline expression: a single heavy cone, leaves {acc,x}
                    let mut e = acc.value();
                    for _ in 0..depth {
                        e = e * lit_u(0x9e37_79b9, 32) + x.value();
                    }
                    e
                } else {
                    // each step is its own wire: many cheap cones
                    let mut prev = acc.value();
                    for s in 0..depth {
                        let w = m.wire(format!("w{lane}_{s}"), uint(32));
                        m.assign(w, prev * lit_u(0x9e37_79b9, 32) + x.value());
                        prev = w.value();
                    }
                    prev
                };
                m.next(acc, mux(eni, path, acc.value()));
                let o = m.output(format!("o{lane}"), uint(32));
                m.assign(o, acc);
            }
        }
        design
    };

    for (label, coarse) in [("FINE cones (per-wire)", false), ("COARSE cones (deep inline)", true)] {
        let design = build(coarse);
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Bank").unwrap();
        let machine = lower_to_machine_program(&program);
        let h = |n: &str| -> Signal { compiled.find_module("Bank").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle };
        let idx = |n: &str| program.signal_index(h(n)).unwrap();

        let mut plain = JitSimulator::compile(&machine).expect("compile plain");
        let mut act = JitSimulator::compile_activity(&machine).expect("compile activity");
        let mut act_m = JitSimulator::compile_activity_instrumented(&machine).expect("compile instrumented");

        // correctness: activity must match the oblivious JIT bit-for-bit
        let mut lcg: u64 = 0xabc;
        let mut rng = || { lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1); lcg };
        let mut ok = true;
        for c in 0..200u32 {
            let rst = (c == 0) as u64;
            let xv = (rng() & 0xffff) as u64;
            let env = if c % 8 == 0 { rng() } else { 0 };
            for (n, v) in [("clk", 1u64), ("rst", rst), ("x", xv), ("en", env)] {
                plain.set_signal(idx(n), v); act.set_signal(idx(n), v);
            }
            plain.tick(); act.tick();
            for l in 0..lanes {
                if plain.get_signal(idx(&format!("o{l}"))) != act.get_signal(idx(&format!("o{l}"))) { ok = false; }
            }
        }

        let all = if lanes >= 64 { u64::MAX } else { (1u64 << lanes) - 1 };
        let idle_bench = |label: &str, j: &mut JitSimulator| -> f64 {
            for _ in 0..2000 { j.tick(); }
            let mut best = f64::INFINITY;
            for _ in 0..3 { let t = Instant::now(); j.tick_many(ticks); best = best.min(t.elapsed().as_secs_f64()); }
            let mhz = ticks as f64 / best / 1e6;
            println!("      {label:<16} {:>7.2} M-cycles/s", mhz);
            mhz
        };
        let busy_bench = |j: &mut JitSimulator, idx: &dyn Fn(&str) -> usize| -> f64 {
            let mut lcg: u64 = 1;
            let mut step = |j: &mut JitSimulator| { lcg = lcg.wrapping_mul(2654435761).wrapping_add(1); j.set_signal(idx("x"), lcg & 0xffff_ffff); j.tick(); };
            for _ in 0..2000 { step(j); }
            let mut best = f64::INFINITY;
            for _ in 0..3 { let t = Instant::now(); for _ in 0..ticks { step(j); } best = best.min(t.elapsed().as_secs_f64()); }
            ticks as f64 / best / 1e6
        };

        println!("  {label}: correctness [{}]", if ok { "OK" } else { "MISMATCH" });

        // IDLE
        for j in [&mut plain, &mut act, &mut act_m] {
            j.set_signal(idx("clk"), 1); j.set_signal(idx("rst"), 0);
            j.set_signal(idx("x"), 0x1234_5678); j.set_signal(idx("en"), 0);
        }
        for _ in 0..1000 { act_m.tick(); }
        println!("    IDLE (en=0, x held):");
        let pi = idle_bench("oblivious", &mut plain);
        let ai = idle_bench("activity", &mut act);
        println!("      => activity {:.2}x at {:.1}% skip", ai / pi, act_m.activity_skip_rate().unwrap() * 100.0);

        // BUSY
        for j in [&mut plain, &mut act] { j.set_signal(idx("rst"), 0); j.set_signal(idx("en"), all); }
        let pb = busy_bench(&mut plain, &idx);
        let ab = busy_bench(&mut act, &idx);
        println!("    BUSY (en=all, x churning, incl set_signal): oblivious {pb:.2} / activity {ab:.2} M-cyc/s => {:.2}x", ab / pb);
    }
}
