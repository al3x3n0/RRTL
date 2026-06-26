//! Measures activity-based cone-skipping in PartitionedSimulator: N independent
//! gated accumulators, each with its OWN enable + data (so each register-cone has
//! private leaves). A rotating one-hot enable keeps all but a few cones idle each
//! cycle. Compares tick() (oracle, evaluates everything) vs tick_activity()
//! (skips idle cones), reporting skip rate, speedup, and bit-exactness.
//!
//! Usage: cargo run --release -p rrtl-sim-ir --example cone_skip_check -- [n active lanes cycles]
use std::time::Instant;
use rrtl_core::{compile, mux, uint, Design, Signal};
use rrtl_sim_ir::{lower_to_packed_program, PartitionedSimulator};

fn build(n: usize, depth: usize) -> Design {
    let mut design = Design::new();
    let mut m = design.module("Accs");
    let clk = m.input("clk", uint(1));
    for i in 0..n {
        let en = m.input(format!("en{i}"), uint(1));
        let din = m.input(format!("din{i}"), uint(8));
        let acc = m.reg(format!("acc{i}"), uint(8));
        m.clock(acc, clk);
        // small cone per accumulator (depth adds work so skipping a cone saves more)
        let mut v = acc.value() + din.value();
        for _ in 0..depth {
            v = mux(en.value(), v.clone() + din.value(), v);
        }
        m.next(acc, mux(en.value(), v, acc.value()));
        let o = m.output(format!("o{i}"), uint(8));
        m.assign(o, acc);
    }
    design
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (n, active, lanes, cycles) = (p(1, 32), p(2, 2), p(3, 64), p(4, 400));
    println!("accumulators={n} active/cycle={active} lanes={lanes} cycles={cycles} depth=6");

    let design = build(n, 6);
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Accs").unwrap();
    let h = |name: &str| -> Signal {
        compiled.find_module("Accs").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle
    };
    let clk = h("clk");

    let mut oracle = PartitionedSimulator::new_register_balanced(&program, n, lanes).unwrap();
    let mut act = PartitionedSimulator::new_register_balanced(&program, n, lanes).unwrap();
    oracle.set_parallel(false);
    act.set_parallel(false);
    oracle.set_signal(clk, &vec![1u128; lanes]).unwrap();
    act.set_signal(clk, &vec![1u128; lanes]).unwrap();
    println!("  groups={}", act.group_count());

    // per-accumulator held data; enable rotates a window of `active` accumulators.
    let mut held = vec![vec![0u128; lanes]; n];
    let stim = |cyc: usize, held: &mut Vec<Vec<u128>>| -> Vec<(usize, u128)> {
        let mut ens = vec![0u128; n];
        for k in 0..active {
            let i = (cyc * active + k) % n;
            ens[i] = 1;
            held[i] = (0..lanes as u128).map(|l| (l * 7 + cyc as u128 + i as u128) & 0xff).collect();
        }
        ens.into_iter().enumerate().collect()
    };

    let mut mismatch = false;
    // correctness: run both in lockstep, compare all outputs
    for cyc in 0..40usize {
        let ens = stim(cyc, &mut held);
        for i in 0..n {
            oracle.set_signal(h(&format!("en{i}")), &vec![ens[i].1; lanes]).unwrap();
            act.set_signal(h(&format!("en{i}")), &vec![ens[i].1; lanes]).unwrap();
            oracle.set_signal(h(&format!("din{i}")), &held[i]).unwrap();
            act.set_signal(h(&format!("din{i}")), &held[i]).unwrap();
        }
        oracle.tick();
        act.tick_activity();
        for i in 0..n {
            if oracle.get_signal(h(&format!("o{i}"))).unwrap() != act.get_signal(h(&format!("o{i}"))).unwrap() {
                mismatch = true;
            }
        }
    }
    println!("  correctness (tick vs tick_activity): [{}]", if mismatch {"MISMATCH"} else {"OK"});

    // timing: oracle
    let time = |label: &str, sim: &mut PartitionedSimulator, use_act: bool| -> f64 {
        // warm
        for cyc in 0..20usize { let _ = stim(cyc, &mut vec![vec![0u128; lanes]; n]); if use_act { sim.tick_activity(); } else { sim.tick(); } }
        let mut held = vec![vec![0u128; lanes]; n];
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t = Instant::now();
            for cyc in 0..cycles {
                let ens = stim(cyc, &mut held);
                for i in 0..n {
                    sim.set_signal(h(&format!("en{i}")), &vec![ens[i].1; lanes]).unwrap();
                    sim.set_signal(h(&format!("din{i}")), &held[i]).unwrap();
                }
                if use_act { sim.tick_activity(); } else { sim.tick(); }
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        let _ = label;
        best
    };
    let ot = time("oracle", &mut oracle, false);
    let at = time("activity", &mut act, true);
    println!("  oracle (full)   {:>7.2} ms", ot * 1e3);
    println!("  activity-skip   {:>7.2} ms   ({:.2}x)   skip-rate {:.1}%", at * 1e3, ot / at, 100.0 * act.activity_skip_rate());
}
