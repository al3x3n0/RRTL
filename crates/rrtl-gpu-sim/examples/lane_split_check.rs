//! Validates and benchmarks the CPU‖GPU lane-split simulator: every split must
//! be per-lane identical to running all lanes on the CPU (the oracle), and the
//! combined throughput should approach cpu_rate + gpu_rate near the balanced
//! split.
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example lane_split_check -- [W D lanes steps]

use std::time::Instant;

use rrtl_core::{compile, lit_u, uint, Design, Signal};
use rrtl_gpu_sim::lane_split::LaneSplitSimulator;
use rrtl_sim_ir::{lower_to_packed_program, SimdCpuSimulator};

fn build_wide(width: usize, depth: usize) -> Design {
    let mut design = Design::new();
    let mut m = design.module("Wide");
    let clk = m.input("clk", uint(1));
    let din = m.input("din", uint(32));
    for lane in 0..width {
        let acc = m.reg(format!("acc{lane}"), uint(32));
        m.clock(acc, clk);
        let mut prev = acc;
        for stage in 0..depth {
            let w = m.wire(format!("w{lane}_{stage}"), uint(32));
            m.assign(w, prev * lit_u(0x9e37_79b9, 32) + din);
            prev = w;
        }
        m.next(acc, prev + acc);
        let o = m.output(format!("o{lane}"), uint(32));
        m.assign(o, acc);
    }
    design
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (width, depth, lanes, steps) = (p(1, 64), p(2, 8), p(3, 16384), p(4, 32));
    println!("Wide width={width} depth={depth} total_lanes={lanes} steps={steps}\n");

    let design = build_wide(width, depth);
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Wide").unwrap();
    let sig = |name: &str| -> Signal {
        compiled
            .find_module("Wide")
            .and_then(|m| m.signals.iter().find(|s| s.name == name))
            .map(|s| s.handle)
            .unwrap()
    };
    let (din, clk, o0) = (sig("din"), sig("clk"), sig("o0"));

    let din_v: Vec<u128> = (0..lanes as u128).map(|l| l.wrapping_mul(2_654_435_761) & 0xffff_ffff).collect();
    let clk_v = vec![1u128; lanes];

    // Oracle: all lanes on the CPU.
    let mut oracle = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
    oracle.set_signal(clk, &clk_v).unwrap();
    oracle.set_signal(din, &din_v).unwrap();
    oracle.tick_many(steps).unwrap();
    let ref_out = oracle.get_signal(o0).unwrap();

    let mlc = |secs: f64| (lanes * steps) as f64 / secs / 1.0e6;
    let run = |gpu_lanes: usize| -> (f64, bool) {
        let mut sim = LaneSplitSimulator::new(program.clone(), lanes, gpu_lanes).unwrap();
        sim.set_signal(clk, &clk_v).unwrap();
        sim.set_signal(din, &din_v).unwrap();
        // Correctness on a fresh state.
        sim.tick_many(steps).unwrap();
        let ok = sim.get_signal(o0).unwrap() == ref_out;
        // Throughput: warm up, then best-of-3 (state evolves; timing is what matters).
        sim.tick_many(steps).unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t = Instant::now();
            sim.tick_many(steps).unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        (best, ok)
    };

    println!("{:>12} {:>8} {:>14} {:>8}", "split", "gpu_%", "Mlc/s", "correct");
    let mut cpu_only = 0.0;
    let mut gpu_only = 0.0;
    for &frac in &[0.0f64, 0.25, 0.5, 0.75, 0.875, 1.0] {
        let gpu_lanes = ((lanes as f64 * frac).round() as usize).min(lanes);
        let (secs, ok) = run(gpu_lanes);
        if frac == 0.0 {
            cpu_only = mlc(secs);
        }
        if frac == 1.0 {
            gpu_only = mlc(secs);
        }
        println!(
            "{:>12} {:>7.0}% {:>14.1} {:>8}",
            format!("{}/{}", gpu_lanes, lanes - gpu_lanes),
            frac * 100.0,
            mlc(secs),
            if ok { "OK" } else { "MISMATCH" },
        );
    }

    // Throughput-proportional split from the measured single-device rates.
    if cpu_only > 0.0 && gpu_only > 0.0 {
        let gpu_frac = gpu_only / (cpu_only + gpu_only);
        let gpu_lanes = ((lanes as f64 * gpu_frac).round() as usize).min(lanes);
        let (secs, ok) = run(gpu_lanes);
        println!(
            "\nproportional split (gpu {:.0}%): {:.1} Mlc/s  [{}]   vs cpu-only {:.1}, gpu-only {:.1}, ideal-sum {:.1}",
            gpu_frac * 100.0,
            mlc(secs),
            if ok { "OK" } else { "MISMATCH" },
            cpu_only,
            gpu_only,
            cpu_only + gpu_only,
        );
    }
}
