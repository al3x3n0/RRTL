//! Measures GPU per-dispatch fixed cost vs per-cycle compute for the interp
//! kernel. tick_many(N) is a single dispatch (on-device loop), so timing across
//! N gives: time(N) = dispatch_overhead + N * per_cycle. The intercept is the
//! fixed cost per CPU<->GPU sync; K_min = dispatch_overhead / per_cycle is the
//! minimum boundary slack a heterogeneous design-partition needs to break even.
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example gpu_dispatch_cost -- [W D lanes]

use std::time::Instant;

use rrtl_core::{compile, lit_u, uint, Design, Signal};
use rrtl_gpu_sim::interp::{InterpGpuSimulator, InterpProgram};
use rrtl_sim_ir::lower_to_packed_program;

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
    let (width, depth, lanes) = (p(1, 32), p(2, 4), p(3, 4096));
    println!("Wide width={width} depth={depth} lanes={lanes}");

    let design = build_wide(width, depth);
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Wide").unwrap();
    let off = |name: &str| {
        let s: Signal = compiled
            .find_module("Wide")
            .and_then(|m| m.signals.iter().find(|s| s.name == name))
            .map(|s| s.handle)
            .unwrap();
        program.signals[program.signal_index(s).unwrap()].layout.offset
    };

    let encoded = InterpProgram::encode_design(&program).unwrap();
    let gpu = InterpGpuSimulator::new(&encoded, lanes).unwrap();
    gpu.set_signal(off("clk"), &vec![1u32; lanes]);
    gpu.set_signal(
        off("din"),
        &(0..lanes as u32).map(|l| l.wrapping_mul(2_654_435_761)).collect::<Vec<_>>(),
    );

    // Warm up (first dispatch pays one-time costs).
    gpu.tick_many(64);
    gpu.synchronize();

    // Submission is now async: tick_many returns before the GPU finishes, so a
    // CPU+GPU lane-split can run the CPU's lanes during this window.
    let t = Instant::now();
    gpu.tick_many(256);
    let submit_us = t.elapsed().as_secs_f64() * 1e6;
    gpu.synchronize();
    let synced_us = t.elapsed().as_secs_f64() * 1e6;
    println!(
        "async submit: tick_many(256) returns in {submit_us:.1} us; \
         GPU actually done after {synced_us:.1} us\n"
    );

    println!("{:>8} {:>14} {:>14}", "steps", "sync_us", "ns/cycle");
    let best_of = |steps: usize| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            gpu.tick_many(steps);
            gpu.synchronize();
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };
    for steps in [1usize, 2, 4, 8, 16, 64, 256, 1024] {
        let secs = best_of(steps);
        println!(
            "{:>8} {:>14.2} {:>14.1}",
            steps,
            secs * 1e6,
            secs * 1e9 / steps as f64
        );
    }
    println!("\n(intercept at steps->0 = per-dispatch/per-sync fixed cost; \
              K_min ~= that / per-cycle compute)");
}
