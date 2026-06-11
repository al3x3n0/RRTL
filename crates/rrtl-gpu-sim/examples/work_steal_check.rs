//! Validates and benchmarks the work-stealing CPU+GPU batch executor: the result
//! must be per-lane identical to running all lanes on the CPU, and the throughput
//! should self-balance to roughly cpu_rate + gpu_rate without a static split.
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example work_steal_check -- [W D lanes steps gpu_tile cpu_tile]

use std::time::Instant;

use rrtl_core::{compile, lit_u, uint, Design, Signal};
use rrtl_gpu_sim::work_steal::WorkStealingBatch;
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
    let (width, depth, lanes, steps) = (p(1, 32), p(2, 4), p(3, 8192), p(4, 64));
    let (gpu_tile, cpu_tile) = (p(5, 2048), p(6, 256));
    println!(
        "Wide W={width} D={depth} lanes={lanes} steps={steps} gpu_tile={gpu_tile} cpu_tile={cpu_tile}\n"
    );

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
    let ref_o0 = oracle.get_signal(o0).unwrap();

    let mut ws = WorkStealingBatch::new(program.clone(), lanes, gpu_tile, cpu_tile).unwrap();
    ws.set_input(clk, clk_v.clone());
    ws.set_input(din, din_v.clone());

    // Warm up, then best-of-3.
    let _ = ws.run(steps, &[o0]);
    let mut best = f64::INFINITY;
    let mut out = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        out = ws.run(steps, &[o0]);
        best = best.min(t.elapsed().as_secs_f64());
    }
    let ok = out[0] == ref_o0;
    let st = ws.stats();
    let mlc = (lanes * steps) as f64 / best / 1.0e6;
    println!(
        "work-steal: {:.1} Mlc/s   correctness {}\n  GPU {} tiles / {} lanes ({:.0}%);  CPU {} tiles / {} lanes ({:.0}%)",
        mlc,
        if ok { "OK" } else { "MISMATCH" },
        st.gpu_tiles,
        st.gpu_lanes,
        100.0 * st.gpu_lanes as f64 / lanes as f64,
        st.cpu_tiles,
        st.cpu_lanes,
        100.0 * st.cpu_lanes as f64 / lanes as f64,
    );
}
