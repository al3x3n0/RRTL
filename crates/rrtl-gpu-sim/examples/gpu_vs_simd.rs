//! GPU batch throughput vs the CPU SIMD engine across stimulus-lane counts.
//!
//! One fixed design (width W, comb depth D) run across a sweep of independent
//! stimulus lanes. Reports lane-cycles/sec for `SimdCpuSimulator::tick_many` vs
//! `GpuBatchSimulator::tick_many` (LaneMajor and WordMajor layouts), with a
//! differential check (GPU == CPU). Finds the crossover and the GPU ceiling.
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example gpu_vs_simd -- [W D steps]

use std::time::Instant;

use rrtl_core::{compile, lit_u, uint, Design, Signal};
use rrtl_gpu_sim::{GpuBatchOptions, GpuBatchSimulator, GpuMemoryLayout};
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

fn handle(design_compiled: &rrtl_core::CompiledDesign, name: &str) -> Signal {
    design_compiled
        .find_module("Wide")
        .and_then(|m| m.signals.iter().find(|s| s.name == name))
        .map(|s| s.handle)
        .expect("signal handle")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let width = p(1, 64);
    let depth = p(2, 8);
    let steps = p(3, 256);
    // Optional 4th arg: a single lane count to isolate (else full sweep).
    let lane_counts: Vec<usize> = match args.get(4).and_then(|a| a.parse().ok()) {
        Some(n) => vec![n],
        None => vec![256, 1024, 4096, 16384, 65536],
    };

    println!("Wide design width={width} comb_depth={depth} steps={steps}\n");

    let design = build_wide(width, depth);
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Wide").unwrap();
    let din = handle(&compiled, "din");
    let clk = handle(&compiled, "clk");
    let o0 = handle(&compiled, "o0");

    let mlc = |lanes: usize, secs: f64| (lanes * steps) as f64 / secs / 1.0e6;

    println!(
        "{:>7}  {:>14}  {:>14} {:>6}  {:>14} {:>6}",
        "lanes", "CPU-SIMD Mlc/s", "GPU-lane Mlc/s", "x", "GPU-word Mlc/s", "x"
    );
    for &lanes in &lane_counts {
        let din_u32: Vec<u32> = (0..lanes as u32)
            .map(|l| l.wrapping_mul(2_654_435_761))
            .collect();
        let din_u128: Vec<u128> = din_u32.iter().map(|&v| v as u128).collect();
        let clk_u32 = vec![1u32; lanes];
        let clk_u128 = vec![1u128; lanes];

        eprintln!("[lanes={lanes}] CPU-SIMD ...");
        // CPU SIMD
        let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        cpu.set_signal(din, &din_u128).unwrap();
        cpu.set_signal(clk, &clk_u128).unwrap();
        let t = Instant::now();
        cpu.tick_many(steps).unwrap();
        let cpu_secs = t.elapsed().as_secs_f64();
        let cpu_out = cpu.get_signal(o0).unwrap();

        let mut line = format!("{lanes:>7}  {:>14.0}", mlc(lanes, cpu_secs));

        for (tag, layout) in [
            ("lane", GpuMemoryLayout::LaneMajor),
            ("word", GpuMemoryLayout::WordMajor),
        ] {
            let options = GpuBatchOptions {
                memory_layout: layout,
                ..GpuBatchOptions::default()
            };
            eprintln!("[lanes={lanes}] GPU {tag} construct ...");
            let mut gpu =
                GpuBatchSimulator::new_with_options(&design, "Wide", lanes, options).unwrap();
            gpu.set_input(din, &din_u32).unwrap();
            gpu.set_input(clk, &clk_u32).unwrap();
            eprintln!("[lanes={lanes}] GPU {tag} tick_many ...");
            let t = Instant::now();
            gpu.tick_many(steps).unwrap();
            let gpu_secs = t.elapsed().as_secs_f64();
            eprintln!("[lanes={lanes}] GPU {tag} readback ...");
            let gpu_out = gpu.get_signal(o0).unwrap();
            eprintln!("[lanes={lanes}] GPU {tag} done ({gpu_secs:.3}s)");
            let ok = (0..lanes).all(|i| (cpu_out[i] as u32) == gpu_out[i]);
            let _ = tag;
            line.push_str(&format!(
                "  {:>14.0} {:>5.1}x{}",
                mlc(lanes, gpu_secs),
                cpu_secs / gpu_secs,
                if ok { "" } else { "!" },
            ));
        }
        println!("{line}");
    }
    println!("\n(x = speedup vs CPU-SIMD; trailing ! = GPU/CPU MISMATCH)");
}
