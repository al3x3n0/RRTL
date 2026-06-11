//! Validates the design-as-data interpreter GPU kernel against the CPU SIMD
//! engine, and times it. Crucially runs designs the straight-line codegen
//! cannot compile (its WGSL blows up super-linearly).
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example gpu_interp_check -- [W D lanes steps]

use std::time::Instant;

use rrtl_core::{compile, lit_u, uint, Design, Signal};
use rrtl_gpu_sim::interp::{InterpGpuSimulator, InterpProgram};
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
    let (width, depth, lanes, steps) = (p(1, 8), p(2, 4), p(3, 1024), p(4, 64));
    println!("Wide width={width} depth={depth} lanes={lanes} steps={steps}");

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
    let off = |s: Signal| {
        program.signals[program.signal_index(s).unwrap()]
            .layout
            .offset
    };
    let (din, clk, o0) = (sig("din"), sig("clk"), sig("o0"));

    let din_u32: Vec<u32> = (0..lanes as u32).map(|l| l.wrapping_mul(2_654_435_761)).collect();
    let din_u128: Vec<u128> = din_u32.iter().map(|&v| v as u128).collect();

    eprintln!("encoding ...");
    let encoded = InterpProgram::encode_design(&program).unwrap();
    let words: usize = encoded.blocks.iter().map(|b| b.len()).sum();
    println!("encoded code: {} bytes ({} signals)", words * 4, program.signals.len());

    eprintln!("GPU construct ...");
    let gpu = InterpGpuSimulator::new(&encoded, lanes).unwrap();
    gpu.set_signal(off(clk), &vec![1u32; lanes]);
    gpu.set_signal(off(din), &din_u32);
    eprintln!("GPU tick_many ...");
    let t = Instant::now();
    gpu.tick_many(steps);
    let gpu_out = gpu.get_signal(off(o0));
    let gpu_secs = t.elapsed().as_secs_f64();
    eprintln!("GPU done");

    let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
    cpu.set_signal(clk, &vec![1u128; lanes]).unwrap();
    cpu.set_signal(din, &din_u128).unwrap();
    let t = Instant::now();
    cpu.tick_many(steps).unwrap();
    let cpu_out: Vec<u32> = cpu.get_signal(o0).unwrap().iter().map(|&v| v as u32).collect();
    let cpu_secs = t.elapsed().as_secs_f64();

    let ok = gpu_out == cpu_out;
    let mlc = |secs: f64| (lanes * steps) as f64 / secs / 1.0e6;
    println!(
        "correctness: {}   CPU-SIMD {:.0} Mlc/s   GPU-interp {:.0} Mlc/s   ({:.2}x)",
        if ok { "OK" } else { "MISMATCH" },
        mlc(cpu_secs),
        mlc(gpu_secs),
        cpu_secs / gpu_secs,
    );
}
