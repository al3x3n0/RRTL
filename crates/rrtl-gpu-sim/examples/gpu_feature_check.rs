//! Validates the GPU interpreter's concat + reset support against the CPU SIMD
//! engine. A counter drives reset-deassert timing so a single tick_many exercises
//! both reset-asserted and normal capture, plus a concatenated output.
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example gpu_feature_check -- [lanes steps]

use rrtl_core::{compile, concat, lit_u, uint, Design, Signal};
use rrtl_gpu_sim::interp::{InterpGpuSimulator, InterpProgram};
use rrtl_sim_ir::{lower_to_packed_program, SimdCpuSimulator};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (lanes, steps) = (p(1, 4096), p(2, 64));

    let mut design = Design::new();
    let (clk, din, out);
    {
        let mut m = design.module("Top");
        clk = m.input("clk", uint(1));
        din = m.input("din", uint(8));
        // 1-bit toggle drives the sync reset, so `acc` is reset every other
        // cycle and captures `acc + din` otherwise.
        let tgl = m.reg("tgl", uint(1));
        m.clock(tgl, clk);
        m.next(tgl, tgl + lit_u(1, 1));
        let acc = m.reg("acc", uint(8));
        m.clock(acc, clk);
        m.reset(acc, tgl, 0x55);
        m.next(acc, acc + din);
        // Output = {acc, tgl} via concat (9 bits).
        out = m.output("out", uint(9));
        m.assign(out, concat([acc.value(), tgl.value()]));
    }

    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Top").unwrap();
    let off = |s: Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;

    let din_v: Vec<u32> = (0..lanes as u32).map(|l| (l * 3 + 1) & 0xff).collect();
    let din_v128: Vec<u128> = din_v.iter().map(|&v| v as u128).collect();

    let gpu = InterpGpuSimulator::new(&InterpProgram::encode_design(&program).unwrap(), lanes).unwrap();
    gpu.set_signal(off(clk), &vec![1u32; lanes]);
    gpu.set_signal(off(din), &din_v);
    gpu.tick_many(steps);
    let gpu_out = gpu.get_signal(off(out));

    let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
    cpu.set_signal(clk, &vec![1u128; lanes]).unwrap();
    cpu.set_signal(din, &din_v128).unwrap();
    cpu.tick_many(steps).unwrap();
    let cpu_out: Vec<u32> = cpu.get_signal(out).unwrap().iter().map(|&v| v as u32).collect();

    println!(
        "concat + reset design: lanes={lanes} steps={steps}  GPU-interp vs CPU-SIMD: {}",
        if gpu_out == cpu_out { "OK" } else { "MISMATCH" }
    );
}
