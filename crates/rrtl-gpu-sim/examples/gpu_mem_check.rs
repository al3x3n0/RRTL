//! Validates the GPU interpreter's memory support against SimdCpuSimulator.
//! A 2-bit counter drives the memory address, so a single tick_many (fixed
//! inputs) still exercises distinct read/write addresses each cycle.
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example gpu_mem_check -- [lanes steps]

use rrtl_core::{compile, lit_u, uint, Design, Signal};
use rrtl_gpu_sim::interp::{InterpGpuSimulator, InterpProgram};
use rrtl_sim_ir::{lower_to_packed_program, SimdCpuSimulator};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (lanes, steps) = (p(1, 4096), p(2, 64));

    let mut design = Design::new();
    let (clk, din, rdata);
    {
        let mut m = design.module("Top");
        clk = m.input("clk", uint(1));
        din = m.input("din", uint(2));
        let cnt = m.reg("cnt", uint(2));
        m.clock(cnt, clk);
        m.next(cnt, cnt + lit_u(1, 2));
        let data_w = m.wire("data_w", uint(2));
        m.assign(data_w, din + cnt.value());
        let mem = m.mem("mem", 2, uint(2), 4);
        m.mem_write(mem, clk, lit_u(1, 1), cnt, data_w);
        let rd = m.mem_read(mem, cnt);
        rdata = m.output("rdata", uint(2));
        m.assign(rdata, rd);
    }

    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Top").unwrap();
    let off = |s: Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;

    let din_v: Vec<u32> = (0..lanes as u32).map(|l| l & 0x3).collect();
    let din_v128: Vec<u128> = din_v.iter().map(|&v| v as u128).collect();

    let gpu = InterpGpuSimulator::new(&InterpProgram::encode_design(&program).unwrap(), lanes).unwrap();
    gpu.set_signal(off(clk), &vec![1u32; lanes]);
    gpu.set_signal(off(din), &din_v);
    gpu.tick_many(steps);
    let gpu_out = gpu.get_signal(off(rdata));

    let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
    cpu.set_signal(clk, &vec![1u128; lanes]).unwrap();
    cpu.set_signal(din, &din_v128).unwrap();
    cpu.tick_many(steps).unwrap();
    let cpu_out: Vec<u32> = cpu.get_signal(rdata).unwrap().iter().map(|&v| v as u32).collect();

    let ok = gpu_out == cpu_out;
    println!(
        "memory design: lanes={lanes} steps={steps}  GPU-interp vs CPU-SIMD: {}",
        if ok { "OK" } else { "MISMATCH" }
    );
}
