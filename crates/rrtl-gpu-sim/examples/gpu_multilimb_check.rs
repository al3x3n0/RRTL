//! Validates the multi-limb (>32-bit) GPU interpreter kernel against the CPU
//! SIMD engine on a 64-bit datapath (mul + add + bitwise + slice + concat).
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example gpu_multilimb_check -- [lanes steps]

use rrtl_core::{compile, concat, lit_u, uint, Design, Signal};
use rrtl_gpu_sim::interp::{InterpGpuSimulator, InterpProgram};
use rrtl_sim_ir::{lower_to_packed_program, SimdCpuSimulator};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (lanes, steps) = (p(1, 4096), p(2, 48));

    let mut design = Design::new();
    let (clk, din, o);
    {
        let mut m = design.module("Top");
        clk = m.input("clk", uint(1));
        din = m.input("din", uint(64));
        let acc = m.reg("acc", uint(64));
        m.clock(acc, clk);
        // mul (the multi-limb hard case) + add + xor
        let t = m.wire("t", uint(64));
        m.assign(t, acc * lit_u(0x9e37_79b9_7f4a_7c15, 64) + din);
        let mixed = m.wire("mixed", uint(64));
        m.assign(mixed, t ^ din);
        m.next(acc, mixed);
        // exercise slice + concat: {acc[31:0], acc[63:32]} (rotate halves)
        let lo = m.wire("lo", uint(32));
        let hi = m.wire("hi", uint(32));
        m.assign(lo, acc.value().slice(0, 32));
        m.assign(hi, acc.value().slice(32, 32));
        o = m.output("o", uint(64));
        m.assign(o, concat([lo.value(), hi.value()]));
    }

    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Top").unwrap();
    let encoded = InterpProgram::encode_design(&program).unwrap();
    assert!(encoded.max_limbs >= 2, "expected multi-limb");
    let off = |s: Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;

    let din_v: Vec<u128> = (0..lanes as u128)
        .map(|l| (l.wrapping_mul(0x1234_5678_9abc_def1)) & 0xffff_ffff_ffff_ffff)
        .collect();

    let gpu = InterpGpuSimulator::new(&encoded, lanes).unwrap();
    gpu.set_signal(off(clk), &vec![1u32; lanes]);
    // 64-bit din: set low + high limbs.
    let lo: Vec<u32> = din_v.iter().map(|&v| v as u32).collect();
    let hi: Vec<u32> = din_v.iter().map(|&v| (v >> 32) as u32).collect();
    gpu.set_signal(off(din), &lo);
    gpu.set_signal(off(din) + 1, &hi);
    gpu.tick_many(steps);
    let g_lo = gpu.get_signal(off(o));
    let g_hi = gpu.get_signal(off(o) + 1);
    let gpu_o: Vec<u128> = g_lo
        .iter()
        .zip(&g_hi)
        .map(|(&l, &h)| (l as u128) | ((h as u128) << 32))
        .collect();

    let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
    cpu.set_signal(clk, &vec![1u128; lanes]).unwrap();
    cpu.set_signal(din, &din_v).unwrap();
    cpu.tick_many(steps).unwrap();
    let cpu_o = cpu.get_signal(o).unwrap();

    println!(
        "64-bit design: lanes={lanes} steps={steps} max_limbs={}  GPU-interp vs CPU-SIMD: {}",
        encoded.max_limbs,
        if gpu_o == cpu_o { "OK" } else { "MISMATCH" }
    );
}
