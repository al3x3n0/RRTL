//! GPU-interp vs CPU-SIMD throughput on a wide datapath at a configurable bit
//! width — measures the multi-limb cost (32-bit/1-limb vs 64-bit/2-limb) and
//! whether the GPU still wins on wide designs.
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example gpu_throughput -- [bits W D lanes steps]

use std::time::Instant;

use rrtl_core::{compile, lit_u, uint, Design, Signal};
use rrtl_gpu_sim::interp::{InterpGpuSimulator, InterpProgram};
use rrtl_sim_ir::{lower_to_packed_program, SimdCpuSimulator};

fn build_wide(bits: u32, width: usize, depth: usize, mixed: bool) -> Design {
    let c: u128 = if bits >= 64 { 0x9e37_79b9_7f4a_7c15 } else { 0x9e37_79b9 };
    let mut design = Design::new();
    let mut m = design.module("Wide");
    let clk = m.input("clk", uint(1));
    let din = m.input("din", uint(bits));
    if mixed {
        // One barely-used 64-bit register forces max_limbs=2, so every (mostly
        // 32-bit) value pays the multi-limb tax under uniform-limb storage.
        let din64 = m.input("din64", uint(64));
        let extra = m.reg("extra64", uint(64));
        m.clock(extra, clk);
        m.next(extra, din64);
        let oe = m.output("oextra", uint(64));
        m.assign(oe, extra);
    }
    for lane in 0..width {
        let acc = m.reg(format!("acc{lane}"), uint(bits));
        m.clock(acc, clk);
        let mut prev = acc;
        for stage in 0..depth {
            let w = m.wire(format!("w{lane}_{stage}"), uint(bits));
            m.assign(w, prev * lit_u(c, bits) + din);
            prev = w;
        }
        m.next(acc, prev + acc);
        let o = m.output(format!("o{lane}"), uint(bits));
        m.assign(o, acc);
    }
    design
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let bits = p(1, 32) as u32;
    let (width, depth, lanes, steps) = (p(2, 64), p(3, 8), p(4, 16384), p(5, 64));
    let mixed = p(6, 0) != 0;
    let limbs = ((bits + 31) / 32) as usize;
    println!("bits={bits} (limbs={limbs}) width={width} depth={depth} lanes={lanes} steps={steps} mixed={mixed}");

    let design = build_wide(bits, width, depth, mixed);
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Wide").unwrap();
    let sig = |name: &str| -> Signal {
        compiled.find_module("Wide").and_then(|m| m.signals.iter().find(|s| s.name == name)).map(|s| s.handle).unwrap()
    };
    let off = |s: Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;
    let (din, clk, o0) = (sig("din"), sig("clk"), sig("o0"));

    let mask = if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 };
    let din_v: Vec<u128> = (0..lanes as u128).map(|l| l.wrapping_mul(0x1234_5678_9abc_def1) & mask).collect();
    let mlc = |secs: f64| (lanes * steps) as f64 / secs / 1.0e6;

    // CPU SIMD
    let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
    cpu.set_signal(clk, &vec![1u128; lanes]).unwrap();
    cpu.set_signal(din, &din_v).unwrap();
    cpu.tick_many(8).unwrap();
    let mut cpu_best = f64::INFINITY;
    let mut cpu_o = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        cpu.tick_many(steps).unwrap();
        cpu_best = cpu_best.min(t.elapsed().as_secs_f64());
        cpu_o = cpu.get_signal(o0).unwrap();
    }

    // GPU interp
    let gpu = InterpGpuSimulator::new(&InterpProgram::encode_design(&program).unwrap(), lanes).unwrap();
    gpu.set_signal(off(clk), &vec![1u32; lanes]);
    for l in 0..limbs {
        let limb: Vec<u32> = din_v.iter().map(|&v| (v >> (32 * l)) as u32).collect();
        gpu.set_signal(off(din) + l, &limb);
    }
    gpu.tick_many(8);
    gpu.synchronize();
    let mut gpu_best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        gpu.tick_many(steps);
        gpu.synchronize();
        gpu_best = gpu_best.min(t.elapsed().as_secs_f64());
    }
    let limb_vecs: Vec<Vec<u32>> = (0..limbs).map(|l| gpu.get_signal(off(o0) + l)).collect();
    let gpu_o: Vec<u128> = (0..lanes)
        .map(|lane| {
            let mut v = 0u128;
            for (l, lv) in limb_vecs.iter().enumerate() {
                v |= (lv[lane] as u128) << (32 * l);
            }
            v
        })
        .collect();

    let ok = gpu_o == cpu_o;
    println!(
        "  CPU-SIMD {:>8.1} Mlc/s   GPU-interp {:>8.1} Mlc/s   ({:.1}x)   [{}]",
        mlc(cpu_best),
        mlc(gpu_best),
        cpu_best / gpu_best,
        if ok { "OK" } else { "MISMATCH" },
    );
}
