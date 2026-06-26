//! Measures the two-level standard-cell fusions (AOI21/OAI21) on an AOI/OAI-tree
//! design. Each `~((x op a) op2 b)` is three ops today (binop + binop + not);
//! the fused cell is one op. Reports GPU throughput fusion on vs off + correctness.
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example aoi_check -- [bits width depth lanes steps]

use std::time::Instant;
use rrtl_core::{compile, uint, Design, Signal};
use rrtl_gpu_sim::interp::{InterpGpuSimulator, InterpProgram};
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};

fn build(bits: u32, width: usize, depth: usize) -> Design {
    let mut design = Design::new();
    let mut m = design.module("Cells");
    let clk = m.input("clk", uint(1));
    let a = m.input("a", uint(bits));
    let b = m.input("b", uint(bits));
    for lane in 0..width {
        let acc = m.reg(format!("g{lane}"), uint(bits));
        m.clock(acc, clk);
        let mut x = acc.value();
        for k in 0..depth {
            x = if k % 2 == 0 { !((x & a.value()) | b.value()) } else { !((x | a.value()) & b.value()) };
        }
        m.next(acc, x);
        let o = m.output(format!("o{lane}"), uint(bits));
        m.assign(o, acc);
    }
    design
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let bits = p(1, 32) as u32;
    let (width, depth, lanes, steps) = (p(2, 64), p(3, 10), p(4, 16384), p(5, 64));
    println!("bits={bits} width={width} depth={depth} lanes={lanes} steps={steps}");
    let design = build(bits, width, depth);
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Cells").unwrap();
    let sig = |name: &str| -> Signal { compiled.find_module("Cells").and_then(|mm| mm.signals.iter().find(|s| s.name == name)).map(|s| s.handle).unwrap() };
    let off = |s: Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;
    let (a, bb, clk, o0) = (sig("a"), sig("b"), sig("clk"), sig("o0"));
    let machine = lower_to_machine_program(&program);
    let plain = InterpProgram::encode_opts(&machine, false).unwrap();
    let fused = InterpProgram::encode_opts(&machine, true).unwrap();
    println!("  records: plain {} -> fused {}  ({:.1}% fewer)", plain.total_code_words()/6, fused.total_code_words()/6,
        100.0*(plain.total_code_words()-fused.total_code_words()) as f64/plain.total_code_words() as f64);
    let mlc = |s: f64| (lanes*steps) as f64 / s / 1.0e6;
    let run = |prog: &InterpProgram| -> (f64, Vec<u32>) {
        let gpu = InterpGpuSimulator::new(prog, lanes).unwrap();
        gpu.set_signal(off(clk), &vec![1u32; lanes]);
        gpu.set_signal(off(a), &(0..lanes as u32).map(|l| l.wrapping_mul(2654435761)).collect::<Vec<_>>());
        gpu.set_signal(off(bb), &(0..lanes as u32).map(|l| l.wrapping_mul(40503)+1).collect::<Vec<_>>());
        gpu.tick_many(8); gpu.synchronize();
        let mut best = f64::INFINITY;
        for _ in 0..3 { let t = Instant::now(); gpu.tick_many(steps); gpu.synchronize(); best = best.min(t.elapsed().as_secs_f64()); }
        (best, gpu.get_signal(off(o0)))
    };
    let (pt, po) = run(&plain); let (ft, fo) = run(&fused);
    println!("  GPU plain {:>6.2} Mlc/s   fused {:>6.2} Mlc/s   ({:.2}x)   [{}]", mlc(pt), mlc(ft), pt/ft, if po==fo {"OK"} else {"MISMATCH"});
}
