//! Measures constant-multiply strength reduction on a 64-bit datapath
//! `acc = acc * C + din` for several multiplier constants. Strength reduction
//! only fires for multi-limb multiplies by sparse constants (few NAF terms), so
//! we sweep sparse (×4, ×5, ×9, ×640) and dense (×0x9e37…) multipliers and report
//! GPU-interp throughput original vs reduced, plus a correctness check.
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example strength_check -- [width depth lanes steps]

use std::time::Instant;

use rrtl_core::{compile, lit_u, uint, Design, Signal};
use rrtl_gpu_sim::interp::{InterpGpuSimulator, InterpProgram};
use rrtl_sim_ir::{lower_to_packed_program, strength_reduce_program};

fn build(cmul: u128, width: usize, depth: usize) -> Design {
    let mut design = Design::new();
    let mut m = design.module("Mul");
    let clk = m.input("clk", uint(1));
    let din = m.input("din", uint(64));
    for lane in 0..width {
        let acc = m.reg(format!("acc{lane}"), uint(64));
        m.clock(acc, clk);
        let mut prev = acc;
        for stage in 0..depth {
            let w = m.wire(format!("w{lane}_{stage}"), uint(64));
            m.assign(w, prev * lit_u(cmul, 64) + din);
            prev = w;
        }
        m.next(acc, prev + acc);
        let o = m.output(format!("o{lane}"), uint(64));
        m.assign(o, acc);
    }
    design
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (width, depth, lanes, steps) = (p(1, 64), p(2, 8), p(3, 16384), p(4, 64));
    println!("64-bit  width={width} depth={depth} lanes={lanes} steps={steps}");

    let muls: [u128; 5] = [4, 5, 9, 640, 0x9e37_79b9_7f4a_7c15];
    let din_v: Vec<u128> = (0..lanes as u128).map(|l| l.wrapping_mul(0x1234_5678_9abc) + 1).collect();
    let mlc = |secs: f64| (lanes * steps) as f64 / secs / 1.0e6;

    for &cmul in &muls {
        let design = build(cmul, width, depth);
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Mul").unwrap();
        let (reduced, stats) = strength_reduce_program(&program);
        let sig = |name: &str| -> Signal {
            compiled.find_module("Mul").and_then(|mm| mm.signals.iter().find(|s| s.name == name)).map(|s| s.handle).unwrap()
        };
        let off = |s: Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;
        let (din, clk, o0) = (sig("din"), sig("clk"), sig("o0"));

        let run = |prog: &InterpProgram| -> (f64, Vec<u128>) {
            let gpu = InterpGpuSimulator::new(prog, lanes).unwrap();
            gpu.set_signal(off(clk), &vec![1u32; lanes]);
            gpu.set_signal(off(din), &din_v.iter().map(|&v| v as u32).collect::<Vec<_>>());
            gpu.set_signal(off(din) + 1, &din_v.iter().map(|&v| (v >> 32) as u32).collect::<Vec<_>>());
            gpu.tick_many(8);
            gpu.synchronize();
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t = Instant::now();
                gpu.tick_many(steps);
                gpu.synchronize();
                best = best.min(t.elapsed().as_secs_f64());
            }
            let lo = gpu.get_signal(off(o0));
            let hi = gpu.get_signal(off(o0) + 1);
            (best, lo.iter().zip(&hi).map(|(&l, &h)| (l as u128) | ((h as u128) << 32)).collect())
        };

        let orig = InterpProgram::encode_design(&program).unwrap();
        let red = InterpProgram::encode_design(&reduced).unwrap();
        let (orig_t, orig_o) = run(&orig);
        let (red_t, red_o) = run(&red);
        let ok = orig_o == red_o;
        println!(
            "  *{:#018x}  reduced={} zeroed={}  orig {:>6.2} -> red {:>6.2} Mlc/s  ({:.2}x)  [{}]",
            cmul,
            stats.reduced,
            stats.zeroed,
            mlc(orig_t),
            mlc(red_t),
            orig_t / red_t,
            if ok { "OK" } else { "MISMATCH" },
        );
    }
}
