//! Measures the curated fused superoperators (Add3, AndOr) on an adder/logic-heavy
//! datapath, comparing GPU throughput with fusion on vs off (same program), and
//! checking bit-exactness. Each fusion removes one intermediate value round-trip,
//! so per the access-count thesis it is a universal win on designs that exercise it.
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example fused_ops_check -- [bits width depth lanes steps]

use std::time::Instant;

use rrtl_core::{compile, uint, Design, Signal};
use rrtl_gpu_sim::interp::{InterpGpuSimulator, InterpProgram};
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};

fn build(bits: u32, width: usize, depth: usize) -> Design {
    let mut design = Design::new();
    let mut m = design.module("Logic");
    let clk = m.input("clk", uint(1));
    let a = m.input("a", uint(bits));
    let b = m.input("b", uint(bits));
    for lane in 0..width {
        // adder reduction -> Add3 fusions
        let s = m.reg(format!("s{lane}"), uint(bits));
        m.clock(s, clk);
        let mut acc = s.value() + a.value();
        for _ in 0..depth {
            acc = acc + a.value() + b.value();
        }
        m.next(s, acc);
        let so = m.output(format!("so{lane}"), uint(bits));
        m.assign(so, s);
        // AOI reduction -> AndOr fusions
        let r = m.reg(format!("r{lane}"), uint(bits));
        m.clock(r, clk);
        let mut acc2 = r.value();
        for _ in 0..depth {
            acc2 = (acc2 & a.value()) | (b.value() & a.value());
        }
        m.next(r, acc2);
        let ro = m.output(format!("ro{lane}"), uint(bits));
        m.assign(ro, r);
    }
    design
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let bits = p(1, 32) as u32;
    let (width, depth, lanes, steps) = (p(2, 64), p(3, 8), p(4, 16384), p(5, 64));
    println!("bits={bits} width={width} depth={depth} lanes={lanes} steps={steps}");

    let design = build(bits, width, depth);
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Logic").unwrap();
    let sig = |name: &str| -> Signal {
        compiled.find_module("Logic").and_then(|mm| mm.signals.iter().find(|s| s.name == name)).map(|s| s.handle).unwrap()
    };
    let off = |s: Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;
    let (a, bb, clk, so0) = (sig("a"), sig("b"), sig("clk"), sig("so0"));

    let machine = lower_to_machine_program(&program);
    let plain = InterpProgram::encode_opts(&machine, false).unwrap();
    let fused = InterpProgram::encode_opts(&machine, true).unwrap();
    println!(
        "  records: plain {} -> fused {}  ({} fused away, {:.1}% fewer)",
        plain.total_code_words() / 6,
        fused.total_code_words() / 6,
        (plain.total_code_words() - fused.total_code_words()) / 6,
        100.0 * (plain.total_code_words() - fused.total_code_words()) as f64 / plain.total_code_words() as f64,
    );

    let mlc = |secs: f64| (lanes * steps) as f64 / secs / 1.0e6;
    let run = |prog: &InterpProgram| -> (f64, Vec<u32>) {
        let gpu = InterpGpuSimulator::new(prog, lanes).unwrap();
        gpu.set_signal(off(clk), &vec![1u32; lanes]);
        gpu.set_signal(off(a), &(0..lanes as u32).map(|l| l.wrapping_mul(2654435761)).collect::<Vec<_>>());
        gpu.set_signal(off(bb), &(0..lanes as u32).map(|l| l.wrapping_mul(40503) + 1).collect::<Vec<_>>());
        gpu.tick_many(8);
        gpu.synchronize();
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t = Instant::now();
            gpu.tick_many(steps);
            gpu.synchronize();
            best = best.min(t.elapsed().as_secs_f64());
        }
        (best, gpu.get_signal(off(so0)))
    };

    let (pt, po) = run(&plain);
    let (ft, fo) = run(&fused);
    println!(
        "  GPU plain {:>6.2} Mlc/s   fused {:>6.2} Mlc/s   ({:.2}x)   [{}]",
        mlc(pt),
        mlc(ft),
        pt / ft,
        if po == fo { "OK" } else { "MISMATCH" },
    );
}
