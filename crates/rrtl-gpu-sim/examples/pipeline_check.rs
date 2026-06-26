//! Capstone: stacks the design-specializer pipeline on a mixed datapath (AOI gate
//! logic + multiply-accumulate arithmetic + constant-foldable config) and reports
//! the cumulative GPU throughput from the oblivious baseline through
//! constant-folding and fusion to the full pipeline. All combos are checked
//! bit-identical to the baseline.
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example pipeline_check -- [bits width depth lanes steps]
use std::time::Instant;
use rrtl_core::{compile, lit_u, uint, Design, Signal};
use rrtl_gpu_sim::interp::{InterpGpuSimulator, InterpProgram};
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program, specialize_program};

fn build(bits: u32, width: usize, depth: usize) -> Design {
    let mut design = Design::new();
    let mut m = design.module("Mix");
    let clk = m.input("clk", uint(1));
    let a = m.input("a", uint(bits));
    let b = m.input("b", uint(bits));
    let din = m.input("din", uint(bits));
    for lane in 0..width {
        // constant config that folds to a literal: (5+7)*1, masked by all-ones.
        let k = m.wire(format!("k{lane}"), uint(bits));
        let allones: u128 = if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 };
        m.assign(k, ((lit_u(5, bits) + lit_u(7, bits)) * lit_u(1, bits)) & lit_u(allones, bits));
        // gate chain: AOI22 cells.
        let g = m.reg(format!("g{lane}"), uint(bits));
        m.clock(g, clk);
        let mut x = g.value();
        for _ in 0..depth { x = !((x & a.value()) | (b.value() & din.value())); }
        m.next(g, x);
        let go = m.output(format!("go{lane}"), uint(bits));
        m.assign(go, g);
        // arithmetic chain: multiply-accumulate with the folded constant.
        let acc = m.reg(format!("acc{lane}"), uint(bits));
        m.clock(acc, clk);
        let mut y = acc.value();
        for _ in 0..depth { y = y * lit_u(0x9e37_79b9, bits) + din.value() + k.value(); }
        m.next(acc, y + acc.value());
        let ao = m.output(format!("ao{lane}"), uint(bits));
        m.assign(ao, acc);
    }
    design
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let bits = p(1, 32) as u32;
    let (width, depth, lanes, steps) = (p(2, 64), p(3, 6), p(4, 16384), p(5, 64));
    println!("bits={bits} width={width} depth={depth} lanes={lanes} steps={steps}");

    let design = build(bits, width, depth);
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Mix").unwrap();
    let sig = |n: &str| -> Signal { compiled.find_module("Mix").and_then(|mm| mm.signals.iter().find(|s| s.name == n)).map(|s| s.handle).unwrap() };
    let off = |s: Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;
    let (a, b, din, clk, ao0) = (sig("a"), sig("b"), sig("din"), sig("clk"), sig("ao0"));

    let machine = lower_to_machine_program(&program);
    let spec = specialize_program(&machine).0;
    let baseline = InterpProgram::encode_opts(&machine, false).unwrap();
    let folded = InterpProgram::encode_opts(&spec, false).unwrap();
    let fused = InterpProgram::encode_opts(&machine, true).unwrap();
    let full = InterpProgram::encode_opts(&spec, true).unwrap();
    println!("  records: baseline {}  fold {}  fuse {}  full {}",
        baseline.total_code_words()/6, folded.total_code_words()/6, fused.total_code_words()/6, full.total_code_words()/6);

    let mlc = |s: f64| (lanes * steps) as f64 / s / 1.0e6;
    let progs: [(&str, &InterpProgram); 4] = [("baseline", &baseline), ("+fold", &folded), ("+fuse", &fused), ("+full", &full)];
    // Build + warm all sims, then interleave timed runs so every combo shares the
    // same thermal profile (best-of-N per combo).
    let sims: Vec<_> = progs.iter().map(|(_, prog)| {
        let gpu = InterpGpuSimulator::new(prog, lanes).unwrap();
        gpu.set_signal(off(clk), &vec![1u32; lanes]);
        for s in [a, b, din] { gpu.set_signal(off(s), &(0..lanes as u32).map(|l| l.wrapping_mul(2654435761)+1).collect::<Vec<_>>()); }
        gpu.tick_many(16); gpu.synchronize();
        gpu
    }).collect();
    let mut best = vec![f64::INFINITY; 4];
    for _ in 0..6 {
        for (i, gpu) in sims.iter().enumerate() {
            let t = Instant::now();
            gpu.tick_many(steps); gpu.synchronize();
            best[i] = best[i].min(t.elapsed().as_secs_f64());
        }
    }
    let bo = sims[0].get_signal(off(ao0));
    let b0 = best[0];
    for (i, (n, _)) in progs.iter().enumerate() {
        let o = sims[i].get_signal(off(ao0));
        println!("  {n:<13} {:>6.2} Mlc/s   ({:.2}x)   [{}]", mlc(best[i]), b0 / best[i], if o == bo {"OK"} else {"MISMATCH"});
    }
}
