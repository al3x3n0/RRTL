//! Auto-detected instance folding end-to-end: a Top of N independent instances of
//! a datapath module M. Compares naive (flattened Top at L lanes) vs folded
//! (InstanceFoldSimulator: M at N*L lanes, auto-detected) — identical total work,
//! so datapath-megacycles/sec is directly comparable. Verifies per-instance
//! correctness and sweeps L to show the under-saturation win.
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example instance_fold_check -- [insts depth steps]
use std::time::Instant;
use rrtl_core::{compile, lit_u, uint, Design, Signal};
use rrtl_gpu_sim::instance_fold::InstanceFoldSimulator;
use rrtl_gpu_sim::interp::{InterpGpuSimulator, InterpProgram};
use rrtl_sim_ir::{analyze_instance_fold, lower_to_packed_program};

fn build(insts: usize, depth: usize) -> Design {
    let mut design = Design::new();
    {
        let mut m = design.module("M");
        let clk = m.input("clk", uint(1));
        let din = m.input("din", uint(32));
        let acc = m.reg("acc", uint(32));
        m.clock(acc, clk);
        let mut prev = acc;
        for s in 0..depth {
            let w = m.wire(format!("w{s}"), uint(32));
            m.assign(w, prev * lit_u(0x9e37_79b9, 32) + din);
            prev = w;
        }
        m.next(acc, prev + acc);
        let o = m.output("o", uint(32));
        m.assign(o, acc);
    }
    {
        let mut m = design.module("Top");
        let clk = m.input("clk", uint(1));
        for i in 0..insts {
            let din = m.input(format!("din{i}"), uint(32));
            let o = m.output(format!("o{i}"), uint(32));
            m.instance(format!("u{i}"), "M", [("clk", clk), ("din", din), ("o", o)]);
        }
    }
    design
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (insts, depth, steps) = (p(1, 64), p(2, 8), p(3, 64));
    println!("instances={insts} depth={depth} steps={steps}");

    let design = build(insts, depth);
    let compiled = compile(&design).unwrap();
    let top = lower_to_packed_program(&compiled, "Top").unwrap();
    let fold = analyze_instance_fold(&compiled, "Top").expect("foldable");
    let m = lower_to_packed_program(&compiled, &fold.child_module).unwrap();
    let nprog = InterpProgram::encode_design(&top).unwrap();
    let fprog = InterpProgram::encode_design(&m).unwrap();
    println!("  code words: naive Top {} -> folded M {}  ({:.1}x smaller)",
        nprog.total_code_words(), fprog.total_code_words(),
        nprog.total_code_words() as f64 / fprog.total_code_words().max(1) as f64);

    let toff = |name: &str| top.signals.iter().find(|s| s.name.rsplit('.').next() == Some(name)).unwrap().layout.offset;
    let din_at = |i: usize, l: usize| ((i as u32) * 2654435761).wrapping_add((l as u32) * 40503 + 1);

    // correctness at a small lane count
    {
        let base = 64usize;
        let mut naive = InterpGpuSimulator::new(&nprog, base).unwrap();
        naive.set_signal(toff("clk"), &vec![1u32; base]);
        for i in 0..insts { naive.set_signal(toff(&format!("din{i}")), &(0..base).map(|l| din_at(i,l)).collect::<Vec<_>>()); }
        naive.tick_many(8); naive.synchronize();
        let folded = InstanceFoldSimulator::new(&compiled, "Top", base).unwrap();
        folded.set_port("clk", 0, &vec![1u32; folded.total_lanes()]).unwrap();
        folded.set_port("din", 0, &(0..folded.total_lanes()).map(|lane| din_at(lane/base, lane%base)).collect::<Vec<_>>()).unwrap();
        folded.tick_many(8); folded.synchronize();
        let fo = folded.get_port("o", 0).unwrap();
        let mut ok = true;
        for i in 0..insts { if naive.get_signal(toff(&format!("o{i}"))) != fo[i*base..(i+1)*base] { ok = false; } }
        println!("  correctness (naive Top vs folded M, base=64): [{}]", if ok {"OK"} else {"MISMATCH"});
    }

    println!("  base_lanes |  naive (NxL)   |  folded (1 x N*L)  |  speedup");
    for &base in &[64usize, 256, 1024, 4096] {
        let work = (insts * base * steps) as f64 / 1.0e6;
        let mut naive = InterpGpuSimulator::new(&nprog, base).unwrap();
        naive.set_signal(toff("clk"), &vec![1u32; base]);
        for i in 0..insts { naive.set_signal(toff(&format!("din{i}")), &(0..base).map(|l| din_at(i,l)).collect::<Vec<_>>()); }
        naive.tick_many(8); naive.synchronize();
        let mut nt = f64::INFINITY;
        for _ in 0..3 { let t = Instant::now(); naive.tick_many(steps); naive.synchronize(); nt = nt.min(t.elapsed().as_secs_f64()); }

        let folded = InstanceFoldSimulator::new(&compiled, "Top", base).unwrap();
        folded.set_port("clk", 0, &vec![1u32; folded.total_lanes()]).unwrap();
        folded.set_port("din", 0, &(0..folded.total_lanes()).map(|lane| din_at(lane/base, lane%base)).collect::<Vec<_>>()).unwrap();
        folded.tick_many(8); folded.synchronize();
        let mut ft = f64::INFINITY;
        for _ in 0..3 { let t = Instant::now(); folded.tick_many(steps); folded.synchronize(); ft = ft.min(t.elapsed().as_secs_f64()); }
        println!("  {base:>9}  |  {:>6.1} Mdc/s |  {:>6.1} Mdc/s    |  {:.2}x", work/nt, work/ft, (work/ft)/(work/nt));
    }
    let _: Option<Signal> = None;
}
