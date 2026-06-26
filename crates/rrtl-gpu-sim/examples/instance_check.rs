//! Measures instance-level data parallelism: folding N independent instances of
//! a module into the lane dimension. "Naive" = N flattened datapaths at L lanes
//! (what hierarchy flattening produces today); "folded" = 1 datapath at N*L lanes
//! (instances folded into parallelism). Both do identical total work
//! (N*L datapath-instances), so throughput in datapath-megacycles/sec is directly
//! comparable, and the code stream shrinks N-fold.
//!
//! The hypothesis (from the access-count thesis): folding wins only when the base
//! lane count is below GPU saturation (it fills idle threads); at saturation it is
//! a code-size / capacity win, not a throughput win. We sweep L to find the line.
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example instance_check -- [bits insts depth steps]

use std::time::Instant;

use rrtl_core::{compile, lit_u, uint, Design};
use rrtl_gpu_sim::interp::{InterpGpuSimulator, InterpProgram};
use rrtl_sim_ir::lower_to_packed_program;

fn build(bits: u32, datapaths: usize, depth: usize) -> Design {
    let c: u128 = if bits >= 64 { 0x9e37_79b9_7f4a_7c15 } else { 0x9e37_79b9 };
    let mut design = Design::new();
    let mut m = design.module("M");
    let clk = m.input("clk", uint(1));
    let din = m.input("din", uint(bits));
    for lane in 0..datapaths {
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
    let (insts, depth, steps) = (p(2, 64), p(3, 8), p(4, 64));
    let limbs = ((bits + 31) / 32) as usize;
    println!("bits={bits} instances={insts} depth={depth} steps={steps}");

    // Build once to report code-size reduction.
    let naive_design = build(bits, insts, depth);
    let folded_design = build(bits, 1, depth);
    let nc = compile(&naive_design).unwrap();
    let fc = compile(&folded_design).unwrap();
    let np = lower_to_packed_program(&nc, "M").unwrap();
    let fp = lower_to_packed_program(&fc, "M").unwrap();
    let noff = |name: &str| np.signals[np.signal_index(nc.find_module("M").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle).unwrap()].layout.offset;
    let foff = |name: &str| fp.signals[fp.signal_index(fc.find_module("M").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle).unwrap()].layout.offset;
    let nprog = InterpProgram::encode_design(&np).unwrap();
    let fprog = InterpProgram::encode_design(&fp).unwrap();
    println!(
        "  code words: naive(x{insts}) {} -> folded(x1) {}  ({:.1}x smaller)",
        nprog.total_code_words(),
        fprog.total_code_words(),
        nprog.total_code_words() as f64 / fprog.total_code_words().max(1) as f64,
    );

    let run = |prog: &InterpProgram, din_off: usize, clk_off: usize, lanes: usize| -> f64 {
        let gpu = InterpGpuSimulator::new(prog, lanes).unwrap();
        gpu.set_signal(clk_off, &vec![1u32; lanes]);
        let din_v: Vec<u32> = (0..lanes as u32).map(|l| l.wrapping_mul(2654435761)).collect();
        for l in 0..limbs {
            let limb: Vec<u32> = din_v.iter().map(|&v| if l == 0 { v } else { 0 }).collect();
            gpu.set_signal(din_off + l, &limb);
        }
        gpu.tick_many(8);
        gpu.synchronize();
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t = Instant::now();
            gpu.tick_many(steps);
            gpu.synchronize();
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };

    // datapath-megacycles/sec = datapaths * lanes * steps / sec.
    println!("  base_lanes |  naive (NxL)        |  folded (1 x N*L)    |  speedup");
    for &l in &[64usize, 256, 1024, 4096, 16384] {
        let nt = run(&nprog, noff("din"), noff("clk"), l);
        let ft = run(&fprog, foff("din"), foff("clk"), insts * l);
        let work = (insts * l * steps) as f64 / 1.0e6; // identical total datapath-cycles
        let (nm, fm) = (work / nt, work / ft);
        println!("  {l:>9}  |  {nm:>7.1} Mdc/s        |  {fm:>7.1} Mdc/s        |  {:.2}x", fm / nm);
    }
}
