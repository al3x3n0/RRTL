//! Measures the AOT design-specializer (constant folding + algebraic identities
//! + DCE) on a datapath whose per-stage logic is partly constant-foldable —
//! standing in for the tie-off / config logic that pervades real SoCs. Reports
//! instruction-count reduction and GPU-interp throughput, original vs specialized,
//! and checks the two produce identical outputs.
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example specialize_check -- [bits width depth lanes steps]

use std::time::Instant;

use rrtl_core::{compile, lit_u, uint, Design, Signal};
use rrtl_gpu_sim::interp::{InterpGpuSimulator, InterpProgram};
use rrtl_sim_ir::lower_to_packed_program;

/// Each stage computes `(prev & ALLONES) * 1 + (A + B) + din`, which the
/// specializer collapses to `prev + (A+B folded) + din`. The `& ALLONES` and
/// `* 1` are identities; `A + B` folds to a constant.
fn build(bits: u32, width: usize, depth: usize) -> Design {
    let allones: u128 = if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 };
    let mut design = Design::new();
    let mut m = design.module("Cfg");
    let clk = m.input("clk", uint(1));
    let din = m.input("din", uint(bits));
    for lane in 0..width {
        let acc = m.reg(format!("acc{lane}"), uint(bits));
        m.clock(acc, clk);
        let mut prev = acc;
        for stage in 0..depth {
            let w = m.wire(format!("w{lane}_{stage}"), uint(bits));
            // identities (& allones, * 1) + constant fold (A + B)
            let masked = prev & lit_u(allones, bits);
            let scaled = masked * lit_u(1, bits);
            let konst = lit_u(0x55 + stage as u128, bits) + lit_u(0xAA, bits);
            m.assign(w, scaled + konst + din);
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
    let limbs = ((bits + 31) / 32) as usize;
    println!("bits={bits} width={width} depth={depth} lanes={lanes} steps={steps}");

    let design = build(bits, width, depth);
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Cfg").unwrap();
    let sig = |name: &str| -> Signal {
        compiled
            .find_module("Cfg")
            .and_then(|mm| mm.signals.iter().find(|s| s.name == name))
            .map(|s| s.handle)
            .unwrap()
    };
    let off = |s: Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;
    let (din, clk, o0) = (sig("din"), sig("clk"), sig("o0"));

    let orig = InterpProgram::encode_design(&program).unwrap();
    let (spec, stats) = InterpProgram::encode_design_specialized(&program).unwrap();
    println!(
        "  instrs: {} -> {}  ({} folded, {} copies, {} dead, {:.1}% removed)",
        stats.instrs_before,
        stats.instrs_after,
        stats.folded,
        stats.copies,
        stats.dead,
        100.0 * stats.instrs_removed() as f64 / stats.instrs_before.max(1) as f64,
    );

    let din_v: Vec<u32> = (0..lanes as u32).map(|l| l.wrapping_mul(2654435761)).collect();
    let mlc = |secs: f64| (lanes * steps) as f64 / secs / 1.0e6;

    let run = |prog: &InterpProgram| -> (f64, Vec<u32>) {
        let gpu = InterpGpuSimulator::new(prog, lanes).unwrap();
        gpu.set_signal(off(clk), &vec![1u32; lanes]);
        for l in 0..limbs {
            let limb: Vec<u32> = din_v.iter().map(|&v| if l == 0 { v } else { 0 }).collect();
            gpu.set_signal(off(din) + l, &limb);
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
        (best, gpu.get_signal(off(o0)))
    };

    let (orig_t, orig_o) = run(&orig);
    let (spec_t, spec_o) = run(&spec);
    let ok = orig_o == spec_o;
    println!(
        "  GPU orig {:>7.1} Mlc/s   spec {:>7.1} Mlc/s   ({:.2}x)   [{}]",
        mlc(orig_t),
        mlc(spec_t),
        orig_t / spec_t,
        if ok { "OK" } else { "MISMATCH" },
    );
}
