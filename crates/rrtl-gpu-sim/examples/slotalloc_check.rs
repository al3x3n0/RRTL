//! Isolates liveness-based value-slot allocation. Slot allocation reuses value
//! ids across non-overlapping lifetimes, shrinking the value workspace. It does
//! NOT compose with multiply-add fusion (fusion assumes SSA ids), so to measure
//! it cleanly we compare fusion-off ± slot-allocation (both correct), and also
//! show the fusion baseline for reference.
//!
//! Combos (all bit-exact-checked against the fusion baseline):
//!   A  fuse=on,  slotalloc=off   (shipped default)
//!   B  fuse=off, slotalloc=off
//!   C  fuse=off, slotalloc=on    (isolates slot allocation vs B)
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example slotalloc_check -- [bits width depth lanes steps]

use std::time::Instant;

use rrtl_core::{compile, lit_u, uint, Design, Signal};
use rrtl_gpu_sim::interp::{InterpGpuSimulator, InterpProgram};
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program, slot_allocate_program, specialize_program};

fn build(bits: u32, width: usize, depth: usize) -> Design {
    let c: u128 = if bits >= 64 { 0x9e37_79b9_7f4a_7c15 } else { 0x9e37_79b9 };
    let mut design = Design::new();
    let mut m = design.module("Wide");
    let clk = m.input("clk", uint(1));
    let din = m.input("din", uint(bits));
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
    let limbs = ((bits + 31) / 32) as usize;
    println!("bits={bits} width={width} depth={depth} lanes={lanes} steps={steps}");

    let design = build(bits, width, depth);
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Wide").unwrap();
    let sig = |name: &str| -> Signal {
        compiled.find_module("Wide").and_then(|mm| mm.signals.iter().find(|s| s.name == name)).map(|s| s.handle).unwrap()
    };
    let off = |s: Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;
    let (din, clk, o0) = (sig("din"), sig("clk"), sig("o0"));

    let machine = lower_to_machine_program(&program);
    let (spec, _) = specialize_program(&machine);
    let (alloc, slot_stats) = slot_allocate_program(&spec);

    let a = InterpProgram::encode_opts(&spec, true).unwrap();
    let b = InterpProgram::encode_opts(&spec, false).unwrap();
    let c_enc = InterpProgram::encode_opts(&alloc, false).unwrap();
    println!(
        "  value words/lane: spec={} slotalloc={}  ({} -> {} live slots, {:.0}% smaller)",
        b.total_value_words,
        c_enc.total_value_words,
        slot_stats.slots_before,
        slot_stats.slots_after,
        100.0 * (b.total_value_words.saturating_sub(c_enc.total_value_words)) as f64 / b.total_value_words.max(1) as f64,
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

    let (ta, oa) = run(&a);
    let (tb, ob) = run(&b);
    let (tc, oc) = run(&c_enc);
    let ok = oa == ob && oa == oc;
    println!("  A fuse        {:>6.2} Mlc/s", mlc(ta));
    println!("  B plain       {:>6.2} Mlc/s", mlc(tb));
    println!("  C slotalloc   {:>6.2} Mlc/s   (C/B {:.2}x)   [{}]", mlc(tc), tb / tc, if ok { "OK" } else { "MISMATCH" });
}
