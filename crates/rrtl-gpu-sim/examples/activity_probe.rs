//! Measures activity-based skip POTENTIAL at register-cone granularity, before
//! building the skipping machinery. A gated shift-chain models bursty/idle logic
//! (data held stable between enable pulses). Reports per-lane skip rate and the
//! GPU-relevant all-lanes-in-a-tile skip rate, for CORRELATED (same enable across
//! lanes) vs DECORRELATED (per-lane phase) stimulus — the regimes that decide
//! whether GPU tile-skipping can ever pay.
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example activity_probe -- [depth lanes cycles duty]
use rrtl_core::{compile, mux, uint, Design};
use rrtl_gpu_sim::interp::{InterpProgram, InterpRunner};
use rrtl_sim_ir::{lower_to_packed_program, register_support};

// Gated shift chain: when en==0 every register holds, so (en, din, regs) are all
// stable -> the whole design is idle and skippable. A free-running counter is
// always active.
fn build(depth: usize) -> Design {
    let mut design = Design::new();
    let mut m = design.module("Chain");
    let clk = m.input("clk", uint(1));
    let en = m.input("en", uint(8));
    let din = m.input("din", uint(8));
    // free-running counter (always active, never skippable)
    let ctr = m.reg("ctr", uint(8));
    m.clock(ctr, clk);
    m.next(ctr, ctr.value() + rrtl_core::lit_u(1, 8));
    let octr = m.output("octr", uint(8));
    m.assign(octr, ctr);
    // gated chain
    let mut prev = din;
    for i in 0..depth {
        let s = m.reg(format!("s{i}"), uint(8));
        m.clock(s, clk);
        // s <= en[0] ? prev : s    (enable held stable when idle)
        let en0 = en.value().slice(0, 1);
        m.next(s, mux(en0, prev.value(), s.value()));
        prev = s;
    }
    let o = m.output("o", uint(8));
    m.assign(o, prev);
    design
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (depth, lanes, cycles, duty) = (p(1, 8), p(2, 256), p(3, 200), p(4, 8));
    println!("depth={depth} lanes={lanes} cycles={cycles} duty=1/{duty} (enable on 1 of every {duty} cycles)");

    let design = build(depth);
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Chain").unwrap();
    let supports = register_support(&program);
    let off = |idx: usize| program.signals[idx].layout.offset;
    let limbs = |idx: usize| program.signals[idx].layout.limbs;
    let sig = |name: &str| -> usize { program.signals.iter().position(|s| s.name.ends_with(name)).unwrap() };
    let en_off = off(sig(".en"));
    let din_off = off(sig(".din"));
    let clk_off = off(sig(".clk"));

    // union of all support signals to snapshot each cycle
    let mut leaves: Vec<usize> = supports.iter().flat_map(|r| r.support.iter().copied()).collect();
    leaves.sort_unstable();
    leaves.dedup();

    for &correlated in &[true, false] {
        let mut runner = InterpRunner::new(InterpProgram::encode_design(&program).unwrap(), lanes);
        runner.set_signal(clk_off, &vec![1u32; lanes]);
        let mut prev_snap: Vec<Vec<u128>> = vec![Vec::new(); program.signals.len()];
        let (mut lane_skip, mut lane_tot) = (0u64, 0u64);
        let (mut tile_skip, mut tile_tot) = (0u64, 0u64);
        let mut lcg: u64 = 0x1234;
        let mut rng = || { lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1); lcg >> 32 };

        for cyc in 0..cycles {
            // enable: pulses 1-of-`duty`. correlated => same phase all lanes; decorrelated => per-lane phase.
            let en_v: Vec<u32> = (0..lanes as u32).map(|l| {
                let phase = if correlated { 0 } else { (l as usize) % duty };
                (((cyc + phase) % duty == 0) as u32)
            }).collect();
            // din changes only when enabled (held stable when idle).
            let din_v: Vec<u32> = (0..lanes).map(|l| if en_v[l] != 0 { (rng() & 0xff) as u32 } else {
                prev_snap.get(sig(".din")).and_then(|v| v.get(l)).map(|&x| x as u32).unwrap_or(0)
            }).collect();
            runner.set_signal(en_off, &en_v);
            runner.set_signal(din_off, &din_v);
            runner.tick();

            // snapshot leaf signals
            let mut snap: Vec<Vec<u128>> = vec![Vec::new(); program.signals.len()];
            for &idx in &leaves {
                snap[idx] = runner.get_signal_wide(off(idx), limbs(idx));
            }
            if cyc > 0 {
                // changed[idx][lane]
                let changed = |idx: usize, lane: usize| -> bool {
                    prev_snap[idx].get(lane) != snap[idx].get(lane)
                };
                for rs in &supports {
                    if rs.reads_memory { continue; }
                    // tile-level: skippable iff no support signal changed in ANY lane
                    let tile = !rs.support.iter().any(|&s| (0..lanes).any(|l| changed(s, l)));
                    tile_skip += tile as u64; tile_tot += 1;
                    // per-lane
                    for l in 0..lanes {
                        let sk = !rs.support.iter().any(|&s| changed(s, l));
                        lane_skip += sk as u64; lane_tot += 1;
                    }
                }
            }
            prev_snap = snap;
            // keep din snapshot for hold-when-idle even on cyc 0
            prev_snap[sig(".din")] = (0..lanes).map(|l| din_v[l] as u128).collect();
        }
        println!("  {:<12} per-lane skip {:>5.1}%   tile (all-lanes-idle) skip {:>5.1}%",
            if correlated {"correlated"} else {"decorrelated"},
            100.0 * lane_skip as f64 / lane_tot.max(1) as f64,
            100.0 * tile_skip as f64 / tile_tot.max(1) as f64);
    }
    println!("  (one register is a free-running counter: never skippable -> caps the rate)");
}
