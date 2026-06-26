//! Comb-fusion on the SIMD CPU batch engine: `tick()` runs the combinational
//! stream twice per cycle (comb→capture/commit→comb); within `tick_many` the
//! trailing comb is redundant with the next leading comb, so the fused `tick_many`
//! evaluates comb once per cycle. This A/Bs the naive per-`tick()` loop against the
//! fused `tick_many` on a comb-heavy batch design and checks bit-exactness. The win
//! scales with the comb fraction of a tick (≈2x for comb-dominated designs).
//! Build: cargo run --release -p rrtl-sim-ir --example comb_fusion_check -- [depth] [lanes] [cycles]
use rrtl_sim_ir::{lower_to_packed_program, SimdCpuSimulator};
use rrtl_sv_frontend::import_sv;
use std::time::Instant;

fn gen_sv(depth: usize) -> String {
    // One deep combinational pipeline of `depth` wires reduced into one register:
    // comb-dominated, so halving comb is most visible.
    let consts = ["9e3779b1", "85ebca77", "c2b2ae35", "27d4eb2f", "165667b1", "ff51afd7"];
    let mut s = String::from("module Comb(input clk, input [31:0] din, output [31:0] o);\n");
    s.push_str("  wire [31:0] ");
    s.push_str(&(0..depth).map(|j| format!("w{j}")).collect::<Vec<_>>().join(","));
    s.push_str(";\n  reg [31:0] acc;\n");
    s.push_str(&format!("  assign w0 = (din ^ 32'd1) * 32'h{} + 32'd1;\n", consts[0]));
    for j in 1..depth {
        let c = consts[j % consts.len()];
        s.push_str(&format!(
            "  assign w{j} = (w{} ^ (w{}<<{}) ^ (w{}>>{})) * 32'h{c} + w{};\n",
            j - 1, j - 1, 3 + (j % 7), j - 1, 2 + (j % 5), j - 1
        ));
    }
    s.push_str(&format!("  always @(posedge clk) acc <= acc + w{};\n", depth - 1));
    s.push_str("  assign o = acc;\nendmodule\n");
    s
}

fn main() {
    let depth: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(48);
    let lanes: usize = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(256);
    let cycles: usize = std::env::args().nth(3).and_then(|a| a.parse().ok()).unwrap_or(20_000);

    let src = gen_sv(depth);
    let imported = import_sv(&src, Some("Comb")).unwrap();
    let compiled = rrtl_core::compile(&imported.design).unwrap();
    let program = lower_to_packed_program(&compiled, "Comb").unwrap();
    let sig = |n: &str| {
        compiled.find_module("Comb").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle
    };
    let (din, o) = (sig("din"), sig("o"));
    let dvals: Vec<u128> = (0..lanes).map(|l| (l as u128).wrapping_mul(0x9e3779b1) & 0xffff_ffff).collect();

    // naive = loop tick() (unchanged: 2 combs/cycle); fused = tick_many (1/cycle).
    let run = |fused: bool| -> (f64, Vec<u128>) {
        let mut sim = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        sim.set_signal(din, &dvals).unwrap();
        sim.tick_many(1000).unwrap(); // warm
        let t = Instant::now();
        if fused {
            sim.tick_many(cycles).unwrap();
        } else {
            for _ in 0..cycles {
                sim.tick().unwrap();
            }
        }
        let secs = t.elapsed().as_secs_f64();
        let mlc = (cycles * lanes) as f64 / secs / 1e6;
        (mlc, sim.get_signal(o).unwrap())
    };

    // interleaved best-of-3 (shared thermal profile)
    let (mut naive, mut fused, mut hn, mut hf) = (0f64, 0f64, vec![], vec![]);
    for _ in 0..3 {
        let a = run(false);
        let b = run(true);
        naive = naive.max(a.0);
        fused = fused.max(b.0);
        hn = a.1;
        hf = b.1;
    }
    println!("comb-fusion on SIMD CPU batch — depth {depth}, {lanes} lanes, {cycles} cycles");
    println!("  naive (tick loop) : {naive:.1} M-lane-cyc/s");
    println!("  fused (tick_many) : {fused:.1} M-lane-cyc/s   ({:.2}x)", fused / naive);
    println!("  bit-exact (acc per lane): {}", if hn == hf { "YES" } else { "NO — MISMATCH" });
    assert_eq!(hn, hf, "fused diverged from naive");
}
