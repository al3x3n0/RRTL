//! First-round measurement of register-cone partitioned simulation vs the
//! whole-design packed simulator.
//!
//! Builds `W` independent accumulator lanes, each with a combinational chain of
//! depth `D` feeding its register. The lanes share no combinational logic, so a
//! register-cone partition into `K` groups is perfectly parallel (all groups at
//! topological level 0). Both simulators use the same per-group `PackedSimulator`
//! engine, so the comparison isolates the partitioning + threading effect.
//!
//! Usage: cargo run --release --example partition_bench -p rrtl-sim-ir -- [W D lanes cycles]

use std::time::Instant;

use rrtl_core::{compile, lit_u, uint, Design};
use rrtl_sim_ir::{
    lower_to_packed_program, PackedSimulator, PartitionedSimulator, SimdCpuSimulator,
};

fn build_wide_design(width: usize, depth: usize) -> Design {
    let mut design = Design::new();
    let mut m = design.module("Wide");
    let clk = m.input("clk", uint(1));
    let din = m.input("din", uint(32));
    for lane in 0..width {
        let acc = m.reg(format!("acc{lane}"), uint(32));
        m.clock(acc, clk);
        // Combinational chain of `depth` named wires feeding this register.
        let mut prev = acc;
        for stage in 0..depth {
            let w = m.wire(format!("w{lane}_{stage}"), uint(32));
            m.assign(w, prev * lit_u(0x9e37_79b9, 32) + din);
            prev = w;
        }
        m.next(acc, prev + acc);
        let o = m.output(format!("o{lane}"), uint(32));
        m.assign(o, acc);
    }
    design
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let parse = |i: usize, default: usize| {
        args.get(i)
            .and_then(|a| a.parse().ok())
            .unwrap_or(default)
    };
    let width = parse(1, 256);
    let depth = parse(2, 12);
    let lanes = parse(3, 16);
    let cycles = parse(4, 400);

    println!(
        "Wide design: width={width} lanes(stimulus)={lanes} comb_depth={depth} cycles={cycles}"
    );

    let design = build_wide_design(width, depth);
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Wide").unwrap();
    let din = compiled
        .find_module("Wide")
        .and_then(|m| m.signals.iter().find(|s| s.name == "din"))
        .map(|s| s.handle)
        .unwrap();
    let o0 = compiled
        .find_module("Wide")
        .and_then(|m| m.signals.iter().find(|s| s.name == "o0"))
        .map(|s| s.handle)
        .unwrap();
    println!(
        "packed program: {} signals, {} comb packets\n",
        program.signals.len(),
        program.streams.comb.len()
    );

    let din_lanes: Vec<u128> = (0..lanes as u128)
        .map(|l| l.wrapping_mul(2_654_435_761) & 0xffff_ffff)
        .collect();
    let clk_lanes = vec![1u128; lanes];
    let clk = clk_handle(&program);

    // Best-of-N timing with a warm-up run and cooldown between reps, to blunt
    // thermal throttling and scheduler/background noise on a laptop.
    let reps = 2;
    let bench = |make: &dyn Fn() -> (f64, Vec<u128>)| -> (f64, Vec<u128>) {
        let _ = make(); // warm caches / pool
        let mut best = f64::INFINITY;
        let mut check = Vec::new();
        for _ in 0..reps {
            std::thread::sleep(std::time::Duration::from_millis(150));
            let (secs, out) = make();
            best = best.min(secs);
            check = out;
        }
        (best, check)
    };

    // Scalar whole-design baseline (tree-walking packed simulator).
    let run_scalar = || {
        let mut sim = PackedSimulator::new(program.clone(), lanes).unwrap();
        sim.set_signal(din, &din_lanes).unwrap();
        sim.set_signal(clk, &clk_lanes).unwrap();
        let start = Instant::now();
        sim.tick_many(cycles);
        (start.elapsed().as_secs_f64(), sim.get_signal(o0).unwrap())
    };
    let (scalar, base_check) = bench(&run_scalar);
    println!(
        "whole-design scalar     : {:.3} s   {:>9.0} cyc/s   1.00x",
        scalar,
        cycles as f64 / scalar
    );

    // SIMD whole-design baseline (lane-vectorized, single-threaded). This is the
    // fair single-thread reference for the partitioned (SIMD) runs.
    let run_simd = || {
        let mut sim = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        sim.set_signal(din, &din_lanes).unwrap();
        sim.set_signal(clk, &clk_lanes).unwrap();
        let start = Instant::now();
        sim.tick_many(cycles).unwrap();
        (start.elapsed().as_secs_f64(), sim.get_signal(o0).unwrap())
    };
    let (base, simd_check) = bench(&run_simd);
    assert_eq!(simd_check, base_check, "SIMD whole-design mismatch");
    println!(
        "whole-design SIMD       : {:.3} s   {:>9.0} cyc/s   {:.2}x vs scalar  (partition baseline)",
        base,
        cycles as f64 / base,
        scalar / base,
    );

    for groups in [1usize, 2, 4, 8] {
        for (tag, parallel) in [("serial  ", false), ("parallel", true)] {
            let run = || {
                let mut sim =
                    PartitionedSimulator::new_register_balanced(&program, groups, lanes).unwrap();
                sim.set_parallel(parallel);
                sim.set_signal(din, &din_lanes).unwrap();
                sim.set_signal(clk, &clk_lanes).unwrap();
                let start = Instant::now();
                sim.tick_many(cycles);
                (start.elapsed().as_secs_f64(), sim.get_signal(o0).unwrap())
            };
            let (secs, check) = bench(&run);
            let ok = if check == base_check { "ok" } else { "MISMATCH" };
            println!(
                "partitioned K={groups:<2} {tag}: {:.3} s   {:>9.0} cyc/s   {:.2}x  [{}]",
                secs,
                cycles as f64 / secs,
                base / secs,
                ok,
            );
        }
    }
}

fn clk_handle(program: &rrtl_sim_ir::PackedProgram) -> rrtl_ir::Signal {
    // clk is the sole 1-bit top input; find it by name.
    program
        .signals
        .iter()
        .find(|s| s.name.ends_with(".clk") || s.name == "clk")
        .and_then(|s| s.source)
        .expect("clk signal handle")
}
