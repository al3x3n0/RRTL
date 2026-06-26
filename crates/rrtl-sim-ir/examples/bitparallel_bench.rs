//! Validate + benchmark the bit-parallel gate-level batch engine: a NAND/NOR/XNOR
//! netlist (all 1-bit) run on `BitParallelSimulator` (64 lanes/u64) vs the SIMD
//! CPU batch engine, bit-exact, with a lane-cycles/s comparison.
//! Build: cargo run --release -p rrtl-sim-ir --example bitparallel_bench -- [depth cones lanes steps]
use rrtl_sim_ir::bitparallel::BitParallelSimulator;
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program, SimdCpuSimulator};
use rrtl_core::{compile, uint, Design, Signal};
use std::time::Instant;

fn build(depth: usize, cones: usize) -> Design {
    let mut design = Design::new();
    {
        let mut m = design.module("Gates");
        let clk = m.input("clk", uint(1));
        let a = m.input("a", uint(1));
        let b = m.input("b", uint(1));
        for c in 0..cones {
            let acc = m.reg(format!("g{c}"), uint(1));
            m.clock(acc, clk);
            let mut x = acc.value();
            for d in 0..depth {
                x = match (c + d) % 3 {
                    0 => !(x & a.value()),
                    1 => !(x | b.value()),
                    _ => !(x ^ a.value()),
                };
            }
            m.next(acc, x);
            let o = m.output(format!("o{c}"), uint(1));
            m.assign(o, acc);
        }
    }
    design
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (depth, cones, lanes, steps) = (p(1, 16), p(2, 64), p(3, 1024), p(4, 20000));

    let design = build(depth, cones);
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Gates").unwrap();
    let machine = lower_to_machine_program(&program);
    let sig = |n: &str| -> Signal {
        compiled.find_module("Gates").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle
    };
    let idx = |n: &str| program.signal_index(sig(n)).unwrap();
    let (a, b) = (idx("a"), idx("b"));
    // bit-parallel reads by packed index; the SIMD CPU reads by Signal handle.
    let out_idx: Vec<usize> = (0..cones).map(|c| idx(&format!("o{c}"))).collect();
    let out_sig: Vec<Signal> = (0..cones).map(|c| sig(&format!("o{c}"))).collect();

    // Per-lane stimulus: a = lane&1, b = (lane>>1)&1 (distinct per lane).
    let mut bp = BitParallelSimulator::new(&machine, lanes).expect("bit-parallel applies");
    let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
    cpu.set_signal(sig("clk"), &vec![1u128; lanes]).unwrap();
    for lane in 0..lanes {
        bp.set_signal(a, lane, lane & 1 == 1);
        bp.set_signal(b, lane, (lane >> 1) & 1 == 1);
    }
    cpu.set_signal(sig("a"), &(0..lanes).map(|l| (l & 1) as u128).collect::<Vec<_>>()).unwrap();
    cpu.set_signal(sig("b"), &(0..lanes).map(|l| ((l >> 1) & 1) as u128).collect::<Vec<_>>()).unwrap();

    // Warm + bit-exactness check (a few cycles), comparing every output on every lane.
    for _ in 0..7 {
        bp.tick();
        cpu.tick().unwrap();
    }
    let mut mismatches = 0usize;
    for (oi, os) in out_idx.iter().zip(&out_sig) {
        let cv = cpu.get_signal(*os).unwrap();
        for lane in 0..lanes {
            if bp.get_signal(*oi, lane) != (cv[lane] & 1 == 1) {
                mismatches += 1;
            }
        }
    }
    println!("bit-parallel gate-level engine (depth={depth} cones={cones} lanes={lanes})");
    println!("  signals: {}, words/signal: {}", bp.signal_count(), lanes.div_ceil(64));
    println!("  bit-exact vs SIMD CPU (all {} outputs × {lanes} lanes): {}",
        out_idx.len(), if mismatches == 0 { "YES".into() } else { format!("NO ({mismatches} mismatches)") });

    // Throughput: lane-cycles/s for each engine.
    let t = Instant::now();
    bp.tick_many(steps);
    let bp_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    cpu.tick_many(steps).unwrap();
    let cpu_s = t.elapsed().as_secs_f64();
    let mlc = |s: f64| (lanes * steps) as f64 / s / 1e6;
    println!("  bit-parallel    : {:.1} M-lane-cyc/s", mlc(bp_s));
    println!("  SIMD CPU interp : {:.1} M-lane-cyc/s  ({:.1}x)", mlc(cpu_s), cpu_s / bp_s);

    // The fairest "best existing engine" baseline: the COMPILED vector JIT batch,
    // and the new bit-parallel JIT (the payoff).
    #[cfg(feature = "jit")]
    {
        use rrtl_sim_ir::jit::{BitParallelJitSimulator, SimdJitSimulator};
        // Bit-parallel JIT: validate bit-exact vs the interpreter, then time it.
        let mut bpj = BitParallelJitSimulator::compile_lanes(&machine, lanes).expect("bp jit");
        for lane in 0..lanes {
            bpj.set_signal(lane, a, lane & 1 == 1);
            bpj.set_signal(lane, b, (lane >> 1) & 1 == 1);
        }
        for _ in 0..7 {
            bpj.tick_many(1);
        }
        let mut jm = 0usize;
        for oi in &out_idx {
            for lane in 0..lanes {
                if bpj.get_signal(lane, *oi) != bp.get_signal(*oi, lane) {
                    jm += 1;
                }
            }
        }
        let t = Instant::now();
        bpj.tick_many(steps);
        let bpj_s = t.elapsed().as_secs_f64();
        println!("  bit-parallel JIT: {:.1} M-lane-cyc/s  ({:.1}x)  [bit-exact: {}]",
            mlc(bpj_s), bp_s / bpj_s, if jm == 0 { "YES" } else { "NO" });

        #[cfg(feature = "aot")]
        {
            use rrtl_sim_ir::bitparallel::BitParallelAot;
            let mut bpa = BitParallelAot::compile_lanes(&machine, lanes).expect("bp aot");
            for lane in 0..lanes {
                bpa.set_signal(lane, a, lane & 1 == 1);
                bpa.set_signal(lane, b, (lane >> 1) & 1 == 1);
            }
            for _ in 0..7 {
                bpa.tick_many(1);
            }
            let mut am = 0usize;
            for oi in &out_idx {
                for lane in 0..lanes {
                    if bpa.get_signal(lane, *oi) != bp.get_signal(*oi, lane) {
                        am += 1;
                    }
                }
            }
            let t = Instant::now();
            bpa.tick_many(steps);
            let bpa_s = t.elapsed().as_secs_f64();
            println!("  bit-parallel AOT: {:.1} M-lane-cyc/s  ({:.1}x)  [bit-exact: {}]",
                mlc(bpa_s), bp_s / bpa_s, if am == 0 { "YES" } else { "NO" });

            // Multicore (rayon over independent groups).
            let mut bpm = BitParallelAot::compile_lanes(&machine, lanes).expect("bp aot");
            for lane in 0..lanes {
                bpm.set_signal(lane, a, lane & 1 == 1);
                bpm.set_signal(lane, b, (lane >> 1) & 1 == 1);
            }
            for _ in 0..7 { bpm.tick_many_parallel(1); }
            let mut pm = 0usize;
            for oi in &out_idx {
                for lane in 0..lanes {
                    if bpm.get_signal(lane, *oi) != bp.get_signal(*oi, lane) { pm += 1; }
                }
            }
            let t = Instant::now();
            bpm.tick_many_parallel(steps);
            let bpm_s = t.elapsed().as_secs_f64();
            println!("  bit-parallel AOT ×{} cores: {:.1} M-lane-cyc/s  ({:.1}x serial AOT)  [bit-exact: {}]",
                rayon::current_num_threads(), mlc(bpm_s), bpa_s / bpm_s, if pm == 0 { "YES" } else { "NO" });
        }

        if let Ok(mut vjit) = SimdJitSimulator::compile_lanes(&machine, lanes) {
            for lane in 0..lanes {
                vjit.set_signal(lane, a, (lane & 1) as u32);
                vjit.set_signal(lane, b, ((lane >> 1) & 1) as u32);
            }
            for _ in 0..7 {
                vjit.tick_many(1);
            }
            let t = Instant::now();
            vjit.tick_many(steps);
            let v_s = t.elapsed().as_secs_f64();
            println!("  vector JIT      : {:.1} M-lane-cyc/s  ({:.1}x)", mlc(v_s), v_s / bp_s);
        }
    }
    assert_eq!(mismatches, 0, "bit-parallel diverged from SIMD CPU");
}
