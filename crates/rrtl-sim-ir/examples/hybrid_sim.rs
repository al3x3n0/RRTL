//! Hybrid simulation of a MIXED design: offload the GF(2)-linear register cones
//! to the matrix XOR-AOT ([`LinearAot`]) and run the non-linear cones on a
//! general engine (SIMD CPU), then glue the disjoint state back together. On
//! mixed.sv the linear CRC block is ~62% of the sequential bits; the multiply-
//! accumulate and compare-gated counter are not linear and stay general.
//!
//! The non-linear engine runs an OBSERVABILITY-SLICED program (observe acc+count)
//! so the CRC cone is pruned from it — otherwise the general engine would still
//! pay for the work the matrix AOT is meant to replace. Bit-exact vs the full
//! SIMD CPU oracle; throughput is full-general vs (linear-AOT + sliced-general).
//! Build: cargo run --release --features "aot jit" -p rrtl-sim-ir --example hybrid_sim -- [lanes steps]
use rrtl_sim_ir::linear_aot::LinearAot;
use rrtl_sim_ir::{
    cone_of_influence, lower_to_machine_program, lower_to_packed_program, slice_present, SimdCpuSimulator,
};
use rrtl_sv_frontend::import_sv;
use std::time::Instant;

fn count_ops(p: &rrtl_sim_ir::PackedProgram) -> usize {
    [&p.streams.async_reset_comb, &p.streams.comb, &p.streams.tick_next, &p.streams.tick_commit]
        .iter()
        .flat_map(|s| s.iter())
        .map(|pkt| pkt.ops.len())
        .sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lanes: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20000);

    let src = std::fs::read_to_string("bench/sv/mixed.sv").expect("read mixed.sv");
    let imported = import_sv(&src, Some("mixed")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "mixed").expect("lower");
    let h = |n: &str| compiled.find_module("mixed").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
    let pos = |n: &str| program.signals.iter().position(|s| s.name == n || s.name.ends_with(&format!(".{n}"))).unwrap();

    // Linear part: the CRC cone(s) → matrix XOR-AOT.
    let mut lin = LinearAot::compile(&program, lanes).expect("linear cones");
    let crc_dst = *lin
        .linear_signals()
        .iter()
        .find(|&&d| program.signals[d].name.rsplit('.').next().unwrap().starts_with("crc"))
        .expect("crc cone");
    println!(
        "mixed: linear offload = {:?}",
        lin.linear_signals().iter().map(|&d| program.signals[d].name.clone()).collect::<Vec<_>>()
    );

    // Non-linear part: observability-slice to acc+count → the CRC cone is pruned.
    let (present_sig, present_mem) = cone_of_influence(&program, &[pos("acc"), pos("count")], &[]);
    let nl_program = slice_present(&program, &present_sig, &present_mem).expect("slice").program;
    println!(
        "  full program: {} signals / {} ops   nonlinear slice: {} signals / {} ops (CRC pruned)",
        program.signals.len(),
        count_ops(&program),
        nl_program.signals.len(),
        count_ops(&nl_program),
    );

    let mut full = SimdCpuSimulator::new(program.clone(), lanes).unwrap(); // oracle + general baseline
    let mut nl = SimdCpuSimulator::new(nl_program.clone(), lanes).unwrap();
    full.set_signal(h("clk"), &vec![1u128; lanes]).unwrap();
    let _ = nl.set_signal(h("clk"), &vec![1u128; lanes]);

    let inval = |seed: u64, lane: usize, m: u128| ((seed.wrapping_mul(2654435761).wrapping_add(lane as u64)) as u128) & m;

    // Bit-exact: crc from the linear AOT, acc+count from the sliced general engine.
    let mut mism = 0usize;
    for cyc in 0..40u64 {
        let drv: Vec<(&str, u128)> = vec![("rst", 1), ("din", 0xff), ("a", 0xffff), ("b", 0xffff)];
        for (name, m) in &drv {
            let vals: Vec<u128> = (0..lanes)
                .map(|l| if *name == "rst" { (cyc < 1) as u128 } else { inval(cyc + name.len() as u64, l, *m) })
                .collect();
            full.set_signal(h(name), &vals).unwrap();
            let _ = nl.set_signal(h(name), &vals); // a/b/rst present in the slice; din pruned
            if *name == "din" || *name == "rst" {
                for (l, v) in vals.iter().enumerate() {
                    lin.set_signal(pos(name), l, *v);
                }
            }
        }
        full.tick().unwrap();
        nl.tick().unwrap();
        lin.tick();
        let (fc, fa, fco) = (
            full.get_signal(h("crc")).unwrap(),
            full.get_signal(h("acc")).unwrap(),
            full.get_signal(h("count")).unwrap(),
        );
        let (na, nco) = (nl.get_signal(h("acc")).unwrap(), nl.get_signal(h("count")).unwrap());
        for l in 0..lanes {
            mism += (lin.get_signal(crc_dst, l) != fc[l]) as usize;
            mism += (na[l] != fa[l]) as usize;
            mism += (nco[l] != fco[l]) as usize;
        }
    }
    println!("  bit-exact vs full SIMD CPU (crc,acc,count × {lanes} lanes × 40 cyc): {}", if mism == 0 { "YES" } else { "NO" });
    assert_eq!(mism, 0, "hybrid diverged");

    // Throughput: full general engine vs (linear AOT + sliced general engine).
    let mlc = |s: f64| (lanes * steps) as f64 / s / 1e6;
    let t = Instant::now();
    full.tick_many(steps).unwrap();
    let full_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    lin.tick_many(steps);
    let lin_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    nl.tick_many(steps).unwrap();
    let nl_s = t.elapsed().as_secs_f64();
    let hybrid_s = lin_s + nl_s; // sequential; the two engines could also run concurrently
    println!("  full general (SIMD CPU)     : {:.1} M-lane-cyc/s", mlc(full_s));
    println!("  hybrid: linear-AOT + sliced : {:.1} M-lane-cyc/s  ({:.2}x)  [linear {:.1} + nonlinear {:.1} M/s]", mlc(hybrid_s), full_s / hybrid_s, mlc(lin_s), mlc(nl_s));

    // Compiled baseline: the honest comparison is vs the vector JIT (not the
    // interpreter). full vector-JIT vs (linear-AOT + sliced vector-JIT).
    #[cfg(feature = "jit")]
    {
        use rrtl_sim_ir::jit::SimdJitSimulator;
        let machine_full = lower_to_machine_program(&program);
        let machine_nl = lower_to_machine_program(&nl_program);
        match (
            SimdJitSimulator::compile_lanes(&machine_full, lanes),
            SimdJitSimulator::compile_lanes(&machine_nl, lanes),
        ) {
            (Ok(mut jf), Ok(mut jn)) => {
                let mut lj = LinearAot::compile(&program, lanes).unwrap();
                let fi = |n: &str| program.signal_index(h(n)).unwrap();
                let ni = |n: &str| nl_program.signal_index(h(n)).unwrap();
                // self-consistent bit-exact: hybrid-JIT == full-JIT (full-JIT == oracle
                // is established by Phase A + the JIT being an independently-validated engine).
                let mut jmis = 0usize;
                for cyc in 0..20u64 {
                    for (name, m) in [("rst", 1u128), ("din", 0xff), ("a", 0xffff), ("b", 0xffff)] {
                        for l in 0..lanes {
                            let v = if name == "rst" { (cyc < 1) as u128 } else { inval(cyc + name.len() as u64, l, m) };
                            jf.set_signal(l, fi(name), v as u32);
                            if name == "a" || name == "b" || name == "rst" {
                                jn.set_signal(l, ni(name), v as u32);
                            }
                            if name == "din" || name == "rst" {
                                lj.set_signal(pos(name), l, v);
                            }
                        }
                    }
                    jf.tick();
                    jn.tick();
                    lj.tick();
                    for l in 0..lanes {
                        jmis += (lj.get_signal(crc_dst, l) as u32 != jf.get_signal(l, fi("crc"))) as usize;
                        jmis += (jn.get_signal(l, ni("acc")) != jf.get_signal(l, fi("acc"))) as usize;
                        jmis += (jn.get_signal(l, ni("count")) != jf.get_signal(l, fi("count"))) as usize;
                    }
                }
                println!("  [compiled] hybrid-JIT == full-JIT (crc,acc,count × {lanes} × 20 cyc): {}", if jmis == 0 { "YES" } else { "NO" });
                assert_eq!(jmis, 0, "hybrid-JIT diverged from full-JIT");

                // Thermally-robust timing: warm up, then INTERLEAVE the three
                // measurements each trial (so they share thermal state) and take
                // best-of-N (the least-throttled sample). REAL concurrency: linear
                // on a worker thread (LinearAot is Send), non-linear JIT on main
                // (it never crosses threads); disjoint state, no synchronization.
                jf.tick_many(steps);
                lj.tick_many(steps);
                jn.tick_many(steps); // warmup
                let (mut bf, mut bseq, mut bconc) = (f64::MAX, f64::MAX, f64::MAX);
                // ROTATE the order each trial so no measurement is systematically
                // last (= warmest); best-of-N over rotated positions.
                macro_rules! t_full { () => {{ let t = Instant::now(); jf.tick_many(steps); bf = bf.min(t.elapsed().as_secs_f64()); }}; }
                macro_rules! t_seq { () => {{ let t = Instant::now(); lj.tick_many(steps); jn.tick_many(steps); bseq = bseq.min(t.elapsed().as_secs_f64()); }}; }
                macro_rules! t_conc { () => {{ let t = Instant::now(); std::thread::scope(|sc| { sc.spawn(|| lj.tick_many(steps)); jn.tick_many(steps); }); bconc = bconc.min(t.elapsed().as_secs_f64()); }}; }
                for trial in 0..9 {
                    match trial % 3 {
                        0 => { t_full!(); t_seq!(); t_conc!(); }
                        1 => { t_conc!(); t_full!(); t_seq!(); }
                        _ => { t_seq!(); t_conc!(); t_full!(); }
                    }
                }
                println!("  full general (vector JIT)   : {:.1} M-lane-cyc/s", mlc(bf));
                println!("  hybrid seq  (1 thread)      : {:.1} M-lane-cyc/s  ({:.2}x vs full)", mlc(bseq), bf / bseq);
                println!("  hybrid conc (2 threads)     : {:.1} M-lane-cyc/s  ({:.2}x vs full, {:.2}x vs seq)  [best-of-9, rotated order]", mlc(bconc), bf / bconc, bseq / bconc);
            }
            _ => println!("  [compiled] vector JIT cannot compile this design (skipped)"),
        }
    }

    // The auto-partitioning HybridSimulator: one unified engine that detects the
    // linear cones, auto-picks the best general backend for the rest, and runs
    // them concurrently. Validate it against the full SIMD CPU and time it.
    {
        use rrtl_sim_ir::hybrid::HybridSimulator;
        let mut hyb = HybridSimulator::new(&program, lanes).unwrap();
        let mut oracle = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        oracle.set_signal(h("clk"), &vec![1u128; lanes]).unwrap();
        hyb.set_signal(h("clk"), &vec![1u128; lanes]).unwrap();
        let mut mism = 0usize;
        for cyc in 0..40u64 {
            for (name, m) in [("rst", 1u128), ("din", 0xff), ("a", 0xffff), ("b", 0xffff)] {
                let vals: Vec<u128> = (0..lanes)
                    .map(|l| if name == "rst" { (cyc < 1) as u128 } else { inval(cyc + name.len() as u64, l, m) })
                    .collect();
                oracle.set_signal(h(name), &vals).unwrap();
                hyb.set_signal(h(name), &vals).unwrap();
            }
            oracle.tick().unwrap();
            hyb.tick().unwrap();
            for out in ["crc", "acc", "count"] {
                let (o, hv) = (oracle.get_signal(h(out)).unwrap(), hyb.get_signal(h(out)).unwrap());
                for l in 0..lanes {
                    mism += (o[l] != hv[l]) as usize;
                }
            }
        }
        let t = Instant::now();
        hyb.tick_many(steps).unwrap();
        let s = t.elapsed().as_secs_f64();
        println!(
            "  HybridSimulator: linear=matrix-AOT + general={} (auto), concurrent; bit-exact: {}; {:.1} M-lane-cyc/s",
            hyb.general_backend(),
            if mism == 0 { "YES" } else { "NO" },
            mlc(s),
        );
        assert_eq!(mism, 0, "HybridSimulator diverged");
    }
}
