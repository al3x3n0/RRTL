//! Validate + measure the automatic fused-superoperator (macro-op) pass:
//! `fuse_macros` rewrites maximal single-use simple-op cones into OP_MACRO
//! records (intermediates kept in local registers, not the value buffer). Runs
//! the unfused and macro-fused programs on the CPU InterpRunner with identical
//! random stimulus, asserts bit-exact, and reports the record reduction (the
//! GPU/interp value-traffic metric) + CPU-interp throughput.
//! Build: cargo run --release -p rrtl-gpu-sim --example fused_macro_check -- [v] [top] [lanes] [cycles]
use rrtl_gpu_sim::interp::{fuse_macros, InterpGpuSimulator, InterpProgram, InterpRunner, RECORD_WORDS};
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
use rrtl_sv_frontend::import_sv;
use rrtl_ir::SignalKind;
use std::time::Instant;

fn records(p: &InterpProgram) -> usize {
    p.blocks.iter().map(|b| b.len() / RECORD_WORDS).sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "bench/sv/picorv32.v".into());
    let top = args.get(2).cloned().unwrap_or_else(|| "picorv32".into());
    let lanes: usize = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(64);
    let cycles: usize = args.get(4).and_then(|a| a.parse().ok()).unwrap_or(200000);

    let src = std::fs::read_to_string(&path).expect("read");
    let imported = import_sv(&src, Some(&top)).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, &top).expect("lower");
    let machine = lower_to_machine_program(&program);

    let base = InterpProgram::encode_opts(&machine, false).expect("encode"); // menu OFF
    let macro_p = fuse_macros(&base);

    let module = compiled.find_module(&top).unwrap();
    let off = |name: &str| -> Option<usize> {
        let h = module.signals.iter().find(|s| s.name == name)?.handle;
        program.signal_index(h).map(|i| program.signals[i].layout.offset)
    };
    let ports = |kind: SignalKind| -> Vec<(usize, u32)> {
        module
            .signals
            .iter()
            .filter(|s| s.kind == kind)
            .filter_map(|s| program.signal_index(s.handle).map(|i| {
                (program.signals[i].layout.offset, program.signals[i].layout.width)
            }))
            .collect()
    };
    let inputs = ports(SignalKind::Input);
    let outputs = ports(SignalKind::Output);

    println!("{top}: records unfused {} → macro-fused {} ({:.1}% fewer); {} inputs, {} outputs, {lanes} lanes",
        records(&base), records(&macro_p),
        100.0 * (records(&base) - records(&macro_p)) as f64 / records(&base).max(1) as f64,
        inputs.len(), outputs.len());

    // CPU InterpRunner only does a quick low-lane bit-exact check (its throughput
    // is already characterized + neutral; the GPU is the real throughput test).
    let cl = lanes.min(64);
    let mut a = InterpRunner::new(base.clone(), cl);
    let mut b = InterpRunner::new(macro_p.clone(), cl);
    if let Some(clk) = off("clk") {
        a.set_signal(clk, &vec![1u32; cl]);
        b.set_signal(clk, &vec![1u32; cl]);
    }
    let mut lcg = 0x2545_f491u32;
    let mut mism = 0usize;
    for _ in 0..64 {
        for &(o, w) in &inputs {
            let vals: Vec<u32> = (0..cl)
                .map(|_| {
                    lcg = lcg.wrapping_mul(1664525).wrapping_add(1013904223);
                    let m = if w >= 32 { u32::MAX } else { (1u32 << w) - 1 };
                    lcg & m
                })
                .collect();
            a.set_signal(o, &vals);
            b.set_signal(o, &vals);
        }
        a.tick();
        b.tick();
        for &(o, _) in &outputs {
            if a.get_signal(o) != b.get_signal(o) {
                mism += 1;
            }
        }
    }
    let verdict = if mism == 0 { "YES".to_string() } else { format!("NO ({mism})") };
    println!("  CPU bit-exact (macro-fused == unfused, {cl} lanes × 64 cyc): {verdict}");
    let mlc = |s: f64| (lanes * cycles) as f64 / s / 1e6;
    assert_eq!(mism, 0, "macro-fused diverged from unfused");

    // ---- GPU: the real test (value buffer = global memory; fused records =
    // global-memory writes avoided). ----
    let (ga, gb) = match (InterpGpuSimulator::new(&base, lanes), InterpGpuSimulator::new(&macro_p, lanes)) {
        (Ok(a), Ok(b)) => (a, b),
        _ => {
            println!("  GPU: unavailable — CPU-only");
            return;
        }
    };
    // Same random inputs on both; drive a few cycles and check outputs bit-exact.
    let mut lcg2 = 0x9e37_79b9u32;
    let mut gmism = 0usize;
    if let Some(clk) = off("clk") {
        ga.set_signal(clk, &vec![1u32; lanes]);
        gb.set_signal(clk, &vec![1u32; lanes]);
    }
    for _ in 0..16 {
        for &(o, w) in &inputs {
            let vals: Vec<u32> = (0..lanes).map(|_| {
                lcg2 = lcg2.wrapping_mul(1664525).wrapping_add(1013904223);
                let m = if w >= 32 { u32::MAX } else { (1u32 << w) - 1 };
                lcg2 & m
            }).collect();
            ga.set_signal(o, &vals);
            gb.set_signal(o, &vals);
        }
        ga.tick_many(1);
        gb.tick_many(1);
    }
    ga.synchronize();
    gb.synchronize();
    for &(o, _) in &outputs {
        if ga.get_signal(o) != gb.get_signal(o) {
            gmism += 1;
        }
    }
    println!("  GPU bit-exact (macro-fused == unfused): {}", if gmism == 0 { "YES" } else { "NO" });
    if gmism != 0 {
        // run_block_ml (multi-limb) has no OP_MACRO case yet → hits default (0),
        // so the throughput below would be a fake speedup. Skip it.
        println!("  (multi-limb design: WGSL run_block_ml lacks OP_MACRO — throughput invalid, skipped)");
        return;
    }
    // Throughput (chunked tick_many to stay under the watchdog).
    let bench_gpu = |g: &InterpGpuSimulator| -> f64 {
        g.synchronize();
        let t = Instant::now();
        let mut done = 0;
        while done < cycles {
            let n = 256.min(cycles - done);
            g.tick_many(n);
            g.synchronize();
            done += n;
        }
        t.elapsed().as_secs_f64()
    };
    let ga_s = bench_gpu(&ga);
    let gb_s = bench_gpu(&gb);
    println!("  GPU interp: unfused {:.1} → macro-fused {:.1} M-lane-cyc/s ({:.2}x)", mlc(ga_s), mlc(gb_s), ga_s / gb_s);
}
