//! Single-instance latency: native JIT vs the SIMD interpreter (1 lane) on a deep
//! 32-bit mul-add datapath. Build with `--features jit`.
//! Usage: cargo run --release --features jit -p rrtl-sim-ir --example jit_bench -- [width depth ticks]
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use std::time::Instant;
    use rrtl_core::{compile, lit_u, uint, Design, Signal};
    use rrtl_sim_ir::{jit::{JitSimulator, SimdJitSimulator}, lower_to_machine_program, lower_to_packed_program, SimBackend, SimdCpuSimulator, SingleLaneMachineSimulator};

    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (width, depth, ticks) = (p(1, 16), p(2, 8), p(3, 2_000_000));
    println!("width={width} depth={depth} ticks={ticks}");

    let mut design = Design::new();
    {
        let mut m = design.module("Wide");
        let clk = m.input("clk", uint(1));
        let din = m.input("din", uint(32));
        for lane in 0..width {
            let acc = m.reg(format!("acc{lane}"), uint(32));
            m.clock(acc, clk);
            let mut prev = acc;
            for s in 0..depth {
                let w = m.wire(format!("w{lane}_{s}"), uint(32));
                m.assign(w, prev * lit_u(0x9e37_79b9, 32) + din);
                prev = w;
            }
            m.next(acc, prev.value() + acc.value());
            let o = m.output(format!("o{lane}"), uint(32));
            m.assign(o, acc);
        }
    }
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Wide").unwrap();
    let machine = lower_to_machine_program(&program);
    let h = |n: &str| -> Signal { compiled.find_module("Wide").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle };
    let idx = |n: &str| program.signal_index(h(n)).unwrap();

    let mut jit = JitSimulator::compile(&machine).expect("compile");
    let mut cpu = SimdCpuSimulator::new(program.clone(), 1).unwrap();

    // correctness sanity (10 cycles)
    let mut ok = true;
    for c in 0..10u32 {
        let dv = (c.wrapping_mul(2654435761)) as u64;
        jit.set_signal(idx("clk"), 1); jit.set_signal(idx("din"), dv);
        cpu.set_signal(h("clk"), &[1]).unwrap(); cpu.set_signal(h("din"), &[dv as u128]).unwrap();
        jit.tick(); cpu.tick().unwrap();
        if jit.get_signal(idx("o0")) != cpu.get_signal(h("o0")).unwrap()[0] as u64 { ok = false; }
    }
    println!("  correctness (jit vs simd-cpu/1): [{}]", if ok {"OK"} else {"MISMATCH"});

    jit.set_signal(idx("clk"), 1); jit.set_signal(idx("din"), 12345);
    cpu.set_signal(h("clk"), &[1]).unwrap(); cpu.set_signal(h("din"), &[12345]).unwrap();

    let time = |label: &str, f: &mut dyn FnMut()| -> f64 {
        for _ in 0..1000 { f(); } // warm
        let mut best = f64::INFINITY;
        for _ in 0..3 { let t = Instant::now(); for _ in 0..ticks { f(); } best = best.min(t.elapsed().as_secs_f64()); }
        let mhz = ticks as f64 / best / 1e6;
        println!("  {label:<14} {:>7.1} ms   {:>7.2} M-cycles/s", best * 1e3, mhz);
        mhz
    };
    let mut single = SingleLaneMachineSimulator::new(program.clone()).unwrap();
    single.set_signals_replicated(&[(h("clk"), 1), (h("din"), 12345)]).unwrap();
    let jm = time("JIT (native)", &mut || jit.tick());
    let mut bm = f64::INFINITY;
    for _ in 0..3 { let t = Instant::now(); jit.tick_many(ticks); bm = bm.min(t.elapsed().as_secs_f64()); }
    let mm = ticks as f64 / bm / 1e6;
    println!("  tick_many loop {:>7.1} ms   {:>7.2} M-cycles/s   ({:.2}x vs per-call tick)", bm * 1e3, mm, mm / jm);
    let sm = time("scalar-interp", &mut || { single.tick(); });
    let cm = time("SIMD-cpu/1", &mut || { cpu.tick().unwrap(); });
    println!("  speedup vs scalar-interp: {:.2}x   vs SIMD/1: {:.2}x", jm / sm, jm / cm);

    // --- vector JIT: native code + 4-lane parallelism, in lane-cycles/s ---
    if let Ok(mut vjit) = SimdJitSimulator::compile(&machine) {
        let lanes = vjit.lanes();
        for l in 0..lanes { vjit.set_signal(l, idx("clk"), 1); vjit.set_signal(l, idx("din"), 12345 + l as u32); }
        let mut best = f64::INFINITY;
        for _ in 0..3 { let t = Instant::now(); vjit.tick_many(ticks); best = best.min(t.elapsed().as_secs_f64()); }
        let vmhz = ticks as f64 / best / 1e6;             // vector passes / s
        let vlane = vmhz * lanes as f64;                  // lane-cycles / s
        let mut cpu4 = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        cpu4.set_signal(h("clk"), &vec![1u128; lanes]).unwrap();
        cpu4.set_signal(h("din"), &vec![12345u128; lanes]).unwrap();
        let mut bc = f64::INFINITY;
        for _ in 0..3 { let t = Instant::now(); for _ in 0..ticks { cpu4.tick().unwrap(); } bc = bc.min(t.elapsed().as_secs_f64()); }
        let cpu4_lane = ticks as f64 / bc / 1e6 * lanes as f64;
        println!("  --- throughput (lane-cycles/s) ---");
        println!("  scalar JIT ×1 lane     {:>7.2} M-lane-cyc/s", jm);
        println!("  vector JIT ×{lanes} lanes    {:>7.2} M-lane-cyc/s   ({:.2}x vs scalar JIT, {:.2}x/lane)", vlane, vlane / jm, vmhz / jm);
        println!("  SIMD-interp ×{lanes} lanes   {:>7.2} M-lane-cyc/s   ({:.2}x vs vector JIT)", cpu4_lane, vlane / cpu4_lane);
    }

    // --- baked harness vs manual per-cycle set/tick/get loop ---
    let in_idx = jit.input_indices().to_vec();
    let out_idx = jit.output_indices().to_vec();
    let in_names: Vec<String> = jit.input_ports().to_vec();
    let (nin, nout) = (in_idx.len(), out_idx.len());
    let hc = 300_000usize;
    let mut inbuf = vec![0u64; hc * nin];
    for c in 0..hc {
        for (k, name) in in_names.iter().enumerate() {
            let local = name.rsplit('.').next().unwrap();
            inbuf[c * nin + k] = if local == "clk" { 1 } else { (c as u64).wrapping_mul(2654435761) & 0xffff_ffff };
        }
    }
    let mut best_trace = f64::INFINITY;
    for _ in 0..3 { let t = Instant::now(); let _ = jit.run_trace(&inbuf, hc); best_trace = best_trace.min(t.elapsed().as_secs_f64()); }
    let mut best_manual = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        for c in 0..hc {
            for k in 0..nin { jit.set_signal(in_idx[k], inbuf[c * nin + k]); }
            jit.tick();
            for k in 0..nout { std::hint::black_box(jit.get_signal(out_idx[k])); }
        }
        best_manual = best_manual.min(t.elapsed().as_secs_f64());
    }
    println!("  harness ({hc} cyc, {nin} in/{nout} out):");
    println!("    manual set/tick/get   {:>7.1} ms   {:>6.2} M-cycles/s", best_manual * 1e3, hc as f64 / best_manual / 1e6);
    println!("    baked run_trace       {:>7.1} ms   {:>6.2} M-cycles/s   ({:.2}x)", best_trace * 1e3, hc as f64 / best_trace / 1e6, best_manual / best_trace);
}
