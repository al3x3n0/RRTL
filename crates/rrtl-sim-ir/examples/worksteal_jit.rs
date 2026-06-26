//! Work-stealing for the bulk-synchronous batch (§4g.13). When instances diverge
//! in cost — some run far longer than others (early-exit, a triggered slow path,
//! variable program length) — STATIC partitioning (fixed contiguous chunks) load-
//! imbalances: whichever core gets the heavy instances gates the whole batch while
//! the others idle. DYNAMIC dispatch — a shared atomic "next instance" counter,
//! the simplest form of work-stealing — hands each core the next pending instance
//! as soon as it's free, so the heavy ones spread out and all cores finish together.
//!
//! Same one-compiled-design / disjoint-per-instance-state setup as batch_jit; the
//! only change is HOW instances are assigned to cores. Bit-exact either way.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example worksteal_jit
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::jit::JitSimulator;
    use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv");
    let src = std::fs::read_to_string(format!("{base}/cfgdsp.sv")).unwrap();
    let imported = import_sv(&src, Some("cfgdsp")).unwrap();
    let compiled = rrtl_core::compile(&imported.design).unwrap();
    let program = lower_to_packed_program(&compiled, "cfgdsp").unwrap();
    let machine = lower_to_machine_program(&program);
    let idx = |n: &str| program.signals.iter().position(|s| s.name.ends_with(n)).unwrap();
    let (clk, cfg_we, cfg_in, a, b, acc) =
        (idx(".clk"), idx(".cfg_we"), idx(".cfg_in"), idx(".a"), idx(".b"), idx(".acc"));

    let jit = JitSimulator::compile(&machine).unwrap();
    let tick = jit.tick_many_fn_ptr();
    let slen = jit.state_words().len();
    let cores = rayon::current_num_threads();
    let m = cores * 64;
    // SKEWED + CLUSTERED cost: the first 1/8 of instances run 32× longer, and they
    // sit at the front so static chunking dumps them onto the first core(s).
    let base_cycles = 3_000i64;
    let cost = |i: usize| if i < m / 8 { base_cycles * 32 } else { base_cycles };
    let total_cycles: f64 = (0..m).map(|i| cost(i) as f64).sum();

    let seed = |p: *mut i64, i: usize| unsafe {
        for w in 0..slen {
            *p.add(w) = 0;
        }
        *p.add(clk * 2) = 1;
        *p.add(cfg_we * 2) = 1;
        *p.add(cfg_in * 2) = (i % 4) as i64;
        *p.add(a * 2) = (i as i64).wrapping_mul(2_654_435_761) & 0xffff_ffff;
        *p.add(b * 2) = (i as i64).wrapping_mul(40_503).wrapping_add(7) & 0xffff_ffff;
    };

    let make_states = || -> (Vec<Vec<i64>>, Vec<usize>) {
        let mut st: Vec<Vec<i64>> = (0..m).map(|_| vec![0i64; slen]).collect();
        let ptrs: Vec<usize> = st.iter_mut().map(|s| s.as_mut_ptr() as usize).collect();
        (st, ptrs)
    };
    let collect = |st: &[Vec<i64>]| -> Vec<i64> { st.iter().map(|s| s[acc * 2]).collect() };

    // ---- STATIC: fixed contiguous chunk per core (no rebalancing) ----
    let (st_static, ptrs) = make_states();
    let chunk = m.div_ceil(cores);
    let t = Instant::now();
    std::thread::scope(|s| {
        for tid in 0..cores {
            let ptrs = &ptrs;
            s.spawn(move || {
                for i in (tid * chunk)..((tid + 1) * chunk).min(m) {
                    let p = ptrs[i] as *mut i64;
                    seed(p, i);
                    tick(p, cost(i));
                }
            });
        }
    });
    let static_secs = t.elapsed().as_secs_f64();
    let static_out = collect(&st_static);

    // ---- DYNAMIC: shared atomic counter, grab the next instance when free ----
    let (st_dyn, ptrs) = make_states();
    let next = AtomicUsize::new(0);
    let t = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..cores {
            let (ptrs, next) = (&ptrs, &next);
            s.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= m {
                    break;
                }
                let p = ptrs[i] as *mut i64;
                seed(p, i);
                tick(p, cost(i));
            });
        }
    });
    let dyn_secs = t.elapsed().as_secs_f64();
    let dyn_out = collect(&st_dyn);

    let mc = |s: f64| total_cycles / s / 1e6;
    let ideal = total_cycles / cores as f64; // perfect-balance cycle count per core
    println!("work-stealing batch — cfgdsp, {m} instances ({cores} cores), 1/8 run 32× longer (clustered)");
    println!("  ideal per-core work = {:.0} M cycles; static gives core 0 ~{:.0} M (the heavy chunk)\n",
        ideal / 1e6, (m / 8) as f64 * (base_cycles * 32) as f64 / 1e6);
    println!("  static (fixed chunks)   : {:.0} M cyc/s", mc(static_secs));
    println!("  dynamic (work-stealing) : {:.0} M cyc/s   =>  {:.2}x over static", mc(dyn_secs), static_secs / dyn_secs);
    println!("  bit-exact (static == dynamic per instance): {}", if static_out == dyn_out { "YES" } else { "NO" });
    println!("\n  => per-instance dynamic dispatch spreads the heavy instances across all cores; static");
    println!("     chunking strands them on one core. Same results, ~{:.1}x better core utilization.",
        static_secs / dyn_secs);
}
