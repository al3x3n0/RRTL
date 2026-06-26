//! Work-stealing on a batch with GENUINE data-driven divergence (§4g.14, made
//! real). Each instance is seeded with its own run length and HALTS when its
//! internal counter reaches 0 — so instances run different numbers of cycles by
//! data, not by an assigned cost. Execution is the fast baked tick in chunks of K,
//! checking the (register) counter between chunks to early-exit. We compare STATIC
//! chunking vs an atomic-counter WORK-STEALING schedule; bit-exact either way.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example divergent_batch
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
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::time::Instant;

    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv");
    let src = std::fs::read_to_string(format!("{base}/workload.sv")).unwrap();
    let imported = import_sv(&src, Some("workload")).unwrap();
    let compiled = rrtl_core::compile(&imported.design).unwrap();
    let program = lower_to_packed_program(&compiled, "workload").unwrap();
    let machine = lower_to_machine_program(&program);
    let idx = |n: &str| program.signals.iter().position(|s| s.name.ends_with(n)).unwrap();
    let (clk, load, seed_in, din, acc, counter) =
        (idx(".clk"), idx(".load"), idx(".seed"), idx(".din"), idx(".acc"), idx(".counter"));

    let jit = JitSimulator::compile(&machine).unwrap();
    let tick = jit.tick_many_fn_ptr();
    let slen = jit.state_words().len();
    let cores = rayon::current_num_threads();
    let m = cores * 64;

    // Per-instance run length: most short, a clustered front 1/8 ~250× longer, so
    // STATIC chunking dumps the long runs onto the first core(s).
    // run lengths ≫ K so the chunk-granular early-exit overshoots only marginally.
    let run_len = |i: usize| -> i64 {
        if i < m / 8 { 100_000 } else { 4_000 }
    };
    const K: i64 = 1024; // early-exit granularity

    // Advance one instance to its data-driven halt; returns cycles actually run.
    let work = move |p: *mut i64, i: usize| -> i64 {
        unsafe {
            for w in 0..slen {
                *p.add(w) = 0;
            }
            *p.add(clk * 2) = 1;
            *p.add(din * 2) = (i as i64).wrapping_mul(2_654_435_761) | 1;
            *p.add(load * 2) = 1;
            *p.add(seed_in * 2) = run_len(i);
            tick(p, 1); // latch the seed into counter
            *p.add(load * 2) = 0;
            let mut ran = 1i64;
            // run in fast baked chunks, stopping once the counter register hits 0
            while *p.add(counter * 2) != 0 {
                tick(p, K);
                ran += K;
            }
            ran
        }
    };

    let make_ptrs = |st: &mut [Vec<i64>]| -> Vec<usize> { st.iter_mut().map(|s| s.as_mut_ptr() as usize).collect() };
    let collect = |st: &[Vec<i64>]| -> Vec<i64> { st.iter().map(|s| s[acc * 2]).collect() };

    // ---- STATIC: fixed contiguous chunk per core ----
    let mut st_static: Vec<Vec<i64>> = (0..m).map(|_| vec![0i64; slen]).collect();
    let ptrs = make_ptrs(&mut st_static);
    let chunk = m.div_ceil(cores);
    let stat_cyc = AtomicU64::new(0);
    let t = Instant::now();
    std::thread::scope(|s| {
        for tid in 0..cores {
            let (ptrs, stat_cyc) = (&ptrs, &stat_cyc);
            s.spawn(move || {
                let mut c = 0u64;
                for i in (tid * chunk)..((tid + 1) * chunk).min(m) {
                    c += work(ptrs[i] as *mut i64, i) as u64;
                }
                stat_cyc.fetch_add(c, Ordering::Relaxed);
            });
        }
    });
    let static_secs = t.elapsed().as_secs_f64();
    let static_out = collect(&st_static);

    // ---- DYNAMIC: atomic-counter work-stealing ----
    let mut st_dyn: Vec<Vec<i64>> = (0..m).map(|_| vec![0i64; slen]).collect();
    let ptrs = make_ptrs(&mut st_dyn);
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
                work(ptrs[i] as *mut i64, i);
            });
        }
    });
    let dyn_secs = t.elapsed().as_secs_f64();
    let dyn_out = collect(&st_dyn);

    let total_cyc = stat_cyc.load(Ordering::Relaxed) as f64;
    let mc = |s: f64| total_cyc / s / 1e6;
    println!("divergent batch (data-driven halt) — workload, {m} instances ({cores} cores)");
    println!("  run lengths: {} short (4000 cyc) + {} long (100000 cyc, clustered front)\n", m - m / 8, m / 8);
    println!("  static (fixed chunks)   : {:.0} M cyc/s", mc(static_secs));
    println!("  dynamic (work-stealing) : {:.0} M cyc/s   =>  {:.2}x over static", mc(dyn_secs), static_secs / dyn_secs);
    println!("  bit-exact (static == dynamic per instance): {}", if static_out == dyn_out { "YES" } else { "NO" });
    println!("\n  => instances halt at data-dependent cycles; static strands the long runs on the first");
    println!("     core(s), work-stealing spreads them — same results, ~{:.1}x better utilization.", static_secs / dyn_secs);
}
