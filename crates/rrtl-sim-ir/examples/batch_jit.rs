//! Bulk-synchronous batch on the scalar JIT — the way past the per-cycle barrier
//! wall (§4g.12). Instead of parallelizing ONE design's partitions and syncing
//! every cycle, run MANY independent instances (different stimulus/seeds) and let
//! each advance its full run on a core with NO per-cycle synchronization. The work
//! between syncs becomes N cycles × M instances, so the sync cost (one join)
//! vanishes and throughput scales ~linearly with cores.
//!
//! One compiled design, M per-instance state buffers; `par_iter_mut` hands each
//! thread disjoint `&mut` buffers (no unsafe, no shared state). Unlike the SIMD
//! batch path, this is scalar so it tolerates control-flow divergence between
//! instances — the regression/fuzzing throughput regime.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example batch_jit
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use rayon::prelude::*;
    use rrtl_sim_ir::jit::JitSimulator;
    use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;
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

    // One compiled design; the tick_many code is reentrant over its state pointer.
    let jit = JitSimulator::compile(&machine).unwrap();
    let tick = jit.tick_many_fn_ptr();
    let slen = jit.state_words().len();

    let cores = rayon::current_num_threads();
    let instances = cores * 64; // many independent instances per core
    let cycles = 200_000i64;

    // per-instance seed → a distinct, deterministic run (held inputs).
    let seed = |st: &mut [i64], i: usize| {
        st.iter_mut().for_each(|w| *w = 0);
        st[clk * 2] = 1;
        st[cfg_we * 2] = 1;
        st[cfg_in * 2] = (i % 4) as i64; // mode 0..3
        st[a * 2] = (i as i64).wrapping_mul(2_654_435_761) & 0xffff_ffff;
        st[b * 2] = (i as i64).wrapping_mul(40_503).wrapping_add(7) & 0xffff_ffff;
    };

    // ---- single-core (serial over instances) ----
    let mut states: Vec<Vec<i64>> = (0..instances).map(|_| vec![0i64; slen]).collect();
    let t = Instant::now();
    for (i, st) in states.iter_mut().enumerate() {
        seed(st, i);
        tick(st.as_mut_ptr(), cycles);
    }
    let serial_secs = t.elapsed().as_secs_f64();
    let serial_out: Vec<i64> = states.iter().map(|st| st[acc * 2]).collect();

    // ---- multi-core (rayon par_iter_mut: each thread owns disjoint buffers) ----
    let mut states2: Vec<Vec<i64>> = (0..instances).map(|_| vec![0i64; slen]).collect();
    let t = Instant::now();
    states2.par_iter_mut().enumerate().for_each(|(i, st)| {
        seed(st, i);
        tick(st.as_mut_ptr(), cycles);
    });
    let par_secs = t.elapsed().as_secs_f64();
    let par_out: Vec<i64> = states2.iter().map(|st| st[acc * 2]).collect();

    let total = instances as f64 * cycles as f64;
    let bit_exact = serial_out == par_out;
    println!("bulk-synchronous batch on the scalar JIT — cfgdsp, {instances} instances × {cycles} cycles");
    println!("rayon threads (cores): {cores}\n");
    println!("  single-core : {:.0} M inst-cyc/s", total / serial_secs / 1e6);
    println!("  {cores}-core    : {:.0} M inst-cyc/s   =>  {:.2}x over 1 core (~linear)",
        total / par_secs / 1e6, serial_secs / par_secs);
    println!("  bit-exact (serial == parallel per instance): {}", if bit_exact { "YES" } else { "NO" });
    println!("  (note: a >cores speedup reflects M3 P/E-core heterogeneity — the serial baseline is not");
    println!("   pinned to a fast P-core; the takeaway is at-or-near-linear scaling, not super-linearity.)");
    println!("\n  => batch over independent instances syncs ONCE (at join), not per cycle — the work");
    println!("     between syncs is {cycles}×{instances} cycles, so multi-core scales ~linearly.");
}
