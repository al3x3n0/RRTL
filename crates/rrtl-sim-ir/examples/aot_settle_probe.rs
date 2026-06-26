//! Is the picorv32 AOT gap the per-cycle SETTLE, not codegen? The mem-bus harness
//! reads outputs every cycle → each cycle pays tick_many(1)'s leading comb PLUS a
//! fresh settle() (a 2nd full comb pass) triggered by get_signal. This probe
//! isolates the engine's raw tick rate (tick_many(N), no per-cycle observation)
//! from the observed rate. If raw >> observed, the redundant comb pass is the gap
//! and a settle-once tick model (clang can't do this — it's the model, not codegen)
//! is the lever. Also reports the comb-stream fraction of total tick work.
//! Build: cargo run --release --features aot -p rrtl-sim-ir --example aot_settle_probe -- bench/sv/picorv32.v
fn main() {
    #[cfg(not(feature = "aot"))]
    println!("build with --features aot");
    #[cfg(feature = "aot")]
    run();
}

#[cfg(feature = "aot")]
fn run() {
    use rrtl_sim_ir::{aot::AotSimulator, lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;
    use std::time::Instant;

    let path = std::env::args().nth(1).unwrap_or_else(|| "bench/sv/picorv32.v".into());
    let top = std::env::args().nth(2).unwrap_or_else(|| "picorv32".into());
    let n: usize = std::env::args().nth(3).and_then(|a| a.parse().ok()).unwrap_or(2_000_000);
    let src = std::fs::read_to_string(&path).expect("read top");
    let imported = import_sv(&src, Some(&top)).unwrap();
    let compiled = rrtl_core::compile(&imported.design).unwrap();
    let program = lower_to_packed_program(&compiled, &top).unwrap();
    let machine = lower_to_machine_program(&program);

    let blk = |b: &rrtl_sim_ir::PackedBlock| b.packets.iter().map(|p| p.instrs.len()).sum::<usize>();
    let comb = blk(&machine.streams.async_reset_comb) + blk(&machine.streams.comb);
    let nxt = blk(&machine.streams.tick_next) + blk(&machine.streams.tick_commit);
    println!("`{top}` stream sizes: comb {comb} instrs, tick_next+commit {nxt} instrs");
    println!("  comb is {:.0}% of one tick's instrs → a redundant settle ≈ that much extra work",
        100.0 * comb as f64 / (comb + nxt) as f64);

    let mut sim = AotSimulator::compile(&machine).expect("aot compile");
    let idx = |name: &str| {
        let h = compiled.find_module(&top).unwrap().signals.iter().find(|s| s.name == name).unwrap().handle;
        program.signal_index(h).unwrap()
    };
    // Hardwire a benign input set so the core free-runs; we are measuring TICK RATE,
    // not correctness (mem responses are not serviced here).
    sim.set_signal(idx("clk"), 1);
    sim.set_signal(idx("resetn"), 1);
    sim.set_signal(idx("mem_ready"), 0);
    sim.set_signal(idx("mem_rdata"), 0);

    // (A) raw tick_many(N): ONE comb per cycle, no observation settle.
    sim.tick_many(50_000); // warm
    let t = Instant::now();
    sim.tick_many(n);
    let raw = n as f64 / t.elapsed().as_secs_f64() / 1e6;

    // (B) per-cycle observed: tick(); then read 4 outputs (first read triggers a
    // full settle) — mirrors the mem-bus harness's comb cost exactly.
    let (o_v, o_a, o_w, o_d) = (idx("mem_valid"), idx("mem_addr"), idx("mem_wstrb"), idx("mem_wdata"));
    let mut acc = 0u64;
    let t = Instant::now();
    for _ in 0..n {
        sim.tick();
        acc ^= sim.get_signal(o_v) ^ sim.get_signal(o_a) ^ sim.get_signal(o_w) ^ sim.get_signal(o_d);
    }
    let obs = n as f64 / t.elapsed().as_secs_f64() / 1e6;

    println!("  raw tick_many   : {raw:.2} Mcyc/s  (1 comb/cycle)");
    println!("  per-cycle observ: {obs:.2} Mcyc/s  (1 comb + 1 settle/cycle) [acc={}]", acc & 1);
    println!("  settle overhead : {:.2}x slowdown from the redundant comb pass", raw / obs);
}
