//! Diagnostic: how much static headroom does the IR-level specializer
//! (const-fold + algebraic identities + copy-prop + DCE) find in a real core's
//! machine program — the headroom clang -O3 cannot recover because it can't see
//! the RTL structure? Counts machine instrs and generated-C size before/after.
//! Build: cargo run --release --features aot -p rrtl-sim-ir --example aot_specialize_probe -- bench/sv/picorv32.v
fn main() {
    #[cfg(not(feature = "aot"))]
    println!("build with --features aot");
    #[cfg(feature = "aot")]
    run();
}

#[cfg(feature = "aot")]
fn run() {
    use rrtl_sim_ir::specialize::specialize_program;
    use rrtl_sim_ir::{aot, lower_to_machine_program, lower_to_packed_program, PackedMachineProgram};
    use rrtl_sv_frontend::import_sv;

    let path = std::env::args().nth(1).unwrap_or_else(|| "bench/sv/picorv32.v".into());
    let top = std::env::args().nth(2).unwrap_or_else(|| "picorv32".into());
    let src = std::fs::read_to_string(&path).expect("read top");
    let imported = import_sv(&src, Some(&top)).unwrap();
    let compiled = rrtl_core::compile(&imported.design).unwrap();
    let program = lower_to_packed_program(&compiled, &top).unwrap();
    let machine = lower_to_machine_program(&program);

    let instrs = |m: &PackedMachineProgram| -> usize {
        [&m.streams.async_reset_comb, &m.streams.comb, &m.streams.tick_next, &m.streams.tick_commit]
            .iter().map(|b| b.packets.iter().map(|p| p.instrs.len()).sum::<usize>()).sum()
    };

    let (spec, stats) = specialize_program(&machine);
    let (i0, i1) = (instrs(&machine), instrs(&spec));
    let c0 = aot::generate_c(&machine).unwrap();
    let c1 = aot::generate_c(&spec).unwrap();

    println!("IR-level specializer headroom on `{top}` ({path})");
    println!("  signals               : {}", program.signals.len());
    println!("  machine instrs        : {i0} → {i1}  ({:.1}% removed)", 100.0 * (i0 - i1) as f64 / i0 as f64);
    println!("  specializer stats     : {} instrs removed (folded/identity/copy/dead)", stats.instrs_removed());
    println!("  generated C size      : {} → {} bytes  ({:.1}% smaller)", c0.len(), c1.len(),
        100.0 * (c0.len() as f64 - c1.len() as f64) / c0.len() as f64);
    println!("  state-buffer signals  : {} (width-packed; clang cannot reorder/prune the buffer)", program.signals.len());
}
