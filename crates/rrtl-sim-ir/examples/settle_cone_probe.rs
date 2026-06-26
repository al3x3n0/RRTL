//! How small is the cone the observation settle ACTUALLY needs? The mem-bus harness
//! reads only the output ports each cycle, but settle() recomputes the WHOLE comb.
//! Observability-slicing the settle (cone of the observed outputs) should shrink it
//! to near-nothing for a core whose outputs are mostly registered. This probe
//! measures the comb-instr count of the full design vs the output cone.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example settle_cone_probe -- bench/sv/picorv32.v picorv32
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::{cone_of_influence, lower_to_machine_program, lower_to_packed_program, slice_present, PackedMachineProgram, PackedSignalKind};
    use rrtl_sv_frontend::import_sv;

    let path = std::env::args().nth(1).unwrap_or_else(|| "bench/sv/picorv32.v".into());
    let top = std::env::args().nth(2).unwrap_or_else(|| "picorv32".into());
    let src = std::fs::read_to_string(&path).expect("read top");
    let imported = import_sv(&src, Some(&top)).unwrap();
    let compiled = rrtl_core::compile(&imported.design).unwrap();
    let program = lower_to_packed_program(&compiled, &top).unwrap();
    let full = lower_to_machine_program(&program);

    let comb_instrs = |m: &PackedMachineProgram| -> usize {
        m.streams.async_reset_comb.packets.iter().map(|p| p.instrs.len()).sum::<usize>()
            + m.streams.comb.packets.iter().map(|p| p.instrs.len()).sum::<usize>()
    };

    // Observed set = every OUTPUT-kind signal (what a testbench reads).
    let outputs: Vec<usize> = program.signals.iter().enumerate()
        .filter(|(_, s)| s.kind == PackedSignalKind::Output)
        .map(|(i, _)| i).collect();
    println!("`{top}`: {} signals, {} outputs", program.signals.len(), outputs.len());

    let (sig_mask, mem_mask) = cone_of_influence(&program, &outputs, &[]);
    let slice = slice_present(&program, &sig_mask, &mem_mask).unwrap();
    let sliced = lower_to_machine_program(&slice.program);

    let (cf, cs) = (comb_instrs(&full), comb_instrs(&sliced));
    println!("  full comb instrs        : {cf}");
    println!("  output-cone comb instrs : {cs}  ({:.0}% of full)", 100.0 * cs as f64 / cf.max(1) as f64);
    println!("  => a settle restricted to the output cone does ~{:.0}% of the work;", 100.0 * cs as f64 / cf.max(1) as f64);
    println!("     the redundant full settle (1.51x slowdown measured) collapses toward free.");
}
