//! Quantify the AOT's settle→capture FUSION structurally (throttle-robust): the
//! fused tick_many keeps combinational values in C registers and feeds register
//! next-states directly, skipping the per-cycle comb→state store traffic that the
//! un-fused path (AOT_NOFUSE) round-trips. We count the state-store statements in
//! the emitted tick_many for both, on picorv32.
//! Build: cargo run --release --features aot -p rrtl-sim-ir --example aot_fusion_probe -- [design top]
use rrtl_sim_ir::aot::generate_c;
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
use rrtl_sv_frontend::import_sv;

// (state-store statements, total lines) inside the tick_many function body.
fn tick_many_stats(c: &str) -> (usize, usize) {
    let body = c
        .split("void tick_many")
        .nth(1)
        .and_then(|s| s.split("\nvoid ").next())
        .unwrap_or("");
    let lines: Vec<&str> = body.lines().collect();
    let stores = lines
        .iter()
        .filter(|l| l.find(" = ").map_or(false, |eq| l[..eq].contains("(st+")))
        .count();
    (stores, lines.len())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "bench/sv/picorv32.v".into());
    let top = args.get(2).cloned().unwrap_or_else(|| "picorv32".into());
    let src = std::fs::read_to_string(&path).expect("read design");
    let imported = import_sv(&src, Some(&top)).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, &top).expect("lower");
    let machine = lower_to_machine_program(&program);

    std::env::remove_var("AOT_NOFUSE");
    let fused = generate_c(&machine).unwrap();
    std::env::set_var("AOT_NOFUSE", "1");
    let nofuse = generate_c(&machine).unwrap();

    let (fs, fl) = tick_many_stats(&fused);
    let (ns, nl) = tick_many_stats(&nofuse);
    println!("[{top}] AOT tick_many state-stores/cycle:");
    println!("  un-fused (AOT_NOFUSE): {ns} stores, {nl} lines");
    println!("  fused (default):       {fs} stores, {fl} lines");
    println!(
        "  fusion removes {} comb state-stores/cycle ({:.0}% fewer stores, {:.0}% smaller body) — kept in C registers",
        ns.saturating_sub(fs),
        100.0 * (ns.saturating_sub(fs)) as f64 / ns.max(1) as f64,
        100.0 * (nl.saturating_sub(fl)) as f64 / nl.max(1) as f64,
    );
}
