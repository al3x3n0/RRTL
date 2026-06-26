//! Smoke test: import YosysHQ picorv32 (a full production RISC-V core) via the
//! SV frontend, lower it through the whole pipeline (compile -> packed ->
//! machine -> Cranelift JIT), and tick it with a trivial always-ready memory
//! feeding `addi x0,x0,0` (NOP) so the core fetches/executes without faulting.
//! Confirms the end-to-end path works on a real design. Not a correctness oracle.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example picorv32_smoke -- [path] [N]
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::{jit::JitSimulator, lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("/tmp/picorv32.v");
    let n: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(200_000);

    let src = std::fs::read_to_string(path).expect("read picorv32.v");
    let imported = import_sv(&src, Some("picorv32")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "picorv32").expect("lower packed");
    let machine = lower_to_machine_program(&program);
    println!(
        "imported picorv32: top={} signals={} state_words={}",
        imported.top_name,
        program.signals.len(),
        program.total_signal_words
    );

    let sig = |name: &str| {
        compiled
            .find_module("picorv32")
            .unwrap()
            .signals
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("no signal {name}"))
            .handle
    };
    let idx = |name: &str| program.signal_index(sig(name)).unwrap();

    let mut jit = JitSimulator::compile(&machine).expect("jit compile");

    let (i_clk, i_rst) = (idx("clk"), idx("resetn"));
    let (i_mem_ready, i_mem_rdata) = (idx("mem_ready"), idx("mem_rdata"));
    let o_mem_valid = idx("mem_valid");
    let o_trap = idx("trap");

    // Reset low for a few cycles, then run with an always-ready memory that
    // returns a NOP (addi x0,x0,0 = 0x00000013) for every fetch.
    let nop: u64 = 0x0000_0013;
    let start = Instant::now();
    let mut traps = 0u64;
    let mut fetches = 0u64;
    for c in 0..n {
        let resetn = (c >= 4) as u64;
        // memory is combinationally ready and returns NOP whenever requested
        let mem_valid = jit.get_signal(o_mem_valid);
        jit.set_signal(i_clk, 1);
        jit.set_signal(i_rst, resetn);
        jit.set_signal(i_mem_ready, mem_valid);
        jit.set_signal(i_mem_rdata, nop);
        jit.tick();
        if mem_valid != 0 {
            fetches += 1;
        }
        if jit.get_signal(o_trap) != 0 {
            traps += 1;
        }
    }
    let dt = start.elapsed();
    let mcyc = n as f64 / dt.as_secs_f64() / 1e6;
    println!(
        "  ran {n} cycles, mem requests={fetches}, trap-cycles={traps}, {mcyc:.2} Mcyc/s {:.1} ms",
        dt.as_secs_f64() * 1e3
    );
    println!("  [OK] picorv32 compiled and executed through the JIT");
}
