//! Real-design benchmark: import `bench/sv/cpu.sv` (a RISC-V execute unit) via
//! the SystemVerilog frontend, run it through the Cranelift JIT, and cross-check
//! it against the SIMD-CPU interpreter oracle (same program). Prints throughput.
//! The same cpu.sv is run through Verilator by bench/sv/run.sh for the headline.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example sv_cpu_bench -- [N]
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::{jit::JitSimulator, lower_to_machine_program, lower_to_packed_program, SimdCpuSimulator};
    use rrtl_sv_frontend::import_sv;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(2_000_000);

    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv/cpu.sv"))
        .expect("read cpu.sv");
    let imported = import_sv(&src, Some("cpu")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "cpu").expect("lower packed");
    let machine = lower_to_machine_program(&program);

    let handle = |name: &str| {
        compiled.find_module("cpu").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle
    };
    let idx = |name: &str| program.signal_index(handle(name)).unwrap();
    let (i_clk, i_rst, i_instr) = (idx("clk"), idx("rst"), idx("instr"));
    let (o_pc, o_x10) = (idx("pc"), idx("x10"));

    let mut jit = JitSimulator::compile(&machine).expect("jit compile");
    let mut cpu = SimdCpuSimulator::new(program.clone(), 1).expect("interp");
    println!("imported cpu.sv: top={} modules={:?}", imported.top_name, imported.modules);

    // deterministic pseudo-random instruction stream
    let instr_of = |c: usize| ((c as u64).wrapping_mul(2654435761) ^ (c as u64) << 13) as u32;

    // ---- correctness: JIT vs interpreter oracle, per cycle (cycle 0 = reset) ----
    let mut ok = true;
    for c in 0..3000usize {
        let rst = (c == 0) as u64;
        let instr = instr_of(c) as u64;
        jit.set_signal(i_clk, 1);
        jit.set_signal(i_rst, rst);
        jit.set_signal(i_instr, instr);
        cpu.set_signal(handle("clk"), &[1]).unwrap();
        cpu.set_signal(handle("rst"), &[rst as u128]).unwrap();
        cpu.set_signal(handle("instr"), &[instr as u128]).unwrap();
        jit.tick();
        cpu.tick().unwrap();
        let jp = jit.get_signal(o_pc);
        let jx = jit.get_signal(o_x10);
        let cp = cpu.get_signal(handle("pc")).unwrap()[0] as u64;
        let cx = cpu.get_signal(handle("x10")).unwrap()[0] as u64;
        if jp != cp || jx != cx {
            ok = false;
            println!("  MISMATCH cycle {c}: jit pc={jp} x10={jx} | interp pc={cp} x10={cx}");
            break;
        }
    }
    println!("  correctness (JIT vs SIMD-interp): [{}]", if ok { "OK" } else { "MISMATCH" });

    // ---- throughput: scalar JIT, per-cycle instruction stream ----
    jit.set_signal(i_rst, 0);
    let runit = |jit: &mut JitSimulator| {
        for c in 0..n {
            jit.set_signal(i_instr, instr_of(c) as u64);
            jit.tick();
        }
    };
    runit(&mut jit); // warm
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        runit(&mut jit);
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!(
        "  rrtl-jit: {n} cycles, pc={} x10={}, {:.2} Mcyc/s {:.1} ms",
        jit.get_signal(o_pc),
        jit.get_signal(o_x10),
        n as f64 / best / 1e6,
        best * 1e3
    );
}
