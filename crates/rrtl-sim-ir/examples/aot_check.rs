//! Validate + benchmark the AOT (clang -O3) backend against the Cranelift JIT on
//! cpu.sv: same stimulus, compare outputs every cycle (bit-exact), then time both.
//! Build: cargo run --release --features "jit aot" -p rrtl-sim-ir --example aot_check
fn main() {
    #[cfg(not(all(feature = "jit", feature = "aot")))]
    println!("build with --features \"jit aot\"");
    #[cfg(all(feature = "jit", feature = "aot"))]
    run();
}

#[cfg(all(feature = "jit", feature = "aot"))]
fn run() {
    use rrtl_sim_ir::{aot::AotSimulator, jit::JitSimulator, lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;
    use std::time::Instant;

    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv/cpu.sv"))
        .expect("read cpu.sv");
    let imported = import_sv(&src, Some("cpu")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "cpu").expect("lower packed");
    let machine = lower_to_machine_program(&program);

    let idx = |name: &str| {
        let h = compiled.find_module("cpu").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle;
        program.signal_index(h).unwrap()
    };
    let (i_clk, i_rst, i_instr) = (idx("clk"), idx("rst"), idx("instr"));
    let (o_pc, o_x10) = (idx("pc"), idx("x10"));

    let t0 = Instant::now();
    let mut aot = AotSimulator::compile(&machine).expect("aot compile");
    let aot_compile = t0.elapsed();
    let mut jit = JitSimulator::compile(&machine).expect("jit compile");
    println!("AOT clang -O3 compile: {:.0} ms ({} signals)", aot_compile.as_secs_f64() * 1e3, aot.signal_count());

    let instr_of = |c: usize| ((c as u64).wrapping_mul(2654435761) ^ (c as u64) << 13) as u32;

    // correctness: AOT vs JIT, per cycle
    let mut ok = true;
    for c in 0..5000usize {
        let (rst, instr) = ((c == 0) as u64, instr_of(c) as u64);
        aot.set_signal(i_clk, 1); aot.set_signal(i_rst, rst); aot.set_signal(i_instr, instr);
        jit.set_signal(i_clk, 1); jit.set_signal(i_rst, rst); jit.set_signal(i_instr, instr);
        aot.tick();
        jit.tick();
        if aot.get_signal(o_pc) != jit.get_signal(o_pc) || aot.get_signal(o_x10) != jit.get_signal(o_x10) {
            println!(
                "  MISMATCH at cycle {c}: aot pc={} x10={} | jit pc={} x10={}",
                aot.get_signal(o_pc), aot.get_signal(o_x10), jit.get_signal(o_pc), jit.get_signal(o_x10)
            );
            ok = false;
            break;
        }
    }
    println!("  correctness (AOT vs JIT): [{}]", if ok { "OK" } else { "FAIL" });

    // throughput (steady NOP stream)
    let n = 2_000_000usize;
    aot.set_signal(i_rst, 1); aot.tick(); aot.set_signal(i_rst, 0);
    let s = Instant::now();
    for _ in 0..n { aot.set_signal(i_instr, 0x13); aot.tick(); }
    let da = s.elapsed().as_secs_f64();

    jit.set_signal(i_rst, 1); jit.tick(); jit.set_signal(i_rst, 0);
    let s = Instant::now();
    for _ in 0..n { jit.set_signal(i_instr, 0x13); jit.tick(); }
    let dj = s.elapsed().as_secs_f64();

    println!("  AOT (clang -O3): {:.2} Mcyc/s", n as f64 / da / 1e6);
    println!("  JIT (Cranelift): {:.2} Mcyc/s", n as f64 / dj / 1e6);
}
