//! Does the IR-level specializer (const-fold/DCE/copy-prop/algebraic — the V3-pass
//! analog) make the AOT (clang -O3) faster on a REAL core, beyond what clang can
//! recover itself? Runs picorv32 through the same mem-bus handshake on the plain
//! AOT and the specialized AOT, asserts a bit-exact mem-transaction trace, and
//! reports throughput. The win clang can't get: the specializer prunes register
//! next-state stores (to the observable `st` buffer) that clang must conservatively
//! keep. Best-of-N interleaved for a fair thermal profile.
//! Build: cargo run --release --features aot -p rrtl-sim-ir --example picorv32_aot_spec -- [N]
fn main() {
    #[cfg(not(feature = "aot"))]
    println!("build with --features aot");
    #[cfg(feature = "aot")]
    run();
}

#[cfg(feature = "aot")]
fn run() {
    use rrtl_sim_ir::specialize::specialize_program;
    use rrtl_sim_ir::{aot::AotSimulator, lower_to_machine_program, lower_to_packed_program, PackedMachineProgram};
    use rrtl_sv_frontend::import_sv;
    use std::time::Instant;

    let n: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(2_000_000);

    // A small mixed program: arithmetic + branch loop (the picorv32_bench workload).
    let prog = rrtl_riscv_asm::assemble(
        "li x1,0\n li x2,0\n loop: addi x1,x1,1\n addi x2,x2,3\n xor x3,x1,x2\n j loop\n",
    )
    .expect("assemble");

    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv/picorv32.v"))
        .expect("read picorv32.v");
    let imported = import_sv(&src, Some("picorv32")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "picorv32").expect("lower packed");
    let machine = lower_to_machine_program(&program);
    let (spec, stats) = specialize_program(&machine);

    let instrs = |m: &PackedMachineProgram| -> usize {
        [&m.streams.async_reset_comb, &m.streams.comb, &m.streams.tick_next, &m.streams.tick_commit]
            .iter().map(|b| b.packets.iter().map(|p| p.instrs.len()).sum::<usize>()).sum()
    };
    println!("picorv32 AOT: plain vs IR-specialized (clang -O3 both)");
    println!("  machine instrs : {} → {} ({:.1}% removed; {} folded/dead)",
        instrs(&machine), instrs(&spec),
        100.0 * (instrs(&machine) - instrs(&spec)) as f64 / instrs(&machine) as f64, stats.instrs_removed());

    let idx = |name: &str| {
        let h = compiled.find_module("picorv32").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle;
        program.signal_index(h).unwrap()
    };
    let (i_clk, i_resetn) = (idx("clk"), idx("resetn"));
    let (i_mem_ready, i_mem_rdata) = (idx("mem_ready"), idx("mem_rdata"));
    let (o_mem_valid, o_mem_addr) = (idx("mem_valid"), idx("mem_addr"));
    let (o_mem_wstrb, o_mem_wdata) = (idx("mem_wstrb"), idx("mem_wdata"));

    // Run the host mem-bus handshake on one AOT engine; return (Mcyc/s, trace-hash,
    // fetches). The trace folds every ready-cycle (addr,wstrb,wdata) so two engines
    // are bit-exact iff their full bus transaction streams match.
    let bench = |m: &PackedMachineProgram| -> (f64, u64, u64) {
        let mut sim = AotSimulator::compile(m).expect("aot compile");
        let mut mem = vec![0u32; 4096];
        mem[..prog.len()].copy_from_slice(&prog);
        let mut prev_ready = 0u64;
        let mut fetches = 0u64;
        let mut hash = 0xcbf29ce484222325u64; // fnv-1a over the bus trace
        let start = Instant::now();
        for c in 0..n {
            let resetn = (c >= 4) as u64;
            let valid = sim.get_signal(o_mem_valid);
            let addr = sim.get_signal(o_mem_addr);
            let wstrb = sim.get_signal(o_mem_wstrb);
            let wdata = sim.get_signal(o_mem_wdata);
            let ready = (valid != 0 && prev_ready == 0) as u64;
            let mut rdata = 0u64;
            if ready != 0 {
                for v in [addr, wstrb, wdata] {
                    hash = (hash ^ v).wrapping_mul(0x100000001b3);
                }
                let widx = ((addr >> 2) as usize) & (mem.len() - 1);
                if wstrb != 0 {
                    let mut w = mem[widx];
                    for b in 0..4 {
                        if wstrb & (1 << b) != 0 {
                            let sh = b * 8;
                            w = (w & !(0xFFu32 << sh)) | ((((wdata >> sh) & 0xFF) as u32) << sh);
                        }
                    }
                    mem[widx] = w;
                } else {
                    rdata = mem[widx] as u64;
                    fetches += 1;
                }
            }
            prev_ready = ready;
            sim.set_signal(i_clk, 1);
            sim.set_signal(i_resetn, resetn);
            sim.set_signal(i_mem_ready, ready);
            sim.set_signal(i_mem_rdata, rdata);
            sim.tick();
        }
        (n as f64 / start.elapsed().as_secs_f64() / 1e6, hash, fetches)
    };

    // Interleaved best-of-3 so both share the same thermal envelope.
    let (mut plain, mut sp) = (0f64, 0f64);
    let (mut h0, mut h1, mut f0, mut f1) = (0u64, 0u64, 0u64, 0u64);
    for _ in 0..3 {
        let a = bench(&machine);
        let b = bench(&spec);
        plain = plain.max(a.0);
        sp = sp.max(b.0);
        h0 = a.1; h1 = b.1; f0 = a.2; f1 = b.2;
    }
    println!("  plain AOT      : {plain:.2} Mcyc/s  ({f0} mem reads)");
    println!("  specialized AOT: {sp:.2} Mcyc/s  ({f1} mem reads)   => {:.2}x", sp / plain);
    println!("  bit-exact bus trace: {}", if h0 == h1 { "YES" } else { "NO — MISMATCH" });
    assert_eq!(h0, h1, "specialized AOT diverged from plain AOT");
}
