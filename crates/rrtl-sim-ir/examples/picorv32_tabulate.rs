//! Decoder tabulation on the Cranelift JIT: replace picorv32's deep decode mux
//! cone with a precomputed ROM keyed on the instruction bits, then run the core
//! through the host mem-bus and check it is bit-exact vs the plain JIT.
//!
//! This is the lever §4h.1 credits to Verilator ("table-izing decoders"), built
//! for the JIT where collapsing the ~48-deep decode chain should pay (the AOT's
//! clang already schedules it; cf. the mux-rebalance JIT-win/AOT-loss).
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example picorv32_tabulate -- [N] [KEY_MAX]
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
struct Io {
    clk: usize,
    resetn: usize,
    mem_ready: usize,
    mem_rdata: usize,
    mem_valid: usize,
    mem_addr: usize,
    mem_wstrb: usize,
    mem_wdata: usize,
    trap: usize,
}

#[cfg(feature = "jit")]
struct RunResult {
    bus: Vec<(u32, u32, u32)>,
    secs: f64,
    trap: u64,
}

/// Run picorv32 from reset for `cycles` on a JIT engine; if `rom` is set, load
/// it into the tabulated ROM memory before running. Times only the cycle loop.
#[cfg(feature = "jit")]
fn run_pico(
    machine: &rrtl_sim_ir::PackedMachineProgram,
    io: &Io,
    prog: &[u32],
    cycles: usize,
    rom: Option<(usize, &[u128])>,
) -> RunResult {
    use rrtl_sim_ir::jit::JitSimulator;
    use std::time::Instant;

    let mut jit = JitSimulator::compile(machine).expect("jit compile");
    if let Some((idx, data)) = rom {
        jit.set_memory(idx, data);
    }
    let mut mem = vec![0u32; 8192];
    mem[..prog.len()].copy_from_slice(prog);
    let mut bus = Vec::new();
    let mut prev_ready = 0u64;

    let t = Instant::now();
    for c in 0..cycles {
        let resetn = (c >= 4) as u64;
        let valid = jit.get_signal(io.mem_valid);
        let addr = jit.get_signal(io.mem_addr);
        let wstrb = jit.get_signal(io.mem_wstrb);
        let wdata = jit.get_signal(io.mem_wdata);

        let ready = (valid != 0 && prev_ready == 0) as u64;
        let mut rdata = 0u64;
        if ready != 0 {
            let widx = (addr >> 2) as usize;
            if wstrb != 0 {
                if widx < mem.len() {
                    let mut w = mem[widx];
                    for b in 0..4 {
                        if wstrb & (1 << b) != 0 {
                            let sh = b * 8;
                            w = (w & !(0xFFu32 << sh)) | ((((wdata >> sh) & 0xFF) as u32) << sh);
                        }
                    }
                    mem[widx] = w;
                }
            } else {
                rdata = mem.get(widx).copied().unwrap_or(0) as u64;
            }
            bus.push((addr as u32, wdata as u32, wstrb as u32));
        }
        prev_ready = ready;

        jit.set_signal(io.clk, 1);
        jit.set_signal(io.resetn, resetn);
        jit.set_signal(io.mem_ready, ready);
        jit.set_signal(io.mem_rdata, rdata);
        jit.tick();
    }
    RunResult { bus, secs: t.elapsed().as_secs_f64(), trap: jit.get_signal(io.trap) }
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::tabulate::tabulate_decode_program;
    use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;
    use std::time::Instant;

    let cycles: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(2_000_000);
    let key_max: u32 = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(20);

    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv/picorv32.v");
    let src = std::fs::read_to_string(path).expect("read picorv32.v");
    let imported = import_sv(&src, Some("picorv32")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "picorv32").expect("lower packed");
    let machine = lower_to_machine_program(&program);
    let module = compiled.find_module("picorv32").unwrap();

    let idx = |name: &str| {
        let h = module.signals.iter().find(|s| s.name == name).unwrap().handle;
        program.signal_index(h).unwrap()
    };
    let io = Io {
        clk: idx("clk"),
        resetn: idx("resetn"),
        mem_ready: idx("mem_ready"),
        mem_rdata: idx("mem_rdata"),
        mem_valid: idx("mem_valid"),
        mem_addr: idx("mem_addr"),
        mem_wstrb: idx("mem_wstrb"),
        mem_wdata: idx("mem_wdata"),
        trap: idx("trap"),
    };

    // Same nested add/branch workload as the other picorv32 examples.
    let prog = rrtl_riscv_asm::assemble(
        "
        li   x1, 0
        li   x4, 0
        li   x5, 100
    outer:
        li   x2, 0
        li   x3, 100
    inner:
        add  x1, x1, x2
        addi x2, x2, 1
        blt  x2, x3, inner
        addi x4, x4, 1
        blt  x4, x5, outer
        sw   x1, 0x40(x0)
    spin:
        j spin
        ",
    )
    .expect("assemble workload");

    let t0 = Instant::now();
    let (tab, stats) = tabulate_decode_program(&machine, key_max);
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;
    let Some(stats) = stats else {
        println!("no tabulation group found (key_max={key_max})");
        return;
    };

    println!("picorv32 decoder tabulation (Cranelift JIT)");
    println!(
        "  tabulated {} decode flags, key = {} bits of signal #{} → ROM {} entries ({} KiB dense)",
        stats.tabulated_flags, stats.key_bits.len(), stats.key_signal, stats.rom_depth,
        stats.rom_depth * 16 / 1024,
    );
    let signame = |i: usize| program.signals.get(i).map(|s| s.name.as_str()).unwrap_or("?");
    println!("  key signal     : {}", signame(stats.key_signal));
    println!("  tabulated flags: {}", stats.flag_regs.iter().map(|&r| signame(r)).collect::<Vec<_>>().join(" "));
    println!("  machine instrs : {} → {} ({} removed)", stats.instrs_before, stats.instrs_after,
        stats.instrs_before as i64 - stats.instrs_after as i64);
    println!("  ROM build time : {build_ms:.0} ms\n");

    let rom = (stats.rom_mem_index, stats.rom_data.as_slice());
    // Interleaved best-of-3 for a shared thermal profile.
    let bench = |m: &rrtl_sim_ir::PackedMachineProgram, r: Option<(usize, &[u128])>| -> RunResult {
        let mut best = run_pico(m, &io, &prog, cycles, r);
        for _ in 0..2 {
            let x = run_pico(m, &io, &prog, cycles, r);
            if x.secs < best.secs {
                best = x;
            }
        }
        best
    };
    let g = bench(&machine, None);
    let s = bench(&tab, Some(rom));
    let bit_exact = g.bus == s.bus && g.trap == s.trap;
    if !bit_exact {
        println!("  trace lens: plain {} tab {}; traps {} {}", g.bus.len(), s.bus.len(), g.trap, s.trap);
        for (i, (a, b)) in g.bus.iter().zip(&s.bus).enumerate() {
            if a != b {
                println!("  first divergence at handshake #{i}: plain {a:?} tab {b:?}");
                for j in i.saturating_sub(2)..(i + 1).min(g.bus.len()) {
                    println!("    #{j}: plain {:?} tab {:?}", g.bus[j], s.bus.get(j));
                }
                break;
            }
        }
    }
    let result = g.bus.iter().rev().find(|(a, _, w)| *a == 0x40 && *w != 0).map(|(_, d, _)| *d);

    println!("  workload result (sum=mem[0x40]) : {result:?}");
    println!("  plain JIT      : {:.2} Mcyc/s", cycles as f64 / g.secs / 1e6);
    println!("  tabulated JIT  : {:.2} Mcyc/s   =>  {:.2}x", cycles as f64 / s.secs / 1e6, g.secs / s.secs);
    println!("  bus-trace bit-exact (plain == tabulated): {}", if bit_exact { "YES" } else { "NO — MISMATCH" });
    assert!(bit_exact, "tabulated JIT diverged from plain JIT");
    assert!(result.is_some(), "workload did not store its result");
}
