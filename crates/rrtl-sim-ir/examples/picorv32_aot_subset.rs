//! Dynamic instruction-subset specialization of a real RISC-V core (YosysHQ
//! picorv32) on the AOT (clang -O3) backend — the profile-guided complement to
//! the JIT version (`picorv32_specialize`), and the AOT lever the *static*
//! specializer could not deliver.
//!
//! `aot_specialize_probe`/`picorv32_aot_spec` showed that running the STATIC
//! specializer (const-fold/DCE) before AOT is NEUTRAL (~0.98x): clang -O3 already
//! recovers every value-level redundancy. But FREEZING profiled quasi-constant
//! decode flags is categorically different — it injects information clang
//! *cannot* derive (that `instr_sll`/`instr_mul`/… are 0 for this run depends on
//! the program bytes in RAM, not the RTL). Freezing them to 0 makes const-fold
//! DCE whole functional units — the barrel shifter, the comparators, the
//! load/store byte lanes, the CSR/IRQ logic — *before* clang ever sees the C, so
//! clang simply has half as much to compile and run.
//!
//! We profile a run to find the always-0 flags, freeze them, AOT-compile both the
//! plain and specialized programs, and report throughput + a bit-exact bus-trace
//! check. Best-of-N interleaved for a fair thermal profile.
//! Build: cargo run --release --features aot -p rrtl-sim-ir --example picorv32_aot_subset -- [N]
fn main() {
    #[cfg(not(feature = "aot"))]
    println!("build with --features aot");
    #[cfg(feature = "aot")]
    run();
}

#[cfg(feature = "aot")]
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

#[cfg(feature = "aot")]
struct RunResult {
    bus: Vec<(u32, u32, u32)>, // (addr, wdata, wstrb) per ready handshake
    ever1: Vec<bool>,
    secs: f64,
    trap: u64,
}

/// Run picorv32 from reset for `cycles` on an AOT engine compiled from `machine`,
/// driving a 1-cycle-latency RAM holding `prog`. Records the bus transaction
/// trace; if `track` is set, records which of those signals was ever non-zero
/// (for profiling). Times only the cycle loop, not compilation.
#[cfg(feature = "aot")]
fn run_pico(
    machine: &rrtl_sim_ir::PackedMachineProgram,
    io: &Io,
    prog: &[u32],
    cycles: usize,
    track: Option<&[usize]>,
) -> RunResult {
    use rrtl_sim_ir::aot::AotSimulator;
    use std::time::Instant;

    let mut sim = AotSimulator::compile(machine).expect("aot compile");
    let mut mem = vec![0u32; 8192];
    mem[..prog.len()].copy_from_slice(prog);
    let mut ever1 = vec![false; track.map_or(0, |t| t.len())];
    let mut bus = Vec::new();
    let mut prev_ready = 0u64;

    let t = Instant::now();
    for c in 0..cycles {
        let resetn = (c >= 4) as u64;
        let valid = sim.get_signal(io.mem_valid);
        let addr = sim.get_signal(io.mem_addr);
        let wstrb = sim.get_signal(io.mem_wstrb);
        let wdata = sim.get_signal(io.mem_wdata);

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

        sim.set_signal(io.clk, 1);
        sim.set_signal(io.resetn, resetn);
        sim.set_signal(io.mem_ready, ready);
        sim.set_signal(io.mem_rdata, rdata);
        sim.tick();

        if let Some(flags) = track {
            for (k, &fi) in flags.iter().enumerate() {
                if sim.get_signal(fi) != 0 {
                    ever1[k] = true;
                }
            }
        }
    }
    let secs = t.elapsed().as_secs_f64();
    let trap = sim.get_signal(io.trap);
    RunResult { bus, ever1, secs, trap }
}

#[cfg(feature = "aot")]
fn run() {
    use rrtl_sim_ir::specialize::freeze_signals_program;
    use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;
    use std::collections::HashMap;

    let cycles: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(2_000_000);

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

    // Decode-flag registers: the per-instruction `instr_*` / `is_*` latches.
    let flags: Vec<(String, usize)> = module
        .signals
        .iter()
        .filter(|s| s.name.starts_with("instr_") || s.name.starts_with("is_"))
        .filter_map(|s| program.signal_index(s.handle).map(|i| (s.name.clone(), i)))
        .collect();

    // Workload: a nested add/branch loop (uses only addi/add/blt/sw/j). Small
    // constants keep `li` as addi (no lui). Every other instruction class — the
    // whole shifter, comparators, loads, logic ops, CSRs, IRQ — stays unused.
    let prog = rrtl_riscv_asm::assemble(
        "
        li   x1, 0          # sum
        li   x4, 0          # outer i
        li   x5, 100        # outer limit
    outer:
        li   x2, 0          # inner j
        li   x3, 100        # inner limit
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

    // ---- PROFILE: discover which decode flags stay 0 over the whole run ----
    // A short profiling window is enough — the live ISA subset is established in
    // the first few loop iterations.
    let prof_cycles = cycles.min(300_000);
    let flag_idx: Vec<usize> = flags.iter().map(|(_, i)| *i).collect();
    let prof = run_pico(&machine, &io, &prog, prof_cycles, Some(&flag_idx));
    let always0: Vec<(String, usize)> = flags
        .iter()
        .zip(&prof.ever1)
        .filter(|(_, &seen)| !seen)
        .map(|((n, i), _)| (n.clone(), *i))
        .collect();
    let used: Vec<&str> = flags
        .iter()
        .zip(&prof.ever1)
        .filter(|(_, &seen)| seen)
        .map(|((n, _), _)| n.as_str())
        .collect();

    // ---- FREEZE the always-0 flags to 0 and re-specialize ----
    let frozen: HashMap<usize, u128> = always0.iter().map(|(_, i)| (*i, 0u128)).collect();
    let (spec, fstats) = freeze_signals_program(&machine, &frozen);
    let total_instrs: usize = [
        &machine.streams.async_reset_comb, &machine.streams.comb,
        &machine.streams.tick_next, &machine.streams.tick_commit,
    ].iter().map(|b| b.packets.iter().map(|p| p.instrs.len()).sum::<usize>()).sum();

    // ---- BENCH plain vs specialized AOT (interleaved best-of-3), bit-exact ----
    let bench = |m: &rrtl_sim_ir::PackedMachineProgram| -> RunResult {
        run_pico(m, &io, &prog, cycles, None)
    };
    let (mut g, mut s) = (bench(&machine), bench(&spec));
    for _ in 0..2 {
        let a = bench(&machine);
        if a.secs < g.secs {
            g = a;
        }
        let b = bench(&spec);
        if b.secs < s.secs {
            s = b;
        }
    }
    let bit_exact = g.bus == s.bus && g.trap == s.trap;
    let result = g.bus.iter().rev().find(|(a, _, w)| *a == 0x40 && *w != 0).map(|(_, d, _)| *d);

    println!("picorv32 dynamic instruction-subset specialization (AOT clang -O3)");
    println!("workload: nested add/branch loop, {cycles} cycles, sum=mem[0x40]={result:?}\n");
    println!("{} decode-flag registers; used by workload ({}): {}", flags.len(), used.len(), used.join(" "));
    println!("froze {} unused flags to 0 → {} → {} machine instrs ({} removed, {:.1}%)\n",
        always0.len(), total_instrs, fstats.specialize.instrs_after,
        fstats.specialize.instrs_removed(),
        100.0 * fstats.specialize.instrs_removed() as f64 / total_instrs as f64);
    println!("  generic AOT     : {:.2} Mcyc/s", cycles as f64 / g.secs / 1e6);
    println!("  specialized AOT : {:.2} Mcyc/s   =>  {:.2}x", cycles as f64 / s.secs / 1e6, g.secs / s.secs);
    println!("  bus-trace bit-exact (generic == specialized): {}", if bit_exact { "YES" } else { "NO" });
    assert!(bit_exact, "specialized AOT diverged from generic AOT");
    assert!(result.is_some(), "workload did not store its result");
}
