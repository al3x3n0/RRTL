//! Dynamic instruction-subset specialization of a real RISC-V core (YosysHQ
//! picorv32), on the Cranelift JIT.
//!
//! A workload exercises only a subset of the ISA, so picorv32's per-instruction
//! decode-flag registers (`instr_sll`, `instr_mul`, `instr_slt`, `instr_lw`, …)
//! for the *unused* instruction classes stay 0 for the entire run — a runtime
//! quasi-constant that static analysis cannot know (it depends on the program in
//! RAM, not the RTL). We profile a run to discover those always-0 flags, freeze
//! them to 0, and re-JIT: const-folding then DCEs the barrel shifter, the
//! comparators, the load/store paths, etc. — whole datapaths the workload never
//! activates. Unlike the cfgdsp microbenchmarks, picorv32's tick is heavy, so the
//! eliminated logic shows up in wall-clock.
//!
//! We measure generic vs specialized from reset on the same program, and check
//! bit-exactness by comparing the full memory-bus transaction trace.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example picorv32_specialize -- bench/sv/picorv32.v
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
    bus: Vec<(u32, u32, u32)>, // (addr, wdata, wstrb) per ready handshake
    ever1: Vec<bool>,
    secs: f64,
    trap: u64,
}

/// Run picorv32 from reset for `cycles`, driving a 1-cycle-latency RAM holding
/// `prog`. Records the bus transaction trace; if `track` is set, records which of
/// those signals was ever non-zero (for profiling). Times only the cycle loop.
#[cfg(feature = "jit")]
fn run_pico(
    machine: &rrtl_sim_ir::PackedMachineProgram,
    io: &Io,
    prog: &[u32],
    cycles: usize,
    track: Option<&[usize]>,
) -> RunResult {
    use rrtl_sim_ir::jit::JitSimulator;
    use std::time::Instant;

    let mut jit = JitSimulator::compile(machine).expect("jit compile");
    let mut mem = vec![0u32; 8192];
    mem[..prog.len()].copy_from_slice(prog);
    let mut ever1 = vec![false; track.map_or(0, |t| t.len())];
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

        if let Some(flags) = track {
            for (k, &fi) in flags.iter().enumerate() {
                if jit.get_signal(fi) != 0 {
                    ever1[k] = true;
                }
            }
        }
    }
    let secs = t.elapsed().as_secs_f64();
    let trap = jit.get_signal(io.trap);
    RunResult { bus, ever1, secs, trap }
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::specialize::freeze_signals_program;
    use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;
    use std::collections::HashMap;

    let path = std::env::args().nth(1).unwrap_or_else(|| "bench/sv/picorv32.v".into());
    let src = std::fs::read_to_string(&path).expect("read picorv32.v");
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

    let cycles = 300_000usize;

    // ---- PROFILE: discover which decode flags stay 0 over the whole run ----
    let flag_idx: Vec<usize> = flags.iter().map(|(_, i)| *i).collect();
    let prof = run_pico(&machine, &io, &prog, cycles, Some(&flag_idx));
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

    // ---- FREEZE the always-0 flags to 0 and re-JIT ----
    let frozen: HashMap<usize, u128> = always0.iter().map(|(_, i)| (*i, 0u128)).collect();
    let (spec, fstats) = freeze_signals_program(&machine, &frozen);
    let total_instrs: usize = [
        &machine.streams.async_reset_comb, &machine.streams.comb,
        &machine.streams.tick_next, &machine.streams.tick_commit,
    ].iter().map(|b| b.packets.iter().map(|p| p.instrs.len()).sum::<usize>()).sum();

    // ---- BENCH generic vs specialized (best-of-3), bus-trace bit-exactness ----
    let bench = |m: &rrtl_sim_ir::PackedMachineProgram| -> RunResult {
        let mut best = run_pico(m, &io, &prog, cycles, None);
        for _ in 0..2 {
            let r = run_pico(m, &io, &prog, cycles, None);
            if r.secs < best.secs {
                best = r;
            }
        }
        best
    };
    let g = bench(&machine);
    let s = bench(&spec);
    let bit_exact = g.bus == s.bus && g.trap == s.trap;
    let result = g.bus.iter().rev().find(|(a, _, w)| *a == 0x40 && *w != 0).map(|(_, d, _)| *d);

    println!("picorv32 dynamic instruction-subset specialization (Cranelift JIT)");
    println!("workload: nested add/branch loop, {cycles} cycles, sum=mem[0x40]={:?}\n", result);
    println!("{} decode-flag registers; used by workload ({}): {}", flags.len(), used.len(), used.join(" "));
    println!("froze {} unused flags to 0 → {} → {} machine instrs ({} removed, {:.1}%)\n",
        always0.len(), total_instrs, fstats.specialize.instrs_after,
        fstats.specialize.instrs_removed(),
        100.0 * fstats.specialize.instrs_removed() as f64 / total_instrs as f64);
    println!("  generic JIT     : {:.2} Mcyc/s", cycles as f64 / g.secs / 1e6);
    println!("  specialized JIT : {:.2} Mcyc/s   =>  {:.2}x", cycles as f64 / s.secs / 1e6, g.secs / s.secs);
    println!("  bus-trace bit-exact (generic == specialized): {}", if bit_exact { "YES" } else { "NO" });
    println!("\n  => {}", if bit_exact && result.is_some() {
        "profiled the live ISA subset, froze the rest, DCE'd unused datapaths — bit-exact"
    } else {
        "FAIL (no result stored or traces diverged)"
    });
}
