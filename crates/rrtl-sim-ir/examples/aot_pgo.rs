//! Profile-guided AOT: build the C backend instrumented (`-fprofile-generate`),
//! run a representative workload, `llvm-profdata merge`, then rebuild with
//! `-fprofile-use` so clang lays out branches/inlining for the hot paths (the way
//! Verilator users PGO their generated C++). Validates the PGO build is bit-exact
//! vs the plain -O3 build on cpu.sv (a branchy RISC-V execute unit).
//!
//! The SPEEDUP is a throughput win — NOT measured here, because this machine is
//! thermally throttled; run on a stable machine to A/B plain vs PGO cycles/s.
//! Build: cargo run --release --features aot -p rrtl-sim-ir --example aot_pgo
use rrtl_sim_ir::aot::AotSimulator;
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
use rrtl_sv_frontend::import_sv;

fn main() {
    let src = std::fs::read_to_string("bench/sv/cpu.sv").expect("read cpu.sv");
    let imported = import_sv(&src, Some("cpu")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "cpu").expect("lower");
    let machine = lower_to_machine_program(&program);
    let h = |n: &str| compiled.find_module("cpu").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
    let idx = |n: &str| program.signal_index(h(n)).unwrap();
    let (clk, rst, instr, pc, x10) = (idx("clk"), idx("rst"), idx("instr"), idx("pc"), idx("x10"));

    // deterministic instruction stream (same for profile + validation).
    let stim = |c: u64| -> u128 { ((c.wrapping_mul(6364136223846793005).wrapping_add(1)) >> 24) as u128 & 0xffff_ffff };

    println!("cpu.sv AOT PGO: building plain (-O3) and profile-guided (-fprofile-use)…");
    let mut plain = AotSimulator::compile(&machine).expect("plain AOT");
    // The PGO build runs `profile` on the instrumented sim, then rebuilds -fprofile-use.
    let mut pgo = AotSimulator::compile_pgo(&machine, |sim| {
        sim.set_signal(clk, 1);
        for c in 0..20_000u64 {
            sim.set_signal(rst, (c < 1) as u64);
            sim.set_signal_u128(instr, stim(c));
            sim.tick_many(1);
        }
    })
    .expect("PGO AOT");
    println!("  PGO pipeline (instrument → run → llvm-profdata merge → -fprofile-use): OK");

    // bit-exact: identical stimulus into both, compare pc/x10 every cycle.
    plain.set_signal(clk, 1);
    pgo.set_signal(clk, 1);
    let mut mism = 0u64;
    for c in 0..8_000u64 {
        for sim in [&mut plain, &mut pgo] {
            sim.set_signal(rst, (c < 1) as u64);
            sim.set_signal_u128(instr, stim(c));
            sim.tick_many(1);
        }
        if plain.get_signal_u128(pc) != pgo.get_signal_u128(pc) || plain.get_signal_u128(x10) != pgo.get_signal_u128(x10) {
            mism += 1;
        }
    }
    println!("  PGO vs plain AOT (pc,x10 × 8000 cyc): {}", if mism == 0 { "BIT-EXACT" } else { "MISMATCH" });
    println!("  (throughput A/B deferred — this machine is thermally throttled)");
    assert_eq!(mism, 0, "PGO build diverged from plain");
}
