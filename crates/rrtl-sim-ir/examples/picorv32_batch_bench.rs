//! Batch throughput of N picorv32 SoC instances in RRTL's SIMD CPU lanes — the
//! moat regime (regression/fuzzing/DSE: many CPUs, independent stimulus, lockstep
//! lanes, divergence-tolerant since the engine is oblivious). Reports instance-
//! cycles/s to compare against N Verilator processes (each ~one core).
//! Build: cargo run --release -p rrtl-sim-ir --example picorv32_batch_bench -- [lanes] [cycles]
use rrtl_sim_ir::{lower_to_packed_program, SimdCpuSimulator};
use rrtl_sv_frontend::import_sv;
use std::time::Instant;

fn main() {
    let lanes: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(256);
    let cycles: usize = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(20_000);

    // A compute loop so every lane runs many cycles of real RISC-V (sum 1..N, vary
    // the limit per lane so instances diverge — the realistic batch case).
    let prog_for = |limit: u32| {
        rrtl_riscv_asm::assemble(&format!(
            "li x5,0\n li x6,1\n li x7,{limit}\n loop: add x5,x5,x6\n addi x6,x6,1\n blt x6,x7,loop\n li x8,0x100\n sw x5,0(x8)\n spin: j spin\n"
        )).expect("assemble")
    };

    let core = std::fs::read_to_string("bench/sv/picorv32.v").expect("read picorv32.v");
    let soc = std::fs::read_to_string("bench/sv/picorv32_soc.v").expect("read picorv32_soc.v");
    let imported = import_sv(&format!("{core}\n{soc}\n"), Some("picorv32_soc")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "picorv32_soc").expect("lower");
    let module = compiled.find_module("picorv32_soc").unwrap();
    let sig = |n: &str| module.signals.iter().find(|s| s.name == n).unwrap().handle;
    let (clk, resetn) = (sig("clk"), sig("resetn"));
    let mem = program.memories.iter().find(|m| m.name == "mem" || m.name.ends_with(".mem")).unwrap().source;

    let mut sim = SimdCpuSimulator::new(program, lanes).unwrap();
    // each lane gets its own program (different loop limit) — independent instances
    let lane_mem: Vec<Vec<u128>> = (0..lanes)
        .map(|l| {
            let prog = prog_for(4 + (l as u32 % 60));
            let mut w = vec![0u128; 1024];
            for (i, x) in prog.iter().enumerate() {
                w[i] = *x as u128;
            }
            w
        })
        .collect();
    sim.set_memory(mem, &lane_mem).unwrap();
    sim.set_signal(clk, &vec![1u128; lanes]).unwrap();
    sim.set_signal(resetn, &vec![0u128; lanes]).unwrap();
    sim.tick_many(8).unwrap();
    sim.set_signal(resetn, &vec![1u128; lanes]).unwrap();

    sim.tick_many(2000).unwrap(); // warm
    let t = Instant::now();
    sim.tick_many(cycles).unwrap();
    let secs = t.elapsed().as_secs_f64();
    let inst_cyc = (lanes * cycles) as f64;
    println!("picorv32 SoC batch on SIMD CPU — {lanes} lanes (independent programs), {cycles} cycles");
    println!("  total: {:.1} M-instance-cycles/s  ({:.3} M-cyc/s per lane)", inst_cyc / secs / 1e6, cycles as f64 / secs / 1e6);
    println!("  (compare: N Verilator processes each ~one core; this is 1 RRTL process, SIMD lanes, 1 thread)");
}
