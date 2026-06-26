//! Validate the self-contained picorv32 SoC (core + internal RAM) running
//! autonomously in the SIMD CPU batch engine: load a program into each lane's RAM,
//! run, and read the per-lane `result`/`done`. This is the foundation of the batch
//! moat demo — N independent CPUs in lockstep lanes, each with its own memory.
//! Build: cargo run --release -p rrtl-sim-ir --example picorv32_soc_check -- [lanes] [cycles]
use rrtl_sim_ir::{lower_to_packed_program, SimdCpuSimulator};
use rrtl_sv_frontend::import_sv;

fn main() {
    let lanes: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(4);
    let cycles: usize = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(3000);

    // sum 1..5 = 15, store to magic addr 0x100 (latched into `result`), then spin.
    let prog = rrtl_riscv_asm::assemble(
        "
        li   x5, 0          # sum
        li   x6, 1          # i
        li   x7, 6          # limit
    loop:
        add  x5, x5, x6
        addi x6, x6, 1
        blt  x6, x7, loop
        li   x8, 0x100
        sw   x5, 0(x8)      # result = 15 -> 0x100 (latched)
    spin:
        j spin
        ",
    )
    .expect("assemble");

    let core = std::fs::read_to_string("bench/sv/picorv32.v").expect("read picorv32.v");
    let soc = std::fs::read_to_string("bench/sv/picorv32_soc.v").expect("read picorv32_soc.v");
    let src = format!("{core}\n{soc}\n");
    let imported = import_sv(&src, Some("picorv32_soc")).expect("import_sv picorv32_soc");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "picorv32_soc").expect("lower");
    println!(
        "lowered picorv32_soc: {} signals, {} memories",
        program.signals.len(),
        program.memories.len()
    );

    let module = compiled.find_module("picorv32_soc").unwrap();
    let sig = |n: &str| module.signals.iter().find(|s| s.name == n).unwrap_or_else(|| panic!("no signal {n}")).handle;
    let (clk, resetn, result, done, trap) = (sig("clk"), sig("resetn"), sig("result"), sig("done"), sig("trap"));
    let mem = program.memories.iter().find(|m| m.name == "mem" || m.name.ends_with(".mem"))
        .expect("no `mem` memory").source;

    let mut sim = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
    let mut words = vec![0u128; 1024]; // full RAM depth
    for (i, w) in prog.iter().enumerate() {
        words[i] = *w as u128;
    }
    sim.set_memory_replicated(mem, &words).unwrap();
    sim.set_signal(clk, &vec![1u128; lanes]).unwrap();

    // reset: resetn=0 for a few cycles, then 1.
    sim.set_signal(resetn, &vec![0u128; lanes]).unwrap();
    sim.tick_many(8).unwrap();
    sim.set_signal(resetn, &vec![1u128; lanes]).unwrap();
    sim.tick_many(cycles).unwrap();

    let res = sim.get_signal(result).unwrap();
    let dn = sim.get_signal(done).unwrap();
    let tr = sim.get_signal(trap).unwrap();
    println!("after {cycles} cycles ({lanes} lanes, same program):");
    println!("  result per lane: {:?}", res);
    println!("  done   per lane: {:?}", dn);
    println!("  trap   per lane: {:?}", tr);
    let ok = res.iter().all(|&r| r == 15) && dn.iter().all(|&d| d == 1) && tr.iter().all(|&t| t == 0);
    println!("  {}", if ok { "[PASS] all lanes computed sum 1..5 = 15, no trap" } else { "[FAIL]" });
    assert!(ok, "picorv32 SoC did not execute correctly in SIMD lanes");
}
