//! Measure the temporal sparsity of picorv32 (a multi-cycle core): how many signal
//! slots actually change value each cycle? High sparsity ⇒ most register-update work
//! is redundant ⇒ activity/event skipping has a high ceiling. Low sparsity ⇒ skipping
//! can't help. This de-risks the skipping build before writing the mechanism.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example activity_sparsity_probe -- [cycles]
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::jit::JitSimulator;
    use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;

    let cycles: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(5000);
    let prog = rrtl_riscv_asm::assemble(
        "li x5,0\n li x6,1\n li x7,200\n loop: add x5,x5,x6\n addi x6,x6,1\n blt x6,x7,loop\n li x8,0x100\n sw x5,0(x8)\n spin: j spin\n",
    ).expect("assemble");

    let core = std::fs::read_to_string("bench/sv/picorv32.v").expect("read picorv32.v");
    let soc = std::fs::read_to_string("bench/sv/picorv32_soc.v").expect("read picorv32_soc.v");
    let imported = import_sv(&format!("{core}\n{soc}\n"), Some("picorv32_soc")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "picorv32_soc").expect("lower");
    let n_signals = program.signals.len();
    let machine = lower_to_machine_program(&program);
    let idx = |n: &str| {
        let h = compiled.find_module("picorv32_soc").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
        program.signal_index(h).unwrap()
    };
    let (clk, resetn) = (idx("clk"), idx("resetn"));
    let mem_idx = program.memories.iter().position(|m| m.name == "mem" || m.name.ends_with(".mem")).unwrap();

    let mut sim = JitSimulator::compile(&machine).unwrap();
    let words: Vec<u128> = {
        let mut w = vec![0u128; 1024];
        for (i, x) in prog.iter().enumerate() { w[i] = *x as u128; }
        w
    };
    sim.set_memory(mem_idx, &words);
    sim.set_signal(clk, 1);
    sim.set_signal(resetn, 0);
    sim.tick_many(8);
    sim.set_signal(resetn, 1);

    // Snapshot the per-signal low word each cycle; count how many change. Also track
    // per-signal change counts to see the hot/cold split.
    let snap = |s: &JitSimulator| -> Vec<i64> {
        let w = s.state_words();
        (0..n_signals).map(|i| w[i * 2]).collect()
    };
    let mut prev = snap(&sim);
    let mut changed_per_cycle = 0u64;
    let mut per_signal = vec![0u64; n_signals];
    for _ in 0..cycles {
        sim.tick();
        let cur = snap(&sim);
        for i in 0..n_signals {
            if cur[i] != prev[i] {
                changed_per_cycle += 1;
                per_signal[i] += 1;
            }
        }
        prev = cur;
    }

    let avg_changed = changed_per_cycle as f64 / cycles as f64;
    let never = per_signal.iter().filter(|&&c| c == 0).count();
    let rare = per_signal.iter().filter(|&&c| c > 0 && (c as f64) < cycles as f64 * 0.05).count();
    let hot = per_signal.iter().filter(|&&c| (c as f64) >= cycles as f64 * 0.5).count();
    println!("picorv32 SoC temporal sparsity over {cycles} cycles ({n_signals} signals):");
    println!("  avg signals changing per cycle: {avg_changed:.1} / {n_signals} ({:.1}%)", 100.0 * avg_changed / n_signals as f64);
    println!("  signals NEVER changing        : {never} ({:.0}%)", 100.0 * never as f64 / n_signals as f64);
    println!("  signals changing <5% of cycles: {rare} ({:.0}%)", 100.0 * rare as f64 / n_signals as f64);
    println!("  signals changing >=50% of cyc : {hot} ({:.0}%)", 100.0 * hot as f64 / n_signals as f64);
    println!("  => activity-skip ceiling ≈ {:.0}% of per-cycle signal work is redundant", 100.0 * (1.0 - avg_changed / n_signals as f64));
}
