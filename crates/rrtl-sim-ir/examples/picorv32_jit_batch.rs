//! Batch N picorv32 SoC instances on the SCALAR JIT, one independent native-code
//! instance per core (bulk-synchronous: each runs its whole trace, sync once at the
//! join). This is the RIGHT batch engine for a control-heavy CPU — unlike SIMD
//! lanes it has no per-lane memory gather and tolerates divergence. The fair
//! comparison vs N Verilator processes (both native, both parallel across cores).
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example picorv32_jit_batch -- [instances] [cycles]
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use rayon::prelude::*;
    use rrtl_sim_ir::jit::JitSimulator;
    use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;
    use std::time::Instant;

    let instances: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(256);
    let cycles: usize = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(50_000);

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
    let machine = lower_to_machine_program(&program);
    let module = compiled.find_module("picorv32_soc").unwrap();
    let idx = |n: &str| {
        let h = module.signals.iter().find(|s| s.name == n).unwrap().handle;
        program.signal_index(h).unwrap()
    };
    let (clk, resetn, result, trap) = (idx("clk"), idx("resetn"), idx("result"), idx("trap"));
    let mem_index = program.memories.iter().position(|m| m.name == "mem" || m.name.ends_with(".mem")).unwrap();

    let template = JitSimulator::compile(&machine).expect("jit compile");
    let mem_base = template.memory_word_base(mem_index);
    let state_len = template.state_words().len();
    let tick = template.tick_many_fn_ptr();

    // one independent state buffer per instance, each preloaded with its own program
    let mut buffers: Vec<Vec<i64>> = (0..instances)
        .map(|l| {
            let mut buf = vec![0i64; state_len];
            buf[clk * 2] = 1; // clk=1 (each tick is an edge)
            let prog = prog_for(4 + (l as u32 % 60));
            for (k, w) in prog.iter().enumerate() {
                buf[mem_base + k * 2] = *w as i64;
            }
            buf
        })
        .collect();

    let run_one = |buf: &mut [i64]| unsafe {
        buf[resetn * 2] = 0;
        tick(buf.as_mut_ptr(), 8);
        buf[resetn * 2] = 1;
        tick(buf.as_mut_ptr(), cycles as i64);
    };

    // correctness: instance 0 (limit 4 ⇒ sum 1..3 = 6); validate before timing
    {
        let mut b0 = buffers[0].clone();
        run_one(&mut b0);
        assert_eq!(b0[trap * 2], 0, "instance 0 trapped");
        // limit=4 ⇒ sum of i=1,2,3 = 6
        assert_eq!(b0[result * 2], 6, "instance 0 wrong result: {}", b0[result * 2]);
    }

    let t = Instant::now();
    buffers.par_iter_mut().for_each(|buf| run_one(buf));
    let secs = t.elapsed().as_secs_f64();
    let no_trap = buffers.iter().filter(|b| b[trap * 2] == 0).count();
    let inst_cyc = (instances * cycles) as f64;
    println!("picorv32 SoC batch on SCALAR JIT — {instances} instances (independent programs), {cycles} cycles, rayon");
    println!("  {no_trap}/{instances} ran without trap");
    println!("  total: {:.1} M-instance-cycles/s  ({:.2} M-cyc/s per instance avg)",
        inst_cyc / secs / 1e6, inst_cyc / secs / 1e6 / instances as f64);
}
