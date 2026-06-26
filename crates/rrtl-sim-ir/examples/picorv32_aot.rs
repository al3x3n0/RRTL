//! picorv32 throughput on the AOT (clang -O3) backend, same program + memory
//! protocol as picorv32_bench / the Verilator harness — the apples-to-apples
//! "RRTL IR through -O3" number for the paper.
//! Build: cargo run --release --features aot -p rrtl-sim-ir --example picorv32_aot -- [N]
fn main() {
    #[cfg(not(feature = "aot"))]
    println!("build with --features aot");
    #[cfg(feature = "aot")]
    run();
}

#[cfg(feature = "aot")]
fn run() {
    use rrtl_sim_ir::{aot::AotSimulator, lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;
    use std::time::Instant;

    let n: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(5_000_000);
    let mem_lat: u64 = std::env::var("MEM_LAT").ok().and_then(|s| s.parse().ok()).unwrap_or(0);

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

    let idx = |name: &str| {
        let h = compiled.find_module("picorv32").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle;
        program.signal_index(h).unwrap()
    };
    let (i_clk, i_resetn) = (idx("clk"), idx("resetn"));
    let (i_mem_ready, i_mem_rdata) = (idx("mem_ready"), idx("mem_rdata"));
    let (o_mem_valid, o_mem_addr) = (idx("mem_valid"), idx("mem_addr"));
    let (o_mem_wstrb, o_mem_wdata) = (idx("mem_wstrb"), idx("mem_wdata"));

    let t0 = Instant::now();
    let mut sim = AotSimulator::compile(&machine).expect("aot compile");
    println!("AOT clang -O3 compile: {:.0} ms ({} signals)", t0.elapsed().as_secs_f64() * 1e3, sim.signal_count());

    let mut mem = vec![0u32; 4096];
    mem[..prog.len()].copy_from_slice(&prog);
    let mut prev_ready = 0u64;
    let mut wait = 0u64;
    let mut fetches = 0u64;
    let start = Instant::now();
    for c in 0..n {
        let resetn = (c >= 4) as u64;
        let valid = sim.get_signal(o_mem_valid);
        let addr = sim.get_signal(o_mem_addr);
        let wstrb = sim.get_signal(o_mem_wstrb);
        let wdata = sim.get_signal(o_mem_wdata);
        if valid != 0 && prev_ready == 0 && wait < mem_lat {
            wait += 1;
        }
        let ready = (valid != 0 && prev_ready == 0 && wait >= mem_lat) as u64;
        if ready != 0 {
            wait = 0;
        }
        let mut rdata = 0u64;
        if ready != 0 {
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
    let dt = start.elapsed().as_secs_f64();
    println!(
        "AOT picorv32: {n} cycles, {fetches} mem reads, {:.2} Mcyc/s (MEM_LAT={mem_lat})",
        n as f64 / dt / 1e6
    );
}
