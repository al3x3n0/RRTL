//! Throughput benchmark: run the full YosysHQ picorv32 core through the RRTL
//! Cranelift JIT for a fixed number of clock cycles and report Mcyc/s. Emits the
//! program as a `$readmemh` hex file so the Verilator harness (bench/sv/
//! run_picorv32.sh) runs the *identical* program + memory protocol for a fair
//! comparison.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example picorv32_bench -- [N]
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

// Shared workload: an infinite compute loop (3 ALU ops + a jump per iteration).
// Runs forever so the benchmark can clock it for an arbitrary cycle count.
#[cfg(feature = "jit")]
const PROGRAM_ASM: &str = "
    li   x1, 0
    li   x2, 0
loop:
    addi x1, x1, 1
    addi x2, x2, 3
    xor  x3, x1, x2
    j    loop
";

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::{jit::JitSimulator, lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;

    let n: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(5_000_000);

    let prog = rrtl_riscv_asm::assemble(PROGRAM_ASM).expect("assemble program");
    // Emit the shared $readmemh program (one 32-bit word per line).
    let hex: String = prog.iter().map(|w| format!("{w:08x}\n")).collect();
    let hex_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv/picorv32_prog.hex");
    std::fs::write(hex_path, &hex).expect("write hex");

    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv/picorv32.v"))
        .expect("read picorv32.v");
    let imported = import_sv(&src, Some("picorv32")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "picorv32").expect("lower packed");
    let machine = lower_to_machine_program(&program);

    let idx = |name: &str| {
        let h = compiled
            .find_module("picorv32")
            .unwrap()
            .signals
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("no signal {name}"))
            .handle;
        program.signal_index(h).unwrap()
    };
    let (i_clk, i_resetn) = (idx("clk"), idx("resetn"));
    let (i_mem_ready, i_mem_rdata) = (idx("mem_ready"), idx("mem_rdata"));
    let (o_mem_valid, o_mem_addr) = (idx("mem_valid"), idx("mem_addr"));
    let (o_mem_wstrb, o_mem_wdata) = (idx("mem_wstrb"), idx("mem_wdata"));

    // Optional fixed memory latency (stall cycles per access) to contrast a busy
    // tight loop (MEM_LAT=0) with a stall-heavy workload (MEM_LAT=16).
    let mem_lat: u64 = std::env::var("MEM_LAT").ok().and_then(|s| s.parse().ok()).unwrap_or(0);

    let ports = Ports {
        i_clk, i_resetn, i_mem_ready, i_mem_rdata,
        o_mem_valid, o_mem_addr, o_mem_wstrb, o_mem_wdata,
    };

    // Oblivious JIT (the headline number written to compare vs Verilator).
    let mut plain = JitSimulator::compile(&machine).expect("jit compile");
    let (mc0, f0, _) = drive(&mut plain, &prog, &ports, n, mem_lat);
    println!(
        "RRTL-JIT picorv32: {n} cycles, {f0} mem reads, {mc0:.2} Mcyc/s (MEM_LAT={mem_lat})"
    );

    // Activity-skipping JIT (instrumented to report realized skip rate).
    let mut act =
        JitSimulator::compile_activity_instrumented(&machine).expect("activity jit compile");
    let (mc1, f1, _) = drive(&mut act, &prog, &ports, n, mem_lat);
    let rate = act.activity_skip_rate().unwrap_or(0.0);
    let agree = if f0 == f1 { "OK" } else { "MISMATCH!" };
    println!(
        "RRTL-JIT+activity:  {n} cycles, {f1} mem reads ({agree}), {mc1:.2} Mcyc/s, skip {:.1}% => {:.2}x vs oblivious",
        rate * 100.0,
        mc1 / mc0,
    );
}

#[cfg(feature = "jit")]
struct Ports {
    i_clk: usize,
    i_resetn: usize,
    i_mem_ready: usize,
    i_mem_rdata: usize,
    o_mem_valid: usize,
    o_mem_addr: usize,
    o_mem_wstrb: usize,
    o_mem_wdata: usize,
}

#[cfg(feature = "jit")]
fn drive(
    jit: &mut rrtl_sim_ir::jit::JitSimulator,
    prog: &[u32],
    p: &Ports,
    n: usize,
    mem_lat: u64,
) -> (f64, u64, u64) {
    use std::time::Instant;
    let mut mem = vec![0u32; 4096];
    mem[..prog.len()].copy_from_slice(prog);
    let mut prev_ready = 0u64;
    let mut wait = 0u64;
    let mut fetches = 0u64;
    let start = Instant::now();
    for c in 0..n {
        let resetn = (c >= 4) as u64;
        let valid = jit.get_signal(p.o_mem_valid);
        let addr = jit.get_signal(p.o_mem_addr);
        let wstrb = jit.get_signal(p.o_mem_wstrb);
        let wdata = jit.get_signal(p.o_mem_wdata);

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

        jit.set_signal(p.i_clk, 1);
        jit.set_signal(p.i_resetn, resetn);
        jit.set_signal(p.i_mem_ready, ready);
        jit.set_signal(p.i_mem_rdata, rdata);
        jit.tick();
    }
    let dt = start.elapsed().as_secs_f64();
    (n as f64 / dt / 1e6, fetches, 0)
}
