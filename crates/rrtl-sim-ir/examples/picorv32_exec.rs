//! Correctness check: run real RISC-V programs on the imported YosysHQ picorv32
//! core via the Cranelift JIT, driving its native memory interface with a simple
//! 1-cycle-latency RAM. Each program computes a value and stores it to a known
//! address; we observe the store transaction on the memory bus and check it —
//! proving real fetch/decode/ALU/branch/store execution end to end.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example picorv32_exec
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;

    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/picorv32.v".into());
    let src = std::fs::read_to_string(&path).expect("read picorv32.v");
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
    let try_idx = |name: &str| -> Option<usize> {
        let s = compiled
            .find_module("picorv32")
            .unwrap()
            .signals
            .iter()
            .find(|s| s.name == name)?;
        program.signal_index(s.handle)
    };
    let trace_sigs: Vec<(String, usize)> = [
        "reg_op1", "reg_op2", "alu_lts", "alu_eq", "instr_blt", "instr_add", "decoded_rd",
        "reg_out", "latched_rd",
    ]
    .iter()
    .filter_map(|n| try_idx(n).map(|i| (n.to_string(), i)))
    .collect();

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

    // (7 + 5) << 2 = 48 -> mem[0x40]; straight-line ALU + store.
    let prog_alu = rrtl_riscv_asm::assemble(
        "
        addi x1, x0, 7
        addi x2, x0, 5
        add  x3, x1, x2     # 12
        slli x3, x3, 2      # 48
        sw   x3, 0x40(x0)
    spin:
        j spin
        ",
    )
    .expect("assemble prog_alu");
    // sum 1..10 = 55 -> mem[0x44]; backward branch (blt) loop.
    let prog_loop = rrtl_riscv_asm::assemble(
        "
        li   x1, 0          # sum
        li   x2, 1          # i
        li   x3, 11         # limit
    loop:
        add  x1, x1, x2     # sum += i
        addi x2, x2, 1      # i++
        blt  x2, x3, loop   # if i < 11 repeat
        sw   x1, 0x44(x0)
    spin:
        j spin
        ",
    )
    .expect("assemble prog_loop");

    let mut all_ok = true;
    all_ok &= run_program(&machine, &io, &trace_sigs, &prog_alu, 0x40, 48, "(7+5)<<2");
    all_ok &= run_program(&machine, &io, &trace_sigs, &prog_loop, 0x44, 55, "sum 1..10 (branch loop)");
    println!("{}", if all_ok { "ALL [PASS]" } else { "SOME [FAIL]" });
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
fn run_program(
    machine: &rrtl_sim_ir::PackedMachineProgram,
    io: &Io,
    trace_sigs: &[(String, usize)],
    prog: &[u32],
    expect_addr: u64,
    expect_val: u64,
    label: &str,
) -> bool {
    use rrtl_sim_ir::jit::JitSimulator;

    let mut mem = vec![0u32; 4096];
    mem[..prog.len()].copy_from_slice(prog);
    let mut jit = JitSimulator::compile(machine).expect("jit compile");

    let trace = std::env::var("TRACE").is_ok();
    let mut prev_ready = 0u64;
    let max_cycles = 4000usize;
    for c in 0..max_cycles {
        let resetn = (c >= 4) as u64;
        let valid = jit.get_signal(io.mem_valid);
        let addr = jit.get_signal(io.mem_addr);
        if trace && c < 400 && valid != 0 && prev_ready == 0 {
            let ints: String = trace_sigs
                .iter()
                .map(|(n, i)| format!(" {n}={}", jit.get_signal(*i)))
                .collect();
            println!(
                "c={c:3} req addr=0x{addr:x} wstrb={}{ints}",
                jit.get_signal(io.mem_wstrb)
            );
        }
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
                if addr == expect_addr {
                    let pass = wdata == expect_val;
                    println!(
                        "  {label}: mem[0x{addr:x}] <= {wdata} (want {expect_val})  =>  {}",
                        if pass { "[PASS]" } else { "[FAIL]" }
                    );
                    return pass;
                }
            } else {
                rdata = mem.get(widx).copied().unwrap_or(0) as u64;
            }
        }
        prev_ready = ready;

        jit.set_signal(io.clk, 1);
        jit.set_signal(io.resetn, resetn);
        jit.set_signal(io.mem_ready, ready);
        jit.set_signal(io.mem_rdata, rdata);
        jit.tick();
        assert!(jit.get_signal(io.trap) == 0, "{label}: core trapped at cycle {c}");
    }
    println!("  {label}: [FAIL] no store to 0x{expect_addr:x} within {max_cycles} cycles");
    false
}
