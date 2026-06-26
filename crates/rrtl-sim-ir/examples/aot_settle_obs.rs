//! Validate + measure the observability-sliced settle: picorv32 on the AOT (whose
//! per-cycle output reads now use settle_obs — stores only output ports) vs the JIT
//! oracle, asserting a bit-exact mem-bus transaction trace, and reporting AOT
//! throughput. Run twice to A/B: normal (settle_obs) vs AOT_NOOBS=1 (full settle).
//! Build: cargo run --release --features "aot jit" -p rrtl-sim-ir --example aot_settle_obs -- [N]
fn main() {
    #[cfg(not(all(feature = "aot", feature = "jit")))]
    println!("build with --features \"aot jit\"");
    #[cfg(all(feature = "aot", feature = "jit"))]
    run();
}

#[cfg(all(feature = "aot", feature = "jit"))]
fn run() {
    use rrtl_sim_ir::{aot::AotSimulator, jit::JitSimulator, lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;
    use std::time::Instant;

    let n: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(2_000_000);
    let prog = rrtl_riscv_asm::assemble(
        "li x1,0\n li x2,0\n loop: addi x1,x1,1\n addi x2,x2,3\n xor x3,x1,x2\n j loop\n",
    ).expect("assemble");

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
    let (i_clk, i_resetn, i_ready, i_rdata) = (idx("clk"), idx("resetn"), idx("mem_ready"), idx("mem_rdata"));
    let (o_v, o_a, o_w, o_d) = (idx("mem_valid"), idx("mem_addr"), idx("mem_wstrb"), idx("mem_wdata"));

    // Drive the mem-bus handshake; engine is a closure over get/set/tick so the same
    // loop runs the AOT and the JIT. Returns (Mcyc/s, fnv1a over the bus trace).
    macro_rules! run_engine {
        ($sim:expr, $get:ident, $set:ident, $tick:ident) => {{
            let sim = &mut $sim;
            let mut mem = vec![0u32; 4096];
            mem[..prog.len()].copy_from_slice(&prog);
            let mut prev_ready = 0u64;
            let mut hash = 0xcbf29ce484222325u64;
            let start = Instant::now();
            for c in 0..n {
                let resetn = (c >= 4) as u64;
                let valid = sim.$get(o_v);
                let addr = sim.$get(o_a);
                let wstrb = sim.$get(o_w);
                let wdata = sim.$get(o_d);
                let ready = (valid != 0 && prev_ready == 0) as u64;
                let mut rdata = 0u64;
                if ready != 0 {
                    for v in [addr, wstrb, wdata] { hash = (hash ^ v).wrapping_mul(0x100000001b3); }
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
                    } else { rdata = mem[widx] as u64; }
                }
                prev_ready = ready;
                sim.$set(i_clk, 1);
                sim.$set(i_resetn, resetn);
                sim.$set(i_ready, ready);
                sim.$set(i_rdata, rdata);
                sim.$tick();
            }
            (n as f64 / start.elapsed().as_secs_f64() / 1e6, hash)
        }};
    }

    let mut aot = AotSimulator::compile(&machine).expect("aot compile");
    let mut jit = JitSimulator::compile(&machine).expect("jit compile");
    let (aot_mc, aot_h) = run_engine!(aot, get_signal, set_signal, tick);
    let (_jit_mc, jit_h) = run_engine!(jit, get_signal, set_signal, tick);

    let mode = if std::env::var("AOT_NOOBS").is_ok() { "full settle (AOT_NOOBS)" } else { "settle_obs (obs-sliced)" };
    println!("picorv32 AOT settle mode: {mode}");
    println!("  AOT throughput      : {aot_mc:.2} Mcyc/s");
    println!("  bit-exact vs JIT    : {}", if aot_h == jit_h { "YES" } else { "NO — MISMATCH" });
    assert_eq!(aot_h, jit_h, "AOT (settle_obs) diverged from JIT oracle");
}
