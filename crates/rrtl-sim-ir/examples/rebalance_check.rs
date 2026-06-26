//! Does priority-mux-chain rebalancing (depth N→log N) raise picorv32 throughput?
//! Rebalances the lowered machine program, reports the mux-chain depth reduction,
//! and runs the same mem-bus handshake on the plain vs rebalanced program on the
//! JIT (and AOT if built), asserting a bit-exact bus trace. A win means picorv32
//! is latency-bound on the comb mux spine; a wash means it is op-count-bound.
//! Build: cargo run --release --features "jit aot" -p rrtl-sim-ir --example rebalance_check -- [N]
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit (optionally aot)");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program, rebalance_mux_chains_program, PackedBlock, PackedInstrKind, PackedMachineProgram};
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
    let (rb, stats) = rebalance_mux_chains_program(&machine);

    let mux_depth = |b: &PackedBlock| -> usize {
        let mut md = std::collections::HashMap::new();
        let mut best = 0;
        for p in &b.packets {
            for instr in &p.instrs {
                let ops = match &instr.kind {
                    PackedInstrKind::Mux { cond, then_value, else_value } => vec![cond.0, then_value.0, else_value.0],
                    PackedInstrKind::And(a, b) | PackedInstrKind::Or(a, b) | PackedInstrKind::Xor(a, b)
                    | PackedInstrKind::Add(a, b) | PackedInstrKind::Sub(a, b) | PackedInstrKind::Mul(a, b)
                    | PackedInstrKind::Eq(a, b) | PackedInstrKind::Ne(a, b) => vec![a.0, b.0],
                    PackedInstrKind::Lt { lhs, rhs, .. } => vec![lhs.0, rhs.0],
                    PackedInstrKind::Not(a) | PackedInstrKind::Zext(a) | PackedInstrKind::Sext(a)
                    | PackedInstrKind::Trunc(a) | PackedInstrKind::Cast(a) | PackedInstrKind::Slice { value: a, .. } => vec![a.0],
                    PackedInstrKind::Concat(ps) => ps.iter().map(|p| p.0).collect(),
                    PackedInstrKind::MemRead { addr, .. } => vec![addr.0],
                    _ => vec![],
                };
                let child = ops.iter().map(|o| *md.get(o).unwrap_or(&0)).max().unwrap_or(0);
                let d = if matches!(instr.kind, PackedInstrKind::Mux { .. }) { child + 1 } else { child };
                md.insert(instr.dst.0, d);
                best = best.max(d);
            }
        }
        best
    };
    let n_instrs = |m: &PackedMachineProgram| -> usize {
        [&m.streams.async_reset_comb, &m.streams.comb, &m.streams.tick_next, &m.streams.tick_commit]
            .iter().map(|b| b.packets.iter().map(|p| p.instrs.len()).sum::<usize>()).sum()
    };
    println!("picorv32 mux-chain rebalancing:");
    println!("  chains rebalanced     : {} (deepest {}), {} hit-ORs added", stats.chains, stats.deepest, stats.ors_added);
    println!("  comb mux-depth        : {} → {}", mux_depth(&machine.streams.comb), mux_depth(&rb.streams.comb));
    println!("  tick_next mux-depth   : {} → {}", mux_depth(&machine.streams.tick_next), mux_depth(&rb.streams.tick_next));
    println!("  total machine instrs  : {} → {}", n_instrs(&machine), n_instrs(&rb));

    let idx = |name: &str| {
        let h = compiled.find_module("picorv32").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle;
        program.signal_index(h).unwrap()
    };
    let (i_clk, i_resetn, i_ready, i_rdata) = (idx("clk"), idx("resetn"), idx("mem_ready"), idx("mem_rdata"));
    let (o_v, o_a, o_w, o_d) = (idx("mem_valid"), idx("mem_addr"), idx("mem_wstrb"), idx("mem_wdata"));

    macro_rules! bench {
        ($compile:expr) => {{
            let mut sim = $compile;
            let mut mem = vec![0u32; 4096];
            mem[..prog.len()].copy_from_slice(&prog);
            let mut prev_ready = 0u64;
            let mut hash = 0xcbf29ce484222325u64;
            let start = Instant::now();
            for c in 0..n {
                let resetn = (c >= 4) as u64;
                let (valid, addr, wstrb, wdata) = (sim.get_signal(o_v), sim.get_signal(o_a), sim.get_signal(o_w), sim.get_signal(o_d));
                let ready = (valid != 0 && prev_ready == 0) as u64;
                let mut rdata = 0u64;
                if ready != 0 {
                    for v in [addr, wstrb, wdata] { hash = (hash ^ v).wrapping_mul(0x100000001b3); }
                    let widx = ((addr >> 2) as usize) & (mem.len() - 1);
                    if wstrb != 0 {
                        let mut w = mem[widx];
                        for b in 0..4 { if wstrb & (1 << b) != 0 { let sh = b * 8; w = (w & !(0xFFu32 << sh)) | ((((wdata >> sh) & 0xFF) as u32) << sh); } }
                        mem[widx] = w;
                    } else { rdata = mem[widx] as u64; }
                }
                prev_ready = ready;
                sim.set_signal(i_clk, 1); sim.set_signal(i_resetn, resetn);
                sim.set_signal(i_ready, ready); sim.set_signal(i_rdata, rdata);
                sim.tick();
            }
            (n as f64 / start.elapsed().as_secs_f64() / 1e6, hash)
        }};
    }

    use rrtl_sim_ir::jit::JitSimulator;
    let (mut jp, mut jr, mut hp, mut hr) = (0f64, 0f64, 0u64, 0u64);
    for _ in 0..3 {
        let a = bench!(JitSimulator::compile(&machine).unwrap());
        let b = bench!(JitSimulator::compile(&rb).unwrap());
        jp = jp.max(a.0); jr = jr.max(b.0); hp = a.1; hr = b.1;
    }
    println!("  JIT plain → rebalanced: {jp:.2} → {jr:.2} Mcyc/s  ({:.2}x)", jr / jp);
    println!("  JIT bit-exact bus trace: {}", if hp == hr { "YES" } else { "NO — MISMATCH" });
    assert_eq!(hp, hr, "rebalanced JIT diverged");

    #[cfg(feature = "aot")]
    {
        use rrtl_sim_ir::aot::AotSimulator;
        let (mut ap, mut ar, mut ahp, mut ahr) = (0f64, 0f64, 0u64, 0u64);
        for _ in 0..3 {
            let a = bench!(AotSimulator::compile(&machine).unwrap());
            let b = bench!(AotSimulator::compile(&rb).unwrap());
            ap = ap.max(a.0); ar = ar.max(b.0); ahp = a.1; ahr = b.1;
        }
        println!("  AOT plain → rebalanced: {ap:.2} → {ar:.2} Mcyc/s  ({:.2}x)", ar / ap);
        println!("  AOT bit-exact bus trace: {}", if ahp == ahr { "YES" } else { "NO — MISMATCH" });
        assert_eq!(ahp, ahr, "rebalanced AOT diverged");
    }
}
