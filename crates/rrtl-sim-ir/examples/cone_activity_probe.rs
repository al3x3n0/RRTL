//! Does picorv32's 91% signal-sparsity translate to skippable register CONES? A
//! cone is "active" this cycle if any of its combinational fan-in (traced back to
//! registers/inputs) changed last cycle — that is what an activity-skip mechanism
//! could actually skip, vs the looser signal-change rate. If active-cone rate is
//! low, the skip win is real; if cones are interconnected (decoder feeds all), high.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example cone_activity_probe -- [cycles]
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::jit::JitSimulator;
    use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program, PackedBlock, PackedEffect, PackedInstrKind};
    use rrtl_sv_frontend::import_sv;
    use std::collections::{HashMap, HashSet};

    let cycles: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(5000);
    let prog = rrtl_riscv_asm::assemble(
        "li x5,0\n li x6,1\n li x7,200\n loop: add x5,x5,x6\n addi x6,x6,1\n blt x6,x7,loop\n li x8,0x100\n sw x5,0(x8)\n spin: j spin\n",
    ).expect("assemble");
    let core = std::fs::read_to_string("bench/sv/picorv32.v").expect("read picorv32.v");
    let soc = std::fs::read_to_string("bench/sv/picorv32_soc.v").expect("read picorv32_soc.v");
    let imported = import_sv(&format!("{core}\n{soc}\n"), Some("picorv32_soc")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "picorv32_soc").expect("lower");
    let n = program.signals.len();
    let machine = lower_to_machine_program(&program);

    // operand value-ids of an instr kind
    let ops = |k: &PackedInstrKind| -> Vec<usize> {
        use PackedInstrKind::*;
        match k {
            Lit(_) | Signal(_) => vec![],
            Not(a) | Zext(a) | Sext(a) | Trunc(a) | Cast(a) | Slice { value: a, .. } => vec![a.0],
            And(a,b)|Or(a,b)|Xor(a,b)|Add(a,b)|Sub(a,b)|Mul(a,b)|Eq(a,b)|Ne(a,b) => vec![a.0,b.0],
            Lt{lhs,rhs,..} => vec![lhs.0,rhs.0],
            Mux{cond,then_value,else_value} => vec![cond.0,then_value.0,else_value.0],
            Concat(p) => p.iter().map(|x| x.0).collect(),
            MemRead{addr,..} => vec![addr.0],
        }
    };
    // signals each value-id transitively reads, within a block
    let value_sig_reads = |block: &PackedBlock| -> HashMap<usize, HashSet<usize>> {
        let mut vr: HashMap<usize, HashSet<usize>> = HashMap::new();
        for p in &block.packets {
            for instr in &p.instrs {
                let mut s = HashSet::new();
                if let PackedInstrKind::Signal(sig) = &instr.kind {
                    s.insert(*sig);
                }
                for o in ops(&instr.kind) {
                    if let Some(os) = vr.get(&o) {
                        s.extend(os.iter().copied());
                    }
                }
                vr.insert(instr.dst.0, s);
            }
        }
        vr
    };

    // direct signal-deps: comb signal -> signals it reads; register -> next-state reads
    let comb_vr = value_sig_reads(&machine.streams.comb);
    let arc_vr = value_sig_reads(&machine.streams.async_reset_comb);
    let next_vr = value_sig_reads(&machine.streams.tick_next);
    let mut direct: HashMap<usize, HashSet<usize>> = HashMap::new(); // signal -> directly-read signals
    let mut is_reg = vec![false; n];
    for (block, vr) in [(&machine.streams.comb, &comb_vr), (&machine.streams.async_reset_comb, &arc_vr)] {
        for p in &block.packets {
            for e in &p.effects {
                if let PackedEffect::StoreSignal { dst, value } = e {
                    direct.entry(*dst).or_default().extend(vr.get(&value.0).cloned().unwrap_or_default());
                }
            }
        }
    }
    let mut reg_next: HashMap<usize, HashSet<usize>> = HashMap::new();
    for p in &machine.streams.tick_next.packets {
        for e in &p.effects {
            if let PackedEffect::CaptureReg { dst, value, .. } = e {
                is_reg[*dst] = true;
                reg_next.insert(*dst, next_vr.get(&value.0).cloned().unwrap_or_default());
            }
        }
    }

    // cone(reg) = registers/inputs reached by tracing next-state deps through comb
    let regs: Vec<usize> = (0..n).filter(|&i| is_reg[i]).collect();
    let cone_of = |start: &HashSet<usize>| -> HashSet<usize> {
        let mut leaves = HashSet::new();
        let mut seen = HashSet::new();
        let mut stack: Vec<usize> = start.iter().copied().collect();
        while let Some(s) = stack.pop() {
            if !seen.insert(s) { continue; }
            if is_reg[s] || direct.get(&s).map_or(true, |d| d.is_empty()) {
                leaves.insert(s); // a register or a primary input/leaf
                if let Some(d) = direct.get(&s) { for &x in d { stack.push(x); } }
            } else {
                for &x in &direct[&s] { stack.push(x); }
            }
        }
        leaves
    };
    let cones: Vec<(usize, HashSet<usize>)> = regs.iter()
        .map(|&r| (r, cone_of(reg_next.get(&r).unwrap_or(&HashSet::new())))).collect();
    let self_driving = cones.iter().filter(|(r, c)| c.contains(r)).count();

    // run; each cycle, mark cone active if any fan-in signal changed last cycle
    let idx = |nm: &str| {
        let h = compiled.find_module("picorv32_soc").unwrap().signals.iter().find(|s| s.name == nm).unwrap().handle;
        program.signal_index(h).unwrap()
    };
    let mem_idx = program.memories.iter().position(|m| m.name == "mem" || m.name.ends_with(".mem")).unwrap();
    let mut sim = JitSimulator::compile(&machine).unwrap();
    let mut w = vec![0u128; 1024];
    for (i, x) in prog.iter().enumerate() { w[i] = *x as u128; }
    sim.set_memory(mem_idx, &w);
    sim.set_signal(idx("clk"), 1);
    sim.set_signal(idx("resetn"), 0);
    sim.tick_many(8);
    sim.set_signal(idx("resetn"), 1);

    let snap = |s: &JitSimulator| -> Vec<i64> { let sw = s.state_words(); (0..n).map(|i| sw[i*2]).collect() };
    let mut prev = snap(&sim);
    let mut active_sum = 0u64;
    for _ in 0..cycles {
        sim.tick();
        let cur = snap(&sim);
        let changed: HashSet<usize> = (0..n).filter(|&i| cur[i] != prev[i]).collect();
        for (r, c) in &cones {
            if c.contains(r) || c.iter().any(|s| changed.contains(s)) {
                active_sum += 1;
                let _ = r;
            }
        }
        prev = cur;
    }
    let avg_active = active_sum as f64 / cycles as f64;
    println!("picorv32 cone activity over {cycles} cycles ({} register cones, {self_driving} self-driving):", regs.len());
    println!("  avg ACTIVE cones per cycle: {avg_active:.1} / {} ({:.1}%)", regs.len(), 100.0 * avg_active / regs.len() as f64);
    println!("  => capturable cone-skip ceiling ≈ {:.0}% (vs 91% at signal grain)", 100.0 * (1.0 - avg_active / regs.len() as f64));
}
