//! Fault simulation at SCALE on a real RISC-V core: the synthesized picorv32
//! gate netlist (11,735 cells, 1,597 flops → 3,194 stuck-at faults) graded in a
//! SINGLE batch sweep on the bit-parallel engine — 3,194 faults for the cost of
//! one fault-free simulation (§4g.15 mechanism, §4g.4b engine).
//!
//! Unlike crc32 (every flop a fully-observed CRC output → 100% coverage), a CPU
//! under a bounded stimulus leaves many nets un-excited or un-observable, so
//! coverage is genuinely < 100% — which is the point: it shows the sim
//! *discriminates* detected from undetected faults, and reports the latter.
//!
//! Stimulus: a fixed pseudo-random "instruction" stream on mem_rdata with
//! mem_ready held high — identical across all lanes (same-stimulus is the
//! fault-sim invariant; only the injected fault differs per lane). We observe the
//! architecturally-visible memory interface (mem_valid/addr/wdata/wstrb/instr,
//! trap). This is a scale + discrimination demo, NOT an ATPG-grade test set;
//! coverage is stimulus-limited by construction.
//!
//! Regenerate the netlist: yosys -q bench/sv/synth_picorv32.ys  (gitignored, 6.8MB)
//! Build: cargo run --release -p rrtl-sim-ir --example fault_sim_picorv32
use rrtl_sim_ir::bitparallel::BitParallelSimulator;
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program, PackedSignalKind};

fn main() {
    let path = "bench/sv/picorv32_gates.json";
    let Ok(json) = std::fs::read_to_string(path) else {
        println!("skip: {path} not present (gitignored 6.8MB). Generate: yosys -q bench/sv/synth_picorv32.ys");
        return;
    };
    let gates = rrtl_sv_frontend::import_yosys_netlist(&json, "picorv32").expect("import netlist");
    let gc = rrtl_core::compile(&gates).unwrap();
    let program = lower_to_packed_program(&gc, "picorv32").unwrap();
    let machine = lower_to_machine_program(&program);
    let gh = |n: &str| gc.find_module("picorv32").unwrap().signals.iter().find(|s| s.name == n).map(|s| s.handle);
    let idx = |n: &str| gh(n).and_then(|h| program.signal_index(h));

    // Fault sites: every flop, both polarities. Position in program.signals ==
    // bit-parallel state index == signal_index.
    let regs: Vec<usize> = program
        .signals
        .iter()
        .enumerate()
        .filter(|(_, s)| s.kind == PackedSignalKind::Reg)
        .map(|(i, _)| i)
        .collect();
    let faults: Vec<(usize, bool)> = regs.iter().flat_map(|&i| [(i, false), (i, true)]).collect();
    let lanes = faults.len() + 1; // lane 0 = golden

    // Observation: the architecturally-visible memory interface (multi-bit ports
    // are split into name_i by the importer).
    let mut obs: Vec<usize> = Vec::new();
    let mut push = |n: &str, out: &mut Vec<usize>| {
        if let Some(i) = idx(n) {
            out.push(i);
        }
    };
    for n in ["mem_valid", "mem_instr", "trap"] {
        push(n, &mut obs);
    }
    for (base, w) in [("mem_addr", 32), ("mem_wdata", 32), ("mem_wstrb", 4)] {
        for b in 0..w {
            push(&format!("{base}_{b}"), &mut obs);
        }
    }
    assert!(obs.len() <= 128, "observation vector must fit in u128");
    println!(
        "picorv32 gate netlist: {} cells, {} flops → {} stuck-at faults in ONE {}-lane batch",
        11_735,
        regs.len(),
        faults.len(),
        lanes
    );
    println!("  observing {} memory-interface output bits", obs.len());

    let mut bp = BitParallelSimulator::new(&machine, lanes).expect("gate-level → bit-parallel");
    for (f, &(sig, stuck)) in faults.iter().enumerate() {
        bp.set_force(sig, f + 1, stuck);
    }

    // Fault-free scalar oracle for the golden lane.
    let mut gold = rrtl_core::Simulator::new(&gates, "picorv32").unwrap();
    let set_gold = |g: &mut rrtl_core::Simulator, n: &str, v: u128| {
        if let Some(h) = gh(n) {
            g.set(h, v);
        }
    };
    set_gold(&mut gold, "clk", 1);
    if let Some(c) = idx("clk") {
        bp.set_signal_all(c, true);
    }

    // Sustained LEGAL compute-AND-STORE stimulus (deterministic; identical across
    // lanes). Each instruction is held 8 cycles so every fetch sees a stable word
    // regardless of fetch timing; we ignore the fetch address (infinite straight-
    // line program). The 4-instruction loop pins an aligned base pointer, evolves
    // an accumulator, mixes it, then STORES it — the store drives the computed
    // value onto the observed `mem_wdata`/`mem_wstrb` bus, so register-file and ALU
    // faults become architecturally observable (an addi-only stream exposes only
    // fetch/PC/control faults, since nothing computed ever reaches the bus).
    //   x2 is only ever written by slot 0 (= 16), so `sw x1,0(x2)` is always
    //   word-aligned even if fetch timing repeats/skips a slot — no misalign trap.
    let stim = |c: u64| -> u32 {
        let n = c / 32; // loop iteration (4 instrs × 8-cycle hold)
        match (c / 8) % 4 {
            0 => 0x0100_0113,                                        // addi x2, x0, 16   (pin aligned base)
            1 => (((n.wrapping_mul(2654435761) >> 20) as u32 & 0xfff) << 20) | 0x0000_8093, // addi x1, x1, imm
            2 => 0x0020_C0B3,                                        // xor  x1, x1, x2
            _ => 0x0011_2023,                                        // sw   x1, 0(x2)    (expose x1 on the bus)
        }
    };
    let read_obs = |bp: &BitParallelSimulator, lane: usize| -> u128 {
        obs.iter().enumerate().fold(0u128, |acc, (b, &o)| acc | ((bp.get_signal(o, lane) as u128) << b))
    };
    let read_obs_gold = |g: &rrtl_core::Simulator| -> u128 {
        let names: Vec<String> = ["mem_valid", "mem_instr", "trap"]
            .iter()
            .map(|s| s.to_string())
            .chain((0..32).map(|b| format!("mem_addr_{b}")))
            .chain((0..32).map(|b| format!("mem_wdata_{b}")))
            .chain((0..4).map(|b| format!("mem_wstrb_{b}")))
            .collect();
        names.iter().enumerate().fold(0u128, |acc, (b, n)| {
            let v = gh(n).map(|h| g.get(h)).unwrap_or(0);
            acc | ((v & 1) << b)
        })
    };

    let mut detected = vec![false; faults.len()];
    let mut first_cycle = vec![0u64; faults.len()];
    let mut golden_mism = 0u64;
    let cycles = 400u64;
    for cyc in 0..cycles {
        let resetn = (cyc >= 4) as u128; // active-low reset for the first 4 cycles
        let rd = stim(cyc);
        set_gold(&mut gold, "resetn", resetn);
        set_gold(&mut gold, "mem_ready", 1);
        for i in 0..32 {
            set_gold(&mut gold, &format!("mem_rdata_{i}"), ((rd >> i) & 1) as u128);
        }
        if let Some(i) = idx("resetn") {
            bp.set_signal_all(i, resetn == 1);
        }
        if let Some(i) = idx("mem_ready") {
            bp.set_signal_all(i, true);
        }
        for i in 0..32 {
            if let Some(s) = idx(&format!("mem_rdata_{i}")) {
                bp.set_signal_all(s, (rd >> i) & 1 == 1);
            }
        }
        bp.tick();
        gold.tick();

        if read_obs(&bp, 0) != read_obs_gold(&gold) {
            golden_mism += 1;
        }
        let g = read_obs(&bp, 0);
        for f in 0..faults.len() {
            if !detected[f] && read_obs(&bp, f + 1) != g {
                detected[f] = true;
                first_cycle[f] = cyc;
            }
        }
    }

    let n_det = detected.iter().filter(|d| **d).count();
    println!(
        "  golden lane vs fault-free scalar ({cycles} cyc): {}",
        if golden_mism == 0 { "BIT-EXACT" } else { "MISMATCH" }
    );
    assert_eq!(golden_mism, 0, "fault injection corrupted the golden lane");
    println!(
        "  fault coverage: {}/{} detected = {:.1}%  (stimulus-limited; {} undetected under this stream)",
        n_det,
        faults.len(),
        100.0 * n_det as f64 / faults.len() as f64,
        faults.len() - n_det
    );
    let lats: Vec<u64> = (0..faults.len()).filter(|&f| detected[f]).map(|f| first_cycle[f] + 1).collect();
    if !lats.is_empty() {
        let (mn, mx) = (*lats.iter().min().unwrap(), *lats.iter().max().unwrap());
        let mean = lats.iter().sum::<u64>() as f64 / lats.len() as f64;
        println!("  detection latency (cycles to first divergence): min {mn}, max {mx}, mean {mean:.1}");
    }
    println!(
        "  → {} faults graded in one batch sweep = the cost of a single fault-free simulation",
        faults.len()
    );
}
