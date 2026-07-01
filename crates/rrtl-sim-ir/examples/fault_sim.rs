//! Fault simulation on the bit-parallel batch engine — the "F" in the batch-axis
//! thesis (fuzz / DSE / **fault-sim**). One stuck-at fault per lane runs in
//! parallel against a fault-free golden lane; a fault is *detected* when its
//! observed outputs ever diverge from golden. This is the moat's canonical
//! killer app: a P-lane u64 engine grades P faults for the price of one
//! (fault-free) simulation, which is exactly what an ATPG coverage sweep needs.
//!
//! Mechanism: `BitParallelSimulator::set_force(sig, lane, value)` clamps a
//! register net every cycle (re-applied after commit so the stuck value
//! propagates through the combinational cone into the outputs). Lane 0 is
//! golden; lanes 1.. each carry one stuck-at-0/1 fault on a flop.
//!
//! Design: bench/sv/crc32_gates.json (195 cells, 32 flops). Validated: (a) the
//! golden lane matches a fault-free scalar reference, (b) coverage is reported.
//! Build: cargo run --release -p rrtl-sim-ir --example fault_sim
use rrtl_sim_ir::bitparallel::BitParallelSimulator;
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program, PackedSignalKind};

fn main() {
    let json = std::fs::read_to_string("bench/sv/crc32_gates.json").expect("read netlist");
    let gates = import_or_die(&json);
    let gc = rrtl_core::compile(&gates).unwrap();
    let program = lower_to_packed_program(&gc, "crc32").unwrap();
    let machine = lower_to_machine_program(&program);

    // Enumerate fault sites (flops) and observation points (output ports) by kind.
    // Position in `program.signals` IS the bit-parallel state index.
    let regs: Vec<(usize, &str)> = program
        .signals
        .iter()
        .enumerate()
        .filter(|(_, s)| s.kind == PackedSignalKind::Reg)
        .map(|(i, s)| (i, s.name.as_str()))
        .collect();
    let gh = |n: &str| gc.find_module("crc32").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
    let idx = |n: &str| program.signal_index(gh(n)).unwrap();
    let (clk, rst) = (idx("clk"), idx("rst"));
    let din: Vec<usize> = (0..8).map(|i| idx(&format!("din_{i}"))).collect();
    let out_idx: Vec<usize> = (0..32).map(|i| idx(&format!("crc_{i}"))).collect();

    // Fault list: stuck-at-0 and stuck-at-1 on every flop → lane f+1 carries fault f.
    let faults: Vec<(usize, &str, bool)> = regs
        .iter()
        .flat_map(|&(i, n)| [(i, n, false), (i, n, true)])
        .collect();
    let lanes = faults.len() + 1; // lane 0 = golden
    println!(
        "crc32 gate netlist: {} flops → {} stuck-at faults, {} lanes ({} golden + faulty)",
        regs.len(),
        faults.len(),
        lanes,
        1
    );

    let mut bp = BitParallelSimulator::new(&machine, lanes).expect("gate-level → bit-parallel");
    for (f, &(sig, _, stuck)) in faults.iter().enumerate() {
        bp.set_force(sig, f + 1, stuck); // lane 0 left untouched (golden)
    }

    // Fault-free scalar oracle for the golden lane.
    let mut gold = rrtl_core::Simulator::new(&gates, "crc32").unwrap();
    gold.set(gh("clk"), 1);
    bp.set_signal_all(clk, true);

    let mut detected = vec![false; faults.len()];
    let mut first_cycle = vec![0u64; faults.len()];
    let mut golden_mism = 0u64;
    let cycles = 96u64;
    for cyc in 0..cycles {
        let rst_v = cyc < 1;
        let din_v = (cyc.wrapping_mul(2654435761) >> 13) & 0xff;
        bp.set_signal_all(rst, rst_v);
        gold.set(gh("rst"), rst_v as u128);
        for i in 0..8 {
            bp.set_signal_all(din[i], (din_v >> i) & 1 == 1);
            gold.set(gh(&format!("din_{i}")), ((din_v >> i) & 1) as u128);
        }
        bp.tick();
        gold.tick();

        // Golden lane must equal the fault-free scalar sim.
        let read_bp = |bp: &BitParallelSimulator, lane: usize| -> u128 {
            out_idx.iter().enumerate().fold(0u128, |acc, (b, &o)| acc | ((bp.get_signal(o, lane) as u128) << b))
        };
        let g_scalar = (0..32).fold(0u128, |acc, b| acc | (gold.get(gh(&format!("crc_{b}"))) << b));
        if read_bp(&bp, 0) != g_scalar {
            golden_mism += 1;
        }
        // A fault is detected the first cycle its lane's outputs differ from golden.
        let g = read_bp(&bp, 0);
        for f in 0..faults.len() {
            if !detected[f] && read_bp(&bp, f + 1) != g {
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
        "  fault coverage: {}/{} detected = {:.1}%  (in ONE {}-lane batch sweep)",
        n_det,
        faults.len(),
        100.0 * n_det as f64 / faults.len() as f64,
        lanes
    );
    // Detection latency proves the faults propagate through the logic over
    // varying cone depths (not a trivial one-cycle mask).
    let lats: Vec<u64> = (0..faults.len()).filter(|&f| detected[f]).map(|f| first_cycle[f] + 1).collect();
    if !lats.is_empty() {
        let (mn, mx) = (*lats.iter().min().unwrap(), *lats.iter().max().unwrap());
        let mean = lats.iter().sum::<u64>() as f64 / lats.len() as f64;
        println!("  detection latency (cycles to first divergence): min {mn}, max {mx}, mean {mean:.1}");
    }
    // Report any undetected faults (untestable / redundant under this stimulus).
    for (f, &(_, name, stuck)) in faults.iter().enumerate() {
        if !detected[f] {
            println!("    UNDETECTED: {name} stuck-at-{}", stuck as u8);
        }
    }
}

fn import_or_die(json: &str) -> rrtl_core::Design {
    rrtl_sv_frontend::import_yosys_netlist(json, "crc32").expect("import netlist")
}
