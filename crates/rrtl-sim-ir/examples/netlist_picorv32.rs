//! Import a REAL picorv32 RISC-V core synthesized to gates (11,735 cells) and run
//! it on the bit-parallel batch engine — a scale test of both the Yosys-JSON
//! importer and the comb-wire-chain settle fix on a full core. Validated by
//! cross-engine self-consistency (bit-parallel == scalar) over a reset sequence.
//! Regenerate: yosys -q bench/sv/synth_picorv32.ys
//! Build: cargo run --release -p rrtl-sim-ir --example netlist_picorv32
use rrtl_sim_ir::bitparallel::BitParallelSimulator;
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
use rrtl_sv_frontend::import_yosys_netlist;
use std::time::Instant;

fn main() {
    // The 6.8 MB netlist is regenerated (not committed): yosys -q bench/sv/synth_picorv32.ys
    let json = match std::fs::read_to_string("bench/sv/picorv32_gates.json") {
        Ok(j) => j,
        Err(_) => {
            println!("picorv32_gates.json not found — regenerate with: yosys -q bench/sv/synth_picorv32.ys");
            return;
        }
    };
    let t = Instant::now();
    let design = import_yosys_netlist(&json, "picorv32").expect("import");
    let compiled = rrtl_core::compile(&design).expect("compile");
    let program = lower_to_packed_program(&compiled, "picorv32").expect("lower");
    let machine = lower_to_machine_program(&program);
    let import_s = t.elapsed().as_secs_f64();
    let sigs = compiled.find_module("picorv32").unwrap().signals.clone();
    let h = |n: &str| sigs.iter().find(|s| s.name == n).map(|s| s.handle);
    println!("picorv32 gate netlist: 11735 cells → {} RRTL signals, imported+lowered in {:.2}s", sigs.len(), import_s);

    let lanes = 16;
    let mut bp = BitParallelSimulator::new(&machine, lanes).expect("gate-level → bit-parallel");
    let idx = |n: &str| h(n).and_then(|hh| program.signal_index(hh));
    let mut scal: Vec<rrtl_core::Simulator> = (0..lanes).map(|_| rrtl_core::Simulator::new(&design, "picorv32").unwrap()).collect();
    // observe a spread of outputs
    let obs: Vec<String> = ["trap", "mem_valid", "mem_instr"].iter().map(|s| s.to_string())
        .chain((0..32).map(|i| format!("mem_addr_{i}")))
        .filter(|n| idx(n).is_some())
        .collect();

    let set_all = |bp: &mut BitParallelSimulator, scal: &mut [rrtl_core::Simulator], name: &str, v: bool| {
        if let Some(ix) = idx(name) {
            for l in 0..lanes {
                bp.set_signal(ix, l, v);
                if let Some(hh) = h(name) {
                    scal[l].set(hh, v as u128);
                }
            }
        }
    };
    // resetn active-low: hold low 3 cycles, then release; clk high; bus idle.
    let mut mism = 0usize;
    let t = Instant::now();
    for cyc in 0..48u64 {
        set_all(&mut bp, &mut scal, "clk", true);
        set_all(&mut bp, &mut scal, "resetn", cyc >= 3);
        bp.tick();
        for l in 0..lanes {
            scal[l].tick();
            for name in &obs {
                let ix = idx(name).unwrap();
                let hh = h(name).unwrap();
                if (bp.get_signal(ix, l) as u128) != (scal[l].get(hh) & 1) {
                    mism += 1;
                }
            }
        }
    }
    let run_s = t.elapsed().as_secs_f64();
    println!("  bit-parallel vs scalar on {} observed outputs, {lanes} lanes × 48 cyc: {} ({:.2}s)",
        obs.len(), if mism == 0 { "BIT-EXACT" } else { "MISMATCH" }, run_s);
    assert_eq!(mism, 0, "picorv32 netlist diverged across engines");
}
