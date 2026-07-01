//! Import a REAL Yosys-synthesized gate netlist (bench/sv/crc32_gates.json, 195
//! cells from `synth; dfflegalize; abc -g AND,OR,XOR,MUX`) and check it computes
//! the correct CRC-32 (vs the algorithm) cycle-for-cycle. Then run the imported
//! netlist on the bit-parallel batch engine (the ~96×-Verilator moat).
//! Regenerate the netlist with: yosys -q bench/sv/synth_crc32.ys
//! Build: cargo run --release -p rrtl-sim-ir --example netlist_crc32
use rrtl_sim_ir::bitparallel::BitParallelSimulator;
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
use rrtl_sv_frontend::import_yosys_netlist;

fn main() {
    // Gate netlist (1-bit ports: din_0..din_7, crc_0..crc_31, clk, rst).
    let json = std::fs::read_to_string("bench/sv/crc32_gates.json").expect("read netlist");
    let gates = import_yosys_netlist(&json, "crc32").expect("import netlist");
    let gc = rrtl_core::compile(&gates).unwrap();
    let gh = |n: &str| gc.find_module("crc32").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;

    // Reference: crc32.sv's exact 1-cycle recurrence (Yosys keeps the blocking
    // temp `c` combinational, so the netlist is a 1-cycle CRC — RRTL's frontend
    // instead registers `c`, giving a 2-cycle pipeline, so we compare against the
    // algorithm directly rather than the RRTL-behavioral build).
    fn crc_step(mut c: u32, din: u32) -> u32 {
        for i in 0..8 {
            let bit = ((c >> 31) & 1) ^ ((din >> i) & 1);
            c = (c << 1) ^ if bit == 1 { 0x04C1_1DB7 } else { 0 };
        }
        c
    }
    println!("imported crc32 gate netlist: {} cells → {} RRTL signals", 195, gc.find_module("crc32").unwrap().signals.len());

    let mut gsim = rrtl_core::Simulator::new(&gates, "crc32").unwrap();
    gsim.set(gh("clk"), 1);
    let read_gate_crc = |s: &rrtl_core::Simulator| -> u128 {
        (0..32).fold(0u128, |acc, i| acc | (s.get(gh(&format!("crc_{i}"))) << i))
    };
    let mut refc: u32 = 0;
    let mut mism = 0;
    for cyc in 0..64u64 {
        let rst = (cyc < 1) as u128;
        let din = (cyc.wrapping_mul(2654435761) >> 13) & 0xff;
        gsim.set(gh("rst"), rst);
        for i in 0..8 {
            gsim.set(gh(&format!("din_{i}")), ((din >> i) & 1) as u128);
        }
        gsim.tick();
        refc = if rst == 1 { 0xFFFF_FFFF } else { crc_step(refc, din as u32) };
        if read_gate_crc(&gsim) != refc as u128 {
            mism += 1;
        }
    }
    println!("  netlist vs CRC-32 reference (64 cyc): {}", if mism == 0 { "BIT-EXACT" } else { "MISMATCH" });
    assert_eq!(mism, 0, "synthesized netlist diverged from the CRC-32 algorithm");

    // The netlist is all-1-bit → runs on the bit-parallel moat engine.
    let program = lower_to_packed_program(&gc, "crc32").unwrap();
    let machine = lower_to_machine_program(&program);
    let lanes = 256;
    let mut bp = BitParallelSimulator::new(&machine, lanes).expect("gate-level → bit-parallel");
    let idx = |n: &str| program.signal_index(gh(n)).unwrap();
    let mut scal: Vec<rrtl_core::Simulator> = (0..lanes).map(|_| rrtl_core::Simulator::new(&gates, "crc32").unwrap()).collect();
    for l in 0..lanes {
        bp.set_signal(idx("clk"), l, true);
        scal[l].set(gh("clk"), 1);
    }
    let mut bmism = 0;
    for cyc in 0..64u64 {
        let rst = cyc < 1;
        for l in 0..lanes {
            let din = ((cyc + l as u64).wrapping_mul(2654435761) >> 13) & 0xff;
            bp.set_signal(idx("rst"), l, rst);
            scal[l].set(gh("rst"), rst as u128);
            for i in 0..8 {
                bp.set_signal(idx(&format!("din_{i}")), l, (din >> i) & 1 == 1);
                scal[l].set(gh(&format!("din_{i}")), ((din >> i) & 1) as u128);
            }
        }
        bp.tick();
        for l in 0..lanes {
            scal[l].tick();
            let bpcrc = (0..32).fold(0u128, |acc, i| acc | ((bp.get_signal(idx(&format!("crc_{i}")), l) as u128) << i));
            let scrc = (0..32).fold(0u128, |acc, i| acc | (scal[l].get(gh(&format!("crc_{i}"))) << i));
            if bpcrc != scrc {
                bmism += 1;
            }
        }
    }
    println!("  netlist on bit-parallel batch ({lanes} lanes) vs scalar: {}", if bmism == 0 { "BIT-EXACT" } else { "MISMATCH" });
    assert_eq!(bmism, 0);
}
