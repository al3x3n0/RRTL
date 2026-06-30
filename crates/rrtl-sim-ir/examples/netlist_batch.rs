//! Import a gate-level (Yosys-JSON) netlist and run it on the bit-parallel batch
//! engine — the concrete link from "any synthesized design" (Yosys
//! `synth; abc; write_json`) to RRTL's ~96×-Verilator gate-level moat. All nets
//! are 1-bit, so BitParallelSimulator applies directly; checked bit-exact vs the
//! scalar oracle. (The multi-level comb-wire chains a netlist produces now settle
//! correctly — the interp applies each comb packet's stores before the next.)
//! Build: cargo run --release -p rrtl-sim-ir --example netlist_batch
use rrtl_sim_ir::bitparallel::BitParallelSimulator;
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
use rrtl_sv_frontend::import_yosys_netlist;

fn main() {
    // q <= (a & b) ^ q ; output q. nets: 1=clk 2=a 3=b 4=t 5=q 6=d
    let json = r#"{ "modules": { "acc1": {
        "ports": {
          "clk": {"direction":"input","bits":[1]},
          "a":   {"direction":"input","bits":[2]},
          "b":   {"direction":"input","bits":[3]},
          "q":   {"direction":"output","bits":[5]}
        },
        "cells": {
          "u_and": {"type":"$_AND_","connections":{"A":[2],"B":[3],"Y":[4]}},
          "u_xor": {"type":"$_XOR_","connections":{"A":[4],"B":[5],"Y":[6]}},
          "u_dff": {"type":"$_DFF_P_","connections":{"C":[1],"D":[6],"Q":[5]}}
        }
    }}}"#;
    let design = import_yosys_netlist(json, "acc1").expect("import netlist");
    let compiled = rrtl_core::compile(&design).expect("compile");
    let program = lower_to_packed_program(&compiled, "acc1").expect("lower");
    let machine = lower_to_machine_program(&program);
    let h = |n: &str| compiled.find_module("acc1").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
    let idx = |n: &str| program.signal_index(h(n)).unwrap();
    let lanes = 256;

    let mut bp = BitParallelSimulator::new(&machine, lanes).expect("gate-level → bit-parallel");
    let av: Vec<u128> = (0..lanes).map(|l| (l % 2) as u128).collect();
    let bv: Vec<u128> = (0..lanes).map(|l| ((l / 2) % 2) as u128).collect();
    for l in 0..lanes {
        bp.set_signal(idx("clk"), l, true);
        bp.set_signal(idx("a"), l, av[l] != 0);
        bp.set_signal(idx("b"), l, bv[l] != 0);
    }
    // ground-truth scalar oracle per distinct input combo
    let mut scal: Vec<rrtl_core::Simulator> = (0..lanes)
        .map(|l| {
            let mut s = rrtl_core::Simulator::new(&design, "acc1").unwrap();
            s.set(h("clk"), 1);
            s.set(h("a"), av[l]);
            s.set(h("b"), bv[l]);
            s
        })
        .collect();

    let mut mism = 0;
    for _ in 0..32 {
        bp.tick();
        for l in 0..lanes {
            scal[l].tick();
            if (bp.get_signal(idx("q"), l) as u128) != (scal[l].get(h("q")) & 1) {
                mism += 1;
            }
        }
    }
    println!("imported gate netlist `acc1` ({} nets) → BIT-PARALLEL batch, {lanes} lanes × 32 cyc", program.signals.len());
    println!("  bit-parallel vs scalar oracle (per lane): {}", if mism == 0 { "BIT-EXACT" } else { "MISMATCH" });
    assert_eq!(mism, 0);
}
