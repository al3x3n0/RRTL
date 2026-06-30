//! Import a gate-level (Yosys-JSON) netlist and run it batched — the link from
//! "any synthesized design" (Yosys `synth; abc; write_json`) to RRTL's batch
//! engines. All nets are 1-bit, so the design is also bit-parallel-eligible (the
//! ~96×-Verilator gate-level moat); we run it on the verified SIMD CPU batch
//! engine here and check every lane bit-exact against the scalar oracle.
//!
//! NOTE: the bit-parallel *interp* currently mis-settles the multi-level comb-
//! wire chains a netlist produces (gate-by-gate `n4 = a&b; n6 = n4^q; reg <= n6`)
//! — it captures the register from a not-yet-settled wire. The hand-written gate
//! benches use inline exprs and never hit it; importing exposes the gap. Tracked
//! separately; the SIMD/JIT/AOT batch engines settle the chains correctly.
//! Build: cargo run --release -p rrtl-sim-ir --example netlist_batch
use rrtl_sim_ir::{lower_to_packed_program, SimdCpuSimulator};
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
    let h = |n: &str| compiled.find_module("acc1").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
    let lanes = 256;

    let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
    cpu.set_signal(h("clk"), &vec![1u128; lanes]).unwrap();
    let av: Vec<u128> = (0..lanes).map(|l| (l % 2) as u128).collect();
    let bv: Vec<u128> = (0..lanes).map(|l| ((l / 2) % 2) as u128).collect();
    cpu.set_signal(h("a"), &av).unwrap();
    cpu.set_signal(h("b"), &bv).unwrap();

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
        cpu.tick().unwrap();
        let cq = cpu.get_signal(h("q")).unwrap();
        for l in 0..lanes {
            scal[l].tick();
            if (cq[l] & 1) != (scal[l].get(h("q")) & 1) {
                mism += 1;
            }
        }
    }
    println!("imported gate netlist `acc1` ({} nets) → SIMD CPU batch, {lanes} lanes × 32 cyc", program.signals.len());
    println!("  batch vs scalar oracle (per lane): {}", if mism == 0 { "BIT-EXACT" } else { "MISMATCH" });
    assert_eq!(mism, 0);
}
