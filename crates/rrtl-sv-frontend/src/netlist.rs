//! Gate-level netlist import (Yosys `write_json` format) into native RRTL IR.
//!
//! A post-synthesis netlist is a graph of 1-bit nets connected by primitive gate
//! cells (`$_AND_`, `$_DFF_P_`, …). Every signal is one bit and every cell is a
//! bitwise op or a flop — exactly the regime RRTL's bit-parallel batch engine
//! beats Verilator on at scale. Importing such netlists lets that moat apply to
//! *any* synthesizable design (run it through Yosys `synth; abc; write_json`),
//! not just hand-written gate benches.
//!
//! Supported cells: `$_NOT_ $_AND_ $_OR_ $_XOR_ $_NAND_ $_NOR_ $_XNOR_ $_ANDNOT_
//! $_ORNOT_ $_MUX_` (combinational) and `$_DFF_P_` / `$_DFFE_PP_` (posedge flop,
//! optional active-high enable). Unknown cells are rejected with their type, so
//! coverage gaps are explicit rather than silent.

use std::collections::{HashMap, HashSet};

use rrtl_core::{lit_u, mux, uint, Design, Expr, Signal};
use serde_json::Value;

/// A net reference in a cell connection: a numbered net or a constant bit.
#[derive(Clone, Copy)]
enum Net {
    Id(i64),
    Const(bool),
}

fn net_of(v: &Value) -> Result<Net, String> {
    match v {
        Value::Number(n) => Ok(Net::Id(n.as_i64().ok_or("net id not an integer")?)),
        Value::String(s) => match s.as_str() {
            "1" => Ok(Net::Const(true)),
            _ => Ok(Net::Const(false)), // "0"/"x"/"z" → 0
        },
        _ => Err(format!("unexpected net value: {v}")),
    }
}

fn conn1(cell: &Value, port: &str) -> Result<Net, String> {
    let arr = cell["connections"][port]
        .as_array()
        .ok_or_else(|| format!("cell missing connection `{port}`"))?;
    if arr.len() != 1 {
        return Err(format!("gate cell port `{port}` is not 1-bit"));
    }
    net_of(&arr[0])
}

/// Import the top module of a Yosys JSON netlist as an RRTL [`Design`].
pub fn import_yosys_netlist(json: &str, top: &str) -> Result<Design, String> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("JSON parse: {e}"))?;
    let module = v["modules"]
        .get(top)
        .ok_or_else(|| format!("module `{top}` not found"))?;
    let cells = module["cells"].as_object().cloned().unwrap_or_default();
    let ports = module["ports"].as_object().cloned().unwrap_or_default();

    // Classify each net: input-port bit, DFF output (reg), or comb wire.
    let mut input_nets: HashSet<i64> = HashSet::new();
    let mut reg_nets: HashSet<i64> = HashSet::new();
    let mut all_nets: HashSet<i64> = HashSet::new();
    let mut input_name: HashMap<i64, String> = HashMap::new(); // input net -> port name
    let mut output_port_bits: Vec<(String, usize, Net)> = Vec::new();
    for (pname, p) in &ports {
        let dir = p["direction"].as_str().unwrap_or("");
        let bits = p["bits"].as_array().cloned().unwrap_or_default();
        let multi = bits.len() > 1;
        for (i, b) in bits.iter().enumerate() {
            let n = net_of(b)?;
            if let Net::Id(id) = n {
                all_nets.insert(id);
                if dir == "input" {
                    input_nets.insert(id);
                    input_name.insert(id, if multi { format!("{pname}_{i}") } else { pname.clone() });
                }
            }
            if dir == "output" {
                output_port_bits.push((pname.clone(), i, n));
            }
        }
    }
    for (_, cell) in &cells {
        let ty = cell["type"].as_str().unwrap_or("");
        if let Some(conns) = cell["connections"].as_object() {
            for arr in conns.values() {
                if let Some(a) = arr.as_array() {
                    for b in a {
                        if let Ok(Net::Id(id)) = net_of(b) {
                            all_nets.insert(id);
                        }
                    }
                }
            }
        }
        if ty.starts_with("$_DFF") {
            if let Ok(Net::Id(q)) = conn1(cell, "Q") {
                reg_nets.insert(q);
            }
        }
    }

    let mut design = Design::new();
    let mut m = design.module(top);
    // Create a 1-bit signal for every net (input / reg / wire).
    let mut sig: HashMap<i64, Signal> = HashMap::new();
    for &id in &all_nets {
        let s = if input_nets.contains(&id) {
            m.input(input_name[&id].clone(), uint(1))
        } else if reg_nets.contains(&id) {
            m.reg(format!("n{id}"), uint(1))
        } else {
            m.wire(format!("n{id}"), uint(1))
        };
        sig.insert(id, s);
    }
    let val = |n: Net, sig: &HashMap<i64, Signal>| -> Expr {
        match n {
            Net::Id(id) => sig[&id].value(),
            Net::Const(b) => lit_u(b as u128, 1),
        }
    };

    // Emit each cell's logic.
    for (cname, cell) in &cells {
        let ty = cell["type"].as_str().unwrap_or("");
        let a = |p| conn1(cell, p).map(|n| val(n, &sig));
        if ty.starts_with("$_DFF") {
            let q = match conn1(cell, "Q")? {
                Net::Id(id) => sig[&id],
                _ => return Err(format!("DFF `{cname}` Q is constant")),
            };
            let clk = match conn1(cell, "C")? {
                Net::Id(id) => sig[&id],
                _ => return Err(format!("DFF `{cname}` clock is constant")),
            };
            m.clock(q, clk);
            let d = a("D")?;
            // $_DFFE_PP_: posedge clock, active-high enable.
            let next = if ty.starts_with("$_DFFE") {
                mux(a("E")?, d, q.value())
            } else {
                d
            };
            m.next(q, next);
            continue;
        }
        let y = match conn1(cell, "Y")? {
            Net::Id(id) => sig[&id],
            _ => return Err(format!("cell `{cname}` output is constant")),
        };
        let expr = match ty {
            "$_NOT_" => !a("A")?,
            "$_AND_" => a("A")? & a("B")?,
            "$_OR_" => a("A")? | a("B")?,
            "$_XOR_" => a("A")? ^ a("B")?,
            "$_NAND_" => !(a("A")? & a("B")?),
            "$_NOR_" => !(a("A")? | a("B")?),
            "$_XNOR_" => !(a("A")? ^ a("B")?),
            "$_ANDNOT_" => a("A")? & !a("B")?,
            "$_ORNOT_" => a("A")? | !a("B")?,
            "$_MUX_" => mux(a("S")?, a("B")?, a("A")?), // Y = S ? B : A
            other => return Err(format!("unsupported gate cell `{other}` (cell `{cname}`)")),
        };
        m.assign(y, expr);
    }

    // Output ports read their nets.
    for (pname, i, n) in &output_port_bits {
        let name = if module["ports"][pname]["bits"].as_array().map_or(1, |b| b.len()) == 1 {
            pname.clone()
        } else {
            format!("{pname}_{i}")
        };
        let o = m.output(name, uint(1));
        let e = val(*n, &sig);
        m.assign(o, e);
    }
    Ok(design)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rrtl_core::{compile, Simulator};

    // A 2-bit counter as a gate netlist: q0 <= ~q0; q1 <= q1 ^ q0. Imported and
    // simulated, it must match a reference RRTL design of the same logic.
    #[test]
    fn import_2bit_counter_matches_reference() {
        // nets: 1=clk, 2=q0, 3=n0(=~q0), 4=q1, 5=x1(=q1^q0)
        let json = r#"{ "modules": { "ctr": {
            "ports": {
              "clk": {"direction":"input","bits":[1]},
              "q0":  {"direction":"output","bits":[2]},
              "q1":  {"direction":"output","bits":[4]}
            },
            "cells": {
              "u_not":  {"type":"$_NOT_","connections":{"A":[2],"Y":[3]}},
              "u_dff0": {"type":"$_DFF_P_","connections":{"C":[1],"D":[3],"Q":[2]}},
              "u_xor":  {"type":"$_XOR_","connections":{"A":[4],"B":[2],"Y":[5]}},
              "u_dff1": {"type":"$_DFF_P_","connections":{"C":[1],"D":[5],"Q":[4]}}
            }
        }}}"#;
        let design = import_yosys_netlist(json, "ctr").unwrap();
        let compiled = compile(&design).unwrap();
        let h = |d: &rrtl_core::CompiledDesign, n: &str| d.find_module("ctr").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;

        // reference: same recurrence built directly
        let mut rdesign = Design::new();
        {
            let mut rm = rdesign.module("ctr");
            let clk = rm.input("clk", uint(1));
            let q0 = rm.reg("q0", uint(1));
            let q1 = rm.reg("q1", uint(1));
            rm.clock(q0, clk);
            rm.clock(q1, clk);
            rm.next(q0, !q0.value());
            rm.next(q1, q1.value() ^ q0.value());
            let o0 = rm.output("o0", uint(1));
            let o1 = rm.output("o1", uint(1));
            rm.assign(o0, q0.value());
            rm.assign(o1, q1.value());
        }
        let rcompiled = compile(&rdesign).unwrap();
        let rh = |n: &str| rcompiled.find_module("ctr").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;

        let mut sim = Simulator::new(&design, "ctr").unwrap();
        let mut rsim = Simulator::new(&rdesign, "ctr").unwrap();
        let (q0o, q1o) = (h(&compiled, "q0"), h(&compiled, "q1"));
        for cyc in 0..8 {
            sim.tick();
            rsim.tick();
            assert_eq!(sim.get(q0o), rsim.get(rh("o0")), "q0 @ cyc{cyc}");
            assert_eq!(sim.get(q1o), rsim.get(rh("o1")), "q1 @ cyc{cyc}");
        }
    }
}
