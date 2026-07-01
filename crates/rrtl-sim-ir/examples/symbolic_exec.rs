//! RTL symbolic execution via BDDs, and its payoff: symbolic ATPG.
//!
//! The symbolic engine runs the same gate schedule as the bit-parallel batch
//! engine, but each signal carries a BDD (a boolean function of symbolic inputs)
//! instead of concrete bits. Three acts:
//!   1. Symbolic-exec a REAL synthesized netlist (crc32) and prove it agrees with
//!      a concrete sim on *every* input assignment (exhaustive equivalence).
//!   2. Symbolic ATPG on a reconvergent-fanout cone: generate a test vector for a
//!      testable fault, and *prove a redundant fault untestable* (its detection
//!      function is the false BDD) — a negative no amount of simulation can show.
//!   3. Generate a test vector for a real crc32 register fault symbolically, then
//!      confirm it on the concrete bit-parallel engine — closing the ATPG →
//!      fault-sim loop (§4g.15).
//!
//! Build: cargo run --release -p rrtl-sim-ir --example symbolic_exec
use rrtl_sim_ir::bitparallel::BitParallelSimulator;
use rrtl_sim_ir::symbolic::SymbolicSimulator;
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
use rrtl_core::{compile, uint, Design};

fn main() {
    act1_exhaustive_equivalence();
    act2_symbolic_atpg_redundancy();
    act3_symbolic_test_meets_concrete();
}

// ─── Act 1: symbolic == concrete on a real netlist, over ALL inputs ──────────
fn act1_exhaustive_equivalence() {
    println!("── Act 1: symbolic exec of the crc32 netlist ≡ concrete, exhaustively ──");
    let Ok(json) = std::fs::read_to_string("bench/sv/crc32_gates.json") else {
        println!("  skip: bench/sv/crc32_gates.json not present");
        return;
    };
    let gates = rrtl_sv_frontend::import_yosys_netlist(&json, "crc32").expect("import");
    let gc = compile(&gates).unwrap();
    let program = lower_to_packed_program(&gc, "crc32").unwrap();
    let machine = lower_to_machine_program(&program);
    let gh = |n: &str| gc.find_module("crc32").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
    let idx = |n: &str| program.signal_index(gh(n)).unwrap();

    // Symbolic: din_0..7 are variables 0..7; initial crc state = 0 (all FALSE);
    // rst deasserted, one clock. The 32 crc outputs become BDDs over the 8 din vars.
    let mut sym = SymbolicSimulator::new(&machine).unwrap();
    sym.set_const(idx("clk"), true);
    sym.set_const(idx("rst"), false);
    for i in 0..8 {
        sym.set_var(idx(&format!("din_{i}")), i as u32);
    }
    sym.tick();
    let out: Vec<_> = (0..32).map(|i| sym.get(idx(&format!("crc_{i}")))).collect();
    let sizes: usize = out.iter().map(|&b| sym.mgr.size(b)).sum();
    println!("  32 crc output BDDs over 8 din vars, total {sizes} nodes");

    // Concrete oracle: rrtl_core scalar sim, same initial state, all 256 din.
    let mut mism = 0;
    for din in 0..256u32 {
        let mut c = rrtl_core::Simulator::new(&gates, "crc32").unwrap();
        c.set(gh("clk"), 1);
        c.set(gh("rst"), 0);
        for i in 0..8 {
            c.set(gh(&format!("din_{i}")), ((din >> i) & 1) as u128);
        }
        c.tick();
        for i in 0..32 {
            let sym_bit = sym.mgr.eval(out[i], |v| (din >> v) & 1 == 1);
            let con_bit = c.get(gh(&format!("crc_{i}"))) & 1 == 1;
            if sym_bit != con_bit {
                mism += 1;
            }
        }
    }
    println!("  symbolic vs concrete over all 256 din × 32 outputs: {}", if mism == 0 { "BIT-EXACT" } else { "MISMATCH" });
    assert_eq!(mism, 0);
}

// ─── Act 2: symbolic ATPG — test generation and a provable redundancy ────────
fn act2_symbolic_atpg_redundancy() {
    println!("── Act 2: symbolic ATPG on a reconvergent cone (y = (a&b) | (a&~b) = a) ──");
    // y is functionally `a`; the ~b→(a&~b) path is redundant, so a stuck-at-1 on
    // the `nb` wire is UNTESTABLE, while stuck-at-0 there IS testable.
    let mut design = Design::new();
    {
        let mut d = design.module("Redund");
        let a = d.input("a", uint(1));
        let b = d.input("b", uint(1));
        let nb = d.wire("nb", uint(1));
        let w = d.wire("w", uint(1));
        let v = d.wire("v", uint(1));
        let y = d.output("y", uint(1));
        d.assign(nb, !b.value());
        d.assign(w, a.value() & b.value());
        d.assign(v, a.value() & nb.value());
        d.assign(y, w.value() | v.value());
    }
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Redund").unwrap();
    let machine = lower_to_machine_program(&program);
    let h = |n: &str| compiled.find_module("Redund").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
    let idx = |n: &str| program.signal_index(h(n)).unwrap();

    // A detection function for a stuck-at fault on `signal`: golden_y XOR faulty_y,
    // both BDDs in the same manager so the XOR is valid.
    let detect = |signal: usize, stuck: bool| -> (u32, SymbolicSimulator, Vec<String>) {
        let mut s = SymbolicSimulator::new(&machine).unwrap();
        s.set_var(idx("a"), 0);
        s.set_var(idx("b"), 1);
        s.settle();
        let golden = s.get(idx("y"));
        s.set_force_const(signal, stuck);
        s.settle();
        let faulty = s.get(idx("y"));
        let d = s.mgr.xor(golden, faulty);
        let vecs = s.mgr.sat_one(d).map(|a| a.iter().map(|(v, x)| format!("{}={}", ["a", "b"][*v as usize], *x as u8)).collect()).unwrap_or_default();
        (d, s, vecs)
    };

    // Faults on internal wires (where a stuck-at clamp is well-defined; forcing a
    // primary input is just constraining that input, not a fault, so we don't).
    for (name, signal, stuck) in [
        ("nb", idx("nb"), true),  // redundant: y already = a
        ("nb", idx("nb"), false), // testable
        ("w", idx("w"), false),   // testable
        ("v", idx("v"), true),    // testable
    ] {
        let (d, s, test) = detect(signal, stuck);
        if s.mgr.is_false(d) {
            println!("  {name} stuck-at-{}: detection BDD = FALSE → PROVABLY UNTESTABLE (redundant)", stuck as u8);
        } else {
            println!("  {name} stuck-at-{}: testable — generated test vector [{}]", stuck as u8, test.join(", "));
        }
    }
    // Correctness assertions on the known-redundant structure.
    assert!(detect(idx("nb"), true).1.mgr.is_false(detect(idx("nb"), true).0), "nb s-a-1 must be redundant");
    assert!(!detect(idx("nb"), false).1.mgr.is_false(detect(idx("nb"), false).0), "nb s-a-0 must be testable");
}

// ─── Act 3: symbolic-generated test, confirmed on the concrete engine ────────
fn act3_symbolic_test_meets_concrete() {
    println!("── Act 3: symbolic ATPG for a crc32 register fault → confirmed concretely ──");
    let Ok(json) = std::fs::read_to_string("bench/sv/crc32_gates.json") else {
        println!("  skip: netlist absent");
        return;
    };
    let gates = rrtl_sv_frontend::import_yosys_netlist(&json, "crc32").expect("import");
    let gc = compile(&gates).unwrap();
    let program = lower_to_packed_program(&gc, "crc32").unwrap();
    let machine = lower_to_machine_program(&program);
    let gh = |n: &str| gc.find_module("crc32").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
    let idx = |n: &str| program.signal_index(gh(n)).unwrap();

    // Symbolic over din (vars 0..7) AND the 32 initial crc flops (var 8+k for crc
    // bit k), so a test is a full (state, din) vector. crc is GF(2)-linear → the
    // BDDs stay small. Setup helper drives inputs + symbolic initial state.
    let setup = |s: &mut SymbolicSimulator| {
        s.set_const(idx("clk"), true);
        s.set_const(idx("rst"), false);
        for i in 0..8 {
            s.set_var(idx(&format!("din_{i}")), i as u32);
        }
        for k in 0..32 {
            s.set_var(idx(&format!("crc_{k}")), 8 + k as u32);
        }
    };

    // One sim, one manager: settle golden, snapshot output BDDs, THEN inject the
    // fault and settle again — both live in the same manager so the XOR is valid.
    let mut s = SymbolicSimulator::new(&machine).unwrap();
    setup(&mut s);
    s.tick();
    let golden: Vec<u32> = (0..32).map(|i| s.get(idx(&format!("crc_{i}")))).collect();

    let fault_bit = 5usize;
    let fault_sig = idx(&format!("crc_{fault_bit}")); // the crc flop == output net
    setup(&mut s);
    s.set_force_const(fault_sig, false); // crc_5 stuck-at-0
    s.tick();
    let faulty: Vec<u32> = (0..32).map(|i| s.get(idx(&format!("crc_{i}")))).collect();

    // Detection = OR over outputs of (golden_o XOR faulty_o).
    let mut d = 0u32; // FALSE
    for i in 0..32 {
        let x = s.mgr.xor(golden[i], faulty[i]);
        d = s.mgr.or(d, x);
    }
    let test = s.mgr.sat_one(d).expect("crc bit-5 stuck-at-0 must be testable");
    let bit = |v: u32| test.iter().find(|(x, _)| *x == v).map(|(_, x)| *x).unwrap_or(false);
    let din: u32 = (0..8).fold(0, |acc, i| acc | ((bit(i) as u32) << i));
    let state_word: u32 = (0..32).fold(0, |acc, k| acc | ((bit(8 + k) as u32) << k));
    println!("  symbolic test for crc_{fault_bit} stuck-at-0: din=0x{din:02x}, init_state=0x{state_word:08x}");

    // ── Concrete confirmation on the bit-parallel engine (register force works) ──
    let mut bp = BitParallelSimulator::new(&machine, 2).unwrap();
    // lane 0 golden, lane 1 faulted; drive the symbolic test into both.
    for l in 0..2 {
        bp.set_signal(idx("clk"), l, true);
        bp.set_signal(idx("rst"), l, false);
        for i in 0..8 {
            bp.set_signal(idx(&format!("din_{i}")), l, (din >> i) & 1 == 1);
        }
        for k in 0..32 {
            bp.set_signal(idx(&format!("crc_{k}")), l, (state_word >> k) & 1 == 1);
        }
    }
    bp.set_force(fault_sig, 1, false); // lane 1: crc_5 stuck-at-0
    bp.tick();
    let read = |bp: &BitParallelSimulator, l: usize| -> u32 {
        (0..32).fold(0, |acc, i| acc | ((bp.get_signal(idx(&format!("crc_{i}")), l) as u32) << i))
    };
    let (g, f) = (read(&bp, 0), read(&bp, 1));
    println!("  concrete bit-parallel: golden=0x{g:08x}, faulty=0x{f:08x} → {}", if g != f { "DETECTED (symbolic test confirmed)" } else { "NOT detected" });
    assert_ne!(g, f, "symbolic-generated test failed to detect the fault concretely");
}
