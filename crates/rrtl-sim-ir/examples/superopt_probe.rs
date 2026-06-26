//! Measure-first sizing for an RTL SUPEROPTIMIZER: does optimal rewriting find
//! op savings BEYOND the existing fusion/specialize passes?
//!
//! Method: build the ground-truth optimal min-op-count form of EVERY boolean
//! function of ≤3 variables (a complete DP over all 256 functions — this is the
//! exact answer a superoptimizer would synthesize for small cones). Then harvest
//! the actual 1-bit combinational cones from a real design's machine IR (AFTER
//! running the current specialize+fuse pipeline), and for each cone with ≤3
//! distinct leaves, compare its op-count to the optimal. The residual savings is
//! the superoptimizer's headroom on small cones (it finds MORE on larger ones).
//! Build: cargo run --release --features aot -p rrtl-sim-ir --example superopt_probe -- [picorv32.v] [top]
use rrtl_sim_ir::{
    lower_to_machine_program, lower_to_packed_program, PackedBlock, PackedInstr, PackedInstrKind,
    PackedMachineProgram, PackedValueId,
};
use rrtl_sim_ir::specialize::specialize_program;
use rrtl_sv_frontend::import_sv;
use std::collections::HashMap;

/// Optimal AND/OR/XOR/NOT op-count for every 3-variable boolean function.
/// Truth table = u8 (8 rows = 2^3); 256 functions. Cost = #binary/unary gates.
fn optimal_costs() -> [u8; 256] {
    let mut cost = [u8::MAX; 256];
    // Standard 3-var leaf truth tables + the two constants (cost 0).
    for l in [0xAAu8, 0xCC, 0xF0, 0x00, 0xFF] {
        cost[l as usize] = 0;
    }
    loop {
        let mut changed = false;
        let known: Vec<u8> = (0..256u16).filter(|&i| cost[i as usize] != u8::MAX).map(|i| i as u8).collect();
        for &a in &known {
            let ca = cost[a as usize];
            let n = !a;
            if cost[n as usize] > ca + 1 {
                cost[n as usize] = ca + 1;
                changed = true;
            }
            for &b in &known {
                let c = ca.saturating_add(cost[b as usize]).saturating_add(1);
                for r in [a & b, a | b, a ^ b] {
                    if cost[r as usize] > c {
                        cost[r as usize] = c;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    cost
}

fn def_map(block: &PackedBlock) -> (HashMap<PackedValueId, usize>, Vec<PackedInstr>) {
    let mut flat = Vec::new();
    let mut def = HashMap::new();
    for p in &block.packets {
        for instr in &p.instrs {
            def.insert(instr.dst, flat.len());
            flat.push(instr.clone());
        }
    }
    (def, flat)
}

/// Recursively evaluate a 1-bit cone for a given assignment of its leaves
/// (leaf value-id → bit). Non-boolean / multi-bit / unknown ops are treated as
/// opaque leaves. Returns the bit, or None if the cone isn't pure-boolean.
fn eval(
    v: PackedValueId,
    def: &HashMap<PackedValueId, usize>,
    instrs: &[PackedInstr],
    leaf_bits: &HashMap<PackedValueId, u8>,
) -> Option<u8> {
    if let Some(&b) = leaf_bits.get(&v) {
        return Some(b);
    }
    let instr = &instrs[*def.get(&v)?];
    use PackedInstrKind::*;
    Some(match &instr.kind {
        And(a, b) => eval(*a, def, instrs, leaf_bits)? & eval(*b, def, instrs, leaf_bits)?,
        Or(a, b) => eval(*a, def, instrs, leaf_bits)? | eval(*b, def, instrs, leaf_bits)?,
        Xor(a, b) => eval(*a, def, instrs, leaf_bits)? ^ eval(*b, def, instrs, leaf_bits)?,
        Not(a) => 1 - eval(*a, def, instrs, leaf_bits)?,
        Eq(a, b) => (eval(*a, def, instrs, leaf_bits)? == eval(*b, def, instrs, leaf_bits)?) as u8,
        Ne(a, b) => (eval(*a, def, instrs, leaf_bits)? != eval(*b, def, instrs, leaf_bits)?) as u8,
        Mux { cond, then_value, else_value } => {
            if eval(*cond, def, instrs, leaf_bits)? == 1 {
                eval(*then_value, def, instrs, leaf_bits)?
            } else {
                eval(*else_value, def, instrs, leaf_bits)?
            }
        }
        _ => return None,
    })
}

/// Collect the boolean cone of a 1-bit value: its op-count and the distinct
/// leaves (value-ids whose op is not a 1-bit boolean op). None if not boolean.
fn cone(
    v: PackedValueId,
    def: &HashMap<PackedValueId, usize>,
    instrs: &[PackedInstr],
    widths: &[u32],
    leaves: &mut Vec<PackedValueId>,
    seen: &mut std::collections::HashSet<PackedValueId>,
    ops: &mut usize,
) -> Option<()> {
    let Some(&di) = def.get(&v) else {
        if !leaves.contains(&v) {
            leaves.push(v);
        }
        return Some(());
    };
    let instr = &instrs[di];
    use PackedInstrKind::*;
    let kids: Vec<PackedValueId> = match &instr.kind {
        And(a, b) | Or(a, b) | Xor(a, b) | Eq(a, b) | Ne(a, b) => vec![*a, *b],
        Not(a) => vec![*a],
        Mux { cond, then_value, else_value } => vec![*cond, *then_value, *else_value],
        // any other op (Slice/Concat/Signal/Lit/arith…) → opaque leaf
        _ => {
            if !leaves.contains(&v) {
                leaves.push(v);
            }
            return Some(());
        }
    };
    // Eq/Ne are only boolean when 1-bit operands; require the result is 1-bit.
    if instr.ty.width != 1 {
        if !leaves.contains(&v) {
            leaves.push(v);
        }
        return Some(());
    }
    if seen.insert(v) {
        *ops += 1;
        for k in kids {
            cone(k, def, instrs, widths, leaves, seen, ops)?;
        }
    }
    Some(())
}

fn run(machine: &PackedMachineProgram, label: &str, opt: &[u8; 256]) {
    let widths: Vec<u32> = machine.source.signals.iter().map(|s| s.layout.width).collect();
    let mut harvested = 0usize;
    let mut le3 = 0usize;
    let mut wins = 0usize;
    let mut saved = 0usize;
    let mut cur_ops = 0usize;
    let mut big = 0usize; // pure-boolean cones with >3 leaves (the egg/SMT domain)
    let mut big_ops = 0usize;
    let mut big_leaves = 0usize;
    for blk in [&machine.streams.comb, &machine.streams.tick_next] {
        let (def, instrs) = def_map(blk);
        // roots = each instr that produces a 1-bit boolean op (the cone tops).
        for instr in &instrs {
            if instr.ty.width != 1 {
                continue;
            }
            if !matches!(
                instr.kind,
                PackedInstrKind::And(..) | PackedInstrKind::Or(..) | PackedInstrKind::Xor(..)
                    | PackedInstrKind::Not(..) | PackedInstrKind::Mux { .. }
            ) {
                continue;
            }
            harvested += 1;
            let mut leaves = Vec::new();
            let mut seen = std::collections::HashSet::new();
            let mut ops = 0usize;
            if cone(instr.dst, &def, &instrs, &widths, &mut leaves, &mut seen, &mut ops).is_none() {
                continue;
            }
            if ops < 2 {
                continue;
            }
            if leaves.len() > 3 {
                // larger pure-boolean cone — the multi-input restructuring domain
                // a peephole DP can't reach but egg/SMT synthesis can.
                big += 1;
                big_ops += ops;
                big_leaves += leaves.len();
                continue;
            }
            le3 += 1;
            // truth table over the (≤3) leaves using the canonical 3-var patterns.
            let pat = [0xAAu8, 0xCC, 0xF0];
            let mut tt = 0u8;
            let mut ok = true;
            for row in 0..8u8 {
                let mut lb = HashMap::new();
                for (i, &leaf) in leaves.iter().enumerate() {
                    lb.insert(leaf, (row >> i) & 1);
                }
                match eval(instr.dst, &def, &instrs, &lb) {
                    Some(b) => {
                        if b == 1 {
                            tt |= 1 << row;
                        }
                    }
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            let _ = pat;
            if !ok {
                continue;
            }
            let best = opt[tt as usize] as usize;
            cur_ops += ops;
            if best < ops {
                wins += 1;
                saved += ops - best;
            }
        }
    }
    println!("[{label}] {harvested} 1-bit boolean cones; {le3} have ≤3 leaves & ≥2 ops");
    let wpct = 100.0 * wins as f64 / le3.max(1) as f64;
    let spct = 100.0 * saved as f64 / cur_ops.max(1) as f64;
    println!("  ≤3-leaf: of those, {wins} ({wpct:.0}%) have a CHEAPER optimal form; {saved} ops saveable (of {cur_ops} = {spct:.1}%)");
    println!("  >3-leaf (egg/SMT domain): {big} cones, {big_ops} ops, avg {:.1} leaves — UNMEASURED by the ≤3-leaf DP",
        big_leaves as f64 / big.max(1) as f64);
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "bench/sv/picorv32.v".into());
    let top = std::env::args().nth(2).unwrap_or_else(|| "picorv32".into());
    let opt = optimal_costs();
    println!("optimal 3-var boolean DP built ({} functions reachable)", opt.iter().filter(|&&c| c != u8::MAX).count());

    let src = std::fs::read_to_string(&path).expect("read");
    let imported = import_sv(&src, Some(&top)).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, &top).expect("lower");
    let machine = lower_to_machine_program(&program);

    // Measure the RESIDUAL opportunity: after the existing specialize pass.
    let (spec, _) = specialize_program(&machine);
    run(&machine, "raw", &opt);
    run(&spec, "after specialize", &opt);
}
