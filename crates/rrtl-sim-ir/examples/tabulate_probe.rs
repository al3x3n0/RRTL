//! Measure-first sizing for DECODER / CASE TABULATION (a single-thread latency
//! lever). §4h.1 attributes part of Verilator's residual edge to "table-izing
//! decoders" — a structural pass clang cannot reconstruct from flattened C.
//!
//! The idea: a combinational cone (or register next-state) that is a PURE
//! function of a NARROW input (e.g. picorv32's decode flags depend on ~12
//! instruction bits) can be replaced at compile time by a precomputed lookup
//! table — one `MemRead` into a const ROM instead of a deep mux/AOI cone.
//!
//! This probe does NOT build the pass. It measures whether the opportunity
//! exists: for every comb-wire and register next-state root, it computes the
//! backward instruction cone, the BIT-LEVEL input support (tracking the dominant
//! `Slice(Signal,lsb)` field-extraction pattern precisely so a 32-bit
//! instruction word that is only sliced for ~12 bits is counted as ~12 bits),
//! the cone op-count, and whether the cone is pure (no MemRead) and
//! feedback-free (does not read its own destination). A root with a high
//! op-count and a small input-bit support is a tabulation candidate.
//! Build: cargo run --release -p rrtl-sim-ir --example tabulate_probe -- [picorv32.v] [top]
use rrtl_sim_ir::{
    lower_to_machine_program, lower_to_packed_program, PackedBlock, PackedInstrKind,
    PackedMachineProgram, PackedValueId,
};
use rrtl_sv_frontend::import_sv;
use std::collections::{HashMap, HashSet};

/// Value-id operands of an instruction kind (everything it reads).
fn operands(kind: &PackedInstrKind) -> Vec<PackedValueId> {
    use PackedInstrKind::*;
    match kind {
        Lit(_) | Signal(_) => vec![],
        Not(a) | Zext(a) | Sext(a) | Trunc(a) | Cast(a) | Slice { value: a, .. } => vec![*a],
        And(a, b) | Or(a, b) | Xor(a, b) | Add(a, b) | Sub(a, b) | Mul(a, b) | Eq(a, b)
        | Ne(a, b) => vec![*a, *b],
        Lt { lhs, rhs, .. } => vec![*lhs, *rhs],
        Mux { cond, then_value, else_value } => vec![*cond, *then_value, *else_value],
        Concat(vs) => vs.clone(),
        MemRead { addr, .. } => vec![*addr],
    }
}

struct Cone {
    ops: usize,
    /// Bit-level input support: (signal_index, bit).
    support: HashSet<(usize, u32)>,
    has_memread: bool,
}

/// Backward cone of value `root` within one block's def-map. `width_of` gives a
/// value's bit-width (for Signal/Slice leaf bit accounting).
fn cone_of(
    root: PackedValueId,
    def: &HashMap<PackedValueId, usize>,
    instrs: &[rrtl_sim_ir::PackedInstr],
) -> Cone {
    let mut seen: HashSet<PackedValueId> = HashSet::new();
    let mut stack = vec![root];
    let mut ops = 0usize;
    let mut support = HashSet::new();
    let mut has_memread = false;
    while let Some(v) = stack.pop() {
        if !seen.insert(v) {
            continue;
        }
        let Some(&ii) = def.get(&v) else { continue };
        let instr = &instrs[ii];
        ops += 1;
        match &instr.kind {
            PackedInstrKind::Lit(_) => {}
            PackedInstrKind::Signal(idx) => {
                for b in 0..instr.ty.width as u32 {
                    support.insert((*idx, b));
                }
            }
            // Precise field extraction: Slice(Signal(idx), lsb) reads exactly
            // bits [lsb, lsb+width) of the signal — the decode-extraction shape.
            PackedInstrKind::Slice { value, lsb } => {
                if let Some(&vi) = def.get(value) {
                    if let PackedInstrKind::Signal(idx) = &instrs[vi].kind {
                        for b in 0..instr.ty.width as u32 {
                            support.insert((*idx, *lsb as u32 + b));
                        }
                        continue; // do not recurse into the (fully-read) signal
                    }
                }
                stack.push(*value);
            }
            PackedInstrKind::MemRead { addr, .. } => {
                has_memread = true;
                stack.push(*addr);
            }
            other => {
                for op in operands(other) {
                    stack.push(op);
                }
            }
        }
    }
    Cone { ops, support, has_memread }
}

/// Build dst→instr-index map for a block (value-ids are block-local but flow
/// across packets within the block).
fn def_map(block: &PackedBlock) -> (HashMap<PackedValueId, usize>, Vec<rrtl_sim_ir::PackedInstr>) {
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

/// Shallow pretty-print of a value's defining expression tree, to depth `d`.
fn dump(
    v: PackedValueId,
    def: &HashMap<PackedValueId, usize>,
    instrs: &[rrtl_sim_ir::PackedInstr],
    name: &dyn Fn(usize) -> String,
    d: usize,
    indent: usize,
) {
    let pad = "  ".repeat(indent);
    let Some(&ii) = def.get(&v) else {
        println!("{pad}<input v{}>", v.0);
        return;
    };
    let k = &instrs[ii].kind;
    let w = instrs[ii].ty.width;
    let tag = match k {
        PackedInstrKind::Lit(l) => format!("Lit({l:?})"),
        PackedInstrKind::Signal(s) => format!("Signal({})", name(*s)),
        PackedInstrKind::Slice { lsb, .. } => format!("Slice[lsb={lsb}]"),
        PackedInstrKind::Mux { .. } => "Mux".into(),
        PackedInstrKind::Eq(..) => "Eq".into(),
        PackedInstrKind::And(..) => "And".into(),
        PackedInstrKind::Or(..) => "Or".into(),
        PackedInstrKind::Concat(..) => "Concat".into(),
        other => format!("{other:?}").chars().take(24).collect(),
    };
    println!("{pad}{tag} :{w}");
    if d == 0 {
        return;
    }
    if let PackedInstrKind::Signal(_) | PackedInstrKind::Lit(_) = k {
        return;
    }
    for op in operands(k) {
        dump(op, def, instrs, name, d - 1, indent + 1);
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "bench/sv/picorv32.v".into());
    let top = std::env::args().nth(2).unwrap_or_else(|| "picorv32".into());
    let dump_sig = std::env::args().nth(3); // optional: dump this register's next-state tree
    let src = std::fs::read_to_string(&path).expect("read top");
    let imported = import_sv(&src, Some(&top)).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, &top).expect("lower packed");
    let machine: PackedMachineProgram = lower_to_machine_program(&program);

    let name = |idx: usize| program.signals.get(idx).map(|s| s.name.as_str()).unwrap_or("?");

    // Roots: comb StoreSignal values + register CaptureReg next-states.
    let (comb_def, comb_instrs) = def_map(&machine.streams.comb);
    let (next_def, next_instrs) = def_map(&machine.streams.tick_next);

    // Optional: dump the next-state tree of a named register and exit.
    if let Some(target) = &dump_sig {
        let nm = |i: usize| program.signals.get(i).map(|s| s.name.clone()).unwrap_or_else(|| "?".into());
        for p in &machine.streams.tick_next.packets {
            for e in &p.effects {
                if let rrtl_sim_ir::PackedEffect::CaptureReg { dst, value, .. } = e {
                    if nm(*dst).ends_with(target.as_str()) {
                        println!("next-state of {} (v{}):", nm(*dst), value.0);
                        dump(*value, &next_def, &next_instrs, &nm, 4, 1);
                        println!();
                    }
                }
            }
        }
        return;
    }

    struct Cand {
        dst: usize,
        ops: usize,
        bits: usize,
        sigs: usize,
        pure: bool,
        self_fb: bool,
        leaves: Vec<usize>,
    }
    let mut cands: Vec<Cand> = Vec::new();

    let consider = |dst: usize, value: PackedValueId,
                        def: &HashMap<PackedValueId, usize>,
                        instrs: &[rrtl_sim_ir::PackedInstr],
                        cands: &mut Vec<Cand>| {
        let c = cone_of(value, def, instrs);
        let sigs: HashSet<usize> = c.support.iter().map(|(s, _)| *s).collect();
        let self_fb = sigs.contains(&dst);
        let mut leaves: Vec<usize> = sigs.iter().copied().collect();
        leaves.sort_unstable();
        cands.push(Cand {
            dst,
            ops: c.ops,
            bits: c.support.len(),
            sigs: sigs.len(),
            pure: !c.has_memread,
            self_fb,
            leaves,
        });
    };

    for p in &machine.streams.comb.packets {
        for e in &p.effects {
            if let rrtl_sim_ir::PackedEffect::StoreSignal { dst, value } = e {
                consider(*dst, *value, &comb_def, &comb_instrs, &mut cands);
            }
        }
    }
    for p in &machine.streams.tick_next.packets {
        for e in &p.effects {
            if let rrtl_sim_ir::PackedEffect::CaptureReg { dst, value, .. } = e {
                consider(*dst, *value, &next_def, &next_instrs, &mut cands);
            }
        }
    }

    let total_ops = comb_instrs.len() + next_instrs.len();
    println!("Tabulation-opportunity probe for `{top}` ({path})");
    println!("  total comb+next instrs: {total_ops}  (comb {}, next {})\n", comb_instrs.len(), next_instrs.len());

    // A tabulation candidate: pure (no MemRead), meaningful op-count, and a
    // narrow input-bit support that fits a ROM (≤ 20 bits ≈ 1M entries).
    const MIN_OPS: usize = 6;
    const MAX_BITS: usize = 20;
    let mut viable: Vec<&Cand> = cands
        .iter()
        .filter(|c| c.pure && c.ops >= MIN_OPS && c.bits <= MAX_BITS && c.bits > 0)
        .collect();
    viable.sort_by(|a, b| b.ops.cmp(&a.ops));

    let viable_ops: usize = viable.iter().map(|c| c.ops).sum();
    println!(
        "TABULATABLE roots (pure, ops≥{MIN_OPS}, input-bits≤{MAX_BITS}): {} roots, {viable_ops} cone-ops \
         ({:.1}% of comb+next)\n",
        viable.len(),
        100.0 * viable_ops as f64 / total_ops.max(1) as f64
    );
    println!("  {:<28} {:>5} {:>5} {:>5} {:>5}  inputs", "root signal", "ops", "bits", "sigs", "fb");
    for c in viable.iter().take(30) {
        let inputs: Vec<String> = c.leaves.iter().take(6).map(|&s| name(s).to_string()).collect();
        println!(
            "  {:<28} {:>5} {:>5} {:>5} {:>5}  {}",
            name(c.dst), c.ops, c.bits, c.sigs, if c.self_fb { "yes" } else { "no" },
            inputs.join(", ") + if c.leaves.len() > 6 { ", …" } else { "" }
        );
    }

    // JOINT decode-group sizing: the real tabulation target is the SHARED
    // decode function — one ROM indexed by the instruction bits driving ALL the
    // decode-flag registers at once (the hold-mux `flag <= trig ? decode : flag`
    // stays cheap). Union the cones of the instr_*/is_*/latched_is_* register
    // next-states, but STOP at register/control leaves: we measure the pure
    // decode sub-cone reachable from the instruction, factoring out self-hold.
    let decode_like = |n: &str| {
        n.starts_with("picorv32.instr_")
            || n.starts_with("picorv32.is_")
            || n.starts_with("picorv32.latched_is_")
    };
    // Union-cone op-count over decode-flag roots (unique instrs across all),
    // measured on the tick_next def-map.
    let mut union_seen: HashSet<PackedValueId> = HashSet::new();
    let mut union_support: HashSet<(usize, u32)> = HashSet::new();
    let mut roots = 0usize;
    let mut hold_matched = 0usize;
    for p in &machine.streams.tick_next.packets {
        for e in &p.effects {
            if let rrtl_sim_ir::PackedEffect::CaptureReg { dst, value, .. } = e {
                if decode_like(name(*dst)) {
                    roots += 1;
                    // Detect the hold-mux `Mux{cond, then, else=Signal(self)}`
                    // and tabulate only the THEN (decode) branch — the self-hold
                    // lives in else and must NOT enter the table's input.
                    let mut decode_root = *value;
                    if let Some(&ri) = next_def.get(value) {
                        if let PackedInstrKind::Mux { then_value, else_value, .. } = &next_instrs[ri].kind {
                            if let Some(&ei) = next_def.get(else_value) {
                                if matches!(&next_instrs[ei].kind, PackedInstrKind::Signal(s) if *s == *dst) {
                                    decode_root = *then_value;
                                    hold_matched += 1;
                                }
                            }
                        }
                    }
                    // walk the cone, accumulating into the shared seen-set so
                    // shared sub-expressions are counted ONCE.
                    let mut stack = vec![decode_root];
                    while let Some(v) = stack.pop() {
                        if !union_seen.insert(v) {
                            continue;
                        }
                        let Some(&ii) = next_def.get(&v) else { continue };
                        match &next_instrs[ii].kind {
                            PackedInstrKind::Signal(idx) => {
                                for b in 0..next_instrs[ii].ty.width as u32 {
                                    union_support.insert((*idx, b));
                                }
                            }
                            PackedInstrKind::Slice { value, lsb } => {
                                if let Some(&vi) = next_def.get(value) {
                                    if let PackedInstrKind::Signal(idx) = &next_instrs[vi].kind {
                                        for b in 0..next_instrs[ii].ty.width as u32 {
                                            union_support.insert((*idx, *lsb as u32 + b));
                                        }
                                        continue;
                                    }
                                }
                                stack.push(*value);
                            }
                            other => {
                                for op in operands(other) {
                                    stack.push(op);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // LEAF opcode-matchers: hold-matched decode regs whose THEN-branch support
    // is a pure function of the instruction word (+ a tiny control set), i.e. it
    // does NOT read other decode flags. These are the cleanly-tabulatable ones;
    // their joint instruction-bit key decides the ROM size.
    let instr_sigs: HashSet<&str> = ["picorv32.mem_rdata_latched"].into_iter().collect();
    let ctrl_ok = |n: &str| n == "picorv32.cpu_state" || n == "picorv32.resetn";
    let mut leaf_support: HashSet<(usize, u32)> = HashSet::new();
    let mut leaf_count = 0usize;
    for p in &machine.streams.tick_next.packets {
        for e in &p.effects {
            if let rrtl_sim_ir::PackedEffect::CaptureReg { dst, value, .. } = e {
                if !decode_like(name(*dst)) { continue; }
                let Some(&ri) = next_def.get(value) else { continue };
                let PackedInstrKind::Mux { then_value, else_value, .. } = &next_instrs[ri].kind else { continue };
                let is_hold = next_def.get(else_value)
                    .map(|&ei| matches!(&next_instrs[ei].kind, PackedInstrKind::Signal(s) if *s == *dst))
                    .unwrap_or(false);
                if !is_hold { continue; }
                let c = cone_of(*then_value, &next_def, &next_instrs);
                // pure-instruction iff every input signal is an instruction word
                // or an allowed tiny control signal.
                let pure_instr = c.support.iter().all(|(s, _)| {
                    let nm = name(*s);
                    instr_sigs.contains(nm) || ctrl_ok(nm)
                });
                if pure_instr {
                    leaf_count += 1;
                    for b in &c.support { leaf_support.insert(*b); }
                    // per-flag instruction-source breakdown
                    let lat = c.support.iter().filter(|(s, _)| name(*s) == "picorv32.mem_rdata_latched").count();
                    let q = c.support.iter().filter(|(s, _)| name(*s) == "picorv32.mem_rdata_q").count();
                    let cs = c.support.iter().filter(|(s, _)| ctrl_ok(name(*s))).count();
                    println!("    {:<28} ops {:>3}  latched {:>2}b  q {:>2}b  ctrl {}b", name(*dst), c.ops, lat, q, cs);
                }
            }
        }
    }
    let leaf_instr_bits: usize = leaf_support.iter()
        .filter(|(s, _)| instr_sigs.contains(name(*s))).count();
    println!("\nLEAF opcode-matchers (then = pure fn of instruction + cpu_state/resetn):");
    println!("  count                    : {leaf_count} flags");
    println!("  joint key width          : {} bits ({} instruction bits + {} control) → ROM 2^{} entries",
        leaf_support.len(), leaf_instr_bits, leaf_support.len() - leaf_instr_bits, leaf_support.len());

    let union_ops = union_seen.iter().filter(|v| next_def.contains_key(v)).count();
    let union_sigs: HashSet<usize> = union_support.iter().map(|(s, _)| *s).collect();
    let mut union_leaves: Vec<usize> = union_sigs.iter().copied().collect();
    union_leaves.sort_by_key(|&s| std::cmp::Reverse(union_support.iter().filter(|(x, _)| *x == s).count()));
    println!("\nJOINT decode-group cone ({roots} instr_*/is_*/latched_is_* registers, {hold_matched} hold-mux matched):");
    println!("  unique cone ops (then-only): {union_ops}  ({:.1}% of tick_next)", 100.0 * union_ops as f64 / next_instrs.len().max(1) as f64);
    println!("  joint input-bit support  : {} bits across {} signals", union_support.len(), union_sigs.len());
    println!("  top input signals (by bits used):");
    for &s in union_leaves.iter().take(8) {
        let bits = union_support.iter().filter(|(x, _)| *x == s).count();
        println!("    {:<32} {bits} bits", name(s));
    }

    // Also show the biggest cones we CANNOT tabulate, and why — the honest
    // counter-picture (wide input or self-feedback).
    let mut wide: Vec<&Cand> = cands.iter().filter(|c| c.ops >= 20).collect();
    wide.sort_by(|a, b| b.ops.cmp(&a.ops));
    println!("\n  Largest cones overall (tabulatable or not):");
    println!("  {:<28} {:>5} {:>6} {:>5} {:>6} {:>5}", "root signal", "ops", "bits", "sigs", "pure", "fb");
    for c in wide.iter().take(12) {
        println!(
            "  {:<28} {:>5} {:>6} {:>5} {:>6} {:>5}",
            name(c.dst), c.ops, c.bits, c.sigs,
            if c.pure { "yes" } else { "no" }, if c.self_fb { "yes" } else { "no" }
        );
    }
}
