//! Measure-first sizing for AUTOMATIC FUSED-SUPEROPERATOR DISCOVERY (the GPU/
//! batch value-traffic axis of an RTL superoptimizer). Each op in the GPU interp
//! is a value-buffer write + operand reads; fusing a SINGLE-USE intermediate into
//! its consumer eliminates that write+read (a register pass). The hand-built
//! 13-op fusion menu captures specific patterns; this probe sizes how much
//! fusion headroom REMAINS beyond the menu — i.e. whether auto-discovering
//! (or a generic macro-op) would help.
//!
//! Metrics (records = value-buffer-writing ops): unfused, menu-fused, and the
//! THEORETICAL MIN (every single-use intermediate fused). The gap menu→min is
//! the discovery opportunity; the residual op-mix shows what superops it needs.
//! Build: cargo run --release -p rrtl-gpu-sim --example fused_superop_probe -- [v] [top]
use rrtl_gpu_sim::interp::{InterpProgram, RECORD_WORDS};
use rrtl_sim_ir::specialize::specialize_program;
use rrtl_sim_ir::{
    lower_to_machine_program, lower_to_packed_program, PackedBlock, PackedEffect, PackedInstrKind,
    PackedMachineProgram, PackedValueId,
};
use rrtl_sv_frontend::import_sv;
use std::collections::HashMap;

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

fn eff_operands(eff: &PackedEffect) -> Vec<PackedValueId> {
    match eff {
        PackedEffect::StoreSignal { value, .. } => vec![*value],
        PackedEffect::CaptureReg { value, .. } => vec![*value],
        PackedEffect::MemoryWrite { enable, addr, data, .. } => vec![*enable, *addr, *data],
    }
}

/// For one block: total instrs, # single-use-eliminable intermediates (used
/// exactly once, by another INSTR not an effect), and the op-mix of the
/// CONSUMER ops that such an intermediate feeds (the superops a generic macro-op
/// or auto-discovery would cover).
fn analyze(block: &PackedBlock) -> (usize, usize, HashMap<String, usize>) {
    let mut use_in_instr: HashMap<PackedValueId, usize> = HashMap::new();
    let mut use_in_eff: HashMap<PackedValueId, usize> = HashMap::new();
    let mut def: HashMap<PackedValueId, PackedInstrKind> = HashMap::new();
    let mut total = 0usize;
    for p in &block.packets {
        for instr in &p.instrs {
            total += 1;
            def.insert(instr.dst, instr.kind.clone());
            for op in operands(&instr.kind) {
                *use_in_instr.entry(op).or_default() += 1;
            }
        }
        for eff in &p.effects {
            for op in eff_operands(eff) {
                *use_in_eff.entry(op).or_default() += 1;
            }
        }
    }
    // A value is fusion-eliminable iff used exactly once total and that use is in
    // an instruction (so it can be inlined into its consumer's macro-op).
    let mut eliminable = 0usize;
    let mut consumer_mix: HashMap<String, usize> = HashMap::new();
    // Map operand -> the consumer instr kind (for op-mix of the boundary). Build a
    // reverse: for each instr, its operands; if an operand is eliminable, tally
    // this instr's kind.
    let elig: std::collections::HashSet<PackedValueId> = def
        .keys()
        .copied()
        .filter(|v| use_in_instr.get(v).copied().unwrap_or(0) == 1 && use_in_eff.get(v).copied().unwrap_or(0) == 0)
        .collect();
    for p in &block.packets {
        for instr in &p.instrs {
            for op in operands(&instr.kind) {
                if elig.contains(&op) {
                    eliminable += 1;
                    let k = format!("{:?}", std::mem::discriminant(&instr.kind));
                    let name = match &instr.kind {
                        PackedInstrKind::And(..) => "And", PackedInstrKind::Or(..) => "Or",
                        PackedInstrKind::Xor(..) => "Xor", PackedInstrKind::Not(..) => "Not",
                        PackedInstrKind::Add(..) => "Add", PackedInstrKind::Sub(..) => "Sub",
                        PackedInstrKind::Mul(..) => "Mul", PackedInstrKind::Mux { .. } => "Mux",
                        PackedInstrKind::Eq(..) => "Eq", PackedInstrKind::Ne(..) => "Ne",
                        PackedInstrKind::Lt { .. } => "Lt", PackedInstrKind::Slice { .. } => "Slice",
                        PackedInstrKind::Concat(..) => "Concat", PackedInstrKind::Zext(..) => "Zext",
                        PackedInstrKind::Sext(..) => "Sext", PackedInstrKind::Trunc(..) => "Trunc",
                        PackedInstrKind::Cast(..) => "Cast", PackedInstrKind::MemRead { .. } => "MemRead",
                        _ => "?",
                    };
                    let _ = k;
                    *consumer_mix.entry(name.to_string()).or_default() += 1;
                }
            }
        }
    }
    (total, eliminable, consumer_mix)
}

fn records(p: &InterpProgram) -> usize {
    p.total_code_words() / RECORD_WORDS
}

fn report(machine: &PackedMachineProgram, label: &str) {
    let mut total = 0;
    let mut elim = 0;
    let mut mix: HashMap<String, usize> = HashMap::new();
    for blk in [&machine.streams.comb, &machine.streams.tick_next, &machine.streams.async_reset_comb] {
        let (t, e, m) = analyze(blk);
        total += t;
        elim += e;
        for (k, v) in m {
            *mix.entry(k).or_default() += v;
        }
    }
    let unfused = records(&InterpProgram::encode_opts(machine, false).unwrap());
    let menu = records(&InterpProgram::encode_opts(machine, true).unwrap());
    let theoretical_min = total.saturating_sub(elim);
    println!("[{label}] {total} instrs");
    println!("  records: unfused {unfused} → menu-fused {menu} ({:.1}% fused) → theoretical-min {theoretical_min}",
        100.0 * (unfused - menu) as f64 / unfused.max(1) as f64);
    let menu_saved = unfused.saturating_sub(menu);
    let max_saved = unfused.saturating_sub(theoretical_min);
    println!("  fusion saved: menu {menu_saved} of max {max_saved} possible → HEADROOM {} records ({:.1}% of unfused)",
        menu.saturating_sub(theoretical_min), 100.0 * menu.saturating_sub(theoretical_min) as f64 / unfused.max(1) as f64);
    let mut sorted: Vec<(&String, &usize)> = mix.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    let top: Vec<String> = sorted.iter().take(8).map(|(k, v)| format!("{k}:{v}")).collect();
    println!("  consumer op-mix of fusible intermediates (superops needed): {}", top.join(" "));
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "bench/sv/picorv32.v".into());
    let top = std::env::args().nth(2).unwrap_or_else(|| "picorv32".into());
    let src = std::fs::read_to_string(&path).expect("read");
    let imported = import_sv(&src, Some(&top)).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, &top).expect("lower");
    let machine = lower_to_machine_program(&program);
    let (spec, _) = specialize_program(&machine);
    report(&machine, "raw");
    report(&spec, "after specialize");
}
