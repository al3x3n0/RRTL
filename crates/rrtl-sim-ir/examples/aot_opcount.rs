//! Size the IR-scheduling prize: count the primitive ops RRTL evaluates per
//! picorv32 cycle (across the four machine streams the AOT emits) and look for
//! cross-stream redundancy (identical (kind, operands) instructions computed in
//! more than one stream — candidates for cross-stream CSE). Compare against
//! Verilator's generated eval (counted separately from obj_dir).
//! Build: cargo run --release -p rrtl-sim-ir --example aot_opcount
use std::collections::HashMap;

use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program, PackedBlock, PackedInstrKind};
use rrtl_sv_frontend::import_sv;

fn count(block: &PackedBlock) -> (usize, usize) {
    let mut instrs = 0;
    let mut effects = 0;
    for p in &block.packets {
        instrs += p.instrs.len();
        effects += p.effects.len();
    }
    (instrs, effects)
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "bench/sv/picorv32.v".into());
    let src = std::fs::read_to_string(&path).expect("read");
    let imported = import_sv(&src, Some("picorv32")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "picorv32").expect("packed");
    let m = lower_to_machine_program(&program);

    let streams = [
        ("async_reset_comb", &m.streams.async_reset_comb),
        ("comb", &m.streams.comb),
        ("tick_next", &m.streams.tick_next),
        ("tick_commit", &m.streams.tick_commit),
    ];
    let mut total_instr = 0;
    let mut total_eff = 0;
    println!("picorv32 machine-program ops per cycle:");
    for (name, b) in streams {
        let (i, e) = count(b);
        total_instr += i;
        total_eff += e;
        println!("  {name:16}: {i:5} instrs, {e:4} effects");
    }
    println!("  {:16}: {total_instr:5} instrs, {total_eff:4} effects  (per-cycle hot path)", "TOTAL");

    // Cross-stream redundancy: a (kind, dst-widths-aside) op recomputed in >1 stream.
    // Approximate identity by (discriminant + operand value-ids) — value-ids are
    // stream-local so this catches structurally-identical recomputation only when
    // they happen to share ids; a coarse lower bound. Better: count opcode-kind
    // histogram to show op mix.
    let mut kinds: HashMap<&str, usize> = HashMap::new();
    for (_, b) in streams {
        for p in &b.packets {
            for ins in &p.instrs {
                let k = match &ins.kind {
                    PackedInstrKind::Lit(_) => "Lit",
                    PackedInstrKind::Signal(_) => "Signal(load)",
                    PackedInstrKind::Not(_) => "Not",
                    PackedInstrKind::And(..) => "And",
                    PackedInstrKind::Or(..) => "Or",
                    PackedInstrKind::Xor(..) => "Xor",
                    PackedInstrKind::Add(..) => "Add",
                    PackedInstrKind::Sub(..) => "Sub",
                    PackedInstrKind::Mul(..) => "Mul",
                    PackedInstrKind::Eq(..) => "Eq",
                    PackedInstrKind::Ne(..) => "Ne",
                    PackedInstrKind::Lt { .. } => "Lt",
                    PackedInstrKind::Mux { .. } => "Mux",
                    PackedInstrKind::Slice { .. } => "Slice",
                    PackedInstrKind::Zext(_) => "Zext",
                    PackedInstrKind::Sext(_) => "Sext",
                    PackedInstrKind::Trunc(_) => "Trunc",
                    PackedInstrKind::Cast(_) => "Cast",
                    PackedInstrKind::Concat(_) => "Concat",
                    PackedInstrKind::MemRead { .. } => "MemRead",
                };
                *kinds.entry(k).or_default() += 1;
            }
        }
    }
    let mut v: Vec<_> = kinds.into_iter().collect();
    v.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    println!("op mix:");
    for (k, c) in v {
        println!("  {k:14}: {c}");
    }

    // Control-flow-specialization sizing: a value-id that is used exactly once and
    // whose sole consumer is a Mux then/else arm can be SUNK into that branch and
    // skipped when the arm isn't taken. Recursively, its single-use fan-in sinks
    // too. Measure the total sinkable op count (the per-cycle work a branchy,
    // eval-taken-only single-instance backend would save on the un-taken arms).
    use rrtl_sim_ir::PackedValueId;
    let mut use_count: HashMap<usize, usize> = HashMap::new();
    let mut def: HashMap<usize, &PackedInstrKind> = HashMap::new();
    let mut mux_arms: std::collections::HashSet<usize> = Default::default();
    let operands = |k: &PackedInstrKind, out: &mut Vec<usize>| {
        let mut p = |a: &PackedValueId| out.push(a.0);
        match k {
            PackedInstrKind::Not(a) | PackedInstrKind::Zext(a) | PackedInstrKind::Sext(a)
            | PackedInstrKind::Trunc(a) | PackedInstrKind::Cast(a) | PackedInstrKind::Slice { value: a, .. }
            | PackedInstrKind::MemRead { addr: a, .. } => p(a),
            PackedInstrKind::And(a, b) | PackedInstrKind::Or(a, b) | PackedInstrKind::Xor(a, b)
            | PackedInstrKind::Add(a, b) | PackedInstrKind::Sub(a, b) | PackedInstrKind::Mul(a, b)
            | PackedInstrKind::Eq(a, b) | PackedInstrKind::Ne(a, b) => { p(a); p(b); }
            PackedInstrKind::Lt { lhs, rhs, .. } => { p(lhs); p(rhs); }
            PackedInstrKind::Mux { cond, then_value, else_value } => { p(cond); p(then_value); p(else_value); }
            PackedInstrKind::Concat(parts) => parts.iter().for_each(|x| out.push(x.0)),
            PackedInstrKind::Lit(_) | PackedInstrKind::Signal(_) => {}
        }
    };
    for (_, b) in streams {
        for pk in &b.packets {
            for ins in &pk.instrs {
                def.insert(ins.dst.0, &ins.kind);
                let mut ops = Vec::new();
                operands(&ins.kind, &mut ops);
                for o in ops { *use_count.entry(o).or_default() += 1; }
                if let PackedInstrKind::Mux { then_value, else_value, .. } = &ins.kind {
                    mux_arms.insert(then_value.0);
                    mux_arms.insert(else_value.0);
                }
            }
        }
    }
    // recursively count the single-use cone rooted at each mux arm
    let mut sunk: std::collections::HashSet<usize> = Default::default();
    fn sink(id: usize, uc: &HashMap<usize, usize>, def: &HashMap<usize, &PackedInstrKind>,
            sunk: &mut std::collections::HashSet<usize>) -> usize {
        if uc.get(&id).copied().unwrap_or(0) != 1 || !sunk.insert(id) { return 0; }
        let Some(k) = def.get(&id) else { return 0; };
        let mut ops = Vec::new();
        let mut p = |a: &rrtl_sim_ir::PackedValueId| ops.push(a.0);
        match k {
            PackedInstrKind::Not(a)|PackedInstrKind::Zext(a)|PackedInstrKind::Sext(a)|PackedInstrKind::Trunc(a)
            |PackedInstrKind::Cast(a)|PackedInstrKind::Slice{value:a,..}|PackedInstrKind::MemRead{addr:a,..}=>p(a),
            PackedInstrKind::And(a,b)|PackedInstrKind::Or(a,b)|PackedInstrKind::Xor(a,b)|PackedInstrKind::Add(a,b)
            |PackedInstrKind::Sub(a,b)|PackedInstrKind::Mul(a,b)|PackedInstrKind::Eq(a,b)|PackedInstrKind::Ne(a,b)=>{p(a);p(b);}
            PackedInstrKind::Lt{lhs,rhs,..}=>{p(lhs);p(rhs);}
            PackedInstrKind::Mux{cond,then_value,else_value}=>{p(cond);p(then_value);p(else_value);}
            PackedInstrKind::Concat(parts)=>parts.iter().for_each(|x|ops.push(x.0)),
            _=>{}
        }
        1 + ops.into_iter().map(|o| sink(o, uc, def, sunk)).sum::<usize>()
    }
    let mut sinkable = 0usize;
    for &arm in &mux_arms { sinkable += sink(arm, &use_count, &def, &mut sunk); }
    println!(
        "control-flow specialization: {} of {} instrs ({:.0}%) are single-use mux-arm cones",
        sinkable, total_instr, sinkable as f64 / total_instr.max(1) as f64 * 100.0
    );
    println!(
        "  ~half evaluated per cycle on the un-taken arm ⇒ ~{:.0}% potential op saving from eval-taken-only",
        sinkable as f64 / total_instr.max(1) as f64 * 100.0 / 2.0
    );
}
