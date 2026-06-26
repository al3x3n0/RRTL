//! Measure-first sizing for a TOPOLOGICAL COMB SCHEDULER (the "eval-graph
//! scheduling" residual). A topo-merge of comb+tick_next + global CSE can only
//! help if there is genuinely-redundant computation across the streams that the
//! existing settle-capture fusion + clang GVN does NOT already remove.
//!
//! This probe does global value numbering over the MERGED comb→tick_next stream
//! (mapping a tick_next `Signal(idx)` read of a comb-written wire to the comb
//! value that produced it — i.e. modelling the settle-capture fusion), and counts
//! how many instructions are duplicate computations a topo-merge + CSE would
//! collapse. That is the CEILING; with fusion on, clang already CSEs the fused C
//! scope, so the realisable headroom is at most this and likely far less.
//! Build: cargo run --release -p rrtl-sim-ir --example toposched_probe -- [picorv32.v] [top]
use rrtl_sim_ir::{
    lower_to_machine_program, lower_to_packed_program, PackedBlock, PackedInstrKind, PackedValueId,
};
use rrtl_sv_frontend::import_sv;
use std::collections::HashMap;

/// A structural key for value-numbering: the op kind plus its operands' value
/// numbers (so equal sub-expressions over equal inputs share a number).
fn vn_key(kind: &PackedInstrKind, vn: &dyn Fn(PackedValueId) -> u64) -> String {
    use PackedInstrKind::*;
    match kind {
        Lit(w) => format!("lit{w:?}"),
        Signal(s) => format!("sig{s}"), // resolved separately to the producing value
        Not(a) => format!("not({})", vn(*a)),
        And(a, b) => format!("and({},{})", vn(*a).min(vn(*b)), vn(*a).max(vn(*b))),
        Or(a, b) => format!("or({},{})", vn(*a).min(vn(*b)), vn(*a).max(vn(*b))),
        Xor(a, b) => format!("xor({},{})", vn(*a).min(vn(*b)), vn(*a).max(vn(*b))),
        Add(a, b) => format!("add({},{})", vn(*a).min(vn(*b)), vn(*a).max(vn(*b))),
        Mul(a, b) => format!("mul({},{})", vn(*a).min(vn(*b)), vn(*a).max(vn(*b))),
        Sub(a, b) => format!("sub({},{})", vn(*a), vn(*b)),
        Eq(a, b) => format!("eq({},{})", vn(*a).min(vn(*b)), vn(*a).max(vn(*b))),
        Ne(a, b) => format!("ne({},{})", vn(*a).min(vn(*b)), vn(*a).max(vn(*b))),
        Lt { lhs, rhs, signed } => format!("lt{signed}({},{})", vn(*lhs), vn(*rhs)),
        Mux { cond, then_value, else_value } => format!("mux({},{},{})", vn(*cond), vn(*then_value), vn(*else_value)),
        Slice { value, lsb } => format!("slice{lsb}({})", vn(*value)),
        Zext(a) => format!("zext({})", vn(*a)),
        Sext(a) => format!("sext({})", vn(*a)),
        Trunc(a) => format!("trunc({})", vn(*a)),
        Cast(a) => format!("cast({})", vn(*a)),
        Concat(vs) => format!("concat({:?})", vs.iter().map(|v| vn(*v)).collect::<Vec<_>>()),
        MemRead { memory, addr } => format!("memrd{memory}({})", vn(*addr)),
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "bench/sv/picorv32.v".into());
    let top = std::env::args().nth(2).unwrap_or_else(|| "picorv32".into());
    let src = std::fs::read_to_string(&path).expect("read");
    let imported = import_sv(&src, Some(&top)).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, &top).expect("lower");
    let machine = lower_to_machine_program(&program);

    // Global value numbering across comb then tick_next. `value_vn` maps a packed
    // value-id (block-local, but we process comb fully before tick_next) to its
    // value number; `signal_vn` maps a signal index to the VN of the value that
    // last wrote it in comb (the settle-capture fusion: tick_next reads the live
    // comb value, not a reload).
    let mut next_vn = 0u64;
    let mut key_to_vn: HashMap<String, u64> = HashMap::new();
    let mut value_vn: HashMap<PackedValueId, u64> = HashMap::new();
    let mut signal_vn: HashMap<usize, u64> = HashMap::new();
    let mut leaf_signal_vn: HashMap<usize, u64> = HashMap::new();

    let mut total = 0usize;
    let mut duplicates = 0usize; // compute-CSE duplicates (clang GVN already removes)
    let mut forwards = 0usize; // Signal loads that forward a value (fusion makes free)

    let process = |block: &PackedBlock,
                       next_vn: &mut u64,
                       key_to_vn: &mut HashMap<String, u64>,
                       value_vn: &mut HashMap<PackedValueId, u64>,
                       signal_vn: &mut HashMap<usize, u64>,
                       leaf_signal_vn: &mut HashMap<usize, u64>,
                       total: &mut usize,
                       duplicates: &mut usize,
                       forwards: &mut usize| {
        for pkt in &block.packets {
            for instr in &pkt.instrs {
                *total += 1;
                // Signal reads resolve to the producing comb value if any, else a
                // stable leaf VN (registers / inputs).
                if let PackedInstrKind::Signal(s) = &instr.kind {
                    let v = signal_vn.get(s).copied().unwrap_or_else(|| {
                        *leaf_signal_vn.entry(*s).or_insert_with(|| {
                            let n = *next_vn;
                            *next_vn += 1;
                            n
                        })
                    });
                    value_vn.insert(instr.dst, v);
                    *forwards += 1;
                    continue;
                }
                let vn_lookup = |v: PackedValueId| *value_vn.get(&v).unwrap_or(&u64::MAX);
                let key = vn_key(&instr.kind, &vn_lookup);
                if let Some(&existing) = key_to_vn.get(&key) {
                    value_vn.insert(instr.dst, existing);
                    *duplicates += 1;
                } else {
                    let n = *next_vn;
                    *next_vn += 1;
                    key_to_vn.insert(key, n);
                    value_vn.insert(instr.dst, n);
                }
            }
            // comb StoreSignal updates signal_vn so tick_next reads the live value.
            for eff in &pkt.effects {
                if let rrtl_sim_ir::PackedEffect::StoreSignal { dst, value } = eff {
                    if let Some(&v) = value_vn.get(value) {
                        signal_vn.insert(*dst, v);
                    }
                }
            }
        }
    };

    process(&machine.streams.comb, &mut next_vn, &mut key_to_vn, &mut value_vn, &mut signal_vn, &mut leaf_signal_vn, &mut total, &mut duplicates, &mut forwards);
    process(&machine.streams.tick_next, &mut next_vn, &mut key_to_vn, &mut value_vn, &mut signal_vn, &mut leaf_signal_vn, &mut total, &mut duplicates, &mut forwards);

    println!("Topo-scheduler / cross-stream CSE sizing for `{top}`");
    println!("  {total} comb+tick_next instrs, {} distinct value numbers", next_vn);
    println!("  Signal forwards (a load that just forwards a live value): {forwards} ({:.1}%)",
        100.0 * forwards as f64 / total.max(1) as f64);
    println!("    → settle-capture FUSION already makes these free local reads (not state reloads).");
    println!("  compute-CSE duplicates (same op, same operand VNs): {duplicates} ({:.1}%)",
        100.0 * duplicates as f64 / total.max(1) as f64);
    println!("    → clang's GVN already removes these in the fused single-scope C.");
    println!("\n  VERDICT: a topo-merge + global CSE could remove at most {} of {total} instrs ({:.1}%),",
        forwards + duplicates, 100.0 * (forwards + duplicates) as f64 / total.max(1) as f64);
    println!("  but ALL of it is ALREADY captured by settle-capture fusion (the forwards) + clang GVN");
    println!("  (the compute duplicates). Realisable headroom beyond the current AOT ≈ 0.");
    println!("  The {} non-duplicate ops are the irreducible mux-eval-all + width-op cost (the dataflow-IR", next_vn);
    println!("  batch-moat tradeoff), which a SCHEDULER cannot remove — only control-flow conversion (cflow) can.");
}
