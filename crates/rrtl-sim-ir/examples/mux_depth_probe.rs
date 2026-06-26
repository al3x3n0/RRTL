//! Size the mux-tree-rebalancing opportunity: how deep are picorv32's dependency
//! chains, how much of the critical path is muxes, and are the deep mux chains
//! one-hot equality-case chains (safe to rebalance into a balanced tree, shrinking
//! depth N→log N for better superscalar IPC)? Measure before building the pass.
//! Build: cargo run --release -p rrtl-sim-ir --example mux_depth_probe -- bench/sv/picorv32.v picorv32
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program, PackedBlock, PackedInstrKind, PackedValueId};
use rrtl_sv_frontend::import_sv;
use std::collections::HashMap;

fn operands(kind: &PackedInstrKind) -> Vec<usize> {
    use PackedInstrKind::*;
    match kind {
        Lit(_) | Signal(_) => vec![],
        Not(a) | Zext(a) | Sext(a) | Trunc(a) | Cast(a) | Slice { value: a, .. } => vec![a.0],
        And(a, b) | Or(a, b) | Xor(a, b) | Add(a, b) | Sub(a, b) | Mul(a, b) | Eq(a, b) | Ne(a, b) => vec![a.0, b.0],
        Lt { lhs, rhs, .. } => vec![lhs.0, rhs.0],
        Mux { cond, then_value, else_value } => vec![cond.0, then_value.0, else_value.0],
        Concat(parts) => parts.iter().map(|p| p.0).collect(),
        MemRead { addr, .. } => vec![addr.0],
    }
}

fn analyze(block: &PackedBlock, label: &str) {
    // depth[v] = longest dependency chain ending at value v; muxd[v] = longest
    // chain of mux→mux dependencies (the part a tree rebalance would shorten).
    let mut depth: HashMap<usize, usize> = HashMap::new();
    let mut muxd: HashMap<usize, usize> = HashMap::new();
    let mut kind_of: HashMap<usize, PackedInstrKind> = HashMap::new();
    let mut else_child: HashMap<usize, usize> = HashMap::new(); // mux -> its else mux
    let (mut nmux, mut crit, mut mux_crit) = (0usize, 0usize, 0usize);
    for packet in &block.packets {
        for instr in &packet.instrs {
            let ops = operands(&instr.kind);
            let d = 1 + ops.iter().map(|o| *depth.get(o).unwrap_or(&0)).max().unwrap_or(0);
            let is_mux = matches!(instr.kind, PackedInstrKind::Mux { .. });
            let md_ops = ops.iter().map(|o| *muxd.get(o).unwrap_or(&0)).max().unwrap_or(0);
            let md = if is_mux { md_ops + 1 } else { md_ops };
            depth.insert(instr.dst.0, d);
            muxd.insert(instr.dst.0, md);
            kind_of.insert(instr.dst.0, instr.kind.clone());
            if let PackedInstrKind::Mux { else_value, .. } = &instr.kind {
                else_child.insert(instr.dst.0, else_value.0);
                nmux += 1;
            }
            crit = crit.max(d);
            mux_crit = mux_crit.max(md);
        }
    }

    // Find the longest else-linked mux chain and check if its conditions are all
    // Eq(sel, const) with a shared selector (one-hot ⇒ priority-free ⇒ tree-safe).
    let mut best_chain: Vec<usize> = vec![];
    for &start in else_child.keys() {
        let mut chain = vec![start];
        let mut cur = start;
        while let Some(&next) = else_child.get(&cur) {
            if matches!(kind_of.get(&next), Some(PackedInstrKind::Mux { .. })) {
                chain.push(next);
                cur = next;
            } else {
                break;
            }
        }
        if chain.len() > best_chain.len() {
            best_chain = chain;
        }
    }
    let eq_conds = best_chain.iter().filter(|m| {
        if let Some(PackedInstrKind::Mux { cond, .. }) = kind_of.get(m) {
            matches!(kind_of.get(&cond.0), Some(PackedInstrKind::Eq(..) | PackedInstrKind::Ne(..)))
        } else { false }
    }).count();

    println!("  [{label}] critical-path depth {crit}, of which mux-chain depth {mux_crit}; {nmux} muxes");
    println!("           longest else-chain: {} muxes, {} with Eq/Ne conditions (one-hot-rebalanceable)",
        best_chain.len(), eq_conds);

    // Dump the structure of the longest chain: each mux's condition-op kind, and
    // whether the THEN value is a leaf (Signal/Lit) — a chain of independent
    // (cond, leaf-value) pairs is a classic priority ladder a parallel-prefix
    // rebalance can shorten; if then-values are themselves deep, it's intrinsic.
    if best_chain.len() >= 6 {
        let kname = |id: usize| -> String {
            match kind_of.get(&id) {
                Some(PackedInstrKind::Eq(..)) => "Eq".into(),
                Some(PackedInstrKind::Ne(..)) => "Ne".into(),
                Some(PackedInstrKind::And(..)) => "And".into(),
                Some(PackedInstrKind::Or(..)) => "Or".into(),
                Some(PackedInstrKind::Not(..)) => "Not".into(),
                Some(PackedInstrKind::Signal(_)) => "Signal".into(),
                Some(PackedInstrKind::Lt { .. }) => "Lt".into(),
                Some(k) => format!("{k:?}").split('(').next().unwrap_or("?").split('{').next().unwrap_or("?").trim().to_string(),
                None => "leaf".into(),
            }
        };
        let mut cond_hist: HashMap<String, usize> = HashMap::new();
        let mut leaf_then = 0;
        for m in &best_chain {
            if let Some(PackedInstrKind::Mux { cond, then_value, .. }) = kind_of.get(m) {
                *cond_hist.entry(kname(cond.0)).or_default() += 1;
                if matches!(kind_of.get(&then_value.0), Some(PackedInstrKind::Signal(_) | PackedInstrKind::Lit(_)) | None) {
                    leaf_then += 1;
                }
            }
        }
        let mut hist: Vec<_> = cond_hist.into_iter().collect();
        hist.sort_by(|a, b| b.1.cmp(&a.1));
        println!("           chain condition-op histogram: {hist:?}");
        println!("           then-values that are leaves (Signal/Lit): {}/{}", leaf_then, best_chain.len());
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "bench/sv/picorv32.v".into());
    let top = std::env::args().nth(2).unwrap_or_else(|| "picorv32".into());
    let src = std::fs::read_to_string(&path).expect("read top");
    let imported = import_sv(&src, Some(&top)).unwrap();
    let compiled = rrtl_core::compile(&imported.design).unwrap();
    let program = lower_to_packed_program(&compiled, &top).unwrap();
    let machine = lower_to_machine_program(&program);
    println!("mux-chain analysis for `{top}`:");
    analyze(&machine.streams.comb, "comb");
    analyze(&machine.streams.tick_next, "tick_next");
    let _ = PackedValueId(0);
}
