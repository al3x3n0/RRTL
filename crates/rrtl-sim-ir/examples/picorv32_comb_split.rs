//! Sizes the topo-ordered-comb-split prize for picorv32: of all combinational
//! signals, how many feed a register's (or memory's) next-state — and thus MUST
//! be evaluated pre-commit — vs feed only outputs/observation, which a split
//! could defer to the lazy post-commit settle (skipped entirely when a harness
//! reads only registered/aliased outputs). The deferrable fraction is the
//! ceiling on what splitting `tick_many`'s pre-settle saves.
//! Build: cargo run --release -p rrtl-sim-ir --example picorv32_comb_split
use std::collections::{HashMap, HashSet};

use rrtl_sim_ir::activity::expr_direct_reads;
use rrtl_sim_ir::{lower_to_packed_program, PackedOp};
use rrtl_sv_frontend::import_sv;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "bench/sv/picorv32.v".into());
    let src = std::fs::read_to_string(&path).expect("read picorv32.v");
    let imported = import_sv(&src, Some("picorv32")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "picorv32").expect("lower packed");

    // comb_def: signal -> its combinational definition expr; cost = expr node count.
    let mut comb_def: HashMap<usize, usize> = HashMap::new(); // dst -> cost
    let mut comb_reads: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut total_cost = 0usize;
    for stream in [&program.streams.async_reset_comb, &program.streams.comb] {
        for packet in stream {
            for op in &packet.ops {
                if let PackedOp::Assign { dst, expr } = op {
                    let (reads, _) = expr_direct_reads(expr);
                    let cost = 1 + reads.len(); // crude node-count proxy
                    comb_def.insert(*dst, cost);
                    comb_reads.insert(*dst, reads);
                    total_cost += cost;
                }
            }
        }
    }

    // Seed the "feeds next-state" set from everything read by register next-state
    // exprs and memory writes (tick_next + tick_commit), then close backward over
    // comb definitions.
    let mut feeds_reg: HashSet<usize> = HashSet::new();
    let mut work: Vec<usize> = Vec::new();
    let seed = |reads: Vec<usize>, feeds: &mut HashSet<usize>, work: &mut Vec<usize>| {
        for r in reads {
            if comb_def.contains_key(&r) && feeds.insert(r) {
                work.push(r);
            }
        }
    };
    for packet in &program.streams.tick_next {
        for op in &packet.ops {
            if let PackedOp::CaptureReg { next, .. } = op {
                seed(expr_direct_reads(next).0, &mut feeds_reg, &mut work);
            }
        }
    }
    for packet in &program.streams.tick_commit {
        for op in &packet.ops {
            // memory writes (and any committed assigns) read comb that must settle pre-commit
            match op {
                PackedOp::Assign { expr, .. } => seed(expr_direct_reads(expr).0, &mut feeds_reg, &mut work),
                PackedOp::CaptureReg { next, .. } => seed(expr_direct_reads(next).0, &mut feeds_reg, &mut work),
                _ => {}
            }
        }
    }
    while let Some(s) = work.pop() {
        if let Some(reads) = comb_reads.get(&s).cloned() {
            seed(reads, &mut feeds_reg, &mut work);
        }
    }

    let total_sigs = comb_def.len();
    let reg_sigs = feeds_reg.len();
    let out_sigs = total_sigs - reg_sigs;
    let reg_cost: usize = feeds_reg.iter().filter_map(|s| comb_def.get(s)).sum();
    let out_cost = total_cost - reg_cost;

    println!("picorv32 combinational split:");
    println!(
        "  signals : {total_sigs} total | {reg_sigs} feed next-state ({:.1}%) | {out_sigs} output-only ({:.1}%)",
        reg_sigs as f64 / total_sigs.max(1) as f64 * 100.0,
        out_sigs as f64 / total_sigs.max(1) as f64 * 100.0,
    );
    println!(
        "  cost    : {total_cost} total | {reg_cost} feed next-state ({:.1}%) | {out_cost} output-only ({:.1}%)",
        reg_cost as f64 / total_cost.max(1) as f64 * 100.0,
        out_cost as f64 / total_cost.max(1) as f64 * 100.0,
    );
    println!(
        "  => splitting can defer ~{:.1}% of the pre-commit comb settle to lazy phase B",
        out_cost as f64 / total_cost.max(1) as f64 * 100.0
    );

    // Register-residency (mem2reg) prize: each comb signal is StoreSignal'd to the
    // state buffer (a store) and each read of a comb signal is a load. Reads of
    // *other comb signals* could instead stay in CPU registers (like Verilator's
    // C++ locals); reads of leaves (regs/inputs) must load from state.
    let stores = comb_def.len(); // one store per comb signal definition
    let mut reads_of_comb = 0usize;
    let mut reads_of_leaf = 0usize;
    for reads in comb_reads.values() {
        for r in reads {
            if comb_def.contains_key(r) {
                reads_of_comb += 1;
            } else {
                reads_of_leaf += 1;
            }
        }
    }
    let mem_ops = stores + reads_of_comb + reads_of_leaf;
    println!("memory traffic / comb settle (state_ptr load+store):");
    println!(
        "  {mem_ops} ops = {stores} stores + {reads_of_comb} comb-reads + {reads_of_leaf} leaf-reads",
    );
    println!(
        "  => mem2reg (promote intra-block comb to SSA) could remove up to {} ops ({:.0}%): the {stores} stores + {reads_of_comb} comb-reads",
        stores + reads_of_comb,
        (stores + reads_of_comb) as f64 / mem_ops.max(1) as f64 * 100.0,
    );
}
