//! Measure-first feasibility probe for C-slow / temporal lane batching: can we
//! simulate K consecutive cycles of ONE instance in K parallel lanes?
//!
//! That is valid only where cycles are independent — i.e. the register
//! dependency graph (edge R_i -> R_j iff R_i feeds R_j's next-state cone) is
//! FEED-FORWARD (a DAG). Any feedback SCC (a self-driving accumulator/counter, an
//! FSM, a PC, mutual recurrence) is a sequential recurrence that cannot be
//! parallelized across cycles. So the gate is: what fraction of registers live in
//! feed-forward (parallelizable) position vs feedback SCCs (sequential), and how
//! deep is the feed-forward pipeline (= latency that becomes lane-parallel).
//! Build: cargo run --release -p rrtl-sim-ir --example cslow_probe -- [design top]
use rrtl_sim_ir::{lower_to_packed_program, register_support, PackedProgram, PackedSignalKind};
use rrtl_sv_frontend::import_sv;
use std::collections::HashMap;

/// Tarjan SCC over the register dependency graph. Returns component id per node
/// and the number of components.
fn sccs(n: usize, adj: &[Vec<usize>]) -> (Vec<usize>, usize) {
    let mut index = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut comp = vec![usize::MAX; n];
    let mut idx = 0usize;
    let mut ncomp = 0usize;
    // iterative Tarjan (designs can be deep; avoid stack overflow)
    for start in 0..n {
        if index[start] != usize::MAX {
            continue;
        }
        let mut call: Vec<(usize, usize)> = vec![(start, 0)];
        while let Some(&(v, pi)) = call.last() {
            if pi == 0 {
                index[v] = idx;
                low[v] = idx;
                idx += 1;
                stack.push(v);
                on_stack[v] = true;
            }
            if pi < adj[v].len() {
                call.last_mut().unwrap().1 += 1;
                let w = adj[v][pi];
                if index[w] == usize::MAX {
                    call.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
            } else {
                if low[v] == index[v] {
                    loop {
                        let w = stack.pop().unwrap();
                        on_stack[w] = false;
                        comp[w] = ncomp;
                        if w == v {
                            break;
                        }
                    }
                    ncomp += 1;
                }
                call.pop();
                if let Some(&(p, _)) = call.last() {
                    low[p] = low[p].min(low[v]);
                }
            }
        }
    }
    (comp, ncomp)
}

fn run(path: &str, top: &str) {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => return println!("[{top}] (read failed: {e})"),
    };
    let program = match import_sv(&src, Some(top))
        .and_then(|i| rrtl_core::compile(&i.design))
        .and_then(|c| lower_to_packed_program(&c, top))
    {
        Ok(p) => p,
        Err(_) => return println!("[{top}] (frontend cannot import this design — skipped)"),
    };
    analyze(&program, top);
}

fn analyze(program: &PackedProgram, top: &str) {
    let supports = register_support(program);

    // Register dependency graph: compact register indices.
    let regs: Vec<usize> = supports.iter().map(|rs| rs.reg).collect();
    let ri: HashMap<usize, usize> = regs.iter().enumerate().map(|(i, &r)| (r, i)).collect();
    let n = regs.len();
    let mut adj = vec![Vec::new(); n];
    let mut self_loop = vec![false; n];
    for rs in &supports {
        let j = ri[&rs.reg];
        for &s in &rs.support {
            if program.signals[s].kind == PackedSignalKind::Reg {
                if let Some(&i) = ri.get(&s) {
                    if i == j {
                        self_loop[i] = true;
                    } else {
                        adj[i].push(j); // i feeds j
                    }
                }
            }
        }
    }
    let (comp, ncomp) = sccs(n, &adj);
    // A register is SEQUENTIAL (feedback) if it self-loops or its SCC has >1 member.
    let mut comp_size = vec![0usize; ncomp];
    for &c in &comp {
        comp_size[c] += 1;
    }
    let feedback: Vec<bool> = (0..n).map(|i| self_loop[i] || comp_size[comp[i]] > 1).collect();
    let n_feedback = feedback.iter().filter(|&&b| b).count();

    // Feed-forward pipeline depth = longest path through feed-forward registers
    // (the latency that C-slow converts into lane parallelism).
    let mut depth = vec![0u32; n];
    // process in reverse-topo by SCC order (Tarjan emits in reverse topological order)
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| comp[i]); // SCC id increases along reverse-topo
    for &i in &order {
        if feedback[i] {
            continue;
        }
        let mut d = 0;
        for &j in &adj[i] {
            if !feedback[j] {
                d = d.max(depth[j] + 1);
            }
        }
        depth[i] = d;
    }
    let ff_depth = depth.iter().max().copied().unwrap_or(0);

    let pct = 100.0 * (n - n_feedback) as f64 / n.max(1) as f64;
    println!(
        "[{top}] {n} regs | feed-forward {} ({:.0}%) | feedback {} ({:.0}%) | ff pipeline depth {ff_depth}",
        n - n_feedback, pct, n_feedback, 100.0 - pct,
    );
    let verdict = if pct >= 70.0 && ff_depth >= 3 {
        "C-SLOW VIABLE (deep feed-forward pipeline -> cycles parallelize)"
    } else if pct >= 40.0 {
        "PARTIAL (feed-forward datapath + a sequential control kernel)"
    } else {
        "C-SLOW DEAD (feedback-dominated: PC/FSM/accumulator recurrence is serial)"
    };
    println!("    -> {verdict}");
}

/// Synthetic depth-D pipeline: r0 <= din; r_i <= r_{i-1} ^ k. Pure feed-forward
/// (each stage reads only the previous) — the regime C-slow is built for.
fn feedforward_pipeline(depth: usize) -> PackedProgram {
    use rrtl_core::{compile, lit_u, uint, Design};
    let mut design = Design::new();
    {
        let mut m = design.module("ffpipe");
        let clk = m.input("clk", uint(1));
        let din = m.input("din", uint(8));
        let mut prev = din.value();
        for i in 0..depth {
            let r = m.reg(format!("r{i}"), uint(8));
            m.clock(r, clk);
            m.next(r, prev.clone() ^ lit_u(0x5a, 8));
            prev = r.value();
        }
        let o = m.output("o", uint(8));
        m.assign(o, prev);
    }
    let compiled = compile(&design).unwrap();
    lower_to_packed_program(&compiled, "ffpipe").unwrap()
}

/// Synthetic accumulator bank: each r_i <= r_i + din. Self-loop (feedback) — the
/// regime C-slow CANNOT touch.
fn accumulator_bank(n: usize) -> PackedProgram {
    use rrtl_core::{compile, uint, Design};
    let mut design = Design::new();
    {
        let mut m = design.module("accbank");
        let clk = m.input("clk", uint(1));
        let din = m.input("din", uint(8));
        for i in 0..n {
            let r = m.reg(format!("acc{i}"), uint(8));
            m.clock(r, clk);
            m.next(r, r.value() + din.value());
            let o = m.output(format!("o{i}"), uint(8));
            m.assign(o, r);
        }
    }
    let compiled = compile(&design).unwrap();
    lower_to_packed_program(&compiled, "accbank").unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 3 {
        run(&args[1], &args[2]);
        return;
    }
    println!("== real designs (bench/sv) ==");
    for (f, top) in [
        ("bench/sv/picorv32.v", "picorv32"),
        ("bench/sv/cpu.sv", "cpu"),
        ("bench/sv/crc32.sv", "crc32"),
        ("bench/sv/mixed.sv", "mixed"),
        ("bench/sv/aimac.v", "aimac"),
        ("bench/sv/cfgdsp.sv", "cfgdsp"),
        ("bench/sv/workload.sv", "workload"),
    ] {
        run(f, top);
    }
    println!("== synthetic controls (validate the probe) ==");
    analyze(&feedforward_pipeline(8), "ffpipe-d8 (synthetic feed-forward)");
    analyze(&accumulator_bank(8), "accbank-8 (synthetic feedback)");
}
