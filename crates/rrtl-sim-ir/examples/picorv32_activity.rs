//! Activity profiler for picorv32: how idle is the design per cycle? Sizes the
//! activity-skipping prize before building the mechanism. Uses the existing
//! `activity::register_support` cone analysis + the real JIT execution (same
//! program/mem model as picorv32_bench), and reports:
//!   - register-count skip rate: mean fraction of register cones whose support
//!     leaves didn't change this cycle (so their next-state is provably stable),
//!   - leaf-weighted skip rate: the same weighted by cone leaf count (a rough
//!     work proxy), and the cone-size distribution (coarse vs fine cones).
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example picorv32_activity -- [N]
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::activity::register_support;
    use rrtl_sim_ir::{jit::JitSimulator, lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;

    let n: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(200_000);

    let prog = rrtl_riscv_asm::assemble(
        "
        li   x1, 0
        li   x2, 0
    loop:
        addi x1, x1, 1
        addi x2, x2, 3
        xor  x3, x1, x2
        j    loop
        ",
    )
    .expect("assemble");

    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv/picorv32.v"))
        .expect("read picorv32.v");
    let imported = import_sv(&src, Some("picorv32")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "picorv32").expect("lower packed");
    let supports = register_support(&program);
    let machine = lower_to_machine_program(&program);

    // cone-size (leaf-count) distribution
    let total_leaves: usize = supports.iter().map(|s| s.support.len()).sum();
    let mem_cones = supports.iter().filter(|s| s.reads_memory).count();
    let mut sizes: Vec<usize> = supports.iter().map(|s| s.support.len()).collect();
    sizes.sort_unstable();
    let pct = |p: f64| sizes.get(((sizes.len() as f64 * p) as usize).min(sizes.len().saturating_sub(1))).copied().unwrap_or(0);
    println!(
        "picorv32: {} register cones, {} leaf-edges total, avg {:.1} leaves/cone (median {}, p90 {}, max {}); {} cones read memory",
        supports.len(),
        total_leaves,
        total_leaves as f64 / supports.len().max(1) as f64,
        pct(0.5),
        pct(0.9),
        sizes.last().copied().unwrap_or(0),
        mem_cones,
    );

    // union of all leaves we must watch
    let mut leaves: Vec<usize> = supports.iter().flat_map(|s| s.support.iter().copied()).collect();
    leaves.sort_unstable();
    leaves.dedup();

    let idx = |name: &str| {
        let h = compiled.find_module("picorv32").unwrap().signals.iter()
            .find(|s| s.name == name).unwrap().handle;
        program.signal_index(h).unwrap()
    };
    let (i_clk, i_resetn) = (idx("clk"), idx("resetn"));
    let (i_mem_ready, i_mem_rdata) = (idx("mem_ready"), idx("mem_rdata"));
    let (o_mem_valid, o_mem_addr) = (idx("mem_valid"), idx("mem_addr"));
    let (o_mem_wstrb, o_mem_wdata) = (idx("mem_wstrb"), idx("mem_wdata"));

    let mut mem = vec![0u32; 4096];
    mem[..prog.len()].copy_from_slice(&prog);
    let mut jit = JitSimulator::compile(&machine).expect("jit compile");

    // Optional fixed memory latency (cycles the core stalls per access) to
    // contrast busy vs stall-heavy workloads. MEM_LAT=0 => 1-cycle pulse.
    let mem_lat: u64 = std::env::var("MEM_LAT").ok().and_then(|s| s.parse().ok()).unwrap_or(0);

    let mut prev_leaf = vec![0u64; leaves.len()];
    let mut prev_ready = 0u64;
    let mut wait = 0u64;
    let mut active_cone_sum = 0.0f64; // register-count weighted
    let mut active_leaf_sum = 0.0f64; // leaf-count weighted
    let mut measured = 0usize;

    for c in 0..n {
        let resetn = (c >= 4) as u64;
        let valid = jit.get_signal(o_mem_valid);
        let addr = jit.get_signal(o_mem_addr);
        let wstrb = jit.get_signal(o_mem_wstrb);
        let wdata = jit.get_signal(o_mem_wdata);
        // hold the request for `mem_lat` cycles before acknowledging
        if valid != 0 && prev_ready == 0 && wait < mem_lat {
            wait += 1;
        }
        let ready = (valid != 0 && prev_ready == 0 && wait >= mem_lat) as u64;
        if ready != 0 {
            wait = 0;
        }
        let mut rdata = 0u64;
        if ready != 0 {
            let widx = ((addr >> 2) as usize) & (mem.len() - 1);
            if wstrb != 0 {
                let mut w = mem[widx];
                for b in 0..4 {
                    if wstrb & (1 << b) != 0 {
                        let sh = b * 8;
                        w = (w & !(0xFFu32 << sh)) | ((((wdata >> sh) & 0xFF) as u32) << sh);
                    }
                }
                mem[widx] = w;
            } else {
                rdata = mem[widx] as u64;
            }
        }
        prev_ready = ready;
        jit.set_signal(i_clk, 1);
        jit.set_signal(i_resetn, resetn);
        jit.set_signal(i_mem_ready, ready);
        jit.set_signal(i_mem_rdata, rdata);
        jit.tick();

        // snapshot leaves, mark which changed
        let mut changed = vec![false; leaves.len()];
        for (k, &li) in leaves.iter().enumerate() {
            let v = jit.get_signal(li);
            if v != prev_leaf[k] {
                changed[k] = true;
                prev_leaf[k] = v;
            }
        }
        if c < 64 {
            continue; // warm up past reset before measuring
        }
        // leaf index -> position in `leaves`
        let pos = |li: usize| leaves.binary_search(&li).unwrap();
        let mut active = 0usize;
        let mut active_leaves = 0usize;
        for s in &supports {
            let act = s.reads_memory || s.support.iter().any(|&li| changed[pos(li)]);
            if act {
                active += 1;
                active_leaves += s.support.len();
            }
        }
        active_cone_sum += active as f64 / supports.len().max(1) as f64;
        active_leaf_sum += active_leaves as f64 / total_leaves.max(1) as f64;
        measured += 1;
    }

    let act_cone = active_cone_sum / measured.max(1) as f64;
    let act_leaf = active_leaf_sum / measured.max(1) as f64;
    println!(
        "per-cycle ACTIVE: {:.1}% of cones ({:.1}% skippable), {:.1}% of leaf-work ({:.1}% skippable)",
        act_cone * 100.0,
        (1.0 - act_cone) * 100.0,
        act_leaf * 100.0,
        (1.0 - act_leaf) * 100.0,
    );
    println!(
        "  => structural ceiling on cone-skip ~{:.1}% of register cones idle/cycle (value-level ESSENT would be >=)",
        (1.0 - act_cone) * 100.0
    );
}
