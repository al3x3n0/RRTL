//! Measure-first for GPU **workgroup-uniform** activity skip — the ONLY skip
//! granularity the design-as-data interp kernel can do cheaply (one thread/lane,
//! whole tick loop on-device, fully uniform PC). A workgroup (tile) can only skip
//! a cycle if EVERY lane in it agrees the design is idle, and the kernel has no
//! cone structure, so the tractable test is WHOLE-DESIGN: skip a tile's cycle iff
//! all design leaves (inputs + registers) are stable across the whole tile.
//!
//! The 67-78% figure from `activity_probe` was CONE-level and per-lane. This probe
//! answers: (1) how much does the coarse whole-design tile granularity lose vs the
//! cone ceiling, and (2) does a free-running register collapse it? It decides
//! whether the (hard) WGSL barrier mechanism is worth building.
//!
//! Key insight vs the CPU/JIT activity-skip negatives: on GPU a UNIFORM branch is
//! ~free (no divergence), so unlike the per-lane guard whose cost grew with lanes,
//! a tile-uniform skip keeps its benefit — IF the skip rate survives the coarse
//! granularity. That survival is exactly what this measures.
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example gpu_uniform_skip_probe -- [depth lanes cycles duty tile]
use rrtl_core::{compile, lit_u, mux, uint, Design};
use rrtl_gpu_sim::interp::{InterpProgram, InterpRunner};
use rrtl_sim_ir::{lower_to_packed_program, register_support};

// Gated shift chain. `with_counter` adds a free-running counter (always active)
// to model a design that is never GLOBALLY idle even while its datapath stalls.
fn build(depth: usize, with_counter: bool) -> Design {
    let mut design = Design::new();
    let mut m = design.module("Chain");
    let clk = m.input("clk", uint(1));
    let en = m.input("en", uint(8));
    let din = m.input("din", uint(8));
    if with_counter {
        let ctr = m.reg("ctr", uint(8));
        m.clock(ctr, clk);
        m.next(ctr, ctr.value() + lit_u(1, 8));
        let octr = m.output("octr", uint(8));
        m.assign(octr, ctr);
    }
    let mut prev = din;
    for i in 0..depth {
        let s = m.reg(format!("s{i}"), uint(8));
        m.clock(s, clk);
        let en0 = en.value().slice(0, 1);
        m.next(s, mux(en0, prev.value(), s.value())); // hold when idle
        prev = s;
    }
    let o = m.output("o", uint(8));
    m.assign(o, prev);
    design
}

fn run(label: &str, with_counter: bool, depth: usize, lanes: usize, cycles: usize, duty: usize, tile: usize) {
    let design = build(depth, with_counter);
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Chain").unwrap();
    let supports = register_support(&program);
    let off = |idx: usize| program.signals[idx].layout.offset;
    let sig = |name: &str| -> usize { program.signals.iter().position(|s| s.name.ends_with(name)).unwrap() };
    let (en_off, din_off, clk_off) = (off(sig(".en")), off(sig(".din")), off(sig(".clk")));

    // union of all cone-support leaves (inputs + regs the next-states read)
    let mut leaves: Vec<usize> = supports.iter().flat_map(|r| r.support.iter().copied()).collect();
    leaves.sort_unstable();
    leaves.dedup();
    let leaf_offsets: Vec<usize> = leaves.iter().map(|&i| off(i)).collect();

    for &correlated in &[true, false] {
        let mut runner = InterpRunner::new(InterpProgram::encode_design(&program).unwrap(), lanes);
        runner.set_signal(clk_off, &vec![1u32; lanes]);

        // per-lane previous leaf snapshot (one u32/leaf — these designs are 8-bit)
        let mut prev_leaf: Vec<Vec<u32>> = vec![vec![0u32; lanes]; leaf_offsets.len()];
        let mut prev_din = vec![0u32; lanes];
        let mut have_prev = false;
        let (mut global_skip, mut global_tot) = (0u64, 0u64); // tile-uniform whole-design
        let (mut cone_skip, mut cone_tot) = (0u64, 0u64); // per-lane cone ceiling
        let mut lcg: u64 = 0x1234_5678;
        let mut rng = || { lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1); (lcg >> 33) as u32 };

        for cyc in 0..cycles {
            let en_v: Vec<u32> = (0..lanes).map(|l| {
                let phase = if correlated { 0 } else { l % duty };
                ((cyc + phase) % duty == 0) as u32
            }).collect();
            let din_v: Vec<u32> = (0..lanes).map(|l| if en_v[l] != 0 { rng() & 0xff } else { prev_din[l] }).collect();
            runner.set_signal(en_off, &en_v);
            runner.set_signal(din_off, &din_v);
            runner.tick();

            // snapshot current leaves
            let cur_leaf: Vec<Vec<u32>> = leaf_offsets.iter().map(|&o| runner.get_signal(o)).collect();

            if have_prev {
                // per-lane stable bit (all leaves equal to last cycle)
                let stable: Vec<bool> = (0..lanes).map(|l| {
                    cur_leaf.iter().zip(&prev_leaf).all(|(c, p)| c[l] == p[l])
                }).collect();
                // (1) whole-design tile-uniform skip: a tile skips iff ALL its lanes stable
                for t in (0..lanes).step_by(tile) {
                    let end = (t + tile).min(lanes);
                    global_tot += 1;
                    if (t..end).all(|l| stable[l]) { global_skip += 1; }
                }
                // (2) cone ceiling: per (cone, lane), stable iff that cone's support leaves stable
                for rs in &supports {
                    let cone_leaf_offs: Vec<usize> = rs.support.iter().map(|&i| off(i)).collect();
                    for l in 0..lanes {
                        cone_tot += 1;
                        let stable_cone = cone_leaf_offs.iter().all(|&o| {
                            let li = leaf_offsets.iter().position(|&x| x == o).unwrap();
                            cur_leaf[li][l] == prev_leaf[li][l]
                        });
                        if stable_cone { cone_skip += 1; }
                    }
                }
            }
            prev_leaf = cur_leaf;
            prev_din = din_v;
            have_prev = true;
        }
        let pct = |a: u64, b: u64| 100.0 * a as f64 / b.max(1) as f64;
        println!(
            "  [{label} | {}] whole-design tile-skip(tile={tile}): {:.1}%   cone ceiling(per-lane): {:.1}%   gap: {:.1} pts",
            if correlated { "correlated" } else { "decorrelated" },
            pct(global_skip, global_tot), pct(cone_skip, cone_tot),
            pct(cone_skip, cone_tot) - pct(global_skip, global_tot),
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (depth, lanes, cycles, duty, tile) = (p(1, 8), p(2, 256), p(3, 400), p(4, 16), p(5, 64));
    println!("depth={depth} lanes={lanes} cycles={cycles} duty=1/{duty} tile(workgroup)={tile}");
    println!("Q: does whole-design tile-uniform skip (the GPU-tractable granularity) survive vs the cone ceiling?");
    run("stall-only", false, depth, lanes, cycles, duty, tile);
    run("stall+counter", true, depth, lanes, cycles, duty, tile);
}
