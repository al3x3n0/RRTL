//! Measure-first for state-layout optimization, in STRUCTURAL terms (cache-line
//! footprint) since single-instance throughput is thermal-noise on this machine.
//!
//! Three layouts, scored by (a) total working-set bytes and (b) the average
//! number of distinct 64-byte cache lines a register cone touches per cycle
//! (its support leaves + the register itself) — a direct proxy for L1 misses:
//!   - FAT:      uniform 16-byte slots (the original AOT layout).
//!   - PACKED:   per-width slots, grouped by size class (the current AOT layout).
//!   - AFFINITY: per-width slots ordered so co-accessed signals are adjacent
//!               (the NEW lever) — cluster each cone's support contiguously.
//! Build: cargo run --release -p rrtl-sim-ir --example state_layout_probe -- [design.sv top]
use rrtl_sim_ir::{lower_to_packed_program, register_support};
use rrtl_sv_frontend::import_sv;
use std::collections::HashSet;

fn store_bytes(w: u32) -> usize {
    match w {
        0..=8 => 1,
        9..=16 => 2,
        17..=32 => 4,
        33..=64 => 8,
        _ => 16,
    }
}

fn align(x: usize, a: usize) -> usize {
    (x + a - 1) / a * a
}

/// Offsets for the current PACKED layout: each size class placed contiguously,
/// largest first (naturally aligned, zero inter-slot padding).
fn packed_offsets(widths: &[u32]) -> Vec<usize> {
    let mut off = vec![0usize; widths.len()];
    let mut cur = 0usize;
    for size in [16usize, 8, 4, 2, 1] {
        for (i, &w) in widths.iter().enumerate() {
            if store_bytes(w) == size {
                cur = align(cur, size);
                off[i] = cur;
                cur += size;
            }
        }
    }
    off
}

/// Offsets for the AFFINITY layout: visit cones in order, assign each cone's
/// not-yet-placed signals contiguous (naturally-aligned) slots, so signals read
/// together land on the same / adjacent cache lines.
fn affinity_offsets(widths: &[u32], cones: &[Vec<usize>]) -> Vec<usize> {
    let mut order: Vec<usize> = Vec::new();
    let mut seen = vec![false; widths.len()];
    for cone in cones {
        for &s in cone {
            if !seen[s] {
                seen[s] = true;
                order.push(s);
            }
        }
    }
    for s in 0..widths.len() {
        if !seen[s] {
            order.push(s);
        }
    }
    let mut off = vec![0usize; widths.len()];
    let mut cur = 0usize;
    for &s in &order {
        let sz = store_bytes(widths[s]);
        cur = align(cur, sz);
        off[s] = cur;
        cur += sz;
    }
    off
}

const LINE: usize = 64;

/// Average distinct cache lines a cone touches under `off`.
fn avg_lines(cones: &[Vec<usize>], off: &[usize]) -> f64 {
    let mut total = 0usize;
    for cone in cones {
        let lines: HashSet<usize> = cone.iter().map(|&s| off[s] / LINE).collect();
        total += lines.len();
    }
    total as f64 / cones.len().max(1) as f64
}

fn run(path: &str, top: &str) {
    let src = std::fs::read_to_string(path).expect("read");
    let imported = import_sv(&src, Some(top)).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, top).expect("lower");
    let widths: Vec<u32> = program.signals.iter().map(|s| s.layout.width).collect();
    let n = widths.len();

    // Each register cone's touched slot-set = its support leaves + the register.
    let supports = register_support(&program);
    let cones: Vec<Vec<usize>> = supports
        .iter()
        .map(|rs| {
            let mut v = rs.support.clone();
            v.push(rs.reg);
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect();

    let w32 = widths.iter().filter(|&&w| w > 16 && w <= 32).count();
    println!("[{top}] {n} signals ({w32} are 17-32 bit), {} register cones, avg cone fan-in {:.1}",
        cones.len(), cones.iter().map(|c| c.len()).sum::<usize>() as f64 / cones.len().max(1) as f64);

    let fat: Vec<usize> = (0..n).map(|i| i * 16).collect();
    let packed = packed_offsets(&widths);
    let affinity = affinity_offsets(&widths, &cones);
    let bytes = |off: &[usize]| off.iter().zip(&widths).map(|(o, w)| o + store_bytes(*w)).max().unwrap_or(0);

    println!("  working set:  fat {:>6} B | packed {:>6} B ({:.2}x smaller) | affinity {:>6} B",
        bytes(&fat), bytes(&packed), bytes(&fat) as f64 / bytes(&packed).max(1) as f64, bytes(&affinity));
    println!("  cache lines touched / cone (avg):  fat {:.2} | packed {:.2} | affinity {:.2} ({:.1}% fewer than packed)",
        avg_lines(&cones, &fat), avg_lines(&cones, &packed), avg_lines(&cones, &affinity),
        100.0 * (avg_lines(&cones, &packed) - avg_lines(&cones, &affinity)) / avg_lines(&cones, &packed).max(1e-9));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 3 {
        run(&args[1], &args[2]);
    } else {
        run("bench/sv/picorv32.v", "picorv32");
        run("bench/sv/cpu.sv", "cpu");
        run("bench/sv/mixed.sv", "mixed");
    }
}
