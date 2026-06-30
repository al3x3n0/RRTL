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
use rrtl_sim_ir::{lower_to_packed_program, packed_signal_layout, register_support, state_store_bytes as store_bytes};
use rrtl_sv_frontend::import_sv;
use std::collections::HashSet;

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

    // Affinity order: each cone's signals, first-seen.
    let mut affinity_order: Vec<usize> = Vec::new();
    let mut seen = vec![false; n];
    for cone in &cones {
        for &s in cone {
            if !seen[s] {
                seen[s] = true;
                affinity_order.push(s);
            }
        }
    }
    let fat: Vec<usize> = (0..n).map(|i| i * 16).collect();
    let (packed, _) = packed_signal_layout(&widths, None);
    let (affinity, _) = packed_signal_layout(&widths, Some(&affinity_order));
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
