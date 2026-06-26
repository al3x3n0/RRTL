//! Measure-first sizing for a BIT-SLICED multi-bit batch engine. A W-bit signal
//! across L lanes is stored as W bit-planes (each plane = L lanes packed in
//! machine words), so EVERY width runs at the full 128-lane (NEON) density —
//! unlike the vector JIT, which picks ONE width-class by the design's MAX width
//! (picorv32 = 32-bit → I32X4 = only 4 lanes) and runs even the many 1-bit
//! control signals at that poor density.
//!
//! Plane-cost per op: bitwise/mux/cmp ≈ W plane-ops; add/sub ≈ W (ripple carry);
//! mul ≈ W² (bit-serial); slice/concat/ext ≈ free (plane reindex). The win is
//! `128 / Σ plane_ops` vs the vector JIT's `cfg_lanes / num_ops`. This probe
//! computes that projected ratio (build only if it pays).
//! Build: cargo run --release -p rrtl-sim-ir --example bitslice_probe -- [v] [top]
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program, PackedInstrKind, PackedMachineProgram};
use rrtl_sv_frontend::import_sv;

fn plane_cost(kind: &PackedInstrKind, w: u32) -> (u64, u64) {
    // returns (bitwise/ripple plane-ops, mul plane-ops) — width ops are free.
    use PackedInstrKind::*;
    let w = w as u64;
    match kind {
        And(..) | Or(..) | Xor(..) | Not(..) | Mux { .. } => (w, 0),
        Eq(..) | Ne(..) | Lt { .. } => (w, 0),       // xor + reduce / comparator ≈ W
        Add(..) | Sub(..) => (w, 0),                 // ripple carry ≈ W
        Mul(..) => (0, w * w),                        // bit-serial multiply ≈ W²
        Signal(_) | Lit(_) => (w, 0),                 // touch W planes
        Slice { .. } | Concat(_) | Zext(_) | Sext(_) | Trunc(_) | Cast(_) => (0, 0), // plane reindex
        MemRead { .. } => (w, 0),
    }
}

fn vec_lanes(maxw: u32) -> Option<u32> {
    match maxw {
        0..=8 => Some(16),
        9..=16 => Some(8),
        17..=32 => Some(4),
        _ => None, // vector JIT can't do >32-bit — bit-slice's exclusive domain
    }
}

fn run(machine: &PackedMachineProgram, label: &str) {
    let mut num_ops = 0u64;
    let mut plane = 0u64;
    let mut mul_plane = 0u64;
    let mut maxw = 1u32;
    let mut wsum = 0u64;
    let mut w1 = 0u64; // # of 1-bit ops (where bit-slice beats the wide vector cfg most)
    for blk in [&machine.streams.comb, &machine.streams.tick_next, &machine.streams.async_reset_comb] {
        for p in &blk.packets {
            for instr in &p.instrs {
                num_ops += 1;
                let w = instr.ty.width;
                maxw = maxw.max(w);
                wsum += w as u64;
                if w == 1 {
                    w1 += 1;
                }
                let (b, m) = plane_cost(&instr.kind, w);
                plane += b;
                mul_plane += m;
            }
        }
    }
    let total_plane = plane + mul_plane;
    println!("[{label}] {num_ops} ops, max width {maxw}, avg op width {:.1}, {w1} are 1-bit ({:.0}%)",
        wsum as f64 / num_ops.max(1) as f64, 100.0 * w1 as f64 / num_ops.max(1) as f64);
    println!("  bit-slice plane-ops: {total_plane} (bitwise/ripple {plane} + mul {mul_plane}, {:.0}% mul)",
        100.0 * mul_plane as f64 / total_plane.max(1) as f64);
    // throughput ∝ lanes_per_op / ops_per_cycle. bit-slice = 128 / total_plane.
    let bs = 128.0 / total_plane.max(1) as f64;
    match vec_lanes(maxw) {
        Some(vl) => {
            let vec = vl as f64 / num_ops as f64;
            println!("  projected vs vector JIT (I*X*, {vl} lanes): bit-slice {:.2}x  ({} plane-ops vs {} vec-ops)",
                bs / vec, total_plane, num_ops);
        }
        None => println!("  vector JIT CANNOT run this (>32-bit); bit-slice is the only lane-parallel option"),
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
    run(&machine, &top);
}
