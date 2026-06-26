//! Measure-first sizing for a BIT-PARALLEL gate-level batch engine.
//!
//! Idea: for a gate-level / boolean design, store each 1-bit signal's state
//! across L lanes as ceil(L/64) u64 words (one bit per lane) and evaluate gates
//! (And/Or/Xor/Not/Mux) as plain scalar bitwise ops — one u64 op advances 64
//! lanes. vs the SIMD vector JIT's I8X16 (16 lanes, each wasting 7/8 bits on a
//! 1-bit signal), this is ~4x the lane density with cheaper ops.
//!
//! This probe does NOT build the engine. It measures whether a gate-level design
//! is actually all-1-bit and all-bitwise (so the model applies), the op mix, and
//! the projected lane-density gain.
//! Build: cargo run --release -p rrtl-sim-ir --example bitparallel_probe -- [bits depth cones]
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program, PackedInstrKind, PackedMachineProgram};
use rrtl_core::{compile, uint, Design};

/// A gate-level-style netlist: `cones` register cones, each a depth-`depth` tree
/// of NAND/NOR/XNOR over `bits`-wide signals (bits=1 = a true gate netlist).
fn build(bits: u32, depth: usize, cones: usize) -> Design {
    let mut design = Design::new();
    {
        let mut m = design.module("Gates");
        let clk = m.input("clk", uint(1));
        let a = m.input("a", uint(bits));
        let b = m.input("b", uint(bits));
        for c in 0..cones {
            let acc = m.reg(format!("g{c}"), uint(bits));
            m.clock(acc, clk);
            let mut x = acc.value();
            for d in 0..depth {
                x = match (c + d) % 3 {
                    0 => !(x & a.value()),
                    1 => !(x | b.value()),
                    _ => !(x ^ a.value()),
                };
            }
            m.next(acc, x);
            let o = m.output(format!("o{c}"), uint(bits));
            m.assign(o, acc);
        }
    }
    design
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let bits = p(1, 1) as u32;
    let (depth, cones) = (p(2, 16), p(3, 64));

    let design = build(bits, depth, cones);
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Gates").unwrap();
    let machine: PackedMachineProgram = lower_to_machine_program(&program);

    // Signal width distribution.
    let widths: Vec<u32> = program.signals.iter().map(|s| s.layout.width).collect();
    let n1 = widths.iter().filter(|&&w| w == 1).count();
    let nwide = widths.len() - n1;
    let maxw = widths.iter().copied().max().unwrap_or(0);

    // Op-kind distribution across all four streams.
    let mut total = 0usize;
    let mut bitwise = 0usize; // And/Or/Xor/Not/Mux/Signal/Lit — bit-parallel-able
    let mut arith = 0usize; // Add/Sub/Mul/Eq/Ne/Lt — need bit-sliced adders
    let mut widthop = 0usize; // Slice/Concat/Zext/Sext/Trunc/Cast/MemRead
    let mut counts: std::collections::BTreeMap<&str, usize> = Default::default();
    for blk in [
        &machine.streams.async_reset_comb,
        &machine.streams.comb,
        &machine.streams.tick_next,
        &machine.streams.tick_commit,
    ] {
        for pkt in &blk.packets {
            for instr in &pkt.instrs {
                total += 1;
                use PackedInstrKind::*;
                let k = match &instr.kind {
                    And(..) => "And", Or(..) => "Or", Xor(..) => "Xor", Not(..) => "Not",
                    Mux { .. } => "Mux", Signal(_) => "Signal", Lit(_) => "Lit",
                    Add(..) => "Add", Sub(..) => "Sub", Mul(..) => "Mul",
                    Eq(..) => "Eq", Ne(..) => "Ne", Lt { .. } => "Lt",
                    Slice { .. } => "Slice", Concat(_) => "Concat", Zext(_) => "Zext",
                    Sext(_) => "Sext", Trunc(_) => "Trunc", Cast(_) => "Cast", MemRead { .. } => "MemRead",
                };
                *counts.entry(k).or_default() += 1;
                match &instr.kind {
                    And(..) | Or(..) | Xor(..) | Not(..) | Mux { .. } | Signal(_) | Lit(_) => bitwise += 1,
                    Add(..) | Sub(..) | Mul(..) | Eq(..) | Ne(..) | Lt { .. } => arith += 1,
                    _ => widthop += 1,
                }
            }
        }
    }

    println!("Bit-parallel sizing for a gate-level design (bits={bits}, depth={depth}, cones={cones})");
    println!("  signals          : {} total — {n1} are 1-bit ({:.0}%), {nwide} wider (max width {maxw})",
        widths.len(), 100.0 * n1 as f64 / widths.len().max(1) as f64);
    println!("  machine ops      : {total} total");
    println!("    bit-parallel-able (And/Or/Xor/Not/Mux/Signal/Lit): {bitwise} ({:.0}%)",
        100.0 * bitwise as f64 / total.max(1) as f64);
    println!("    arithmetic (need bit-sliced adders)              : {arith} ({:.0}%)",
        100.0 * arith as f64 / total.max(1) as f64);
    println!("    width ops (slice/concat/ext)                     : {widthop} ({:.0}%)",
        100.0 * widthop as f64 / total.max(1) as f64);
    println!("  op mix: {:?}", counts);

    // Projected lane density: bit-parallel packs 64 lanes per u64 word (or 128
    // per u128 / 512 per AVX-512). The SIMD vector JIT uses I8X16 = 16 lanes for
    // ≤8-bit signals. So for an all-1-bit design, bit-parallel = 4x the lanes per
    // 128-bit register, with bitwise (cheaper) ops.
    let applicable = bits == 1 && nwide == 0 && arith == 0;
    println!("\n  VERDICT: {}", if applicable {
        "PURE gate-level → bit-parallel applies directly (64 lanes/u64, ~4x the I8X16 SIMD density)"
    } else if arith > 0 {
        "has arithmetic → needs bit-sliced adders for those ops (partial: bitwise cones still pack)"
    } else {
        "has multi-bit signals → needs per-bit-position slicing (bit-slicing), more complex"
    });
}
