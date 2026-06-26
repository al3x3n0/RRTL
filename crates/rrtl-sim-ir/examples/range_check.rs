//! Estimates the limb-reduction opportunity (range analysis) across design shapes:
//! how many declared value-words are provably reducible to fewer limbs. This gates
//! whether the (invasive) kernel-level narrowing is worth implementing.
//!
//! Usage: cargo run --release -p rrtl-sim-ir --example range_check
use rrtl_core::{compile, lit_u, uint, Design, Signal};
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program, range_reduction_stats};

fn report(name: &str, design: &Design, top: &str) {
    let compiled = compile(design).unwrap();
    let program = lower_to_packed_program(&compiled, top).unwrap();
    let machine = lower_to_machine_program(&program);
    let s = range_reduction_stats(&machine);
    println!(
        "  {name:<16} values={:<5} narrowable={:<5} ({:>4.1}%)  words {} -> {}  ({:.1}% saved)",
        s.total_values, s.narrowable_values,
        100.0 * s.narrowable_values as f64 / s.total_values.max(1) as f64,
        s.declared_words, s.effective_words,
        100.0 * s.words_saved() as f64 / s.declared_words.max(1) as f64,
    );
}

fn genuine64(width: usize, depth: usize) -> Design {
    let mut d = Design::new();
    let mut m = d.module("M");
    let clk = m.input("clk", uint(1));
    let din = m.input("din", uint(64));
    for lane in 0..width {
        let acc = m.reg(format!("acc{lane}"), uint(64));
        m.clock(acc, clk);
        let mut prev = acc;
        for s in 0..depth { let w = m.wire(format!("w{lane}_{s}"), uint(64)); m.assign(w, prev * lit_u(0x9e37_79b9_7f4a_7c15,64) + din); prev = w; }
        m.next(acc, prev + acc);
        let o = m.output(format!("o{lane}"), uint(64)); m.assign(o, acc);
    }
    d
}

// 64-bit-typed datapath where the data is really zero-extended 16-bit (common for
// packed-field / narrow-counter logic carried in wide buses).
fn zext_heavy(width: usize, depth: usize) -> Design {
    let mut d = Design::new();
    let mut m = d.module("M");
    let clk = m.input("clk", uint(1));
    let din = m.input("din", uint(16));
    for lane in 0..width {
        let acc = m.reg(format!("acc{lane}"), uint(64));
        m.clock(acc, clk);
        let mut narrow: Signal = din;
        for s in 0..depth { let w = m.wire(format!("n{lane}_{s}"), uint(16)); m.assign(w, narrow + lit_u(1,16)); narrow = w; }
        let wide = m.wire(format!("wide{lane}"), uint(64));
        m.assign(wide, narrow.value().zext(64));
        m.next(acc, wide);
        let o = m.output(format!("o{lane}"), uint(64)); m.assign(o, acc);
    }
    d
}

fn main() {
    report("genuine-64", &genuine64(32, 8), "M");
    report("zext-heavy", &zext_heavy(32, 8), "M");
}
