//! Zero-copy partitioned JIT: each partition is compiled over the GLOBAL signal
//! layout and shares one pair of state buffers (cur/nxt). A partition reads every
//! other partition's last-cycle registers DIRECTLY from `cur` (no boundary copy at
//! all) and commits its own registers to `nxt`; the runner swaps cur/nxt each
//! cycle — replacing the per-boundary exchange entirely. Because all reads come
//! from the read-only `cur`, partitions are also order-independent (parallelizable).
//!
//! The design is FLAT and partitioned by register cones (partition_registers_
//! balanced), so every cross-partition boundary is a true register (not an output-
//! port comb alias), read straight from `cur`. Each partition's program is the full
//! (already global-indexed) program with only the ops it owns retained — no remap.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example zerocopy_jit -- [stages] [groups]
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn gen_sv(n: usize) -> String {
    // A flat N-stage register pipeline (no module instances): stage i feeds i+1, so
    // a register-cone partition cut leaves a true register on the boundary.
    let consts = ["9e3779b1", "85ebca77", "c2b2ae35", "27d4eb2f", "165667b1", "ff51afd7"];
    let mut s = String::from("module top(input clk, input [31:0] din, output reg [31:0] dout);\n  reg [31:0] ");
    s.push_str(&(0..n).map(|i| format!("r{i}")).collect::<Vec<_>>().join(","));
    s.push_str(";\n  always @(posedge clk) begin\n");
    s.push_str(&format!("    r0 <= din * 32'h{} + 32'd1;\n", consts[0]));
    for i in 1..n {
        let c = consts[i % consts.len()];
        s.push_str(&format!("    r{i} <= (r{} ^ (r{}<<{}) ^ (r{}>>{})) * 32'h{c} + r{};\n",
            i - 1, i - 1, 3 + (i % 7), i - 1, 2 + (i % 5), i - 1));
    }
    s.push_str(&format!("    dout <= r{} + r{};\n  end\nendmodule\n", n - 1, n - 2));
    s
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::jit::JitSimulator;
    use rrtl_sim_ir::{
        lower_to_machine_program, lower_to_packed_program, partition_registers_balanced,
        PackedEffect, PackedOp, PackedProgram,
    };
    use rrtl_sv_frontend::import_sv;
    use std::time::Instant;

    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let groups: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let src = gen_sv(n);
    let imported = import_sv(&src, Some("top")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "top").expect("lower packed");
    let full_machine = lower_to_machine_program(&program);
    let nsig = program.signals.len();
    let gidx = |nm: &str| {
        let h = compiled.find_module("top").unwrap().signals.iter().find(|s| s.name == nm).unwrap().handle;
        program.signal_index(h).unwrap()
    };
    let (clk_g, din_g) = (gidx("clk"), gidx("din"));
    // Observe a true internal register (not the `output reg` port, which is a comb
    // alias of its register and would need a post-commit settle to be current).
    let dout_g = gidx(&format!("r{}", n - 1));

    // GLOBAL register mask (machine-level CaptureReg destinations).
    let mut is_reg = vec![false; nsig];
    for block in [&full_machine.streams.async_reset_comb, &full_machine.streams.tick_next] {
        for packet in &block.packets {
            for eff in &packet.effects {
                if let PackedEffect::CaptureReg { dst, .. } = eff {
                    is_reg[*dst] = true;
                }
            }
        }
    }

    // RepCut-style register-cone partition: per-group present mask over signals.
    let masks = partition_registers_balanced(&program, groups).expect("partition");

    // Each partition's program = the full (global) program with only the ops whose
    // destination its mask owns retained (replicated comb stays in each group).
    let filtered = |keep: &[bool]| -> PackedProgram {
        let mut p = program.clone();
        for stream in [&mut p.streams.async_reset_comb, &mut p.streams.comb, &mut p.streams.tick_next, &mut p.streams.tick_commit] {
            for packet in stream.iter_mut() {
                packet.ops.retain(|op| match op {
                    PackedOp::Assign { dst, .. } | PackedOp::CaptureReg { dst, .. } => keep[*dst],
                    PackedOp::MemoryWrite { .. } => false,
                });
            }
        }
        p
    };
    let jits: Vec<JitSimulator> = masks.iter().map(|mask| {
        let m = lower_to_machine_program(&filtered(mask));
        JitSimulator::compile_db(&m, &is_reg).unwrap()
    }).collect();
    let state_words = jits[0].state_words().len();
    let cuts = is_reg.iter().filter(|&&r| r).count();

    let cycles = 200_000u64;
    let stim = |c: u64| c.wrapping_mul(2_654_435_761) & 0xffff_ffff;

    // reference: monolithic single-buffer
    let mut mono = JitSimulator::compile(&full_machine).unwrap();
    let mut ref_trace = vec![0u64; cycles as usize];
    let t = Instant::now();
    for c in 0..cycles {
        mono.set_signal(clk_g, 1);
        mono.set_signal(din_g, stim(c));
        mono.tick();
        ref_trace[c as usize] = mono.get_signal(dout_g);
    }
    let mono_secs = t.elapsed().as_secs_f64();

    // zero-copy partitioned: shared cur/nxt, NO exchange, swap each cycle
    let (mut cur, mut nxt) = (vec![0i64; state_words], vec![0i64; state_words]);
    let mut mism = 0u64;
    let t = Instant::now();
    for c in 0..cycles {
        nxt[clk_g * 2] = 1;
        nxt[din_g * 2] = stim(c) as i64;
        nxt[din_g * 2 + 1] = 0;
        for j in &jits {
            j.tick_db(&cur, &mut nxt);
        }
        std::mem::swap(&mut cur, &mut nxt);
        if (cur[dout_g * 2] as u64) != ref_trace[c as usize] {
            mism += 1;
        }
    }
    let zc_secs = t.elapsed().as_secs_f64();

    println!("zero-copy partitioned JIT — flat {n}-stage pipeline, {} partitions, {cuts} register signals, {cycles} cycles", masks.len());
    println!("  monolithic (reference) : {:.2} Mcyc/s", cycles as f64 / mono_secs / 1e6);
    println!("  zero-copy partitioned  : {:.2} Mcyc/s | mismatches {mism}", cycles as f64 / zc_secs / 1e6);
    println!("  => {}", if mism == 0 {
        "partitions share one state, read boundary registers directly from cur — NO exchange, bit-exact"
    } else {
        "FAIL (diverged)"
    });
}
