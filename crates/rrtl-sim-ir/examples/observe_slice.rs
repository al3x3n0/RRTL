//! Observability slicing — "compute only what you observe." A design computes K
//! independent results but the testbench checks only ONE. The backward cone-of-
//! influence from the observed signal keeps just its logic; everything else is
//! dead and is pruned. Static (the observed set is fixed), so the pruned program
//! is bit-exact on the observation and runs faster on every backend — here the JIT.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example observe_slice
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn gen_sv(k: usize, d: usize) -> String {
    // K independent depth-d multiply pipelines, each driving its own output.
    let consts = ["9e3779b1", "85ebca77", "c2b2ae35", "27d4eb2f", "165667b1", "ff51afd7"];
    let mut s = String::from("module top(input clk, input [31:0] din");
    for i in 0..k {
        s.push_str(&format!(", output [31:0] o{i}"));
    }
    s.push_str(");\n  reg [31:0] ");
    let regs: Vec<String> = (0..k).flat_map(|i| (0..d).map(move |j| format!("p{i}_{j}"))).collect();
    s.push_str(&regs.join(","));
    s.push_str(";\n  always @(posedge clk) begin\n");
    for i in 0..k {
        s.push_str(&format!("    p{i}_0 <= (din ^ 32'd{i}) * 32'h{} + 32'd1;\n", consts[0]));
        for j in 1..d {
            let c = consts[j % consts.len()];
            s.push_str(&format!("    p{i}_{j} <= (p{i}_{} ^ (p{i}_{}<<{}) ^ (p{i}_{}>>{})) * 32'h{c} + p{i}_{};\n",
                j - 1, j - 1, 3 + (j % 7), j - 1, 2 + (j % 5), j - 1));
        }
    }
    s.push_str("  end\n");
    for i in 0..k {
        s.push_str(&format!("  assign o{i} = p{i}_{};\n", d - 1));
    }
    s.push_str("endmodule\n");
    s
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::jit::JitSimulator;
    use rrtl_sim_ir::{cone_of_influence, lower_to_machine_program, lower_to_packed_program, slice_present, PackedMachineProgram};
    use rrtl_sv_frontend::import_sv;

    let (k, d) = (16usize, 16usize);
    let src = gen_sv(k, d);
    let imported = import_sv(&src, Some("top")).unwrap();
    let compiled = rrtl_core::compile(&imported.design).unwrap();
    let program = lower_to_packed_program(&compiled, "top").unwrap();
    let full = lower_to_machine_program(&program);
    let gidx = |nm: &str| {
        let h = compiled.find_module("top").unwrap().signals.iter().find(|s| s.name == nm).unwrap().handle;
        program.signal_index(h).unwrap()
    };
    // observe ONLY pipeline 0's last register (a true register, not an output-port
    // comb alias) → cone-of-influence → slice away the other 15 pipelines.
    let (din_g, o0_g) = (gidx("din"), gidx(&format!("p0_{}", d - 1)));
    let (sig_mask, mem_mask) = cone_of_influence(&program, &[o0_g], &[]);
    let kept_sigs = sig_mask.iter().filter(|x| **x).count();
    let slice = slice_present(&program, &sig_mask, &mem_mask).unwrap();
    let sliced = lower_to_machine_program(&slice.program);
    // observed + din local indices inside the sliced program. (clk is correctly NOT
    // in the cone — it drives no data path; RRTL is cycle-based, each tick is an edge.)
    let o0_local = slice.signal_origin.iter().position(|&g| g == o0_g).unwrap();
    let din_l = slice.signal_origin.iter().position(|&g| g == din_g).unwrap();

    let instrs = |m: &PackedMachineProgram| -> usize {
        [&m.streams.async_reset_comb, &m.streams.comb, &m.streams.tick_next, &m.streams.tick_commit]
            .iter().map(|b| b.packets.iter().map(|p| p.instrs.len()).sum::<usize>()).sum()
    };
    let (fi, si) = (instrs(&full), instrs(&sliced));

    // throughput + bit-exactness on the observed register
    let bench = |m: &PackedMachineProgram, din: usize, o0: usize| -> (f64, u64) {
        let mut j = JitSimulator::compile(m).unwrap();
        j.set_signal(din, 0x9e37_79b9);
        j.tick_many(20_000); // warm
        let t = std::time::Instant::now();
        j.tick_many(1_000_000);
        let secs = t.elapsed().as_secs_f64();
        (1_000_000.0 / secs / 1e6, j.get_signal(o0))
    };
    let (full_mc, full_o0) = bench(&full, din_g, o0_g);
    let (slice_mc, slice_o0) = bench(&sliced, din_l, o0_local);

    println!("observability slicing — {k} independent pipelines (depth {d}), observe only o0");
    println!("  signals kept in cone : {kept_sigs}/{} ({:.0}% pruned)", program.signals.len(),
        100.0 * (1.0 - kept_sigs as f64 / program.signals.len() as f64));
    println!("  machine instrs       : {fi} → {si}  ({:.0}% pruned)", 100.0 * (1.0 - si as f64 / fi as f64));
    println!("  JIT throughput       : {full_mc:.1} → {slice_mc:.1} Mcyc/s  ({:.2}x)", slice_mc / full_mc);
    println!("  o0 bit-exact (full == sliced): {}", if full_o0 == slice_o0 { "YES" } else { "NO" });
    println!("\n  => the cone-of-influence keeps only the observed pipeline; the other {} are dead and",
        k - 1);
    println!("     pruned — a static win that also multiplies the batch/GPU throughput.");
}
