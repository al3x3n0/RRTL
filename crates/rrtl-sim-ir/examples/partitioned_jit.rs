//! Partitioned / parallel / incremental Cranelift JIT for large designs.
//!
//! A huge design lowered into ONE Cranelift function blows up compile time and
//! code size (Verilator's C++ wall, relocated to the JIT). Instead we slice the
//! design into independent register-cone partitions (the existing RepCut-style
//! slicer), compile each as a SEPARATE function — in PARALLEL across cores, and
//! INCREMENTALLY re-compilable (edit one module → recompile one cone, not the
//! world) — and run them with register-stable boundary exchange.
//!
//! Demonstrated on a large memory-free design (K independent pipelined PEs):
//!   - parallel compile time vs the monolithic one-function compile,
//!   - incremental recompile of a single partition,
//!   - bit-exact partitioned execution vs the monolithic JIT.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example partitioned_jit -- [K]
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

/// Generate a large memory-free design: `k` independent PEs (deep multiply
/// pipelines) fed from `din`, outputs XOR-reduced to `dout`. Partitions cleanly
/// by instance (`top.pe{i}`), with all cross-partition signals register-stable.
#[cfg(feature = "jit")]
fn gen_sv(k: usize) -> String {
    let mut s = String::new();
    // SALT is internal (parameter), so each PE's only boundary input is `din`
    // (a stable top input). Outputs are registers. → all boundaries register-stable.
    s.push_str("module pe #(parameter [31:0] SALT = 0) (input clk, input [31:0] din, output reg [31:0] dout);\n");
    s.push_str("  reg [31:0] s1,s2,s3,s4,s5,s6;\n");
    s.push_str("  always @(posedge clk) begin\n");
    s.push_str("    s1 <= (din ^ SALT) * 32'h9e3779b1 + 32'd1;\n");
    s.push_str("    s2 <= (s1 ^ (s1<<5) ^ (s1>>3)) * 32'h85ebca77;\n");
    s.push_str("    s3 <= s2 * 32'hc2b2ae35 + s1;\n");
    s.push_str("    s4 <= (s3 ^ (s3<<7)) * 32'h27d4eb2f;\n");
    s.push_str("    s5 <= s4 * 32'h165667b1 + s3;\n");
    s.push_str("    s6 <= (s5 ^ (s5>>11)) * 32'h9e3779b1;\n");
    s.push_str("    dout <= s6 + s5;\n");
    s.push_str("  end\nendmodule\n\n");
    s.push_str("module top(input clk, input [31:0] din, output reg [31:0] dout);\n");
    for i in 0..k {
        s.push_str(&format!("  wire [31:0] o{i};\n"));
    }
    for i in 0..k {
        s.push_str(&format!("  pe #(.SALT(32'd{i})) pe{i}(.clk(clk), .din(din), .dout(o{i}));\n"));
    }
    s.push_str("  always @(posedge clk) dout <= ");
    s.push_str(&(0..k).map(|i| format!("o{i}")).collect::<Vec<_>>().join(" ^ "));
    s.push_str(";\nendmodule\n");
    s
}

#[cfg(feature = "jit")]
fn run() {
    use rayon::prelude::*;
    use rrtl_sim_ir::jit::JitSimulator;
    use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program, slice_packed_program, PackedSliceGroup};
    use rrtl_sv_frontend::import_sv;
    use std::collections::HashSet;
    use std::time::Instant;

    let k: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let src = gen_sv(k);
    let imported = import_sv(&src, Some("top")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "top").expect("lower packed");
    let full_machine = lower_to_machine_program(&program);
    let total_instrs: usize = [
        &full_machine.streams.async_reset_comb, &full_machine.streams.comb,
        &full_machine.streams.tick_next, &full_machine.streams.tick_commit,
    ].iter().map(|b| b.packets.iter().map(|p| p.instrs.len()).sum::<usize>()).sum();

    let gidx = |name: &str| {
        let h = compiled.find_module("top").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle;
        program.signal_index(h).unwrap()
    };
    let (clk_g, din_g, dout_g) = (gidx("clk"), gidx("din"), gidx("dout"));

    // ---- partition: one group for the top glue + one per PE instance ----
    let mut groups = vec![PackedSliceGroup { id: "top".into(), owned_paths: vec!["top".into()] }];
    for i in 0..k {
        groups.push(PackedSliceGroup { id: format!("pe{i}"), owned_paths: vec![format!("top.pe{i}")] });
    }
    let slices = slice_packed_program(&program, &groups).expect("slice");

    // ---- COMPILE: monolithic vs partitioned (parallel) ----
    let t = Instant::now();
    let _mono0 = JitSimulator::compile(&full_machine).expect("mono compile");
    let mono_secs = t.elapsed().as_secs_f64();

    let slice_machines: Vec<_> = slices.iter().map(|s| lower_to_machine_program(&s.program)).collect();
    // serial sum (total compile work) ...
    let t = Instant::now();
    let _serial: Vec<JitSimulator> = slice_machines.iter().map(|m| JitSimulator::compile(m).unwrap()).collect();
    let serial_secs = t.elapsed().as_secs_f64();
    // ... and parallel wall-clock (rayon across cores)
    let t = Instant::now();
    let mut jits: Vec<JitSimulator> = slice_machines.par_iter().map(|m| JitSimulator::compile(m).unwrap()).collect();
    let par_secs = t.elapsed().as_secs_f64();
    // incremental: recompile a single partition
    let t = Instant::now();
    let _one = JitSimulator::compile(&slice_machines[1]).unwrap();
    let incr_secs = t.elapsed().as_secs_f64();

    let max_slice_instrs = slice_machines.iter().map(|m| {
        [&m.streams.async_reset_comb, &m.streams.comb, &m.streams.tick_next, &m.streams.tick_commit]
            .iter().map(|b| b.packets.iter().map(|p| p.instrs.len()).sum::<usize>()).sum::<usize>()
    }).max().unwrap();

    // ---- build the boundary-exchange plan (all boundaries register-stable) ----
    // local index of global `g` within slice `s`, if present
    let local_of = |s: usize, g: usize| slices[s].signal_origin.iter().position(|&x| x == g);
    let inputs = [clk_g, din_g];
    // per slice: (input_global, local) to set each cycle
    let input_sets: Vec<Vec<(usize, usize)>> = (0..slices.len())
        .map(|s| inputs.iter().filter_map(|&g| local_of(s, g).map(|l| (g, l))).collect())
        .collect();
    // owner of each register global = the slice where it's present and NOT a boundary input
    let owner = |g: usize| -> (usize, usize) {
        for (s, sl) in slices.iter().enumerate() {
            let bset: HashSet<usize> = sl.boundary_inputs.iter().copied().collect();
            if let Some(l) = sl.signal_origin.iter().position(|&x| x == g) {
                if !bset.contains(&l) {
                    return (s, l);
                }
            }
        }
        panic!("no owner for global {g}");
    };
    // per slice: routes (owner_slice, owner_local, my_boundary_local) for non-input boundaries
    let routes: Vec<Vec<(usize, usize, usize)>> = slices.iter().map(|sl| {
        sl.boundary_inputs.iter().filter_map(|&bl| {
            let g = sl.signal_origin[bl];
            if inputs.contains(&g) { return None; } // inputs set by host directly
            let (os, ol) = owner(g);
            Some((os, ol, bl))
        }).collect()
    }).collect();
    let (dout_s, dout_l) = owner(dout_g);

    // ---- run: monolithic reference vs partitioned, compare dout ----
    let cycles = 400u64;
    let stim = |c: u64| (1u64, c.wrapping_mul(2_654_435_761) & 0xffff_ffff); // (clk, din)
    let mut ref_trace = vec![0u64; cycles as usize];
    let mut mono = JitSimulator::compile(&full_machine).unwrap();
    for c in 0..cycles {
        let (clk, din) = stim(c);
        mono.set_signal(clk_g, clk);
        mono.set_signal(din_g, din);
        mono.tick();
        ref_trace[c as usize] = mono.get_signal(dout_g);
    }

    let mut mism = 0u64;
    for c in 0..cycles {
        let (clk, din) = stim(c);
        // set inputs on every slice that has them
        for (s, sets) in input_sets.iter().enumerate() {
            for &(g, l) in sets {
                jits[s].set_signal(l, if g == clk_g { clk } else { din });
            }
        }
        // exchange register boundaries (producers' current reg values)
        for s in 0..jits.len() {
            for &(os, ol, bl) in &routes[s] {
                let v = jits[os].get_signal(ol);
                jits[s].set_signal(bl, v);
            }
        }
        // tick all partitions
        for j in jits.iter_mut() {
            j.tick();
        }
        if jits[dout_s].get_signal(dout_l) != ref_trace[c as usize] {
            mism += 1;
        }
    }

    let mc = |s: f64| if s > 0.0 { s * 1e3 } else { 0.0 };
    println!("partitioned JIT — large memory-free design: top + {k} PEs");
    println!("{total_instrs} machine instrs, sliced into {} partitions (max {max_slice_instrs} instrs each)\n", slices.len());
    println!("compile time:");
    println!("  monolithic (1 function)      : {:.1} ms", mc(mono_secs));
    println!("  partitioned serial (sum)     : {:.1} ms", mc(serial_secs));
    println!("  partitioned PARALLEL (rayon) : {:.1} ms   =>  {:.2}x vs monolithic", mc(par_secs), mono_secs / par_secs);
    println!("  incremental (recompile 1 PE) : {:.1} ms   =>  {:.0}x cheaper than full", mc(incr_secs), mono_secs / incr_secs.max(1e-9));
    println!("\nexecution ({cycles} cycles): partitioned vs monolithic mismatches = {mism}");
    println!("  => {}", if mism == 0 {
        "parallel + incremental compile, bit-exact partitioned execution"
    } else {
        "FAIL (partitioned execution diverged)"
    });
}
