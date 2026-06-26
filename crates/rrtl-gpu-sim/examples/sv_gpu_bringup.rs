//! End-to-end bringup of the SystemVerilog frontend through the GPU interpreter
//! path: SV source -> `import_sv` -> `compile` -> `lower_to_packed_program` ->
//! `InterpProgram::encode_design` -> InterpRunner (CPU interp) and
//! InterpGpuSimulator (GPU). Both are validated cycle-by-cycle against the gold
//! `rrtl_core::Simulator` reference on the same imported design.
//!
//! Usage: cargo run --release -p rrtl-gpu-sim --example sv_gpu_bringup

use std::time::Instant;

use rrtl_core::{compile, Signal, Simulator};
use rrtl_gpu_sim::interp::{InterpGpuSimulator, InterpProgram, InterpRunner};
use rrtl_sim_ir::lower_to_packed_program;
use rrtl_sv_frontend::import_sv;

/// A small mixed design: combinational add + compare, plus a sequential
/// accumulator with synchronous reset and an enable.
const SV: &str = r#"
module Dut(
  input  logic       clk,
  input  logic       rst,
  input  logic       en,
  input  logic [7:0] a,
  input  logic [7:0] b,
  output logic [7:0] sum,
  output logic       eq,
  output logic [7:0] acc
);
  assign sum = a + b;
  assign eq  = (a == b);

  logic [7:0] acc_r;
  assign acc = acc_r;

  always_ff @(posedge clk) begin
    if (rst)      acc_r <= 8'd0;
    else if (en)  acc_r <= acc_r + sum;
  end
endmodule
"#;

fn main() {
    // ---- Frontend: parse + elaborate SystemVerilog into a Design. ----
    let imported = import_sv(SV, Some("Dut")).expect("SV import failed");
    let top = imported.top_name.clone();
    println!("imported top=`{top}` modules={:?}", imported.modules);

    // ---- Lower the imported design through the packed/interp path. ----
    let compiled = compile(&imported.design).expect("compile failed");
    let program = lower_to_packed_program(&compiled, &top).expect("lowering failed");
    let encoded = InterpProgram::encode_design(&program).expect("encode failed");
    println!(
        "lowered: {} signals, {} values, {} instr-words",
        program.signals.len(),
        encoded.num_values,
        encoded.total_code_words(),
    );

    // ---- Signal handles (for the gold sim) and storage offsets (for interp). ----
    let handle = |name: &str| -> Signal {
        compiled
            .find_module(&top)
            .unwrap()
            .signals
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("no signal `{name}`"))
            .handle
    };
    let off = |s: Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;
    let inputs = ["rst", "en", "a", "b"];
    let outputs = ["sum", "eq", "acc"];

    // ---- Reference: gold CPU Simulator on the imported design. ----
    let mut gold = Simulator::new(&imported.design, &top).unwrap();
    // ---- Under test: the interpreter CPU reference (deterministic). ----
    let mut interp = InterpRunner::new(encoded.clone(), 1);
    interp.set_signal(off(handle("clk")), &[1]);

    // Deterministic pseudo-random stimulus.
    let mut lcg: u32 = 0x1234_5678;
    let mut next = || {
        lcg = lcg.wrapping_mul(1664525).wrapping_add(1013904223);
        lcg
    };

    let mut mismatches = 0usize;
    for cycle in 0..40u32 {
        let stim: Vec<(&str, u128)> = inputs
            .iter()
            .map(|&n| {
                let v = match n {
                    "rst" => (cycle == 0) as u128, // reset on cycle 0 only
                    "en" => (next() & 1) as u128,
                    _ => (next() & 0xff) as u128,
                };
                (n, v)
            })
            .collect();
        for &(n, v) in &stim {
            gold.set(handle(n), v);
            interp.set_signal(off(handle(n)), &[v as u32]);
        }
        gold.tick();
        interp.tick();
        for &o in &outputs {
            let g = gold.get(handle(o));
            let i = interp.get_signal(off(handle(o)))[0] as u128;
            if g != i {
                if mismatches < 5 {
                    println!("  cycle {cycle} {o}: gold={g} interp={i}");
                }
                mismatches += 1;
            }
        }
    }
    println!(
        "CPU interp vs gold Simulator: {}",
        if mismatches == 0 { "OK (40 cycles)" } else { "MISMATCH" }
    );

    // ---- GPU bringup: same design on the wgpu backend, replicated across lanes. ----
    let lanes = 256;
    let gpu = match InterpGpuSimulator::new(&encoded, lanes) {
        Ok(g) => g,
        Err(e) => {
            println!("GPU unavailable ({e:?}); CPU path validated above.");
            return;
        }
    };
    let mut gold2 = Simulator::new(&imported.design, &top).unwrap();
    gpu.set_signal(off(handle("clk")), &vec![1u32; lanes]);
    let mut gpu_mismatch = 0usize;
    let t0 = Instant::now();
    for cycle in 0..20u32 {
        for &n in &inputs {
            let v = match n {
                "rst" => (cycle == 0) as u32,
                "en" => next() & 1,
                _ => next() & 0xff,
            };
            gold2.set(handle(n), v as u128);
            gpu.set_signal(off(handle(n)), &vec![v; lanes]); // same value on every lane
        }
        gold2.tick();
        gpu.tick_many(1);
        gpu.synchronize();
        for &o in &outputs {
            let g = gold2.get(handle(o)) as u32;
            let lane_vals = gpu.get_signal(off(handle(o)));
            if lane_vals.iter().any(|&v| v != g) {
                gpu_mismatch += 1;
                if gpu_mismatch <= 3 {
                    println!("  GPU cycle {cycle} {o}: gold={g} got={:?}", &lane_vals[..4.min(lanes)]);
                }
            }
        }
    }
    println!(
        "GPU ({lanes} lanes) vs gold Simulator: {}  [{:.1} ms]",
        if gpu_mismatch == 0 { "OK (20 cycles)" } else { "MISMATCH" },
        t0.elapsed().as_secs_f64() * 1e3,
    );
}
