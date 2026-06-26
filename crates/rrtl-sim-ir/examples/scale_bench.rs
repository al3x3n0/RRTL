//! Compile-time / scale moat (Axis 3): as a design grows, Verilator must compile
//! a giant C++ model (super-linear in gcc/clang), while RRTL's interpreter is ready
//! after just lowering (no compile). This emits a flat W×D mul-add datapath to an
//! SV file and times RRTL's "time to first cycle" (import → lower → interp ready).
//! A companion shell step verilates the same file and times that.
//! Build: cargo run --release -p rrtl-sim-ir --example scale_bench -- <W> <D> <out.v>
use rrtl_sim_ir::{lower_to_packed_program, PackedSimulator};
use rrtl_sv_frontend::import_sv;
use std::time::Instant;

fn gen_sv(w: usize, d: usize) -> String {
    let consts = ["9e3779b1", "85ebca77", "c2b2ae35", "27d4eb2f", "165667b1", "ff51afd7"];
    let mut s = String::from("module scale(input clk, input [31:0] din, output [31:0] out);\n");
    s.push_str("  reg [31:0] ");
    s.push_str(&(0..w).map(|i| format!("acc{i}")).collect::<Vec<_>>().join(","));
    s.push_str(";\n  wire [31:0] ");
    let wires: Vec<String> = (0..w).flat_map(|i| (0..d).map(move |j| format!("w{i}_{j}"))).collect();
    s.push_str(&wires.join(","));
    s.push_str(";\n");
    for i in 0..w {
        s.push_str(&format!("  assign w{i}_0 = (acc{i} ^ din) * 32'h{} + din;\n", consts[i % consts.len()]));
        for j in 1..d {
            let c = consts[(i + j) % consts.len()];
            s.push_str(&format!("  assign w{i}_{j} = (w{i}_{} ^ (w{i}_{}<<{})) * 32'h{c} + din;\n",
                j - 1, j - 1, 1 + (j % 7)));
        }
    }
    s.push_str("  always @(posedge clk) begin\n");
    for i in 0..w {
        s.push_str(&format!("    acc{i} <= w{i}_{};\n", d - 1));
    }
    s.push_str("  end\n");
    // balanced XOR reduction tree (log depth, O(W) wires) — a real design's
    // reduction, not a W-deep expression (which makes the parser quadratic).
    let mut level: Vec<String> = (0..w).map(|i| format!("acc{i}")).collect();
    let mut tier = 0;
    while level.len() > 1 {
        let mut next = Vec::new();
        let mut k = 0;
        let mut i = 0;
        while i < level.len() {
            if i + 1 < level.len() {
                let name = format!("r{tier}_{k}");
                s.push_str(&format!("  wire [31:0] {name} = {} ^ {};\n", level[i], level[i + 1]));
                next.push(name);
                k += 1;
                i += 2;
            } else {
                next.push(level[i].clone());
                i += 1;
            }
        }
        level = next;
        tier += 1;
    }
    s.push_str(&format!("  assign out = {};\nendmodule\n", level[0]));
    s
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| a.get(i).and_then(|x| x.parse().ok()).unwrap_or(d);
    let (w, d) = (p(1, 256), p(2, 4));
    let out = a.get(3).cloned().unwrap_or_else(|| "/tmp/scale.v".into());

    let src = gen_sv(w, d);
    std::fs::write(&out, &src).expect("write SV");

    // RRTL "time to first cycle": parse -> compile -> lower -> interpreter ready.
    let t = Instant::now();
    let imported = import_sv(&src, Some("scale")).expect("import_sv");
    let t_imp = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let t_comp = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let program = lower_to_packed_program(&compiled, "scale").expect("lower");
    let t_low = t.elapsed().as_secs_f64() * 1e3;
    let nsig = program.signals.len();
    let t = Instant::now();
    let _sim = PackedSimulator::new(program, 1).expect("interp");
    let t_interp = t.elapsed().as_secs_f64() * 1e3;
    let ms = t_imp + t_comp + t_low + t_interp;

    println!("scale W={w} D={d}: {nsig} signals, {} bytes SV", src.len());
    println!("  RRTL time-to-first-cycle: {ms:.0} ms (import {t_imp:.0} + compile {t_comp:.0} + lower {t_low:.0} + interp {t_interp:.0})  -> wrote {out}");
}
