//! Maps the SystemVerilog frontend's feature frontier: each design is pushed
//! through import -> compile -> lower -> encode -> validate-vs-gold, reporting the
//! first stage that fails. Non-fatal: prints a coverage table.
use rrtl_core::{compile, Signal, SignalKind, Simulator};
use rrtl_gpu_sim::interp::{InterpProgram, InterpRunner};
use rrtl_sim_ir::lower_to_packed_program;
use rrtl_sv_frontend::import_sv;

fn try_validate(sv: &str, top: &str, cycles: u32) -> Result<usize, String> {
    let imported = import_sv(sv, Some(top)).map_err(|e| format!("IMPORT: {e:?}"))?;
    let compiled = compile(&imported.design).map_err(|e| format!("COMPILE: {e:?}"))?;
    let program = lower_to_packed_program(&compiled, top).map_err(|e| format!("LOWER: {e:?}"))?;
    let encoded = InterpProgram::encode_design(&program).map_err(|e| format!("ENCODE: {e:?}"))?;
    let module = compiled.find_module(top).ok_or("no top module")?;
    let off = |s: Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;
    let limbs = |w: u32| (((w + 31) / 32).max(1)) as usize;
    let mask = |w: u32| if w >= 128 { u128::MAX } else { (1u128 << w) - 1 };
    let mut ins = Vec::new();
    let mut outs = Vec::new();
    for s in &module.signals {
        match s.kind {
            SignalKind::Input => ins.push((s.name.clone(), s.handle, s.width)),
            SignalKind::Output => outs.push((s.name.clone(), s.handle, s.width)),
            _ => {}
        }
    }
    if outs.is_empty() { return Err("no outputs".into()); }
    let mut gold = Simulator::new(&imported.design, top).map_err(|e| format!("GOLD: {e:?}"))?;
    let mut interp = InterpRunner::new(encoded, 1);
    let mut lcg: u128 = 0x9e37_79b9_7f4a_7c15;
    let mut rng = || { lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1); lcg >> 64 };
    for cycle in 0..cycles {
        for (n, h, w) in &ins {
            let v = if n == "clk" { 1 } else { rng() & mask(*w) };
            gold.set(*h, v);
            interp.set_signal_wide(off(*h), limbs(*w), &[v]);
        }
        gold.tick();
        interp.tick();
        for (n, h, w) in &outs {
            let g = gold.get(*h) & mask(*w);
            let i = interp.get_signal_wide(off(*h), limbs(*w))[0] & mask(*w);
            if g != i { return Err(format!("MISMATCH {n} @cyc{cycle}: gold={g} interp={i}")); }
        }
    }
    Ok(program.signals.len())
}

fn main() {
    let cases: Vec<(&str, &str, &str)> = vec![
        ("param_width", "Inc", "module Inc #(parameter W=8)(input logic [W-1:0] a, output logic [W-1:0] y); assign y = a + 1; endmodule"),
        ("localparam", "L", "module L(input logic [7:0] a, output logic [7:0] y); localparam logic [7:0] K = 8'd5; assign y = a + K; endmodule"),
        ("reduction_and", "R", "module R(input logic [7:0] a, output logic y); assign y = &a; endmodule"),
        ("reduction_or", "R", "module R(input logic [7:0] a, output logic y); assign y = |a; endmodule"),
        ("reduction_xor", "R", "module R(input logic [7:0] a, output logic y); assign y = ^a; endmodule"),
        ("replication", "R", "module R(input logic [1:0] a, output logic [7:0] y); assign y = {4{a}}; endmodule"),
        ("shift_left", "S", "module S(input logic [7:0] a, output logic [7:0] y); assign y = a << 2; endmodule"),
        ("shift_right", "S", "module S(input logic [7:0] a, output logic [7:0] y); assign y = a >> 2; endmodule"),
        ("var_shift", "S", "module S(input logic [7:0] a, input logic [2:0] n, output logic [7:0] y); assign y = a << n; endmodule"),
        ("arith_shift", "S", "module S(input logic signed [7:0] a, output logic signed [7:0] y); assign y = a >>> 1; endmodule"),
        ("part_select_var", "P", "module P(input logic [15:0] a, input logic [1:0] i, output logic [3:0] y); assign y = a[i*4 +: 4]; endmodule"),
        ("always_comb_if", "C", "module C(input logic s, input logic [7:0] a, b, output logic [7:0] y); always_comb begin if (s) y = a; else y = b; end endmodule"),
        ("always_comb_case", "D", "module D(input logic [1:0] s, output logic [3:0] y); always_comb begin case (s) 2'd0:y=4'd1; 2'd1:y=4'd2; 2'd2:y=4'd4; default:y=4'd8; endcase end endmodule"),
        ("for_popcount", "PC", "module PC(input logic [7:0] a, output logic [3:0] cnt); always_comb begin cnt = 0; for (int i=0;i<8;i++) cnt = cnt + a[i]; end endmodule"),
        ("modulo", "M", "module M(input logic [7:0] a, b, output logic [7:0] y); assign y = a % (b | 8'd1); endmodule"),
        ("divide", "M", "module M(input logic [7:0] a, b, output logic [7:0] y); assign y = a / (b | 8'd1); endmodule"),
        ("casez", "Z", "module Z(input logic [3:0] a, output logic [1:0] y); always_comb begin casez (a) 4'b1???:y=2'd3; 4'b01??:y=2'd2; 4'b001?:y=2'd1; default:y=2'd0; endcase end endmodule"),
        ("genfor", "G", "module G(input logic [3:0] a, b, output logic [3:0] y); genvar i; generate for (i=0;i<4;i=i+1) begin: g assign y[i] = a[i] & b[i]; end endgenerate endmodule"),
        ("generate_if", "GI", "module GI #(parameter EN=1)(input logic [7:0] a, output logic [7:0] y); generate if (EN) begin assign y = a + 1; end else begin assign y = a; end endgenerate endmodule"),
        ("function", "F", "module F(input logic [7:0] a, output logic [7:0] y); function automatic logic [7:0] inc(input logic [7:0] x); inc = x + 8'd1; endfunction assign y = inc(a); endmodule"),
        ("param_inst", "T", "module Sub #(parameter W=8)(input logic [W-1:0] a, output logic [W-1:0] y); assign y=a+1; endmodule\nmodule T(input logic [15:0] a, output logic [15:0] y); Sub #(.W(16)) u(.a(a),.y(y)); endmodule"),
        ("enum_typedef", "E", "module E(input logic [1:0] s, output logic [1:0] y); typedef enum logic [1:0] {A,B,C} st; always_comb begin case (s) 2'd0:y=2'd1; default:y=2'd0; endcase end endmodule"),
        ("packed_struct", "PS", "module PS(input logic [7:0] a, output logic [7:0] y); typedef struct packed { logic [3:0] hi; logic [3:0] lo; } pair; pair p; assign p = a; assign y = {p.lo, p.hi}; endmodule"),
        ("signed_mul", "SM", "module SM(input logic signed [7:0] a, b, output logic signed [15:0] y); assign y = a * b; endmodule"),
    ];
    let mut ok = 0; let mut fail = 0;
    println!("{:<18} {:<10} detail", "case", "status");
    println!("{}", "-".repeat(70));
    for (name, top, sv) in &cases {
        match try_validate(sv, top, 20) {
            Ok(_) => { println!("{name:<18} {:<10}", "OK"); ok += 1; }
            Err(e) => {
                let stage = e.split(':').next().unwrap_or("?");
                let short: String = e.chars().take(64).collect();
                println!("{name:<18} {stage:<10} {short}");
                fail += 1;
            }
        }
    }
    println!("{}", "-".repeat(70));
    println!("{ok} OK, {fail} fail / {} total", cases.len());
}
