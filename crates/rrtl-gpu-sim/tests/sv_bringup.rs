//! SystemVerilog frontend → interpreter bringup suite. Each design is imported
//! from SV, lowered through the packed/interp path, and validated cycle-by-cycle
//! against the gold `rrtl_core::Simulator` on random stimulus. Deterministic
//! (CPU interpreter, no GPU), so it runs as a regression test; the GPU path is
//! exercised separately by the `sv_gpu_bringup` example.

use rrtl_core::{compile, Signal, SignalKind, Simulator};
use rrtl_gpu_sim::interp::{InterpProgram, InterpRunner};
use rrtl_sim_ir::lower_to_packed_program;
use rrtl_sv_frontend::import_sv;

/// Import an SV design, lower it through the interpreter, and assert it matches
/// the gold simulator for `cycles` cycles of pseudo-random stimulus.
fn validate(name: &str, sv: &str, top: &str, cycles: u32) {
    let imported =
        import_sv(sv, Some(top)).unwrap_or_else(|e| panic!("{name}: SV import failed: {e:?}"));
    let compiled =
        compile(&imported.design).unwrap_or_else(|e| panic!("{name}: compile failed: {e:?}"));
    let program = lower_to_packed_program(&compiled, top)
        .unwrap_or_else(|e| panic!("{name}: lowering failed: {e:?}"));
    let encoded = InterpProgram::encode_design(&program)
        .unwrap_or_else(|e| panic!("{name}: encode failed: {e:?}"));
    let module = compiled.find_module(top).unwrap();

    let off = |s: Signal| program.signals[program.signal_index(s).unwrap()].layout.offset;
    let limbs = |w: u32| (((w + 31) / 32).max(1)) as usize;
    let mask = |w: u32| if w >= 128 { u128::MAX } else { (1u128 << w) - 1 };

    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for s in &module.signals {
        match s.kind {
            SignalKind::Input => inputs.push((s.name.clone(), s.handle, s.width)),
            SignalKind::Output => outputs.push((s.name.clone(), s.handle, s.width)),
            _ => {}
        }
    }
    assert!(!outputs.is_empty(), "{name}: design has no outputs to check");

    let mut gold = Simulator::new(&imported.design, top).unwrap();
    let mut interp = InterpRunner::new(encoded, 1);

    let mut lcg: u128 = 0x9e37_79b9_7f4a_7c15 ^ name.len() as u128;
    let mut rng = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        lcg >> 64
    };

    for cycle in 0..cycles {
        for (n, h, w) in &inputs {
            // clk is a tick convention for the interpreter; everything else
            // (including resets) is driven randomly — both engines see the same
            // value, so the comparison stays valid.
            let v = if n == "clk" { 1 } else { rng() & mask(*w) };
            gold.set(*h, v);
            interp.set_signal_wide(off(*h), limbs(*w), &[v]);
        }
        gold.tick();
        interp.tick();
        for (n, h, w) in &outputs {
            let g = gold.get(*h) & mask(*w);
            let i = interp.get_signal_wide(off(*h), limbs(*w))[0] & mask(*w);
            assert_eq!(g, i, "{name}: output `{n}` mismatch at cycle {cycle}");
        }
    }
}

#[test]
fn sv_combinational_alu() {
    validate(
        "alu",
        r#"
        module Alu(
          input  logic [7:0] a, b,
          input  logic [1:0] op,
          output logic [7:0] y,
          output logic       eq, ne, lt
        );
          assign y  = (op == 2'd0) ? (a + b)
                    : (op == 2'd1) ? (a - b)
                    : (op == 2'd2) ? (a & b)
                    :                (a ^ b);
          assign eq = (a == b);
          assign ne = (a != b);
          assign lt = (a < b);
        endmodule
        "#,
        "Alu",
        30,
    );
}

#[test]
fn sv_counter_sync_reset_enable() {
    validate(
        "counter",
        r#"
        module Counter(
          input  logic       clk, rst, en,
          input  logic [7:0] step,
          output logic [7:0] q
        );
          logic [7:0] q_r;
          assign q = q_r;
          always_ff @(posedge clk) begin
            if (rst)     q_r <= 8'd0;
            else if (en) q_r <= q_r + step;
          end
        endmodule
        "#,
        "Counter",
        40,
    );
}

#[test]
fn sv_fsm_case() {
    validate(
        "fsm",
        r#"
        module Fsm(
          input  logic       clk, rst, go, done,
          output logic [1:0] state,
          output logic       busy
        );
          logic [1:0] s;
          assign state = s;
          assign busy = (s != 2'd0);
          always_ff @(posedge clk) begin
            if (rst) s <= 2'd0;
            else begin
              case (s)
                2'd0: s <= go   ? 2'd1 : 2'd0;
                2'd1: s <= 2'd2;
                2'd2: s <= done ? 2'd0 : 2'd2;
                default: s <= 2'd0;
              endcase
            end
          end
        endmodule
        "#,
        "Fsm",
        40,
    );
}

#[test]
fn sv_async_active_low_reset() {
    validate(
        "async_rst",
        r#"
        module Counter(
          input  logic       clk, rst_n, en,
          output logic [3:0] out
        );
          logic [3:0] r;
          assign out = r;
          always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n)  r <= 4'd0;
            else if (en) r <= r + 4'd1;
          end
        endmodule
        "#,
        "Counter",
        40,
    );
}

#[test]
fn sv_slice_concat_wide() {
    validate(
        "shifter",
        r#"
        module Shifter(
          input  logic        clk,
          input  logic [31:0] din,
          input  logic        ld,
          output logic [31:0] dout,
          output logic [7:0]  hi
        );
          logic [31:0] r;
          assign dout = r;
          assign hi   = r[31:24];
          always_ff @(posedge clk) begin
            if (ld) r <= din;
            else    r <= {r[23:0], r[31:24]};   // rotate left by 8
          end
        endmodule
        "#,
        "Shifter",
        40,
    );
}

#[test]
fn sv_memory_regfile() {
    validate(
        "regfile",
        r#"
        module RegFile(
          input  logic       clk, we,
          input  logic [3:0] waddr, raddr,
          input  logic [7:0] wdata,
          output logic [7:0] rdata
        );
          logic [7:0] mem [0:15];
          assign rdata = mem[raddr];
          always_ff @(posedge clk) begin
            if (we) mem[waddr] <= wdata;
          end
        endmodule
        "#,
        "RegFile",
        50,
    );
}

#[test]
fn sv_signed_compare() {
    validate(
        "signed",
        r#"
        module SCmp(
          input  logic signed [7:0] a, b,
          output logic              slt,
          output logic signed [8:0] ssum
        );
          assign slt  = (a < b);
          assign ssum = a + b;
        endmodule
        "#,
        "SCmp",
        30,
    );
}

#[test]
fn sv_reduction_operators() {
    validate(
        "reduction",
        r#"
        module Red(
          input  logic [7:0] a,
          output logic       all_set, any_set, parity
        );
          assign all_set = &a;   // reduction-AND
          assign any_set = |a;   // reduction-OR
          assign parity  = ^a;   // reduction-XOR
        endmodule
        "#,
        "Red",
        30,
    );
}

#[test]
fn sv_shift_operators() {
    validate(
        "shifts",
        r#"
        module Shift(
          input  logic [7:0]        a,
          input  logic signed [7:0] sa,
          input  logic [2:0]        n,
          output logic [7:0]        shl_c, shr_c, shl_v, shr_v,
          output logic signed [7:0] ashr_c
        );
          assign shl_c  = a << 3;     // constant left
          assign shr_c  = a >> 2;     // constant right
          assign shl_v  = a << n;     // variable (barrel)
          assign shr_v  = a >> n;     // variable (barrel)
          assign ashr_c = sa >>> 2;   // arithmetic right
        endmodule
        "#,
        "Shift",
        40,
    );
}

#[test]
fn sv_parameters_and_unsized_literals() {
    validate(
        "params",
        r#"
        module Counter #(parameter W = 8) (
          input  logic         clk, en,
          input  logic [W-1:0] step,
          output logic [W-1:0] q
        );
          logic [W-1:0] q_r;
          assign q = q_r;
          always_ff @(posedge clk) begin
            if (en) q_r <= q_r + step + 1;   // unsized literal mixed with W-bit
          end
        endmodule
        module Top(
          input  logic        clk, en,
          input  logic [15:0] s16,
          input  logic [3:0]  s4,
          output logic [15:0] q16,
          output logic [3:0]  q4
        );
          Counter #(.W(16)) u16(.clk(clk), .en(en), .step(s16), .q(q16));
          Counter #(.W(4))  u4 (.clk(clk), .en(en), .step(s4),  .q(q4));
        endmodule
        "#,
        "Top",
        40,
    );
}

#[test]
fn sv_module_hierarchy() {
    validate(
        "hier",
        r#"
        module AddOne(input logic [7:0] a, output logic [7:0] y);
          assign y = a + 8'd1;
        endmodule
        module Top(
          input  logic [7:0] a, b,
          output logic [7:0] ya, yb, sum
        );
          AddOne ua(.a(a), .y(ya));
          AddOne ub(.a(b), .y(yb));
          assign sum = ya + yb;
        endmodule
        "#,
        "Top",
        30,
    );
}
