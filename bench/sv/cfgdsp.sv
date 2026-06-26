// A configuration-gated DSP: a 2-bit `mode` register, latched once from `cfg_in`
// when `cfg_we` pulses and then held for the rest of the run, selects among four
// multiply-heavy 64-bit datapaths feeding the accumulator. This is the canonical
// target for *dynamic* specialization: `mode` is a runtime quasi-constant that
// static analysis cannot prove constant, but a profiler observing a long run can
// — and once frozen, const-folding collapses the select and DCEs the three
// unused (multiply-heavy) arms, the bulk of the per-cycle cost. `acc` is
// genuinely dynamic and must NOT be frozen.
module cfgdsp (
    input             clk,
    input             rst,
    input             cfg_we,
    input      [1:0]  cfg_in,
    input      [63:0] a,
    input      [63:0] b,
    output reg [63:0] acc
);
    reg [1:0] mode;
    always @(posedge clk) begin
        if (rst) begin
            mode <= 2'd0;
            acc  <= 64'd0;
        end else begin
            if (cfg_we) mode <= cfg_in;
            case (mode)
                2'd0:    acc <= acc + a*b*a + b*a*b + a*a*b + b*b*a + a*b;
                2'd1:    acc <= acc + a*a*a + b*b*b + a*b*a + b*a*b + a*a;
                2'd2:    acc <= acc + ((a*b*a) ^ (a*a*b) ^ (b*b*a) ^ (a*b*b));
                default: acc <= acc + a*b*a*b + a*a*b + b*b*a + a*b + b;
            endcase
        end
    end
endmodule
