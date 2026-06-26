// Cross-simulator benchmark DUT: W channels of a depth-D 32-bit mul-add
// accumulator, XOR-reduced to one output (so no channel is dead-code-eliminated).
// Must stay semantically identical to the RRTL and PyRTL versions.
//
// The mul-add chain is a procedural always@(*) with blocking assignments (clear
// sequential ordering) rather than an unpacked wire array t[j+1]=f(t[j]) — the
// latter makes Verilator flag the array as circular (UNOPTFLAT) and fall back to
// slow iterative settling.
module dut #(
    parameter W = 16,
    parameter D = 8
) (
    input         clk,
    input  [31:0] din,
    output [31:0] out
);
  localparam [31:0] C = 32'h9e3779b9;
  wire [31:0] accv [0:W-1];
  genvar i;
  generate
    for (i = 0; i < W; i = i + 1) begin : ch
      reg [31:0] acc;
      reg [31:0] nxt;
      integer j;
      always @(*) begin
        nxt = acc;
        for (j = 0; j < D; j = j + 1) nxt = nxt * C + din;
      end
      always @(posedge clk) acc <= nxt + acc + i;  // per-channel salt
      assign accv[i] = acc;
    end
  endgenerate
  // XOR reduction (procedural, accv has no feedthrough → no UNOPTFLAT).
  reg [31:0] x;
  integer k;
  always @(*) begin
    x = 32'b0;
    for (k = 0; k < W; k = k + 1) x = x ^ accv[k];
  end
  assign out = x;
endmodule
