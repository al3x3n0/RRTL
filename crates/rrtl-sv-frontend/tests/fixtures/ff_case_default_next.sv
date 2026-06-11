module FfCaseDefaultNext(
  input logic clk,
  input logic [1:0] op,
  input logic [7:0] a,
  input logic [7:0] b,
  output logic [7:0] q
);
  logic [7:0] q_r;
  assign q = q_r;

  always_ff @(posedge clk) begin
    q_r <= a;
    priority case (op)
      2'd1: q_r <= b;
      2'd2: q_r <= q_r + 8'd1;
    endcase
  end
endmodule
