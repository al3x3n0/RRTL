module FfIfElseCounter(
  input logic clk,
  input logic en,
  input logic load,
  input logic [3:0] din,
  output logic [3:0] q
);
  logic [3:0] q_r;
  assign q = q_r;

  always_ff @(posedge clk) begin
    if (load) begin
      q_r <= din;
    end else if (en) begin
      q_r <= q_r + 4'd1;
    end else begin
      q_r <= q_r;
    end
  end
endmodule
