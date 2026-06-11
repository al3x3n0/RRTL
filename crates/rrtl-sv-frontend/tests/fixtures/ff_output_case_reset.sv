module FfOutputCaseReset(
  input logic clk,
  input logic rst,
  input logic [1:0] op,
  input logic [3:0] d,
  output [3:0] q
);
  always_ff @(posedge clk) begin
    if (rst) begin
      q <= 4'd0;
    end else begin
      case (op)
        2'd1: q <= d;
        2'd2: q <= q + 4'd1;
        default: q <= q;
      endcase
    end
  end
endmodule
