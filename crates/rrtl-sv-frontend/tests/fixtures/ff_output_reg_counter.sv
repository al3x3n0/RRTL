module FfOutputRegCounter(
  input logic clk,
  input logic en,
  output reg [7:0] q
);
  always_ff @(posedge clk) begin
    if (en) begin
      q <= q + 8'd1;
    end
  end
endmodule
