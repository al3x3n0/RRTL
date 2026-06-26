module aimac #(parameter W=64) (input clk, input [8*W-1:0] a, input [8*W-1:0] b, output [31:0] o);
  reg [31:0] acc [0:W-1];
  reg [31:0] xacc;
  integer i;
  always @(posedge clk) for (i=0;i<W;i=i+1) acc[i] <= acc[i] + a[8*i +: 8] * b[8*i +: 8];
  always @(*) begin xacc=0; for(i=0;i<W;i=i+1) xacc=xacc^acc[i]; end
  assign o = xacc;
endmodule
