module CombIfDefaults(
  input logic en,
  input logic sel,
  input logic [7:0] a,
  input logic [7:0] b,
  input logic [7:0] c,
  output logic [7:0] y,
  output logic [7:0] z
);
  always_comb begin
    y = a;
    z = 8'd0;
    if (en) begin
      y = b;
      if (sel) begin
        z = c;
      end
    end
  end
endmodule
