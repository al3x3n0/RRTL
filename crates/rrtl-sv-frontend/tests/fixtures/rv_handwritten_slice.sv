module RvHandwrittenSlice(
  input logic in_valid,
  output logic in_ready,
  input logic [7:0] in_bits,
  output logic out_valid,
  input logic out_ready,
  output logic [7:0] out_bits,
  output logic stall
);
  always_comb begin
    in_ready = out_ready;
    out_valid = in_valid;
    out_bits = in_bits;
    stall = 1'd0;
    if (in_valid && !out_ready) begin
      stall = 1'd1;
    end
  end
endmodule
