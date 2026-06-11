module CombCaseDecoder(
  input logic [1:0] op,
  output logic [3:0] y
);
  always_comb begin
    y = 4'd0;
    unique case (op)
      2'd0: y = 4'd1;
      2'd1, 2'd2: y = 4'd7;
      default: y = 4'd15;
    endcase
  end
endmodule
