module FfCaseFsm(
  input logic clk,
  input logic rst,
  input logic start,
  input logic done,
  output logic [1:0] state
);
  logic [1:0] state_r;
  assign state = state_r;

  always_ff @(posedge clk) begin
    if (rst) begin
      state_r <= 2'd0;
    end else begin
      case (state_r)
        2'd0: begin
          if (start) begin
            state_r <= 2'd1;
          end else begin
            state_r <= 2'd0;
          end
        end
        2'd1: begin
          if (done) begin
            state_r <= 2'd2;
          end else begin
            state_r <= 2'd1;
          end
        end
        default: state_r <= 2'd0;
      endcase
    end
  end
endmodule
