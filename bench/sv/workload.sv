// A per-instance variable-length workload: load a `seed` (the run length) into a
// counter, then do real per-cycle work (a multiply-accumulate) until the counter
// reaches 0. Different instances are loaded with different seeds, so they HALT at
// data-dependent cycles — genuine divergence (not an artificially assigned cost),
// the natural shape of a fuzzing / regression / design-space batch.
module workload (
    input             clk,
    input             load,
    input      [31:0] seed,
    input      [31:0] din,
    output reg [31:0] acc
);
    reg [31:0] counter;
    always @(posedge clk) begin
        if (load) begin
            counter <= seed;
            acc     <= 32'd0;
        end else if (counter != 32'd0) begin
            counter <= counter - 32'd1;
            acc     <= acc * din + counter; // real per-cycle work; frozen once counter==0
        end
    end
endmodule
