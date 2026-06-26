// A deliberately MIXED design: a GF(2)-linear CRC block alongside non-linear
// logic (a multiply-accumulate and a compare-gated counter). Exercises cone
// partitioning — the linear `crc` register cone is extractable as a binary
// matrix (tensor-core/GEMM offloadable); `acc` and `count` are not and stay on
// the general lane/JIT path.
module mixed (
    input             clk,
    input             rst,
    input      [7:0]  din,
    input      [15:0] a,
    input      [15:0] b,
    output reg [31:0] crc,
    output reg [31:0] acc,
    output reg [7:0]  count
);
    integer i;
    reg [31:0] c;
    always @(posedge clk) begin
        if (rst) begin
            crc   <= 32'hFFFFFFFF;
            acc   <= 32'h0;
            count <= 8'h0;
        end else begin
            // linear: byte-parallel CRC-32
            c = crc;
            for (i = 0; i < 8; i = i + 1) begin
                c = (c << 1) ^ ({32{c[31] ^ din[i]}} & 32'h04C11DB7);
            end
            crc <= c;
            // non-linear: multiply-accumulate
            acc <= acc + a * b;
            // non-linear: compare-gated counter
            count <= (a > b) ? count + 8'h1 : count;
        end
    end
endmodule
