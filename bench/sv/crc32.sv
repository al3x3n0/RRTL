// A byte-parallel CRC-32 accumulator — a canonical GF(2)-linear block: the
// next-state of `crc` is a fixed binary matrix times {crc, din} over GF(2)
// (only XOR, AND-with-constant, and bit shifts). Used to exercise the
// linearity analysis (the tensor-core specialization prize).
module crc32 (
    input            clk,
    input            rst,
    input      [7:0] din,
    output reg [31:0] crc
);
    integer i;
    reg [31:0] c;
    always @(posedge clk) begin
        if (rst) begin
            crc <= 32'hFFFFFFFF;
        end else begin
            c = crc;
            for (i = 0; i < 8; i = i + 1) begin
                c = (c << 1) ^ ({32{c[31] ^ din[i]}} & 32'h04C11DB7);
            end
            crc <= c;
        end
    end
endmodule
