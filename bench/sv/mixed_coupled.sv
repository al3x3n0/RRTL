// A COUPLED mixed design: the linear CRC block and the non-linear accumulator
// read each other's registers across the partition cut, so a hybrid simulator
// must exchange boundary register values every cycle.
//   crc  <= linearCRC(crc XOR acc, din)   -- linear, but reads `acc` (non-linear reg)
//   acc  <= acc + a*b + crc               -- non-linear, reads `crc` (linear reg)
// crc stays GF(2)-linear in {crc, acc, din} (XOR/shift/AND-const only); the a*b
// keeps acc non-linear. Mutual register coupling = boundary exchange required.
module mixed_coupled (
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
            // linear CRC seeded with crc XOR acc (still GF(2)-linear in crc,acc,din)
            c = crc ^ acc;
            for (i = 0; i < 8; i = i + 1) begin
                c = (c << 1) ^ ({32{c[31] ^ din[i]}} & 32'h04C11DB7);
            end
            crc <= c;
            // non-linear MAC that also reads the linear register crc
            acc <= acc + a * b + crc;
            count <= (a > b) ? count + 8'h1 : count;
        end
    end
endmodule
