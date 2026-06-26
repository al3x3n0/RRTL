// Self-contained picorv32 SoC: the core + an internal SYNCHRONOUS RAM + handshake,
// so the whole instance runs AUTONOMOUSLY (no host bus). This is what lets N
// independent instances run in RRTL's SIMD/GPU lanes — each lane carries its own
// RAM (its own program/stimulus), all in lockstep. A store to MAGIC_ADDR (0x100)
// latches into `result`/`done`, so the per-lane outcome is read from a signal.
//
// The RAM is synchronous: `mem_ready` and `mem_rdata` are registered (valid the
// cycle after a request), which both matches a real single-cycle SRAM and breaks
// the combinational read→core→address loop that a comb read would create. picorv32
// tolerates variable memory latency (it waits for mem_ready). Compile with picorv32.v.
module picorv32_soc (
    input  wire        clk,
    input  wire        resetn,
    output wire        trap,
    output reg  [31:0] result,
    output reg         done
);
    wire        mem_valid;
    wire        mem_instr;
    wire [31:0] mem_addr;
    wire [31:0] mem_wdata;
    wire [ 3:0] mem_wstrb;

    // Registered memory response (synchronous RAM).
    reg         mem_ready;
    reg  [31:0] mem_rdata;
    reg  [31:0] mem [0:1023];          // 4 KiB RAM (word-addressed)

    wire [9:0]  widx  = mem_addr[11:2];
    wire [31:0] wmask = {{8{mem_wstrb[3]}}, {8{mem_wstrb[2]}}, {8{mem_wstrb[1]}}, {8{mem_wstrb[0]}}};

    // Unused core outputs (the frontend requires every instance port connected).
    wire        mem_la_read;
    wire        mem_la_write;
    wire [31:0] mem_la_addr;
    wire [31:0] mem_la_wdata;
    wire [ 3:0] mem_la_wstrb;
    wire        pcpi_valid;
    wire [31:0] pcpi_insn;
    wire [31:0] pcpi_rs1;
    wire [31:0] pcpi_rs2;
    wire [31:0] eoi;
    wire        trace_valid;
    wire [35:0] trace_data;

    picorv32 #(
        .ENABLE_COUNTERS(0),
        .ENABLE_COUNTERS64(0),
        .ENABLE_IRQ(0),
        .ENABLE_IRQ_TIMER(0),
        .ENABLE_IRQ_QREGS(0),
        .CATCH_MISALIGN(0),
        .CATCH_ILLINSN(0),
        .BARREL_SHIFTER(0),
        .ENABLE_MUL(0),
        .ENABLE_DIV(0),
        .COMPRESSED_ISA(0)
    ) cpu (
        .clk(clk),
        .resetn(resetn),
        .trap(trap),
        .mem_valid(mem_valid),
        .mem_instr(mem_instr),
        .mem_ready(mem_ready),
        .mem_addr(mem_addr),
        .mem_wdata(mem_wdata),
        .mem_wstrb(mem_wstrb),
        .mem_rdata(mem_rdata),
        .mem_la_read(mem_la_read),
        .mem_la_write(mem_la_write),
        .mem_la_addr(mem_la_addr),
        .mem_la_wdata(mem_la_wdata),
        .mem_la_wstrb(mem_la_wstrb),
        .pcpi_valid(pcpi_valid),
        .pcpi_insn(pcpi_insn),
        .pcpi_rs1(pcpi_rs1),
        .pcpi_rs2(pcpi_rs2),
        .pcpi_wr(1'b0),
        .pcpi_rd(32'b0),
        .pcpi_wait(1'b0),
        .pcpi_ready(1'b0),
        .irq(32'b0),
        .eoi(eoi),
        .trace_valid(trace_valid),
        .trace_data(trace_data)
    );

    always @(posedge clk) begin
        if (~resetn) begin
            mem_ready <= 1'b0;
            mem_rdata <= 32'b0;
            result    <= 32'b0;
            done      <= 1'b0;
        end else begin
            mem_ready <= 1'b0;
            if (mem_valid & ~mem_ready) begin           // a pending request — answer it
                mem_ready <= 1'b1;
                mem_rdata <= mem[widx];                  // registered read
                if (mem_wstrb != 4'b0) begin
                    mem[widx] <= (mem[widx] & ~wmask) | (mem_wdata & wmask);
                    if (mem_addr == 32'h0000_0100) begin
                        result <= mem_wdata;
                        done   <= 1'b1;
                    end
                end
            end
        end
    end
endmodule
