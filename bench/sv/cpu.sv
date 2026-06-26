// A RISC-V (RV32I) style execute unit: instruction decode + ALU + a 32x32
// register file + branch/PC logic. The instruction word is fed in each cycle
// (so we exercise the whole datapath without an instruction ROM); the register
// file and PC are internal state. Realistic control+datapath RTL — the same
// source is run through RRTL's SV frontend→JIT and through Verilator, and their
// outputs are cross-checked bit-for-bit.
//
// Unsigned ALU variant (the frontend's signed↔unsigned assignment coercion is a
// known gap), but otherwise a faithful decode/ALU/regfile/branch datapath.
module cpu (
    input  logic        clk,
    input  logic        rst,
    input  logic [31:0] instr,
    output logic [31:0] pc,
    output logic [31:0] x10
);
  // Register file (x0 is hardwired zero: never written, starts 0).
  logic [31:0] xr [0:31];

  // Instruction fields (declared separately so widths are unambiguous).
  logic [6:0] opcode;
  logic [4:0] rd;
  logic [4:0] rs1;
  logic [4:0] rs2;
  logic [2:0] funct3;
  logic [6:0] funct7;
  assign opcode = instr[6:0];
  assign rd     = instr[11:7];
  assign funct3 = instr[14:12];
  assign rs1    = instr[19:15];
  assign rs2    = instr[24:20];
  assign funct7 = instr[31:25];

  // Immediates (sign-extended).
  logic [31:0] imm_i;
  logic [31:0] imm_b;
  assign imm_i = {{20{instr[31]}}, instr[31:20]};
  assign imm_b = {{19{instr[31]}}, instr[31], instr[7], instr[30:25], instr[11:8], 1'b0};

  localparam logic [6:0] OP     = 7'b0110011; // R-type
  localparam logic [6:0] OP_IMM = 7'b0010011; // I-type ALU
  localparam logic [6:0] BRANCH = 7'b1100011;

  // Combinational register-file read ports.
  logic [31:0] a;
  logic [31:0] b;
  logic [31:0] opnd;
  assign a    = xr[rs1];
  assign b    = xr[rs2];
  assign opnd = (opcode == OP_IMM) ? imm_i : b;

  // ALU.
  logic [4:0]  shamt;
  logic [31:0] alu;
  assign shamt = opnd[4:0];
  always_comb begin
    case (funct3)
      3'b000:  alu = (opcode == OP && funct7[5]) ? (a - opnd) : (a + opnd);
      3'b001:  alu = a << shamt;
      3'b010:  alu = (a < opnd) ? 32'd1 : 32'd0;
      3'b011:  alu = (a < opnd) ? 32'd1 : 32'd0;
      3'b100:  alu = a ^ opnd;
      3'b101:  alu = a >> shamt;
      3'b110:  alu = a | opnd;
      default: alu = a & opnd;
    endcase
  end

  // Branch decision.
  logic taken;
  always_comb begin
    case (funct3)
      3'b000:  taken = (a == b);  // BEQ
      3'b001:  taken = (a != b);  // BNE
      3'b110:  taken = (a < b);   // BLTU
      default: taken = 1'b0;
    endcase
  end

  logic is_alu;
  logic do_wb;
  assign is_alu = (opcode == OP) || (opcode == OP_IMM);
  assign do_wb  = is_alu && (rd != 5'd0);

  // Sequential state: PC and register-file writeback.
  always_ff @(posedge clk) begin
    if (rst) begin
      pc <= 32'd0;
    end else if (opcode == BRANCH && taken) begin
      pc <= pc + imm_b;
    end else begin
      pc <= pc + 32'd4;
    end
    if (do_wb) xr[rd] <= alu;
  end

  assign x10 = xr[10];
endmodule
