//! A small, dependency-free RV32I assembler for test harnesses.
//!
//! Hand-encoding RISC-V machine code is error-prone; this lets harnesses write
//! readable assembly with labels instead. Two passes: pass 1 assigns each
//! (possibly pseudo-expanded) instruction a byte address and records labels;
//! pass 2 encodes, resolving label references to PC-relative or absolute forms.
//!
//! ```
//! let words = rrtl_riscv_asm::assemble("
//!     addi x1, x0, 7
//!     addi x2, x0, 5
//!     add  x3, x1, x2
//!     sw   x3, 0x40(x0)
//! loop:
//!     j loop
//! ").unwrap();
//! ```
//!
//! Supported: the RV32I base integer set plus the common pseudo-instructions
//! (`nop`, `li`, `mv`, `not`, `neg`, `j`, `jr`, `ret`, `call` is NOT included,
//! `beqz`, `bnez`, `blez`, `bgez`). Registers accept `x0..x31` or ABI names.

use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsmError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for AsmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "asm error on line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for AsmError {}

/// Assemble `src` with a base load address of 0.
pub fn assemble(src: &str) -> Result<Vec<u32>, AsmError> {
    assemble_at(src, 0)
}

/// Assemble `src` as if loaded at byte address `base` (affects label math for
/// absolute references; PC-relative branches/jumps are unaffected).
pub fn assemble_at(src: &str, base: u32) -> Result<Vec<u32>, AsmError> {
    // ---- tokenize lines: strip comments, split labels from instructions ----
    struct Line {
        no: usize,
        op: String,
        args: Vec<String>,
    }
    let mut lines: Vec<Line> = Vec::new();
    let mut labels: HashMap<String, u32> = HashMap::new();
    let mut addr = base;

    for (i, raw) in src.lines().enumerate() {
        let no = i + 1;
        // strip comments (`#`, `//`, `;`)
        let mut text = raw;
        for marker in ["#", "//", ";"] {
            if let Some(p) = text.find(marker) {
                text = &text[..p];
            }
        }
        let mut text = text.trim();
        if text.is_empty() {
            continue;
        }
        // leading `label:` (possibly several)
        while let Some(colon) = text.find(':') {
            let (lbl, rest) = text.split_at(colon);
            let lbl = lbl.trim();
            if lbl.is_empty() || !is_ident(lbl) {
                break;
            }
            if labels.insert(lbl.to_string(), addr).is_some() {
                return Err(AsmError { line: no, message: format!("duplicate label `{lbl}`") });
            }
            text = rest[1..].trim();
            if text.is_empty() {
                break;
            }
        }
        if text.is_empty() {
            continue;
        }
        let (op, rest) = split_mnemonic(text);
        let args = split_args(rest);
        let size = insn_size(&op, &args).map_err(|m| AsmError { line: no, message: m })?;
        lines.push(Line { no, op: op.to_lowercase(), args });
        addr = addr.wrapping_add(4 * size);
    }

    // ---- pass 2: encode ----
    let mut words = Vec::new();
    let mut pc = base;
    for line in &lines {
        let ctx = Ctx { pc, labels: &labels, base };
        let encoded = encode(&line.op, &line.args, &ctx)
            .map_err(|m| AsmError { line: line.no, message: m })?;
        for w in encoded {
            words.push(w);
            pc = pc.wrapping_add(4);
        }
    }
    Ok(words)
}

struct Ctx<'a> {
    pc: u32,
    labels: &'a HashMap<String, u32>,
    base: u32,
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_' || c == '.')
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

fn split_mnemonic(text: &str) -> (String, &str) {
    match text.find(|c: char| c.is_whitespace()) {
        Some(p) => (text[..p].to_string(), text[p..].trim()),
        None => (text.to_string(), ""),
    }
}

fn split_args(rest: &str) -> Vec<String> {
    if rest.trim().is_empty() {
        return Vec::new();
    }
    rest.split(',').map(|a| a.trim().to_string()).collect()
}

/// Number of 4-byte words an instruction (or pseudo) occupies.
fn insn_size(op: &str, args: &[String]) -> Result<u32, String> {
    Ok(match op.to_lowercase().as_str() {
        "li" => {
            let imm = parse_int(args.get(1).ok_or("li needs an immediate")?)?;
            if fits_signed(imm, 12) { 1 } else { 2 }
        }
        _ => 1,
    })
}

fn reg(name: &str) -> Result<u32, String> {
    let n = name.trim();
    let idx = match n {
        "zero" => 0,
        "ra" => 1,
        "sp" => 2,
        "gp" => 3,
        "tp" => 4,
        "t0" => 5,
        "t1" => 6,
        "t2" => 7,
        "s0" | "fp" => 8,
        "s1" => 9,
        "a0" => 10,
        "a1" => 11,
        "a2" => 12,
        "a3" => 13,
        "a4" => 14,
        "a5" => 15,
        "a6" => 16,
        "a7" => 17,
        "s2" => 18,
        "s3" => 19,
        "s4" => 20,
        "s5" => 21,
        "s6" => 22,
        "s7" => 23,
        "s8" => 24,
        "s9" => 25,
        "s10" => 26,
        "s11" => 27,
        "t3" => 28,
        "t4" => 29,
        "t5" => 30,
        "t6" => 31,
        other => {
            let rest = other
                .strip_prefix('x')
                .ok_or_else(|| format!("invalid register `{other}`"))?;
            let v: u32 = rest.parse().map_err(|_| format!("invalid register `{other}`"))?;
            if v > 31 {
                return Err(format!("register out of range `{other}`"));
            }
            v
        }
    };
    Ok(idx)
}

fn parse_int(s: &str) -> Result<i64, String> {
    let s = s.trim();
    let (neg, body) = match s.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let val: i64 = if let Some(hex) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        i64::from_str_radix(&hex.replace('_', ""), 16).map_err(|_| format!("bad number `{s}`"))?
    } else if let Some(bin) = body.strip_prefix("0b").or_else(|| body.strip_prefix("0B")) {
        i64::from_str_radix(&bin.replace('_', ""), 2).map_err(|_| format!("bad number `{s}`"))?
    } else {
        body.replace('_', "").parse().map_err(|_| format!("bad number `{s}`"))?
    };
    Ok(if neg { -val } else { val })
}

fn fits_signed(v: i64, bits: u32) -> bool {
    let lo = -(1i64 << (bits - 1));
    let hi = (1i64 << (bits - 1)) - 1;
    v >= lo && v <= hi
}

/// Resolve an immediate-or-label operand to a value. Labels resolve to their
/// absolute address; `relative` requests `label - pc` instead (for branches/jal).
fn resolve(arg: &str, ctx: &Ctx, relative: bool) -> Result<i64, String> {
    if let Some(&target) = ctx.labels.get(arg) {
        Ok(if relative {
            target as i64 - ctx.pc as i64
        } else {
            target as i64
        })
    } else if is_ident(arg) {
        Err(format!("unknown label `{arg}`"))
    } else {
        let v = parse_int(arg)?;
        Ok(if relative { v } else { v.wrapping_add(ctx.base as i64) })
    }
}

/// Parse `imm(rs1)` (load/store/jalr addressing). Returns (imm, rs1).
fn parse_offset(arg: &str) -> Result<(i64, u32), String> {
    let open = arg.find('(').ok_or_else(|| format!("expected imm(reg), got `{arg}`"))?;
    let close = arg.rfind(')').ok_or_else(|| format!("expected imm(reg), got `{arg}`"))?;
    let imm_s = arg[..open].trim();
    let reg_s = arg[open + 1..close].trim();
    let imm = if imm_s.is_empty() { 0 } else { parse_int(imm_s)? };
    Ok((imm, reg(reg_s)?))
}

fn r_type(funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}
fn i_type(imm: i64, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    ((imm as u32 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}
fn s_type(imm: i64, rs2: u32, rs1: u32, funct3: u32, opcode: u32) -> u32 {
    let imm = imm as u32;
    ((imm >> 5 & 0x7F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | ((imm & 0x1F) << 7)
        | opcode
}
fn b_type(imm: i64, rs2: u32, rs1: u32, funct3: u32, opcode: u32) -> u32 {
    let imm = imm as u32;
    ((imm >> 12 & 1) << 31)
        | ((imm >> 5 & 0x3F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | ((imm >> 1 & 0xF) << 8)
        | ((imm >> 11 & 1) << 7)
        | opcode
}
fn u_type(imm: i64, rd: u32, opcode: u32) -> u32 {
    ((imm as u32) & 0xFFFFF000) | (rd << 7) | opcode
}
fn j_type(imm: i64, rd: u32, opcode: u32) -> u32 {
    let imm = imm as u32;
    ((imm >> 20 & 1) << 31)
        | ((imm >> 1 & 0x3FF) << 21)
        | ((imm >> 11 & 1) << 20)
        | ((imm >> 12 & 0xFF) << 12)
        | (rd << 7)
        | opcode
}

fn encode(op: &str, args: &[String], ctx: &Ctx) -> Result<Vec<u32>, String> {
    let need = |n: usize| -> Result<(), String> {
        if args.len() == n {
            Ok(())
        } else {
            Err(format!("`{op}` expects {n} operands, got {}", args.len()))
        }
    };
    let a = |i: usize| args[i].as_str();

    // R-type: op rd, rs1, rs2
    let r = |f7: u32, f3: u32| -> Result<Vec<u32>, String> {
        need(3)?;
        Ok(vec![r_type(f7, reg(a(2))?, reg(a(1))?, f3, reg(a(0))?, 0x33)])
    };
    // I-type arithmetic: op rd, rs1, imm
    let i_arith = |f3: u32| -> Result<Vec<u32>, String> {
        need(3)?;
        let imm = resolve(a(2), ctx, false)?;
        if !fits_signed(imm, 12) {
            return Err(format!("immediate {imm} out of range for `{op}` (12-bit signed)"));
        }
        Ok(vec![i_type(imm, reg(a(1))?, f3, reg(a(0))?, 0x13)])
    };
    // I-type shift: op rd, rs1, shamt
    let i_shift = |f3: u32, f7: u32| -> Result<Vec<u32>, String> {
        need(3)?;
        let sh = parse_int(a(2))?;
        if !(0..32).contains(&sh) {
            return Err(format!("shift amount {sh} out of range 0..31"));
        }
        Ok(vec![i_type(((f7 as i64) << 5) | sh, reg(a(1))?, f3, reg(a(0))?, 0x13)])
    };
    // load: op rd, imm(rs1)
    let load = |f3: u32| -> Result<Vec<u32>, String> {
        need(2)?;
        let (imm, rs1) = parse_offset(a(1))?;
        Ok(vec![i_type(imm, rs1, f3, reg(a(0))?, 0x03)])
    };
    // store: op rs2, imm(rs1)
    let store = |f3: u32| -> Result<Vec<u32>, String> {
        need(2)?;
        let (imm, rs1) = parse_offset(a(1))?;
        Ok(vec![s_type(imm, reg(a(0))?, rs1, f3, 0x23)])
    };
    // branch: op rs1, rs2, label
    let branch = |f3: u32| -> Result<Vec<u32>, String> {
        need(3)?;
        let off = resolve(a(2), ctx, true)?;
        if !fits_signed(off, 13) || off % 2 != 0 {
            return Err(format!("branch target offset {off} out of range/misaligned"));
        }
        Ok(vec![b_type(off, reg(a(1))?, reg(a(0))?, f3, 0x63)])
    };

    Ok(match op {
        // R-type
        "add" => r(0x00, 0x0)?,
        "sub" => r(0x20, 0x0)?,
        "sll" => r(0x00, 0x1)?,
        "slt" => r(0x00, 0x2)?,
        "sltu" => r(0x00, 0x3)?,
        "xor" => r(0x00, 0x4)?,
        "srl" => r(0x00, 0x5)?,
        "sra" => r(0x20, 0x5)?,
        "or" => r(0x00, 0x6)?,
        "and" => r(0x00, 0x7)?,
        // I-type arithmetic
        "addi" => i_arith(0x0)?,
        "slti" => i_arith(0x2)?,
        "sltiu" => i_arith(0x3)?,
        "xori" => i_arith(0x4)?,
        "ori" => i_arith(0x6)?,
        "andi" => i_arith(0x7)?,
        // I-type shift
        "slli" => i_shift(0x1, 0x00)?,
        "srli" => i_shift(0x5, 0x00)?,
        "srai" => i_shift(0x5, 0x20)?,
        // loads / stores
        "lb" => load(0x0)?,
        "lh" => load(0x1)?,
        "lw" => load(0x2)?,
        "lbu" => load(0x4)?,
        "lhu" => load(0x5)?,
        "sb" => store(0x0)?,
        "sh" => store(0x1)?,
        "sw" => store(0x2)?,
        // branches
        "beq" => branch(0x0)?,
        "bne" => branch(0x1)?,
        "blt" => branch(0x4)?,
        "bge" => branch(0x5)?,
        "bltu" => branch(0x6)?,
        "bgeu" => branch(0x7)?,
        // U-type
        "lui" => {
            need(2)?;
            vec![u_type(resolve(a(1), ctx, false)? << 12, reg(a(0))?, 0x37)]
        }
        "auipc" => {
            need(2)?;
            vec![u_type(resolve(a(1), ctx, false)? << 12, reg(a(0))?, 0x17)]
        }
        // jumps
        "jal" => {
            // `jal rd, target` or `jal target` (rd = ra)
            let (rd, tgt) = if args.len() == 1 { (1, a(0)) } else { need(2).map(|_| ())?; (reg(a(0))?, a(1)) };
            let off = resolve(tgt, ctx, true)?;
            if !fits_signed(off, 21) || off % 2 != 0 {
                return Err(format!("jal target offset {off} out of range/misaligned"));
            }
            vec![j_type(off, rd, 0x6F)]
        }
        "jalr" => {
            // `jalr rd, imm(rs1)` or `jalr rd, rs1, imm` or `jalr rs1`
            match args.len() {
                1 => vec![i_type(0, reg(a(0))?, 0x0, 1, 0x67)],
                2 => {
                    let (imm, rs1) = parse_offset(a(1))?;
                    vec![i_type(imm, rs1, 0x0, reg(a(0))?, 0x67)]
                }
                3 => vec![i_type(parse_int(a(2))?, reg(a(1))?, 0x0, reg(a(0))?, 0x67)],
                n => return Err(format!("`jalr` expects 1..3 operands, got {n}")),
            }
        }
        // system
        "ecall" => { need(0)?; vec![0x0000_0073] }
        "ebreak" => { need(0)?; vec![0x0010_0073] }
        "fence" => { vec![0x0FF0_000F] }
        // ---- pseudo-instructions ----
        "nop" => { need(0)?; vec![i_type(0, 0, 0x0, 0, 0x13)] }
        "mv" => { need(2)?; vec![i_type(0, reg(a(1))?, 0x0, reg(a(0))?, 0x13)] }
        "not" => { need(2)?; vec![i_type(-1, reg(a(1))?, 0x4, reg(a(0))?, 0x13)] }
        "neg" => { need(2)?; vec![r_type(0x20, reg(a(1))?, 0, 0x0, reg(a(0))?, 0x33)] }
        "seqz" => { need(2)?; vec![i_type(1, reg(a(1))?, 0x3, reg(a(0))?, 0x13)] }
        "snez" => { need(2)?; vec![r_type(0x00, reg(a(1))?, 0, 0x3, reg(a(0))?, 0x33)] }
        "j" => {
            need(1)?;
            let off = resolve(a(0), ctx, true)?;
            vec![j_type(off, 0, 0x6F)]
        }
        "jr" => { need(1)?; vec![i_type(0, reg(a(0))?, 0x0, 0, 0x67)] }
        "ret" => { need(0)?; vec![i_type(0, 1, 0x0, 0, 0x67)] }
        "beqz" => { need(2)?; let o = resolve(a(1), ctx, true)?; vec![b_type(o, 0, reg(a(0))?, 0x0, 0x63)] }
        "bnez" => { need(2)?; let o = resolve(a(1), ctx, true)?; vec![b_type(o, 0, reg(a(0))?, 0x1, 0x63)] }
        "blez" => { need(2)?; let o = resolve(a(1), ctx, true)?; vec![b_type(o, reg(a(0))?, 0, 0x5, 0x63)] }
        "bgez" => { need(2)?; let o = resolve(a(1), ctx, true)?; vec![b_type(o, 0, reg(a(0))?, 0x5, 0x63)] }
        "li" => {
            need(2)?;
            let rd = reg(a(0))?;
            let imm = parse_int(a(1))?;
            if fits_signed(imm, 12) {
                vec![i_type(imm, 0, 0x0, rd, 0x13)]
            } else {
                // lui rd, hi ; addi rd, rd, lo  (lo sign-extended -> hi adjusted)
                let lo = ((imm & 0xFFF) << 52 >> 52) as i64; // sign-extend 12 bits
                let hi = (imm - lo) >> 12;
                vec![
                    u_type(hi << 12, rd, 0x37),
                    i_type(lo, rd, 0x0, rd, 0x13),
                ]
            }
        }
        other => return Err(format!("unknown instruction `{other}`")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_basic_itype_and_rtype() {
        let w = assemble("addi x1, x0, 7\nadd x3, x1, x2").unwrap();
        assert_eq!(w, vec![0x0070_0093, 0x0020_81B3]);
    }

    #[test]
    fn encodes_store_with_offset() {
        // sw x1, 0x44(x0)
        assert_eq!(assemble("sw x1, 0x44(x0)").unwrap(), vec![0x0410_2223]);
        // sw x3, 0x40(x0)
        assert_eq!(assemble("sw x3, 0x40(x0)").unwrap(), vec![0x0430_2023]);
    }

    #[test]
    fn slli_and_shifts() {
        assert_eq!(assemble("slli x3, x3, 2").unwrap(), vec![0x0021_9193]);
    }

    #[test]
    fn backward_branch_label() {
        // mirrors the picorv32 sum-loop: blt back by two instructions
        let w = assemble(
            "loop:\n add x1, x1, x2\n addi x2, x2, 1\n blt x2, x3, loop",
        )
        .unwrap();
        assert_eq!(w[2], 0xFE31_4CE3);
    }

    #[test]
    fn forward_branch_and_j() {
        let w = assemble("beq x1, x2, done\n nop\ndone:\n j done").unwrap();
        // beq offset = +8 (two instructions ahead)
        assert_eq!(w[0], b_type(8, 2, 1, 0x0, 0x63));
        // j done -> offset 0
        assert_eq!(w[2], j_type(0, 0, 0x6F));
    }

    #[test]
    fn abi_register_names() {
        // addi sp, sp, -16  == addi x2, x2, -16
        assert_eq!(
            assemble("addi sp, sp, -16").unwrap(),
            assemble("addi x2, x2, -16").unwrap()
        );
    }

    #[test]
    fn li_small_and_large() {
        assert_eq!(assemble("li a0, 5").unwrap(), vec![i_type(5, 0, 0, 10, 0x13)]);
        let big = assemble("li a0, 0x12345").unwrap();
        assert_eq!(big.len(), 2); // lui + addi
    }

    #[test]
    fn pseudo_nop_mv_ret() {
        assert_eq!(assemble("nop").unwrap(), vec![0x0000_0013]);
        assert_eq!(assemble("mv a0, a1").unwrap(), vec![i_type(0, 11, 0, 10, 0x13)]);
        assert_eq!(assemble("ret").unwrap(), vec![i_type(0, 1, 0, 0, 0x67)]);
    }

    #[test]
    fn errors_carry_line_numbers() {
        let e = assemble("addi x1, x0, 7\n bogus x1").unwrap_err();
        assert_eq!(e.line, 2);
    }
}
