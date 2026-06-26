//! SystemVerilog AST produced by the parser and consumed by lowering.
use rrtl_core::Design;
use rrtl_ir::Width;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SvSource {
    pub modules: Vec<SvModule>,
}

#[derive(Clone, Debug)]
pub struct SvImport {
    pub design: Design,
    pub top_name: String,
    pub modules: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SvModule {
    pub name: String,
    pub ports: Vec<String>,
    pub params: Vec<SvParam>,
    pub items: Vec<SvItem>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvItem {
    Param(SvParam),
    TypeDef(SvTypeDef),
    Decl(SvDecl),
    Genvar(Vec<String>),
    Generate(Vec<SvItem>),
    GenerateFor {
        var: String,
        init: SvExpr,
        cmp: SvForCmp,
        bound: SvExpr,
        step: SvForStep,
        label: Option<String>,
        items: Vec<SvItem>,
    },
    Assign {
        dst: SvLvalue,
        expr: SvExpr,
    },
    Initial(Vec<SvInitialAssign>),
    AlwaysComb(Vec<SvStmt>),
    AlwaysFf(SvAlwaysFf),
    Instance(SvInstance),
    Function(SvFunction),
}

/// A SystemVerilog `function … endfunction` — a pure combinational subroutine that
/// is inline-expanded at every call site (functions cannot have side effects).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SvFunction {
    pub name: String,
    pub return_type: SvType,
    /// `(input [..] a, input [..] b)` — only inputs are meaningful for a function.
    pub inputs: Vec<(String, SvType)>,
    /// Local variable declarations in the function body.
    pub locals: Vec<SvDecl>,
    pub body: Vec<SvStmt>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SvParam {
    pub name: String,
    pub value: SvExpr,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SvTypeDef {
    pub name: String,
    pub ty: SvType,
    /// Enum variants `(name, optional explicit value)`. An omitted value
    /// auto-increments from the previous variant (or 0 for the first). Empty for
    /// a plain (non-enum) typedef.
    pub variants: Vec<(String, Option<SvExpr>)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SvDecl {
    pub direction: Option<SvDirection>,
    pub kind: SvDeclKind,
    pub ty: SvType,
    pub names: Vec<SvDeclarator>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvDirection {
    Input,
    Output,
    Inout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvDeclKind {
    Logic,
    Wire,
    Reg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SvType {
    pub width: Width,
    pub signed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SvDeclarator {
    pub name: String,
    /// Total memory depth: D for `[0:D-1]`, or D1*D2 for a 2-D unpacked array
    /// `[0:D1-1][0:D2-1]` (flattened row-major).
    pub memory_depth: Option<usize>,
    /// Inner dimension D2 of a 2-D unpacked array (the row-major flatten factor);
    /// `None` for a 1-D memory.
    pub memory_inner: Option<usize>,
    /// `wire/logic name = expr;` — an inline continuous assignment.
    pub init: Option<SvExpr>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvInitialAssign {
    /// `dst = expr;` — a power-on initial value.
    Assign { dst: SvLvalue, expr: SvExpr },
    /// `$readmemh("file", mem);` / `$readmemb(...)` — load a memory from a hex/bin
    /// file at lowering time.
    ReadMem { hex: bool, file: String, mem: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SvAlwaysFf {
    pub clock: String,
    pub async_reset: Option<SvResetEdge>,
    pub body: Vec<SvStmt>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SvResetEdge {
    pub signal: String,
    pub active_low: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvStmt {
    Assign {
        dst: SvLvalue,
        nonblocking: bool,
        expr: SvExpr,
    },
    /// `{a, b[3:0], c} = expr;` — the RHS is split MSB-first across the parts by
    /// their widths (desugared to per-part assigns at lowering time).
    ConcatAssign {
        parts: Vec<SvLvalue>,
        nonblocking: bool,
        expr: SvExpr,
    },
    If {
        cond: SvExpr,
        then_stmts: Vec<SvStmt>,
        else_stmts: Vec<SvStmt>,
    },
    For {
        var: String,
        init: SvExpr,
        cmp: SvForCmp,
        bound: SvExpr,
        step: SvForStep,
        body: Vec<SvStmt>,
    },
    Case {
        kind: SvCaseKind,
        expr: SvExpr,
        items: Vec<SvCaseItem>,
    },
    Assert {
        name: String,
        cond: SvExpr,
        message: Option<String>,
    },
    Cover {
        name: String,
        cond: SvExpr,
    },
    /// A statement with no datapath effect (a task / system-task call such as
    /// `empty_statement;`, `$display(...);`, `$finish;`).
    Nop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvForCmp {
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvForStep {
    Inc,
    Dec,
    Add(SvExpr),
    Sub(SvExpr),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SvCaseItem {
    pub labels: Vec<SvCaseLabel>,
    pub stmts: Vec<SvStmt>,
    pub is_default: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvCaseKind {
    Normal,
    CaseZ,
    CaseX,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvCaseLabel {
    Expr(SvExpr),
    Wildcard {
        value: u128,
        mask: u128,
        width: Width,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SvInstance {
    pub module: String,
    pub name: String,
    pub params: Vec<SvParamOverride>,
    pub connections: Vec<SvConnection>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvParamOverride {
    Named { name: String, value: SvExpr },
    Positional { value: SvExpr },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SvConnection {
    pub port: String,
    pub expr: SvExpr,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvLvalue {
    Signal(String),
    Bit {
        name: String,
        index: SvExpr,
    },
    Slice {
        name: String,
        msb: SvExpr,
        lsb: SvExpr,
    },
    Memory {
        name: String,
        addr: SvExpr,
    },
    /// `mem[i][j]` write to a 2-D unpacked memory (flattened row-major at lowering).
    Memory2D {
        name: String,
        outer: SvExpr,
        inner: SvExpr,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvExpr {
    Ident(String),
    Lit {
        value: u128,
        width: Width,
        signed: bool,
    },
    Unary {
        op: SvUnaryOp,
        expr: Box<SvExpr>,
    },
    Binary {
        op: SvBinaryOp,
        lhs: Box<SvExpr>,
        rhs: Box<SvExpr>,
    },
    Ternary {
        cond: Box<SvExpr>,
        then_expr: Box<SvExpr>,
        else_expr: Box<SvExpr>,
    },
    Concat(Vec<SvExpr>),
    Repeat {
        count: Width,
        expr: Box<SvExpr>,
    },
    Cast {
        signed: bool,
        expr: Box<SvExpr>,
    },
    Index {
        expr: Box<SvExpr>,
        index: Width,
    },
    Slice {
        expr: Box<SvExpr>,
        msb: Width,
        lsb: Width,
    },
    MemRead {
        name: String,
        addr: Box<SvExpr>,
    },
    Bracket {
        expr: Box<SvExpr>,
        index: Box<SvExpr>,
    },
    /// A user-function call `name(arg0, arg1, …)` — inline-expanded at lowering
    /// (SystemVerilog functions are pure combinational).
    Call {
        name: String,
        args: Vec<SvExpr>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvUnaryOp {
    Not,
    BitNot,
    Neg,
    /// Reduction-AND `&a`, reduction-OR `|a`, reduction-XOR `^a` (1-bit result).
    RedAnd,
    RedOr,
    RedXor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvBinaryOp {
    Add,
    Sub,
    Mul,
    /// Division, modulo, and exponentiation — supported only in constant
    /// expressions (the IR has no runtime divide).
    Div,
    Mod,
    Pow,
    And,
    Or,
    Xor,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    LogAnd,
    LogOr,
    /// Logical shifts `<<` `>>` and arithmetic right shift `>>>`.
    Shl,
    Shr,
    Ashr,
}
