//! SystemVerilog lexer + recursive-descent parser (tokens → SvSource AST).
use std::collections::HashMap;

use rrtl_ir::{ErrorReport, Width};

use crate::ast::*;
use crate::err;
use crate::lower::{const_eval, iface_array_ref, resize_u128_to_width};

/// Separator joining an interface-instance name with a member name when an
/// interface is flattened into individual signals (`bus` + `req_valid` →
/// `bus.req_valid`). A `.` keeps the SV hierarchical feel; signal names are
/// opaque strings downstream so the dot is preserved verbatim.
const IFACE_SEP: &str = ".";

fn mangle(inst: &str, member: &str) -> String {
    format!("{inst}{IFACE_SEP}{member}")
}

/// A parsed `interface … endinterface`. Interfaces are fully desugared into flat
/// signals at parse time, so they never reach lowering.
#[derive(Clone, Debug)]
struct IfaceDef {
    /// Interface parameters (name + default value), which scope member widths.
    /// The first `overridable` entries are the `#()` params; any after are body
    /// localparams (computed, not overridable).
    params: Vec<SvParam>,
    /// Count of leading `#()` parameters (overridable per instance).
    overridable: usize,
    /// Member signals in declaration order.
    members: Vec<IfaceMember>,
    /// `modport name → (member → direction)`.
    modports: HashMap<String, HashMap<String, SvDirection>>,
}

/// One interface member. The width is kept as an expression (not a resolved
/// `Width`) so it can be re-evaluated against per-instance parameter overrides.
#[derive(Clone, Debug)]
struct IfaceMember {
    name: String,
    /// Expression evaluating to the member's bit width (e.g. `DATA_WIDTH`).
    width: SvExpr,
    signed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Token {
    Ident(String),
    Num(String),
    Str(String),
    Sym(String),
    Eof,
}

struct Lexer<'a> {
    source: &'a str,
    chars: Vec<char>,
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            chars: source.chars().collect(),
            pos: 0,
        }
    }

    fn tokenize(mut self) -> Result<Vec<Token>, ErrorReport> {
        let mut tokens = Vec::new();
        loop {
            self.skip_ws_and_comments()?;
            let Some(ch) = self.peek_char() else {
                break;
            };
            if is_ident_start(ch) {
                let start = self.pos;
                self.pos += 1;
                while self.peek_char().is_some_and(is_ident_continue) {
                    self.pos += 1;
                }
                tokens.push(Token::Ident(self.slice(start)));
            } else if ch.is_ascii_digit() {
                let start = self.pos;
                self.pos += 1;
                while self
                    .peek_char()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || c == '\'' || c == '_' || c == '?')
                {
                    self.pos += 1;
                }
                if self.slice(start).contains('\'') && self.peek_char() == Some('-') {
                    self.pos += 1;
                    while self
                        .peek_char()
                        .is_some_and(|c| c.is_ascii_digit() || c == '_')
                    {
                        self.pos += 1;
                    }
                }
                // Verilog allows whitespace between the base and value: `32'h 00ff`.
                let s = self.slice(start);
                if let Some(q) = s.rfind('\'') {
                    let after = &s[q + 1..];
                    let tail = after.strip_prefix(['s', 'S']).unwrap_or(after);
                    if tail.len() == 1 && "bBoOdDhH".contains(tail) {
                        while matches!(self.peek_char(), Some(' ') | Some('\t')) {
                            self.pos += 1;
                        }
                        while self
                            .peek_char()
                            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_' || c == '?')
                        {
                            self.pos += 1;
                        }
                    }
                }
                tokens.push(Token::Num(self.slice(start)));
            } else if ch == '\''
                && matches!(self.chars.get(self.pos + 1), Some(c) if "sSbBoOdDhH".contains(*c))
            {
                // unsized based literal: `'b0`, `'hff`, `'sd5`, `'bx`
                let start = self.pos;
                self.pos += 1;
                while self
                    .peek_char()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_' || c == '?')
                {
                    self.pos += 1;
                }
                tokens.push(Token::Num(self.slice(start)));
            } else if ch == '"' {
                tokens.push(Token::Str(self.read_string()?));
            } else {
                let three = self.peek_three();
                let two = self.peek_two();
                let sym = if matches!(three.as_deref(), Some(">>>" | "<<<")) {
                    self.pos += 3;
                    three.unwrap()
                } else if matches!(
                    two.as_deref(),
                    Some(
                        "<=" | ">=" | "==" | "!=" | "&&" | "||" | "<<" | ">>" | "++" | "--" | "+:"
                            | "-:" | "::" | "**"
                    )
                ) {
                    self.pos += 2;
                    two.unwrap()
                } else {
                    self.pos += 1;
                    ch.to_string()
                };
                tokens.push(Token::Sym(sym));
            }
        }
        tokens.push(Token::Eof);
        Ok(tokens)
    }

    fn skip_ws_and_comments(&mut self) -> Result<(), ErrorReport> {
        loop {
            while self.peek_char().is_some_and(char::is_whitespace) {
                self.pos += 1;
            }
            if self.peek_two().as_deref() == Some("//") {
                self.pos += 2;
                while self.peek_char().is_some_and(|ch| ch != '\n') {
                    self.pos += 1;
                }
            } else if self.peek_two().as_deref() == Some("/*") {
                self.pos += 2;
                while self.pos + 1 < self.chars.len() && self.peek_two().as_deref() != Some("*/") {
                    self.pos += 1;
                }
                if self.pos + 1 >= self.chars.len() {
                    return Err(err("E_SV_LEX", "unterminated block comment"));
                }
                self.pos += 2;
            } else if self.peek_two().as_deref() == Some("(*")
                && self.chars.get(self.pos + 2) != Some(&')')
            {
                // attribute instance `(* ... *)` — skipped like a comment.
                // (Not `(*)`, which is the `always @(*)` sensitivity list.)
                self.pos += 2;
                while self.pos + 1 < self.chars.len() && self.peek_two().as_deref() != Some("*)") {
                    self.pos += 1;
                }
                if self.pos + 1 >= self.chars.len() {
                    return Err(err("E_SV_LEX", "unterminated attribute `(* ... *)`"));
                }
                self.pos += 2;
            } else {
                return Ok(());
            }
        }
    }

    fn read_string(&mut self) -> Result<String, ErrorReport> {
        self.pos += 1;
        let mut value = String::new();
        while let Some(ch) = self.peek_char() {
            self.pos += 1;
            match ch {
                '"' => return Ok(value),
                '\\' => {
                    let Some(escaped) = self.peek_char() else {
                        return Err(err("E_SV_LEX", "unterminated string literal"));
                    };
                    self.pos += 1;
                    value.push(match escaped {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        '"' => '"',
                        '\\' => '\\',
                        other => other,
                    });
                }
                other => value.push(other),
            }
        }
        Err(err("E_SV_LEX", "unterminated string literal"))
    }

    fn peek_char(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_two(&self) -> Option<String> {
        Some(format!(
            "{}{}",
            self.chars.get(self.pos)?,
            self.chars.get(self.pos + 1)?
        ))
    }

    fn peek_three(&self) -> Option<String> {
        Some(format!(
            "{}{}{}",
            self.chars.get(self.pos)?,
            self.chars.get(self.pos + 1)?,
            self.chars.get(self.pos + 2)?
        ))
    }

    fn slice(&self, start: usize) -> String {
        self.chars[start..self.pos].iter().collect()
    }
}

pub(crate) struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    consts: HashMap<String, u128>,
    types: HashMap<String, SvType>,
    /// Packed-struct layouts: type name → fields `(name, width)` in declaration
    /// order (first field occupies the most-significant bits).
    structs: HashMap<String, Vec<(String, Width)>>,
    /// Declared struct variables: variable name → struct type name, so `p.field`
    /// can be resolved to a bit-slice at parse time.
    var_struct: HashMap<String, String>,
    /// Package contents flattened into global namespaces (declared once, visible
    /// to every module): constants, types, and struct layouts.
    package_consts: HashMap<String, u128>,
    package_types: HashMap<String, SvType>,
    package_structs: HashMap<String, Vec<(String, Width)>>,
    /// Interface definitions (global — declared once, used by many modules).
    interfaces: HashMap<String, IfaceDef>,
    /// Interface variables in the current module (instance/port name → iface type),
    /// so `bus.member` resolves to the flat signal `bus.member`.
    var_iface: HashMap<String, String>,
    /// Interface-array variables (array name → iface type), so `bus[i].member`
    /// resolves to `bus.<i>.member` (with `i` folded during generate expansion).
    var_iface_array: HashMap<String, String>,
    /// Packed multi-dimensional vectors (`[A-1:0][B-1:0] x`) flattened to a wide
    /// signal: variable name → element bit-width, so `x[i]` resolves to the slice
    /// `x[i*B +: B]`.
    packed_elem_width: HashMap<String, Width>,
    /// Interface parameters discovered on the current module's interface ports,
    /// promoted to module parameters so port widths are specializable.
    pending_iface_params: Vec<SvParam>,
    /// Per-instance interface parameter overrides (`bus` → [(param, value)]), so a
    /// `.bus(bus)` connection can forward them to the connected child instance.
    iface_inst_overrides: HashMap<String, Vec<(String, u128)>>,
    module_param_overrides: HashMap<String, HashMap<String, u128>>,
    active_param_overrides: HashMap<String, u128>,
}

impl Parser {
    pub(crate) fn new(source: &str) -> Result<Self, ErrorReport> {
        Self::new_with_module_param_overrides(source, HashMap::new())
    }

    pub(crate) fn new_with_module_param_overrides(
        source: &str,
        module_param_overrides: HashMap<String, HashMap<String, u128>>,
    ) -> Result<Self, ErrorReport> {
        let lexer = Lexer::new(source);
        let _ = lexer.source;
        Ok(Self {
            tokens: Lexer::new(source).tokenize()?,
            pos: 0,
            consts: HashMap::new(),
            types: HashMap::new(),
            structs: HashMap::new(),
            var_struct: HashMap::new(),
            package_consts: HashMap::new(),
            package_types: HashMap::new(),
            package_structs: HashMap::new(),
            interfaces: HashMap::new(),
            var_iface: HashMap::new(),
            var_iface_array: HashMap::new(),
            packed_elem_width: HashMap::new(),
            pending_iface_params: Vec::new(),
            iface_inst_overrides: HashMap::new(),
            module_param_overrides,
            active_param_overrides: HashMap::new(),
        })
    }

    pub(crate) fn parse_source(&mut self) -> Result<SvSource, ErrorReport> {
        let mut modules = Vec::new();
        while !self.is_eof() {
            if self.peek_ident_value("package") {
                self.parse_package()?;
            } else if self.peek_ident_value("import") {
                self.skip_import()?;
            } else if self.peek_ident_value("interface") {
                self.parse_interface()?;
            } else {
                modules.push(self.parse_module()?);
            }
        }
        Ok(SvSource { modules })
    }

    /// `import pkg::*;` / `import pkg::name;` — a no-op: package members are
    /// already flattened into the global namespaces.
    fn skip_import(&mut self) -> Result<(), ErrorReport> {
        while !self.eat_sym(";") {
            if self.is_eof() {
                return Err(err("E_SV_PACKAGE", "unterminated import statement"));
            }
            self.next();
        }
        Ok(())
    }

    /// `package NAME; localparams / typedefs endpackage` — its constants, enum
    /// variants, and types are flattened into the global namespaces so that both
    /// `pkg::NAME` and (after `import pkg::*`) bare `NAME` resolve.
    fn parse_package(&mut self) -> Result<(), ErrorReport> {
        self.expect_ident_value("package")?;
        let _name = self.expect_ident()?;
        self.expect_sym(";")?;
        while !self.eat_ident_value("endpackage") {
            if self.is_eof() {
                return Err(err("E_SV_PACKAGE", "unterminated package"));
            }
            if self.peek_ident_value("parameter") || self.peek_ident_value("localparam") {
                let param = self.parse_param_decl(true)?;
                let value = const_eval(&param.value, &self.consts)?;
                self.consts.insert(param.name.clone(), value);
                self.package_consts.insert(param.name, value);
            } else if self.peek_ident_value("typedef") {
                let td = self.parse_typedef()?;
                self.package_types.insert(td.name.clone(), td.ty);
                // Enum variants become package constants (auto-incrementing).
                let mut next = 0u128;
                for (variant, value) in &td.variants {
                    let value = match value {
                        Some(expr) => const_eval(expr, &self.consts)?,
                        None => next,
                    };
                    self.consts.insert(variant.clone(), value);
                    self.package_consts.insert(variant.clone(), value);
                    next = value.wrapping_add(1);
                }
                if let Some(fields) = self.structs.get(&td.name) {
                    self.package_structs.insert(td.name.clone(), fields.clone());
                }
            } else if self.peek_ident_value("function") {
                // Package functions are not modeled yet (members would need to be
                // injected into each using module); skip the definition.
                self.skip_to_keyword("endfunction")?;
            } else if self.peek_ident_value("import") {
                self.skip_import()?;
            } else {
                return Err(err(
                    "E_SV_PACKAGE",
                    "unsupported package item (expected localparam/typedef/function)",
                ));
            }
        }
        Ok(())
    }

    fn parse_module(&mut self) -> Result<SvModule, ErrorReport> {
        self.consts.clear();
        self.types.clear();
        self.structs.clear();
        self.var_struct.clear();
        self.var_iface.clear();
        self.var_iface_array.clear();
        self.packed_elem_width.clear();
        self.pending_iface_params.clear();
        self.iface_inst_overrides.clear();
        // Seed package-level constants and types (visible to every module).
        self.consts.extend(
            self.package_consts
                .iter()
                .map(|(name, value)| (name.clone(), *value)),
        );
        self.types
            .extend(self.package_types.iter().map(|(n, t)| (n.clone(), *t)));
        self.structs
            .extend(self.package_structs.iter().map(|(n, f)| (n.clone(), f.clone())));
        self.expect_ident_value("module")?;
        let name = self.expect_ident()?;
        self.active_param_overrides = self
            .module_param_overrides
            .get(&name)
            .cloned()
            .unwrap_or_default();
        self.consts.extend(
            self.active_param_overrides
                .iter()
                .map(|(name, value)| (name.clone(), *value)),
        );
        let mut params = if self.eat_sym("#") {
            self.parse_parameter_port_list()?
        } else {
            Vec::new()
        };
        self.expect_sym("(")?;
        let mut ports = Vec::new();
        let mut items = Vec::new();
        if !self.eat_sym(")") {
            loop {
                if self.starts_decl() {
                    let decl = self.parse_ansi_port_decl()?;
                    for declarator in &decl.names {
                        ports.push(declarator.name.clone());
                    }
                    items.push(SvItem::Decl(decl));
                } else if self.peek_interface_type() {
                    self.parse_interface_port(&mut ports, &mut items)?;
                } else {
                    ports.push(self.expect_ident()?);
                }
                if self.eat_sym(")") {
                    break;
                }
                self.expect_sym(",")?;
            }
        }
        self.expect_sym(";")?;
        while !self.eat_ident_value("endmodule") {
            if self.is_eof() {
                return Err(err("E_SV_PARSE", "unterminated module"));
            }
            items.push(self.parse_item()?);
        }
        // Promote interface-port parameters to module parameters (deduped, keeping
        // the explicit module params) so port widths can be specialized per the
        // connected interface instance's overrides.
        for param in self.pending_iface_params.drain(..) {
            if !params.iter().any(|p| p.name == param.name) {
                params.push(param);
            }
        }
        Ok(SvModule {
            name,
            ports,
            params,
            items,
        })
    }

    fn parse_item(&mut self) -> Result<SvItem, ErrorReport> {
        // Tolerate stray empty items (`;;`).
        if self.eat_sym(";") {
            return Ok(SvItem::Generate(Vec::new()));
        }
        if self.peek_ident_value("import") {
            // `import pkg::*;` inside a module — no-op (package members are global).
            self.skip_import()?;
            return Ok(SvItem::Generate(Vec::new()));
        }
        if self.peek_ident_value("parameter") || self.peek_ident_value("localparam") {
            return Ok(SvItem::Param(self.parse_param_decl(true)?));
        }
        if self.peek_ident_value("genvar") {
            self.pos += 1;
            let mut names = Vec::new();
            loop {
                names.push(self.expect_ident()?);
                if self.eat_sym(";") {
                    break;
                }
                self.expect_sym(",")?;
            }
            return Ok(SvItem::Genvar(names));
        }
        if self.peek_ident_value("generate") {
            self.pos += 1;
            let mut items = Vec::new();
            while !self.eat_ident_value("endgenerate") {
                if self.is_eof() {
                    return Err(err("E_SV_GENERATE", "unterminated generate block"));
                }
                items.push(self.parse_item()?);
            }
            return Ok(SvItem::Generate(items));
        }
        if self.peek_ident_value("for") {
            self.pos += 1;
            return self.parse_generate_for_item();
        }
        // Conditional generate: the condition is a compile-time constant, so
        // evaluate it now and keep only the taken branch (the other is parsed
        // and discarded — its items are never elaborated).
        if self.peek_ident_value("if") {
            self.pos += 1;
            self.expect_sym("(")?;
            let cond = self.parse_expr()?;
            self.expect_sym(")")?;
            let taken = const_eval(&cond, &self.consts)? != 0;
            let then_items = self.parse_generate_item_block()?.1;
            let else_items = if self.eat_ident_value("else") {
                if self.peek_ident_value("if") {
                    vec![self.parse_item()?]
                } else {
                    self.parse_generate_item_block()?.1
                }
            } else {
                Vec::new()
            };
            return Ok(SvItem::Generate(if taken { then_items } else { else_items }));
        }
        // Tasks/functions are skipped (e.g. picorv32's no-op `empty_statement`
        // task that the `assert directive expands to in non-formal mode).
        if self.peek_ident_value("task") {
            self.skip_to_keyword("endtask")?;
            return Ok(SvItem::Initial(Vec::new()));
        }
        if self.peek_ident_value("function") {
            return Ok(SvItem::Function(self.parse_function()?));
        }
        if self.peek_ident_value("typedef") {
            return Ok(SvItem::TypeDef(self.parse_typedef()?));
        }
        if self.peek_ident_value("assign") {
            self.pos += 1;
            let dst = self.parse_lvalue()?;
            self.expect_sym("=")?;
            let expr = self.parse_expr()?;
            self.expect_sym(";")?;
            return Ok(SvItem::Assign { dst, expr });
        }
        if self.peek_ident_value("always_comb") {
            self.pos += 1;
            return Ok(SvItem::AlwaysComb(self.parse_stmt_block()?));
        }
        if self.peek_ident_value("initial") {
            self.pos += 1;
            return Ok(SvItem::Initial(self.parse_initial_block()?));
        }
        if self.peek_ident_value("always_ff") {
            self.pos += 1;
            return Ok(SvItem::AlwaysFf(self.parse_always_ff()?));
        }
        if self.peek_ident_value("always_latch") {
            return Err(err(
                "E_SV_ALWAYS",
                "always_latch is not supported by the SV frontend",
            ));
        }
        if self.peek_ident_value("always") {
            self.pos += 1;
            return self.parse_legacy_always();
        }
        if self.starts_decl() {
            return Ok(SvItem::Decl(self.parse_decl()?));
        }
        if self.peek_typedef_type() {
            return Ok(SvItem::Decl(self.parse_typedef_decl()?));
        }
        if self.peek_interface_type() {
            return self.parse_interface_instance();
        }
        self.parse_instance().map(SvItem::Instance)
    }

    fn parse_generate_for_item(&mut self) -> Result<SvItem, ErrorReport> {
        self.expect_sym("(")?;
        let _ = self.eat_ident_value("genvar")
            || self.eat_ident_value("int")
            || self.eat_ident_value("integer");
        let var = self.expect_ident()?;
        self.expect_sym("=")?;
        let init = self.parse_expr()?;
        self.expect_sym(";")?;

        let cond_lhs = self.expect_ident()?;
        if cond_lhs != var {
            return Err(err(
                "E_SV_GENERATE",
                format!("generate-for condition must compare genvar `{var}`"),
            ));
        }
        let cmp = self.parse_for_cmp().map_err(|_| {
            err(
                "E_SV_GENERATE",
                "unsupported generate-for comparison operator",
            )
        })?;
        let bound = self.parse_expr()?;
        self.expect_sym(";")?;

        let step = self.parse_for_update(&var).map_err(|_| {
            err(
                "E_SV_GENERATE",
                "unsupported generate-for update expression",
            )
        })?;
        self.expect_sym(")")?;
        let (label, items) = self.parse_generate_item_block()?;
        Ok(SvItem::GenerateFor {
            var,
            init,
            cmp,
            bound,
            step,
            label,
            items,
        })
    }

    fn parse_generate_item_block(&mut self) -> Result<(Option<String>, Vec<SvItem>), ErrorReport> {
        if self.eat_ident_value("begin") {
            let label = if self.eat_sym(":") {
                Some(self.expect_ident()?)
            } else {
                None
            };
            let mut items = Vec::new();
            while !self.eat_ident_value("end") {
                if self.is_eof() {
                    return Err(err("E_SV_GENERATE", "unterminated generate-for block"));
                }
                items.push(self.parse_item()?);
            }
            Ok((label, items))
        } else {
            Ok((None, vec![self.parse_item()?]))
        }
    }

    fn parse_typedef(&mut self) -> Result<SvTypeDef, ErrorReport> {
        self.expect_ident_value("typedef")?;
        if self.peek_ident_value("struct") {
            return self.parse_struct_typedef();
        }
        self.expect_ident_value("enum")?;
        // Optional base type: `logic`/`reg`/`bit` (default width 1, range overrides)
        // or `int`/`integer` (32-bit). A bare `enum { … }` defaults to int width.
        let mut width: Width = 32;
        if self.eat_ident_value("logic") || self.eat_ident_value("reg") || self.eat_ident_value("bit") {
            width = self.parse_optional_range()?.unwrap_or(1);
        } else if self.eat_ident_value("int") || self.eat_ident_value("integer") {
            width = 32;
        }
        self.expect_sym("{")?;
        let mut variants = Vec::new();
        loop {
            let variant = self.expect_ident()?;
            // The value is optional — an omitted value auto-increments at lowering.
            let value = if self.eat_sym("=") {
                Some(self.parse_expr()?)
            } else {
                None
            };
            variants.push((variant, value));
            if self.eat_sym("}") {
                break;
            }
            self.expect_sym(",")?;
        }
        let name = self.expect_ident()?;
        self.expect_sym(";")?;
        let ty = SvType {
            width,
            signed: false,
        };
        self.types.insert(name.clone(), ty);
        Ok(SvTypeDef { name, ty, variants })
    }

    /// `typedef struct packed [signed] { field … } name;` — a packed struct is
    /// modeled as a flat bit-vector; the first-declared field occupies the
    /// most-significant bits, and `var.field` resolves to a bit-slice.
    fn parse_struct_typedef(&mut self) -> Result<SvTypeDef, ErrorReport> {
        self.expect_ident_value("struct")?;
        // `packed` is required for a bit-vector layout (unpacked structs have no
        // defined bit ordering, so we can't flatten them).
        if !self.eat_ident_value("packed") {
            return Err(err(
                "E_SV_STRUCT",
                "only `struct packed` is supported (unpacked structs have no bit layout)",
            ));
        }
        let signed = self.eat_ident_value("signed");
        self.expect_sym("{")?;
        let mut fields: Vec<(String, Width)> = Vec::new();
        while !self.eat_sym("}") {
            // Field base type: logic/reg/bit [range], or a previously-typedef'd
            // type (enum/struct) used by its total width.
            let field_width = if self.eat_ident_value("logic")
                || self.eat_ident_value("reg")
                || self.eat_ident_value("bit")
            {
                let _ = self.eat_ident_value("signed");
                self.parse_optional_range()?.unwrap_or(1)
            } else if self.peek_typedef_type() {
                let tn = self.expect_ident()?;
                self.types[&tn].width
            } else {
                return Err(err(
                    "E_SV_STRUCT",
                    "unsupported struct field type (expected logic/reg/bit or a typedef)",
                ));
            };
            loop {
                let fname = self.expect_ident()?;
                fields.push((fname, field_width));
                if self.eat_sym(";") {
                    break;
                }
                self.expect_sym(",")?;
            }
        }
        let name = self.expect_ident()?;
        self.expect_sym(";")?;
        let width: Width = fields.iter().map(|(_, w)| *w).sum();
        let ty = SvType { width, signed };
        self.types.insert(name.clone(), ty);
        self.structs.insert(name.clone(), fields);
        Ok(SvTypeDef {
            name,
            ty,
            variants: Vec::new(),
        })
    }

    /// Bit range `(msb, lsb)` of `field` within a packed struct: the first field
    /// is most-significant, fields pack downward toward bit 0.
    fn struct_field_bits(fields: &[(String, Width)], field: &str) -> Option<(Width, Width)> {
        let total: Width = fields.iter().map(|(_, w)| *w).sum();
        let mut hi = total;
        for (fname, w) in fields {
            let lo = hi - w;
            if fname == field {
                return Some((hi - 1, lo));
            }
            hi = lo;
        }
        None
    }

    /// Resolve `var.field` to the bit range of that field, erroring if `var` is
    /// not a known struct variable or `field` is not a member.
    fn member_field_bits(&self, var: &str, field: &str) -> Result<(Width, Width), ErrorReport> {
        let struct_name = self.var_struct.get(var).ok_or_else(|| {
            err(
                "E_SV_STRUCT",
                format!("`{var}` is not a struct variable (member access `.{field}`)"),
            )
        })?;
        let fields = &self.structs[struct_name];
        Self::struct_field_bits(fields, field).ok_or_else(|| {
            err(
                "E_SV_STRUCT",
                format!("struct `{struct_name}` has no field `{field}`"),
            )
        })
    }

    /// Verify `member` exists on the interface bound to `var` (good error
    /// messages for `bus.typo`).
    fn check_iface_member(&self, var: &str, member: &str) -> Result<(), ErrorReport> {
        let iface = &self.var_iface[var];
        let def = &self.interfaces[iface];
        if def.members.iter().any(|m| m.name == member) {
            Ok(())
        } else {
            Err(err(
                "E_SV_INTERFACE",
                format!("interface `{iface}` (`{var}`) has no member `{member}`"),
            ))
        }
    }

    /// Verify `member` exists on the interface bound to interface-array `arr`.
    fn check_iface_array_member(&self, arr: &str, member: &str) -> Result<(), ErrorReport> {
        let iface = &self.var_iface_array[arr];
        let def = &self.interfaces[iface];
        if def.members.iter().any(|m| m.name == member) {
            Ok(())
        } else {
            Err(err(
                "E_SV_INTERFACE",
                format!("interface `{iface}` (`{arr}`) has no member `{member}`"),
            ))
        }
    }

    fn peek_interface_type(&self) -> bool {
        matches!(self.peek(), Token::Ident(value) if self.interfaces.contains_key(value))
    }

    /// `interface NAME #(params) (); members + modports endinterface` — recorded
    /// in `self.interfaces` for later flattening. Members are plain signals; an
    /// optional set of modports assigns per-member directions for ports.
    fn parse_interface(&mut self) -> Result<(), ErrorReport> {
        self.expect_ident_value("interface")?;
        let name = self.expect_ident()?;
        // Member widths are kept as expressions over the interface parameters, so
        // they re-evaluate against per-instance overrides. `#()` params and body
        // localparams are collected in declaration order (later ones may use
        // earlier ones); only `#()` params are overridable.
        let mut params: Vec<SvParam> = Vec::new();
        let saved_consts = self.consts.clone();
        if self.eat_sym("#") {
            params = self.parse_parameter_port_list()?;
            for param in &params {
                let value = const_eval(&param.value, &self.consts)?;
                self.consts.insert(param.name.clone(), value);
            }
        }
        let overridable = params.len();
        if self.eat_sym("(") {
            self.expect_sym(")")?;
        }
        self.expect_sym(";")?;
        let mut members: Vec<IfaceMember> = Vec::new();
        let mut modports: HashMap<String, HashMap<String, SvDirection>> = HashMap::new();
        while !self.eat_ident_value("endinterface") {
            if self.is_eof() {
                return Err(err("E_SV_INTERFACE", "unterminated interface"));
            }
            if self.peek_ident_value("modport") {
                let (mp_name, dirs) = self.parse_modport()?;
                modports.insert(mp_name, dirs);
            } else if self.peek_ident_value("parameter") || self.peek_ident_value("localparam") {
                let param = self.parse_param_decl(true)?;
                let value = const_eval(&param.value, &self.consts)?;
                self.consts.insert(param.name.clone(), value);
                params.push(param);
            } else {
                self.parse_interface_members(&mut members)?;
            }
        }
        self.consts = saved_consts;
        self.interfaces.insert(
            name,
            IfaceDef {
                params,
                overridable,
                members,
                modports,
            },
        );
        Ok(())
    }

    /// One interface member declaration `[logic|reg|bit] [signed] [range] a, b;`.
    /// The width is captured as an expression so it can track parameter overrides.
    fn parse_interface_members(
        &mut self,
        members: &mut Vec<IfaceMember>,
    ) -> Result<(), ErrorReport> {
        let _ = self.eat_ident_value("logic")
            || self.eat_ident_value("reg")
            || self.eat_ident_value("bit");
        let signed = self.eat_ident_value("signed");
        let width = if self.eat_sym("[") {
            let msb = self.parse_expr()?;
            self.expect_sym(":")?;
            let lsb = self.parse_expr()?;
            self.expect_sym("]")?;
            // width = msb - lsb + 1
            SvExpr::Binary {
                op: SvBinaryOp::Add,
                lhs: Box::new(SvExpr::Binary {
                    op: SvBinaryOp::Sub,
                    lhs: Box::new(msb),
                    rhs: Box::new(lsb),
                }),
                rhs: Box::new(SvExpr::Lit {
                    value: 1,
                    width: 32,
                    signed: false,
                }),
            }
        } else {
            SvExpr::Lit {
                value: 1,
                width: 32,
                signed: false,
            }
        };
        loop {
            let member = self.expect_ident()?;
            if self.peek_sym("[") {
                return Err(err(
                    "E_SV_INTERFACE",
                    "interface memory/array members are not supported yet",
                ));
            }
            members.push(IfaceMember {
                name: member,
                width: width.clone(),
                signed,
            });
            if self.eat_sym(";") {
                break;
            }
            self.expect_sym(",")?;
        }
        Ok(())
    }

    /// Resolve an interface's members to concrete `(name, type)` given parameter
    /// overrides (named param → value). Parameters evaluate in declaration order,
    /// each override replacing the default.
    fn resolve_iface_members(
        &self,
        def: &IfaceDef,
        overrides: &HashMap<String, u128>,
    ) -> Result<Vec<(String, SvType)>, ErrorReport> {
        let mut consts = HashMap::new();
        for param in &def.params {
            let value = match overrides.get(&param.name) {
                Some(value) => *value,
                None => const_eval(&param.value, &consts)?,
            };
            consts.insert(param.name.clone(), value);
        }
        let mut out = Vec::with_capacity(def.members.len());
        for member in &def.members {
            let width = const_eval(&member.width, &consts)?;
            let width = Width::try_from(width).map_err(|_| {
                err("E_SV_INTERFACE", "interface member width is out of range")
            })?;
            out.push((
                member.name.clone(),
                SvType {
                    width,
                    signed: member.signed,
                },
            ));
        }
        Ok(out)
    }

    /// `modport NAME (input a, output b, c, …);` — a comma list of members, each
    /// optionally re-stating the direction (which carries to following members).
    fn parse_modport(&mut self) -> Result<(String, HashMap<String, SvDirection>), ErrorReport> {
        self.expect_ident_value("modport")?;
        let name = self.expect_ident()?;
        self.expect_sym("(")?;
        let mut dirs = HashMap::new();
        let mut current: Option<SvDirection> = None;
        loop {
            if self.eat_ident_value("input") {
                current = Some(SvDirection::Input);
            } else if self.eat_ident_value("output") {
                current = Some(SvDirection::Output);
            } else if self.eat_ident_value("inout") {
                current = Some(SvDirection::Inout);
            }
            let member = self.expect_ident()?;
            let dir = current.ok_or_else(|| {
                err("E_SV_INTERFACE", "modport member is missing a direction")
            })?;
            dirs.insert(member, dir);
            if self.eat_sym(")") {
                break;
            }
            self.expect_sym(",")?;
        }
        self.expect_sym(";")?;
        Ok((name, dirs))
    }

    /// An interface used as a module port: `IFACE[.modport] name`. Flattens into
    /// one flat port per (exposed) member, named `name.member`, with the
    /// modport's direction (or `inout` for every member if no modport is given).
    fn parse_interface_port(
        &mut self,
        ports: &mut Vec<String>,
        items: &mut Vec<SvItem>,
    ) -> Result<(), ErrorReport> {
        let iface_name = self.expect_ident()?;
        let modport = if self.eat_sym(".") {
            Some(self.expect_ident()?)
        } else {
            None
        };
        let inst = self.expect_ident()?;
        let def = self.interfaces[&iface_name].clone();
        self.var_iface.insert(inst.clone(), iface_name.clone());
        // Promote the interface's overridable params to module params (so the
        // module can be specialized per connected instance), and use this module's
        // effective value (specialization override, else the interface default)
        // for the port widths.
        let mut overrides = HashMap::new();
        let mut iconsts = HashMap::new();
        for (i, param) in def.params.iter().enumerate() {
            let default = const_eval(&param.value, &iconsts)?;
            iconsts.insert(param.name.clone(), default);
            let effective = self
                .active_param_overrides
                .get(&param.name)
                .copied()
                .unwrap_or(default);
            overrides.insert(param.name.clone(), effective);
            if i < def.overridable {
                self.consts.entry(param.name.clone()).or_insert(effective);
                if !self.pending_iface_params.iter().any(|p| p.name == param.name) {
                    self.pending_iface_params.push(SvParam {
                        name: param.name.clone(),
                        value: param.value.clone(),
                    });
                }
            }
        }
        let members = self.resolve_iface_members(&def, &overrides)?;
        let dirs = match &modport {
            Some(mp) => Some(def.modports.get(mp).cloned().ok_or_else(|| {
                err(
                    "E_SV_INTERFACE",
                    format!("interface `{iface_name}` has no modport `{mp}`"),
                )
            })?),
            None => None,
        };
        for (member, ty) in &members {
            // With a modport, only the members it lists are exposed.
            let direction = match &dirs {
                Some(map) => match map.get(member) {
                    Some(dir) => *dir,
                    None => continue,
                },
                None => SvDirection::Inout,
            };
            let port_name = mangle(&inst, member);
            ports.push(port_name.clone());
            items.push(SvItem::Decl(SvDecl {
                direction: Some(direction),
                kind: SvDeclKind::Logic,
                ty: *ty,
                names: vec![SvDeclarator {
                    name: port_name,
                    memory_depth: None,
                    memory_inner: None,
                    init: None,
                }],
            }));
        }
        Ok(())
    }

    /// An interface instance inside a module: `IFACE name ();`. Expands into one
    /// local signal per member (`name.member`), grouped under a Generate item.
    fn parse_interface_instance(&mut self) -> Result<SvItem, ErrorReport> {
        let iface_name = self.expect_ident()?;
        let def = self.interfaces[&iface_name].clone();
        // Per-instance parameter overrides (`VX_if #(.W(64)) bus()`), evaluated in
        // this module's const scope and matched against the interface's `#()`
        // params (named, or positional by declaration order).
        let mut overrides: HashMap<String, u128> = HashMap::new();
        let mut override_list: Vec<(String, u128)> = Vec::new();
        if self.eat_sym("#") {
            let raw = self.parse_instance_param_overrides()?;
            for (i, ov) in raw.iter().enumerate() {
                let (name, value_expr) = match ov {
                    SvParamOverride::Named { name, value } => (name.clone(), value),
                    SvParamOverride::Positional { value } => {
                        let param = def.params.get(i).filter(|_| i < def.overridable).ok_or_else(
                            || err("E_SV_INTERFACE", "too many positional interface parameters"),
                        )?;
                        (param.name.clone(), value)
                    }
                };
                if !def.params.iter().take(def.overridable).any(|p| p.name == name) {
                    return Err(err(
                        "E_SV_INTERFACE",
                        format!("interface `{iface_name}` has no parameter `{name}`"),
                    ));
                }
                let value = const_eval(value_expr, &self.consts)?;
                overrides.insert(name.clone(), value);
                override_list.push((name, value));
            }
        }
        let inst = self.expect_ident()?;
        // `VX_if bus[N] ();` — an array of N bundles, each flattened to its own
        // set of members named `bus.<i>.member`.
        let array_size = if self.eat_sym("[") {
            let size = self.parse_expr()?;
            self.expect_sym("]")?;
            let n = const_eval(&size, &self.consts)?;
            Some(usize::try_from(n).map_err(|_| {
                err("E_SV_INTERFACE", "interface array size is out of range")
            })?)
        } else {
            None
        };
        self.expect_sym("(")?;
        self.expect_sym(")")?;
        self.expect_sym(";")?;
        // Record the overrides so a `.port(inst)` connection can forward them as
        // module-parameter overrides onto the connected child instance.
        if !override_list.is_empty() {
            self.iface_inst_overrides.insert(inst.clone(), override_list);
        }
        let members = self.resolve_iface_members(&def, &overrides)?;
        let make_decl = |name: String, ty: SvType| {
            SvItem::Decl(SvDecl {
                direction: None,
                kind: SvDeclKind::Logic,
                ty,
                names: vec![SvDeclarator {
                    name,
                    memory_depth: None,
                    memory_inner: None,
                    init: None,
                }],
            })
        };
        let decls = match array_size {
            None => {
                self.var_iface.insert(inst.clone(), iface_name.clone());
                members
                    .iter()
                    .map(|(member, ty)| make_decl(mangle(&inst, member), *ty))
                    .collect()
            }
            Some(n) => {
                self.var_iface_array.insert(inst.clone(), iface_name.clone());
                let mut decls = Vec::with_capacity(n * members.len());
                for i in 0..n {
                    for (member, ty) in &members {
                        decls.push(make_decl(format!("{inst}.{i}.{member}"), *ty));
                    }
                }
                decls
            }
        };
        Ok(SvItem::Generate(decls))
    }

    fn parse_parameter_port_list(&mut self) -> Result<Vec<SvParam>, ErrorReport> {
        self.expect_sym("(")?;
        let mut params = Vec::new();
        if self.eat_sym(")") {
            return Ok(params);
        }
        loop {
            if self.peek_ident_value("parameter") || self.peek_ident_value("localparam") {
                params.push(self.parse_param_decl(false)?);
            } else {
                params.push(self.parse_param_body()?);
            }
            if self.eat_sym(")") {
                break;
            }
            self.expect_sym(",")?;
        }
        Ok(params)
    }

    fn parse_param_decl(&mut self, expect_semicolon: bool) -> Result<SvParam, ErrorReport> {
        if !self.eat_ident_value("parameter") {
            self.expect_ident_value("localparam")?;
        }
        let param = self.parse_param_body()?;
        if expect_semicolon {
            self.expect_sym(";")?;
        }
        Ok(param)
    }

    fn parse_param_body(&mut self) -> Result<SvParam, ErrorReport> {
        self.skip_optional_param_type()?;
        let name = self.expect_ident()?;
        self.expect_sym("=")?;
        let default_value = self.parse_expr()?;
        let (value, evaluated) = if let Some(overridden) = self.active_param_overrides.get(&name) {
            (
                SvExpr::Lit {
                    value: *overridden,
                    width: 32,
                    signed: false,
                },
                *overridden,
            )
        } else {
            // A generate-scope localparam may depend on a genvar that is unknown
            // until generate unrolling — defer evaluation in that case (it will be
            // resolved per-iteration during generate expansion).
            match const_eval(&default_value, &self.consts) {
                Ok(evaluated) => {
                    self.consts.insert(name.clone(), evaluated);
                    return Ok(SvParam {
                        name,
                        value: SvExpr::Lit {
                            value: evaluated,
                            width: 32,
                            signed: false,
                        },
                    });
                }
                Err(_) => return Ok(SvParam { name, value: default_value }),
            }
        };
        self.consts.insert(name.clone(), evaluated);
        Ok(SvParam { name, value })
    }

    fn skip_optional_param_type(&mut self) -> Result<(), ErrorReport> {
        if self.eat_ident_value("int") || self.eat_ident_value("integer") {
            return Ok(());
        }
        // `parameter string X = "..."` — string literals lower to packed bit
        // vectors, so a string param is just a (wide) constant; skip the keyword.
        if self.eat_ident_value("string") {
            return Ok(());
        }
        if self.peek_ident_value("logic")
            || self.peek_ident_value("wire")
            || self.peek_ident_value("reg")
            || self.peek_ident_value("bit")
        {
            self.pos += 1;
        }
        // optional signedness then an optional packed range, e.g. `parameter [0:0] X`
        self.eat_ident_value("signed");
        self.eat_ident_value("unsigned");
        let _ = self.parse_optional_range()?;
        Ok(())
    }

    fn parse_ansi_port_decl(&mut self) -> Result<SvDecl, ErrorReport> {
        let direction = if self.eat_ident_value("input") {
            Some(SvDirection::Input)
        } else if self.eat_ident_value("output") {
            Some(SvDirection::Output)
        } else if self.eat_ident_value("inout") {
            Some(SvDirection::Inout)
        } else {
            None
        };
        let kind = if self.eat_ident_value("wire") {
            SvDeclKind::Wire
        } else if self.eat_ident_value("reg") {
            SvDeclKind::Reg
        } else {
            self.eat_ident_value("logic");
            SvDeclKind::Logic
        };
        let signed = self.eat_ident_value("signed");
        let (width, packed_elem) = self.parse_packed_dims()?;
        let mut names = Vec::new();
        loop {
            let name = self.expect_ident()?;
            if let Some(elem) = packed_elem {
                self.packed_elem_width.insert(name.clone(), elem);
            }
            names.push(SvDeclarator {
                name,
                memory_depth: None,
                memory_inner: None,
                init: None,
            });
            if !self.eat_sym(",") {
                break;
            }
            if self.peek_sym(")") || self.starts_decl() || self.peek_interface_type() {
                self.pos -= 1;
                break;
            }
        }
        Ok(SvDecl {
            direction,
            kind,
            ty: SvType { width, signed },
            names,
        })
    }

    /// `function [automatic] [type] name (input … , …); [locals] [stmts] endfunction`
    /// A pure combinational subroutine; `return e;` is recorded as `name = e;`.
    fn parse_function(&mut self) -> Result<SvFunction, ErrorReport> {
        self.expect_ident_value("function")?;
        self.eat_ident_value("automatic");
        self.eat_ident_value("static");
        // Optional return type: `int`/`integer` (32-bit signed), or `[logic|reg|bit]
        // [signed] [range]`.
        let int_w = if self.eat_ident_value("int") || self.eat_ident_value("integer") {
            Some(32)
        } else {
            self.eat_ident_value("logic");
            self.eat_ident_value("reg");
            self.eat_ident_value("bit");
            None
        };
        let mut signed = int_w.is_some() || self.eat_ident_value("signed");
        self.eat_ident_value("unsigned");
        let width = match int_w {
            Some(w) => w,
            None => self.parse_optional_range()?.unwrap_or(1),
        };
        signed = signed || self.eat_ident_value("signed");
        let return_type = SvType { width, signed };
        let name = self.expect_ident()?;

        // Input ports `(input [..] a, input [..] b)` (functions only have inputs).
        let mut inputs = Vec::new();
        if self.eat_sym("(") && !self.eat_sym(")") {
            loop {
                self.eat_ident_value("input");
                let pint = if self.eat_ident_value("int") || self.eat_ident_value("integer") {
                    Some(32)
                } else {
                    self.eat_ident_value("logic");
                    self.eat_ident_value("reg");
                    self.eat_ident_value("bit");
                    None
                };
                let psigned = pint.is_some() || self.eat_ident_value("signed");
                self.eat_ident_value("unsigned");
                let pwidth = match pint {
                    Some(w) => w,
                    None => self.parse_optional_range()?.unwrap_or(1),
                };
                let pname = self.expect_ident()?;
                inputs.push((pname, SvType { width: pwidth, signed: psigned }));
                if self.eat_sym(")") {
                    break;
                }
                self.expect_sym(",")?;
            }
        }
        self.expect_sym(";")?;

        // Body: local declarations and statements until `endfunction`.
        let mut locals = Vec::new();
        let mut body = Vec::new();
        loop {
            if self.eat_ident_value("endfunction") {
                break;
            }
            if self.is_eof() {
                return Err(err(
                    "E_SV_FUNCTION",
                    "unterminated function, expected `endfunction`",
                ));
            }
            if self.eat_ident_value("return") {
                let expr = self.parse_expr()?;
                self.expect_sym(";")?;
                body.push(SvStmt::Assign {
                    dst: SvLvalue::Signal(name.clone()),
                    nonblocking: false,
                    expr,
                });
                continue;
            }
            if self.peek_ident_value("logic")
                || self.peek_ident_value("reg")
                || self.peek_ident_value("int")
                || self.peek_ident_value("integer")
                || self.peek_ident_value("bit")
                || self.peek_ident_value("wire")
            {
                locals.push(self.parse_decl()?);
                continue;
            }
            body.push(self.parse_stmt()?);
        }
        Ok(SvFunction { name, return_type, inputs, locals, body })
    }

    fn parse_decl(&mut self) -> Result<SvDecl, ErrorReport> {
        let direction = if self.eat_ident_value("input") {
            Some(SvDirection::Input)
        } else if self.eat_ident_value("output") {
            Some(SvDirection::Output)
        } else if self.eat_ident_value("inout") {
            Some(SvDirection::Inout)
        } else {
            None
        };
        // `integer`/`int` are 32-bit signed variables (no range follows).
        let (kind, int_width) = if self.eat_ident_value("integer") || self.eat_ident_value("int") {
            (SvDeclKind::Reg, Some(32))
        } else if self.eat_ident_value("wire") {
            (SvDeclKind::Wire, None)
        } else if self.eat_ident_value("reg") {
            (SvDeclKind::Reg, None)
        } else if direction.is_some() {
            self.eat_ident_value("logic");
            (SvDeclKind::Logic, None)
        } else {
            self.expect_ident_value("logic")?;
            (SvDeclKind::Logic, None)
        };
        let signed = int_width.is_some() || self.eat_ident_value("signed");
        let (width, packed_elem) = match int_width {
            Some(w) => (w, None),
            None => self.parse_packed_dims()?,
        };
        let mut names = Vec::new();
        loop {
            let name = self.expect_ident()?;
            if let Some(elem) = packed_elem {
                self.packed_elem_width.insert(name.clone(), elem);
            }
            // Unpacked array dimensions: `[0:D1-1][0:D2-1]` (up to 2-D), flattened
            // row-major into a memory of depth D1*D2.
            let mut dims: Vec<usize> = Vec::new();
            while self.eat_sym("[") {
                let first = self.expect_usize_const()?;
                // `[N]` is shorthand for `[0:N-1]` (size N); `[a:b]` is a range.
                let d = if self.eat_sym(":") {
                    let second = self.expect_usize_const()?;
                    self.expect_sym("]")?;
                    if first == 0 && second > 0 {
                        second + 1
                    } else if second == 0 && first > 0 {
                        first + 1
                    } else if first == 0 && second == 0 {
                        1
                    } else {
                        return Err(err(
                            "E_SV_MEMORY_RANGE",
                            "memory ranges must use [0:D-1] or [D-1:0] depth syntax",
                        ));
                    }
                } else {
                    self.expect_sym("]")?;
                    first
                };
                dims.push(d);
                if dims.len() == 2 && self.peek_sym("[") {
                    return Err(err(
                        "E_SV_MEMORY_RANGE",
                        "arrays with more than 2 unpacked dimensions are not supported",
                    ));
                }
            }
            let (memory_depth, memory_inner) = match dims.as_slice() {
                [] => (None, None),
                [d] => (Some(*d), None),
                [d1, d2] => (Some(d1 * d2), Some(*d2)),
                _ => unreachable!(),
            };
            let init = if memory_depth.is_none() && self.eat_sym("=") {
                Some(self.parse_expr()?)
            } else {
                None
            };
            names.push(SvDeclarator { name, memory_depth, memory_inner, init });
            if self.eat_sym(";") {
                break;
            }
            self.expect_sym(",")?;
        }
        Ok(SvDecl {
            direction,
            kind,
            ty: SvType { width, signed },
            names,
        })
    }

    fn parse_typedef_decl(&mut self) -> Result<SvDecl, ErrorReport> {
        let mut type_name = self.expect_ident()?;
        // Drop a package qualifier: `pkg::type_t` → `type_t` (types are global).
        if self.eat_sym("::") {
            type_name = self.expect_ident()?;
        }
        let ty = *self.types.get(&type_name).ok_or_else(|| {
            err(
                "E_SV_TYPE",
                format!("unknown SystemVerilog type `{type_name}`"),
            )
        })?;
        let is_struct = self.structs.contains_key(&type_name);
        let mut names = Vec::new();
        loop {
            let var = self.expect_ident()?;
            // Remember struct-typed variables so `var.field` resolves to a slice.
            if is_struct {
                self.var_struct.insert(var.clone(), type_name.clone());
            }
            names.push(SvDeclarator {
                name: var,
                memory_depth: None,
                memory_inner: None,
                init: None,
            });
            if self.eat_sym(";") {
                break;
            }
            self.expect_sym(",")?;
        }
        Ok(SvDecl {
            direction: None,
            kind: SvDeclKind::Logic,
            ty,
            names,
        })
    }

    /// Forward an interface instance/array's parameter overrides as named module-
    /// parameter overrides on the child instance, so it specializes to the
    /// matching member widths.
    fn forward_iface_overrides(&self, name: &str, params: &mut Vec<SvParamOverride>) {
        if let Some(overrides) = self.iface_inst_overrides.get(name) {
            for (pname, pvalue) in overrides {
                params.push(SvParamOverride::Named {
                    name: pname.clone(),
                    value: SvExpr::Lit {
                        value: *pvalue,
                        width: 32,
                        signed: false,
                    },
                });
            }
        }
    }

    fn parse_instance(&mut self) -> Result<SvInstance, ErrorReport> {
        let module = self.expect_ident()?;
        let mut params = if self.eat_sym("#") {
            self.parse_instance_param_overrides()?
        } else {
            Vec::new()
        };
        let name = self.expect_ident()?;
        self.expect_sym("(")?;
        let mut connections = Vec::new();
        if !self.eat_sym(")") {
            loop {
                self.expect_sym(".")?;
                let port = self.expect_ident()?;
                // `.clk` shorthand connects the port to the same-named signal.
                if !self.eat_sym("(") {
                    connections.push(SvConnection {
                        port: port.clone(),
                        expr: SvExpr::Ident(port),
                    });
                    if self.eat_sym(")") {
                        break;
                    }
                    self.expect_sym(",")?;
                    continue;
                }
                // `.port()` — an explicitly unconnected port; drop the connection.
                if self.eat_sym(")") {
                    if self.eat_sym(")") {
                        break;
                    }
                    self.expect_sym(",")?;
                    continue;
                }
                let expr = self.parse_expr()?;
                self.expect_sym(")")?;
                // Connecting an interface bundle expands member-wise: `.bus(x)`
                // becomes `.bus.m0(x.m0), .bus.m1(x.m1), …`.
                match &expr {
                    SvExpr::Ident(arg) if self.var_iface.contains_key(arg) => {
                        let iface = self.var_iface[arg].clone();
                        for member in &self.interfaces[&iface].members {
                            connections.push(SvConnection {
                                port: mangle(&port, &member.name),
                                expr: SvExpr::Ident(mangle(arg, &member.name)),
                            });
                        }
                        self.forward_iface_overrides(arg, &mut params);
                    }
                    // `.port(bus[i])` — connect one element of an interface array.
                    // The index stays symbolic (a genvar) and folds during generate
                    // unrolling; each member becomes a sentinel-indexed connection.
                    SvExpr::Bracket { expr: inner, index }
                        if matches!(inner.as_ref(), SvExpr::Ident(a) if self.var_iface_array.contains_key(a)) =>
                    {
                        let SvExpr::Ident(arr) = inner.as_ref() else { unreachable!() };
                        let iface = self.var_iface_array[arr].clone();
                        for member in &self.interfaces[&iface].members {
                            connections.push(SvConnection {
                                port: mangle(&port, &member.name),
                                expr: SvExpr::Bracket {
                                    expr: Box::new(SvExpr::Ident(iface_array_ref(arr, &member.name))),
                                    index: index.clone(),
                                },
                            });
                        }
                        self.forward_iface_overrides(arr, &mut params);
                    }
                    _ => connections.push(SvConnection { port, expr }),
                }
                if self.eat_sym(")") {
                    break;
                }
                self.expect_sym(",")?;
            }
        }
        self.expect_sym(";")?;
        Ok(SvInstance {
            module,
            name,
            params,
            connections,
        })
    }

    fn parse_instance_param_overrides(&mut self) -> Result<Vec<SvParamOverride>, ErrorReport> {
        self.expect_sym("(")?;
        let mut params = Vec::new();
        if self.eat_sym(")") {
            return Ok(params);
        }
        let mut mode = None::<bool>;
        loop {
            let is_named = self.eat_sym(".");
            match mode {
                Some(existing) if existing != is_named => {
                    return Err(err(
                        "E_SV_PARAM_OVERRIDE",
                        "mixed named and positional instance parameter overrides are not supported",
                    ));
                }
                None => mode = Some(is_named),
                Some(_) => {}
            }
            if is_named {
                let name = self.expect_ident()?;
                self.expect_sym("(")?;
                let value = self.parse_expr()?;
                self.expect_sym(")")?;
                params.push(SvParamOverride::Named { name, value });
            } else {
                let value = self.parse_expr()?;
                params.push(SvParamOverride::Positional { value });
            }
            if self.eat_sym(")") {
                break;
            }
            self.expect_sym(",")?;
        }
        Ok(params)
    }

    fn parse_always_ff(&mut self) -> Result<SvAlwaysFf, ErrorReport> {
        let (clock, async_reset) = self.parse_clocked_event_control()?;
        Ok(SvAlwaysFf {
            clock,
            async_reset,
            body: self.parse_stmt_block()?,
        })
    }

    fn parse_legacy_always(&mut self) -> Result<SvItem, ErrorReport> {
        self.expect_sym("@")?;
        if self.eat_sym("*") {
            return Ok(SvItem::AlwaysComb(self.parse_stmt_block()?));
        }
        if self.eat_sym("(") {
            if self.eat_sym("*") {
                self.expect_sym(")")?;
                return Ok(SvItem::AlwaysComb(self.parse_stmt_block()?));
            }
            if self.peek_ident_value("posedge") {
                let (clock, async_reset) = self.parse_clocked_event_control_after_open_paren()?;
                return Ok(SvItem::AlwaysFf(SvAlwaysFf {
                    clock,
                    async_reset,
                    body: self.parse_stmt_block()?,
                }));
            }
            if self.peek_ident_value("negedge") {
                return Err(err(
                    "E_SV_ALWAYS",
                    "negative-edge primary clocks are not supported",
                ));
            }
            return Err(err(
                "E_SV_ALWAYS",
                "level-sensitive always event controls are not supported",
            ));
        }
        Err(err(
            "E_SV_ALWAYS",
            format!(
                "expected always event control, found `{}`",
                self.peek().display()
            ),
        ))
    }

    fn parse_clocked_event_control(
        &mut self,
    ) -> Result<(String, Option<SvResetEdge>), ErrorReport> {
        self.expect_sym("@")?;
        self.expect_sym("(")?;
        self.parse_clocked_event_control_after_open_paren()
    }

    fn parse_clocked_event_control_after_open_paren(
        &mut self,
    ) -> Result<(String, Option<SvResetEdge>), ErrorReport> {
        self.expect_ident_value("posedge")?;
        let clock = self.expect_ident()?;
        let async_reset = if self.eat_ident_value("or") {
            let active_low = if self.eat_ident_value("negedge") {
                true
            } else {
                self.expect_ident_value("posedge")?;
                false
            };
            Some(SvResetEdge {
                signal: self.expect_ident()?,
                active_low,
            })
        } else {
            None
        };
        self.expect_sym(")")?;
        Ok((clock, async_reset))
    }

    fn parse_initial_block(&mut self) -> Result<Vec<SvInitialAssign>, ErrorReport> {
        // Parse the full statement block, then keep only the top-level simple
        // assignments as power-on initial values. Control flow (if/for/case —
        // e.g. a conditional register zero-init) is tolerated and ignored: it
        // either is dead or only sets the already-zero default state.
        let begin = self.eat_ident_value("begin");
        let mut assigns = Vec::new();
        loop {
            if begin {
                if self.eat_ident_value("end") {
                    break;
                }
                if self.is_eof() {
                    return Err(err("E_SV_INITIAL", "unterminated initial block"));
                }
            }
            // `$readmemh("file", mem [, start [, end]]);` / `$readmemb(...)`
            if self.peek_sym("$") {
                self.next();
                let func = self.expect_ident()?;
                if func == "readmemh" || func == "readmemb" {
                    self.expect_sym("(")?;
                    let file = self.expect_string()?;
                    self.expect_sym(",")?;
                    let mem = self.expect_ident()?;
                    while !self.eat_sym(")") {
                        if self.is_eof() {
                            return Err(err("E_SV_READMEM", "unterminated $readmem call"));
                        }
                        self.next(); // ignore optional start/end address args
                    }
                    self.expect_sym(";")?;
                    assigns.push(SvInitialAssign::ReadMem {
                        hex: func == "readmemh",
                        file,
                        mem,
                    });
                } else {
                    // other system task ($display/$finish/…): skip to `;`
                    while !self.eat_sym(";") {
                        if self.is_eof() {
                            break;
                        }
                        self.next();
                    }
                }
                if !begin {
                    break;
                }
                continue;
            }
            // A simple `dst = expr;` is a power-on value; other control flow is
            // tolerated and ignored (it only re-sets the already-zero default).
            let stmt = self.parse_stmt()?;
            if let SvStmt::Assign { dst, expr, .. } = stmt {
                assigns.push(SvInitialAssign::Assign { dst, expr });
            }
            if !begin {
                break;
            }
        }
        Ok(assigns)
    }

    fn parse_stmt_block(&mut self) -> Result<Vec<SvStmt>, ErrorReport> {
        if self.eat_ident_value("begin") {
            let mut stmts = Vec::new();
            while !self.eat_ident_value("end") {
                stmts.push(self.parse_stmt()?);
            }
            Ok(stmts)
        } else {
            Ok(vec![self.parse_stmt()?])
        }
    }

    fn parse_stmt(&mut self) -> Result<SvStmt, ErrorReport> {
        if self.eat_ident_value("for") {
            return self.parse_for_stmt();
        }
        if self.eat_ident_value("if") {
            self.expect_sym("(")?;
            let cond = self.parse_expr()?;
            self.expect_sym(")")?;
            let then_stmts = self.parse_stmt_block()?;
            let else_stmts = if self.eat_ident_value("else") {
                self.parse_stmt_block()?
            } else {
                Vec::new()
            };
            return Ok(SvStmt::If {
                cond,
                then_stmts,
                else_stmts,
            });
        }
        if self.peek_ident_value("unique")
            || self.peek_ident_value("unique0")
            || self.peek_ident_value("priority")
        {
            self.next();
            if self.peek_ident_value("case")
                || self.peek_ident_value("casez")
                || self.peek_ident_value("casex")
            {
                return self.parse_case_stmt();
            }
            return Err(err(
                "E_SV_PARSE",
                "unique/priority statement qualifier is only supported before case",
            ));
        }
        if self.peek_ident_value("case") {
            return self.parse_case_stmt();
        }
        if self.peek_ident_value("casez") || self.peek_ident_value("casex") {
            return self.parse_case_stmt();
        }
        if matches!(self.peek(), Token::Ident(_)) && self.peek_next_sym(":") {
            return self.parse_property_stmt();
        }
        // concat-lvalue assignment: `{a, b[3:0], c} = expr;`
        if self.peek_sym("{") {
            self.pos += 1;
            let mut parts = Vec::new();
            loop {
                parts.push(self.parse_lvalue()?);
                if self.eat_sym("}") {
                    break;
                }
                self.expect_sym(",")?;
            }
            let nonblocking = if self.eat_sym("<=") {
                true
            } else {
                self.expect_sym("=")?;
                false
            };
            let expr = self.parse_expr()?;
            self.expect_sym(";")?;
            return Ok(SvStmt::ConcatAssign {
                parts,
                nonblocking,
                expr,
            });
        }
        // Task / system-task call statement (`empty_statement;`, `$display(...);`,
        // `$finish;`) — no datapath effect, treated as a no-op.
        if matches!(self.peek(), Token::Ident(_)) && (self.peek_next_sym(";") || self.peek_next_sym("(")) {
            self.next();
            if self.eat_sym("(") {
                let mut depth = 1;
                while depth > 0 {
                    if self.is_eof() {
                        return Err(err("E_SV_PARSE", "unterminated call statement"));
                    }
                    if self.eat_sym("(") {
                        depth += 1;
                    } else if self.eat_sym(")") {
                        depth -= 1;
                    } else {
                        self.next();
                    }
                }
            }
            self.expect_sym(";")?;
            return Ok(SvStmt::Nop);
        }
        let dst = self.parse_lvalue()?;
        let nonblocking = if self.eat_sym("<=") {
            true
        } else {
            self.expect_sym("=")?;
            false
        };
        let expr = self.parse_expr()?;
        self.expect_sym(";")?;
        Ok(SvStmt::Assign {
            dst,
            nonblocking,
            expr,
        })
    }

    fn parse_for_stmt(&mut self) -> Result<SvStmt, ErrorReport> {
        self.expect_sym("(")?;
        let _ = self.eat_ident_value("int") || self.eat_ident_value("integer");
        let var = self.expect_ident()?;
        self.expect_sym("=")?;
        let init = self.parse_expr()?;
        self.expect_sym(";")?;

        let cond_lhs = self.expect_ident()?;
        if cond_lhs != var {
            return Err(err(
                "E_SV_FOR",
                format!("for-loop condition must compare loop variable `{var}`"),
            ));
        }
        let cmp = self.parse_for_cmp()?;
        let bound = self.parse_expr()?;
        self.expect_sym(";")?;

        let step = self.parse_for_update(&var)?;
        self.expect_sym(")")?;
        let body = self.parse_stmt_block()?;
        Ok(SvStmt::For {
            var,
            init,
            cmp,
            bound,
            step,
            body,
        })
    }

    fn parse_for_cmp(&mut self) -> Result<SvForCmp, ErrorReport> {
        if self.eat_sym("<") {
            Ok(SvForCmp::Lt)
        } else if self.eat_sym("<=") {
            Ok(SvForCmp::Le)
        } else if self.eat_sym(">") {
            Ok(SvForCmp::Gt)
        } else if self.eat_sym(">=") {
            Ok(SvForCmp::Ge)
        } else {
            Err(err(
                "E_SV_FOR",
                format!(
                    "unsupported for-loop comparison `{}`",
                    self.peek().display()
                ),
            ))
        }
    }

    /// A full for-loop update, accepting both pre- (`++i`) and post-increment
    /// (`i++`), plus `i = i + k`.
    fn parse_for_update(&mut self, var: &str) -> Result<SvForStep, ErrorReport> {
        // Pre-increment/decrement: `++i` / `--i`.
        if self.eat_sym("++") {
            self.expect_var(var)?;
            return Ok(SvForStep::Inc);
        }
        if self.eat_sym("--") {
            self.expect_var(var)?;
            return Ok(SvForStep::Dec);
        }
        self.expect_var(var)?;
        self.parse_for_step(var)
    }

    fn expect_var(&mut self, var: &str) -> Result<(), ErrorReport> {
        let got = self.expect_ident()?;
        if got != var {
            return Err(err(
                "E_SV_FOR",
                format!("for-loop update must update `{var}`"),
            ));
        }
        Ok(())
    }

    fn parse_for_step(&mut self, var: &str) -> Result<SvForStep, ErrorReport> {
        if self.eat_sym("++") {
            return Ok(SvForStep::Inc);
        }
        if self.eat_sym("--") {
            return Ok(SvForStep::Dec);
        }
        self.expect_sym("=")?;
        let lhs = self.expect_ident()?;
        if lhs != var {
            return Err(err(
                "E_SV_FOR",
                format!("for-loop update expression must start from `{var}`"),
            ));
        }
        if self.eat_sym("+") {
            Ok(SvForStep::Add(self.parse_expr()?))
        } else if self.eat_sym("-") {
            Ok(SvForStep::Sub(self.parse_expr()?))
        } else {
            Err(err(
                "E_SV_FOR",
                "for-loop update only supports ++, --, +, and - steps",
            ))
        }
    }

    fn parse_case_stmt(&mut self) -> Result<SvStmt, ErrorReport> {
        let kind = if self.eat_ident_value("case") {
            SvCaseKind::Normal
        } else if self.eat_ident_value("casez") {
            SvCaseKind::CaseZ
        } else {
            self.expect_ident_value("casex")?;
            SvCaseKind::CaseX
        };
        self.expect_sym("(")?;
        let expr = self.parse_expr()?;
        self.expect_sym(")")?;
        let mut items = Vec::new();
        while !self.eat_ident_value("endcase") {
            if self.eat_ident_value("default") {
                self.eat_sym(":");
                let stmts = self.parse_stmt_block()?;
                items.push(SvCaseItem {
                    labels: Vec::new(),
                    stmts,
                    is_default: true,
                });
                continue;
            }

            let mut labels = vec![self.parse_case_label(kind)?];
            while self.eat_sym(",") {
                labels.push(self.parse_case_label(kind)?);
            }
            self.expect_sym(":")?;
            let stmts = self.parse_stmt_block()?;
            items.push(SvCaseItem {
                labels,
                stmts,
                is_default: false,
            });
        }
        Ok(SvStmt::Case { kind, expr, items })
    }

    fn parse_case_label(&mut self, kind: SvCaseKind) -> Result<SvCaseLabel, ErrorReport> {
        if matches!(self.peek(), Token::Num(raw) if raw.contains('\'') && has_wildcard_digit(raw)) {
            let Token::Num(raw) = self.next().clone() else {
                unreachable!("peek checked numeric token")
            };
            return parse_wildcard_case_label(&raw, kind);
        }
        Ok(SvCaseLabel::Expr(self.parse_expr()?))
    }

    fn parse_property_stmt(&mut self) -> Result<SvStmt, ErrorReport> {
        let name = self.expect_ident()?;
        self.expect_sym(":")?;
        if self.eat_ident_value("assert") {
            self.expect_sym("(")?;
            let cond = self.parse_expr()?;
            self.expect_sym(")")?;
            let message = if self.eat_ident_value("else") {
                self.expect_sym("$")?;
                self.expect_ident_value("error")?;
                self.expect_sym("(")?;
                let message = self.expect_string()?;
                self.expect_sym(")")?;
                Some(message)
            } else {
                None
            };
            self.expect_sym(";")?;
            return Ok(SvStmt::Assert {
                name,
                cond,
                message,
            });
        }
        if self.eat_ident_value("cover") {
            self.expect_sym("(")?;
            let cond = self.parse_expr()?;
            self.expect_sym(")")?;
            self.expect_sym(";")?;
            return Ok(SvStmt::Cover { name, cond });
        }
        Err(err(
            "E_SV_PROPERTY",
            "labeled statements only support emitted assert/cover forms",
        ))
    }

    fn parse_lvalue(&mut self) -> Result<SvLvalue, ErrorReport> {
        let name = self.expect_ident()?;
        // `var.field = …` — interface member is a flat signal; struct member is
        // a bit-slice.
        if self.eat_sym(".") {
            let field = self.expect_ident()?;
            if self.var_iface.contains_key(&name) {
                self.check_iface_member(&name, &field)?;
                return Ok(SvLvalue::Signal(mangle(&name, &field)));
            }
            let (msb, lsb) = self.member_field_bits(&name, &field)?;
            let lit = |w: Width| SvExpr::Lit {
                value: w as u128,
                width: 32,
                signed: false,
            };
            return Ok(SvLvalue::Slice {
                name,
                msb: lit(msb),
                lsb: lit(lsb),
            });
        }
        if self.eat_sym("[") {
            let first = self.parse_expr()?;
            // Indexed part-select `[base +: W]` / `[base -: W]` (W constant). The
            // bounds stay symbolic (e.g. a loop variable) and const-fold later.
            if self.eat_sym("+:") {
                let w = self.expect_width_const()?;
                self.expect_sym("]")?;
                return Ok(SvLvalue::Slice {
                    name,
                    msb: add_const(first.clone(), w - 1),
                    lsb: first,
                });
            }
            if self.eat_sym("-:") {
                let w = self.expect_width_const()?;
                self.expect_sym("]")?;
                return Ok(SvLvalue::Slice {
                    name,
                    lsb: sub_const(first.clone(), w - 1),
                    msb: first,
                });
            }
            if self.eat_sym(":") {
                let lsb = self.parse_expr()?;
                self.expect_sym("]")?;
                return Ok(SvLvalue::Slice {
                    name,
                    msb: first,
                    lsb,
                });
            }
            self.expect_sym("]")?;
            // `arr[i].member <= …` — interface-array element member write. Encoded
            // as a sentinel Bit; resolves to `arr.<i>.member` after generate unroll.
            if self.var_iface_array.contains_key(&name) && self.eat_sym(".") {
                let member = self.expect_ident()?;
                self.check_iface_array_member(&name, &member)?;
                return Ok(SvLvalue::Bit {
                    name: iface_array_ref(&name, &member),
                    index: first,
                });
            }
            if self.eat_sym("[") {
                // 2-D unpacked memory write `mem[i][j]`.
                let inner = self.parse_expr()?;
                self.expect_sym("]")?;
                return Ok(SvLvalue::Memory2D { name, outer: first, inner });
            }
            // `x[i] <= …` on a packed multi-dim vector writes element `i`'s slice.
            if let Some(&elem) = self.packed_elem_width.get(&name) {
                let base = mul_const(first, elem);
                return Ok(SvLvalue::Slice {
                    name,
                    msb: add_const(base.clone(), elem - 1),
                    lsb: base,
                });
            }
            Ok(SvLvalue::Bit { name, index: first })
        } else {
            Ok(SvLvalue::Signal(name))
        }
    }

    fn parse_expr(&mut self) -> Result<SvExpr, ErrorReport> {
        let cond = self.parse_binary(0)?;
        if self.eat_sym("?") {
            let then_expr = self.parse_expr()?;
            self.expect_sym(":")?;
            let else_expr = self.parse_expr()?;
            Ok(SvExpr::Ternary {
                cond: Box::new(cond),
                then_expr: Box::new(then_expr),
                else_expr: Box::new(else_expr),
            })
        } else {
            Ok(cond)
        }
    }

    fn parse_binary(&mut self, min_bp: u8) -> Result<SvExpr, ErrorReport> {
        let mut lhs = self.parse_unary()?;
        while let Some((op, left_bp, right_bp)) = self.peek_binary_op() {
            if left_bp < min_bp {
                break;
            }
            self.pos += 1;
            let rhs = self.parse_binary(right_bp)?;
            lhs = SvExpr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<SvExpr, ErrorReport> {
        if self.eat_sym("!") {
            Ok(SvExpr::Unary {
                op: SvUnaryOp::Not,
                expr: Box::new(self.parse_unary()?),
            })
        } else if self.eat_sym("~") {
            Ok(SvExpr::Unary {
                op: SvUnaryOp::BitNot,
                expr: Box::new(self.parse_unary()?),
            })
        } else if self.eat_sym("-") {
            Ok(SvExpr::Unary {
                op: SvUnaryOp::Neg,
                expr: Box::new(self.parse_unary()?),
            })
        } else if self.eat_sym("&") {
            // Reduction operators in prefix position (`&a`, `|a`, `^a`).
            Ok(SvExpr::Unary {
                op: SvUnaryOp::RedAnd,
                expr: Box::new(self.parse_unary()?),
            })
        } else if self.eat_sym("|") {
            Ok(SvExpr::Unary {
                op: SvUnaryOp::RedOr,
                expr: Box::new(self.parse_unary()?),
            })
        } else if self.eat_sym("^") {
            Ok(SvExpr::Unary {
                op: SvUnaryOp::RedXor,
                expr: Box::new(self.parse_unary()?),
            })
        } else {
            self.parse_postfix()
        }
    }

    fn parse_postfix(&mut self) -> Result<SvExpr, ErrorReport> {
        let mut expr = self.parse_primary()?;
        loop {
            // Sized cast `W'(e)` — resize `e` to W bits (unsigned: mask off the
            // upper bits). The width `W` is the constant just parsed.
            if self.peek_sym("'") && self.peek_next_sym("(") {
                let width = const_eval(&expr, &self.consts)?;
                self.expect_sym("'")?;
                self.expect_sym("(")?;
                let inner = self.parse_expr()?;
                self.expect_sym(")")?;
                let mask = if width >= 128 {
                    u128::MAX
                } else {
                    (1u128 << width) - 1
                };
                expr = SvExpr::Binary {
                    op: SvBinaryOp::And,
                    lhs: Box::new(inner),
                    rhs: Box::new(SvExpr::Lit {
                        value: mask,
                        width: width.min(128) as Width,
                        signed: false,
                    }),
                };
                continue;
            }
            // `var.field` — struct member resolves to a bit-slice; interface
            // member resolves to the flat signal `var.field`; `arr[i].field`
            // resolves to `arr.<i>.field` (index folded after generate unroll).
            if self.eat_sym(".") {
                let field = self.expect_ident()?;
                // Interface-array element access: `arr[i].field`.
                if let SvExpr::Bracket { expr: inner, index } = &expr {
                    if let SvExpr::Ident(arr) = inner.as_ref() {
                        if self.var_iface_array.contains_key(arr) {
                            self.check_iface_array_member(arr, &field)?;
                            expr = SvExpr::Bracket {
                                expr: Box::new(SvExpr::Ident(iface_array_ref(arr, &field))),
                                index: index.clone(),
                            };
                            continue;
                        }
                    }
                }
                let SvExpr::Ident(var) = &expr else {
                    return Err(err(
                        "E_SV_STRUCT",
                        "member access `.field` requires a struct or interface variable",
                    ));
                };
                if self.var_iface.contains_key(var) {
                    self.check_iface_member(var, &field)?;
                    expr = SvExpr::Ident(mangle(var, &field));
                    continue;
                }
                let (msb, lsb) = self.member_field_bits(var, &field)?;
                expr = SvExpr::Slice {
                    expr: Box::new(expr),
                    msb,
                    lsb,
                };
                continue;
            }
            if !self.eat_sym("[") {
                break;
            }
            let first = self.parse_expr()?;
            // Indexed part-select: `a[base +: W]` == `(a >> base)[W-1:0]`,
            // `a[base -: W]` == `(a >> (base - (W-1)))[W-1:0]` (W constant, base
            // may be symbolic — the shift carries it; the slice is fixed-width).
            if self.eat_sym("+:") {
                let w = self.expect_width_const()?;
                self.expect_sym("]")?;
                expr = SvExpr::Slice {
                    expr: Box::new(shr(expr, first)),
                    msb: w - 1,
                    lsb: 0,
                };
                continue;
            }
            if self.eat_sym("-:") {
                let w = self.expect_width_const()?;
                self.expect_sym("]")?;
                expr = SvExpr::Slice {
                    expr: Box::new(shr(expr, sub_const(first, w - 1))),
                    msb: w - 1,
                    lsb: 0,
                };
                continue;
            }
            if self.eat_sym(":") {
                let msb = const_eval(&first, &self.consts)?;
                let lsb = self.expect_width_const()?;
                self.expect_sym("]")?;
                expr = SvExpr::Slice {
                    expr: Box::new(expr),
                    msb: Width::try_from(msb)
                        .map_err(|_| err("E_SV_SLICE", "part-select msb exceeds width range"))?,
                    lsb,
                };
            } else {
                self.expect_sym("]")?;
                // `x[i]` on a packed multi-dim vector selects element `i`:
                // `(x >> (i*ELEM))[ELEM-1:0]`.
                if let SvExpr::Ident(var) = &expr {
                    if let Some(&elem) = self.packed_elem_width.get(var) {
                        let base = mul_const(first, elem);
                        expr = SvExpr::Slice {
                            expr: Box::new(shr(expr, base)),
                            msb: elem - 1,
                            lsb: 0,
                        };
                        continue;
                    }
                }
                expr = SvExpr::Bracket {
                    expr: Box::new(expr),
                    index: Box::new(first),
                };
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<SvExpr, ErrorReport> {
        match self.next().clone() {
            Token::Ident(name) => {
                // `pkg::name` — package-scoped reference. Package members are
                // flattened into the global tables, so the qualifier is dropped.
                let name = if self.eat_sym("::") {
                    self.expect_ident()?
                } else {
                    name
                };
                // Package constants (localparams + enum variants) fold to literals
                // here so they survive to lowering, like module-level enum consts.
                if let Some(value) = self.package_consts.get(&name) {
                    let width = (128 - value.leading_zeros()).max(32) as Width;
                    return Ok(SvExpr::Lit {
                        value: *value,
                        width,
                        signed: false,
                    });
                }
                // `name(args)` is a user-function call; `name` alone is an identifier.
                if self.eat_sym("(") {
                    let mut args = Vec::new();
                    if !self.eat_sym(")") {
                        loop {
                            args.push(self.parse_expr()?);
                            if self.eat_sym(")") {
                                break;
                            }
                            self.expect_sym(",")?;
                        }
                    }
                    Ok(SvExpr::Call { name, args })
                } else {
                    Ok(SvExpr::Ident(name))
                }
            }
            Token::Num(raw) => parse_number(&raw),
            Token::Str(s) => {
                // A string literal is a bit-vector: 8 bits per character, MSB-first.
                let bytes = s.as_bytes();
                let mut value: u128 = 0;
                for &b in bytes.iter().rev().take(16).rev() {
                    value = (value << 8) | b as u128;
                }
                let width = ((bytes.len() * 8).clamp(1, 128)) as Width;
                Ok(SvExpr::Lit {
                    value,
                    width,
                    signed: false,
                })
            }
            Token::Sym(sym) if sym == "'" => {
                // Unsized fill literals `'0` / `'1` / `'x` / `'z`. Width is context-
                // determined; coercion truncates to the target. `'1` is all-ones,
                // so use a max-width value that masks down to all-ones anywhere.
                let fill = self.next().clone();
                let all_ones = matches!(&fill, Token::Num(n) if n == "1");
                let (value, width) = if all_ones {
                    (u128::MAX, 128)
                } else {
                    (0, 1)
                };
                Ok(SvExpr::Lit {
                    value,
                    width,
                    signed: false,
                })
            }
            Token::Sym(sym) if sym == "$" => {
                let func = self.expect_ident()?;
                // `$clog2(x)` is a compile-time constant (ceil(log2 x), x≥1 → 0).
                if func == "clog2" {
                    self.expect_sym("(")?;
                    let arg = self.parse_expr()?;
                    self.expect_sym(")")?;
                    let x = const_eval(&arg, &self.consts)?;
                    let value = if x <= 1 {
                        0
                    } else {
                        (128 - (x - 1).leading_zeros()) as u128
                    };
                    return Ok(SvExpr::Lit {
                        value,
                        width: 32,
                        signed: false,
                    });
                }
                let signed = match func.as_str() {
                    "signed" => true,
                    "unsigned" => false,
                    other => {
                        return Err(err(
                            "E_SV_SYSTEM_FUNC",
                            format!("unsupported SystemVerilog function `${other}`"),
                        ));
                    }
                };
                self.expect_sym("(")?;
                let expr = self.parse_expr()?;
                self.expect_sym(")")?;
                Ok(SvExpr::Cast {
                    signed,
                    expr: Box::new(expr),
                })
            }
            Token::Sym(sym) if sym == "(" => {
                let expr = self.parse_expr()?;
                self.expect_sym(")")?;
                Ok(expr)
            }
            Token::Sym(sym) if sym == "{" => {
                let first = self.parse_expr()?;
                if self.eat_sym("{") {
                    let count = const_eval(&first, &self.consts)?;
                    let count = Width::try_from(count)
                        .map_err(|_| err("E_SV_REPEAT", "replication count exceeds width range"))?;
                    let expr = self.parse_expr()?;
                    self.expect_sym("}")?;
                    self.expect_sym("}")?;
                    return Ok(SvExpr::Repeat {
                        count,
                        expr: Box::new(expr),
                    });
                }
                let mut parts = vec![first];
                loop {
                    if self.eat_sym("}") {
                        break;
                    }
                    self.expect_sym(",")?;
                    parts.push(self.parse_expr()?);
                }
                Ok(SvExpr::Concat(parts))
            }
            token => Err(err(
                "E_SV_EXPR",
                format!(
                    "expected expression, found `{}`  [near: {}]",
                    token.display(),
                    self.context()
                ),
            )),
        }
    }

    fn parse_optional_range(&mut self) -> Result<Option<Width>, ErrorReport> {
        if !self.eat_sym("[") {
            return Ok(None);
        }
        let msb = self.expect_width_const()?;
        self.expect_sym(":")?;
        let lsb = self.expect_width_const()?;
        self.expect_sym("]")?;
        if msb < lsb {
            return Err(err("E_SV_RANGE", "packed range msb must be >= lsb"));
        }
        Ok(Some(msb - lsb + 1))
    }

    /// Parse zero or more packed dimensions `[a:b][c:d]…`. Returns the total flat
    /// width (product of dim sizes, 1 if none) and, for a multi-dimensional packed
    /// vector, the element bit-width (product of all but the outermost dim) so
    /// `x[i]` can be sliced out.
    fn parse_packed_dims(&mut self) -> Result<(Width, Option<Width>), ErrorReport> {
        let mut dims = Vec::new();
        while let Some(width) = self.parse_optional_range()? {
            dims.push(width);
        }
        match dims.as_slice() {
            [] => Ok((1, None)),
            [w] => Ok((*w, None)),
            _ => {
                let total: Width = dims.iter().product();
                let elem: Width = dims[1..].iter().product();
                Ok((total, Some(elem)))
            }
        }
    }

    fn starts_decl(&self) -> bool {
        self.peek_ident_value("input")
            || self.peek_ident_value("output")
            || self.peek_ident_value("inout")
            || self.peek_ident_value("logic")
            || self.peek_ident_value("wire")
            || self.peek_ident_value("reg")
            || self.peek_ident_value("integer")
            || self.peek_ident_value("int")
    }

    fn peek_typedef_type(&self) -> bool {
        // Plain `type_t …` or package-scoped `pkg::type_t …`.
        if let Token::Ident(value) = self.peek() {
            if self.types.contains_key(value) {
                return true;
            }
        }
        if self.peek_next_sym("::") {
            if let Some(Token::Ident(value)) = self.tokens.get(self.pos + 2) {
                return self.types.contains_key(value);
            }
        }
        false
    }

    fn peek_binary_op(&self) -> Option<(SvBinaryOp, u8, u8)> {
        let sym = match self.peek() {
            Token::Sym(sym) => sym.as_str(),
            _ => return None,
        };
        let (op, bp) = match sym {
            "||" => (SvBinaryOp::LogOr, 1),
            "&&" => (SvBinaryOp::LogAnd, 2),
            "|" => (SvBinaryOp::Or, 3),
            "^" => (SvBinaryOp::Xor, 4),
            "&" => (SvBinaryOp::And, 5),
            "==" => (SvBinaryOp::Eq, 6),
            "!=" => (SvBinaryOp::Ne, 6),
            "<" => (SvBinaryOp::Lt, 7),
            "<=" => (SvBinaryOp::Le, 7),
            ">" => (SvBinaryOp::Gt, 7),
            ">=" => (SvBinaryOp::Ge, 7),
            // Shifts bind looser than additive, tighter than relational.
            "<<" => (SvBinaryOp::Shl, 8),
            ">>" => (SvBinaryOp::Shr, 8),
            ">>>" => (SvBinaryOp::Ashr, 8),
            "+" => (SvBinaryOp::Add, 9),
            "-" => (SvBinaryOp::Sub, 9),
            "*" => (SvBinaryOp::Mul, 10),
            "/" => (SvBinaryOp::Div, 10),
            "%" => (SvBinaryOp::Mod, 10),
            // Exponentiation binds tighter than multiplicative.
            "**" => (SvBinaryOp::Pow, 11),
            _ => return None,
        };
        Some((op, bp, bp + 1))
    }

    fn expect_ident(&mut self) -> Result<String, ErrorReport> {
        match self.next().clone() {
            Token::Ident(value) => Ok(value),
            token => Err(err(
                "E_SV_PARSE",
                format!(
                    "expected identifier, found `{}`  [near: {}]",
                    token.display(),
                    self.context()
                ),
            )),
        }
    }

    fn expect_string(&mut self) -> Result<String, ErrorReport> {
        match self.next().clone() {
            Token::Str(value) => Ok(value),
            token => Err(err(
                "E_SV_PARSE",
                format!("expected string literal, found `{}`", token.display()),
            )),
        }
    }

    fn expect_ident_value(&mut self, expected: &str) -> Result<(), ErrorReport> {
        let actual = self.expect_ident()?;
        if actual == expected {
            Ok(())
        } else {
            Err(err(
                "E_SV_PARSE",
                format!("expected `{expected}`, found `{actual}`"),
            ))
        }
    }

    fn eat_ident_value(&mut self, expected: &str) -> bool {
        if self.peek_ident_value(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn peek_ident_value(&self, expected: &str) -> bool {
        matches!(self.peek(), Token::Ident(value) if value == expected)
    }

    fn expect_sym(&mut self, expected: &str) -> Result<(), ErrorReport> {
        if self.eat_sym(expected) {
            Ok(())
        } else {
            Err(err(
                "E_SV_PARSE",
                format!(
                    "expected `{expected}`, found `{}`  [near: {}]",
                    self.peek().display(),
                    self.context()
                ),
            ))
        }
    }

    /// Consume the current (opening) keyword and skip to just past `end`.
    fn skip_to_keyword(&mut self, end: &str) -> Result<(), ErrorReport> {
        self.pos += 1;
        while !self.eat_ident_value(end) {
            if self.is_eof() {
                return Err(err("E_SV_PARSE", format!("unterminated block, expected `{end}`")));
            }
            self.pos += 1;
        }
        Ok(())
    }

    /// The next few tokens, for error context.
    fn context(&self) -> String {
        self.tokens
            .iter()
            .skip(self.pos.saturating_sub(4))
            .take(10)
            .map(|t| t.display())
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn eat_sym(&mut self, expected: &str) -> bool {
        if matches!(self.peek(), Token::Sym(value) if value == expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn peek_sym(&self, expected: &str) -> bool {
        matches!(self.peek(), Token::Sym(value) if value == expected)
    }

    fn peek_next_sym(&self, expected: &str) -> bool {
        matches!(self.tokens.get(self.pos + 1), Some(Token::Sym(value)) if value == expected)
    }

    fn expect_width_const(&mut self) -> Result<Width, ErrorReport> {
        let value = self.expect_usize_const()?;
        Width::try_from(value).map_err(|_| err("E_SV_CONST", "constant exceeds width range"))
    }

    fn expect_usize_const(&mut self) -> Result<usize, ErrorReport> {
        let expr = self.parse_expr()?;
        let value = const_eval(&expr, &self.consts)?;
        usize::try_from(value).map_err(|_| err("E_SV_CONST", "constant exceeds usize range"))
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn next(&mut self) -> &Token {
        let token = self.tokens.get(self.pos).unwrap_or(&Token::Eof);
        self.pos += 1;
        token
    }

    fn is_eof(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }
}

impl Token {
    fn display(&self) -> String {
        match self {
            Token::Ident(value) | Token::Num(value) | Token::Sym(value) => value.clone(),
            Token::Str(value) => format!("\"{value}\""),
            Token::Eof => "<eof>".to_string(),
        }
    }
}

/// A 32-bit unsigned literal `SvExpr`.
fn lit_width(k: Width) -> SvExpr {
    SvExpr::Lit {
        value: k as u128,
        width: 32,
        signed: false,
    }
}

/// `e + k`, `e - k`, `e >> amt` as `SvExpr`s (for desugaring `+:`/`-:`).
fn add_const(e: SvExpr, k: Width) -> SvExpr {
    SvExpr::Binary {
        op: SvBinaryOp::Add,
        lhs: Box::new(e),
        rhs: Box::new(lit_width(k)),
    }
}
fn sub_const(e: SvExpr, k: Width) -> SvExpr {
    SvExpr::Binary {
        op: SvBinaryOp::Sub,
        lhs: Box::new(e),
        rhs: Box::new(lit_width(k)),
    }
}
fn shr(e: SvExpr, amt: SvExpr) -> SvExpr {
    SvExpr::Binary {
        op: SvBinaryOp::Shr,
        lhs: Box::new(e),
        rhs: Box::new(amt),
    }
}
fn mul_const(e: SvExpr, k: Width) -> SvExpr {
    SvExpr::Binary {
        op: SvBinaryOp::Mul,
        lhs: Box::new(e),
        rhs: Box::new(lit_width(k)),
    }
}

fn parse_number(raw: &str) -> Result<SvExpr, ErrorReport> {
    let (value, width, signed) = parse_number_parts(raw)?;
    Ok(SvExpr::Lit {
        value,
        width,
        signed,
    })
}

fn parse_number_parts(raw: &str) -> Result<(u128, Width, bool), ErrorReport> {
    let raw: String = raw.chars().filter(|c| *c != '_' && !c.is_whitespace()).collect();
    if let Some((width_raw, rest)) = raw.split_once('\'') {
        // unsized literal (`'b0`, `'hff`) defaults to 32 bits
        let width = if width_raw.is_empty() {
            32
        } else {
            width_raw
                .parse::<Width>()
                .map_err(|_| err("E_SV_CONST", format!("invalid literal width `{width_raw}`")))?
        };
        let mut chars = rest.chars();
        let mut signed = false;
        let base_char = match chars.next() {
            Some('s') | Some('S') => {
                signed = true;
                chars
                    .next()
                    .ok_or_else(|| err("E_SV_CONST", "missing literal base"))?
            }
            Some(ch) => ch,
            None => return Err(err("E_SV_CONST", "missing literal base")),
        };
        // 2-state interpretation: x/z/? unknown bits map to 0 (as Verilator's
        // --x-assign 0 / --x-initial 0).
        let digits = chars
            .collect::<String>()
            .replace(['x', 'X', 'z', 'Z', '?'], "0");
        let radix = match base_char {
            'b' | 'B' => 2,
            'd' | 'D' => 10,
            'h' | 'H' => 16,
            'o' | 'O' => 8,
            _ => return Err(err("E_SV_CONST", format!("unsupported base `{base_char}`"))),
        };
        let value = if signed && digits.starts_with('-') {
            if radix != 10 {
                return Err(err(
                    "E_SV_CONST",
                    format!("negative literal `{raw}` must use decimal base"),
                ));
            }
            digits
                .parse::<i128>()
                .map(|value| value as u128)
                .map_err(|_| err("E_SV_CONST", format!("invalid literal `{raw}`")))?
        } else {
            u128::from_str_radix(&digits, radix)
                .map_err(|_| err("E_SV_CONST", format!("invalid literal `{raw}`")))?
        };
        Ok((value, width, signed))
    } else {
        let value = raw
            .parse::<u128>()
            .map_err(|_| err("E_SV_CONST", format!("invalid number `{raw}`")))?;
        Ok((value, 32, false))
    }
}

fn has_wildcard_digit(raw: &str) -> bool {
    raw.chars()
        .any(|ch| matches!(ch, 'x' | 'X' | 'z' | 'Z' | '?'))
}

fn parse_wildcard_case_label(raw: &str, kind: SvCaseKind) -> Result<SvCaseLabel, ErrorReport> {
    if kind == SvCaseKind::Normal {
        return Err(err(
            "E_SV_CASE",
            format!("wildcard literal `{raw}` is only supported in casez/casex"),
        ));
    }
    let raw = raw.replace('_', "");
    let Some((width_raw, rest)) = raw.split_once('\'') else {
        return Err(err(
            "E_SV_CASE",
            format!("wildcard case label `{raw}` must be a based literal"),
        ));
    };
    let width = width_raw
        .parse::<Width>()
        .map_err(|_| err("E_SV_CONST", format!("invalid literal width `{width_raw}`")))?;
    let mut chars = rest.chars();
    let signed_prefix = matches!(chars.clone().next(), Some('s' | 'S'));
    if signed_prefix {
        chars.next();
    }
    let base_char = chars
        .next()
        .ok_or_else(|| err("E_SV_CONST", "missing literal base"))?;
    let bits_per_digit = match base_char {
        'b' | 'B' => 1,
        'o' | 'O' => 3,
        'h' | 'H' => 4,
        'd' | 'D' => {
            return Err(err(
                "E_SV_CASE",
                "wildcard case labels must use binary, octal, or hex base",
            ));
        }
        _ => return Err(err("E_SV_CONST", format!("unsupported base `{base_char}`"))),
    };
    let digits = chars.collect::<Vec<_>>();
    let mut value = 0u128;
    let mut mask = 0u128;
    for digit in &digits {
        let (digit_value, digit_mask) = wildcard_digit_value_mask(*digit, bits_per_digit, kind)?;
        value = (value << bits_per_digit) | digit_value;
        mask = (mask << bits_per_digit) | digit_mask;
    }
    let raw_bit_width = digits.len() as Width * bits_per_digit;
    let value = resize_u128_to_width(value, raw_bit_width, width);
    let mask = resize_u128_to_width(mask, raw_bit_width, width);
    Ok(SvCaseLabel::Wildcard { value, mask, width })
}

fn wildcard_digit_value_mask(
    digit: char,
    bits_per_digit: Width,
    kind: SvCaseKind,
) -> Result<(u128, u128), ErrorReport> {
    let full_mask = if bits_per_digit >= 128 {
        u128::MAX
    } else {
        (1u128 << bits_per_digit) - 1
    };
    match digit {
        '?' => Ok((0, 0)),
        'z' | 'Z' => Ok((0, 0)),
        'x' | 'X' if kind == SvCaseKind::CaseX => Ok((0, 0)),
        'x' | 'X' => Err(err(
            "E_SV_CASE",
            "casez wildcard labels do not treat x bits as don't-care",
        )),
        _ => {
            let radix = match bits_per_digit {
                1 => 2,
                3 => 8,
                4 => 16,
                _ => unreachable!("unsupported wildcard literal radix"),
            };
            let value = digit.to_digit(radix).ok_or_else(|| {
                err(
                    "E_SV_CONST",
                    format!("invalid wildcard case digit `{digit}`"),
                )
            })?;
            Ok((u128::from(value), full_mask))
        }
    }
}

fn is_ident_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_'
}

fn is_ident_continue(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_' || ch == '$'
}
