use std::collections::{HashMap, HashSet};

use rrtl_core::{
    as_sint, as_uint, compile, concat, lit_u, mem_read, mux, sint, uint, BitType, Design, Expr,
    Signal,
};
use rrtl_ir::{Diagnostic, ErrorReport, Signedness, Width};
use serde::{Deserialize, Serialize};

const SV_FOR_UNROLL_LIMIT: usize = 4096;

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
    pub memory_depth: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SvInitialAssign {
    pub dst: SvLvalue,
    pub expr: SvExpr,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvUnaryOp {
    Not,
    BitNot,
    Neg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvBinaryOp {
    Add,
    Sub,
    Mul,
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
}

pub fn parse_sv(source: &str) -> Result<SvSource, ErrorReport> {
    Parser::new(source)?.parse_source()
}

pub fn import_sv(source: &str, top: Option<&str>) -> Result<SvImport, ErrorReport> {
    let parsed = parse_sv(source)?;
    import_source(specialize_source(source, parsed)?, top)
}

pub fn import_source(source: SvSource, top: Option<&str>) -> Result<SvImport, ErrorReport> {
    if source.modules.is_empty() {
        return Err(err(
            "E_SV_NO_MODULES",
            "SystemVerilog source has no modules",
        ));
    }
    let top_name = select_top(&source, top)?;
    reject_defined_param_overrides_without_source(&source)?;
    let mut design = Design::new();
    let defined = source
        .modules
        .iter()
        .map(|module| module.name.clone())
        .collect::<HashSet<_>>();
    let port_maps = module_port_maps(&source.modules);
    let mut externs = HashMap::<String, HashMap<String, (BitType, SvDirection)>>::new();

    for module in &source.modules {
        lower_module(module, &mut design, &defined, &port_maps, &mut externs)?;
    }
    for (module_name, ports) in externs {
        if defined.contains(&module_name) {
            continue;
        }
        let mut ext = design.extern_module(module_name);
        let mut ports = ports.into_iter().collect::<Vec<_>>();
        ports.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
        for (port, (ty, dir)) in ports {
            match dir {
                SvDirection::Input => {
                    ext.input(port, ty);
                }
                SvDirection::Output => {
                    ext.output(port, ty);
                }
                SvDirection::Inout => {
                    ext.inout(port, ty);
                }
            }
        }
    }
    compile(&design).map_err(|report| report)?;
    Ok(SvImport {
        modules: source
            .modules
            .into_iter()
            .map(|module| module.name)
            .collect(),
        design,
        top_name,
    })
}

fn reject_defined_param_overrides_without_source(source: &SvSource) -> Result<(), ErrorReport> {
    let defined = source
        .modules
        .iter()
        .map(|module| module.name.clone())
        .collect::<HashSet<_>>();
    for module in &source.modules {
        for item in &module.items {
            let SvItem::Instance(instance) = item else {
                continue;
            };
            if !instance.params.is_empty() && defined.contains(&instance.module) {
                return Err(err(
                    "E_SV_PARAM_OVERRIDE",
                    format!(
                        "instance `{}` of defined module `{}` has parameter overrides, but import_source cannot recompute parsed widths; use import_sv",
                        instance.name, instance.module
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn specialize_source(source_text: &str, source: SvSource) -> Result<SvSource, ErrorReport> {
    let original_modules = source
        .modules
        .iter()
        .map(|module| module.name.clone())
        .collect::<HashSet<_>>();
    let base_param_names = source
        .modules
        .iter()
        .map(|module| {
            (
                module.name.clone(),
                module
                    .params
                    .iter()
                    .map(|param| param.name.clone())
                    .collect::<HashSet<_>>(),
            )
        })
        .collect::<HashMap<_, _>>();
    let base_param_orders = source
        .modules
        .iter()
        .map(|module| {
            (
                module.name.clone(),
                module
                    .params
                    .iter()
                    .map(|param| param.name.clone())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut occupied_names = original_modules.clone();
    let mut specializations = HashMap::<(String, Vec<(String, u128)>), String>::new();
    let mut modules = source.modules;
    let mut index = 0usize;

    while index < modules.len() {
        let consts = module_consts(&modules[index])?;
        let mut updates = Vec::<(usize, String)>::new();
        let mut new_modules = Vec::<SvModule>::new();

        for (item_index, item) in modules[index].items.iter().enumerate() {
            let SvItem::Instance(instance) = item else {
                continue;
            };
            if instance.params.is_empty() || !original_modules.contains(&instance.module) {
                continue;
            }
            let values = eval_param_overrides(instance, &consts, &base_param_orders)?;
            validate_param_override_names(instance, &values, &base_param_names)?;
            let key = (instance.module.clone(), values.clone());
            let specialized_name = if let Some(name) = specializations.get(&key) {
                name.clone()
            } else {
                let name =
                    fresh_specialized_module_name(&instance.module, &values, &mut occupied_names);
                let specialized = parse_specialized_module(
                    source_text,
                    &instance.module,
                    &name,
                    values.iter().cloned().collect(),
                )?;
                specializations.insert(key, name.clone());
                new_modules.push(specialized);
                name
            };
            updates.push((item_index, specialized_name));
        }

        for (item_index, specialized_name) in updates {
            if let SvItem::Instance(instance) = &mut modules[index].items[item_index] {
                instance.module = specialized_name;
                instance.params.clear();
            }
        }
        modules.extend(new_modules);
        index += 1;
    }

    Ok(SvSource { modules })
}

fn module_consts(module: &SvModule) -> Result<HashMap<String, u128>, ErrorReport> {
    let mut consts = HashMap::new();
    for param in &module.params {
        let value = const_eval(&param.value, &consts)?;
        consts.insert(param.name.clone(), value);
    }
    for item in &module.items {
        if let SvItem::Param(param) = item {
            let value = const_eval(&param.value, &consts)?;
            consts.insert(param.name.clone(), value);
        }
    }
    Ok(consts)
}

fn eval_param_overrides(
    instance: &SvInstance,
    consts: &HashMap<String, u128>,
    base_param_orders: &HashMap<String, Vec<String>>,
) -> Result<Vec<(String, u128)>, ErrorReport> {
    let mut seen = HashSet::new();
    let mut values = Vec::with_capacity(instance.params.len());
    let has_named = instance
        .params
        .iter()
        .any(|param| matches!(param, SvParamOverride::Named { .. }));
    let has_positional = instance
        .params
        .iter()
        .any(|param| matches!(param, SvParamOverride::Positional { .. }));
    if has_named && has_positional {
        return Err(err(
            "E_SV_PARAM_OVERRIDE",
            format!(
                "instance `{}` mixes named and positional parameter overrides",
                instance.name
            ),
        ));
    }
    if has_positional {
        let Some(order) = base_param_orders.get(&instance.module) else {
            return Err(err(
                "E_SV_PARAM_OVERRIDE",
                format!(
                    "instance `{}` has positional parameter overrides, but module `{}` parameter order is unknown",
                    instance.name, instance.module
                ),
            ));
        };
        if instance.params.len() > order.len() {
            return Err(err(
                "E_SV_PARAM_OVERRIDE",
                format!(
                    "instance `{}` provides {} positional parameter overrides for module `{}` with {} parameters",
                    instance.name,
                    instance.params.len(),
                    instance.module,
                    order.len()
                ),
            ));
        }
        for (index, param) in instance.params.iter().enumerate() {
            let SvParamOverride::Positional { value } = param else {
                unreachable!("mixed overrides were rejected");
            };
            values.push((order[index].clone(), const_eval(value, consts)?));
        }
    } else {
        for param in &instance.params {
            let SvParamOverride::Named { name, value } = param else {
                unreachable!("positional overrides were handled above");
            };
            if !seen.insert(name.clone()) {
                return Err(err(
                    "E_SV_PARAM_OVERRIDE",
                    format!(
                        "instance `{}` overrides parameter `{}` more than once",
                        instance.name, name
                    ),
                ));
            }
            values.push((name.clone(), const_eval(value, consts)?));
        }
    }
    values.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
    Ok(values)
}

fn validate_param_override_names(
    instance: &SvInstance,
    values: &[(String, u128)],
    base_param_names: &HashMap<String, HashSet<String>>,
) -> Result<(), ErrorReport> {
    let Some(names) = base_param_names.get(&instance.module) else {
        return Ok(());
    };
    for (name, _) in values {
        if !names.contains(name) {
            return Err(err(
                "E_SV_PARAM_OVERRIDE",
                format!(
                    "instance `{}` overrides unknown parameter `{}` on module `{}`",
                    instance.name, name, instance.module
                ),
            ));
        }
    }
    Ok(())
}

fn parse_specialized_module(
    source_text: &str,
    base_name: &str,
    specialized_name: &str,
    values: HashMap<String, u128>,
) -> Result<SvModule, ErrorReport> {
    let mut overrides = HashMap::new();
    overrides.insert(base_name.to_string(), values);
    let parsed = Parser::new_with_module_param_overrides(source_text, overrides)?.parse_source()?;
    let mut module = parsed
        .modules
        .into_iter()
        .find(|module| module.name == base_name)
        .ok_or_else(|| {
            err(
                "E_SV_PARAM_OVERRIDE",
                format!("cannot find module `{base_name}` to specialize"),
            )
        })?;
    module.name = specialized_name.to_string();
    Ok(module)
}

fn fresh_specialized_module_name(
    base: &str,
    values: &[(String, u128)],
    occupied: &mut HashSet<String>,
) -> String {
    let suffix = values
        .iter()
        .map(|(name, value)| format!("{}_{}", sanitize_sv_name_part(name), value))
        .collect::<Vec<_>>()
        .join("__");
    let first = if suffix.is_empty() {
        format!("{base}__param")
    } else {
        format!("{base}__{suffix}")
    };
    if occupied.insert(first.clone()) {
        return first;
    }
    let mut index = 1usize;
    loop {
        let candidate = format!("{first}_{index}");
        if occupied.insert(candidate.clone()) {
            return candidate;
        }
        index += 1;
    }
}

fn sanitize_sv_name_part(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn lower_module(
    module: &SvModule,
    design: &mut Design,
    defined: &HashSet<String>,
    port_maps: &HashMap<String, HashMap<String, (BitType, SvDirection)>>,
    externs: &mut HashMap<String, HashMap<String, (BitType, SvDirection)>>,
) -> Result<(), ErrorReport> {
    let mut builder = design.module(&module.name);
    let mut symbols = HashMap::<String, SvSymbol>::new();
    let mut consts = HashMap::<String, u128>::new();
    for param in &module.params {
        let value = const_eval(&param.value, &consts)?;
        consts.insert(param.name.clone(), value);
    }
    let items = expand_generate_items(&module.items, &consts)?;
    let ff_targets = always_ff_signal_targets(&items);
    let mut occupied_names = module_signal_names(&items);
    for item in &items {
        match item {
            SvItem::Param(param) => {
                let value = const_eval(&param.value, &consts)?;
                consts.insert(param.name.clone(), value);
            }
            SvItem::TypeDef(_) => {}
            SvItem::Genvar(_) => {}
            SvItem::Generate(_) | SvItem::GenerateFor { .. } => {}
            SvItem::Initial(_) => {}
            SvItem::Decl(decl) => {
                for declarator in &decl.names {
                    let ty = sv_type_to_rrtl(decl.ty);
                    let signal = match (decl.direction, declarator.memory_depth) {
                        (_, Some(depth)) => {
                            let addr_width = ceil_log2(depth);
                            builder.mem(&declarator.name, addr_width, ty, depth)
                        }
                        (Some(SvDirection::Input), None) => builder.input(&declarator.name, ty),
                        (Some(SvDirection::Output), None)
                            if ff_targets.contains(&declarator.name) =>
                        {
                            let output = builder.output(&declarator.name, ty);
                            let reg_name = fresh_sv_reg_name(&declarator.name, &mut occupied_names);
                            let reg = builder.reg(reg_name, ty);
                            builder.assign(output, reg);
                            reg
                        }
                        (Some(SvDirection::Output), None) => builder.output(&declarator.name, ty),
                        (Some(SvDirection::Inout), None) => builder.inout(&declarator.name, ty),
                        (None, None) => match decl.kind {
                            SvDeclKind::Reg => builder.reg(&declarator.name, ty),
                            SvDeclKind::Logic if ff_targets.contains(&declarator.name) => {
                                builder.reg(&declarator.name, ty)
                            }
                            SvDeclKind::Logic | SvDeclKind::Wire => {
                                builder.wire(&declarator.name, ty)
                            }
                        },
                    };
                    if symbols
                        .insert(
                            declarator.name.clone(),
                            SvSymbol {
                                signal,
                                ty,
                                direction: decl.direction,
                                is_memory: declarator.memory_depth.is_some(),
                            },
                        )
                        .is_some()
                    {
                        return Err(err(
                            "E_SV_DUPLICATE_SIGNAL",
                            format!(
                                "module `{}` declares `{}` more than once",
                                module.name, declarator.name
                            ),
                        ));
                    }
                }
            }
            _ => {}
        }
    }

    for port in &module.ports {
        if !symbols.contains_key(port) {
            return Err(err(
                "E_SV_PORT_DECL",
                format!("module `{}` port `{port}` has no declaration", module.name),
            ));
        }
    }

    let mut continuous_state = CombState::default();
    let mut initial_state = InitialState::default();
    for item in &items {
        match item {
            SvItem::Param(_) => {}
            SvItem::TypeDef(_) => {}
            SvItem::Genvar(_) => {}
            SvItem::Generate(_) | SvItem::GenerateFor { .. } => {}
            SvItem::Decl(_) => {}
            SvItem::Initial(assigns) => {
                for assign in assigns {
                    lower_initial_assign(
                        assign,
                        &symbols,
                        &consts,
                        &mut builder,
                        &mut initial_state,
                    )?;
                }
            }
            SvItem::Assign { dst, expr } => {
                lower_continuous_lvalue_assign(
                    dst,
                    expr,
                    &symbols,
                    &consts,
                    &mut continuous_state,
                )?;
            }
            SvItem::AlwaysComb(stmts) => {
                lower_always_comb(stmts, &symbols, &consts, &mut builder)?;
            }
            SvItem::AlwaysFf(always) => {
                lower_always_ff(always, &symbols, &consts, &mut builder)?;
            }
            SvItem::Instance(instance) => {
                if !defined.contains(&instance.module)
                    && instance
                        .params
                        .iter()
                        .any(|param| matches!(param, SvParamOverride::Positional { .. }))
                {
                    return Err(err(
                        "E_SV_PARAM_OVERRIDE",
                        format!(
                            "instance `{}` of external module `{}` uses positional parameter overrides, but external parameter order is unknown",
                            instance.name, instance.module
                        ),
                    ));
                }
                let mut connections = Vec::with_capacity(instance.connections.len());
                for connection in &instance.connections {
                    let port_info = port_maps
                        .get(&instance.module)
                        .and_then(|ports| ports.get(&connection.port))
                        .copied();
                    let signal = lower_instance_connection(
                        instance,
                        connection,
                        port_info,
                        &symbols,
                        &consts,
                        &mut occupied_names,
                        &mut builder,
                        &mut continuous_state,
                    )?;
                    connections.push((connection.port.clone(), signal));
                    if !defined.contains(&instance.module) {
                        let (ty, dir) = infer_external_port(connection, &symbols, &consts)?;
                        externs
                            .entry(instance.module.clone())
                            .or_default()
                            .entry(connection.port.clone())
                            .or_insert((ty, dir));
                    }
                }
                builder.instance(&instance.name, &instance.module, connections);
            }
        }
    }
    initial_state.emit(&mut builder);
    for name in &continuous_state.order {
        let symbol = symbols
            .get(name)
            .filter(|symbol| !symbol.is_memory)
            .ok_or_else(|| err("E_SV_LVALUE", format!("unknown signal `{name}`")))?;
        let expr = continuous_state.values.get(name).cloned().ok_or_else(|| {
            err(
                "E_SV_ASSIGN",
                format!("missing continuous value for `{name}`"),
            )
        })?;
        builder.assign(
            symbol.signal,
            coerce_expr_to_signal_type(expr, symbol.signal, &symbols)?,
        );
    }
    Ok(())
}

fn expand_generate_items(
    items: &[SvItem],
    consts: &HashMap<String, u128>,
) -> Result<Vec<SvItem>, ErrorReport> {
    expand_generate_items_with_prefix(items, consts, "", &HashMap::new())
}

fn expand_generate_items_with_prefix(
    items: &[SvItem],
    consts: &HashMap<String, u128>,
    prefix: &str,
    renames: &HashMap<String, String>,
) -> Result<Vec<SvItem>, ErrorReport> {
    let mut scope_renames = renames.clone();
    if !prefix.is_empty() {
        collect_generated_decl_renames(items, prefix, &mut scope_renames)?;
    }
    let mut expanded = Vec::new();
    for item in items {
        match item {
            SvItem::Generate(items) => {
                expanded.extend(expand_generate_items_with_prefix(
                    items,
                    consts,
                    prefix,
                    &scope_renames,
                )?);
            }
            SvItem::GenerateFor {
                var,
                init,
                cmp,
                bound,
                step,
                label,
                items,
            } => {
                for iter_consts in for_loop_iteration_consts(var, init, *cmp, bound, step, consts)?
                {
                    let index = iter_consts
                        .get(var)
                        .copied()
                        .ok_or_else(|| err("E_SV_GENERATE", format!("missing genvar `{var}`")))?;
                    let scope = label
                        .as_ref()
                        .map(|label| format!("{}{}__{}__", prefix, label, index))
                        .unwrap_or_else(|| format!("{}__gen_{}__{}__", prefix, var, index));
                    expanded.extend(expand_generate_items_with_prefix(
                        items,
                        &iter_consts,
                        &scope,
                        &scope_renames,
                    )?);
                }
            }
            SvItem::Decl(decl) if !prefix.is_empty() => {
                expanded.push(SvItem::Decl(rename_generated_decl(decl, &scope_renames)?));
            }
            SvItem::TypeDef(_) | SvItem::Param(_) if !prefix.is_empty() => {
                return Err(err(
                    "E_SV_GENERATE",
                    "generated typedefs and parameters are not supported",
                ));
            }
            SvItem::Instance(instance) => {
                let mut instance = substitute_instance(instance, consts, &scope_renames)?;
                if !prefix.is_empty() {
                    instance.name = format!("{prefix}{}", instance.name);
                }
                expanded.push(SvItem::Instance(instance));
            }
            _ => expanded.push(substitute_item(item, consts, &scope_renames)?),
        }
    }
    Ok(expanded)
}

fn collect_generated_decl_renames(
    items: &[SvItem],
    prefix: &str,
    renames: &mut HashMap<String, String>,
) -> Result<(), ErrorReport> {
    for item in items {
        match item {
            SvItem::Decl(decl) => {
                for declarator in &decl.names {
                    if declarator.memory_depth.is_some() {
                        return Err(err(
                            "E_SV_GENERATE",
                            "generated memory declarations are not supported",
                        ));
                    }
                    renames.insert(
                        declarator.name.clone(),
                        format!("{prefix}{}", declarator.name),
                    );
                }
            }
            SvItem::Param(_) | SvItem::TypeDef(_) => {
                return Err(err(
                    "E_SV_GENERATE",
                    "generated typedefs and parameters are not supported",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn rename_generated_decl(
    decl: &SvDecl,
    renames: &HashMap<String, String>,
) -> Result<SvDecl, ErrorReport> {
    let mut decl = decl.clone();
    for declarator in &mut decl.names {
        if declarator.memory_depth.is_some() {
            return Err(err(
                "E_SV_GENERATE",
                "generated memory declarations are not supported",
            ));
        }
        declarator.name = renames
            .get(&declarator.name)
            .cloned()
            .unwrap_or_else(|| declarator.name.clone());
    }
    Ok(decl)
}

fn substitute_item(
    item: &SvItem,
    consts: &HashMap<String, u128>,
    renames: &HashMap<String, String>,
) -> Result<SvItem, ErrorReport> {
    Ok(match item {
        SvItem::Param(param) => SvItem::Param(SvParam {
            name: param.name.clone(),
            value: substitute_expr(&param.value, consts, renames)?,
        }),
        SvItem::TypeDef(ty) => SvItem::TypeDef(ty.clone()),
        SvItem::Decl(decl) => SvItem::Decl(decl.clone()),
        SvItem::Genvar(names) => SvItem::Genvar(names.clone()),
        SvItem::Generate(items) => SvItem::Generate(items.clone()),
        SvItem::GenerateFor { .. } => item.clone(),
        SvItem::Assign { dst, expr } => SvItem::Assign {
            dst: substitute_lvalue(dst, consts, renames)?,
            expr: substitute_expr(expr, consts, renames)?,
        },
        SvItem::Initial(assigns) => SvItem::Initial(
            assigns
                .iter()
                .map(|assign| {
                    Ok(SvInitialAssign {
                        dst: substitute_lvalue(&assign.dst, consts, renames)?,
                        expr: substitute_expr(&assign.expr, consts, renames)?,
                    })
                })
                .collect::<Result<Vec<_>, ErrorReport>>()?,
        ),
        SvItem::AlwaysComb(stmts) => SvItem::AlwaysComb(substitute_stmts(stmts, consts, renames)?),
        SvItem::AlwaysFf(always) => SvItem::AlwaysFf(SvAlwaysFf {
            clock: rename_ident(&always.clock, renames),
            async_reset: always.async_reset.as_ref().map(|reset| SvResetEdge {
                signal: rename_ident(&reset.signal, renames),
                active_low: reset.active_low,
            }),
            body: substitute_stmts(&always.body, consts, renames)?,
        }),
        SvItem::Instance(instance) => {
            SvItem::Instance(substitute_instance(instance, consts, renames)?)
        }
    })
}

fn substitute_instance(
    instance: &SvInstance,
    consts: &HashMap<String, u128>,
    renames: &HashMap<String, String>,
) -> Result<SvInstance, ErrorReport> {
    Ok(SvInstance {
        module: instance.module.clone(),
        name: instance.name.clone(),
        params: instance
            .params
            .iter()
            .map(|param| {
                Ok(match param {
                    SvParamOverride::Named { name, value } => SvParamOverride::Named {
                        name: name.clone(),
                        value: substitute_expr(value, consts, renames)?,
                    },
                    SvParamOverride::Positional { value } => SvParamOverride::Positional {
                        value: substitute_expr(value, consts, renames)?,
                    },
                })
            })
            .collect::<Result<Vec<_>, ErrorReport>>()?,
        connections: instance
            .connections
            .iter()
            .map(|connection| {
                Ok(SvConnection {
                    port: connection.port.clone(),
                    expr: substitute_expr(&connection.expr, consts, renames)?,
                })
            })
            .collect::<Result<Vec<_>, ErrorReport>>()?,
    })
}

fn substitute_stmts(
    stmts: &[SvStmt],
    consts: &HashMap<String, u128>,
    renames: &HashMap<String, String>,
) -> Result<Vec<SvStmt>, ErrorReport> {
    stmts
        .iter()
        .map(|stmt| {
            Ok(match stmt {
                SvStmt::Assign {
                    dst,
                    nonblocking,
                    expr,
                } => SvStmt::Assign {
                    dst: substitute_lvalue(dst, consts, renames)?,
                    nonblocking: *nonblocking,
                    expr: substitute_expr(expr, consts, renames)?,
                },
                SvStmt::If {
                    cond,
                    then_stmts,
                    else_stmts,
                } => SvStmt::If {
                    cond: substitute_expr(cond, consts, renames)?,
                    then_stmts: substitute_stmts(then_stmts, consts, renames)?,
                    else_stmts: substitute_stmts(else_stmts, consts, renames)?,
                },
                SvStmt::For {
                    var,
                    init,
                    cmp,
                    bound,
                    step,
                    body,
                } => SvStmt::For {
                    var: var.clone(),
                    init: substitute_expr(init, consts, renames)?,
                    cmp: *cmp,
                    bound: substitute_expr(bound, consts, renames)?,
                    step: substitute_for_step(step, consts, renames)?,
                    body: substitute_stmts(body, consts, renames)?,
                },
                SvStmt::Case { kind, expr, items } => SvStmt::Case {
                    kind: *kind,
                    expr: substitute_expr(expr, consts, renames)?,
                    items: items
                        .iter()
                        .map(|item| {
                            Ok(SvCaseItem {
                                labels: item
                                    .labels
                                    .iter()
                                    .map(|label| substitute_case_label(label, consts, renames))
                                    .collect::<Result<Vec<_>, ErrorReport>>()?,
                                stmts: substitute_stmts(&item.stmts, consts, renames)?,
                                is_default: item.is_default,
                            })
                        })
                        .collect::<Result<Vec<_>, ErrorReport>>()?,
                },
                SvStmt::Assert {
                    name,
                    cond,
                    message,
                } => SvStmt::Assert {
                    name: name.clone(),
                    cond: substitute_expr(cond, consts, renames)?,
                    message: message.clone(),
                },
                SvStmt::Cover { name, cond } => SvStmt::Cover {
                    name: name.clone(),
                    cond: substitute_expr(cond, consts, renames)?,
                },
            })
        })
        .collect()
}

fn substitute_for_step(
    step: &SvForStep,
    consts: &HashMap<String, u128>,
    renames: &HashMap<String, String>,
) -> Result<SvForStep, ErrorReport> {
    Ok(match step {
        SvForStep::Inc => SvForStep::Inc,
        SvForStep::Dec => SvForStep::Dec,
        SvForStep::Add(expr) => SvForStep::Add(substitute_expr(expr, consts, renames)?),
        SvForStep::Sub(expr) => SvForStep::Sub(substitute_expr(expr, consts, renames)?),
    })
}

fn substitute_case_label(
    label: &SvCaseLabel,
    consts: &HashMap<String, u128>,
    renames: &HashMap<String, String>,
) -> Result<SvCaseLabel, ErrorReport> {
    Ok(match label {
        SvCaseLabel::Expr(expr) => SvCaseLabel::Expr(substitute_expr(expr, consts, renames)?),
        SvCaseLabel::Wildcard { value, mask, width } => SvCaseLabel::Wildcard {
            value: *value,
            mask: *mask,
            width: *width,
        },
    })
}

fn substitute_lvalue(
    dst: &SvLvalue,
    consts: &HashMap<String, u128>,
    renames: &HashMap<String, String>,
) -> Result<SvLvalue, ErrorReport> {
    Ok(match dst {
        SvLvalue::Signal(name) => SvLvalue::Signal(rename_ident(name, renames)),
        SvLvalue::Bit { name, index } => SvLvalue::Bit {
            name: rename_ident(name, renames),
            index: substitute_expr(index, consts, renames)?,
        },
        SvLvalue::Slice { name, msb, lsb } => SvLvalue::Slice {
            name: rename_ident(name, renames),
            msb: substitute_expr(msb, consts, renames)?,
            lsb: substitute_expr(lsb, consts, renames)?,
        },
        SvLvalue::Memory { name, addr } => SvLvalue::Memory {
            name: rename_ident(name, renames),
            addr: substitute_expr(addr, consts, renames)?,
        },
    })
}

fn substitute_expr(
    expr: &SvExpr,
    consts: &HashMap<String, u128>,
    renames: &HashMap<String, String>,
) -> Result<SvExpr, ErrorReport> {
    Ok(match expr {
        SvExpr::Ident(name) => consts
            .get(name)
            .map(|value| SvExpr::Lit {
                value: *value,
                width: 32,
                signed: false,
            })
            .unwrap_or_else(|| SvExpr::Ident(rename_ident(name, renames))),
        SvExpr::Lit {
            value,
            width,
            signed,
        } => SvExpr::Lit {
            value: *value,
            width: *width,
            signed: *signed,
        },
        SvExpr::Unary { op, expr } => SvExpr::Unary {
            op: *op,
            expr: Box::new(substitute_expr(expr, consts, renames)?),
        },
        SvExpr::Binary { op, lhs, rhs } => SvExpr::Binary {
            op: *op,
            lhs: Box::new(substitute_expr(lhs, consts, renames)?),
            rhs: Box::new(substitute_expr(rhs, consts, renames)?),
        },
        SvExpr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => SvExpr::Ternary {
            cond: Box::new(substitute_expr(cond, consts, renames)?),
            then_expr: Box::new(substitute_expr(then_expr, consts, renames)?),
            else_expr: Box::new(substitute_expr(else_expr, consts, renames)?),
        },
        SvExpr::Concat(parts) => SvExpr::Concat(
            parts
                .iter()
                .map(|part| substitute_expr(part, consts, renames))
                .collect::<Result<Vec<_>, ErrorReport>>()?,
        ),
        SvExpr::Repeat { count, expr } => SvExpr::Repeat {
            count: *count,
            expr: Box::new(substitute_expr(expr, consts, renames)?),
        },
        SvExpr::Cast { signed, expr } => SvExpr::Cast {
            signed: *signed,
            expr: Box::new(substitute_expr(expr, consts, renames)?),
        },
        SvExpr::Index { expr, index } => SvExpr::Index {
            expr: Box::new(substitute_expr(expr, consts, renames)?),
            index: *index,
        },
        SvExpr::Slice { expr, msb, lsb } => SvExpr::Slice {
            expr: Box::new(substitute_expr(expr, consts, renames)?),
            msb: *msb,
            lsb: *lsb,
        },
        SvExpr::MemRead { name, addr } => SvExpr::MemRead {
            name: rename_ident(name, renames),
            addr: Box::new(substitute_expr(addr, consts, renames)?),
        },
        SvExpr::Bracket { expr, index } => SvExpr::Bracket {
            expr: Box::new(substitute_expr(expr, consts, renames)?),
            index: Box::new(substitute_expr(index, consts, renames)?),
        },
    })
}

fn rename_ident(name: &str, renames: &HashMap<String, String>) -> String {
    renames
        .get(name)
        .cloned()
        .unwrap_or_else(|| name.to_string())
}

fn module_port_maps(
    modules: &[SvModule],
) -> HashMap<String, HashMap<String, (BitType, SvDirection)>> {
    let mut maps = HashMap::new();
    for module in modules {
        let declared_ports = module.ports.iter().cloned().collect::<HashSet<_>>();
        let mut ports = HashMap::new();
        for item in &module.items {
            let SvItem::Decl(decl) = item else {
                continue;
            };
            let Some(direction) = decl.direction else {
                continue;
            };
            for declarator in &decl.names {
                if declared_ports.contains(&declarator.name) && declarator.memory_depth.is_none() {
                    ports.insert(
                        declarator.name.clone(),
                        (sv_type_to_rrtl(decl.ty), direction),
                    );
                }
            }
        }
        maps.insert(module.name.clone(), ports);
    }
    maps
}

fn lower_instance_connection(
    instance: &SvInstance,
    connection: &SvConnection,
    port_info: Option<(BitType, SvDirection)>,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    occupied_names: &mut HashSet<String>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
    continuous_state: &mut CombState,
) -> Result<Signal, ErrorReport> {
    match port_info.map(|(_, direction)| direction) {
        Some(SvDirection::Output) => {
            if let Some(signal) = direct_instance_signal(&connection.expr, symbols)? {
                return Ok(signal);
            }
            let Some(dst) = instance_output_lvalue(&connection.expr) else {
                return Err(err(
                    "E_SV_INSTANCE_CONN",
                    format!(
                        "instance `{}` output port `{}` must connect to a signal or selected signal",
                        instance.name, connection.port
                    ),
                ));
            };
            let (ty, _) = port_info.expect("port_info is present for known output");
            let name = fresh_sv_inst_wire_name(&instance.name, &connection.port, occupied_names);
            let signal = builder.wire(name, ty);
            lower_continuous_lvalue_assign_value(
                &dst,
                signal.value(),
                symbols,
                consts,
                continuous_state,
            )?;
            Ok(signal)
        }
        Some(SvDirection::Inout) => instance_connection_signal(instance, connection, symbols),
        Some(SvDirection::Input) => {
            if let Some(signal) = direct_instance_signal(&connection.expr, symbols)? {
                return Ok(signal);
            }
            let (ty, _) = port_info.expect("port_info is present for known input");
            Ok(synthesize_instance_input(
                instance,
                connection,
                ty,
                symbols,
                consts,
                occupied_names,
                builder,
            )?)
        }
        None => {
            if let Some(signal) = direct_instance_signal(&connection.expr, symbols)? {
                return Ok(signal);
            }
            let ty = sv_expr_type(&connection.expr, symbols, consts)?;
            Ok(synthesize_instance_input(
                instance,
                connection,
                ty,
                symbols,
                consts,
                occupied_names,
                builder,
            )?)
        }
    }
}

fn instance_output_lvalue(expr: &SvExpr) -> Option<SvLvalue> {
    match expr {
        SvExpr::Ident(name) => Some(SvLvalue::Signal(name.clone())),
        SvExpr::Bracket { expr, index } => match expr.as_ref() {
            SvExpr::Ident(name) => Some(SvLvalue::Bit {
                name: name.clone(),
                index: index.as_ref().clone(),
            }),
            _ => None,
        },
        SvExpr::Slice { expr, msb, lsb } => match expr.as_ref() {
            SvExpr::Ident(name) => Some(SvLvalue::Slice {
                name: name.clone(),
                msb: SvExpr::Lit {
                    value: *msb as u128,
                    width: 32,
                    signed: false,
                },
                lsb: SvExpr::Lit {
                    value: *lsb as u128,
                    width: 32,
                    signed: false,
                },
            }),
            _ => None,
        },
        _ => None,
    }
}

fn direct_instance_signal(
    expr: &SvExpr,
    symbols: &HashMap<String, SvSymbol>,
) -> Result<Option<Signal>, ErrorReport> {
    let SvExpr::Ident(name) = expr else {
        return Ok(None);
    };
    let Some(symbol) = symbols.get(name) else {
        return Ok(None);
    };
    if symbol.is_memory {
        return Err(err(
            "E_SV_INSTANCE_SIGNAL",
            format!("instance connection cannot use memory `{name}` as a port signal"),
        ));
    }
    Ok(Some(symbol.signal))
}

fn instance_connection_signal(
    instance: &SvInstance,
    connection: &SvConnection,
    symbols: &HashMap<String, SvSymbol>,
) -> Result<Signal, ErrorReport> {
    let SvExpr::Ident(name) = &connection.expr else {
        return Err(err(
            "E_SV_INSTANCE_CONN",
            format!(
                "instance `{}` port `{}` is output/inout and must connect to a signal",
                instance.name, connection.port
            ),
        ));
    };
    let Some(symbol) = symbols.get(name) else {
        return Err(err(
            "E_SV_INSTANCE_SIGNAL",
            format!(
                "instance `{}` connects unknown signal `{name}`",
                instance.name
            ),
        ));
    };
    if symbol.is_memory {
        return Err(err(
            "E_SV_INSTANCE_SIGNAL",
            format!(
                "instance `{}` connects memory `{name}` as a port signal",
                instance.name
            ),
        ));
    }
    Ok(symbol.signal)
}

fn synthesize_instance_input(
    instance: &SvInstance,
    connection: &SvConnection,
    ty: BitType,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    occupied_names: &mut HashSet<String>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
) -> Result<Signal, ErrorReport> {
    let name = fresh_sv_inst_wire_name(&instance.name, &connection.port, occupied_names);
    let signal = builder.wire(name, ty);
    let expr_ty = sv_expr_type(&connection.expr, symbols, consts)?;
    let expr = coerce_expr_to_type(lower_expr(&connection.expr, symbols, consts)?, expr_ty, ty);
    builder.assign(signal, expr);
    Ok(signal)
}

fn infer_external_port(
    connection: &SvConnection,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
) -> Result<(BitType, SvDirection), ErrorReport> {
    if let SvExpr::Ident(name) = &connection.expr {
        if let Some(symbol) = symbols.get(name) {
            return Ok((symbol.ty, symbol.direction.unwrap_or(SvDirection::Input)));
        }
    }
    Ok((
        sv_expr_type(&connection.expr, symbols, consts)?,
        SvDirection::Input,
    ))
}

fn fresh_sv_inst_wire_name(
    instance_name: &str,
    port_name: &str,
    occupied: &mut HashSet<String>,
) -> String {
    let base = format!("__sv_inst_{instance_name}_{port_name}");
    if occupied.insert(base.clone()) {
        return base;
    }
    let mut index = 1usize;
    loop {
        let candidate = format!("{base}_{index}");
        if occupied.insert(candidate.clone()) {
            return candidate;
        }
        index += 1;
    }
}

fn always_ff_signal_targets(items: &[SvItem]) -> HashSet<String> {
    let mut targets = HashSet::new();
    for item in items {
        if let SvItem::AlwaysFf(always) = item {
            collect_stmt_signal_targets(&always.body, &mut targets);
        }
    }
    targets
}

fn module_signal_names(items: &[SvItem]) -> HashSet<String> {
    let mut names = HashSet::new();
    for item in items {
        if let SvItem::Decl(decl) = item {
            for declarator in &decl.names {
                names.insert(declarator.name.clone());
            }
        }
    }
    names
}

fn fresh_sv_reg_name(base: &str, occupied: &mut HashSet<String>) -> String {
    let first = format!("{base}__sv_reg");
    if occupied.insert(first.clone()) {
        return first;
    }
    let mut index = 1usize;
    loop {
        let candidate = format!("{base}__sv_reg_{index}");
        if occupied.insert(candidate.clone()) {
            return candidate;
        }
        index += 1;
    }
}

fn collect_stmt_signal_targets(stmts: &[SvStmt], targets: &mut HashSet<String>) {
    for stmt in stmts {
        match stmt {
            SvStmt::Assign {
                dst:
                    SvLvalue::Signal(name) | SvLvalue::Bit { name, .. } | SvLvalue::Slice { name, .. },
                ..
            } => {
                targets.insert(name.clone());
            }
            SvStmt::Assign { .. } => {}
            SvStmt::If {
                then_stmts,
                else_stmts,
                ..
            } => {
                collect_stmt_signal_targets(then_stmts, targets);
                collect_stmt_signal_targets(else_stmts, targets);
            }
            SvStmt::For { body, .. } => collect_stmt_signal_targets(body, targets),
            SvStmt::Case { items, .. } => {
                for item in items {
                    collect_stmt_signal_targets(&item.stmts, targets);
                }
            }
            SvStmt::Assert { .. } | SvStmt::Cover { .. } => {}
        }
    }
}

#[derive(Clone, Copy)]
struct SvSymbol {
    signal: Signal,
    ty: BitType,
    direction: Option<SvDirection>,
    is_memory: bool,
}

#[derive(Clone, Default)]
struct CombState {
    values: HashMap<String, Expr>,
    order: Vec<String>,
}

impl CombState {
    fn set(&mut self, name: String, expr: Expr) {
        if !self.values.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.values.insert(name, expr);
    }

    fn expr_for(
        &self,
        name: &str,
        symbols: &HashMap<String, SvSymbol>,
    ) -> Result<Expr, ErrorReport> {
        if let Some(expr) = self.values.get(name) {
            return Ok(expr.clone());
        }
        let signal = symbols
            .get(name)
            .filter(|symbol| !symbol.is_memory)
            .ok_or_else(|| err("E_SV_LVALUE", format!("unknown signal `{name}`")))?
            .signal;
        Ok(signal.value())
    }
}

fn lower_always_comb(
    stmts: &[SvStmt],
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
) -> Result<(), ErrorReport> {
    let mut state = CombState::default();
    lower_comb_stmts(stmts, symbols, consts, builder, &mut state, None)?;
    for name in &state.order {
        let symbol = symbols
            .get(name)
            .filter(|symbol| !symbol.is_memory)
            .ok_or_else(|| err("E_SV_LVALUE", format!("unknown signal `{name}`")))?;
        let expr = state
            .values
            .get(name)
            .cloned()
            .ok_or_else(|| err("E_SV_ALWAYS_COMB", format!("missing value for `{name}`")))?;
        builder.assign(
            symbol.signal,
            coerce_expr_to_signal_type(expr, symbol.signal, symbols)?,
        );
    }
    Ok(())
}

fn lower_comb_stmts(
    stmts: &[SvStmt],
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
    state: &mut CombState,
    enable: Option<Expr>,
) -> Result<HashSet<String>, ErrorReport> {
    let mut assigned = HashSet::new();
    for stmt in stmts {
        assigned.extend(lower_comb_stmt(
            stmt,
            symbols,
            consts,
            builder,
            state,
            enable.clone(),
        )?);
    }
    Ok(assigned)
}

fn lower_comb_stmt(
    stmt: &SvStmt,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
    state: &mut CombState,
    enable: Option<Expr>,
) -> Result<HashSet<String>, ErrorReport> {
    match stmt {
        SvStmt::Assign {
            dst,
            nonblocking: false,
            expr,
        } => {
            let name = lower_comb_lvalue_assign(dst, expr, symbols, consts, state)?;
            Ok(HashSet::from([name]))
        }
        SvStmt::Assign {
            nonblocking: true, ..
        } => Err(err(
            "E_SV_ALWAYS_COMB_ASSIGN",
            "always_comb only supports blocking assignments",
        )),
        SvStmt::If {
            cond,
            then_stmts,
            else_stmts,
        } => lower_comb_if(
            cond, then_stmts, else_stmts, symbols, consts, builder, state, enable,
        ),
        SvStmt::For {
            var,
            init,
            cmp,
            bound,
            step,
            body,
        } => lower_comb_for(
            var, init, *cmp, bound, step, body, symbols, consts, builder, state, enable,
        ),
        SvStmt::Case { kind, expr, items } => {
            lower_comb_case(*kind, expr, items, symbols, consts, builder, state, enable)
        }
        SvStmt::Assert {
            name,
            cond,
            message,
        } => {
            let cond = lower_expr(cond, symbols, consts)?;
            if let Some(enable) = enable {
                if let Some(message) = message {
                    builder.assert_when_msg(name, enable, cond, message);
                } else {
                    builder.assert_when(name, enable, cond);
                }
            } else if let Some(message) = message {
                builder.assert_msg(name, cond, message);
            } else {
                builder.assert(name, cond);
            }
            Ok(HashSet::new())
        }
        SvStmt::Cover { name, cond } => {
            let cond = lower_expr(cond, symbols, consts)?;
            if let Some(enable) = enable {
                builder.cover_when(name, enable, cond);
            } else {
                builder.cover(name, cond);
            }
            Ok(HashSet::new())
        }
    }
}

fn lower_comb_if(
    cond: &SvExpr,
    then_stmts: &[SvStmt],
    else_stmts: &[SvStmt],
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
    state: &mut CombState,
    enable: Option<Expr>,
) -> Result<HashSet<String>, ErrorReport> {
    let cond = lower_expr(cond, symbols, consts)?;
    let base = state.clone();
    let then_enable = combine_enable(enable.clone(), cond.clone());
    let else_enable = combine_enable(enable, not_expr(cond.clone()));

    let mut then_state = base.clone();
    let then_assigned = lower_comb_stmts(
        then_stmts,
        symbols,
        consts,
        builder,
        &mut then_state,
        Some(then_enable),
    )?;
    let mut else_state = base.clone();
    let else_assigned = if else_stmts.is_empty() {
        HashSet::new()
    } else {
        lower_comb_stmts(
            else_stmts,
            symbols,
            consts,
            builder,
            &mut else_state,
            Some(else_enable),
        )?
    };

    let mut assigned = then_assigned;
    assigned.extend(else_assigned);
    for name in assigned.iter().cloned() {
        let then_expr = then_state.expr_for(&name, symbols)?;
        let else_expr = else_state.expr_for(&name, symbols)?;
        state.set(name, mux(cond.clone(), then_expr, else_expr));
    }
    Ok(assigned)
}

fn lower_comb_for(
    var: &str,
    init: &SvExpr,
    cmp: SvForCmp,
    bound: &SvExpr,
    step: &SvForStep,
    body: &[SvStmt],
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
    state: &mut CombState,
    enable: Option<Expr>,
) -> Result<HashSet<String>, ErrorReport> {
    let mut assigned = HashSet::new();
    for iter_consts in for_loop_iteration_consts(var, init, cmp, bound, step, consts)? {
        assigned.extend(lower_comb_stmts(
            body,
            symbols,
            &iter_consts,
            builder,
            state,
            enable.clone(),
        )?);
    }
    Ok(assigned)
}

fn lower_comb_lvalue_assign(
    dst: &SvLvalue,
    expr: &SvExpr,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    state: &mut CombState,
) -> Result<String, ErrorReport> {
    let name = signal_lvalue_name(dst, symbols)?;
    let base = state.expr_for(&name, symbols)?;
    let signal = symbols[&name].signal;
    let next = lower_lvalue_assignment_expr(dst, expr, base, signal, symbols, consts)?;
    state.set(name.clone(), next);
    Ok(name)
}

fn lower_continuous_lvalue_assign(
    dst: &SvLvalue,
    expr: &SvExpr,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    state: &mut CombState,
) -> Result<String, ErrorReport> {
    let name = signal_lvalue_name(dst, symbols)?;
    let signal = symbols[&name].signal;
    let base = if let Some(expr) = state.values.get(&name) {
        expr.clone()
    } else {
        lit_u(0, symbols[&name].ty.width)
    };
    let next = lower_lvalue_assignment_expr(dst, expr, base, signal, symbols, consts)?;
    state.set(name.clone(), next);
    Ok(name)
}

fn lower_continuous_lvalue_assign_value(
    dst: &SvLvalue,
    value: Expr,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    state: &mut CombState,
) -> Result<String, ErrorReport> {
    let name = signal_lvalue_name(dst, symbols)?;
    let signal = symbols[&name].signal;
    let base = if let Some(expr) = state.values.get(&name) {
        expr.clone()
    } else {
        lit_u(0, symbols[&name].ty.width)
    };
    let next = lower_lvalue_assignment_value(dst, value, base, signal, symbols, consts)?;
    state.set(name.clone(), next);
    Ok(name)
}

fn lower_comb_case(
    kind: SvCaseKind,
    expr: &SvExpr,
    items: &[SvCaseItem],
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
    state: &mut CombState,
    enable: Option<Expr>,
) -> Result<HashSet<String>, ErrorReport> {
    let selector = lower_expr(expr, symbols, consts)?;
    let selector_width = sv_expr_width(expr, symbols, consts)?;
    let base = state.clone();
    let mut assigned = HashSet::new();
    let mut any_match: Option<Expr> = None;
    let mut all_matches: Option<Expr> = None;
    for item in items.iter().filter(|item| !item.is_default) {
        let match_expr = case_match_expr(
            kind,
            &selector,
            selector_width,
            &item.labels,
            symbols,
            consts,
        )?;
        all_matches = Some(match all_matches {
            Some(current) => current | match_expr,
            None => match_expr,
        });
    }
    let mut branches = Vec::new();
    let mut default_state = None;
    let mut default_assigned = HashSet::new();

    for item in items {
        let mut branch_state = base.clone();
        if item.is_default {
            if default_state.is_some() {
                return Err(err(
                    "E_SV_CASE",
                    "case statement has multiple default items",
                ));
            }
            let default_enable = case_default_enable(enable.clone(), all_matches.clone());
            default_assigned = lower_comb_stmts(
                &item.stmts,
                symbols,
                consts,
                builder,
                &mut branch_state,
                default_enable,
            )?;
            assigned.extend(default_assigned.iter().cloned());
            default_state = Some(branch_state);
            continue;
        }

        let match_expr = case_match_expr(
            kind,
            &selector,
            selector_width,
            &item.labels,
            symbols,
            consts,
        )?;
        any_match = Some(match any_match {
            Some(current) => current | match_expr.clone(),
            None => match_expr.clone(),
        });
        let branch_enable = combine_enable(enable.clone(), match_expr.clone());
        let branch_assigned = lower_comb_stmts(
            &item.stmts,
            symbols,
            consts,
            builder,
            &mut branch_state,
            Some(branch_enable),
        )?;
        assigned.extend(branch_assigned.iter().cloned());
        branches.push((match_expr, branch_state));
    }

    assigned.extend(default_assigned);
    for name in assigned.iter().cloned() {
        let mut expr = if let Some(default_state) = &default_state {
            default_state.expr_for(&name, symbols)?
        } else {
            base.expr_for(&name, symbols)?
        };
        for (match_expr, branch_state) in branches.iter().rev() {
            expr = mux(
                match_expr.clone(),
                branch_state.expr_for(&name, symbols)?,
                expr,
            );
        }
        state.set(name, expr);
    }
    Ok(assigned)
}

fn case_match_expr(
    kind: SvCaseKind,
    selector: &Expr,
    selector_width: Option<Width>,
    labels: &[SvCaseLabel],
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
) -> Result<Expr, ErrorReport> {
    if labels.is_empty() {
        return Err(err("E_SV_CASE", "case item must have at least one label"));
    }
    let mut expr = None;
    for label in labels {
        let item = lower_case_match(kind, selector, selector_width, label, symbols, consts)?;
        expr = Some(match expr {
            Some(current) => current | item,
            None => item,
        });
    }
    Ok(expr.unwrap())
}

fn lower_case_match(
    kind: SvCaseKind,
    selector: &Expr,
    selector_width: Option<Width>,
    label: &SvCaseLabel,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
) -> Result<Expr, ErrorReport> {
    match label {
        SvCaseLabel::Expr(label) => {
            let label_expr = lower_case_label(label, selector_width, symbols, consts)?;
            Ok(selector.clone().eq_expr(label_expr))
        }
        SvCaseLabel::Wildcard { value, mask, width } => {
            if kind == SvCaseKind::Normal {
                return Err(err(
                    "E_SV_CASE",
                    "wildcard case label is only supported in casez/casex",
                ));
            }
            let label_width = selector_width.unwrap_or(*width);
            let value = resize_u128_to_width(*value, *width, label_width);
            let mask = resize_u128_to_width(*mask, *width, label_width);
            let mask_expr = lit_u(mask, label_width);
            Ok((selector.clone() & mask_expr.clone()).eq_expr(lit_u(value & mask, label_width)))
        }
    }
}

fn resize_u128_to_width(value: u128, from_width: Width, to_width: Width) -> u128 {
    if to_width >= 128 {
        value
    } else if from_width >= to_width {
        value & ((1u128 << to_width) - 1)
    } else {
        value & ((1u128 << from_width) - 1)
    }
}

fn lower_case_label(
    label: &SvExpr,
    selector_width: Option<Width>,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
) -> Result<Expr, ErrorReport> {
    let label_expr = lower_expr(label, symbols, consts)?;
    let Some(selector_width) = selector_width else {
        return Ok(label_expr);
    };
    let Some(label_width) = sv_expr_width(label, symbols, consts)? else {
        return Ok(label_expr);
    };
    if label_width == selector_width {
        Ok(label_expr)
    } else if label_width > selector_width {
        Ok(label_expr.slice(0, selector_width))
    } else {
        Ok(concat([lit_u(0, selector_width - label_width), label_expr]))
    }
}

fn case_default_enable(enable: Option<Expr>, any_match: Option<Expr>) -> Option<Expr> {
    match (enable, any_match) {
        (Some(enable), Some(any_match)) => Some(enable & not_expr(any_match)),
        (Some(enable), None) => Some(enable),
        (None, Some(any_match)) => Some(not_expr(any_match)),
        (None, None) => None,
    }
}

fn combine_enable(enable: Option<Expr>, cond: Expr) -> Expr {
    match enable {
        Some(enable) => enable & cond,
        None => cond,
    }
}

fn not_expr(expr: Expr) -> Expr {
    expr.eq_expr(lit_u(0, 1))
}

fn lower_always_ff(
    always: &SvAlwaysFf,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
) -> Result<(), ErrorReport> {
    let clock = symbols
        .get(&always.clock)
        .ok_or_else(|| err("E_SV_CLOCK", format!("unknown clock `{}`", always.clock)))?
        .signal;
    let mut state = FfState::default();
    for stmt in &always.body {
        lower_ff_stmt(
            stmt,
            clock,
            symbols,
            consts,
            builder,
            &mut state,
            always.async_reset.as_ref(),
            None,
        )?;
    }
    for name in &state.order {
        let reg = symbols
            .get(name)
            .filter(|symbol| !symbol.is_memory)
            .ok_or_else(|| err("E_SV_LVALUE", format!("unknown signal `{name}`")))?
            .signal;
        let expr = state
            .values
            .get(name)
            .cloned()
            .ok_or_else(|| err("E_SV_ALWAYS_FF", format!("missing next value for `{name}`")))?;
        builder.clock(reg, clock);
        builder.next(reg, coerce_expr_to_signal_type(expr, reg, symbols)?);
    }
    Ok(())
}

#[derive(Clone, Default)]
struct FfState {
    values: HashMap<String, Expr>,
    order: Vec<String>,
}

impl FfState {
    fn set(&mut self, name: String, expr: Expr) {
        if !self.values.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.values.insert(name, expr);
    }

    fn expr_for(
        &self,
        name: &str,
        symbols: &HashMap<String, SvSymbol>,
    ) -> Result<Expr, ErrorReport> {
        if let Some(expr) = self.values.get(name) {
            return Ok(expr.clone());
        }
        let signal = symbols
            .get(name)
            .filter(|symbol| !symbol.is_memory)
            .ok_or_else(|| err("E_SV_LVALUE", format!("unknown signal `{name}`")))?
            .signal;
        Ok(signal.value())
    }
}

fn lower_ff_stmts(
    stmts: &[SvStmt],
    clock: Signal,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
    state: &mut FfState,
    async_edge: Option<&SvResetEdge>,
    enable: Option<Expr>,
) -> Result<HashSet<String>, ErrorReport> {
    let mut assigned = HashSet::new();
    for stmt in stmts {
        assigned.extend(lower_ff_stmt(
            stmt,
            clock,
            symbols,
            consts,
            builder,
            state,
            async_edge,
            enable.clone(),
        )?);
    }
    Ok(assigned)
}

fn lower_ff_stmt(
    stmt: &SvStmt,
    clock: Signal,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
    state: &mut FfState,
    async_edge: Option<&SvResetEdge>,
    enable: Option<Expr>,
) -> Result<HashSet<String>, ErrorReport> {
    match stmt {
        SvStmt::Assign {
            dst,
            nonblocking: true,
            expr,
        } if lvalue_is_memory(dst, symbols) => {
            let Some(enable) = enable else {
                return Err(err(
                    "E_SV_MEMORY",
                    "memory writes in always_ff require an enclosing enable if/case",
                ));
            };
            let (name, addr) = memory_lvalue_parts(dst)?;
            let mem = symbols
                .get(name)
                .filter(|symbol| symbol.is_memory)
                .ok_or_else(|| err("E_SV_MEMORY", format!("unknown memory `{name}`")))?
                .signal;
            builder.mem_write(
                mem,
                clock,
                enable,
                lower_expr(addr, symbols, consts)?,
                lower_expr(expr, symbols, consts)?,
            );
            Ok(HashSet::new())
        }
        SvStmt::Assign {
            dst,
            nonblocking: true,
            expr,
        } => {
            let name = lower_ff_lvalue_assign(dst, expr, symbols, consts, state)?;
            Ok(HashSet::from([name]))
        }
        SvStmt::Assign {
            nonblocking: false, ..
        } => Err(err(
            "E_SV_ALWAYS_FF_ASSIGN",
            "always_ff only supports nonblocking assignments",
        )),
        SvStmt::If {
            cond,
            then_stmts,
            else_stmts,
        } => lower_ff_if(
            cond, then_stmts, else_stmts, clock, symbols, consts, builder, state, async_edge,
            enable,
        ),
        SvStmt::For {
            var,
            init,
            cmp,
            bound,
            step,
            body,
        } => lower_ff_for(
            var, init, *cmp, bound, step, body, clock, symbols, consts, builder, state, async_edge,
            enable,
        ),
        SvStmt::Case { kind, expr, items } => lower_ff_case(
            *kind, expr, items, clock, symbols, consts, builder, state, enable,
        ),
        SvStmt::Assert {
            name,
            cond,
            message,
        } => {
            let cond = lower_expr(cond, symbols, consts)?;
            let cond = if let Some(enable) = enable {
                not_expr(enable) | cond
            } else {
                cond
            };
            if let Some(message) = message {
                builder.assert_clocked_msg(name, clock, cond, message);
            } else {
                builder.assert_clocked(name, clock, cond);
            }
            Ok(HashSet::new())
        }
        SvStmt::Cover { name, cond } => {
            let cond = lower_expr(cond, symbols, consts)?;
            let cond = if let Some(enable) = enable {
                enable & cond
            } else {
                cond
            };
            builder.cover_clocked(name, clock, cond);
            Ok(HashSet::new())
        }
    }
}

fn lower_ff_for(
    var: &str,
    init: &SvExpr,
    cmp: SvForCmp,
    bound: &SvExpr,
    step: &SvForStep,
    body: &[SvStmt],
    clock: Signal,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
    state: &mut FfState,
    async_edge: Option<&SvResetEdge>,
    enable: Option<Expr>,
) -> Result<HashSet<String>, ErrorReport> {
    let mut assigned = HashSet::new();
    for iter_consts in for_loop_iteration_consts(var, init, cmp, bound, step, consts)? {
        assigned.extend(lower_ff_stmts(
            body,
            clock,
            symbols,
            &iter_consts,
            builder,
            state,
            async_edge,
            enable.clone(),
        )?);
    }
    Ok(assigned)
}

fn lower_ff_lvalue_assign(
    dst: &SvLvalue,
    expr: &SvExpr,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    state: &mut FfState,
) -> Result<String, ErrorReport> {
    let name = signal_lvalue_name(dst, symbols)?;
    let base = state.expr_for(&name, symbols)?;
    let signal = symbols[&name].signal;
    let next = lower_lvalue_assignment_expr(dst, expr, base, signal, symbols, consts)?;
    state.set(name.clone(), next);
    Ok(name)
}

fn lower_ff_if(
    cond: &SvExpr,
    then_stmts: &[SvStmt],
    else_stmts: &[SvStmt],
    clock: Signal,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
    state: &mut FfState,
    async_edge: Option<&SvResetEdge>,
    enable: Option<Expr>,
) -> Result<HashSet<String>, ErrorReport> {
    if enable.is_none() && (async_edge.is_some() || !else_stmts.is_empty()) {
        if let Some((reset_name, active_low)) = reset_condition(cond) {
            let Some(reset_assignments) = reset_branch_const_assignments(then_stmts, consts)?
            else {
                return lower_ff_if_non_reset(
                    cond, then_stmts, else_stmts, clock, symbols, consts, builder, state, enable,
                );
            };
            if let Some(edge) = async_edge {
                if edge.signal != reset_name || edge.active_low != active_low {
                    return Err(err(
                        "E_SV_RESET_EDGE",
                        "async reset edge and if reset condition disagree",
                    ));
                }
            }
            let mut reset_values = HashMap::<String, i128>::new();
            let mut reset_order = Vec::<String>::new();
            for assignment in reset_assignments {
                let name = signal_lvalue_name(assignment.dst, symbols)?;
                if !reset_values.contains_key(&name) {
                    reset_order.push(name.clone());
                }
                let current = reset_values.get(&name).copied().unwrap_or(0);
                let value = const_lvalue_assignment_value(
                    assignment.dst,
                    assignment.expr,
                    current,
                    symbols,
                    &assignment.consts,
                )?;
                reset_values.insert(name, value);
            }
            for name in reset_order {
                let reg = symbols
                    .get(&name)
                    .filter(|symbol| !symbol.is_memory)
                    .ok_or_else(|| err("E_SV_RESET", format!("unknown reset target `{name}`")))?
                    .signal;
                let reset = symbols
                    .get(&reset_name)
                    .ok_or_else(|| err("E_SV_RESET", format!("unknown reset `{reset_name}`")))?
                    .signal;
                let reset_value = reset_values.get(&name).copied().ok_or_else(|| {
                    err("E_SV_RESET", format!("missing reset value for `{name}`"))
                })?;
                builder.clock(reg, clock);
                if async_edge.is_some() {
                    if active_low {
                        builder.async_reset_low(reg, reset, reset_value);
                    } else {
                        builder.async_reset(reg, reset, reset_value);
                    }
                } else if active_low {
                    builder.reset_low(reg, reset, reset_value);
                } else {
                    builder.reset(reg, reset, reset_value);
                }
            }
            for stmt in else_stmts {
                lower_ff_stmt(stmt, clock, symbols, consts, builder, state, None, None)?;
            }
            return Ok(HashSet::new());
        }
    }

    lower_ff_if_non_reset(
        cond, then_stmts, else_stmts, clock, symbols, consts, builder, state, enable,
    )
}

fn lower_ff_if_non_reset(
    cond: &SvExpr,
    then_stmts: &[SvStmt],
    else_stmts: &[SvStmt],
    clock: Signal,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
    state: &mut FfState,
    enable: Option<Expr>,
) -> Result<HashSet<String>, ErrorReport> {
    if else_stmts.is_empty() {
        if then_stmts.len() == 1 {
            match &then_stmts[0] {
                SvStmt::Assert {
                    name,
                    cond: assertion_cond,
                    message,
                } => {
                    let guard = combine_enable(enable.clone(), lower_expr(cond, symbols, consts)?);
                    let gated = not_expr(guard) | lower_expr(assertion_cond, symbols, consts)?;
                    if let Some(message) = message {
                        builder.assert_clocked_msg(name, clock, gated, message);
                    } else {
                        builder.assert_clocked(name, clock, gated);
                    }
                    return Ok(HashSet::new());
                }
                SvStmt::Cover {
                    name,
                    cond: cover_cond,
                } => {
                    let guard = combine_enable(enable.clone(), lower_expr(cond, symbols, consts)?);
                    let gated = guard & lower_expr(cover_cond, symbols, consts)?;
                    builder.cover_clocked(name, clock, gated);
                    return Ok(HashSet::new());
                }
                _ => {}
            }
        }
    }

    let cond = lower_expr(cond, symbols, consts)?;
    let base = state.clone();
    let mut then_state = base.clone();
    let then_assigned = lower_ff_stmts(
        then_stmts,
        clock,
        symbols,
        consts,
        builder,
        &mut then_state,
        None,
        Some(combine_enable(enable.clone(), cond.clone())),
    )?;
    let mut else_state = base.clone();
    let else_assigned = if else_stmts.is_empty() {
        HashSet::new()
    } else {
        lower_ff_stmts(
            else_stmts,
            clock,
            symbols,
            consts,
            builder,
            &mut else_state,
            None,
            Some(combine_enable(enable, not_expr(cond.clone()))),
        )?
    };

    let mut assigned = then_assigned;
    assigned.extend(else_assigned);
    for name in assigned.iter().cloned() {
        state.set(
            name.clone(),
            mux(
                cond.clone(),
                then_state.expr_for(&name, symbols)?,
                else_state.expr_for(&name, symbols)?,
            ),
        );
    }
    Ok(assigned)
}

struct ResetAssignment<'a> {
    dst: &'a SvLvalue,
    expr: &'a SvExpr,
    consts: HashMap<String, u128>,
}

fn reset_branch_const_assignments<'a>(
    stmts: &'a [SvStmt],
    consts: &HashMap<String, u128>,
) -> Result<Option<Vec<ResetAssignment<'a>>>, ErrorReport> {
    if stmts.is_empty() {
        return Ok(None);
    }
    let mut assignments = Vec::new();
    if collect_reset_branch_const_assignments(stmts, consts, &mut assignments)? {
        Ok(Some(assignments))
    } else {
        Ok(None)
    }
}

fn collect_reset_branch_const_assignments<'a>(
    stmts: &'a [SvStmt],
    consts: &HashMap<String, u128>,
    assignments: &mut Vec<ResetAssignment<'a>>,
) -> Result<bool, ErrorReport> {
    for stmt in stmts {
        match stmt {
            SvStmt::Assign {
                dst,
                nonblocking: true,
                expr,
            } if reset_lvalue_candidate(dst) && const_value(expr, consts).is_ok() => assignments
                .push(ResetAssignment {
                    dst,
                    expr,
                    consts: consts.clone(),
                }),
            SvStmt::For {
                var,
                init,
                cmp,
                bound,
                step,
                body,
            } => {
                for iter_consts in for_loop_iteration_consts(var, init, *cmp, bound, step, consts)?
                {
                    if !collect_reset_branch_const_assignments(body, &iter_consts, assignments)? {
                        return Ok(false);
                    }
                }
            }
            _ => return Ok(false),
        }
    }
    Ok(true)
}

fn reset_lvalue_candidate(dst: &SvLvalue) -> bool {
    matches!(
        dst,
        SvLvalue::Signal(_) | SvLvalue::Bit { .. } | SvLvalue::Slice { .. }
    )
}

fn lower_ff_case(
    kind: SvCaseKind,
    expr: &SvExpr,
    items: &[SvCaseItem],
    clock: Signal,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
    state: &mut FfState,
    enable: Option<Expr>,
) -> Result<HashSet<String>, ErrorReport> {
    let selector = lower_expr(expr, symbols, consts)?;
    let selector_width = sv_expr_width(expr, symbols, consts)?;
    let base = state.clone();
    let mut assigned = HashSet::new();
    let mut all_matches: Option<Expr> = None;
    for item in items.iter().filter(|item| !item.is_default) {
        let match_expr = case_match_expr(
            kind,
            &selector,
            selector_width,
            &item.labels,
            symbols,
            consts,
        )?;
        all_matches = Some(match all_matches {
            Some(current) => current | match_expr,
            None => match_expr,
        });
    }
    let mut branches = Vec::new();
    let mut default_state = None;
    let mut default_assigned = HashSet::new();

    for item in items {
        let mut branch_state = base.clone();
        if item.is_default {
            if default_state.is_some() {
                return Err(err(
                    "E_SV_CASE",
                    "case statement has multiple default items",
                ));
            }
            default_assigned = lower_ff_stmts(
                &item.stmts,
                clock,
                symbols,
                consts,
                builder,
                &mut branch_state,
                None,
                case_default_enable(enable.clone(), all_matches.clone()),
            )?;
            assigned.extend(default_assigned.iter().cloned());
            default_state = Some(branch_state);
            continue;
        }

        let match_expr = case_match_expr(
            kind,
            &selector,
            selector_width,
            &item.labels,
            symbols,
            consts,
        )?;
        let branch_enable = Some(combine_enable(enable.clone(), match_expr.clone()));
        let branch_assigned = lower_ff_stmts(
            &item.stmts,
            clock,
            symbols,
            consts,
            builder,
            &mut branch_state,
            None,
            branch_enable,
        )?;
        assigned.extend(branch_assigned.iter().cloned());
        branches.push((match_expr, branch_state));
    }

    assigned.extend(default_assigned);
    for name in assigned.iter().cloned() {
        let mut expr = if let Some(default_state) = &default_state {
            default_state.expr_for(&name, symbols)?
        } else {
            base.expr_for(&name, symbols)?
        };
        for (match_expr, branch_state) in branches.iter().rev() {
            expr = mux(
                match_expr.clone(),
                branch_state.expr_for(&name, symbols)?,
                expr,
            );
        }
        state.set(name, expr);
    }
    Ok(assigned)
}

fn lower_expr(
    expr: &SvExpr,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
) -> Result<Expr, ErrorReport> {
    Ok(match expr {
        SvExpr::Ident(name) => {
            if let Some(symbol) = symbols.get(name) {
                symbol.signal.value()
            } else if let Some(value) = consts.get(name) {
                lit_u(*value, 32)
            } else {
                return Err(err(
                    "E_SV_EXPR_IDENT",
                    format!("unknown identifier `{name}`"),
                ));
            }
        }
        SvExpr::Lit {
            value,
            width,
            signed,
        } => {
            if *signed {
                rrtl_core::lit_s(*value as i128, *width)
            } else {
                lit_u(*value, *width)
            }
        }
        SvExpr::Unary { op, expr } => {
            let inner = lower_expr(expr, symbols, consts)?;
            match op {
                SvUnaryOp::Not => inner.eq_expr(lit_u(0, 1)),
                SvUnaryOp::BitNot => !inner,
                SvUnaryOp::Neg => lit_u(0, 32) - inner,
            }
        }
        SvExpr::Binary { op, lhs, rhs } => {
            let lhs = lower_expr(lhs, symbols, consts)?;
            let rhs = lower_expr(rhs, symbols, consts)?;
            match op {
                SvBinaryOp::Add => lhs + rhs,
                SvBinaryOp::Sub => lhs - rhs,
                SvBinaryOp::Mul => lhs * rhs,
                SvBinaryOp::And | SvBinaryOp::LogAnd => lhs & rhs,
                SvBinaryOp::Or | SvBinaryOp::LogOr => lhs | rhs,
                SvBinaryOp::Xor => lhs ^ rhs,
                SvBinaryOp::Eq => lhs.eq_expr(rhs),
                SvBinaryOp::Ne => lhs.ne_expr(rhs),
                SvBinaryOp::Lt => lhs.lt_expr(rhs),
                SvBinaryOp::Le => lhs.clone().lt_expr(rhs.clone()) | lhs.eq_expr(rhs),
                SvBinaryOp::Gt => rhs.lt_expr(lhs),
                SvBinaryOp::Ge => rhs.clone().lt_expr(lhs.clone()) | lhs.eq_expr(rhs),
            }
        }
        SvExpr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => mux(
            lower_expr(cond, symbols, consts)?,
            lower_expr(then_expr, symbols, consts)?,
            lower_expr(else_expr, symbols, consts)?,
        ),
        SvExpr::Concat(parts) => concat(
            parts
                .iter()
                .map(|part| lower_concat_part(part, symbols, consts))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        SvExpr::Repeat { count, expr } => {
            if *count == 0 {
                return Err(err("E_SV_REPEAT", "replication count must be nonzero"));
            }
            let expr = lower_concat_part(expr, symbols, consts)?;
            concat(std::iter::repeat_n(expr, *count as usize).collect::<Vec<_>>())
        }
        SvExpr::Cast { signed, expr } => {
            let expr = lower_expr(expr, symbols, consts)?;
            if *signed {
                as_sint(expr)
            } else {
                as_uint(expr)
            }
        }
        SvExpr::Index { expr, index } => lower_expr(expr, symbols, consts)?.slice(*index, 1),
        SvExpr::Slice { expr, msb, lsb } => {
            if msb < lsb {
                return Err(err("E_SV_SLICE", "part-select msb must be >= lsb"));
            }
            lower_expr(expr, symbols, consts)?.slice(*lsb, msb - lsb + 1)
        }
        SvExpr::MemRead { name, addr } => {
            let mem = symbols
                .get(name)
                .filter(|symbol| symbol.is_memory)
                .ok_or_else(|| err("E_SV_MEMORY", format!("unknown memory `{name}`")))?
                .signal;
            mem_read(mem, lower_expr(addr, symbols, consts)?)
        }
        SvExpr::Bracket { expr, index } => match expr.as_ref() {
            SvExpr::Ident(name) => {
                let Some(symbol) = symbols.get(name) else {
                    return Err(err(
                        "E_SV_EXPR_IDENT",
                        format!("unknown identifier `{name}`"),
                    ));
                };
                if symbol.is_memory {
                    mem_read(symbol.signal, lower_expr(index, symbols, consts)?)
                } else {
                    let index = const_eval(index, consts)?;
                    let index = Width::try_from(index)
                        .map_err(|_| err("E_SV_INDEX", "bit-select index exceeds width range"))?;
                    symbol.signal.value().slice(index, 1)
                }
            }
            _ => {
                let index = const_eval(index, consts)?;
                let index = Width::try_from(index)
                    .map_err(|_| err("E_SV_INDEX", "bit-select index exceeds width range"))?;
                lower_expr(expr, symbols, consts)?.slice(index, 1)
            }
        },
    })
}

fn lower_concat_part(
    expr: &SvExpr,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
) -> Result<Expr, ErrorReport> {
    let lowered = lower_expr(expr, symbols, consts)?;
    if let Some(width) = sv_expr_width(expr, symbols, consts)? {
        Ok(lowered.slice(0, width))
    } else {
        Ok(lowered)
    }
}

fn sv_expr_width(
    expr: &SvExpr,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
) -> Result<Option<Width>, ErrorReport> {
    Ok(match expr {
        SvExpr::Ident(name) => symbols.get(name).map(|symbol| symbol.ty.width).or(Some(32)),
        SvExpr::Lit { width, .. } => Some(*width),
        SvExpr::Unary { expr, .. } | SvExpr::Cast { expr, .. } => {
            sv_expr_width(expr, symbols, consts)?
        }
        SvExpr::Binary { lhs, .. } => sv_expr_width(lhs, symbols, consts)?,
        SvExpr::Ternary { then_expr, .. } => sv_expr_width(then_expr, symbols, consts)?,
        SvExpr::Concat(parts) => {
            let mut width = 0;
            for part in parts {
                let Some(part_width) = sv_expr_width(part, symbols, consts)? else {
                    return Ok(None);
                };
                width += part_width;
            }
            Some(width)
        }
        SvExpr::Repeat { count, expr } => {
            sv_expr_width(expr, symbols, consts)?.map(|width| width * *count)
        }
        SvExpr::Index { .. } | SvExpr::Bracket { .. } => Some(1),
        SvExpr::Slice { msb, lsb, .. } => Some(msb - lsb + 1),
        SvExpr::MemRead { name, .. } => symbols.get(name).map(|symbol| symbol.ty.width),
    })
}

fn sv_expr_type(
    expr: &SvExpr,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
) -> Result<BitType, ErrorReport> {
    let width = sv_expr_width(expr, symbols, consts)?.unwrap_or(32);
    let signedness = if sv_expr_is_signed(expr, symbols) {
        Signedness::Signed
    } else {
        Signedness::Unsigned
    };
    Ok(BitType::new(width, signedness))
}

fn sv_expr_is_signed(expr: &SvExpr, symbols: &HashMap<String, SvSymbol>) -> bool {
    match expr {
        SvExpr::Ident(name) => symbols
            .get(name)
            .is_some_and(|symbol| symbol.ty.is_signed()),
        SvExpr::Lit { signed, .. } => *signed,
        SvExpr::Unary { expr, .. } => sv_expr_is_signed(expr, symbols),
        SvExpr::Cast { signed, .. } => *signed,
        SvExpr::Binary { lhs, .. } => sv_expr_is_signed(lhs, symbols),
        SvExpr::Ternary { then_expr, .. } => sv_expr_is_signed(then_expr, symbols),
        SvExpr::Concat(_) | SvExpr::Repeat { .. } => false,
        SvExpr::Index { .. } | SvExpr::Slice { .. } | SvExpr::Bracket { .. } => false,
        SvExpr::MemRead { name, .. } => symbols
            .get(name)
            .is_some_and(|symbol| symbol.ty.is_signed()),
    }
}

fn coerce_expr_to_type(expr: Expr, from: BitType, to: BitType) -> Expr {
    let expr = match from.width.cmp(&to.width) {
        std::cmp::Ordering::Less if from.is_signed() => Expr::Sext {
            expr: Box::new(expr),
            width: to.width,
        },
        std::cmp::Ordering::Less => Expr::Zext {
            expr: Box::new(expr),
            width: to.width,
        },
        std::cmp::Ordering::Greater => Expr::Trunc {
            expr: Box::new(expr),
            width: to.width,
        },
        std::cmp::Ordering::Equal => expr,
    };
    if from.signedness == to.signedness {
        expr
    } else {
        Expr::Cast {
            expr: Box::new(expr),
            signedness: to.signedness,
        }
    }
}

fn coerce_const_to_type(value: u128, from: BitType, to: BitType) -> i128 {
    let mut bits = value & value_mask(from.width);
    if from.width < to.width && from.is_signed() && from.width > 0 {
        let sign_bit = if from.width >= 128 {
            1u128 << 127
        } else {
            1u128 << (from.width - 1)
        };
        if bits & sign_bit != 0 {
            bits |= !value_mask(from.width);
        }
    }
    bits &= value_mask(to.width);
    if to.is_signed() {
        signed_const_bits(bits, to.width)
    } else {
        bits as i128
    }
}

fn signed_const_bits(bits: u128, width: Width) -> i128 {
    if width == 0 {
        return 0;
    }
    if width >= 128 {
        return bits as i128;
    }
    let bits = bits & value_mask(width);
    let sign_bit = 1u128 << (width - 1);
    if bits & sign_bit == 0 {
        bits as i128
    } else {
        (bits as i128) - (1i128 << width)
    }
}

fn lower_lvalue_assignment_expr(
    dst: &SvLvalue,
    expr: &SvExpr,
    base: Expr,
    signal: Signal,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
) -> Result<Expr, ErrorReport> {
    if matches!(dst, SvLvalue::Signal(_)) {
        return lower_expr_for_signal(expr, signal, symbols, consts);
    }
    let value = lower_expr(expr, symbols, consts)?;
    lower_lvalue_assignment_value(dst, value, base, signal, symbols, consts)
}

fn lower_expr_for_signal(
    expr: &SvExpr,
    dst: Signal,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
) -> Result<Expr, ErrorReport> {
    let value = lower_expr(expr, symbols, consts)?;
    let expr_ty = sv_expr_type(expr, symbols, consts)?;
    let dst_ty = symbols
        .values()
        .find(|symbol| symbol.signal == dst)
        .map(|symbol| symbol.ty)
        .ok_or_else(|| err("E_SV_LVALUE", "unknown assignment target"))?;
    Ok(coerce_expr_to_type(value, expr_ty, dst_ty))
}

fn lower_lvalue_assignment_value(
    dst: &SvLvalue,
    value: Expr,
    base: Expr,
    signal: Signal,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
) -> Result<Expr, ErrorReport> {
    match dst {
        SvLvalue::Signal(_) => coerce_expr_to_signal_type(value, signal, symbols),
        SvLvalue::Bit { index, .. } => {
            let index = const_index(index, consts, "E_SV_LVALUE")?;
            splice_lvalue_expr(base, signal, index, 1, value, symbols)
        }
        SvLvalue::Slice { msb, lsb, .. } => {
            let msb = const_index(msb, consts, "E_SV_LVALUE")?;
            let lsb = const_index(lsb, consts, "E_SV_LVALUE")?;
            if msb < lsb {
                return Err(err("E_SV_LVALUE", "part-select msb must be >= lsb"));
            }
            splice_lvalue_expr(base, signal, lsb, msb - lsb + 1, value, symbols)
        }
        SvLvalue::Memory { .. } => Err(err(
            "E_SV_LVALUE",
            "memory lvalue is only valid inside enabled always_ff",
        )),
    }
}

fn splice_lvalue_expr(
    base: Expr,
    signal: Signal,
    lsb: Width,
    width: Width,
    value: Expr,
    symbols: &HashMap<String, SvSymbol>,
) -> Result<Expr, ErrorReport> {
    let signal_width = symbols
        .values()
        .find(|symbol| symbol.signal == signal)
        .map(|symbol| symbol.ty.width)
        .ok_or_else(|| err("E_SV_LVALUE", "unknown lvalue signal"))?;
    if width == 0 || lsb + width > signal_width {
        return Err(err("E_SV_LVALUE", "selected lvalue exceeds signal width"));
    }
    let value = coerce_expr_to_type(value, uint(width), uint(width));
    let mut parts = Vec::new();
    let high_lsb = lsb + width;
    if high_lsb < signal_width {
        parts.push(base.clone().slice(high_lsb, signal_width - high_lsb));
    }
    parts.push(value.slice(0, width));
    if lsb > 0 {
        parts.push(base.slice(0, lsb));
    }
    let spliced = if parts.len() == 1 {
        parts.pop().unwrap()
    } else {
        concat(parts)
    };
    coerce_expr_to_signal_type(spliced, signal, symbols)
}

fn coerce_expr_to_signal_type(
    expr: Expr,
    dst: Signal,
    symbols: &HashMap<String, SvSymbol>,
) -> Result<Expr, ErrorReport> {
    let Some(symbol) = symbols.values().find(|symbol| symbol.signal == dst) else {
        return Ok(expr);
    };
    if symbol.ty.is_signed() {
        Ok(as_sint(expr))
    } else {
        Ok(expr)
    }
}

#[derive(Default)]
struct InitialState {
    order: Vec<Signal>,
    values: HashMap<Signal, i128>,
}

impl InitialState {
    fn set(&mut self, signal: Signal, value: i128) {
        if !self.values.contains_key(&signal) {
            self.order.push(signal);
        }
        self.values.insert(signal, value);
    }

    fn get(&self, signal: Signal) -> i128 {
        self.values.get(&signal).copied().unwrap_or(0)
    }

    fn emit(self, builder: &mut rrtl_core::ModuleBuilder<'_>) {
        for signal in self.order {
            if let Some(value) = self.values.get(&signal).copied() {
                builder.initial(signal, value);
            }
        }
    }
}

fn lower_initial_assign(
    assign: &SvInitialAssign,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
    initial_state: &mut InitialState,
) -> Result<(), ErrorReport> {
    let value = const_value(&assign.expr, consts)?;
    match &assign.dst {
        SvLvalue::Signal(name) => {
            let symbol = symbols
                .get(name)
                .filter(|symbol| !symbol.is_memory)
                .ok_or_else(|| err("E_SV_INITIAL", format!("unknown initial target `{name}`")))?;
            initial_state.set(symbol.signal, value);
            Ok(())
        }
        SvLvalue::Memory { name, addr } => {
            let symbol = symbols
                .get(name)
                .filter(|symbol| symbol.is_memory)
                .ok_or_else(|| err("E_SV_INITIAL", format!("unknown memory `{name}`")))?;
            let addr = const_eval(addr, consts)?;
            let addr = usize::try_from(addr)
                .map_err(|_| err("E_SV_INITIAL", "initial memory address exceeds usize range"))?;
            builder.mem_init(symbol.signal, addr, value);
            Ok(())
        }
        SvLvalue::Bit { name, index } => {
            let symbol = symbols
                .get(name)
                .ok_or_else(|| err("E_SV_INITIAL", format!("unknown initial target `{name}`")))?;
            if symbol.is_memory {
                let addr = const_eval(index, consts)?;
                let addr = usize::try_from(addr).map_err(|_| {
                    err("E_SV_INITIAL", "initial memory address exceeds usize range")
                })?;
                builder.mem_init(symbol.signal, addr, value);
            } else {
                let index = const_index(index, consts, "E_SV_LVALUE")?;
                let base = initial_state.get(symbol.signal);
                let value = splice_initial_value(base, symbol.ty.width, index, 1, value)?;
                initial_state.set(symbol.signal, value);
            }
            Ok(())
        }
        SvLvalue::Slice { name, msb, lsb } => {
            let symbol = symbols
                .get(name)
                .filter(|symbol| !symbol.is_memory)
                .ok_or_else(|| err("E_SV_INITIAL", format!("unknown initial target `{name}`")))?;
            let msb = const_index(msb, consts, "E_SV_LVALUE")?;
            let lsb = const_index(lsb, consts, "E_SV_LVALUE")?;
            if msb < lsb {
                return Err(err("E_SV_LVALUE", "part-select msb must be >= lsb"));
            }
            let base = initial_state.get(symbol.signal);
            let value = splice_initial_value(base, symbol.ty.width, lsb, msb - lsb + 1, value)?;
            initial_state.set(symbol.signal, value);
            Ok(())
        }
    }
}

fn splice_initial_value(
    base: i128,
    signal_width: Width,
    lsb: Width,
    width: Width,
    value: i128,
) -> Result<i128, ErrorReport> {
    let selected_end = lsb
        .checked_add(width)
        .ok_or_else(|| err("E_SV_LVALUE", "selected lvalue exceeds signal width"))?;
    if width == 0 || selected_end > signal_width {
        return Err(err("E_SV_LVALUE", "selected lvalue exceeds signal width"));
    }
    if selected_end > 128 {
        return Err(err(
            "E_SV_INITIAL",
            "selected initial assignments beyond 128 bits are not supported",
        ));
    }
    let signal_mask = value_mask(signal_width);
    let selected_mask = value_mask(width) << lsb;
    let selected = ((value as u128) & value_mask(width)) << lsb;
    let spliced = ((base as u128) & signal_mask & !selected_mask) | selected;
    Ok((spliced & signal_mask) as i128)
}

fn value_mask(width: Width) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

fn signal_lvalue_name(
    dst: &SvLvalue,
    symbols: &HashMap<String, SvSymbol>,
) -> Result<String, ErrorReport> {
    match dst {
        SvLvalue::Signal(name) | SvLvalue::Bit { name, .. } | SvLvalue::Slice { name, .. } => {
            symbols
                .get(name)
                .filter(|symbol| !symbol.is_memory)
                .map(|_| name.clone())
                .ok_or_else(|| err("E_SV_LVALUE", format!("unknown signal `{name}`")))
        }
        SvLvalue::Memory { .. } => Err(err(
            "E_SV_LVALUE",
            "memory lvalue is only valid inside enabled always_ff",
        )),
    }
}

fn lvalue_is_memory(dst: &SvLvalue, symbols: &HashMap<String, SvSymbol>) -> bool {
    match dst {
        SvLvalue::Memory { name, .. } | SvLvalue::Bit { name, .. } => {
            symbols.get(name).is_some_and(|symbol| symbol.is_memory)
        }
        SvLvalue::Signal(_) | SvLvalue::Slice { .. } => false,
    }
}

fn memory_lvalue_parts(dst: &SvLvalue) -> Result<(&str, &SvExpr), ErrorReport> {
    match dst {
        SvLvalue::Memory { name, addr } => Ok((name, addr)),
        SvLvalue::Bit { name, index } => Ok((name, index)),
        _ => Err(err("E_SV_MEMORY", "expected memory lvalue")),
    }
}

fn const_index(
    expr: &SvExpr,
    consts: &HashMap<String, u128>,
    code: &'static str,
) -> Result<Width, ErrorReport> {
    let value = const_eval(expr, consts).map_err(|_| err(code, "index must be constant"))?;
    Width::try_from(value).map_err(|_| err(code, "constant index exceeds width range"))
}

fn const_lvalue_assignment_value(
    dst: &SvLvalue,
    expr: &SvExpr,
    current: i128,
    symbols: &HashMap<String, SvSymbol>,
    consts: &HashMap<String, u128>,
) -> Result<i128, ErrorReport> {
    match dst {
        SvLvalue::Signal(name) => {
            let symbol = symbols
                .get(name)
                .filter(|symbol| !symbol.is_memory)
                .ok_or_else(|| err("E_SV_RESET", format!("unknown reset target `{name}`")))?;
            let value = const_eval(expr, consts)?;
            let expr_ty = sv_expr_type(expr, symbols, consts)?;
            Ok(coerce_const_to_type(value, expr_ty, symbol.ty))
        }
        SvLvalue::Bit { name, index } => {
            let index = const_index(index, consts, "E_SV_RESET")?;
            let symbol = symbols
                .get(name)
                .filter(|symbol| !symbol.is_memory)
                .ok_or_else(|| err("E_SV_RESET", format!("unknown reset target `{name}`")))?;
            if index >= symbol.ty.width {
                return Err(err("E_SV_RESET", "reset bit-select exceeds signal width"));
            }
            let bit = const_eval(expr, consts)? & 1;
            let mask = 1i128
                .checked_shl(index)
                .ok_or_else(|| err("E_SV_RESET", "reset bit-select exceeds i128 range"))?;
            Ok((current & !mask) | ((bit as i128) << index))
        }
        SvLvalue::Slice { name, msb, lsb } => {
            let msb = const_index(msb, consts, "E_SV_RESET")?;
            let lsb = const_index(lsb, consts, "E_SV_RESET")?;
            if msb < lsb {
                return Err(err("E_SV_RESET", "reset part-select msb must be >= lsb"));
            }
            let symbol = symbols
                .get(name)
                .filter(|symbol| !symbol.is_memory)
                .ok_or_else(|| err("E_SV_RESET", format!("unknown reset target `{name}`")))?;
            if msb >= symbol.ty.width {
                return Err(err("E_SV_RESET", "reset part-select exceeds signal width"));
            }
            let width = msb - lsb + 1;
            if width >= 128 {
                return Err(err(
                    "E_SV_RESET",
                    "reset part-selects wider than 127 bits are not supported",
                ));
            }
            let mask = ((1i128 << width) - 1) << lsb;
            let value = (const_eval(expr, consts)? as i128) & ((1i128 << width) - 1);
            Ok((current & !mask) | (value << lsb))
        }
        SvLvalue::Memory { .. } => Err(err(
            "E_SV_RESET",
            "reset branch must assign registers, not memories",
        )),
    }
}

fn reset_condition(expr: &SvExpr) -> Option<(String, bool)> {
    match expr {
        SvExpr::Ident(name) => Some((name.clone(), false)),
        SvExpr::Unary {
            op: SvUnaryOp::Not | SvUnaryOp::BitNot,
            expr,
        } => match expr.as_ref() {
            SvExpr::Ident(name) => Some((name.clone(), true)),
            _ => None,
        },
        _ => None,
    }
}

fn for_loop_iteration_consts(
    var: &str,
    init: &SvExpr,
    cmp: SvForCmp,
    bound: &SvExpr,
    step: &SvForStep,
    consts: &HashMap<String, u128>,
) -> Result<Vec<HashMap<String, u128>>, ErrorReport> {
    let mut current = const_eval_i128(init, consts, "for-loop initial value")?;
    let bound = const_eval_i128(bound, consts, "for-loop bound")?;
    let step = eval_for_step(step, consts)?;
    if step == 0 {
        return Err(err("E_SV_FOR", "for-loop step must be nonzero"));
    }

    let mut iterations = Vec::new();
    while for_condition_holds(current, cmp, bound) {
        if iterations.len() >= SV_FOR_UNROLL_LIMIT {
            return Err(err(
                "E_SV_FOR",
                format!("for-loop exceeds unroll limit of {SV_FOR_UNROLL_LIMIT} iterations"),
            ));
        }
        let value = u128::try_from(current).map_err(|_| {
            err(
                "E_SV_FOR",
                "negative for-loop variable values are not supported",
            )
        })?;
        let mut iter_consts = consts.clone();
        iter_consts.insert(var.to_string(), value);
        iterations.push(iter_consts);
        current = current
            .checked_add(step)
            .ok_or_else(|| err("E_SV_FOR", "for-loop variable overflowed during unroll"))?;
    }
    Ok(iterations)
}

fn eval_for_step(step: &SvForStep, consts: &HashMap<String, u128>) -> Result<i128, ErrorReport> {
    match step {
        SvForStep::Inc => Ok(1),
        SvForStep::Dec => Ok(-1),
        SvForStep::Add(expr) => const_eval_i128(expr, consts, "for-loop step"),
        SvForStep::Sub(expr) => const_eval_i128(expr, consts, "for-loop step").map(|value| -value),
    }
}

fn const_eval_i128(
    expr: &SvExpr,
    consts: &HashMap<String, u128>,
    context: &str,
) -> Result<i128, ErrorReport> {
    let value = const_eval(expr, consts).map_err(|_| {
        err(
            "E_SV_FOR",
            format!("{context} must be a supported constant expression"),
        )
    })?;
    i128::try_from(value).map_err(|_| err("E_SV_FOR", format!("{context} exceeds i128 range")))
}

fn for_condition_holds(current: i128, cmp: SvForCmp, bound: i128) -> bool {
    match cmp {
        SvForCmp::Lt => current < bound,
        SvForCmp::Le => current <= bound,
        SvForCmp::Gt => current > bound,
        SvForCmp::Ge => current >= bound,
    }
}

fn const_value(expr: &SvExpr, consts: &HashMap<String, u128>) -> Result<i128, ErrorReport> {
    const_eval(expr, consts).map(|value| value as i128)
}

fn const_eval(expr: &SvExpr, consts: &HashMap<String, u128>) -> Result<u128, ErrorReport> {
    match expr {
        SvExpr::Ident(name) => consts
            .get(name)
            .copied()
            .ok_or_else(|| err("E_SV_CONST", format!("unknown constant `{name}`"))),
        SvExpr::Lit { value, .. } => Ok(*value),
        SvExpr::Unary { op, expr } => {
            let value = const_eval(expr, consts)?;
            Ok(match op {
                SvUnaryOp::Not => u128::from(value == 0),
                SvUnaryOp::BitNot => !value,
                SvUnaryOp::Neg => value.wrapping_neg(),
            })
        }
        SvExpr::Binary { op, lhs, rhs } => {
            let lhs = const_eval(lhs, consts)?;
            let rhs = const_eval(rhs, consts)?;
            Ok(match op {
                SvBinaryOp::Add => lhs.wrapping_add(rhs),
                SvBinaryOp::Sub => lhs.wrapping_sub(rhs),
                SvBinaryOp::Mul => lhs.wrapping_mul(rhs),
                SvBinaryOp::And | SvBinaryOp::LogAnd => lhs & rhs,
                SvBinaryOp::Or | SvBinaryOp::LogOr => lhs | rhs,
                SvBinaryOp::Xor => lhs ^ rhs,
                SvBinaryOp::Eq => u128::from(lhs == rhs),
                SvBinaryOp::Ne => u128::from(lhs != rhs),
                SvBinaryOp::Lt => u128::from(lhs < rhs),
                SvBinaryOp::Le => u128::from(lhs <= rhs),
                SvBinaryOp::Gt => u128::from(lhs > rhs),
                SvBinaryOp::Ge => u128::from(lhs >= rhs),
            })
        }
        SvExpr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            if const_eval(cond, consts)? != 0 {
                const_eval(then_expr, consts)
            } else {
                const_eval(else_expr, consts)
            }
        }
        SvExpr::Cast { expr, .. } => const_eval(expr, consts),
        SvExpr::Concat(parts) => const_eval_concat(parts, consts),
        SvExpr::Repeat { count, expr } => const_eval_repeat(*count, expr, consts),
        SvExpr::Index { expr, index } => const_eval_select(expr, *index, 1, consts),
        SvExpr::Slice { expr, msb, lsb } => {
            if msb < lsb {
                return Err(err("E_SV_CONST", "constant part-select msb must be >= lsb"));
            }
            const_eval_select(expr, *lsb, msb - lsb + 1, consts)
        }
        SvExpr::Bracket { expr, index } => {
            let index = const_eval(index, consts)?;
            let index = Width::try_from(index).map_err(|_| {
                err(
                    "E_SV_CONST",
                    "constant bit-select index exceeds width range",
                )
            })?;
            const_eval_select(expr, index, 1, consts)
        }
        _ => Err(err(
            "E_SV_CONST",
            format!("expression is not a supported constant: `{expr:?}`"),
        )),
    }
}

fn const_eval_concat(
    parts: &[SvExpr],
    consts: &HashMap<String, u128>,
) -> Result<u128, ErrorReport> {
    let mut total_width: Width = 0;
    for part in parts {
        total_width = total_width
            .checked_add(const_expr_width(part, consts)?)
            .ok_or_else(|| err("E_SV_CONST", "constant concat width exceeds range"))?;
        if total_width > 128 {
            return Err(err("E_SV_CONST", "constant concat exceeds 128 bits"));
        }
    }

    let mut value = 0u128;
    let mut packed_width = 0;
    for part in parts {
        let width = const_expr_width(part, consts)?;
        let part_value = const_eval(part, consts)? & value_mask(width);
        value = append_const_part(value, packed_width, part_value, width)?;
        packed_width += width;
    }
    Ok(value)
}

fn const_eval_repeat(
    count: Width,
    expr: &SvExpr,
    consts: &HashMap<String, u128>,
) -> Result<u128, ErrorReport> {
    let width = const_expr_width(expr, consts)?;
    let total_width = width
        .checked_mul(count)
        .ok_or_else(|| err("E_SV_CONST", "constant repeat width exceeds range"))?;
    if total_width > 128 {
        return Err(err("E_SV_CONST", "constant repeat exceeds 128 bits"));
    }

    let part_value = const_eval(expr, consts)? & value_mask(width);
    let mut value = 0u128;
    let mut packed_width = 0;
    for _ in 0..count {
        value = append_const_part(value, packed_width, part_value, width)?;
        packed_width += width;
    }
    Ok(value)
}

fn append_const_part(
    current: u128,
    current_width: Width,
    part: u128,
    part_width: Width,
) -> Result<u128, ErrorReport> {
    let total_width = current_width
        .checked_add(part_width)
        .ok_or_else(|| err("E_SV_CONST", "constant packed width exceeds range"))?;
    if total_width > 128 {
        return Err(err(
            "E_SV_CONST",
            "constant packed expression exceeds 128 bits",
        ));
    }
    if part_width == 0 {
        return Ok(current);
    }
    if part_width == 128 {
        if current_width == 0 {
            return Ok(part);
        }
        return Err(err(
            "E_SV_CONST",
            "constant packed expression exceeds 128 bits",
        ));
    }
    Ok((current << part_width) | (part & value_mask(part_width)))
}

fn const_eval_select(
    expr: &SvExpr,
    lsb: Width,
    width: Width,
    consts: &HashMap<String, u128>,
) -> Result<u128, ErrorReport> {
    let expr_width = const_expr_width(expr, consts)?;
    let selected_end = lsb
        .checked_add(width)
        .ok_or_else(|| err("E_SV_CONST", "constant select exceeds width range"))?;
    if width == 0 || selected_end > expr_width {
        return Err(err(
            "E_SV_CONST",
            "constant select exceeds expression width",
        ));
    }
    if selected_end > 128 {
        return Err(err(
            "E_SV_CONST",
            "constant select beyond 128 bits is not supported",
        ));
    }
    Ok((const_eval(expr, consts)? >> lsb) & value_mask(width))
}

fn const_expr_width(expr: &SvExpr, consts: &HashMap<String, u128>) -> Result<Width, ErrorReport> {
    match expr {
        SvExpr::Ident(_) => Ok(32),
        SvExpr::Lit { width, .. } => Ok(*width),
        SvExpr::Unary { expr, .. } | SvExpr::Cast { expr, .. } => const_expr_width(expr, consts),
        SvExpr::Binary { lhs, .. } => const_expr_width(lhs, consts),
        SvExpr::Ternary { then_expr, .. } => const_expr_width(then_expr, consts),
        SvExpr::Concat(parts) => {
            let mut width: Width = 0;
            for part in parts {
                width = width
                    .checked_add(const_expr_width(part, consts)?)
                    .ok_or_else(|| err("E_SV_CONST", "constant concat width exceeds range"))?;
            }
            Ok(width)
        }
        SvExpr::Repeat { count, expr } => const_expr_width(expr, consts)?
            .checked_mul(*count)
            .ok_or_else(|| err("E_SV_CONST", "constant repeat width exceeds range")),
        SvExpr::Index { .. } | SvExpr::Bracket { .. } => Ok(1),
        SvExpr::Slice { msb, lsb, .. } => {
            if msb < lsb {
                return Err(err("E_SV_CONST", "constant part-select msb must be >= lsb"));
            }
            Ok(msb - lsb + 1)
        }
        SvExpr::MemRead { .. } => Err(err(
            "E_SV_CONST",
            "memory reads are not supported constant expressions",
        )),
    }
}

fn sv_type_to_rrtl(ty: SvType) -> BitType {
    if ty.signed {
        sint(ty.width)
    } else {
        uint(ty.width)
    }
}

fn select_top(source: &SvSource, top: Option<&str>) -> Result<String, ErrorReport> {
    if let Some(top) = top {
        if source.modules.iter().any(|module| module.name == top) {
            return Ok(top.to_string());
        }
        return Err(err(
            "E_SV_TOP",
            format!("top module `{top}` is not present in source"),
        ));
    }
    if source.modules.len() == 1 {
        Ok(source.modules[0].name.clone())
    } else {
        Err(err(
            "E_SV_TOP",
            "multiple modules present; pass an explicit top module",
        ))
    }
}

fn ceil_log2(value: usize) -> Width {
    if value <= 1 {
        1
    } else {
        usize::BITS - (value - 1).leading_zeros()
    }
}

fn err(code: &'static str, message: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![Diagnostic::new(code, message)])
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
                tokens.push(Token::Num(self.slice(start)));
            } else if ch == '"' {
                tokens.push(Token::Str(self.read_string()?));
            } else {
                let two = self.peek_two();
                let sym = match two.as_deref() {
                    Some("<=" | ">=" | "==" | "!=" | "&&" | "||" | "<<" | ">>" | "++" | "--") => {
                        self.pos += 2;
                        two.unwrap()
                    }
                    _ => {
                        self.pos += 1;
                        ch.to_string()
                    }
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

    fn slice(&self, start: usize) -> String {
        self.chars[start..self.pos].iter().collect()
    }
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    consts: HashMap<String, u128>,
    types: HashMap<String, SvType>,
    module_param_overrides: HashMap<String, HashMap<String, u128>>,
    active_param_overrides: HashMap<String, u128>,
}

impl Parser {
    fn new(source: &str) -> Result<Self, ErrorReport> {
        Self::new_with_module_param_overrides(source, HashMap::new())
    }

    fn new_with_module_param_overrides(
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
            module_param_overrides,
            active_param_overrides: HashMap::new(),
        })
    }

    fn parse_source(&mut self) -> Result<SvSource, ErrorReport> {
        let mut modules = Vec::new();
        while !self.is_eof() {
            modules.push(self.parse_module()?);
        }
        Ok(SvSource { modules })
    }

    fn parse_module(&mut self) -> Result<SvModule, ErrorReport> {
        self.consts.clear();
        self.types.clear();
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
        let params = if self.eat_sym("#") {
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
        Ok(SvModule {
            name,
            ports,
            params,
            items,
        })
    }

    fn parse_item(&mut self) -> Result<SvItem, ErrorReport> {
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

        let step_var = self.expect_ident()?;
        if step_var != var {
            return Err(err(
                "E_SV_GENERATE",
                format!("generate-for update must update genvar `{var}`"),
            ));
        }
        let step = self.parse_for_step(&var).map_err(|_| {
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
        self.expect_ident_value("enum")?;
        self.expect_ident_value("logic")?;
        let width = self.parse_optional_range()?.unwrap_or(1);
        self.expect_sym("{")?;
        loop {
            let _variant = self.expect_ident()?;
            self.expect_sym("=")?;
            let _value = self.parse_expr()?;
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
        Ok(SvTypeDef { name, ty })
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
            let evaluated = const_eval(&default_value, &self.consts)?;
            (default_value, evaluated)
        };
        self.consts.insert(name.clone(), evaluated);
        Ok(SvParam { name, value })
    }

    fn skip_optional_param_type(&mut self) -> Result<(), ErrorReport> {
        if self.eat_ident_value("int") || self.eat_ident_value("integer") {
            return Ok(());
        }
        if self.peek_ident_value("logic")
            || self.peek_ident_value("wire")
            || self.peek_ident_value("reg")
        {
            self.pos += 1;
            self.eat_ident_value("signed");
            let _ = self.parse_optional_range()?;
        }
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
        let width = self.parse_optional_range()?.unwrap_or(1);
        let mut names = Vec::new();
        loop {
            let name = self.expect_ident()?;
            names.push(SvDeclarator {
                name,
                memory_depth: None,
            });
            if !self.eat_sym(",") {
                break;
            }
            if self.peek_sym(")") || self.starts_decl() {
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
        let kind = if self.eat_ident_value("wire") {
            SvDeclKind::Wire
        } else if self.eat_ident_value("reg") {
            SvDeclKind::Reg
        } else if direction.is_some() {
            self.eat_ident_value("logic");
            SvDeclKind::Logic
        } else {
            self.expect_ident_value("logic")?;
            SvDeclKind::Logic
        };
        let signed = self.eat_ident_value("signed");
        let width = self.parse_optional_range()?.unwrap_or(1);
        let mut names = Vec::new();
        loop {
            let name = self.expect_ident()?;
            let memory_depth = if self.eat_sym("[") {
                let first = self.expect_usize_const()?;
                self.expect_sym(":")?;
                let second = self.expect_usize_const()?;
                self.expect_sym("]")?;
                if first == 0 && second > 0 {
                    Some(second + 1)
                } else if second == 0 && first > 0 {
                    Some(first + 1)
                } else {
                    return Err(err(
                        "E_SV_MEMORY_RANGE",
                        "memory ranges must use [0:D-1] or [D-1:0] depth syntax",
                    ));
                }
            } else {
                None
            };
            names.push(SvDeclarator { name, memory_depth });
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
        let type_name = self.expect_ident()?;
        let ty = *self.types.get(&type_name).ok_or_else(|| {
            err(
                "E_SV_TYPE",
                format!("unknown SystemVerilog type `{type_name}`"),
            )
        })?;
        let mut names = Vec::new();
        loop {
            names.push(SvDeclarator {
                name: self.expect_ident()?,
                memory_depth: None,
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

    fn parse_instance(&mut self) -> Result<SvInstance, ErrorReport> {
        let module = self.expect_ident()?;
        let params = if self.eat_sym("#") {
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
                self.expect_sym("(")?;
                let expr = self.parse_expr()?;
                self.expect_sym(")")?;
                connections.push(SvConnection { port, expr });
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
        self.expect_ident_value("begin")?;
        let mut assigns = Vec::new();
        while !self.eat_ident_value("end") {
            let dst = self.parse_lvalue()?;
            self.expect_sym("=")?;
            let expr = self.parse_expr()?;
            self.expect_sym(";")?;
            assigns.push(SvInitialAssign { dst, expr });
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

        let step_var = self.expect_ident()?;
        if step_var != var {
            return Err(err(
                "E_SV_FOR",
                format!("for-loop update must update loop variable `{var}`"),
            ));
        }
        let step = self.parse_for_step(&var)?;
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
        if self.eat_sym("[") {
            let first = self.parse_expr()?;
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
        } else {
            self.parse_postfix()
        }
    }

    fn parse_postfix(&mut self) -> Result<SvExpr, ErrorReport> {
        let mut expr = self.parse_primary()?;
        loop {
            if !self.eat_sym("[") {
                break;
            }
            let first = self.parse_expr()?;
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
            Token::Ident(name) => Ok(SvExpr::Ident(name)),
            Token::Num(raw) => parse_number(&raw),
            Token::Sym(sym) if sym == "$" => {
                let func = self.expect_ident()?;
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
                format!("expected expression, found `{}`", token.display()),
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

    fn starts_decl(&self) -> bool {
        self.peek_ident_value("input")
            || self.peek_ident_value("output")
            || self.peek_ident_value("inout")
            || self.peek_ident_value("logic")
            || self.peek_ident_value("wire")
            || self.peek_ident_value("reg")
    }

    fn peek_typedef_type(&self) -> bool {
        matches!(self.peek(), Token::Ident(value) if self.types.contains_key(value))
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
            "+" => (SvBinaryOp::Add, 8),
            "-" => (SvBinaryOp::Sub, 8),
            "*" => (SvBinaryOp::Mul, 9),
            _ => return None,
        };
        Some((op, bp, bp + 1))
    }

    fn expect_ident(&mut self) -> Result<String, ErrorReport> {
        match self.next().clone() {
            Token::Ident(value) => Ok(value),
            token => Err(err(
                "E_SV_PARSE",
                format!("expected identifier, found `{}`", token.display()),
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
                format!("expected `{expected}`, found `{}`", self.peek().display()),
            ))
        }
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

fn parse_number(raw: &str) -> Result<SvExpr, ErrorReport> {
    let (value, width, signed) = parse_number_parts(raw)?;
    Ok(SvExpr::Lit {
        value,
        width,
        signed,
    })
}

fn parse_number_parts(raw: &str) -> Result<(u128, Width, bool), ErrorReport> {
    let raw = raw.replace('_', "");
    if let Some((width_raw, rest)) = raw.split_once('\'') {
        let width = width_raw
            .parse::<Width>()
            .map_err(|_| err("E_SV_CONST", format!("invalid literal width `{width_raw}`")))?;
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
        let digits = chars.collect::<String>();
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

#[cfg(test)]
mod tests {
    use super::*;
    use rrtl_core::{
        bundle_type, field, iface_input, iface_output, interface_type, lit_s, lit_u as lit, nested,
        rv_bundle, rv_scalar, sint, uint, zext, Design, SignalKind, Simulator,
    };

    fn signal(imported: &SvImport, module: &str, name: &str) -> Signal {
        let compiled = compile(&imported.design).unwrap();
        compiled
            .find_module(module)
            .unwrap()
            .signals
            .iter()
            .find(|signal| signal.name == name)
            .unwrap()
            .handle
    }

    fn reimport_emitted(design: &Design, top: &str) -> SvImport {
        let sv = rrtl_sv::emit(design).unwrap();
        import_sv(&sv, Some(top)).unwrap_or_else(|err| panic!("failed to reimport:\n{sv}\n{err}"))
    }

    #[test]
    fn imports_combinational_alu() {
        let imported = import_sv(
            r#"
            module Alu(a, b, sum, eq);
              input logic [7:0] a, b;
              output logic [7:0] sum;
              output logic eq;
              assign sum = a + b;
              assign eq = a == b;
            endmodule
            "#,
            None,
        )
        .unwrap();
        assert_eq!(imported.top_name, "Alu");
        let mut sim = Simulator::new(&imported.design, "Alu").unwrap();
        let a = signal(&imported, "Alu", "a");
        let b = signal(&imported, "Alu", "b");
        let sum = signal(&imported, "Alu", "sum");
        let eq = signal(&imported, "Alu", "eq");
        sim.set(a, 3);
        sim.set(b, 4);
        assert_eq!(sim.get(sum), 7);
        assert_eq!(sim.get(eq), 0);
    }

    #[test]
    fn imports_counter_with_async_active_low_reset() {
        let imported = import_sv(
            r#"
            module Counter(clk, rst_n, en, out);
              input logic clk, rst_n, en;
              output logic [3:0] out;
              logic [3:0] count;
              assign out = count;
              always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) count <= 4'd0;
                else if (en) count <= count + 4'd1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "Counter").unwrap();
        let rst_n = signal(&imported, "Counter", "rst_n");
        let en = signal(&imported, "Counter", "en");
        let out = signal(&imported, "Counter", "out");
        sim.set(rst_n, 0);
        sim.tick();
        sim.set(rst_n, 1);
        sim.set(en, 1);
        sim.tick();
        assert_eq!(sim.get(out), 1);
    }

    #[test]
    fn imports_simple_memory_write_read() {
        let imported = import_sv(
            r#"
            module Mem(clk, we, waddr, wdata, raddr, rdata);
              input logic clk, we;
              input logic [1:0] waddr, raddr;
              input logic [7:0] wdata;
              output logic [7:0] rdata;
              logic [7:0] regs [0:3];
              assign rdata = regs[raddr];
              always_ff @(posedge clk) begin
                if (we) regs[waddr] <= wdata;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "Mem").unwrap();
        let we = signal(&imported, "Mem", "we");
        let waddr = signal(&imported, "Mem", "waddr");
        let wdata = signal(&imported, "Mem", "wdata");
        let raddr = signal(&imported, "Mem", "raddr");
        let rdata = signal(&imported, "Mem", "rdata");
        sim.set(we, 1);
        sim.set(waddr, 2);
        sim.set(wdata, 0xab);
        sim.tick();
        sim.set(we, 0);
        sim.set(raddr, 2);
        assert_eq!(sim.get(rdata), 0xab);
    }

    #[test]
    fn imports_ansi_port_alu() {
        let imported = import_sv(
            r#"
            module AnsiAlu(
              input logic [7:0] a, b,
              output logic [7:0] sum,
              output logic eq
            );
              assign sum = a + b;
              assign eq = a == b;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "AnsiAlu").unwrap();
        let a = signal(&imported, "AnsiAlu", "a");
        let b = signal(&imported, "AnsiAlu", "b");
        let sum = signal(&imported, "AnsiAlu", "sum");
        let eq = signal(&imported, "AnsiAlu", "eq");
        sim.set(a, 9);
        sim.set(b, 9);
        assert_eq!(sim.get(sum), 18);
        assert_eq!(sim.get(eq), 1);
    }

    #[test]
    fn imports_parameterized_width_counter() {
        let imported = import_sv(
            r#"
            module ParamCounter #(parameter int WIDTH = 4) (
              input logic clk,
              input logic en,
              output logic [WIDTH-1:0] out
            );
              logic [WIDTH-1:0] count;
              assign out = count;
              always_ff @(posedge clk) begin
                if (en) count <= count + 4'd1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "ParamCounter").unwrap();
        let en = signal(&imported, "ParamCounter", "en");
        let out = signal(&imported, "ParamCounter", "out");
        sim.set(en, 1);
        sim.tick();
        sim.tick();
        assert_eq!(sim.get(out), 2);
    }

    #[test]
    fn imports_parameterized_instance_overrides_with_multiple_specializations() {
        let imported = import_sv(
            r#"
            module WidthChild #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a ^ {WIDTH{1'b1}};
            endmodule

            module WidthTop(
              input logic [3:0] a4,
              input logic [7:0] a8,
              output logic [3:0] y4,
              output logic [7:0] y8
            );
              WidthChild #(.WIDTH(4)) u4(.a(a4), .y(y4));
              WidthChild #(.WIDTH(8)) u8(.a(a8), .y(y8));
            endmodule
            "#,
            Some("WidthTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("WidthChild__WIDTH_4").is_some());
        assert!(compiled.find_module("WidthChild__WIDTH_8").is_some());

        let mut sim = Simulator::new(&imported.design, "WidthTop").unwrap();
        let a4 = signal(&imported, "WidthTop", "a4");
        let a8 = signal(&imported, "WidthTop", "a8");
        let y4 = signal(&imported, "WidthTop", "y4");
        let y8 = signal(&imported, "WidthTop", "y8");
        sim.set(a4, 3);
        sim.set(a8, 3);
        assert_eq!(sim.get(y4), 12);
        assert_eq!(sim.get(y8), 252);
    }

    #[test]
    fn imports_nested_parameterized_instance_overrides() {
        let imported = import_sv(
            r#"
            module Leaf #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a ^ {WIDTH{1'b1}};
            endmodule

            module Mid #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              Leaf #(.WIDTH(WIDTH)) u_leaf(.a(a), .y(y));
            endmodule

            module NestedTop(input logic [5:0] a, output logic [5:0] y);
              Mid #(.WIDTH(6)) u_mid(.a(a), .y(y));
            endmodule
            "#,
            Some("NestedTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("Mid__WIDTH_6").is_some());
        assert!(compiled.find_module("Leaf__WIDTH_6").is_some());

        let mut sim = Simulator::new(&imported.design, "NestedTop").unwrap();
        let a = signal(&imported, "NestedTop", "a");
        let y = signal(&imported, "NestedTop", "y");
        sim.set(a, 10);
        assert_eq!(sim.get(y), 53);
    }

    #[test]
    fn imports_parameterized_instance_override_from_parent_localparam() {
        let imported = import_sv(
            r#"
            module LocalWidth #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a ^ {WIDTH{1'b1}};
            endmodule

            module LocalTop(input logic [7:0] a, output logic [7:0] y);
              localparam int BASE = 4;
              LocalWidth #(.WIDTH(BASE * 2)) u_child(.a(a), .y(y));
            endmodule
            "#,
            Some("LocalTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("LocalWidth__WIDTH_8").is_some());

        let mut sim = Simulator::new(&imported.design, "LocalTop").unwrap();
        let a = signal(&imported, "LocalTop", "a");
        let y = signal(&imported, "LocalTop", "y");
        sim.set(a, 2);
        assert_eq!(sim.get(y), 253);
    }

    #[test]
    fn imports_external_instance_parameter_overrides_as_metadata() {
        let imported = import_sv(
            r#"
            module ExtParamTop(input logic [7:0] a, output logic [7:0] y);
              Vendor #(.WIDTH(8)) u_vendor(.a(a), .y(y));
            endmodule
            "#,
            Some("ExtParamTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("Vendor").is_some());
        assert!(compiled.find_module("Vendor__WIDTH_8").is_none());
    }

    #[test]
    fn imports_positional_instance_parameter_override() {
        let imported = import_sv(
            r#"
            module Child #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a;
            endmodule

            module PosTop(input logic [7:0] a, output logic [7:0] y);
              Child #(8) u_child(.a(a), .y(y));
            endmodule
            "#,
            Some("PosTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("Child__WIDTH_8").is_some());
        let mut sim = Simulator::new(&imported.design, "PosTop").unwrap();
        let a = signal(&imported, "PosTop", "a");
        let y = signal(&imported, "PosTop", "y");
        sim.set(a, 0xab);
        assert_eq!(sim.get(y), 0xab);
    }

    #[test]
    fn imports_multiple_positional_instance_parameter_overrides() {
        let imported = import_sv(
            r#"
            module SliceChild #(parameter int WIDTH = 4, parameter int MASK = 0) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a ^ MASK[WIDTH-1:0];
            endmodule

            module PosMultiTop(input logic [7:0] a, output logic [7:0] y);
              SliceChild #(8, 8'hff) u_child(.a(a), .y(y));
            endmodule
            "#,
            Some("PosMultiTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled
            .find_module("SliceChild__MASK_255__WIDTH_8")
            .is_some());
        let mut sim = Simulator::new(&imported.design, "PosMultiTop").unwrap();
        let a = signal(&imported, "PosMultiTop", "a");
        let y = signal(&imported, "PosMultiTop", "y");
        sim.set(a, 0x03);
        assert_eq!(sim.get(y), 0xfc);
    }

    #[test]
    fn imports_nested_positional_instance_parameter_overrides() {
        let imported = import_sv(
            r#"
            module Leaf #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a ^ {WIDTH{1'b1}};
            endmodule

            module Mid #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              Leaf #(WIDTH) u_leaf(.a(a), .y(y));
            endmodule

            module NestedPosTop(input logic [5:0] a, output logic [5:0] y);
              Mid #(6) u_mid(.a(a), .y(y));
            endmodule
            "#,
            Some("NestedPosTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("Mid__WIDTH_6").is_some());
        assert!(compiled.find_module("Leaf__WIDTH_6").is_some());
        let mut sim = Simulator::new(&imported.design, "NestedPosTop").unwrap();
        let a = signal(&imported, "NestedPosTop", "a");
        let y = signal(&imported, "NestedPosTop", "y");
        sim.set(a, 10);
        assert_eq!(sim.get(y), 53);
    }

    #[test]
    fn imports_positional_instance_override_from_parent_localparam() {
        let imported = import_sv(
            r#"
            module LocalWidth #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a ^ {WIDTH{1'b1}};
            endmodule

            module LocalPosTop(input logic [7:0] a, output logic [7:0] y);
              localparam int BASE = 4;
              LocalWidth #(BASE * 2) u_child(.a(a), .y(y));
            endmodule
            "#,
            Some("LocalPosTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("LocalWidth__WIDTH_8").is_some());
        let mut sim = Simulator::new(&imported.design, "LocalPosTop").unwrap();
        let a = signal(&imported, "LocalPosTop", "a");
        let y = signal(&imported, "LocalPosTop", "y");
        sim.set(a, 2);
        assert_eq!(sim.get(y), 253);
    }

    #[test]
    fn rejects_mixed_instance_parameter_overrides() {
        let err = parse_sv(
            r#"
            module Child #(parameter int WIDTH = 4, parameter int MASK = 0) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a;
            endmodule

            module Bad(input logic [7:0] a, output logic [7:0] y);
              Child #(8, .MASK(1)) u_child(.a(a), .y(y));
            endmodule
            "#,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_PARAM_OVERRIDE");
    }

    #[test]
    fn rejects_too_many_positional_instance_parameter_overrides() {
        let err = import_sv(
            r#"
            module Child #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a;
            endmodule

            module Bad(input logic [7:0] a, output logic [7:0] y);
              Child #(8, 2) u_child(.a(a), .y(y));
            endmodule
            "#,
            Some("Bad"),
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_PARAM_OVERRIDE");
    }

    #[test]
    fn rejects_external_positional_instance_parameter_overrides() {
        let err = import_sv(
            r#"
            module Bad(input logic [7:0] a, output logic [7:0] y);
              Vendor #(8) u_vendor(.a(a), .y(y));
            endmodule
            "#,
            Some("Bad"),
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_PARAM_OVERRIDE");
    }

    #[test]
    fn rejects_import_source_defined_parameter_overrides() {
        let source = parse_sv(
            r#"
            module Child #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a;
            endmodule

            module Bad(input logic [7:0] a, output logic [7:0] y);
              Child #(.WIDTH(8)) u_child(.a(a), .y(y));
            endmodule
            "#,
        )
        .unwrap();
        let err = import_source(source, Some("Bad")).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_PARAM_OVERRIDE");
    }

    #[test]
    fn rejects_import_source_defined_positional_parameter_overrides() {
        let source = parse_sv(
            r#"
            module Child #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a;
            endmodule

            module Bad(input logic [7:0] a, output logic [7:0] y);
              Child #(8) u_child(.a(a), .y(y));
            endmodule
            "#,
        )
        .unwrap();
        let err = import_source(source, Some("Bad")).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_PARAM_OVERRIDE");
    }

    #[test]
    fn imports_localparam_memory_depth() {
        let imported = import_sv(
            r#"
            module ParamMem(clk, we, waddr, wdata, raddr, rdata);
              localparam int DEPTH = 4;
              input logic clk, we;
              input logic [1:0] waddr, raddr;
              input logic [7:0] wdata;
              output logic [7:0] rdata;
              logic [7:0] regs [0:DEPTH-1];
              assign rdata = regs[raddr];
              always_ff @(posedge clk) begin
                if (we) regs[waddr] <= wdata;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "ParamMem").unwrap();
        let we = signal(&imported, "ParamMem", "we");
        let waddr = signal(&imported, "ParamMem", "waddr");
        let wdata = signal(&imported, "ParamMem", "wdata");
        let raddr = signal(&imported, "ParamMem", "raddr");
        let rdata = signal(&imported, "ParamMem", "rdata");
        sim.set(we, 1);
        sim.set(waddr, 3);
        sim.set(wdata, 0xcd);
        sim.tick();
        sim.set(we, 0);
        sim.set(raddr, 3);
        assert_eq!(sim.get(rdata), 0xcd);
    }

    #[test]
    fn imports_packed_bit_and_part_selects() {
        let imported = import_sv(
            r#"
            module Selects(
              input logic [7:0] a,
              output logic bit2,
              output logic [3:0] upper
            );
              assign bit2 = a[2];
              assign upper = a[7:4];
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "Selects").unwrap();
        let a = signal(&imported, "Selects", "a");
        let bit2 = signal(&imported, "Selects", "bit2");
        let upper = signal(&imported, "Selects", "upper");
        sim.set(a, 0b1010_0100);
        assert_eq!(sim.get(bit2), 1);
        assert_eq!(sim.get(upper), 0b1010);
    }

    #[test]
    fn imports_localparam_bit_and_part_select_constants() {
        let imported = import_sv(
            r#"
            module ConstSelects(output logic bit2, output logic [2:0] mid);
              localparam int P = 8'b1010_1100;
              localparam int B = P[2];
              localparam int M = P[5:3];
              assign bit2 = B;
              assign mid = M;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "ConstSelects").unwrap();
        let bit2 = signal(&imported, "ConstSelects", "bit2");
        let mid = signal(&imported, "ConstSelects", "mid");
        assert_eq!(sim.get(bit2), 1);
        assert_eq!(sim.get(mid), 0b101);
    }

    #[test]
    fn imports_assignment_truncates_localparam_to_destination_width() {
        let imported = import_sv(
            r#"
            module AssignTrunc(output logic [2:0] y);
              localparam int P = 8'b1010_1100;
              assign y = P;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "AssignTrunc").unwrap();
        let y = signal(&imported, "AssignTrunc", "y");
        assert_eq!(sim.get(y), 0b100);
    }

    #[test]
    fn imports_assignment_zero_extends_unsigned_narrow_expr() {
        let imported = import_sv(
            r#"
            module AssignZext(output logic [7:0] y);
              assign y = 4'b1111;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "AssignZext").unwrap();
        let y = signal(&imported, "AssignZext", "y");
        assert_eq!(sim.get(y), 0x0f);
    }

    #[test]
    fn imports_assignment_sign_extends_signed_narrow_expr() {
        let imported = import_sv(
            r#"
            module AssignSext(output logic signed [7:0] y);
              assign y = $signed(4'b1111);
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "AssignSext").unwrap();
        let y = signal(&imported, "AssignSext", "y");
        assert_eq!(sim.get(y), 0xff);
    }

    #[test]
    fn imports_always_comb_assignment_sizing() {
        let imported = import_sv(
            r#"
            module CombAssignSizing(
              input logic a,
              output logic [2:0] y
            );
              localparam int P = 8'b1010_1100;
              always_comb begin
                if (a) y = P;
                else y = 4'b0010;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "CombAssignSizing").unwrap();
        let a = signal(&imported, "CombAssignSizing", "a");
        let y = signal(&imported, "CombAssignSizing", "y");
        sim.set(a, 0);
        assert_eq!(sim.get(y), 0b010);
        sim.set(a, 1);
        assert_eq!(sim.get(y), 0b100);
    }

    #[test]
    fn imports_always_ff_assignment_sizing() {
        let imported = import_sv(
            r#"
            module FfAssignSizing(
              input logic clk,
              input logic sel,
              output logic [2:0] y
            );
              localparam int P = 8'b1010_1100;
              reg [2:0] q;
              always_ff @(posedge clk) begin
                q <= 4'b0010;
                if (sel) q <= P;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "FfAssignSizing").unwrap();
        let sel = signal(&imported, "FfAssignSizing", "sel");
        let y = signal(&imported, "FfAssignSizing", "y");
        sim.set(sel, 0);
        sim.tick();
        assert_eq!(sim.get(y), 0b010);
        sim.set(sel, 1);
        sim.tick();
        assert_eq!(sim.get(y), 0b100);
    }

    #[test]
    fn imports_localparam_concat_and_repeat_constants() {
        let imported = import_sv(
            r#"
            module ConstPack(output logic [7:0] y, output logic [3:0] ones);
              localparam int C = {2'b10, 2'b01};
              localparam int R = {4{1'b1}};
              assign y = {C[3:0], R[3:0]};
              assign ones = R[3:0];
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "ConstPack").unwrap();
        let y = signal(&imported, "ConstPack", "y");
        let ones = signal(&imported, "ConstPack", "ones");
        assert_eq!(sim.get(y), 0b1001_1111);
        assert_eq!(sim.get(ones), 0b1111);
    }

    #[test]
    fn imports_generate_for_constant_bit_select_bound() {
        let imported = import_sv(
            r#"
            module GenConstSelect(output logic [1:0] y);
              generate
                for (genvar i = 0; i < 2'b10[1:0]; i++) begin : g
                  assign y[i] = i[0];
                end
              endgenerate
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "GenConstSelect").unwrap();
        let y = signal(&imported, "GenConstSelect", "y");
        assert_eq!(sim.get(y), 0b10);
    }

    #[test]
    fn rejects_dynamic_packed_bit_select() {
        let err = import_sv(
            r#"
            module DynamicSelect(
              input logic [7:0] a,
              input logic [2:0] sel,
              output logic y
            );
              assign y = a[sel];
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_CONST");
    }

    #[test]
    fn rejects_constant_bit_select_beyond_expression_width() {
        let err = import_sv(
            r#"
            module BadConstSelect(output logic y);
              localparam int P = 8'b1010_1100[8];
              assign y = P[0];
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_CONST");
    }

    #[test]
    fn rejects_constant_repeat_wider_than_128_bits() {
        let err = import_sv(
            r#"
            module BadConstRepeat(output logic y);
              localparam int P = {129{1'b1}};
              assign y = P[0];
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_CONST");
    }

    #[test]
    fn imports_comb_for_loop_with_localparam_bound() {
        let imported = import_sv(
            r#"
            module CombFor(
              input logic [3:0] mask,
              output logic any
            );
              localparam int N = 4;
              always_comb begin
                any = 1'b0;
                for (int i = 0; i < N; i = i + 1) begin
                  if (mask[i]) any = 1'b1;
                end
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "CombFor").unwrap();
        let mask = signal(&imported, "CombFor", "mask");
        let any = signal(&imported, "CombFor", "any");
        sim.set(mask, 0);
        assert_eq!(sim.get(any), 0);
        sim.set(mask, 0b0100);
        assert_eq!(sim.get(any), 1);
    }

    #[test]
    fn imports_comb_for_loop_decrement() {
        let imported = import_sv(
            r#"
            module CombForDec(
              input logic [3:0] mask,
              output logic any
            );
              always_comb begin
                any = 1'b0;
                for (integer i = 3; i >= 0; i--) begin
                  if (mask[i]) any = 1'b1;
                end
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "CombForDec").unwrap();
        let mask = signal(&imported, "CombForDec", "mask");
        let any = signal(&imported, "CombForDec", "any");
        sim.set(mask, 0);
        assert_eq!(sim.get(any), 0);
        sim.set(mask, 0b1000);
        assert_eq!(sim.get(any), 1);
    }

    #[test]
    fn imports_clocked_for_loop_register_updates() {
        let imported = import_sv(
            r#"
            module FfFor(
              input logic clk,
              output logic [31:0] q
            );
              always_ff @(posedge clk) begin
                for (int i = 0; i < 4; i++) begin
                  q <= i;
                end
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "FfFor").unwrap();
        let q = signal(&imported, "FfFor", "q");
        sim.tick();
        assert_eq!(sim.get(q), 3);
    }

    #[test]
    fn imports_reset_branch_for_loop_assignments() {
        let imported = import_sv(
            r#"
            module ResetFor(
              input logic clk,
              input logic rst_n,
              output logic [3:0] q
            );
              always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                  for (int i = 0; i < 1; i++) begin
                    q <= i;
                  end
                end else begin
                  q <= q + 4'd1;
                end
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "ResetFor").unwrap();
        let rst_n = signal(&imported, "ResetFor", "rst_n");
        let q = signal(&imported, "ResetFor", "q");
        sim.set(rst_n, 0);
        sim.tick();
        assert_eq!(sim.get(q), 0);
        sim.set(rst_n, 1);
        sim.tick();
        assert_eq!(sim.get(q), 1);
    }

    #[test]
    fn rejects_dynamic_for_loop_bound() {
        let err = import_sv(
            r#"
            module DynamicFor(
              input logic [3:0] n,
              output logic y
            );
              always_comb begin
                y = 1'b0;
                for (int i = 0; i < n; i++) y = 1'b1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_FOR");
    }

    #[test]
    fn rejects_zero_for_loop_step() {
        let err = import_sv(
            r#"
            module ZeroStepFor(output logic y);
              always_comb begin
                y = 1'b0;
                for (int i = 0; i < 4; i = i + 0) y = 1'b1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_FOR");
    }

    #[test]
    fn rejects_excessive_for_loop_unroll() {
        let err = import_sv(
            r#"
            module BigFor(output logic y);
              always_comb begin
                y = 1'b0;
                for (int i = 0; i <= 4096; i++) y = 1'b1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_FOR");
    }

    #[test]
    fn imports_generate_for_continuous_bit_assigns() {
        let imported = import_sv(
            r#"
            module GenAssign(
              input logic [3:0] a,
              input logic [3:0] b,
              output logic [3:0] y
            );
              genvar i;
              generate
                for (i = 0; i < 4; i++) begin : g
                  assign y[i] = a[i] ^ b[i];
                end
              endgenerate
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "GenAssign").unwrap();
        let a = signal(&imported, "GenAssign", "a");
        let b = signal(&imported, "GenAssign", "b");
        let y = signal(&imported, "GenAssign", "y");
        sim.set(a, 0b1010);
        sim.set(b, 0b0011);
        assert_eq!(sim.get(y), 0b1001);
    }

    #[test]
    fn imports_generate_for_repeated_instance_inputs() {
        let imported = import_sv(
            r#"
            module Leaf(input logic a);
            endmodule

            module GenInst(input logic [1:0] a);
              generate
                for (genvar i = 0; i < 2; i++) begin : lane
                  Leaf u(.a(a[i]));
                end
              endgenerate
            endmodule
            "#,
            Some("GenInst"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        let top = compiled.find_module("GenInst").unwrap();
        assert!(top
            .instances
            .iter()
            .any(|instance| instance.name == "lane__0__u"));
        assert!(top
            .instances
            .iter()
            .any(|instance| instance.name == "lane__1__u"));
    }

    #[test]
    fn imports_comb_selected_lvalue_assignments() {
        let imported = import_sv(
            r#"
            module SelectedComb(
              input logic a,
              input logic [2:0] mid,
              output logic [3:0] y
            );
              always_comb begin
                y = 4'b0000;
                y[0] = a;
                y[3:1] = mid;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "SelectedComb").unwrap();
        let a = signal(&imported, "SelectedComb", "a");
        let mid = signal(&imported, "SelectedComb", "mid");
        let y = signal(&imported, "SelectedComb", "y");
        sim.set(a, 1);
        sim.set(mid, 0b101);
        assert_eq!(sim.get(y), 0b1011);
    }

    #[test]
    fn imports_clocked_selected_lvalue_assignments() {
        let imported = import_sv(
            r#"
            module SelectedFf(
              input logic clk,
              input logic a,
              input logic [2:0] mid,
              output logic [3:0] q
            );
              always_ff @(posedge clk) begin
                q <= 4'b0000;
                q[0] <= a;
                q[3:1] <= mid;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "SelectedFf").unwrap();
        let a = signal(&imported, "SelectedFf", "a");
        let mid = signal(&imported, "SelectedFf", "mid");
        let q = signal(&imported, "SelectedFf", "q");
        sim.set(a, 1);
        sim.set(mid, 0b110);
        sim.tick();
        assert_eq!(sim.get(q), 0b1101);
    }

    #[test]
    fn imports_reset_selected_lvalue_assignments() {
        let imported = import_sv(
            r#"
            module SelectedReset(
              input logic clk,
              input logic rst,
              output logic [3:0] q
            );
              always_ff @(posedge clk or posedge rst) begin
                if (rst) begin
                  q[0] <= 1'b1;
                  q[3:1] <= 3'b101;
                end else begin
                  q <= 4'b0010;
                end
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "SelectedReset").unwrap();
        let rst = signal(&imported, "SelectedReset", "rst");
        let q = signal(&imported, "SelectedReset", "q");
        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(q), 0b1011);
        sim.set(rst, 0);
        sim.tick();
        assert_eq!(sim.get(q), 0b0010);
    }

    #[test]
    fn imports_sync_reset_assignment_truncation() {
        let imported = import_sv(
            r#"
            module ResetTrunc(
              input logic clk,
              input logic rst,
              output logic [2:0] y
            );
              reg [2:0] q;
              always_ff @(posedge clk) begin
                if (rst) q <= 8'hac;
                else q <= 3'b001;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "ResetTrunc").unwrap();
        let rst = signal(&imported, "ResetTrunc", "rst");
        let y = signal(&imported, "ResetTrunc", "y");
        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(y), 0b100);
        sim.set(rst, 0);
        sim.tick();
        assert_eq!(sim.get(y), 0b001);
    }

    #[test]
    fn imports_async_active_low_reset_assignment_truncation() {
        let imported = import_sv(
            r#"
            module AsyncResetTrunc(
              input logic clk,
              input logic rst_n,
              output logic [2:0] y
            );
              reg [2:0] q;
              always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) q <= 8'hac;
                else q <= 3'b001;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "AsyncResetTrunc").unwrap();
        let rst_n = signal(&imported, "AsyncResetTrunc", "rst_n");
        let y = signal(&imported, "AsyncResetTrunc", "y");
        sim.set(rst_n, 0);
        sim.tick();
        assert_eq!(sim.get(y), 0b100);
        sim.set(rst_n, 1);
        sim.tick();
        assert_eq!(sim.get(y), 0b001);
    }

    #[test]
    fn imports_reset_assignment_sign_extension() {
        let imported = import_sv(
            r#"
            module ResetSext(
              input logic clk,
              input logic rst,
              output logic signed [7:0] y
            );
              reg signed [7:0] q;
              always_ff @(posedge clk) begin
                if (rst) q <= $signed(4'b1111);
                else q <= 8'sd0;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "ResetSext").unwrap();
        let rst = signal(&imported, "ResetSext", "rst");
        let y = signal(&imported, "ResetSext", "y");
        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(y), 0xff);
    }

    #[test]
    fn imports_reset_assignment_zero_extension() {
        let imported = import_sv(
            r#"
            module ResetZext(
              input logic clk,
              input logic rst,
              output logic [7:0] y
            );
              reg [7:0] q;
              always_ff @(posedge clk) begin
                if (rst) q <= 4'b1111;
                else q <= 8'h00;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "ResetZext").unwrap();
        let rst = signal(&imported, "ResetZext", "rst");
        let y = signal(&imported, "ResetZext", "y");
        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(y), 0x0f);
    }

    #[test]
    fn nonconstant_reset_branch_uses_normal_next_path() {
        let imported = import_sv(
            r#"
            module DynamicResetBranch(
              input logic clk,
              input logic rst,
              input logic [2:0] d,
              output logic [2:0] y
            );
              reg [2:0] q;
              always_ff @(posedge clk) begin
                if (rst) q <= d;
                else q <= 3'b001;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "DynamicResetBranch").unwrap();
        let rst = signal(&imported, "DynamicResetBranch", "rst");
        let d = signal(&imported, "DynamicResetBranch", "d");
        let y = signal(&imported, "DynamicResetBranch", "y");
        sim.set(rst, 1);
        sim.set(d, 0b101);
        sim.tick();
        assert_eq!(sim.get(y), 0b101);
    }

    #[test]
    fn imports_initial_bit_select_assignment() {
        let imported = import_sv(
            r#"
            module InitialBit(
              input logic clk,
              output logic [3:0] y
            );
              reg [3:0] q;
              initial begin
                q[0] = 1'b1;
              end
              always_ff @(posedge clk) begin
                q <= q;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "InitialBit").unwrap();
        let y = signal(&imported, "InitialBit", "y");
        assert_eq!(sim.get(y), 0b0001);
    }

    #[test]
    fn imports_initial_slice_assignment() {
        let imported = import_sv(
            r#"
            module InitialSlice(
              input logic clk,
              output logic [3:0] y
            );
              reg [3:0] q;
              initial begin
                q[3:1] = 3'b101;
              end
              always_ff @(posedge clk) begin
                q <= q;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "InitialSlice").unwrap();
        let y = signal(&imported, "InitialSlice", "y");
        assert_eq!(sim.get(y), 0b1010);
    }

    #[test]
    fn imports_initial_whole_and_selected_assignments_in_order() {
        let imported = import_sv(
            r#"
            module InitialMixed(
              input logic clk,
              output logic [3:0] y
            );
              reg [3:0] q;
              initial begin
                q = 4'b1111;
                q[2:1] = 2'b00;
                q[0] = 1'b0;
              end
              always_ff @(posedge clk) begin
                q <= q;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "InitialMixed").unwrap();
        let y = signal(&imported, "InitialMixed", "y");
        assert_eq!(sim.get(y), 0b1000);
    }

    #[test]
    fn imports_multiple_initial_blocks_for_same_register() {
        let imported = import_sv(
            r#"
            module InitialMulti(
              input logic clk,
              output logic [3:0] y
            );
              reg [3:0] q;
              initial begin
                q[1:0] = 2'b01;
              end
              initial begin
                q[3:2] = 2'b10;
              end
              always_ff @(posedge clk) begin
                q <= q;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "InitialMulti").unwrap();
        let y = signal(&imported, "InitialMulti", "y");
        assert_eq!(sim.get(y), 0b1001);
    }

    #[test]
    fn imports_generated_initial_selected_registers() {
        let imported = import_sv(
            r#"
            module InitialGen(
              input logic clk,
              output logic [1:0] y
            );
              generate
                for (genvar i = 0; i < 2; i++) begin : lane
                  reg tmp;
                  initial begin
                    tmp[0] = i[0];
                  end
                  always_ff @(posedge clk) begin
                    tmp <= tmp;
                  end
                  assign y[i] = tmp;
                end
              endgenerate
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "InitialGen").unwrap();
        let y = signal(&imported, "InitialGen", "y");
        assert_eq!(sim.get(y), 0b10);
    }

    #[test]
    fn initial_memory_bit_syntax_still_initializes_memory() {
        let imported = import_sv(
            r#"
            module InitialMemoryCompat(
              input logic [1:0] addr,
              output logic [7:0] y
            );
              logic [7:0] mem [0:3];
              initial begin
                mem[2] = 8'hab;
              end
              assign y = mem[addr];
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "InitialMemoryCompat").unwrap();
        let addr = signal(&imported, "InitialMemoryCompat", "addr");
        let y = signal(&imported, "InitialMemoryCompat", "y");
        sim.set(addr, 2);
        assert_eq!(sim.get(y), 0xab);
    }

    #[test]
    fn rejects_initial_selected_assignment_to_wire() {
        let err = import_sv(
            r#"
            module BadInitialWire(output logic [1:0] y);
              initial begin
                y[0] = 1'b1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_INITIAL_REG_TARGET");
    }

    #[test]
    fn rejects_dynamic_generate_for_bound() {
        let err = import_sv(
            r#"
            module BadGen(input logic [3:0] n, output logic [3:0] y);
              generate
                for (genvar i = 0; i < n; i++) begin : g
                  assign y[i] = 1'b1;
                end
              endgenerate
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_FOR");
    }

    #[test]
    fn imports_generate_for_local_wire() {
        let imported = import_sv(
            r#"
            module GenLocalWire(
              input logic [1:0] a,
              output logic [1:0] y
            );
              generate
                for (genvar i = 0; i < 2; i++) begin : g
                  logic tmp;
                  assign tmp = a[i];
                  assign y[i] = tmp;
                end
              endgenerate
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "GenLocalWire").unwrap();
        let a = signal(&imported, "GenLocalWire", "a");
        let y = signal(&imported, "GenLocalWire", "y");
        sim.set(a, 0b10);
        assert_eq!(sim.get(y), 0b10);
        let compiled = compile(&imported.design).unwrap();
        let top = compiled.find_module("GenLocalWire").unwrap();
        assert!(top.signals.iter().any(|signal| signal.name == "g__0__tmp"));
        assert!(top.signals.iter().any(|signal| signal.name == "g__1__tmp"));
    }

    #[test]
    fn imports_generate_local_wire_used_in_always_comb() {
        let imported = import_sv(
            r#"
            module GenLocalComb(
              input logic [1:0] a,
              output logic [1:0] y
            );
              generate
                for (genvar i = 0; i < 2; i++) begin : g
                  logic tmp;
                  always_comb begin
                    tmp = ~a[i];
                  end
                  assign y[i] = tmp;
                end
              endgenerate
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "GenLocalComb").unwrap();
        let a = signal(&imported, "GenLocalComb", "a");
        let y = signal(&imported, "GenLocalComb", "y");
        sim.set(a, 0b01);
        assert_eq!(sim.get(y), 0b10);
    }

    #[test]
    fn imports_generate_local_reg_used_in_always_ff() {
        let imported = import_sv(
            r#"
            module GenLocalReg(
              input logic clk,
              input logic [1:0] a,
              output logic [1:0] y
            );
              generate
                for (genvar i = 0; i < 2; i++) begin : g
                  reg tmp;
                  always_ff @(posedge clk) begin
                    tmp <= a[i];
                  end
                  assign y[i] = tmp;
                end
              endgenerate
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "GenLocalReg").unwrap();
        let a = signal(&imported, "GenLocalReg", "a");
        let y = signal(&imported, "GenLocalReg", "y");
        sim.set(a, 0b11);
        sim.tick();
        assert_eq!(sim.get(y), 0b11);
    }

    #[test]
    fn imports_nested_generate_local_names() {
        let imported = import_sv(
            r#"
            module NestedGenLocal(output logic [1:0] y);
              generate
                for (genvar i = 0; i < 2; i++) begin : row
                  for (genvar j = 0; j < 1; j++) begin : col
                    logic tmp;
                    assign tmp = i[0];
                    assign y[i] = tmp;
                  end
                end
              endgenerate
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "NestedGenLocal").unwrap();
        let y = signal(&imported, "NestedGenLocal", "y");
        assert_eq!(sim.get(y), 0b10);
        let compiled = compile(&imported.design).unwrap();
        let top = compiled.find_module("NestedGenLocal").unwrap();
        assert!(top
            .signals
            .iter()
            .any(|signal| signal.name == "row__0__col__0__tmp"));
        assert!(top
            .signals
            .iter()
            .any(|signal| signal.name == "row__1__col__0__tmp"));
    }

    #[test]
    fn imports_generated_instance_through_local_wire() {
        let imported = import_sv(
            r#"
            module Leaf(input logic a);
            endmodule

            module GenLocalInst(input logic [1:0] a, output logic [1:0] y);
              generate
                for (genvar i = 0; i < 2; i++) begin : lane
                  logic tmp;
                  assign tmp = a[i];
                  assign y[i] = tmp;
                  Leaf u(.a(tmp));
                end
              endgenerate
            endmodule
            "#,
            Some("GenLocalInst"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "GenLocalInst").unwrap();
        let a = signal(&imported, "GenLocalInst", "a");
        let y = signal(&imported, "GenLocalInst", "y");
        sim.set(a, 0b01);
        assert_eq!(sim.get(y), 0b01);
    }

    #[test]
    fn imports_instance_output_bit_select_connection() {
        let imported = import_sv(
            r#"
            module Leaf(input logic a, output logic y);
              assign y = a;
            endmodule

            module InstOutBit(input logic a, output logic [1:0] y);
              assign y[1] = 1'b0;
              Leaf u(.a(a), .y(y[0]));
            endmodule
            "#,
            Some("InstOutBit"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "InstOutBit").unwrap();
        let a = signal(&imported, "InstOutBit", "a");
        let y = signal(&imported, "InstOutBit", "y");
        sim.set(a, 1);
        assert_eq!(sim.get(y), 0b01);
    }

    #[test]
    fn imports_generate_for_instance_output_bit_selects() {
        let imported = import_sv(
            r#"
            module Leaf(input logic a, output logic y);
              assign y = a;
            endmodule

            module GenInstOut(input logic [3:0] a, output logic [3:0] y);
              generate
                for (genvar i = 0; i < 4; i++) begin : lane
                  Leaf u(.a(a[i]), .y(y[i]));
                end
              endgenerate
            endmodule
            "#,
            Some("GenInstOut"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "GenInstOut").unwrap();
        let a = signal(&imported, "GenInstOut", "a");
        let y = signal(&imported, "GenInstOut", "y");
        sim.set(a, 0b1011);
        assert_eq!(sim.get(y), 0b1011);
    }

    #[test]
    fn imports_instance_output_slice_connection() {
        let imported = import_sv(
            r#"
            module Leaf(input logic [1:0] a, output logic [1:0] y);
              assign y = a;
            endmodule

            module InstOutSlice(input logic [1:0] a, output logic [3:0] y);
              assign y[1:0] = 2'b01;
              Leaf u(.a(a), .y(y[3:2]));
            endmodule
            "#,
            Some("InstOutSlice"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "InstOutSlice").unwrap();
        let a = signal(&imported, "InstOutSlice", "a");
        let y = signal(&imported, "InstOutSlice", "y");
        sim.set(a, 0b10);
        assert_eq!(sim.get(y), 0b1001);
    }

    #[test]
    fn imports_generated_local_wire_to_selected_instance_output() {
        let imported = import_sv(
            r#"
            module Leaf(input logic a, output logic y);
              assign y = a;
            endmodule

            module GenLocalInstOut(input logic [1:0] a, output logic [1:0] y);
              generate
                for (genvar i = 0; i < 2; i++) begin : lane
                  logic tmp;
                  assign tmp = a[i];
                  Leaf u(.a(tmp), .y(y[i]));
                end
              endgenerate
            endmodule
            "#,
            Some("GenLocalInstOut"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "GenLocalInstOut").unwrap();
        let a = signal(&imported, "GenLocalInstOut", "a");
        let y = signal(&imported, "GenLocalInstOut", "y");
        sim.set(a, 0b10);
        assert_eq!(sim.get(y), 0b10);
    }

    #[test]
    fn rejects_selected_inout_instance_connection() {
        let err = import_sv(
            r#"
            module Pad(inout logic p);
            endmodule

            module BadInout(inout logic [1:0] bus);
              Pad u(.p(bus[0]));
            endmodule
            "#,
            Some("BadInout"),
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_INSTANCE_CONN");
    }

    #[test]
    fn rejects_generated_memory_declarations() {
        let err = import_sv(
            r#"
            module BadGenMem(output logic y);
              generate
                for (genvar i = 0; i < 1; i++) begin : g
                  logic tmp [0:1];
                end
              endgenerate
              assign y = 1'b0;
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_GENERATE");
    }

    #[test]
    fn rejects_generated_parameters() {
        let err = import_sv(
            r#"
            module BadGenParam(output logic y);
              generate
                for (genvar i = 0; i < 1; i++) begin : g
                  localparam int P = 1;
                  assign y = P;
                end
              endgenerate
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_GENERATE");
    }

    #[test]
    fn rejects_generated_typedefs() {
        let err = import_sv(
            r#"
            module BadGenTypedef(output logic y);
              generate
                for (genvar i = 0; i < 1; i++) begin : g
                  typedef enum logic [0:0] { A = 1'b0 } t;
                  t tmp;
                end
              endgenerate
              assign y = 1'b0;
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_GENERATE");
    }

    #[test]
    fn rejects_selected_lvalue_out_of_range() {
        let err = import_sv(
            r#"
            module BadSelect(output logic [1:0] y);
              always_comb begin
                y = 2'b00;
                y[2] = 1'b1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_LVALUE");
    }

    #[test]
    fn parses_case_statement_variants() {
        let source = parse_sv(
            r#"
            module Decoder(input logic [1:0] op, output logic [3:0] y);
              always_comb begin
                priority case (op)
                  2'd0, 2'd1: y = 4'd1;
                  default: y = 4'd0;
                endcase
              end
            endmodule
            "#,
        )
        .unwrap();
        let SvItem::AlwaysComb(stmts) = source.modules[0]
            .items
            .iter()
            .find(|item| matches!(item, SvItem::AlwaysComb(_)))
            .unwrap()
        else {
            panic!("expected always_comb item");
        };
        let SvStmt::Case { kind, expr, items } = &stmts[0] else {
            panic!("expected case statement");
        };
        assert_eq!(*kind, SvCaseKind::Normal);
        assert!(matches!(expr, SvExpr::Ident(name) if name == "op"));
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].labels.len(), 2);
        assert!(matches!(items[0].labels[0], SvCaseLabel::Expr(_)));
        assert!(!items[0].is_default);
        assert!(items[1].is_default);
    }

    #[test]
    fn imports_casez_wildcard_decoder() {
        let imported = import_sv(
            r#"
            module WildCaseZ(input logic [3:0] op, output logic [1:0] y);
              always_comb begin
                y = 2'd0;
                unique casez (op)
                  4'b10??: y = 2'd1;
                  4'b0z01, 4'b0?10: y = 2'd2;
                  default: y = 2'd3;
                endcase
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "WildCaseZ").unwrap();
        let op = signal(&imported, "WildCaseZ", "op");
        let y = signal(&imported, "WildCaseZ", "y");
        sim.set(op, 0b1000);
        assert_eq!(sim.get(y), 1);
        sim.set(op, 0b0101);
        assert_eq!(sim.get(y), 2);
        sim.set(op, 0b0010);
        assert_eq!(sim.get(y), 2);
        sim.set(op, 0b1111);
        assert_eq!(sim.get(y), 3);
    }

    #[test]
    fn imports_casex_wildcard_decoder() {
        let imported = import_sv(
            r#"
            module WildCaseX(input logic [3:0] op, output logic [1:0] y);
              always_comb begin
                y = 2'd0;
                priority casex (op)
                  4'b1x0?: y = 2'd1;
                  4'hz: y = 2'd2;
                  default: y = 2'd3;
                endcase
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "WildCaseX").unwrap();
        let op = signal(&imported, "WildCaseX", "op");
        let y = signal(&imported, "WildCaseX", "y");
        sim.set(op, 0b1001);
        assert_eq!(sim.get(y), 1);
        sim.set(op, 0b1111);
        assert_eq!(sim.get(y), 2);
    }

    #[test]
    fn imports_clocked_casez_wildcard_updates() {
        let imported = import_sv(
            r#"
            module WildFf(
              input logic clk,
              input logic [2:0] op,
              output logic [3:0] q
            );
              always_ff @(posedge clk) begin
                casez (op)
                  3'b1??: q <= 4'd8;
                  3'b0?1: q <= 4'd3;
                  default: q <= q;
                endcase
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "WildFf").unwrap();
        let op = signal(&imported, "WildFf", "op");
        let q = signal(&imported, "WildFf", "q");
        sim.set(op, 0b101);
        sim.tick();
        assert_eq!(sim.get(q), 8);
        sim.set(op, 0b001);
        sim.tick();
        assert_eq!(sim.get(q), 3);
        sim.set(op, 0b010);
        sim.tick();
        assert_eq!(sim.get(q), 3);
    }

    #[test]
    fn rejects_casez_x_wildcard_labels() {
        let err = import_sv(
            r#"
            module BadCaseZ(input logic [1:0] op, output logic y);
              always_comb begin
                y = 1'd0;
                casez (op)
                  2'b1x: y = 1'd1;
                endcase
              end
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_CASE");
    }

    #[test]
    fn imports_comb_case_with_defaults_and_comma_labels() {
        let imported = import_sv(
            include_str!("../tests/fixtures/comb_case_decoder.sv"),
            Some("CombCaseDecoder"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "CombCaseDecoder").unwrap();
        let op = signal(&imported, "CombCaseDecoder", "op");
        let y = signal(&imported, "CombCaseDecoder", "y");

        sim.set(op, 0);
        assert_eq!(sim.get(y), 1);
        sim.set(op, 1);
        assert_eq!(sim.get(y), 7);
        sim.set(op, 2);
        assert_eq!(sim.get(y), 7);
        sim.set(op, 3);
        assert_eq!(sim.get(y), 15);
    }

    #[test]
    fn imports_comb_if_defaults_and_nested_updates() {
        let imported = import_sv(
            include_str!("../tests/fixtures/comb_if_defaults.sv"),
            Some("CombIfDefaults"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "CombIfDefaults").unwrap();
        let en = signal(&imported, "CombIfDefaults", "en");
        let sel = signal(&imported, "CombIfDefaults", "sel");
        let a = signal(&imported, "CombIfDefaults", "a");
        let b = signal(&imported, "CombIfDefaults", "b");
        let c = signal(&imported, "CombIfDefaults", "c");
        let y = signal(&imported, "CombIfDefaults", "y");
        let z = signal(&imported, "CombIfDefaults", "z");

        sim.set(a, 0x11);
        sim.set(b, 0x22);
        sim.set(c, 0x33);
        assert_eq!(sim.get(y), 0x11);
        assert_eq!(sim.get(z), 0);

        sim.set(en, 1);
        assert_eq!(sim.get(y), 0x22);
        assert_eq!(sim.get(z), 0);

        sim.set(sel, 1);
        assert_eq!(sim.get(y), 0x22);
        assert_eq!(sim.get(z), 0x33);
    }

    #[test]
    fn imports_legacy_always_star_comb_logic() {
        let imported = import_sv(
            r#"
            module LegacyCombStar(
              input logic en,
              input logic sel,
              input logic [7:0] a,
              input logic [7:0] b,
              input logic [7:0] c,
              output logic [7:0] y
            );
              always @(*) begin
                y = a;
                if (en) begin
                  y = b;
                  if (sel) y = c;
                end
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "LegacyCombStar").unwrap();
        let en = signal(&imported, "LegacyCombStar", "en");
        let sel = signal(&imported, "LegacyCombStar", "sel");
        let a = signal(&imported, "LegacyCombStar", "a");
        let b = signal(&imported, "LegacyCombStar", "b");
        let c = signal(&imported, "LegacyCombStar", "c");
        let y = signal(&imported, "LegacyCombStar", "y");
        sim.set(a, 0x11);
        sim.set(b, 0x22);
        sim.set(c, 0x33);
        assert_eq!(sim.get(y), 0x11);
        sim.set(en, 1);
        assert_eq!(sim.get(y), 0x22);
        sim.set(sel, 1);
        assert_eq!(sim.get(y), 0x33);
    }

    #[test]
    fn imports_legacy_always_at_star_comb_logic() {
        let imported = import_sv(
            r#"
            module LegacyCombAtStar(input logic a, input logic b, output logic y);
              always @* y = a & b;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "LegacyCombAtStar").unwrap();
        let a = signal(&imported, "LegacyCombAtStar", "a");
        let b = signal(&imported, "LegacyCombAtStar", "b");
        let y = signal(&imported, "LegacyCombAtStar", "y");
        sim.set(a, 1);
        sim.set(b, 0);
        assert_eq!(sim.get(y), 0);
        sim.set(b, 1);
        assert_eq!(sim.get(y), 1);
    }

    #[test]
    fn imports_handwritten_ready_valid_comb_control() {
        let imported = import_sv(
            include_str!("../tests/fixtures/rv_handwritten_slice.sv"),
            Some("RvHandwrittenSlice"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "RvHandwrittenSlice").unwrap();
        let in_valid = signal(&imported, "RvHandwrittenSlice", "in_valid");
        let in_ready = signal(&imported, "RvHandwrittenSlice", "in_ready");
        let in_bits = signal(&imported, "RvHandwrittenSlice", "in_bits");
        let out_valid = signal(&imported, "RvHandwrittenSlice", "out_valid");
        let out_ready = signal(&imported, "RvHandwrittenSlice", "out_ready");
        let out_bits = signal(&imported, "RvHandwrittenSlice", "out_bits");
        let stall = signal(&imported, "RvHandwrittenSlice", "stall");

        sim.set(in_valid, 1);
        sim.set(in_bits, 0xa5);
        sim.set(out_ready, 0);
        assert_eq!(sim.get(in_ready), 0);
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_bits), 0xa5);
        assert_eq!(sim.get(stall), 1);

        sim.set(out_ready, 1);
        assert_eq!(sim.get(in_ready), 1);
        assert_eq!(sim.get(stall), 0);
    }

    #[test]
    fn imports_instance_input_expressions_for_defined_child() {
        let imported = import_sv(
            r#"
            module ExprChild(
              input logic [7:0] a,
              input logic en,
              output logic [7:0] y
            );
              assign y = en ? a : 8'd0;
            endmodule

            module ExprTop(
              input logic [7:0] x,
              input logic valid,
              output logic [7:0] y
            );
              ExprChild u_child(
                .a(x + 8'd1),
                .en(valid & 1'b1),
                .y(y)
              );
            endmodule
            "#,
            Some("ExprTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        let top = compiled.find_module("ExprTop").unwrap();
        assert!(top
            .signals
            .iter()
            .any(|signal| signal.name == "__sv_inst_u_child_a"));
        assert!(top
            .signals
            .iter()
            .any(|signal| signal.name == "__sv_inst_u_child_en"));

        let mut sim = Simulator::new(&imported.design, "ExprTop").unwrap();
        let x = signal(&imported, "ExprTop", "x");
        let valid = signal(&imported, "ExprTop", "valid");
        let y = signal(&imported, "ExprTop", "y");
        sim.set(x, 0x12);
        sim.set(valid, 1);
        assert_eq!(sim.get(y), 0x13);
        sim.set(valid, 0);
        assert_eq!(sim.get(y), 0);
    }

    #[test]
    fn imports_instance_constant_tieoff_and_avoids_temp_name_collision() {
        let imported = import_sv(
            r#"
            module ConstChild(input logic [7:0] a, output logic [7:0] y);
              assign y = a;
            endmodule

            module ConstTop(output logic [7:0] y);
              logic [7:0] __sv_inst_u_child_a;
              assign __sv_inst_u_child_a = 8'd0;
              ConstChild u_child(.a(1'b1), .y(y));
            endmodule
            "#,
            Some("ConstTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        let top = compiled.find_module("ConstTop").unwrap();
        assert!(top
            .signals
            .iter()
            .any(|signal| signal.name == "__sv_inst_u_child_a_1"));

        let sim = Simulator::new(&imported.design, "ConstTop").unwrap();
        let y = signal(&imported, "ConstTop", "y");
        assert_eq!(sim.get(y), 1);
    }

    #[test]
    fn imports_external_instance_input_expressions() {
        let imported = import_sv(
            r#"
            module ExtTop(input logic sel, output logic [3:0] y);
              Vendor u_vendor(
                .a(4'hf),
                .en(sel | 1'b0),
                .y(y)
              );
            endmodule
            "#,
            Some("ExtTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        let vendor = compiled.find_module("Vendor").unwrap();
        let a = vendor
            .signals
            .iter()
            .find(|signal| signal.name == "a")
            .unwrap();
        let en = vendor
            .signals
            .iter()
            .find(|signal| signal.name == "en")
            .unwrap();
        let y = vendor
            .signals
            .iter()
            .find(|signal| signal.name == "y")
            .unwrap();
        assert_eq!(a.ty, uint(4));
        assert_eq!(en.ty, uint(1));
        assert_eq!(y.ty, uint(4));
        assert!(matches!(a.kind, SignalKind::Input));
        assert!(matches!(en.kind, SignalKind::Input));
        assert!(matches!(y.kind, SignalKind::Output));
    }

    #[test]
    fn rejects_defined_instance_output_expressions() {
        for expr in ["a & b", "1'b0"] {
            let err = import_sv(
                &format!(
                    r#"
                    module OutChild(output logic y);
                      assign y = 1'b1;
                    endmodule

                    module BadTop(input logic a, input logic b);
                      OutChild u_child(.y({expr}));
                    endmodule
                    "#
                ),
                Some("BadTop"),
            )
            .unwrap_err();
            assert_eq!(err.diagnostics[0].code, "E_SV_INSTANCE_CONN");
        }
    }

    #[test]
    fn imports_legacy_posedge_always_register_update() {
        let imported = import_sv(
            r#"
            module LegacyFf(input logic clk, input logic en, output logic [3:0] q);
              always @(posedge clk) begin
                if (en) q <= q + 4'd1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "LegacyFf").unwrap();
        let en = signal(&imported, "LegacyFf", "en");
        let q = signal(&imported, "LegacyFf", "q");
        sim.set(en, 1);
        sim.tick();
        sim.tick();
        assert_eq!(sim.get(q), 2);
    }

    #[test]
    fn imports_legacy_posedge_always_async_active_low_reset() {
        let imported = import_sv(
            r#"
            module LegacyAsyncReset(
              input logic clk,
              input logic rst_n,
              input logic en,
              output logic [3:0] q
            );
              always @(posedge clk or negedge rst_n) begin
                if (!rst_n) q <= 4'd0;
                else if (en) q <= q + 4'd1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "LegacyAsyncReset").unwrap();
        let rst_n = signal(&imported, "LegacyAsyncReset", "rst_n");
        let en = signal(&imported, "LegacyAsyncReset", "en");
        let q = signal(&imported, "LegacyAsyncReset", "q");
        sim.set(rst_n, 0);
        sim.tick();
        assert_eq!(sim.get(q), 0);
        sim.set(rst_n, 1);
        sim.set(en, 1);
        sim.tick();
        sim.tick();
        assert_eq!(sim.get(q), 2);
        sim.set(rst_n, 0);
        sim.tick();
        assert_eq!(sim.get(q), 0);
    }

    #[test]
    fn rejects_unsupported_legacy_always_event_controls() {
        for source in [
            r#"
            module BadLevel(input logic a, input logic b, output logic y);
              always @(a or b) y = a & b;
            endmodule
            "#,
            r#"
            module BadNegedge(input logic clk, output logic q);
              always @(negedge clk) q <= 1'd1;
            endmodule
            "#,
            r#"
            module BadLatch(input logic a, output logic y);
              always_latch y = a;
            endmodule
            "#,
        ] {
            let err = import_sv(source, None).unwrap_err();
            assert_eq!(err.diagnostics[0].code, "E_SV_ALWAYS");
        }
    }

    #[test]
    fn imports_case_without_default_using_prior_assignment_fallback() {
        let imported = import_sv(
            r#"
            module CaseFallback(
              input logic sel,
              input logic [7:0] a,
              input logic [7:0] b,
              output logic [7:0] y
            );
              always_comb begin
                y = a;
                case (sel)
                  1'd1: y = b;
                endcase
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "CaseFallback").unwrap();
        let sel = signal(&imported, "CaseFallback", "sel");
        let a = signal(&imported, "CaseFallback", "a");
        let b = signal(&imported, "CaseFallback", "b");
        let y = signal(&imported, "CaseFallback", "y");
        sim.set(a, 0x12);
        sim.set(b, 0x34);
        assert_eq!(sim.get(y), 0x12);
        sim.set(sel, 1);
        assert_eq!(sim.get(y), 0x34);
    }

    #[test]
    fn imports_case_labels_from_localparams() {
        let imported = import_sv(
            r#"
            module CaseParams(input logic [1:0] op, output logic [3:0] y);
              localparam int A = 1;
              localparam int B = 2;
              always_comb begin
                y = 4'd0;
                case (op)
                  A: y = 4'd5;
                  B: y = 4'd6;
                endcase
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "CaseParams").unwrap();
        let op = signal(&imported, "CaseParams", "op");
        let y = signal(&imported, "CaseParams", "y");
        sim.set(op, 1);
        assert_eq!(sim.get(y), 5);
        sim.set(op, 2);
        assert_eq!(sim.get(y), 6);
        sim.set(op, 3);
        assert_eq!(sim.get(y), 0);
    }

    #[test]
    fn imports_clocked_if_else_register_updates() {
        let imported = import_sv(
            include_str!("../tests/fixtures/ff_if_else_counter.sv"),
            Some("FfIfElseCounter"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "FfIfElseCounter").unwrap();
        let en = signal(&imported, "FfIfElseCounter", "en");
        let load = signal(&imported, "FfIfElseCounter", "load");
        let din = signal(&imported, "FfIfElseCounter", "din");
        let q = signal(&imported, "FfIfElseCounter", "q");

        sim.set(load, 1);
        sim.set(din, 9);
        sim.tick();
        assert_eq!(sim.get(q), 9);

        sim.set(load, 0);
        sim.set(en, 1);
        sim.tick();
        assert_eq!(sim.get(q), 10);

        sim.set(en, 0);
        sim.tick();
        assert_eq!(sim.get(q), 10);
    }

    #[test]
    fn imports_clocked_case_fsm_updates() {
        let imported = import_sv(
            include_str!("../tests/fixtures/ff_case_fsm.sv"),
            Some("FfCaseFsm"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "FfCaseFsm").unwrap();
        let rst = signal(&imported, "FfCaseFsm", "rst");
        let start = signal(&imported, "FfCaseFsm", "start");
        let done = signal(&imported, "FfCaseFsm", "done");
        let state = signal(&imported, "FfCaseFsm", "state");

        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(state), 0);

        sim.set(rst, 0);
        sim.set(start, 1);
        sim.tick();
        assert_eq!(sim.get(state), 1);

        sim.set(start, 0);
        sim.tick();
        assert_eq!(sim.get(state), 1);

        sim.set(done, 1);
        sim.tick();
        assert_eq!(sim.get(state), 2);

        sim.set(done, 0);
        sim.tick();
        assert_eq!(sim.get(state), 0);
    }

    #[test]
    fn imports_clocked_case_with_prior_default_next() {
        let imported = import_sv(
            include_str!("../tests/fixtures/ff_case_default_next.sv"),
            Some("FfCaseDefaultNext"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "FfCaseDefaultNext").unwrap();
        let op = signal(&imported, "FfCaseDefaultNext", "op");
        let a = signal(&imported, "FfCaseDefaultNext", "a");
        let b = signal(&imported, "FfCaseDefaultNext", "b");
        let q = signal(&imported, "FfCaseDefaultNext", "q");

        sim.set(a, 0x11);
        sim.set(b, 0x22);
        sim.set(op, 0);
        sim.tick();
        assert_eq!(sim.get(q), 0x11);

        sim.set(op, 1);
        sim.tick();
        assert_eq!(sim.get(q), 0x22);

        sim.set(op, 2);
        sim.tick();
        assert_eq!(sim.get(q), 0x23);
    }

    #[test]
    fn imports_clocked_output_logic_as_internal_register() {
        let imported = import_sv(
            include_str!("../tests/fixtures/ff_output_logic.sv"),
            Some("FfOutputLogic"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        let module = compiled.find_module("FfOutputLogic").unwrap();
        assert!(module.signals.iter().any(|signal| signal.name == "q"));
        assert!(module
            .signals
            .iter()
            .any(|signal| signal.name == "q__sv_reg"));

        let mut sim = Simulator::new(&imported.design, "FfOutputLogic").unwrap();
        let d = signal(&imported, "FfOutputLogic", "d");
        let q = signal(&imported, "FfOutputLogic", "q");
        sim.set(d, 1);
        assert_eq!(sim.get(q), 0);
        sim.tick();
        assert_eq!(sim.get(q), 1);
        sim.set(d, 0);
        sim.tick();
        assert_eq!(sim.get(q), 0);
    }

    #[test]
    fn imports_output_reg_self_reference_as_internal_register() {
        let imported = import_sv(
            include_str!("../tests/fixtures/ff_output_reg_counter.sv"),
            Some("FfOutputRegCounter"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "FfOutputRegCounter").unwrap();
        let en = signal(&imported, "FfOutputRegCounter", "en");
        let q = signal(&imported, "FfOutputRegCounter", "q");
        sim.set(en, 1);
        sim.tick();
        sim.tick();
        assert_eq!(sim.get(q), 2);
        sim.set(en, 0);
        sim.tick();
        assert_eq!(sim.get(q), 2);
    }

    #[test]
    fn imports_clocked_output_case_reset_without_explicit_kind() {
        let imported = import_sv(
            include_str!("../tests/fixtures/ff_output_case_reset.sv"),
            Some("FfOutputCaseReset"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "FfOutputCaseReset").unwrap();
        let rst = signal(&imported, "FfOutputCaseReset", "rst");
        let op = signal(&imported, "FfOutputCaseReset", "op");
        let d = signal(&imported, "FfOutputCaseReset", "d");
        let q = signal(&imported, "FfOutputCaseReset", "q");

        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(q), 0);

        sim.set(rst, 0);
        sim.set(op, 1);
        sim.set(d, 9);
        sim.tick();
        assert_eq!(sim.get(q), 9);

        sim.set(op, 2);
        sim.tick();
        assert_eq!(sim.get(q), 10);

        sim.set(op, 0);
        sim.tick();
        assert_eq!(sim.get(q), 10);
    }

    #[test]
    fn reimports_emitted_state_enum_register() {
        let mut design = Design::new();
        {
            let mut m = design.module("Controller");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let start = m.input("start", uint(1));
            let busy = m.output("busy", uint(1));
            let states = rrtl_core::state_type(
                "ControllerState",
                uint(2),
                [("Idle", 0), ("Run", 1), ("Done", 2)],
            );
            let state = m.state_reg("state", states, clk, rst, "Idle");
            m.state_next_hold(&state, [(start.value(), "Run")]);
            m.assign(busy, state.is("Run"));
        }

        let sv = rrtl_sv::emit(&design).unwrap();
        let imported = import_sv(&sv, Some("Controller")).unwrap();
        let mut sim = Simulator::new(&imported.design, "Controller").unwrap();
        let rst = signal(&imported, "Controller", "rst");
        let start = signal(&imported, "Controller", "start");
        let busy = signal(&imported, "Controller", "busy");
        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(busy), 0);
        sim.set(rst, 0);
        sim.set(start, 1);
        sim.tick();
        assert_eq!(sim.get(busy), 1);
    }

    #[test]
    fn reimports_emitted_async_active_low_state_enum_register() {
        let mut design = Design::new();
        {
            let mut m = design.module("ActiveLowState");
            let clk = m.input("clk", uint(1));
            let rst_n = m.input("rst_n", uint(1));
            let start = m.input("start", uint(1));
            let states =
                rrtl_core::state_type("ControllerState", uint(1), [("Idle", 0), ("Run", 1)]);
            let state = m.state_reg_async_reset_low("state", states, clk, rst_n, "Idle");
            m.state_next_hold(&state, [(start.value(), "Run")]);
        }

        let sv = rrtl_sv::emit(&design).unwrap();
        let imported = import_sv(&sv, Some("ActiveLowState")).unwrap();
        let mut sim = Simulator::new(&imported.design, "ActiveLowState").unwrap();
        let rst_n = signal(&imported, "ActiveLowState", "rst_n");
        let start = signal(&imported, "ActiveLowState", "start");
        let state = signal(&imported, "ActiveLowState", "state");
        sim.set(rst_n, 0);
        sim.tick();
        assert_eq!(sim.get(state), 0);
        sim.set(rst_n, 1);
        sim.set(start, 1);
        sim.tick();
        assert_eq!(sim.get(state), 1);
    }

    #[test]
    fn reimports_emitted_assertions_and_covers() {
        let mut design = Design::new();
        {
            let mut m = design.module("Props");
            let clk = m.input("clk", uint(1));
            let en = m.input("en", uint(1));
            let count = m.reg("count", uint(4));
            m.clock(count, clk);
            m.next(count, count.value() + lit(1, 4));
            m.assert_msg(
                "comb_ok",
                count.value().lt_expr(lit(10, 4)),
                "count too high",
            );
            m.assert_when_msg(
                "enabled_ok",
                en.value(),
                count.value().lt_expr(lit(12, 4)),
                "enabled_ok",
            );
            m.assert_clocked_msg(
                "clocked_ok",
                clk,
                count.value().lt_expr(lit(14, 4)),
                "clocked_ok",
            );
            m.cover("comb_hit", count.value().eq_expr(lit(3, 4)));
            m.cover_when("enabled_hit", en.value(), count.value().eq_expr(lit(5, 4)));
            m.cover_clocked("clocked_hit", clk, count.value().eq_expr(lit(7, 4)));
        }

        let sv = rrtl_sv::emit(&design).unwrap();
        let imported = import_sv(&sv, Some("Props")).unwrap();
        let mut sim = Simulator::new(&imported.design, "Props").unwrap();
        let en = signal(&imported, "Props", "en");
        sim.set(en, 1);
        for _ in 0..8 {
            sim.tick_checked().unwrap();
            sim.check_assertions().unwrap();
        }
        assert!(sim.cover_hits("Props.COMB_HIT") > 0);
        assert!(sim.cover_hits("Props.ENABLED_HIT") > 0);
        assert!(sim.cover_hits("Props.CLOCKED_HIT") > 0);
    }

    #[test]
    fn reimports_emitted_initial_register_values() {
        let mut design = Design::new();
        {
            let mut m = design.module("InitialReg");
            let clk = m.input("clk", uint(1));
            let q = m.reg("q", uint(8));
            let out = m.output("out", uint(8));
            m.clock(q, clk);
            m.next(q, q.value());
            m.initial(q, 0x5a);
            m.assign(out, q.value());
        }

        let sv = rrtl_sv::emit(&design).unwrap();
        let imported = import_sv(&sv, Some("InitialReg")).unwrap();
        let sim = Simulator::new(&imported.design, "InitialReg").unwrap();
        let out = signal(&imported, "InitialReg", "out");
        assert_eq!(sim.get(out), 0x5a);
    }

    #[test]
    fn reimports_emitted_initial_memory_values() {
        let mut design = Design::new();
        {
            let mut m = design.module("InitialMem");
            let addr = m.input("addr", uint(2));
            let rdata = m.output("rdata", uint(8));
            let mem = m.mem("regs", 2, uint(8), 4);
            m.mem_init(mem, 2, 0xab);
            m.assign(rdata, rrtl_core::mem_read(mem, addr.value()));
        }

        let sv = rrtl_sv::emit(&design).unwrap();
        let imported = import_sv(&sv, Some("InitialMem")).unwrap();
        let mut sim = Simulator::new(&imported.design, "InitialMem").unwrap();
        let addr = signal(&imported, "InitialMem", "addr");
        let rdata = signal(&imported, "InitialMem", "rdata");
        sim.set(addr, 2);
        assert_eq!(sim.get(rdata), 0xab);
    }

    #[test]
    fn reimports_emitted_signed_casts() {
        let mut design = Design::new();
        {
            let mut m = design.module("Casts");
            let a = m.input("a", uint(8));
            let signed_y = m.output("signed_y", sint(8));
            let unsigned_y = m.output("unsigned_y", uint(8));
            m.assign(signed_y, rrtl_core::as_sint(a.value()));
            m.assign(unsigned_y, rrtl_core::as_uint(signed_y.value()));
        }

        let sv = rrtl_sv::emit(&design).unwrap();
        let imported = import_sv(&sv, Some("Casts")).unwrap();
        let mut sim = Simulator::new(&imported.design, "Casts").unwrap();
        let a = signal(&imported, "Casts", "a");
        let signed_y = signal(&imported, "Casts", "signed_y");
        let unsigned_y = signal(&imported, "Casts", "unsigned_y");
        sim.set(a, 0xf0);
        assert_eq!(sim.get(signed_y), 0xf0);
        assert_eq!(sim.get(unsigned_y), 0xf0);
    }

    #[test]
    fn reimports_emitted_sign_extension_replication() {
        let mut design = Design::new();
        {
            let mut m = design.module("SignExtend");
            let a = m.input("a", sint(8));
            let wide = m.output("wide", sint(16));
            m.assign(wide, rrtl_core::sext(a.value() + lit_s(-1, 8), 16));
        }

        let sv = rrtl_sv::emit(&design).unwrap();
        let imported = import_sv(&sv, Some("SignExtend")).unwrap();
        let mut sim = Simulator::new(&imported.design, "SignExtend").unwrap();
        let a = signal(&imported, "SignExtend", "a");
        let wide = signal(&imported, "SignExtend", "wide");
        sim.set(a, 0);
        assert_eq!(sim.get(wide), 0xffff);
        sim.set(a, 2);
        assert_eq!(sim.get(wide), 1);
    }

    #[test]
    fn reimports_emitted_bundle_and_interface_flattened_ports() {
        let req_ty = bundle_type(
            "Req",
            [
                field("valid", uint(1)),
                field("addr", uint(8)),
                nested("meta", bundle_type("ReqMeta", [field("tag", uint(4))])),
            ],
        );
        let mut bundle_design = Design::new();
        {
            let mut m = bundle_design.module("BundlePorts");
            let req = m.input_bundle("req", req_ty);
            let y = m.output("y", uint(8));
            m.assign(y, zext(req.path(["meta", "tag"]).unwrap(), 8));
        }

        let imported = reimport_emitted(&bundle_design, "BundlePorts");
        let mut sim = Simulator::new(&imported.design, "BundlePorts").unwrap();
        let req_tag = signal(&imported, "BundlePorts", "req_meta_tag");
        let y = signal(&imported, "BundlePorts", "y");
        sim.set(req_tag, 0xa);
        assert_eq!(sim.get(y), 0x0a);

        let req_ty = bundle_type(
            "Req",
            [
                field("valid", uint(1)),
                nested("meta", bundle_type("ReqMeta", [field("tag", uint(4))])),
            ],
        );
        let resp_ty = bundle_type("Resp", [field("data", uint(8))]);
        let bus_ty = interface_type(
            "Bus",
            [
                iface_input("req", req_ty),
                iface_output("resp", resp_ty),
                iface_input("ready", uint(1)),
            ],
        );
        let mut interface_design = Design::new();
        {
            let mut m = interface_design.module("InterfacePorts");
            let bus = m.interface("bus", bus_ty);
            let req = bus.port("req").unwrap().bundle().unwrap().clone();
            let resp = bus.port("resp").unwrap().bundle().unwrap().clone();
            m.assign(
                resp.field("data").unwrap(),
                zext(req.path(["meta", "tag"]).unwrap(), 8),
            );
        }

        let imported = reimport_emitted(&interface_design, "InterfacePorts");
        let mut sim = Simulator::new(&imported.design, "InterfacePorts").unwrap();
        let req_tag = signal(&imported, "InterfacePorts", "bus_req_meta_tag");
        let resp_data = signal(&imported, "InterfacePorts", "bus_resp_data");
        sim.set(req_tag, 0xb);
        assert_eq!(sim.get(resp_data), 0x0b);
    }

    #[test]
    fn reimports_emitted_bundle_assignment() {
        let req_ty = bundle_type(
            "Req",
            [
                field("valid", uint(1)),
                field("addr", uint(8)),
                nested("meta", bundle_type("ReqMeta", [field("tag", uint(4))])),
            ],
        );
        let mut design = Design::new();
        {
            let mut m = design.module("BundleAssign");
            let req = m.input_bundle("req", req_ty.clone());
            let wire = m.wire_bundle("pipe", req_ty.clone());
            let resp = m.output_bundle("resp", req_ty);
            m.assign_bundle(&wire, &req);
            m.assign_bundle(&resp, &wire);
        }

        let imported = reimport_emitted(&design, "BundleAssign");
        let mut sim = Simulator::new(&imported.design, "BundleAssign").unwrap();
        let req_addr = signal(&imported, "BundleAssign", "req_addr");
        let req_tag = signal(&imported, "BundleAssign", "req_meta_tag");
        let resp_addr = signal(&imported, "BundleAssign", "resp_addr");
        let resp_tag = signal(&imported, "BundleAssign", "resp_meta_tag");
        sim.set(req_addr, 0x5a);
        sim.set(req_tag, 0xc);
        assert_eq!(sim.get(resp_addr), 0x5a);
        assert_eq!(sim.get(resp_tag), 0xc);
    }

    #[test]
    fn reimports_emitted_external_and_inout_instances() {
        let mut extern_design = Design::new();
        {
            let mut ext = extern_design.extern_module("VendorAddOne");
            ext.input("a", uint(8));
            ext.output("y", uint(8));
        }
        {
            let mut m = extern_design.module("Top");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.instance("u_vendor", "VendorAddOne", [("a", a), ("y", y)]);
        }
        let imported = reimport_emitted(&extern_design, "Top");
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("Top").is_some());

        let mut inout_design = Design::new();
        {
            let mut ext = inout_design.extern_module("IOBUF");
            ext.inout("PAD", uint(1));
            ext.input("I", uint(1));
            ext.input("T", uint(1));
            ext.output("O", uint(1));
        }
        {
            let mut m = inout_design.module("Top");
            let pad = m.inout("pad", uint(1));
            let i = m.input("i", uint(1));
            let t = m.input("t", uint(1));
            let o = m.output("o", uint(1));
            m.instance(
                "u_iobuf",
                "IOBUF",
                [("PAD", pad), ("I", i), ("T", t), ("O", o)],
            );
        }
        let imported = reimport_emitted(&inout_design, "Top");
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("Top").is_some());
    }

    #[test]
    fn reimports_emitted_ready_valid_connect_and_ports() {
        let payload_ty = bundle_type("Payload", [field("data", uint(8)), field("last", uint(1))]);
        let mut ports_design = Design::new();
        {
            let mut m = ports_design.module("ReadyValidPorts");
            let stream = m.rv_source("stream", rv_bundle(payload_ty));
            let bits = stream.bits_bundle().unwrap();
            let fired = m.output("fired", uint(1));
            m.assign(fired, stream.fire());
            m.assign(bits.field("data").unwrap(), lit(0, 8));
        }
        let imported = reimport_emitted(&ports_design, "ReadyValidPorts");
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("ReadyValidPorts").is_some());

        let mut connect_design = Design::new();
        {
            let mut m = connect_design.module("ReadyValidConnect");
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_connect(&input, &output);
        }

        let imported = reimport_emitted(&connect_design, "ReadyValidConnect");
        let mut sim = Simulator::new(&imported.design, "ReadyValidConnect").unwrap();
        let in_valid = signal(&imported, "ReadyValidConnect", "in_valid");
        let in_ready = signal(&imported, "ReadyValidConnect", "in_ready");
        let in_bits = signal(&imported, "ReadyValidConnect", "in_bits");
        let out_valid = signal(&imported, "ReadyValidConnect", "out_valid");
        let out_ready = signal(&imported, "ReadyValidConnect", "out_ready");
        let out_bits = signal(&imported, "ReadyValidConnect", "out_bits");
        sim.set(in_valid, 1);
        sim.set(in_bits, 0x5a);
        sim.set(out_ready, 0);
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_bits), 0x5a);
        assert_eq!(sim.get(in_ready), 0);
        sim.set(out_ready, 1);
        assert_eq!(sim.get(in_ready), 1);
    }

    #[test]
    fn reimports_emitted_ready_valid_register_slice_and_skid() {
        let mut slice_design = Design::new();
        {
            let mut m = slice_design.module("ReadyValidSlice");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_register_slice("slice", &input, &output, clk, rst);
        }
        let imported = reimport_emitted(&slice_design, "ReadyValidSlice");
        let mut sim = Simulator::new(&imported.design, "ReadyValidSlice").unwrap();
        let in_valid = signal(&imported, "ReadyValidSlice", "in_valid");
        let in_ready = signal(&imported, "ReadyValidSlice", "in_ready");
        let in_bits = signal(&imported, "ReadyValidSlice", "in_bits");
        let out_valid = signal(&imported, "ReadyValidSlice", "out_valid");
        let out_ready = signal(&imported, "ReadyValidSlice", "out_ready");
        let out_bits = signal(&imported, "ReadyValidSlice", "out_bits");
        sim.set(out_ready, 1);
        assert_eq!(sim.get(in_ready), 1);
        sim.set(in_valid, 1);
        sim.set(in_bits, 0xaa);
        sim.tick();
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_bits), 0xaa);

        let mut skid_design = Design::new();
        {
            let mut m = skid_design.module("ReadyValidSkid");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_skid_buffer("skid", &input, &output, clk, rst);
        }
        let imported = reimport_emitted(&skid_design, "ReadyValidSkid");
        let mut sim = Simulator::new(&imported.design, "ReadyValidSkid").unwrap();
        let in_valid = signal(&imported, "ReadyValidSkid", "in_valid");
        let in_ready = signal(&imported, "ReadyValidSkid", "in_ready");
        let in_bits = signal(&imported, "ReadyValidSkid", "in_bits");
        let out_valid = signal(&imported, "ReadyValidSkid", "out_valid");
        let out_ready = signal(&imported, "ReadyValidSkid", "out_ready");
        let out_bits = signal(&imported, "ReadyValidSkid", "out_bits");
        sim.set(out_ready, 0);
        sim.set(in_valid, 1);
        sim.set(in_bits, 0x22);
        assert_eq!(sim.get(in_ready), 1);
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_bits), 0x22);
        sim.tick();
        sim.set(in_bits, 0x33);
        assert_eq!(sim.get(in_ready), 0);
        assert_eq!(sim.get(out_bits), 0x22);
    }

    #[test]
    fn reimports_emitted_ready_valid_fifo_and_mem_fifo() {
        let mut fifo_design = Design::new();
        {
            let mut m = fifo_design.module("ReadyValidFifo");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_fifo("fifo", &input, &output, clk, rst, 2);
        }
        let imported = reimport_emitted(&fifo_design, "ReadyValidFifo");
        let mut sim = Simulator::new(&imported.design, "ReadyValidFifo").unwrap();
        let in_valid = signal(&imported, "ReadyValidFifo", "in_valid");
        let in_ready = signal(&imported, "ReadyValidFifo", "in_ready");
        let in_bits = signal(&imported, "ReadyValidFifo", "in_bits");
        let out_valid = signal(&imported, "ReadyValidFifo", "out_valid");
        let out_ready = signal(&imported, "ReadyValidFifo", "out_ready");
        let out_bits = signal(&imported, "ReadyValidFifo", "out_bits");
        sim.set(out_ready, 0);
        sim.set(in_valid, 1);
        sim.set(in_bits, 0x10);
        assert_eq!(sim.get(in_ready), 1);
        sim.tick();
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_bits), 0x10);

        let payload_ty = bundle_type("Payload", [field("data", uint(8)), field("last", uint(1))]);
        let mut mem_fifo_design = Design::new();
        {
            let mut m = mem_fifo_design.module("ReadyValidMemFifo");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_bundle(payload_ty.clone()));
            let output = m.rv_source("out", rv_bundle(payload_ty));
            m.rv_mem_fifo("fifo", &input, &output, clk, rst, 2);
        }
        let imported = reimport_emitted(&mem_fifo_design, "ReadyValidMemFifo");
        let mut sim = Simulator::new(&imported.design, "ReadyValidMemFifo").unwrap();
        let in_valid = signal(&imported, "ReadyValidMemFifo", "in_valid");
        let in_ready = signal(&imported, "ReadyValidMemFifo", "in_ready");
        let in_data = signal(&imported, "ReadyValidMemFifo", "in_bits_data");
        let in_last = signal(&imported, "ReadyValidMemFifo", "in_bits_last");
        let out_valid = signal(&imported, "ReadyValidMemFifo", "out_valid");
        let out_ready = signal(&imported, "ReadyValidMemFifo", "out_ready");
        let out_data = signal(&imported, "ReadyValidMemFifo", "out_bits_data");
        let out_last = signal(&imported, "ReadyValidMemFifo", "out_bits_last");
        sim.set(out_ready, 0);
        sim.set(in_valid, 1);
        sim.set(in_data, 0xa5);
        sim.set(in_last, 1);
        assert_eq!(sim.get(in_ready), 1);
        sim.tick();
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_data), 0xa5);
        assert_eq!(sim.get(out_last), 1);
    }

    #[test]
    fn reimports_emitted_register_file_and_width_helpers() {
        let mut design = Design::new();
        {
            let mut m = design.module("RegFile");
            let clk = m.input("clk", 1);
            let we = m.input("we", 1);
            let waddr = m.input("waddr", 2);
            let raddr = m.input("raddr", 2);
            let wdata = m.input("wdata", 8);
            let rdata = m.output("rdata", 8);
            let low = m.output("low", 4);
            let mem = m.mem("regs", 2, 8, 4);

            m.mem_write(mem, clk, we, waddr, zext(rrtl_core::trunc(wdata, 4), 8));
            let read = m.mem_read(mem, raddr);
            m.assign(rdata, read);
            m.assign(low, rrtl_core::trunc(wdata, 4));
        }

        let imported = reimport_emitted(&design, "RegFile");
        let mut sim = Simulator::new(&imported.design, "RegFile").unwrap();
        let we = signal(&imported, "RegFile", "we");
        let waddr = signal(&imported, "RegFile", "waddr");
        let wdata = signal(&imported, "RegFile", "wdata");
        let raddr = signal(&imported, "RegFile", "raddr");
        let rdata = signal(&imported, "RegFile", "rdata");
        let low = signal(&imported, "RegFile", "low");
        sim.set(wdata, 0xab);
        assert_eq!(sim.get(low), 0xb);
        sim.set(we, 1);
        sim.set(waddr, 2);
        sim.tick();
        sim.set(we, 0);
        sim.set(raddr, 2);
        assert_eq!(sim.get(rdata), 0x0b);
    }

    #[test]
    fn emitted_sv_can_be_reimported_for_supported_subset() {
        let imported = import_sv(
            r#"
            module Alu(a, b, y);
              input logic [3:0] a, b;
              output logic [3:0] y;
              assign y = (a == b) ? a : (a + b);
            endmodule
            "#,
            None,
        )
        .unwrap();
        let emitted = rrtl_sv::emit(&imported.design).unwrap();
        let reimported = import_sv(&emitted, Some("Alu")).unwrap();
        let compiled = compile(&reimported.design).unwrap();
        assert!(compiled.find_module("Alu").is_some());
    }

    #[test]
    fn rejects_multiple_modules_without_top() {
        let err = import_sv(
            "module A(a); input logic a; endmodule module B(b); input logic b; endmodule",
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_TOP");
    }
}
