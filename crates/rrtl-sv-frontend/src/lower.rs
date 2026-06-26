//! Elaboration & lowering: SvSource AST → rrtl_core::Design.
use std::collections::{HashMap, HashSet};

use rrtl_core::{
    as_sint, as_uint, compile, concat, lit_u, mem_read, mux, sint, uint, BitType, Design, Expr,
    Signal,
};
use rrtl_ir::{ErrorReport, Signedness, Width};

use crate::ast::*;
use crate::err;
use crate::parser::Parser;
use crate::SV_FOR_UNROLL_LIMIT;

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

    // Only elaborate modules reachable from the chosen top via its instance
    // tree. Sibling/wrapper modules (e.g. picorv32_axi, which *instantiates*
    // the top, or param-gated submodules pruned away) are irrelevant to the
    // top's design and must not contribute validation errors.
    let reachable = reachable_modules(&source, &top_name);
    for module in &source.modules {
        if !reachable.contains(&module.name) {
            continue;
        }
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

pub(crate) fn specialize_source(source_text: &str, source: SvSource) -> Result<SvSource, ErrorReport> {
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

// ============================================================================
// Function inlining: SystemVerilog functions are pure combinational, so a call
// `f(args)` is replaced by symbolically executing the body's blocking assignments
// with the inputs bound to the (already-inlined) argument expressions, yielding the
// return value as one expression. Done as a pre-pass so the rest of lowering never
// sees a call.
// ============================================================================

/// Inline a single call: bind inputs to `args`, execute the body, return the
/// expression assigned to the function's name.
fn inline_call(
    func: &SvFunction,
    args: &[SvExpr],
    functions: &HashMap<String, SvFunction>,
) -> Result<SvExpr, ErrorReport> {
    if args.len() != func.inputs.len() {
        return Err(err(
            "E_SV_FUNCTION",
            format!(
                "function `{}` expects {} argument(s), got {}",
                func.name,
                func.inputs.len(),
                args.len()
            ),
        ));
    }
    let mut vals: HashMap<String, SvExpr> = HashMap::new();
    let zero = |ty: SvType| SvExpr::Lit { value: 0, width: ty.width, signed: ty.signed };
    for ((pname, _), arg) in func.inputs.iter().zip(args) {
        vals.insert(pname.clone(), arg.clone());
    }
    for local in &func.locals {
        for d in &local.names {
            vals.insert(d.name.clone(), zero(local.ty));
        }
    }
    vals.insert(func.name.clone(), zero(func.return_type));
    exec_stmts(&func.body, &mut vals, functions)?;
    vals.remove(&func.name).ok_or_else(|| {
        err("E_SV_FUNCTION", format!("function `{}` never assigns a result", func.name))
    })
}

fn exec_stmts(
    stmts: &[SvStmt],
    vals: &mut HashMap<String, SvExpr>,
    functions: &HashMap<String, SvFunction>,
) -> Result<(), ErrorReport> {
    for stmt in stmts {
        exec_stmt(stmt, vals, functions)?;
    }
    Ok(())
}

fn exec_stmt(
    stmt: &SvStmt,
    vals: &mut HashMap<String, SvExpr>,
    functions: &HashMap<String, SvFunction>,
) -> Result<(), ErrorReport> {
    match stmt {
        SvStmt::Assign { dst: SvLvalue::Signal(name), expr, .. } => {
            let v = subst(expr, vals, functions)?;
            vals.insert(name.clone(), v);
        }
        SvStmt::If { cond, then_stmts, else_stmts } => {
            let c = subst(cond, vals, functions)?;
            let mut tv = vals.clone();
            exec_stmts(then_stmts, &mut tv, functions)?;
            let mut ev = vals.clone();
            exec_stmts(else_stmts, &mut ev, functions)?;
            // merge: ternary where the branches disagree (all keys present in both
            // clones, plus pre-seeded locals, so both sides are always defined).
            let keys: HashSet<String> = tv.keys().chain(ev.keys()).cloned().collect();
            for k in keys {
                let t = tv.get(&k).cloned();
                let e = ev.get(&k).cloned();
                let merged = match (t, e) {
                    (Some(t), Some(e)) if t == e => t,
                    (Some(t), Some(e)) => SvExpr::Ternary {
                        cond: Box::new(c.clone()),
                        then_expr: Box::new(t),
                        else_expr: Box::new(e),
                    },
                    (Some(t), None) => t,
                    (None, Some(e)) => e,
                    (None, None) => continue,
                };
                vals.insert(k, merged);
            }
        }
        SvStmt::Nop | SvStmt::Assert { .. } | SvStmt::Cover { .. } => {}
        other => {
            return Err(err(
                "E_SV_FUNCTION",
                format!("unsupported statement in function body: {other:?}"),
            ));
        }
    }
    Ok(())
}

/// Substitute local/input variables (from `vals`) and inline nested calls in an
/// expression, leaving free identifiers (module signals) untouched.
fn subst(
    expr: &SvExpr,
    vals: &HashMap<String, SvExpr>,
    functions: &HashMap<String, SvFunction>,
) -> Result<SvExpr, ErrorReport> {
    let s = |e: &SvExpr| subst(e, vals, functions);
    Ok(match expr {
        SvExpr::Ident(name) => vals.get(name).cloned().unwrap_or_else(|| expr.clone()),
        SvExpr::Call { name, args } => {
            let new_args = args.iter().map(|a| s(a)).collect::<Result<Vec<_>, _>>()?;
            let func = functions.get(name).ok_or_else(|| {
                err("E_SV_FUNCTION", format!("call to unknown function `{name}`"))
            })?;
            inline_call(func, &new_args, functions)?
        }
        SvExpr::Lit { .. } => expr.clone(),
        SvExpr::Unary { op, expr } => SvExpr::Unary { op: *op, expr: Box::new(s(expr)?) },
        SvExpr::Binary { op, lhs, rhs } => SvExpr::Binary {
            op: *op,
            lhs: Box::new(s(lhs)?),
            rhs: Box::new(s(rhs)?),
        },
        SvExpr::Ternary { cond, then_expr, else_expr } => SvExpr::Ternary {
            cond: Box::new(s(cond)?),
            then_expr: Box::new(s(then_expr)?),
            else_expr: Box::new(s(else_expr)?),
        },
        SvExpr::Concat(parts) => {
            SvExpr::Concat(parts.iter().map(|p| s(p)).collect::<Result<Vec<_>, _>>()?)
        }
        SvExpr::Repeat { count, expr } => {
            SvExpr::Repeat { count: *count, expr: Box::new(s(expr)?) }
        }
        SvExpr::Cast { signed, expr } => SvExpr::Cast { signed: *signed, expr: Box::new(s(expr)?) },
        SvExpr::Index { expr, index } => SvExpr::Index { expr: Box::new(s(expr)?), index: *index },
        SvExpr::Slice { expr, msb, lsb } => {
            SvExpr::Slice { expr: Box::new(s(expr)?), msb: *msb, lsb: *lsb }
        }
        SvExpr::MemRead { name, addr } => {
            SvExpr::MemRead { name: name.clone(), addr: Box::new(s(addr)?) }
        }
        SvExpr::Bracket { expr, index } => {
            SvExpr::Bracket { expr: Box::new(s(expr)?), index: Box::new(s(index)?) }
        }
    })
}

/// Pre-pass: collect functions, inline every call in the (generate-expanded) item
/// list, and drop the function definitions.
fn expand_functions(items: Vec<SvItem>) -> Result<Vec<SvItem>, ErrorReport> {
    let mut functions: HashMap<String, SvFunction> = HashMap::new();
    for item in &items {
        if let SvItem::Function(f) = item {
            functions.insert(f.name.clone(), f.clone());
        }
    }
    if functions.is_empty() {
        return Ok(items);
    }
    let empty: HashMap<String, SvExpr> = HashMap::new();
    let ie = |e: &SvExpr| subst(e, &empty, &functions);
    let is = |s: &SvStmt| rewrite_stmt(s, &functions);
    let mut out = Vec::new();
    for item in items {
        let rewritten = match item {
            SvItem::Function(_) => continue,
            SvItem::Assign { dst, expr } => SvItem::Assign { dst, expr: ie(&expr)? },
            SvItem::AlwaysComb(stmts) => SvItem::AlwaysComb(
                stmts.iter().map(&is).collect::<Result<Vec<_>, _>>()?,
            ),
            SvItem::AlwaysFf(mut ff) => {
                ff.body = ff.body.iter().map(&is).collect::<Result<Vec<_>, _>>()?;
                SvItem::AlwaysFf(ff)
            }
            other => other,
        };
        out.push(rewritten);
    }
    Ok(out)
}

fn rewrite_stmt(
    stmt: &SvStmt,
    functions: &HashMap<String, SvFunction>,
) -> Result<SvStmt, ErrorReport> {
    let empty: HashMap<String, SvExpr> = HashMap::new();
    let ie = |e: &SvExpr| subst(e, &empty, functions);
    let rs = |s: &SvStmt| rewrite_stmt(s, functions);
    Ok(match stmt {
        SvStmt::Assign { dst, nonblocking, expr } => SvStmt::Assign {
            dst: dst.clone(),
            nonblocking: *nonblocking,
            expr: ie(expr)?,
        },
        SvStmt::ConcatAssign { parts, nonblocking, expr } => SvStmt::ConcatAssign {
            parts: parts.clone(),
            nonblocking: *nonblocking,
            expr: ie(expr)?,
        },
        SvStmt::If { cond, then_stmts, else_stmts } => SvStmt::If {
            cond: ie(cond)?,
            then_stmts: then_stmts.iter().map(&rs).collect::<Result<Vec<_>, _>>()?,
            else_stmts: else_stmts.iter().map(&rs).collect::<Result<Vec<_>, _>>()?,
        },
        SvStmt::For { var, init, cmp, bound, step, body } => SvStmt::For {
            var: var.clone(),
            init: ie(init)?,
            cmp: *cmp,
            bound: ie(bound)?,
            step: step.clone(),
            body: body.iter().map(&rs).collect::<Result<Vec<_>, _>>()?,
        },
        SvStmt::Case { kind, expr, items } => {
            let mut new_items = Vec::new();
            for it in items {
                new_items.push(rewrite_case_item(it, functions)?);
            }
            SvStmt::Case { kind: *kind, expr: ie(expr)?, items: new_items }
        }
        SvStmt::Assert { name, cond, message } => SvStmt::Assert {
            name: name.clone(),
            cond: ie(cond)?,
            message: message.clone(),
        },
        SvStmt::Cover { name, cond } => SvStmt::Cover { name: name.clone(), cond: ie(cond)? },
        SvStmt::Nop => SvStmt::Nop,
    })
}

fn rewrite_case_item(
    item: &SvCaseItem,
    functions: &HashMap<String, SvFunction>,
) -> Result<SvCaseItem, ErrorReport> {
    let mut it = item.clone();
    it.stmts = it
        .stmts
        .iter()
        .map(|s| rewrite_stmt(s, functions))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(it)
}

/// Register every enum variant as a named constant so `IDLE`/`S_RUN`/… resolve
/// like a localparam in expressions, case labels, and assignments. An omitted
/// `= value` auto-increments from the previous variant (0 for the first).
fn register_enum_consts(
    items: &[SvItem],
    consts: &mut HashMap<String, u128>,
) -> Result<(), ErrorReport> {
    for item in items {
        if let SvItem::TypeDef(td) = item {
            let mut next = 0u128;
            for (name, value) in &td.variants {
                let value = match value {
                    Some(expr) => const_eval(expr, consts)?,
                    None => next,
                };
                consts.insert(name.clone(), value);
                next = value.wrapping_add(1);
            }
        }
    }
    Ok(())
}

fn lower_module(
    module: &SvModule,
    design: &mut Design,
    defined: &HashSet<String>,
    port_maps: &HashMap<String, HashMap<String, (BitType, SvDirection)>>,
    externs: &mut HashMap<String, HashMap<String, (BitType, SvDirection)>>,
) -> Result<(), ErrorReport> {
    let mut builder = design.module(&module.name);
    let mut symbols = Symbols::default();
    let mut consts = HashMap::<String, u128>::new();
    for param in &module.params {
        let value = const_eval(&param.value, &consts)?;
        consts.insert(param.name.clone(), value);
    }
    register_enum_consts(&module.items, &mut consts)?;
    // Module-level localparams must be available to generate-for bounds, which are
    // evaluated during expansion — collect them before expanding (in declaration
    // order, so later params may reference earlier ones).
    for item in &module.items {
        if let SvItem::Param(param) = item {
            let value = const_eval(&param.value, &consts)?;
            consts.insert(param.name.clone(), value);
        }
    }
    let items = expand_generate_items(&module.items, &consts)?;
    let items = expand_functions(items)?; // inline function calls, drop defs
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
                            // A `reg`/`logic` is a clocked register only if driven by an
                            // always_ff; otherwise it is a combinational variable (wire),
                            // even when declared `reg` (Verilog `reg` is just a variable).
                            SvDeclKind::Reg | SvDeclKind::Logic
                                if ff_targets.contains(&declarator.name) =>
                            {
                                builder.reg(&declarator.name, ty)
                            }
                            SvDeclKind::Reg | SvDeclKind::Logic | SvDeclKind::Wire => {
                                builder.wire(&declarator.name, ty)
                            }
                        },
                    };
                    if symbols
                        .by_name
                        .insert(
                            declarator.name.clone(),
                            SvSymbol {
                                signal,
                                ty,
                                direction: decl.direction,
                                is_memory: declarator.memory_depth.is_some(),
                                mem_addr_width: declarator
                                    .memory_depth
                                    .map(ceil_log2)
                                    .unwrap_or(0),
                                mem_inner: declarator.memory_inner.unwrap_or(0),
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
                    symbols.by_signal.insert(signal, ty);
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
            SvItem::Function(_) => {} // inlined away by expand_functions
            SvItem::Generate(_) | SvItem::GenerateFor { .. } => {}
            SvItem::Decl(decl) => {
                // `wire/logic name = expr;` — lower the inline continuous assign now
                // that every signal is declared.
                for declarator in &decl.names {
                    if let Some(init) = &declarator.init {
                        lower_continuous_lvalue_assign(
                            &SvLvalue::Signal(declarator.name.clone()),
                            init,
                            &symbols,
                            &consts,
                            &mut continuous_state,
                        )?;
                    }
                }
            }
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
    // Generate-scope localparams (e.g. `localparam K = I+i-1;`) accumulate here as
    // they are encountered, so later items in the same scope can use them.
    let mut local_consts = consts.clone();
    let mut expanded = Vec::new();
    for item in items {
        match item {
            SvItem::Generate(items) => {
                expanded.extend(expand_generate_items_with_prefix(
                    items,
                    &local_consts,
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
                for iter_consts in
                    for_loop_iteration_consts(var, init, *cmp, bound, step, &local_consts)?
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
            // A generate-scope localparam: fold it into the scope's constants and
            // drop the item (it is not a runtime signal).
            SvItem::Param(param) if !prefix.is_empty() => {
                let value = const_eval(&param.value, &local_consts)?;
                local_consts.insert(param.name.clone(), value);
            }
            SvItem::TypeDef(_) if !prefix.is_empty() => {
                return Err(err("E_SV_GENERATE", "generated typedefs are not supported"));
            }
            SvItem::Instance(instance) => {
                let mut instance = substitute_instance(instance, &local_consts, &scope_renames)?;
                if !prefix.is_empty() {
                    instance.name = format!("{prefix}{}", instance.name);
                }
                expanded.push(SvItem::Instance(instance));
            }
            _ => expanded.push(substitute_item(item, &local_consts, &scope_renames)?),
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
            // Generate-scope localparams are folded into constants during
            // expansion (handled below), not renamed as signals.
            SvItem::Param(_) => {}
            SvItem::TypeDef(_) => {
                return Err(err(
                    "E_SV_GENERATE",
                    "generated typedefs are not supported",
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
        SvItem::Function(f) => SvItem::Function(f.clone()), // inlined after generate expansion
        SvItem::Generate(items) => SvItem::Generate(items.clone()),
        SvItem::GenerateFor { .. } => item.clone(),
        SvItem::Assign { dst, expr } => SvItem::Assign {
            dst: substitute_lvalue(dst, consts, renames)?,
            expr: substitute_expr(expr, consts, renames)?,
        },
        SvItem::Initial(assigns) => SvItem::Initial(
            assigns
                .iter()
                .map(|assign| match assign {
                    SvInitialAssign::Assign { dst, expr } => Ok(SvInitialAssign::Assign {
                        dst: substitute_lvalue(dst, consts, renames)?,
                        expr: substitute_expr(expr, consts, renames)?,
                    }),
                    SvInitialAssign::ReadMem { .. } => Ok(assign.clone()),
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
                SvStmt::ConcatAssign {
                    parts,
                    nonblocking,
                    expr,
                } => SvStmt::ConcatAssign {
                    parts: parts
                        .iter()
                        .map(|p| substitute_lvalue(p, consts, renames))
                        .collect::<Result<_, _>>()?,
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
                SvStmt::Nop => SvStmt::Nop,
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
        SvLvalue::Bit { name, index } => {
            // An interface-array element member write resolves to a flat signal.
            if let Some((array, member)) = parse_iface_array_ref(name) {
                let idx_expr = substitute_expr(index, consts, renames)?;
                let idx = const_eval(&idx_expr, consts)?;
                SvLvalue::Signal(format!("{array}.{idx}.{member}"))
            } else {
                SvLvalue::Bit {
                    name: rename_ident(name, renames),
                    index: substitute_expr(index, consts, renames)?,
                }
            }
        }
        SvLvalue::Slice { name, msb, lsb } => SvLvalue::Slice {
            name: rename_ident(name, renames),
            msb: substitute_expr(msb, consts, renames)?,
            lsb: substitute_expr(lsb, consts, renames)?,
        },
        SvLvalue::Memory { name, addr } => SvLvalue::Memory {
            name: rename_ident(name, renames),
            addr: substitute_expr(addr, consts, renames)?,
        },
        SvLvalue::Memory2D { name, outer, inner } => SvLvalue::Memory2D {
            name: rename_ident(name, renames),
            outer: substitute_expr(outer, consts, renames)?,
            inner: substitute_expr(inner, consts, renames)?,
        },
    })
}

/// Prefix marking an interface-array element member access whose index is only
/// known after generate-for unrolling. The parser encodes `bus[i].member` as a
/// `Bracket { Ident("<sentinel>bus\0member"), i }`; once the genvar `i` is folded
/// to a constant during substitution, it resolves to the flat signal
/// `bus.<i>.member`.
pub(crate) const IFACE_ARRAY_SENTINEL: &str = "\u{0}IFA\u{0}";

pub(crate) fn iface_array_ref(array: &str, member: &str) -> String {
    format!("{IFACE_ARRAY_SENTINEL}{array}\u{0}{member}")
}

fn parse_iface_array_ref(name: &str) -> Option<(&str, &str)> {
    let rest = name.strip_prefix(IFACE_ARRAY_SENTINEL)?;
    let mut parts = rest.splitn(2, '\u{0}');
    Some((parts.next()?, parts.next()?))
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
        SvExpr::Bracket { expr, index } => {
            // Resolve an interface-array element access once its index is constant.
            if let SvExpr::Ident(name) = expr.as_ref() {
                if let Some((array, member)) = parse_iface_array_ref(name) {
                    let idx_expr = substitute_expr(index, consts, renames)?;
                    let idx = const_eval(&idx_expr, consts)?;
                    return Ok(SvExpr::Ident(format!("{array}.{idx}.{member}")));
                }
            }
            SvExpr::Bracket {
                expr: Box::new(substitute_expr(expr, consts, renames)?),
                index: Box::new(substitute_expr(index, consts, renames)?),
            }
        }
        SvExpr::Call { name, args } => SvExpr::Call {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| substitute_expr(a, consts, renames))
                .collect::<Result<Vec<_>, _>>()?,
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
    symbols: &Symbols,
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
                Some(ty.width),
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
    symbols: &Symbols,
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
    symbols: &Symbols,
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
    symbols: &Symbols,
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
    symbols: &Symbols,
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
            SvStmt::ConcatAssign { parts, .. } => {
                for part in parts {
                    let (SvLvalue::Signal(name)
                    | SvLvalue::Bit { name, .. }
                    | SvLvalue::Slice { name, .. }
                    | SvLvalue::Memory { name, .. }
                    | SvLvalue::Memory2D { name, .. }) = part;
                    targets.insert(name.clone());
                }
            }
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
            SvStmt::Assert { .. } | SvStmt::Cover { .. } | SvStmt::Nop => {}
        }
    }
}

#[derive(Clone, Copy)]
struct SvSymbol {
    signal: Signal,
    ty: BitType,
    direction: Option<SvDirection>,
    is_memory: bool,
    /// For memories, the address (index) width = ceil_log2(depth); else 0.
    mem_addr_width: Width,
    /// Inner dimension D2 of a 2-D unpacked array (row-major flatten factor: a 2-D
    /// access `mem[i][j]` → flat address `i*D2 + j`); 0 for a 1-D memory.
    mem_inner: usize,
}

/// Module symbol table: the by-name map (threaded everywhere as the read view via
/// `Deref`) plus a by-signal reverse index, so `signal → type` lookups are O(1)
/// instead of a linear `values().find()` scan — which was O(n²) over a wide module
/// (the `import_sv` scale bottleneck).
#[derive(Default)]
struct Symbols {
    by_name: HashMap<String, SvSymbol>,
    by_signal: HashMap<Signal, BitType>,
}

impl std::ops::Deref for Symbols {
    type Target = HashMap<String, SvSymbol>;
    fn deref(&self) -> &Self::Target {
        &self.by_name
    }
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
        symbols: &Symbols,
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
    symbols: &Symbols,
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
    symbols: &Symbols,
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

/// The width of a concat-lvalue part.
fn lvalue_width(
    lv: &SvLvalue,
    symbols: &Symbols,
    consts: &HashMap<String, u128>,
) -> Result<Width, ErrorReport> {
    Ok(match lv {
        SvLvalue::Signal(name) | SvLvalue::Memory { name, .. } | SvLvalue::Memory2D { name, .. } => symbols
            .get(name)
            .map(|s| s.ty.width)
            .ok_or_else(|| err("E_SV_CONCAT", format!("unknown signal `{name}` in concat lvalue")))?,
        SvLvalue::Bit { .. } => 1,
        SvLvalue::Slice { msb, lsb, .. } => {
            let m = const_eval(msb, consts)?;
            let l = const_eval(lsb, consts)?;
            if m < l {
                return Err(err("E_SV_CONCAT", "concat-lvalue slice msb < lsb"));
            }
            (m - l + 1) as Width
        }
    })
}

/// Desugar `{p0, p1, …} = rhs` into per-part assignments, splitting the RHS
/// MSB-first by each part's width (`p0` is the most-significant).
fn desugar_concat_assign(
    parts: &[SvLvalue],
    nonblocking: bool,
    rhs: &SvExpr,
    symbols: &Symbols,
    consts: &HashMap<String, u128>,
) -> Result<Vec<SvStmt>, ErrorReport> {
    let widths: Vec<Width> = parts
        .iter()
        .map(|p| lvalue_width(p, symbols, consts))
        .collect::<Result<_, _>>()?;
    let total: Width = widths.iter().sum();
    let mut out = Vec::with_capacity(parts.len());
    let mut hi = total;
    for (part, w) in parts.iter().zip(&widths) {
        let lo = hi - w;
        out.push(SvStmt::Assign {
            dst: part.clone(),
            nonblocking,
            expr: SvExpr::Slice {
                expr: Box::new(rhs.clone()),
                msb: hi - 1,
                lsb: lo,
            },
        });
        hi = lo;
    }
    Ok(out)
}

fn lower_comb_stmt(
    stmt: &SvStmt,
    symbols: &Symbols,
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
        SvStmt::ConcatAssign {
            parts,
            nonblocking,
            expr,
        } => {
            let mut assigned = HashSet::new();
            for s in &desugar_concat_assign(parts, *nonblocking, expr, symbols, consts)? {
                assigned.extend(lower_comb_stmt(s, symbols, consts, builder, state, enable.clone())?);
            }
            Ok(assigned)
        }
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
        SvStmt::Nop => Ok(HashSet::new()),
    }
}

fn lower_comb_if(
    cond: &SvExpr,
    then_stmts: &[SvStmt],
    else_stmts: &[SvStmt],
    symbols: &Symbols,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
    state: &mut CombState,
    enable: Option<Expr>,
) -> Result<HashSet<String>, ErrorReport> {
    let cond = lower_cond(cond, symbols, consts)?;
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
    symbols: &Symbols,
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
    symbols: &Symbols,
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
    symbols: &Symbols,
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
    value_width: Option<Width>,
    symbols: &Symbols,
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
    let next =
        lower_lvalue_assignment_value(dst, value, value_width, base, signal, symbols, consts)?;
    state.set(name.clone(), next);
    Ok(name)
}

fn lower_comb_case(
    kind: SvCaseKind,
    expr: &SvExpr,
    items: &[SvCaseItem],
    symbols: &Symbols,
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
    // A comb `case` with no default and incomplete coverage falls through to the
    // signal's prior value — an inferred latch. For a signal first written in this
    // block that prior value is the signal itself, i.e. a false self-loop (the
    // fall-through is unreachable in practice, e.g. picorv32's `full_case` mem
    // decode where mem_wordsize is never 3). Seed such signals with the last
    // branch; signals with a real prior comb value keep it.
    let no_default = default_state.is_none() && !branches.is_empty();
    for name in assigned.iter().cloned() {
        let self_ref = no_default && !base.values.contains_key(&name);
        let (mut expr, fold) = if let Some(default_state) = &default_state {
            (default_state.expr_for(&name, symbols)?, &branches[..])
        } else if self_ref {
            let (_, last_state) = branches.last().unwrap();
            (
                last_state.expr_for(&name, symbols)?,
                &branches[..branches.len() - 1],
            )
        } else {
            (base.expr_for(&name, symbols)?, &branches[..])
        };
        for (match_expr, branch_state) in fold.iter().rev() {
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

/// True when a plain `case` (no default) provably covers every value of its
/// selector: all labels are constants and the distinct count equals 2^width.
/// Conservatively false for casez/casex (wildcards), unknown width, or any
/// non-constant label.
fn case_is_exhaustive(
    kind: SvCaseKind,
    items: &[SvCaseItem],
    selector_width: Option<Width>,
    consts: &HashMap<String, u128>,
) -> bool {
    if kind != SvCaseKind::Normal {
        return false;
    }
    let Some(width) = selector_width else {
        return false;
    };
    if width == 0 || width > 16 {
        return false;
    }
    let mut values = HashSet::new();
    for item in items {
        if item.is_default {
            return true;
        }
        for label in &item.labels {
            match label {
                SvCaseLabel::Expr(e) => match const_eval(e, consts) {
                    Ok(v) => {
                        values.insert(v & value_mask(width));
                    }
                    Err(_) => return false,
                },
                SvCaseLabel::Wildcard { .. } => return false,
            }
        }
    }
    values.len() as u128 == (1u128 << width)
}

fn case_match_expr(
    kind: SvCaseKind,
    selector: &Expr,
    selector_width: Option<Width>,
    labels: &[SvCaseLabel],
    symbols: &Symbols,
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
    symbols: &Symbols,
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

pub(crate) fn resize_u128_to_width(value: u128, from_width: Width, to_width: Width) -> u128 {
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
    symbols: &Symbols,
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

/// The base signal name of an lvalue.
fn lvalue_base_name(lv: &SvLvalue) -> &str {
    match lv {
        SvLvalue::Signal(n)
        | SvLvalue::Bit { name: n, .. }
        | SvLvalue::Slice { name: n, .. }
        | SvLvalue::Memory { name: n, .. }
        | SvLvalue::Memory2D { name: n, .. } => n,
    }
}

/// Replace forwarded identifiers (blocking-assigned-so-far) with their values.
fn subst_forward(expr: &SvExpr, fwd: &HashMap<String, SvExpr>) -> SvExpr {
    match expr {
        SvExpr::Ident(name) => fwd
            .get(name)
            .cloned()
            .unwrap_or_else(|| SvExpr::Ident(name.clone())),
        SvExpr::Lit { .. } => expr.clone(),
        SvExpr::Unary { op, expr } => SvExpr::Unary {
            op: *op,
            expr: Box::new(subst_forward(expr, fwd)),
        },
        SvExpr::Binary { op, lhs, rhs } => SvExpr::Binary {
            op: *op,
            lhs: Box::new(subst_forward(lhs, fwd)),
            rhs: Box::new(subst_forward(rhs, fwd)),
        },
        SvExpr::Ternary { cond, then_expr, else_expr } => SvExpr::Ternary {
            cond: Box::new(subst_forward(cond, fwd)),
            then_expr: Box::new(subst_forward(then_expr, fwd)),
            else_expr: Box::new(subst_forward(else_expr, fwd)),
        },
        SvExpr::Concat(parts) => {
            SvExpr::Concat(parts.iter().map(|p| subst_forward(p, fwd)).collect())
        }
        SvExpr::Repeat { count, expr } => SvExpr::Repeat {
            count: *count,
            expr: Box::new(subst_forward(expr, fwd)),
        },
        SvExpr::Cast { signed, expr } => SvExpr::Cast {
            signed: *signed,
            expr: Box::new(subst_forward(expr, fwd)),
        },
        SvExpr::Index { expr, index } => SvExpr::Index {
            expr: Box::new(subst_forward(expr, fwd)),
            index: *index,
        },
        SvExpr::Slice { expr, msb, lsb } => SvExpr::Slice {
            expr: Box::new(subst_forward(expr, fwd)),
            msb: *msb,
            lsb: *lsb,
        },
        // memory names aren't forwarded (they are written via mem_write)
        SvExpr::MemRead { name, addr } => SvExpr::MemRead {
            name: name.clone(),
            addr: Box::new(subst_forward(addr, fwd)),
        },
        SvExpr::Bracket { expr, index } => SvExpr::Bracket {
            expr: Box::new(subst_forward(expr, fwd)),
            index: Box::new(subst_forward(index, fwd)),
        },
        SvExpr::Call { name, args } => SvExpr::Call {
            name: name.clone(),
            args: args.iter().map(|a| subst_forward(a, fwd)).collect(),
        },
    }
}

/// Eliminate blocking assignments in a clocked block by forward-substituting
/// their values into later reads (so the block lowers as if all-nonblocking).
/// `if` merges forward values with a `Ternary`; `case`/`for` are conservative
/// (their blocking-assigned names stop forwarding past the construct).
/// Build the match condition for a (non-default, non-wildcard) case branch:
/// `(sel == label0) || (sel == label1) || ...`. Returns None if any label is a
/// wildcard (casez/casex), which this path does not merge.
fn case_branch_cond(sel: &SvExpr, labels: &[SvCaseLabel]) -> Option<SvExpr> {
    let mut cond: Option<SvExpr> = None;
    for label in labels {
        let SvCaseLabel::Expr(e) = label else {
            return None;
        };
        let eq = SvExpr::Binary {
            op: SvBinaryOp::Eq,
            lhs: Box::new(sel.clone()),
            rhs: Box::new(e.clone()),
        };
        cond = Some(match cond {
            Some(c) => SvExpr::Binary {
                op: SvBinaryOp::LogOr,
                lhs: Box::new(c),
                rhs: Box::new(eq),
            },
            None => eq,
        });
    }
    cond
}

fn eliminate_blocking(stmts: &[SvStmt], fwd: &mut HashMap<String, SvExpr>) -> Vec<SvStmt> {
    stmts.iter().map(|s| eliminate_blocking_stmt(s, fwd)).collect()
}

fn eliminate_blocking_stmt(stmt: &SvStmt, fwd: &mut HashMap<String, SvExpr>) -> SvStmt {
    match stmt {
        SvStmt::Assign { dst, nonblocking, expr } => {
            let rhs = subst_forward(expr, fwd);
            if !*nonblocking {
                match dst {
                    SvLvalue::Signal(name) => {
                        fwd.insert(name.clone(), rhs.clone());
                    }
                    other => {
                        // partial blocking write: stop forwarding the whole signal
                        fwd.remove(lvalue_base_name(other));
                    }
                }
            }
            SvStmt::Assign { dst: dst.clone(), nonblocking: true, expr: rhs }
        }
        SvStmt::ConcatAssign { parts, nonblocking, expr } => {
            let rhs = subst_forward(expr, fwd);
            if !*nonblocking {
                for p in parts {
                    fwd.remove(lvalue_base_name(p));
                }
            }
            SvStmt::ConcatAssign { parts: parts.clone(), nonblocking: true, expr: rhs }
        }
        SvStmt::If { cond, then_stmts, else_stmts } => {
            let cond_s = subst_forward(cond, fwd);
            // `if (x)` means `x != 0`; the forward-merge mux needs a 1-bit select.
            let mux_cond = SvExpr::Unary {
                op: SvUnaryOp::RedOr,
                expr: Box::new(cond_s.clone()),
            };
            let mut tf = fwd.clone();
            let then_s = eliminate_blocking(then_stmts, &mut tf);
            let mut ef = fwd.clone();
            let else_s = eliminate_blocking(else_stmts, &mut ef);
            let mut keys: Vec<String> = Vec::new();
            for k in tf.keys().chain(ef.keys()) {
                if !keys.contains(k) {
                    keys.push(k.clone());
                }
            }
            for k in keys {
                let pre = fwd
                    .get(&k)
                    .cloned()
                    .unwrap_or_else(|| SvExpr::Ident(k.clone()));
                let tv = tf.get(&k).cloned().unwrap_or_else(|| pre.clone());
                let ev = ef.get(&k).cloned().unwrap_or_else(|| pre.clone());
                let merged = if tv == ev {
                    tv
                } else {
                    SvExpr::Ternary {
                        cond: Box::new(mux_cond.clone()),
                        then_expr: Box::new(tv),
                        else_expr: Box::new(ev),
                    }
                };
                fwd.insert(k, merged);
            }
            SvStmt::If { cond: cond_s, then_stmts: then_s, else_stmts: else_s }
        }
        SvStmt::Case { kind, expr, items } => {
            let sel_s = subst_forward(expr, fwd);
            // Process each branch under a cloned forward map, remembering the
            // per-branch result and its match condition so blocking writes can
            // be merged across branches (priority mux), just like `if`. Falls
            // back to the conservative drop for casez/casex or wildcard labels
            // (a masked compare we don't synthesize here).
            let mergeable = *kind == SvCaseKind::Normal
                && items.iter().all(|item| {
                    item.is_default
                        || item
                            .labels
                            .iter()
                            .all(|l| matches!(l, SvCaseLabel::Expr(_)))
                });
            let mut new_items = Vec::with_capacity(items.len());
            let mut branches: Vec<(Option<SvExpr>, HashMap<String, SvExpr>)> = Vec::new();
            let mut assigned = HashSet::new();
            for item in items {
                let mut bf = fwd.clone();
                let stmts_s = eliminate_blocking(&item.stmts, &mut bf);
                collect_stmt_signal_targets(&item.stmts, &mut assigned);
                let cond = if item.is_default {
                    None
                } else {
                    case_branch_cond(&sel_s, &item.labels)
                };
                branches.push((cond, bf));
                new_items.push(SvCaseItem {
                    labels: item.labels.clone(),
                    stmts: stmts_s,
                    is_default: item.is_default,
                });
            }
            if mergeable {
                let default_fwd = branches
                    .iter()
                    .find(|(cond, _)| cond.is_none())
                    .map(|(_, f)| f.clone());
                for name in assigned {
                    let pre = fwd
                        .get(&name)
                        .cloned()
                        .unwrap_or_else(|| SvExpr::Ident(name.clone()));
                    // Seed with the default branch's value (or the pre-case value
                    // when there is no default), then fold non-default branches in
                    // reverse so earlier branches take priority.
                    let mut merged = default_fwd
                        .as_ref()
                        .and_then(|f| f.get(&name).cloned())
                        .unwrap_or_else(|| pre.clone());
                    for (cond, bf) in branches.iter().rev() {
                        let Some(cond) = cond else { continue };
                        let bv = bf.get(&name).cloned().unwrap_or_else(|| pre.clone());
                        if bv != merged {
                            merged = SvExpr::Ternary {
                                cond: Box::new(cond.clone()),
                                then_expr: Box::new(bv),
                                else_expr: Box::new(merged),
                            };
                        }
                    }
                    fwd.insert(name, merged);
                }
            } else {
                for name in assigned {
                    fwd.remove(&name);
                }
            }
            SvStmt::Case { kind: *kind, expr: sel_s, items: new_items }
        }
        SvStmt::For { var, init, cmp, bound, step, body } => {
            let init_s = subst_forward(init, fwd);
            let bound_s = subst_forward(bound, fwd);
            let mut bf = fwd.clone();
            let body_s = eliminate_blocking(body, &mut bf);
            let mut assigned = HashSet::new();
            collect_stmt_signal_targets(body, &mut assigned);
            for name in assigned {
                fwd.remove(&name);
            }
            SvStmt::For {
                var: var.clone(),
                init: init_s,
                cmp: *cmp,
                bound: bound_s,
                step: step.clone(),
                body: body_s,
            }
        }
        SvStmt::Nop => SvStmt::Nop,
        SvStmt::Assert { name, cond, message } => SvStmt::Assert {
            name: name.clone(),
            cond: subst_forward(cond, fwd),
            message: message.clone(),
        },
        SvStmt::Cover { name, cond } => SvStmt::Cover {
            name: name.clone(),
            cond: subst_forward(cond, fwd),
        },
    }
}

fn lower_always_ff(
    always: &SvAlwaysFf,
    symbols: &Symbols,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
) -> Result<(), ErrorReport> {
    let clock = symbols
        .get(&always.clock)
        .ok_or_else(|| err("E_SV_CLOCK", format!("unknown clock `{}`", always.clock)))?
        .signal;
    let mut state = FfState::default();
    let body = eliminate_blocking(&always.body, &mut HashMap::new());
    for stmt in &body {
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
        symbols: &Symbols,
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
    symbols: &Symbols,
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
    symbols: &Symbols,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
    state: &mut FfState,
    async_edge: Option<&SvResetEdge>,
    enable: Option<Expr>,
) -> Result<HashSet<String>, ErrorReport> {
    match stmt {
        SvStmt::ConcatAssign {
            parts,
            nonblocking,
            expr,
        } => {
            let mut assigned = HashSet::new();
            for s in &desugar_concat_assign(parts, *nonblocking, expr, symbols, consts)? {
                assigned.extend(lower_ff_stmt(
                    s,
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
            let (name, addr) = memory_lvalue_parts(dst, symbols)?;
            let sym = symbols
                .get(name)
                .filter(|symbol| symbol.is_memory)
                .ok_or_else(|| err("E_SV_MEMORY", format!("unknown memory `{name}`")))?;
            let (msig, aw) = (sym.signal, sym.mem_addr_width);
            builder.mem_write(
                msig,
                clock,
                enable,
                lower_mem_index(&addr, aw, symbols, consts)?,
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
        SvStmt::Nop => Ok(HashSet::new()),
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
    symbols: &Symbols,
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
    symbols: &Symbols,
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
    symbols: &Symbols,
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
    symbols: &Symbols,
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
                    let guard = combine_enable(enable.clone(), lower_cond(cond, symbols, consts)?);
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
                    let guard = combine_enable(enable.clone(), lower_cond(cond, symbols, consts)?);
                    let gated = guard & lower_expr(cover_cond, symbols, consts)?;
                    builder.cover_clocked(name, clock, gated);
                    return Ok(HashSet::new());
                }
                _ => {}
            }
        }
    }

    let cond = lower_cond(cond, symbols, consts)?;
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
    symbols: &Symbols,
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
    // A plain `case` whose constant labels cover every value of the selector
    // (Verilog `full_case`) can never fall through, so seeding the mux chain
    // with the prior value is unreachable. For a comb signal first written here
    // that prior value is the signal itself — a false self-loop. Seed with the
    // last branch instead; when exhaustive the seed is never selected anyway.
    let exhaustive = default_state.is_none()
        && !branches.is_empty()
        && case_is_exhaustive(kind, items, selector_width, consts);
    for name in assigned.iter().cloned() {
        let (mut expr, fold) = if let Some(default_state) = &default_state {
            (default_state.expr_for(&name, symbols)?, &branches[..])
        } else if exhaustive {
            let (_, last_state) = branches.last().unwrap();
            (
                last_state.expr_for(&name, symbols)?,
                &branches[..branches.len() - 1],
            )
        } else {
            (base.expr_for(&name, symbols)?, &branches[..])
        };
        for (match_expr, branch_state) in fold.iter().rev() {
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

fn mask_all_ones(width: Width) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

/// Lower a condition to 1 bit: Verilog `if (x)` / `x ? :` means `x != 0`, so a
/// multi-bit condition is reduced with `|x`.
fn lower_cond(
    cond: &SvExpr,
    symbols: &Symbols,
    consts: &HashMap<String, u128>,
) -> Result<Expr, ErrorReport> {
    let w = sv_expr_width(cond, symbols, consts)?.unwrap_or(1);
    let e = lower_expr(cond, symbols, consts)?;
    Ok(if w <= 1 {
        e
    } else {
        lower_reduction(&SvUnaryOp::RedOr, e, w)
    })
}

/// Lower a reduction operator to existing IR: `&a == (a == all-ones)`,
/// `|a == (a != 0)`, `^a == xor of all bits`. Result is 1 bit wide.
fn lower_reduction(op: &SvUnaryOp, inner: Expr, width: Width) -> Expr {
    let w = width.max(1);
    match op {
        SvUnaryOp::RedAnd => inner.eq_expr(lit_u(mask_all_ones(w), w)),
        SvUnaryOp::RedOr => inner.ne_expr(lit_u(0, w)),
        SvUnaryOp::RedXor => {
            let mut acc = Expr::Slice {
                expr: Box::new(inner.clone()),
                lsb: 0,
                width: 1,
            };
            for i in 1..w {
                acc = acc
                    ^ Expr::Slice {
                        expr: Box::new(inner.clone()),
                        lsb: i,
                        width: 1,
                    };
            }
            acc
        }
        _ => unreachable!("non-reduction op in lower_reduction"),
    }
}

/// Lower a shift by a *constant* amount to slice/concat over existing IR ops
/// (the IR has no native shift). `left` selects `<<` vs `>>`; `arith` selects
/// sign-extending (`>>>`) vs zero-filling.
fn shift_const(a: Expr, width: Width, amount: u128, left: bool, arith: bool) -> Expr {
    let w = width.max(1);
    if left {
        if amount == 0 {
            return a;
        }
        if amount >= w as u128 {
            return lit_u(0, w);
        }
        let k = amount as Width;
        // {a[w-k-1:0], k'b0} (MSB-first) == (a << k) truncated to w bits.
        concat(vec![
            Expr::Slice {
                expr: Box::new(a),
                lsb: 0,
                width: w - k,
            },
            lit_u(0, k),
        ])
    } else if arith {
        // Arithmetic right shift: keep at least the sign bit, sign-extend back.
        let k = amount.min(w as u128 - 1) as Width;
        if k == 0 {
            return a;
        }
        Expr::Sext {
            expr: Box::new(Expr::Slice {
                expr: Box::new(a),
                lsb: k,
                width: w - k,
            }),
            width: w,
        }
    } else {
        if amount == 0 {
            return a;
        }
        if amount >= w as u128 {
            return lit_u(0, w);
        }
        let k = amount as Width;
        Expr::Zext {
            expr: Box::new(Expr::Slice {
                expr: Box::new(a),
                lsb: k,
                width: w - k,
            }),
            width: w,
        }
    }
}

/// Lower a shift with a possibly-variable amount. A constant amount lowers
/// directly; a variable amount becomes a barrel shifter (a mux per amount bit,
/// each stage a constant shift by a power of two).
fn lower_shift(
    a: Expr,
    width: Width,
    amount: Expr,
    amount_const: Option<u128>,
    amount_width: Width,
    left: bool,
    arith: bool,
) -> Expr {
    if let Some(k) = amount_const {
        return shift_const(a, width, k, left, arith);
    }
    let mut acc = a;
    for i in 0..amount_width.min(127).max(1) {
        let amt = if i >= 64 { u128::MAX } else { 1u128 << i };
        let shifted = shift_const(acc.clone(), width, amt, left, arith);
        let bit = Expr::Slice {
            expr: Box::new(amount.clone()),
            lsb: i,
            width: 1,
        };
        acc = mux(bit, shifted, acc);
    }
    acc
}

/// Lower the two operands of a width-matching binary op (arithmetic, bitwise,
/// comparison), coercing each to their common self-determined width. This makes
/// e.g. `a[7:0] + 1` (an unsized 32-bit literal) build a width-consistent tree;
/// for already-matched widths the coercions are no-ops.
fn lower_binary_operands(
    lhs: &SvExpr,
    rhs: &SvExpr,
    symbols: &Symbols,
    consts: &HashMap<String, u128>,
) -> Result<(Expr, Expr), ErrorReport> {
    let lt = sv_expr_type(lhs, symbols, consts)?;
    let rt = sv_expr_type(rhs, symbols, consts)?;
    let cw = lt.width.max(rt.width).max(1);
    let signedness = if lt.is_signed() && rt.is_signed() {
        Signedness::Signed
    } else {
        Signedness::Unsigned
    };
    let ct = BitType::new(cw, signedness);
    Ok((
        coerce_expr_to_type(lower_expr(lhs, symbols, consts)?, lt, ct),
        coerce_expr_to_type(lower_expr(rhs, symbols, consts)?, rt, ct),
    ))
}

fn lower_expr(
    expr: &SvExpr,
    symbols: &Symbols,
    consts: &HashMap<String, u128>,
) -> Result<Expr, ErrorReport> {
    Ok(match expr {
        SvExpr::Ident(name) => {
            if let Some(symbol) = symbols.get(name) {
                symbol.signal.value()
            } else if let Some(value) = consts.get(name) {
                // A bare constant identifier is self-determined: size it to hold
                // its value (at least 32 bits), so downstream coercion truncates
                // to the using context rather than over/underflowing a fixed 32.
                let w = (128 - value.leading_zeros() as Width).max(32);
                lit_u(*value, w)
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
                // Verilog truncates an unsigned literal to its declared width.
                lit_u(*value & value_mask(*width), *width)
            }
        }
        SvExpr::Unary { op, expr } => {
            let inner = lower_expr(expr, symbols, consts)?;
            match op {
                // `!x` == (x == 0), compared at x's width → 1-bit result.
                SvUnaryOp::Not => {
                    let w = sv_expr_width(expr, symbols, consts)?.unwrap_or(1);
                    inner.eq_expr(lit_u(0, w))
                }
                SvUnaryOp::BitNot => !inner,
                SvUnaryOp::Neg => {
                    let w = sv_expr_width(expr, symbols, consts)?.unwrap_or(32).max(1);
                    lit_u(0, w) - inner
                }
                SvUnaryOp::RedAnd | SvUnaryOp::RedOr | SvUnaryOp::RedXor => {
                    let w = sv_expr_width(expr, symbols, consts)?.unwrap_or(1);
                    lower_reduction(op, inner, w)
                }
            }
        }
        SvExpr::Binary { op, lhs, rhs } => {
            // Shifts need the left operand's width and signedness (and a constant
            // amount if available), captured before the operands are lowered.
            let shift_meta = if matches!(op, SvBinaryOp::Shl | SvBinaryOp::Shr | SvBinaryOp::Ashr) {
                Some((
                    sv_expr_width(lhs, symbols, consts)?.unwrap_or(32).max(1),
                    sv_expr_is_signed(lhs, symbols),
                    const_eval(rhs, consts).ok(),
                    sv_expr_width(rhs, symbols, consts)?.unwrap_or(32),
                ))
            } else {
                None
            };
            match op {
                // Shifts: left operand keeps its width; the amount is self-
                // determined and not coerced.
                SvBinaryOp::Shl | SvBinaryOp::Shr | SvBinaryOp::Ashr => {
                    let lhs = lower_expr(lhs, symbols, consts)?;
                    let rhs = lower_expr(rhs, symbols, consts)?;
                    let (w, signed, k, rhs_w) = shift_meta.unwrap();
                    let (left, arith) = match op {
                        SvBinaryOp::Shl => (true, false),
                        SvBinaryOp::Shr => (false, false),
                        _ => (false, signed),
                    };
                    lower_shift(lhs, w, rhs, k, rhs_w, left, arith)
                }
                // Logical && / ||: reduce each operand to 1 bit (`!= 0`), then
                // combine — result is 1 bit (not bitwise at the operand width).
                SvBinaryOp::LogAnd => {
                    lower_cond(lhs, symbols, consts)? & lower_cond(rhs, symbols, consts)?
                }
                SvBinaryOp::LogOr => {
                    lower_cond(lhs, symbols, consts)? | lower_cond(rhs, symbols, consts)?
                }
                // The IR has no divide; these are only valid in constant contexts
                // (which are folded before lowering).
                SvBinaryOp::Div | SvBinaryOp::Mod | SvBinaryOp::Pow => {
                    return Err(err(
                        "E_SV_EXPR",
                        "division, modulo, and exponentiation are only supported in constant expressions",
                    ));
                }
                // Width-matching ops: coerce operands to their common width.
                _ => {
                    let (lhs, rhs) = lower_binary_operands(lhs, rhs, symbols, consts)?;
                    match op {
                        SvBinaryOp::Add => lhs + rhs,
                        SvBinaryOp::Sub => lhs - rhs,
                        SvBinaryOp::Mul => lhs * rhs,
                        SvBinaryOp::And => lhs & rhs,
                        SvBinaryOp::Or => lhs | rhs,
                        SvBinaryOp::Xor => lhs ^ rhs,
                        SvBinaryOp::Eq => lhs.eq_expr(rhs),
                        SvBinaryOp::Ne => lhs.ne_expr(rhs),
                        SvBinaryOp::Lt => lhs.lt_expr(rhs),
                        SvBinaryOp::Le => lhs.clone().lt_expr(rhs.clone()) | lhs.eq_expr(rhs),
                        SvBinaryOp::Gt => rhs.lt_expr(lhs),
                        SvBinaryOp::Ge => rhs.clone().lt_expr(lhs.clone()) | lhs.eq_expr(rhs),
                        _ => unreachable!("non-width-matching op in coercion branch"),
                    }
                }
            }
        }
        SvExpr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            // Both arms widen to their common (self-determined) width.
            let tt = sv_expr_type(then_expr, symbols, consts)?;
            let et = sv_expr_type(else_expr, symbols, consts)?;
            let cw = tt.width.max(et.width).max(1);
            let signedness = if tt.is_signed() && et.is_signed() {
                Signedness::Signed
            } else {
                Signedness::Unsigned
            };
            let ct = BitType::new(cw, signedness);
            mux(
                lower_cond(cond, symbols, consts)?,
                coerce_expr_to_type(lower_expr(then_expr, symbols, consts)?, tt, ct),
                coerce_expr_to_type(lower_expr(else_expr, symbols, consts)?, et, ct),
            )
        }
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
            let inner = lower_expr(expr, symbols, consts)?;
            // Part-selecting beyond the value's width reads zeros (Verilog), so
            // zero-extend a too-narrow operand up to `msb+1` first.
            let iw = sv_expr_width(expr, symbols, consts)?.unwrap_or(msb + 1);
            let inner = if iw <= *msb {
                coerce_expr_to_type(
                    inner,
                    BitType::new(iw.max(1), Signedness::Unsigned),
                    BitType::new(msb + 1, Signedness::Unsigned),
                )
            } else {
                inner
            };
            inner.slice(*lsb, msb - lsb + 1)
        }
        SvExpr::MemRead { name, addr } => {
            let sym = symbols
                .get(name)
                .filter(|symbol| symbol.is_memory)
                .ok_or_else(|| err("E_SV_MEMORY", format!("unknown memory `{name}`")))?;
            let (msig, aw) = (sym.signal, sym.mem_addr_width);
            mem_read(msig, lower_mem_index(addr, aw, symbols, consts)?)
        }
        SvExpr::Call { name, .. } => {
            return Err(err(
                "E_SV_CALL",
                format!("unresolved function call to `{name}` (only pure functions in datapath expressions are supported)"),
            ));
        }
        SvExpr::Bracket { expr, index } => {
            // 2-D unpacked memory read `mem[i][j]` → flat read at `i*D2 + j`.
            if let SvExpr::Bracket { expr: inner, index: i } = expr.as_ref() {
                if let SvExpr::Ident(name) = inner.as_ref() {
                    if let Some(sym) = symbols.get(name).filter(|s| s.is_memory && s.mem_inner > 0) {
                        let (msig, aw, d2) = (sym.signal, sym.mem_addr_width, sym.mem_inner);
                        let flat = flat_2d_addr(i, index, d2);
                        return Ok(mem_read(msig, lower_mem_index(&flat, aw, symbols, consts)?));
                    }
                }
            }
            match expr.as_ref() {
            SvExpr::Ident(name) => {
                let Some(symbol) = symbols.get(name) else {
                    return Err(err(
                        "E_SV_EXPR_IDENT",
                        format!("unknown identifier `{name}`"),
                    ));
                };
                if symbol.is_memory {
                    let (msig, aw) = (symbol.signal, symbol.mem_addr_width);
                    mem_read(msig, lower_mem_index(index, aw, symbols, consts)?)
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
            }
        }
    })
}

/// Row-major flatten of a 2-D unpacked access `[i][j]` into a 1-D address `i*D2+j`.
fn flat_2d_addr(i: &SvExpr, j: &SvExpr, d2: usize) -> SvExpr {
    SvExpr::Binary {
        op: SvBinaryOp::Add,
        lhs: Box::new(SvExpr::Binary {
            op: SvBinaryOp::Mul,
            lhs: Box::new(i.clone()),
            rhs: Box::new(SvExpr::Lit { value: d2 as u128, width: 32, signed: false }),
        }),
        rhs: Box::new(j.clone()),
    }
}

fn lower_concat_part(
    expr: &SvExpr,
    symbols: &Symbols,
    consts: &HashMap<String, u128>,
) -> Result<Expr, ErrorReport> {
    let lowered = lower_expr(expr, symbols, consts)?;
    if let Some(width) = sv_expr_width(expr, symbols, consts)? {
        Ok(lowered.slice(0, width))
    } else {
        Ok(lowered)
    }
}

/// Lower a memory index expression and resize it to the memory's address width:
/// Verilog truncates a wider index and zero-extends a narrower one (rather than
/// erroring on a width mismatch, which is what the core checker does).
fn lower_mem_index(
    addr: &SvExpr,
    addr_width: Width,
    symbols: &Symbols,
    consts: &HashMap<String, u128>,
) -> Result<Expr, ErrorReport> {
    let e = lower_expr(addr, symbols, consts)?;
    let cur = sv_expr_width(addr, symbols, consts)?.unwrap_or(32);
    Ok(if cur == addr_width || addr_width == 0 {
        e
    } else if cur > addr_width {
        e.slice(0, addr_width)
    } else {
        rrtl_core::zext(e, addr_width)
    })
}

fn sv_expr_width(
    expr: &SvExpr,
    symbols: &Symbols,
    consts: &HashMap<String, u128>,
) -> Result<Option<Width>, ErrorReport> {
    Ok(match expr {
        SvExpr::Ident(name) => symbols
            .get(name)
            .map(|symbol| symbol.ty.width)
            .or_else(|| consts.get(name).map(|v| (128 - v.leading_zeros() as Width).max(32)))
            .or(Some(32)),
        SvExpr::Lit { width, .. } => Some(*width),
        SvExpr::Unary {
            op:
                SvUnaryOp::RedAnd
                | SvUnaryOp::RedOr
                | SvUnaryOp::RedXor
                | SvUnaryOp::Not,
            ..
        } => Some(1),
        SvExpr::Unary { expr, .. } | SvExpr::Cast { expr, .. } => {
            sv_expr_width(expr, symbols, consts)?
        }
        SvExpr::Binary { op, lhs, rhs } => match op {
            // Arithmetic/bitwise: self-determined width is the max of the operands.
            SvBinaryOp::Add
            | SvBinaryOp::Sub
            | SvBinaryOp::Mul
            | SvBinaryOp::And
            | SvBinaryOp::Or
            | SvBinaryOp::Xor => {
                let l = sv_expr_width(lhs, symbols, consts)?;
                let r = sv_expr_width(rhs, symbols, consts)?;
                match (l, r) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    _ => l.or(r),
                }
            }
            // Comparisons produce a 1-bit result.
            SvBinaryOp::Eq
            | SvBinaryOp::Ne
            | SvBinaryOp::Lt
            | SvBinaryOp::Le
            | SvBinaryOp::Gt
            | SvBinaryOp::Ge => Some(1),
            // Logical &&/|| produce a 1-bit result (lowered as a reduction).
            SvBinaryOp::LogAnd | SvBinaryOp::LogOr => Some(1),
            // Shifts keep the left operand's width.
            _ => sv_expr_width(lhs, symbols, consts)?,
        },
        SvExpr::Ternary {
            then_expr,
            else_expr,
            ..
        } => {
            // Both arms widen to their common width (matches the lowering).
            let t = sv_expr_width(then_expr, symbols, consts)?;
            let e = sv_expr_width(else_expr, symbols, consts)?;
            match (t, e) {
                (Some(a), Some(b)) => Some(a.max(b)),
                _ => t.or(e),
            }
        }
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
        SvExpr::Index { .. } | SvExpr::Bracket { .. } | SvExpr::Call { .. } => Some(1),
        SvExpr::Slice { msb, lsb, .. } => Some(msb - lsb + 1),
        SvExpr::MemRead { name, .. } => symbols.get(name).map(|symbol| symbol.ty.width),
    })
}

fn sv_expr_type(
    expr: &SvExpr,
    symbols: &Symbols,
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

fn sv_expr_is_signed(expr: &SvExpr, symbols: &Symbols) -> bool {
    match expr {
        SvExpr::Ident(name) => symbols
            .get(name)
            .is_some_and(|symbol| symbol.ty.is_signed()),
        SvExpr::Lit { signed, .. } => *signed,
        SvExpr::Unary {
            op: SvUnaryOp::RedAnd | SvUnaryOp::RedOr | SvUnaryOp::RedXor,
            ..
        } => false,
        SvExpr::Unary { expr, .. } => sv_expr_is_signed(expr, symbols),
        SvExpr::Cast { signed, .. } => *signed,
        SvExpr::Binary { lhs, .. } => sv_expr_is_signed(lhs, symbols),
        SvExpr::Ternary { then_expr, .. } => sv_expr_is_signed(then_expr, symbols),
        SvExpr::Concat(_) | SvExpr::Repeat { .. } => false,
        SvExpr::Index { .. } | SvExpr::Slice { .. } | SvExpr::Bracket { .. } | SvExpr::Call { .. } => false,
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
    symbols: &Symbols,
    consts: &HashMap<String, u128>,
) -> Result<Expr, ErrorReport> {
    if matches!(dst, SvLvalue::Signal(_)) {
        return lower_expr_for_signal(expr, signal, symbols, consts);
    }
    let value = lower_expr(expr, symbols, consts)?;
    let value_width = sv_expr_width(expr, symbols, consts)?;
    lower_lvalue_assignment_value(dst, value, value_width, base, signal, symbols, consts)
}

fn lower_expr_for_signal(
    expr: &SvExpr,
    dst: Signal,
    symbols: &Symbols,
    consts: &HashMap<String, u128>,
) -> Result<Expr, ErrorReport> {
    let value = lower_expr(expr, symbols, consts)?;
    let expr_ty = sv_expr_type(expr, symbols, consts)?;
    let dst_ty = symbols
        .by_signal
        .get(&dst)
        .copied()
        .ok_or_else(|| err("E_SV_LVALUE", "unknown assignment target"))?;
    Ok(coerce_expr_to_type(value, expr_ty, dst_ty))
}

fn lower_lvalue_assignment_value(
    dst: &SvLvalue,
    value: Expr,
    value_width: Option<Width>,
    base: Expr,
    signal: Signal,
    symbols: &Symbols,
    consts: &HashMap<String, u128>,
) -> Result<Expr, ErrorReport> {
    match dst {
        SvLvalue::Signal(_) => coerce_expr_to_signal_type(value, signal, symbols),
        SvLvalue::Bit { index, .. } => {
            let index = const_index(index, consts, "E_SV_LVALUE")?;
            splice_lvalue_expr(base, signal, index, 1, value, value_width, symbols)
        }
        SvLvalue::Slice { msb, lsb, .. } => {
            let msb = const_index(msb, consts, "E_SV_LVALUE")?;
            let lsb = const_index(lsb, consts, "E_SV_LVALUE")?;
            if msb < lsb {
                return Err(err("E_SV_LVALUE", "part-select msb must be >= lsb"));
            }
            splice_lvalue_expr(base, signal, lsb, msb - lsb + 1, value, value_width, symbols)
        }
        SvLvalue::Memory { .. } | SvLvalue::Memory2D { .. } => Err(err(
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
    value_width: Option<Width>,
    symbols: &Symbols,
) -> Result<Expr, ErrorReport> {
    let signal_width = symbols
        .by_signal
        .get(&signal)
        .map(|ty| ty.width)
        .ok_or_else(|| err("E_SV_LVALUE", "unknown lvalue signal"))?;
    if width == 0 || lsb + width > signal_width {
        return Err(err("E_SV_LVALUE", "selected lvalue exceeds signal width"));
    }
    // Resize the RHS to exactly the selected width. When the RHS's actual width
    // is known, zero-extend/truncate it (Verilog truncates wide RHS, zero-fills
    // narrow); otherwise fall back to a low-bit slice (assumes value >= width).
    let value = match value_width {
        Some(vw) => coerce_expr_to_type(value, uint(vw.max(1)), uint(width)),
        None => value.slice(0, width),
    };
    let mut parts = Vec::new();
    let high_lsb = lsb + width;
    if high_lsb < signal_width {
        parts.push(base.clone().slice(high_lsb, signal_width - high_lsb));
    }
    parts.push(value);
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
    symbols: &Symbols,
) -> Result<Expr, ErrorReport> {
    let Some(ty) = symbols.by_signal.get(&dst) else {
        return Ok(expr);
    };
    if ty.is_signed() {
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

/// Load a memory's power-on contents from a `$readmemh`/`$readmemb` file at lowering
/// time: whitespace-separated hex/binary words (one per address), with `//` line
/// comments and `@hexaddr` relocation directives honored.
fn lower_readmem(
    hex: bool,
    file: &str,
    mem: &str,
    symbols: &Symbols,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
) -> Result<(), ErrorReport> {
    let symbol = symbols
        .get(mem)
        .filter(|s| s.is_memory)
        .ok_or_else(|| err("E_SV_READMEM", format!("$readmem target `{mem}` is not a memory")))?;
    let contents = std::fs::read_to_string(file)
        .map_err(|e| err("E_SV_READMEM", format!("cannot read `{file}`: {e}")))?;
    let radix = if hex { 16 } else { 2 };
    let mut addr = 0usize;
    for raw in contents.lines() {
        let line = raw.split("//").next().unwrap_or("");
        for tok in line.split_whitespace() {
            if let Some(a) = tok.strip_prefix('@') {
                addr = usize::from_str_radix(a, 16)
                    .map_err(|_| err("E_SV_READMEM", format!("bad @address `{tok}` in `{file}`")))?;
                continue;
            }
            let clean: String = tok.chars().filter(|c| *c != '_').collect();
            let word = u128::from_str_radix(&clean, radix)
                .map_err(|_| err("E_SV_READMEM", format!("bad data word `{tok}` in `{file}`")))?;
            builder.mem_init(symbol.signal, addr, word as i128);
            addr += 1;
        }
    }
    Ok(())
}

fn lower_initial_assign(
    assign: &SvInitialAssign,
    symbols: &Symbols,
    consts: &HashMap<String, u128>,
    builder: &mut rrtl_core::ModuleBuilder<'_>,
    initial_state: &mut InitialState,
) -> Result<(), ErrorReport> {
    let (dst, expr) = match assign {
        SvInitialAssign::Assign { dst, expr } => (dst, expr),
        SvInitialAssign::ReadMem { hex, file, mem } => {
            return lower_readmem(*hex, file, mem, symbols, builder);
        }
    };
    let value = const_value(expr, consts)?;
    match dst {
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
        SvLvalue::Memory2D { name, outer, inner } => {
            let symbol = symbols
                .get(name)
                .filter(|symbol| symbol.is_memory)
                .ok_or_else(|| err("E_SV_INITIAL", format!("unknown memory `{name}`")))?;
            let d2 = symbol.mem_inner as u128;
            let flat = const_eval(outer, consts)? * d2 + const_eval(inner, consts)?;
            let addr = usize::try_from(flat)
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
    symbols: &Symbols,
) -> Result<String, ErrorReport> {
    match dst {
        SvLvalue::Signal(name) | SvLvalue::Bit { name, .. } | SvLvalue::Slice { name, .. } => {
            symbols
                .get(name)
                .filter(|symbol| !symbol.is_memory)
                .map(|_| name.clone())
                .ok_or_else(|| err("E_SV_LVALUE", format!("unknown signal `{name}`")))
        }
        SvLvalue::Memory { .. } | SvLvalue::Memory2D { .. } => Err(err(
            "E_SV_LVALUE",
            "memory lvalue is only valid inside enabled always_ff",
        )),
    }
}

fn lvalue_is_memory(dst: &SvLvalue, symbols: &Symbols) -> bool {
    match dst {
        SvLvalue::Memory { name, .. }
        | SvLvalue::Memory2D { name, .. }
        | SvLvalue::Bit { name, .. } => {
            symbols.get(name).is_some_and(|symbol| symbol.is_memory)
        }
        SvLvalue::Signal(_) | SvLvalue::Slice { .. } => false,
    }
}

fn memory_lvalue_parts<'a>(
    dst: &'a SvLvalue,
    symbols: &Symbols,
) -> Result<(&'a str, SvExpr), ErrorReport> {
    match dst {
        SvLvalue::Memory { name, addr } => Ok((name, addr.clone())),
        SvLvalue::Bit { name, index } => Ok((name, index.clone())),
        SvLvalue::Memory2D { name, outer, inner } => {
            let d2 = symbols.get(name).map(|s| s.mem_inner).unwrap_or(0);
            Ok((name, flat_2d_addr(outer, inner, d2)))
        }
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
    symbols: &Symbols,
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
        SvLvalue::Memory { .. } | SvLvalue::Memory2D { .. } => Err(err(
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

pub(crate) fn const_eval(expr: &SvExpr, consts: &HashMap<String, u128>) -> Result<u128, ErrorReport> {
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
                SvUnaryOp::RedOr => u128::from(value != 0),
                SvUnaryOp::RedXor => (value.count_ones() & 1) as u128,
                SvUnaryOp::RedAnd => {
                    return Err(err(
                        "E_SV_CONST",
                        "reduction-and in a constant expression requires a known width",
                    ))
                }
            })
        }
        SvExpr::Binary { op, lhs, rhs } => {
            let lhs = const_eval(lhs, consts)?;
            let rhs = const_eval(rhs, consts)?;
            Ok(match op {
                SvBinaryOp::Add => lhs.wrapping_add(rhs),
                SvBinaryOp::Sub => lhs.wrapping_sub(rhs),
                SvBinaryOp::Mul => lhs.wrapping_mul(rhs),
                SvBinaryOp::Div => {
                    if rhs == 0 {
                        return Err(err("E_SV_CONST", "division by zero in constant expression"));
                    }
                    lhs / rhs
                }
                SvBinaryOp::Mod => {
                    if rhs == 0 {
                        return Err(err("E_SV_CONST", "modulo by zero in constant expression"));
                    }
                    lhs % rhs
                }
                SvBinaryOp::Pow => lhs.wrapping_pow(rhs as u32),
                SvBinaryOp::And | SvBinaryOp::LogAnd => lhs & rhs,
                SvBinaryOp::Or | SvBinaryOp::LogOr => lhs | rhs,
                SvBinaryOp::Xor => lhs ^ rhs,
                SvBinaryOp::Eq => u128::from(lhs == rhs),
                SvBinaryOp::Ne => u128::from(lhs != rhs),
                SvBinaryOp::Lt => u128::from(lhs < rhs),
                SvBinaryOp::Le => u128::from(lhs <= rhs),
                SvBinaryOp::Gt => u128::from(lhs > rhs),
                SvBinaryOp::Ge => u128::from(lhs >= rhs),
                SvBinaryOp::Shl => lhs.wrapping_shl(rhs as u32),
                SvBinaryOp::Shr | SvBinaryOp::Ashr => lhs.wrapping_shr(rhs as u32),
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
        SvExpr::Index { .. } | SvExpr::Bracket { .. } | SvExpr::Call { .. } => Ok(1),
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

/// Collect the module names instantiated anywhere in `items`, recursing through
/// generate blocks (which may wrap instances).
fn collect_instance_modules(items: &[SvItem], out: &mut Vec<String>) {
    for item in items {
        match item {
            SvItem::Instance(inst) => out.push(inst.module.clone()),
            SvItem::Generate(inner) => collect_instance_modules(inner, out),
            SvItem::GenerateFor { items, .. } => collect_instance_modules(items, out),
            _ => {}
        }
    }
}

/// The set of module names reachable from `top` by following instance
/// references transitively (including `top` itself).
fn reachable_modules(source: &SvSource, top: &str) -> HashSet<String> {
    let by_name: HashMap<&str, &SvModule> =
        source.modules.iter().map(|m| (m.name.as_str(), m)).collect();
    let mut reachable = HashSet::new();
    let mut stack = vec![top.to_string()];
    while let Some(name) = stack.pop() {
        if !reachable.insert(name.clone()) {
            continue;
        }
        if let Some(module) = by_name.get(name.as_str()) {
            let mut instantiated = Vec::new();
            collect_instance_modules(&module.items, &mut instantiated);
            for child in instantiated {
                if !reachable.contains(&child) {
                    stack.push(child);
                }
            }
        }
    }
    reachable
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
