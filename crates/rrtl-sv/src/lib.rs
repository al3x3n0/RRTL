use rrtl_core::{
    compile, compile_ir, CompiledAssertion, CompiledAssignment, CompiledCoverPoint, CompiledDesign,
    CompiledExpr, CompiledMemoryWrite, CompiledModule, Design,
};
use rrtl_ir::{ErrorReport, Expr, Signal, SignalInfo, SignalKind, Width};

pub fn emit(design: &Design) -> Result<String, ErrorReport> {
    let compiled = compile(design)?;
    Ok(emit_compiled(&compiled))
}

pub fn emit_ir(design: &rrtl_ir::Design) -> Result<String, ErrorReport> {
    let compiled = compile_ir(design)?;
    Ok(emit_compiled(&compiled))
}

pub fn emit_compiled(design: &CompiledDesign) -> String {
    let mut out = String::new();
    for module in design.modules() {
        if module.is_external {
            continue;
        }
        emit_module(module, &mut out);
        out.push('\n');
    }
    out
}

fn emit_module(module: &CompiledModule, out: &mut String) {
    let ports = module
        .signals
        .iter()
        .filter(|s| {
            matches!(
                s.kind,
                SignalKind::Input | SignalKind::Output | SignalKind::Inout
            )
        })
        .collect::<Vec<_>>();

    out.push_str(&format!("module {}(", module.name));
    for (index, port) in ports.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&port.name);
    }
    out.push_str(");\n");

    for state_type in &module.state_types {
        emit_state_typedef(state_type, out);
    }

    for port in &ports {
        match port.kind {
            SignalKind::Input => out.push_str(&format!(
                "  input logic {}{};\n",
                sv_type(port.ty),
                port.name
            )),
            SignalKind::Output => out.push_str(&format!(
                "  output logic {}{};\n",
                sv_type(port.ty),
                port.name
            )),
            SignalKind::Inout => out.push_str(&format!(
                "  inout wire {}{};\n",
                sv_type(port.ty),
                port.name
            )),
            _ => unreachable!(),
        }
    }

    for signal in &module.signals {
        match signal.kind {
            SignalKind::Wire => {
                out.push_str(&format!("  logic {}{};\n", sv_type(signal.ty), signal.name))
            }
            SignalKind::Reg { .. } => {
                if let Some(enum_type) = state_signal_enum_type(module, signal.handle) {
                    out.push_str(&format!("  {} {};\n", enum_type, signal.name));
                } else {
                    out.push_str(&format!("  logic {}{};\n", sv_type(signal.ty), signal.name));
                }
            }
            SignalKind::Mem {
                data_width: _,
                depth,
                ..
            } => out.push_str(&format!(
                "  logic {}{} [0:{}];\n",
                sv_type(signal.ty),
                signal.name,
                depth - 1
            )),
            SignalKind::Input | SignalKind::Output | SignalKind::Inout => {}
        }
    }

    for assignment in &module.assignments {
        emit_assignment(module, assignment, out);
    }

    emit_initial_values(module, out);

    for register in &module.registers {
        let clock = signal_name(module, register.clock);
        let register_name = signal_name(module, register.signal);
        let register_ty = module
            .signal(register.signal)
            .map(|s| s.ty)
            .unwrap_or(rrtl_core::uint(1));
        if let Some(reset) = &register.reset {
            let reset_signal = signal_name(module, reset.signal);
            let reset_edge = match reset.polarity {
                rrtl_core::ResetPolarity::ActiveHigh => "posedge",
                rrtl_core::ResetPolarity::ActiveLow => "negedge",
            };
            let reset_condition = match reset.polarity {
                rrtl_core::ResetPolarity::ActiveHigh => reset_signal.to_string(),
                rrtl_core::ResetPolarity::ActiveLow => format!("!{reset_signal}"),
            };
            match reset.kind {
                rrtl_core::ResetKind::Sync => {
                    out.push_str(&format!("  always_ff @(posedge {clock}) begin\n"));
                }
                rrtl_core::ResetKind::Async => {
                    out.push_str(&format!(
                        "  always_ff @(posedge {clock} or {reset_edge} {reset_signal}) begin\n"
                    ));
                }
            }
            out.push_str(&format!(
                "    if ({reset_condition}) {} <= {};\n",
                register_name,
                literal(reset.value, register_ty)
            ));
            out.push_str(&format!(
                "    else {} <= {};\n",
                register_name,
                emit_expr(module, &register.next)
            ));
        } else {
            out.push_str(&format!("  always_ff @(posedge {clock}) begin\n"));
            out.push_str(&format!(
                "    {} <= {};\n",
                register_name,
                emit_expr(module, &register.next)
            ));
        }
        out.push_str("  end\n");
    }

    for write in &module.memory_writes {
        emit_memory_write(module, write, out);
    }

    for assertion in &module.assertions {
        emit_assertion(module, assertion, out);
    }

    for cover in &module.cover_points {
        emit_cover_point(module, cover, out);
    }

    for instance in &module.instances {
        out.push_str(&format!("  {} {} (", instance.module, instance.name));
        for (index, connection) in instance.connections.iter().enumerate() {
            if index > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!(
                ".{}({})",
                connection.port,
                signal_name(module, connection.signal)
            ));
        }
        out.push_str(");\n");
    }

    out.push_str("endmodule\n");
}

fn emit_initial_values(module: &CompiledModule, out: &mut String) {
    if module.initial_register_values.is_empty() && module.initial_memory_values.is_empty() {
        return;
    }

    out.push_str("  initial begin\n");
    for initial in &module.initial_register_values {
        if let Some(signal) = module.signal(initial.signal) {
            out.push_str(&format!(
                "    {} = {};\n",
                signal.name,
                literal(initial.value, signal.ty)
            ));
        }
    }
    for initial in &module.initial_memory_values {
        if let Some(mem) = module.signal(initial.mem) {
            out.push_str(&format!(
                "    {}[{}] = {};\n",
                mem.name,
                initial.addr,
                literal(initial.value, mem.ty)
            ));
        }
    }
    out.push_str("  end\n");
}

fn emit_state_typedef(state_type: &rrtl_core::StateType, out: &mut String) {
    out.push_str(&format!(
        "  typedef enum logic {}{{ ",
        sv_type(state_type.ty)
    ));
    for (index, variant) in state_type.variants.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!(
            "{}_{} = {}",
            sv_ident(&state_type.name),
            sv_ident(&variant.name),
            literal(variant.value, state_type.ty)
        ));
    }
    out.push_str(&format!(
        " }} {};\n",
        state_enum_type_name(&state_type.name)
    ));
}

fn emit_memory_write(module: &CompiledModule, write: &CompiledMemoryWrite, out: &mut String) {
    let clock = signal_name(module, write.clock);
    let mem = signal_name(module, write.mem);
    out.push_str(&format!("  always_ff @(posedge {clock}) begin\n"));
    out.push_str(&format!(
        "    if ({}) {}[{}] <= {};\n",
        emit_expr(module, &write.enable),
        mem,
        emit_expr(module, &write.addr),
        emit_expr(module, &write.data)
    ));
    out.push_str("  end\n");
}

fn emit_assertion(module: &CompiledModule, assertion: &CompiledAssertion, out: &mut String) {
    match assertion.clock {
        Some(clock) => {
            let clock = signal_name(module, clock);
            out.push_str(&format!("  always_ff @(posedge {clock}) begin\n"));
        }
        None => out.push_str("  always_comb begin\n"),
    }

    let condition = emit_expr(module, &assertion.condition);
    let stmt = assertion_statement(&assertion.name, &condition, assertion.message.as_deref());
    if let Some(enable) = &assertion.enable {
        out.push_str(&format!("    if ({}) {stmt}\n", emit_expr(module, enable)));
    } else {
        out.push_str(&format!("    {stmt}\n"));
    }
    out.push_str("  end\n");
}

fn assertion_statement(name: &str, condition: &str, message: Option<&str>) -> String {
    let label = sv_ident(name);
    let message = message.unwrap_or(name);
    format!(
        "{label}: assert ({condition}) else $error(\"{}\");",
        sv_string(message)
    )
}

fn emit_cover_point(module: &CompiledModule, cover: &CompiledCoverPoint, out: &mut String) {
    match cover.clock {
        Some(clock) => {
            let clock = signal_name(module, clock);
            out.push_str(&format!("  always_ff @(posedge {clock}) begin\n"));
        }
        None => out.push_str("  always_comb begin\n"),
    }

    let condition = emit_expr(module, &cover.condition);
    let stmt = cover_statement(&cover.name, &condition);
    if let Some(enable) = &cover.enable {
        out.push_str(&format!("    if ({}) {stmt}\n", emit_expr(module, enable)));
    } else {
        out.push_str(&format!("    {stmt}\n"));
    }
    out.push_str("  end\n");
}

fn cover_statement(name: &str, condition: &str) -> String {
    let label = sv_ident(name);
    format!("{label}: cover ({condition});")
}

fn emit_assignment(module: &CompiledModule, assignment: &CompiledAssignment, out: &mut String) {
    out.push_str(&format!(
        "  assign {} = {};\n",
        signal_name(module, assignment.dst),
        emit_expr(module, &assignment.expr)
    ));
}

fn emit_expr(module: &CompiledModule, expr: &CompiledExpr) -> String {
    emit_raw_expr(module, &expr.expr)
}

fn emit_raw_expr(module: &CompiledModule, expr: &Expr) -> String {
    match expr {
        Expr::Lit { value, ty } => literal(*value, *ty),
        Expr::Signal(signal) => signal_name(module, *signal).to_string(),
        Expr::Not(inner) => format!("~({})", emit_raw_expr(module, inner)),
        Expr::And(lhs, rhs) => bin(module, "&", lhs, rhs),
        Expr::Or(lhs, rhs) => bin(module, "|", lhs, rhs),
        Expr::Xor(lhs, rhs) => bin(module, "^", lhs, rhs),
        Expr::Add(lhs, rhs) => bin(module, "+", lhs, rhs),
        Expr::Sub(lhs, rhs) => bin(module, "-", lhs, rhs),
        Expr::Mul(lhs, rhs) => bin(module, "*", lhs, rhs),
        Expr::Eq(lhs, rhs) => bin(module, "==", lhs, rhs),
        Expr::Ne(lhs, rhs) => bin(module, "!=", lhs, rhs),
        Expr::Lt(lhs, rhs) => bin(module, "<", lhs, rhs),
        Expr::Mux {
            cond,
            then_expr,
            else_expr,
        } => format!(
            "({} ? {} : {})",
            emit_raw_expr(module, cond),
            emit_raw_expr(module, then_expr),
            emit_raw_expr(module, else_expr)
        ),
        Expr::Slice { expr, lsb, width } => {
            let msb = lsb + width - 1;
            if *width == 1 {
                format!("{}[{}]", emit_raw_expr(module, expr), lsb)
            } else {
                format!("{}[{}:{}]", emit_raw_expr(module, expr), msb, lsb)
            }
        }
        Expr::Zext { expr, width } => {
            let Some(inner_width) = expr_width(module, expr) else {
                return emit_raw_expr(module, expr);
            };
            if *width == inner_width {
                emit_raw_expr(module, expr)
            } else {
                format!(
                    "{{{}'d0, {}}}",
                    width - inner_width,
                    emit_raw_expr(module, expr)
                )
            }
        }
        Expr::Sext { expr, width } => {
            let Some(inner_width) = expr_width(module, expr) else {
                return emit_raw_expr(module, expr);
            };
            if *width == inner_width {
                emit_raw_expr(module, expr)
            } else {
                format!(
                    "{{{{{}{{{}[{}]}}}}, {}}}",
                    width - inner_width,
                    emit_raw_expr(module, expr),
                    inner_width - 1,
                    emit_raw_expr(module, expr)
                )
            }
        }
        Expr::Trunc { expr, width } => {
            if *width == 1 {
                format!("{}[0]", emit_raw_expr(module, expr))
            } else {
                format!("{}[{}:0]", emit_raw_expr(module, expr), width - 1)
            }
        }
        Expr::Cast { expr, signedness } => match signedness {
            rrtl_core::Signedness::Unsigned => {
                format!("$unsigned({})", emit_raw_expr(module, expr))
            }
            rrtl_core::Signedness::Signed => format!("$signed({})", emit_raw_expr(module, expr)),
        },
        Expr::Concat(parts) => {
            let body = parts
                .iter()
                .map(|part| emit_raw_expr(module, part))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{body}}}")
        }
        Expr::MemRead { mem, addr } => {
            format!(
                "{}[{}]",
                signal_name(module, *mem),
                emit_raw_expr(module, addr)
            )
        }
    }
}

fn bin(module: &CompiledModule, op: &str, lhs: &Expr, rhs: &Expr) -> String {
    format!(
        "({} {} {})",
        emit_raw_expr(module, lhs),
        op,
        emit_raw_expr(module, rhs)
    )
}

fn sv_type(ty: rrtl_core::BitType) -> String {
    let signed = if ty.is_signed() { "signed " } else { "" };
    if ty.width == 1 {
        signed.to_string()
    } else {
        format!("{}[{}:0] ", signed, ty.width - 1)
    }
}

fn literal(value: i128, ty: rrtl_core::BitType) -> String {
    let signed = if ty.is_signed() { "s" } else { "" };
    if value < 0 {
        format!("{}'{}d{}", ty.width, signed, value)
    } else {
        format!("{}'{}d{}", ty.width, signed, value)
    }
}

fn signal_name(module: &CompiledModule, signal: Signal) -> &str {
    module
        .signal(signal)
        .map(|s| s.name.as_str())
        .unwrap_or("<invalid>")
}

fn state_signal_enum_type(module: &CompiledModule, signal: Signal) -> Option<String> {
    module
        .state_signals
        .iter()
        .find(|state_signal| state_signal.signal == signal)
        .map(|state_signal| state_enum_type_name(&state_signal.state_type.name))
}

fn state_enum_type_name(name: &str) -> String {
    format!("{}_t", sv_ident(name))
}

fn sv_ident(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn sv_string(value: &str) -> String {
    value
        .chars()
        .flat_map(|ch| match ch {
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\n' => "\\n".chars().collect::<Vec<_>>(),
            '\r' => "\\r".chars().collect::<Vec<_>>(),
            '\t' => "\\t".chars().collect::<Vec<_>>(),
            _ => vec![ch],
        })
        .collect()
}

fn expr_width(module: &CompiledModule, expr: &Expr) -> Option<Width> {
    match expr {
        Expr::Lit { ty, .. } => Some(ty.width),
        Expr::Signal(signal) => module.signal(*signal).map(|s| s.width),
        Expr::Not(inner) => expr_width(module, inner),
        Expr::And(lhs, _)
        | Expr::Or(lhs, _)
        | Expr::Xor(lhs, _)
        | Expr::Add(lhs, _)
        | Expr::Sub(lhs, _)
        | Expr::Mul(lhs, _) => expr_width(module, lhs),
        Expr::Eq(_, _) | Expr::Ne(_, _) | Expr::Lt(_, _) => Some(1),
        Expr::Mux { then_expr, .. } => expr_width(module, then_expr),
        Expr::Slice { width, .. }
        | Expr::Zext { width, .. }
        | Expr::Sext { width, .. }
        | Expr::Trunc { width, .. } => Some(*width),
        Expr::Cast { expr, .. } => expr_width(module, expr),
        Expr::Concat(parts) => parts.iter().try_fold(0, |total, part| {
            expr_width(module, part).map(|width| total + width)
        }),
        Expr::MemRead { mem, .. } => module.signal(*mem).map(|s| s.width),
    }
}

#[allow(dead_code)]
fn _signal_debug(signal: &SignalInfo) -> (&str, Width) {
    (&signal.name, signal.width)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rrtl_core::{
        bundle_type, field, iface_input, iface_output, interface_type, lit, lit_s, mux, nested,
        rv_bundle, rv_scalar, sext, sint, trunc, uint, zext,
    };
    use std::fs;
    use std::process::Command;

    #[test]
    fn emits_counter_systemverilog() {
        let mut design = Design::new();
        {
            let mut m = design.module("Counter");
            let clk = m.input("clk", 1);
            let rst = m.input("rst", 1);
            let en = m.input("en", 1);
            let out = m.output("out", 8);
            let count = m.reg("count", 8);
            m.clock(count, clk);
            m.reset(count, rst, 0);
            m.next(count, mux(en, count + lit(1, 8), count));
            m.assign(out, count);
        }

        let sv = emit(&design).unwrap();
        assert!(sv.contains("module Counter(clk, rst, en, out);"));
        assert!(sv.contains("always_ff @(posedge clk)"));
        assert!(sv.contains("assign out = count;"));
    }

    #[test]
    fn emits_counter_golden_systemverilog() {
        let mut design = Design::new();
        {
            let mut m = design.module("Counter");
            let clk = m.input("clk", 1);
            let rst = m.input("rst", 1);
            let en = m.input("en", 1);
            let out = m.output("out", 8);
            let count = m.reg("count", 8);
            m.clock(count, clk);
            m.reset(count, rst, 0);
            m.next(count, mux(en, count + lit(1, 8), count));
            m.assign(out, count);
        }

        let compiled = compile(&design).unwrap();
        let sv = emit_compiled(&compiled);
        let expected = "module Counter(clk, rst, en, out);\n  input logic clk;\n  input logic rst;\n  input logic en;\n  output logic [7:0] out;\n  logic [7:0] count;\n  assign out = count;\n  always_ff @(posedge clk) begin\n    if (rst) count <= 8'd0;\n    else count <= (en ? (count + 8'd1) : count);\n  end\nendmodule\n\n";
        assert_eq!(sv, expected);
    }

    #[test]
    fn emits_async_reset_systemverilog() {
        let mut design = Design::new();
        {
            let mut m = design.module("AsyncCounter");
            let clk = m.input("clk", 1);
            let rst = m.input("rst", 1);
            let en = m.input("en", 1);
            let out = m.output("out", 8);
            let count = m.reg("count", 8);
            m.clock(count, clk);
            m.async_reset(count, rst, 0);
            m.next(count, mux(en, count + lit(1, 8), count));
            m.assign(out, count);
        }

        let sv = emit(&design).unwrap();
        assert!(sv.contains("always_ff @(posedge clk or posedge rst) begin"));
        assert!(sv.contains("if (rst) count <= 8'd0;"));
        assert!(sv.contains("else count <= (en ? (count + 8'd1) : count);"));
    }

    #[test]
    fn emits_active_low_reset_systemverilog() {
        let mut design = Design::new();
        {
            let mut m = design.module("ActiveLowReset");
            let clk = m.input("clk", 1);
            let sync_rst_n = m.input("sync_rst_n", 1);
            let async_rst_n = m.input("async_rst_n", 1);
            let sync_count = m.reg("sync_count", 8);
            let async_count = m.reg("async_count", 8);
            m.clock(sync_count, clk);
            m.reset_low(sync_count, sync_rst_n, 0);
            m.next(sync_count, sync_count + lit(1, 8));
            m.clock(async_count, clk);
            m.async_reset_low(async_count, async_rst_n, 0);
            m.next(async_count, async_count + lit(1, 8));
        }

        let sv = emit(&design).unwrap();
        assert!(sv.contains("always_ff @(posedge clk) begin"));
        assert!(sv.contains("if (!sync_rst_n) sync_count <= 8'd0;"));
        assert!(sv.contains("always_ff @(posedge clk or negedge async_rst_n) begin"));
        assert!(sv.contains("if (!async_rst_n) async_count <= 8'd0;"));
    }

    #[test]
    fn emits_assertion_systemverilog() {
        let mut design = Design::new();
        {
            let mut m = design.module("Assertions");
            let clk = m.input("clk", 1);
            let enable = m.input("enable", 1);
            let count = m.input("count", 4);
            m.assert_msg(
                "comb_ok",
                count.value().lt_expr(rrtl_core::lit_u(10, 4)),
                "count too high",
            );
            m.assert_when(
                "enabled_ok",
                enable,
                count.value().lt_expr(rrtl_core::lit_u(12, 4)),
            );
            m.assert_clocked(
                "clocked_ok",
                clk,
                count.value().lt_expr(rrtl_core::lit_u(14, 4)),
            );
        }

        let sv = emit(&design).unwrap();
        assert!(sv.contains("always_comb begin"));
        assert!(sv.contains("COMB_OK: assert ((count < 4'd10)) else $error(\"count too high\");"));
        assert!(sv.contains(
            "if (enable) ENABLED_OK: assert ((count < 4'd12)) else $error(\"enabled_ok\");"
        ));
        assert!(sv.contains("always_ff @(posedge clk) begin"));
        assert!(sv.contains("CLOCKED_OK: assert ((count < 4'd14)) else $error(\"clocked_ok\");"));
    }

    #[test]
    fn emits_cover_systemverilog() {
        let mut design = Design::new();
        {
            let mut m = design.module("Covers");
            let clk = m.input("clk", 1);
            let enable = m.input("enable", 1);
            let count = m.input("count", 4);
            m.cover("comb_hit", count.value().eq_expr(rrtl_core::lit_u(3, 4)));
            m.cover_when(
                "enabled_hit",
                enable,
                count.value().eq_expr(rrtl_core::lit_u(5, 4)),
            );
            m.cover_clocked(
                "clocked_hit",
                clk,
                count.value().eq_expr(rrtl_core::lit_u(7, 4)),
            );
        }

        let sv = emit(&design).unwrap();
        assert!(sv.contains("always_comb begin"));
        assert!(sv.contains("COMB_HIT: cover ((count == 4'd3));"));
        assert!(sv.contains("if (enable) ENABLED_HIT: cover ((count == 4'd5));"));
        assert!(sv.contains("always_ff @(posedge clk) begin"));
        assert!(sv.contains("CLOCKED_HIT: cover ((count == 4'd7));"));
    }

    #[test]
    fn emits_bundle_fields_as_prefixed_scalars() {
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
            let mut m = design.module("BundlePorts");
            let req = m.input_bundle("req", req_ty);
            let y = m.output("y", uint(8));
            m.assign(y, zext(req.path(["meta", "tag"]).unwrap(), 8));
        }

        let sv = emit(&design).unwrap();
        assert!(sv.contains("module BundlePorts(req_valid, req_addr, req_meta_tag, y);"));
        assert!(sv.contains("input logic req_valid;"));
        assert!(sv.contains("input logic [7:0] req_addr;"));
        assert!(sv.contains("input logic [3:0] req_meta_tag;"));
        assert!(sv.contains("assign y = {4'd0, req_meta_tag};"));
    }

    #[test]
    fn emits_bundle_assignment_systemverilog() {
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

        let sv = emit(&design).unwrap();
        assert!(sv.contains("assign pipe_valid = req_valid;"));
        assert!(sv.contains("assign pipe_addr = req_addr;"));
        assert!(sv.contains("assign pipe_meta_tag = req_meta_tag;"));
        assert!(sv.contains("assign resp_valid = pipe_valid;"));
        assert!(sv.contains("assign resp_addr = pipe_addr;"));
        assert!(sv.contains("assign resp_meta_tag = pipe_meta_tag;"));
    }

    #[test]
    fn emits_interface_ports_as_prefixed_scalars() {
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
        let mut design = Design::new();
        {
            let mut m = design.module("InterfacePorts");
            let bus = m.interface("bus", bus_ty);
            let req = bus.port("req").unwrap().bundle().unwrap().clone();
            let resp = bus.port("resp").unwrap().bundle().unwrap().clone();
            m.assign(
                resp.field("data").unwrap(),
                zext(req.path(["meta", "tag"]).unwrap(), 8),
            );
        }

        let sv = emit(&design).unwrap();
        assert!(sv.contains(
            "module InterfacePorts(bus_req_valid, bus_req_meta_tag, bus_resp_data, bus_ready);"
        ));
        assert!(sv.contains("input logic bus_req_valid;"));
        assert!(sv.contains("input logic [3:0] bus_req_meta_tag;"));
        assert!(sv.contains("output logic [7:0] bus_resp_data;"));
        assert!(sv.contains("input logic bus_ready;"));
        assert!(sv.contains("assign bus_resp_data = {4'd0, bus_req_meta_tag};"));
    }

    #[test]
    fn emits_external_module_instance_without_body() {
        let mut design = Design::new();
        {
            let mut ext = design.extern_module("VendorAddOne");
            ext.input("a", uint(8));
            ext.output("y", uint(8));
        }
        {
            let mut m = design.module("Top");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.instance("u_vendor", "VendorAddOne", [("a", a), ("y", y)]);
        }

        let sv = emit(&design).unwrap();
        assert!(!sv.contains("module VendorAddOne"));
        assert!(sv.contains("module Top(a, y);"));
        assert!(sv.contains("VendorAddOne u_vendor (.a(a), .y(y));"));
    }

    #[test]
    fn emits_inout_ports_and_external_inout_instance() {
        let mut design = Design::new();
        {
            let mut ext = design.extern_module("IOBUF");
            ext.inout("PAD", uint(1));
            ext.input("I", uint(1));
            ext.input("T", uint(1));
            ext.output("O", uint(1));
        }
        {
            let mut m = design.module("Top");
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

        let sv = emit(&design).unwrap();
        assert!(!sv.contains("module IOBUF"));
        assert!(sv.contains("module Top(pad, i, t, o);"));
        assert!(sv.contains("inout wire pad;"));
        assert!(sv.contains("IOBUF u_iobuf (.PAD(pad), .I(i), .T(t), .O(o));"));
    }

    #[test]
    fn emits_ready_valid_channels_as_prefixed_scalars() {
        let payload_ty = bundle_type("Payload", [field("data", uint(8)), field("last", uint(1))]);
        let mut design = Design::new();
        {
            let mut m = design.module("ReadyValidPorts");
            let stream = m.rv_source("stream", rv_bundle(payload_ty));
            let bits = stream.bits_bundle().unwrap();
            let fired = m.output("fired", uint(1));
            m.assign(fired, stream.fire());
            m.assign(bits.field("data").unwrap(), lit(0, 8));
        }

        let sv = emit(&design).unwrap();
        assert!(sv.contains(
            "module ReadyValidPorts(stream_valid, stream_ready, stream_bits_data, stream_bits_last, fired);"
        ));
        assert!(sv.contains("output logic stream_valid;"));
        assert!(sv.contains("input logic stream_ready;"));
        assert!(sv.contains("output logic [7:0] stream_bits_data;"));
        assert!(sv.contains("output logic stream_bits_last;"));
        assert!(sv.contains("assign fired = (stream_valid & stream_ready);"));
    }

    #[test]
    fn emits_ready_valid_connect_systemverilog() {
        let mut design = Design::new();
        {
            let mut m = design.module("ReadyValidConnect");
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_connect(&input, &output);
        }

        let sv = emit(&design).unwrap();
        assert!(sv.contains("assign out_valid = in_valid;"));
        assert!(sv.contains("assign in_ready = out_ready;"));
        assert!(sv.contains("assign out_bits = in_bits;"));
    }

    #[test]
    fn emits_ready_valid_register_slice_systemverilog() {
        let mut design = Design::new();
        {
            let mut m = design.module("ReadyValidSlice");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_register_slice("slice", &input, &output, clk, rst);
        }

        let sv = emit(&design).unwrap();
        assert!(sv.contains("logic slice_valid_q;"));
        assert!(sv.contains("logic [7:0] slice_bits_q;"));
        assert!(sv.contains("assign in_ready = out_ready;"));
        assert!(sv.contains("assign out_valid = slice_valid_q;"));
        assert!(sv.contains("assign out_bits = slice_bits_q;"));
        assert!(sv.contains("if (rst) slice_valid_q <= 1'd0;"));
        assert!(sv.contains("else slice_valid_q <= in_valid;"));
        assert!(sv.contains("if (rst) slice_bits_q <= 8'd0;"));
        assert!(sv.contains("else slice_bits_q <= in_bits;"));
    }

    #[test]
    fn emits_ready_valid_skid_buffer_systemverilog() {
        let mut design = Design::new();
        {
            let mut m = design.module("ReadyValidSkid");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_skid_buffer("skid", &input, &output, clk, rst);
        }

        let sv = emit(&design).unwrap();
        assert!(sv.contains("logic skid_full_q;"));
        assert!(sv.contains("logic [7:0] skid_bits_q;"));
        assert!(sv.contains("assign in_ready = (~(skid_full_q) | out_ready);"));
        assert!(sv.contains("assign out_valid = (skid_full_q | in_valid);"));
        assert!(sv.contains("assign out_bits = (skid_full_q ? skid_bits_q : in_bits);"));
        assert!(sv.contains(
            "else skid_full_q <= (out_ready ? (skid_full_q & in_valid) : (skid_full_q | in_valid));"
        ));
        assert!(sv.contains("if (rst) skid_full_q <= 1'd0;"));
        assert!(sv.contains("if (rst) skid_bits_q <= 8'd0;"));
        assert!(sv.contains("else skid_bits_q <= ((in_valid & ((~(skid_full_q) & ~(out_ready)) | (skid_full_q & out_ready))) ? in_bits : skid_bits_q);"));
    }

    #[test]
    fn emits_ready_valid_fifo_systemverilog() {
        let mut design = Design::new();
        {
            let mut m = design.module("ReadyValidFifo");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_fifo("fifo", &input, &output, clk, rst, 2);
        }

        let sv = emit(&design).unwrap();
        assert!(sv.contains("logic [1:0] fifo_count_q;"));
        assert!(sv.contains("logic fifo_wptr_q;"));
        assert!(sv.contains("logic fifo_rptr_q;"));
        assert!(sv.contains("logic [7:0] fifo_data_0_q;"));
        assert!(sv.contains("logic [7:0] fifo_data_1_q;"));
        assert!(sv.contains("assign in_ready = (fifo_count_q != 2'd2);"));
        assert!(sv.contains("assign out_valid = (fifo_count_q != 2'd0);"));
        assert!(sv.contains(
            "assign out_bits = ((fifo_rptr_q == 1'd1) ? fifo_data_1_q : fifo_data_0_q);"
        ));
        assert!(sv.contains("if (rst) fifo_count_q <= 2'd0;"));
        assert!(sv.contains("if (rst) fifo_wptr_q <= 1'd0;"));
        assert!(sv.contains("if (rst) fifo_rptr_q <= 1'd0;"));
        assert!(sv.contains("if (rst) fifo_data_0_q <= 8'd0;"));
        assert!(sv.contains("if (rst) fifo_data_1_q <= 8'd0;"));
    }

    #[test]
    fn emits_ready_valid_mem_fifo_systemverilog() {
        let payload_ty = bundle_type("Payload", [field("data", uint(8)), field("last", uint(1))]);
        let mut design = Design::new();
        {
            let mut m = design.module("ReadyValidMemFifo");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_bundle(payload_ty.clone()));
            let output = m.rv_source("out", rv_bundle(payload_ty));
            m.rv_mem_fifo("fifo", &input, &output, clk, rst, 2);
        }

        let sv = emit(&design).unwrap();
        assert!(sv.contains("logic [1:0] fifo_count_q;"));
        assert!(sv.contains("logic fifo_wptr_q;"));
        assert!(sv.contains("logic fifo_rptr_q;"));
        assert!(sv.contains("logic [8:0] fifo_mem [0:1];"));
        assert!(sv.contains("logic [8:0] fifo_read_data;"));
        assert!(sv.contains("assign in_ready = (fifo_count_q != 2'd2);"));
        assert!(sv.contains("assign out_valid = (fifo_count_q != 2'd0);"));
        assert!(sv.contains("assign fifo_read_data = fifo_mem[fifo_rptr_q];"));
        assert!(sv.contains("assign out_bits_data = fifo_read_data[8:1];"));
        assert!(sv.contains("assign out_bits_last = fifo_read_data[0];"));
        assert!(
            sv.contains("if ((in_valid & (fifo_count_q != 2'd2))) fifo_mem[fifo_wptr_q] <= {in_bits_data, in_bits_last};")
        );
    }

    #[test]
    fn emits_memory_ports_and_width_helpers() {
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

            m.mem_write(mem, clk, we, waddr, zext(trunc(wdata, 4), 8));
            let read = m.mem_read(mem, raddr);
            m.assign(rdata, read);
            m.assign(low, trunc(wdata, 4));
        }

        let sv = emit(&design).unwrap();
        assert!(sv.contains("logic [7:0] regs [0:3];"));
        assert!(sv.contains("assign rdata = regs[raddr];"));
        assert!(sv.contains("if (we) regs[waddr] <= {4'd0, wdata[3:0]};"));
        assert!(sv.contains("assign low = wdata[3:0];"));
    }

    #[test]
    fn emits_register_file_golden_systemverilog() {
        let mut design = Design::new();
        {
            let mut m = design.module("RegFile");
            let clk = m.input("clk", 1);
            let we = m.input("we", 1);
            let waddr = m.input("waddr", 2);
            let wdata = m.input("wdata", 8);
            let raddr = m.input("raddr", 2);
            let rdata = m.output("rdata", 8);
            let mem = m.mem("regs", 2, 8, 4);

            m.mem_write(mem, clk, we, waddr, wdata);
            let read = m.mem_read(mem, raddr);
            m.assign(rdata, read);
        }

        let sv = emit(&design).unwrap();
        let expected = "module RegFile(clk, we, waddr, wdata, raddr, rdata);\n  input logic clk;\n  input logic we;\n  input logic [1:0] waddr;\n  input logic [7:0] wdata;\n  input logic [1:0] raddr;\n  output logic [7:0] rdata;\n  logic [7:0] regs [0:3];\n  assign rdata = regs[raddr];\n  always_ff @(posedge clk) begin\n    if (we) regs[waddr] <= wdata;\n  end\nendmodule\n\n";
        assert_eq!(sv, expected);
    }

    #[test]
    fn emits_signed_systemverilog() {
        let mut design = Design::new();
        {
            let mut m = design.module("SignedAlu");
            let a = m.input("a", sint(8));
            let b = m.input("b", sint(8));
            let sum = m.output("sum", sint(8));
            let lt = m.output("lt", uint(1));
            let wide = m.output("wide", sint(16));
            m.assign(sum, a + b);
            m.assign(lt, a.value().lt_expr(b));
            m.assign(wide, sext(a + lit_s(-1, 8), 16));
        }

        let sv = emit(&design).unwrap();
        assert!(sv.contains("input logic signed [7:0] a;"));
        assert!(sv.contains("output logic signed [15:0] wide;"));
        assert!(sv.contains("8'sd-1"));
        assert!(sv.contains("{{8{(a + 8'sd-1)[7]}}, (a + 8'sd-1)}"));
    }

    #[test]
    fn emits_state_enums_and_typed_registers() {
        let mut design = Design::new();
        {
            let mut m = design.module("Controller");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let start = m.input("start", uint(1));
            let busy = m.output("busy", uint(1));
            let scratch = m.reg("scratch", uint(2));
            let states = rrtl_core::state_type(
                "ControllerState",
                uint(2),
                [("Idle", 0), ("Run", 1), ("Done", 2)],
            );
            let state = m.state_reg("state", states, clk, rst, "Idle");
            m.state_next_hold(&state, [(start.value(), "Run")]);
            m.assign(busy, state.is("Run"));
            m.clock(scratch, clk);
            m.reset(scratch, rst, 0);
            m.next(scratch, lit(0, 2));
        }

        let sv = emit(&design).unwrap();
        assert!(sv.contains("typedef enum logic [1:0] { CONTROLLERSTATE_IDLE = 2'd0, CONTROLLERSTATE_RUN = 2'd1, CONTROLLERSTATE_DONE = 2'd2 } CONTROLLERSTATE_t;"));
        assert!(sv.contains("CONTROLLERSTATE_t state;"));
        assert!(sv.contains("logic [1:0] scratch;"));
        assert!(sv.contains("if (rst) state <= 2'd0;"));
        assert!(!sv.contains("localparam CONTROLLERSTATE_IDLE"));
        assert!(sv.contains("assign busy = (state == 2'd1);"));

        let typedef_index = sv.find("typedef enum logic").unwrap();
        let state_decl_index = sv.find("CONTROLLERSTATE_t state;").unwrap();
        let always_index = sv.find("always_ff").unwrap();
        assert!(typedef_index < state_decl_index);
        assert!(state_decl_index < always_index);
    }

    #[test]
    fn emits_async_reset_state_enum_registers() {
        let mut design = Design::new();
        {
            let mut m = design.module("AsyncController");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let states =
                rrtl_core::state_type("ControllerState", uint(1), [("Idle", 0), ("Run", 1)]);
            let state = m.state_reg_async_reset("state", states, clk, rst, "Idle");
            m.state_next_hold(&state, std::iter::empty::<(Expr, String)>());
        }

        let sv = emit(&design).unwrap();
        assert!(sv.contains("typedef enum logic { CONTROLLERSTATE_IDLE = 1'd0, CONTROLLERSTATE_RUN = 1'd1 } CONTROLLERSTATE_t;"));
        assert!(sv.contains("CONTROLLERSTATE_t state;"));
        assert!(sv.contains("always_ff @(posedge clk or posedge rst) begin"));
    }

    #[test]
    fn emits_active_low_state_enum_registers() {
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

        let sv = emit(&design).unwrap();
        assert!(sv.contains("always_ff @(posedge clk or negedge rst_n) begin"));
        assert!(sv.contains("if (!rst_n) state <= 1'd0;"));
    }

    #[test]
    #[ignore = "optional external EDA smoke check; runs only when yosys is installed"]
    fn optional_yosys_accepts_counter_sv() {
        if Command::new("yosys").arg("-V").output().is_err() {
            return;
        }
        let sv_path = write_counter_sv("rrtl_counter_yosys.sv");
        let status = Command::new("yosys")
            .args([
                "-q",
                "-p",
                &format!("read_verilog -sv {}; hierarchy -check", sv_path.display()),
            ])
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    #[ignore = "optional external EDA smoke check; runs only when verilator is installed"]
    fn optional_verilator_lints_counter_sv() {
        if Command::new("verilator").arg("--version").output().is_err() {
            return;
        }
        let sv_path = write_counter_sv("rrtl_counter_verilator.sv");
        let status = Command::new("verilator")
            .args(["--lint-only", "--timing"])
            .arg(&sv_path)
            .status()
            .unwrap();
        assert!(status.success());
    }

    fn write_counter_sv(file_name: &str) -> std::path::PathBuf {
        let mut design = Design::new();
        {
            let mut m = design.module("Counter");
            let clk = m.input("clk", 1);
            let rst = m.input("rst", 1);
            let en = m.input("en", 1);
            let out = m.output("out", 8);
            let count = m.reg("count", 8);
            m.clock(count, clk);
            m.reset(count, rst, 0);
            m.next(count, mux(en, count + lit(1, 8), count));
            m.assign(out, count);
        }

        let path = std::env::temp_dir().join(file_name);
        fs::write(&path, emit(&design).unwrap()).unwrap();
        path
    }
}
