//! A minimal Verilog preprocessor (`define / `ifdef / macro expansion).
use std::collections::HashMap;

use rrtl_ir::ErrorReport;

use crate::err;

type MacroTable = HashMap<String, (Option<Vec<String>>, String)>;

/// Minimal Verilog preprocessor: `define (object- and function-like), `undef,
/// `ifdef/`ifndef/`elsif/`else/`endif conditional compilation, macro expansion,
/// and ignored directives (`timescale, `default_nettype, …). No `include.
pub(crate) fn preprocess(src: &str) -> Result<String, ErrorReport> {
    fn split_ident(s: &str) -> (String, &str) {
        let mut end = 0;
        for (i, c) in s.char_indices() {
            if c.is_ascii_alphanumeric() || c == '_' || c == '$' {
                end = i + c.len_utf8();
            } else {
                break;
            }
        }
        (s[..end].to_string(), &s[end..])
    }
    fn parse_params(s: &str) -> Result<(Vec<String>, &str), ErrorReport> {
        let chars: Vec<char> = s.chars().collect();
        let mut depth = 0i32;
        let mut close = None;
        for (i, &c) in chars.iter().enumerate() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let close = close.ok_or_else(|| err("E_SV_PP", "unterminated macro parameter list"))?;
        let inner: String = chars[1..close].iter().collect();
        let params = inner
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect();
        let rest_start: usize = chars[..=close].iter().map(|c| c.len_utf8()).sum();
        Ok((params, &s[rest_start..]))
    }

    let mut defines: MacroTable = HashMap::new();
    let mut cond: Vec<(bool, bool)> = Vec::new(); // (active, a-true-branch-was-taken)
    let mut stage1 = String::new();
    let lines: Vec<&str> = src.lines().collect();
    let mut li = 0;
    while li < lines.len() {
        let raw = lines[li];
        li += 1;
        let active = cond.iter().all(|c| c.0);
        let trimmed = raw.trim_start();
        if let Some(rest) = trimmed.strip_prefix('`') {
            let (name, after) = split_ident(rest);
            match name.as_str() {
                "define" => {
                    let mut body = after.to_string();
                    while body.trim_end().ends_with('\\') {
                        let t = body.trim_end();
                        body = t[..t.len() - 1].to_string();
                        if li < lines.len() {
                            body.push('\n');
                            body.push_str(lines[li]);
                            li += 1;
                        } else {
                            break;
                        }
                    }
                    if active {
                        let (mname, r2) = split_ident(body.trim_start());
                        if mname.is_empty() {
                            return Err(err("E_SV_PP", "`define without a macro name"));
                        }
                        let (params, mbody) = if r2.starts_with('(') {
                            let (p, b) = parse_params(r2)?;
                            (Some(p), b.trim().to_string())
                        } else {
                            (None, r2.trim().to_string())
                        };
                        defines.insert(mname, (params, mbody));
                    }
                }
                "undef" => {
                    if active {
                        let (m, _) = split_ident(after.trim_start());
                        defines.remove(&m);
                    }
                }
                "ifdef" | "ifndef" => {
                    let (m, _) = split_ident(after.trim_start());
                    let want = if name == "ifdef" {
                        defines.contains_key(&m)
                    } else {
                        !defines.contains_key(&m)
                    };
                    let act = active && want;
                    cond.push((act, act));
                }
                "elsif" => {
                    let parent = cond.len() < 2 || cond[..cond.len() - 1].iter().all(|c| c.0);
                    let (m, _) = split_ident(after.trim_start());
                    let has = defines.contains_key(&m);
                    if let Some(top) = cond.last_mut() {
                        if top.1 {
                            top.0 = false;
                        } else if parent && has {
                            top.0 = true;
                            top.1 = true;
                        } else {
                            top.0 = false;
                        }
                    }
                }
                "else" => {
                    let parent = cond.len() < 2 || cond[..cond.len() - 1].iter().all(|c| c.0);
                    if let Some(top) = cond.last_mut() {
                        top.0 = parent && !top.1;
                        top.1 = true;
                    }
                }
                "endif" => {
                    cond.pop();
                }
                "timescale" | "default_nettype" | "resetall" | "celldefine" | "endcelldefine"
                | "unconnected_drive" | "nounconnected_drive" | "line" | "include"
                | "begin_keywords" | "end_keywords" => { /* ignored directive */ }
                _ => {
                    if active {
                        stage1.push_str(raw);
                        stage1.push('\n');
                    }
                }
            }
        } else if active {
            stage1.push_str(raw);
            stage1.push('\n');
        }
    }
    if !cond.is_empty() {
        return Err(err("E_SV_PP", "unterminated `ifdef/`ifndef"));
    }
    expand_uses(&stage1, &defines, 0)
}

/// Expand macro uses (` `name` / ` `name(args)` ), skipping comments and strings.
fn expand_uses(text: &str, defines: &MacroTable, depth: usize) -> Result<String, ErrorReport> {
    if depth > 64 {
        return Err(err("E_SV_PP", "macro expansion too deep (recursive macro?)"));
    }
    let c: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < c.len() {
        // line comment
        if c[i] == '/' && i + 1 < c.len() && c[i + 1] == '/' {
            while i < c.len() && c[i] != '\n' {
                out.push(c[i]);
                i += 1;
            }
            continue;
        }
        // block comment
        if c[i] == '/' && i + 1 < c.len() && c[i + 1] == '*' {
            out.push('/');
            out.push('*');
            i += 2;
            while i + 1 < c.len() && !(c[i] == '*' && c[i + 1] == '/') {
                out.push(c[i]);
                i += 1;
            }
            if i + 1 < c.len() {
                out.push('*');
                out.push('/');
                i += 2;
            }
            continue;
        }
        // string literal
        if c[i] == '"' {
            out.push('"');
            i += 1;
            while i < c.len() && c[i] != '"' {
                if c[i] == '\\' && i + 1 < c.len() {
                    out.push(c[i]);
                    i += 1;
                }
                out.push(c[i]);
                i += 1;
            }
            if i < c.len() {
                out.push('"');
                i += 1;
            }
            continue;
        }
        if c[i] == '`' {
            let mut j = i + 1;
            let mut name = String::new();
            while j < c.len() && (c[j].is_ascii_alphanumeric() || c[j] == '_' || c[j] == '$') {
                name.push(c[j]);
                j += 1;
            }
            if let Some((params, body)) = defines.get(&name) {
                let (expansion, next) = if let Some(params) = params {
                    let mut k = j;
                    while k < c.len() && c[k].is_whitespace() {
                        k += 1;
                    }
                    if k >= c.len() || c[k] != '(' {
                        return Err(err("E_SV_PP", format!("macro `{name} expects arguments")));
                    }
                    let (args, after) = read_macro_args(&c, k)?;
                    if args.len() != params.len() {
                        return Err(err(
                            "E_SV_PP",
                            format!("macro `{name}: expected {} args, got {}", params.len(), args.len()),
                        ));
                    }
                    (substitute_params(body, params, &args), after)
                } else {
                    (body.clone(), j)
                };
                out.push_str(&expand_uses(&expansion, defines, depth + 1)?);
                i = next;
                continue;
            }
            // unknown macro: leave literally (let the parser report if it matters)
            out.push('`');
            out.push_str(&name);
            i = j;
            continue;
        }
        out.push(c[i]);
        i += 1;
    }
    Ok(out)
}

/// Read `(arg, arg, …)` starting at `c[k] == '('`; commas inside nested parens
/// don't split. Returns the args and the index just past the closing `)`.
fn read_macro_args(c: &[char], k: usize) -> Result<(Vec<String>, usize), ErrorReport> {
    let mut depth = 0i32;
    // Commas inside `{…}` / `[…]` (concatenations, indices) don't separate args.
    let mut nest = 0i32;
    let mut i = k;
    let mut args: Vec<String> = Vec::new();
    let mut cur = String::new();
    loop {
        if i >= c.len() {
            return Err(err("E_SV_PP", "unterminated macro arguments"));
        }
        match c[i] {
            '(' => {
                depth += 1;
                if depth > 1 {
                    cur.push('(');
                }
                i += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    args.push(cur.trim().to_string());
                    i += 1;
                    break;
                }
                cur.push(')');
                i += 1;
            }
            ch @ ('{' | '[') => {
                nest += 1;
                cur.push(ch);
                i += 1;
            }
            ch @ ('}' | ']') => {
                nest -= 1;
                cur.push(ch);
                i += 1;
            }
            ',' if depth == 1 && nest == 0 => {
                args.push(cur.trim().to_string());
                cur.clear();
                i += 1;
            }
            ch => {
                cur.push(ch);
                i += 1;
            }
        }
    }
    if args.len() == 1 && args[0].is_empty() {
        args.clear();
    }
    Ok((args, i))
}

/// Substitute macro parameters (whole-identifier) in a body with their args.
fn substitute_params(body: &str, params: &[String], args: &[String]) -> String {
    let c: Vec<char> = body.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < c.len() {
        if c[i].is_ascii_alphabetic() || c[i] == '_' {
            let mut id = String::new();
            while i < c.len() && (c[i].is_ascii_alphanumeric() || c[i] == '_' || c[i] == '$') {
                id.push(c[i]);
                i += 1;
            }
            if let Some(p) = params.iter().position(|p| *p == id) {
                out.push_str(&args[p]);
            } else {
                out.push_str(&id);
            }
        } else {
            out.push(c[i]);
            i += 1;
        }
    }
    out
}
