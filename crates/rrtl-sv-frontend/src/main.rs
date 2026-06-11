use std::env;
use std::fs;
use std::io::{self, Write};

use rrtl_core::compile;
use rrtl_sv_frontend::import_sv;

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help") {
        usage(&mut io::stdout())?;
        return Ok(());
    }
    let command = args.remove(0);
    let path = args
        .first()
        .ok_or("missing SystemVerilog input path")?
        .clone();
    args.remove(0);
    let mut top = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--top" => {
                index += 1;
                top = Some(args.get(index).ok_or("--top requires a value")?.clone());
            }
            other => return Err(format!("unknown argument `{other}`").into()),
        }
        index += 1;
    }

    let source = fs::read_to_string(path)?;
    let imported = import_sv(&source, top.as_deref())?;
    match command.as_str() {
        "check" => {
            println!(
                "ok: imported top `{}` ({} module(s))",
                imported.top_name,
                imported.modules.len()
            );
        }
        "json" => {
            let compiled = compile(&imported.design)?;
            println!("{}", compiled.to_json_pretty()?);
        }
        "emit-sv" => {
            print!("{}", rrtl_sv::emit(&imported.design)?);
        }
        other => return Err(format!("unknown command `{other}`").into()),
    }
    Ok(())
}

fn usage(out: &mut impl Write) -> io::Result<()> {
    writeln!(
        out,
        "usage: sv2rrtl <check|json|emit-sv> <input.sv> [--top TOP]"
    )
}
