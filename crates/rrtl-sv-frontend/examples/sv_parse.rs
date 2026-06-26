//! Triage probe: run `parse_sv` then `import_sv` on a file and report the first
//! failing stage + diagnostics. Usage: sv_parse <path.v> [top]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: sv_parse <path.v> [top]");
    let top = args.get(2).map(|s| s.as_str());
    let src = std::fs::read_to_string(path).expect("read source");
    println!("source: {} ({} lines)", path, src.lines().count());

    match rrtl_sv_frontend::parse_sv(&src) {
        Ok(_) => println!("parse_sv: OK"),
        Err(e) => {
            println!("parse_sv: FAILED");
            for d in e.diagnostics.iter().take(8) {
                println!("  [{}] {}", d.code, d.message);
            }
            return;
        }
    }
    match rrtl_sv_frontend::import_sv(&src, top) {
        Ok(imp) => println!("import_sv: OK  top={} modules={:?}", imp.top_name, imp.modules),
        Err(e) => {
            println!("import_sv: FAILED ({} diagnostics)", e.diagnostics.len());
            for d in e.diagnostics.iter().take(20) {
                println!(
                    "  [{}] {}{}{}",
                    d.code,
                    d.message,
                    d.module.as_ref().map(|m| format!("  (mod {m})")).unwrap_or_default(),
                    d.signal.as_ref().map(|s| format!("  signal={s}")).unwrap_or_default(),
                );
            }
        }
    }
}
