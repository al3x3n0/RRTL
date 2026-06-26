use std::fs;
fn main() {
    let root = "/tmp/vortex/hw/rtl";
    let files = std::env::args().skip(1).collect::<Vec<_>>();
    let (top, mods) = (files[0].clone(), &files[1..]);
    let mut src = String::new();
    src.push_str(&fs::read_to_string(format!("{root}/VX_platform.vh")).unwrap());
    src.push('\n');
    for m in mods { src.push_str(&fs::read_to_string(m).unwrap()); src.push('\n'); }
    match rrtl_sv_frontend::import_sv(&src, Some(&top)) {
        Ok(imp) => match rrtl_core::compile(&imp.design) {
            Ok(c) => println!("OK: `{top}` ({} modules)", c.modules().len()),
            Err(e) => println!("COMPILE-ERR: {e}"),
        },
        Err(e) => println!("IMPORT-ERR: {e}"),
    }
}
