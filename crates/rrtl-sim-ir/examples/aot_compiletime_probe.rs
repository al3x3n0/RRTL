//! The specializer is runtime-neutral on a single CPU instance (clang -O3 subsumes
//! const-fold/DCE) — but it removes 45.6% of the IR and 40% of the generated C,
//! which clang must still PARSE and optimize. Compile time is an RRTL moat (Axis 3:
//! Verilator spends hours on huge designs). This measures clang -O3 wall-time on the
//! plain vs specialized C for a design, best-of-3.
//! Build: cargo run --release --features aot -p rrtl-sim-ir --example aot_compiletime_probe -- bench/sv/picorv32.v picorv32
fn main() {
    #[cfg(not(feature = "aot"))]
    println!("build with --features aot");
    #[cfg(feature = "aot")]
    run();
}

#[cfg(feature = "aot")]
fn run() {
    use rrtl_sim_ir::specialize::specialize_program;
    use rrtl_sim_ir::{aot, lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;
    use std::time::Instant;

    let path = std::env::args().nth(1).unwrap_or_else(|| "bench/sv/picorv32.v".into());
    let top = std::env::args().nth(2).unwrap_or_else(|| "picorv32".into());
    let src = std::fs::read_to_string(&path).expect("read top");
    let imported = import_sv(&src, Some(&top)).unwrap();
    let compiled = rrtl_core::compile(&imported.design).unwrap();
    let program = lower_to_packed_program(&compiled, &top).unwrap();
    let machine = lower_to_machine_program(&program);
    let (spec, _) = specialize_program(&machine);

    let c_plain = aot::generate_c(&machine).unwrap();
    let c_spec = aot::generate_c(&spec).unwrap();

    let cc = std::env::var("CC").unwrap_or_else(|_| "clang".into());
    let time_cc = |c: &str, tag: &str| -> f64 {
        let dir = std::env::temp_dir();
        let cpath = dir.join(format!("rrtl_ct_{tag}.c"));
        let opath = dir.join(format!("rrtl_ct_{tag}.{}", if cfg!(target_os = "macos") { "dylib" } else { "so" }));
        std::fs::write(&cpath, c).unwrap();
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            let out = std::process::Command::new(&cc)
                .args(["-O3", "-shared", "-fPIC", "-o"]).arg(&opath).arg(&cpath)
                .output().unwrap();
            assert!(out.status.success(), "clang failed: {}", String::from_utf8_lossy(&out.stderr));
            best = best.min(t.elapsed().as_secs_f64());
        }
        best * 1e3
    };

    let tp = time_cc(&c_plain, "plain");
    let ts = time_cc(&c_spec, "spec");
    println!("clang -O3 compile time on `{top}` ({} signals):", program.signals.len());
    println!("  plain C ({:>6} bytes): {tp:.0} ms", c_plain.len());
    println!("  spec  C ({:>6} bytes): {ts:.0} ms   => {:.2}x faster compile", c_spec.len(), tp / ts);
}
