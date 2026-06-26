//! Validate SystemVerilog function support: a design using a `function` must lower
//! and simulate bit-identically to the same logic written out by hand (proving the
//! inline-expansion is correct). Exercises inputs, a local, an if, and the return,
//! plus calling the function in both an always_ff and a continuous assign.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example function_check
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::jit::JitSimulator;
    use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;

    // With a function (local var + if + return, called twice).
    let with_fn = r#"
    module ftest(input clk, input [15:0] din, output [15:0] o);
      function [15:0] step(input [15:0] acc, input [15:0] x);
        logic [15:0] t;
        t = x * 16'd3;
        if (t > 16'd1000) step = acc + t - 16'd1000;
        else step = acc + t;
      endfunction
      reg [15:0] acc;
      always @(posedge clk) acc <= step(acc, din);
      assign o = step(acc, din);
    endmodule
    "#;

    // The same logic, manually inlined.
    let manual = r#"
    module ftest(input clk, input [15:0] din, output [15:0] o);
      reg [15:0] acc;
      wire [15:0] t = din * 16'd3;
      wire [15:0] nxt = (t > 16'd1000) ? (acc + t - 16'd1000) : (acc + t);
      always @(posedge clk) acc <= nxt;
      assign o = nxt;
    endmodule
    "#;

    let build = |src: &str| {
        let imported = import_sv(src, Some("ftest")).expect("import_sv");
        let compiled = rrtl_core::compile(&imported.design).expect("compile");
        let program = lower_to_packed_program(&compiled, "ftest").expect("lower");
        let machine = lower_to_machine_program(&program);
        let idx = |n: &str| {
            let h = compiled.find_module("ftest").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
            program.signal_index(h).unwrap()
        };
        (JitSimulator::compile(&machine).unwrap(), idx("clk"), idx("din"), idx("o"))
    };

    let (mut a, ac, ad, ao) = build(with_fn);
    let (mut b, bc, bd, bo) = build(manual);
    println!("function design lowered + JIT-compiled OK");

    let mut ok = true;
    let mut st = 0x1234_5678u32;
    let mut rng = || { st ^= st << 13; st ^= st >> 17; st ^= st << 5; st };
    for cyc in 0..2000 {
        let din = (rng() & 0xffff) as u64;
        a.set_signal(ac, 1); a.set_signal(ad, din);
        b.set_signal(bc, 1); b.set_signal(bd, din);
        a.tick(); b.tick();
        let (oa, ob) = (a.get_signal(ao), b.get_signal(bo));
        if oa != ob {
            println!("  MISMATCH at cycle {cyc}: fn={oa} manual={ob} (din={din})");
            ok = false;
            break;
        }
    }
    println!("  function vs manually-inlined: {}", if ok { "[PASS] bit-identical over 2000 cycles" } else { "[FAIL]" });
    assert!(ok, "function inlining diverged from manual inline");
}
