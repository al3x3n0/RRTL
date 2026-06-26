//! Probe the frontend's signed-arithmetic handling against known SV semantics, to
//! find the concrete gaps before fixing them. Signed compare, arithmetic right
//! shift, and sign-extension are where signed ≠ unsigned at the bit level.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example signed_check
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

    let src = r#"
    module stest(input signed [3:0] s4, input signed [7:0] a, input signed [7:0] b,
                 output [7:0] s4_to_u8, output signed [15:0] smul, output [15:0] umul_of_s);
      assign s4_to_u8  = s4;       // signed 4b -> unsigned 8b: sign-extend (RHS is signed)
      assign smul      = a * b;    // signed * signed -> signed product
      assign umul_of_s = a * b;    // signed product assigned to UNSIGNED 16b
    endmodule
    "#;
    let imported = match import_sv(src, Some("stest")) {
        Ok(i) => i,
        Err(e) => { println!("LOWER FAILED: {e}"); return; }
    };
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "stest").expect("lower");
    let machine = lower_to_machine_program(&program);
    let idx = |n: &str| {
        let h = compiled.find_module("stest").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
        program.signal_index(h).unwrap()
    };
    let mut sim = JitSimulator::compile(&machine).unwrap();
    let (s4, a, b) = (idx("s4"), idx("a"), idx("b"));
    let (s4_to_u8, smul, umul_of_s) = (idx("s4_to_u8"), idx("smul"), idx("umul_of_s"));
    let check = |name: &str, got: u64, want: u64| {
        println!("  {name:10} = {got:#x}  (want {want:#x})  {}", if got == want { "OK" } else { "WRONG" });
        got == want
    };
    let mut ok = true;

    // s4 = -1 (0xF) -> u8 sign-extend = 0xFF; a=-2(0xFE)*b=3 = -6 = 0xFFFA
    sim.set_signal(s4, 0xF);
    sim.set_signal(a, 0xFE);
    sim.set_signal(b, 0x03);
    sim.tick();
    println!("s4=-1, a=-2, b=3:");
    ok &= check("s4_to_u8", sim.get_signal(s4_to_u8) & 0xff, 0xFF);  // sign-extend on assign-to-unsigned
    ok &= check("smul", sim.get_signal(smul) & 0xffff, 0xFFFA);      // -2*3 = -6
    ok &= check("umul_of_s", sim.get_signal(umul_of_s) & 0xffff, 0xFFFA); // same bits

    // s4 = -8 (0x8) -> 0xF8; a=-5(0xFB)*b=-4(0xFC) = 20 = 0x14
    sim.set_signal(s4, 0x8);
    sim.set_signal(a, 0xFB);
    sim.set_signal(b, 0xFC);
    sim.tick();
    println!("s4=-8, a=-5, b=-4:");
    ok &= check("s4_to_u8", sim.get_signal(s4_to_u8) & 0xff, 0xF8);
    ok &= check("smul", sim.get_signal(smul) & 0xffff, 0x14);  // -5 * -4 = 20

    println!("\n  => {}", if ok { "signed coercion CORRECT" } else { "GAPS found (see WRONG above)" });
}
