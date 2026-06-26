//! Validate the double-buffered (zero-copy) tick: run a design via the normal
//! single-buffer tick and via tick_db(cur, nxt)+swap, and check they agree.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example tick_db_check
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

    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv");
    let src = std::fs::read_to_string(format!("{base}/cfgdsp.sv")).unwrap();
    let imported = import_sv(&src, Some("cfgdsp")).unwrap();
    let compiled = rrtl_core::compile(&imported.design).unwrap();
    let program = lower_to_packed_program(&compiled, "cfgdsp").unwrap();
    let machine = lower_to_machine_program(&program);
    let idx = |n: &str| program.signals.iter().position(|s| s.name.ends_with(n)).unwrap();
    let (clk, rst, cfg_we, cfg_in, a, b, acc) =
        (idx(".clk"), idx(".rst"), idx(".cfg_we"), idx(".cfg_in"), idx(".a"), idx(".b"), idx(".acc"));

    let stim = |c: u64| -> [(usize, u64); 6] {
        let r = (c == 0) as u64;
        let (we, ci) = if c == 1 { (1, 1) } else { (0, 0) };
        [(clk, 1), (rst, r), (cfg_we, we), (cfg_in, ci),
         (a, c.wrapping_mul(2_654_435_761)), (b, c.wrapping_mul(40_503) + 7)]
    };

    // reference: single-buffer tick
    let mut refj = JitSimulator::compile(&machine).unwrap();
    let mut ref_trace = vec![0u64; 300];
    for c in 0..300u64 {
        for &(i, v) in &stim(c) {
            refj.set_signal(i, v);
        }
        refj.tick();
        ref_trace[c as usize] = refj.get_signal(acc);
    }

    // double-buffered: tick_db(cur, nxt) + swap, inputs set into nxt, output read from cur
    let dbj = JitSimulator::compile(&machine).unwrap();
    let n = dbj.state_words().len();
    let (mut cur, mut nxt) = (vec![0i64; n], vec![0i64; n]);
    let mut mism = 0;
    for c in 0..300u64 {
        for &(i, v) in &stim(c) {
            nxt[i * 2] = v as i64; // inputs into the write buffer
            nxt[i * 2 + 1] = 0;
        }
        dbj.tick_db(&cur, &mut nxt);
        std::mem::swap(&mut cur, &mut nxt);
        let v = cur[acc * 2] as u64;
        if v != ref_trace[c as usize] {
            mism += 1;
        }
    }
    println!("tick_db vs single-buffer over 300 cycles: {mism} mismatches");
    println!("  => {}", if mism == 0 { "double-buffered tick is bit-exact" } else { "FAIL" });
}
