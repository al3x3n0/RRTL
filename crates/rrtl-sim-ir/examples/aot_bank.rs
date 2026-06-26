//! Demonstrate WHEN the width-packed state layout matters: a synthetic bank of N
//! independent 32-bit accumulators (`acc_i <= acc_i + 1`). Each tick streams the
//! whole register file, so once the state exceeds cache the per-cycle cost is
//! memory-bandwidth-bound — and the 16-byte-uniform layout moves 4x the bytes of
//! the packed layout. picorv32 (~4KB state) fits L1 either way, so packing is
//! neutral there; this is the regime (huge designs) where it pays.
//! Build: cargo run --release --features aot -p rrtl-sim-ir --example aot_bank -- [N_REGS] [CYCLES]
fn main() {
    #[cfg(not(feature = "aot"))]
    println!("build with --features aot");
    #[cfg(feature = "aot")]
    run();
}

#[cfg(feature = "aot")]
fn run() {
    use rrtl_core::{compile, lit_u, uint, Design};
    use rrtl_sim_ir::{aot::AotSimulator, lower_to_machine_program, lower_to_packed_program};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(200_000);
    let cycles: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(2000);

    let mut design = Design::new();
    {
        let mut m = design.module("bank");
        let clk = m.input("clk", uint(1));
        for i in 0..n {
            let acc = m.reg(format!("acc{i}"), uint(32));
            m.clock(acc, clk);
            m.next(acc, acc.value() + lit_u(1, 32));
        }
    }
    let compiled = compile(&design).expect("compile");
    let program = lower_to_packed_program(&compiled, "bank").expect("lower packed");
    let machine = lower_to_machine_program(&program);

    let bench = |label: &str, sim: &mut AotSimulator| {
        let s = Instant::now();
        sim.tick_many(cycles);
        let dt = s.elapsed().as_secs_f64();
        println!(
            "  {label}: state={} KB, {:.2} Mcyc/s, {:.1} G reg-updates/s",
            sim.state_bytes() / 1024,
            cycles as f64 / dt / 1e6,
            (cycles as f64 * n as f64) / dt / 1e9,
        );
    };

    println!("bank of {n} x 32-bit accumulators, {cycles} cycles:");
    std::env::remove_var("AOT_FAT");
    let mut packed = AotSimulator::compile(&machine).expect("aot packed");
    std::env::set_var("AOT_FAT", "1");
    let mut fat = AotSimulator::compile(&machine).expect("aot fat");
    std::env::remove_var("AOT_FAT");

    // interleave to share thermal state
    for _ in 0..3 {
        bench("packed (4B/sig)", &mut packed);
        bench("16-byte slots  ", &mut fat);
    }
}
