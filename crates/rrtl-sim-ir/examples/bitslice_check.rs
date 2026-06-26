//! Validate + benchmark the bit-sliced multi-bit batch engine on a control-ish
//! design (8-bit datapath + 1-bit control + mux/add/xor/eq), bit-exact vs the
//! SIMD CPU engine, with a throughput comparison vs the vector JIT (whose single
//! width-class runs the 1-bit signals at the wide cfg's poor density).
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example bitslice_check -- [lanes steps]
use rrtl_sim_ir::bitslice::BitSliceSimulator;
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program, SimdCpuSimulator};
use rrtl_core::{compile, uint, Design, Signal};
use std::time::Instant;

fn build() -> Design {
    let mut design = Design::new();
    {
        let mut m = design.module("Ctl");
        let clk = m.input("clk", uint(1));
        let sel = m.input("sel", uint(1));
        let a = m.input("a", uint(8));
        let b = m.input("b", uint(8));
        let acc = m.reg("acc", uint(8));
        let flag = m.reg("flag", uint(1));
        m.clock(acc, clk);
        m.clock(flag, clk);
        // acc <= sel ? acc + a : acc ^ b   (mux over add/xor)
        m.next(acc, rrtl_ir::mux(sel.value(), acc.value() + a.value(), acc.value() ^ b.value()));
        // flag <= (acc == a)               (1-bit compare — the density mismatch)
        m.next(flag, acc.value().eq_expr(a.value()));
        let o = m.output("o", uint(8));
        let f = m.output("f", uint(1));
        m.assign(o, acc.value());
        m.assign(f, flag.value());
    }
    design
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lanes: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20000);

    let design = build();
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "Ctl").unwrap();
    let machine = lower_to_machine_program(&program);
    let h = |n: &str| -> Signal {
        compiled.find_module("Ctl").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle
    };
    let idx = |n: &str| program.signal_index(h(n)).unwrap();

    let mut bs = BitSliceSimulator::new(&machine, lanes).expect("bit-slice applies");
    let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
    cpu.set_signal(h("clk"), &vec![1u128; lanes]).unwrap();

    // distinct per-lane stimulus
    let inval = |seed: u64, lane: usize, m: u128| ((seed.wrapping_mul(2654435761).wrapping_add(lane as u64)) as u128) & m;
    let drive = |bs: &mut BitSliceSimulator, cpu: &mut SimdCpuSimulator, seed: u64| {
        for (name, w) in [("sel", 1u128), ("a", 0xff), ("b", 0xff)] {
            let vals: Vec<u128> = (0..lanes).map(|l| inval(seed + name.len() as u64, l, w)).collect();
            for (l, v) in vals.iter().enumerate() {
                bs.set_signal(idx(name), l, *v);
            }
            cpu.set_signal(h(name), &vals).unwrap();
        }
    };

    // bit-exact over several cycles, comparing both outputs on every lane.
    let mut mism = 0usize;
    for c in 0..40 {
        drive(&mut bs, &mut cpu, c as u64);
        bs.tick();
        cpu.tick().unwrap();
        for (name, _) in [("o", 8u128), ("f", 1)] {
            let cv = cpu.get_signal(h(name)).unwrap();
            for l in 0..lanes {
                if bs.get_signal(idx(name), l) != cv[l] {
                    mism += 1;
                }
            }
        }
    }
    println!("bit-slice engine: Ctl (8-bit datapath + 1-bit control), {lanes} lanes");
    println!("  bit-exact vs SIMD CPU (o,f × {lanes} lanes × 40 cyc): {}", if mism == 0 { "YES" } else { "NO" });

    // throughput: bit-slice vs SIMD-CPU vs vector JIT.
    let t = Instant::now();
    bs.tick_many(steps);
    let bs_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    cpu.tick_many(steps).unwrap();
    let cpu_s = t.elapsed().as_secs_f64();
    let mlc = |s: f64| (lanes * steps) as f64 / s / 1e6;
    println!("  bit-slice : {:.1} M-lane-cyc/s", mlc(bs_s));
    println!("  SIMD CPU  : {:.1} M-lane-cyc/s  ({:.2}x)", mlc(cpu_s), cpu_s / bs_s);
    #[cfg(feature = "jit")]
    {
        use rrtl_sim_ir::jit::SimdJitSimulator;
        if let Ok(mut v) = SimdJitSimulator::compile_lanes(&machine, lanes) {
            for l in 0..lanes {
                v.set_signal(l, idx("a"), inval(1, l, 0xff) as u32);
            }
            for _ in 0..7 { v.tick_many(1); }
            let t = Instant::now();
            v.tick_many(steps);
            let v_s = t.elapsed().as_secs_f64();
            println!("  vector JIT: {:.1} M-lane-cyc/s  (bit-slice is {:.2}x)", mlc(v_s), v_s / bs_s);
        }
    }
    assert_eq!(mism, 0, "bit-slice diverged from SIMD CPU");
}
