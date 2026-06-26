//! Validate + benchmark the bit-slice AOT (clang -O3 per-plane C, inner word
//! loop auto-vectorized) on a real mixed-width SV design — crc32 (32-bit
//! datapath, 33% of ops 1-bit, no multiply). This is where the bit-slice win is
//! supposed to land (per the probe's ~2.4x projection vs the vector JIT, whose
//! I32X4 runs even the 1-bit control signals at only 4 lanes). Bit-exact vs the
//! SIMD CPU engine; throughput vs SIMD-CPU / vector-JIT / the bit-slice interp.
//! Build: cargo run --release --features "aot jit" -p rrtl-sim-ir --example bitslice_aot_bench -- [lanes steps]
use rrtl_sim_ir::bitslice::{BitSliceAot, BitSliceSimulator};
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program, SimdCpuSimulator};
use rrtl_sv_frontend::import_sv;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lanes: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20000);

    let src = std::fs::read_to_string("bench/sv/crc32.sv").expect("read crc32.sv");
    let imported = import_sv(&src, Some("crc32")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "crc32").expect("lower");
    let machine = lower_to_machine_program(&program);
    let h = |n: &str| compiled.find_module("crc32").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
    let idx = |n: &str| program.signal_index(h(n)).unwrap();

    let mut aot = BitSliceAot::compile_lanes(&machine, lanes).expect("bit-slice AOT applies");
    let mut interp = BitSliceSimulator::new(&machine, lanes).expect("bit-slice interp");
    let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
    cpu.set_signal(h("clk"), &vec![1u128; lanes]).unwrap();

    let inval = |seed: u64, lane: usize, m: u128| ((seed.wrapping_mul(2654435761).wrapping_add(lane as u64)) as u128) & m;
    let drive = |aot: &mut BitSliceAot, interp: &mut BitSliceSimulator, cpu: &mut SimdCpuSimulator, c: u64| {
        for (name, w) in [("rst", 1u128), ("din", 0xff)] {
            let vals: Vec<u128> = (0..lanes)
                .map(|l| if name == "rst" { (c < 1) as u128 } else { inval(c + name.len() as u64, l, w) })
                .collect();
            for (l, v) in vals.iter().enumerate() {
                aot.set_signal(idx(name), l, *v);
                interp.set_signal(idx(name), l, *v);
            }
            cpu.set_signal(h(name), &vals).unwrap();
        }
    };

    // bit-exact over several cycles on every lane (crc output).
    let mut mism = 0usize;
    for c in 0..40 {
        drive(&mut aot, &mut interp, &mut cpu, c as u64);
        aot.tick();
        interp.tick();
        cpu.tick().unwrap();
        let cv = cpu.get_signal(h("crc")).unwrap();
        for l in 0..lanes {
            if aot.get_signal(idx("crc"), l) != cv[l] {
                mism += 1;
            }
            if interp.get_signal(idx("crc"), l) != cv[l] {
                mism += 1;
            }
        }
    }
    println!("bit-slice AOT: crc32 (32-bit datapath, 33% 1-bit ops, no mul), {lanes} lanes");
    println!("  bit-exact vs SIMD CPU (crc × {lanes} lanes × 40 cyc, AOT+interp): {}", if mism == 0 { "YES" } else { "NO" });

    let mlc = |s: f64| (lanes * steps) as f64 / s / 1e6;
    let t = Instant::now();
    aot.tick_many(steps);
    let aot_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    interp.tick_many(steps / 20); // interp is slow; scale down then normalize
    let interp_s = t.elapsed().as_secs_f64() * 20.0;
    let t = Instant::now();
    cpu.tick_many(steps).unwrap();
    let cpu_s = t.elapsed().as_secs_f64();

    println!("  bit-slice AOT   : {:.1} M-lane-cyc/s", mlc(aot_s));
    println!("  bit-slice interp: {:.1} M-lane-cyc/s  (AOT is {:.1}x)", mlc(interp_s), interp_s / aot_s);
    println!("  SIMD CPU interp : {:.1} M-lane-cyc/s  (AOT is {:.1}x)", mlc(cpu_s), cpu_s / aot_s);
    #[cfg(feature = "jit")]
    {
        use rrtl_sim_ir::jit::SimdJitSimulator;
        if let Ok(mut v) = SimdJitSimulator::compile_lanes(&machine, lanes) {
            for l in 0..lanes {
                v.set_signal(l, idx("din"), inval(1, l, 0xff) as u32);
            }
            for _ in 0..7 {
                v.tick_many(1);
            }
            let t = Instant::now();
            v.tick_many(steps);
            let v_s = t.elapsed().as_secs_f64();
            println!("  vector JIT      : {:.1} M-lane-cyc/s  (AOT is {:.2}x)", mlc(v_s), v_s / aot_s);
        } else {
            println!("  vector JIT      : (cannot compile this design)");
        }
    }

    // multicore AOT
    let t = Instant::now();
    aot.tick_many_parallel(steps);
    let par_s = t.elapsed().as_secs_f64();
    println!("  bit-slice AOT ×cores: {:.1} M-lane-cyc/s  ({:.1}x serial AOT)", mlc(par_s), aot_s / par_s);

    assert_eq!(mism, 0, "bit-slice diverged from SIMD CPU");
}
