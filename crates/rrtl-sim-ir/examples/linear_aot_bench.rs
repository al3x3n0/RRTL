//! GF(2)-linear AOT: the CPU realization of the tensor-core specialization. A
//! linear register cone is `next = M·{inputs,regs} ⊕ c` over GF(2); transposing
//! the column-major matrix gives, per output bit, a flat XOR of input-bit planes.
//! Emitting that as bit-sliced `uint64_t` C (one word = 64 lanes, inner word loop
//! clang -O3 auto-vectorizes) is PURE XOR — no ripple-carry, no width dispatch,
//! no mux — so it should beat the gate-level bit-slice AOT, which must evaluate
//! the full unrolled CRC netlist. Same matrix feeds a 1-bit BMMA dense GEMM on
//! NVIDIA tensor cores; this is the portable XOR-GEMM that also helps SIMD/CPU.
//!
//! Validates bit-exact vs the SIMD CPU oracle and races the matrix AOT against
//! the gate-level bit-slice AOT on crc32.
//! Build: cargo run --release --features "aot jit" -p rrtl-sim-ir --example linear_aot_bench -- [lanes steps]
use std::collections::HashMap;
use std::fmt::Write as _;

use rrtl_sim_ir::bitslice::BitSliceAot;
use rrtl_sim_ir::{linearize, lower_to_machine_program, lower_to_packed_program, PackedOp, SimdCpuSimulator};
use rrtl_sv_frontend::import_sv;
use std::time::Instant;

struct Reg {
    dst: usize,
    width: u32,
    // per output bit: (input-plane indices to XOR, constant bit)
    rows: Vec<(Vec<usize>, bool)>,
    reset: Option<(usize, bool, Vec<bool>)>, // (rst plane, active_low, reset-value bits)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lanes: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20000);

    let src = std::fs::read_to_string("bench/sv/crc32.sv").expect("read crc32.sv");
    let imported = import_sv(&src, Some("crc32")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "crc32").expect("lower");
    let machine = lower_to_machine_program(&program);

    // 1. Collect linear register cones + their GF(2) forms.
    struct Raw {
        dst: usize,
        width: u32,
        form: linearize::LinearForm,
        reset: Option<rrtl_sim_ir::PackedReset>,
    }
    let mut raws: Vec<Raw> = Vec::new();
    for packet in &program.streams.tick_next {
        for op in &packet.ops {
            let PackedOp::CaptureReg { dst, next, reset } = op else { continue };
            assert!(linearize::is_linear(&program, next), "crc32 cone not linear?");
            raws.push(Raw {
                dst: *dst,
                width: next.ty.width,
                form: linearize::extract_linear_form(&program, next),
                reset: reset.clone(),
            });
        }
    }
    println!("crc32: {} linear register cones", raws.len());

    // 2. Bit-sliced plane layout for every tracked signal (regs + input leaves +
    //    reset signals). plane_base[sig] = first plane; total_planes = Σ widths.
    let width_of = |s: usize| program.signals[s].layout.width;
    let mut tracked: Vec<usize> = Vec::new();
    let push = |s: usize, tracked: &mut Vec<usize>| {
        if !tracked.contains(&s) {
            tracked.push(s);
        }
    };
    for r in &raws {
        push(r.dst, &mut tracked);
    }
    for r in &raws {
        for &(s, _) in &r.form.leaves {
            push(s, &mut tracked);
        }
        if let Some(rst) = &r.reset {
            push(rst.signal, &mut tracked);
        }
    }
    let mut plane_base: HashMap<usize, usize> = HashMap::new();
    let mut total_planes = 0usize;
    for &s in &tracked {
        plane_base.insert(s, total_planes);
        total_planes += width_of(s) as usize;
    }

    // 3. Transpose each form's column-major matrix into per-output-bit XOR rows.
    let regs: Vec<Reg> = raws
        .iter()
        .map(|r| {
            let w = r.width as usize;
            let mut rows: Vec<(Vec<usize>, bool)> = (0..w).map(|_| (Vec::new(), false)).collect();
            let mut b = 0usize; // global input-bit index into columns
            for &(sig, sw) in &r.form.leaves {
                for bit in 0..sw {
                    let col = r.form.columns[b];
                    for i in 0..w {
                        if (col >> i) & 1 != 0 {
                            rows[i].0.push(plane_base[&sig] + bit as usize);
                        }
                    }
                    b += 1;
                }
            }
            for i in 0..w {
                rows[i].1 = (r.form.constant >> i) & 1 != 0;
            }
            let reset = r.reset.as_ref().map(|rst| {
                let active_low = matches!(rst.polarity, rrtl_ir::ResetPolarity::ActiveLow);
                let rv: Vec<bool> = (0..w)
                    .map(|i| (rst.value.get(i / 32).copied().unwrap_or(0) >> (i % 32)) & 1 != 0)
                    .collect();
                (plane_base[&rst.signal], active_low, rv)
            });
            Reg { dst: r.dst, width: r.width, rows, reset }
        })
        .collect();

    // 4. Emit bit-sliced C: all next-states (locals) then all commits (feedback-safe).
    let mut c = String::from("typedef unsigned long long u64;\n");
    writeln!(c, "void tick_lin(u64* restrict st, long nw, long nc){{").ok();
    c.push_str(" for(long _c=0;_c<nc;_c++){\n  for(long k=0;k<nw;k++){\n");
    writeln!(c, "   u64* restrict s = st + k*{total_planes};").ok();
    for r in &regs {
        if let Some((rp, lo, _)) = &r.reset {
            let a = if *lo { format!("(~s[{rp}])") } else { format!("s[{rp}]") };
            writeln!(c, "   u64 ra{} = {a};", r.dst).ok();
        }
        for (i, (xs, cbit)) in r.rows.iter().enumerate() {
            let mut terms: Vec<String> = xs.iter().map(|p| format!("s[{p}]")).collect();
            if *cbit {
                terms.push("~0ull".into());
            }
            let lin = if terms.is_empty() { "0ull".into() } else { terms.join(" ^ ") };
            match &r.reset {
                Some((_, _, rv)) => {
                    let rf = if rv[i] { "~0ull" } else { "0ull" };
                    writeln!(c, "   u64 n{}_{i} = (ra{} & {rf}) | (~ra{} & ({lin}));", r.dst, r.dst, r.dst).ok();
                }
                None => {
                    writeln!(c, "   u64 n{}_{i} = {lin};", r.dst).ok();
                }
            }
        }
    }
    for r in &regs {
        for i in 0..r.width as usize {
            writeln!(c, "   s[{}] = n{}_{i};", plane_base[&r.dst] + i, r.dst).ok();
        }
    }
    c.push_str("  }\n }\n}\n");
    let nnz: usize = regs.iter().flat_map(|r| &r.rows).map(|(xs, _)| xs.len()).sum();
    println!("  matrix: {total_planes} planes, {nnz} word-XORs/cycle (vs the full gate netlist)");

    // 5. Compile + dlopen.
    let stamp = {
        let mut h = 0xcbf29ce484222325u64;
        for b in c.as_bytes() {
            h = (h ^ *b as u64).wrapping_mul(0x100000001b3);
        }
        format!("{h:x}")
    };
    let ext = if cfg!(target_os = "macos") { "dylib" } else { "so" };
    let dir = std::env::temp_dir();
    let cpath = dir.join(format!("rrtl_lin_{stamp}.c"));
    let libpath = dir.join(format!("librrtl_lin_{stamp}.{ext}"));
    std::fs::write(&cpath, &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "clang".into());
    let out = std::process::Command::new(&cc)
        .args(["-O3", "-shared", "-fPIC", "-o"])
        .arg(&libpath)
        .arg(&cpath)
        .output()
        .expect("clang");
    assert!(out.status.success(), "clang: {}", String::from_utf8_lossy(&out.stderr));
    let lib = unsafe { libloading::Library::new(&libpath) }.unwrap();
    let tick_lin: libloading::Symbol<extern "C" fn(*mut u64, i64, i64)> = unsafe { lib.get(b"tick_lin").unwrap() };

    // bit-sliced word-major state: st[k*total_planes + plane]
    let words = lanes.div_ceil(64);
    let mut state = vec![0u64; total_planes * words];
    let set = |state: &mut [u64], sig: usize, lane: usize, val: u128| {
        let (k, b) = (lane / 64, lane % 64);
        for p in 0..width_of(sig) as usize {
            let slot = k * total_planes + plane_base[&sig] + p;
            if (val >> p) & 1 == 1 {
                state[slot] |= 1u64 << b;
            } else {
                state[slot] &= !(1u64 << b);
            }
        }
    };
    let get = |state: &[u64], sig: usize, lane: usize| -> u128 {
        let (k, b) = (lane / 64, lane % 64);
        let mut v = 0u128;
        for p in 0..width_of(sig) as usize {
            if (state[k * total_planes + plane_base[&sig] + p] >> b) & 1 == 1 {
                v |= 1u128 << p;
            }
        }
        v
    };

    // Oracle + gate-level bit-slice AOT on the identical design. Three index
    // spaces: `pos` indexes program.signals (the linear cones' dst/leaf space),
    // `idx` is the machine signal index (BitSliceAot), `h` is the core handle (cpu).
    let h = |n: &str| compiled.find_module("crc32").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
    let idx = |n: &str| program.signal_index(h(n)).unwrap();
    let pos = |n: &str| program.signals.iter().position(|s| s.name == n || s.name.ends_with(&format!(".{n}"))).unwrap();
    let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
    cpu.set_signal(h("clk"), &vec![1u128; lanes]).unwrap();
    let mut gate = BitSliceAot::compile_lanes(&machine, lanes).expect("bit-slice AOT");

    // `crc` is both the register and the output port (two program.signals
    // entries); read it from the register's own dst, which is the tracked plane.
    let crc_dst = regs
        .iter()
        .map(|r| r.dst)
        .find(|&d| program.signals[d].name.rsplit('.').next().unwrap().starts_with("crc"))
        .expect("crc register cone");
    let inval = |seed: u64, lane: usize, m: u128| ((seed.wrapping_mul(2654435761).wrapping_add(lane as u64)) as u128) & m;
    let mut mism = 0usize;
    for cyc in 0..40u64 {
        for (name, m) in [("rst", 1u128), ("din", 0xff)] {
            let vals: Vec<u128> = (0..lanes)
                .map(|l| if name == "rst" { (cyc < 1) as u128 } else { inval(cyc + name.len() as u64, l, m) })
                .collect();
            for (l, v) in vals.iter().enumerate() {
                set(&mut state, pos(name), l, *v);
                gate.set_signal(idx(name), l, *v);
            }
            cpu.set_signal(h(name), &vals).unwrap();
        }
        tick_lin(state.as_mut_ptr(), words as i64, 1);
        gate.tick();
        cpu.tick().unwrap();
        let cv = cpu.get_signal(h("crc")).unwrap();
        for l in 0..lanes {
            if get(&state, crc_dst, l) != cv[l] {
                mism += 1;
            }
            if gate.get_signal(idx("crc"), l) != cv[l] {
                mism += 1;
            }
        }
    }
    println!("  bit-exact vs SIMD CPU (crc × {lanes} lanes × 40 cyc, linear-AOT + gate-AOT): {}", if mism == 0 { "YES" } else { "NO" });
    assert_eq!(mism, 0, "linear AOT diverged");

    // 6. Throughput: linear-matrix AOT vs gate-level bit-slice AOT.
    let mlc = |s: f64| (lanes * steps) as f64 / s / 1e6;
    let t = Instant::now();
    tick_lin(state.as_mut_ptr(), words as i64, steps as i64);
    let lin_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    gate.tick_many(steps);
    let gate_s = t.elapsed().as_secs_f64();
    println!("  linear-matrix AOT : {:.1} M-lane-cyc/s", mlc(lin_s));
    println!("  gate bit-slice AOT: {:.1} M-lane-cyc/s  (linear is {:.2}x)", mlc(gate_s), gate_s / lin_s);
    // Same-process comparison vs the incumbent best batch engine (vector JIT),
    // which (like the gate bit-slice) evaluates the full gate netlist.
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
            println!("  vector JIT (gate) : {:.1} M-lane-cyc/s  (linear is {:.2}x)", mlc(v_s), v_s / lin_s);
        }
    }
}
