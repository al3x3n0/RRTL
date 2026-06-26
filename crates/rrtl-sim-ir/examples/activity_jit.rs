//! Activity-driven skipping on a partitioned JIT — the defining huge-design lever
//! (ESSENT/Khronos): most of a big SoC is idle every cycle, so an oblivious
//! simulator wastes the majority of its work re-evaluating unchanged logic.
//!
//! Building on the partitioned JIT (each register-cone is a separate engine), we
//! skip an entire partition's tick when its **activity inputs** — its own
//! registers plus its boundary inputs — are unchanged since its last evaluation.
//! A partition at a fixpoint with stable inputs produces no state change, so the
//! skip is exact. This is the coarse-cone granularity §4g.1 found the JIT skip
//! actually pays at (the per-partition signature check is amortized over the
//! whole cone's work).
//!
//! The design gives partitions a spatial activity *gradient*: a counter drives
//! per-PE registered inputs that change at geometric rates (PE i every 2^i
//! cycles), so low-index PEs are hot and high-index PEs are almost always idle.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example activity_jit -- [K]
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn gen_sv(k: usize) -> String {
    // Heavy PE: a deep dependent multiply chain, so a partition's tick is real
    // work (skipping it saves far more than the activity check costs).
    const D: usize = 20;
    let consts = ["9e3779b1", "85ebca77", "c2b2ae35", "27d4eb2f", "165667b1", "ff51afd7", "d6e8feb8", "a0761d65"];
    let mut s = String::new();
    s.push_str("module pe #(parameter [31:0] SALT = 0) (input clk, input [31:0] din, output reg [31:0] dout);\n");
    s.push_str("  reg [31:0] ");
    s.push_str(&(0..D).map(|i| format!("r{i}")).collect::<Vec<_>>().join(","));
    s.push_str(";\n  always @(posedge clk) begin\n");
    s.push_str(&format!("    r0 <= (din ^ SALT) * 32'h{} + 32'd1;\n", consts[0]));
    for i in 1..D {
        let c = consts[i % consts.len()];
        s.push_str(&format!("    r{i} <= (r{} ^ (r{}<<{}) ^ (r{}>>{})) * 32'h{c} + r{};\n",
            i - 1, i - 1, 3 + (i % 7), i - 1, 2 + (i % 5), i - 1));
    }
    s.push_str(&format!("    dout <= r{} + r{};\n", D - 1, D - 2));
    s.push_str("  end\nendmodule\n\n");
    s.push_str("module top(input clk, input rst, output reg [31:0] dout);\n");
    s.push_str("  reg [31:0] counter;\n");
    for i in 0..k {
        s.push_str(&format!("  reg [31:0] inp{i};\n  wire [31:0] o{i};\n"));
    }
    s.push_str("  always @(posedge clk) begin\n");
    s.push_str("    if (rst) counter <= 32'd0; else counter <= counter + 32'd1;\n");
    for i in 0..k {
        // inp{i} changes every 2^i cycles -> PE i is active ~1/2^i of the time.
        s.push_str(&format!("    inp{i} <= counter >> {i};\n"));
    }
    s.push_str("    dout <= ");
    s.push_str(&(0..k).map(|i| format!("o{i}")).collect::<Vec<_>>().join(" ^ "));
    s.push_str(";\n  end\n");
    for i in 0..k {
        s.push_str(&format!("  pe #(.SALT(32'd{i})) pe{i}(.clk(clk), .din(inp{i}), .dout(o{i}));\n"));
    }
    s.push_str("endmodule\n");
    s
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::jit::JitSimulator;
    use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program, slice_packed_program, PackedEffect, PackedSliceGroup};
    use rrtl_sv_frontend::import_sv;
    use std::collections::HashSet;
    use std::time::Instant;

    let k: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(24);
    let src = gen_sv(k);
    let imported = import_sv(&src, Some("top")).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, "top").expect("lower packed");
    let full_machine = lower_to_machine_program(&program);

    let gidx = |name: &str| {
        let h = compiled.find_module("top").unwrap().signals.iter().find(|s| s.name == name).unwrap().handle;
        program.signal_index(h).unwrap()
    };
    let (clk_g, rst_g, dout_g) = (gidx("clk"), gidx("rst"), gidx("dout"));
    let inputs = [clk_g, rst_g];

    // partition: top glue + one cone per PE
    let mut groups = vec![PackedSliceGroup { id: "top".into(), owned_paths: vec!["top".into()] }];
    for i in 0..k {
        groups.push(PackedSliceGroup { id: format!("pe{i}"), owned_paths: vec![format!("top.pe{i}")] });
    }
    let slices = slice_packed_program(&program, &groups).expect("slice");
    let slice_machines: Vec<_> = slices.iter().map(|s| lower_to_machine_program(&s.program)).collect();

    let local_of = |s: usize, g: usize| slices[s].signal_origin.iter().position(|&x| x == g);
    let owner = |g: usize| -> (usize, usize) {
        for (s, sl) in slices.iter().enumerate() {
            let bset: HashSet<usize> = sl.boundary_inputs.iter().copied().collect();
            if let Some(l) = sl.signal_origin.iter().position(|&x| x == g) {
                if !bset.contains(&l) {
                    return (s, l);
                }
            }
        }
        panic!("no owner for {g}");
    };
    let input_sets: Vec<Vec<(usize, usize)>> = (0..slices.len())
        .map(|s| inputs.iter().filter_map(|&g| local_of(s, g).map(|l| (g, l))).collect())
        .collect();
    let routes: Vec<Vec<(usize, usize, usize)>> = slices.iter().map(|sl| {
        sl.boundary_inputs.iter().filter_map(|&bl| {
            let g = sl.signal_origin[bl];
            if inputs.contains(&g) { return None; }
            let (os, ol) = owner(g);
            Some((os, ol, bl))
        }).collect()
    }).collect();
    let (dout_s, dout_l) = owner(dout_g);

    // per-partition registers (locals) and a settling bound (register count = a
    // safe upper bound on the cycles of stable input to reach a feed-forward fixpoint).
    let reg_locals: Vec<Vec<usize>> = slice_machines.iter().map(|m| {
        let mut v: Vec<usize> = m.streams.tick_next.packets.iter()
            .flat_map(|p| &p.effects)
            .filter_map(|e| if let PackedEffect::CaptureReg { dst, .. } = e { Some(*dst) } else { None })
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }).collect();
    let settle: Vec<u64> = reg_locals.iter().map(|r| r.len() as u64).collect();

    let cycles = 30_000u64;
    let stim = |c: u64| (1u64, (c < 4) as u64); // (clk, rst): reset first 4 cycles

    // Skip a partition's tick once its external drivers (captured FROM the exchange,
    // no extra reads) have been stable for longer than its settling bound — i.e. it
    // has provably reached a fixpoint. This is the cheapest possible activity test:
    // a per-partition counter + an in-hand vector compare, NO get_signal of state.
    let run = |jits: &mut [JitSimulator], skip: bool| -> (Vec<u64>, f64, u64) {
        let n = jits.len();
        let mut stable = vec![0u64; n];
        // skip-eligibility is a static property (feed-forward registers settle to a
        // fixpoint; self-driving state like a counter never does) — verified ONCE
        // per partition with a pre/post register check, then cached.
        let mut eligible: Vec<Option<bool>> = vec![None; n];
        let mut ext_snap: Vec<Vec<u64>> = vec![Vec::new(); n];
        let mut ext_now: Vec<Vec<u64>> = vec![Vec::new(); n];
        let mut xfer: Vec<(usize, usize, i64, i64)> = Vec::new();
        let mut trace = vec![0u64; cycles as usize];
        let mut ticks = 0u64;
        let t = Instant::now();
        for c in 0..cycles {
            let (clk, rst) = stim(c);
            // inputs (raw low word, hi=0 — clk/rst are 1-bit)
            for s in 0..n {
                ext_now[s].clear();
                let st = jits[s].state_words_mut();
                for &(g, l) in &input_sets[s] {
                    let v = if g == clk_g { clk } else { rst };
                    st[l * 2] = v as i64;
                    st[l * 2 + 1] = 0;
                    ext_now[s].push(v);
                }
            }
            // boundary exchange as raw 16-byte word copies. Two phases (gather then
            // scatter) so producer reads never alias consumer writes. ~1-2 ns/word
            // vs ~20 ns for get_signal/set_signal — the exchange is no longer the floor.
            xfer.clear();
            for s in 0..n {
                for &(os, ol, bl) in &routes[s] {
                    let w = jits[os].state_words();
                    let (lo, hi) = (w[ol * 2], w[ol * 2 + 1]);
                    xfer.push((s, bl, lo, hi));
                    ext_now[s].push(lo as u64);
                }
            }
            for &(s, bl, lo, hi) in &xfer {
                let st = jits[s].state_words_mut();
                st[bl * 2] = lo;
                st[bl * 2 + 1] = hi;
            }
            for s in 0..n {
                if skip {
                    if ext_now[s] == ext_snap[s] {
                        stable[s] += 1;
                    } else {
                        stable[s] = 0;
                        ext_snap[s].clone_from(&ext_now[s]);
                    }
                    if stable[s] > settle[s] {
                        match eligible[s] {
                            Some(true) => continue, // known feed-forward → at fixpoint → skip
                            Some(false) => {}        // self-driving (counter) → always tick
                            None => {
                                // verify once: did a tick with stable drivers change state?
                                let old: Vec<u64> = reg_locals[s].iter().map(|&l| jits[s].get_signal(l)).collect();
                                jits[s].tick();
                                let new: Vec<u64> = reg_locals[s].iter().map(|&l| jits[s].get_signal(l)).collect();
                                eligible[s] = Some(old == new);
                                ticks += 1;
                                continue;
                            }
                        }
                    }
                }
                jits[s].tick();
                ticks += 1;
            }
            trace[c as usize] = jits[dout_s].get_signal(dout_l);
        }
        (trace, t.elapsed().as_secs_f64(), ticks)
    };

    let mut mono = JitSimulator::compile(&full_machine).unwrap();
    let mut ref_trace = vec![0u64; cycles as usize];
    for c in 0..cycles {
        let (clk, rst) = stim(c);
        mono.set_signal(clk_g, clk);
        mono.set_signal(rst_g, rst);
        mono.tick();
        ref_trace[c as usize] = mono.get_signal(dout_g);
    }

    let mut jp: Vec<JitSimulator> = slice_machines.iter().map(|m| JitSimulator::compile(m).unwrap()).collect();
    let mut js: Vec<JitSimulator> = slice_machines.iter().map(|m| JitSimulator::compile(m).unwrap()).collect();
    let (t_pl, s_pl, ticks_pl) = run(&mut jp, false);
    let (t_sk, s_sk, ticks_sk) = run(&mut js, true);
    let exact = t_pl == ref_trace && t_sk == ref_trace;
    let total = cycles * (slices.len() as u64);

    println!("activity-driven skipping on a partitioned JIT — top + {k} PEs (geometric activity gradient)");
    println!("{} partitions, {cycles} cycles\n", slices.len());
    println!("  no-skip   : {:.2} Mcyc/s | {ticks_pl} partition-ticks", cycles as f64 / s_pl / 1e6);
    println!("  activity  : {:.2} Mcyc/s | {ticks_sk} partition-ticks ({:.0}% SKIPPED)",
        cycles as f64 / s_sk / 1e6, 100.0 * (1.0 - ticks_sk as f64 / total as f64));
    println!("  speedup   : {:.2}x | bit-exact: {}", s_pl / s_sk, if exact { "YES" } else { "NO" });
}
