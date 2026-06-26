//! A tiered JIT for RTL simulation with lightweight hot-spot detection and
//! value-*speculation* beyond exact constants.
//!
//! Tier 0 is the generic JIT; it runs while a **lightweight sampled profiler**
//! watches only the *narrow* control/state registers (the ones whose value gates
//! logic — wide datapaths are skipped, which bounds the profiler's cost and
//! memory). When a design has run long enough to be "hot" AND a register's value
//! is strongly *dominant* (not necessarily constant), the controller speculates
//! that value, freezes+re-JITs a specialized Tier 1, and runs it under a per-tick
//! guard. On a guard miss it deoptimizes to Tier 0; specialized versions are
//! **cached** keyed by the speculated values, so re-promotion (when the dominant
//! value recurs) is a cheap engine swap, not a recompile.
//!
//! Why "beyond quasi-constants": a register that blips off its dominant value
//! occasionally is NEVER 100%-stable in any window, so the freeze policy
//! (§4g.6) never fires — but it is still, say, 99% dominant, so bias speculation
//! promotes and wins, deopting only on the rare blips.
//!
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example tiered_jit
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::jit::JitSimulator;
    use rrtl_sim_ir::specialize::freeze_signals_program;
    use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program, PackedEffect, PackedMachineProgram};
    use rrtl_sv_frontend::import_sv;
    use std::collections::{HashMap, HashSet};
    use std::time::Instant;

    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv");
    let src = std::fs::read_to_string(format!("{base}/cfgdsp.sv")).unwrap();
    let imported = import_sv(&src, Some("cfgdsp")).unwrap();
    let compiled = rrtl_core::compile(&imported.design).unwrap();
    let program = lower_to_packed_program(&compiled, "cfgdsp").unwrap();
    let machine = lower_to_machine_program(&program);
    let idx = |n: &str| program.signals.iter().position(|s| s.name.ends_with(n)).unwrap();
    let (clk, rst, cfg_we, cfg_in, a, b, acc) =
        (idx(".clk"), idx(".rst"), idx(".cfg_we"), idx(".cfg_in"), idx(".a"), idx(".b"), idx(".acc"));
    let nsig = program.signals.len();

    // Candidate registers: CaptureReg destinations that are NARROW (≤16 bits) —
    // the cheap-to-profile control/state regs; wide datapaths (acc) are excluded.
    let cand: Vec<usize> = {
        let mut regs = Vec::new();
        for p in &machine.streams.tick_next.packets {
            for e in &p.effects {
                if let PackedEffect::CaptureReg { dst, .. } = e {
                    regs.push(*dst);
                }
            }
        }
        regs.sort_unstable();
        regs.dedup();
        regs.retain(|&r| program.signals[r].layout.width <= 16);
        regs
    };

    // --------- the tiered controller ---------
    struct Cached {
        engine: JitSimulator,
        spec: Vec<(usize, u128)>, // sorted (signal, speculated value)
    }
    enum Active {
        Generic,
        Tier1(usize),
    }
    struct Tiered {
        machine: PackedMachineProgram,
        generic: JitSimulator,
        cache: Vec<Cached>,
        rejected: HashSet<Vec<(usize, u128)>>, // specs that yielded no pruning
        active: Active,
        cand: Vec<usize>,
        nsig: usize,
        // lightweight windowed profiler — runs only during a brief DISCOVERY
        // window (until the first useful specialization is interned, or we give
        // up); steady state is the cheap cache-match path, so the expensive
        // sampling/dominant-scan doesn't run every cycle for the whole sim.
        profiling: bool,
        attempts: u64,
        max_attempts: u64,
        scratch: Vec<(usize, u128)>,
        hist: Vec<HashMap<u128, u64>>,
        samples: u64,
        sample_period: u64,
        max_distinct: usize,
        hotness: u64,
        hot_threshold: u64,
        bias: f64,
        // re-promotion hysteresis (consecutive cycles current state matches a cached spec)
        pending: Option<usize>,
        dwell: u64,
        repromote_dwell: u64,
        // stats
        fast: u64,
        slow: u64,
        compiles: u64,
        promotions: u64,
        deopts: u64,
    }
    fn snapshot(e: &mut JitSimulator, nsig: usize) -> Vec<u128> {
        (0..nsig).map(|i| e.get_signal_u128(i)).collect()
    }
    fn restore(e: &mut JitSimulator, s: &[u128]) {
        for (i, &v) in s.iter().enumerate() {
            e.set_signal_u128(i, v);
        }
    }
    impl Tiered {
        /// Fill `self.scratch` with the dominant value per candidate whose dominant
        /// fraction ≥ bias (allocation-free: reuses the scratch buffer).
        fn refresh_dominant(&mut self) {
            self.scratch.clear();
            for (k, &c) in self.cand.iter().enumerate() {
                if let Some((&val, &cnt)) = self.hist[k].iter().max_by_key(|(_, n)| **n) {
                    if cnt as f64 >= self.bias * self.samples as f64 {
                        self.scratch.push((c, val));
                    }
                }
            }
            self.scratch.sort_unstable();
        }
        /// Get or build the cached specialized engine for `spec`; None if useless.
        fn intern(&mut self, spec: &[(usize, u128)]) -> Option<usize> {
            if let Some(i) = self.cache.iter().position(|c| c.spec == spec) {
                return Some(i); // already compiled
            }
            if self.rejected.contains(spec) {
                return None;
            }
            let frozen: HashMap<usize, u128> = spec.iter().copied().collect();
            let (prog, fstats) = freeze_signals_program(&self.machine, &frozen);
            if fstats.specialize.instrs_removed() == 0 {
                self.rejected.insert(spec.to_vec());
                return None;
            }
            let engine = JitSimulator::compile(&prog).unwrap();
            self.compiles += 1;
            self.cache.push(Cached { engine, spec: spec.to_vec() });
            Some(self.cache.len() - 1)
        }
        /// Burst-driven tiered execution: run the *hot* tier in a tight loop until
        /// its guard trips (amortizing all dispatch), then handle the transition.
        /// This is the key to net speedup — a per-cycle `match`/closure dispatch
        /// would cost as much as the cheap tick it guards.
        fn run<F: Fn(&mut JitSimulator, u64)>(&mut self, total: u64, apply: &F, out: usize, trace: &mut [u64]) {
            let mut c = 0u64;
            while c < total {
                match self.active {
                    Active::Tier1(i) => {
                        // tight hot loop: tick + guard, nothing else
                        let mut nfast = 0u64;
                        let mut deopt = false;
                        while c < total {
                            let cc = &mut self.cache[i];
                            apply(&mut cc.engine, c);
                            cc.engine.tick();
                            trace[c as usize] = cc.engine.get_signal(out);
                            c += 1;
                            nfast += 1;
                            if !cc.spec.iter().all(|&(si, sv)| cc.engine.get_signal_u128(si) == sv) {
                                deopt = true;
                                break;
                            }
                        }
                        self.fast += nfast;
                        if deopt {
                            let s = snapshot(&mut self.cache[i].engine, self.nsig);
                            restore(&mut self.generic, &s);
                            self.active = Active::Generic;
                            self.deopts += 1;
                            self.pending = None;
                            self.dwell = 0;
                        }
                    }
                    Active::Generic => {
                        while c < total {
                            apply(&mut self.generic, c);
                            self.generic.tick();
                            trace[c as usize] = self.generic.get_signal(out);
                            c += 1;
                            self.slow += 1;
                            self.hotness += 1;
                            // discovery (rare window; coprime sample period avoids aliasing)
                            if self.profiling && c % self.sample_period == 0 {
                                for (k, &cd) in self.cand.iter().enumerate() {
                                    let v = self.generic.get_signal_u128(cd);
                                    let h = &mut self.hist[k];
                                    if h.len() < self.max_distinct || h.contains_key(&v) {
                                        *h.entry(v).or_insert(0) += 1;
                                    }
                                }
                                self.samples += 1;
                                if self.hotness >= self.hot_threshold && self.samples >= 8 {
                                    self.attempts += 1;
                                    self.refresh_dominant();
                                    if !self.scratch.is_empty() {
                                        let spec = self.scratch.clone();
                                        if self.intern(&spec).is_some() {
                                            self.profiling = false;
                                        }
                                    }
                                    if self.attempts >= self.max_attempts {
                                        self.profiling = false;
                                    }
                                }
                            }
                            // cheap cache-match re-promotion (dwell-gated)
                            if !self.cache.is_empty() {
                                let mut hit = None;
                                for (ci, cc) in self.cache.iter().enumerate() {
                                    if cc.spec.iter().all(|&(si, sv)| self.generic.get_signal_u128(si) == sv) {
                                        hit = Some(ci);
                                        break;
                                    }
                                }
                                match hit {
                                    Some(ci) if self.pending == Some(ci) => self.dwell += 1,
                                    Some(ci) => {
                                        self.pending = Some(ci);
                                        self.dwell = 1;
                                    }
                                    None => {
                                        self.pending = None;
                                        self.dwell = 0;
                                    }
                                }
                                if let Some(ci) = hit {
                                    if self.dwell >= self.repromote_dwell {
                                        let s = snapshot(&mut self.generic, self.nsig);
                                        restore(&mut self.cache[ci].engine, &s);
                                        self.active = Active::Tier1(ci);
                                        self.promotions += 1;
                                        self.pending = None;
                                        self.dwell = 0;
                                        break; // back to the hot loop
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let total = 4_000_000u64;
    let mk = |bias: f64| Tiered {
        machine: machine.clone(),
        generic: JitSimulator::compile(&machine).unwrap(),
        cache: Vec::new(),
        rejected: HashSet::new(),
        active: Active::Generic,
        cand: cand.clone(),
        nsig,
        profiling: true,
        attempts: 0,
        max_attempts: 64,
        scratch: Vec::new(),
        hist: vec![HashMap::new(); cand.len()],
        samples: 0,
        sample_period: 13, // prime — coprime to the blip period to avoid aliasing
        max_distinct: 32,
        hotness: 0,
        hot_threshold: 2000,
        bias,
        pending: None,
        dwell: 0,
        repromote_dwell: 4,
        fast: 0,
        slow: 0,
        compiles: 0,
        promotions: 0,
        deopts: 0,
    };

    // One scenario at a given blip period: build the stimulus, a reference trace,
    // and run generic / bias-0.90 / freeze-1.00 interleaved best-of-3.
    let scenario = |period: u64| {
        // blip stimulus: mode := 1 at boot, a 2-cycle excursion to 2 every `period`.
        let apply = move |e: &mut JitSimulator, cycle: u64| {
            e.set_signal(a, cycle.wrapping_mul(2_654_435_761));
            e.set_signal(b, cycle.wrapping_mul(40_503).wrapping_add(7));
            match cycle {
                0 => {
                    e.set_signal(clk, 1);
                    e.set_signal(rst, 1);
                }
                1 => {
                    e.set_signal(rst, 0);
                    e.set_signal(cfg_we, 1);
                    e.set_signal(cfg_in, 1);
                }
                2 => e.set_signal(cfg_we, 0),
                _ => {
                    let ph = cycle % period;
                    if ph == 0 {
                        e.set_signal(cfg_we, 1);
                        e.set_signal(cfg_in, 2);
                    } else if ph == 1 {
                        e.set_signal(cfg_in, 1);
                    } else if ph == 2 {
                        e.set_signal(cfg_we, 0);
                    }
                }
            }
        };
        let mut gref = JitSimulator::compile(&machine).unwrap();
        let mut ref_trace = vec![0u64; total as usize];
        for c in 0..total {
            apply(&mut gref, c);
            gref.tick();
            ref_trace[c as usize] = gref.get_signal(acc);
        }
        let run_tiered = |mut ctl: Tiered| -> (Tiered, f64, usize) {
            let mut trace = vec![0u64; total as usize];
            let t = Instant::now();
            ctl.run(total, &apply, acc, &mut trace);
            let secs = t.elapsed().as_secs_f64();
            let mism = trace.iter().zip(&ref_trace).filter(|(x, y)| x != y).count();
            (ctl, secs, mism)
        };
        let run_generic = || -> f64 {
            let mut g = JitSimulator::compile(&machine).unwrap();
            let t = Instant::now();
            for c in 0..total {
                apply(&mut g, c);
                g.tick();
                std::hint::black_box(g.get_signal(acc));
            }
            t.elapsed().as_secs_f64()
        };
        let (mut g_best, mut b_best, mut f_best) = (f64::MAX, f64::MAX, f64::MAX);
        let (mut bs, mut fs) = (None, None);
        for _ in 0..3 {
            g_best = g_best.min(run_generic());
            let (bb, t, m) = run_tiered(mk(0.90));
            b_best = b_best.min(t);
            bs = Some((bb, m));
            let (ff, t, m) = run_tiered(mk(1.0));
            f_best = f_best.min(t);
            fs = Some((ff, m));
        }
        let (bb, b_mis) = bs.unwrap();
        let (ff, f_mis) = fs.unwrap();
        let mc = |s: f64| total as f64 / s / 1e6;
        let bpct = 100.0 * bb.fast as f64 / (bb.fast + bb.slow) as f64;
        let fpct = 100.0 * ff.fast as f64 / (ff.fast + ff.slow) as f64;
        println!("--- blip every {period} cycles (mode {:.1}% dominant-1) ---", 100.0 * (period - 2) as f64 / period as f64);
        println!("  generic           : {:.1} Mcyc/s", mc(g_best));
        println!("  bias 0.90  : {:>5.1}% fast | {} compile / {} promotions / {} deopts | {:.1} Mcyc/s ({:.2}x) | exact {}",
            bpct, bb.compiles, bb.promotions, bb.deopts, mc(b_best), g_best / b_best, if b_mis == 0 { "Y" } else { "N" });
        println!("  freeze 1.0 : {:>5.1}% fast | {} compile / {} promotions / {} deopts | {:.1} Mcyc/s ({:.2}x) | exact {}",
            fpct, ff.compiles, ff.promotions, ff.deopts, mc(f_best), g_best / f_best, if f_mis == 0 { "Y" } else { "N" });
        (b_mis == 0 && f_mis == 0, bpct, fpct)
    };

    println!("cfgdsp tiered JIT — lightweight sampled profiling + value speculation");
    println!("{} narrow candidate reg(s), {total} cycles, best-of-3 interleaved\n", cand.len());

    // Scenario A — rare blips: bias promotes with few deopts (net throughput).
    let (ok_a, _, _) = scenario(4096);
    println!();
    // Scenario B — frequent blips: the exact-constant policy can NEVER find a
    // clean discovery window, so it gives up (0% fast); bias still promotes.
    let (ok_b, bpct_b, fpct_b) = scenario(64);

    println!("\n  => beyond quasi-constants: bias speculation keeps a dominant-but-not-constant register");
    println!("     in the fast tier ({:.0}% at 64-cycle blips) where the exact-freeze policy gives up ({:.0}%).",
        bpct_b, fpct_b);
    println!("     Cached specializations make re-promotion free; guard keeps it bit-exact.");
    println!("     Net wall-clock gain is gated by per-tick compute vs per-cycle I/O (see notes). {}",
        if ok_a && ok_b { "All bit-exact." } else { "FAIL" });
}
