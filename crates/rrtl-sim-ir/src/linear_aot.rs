//! AOT-compiled GF(2)-linear backend — the portable XOR-GEMM realization of the
//! tensor-core (binary-matrix) specialization. A linear register cone is
//! `next = M·{inputs,regs} ⊕ c` over GF(2); transposing the column-major matrix
//! gives, per output bit, a flat XOR of input-bit planes. We emit that as
//! bit-sliced `uint64_t` C (one word = 64 lanes, inner word loop clang -O3
//! auto-vectorizes) — pure XOR, no ripple-carry / width dispatch / mux, so it
//! beats evaluating the equivalent gate netlist. The same matrix is one 1-bit
//! BMMA dense GEMM on NVIDIA tensor cores.
//!
//! Scope: the *linear register cones* of a `PackedProgram` (detected via
//! [`crate::linearize`]). Non-linear cones are left to a general backend — see
//! the `hybrid_sim` example for offloading the linear part of a mixed design.

use std::collections::HashMap;
use std::fmt::Write as _;

use rrtl_ir::{Diagnostic, ErrorReport, ResetPolarity};

use crate::{linearize, PackedOp, PackedProgram, PackedReset};

fn lin_err(msg: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![Diagnostic::new("E_LINEAR_AOT", msg)])
}

struct Reg {
    dst: usize,
    width: u32,
    // per output bit: (input-plane indices to XOR, constant bit)
    rows: Vec<(Vec<usize>, bool)>,
    reset: Option<(usize, bool, Vec<bool>)>, // (rst plane, active_low, reset-value bits)
}

/// AOT-compiled simulator for a design's GF(2)-linear register cones. State is
/// bit-sliced word-major (`st[k*total_planes + plane]`, one word = 64 lanes).
pub struct LinearAot {
    _lib: libloading::Library,
    tick_fn: extern "C" fn(*mut u64, i64, i64),
    state: Vec<u64>,
    plane_base: HashMap<usize, usize>, // program.signals index -> first plane
    widths: HashMap<usize, u32>,
    total_planes: usize,
    words: usize,
    lanes: usize,
    linear_dsts: Vec<usize>,
    input_leaves: Vec<usize>,
}

impl LinearAot {
    /// Detect the linear register cones of `program`, extract their GF(2)
    /// matrices, and compile a bit-sliced XOR kernel. Errors if there are no
    /// linear cones (caller should use a general backend) or any cone is >128-bit.
    pub fn compile(program: &PackedProgram, lanes: usize) -> Result<Self, ErrorReport> {
        // 1. Collect linear register cones + GF(2) forms.
        struct Raw {
            dst: usize,
            width: u32,
            form: linearize::LinearForm,
            reset: Option<PackedReset>,
        }
        let mut raws: Vec<Raw> = Vec::new();
        for packet in &program.streams.tick_next {
            for op in &packet.ops {
                let PackedOp::CaptureReg { dst, next, reset } = op else { continue };
                if next.ty.width > 128 || !linearize::is_linear(program, next) {
                    continue;
                }
                raws.push(Raw {
                    dst: *dst,
                    width: next.ty.width,
                    form: linearize::extract_linear_form(program, next),
                    reset: reset.clone(),
                });
            }
        }
        if raws.is_empty() {
            return Err(lin_err("no GF(2)-linear register cones to offload"));
        }

        // 2. Bit-sliced plane layout: every tracked signal (regs + input leaves +
        //    reset signals) gets a contiguous block of `width` planes.
        let width_of = |s: usize| program.signals[s].layout.width;
        let mut tracked: Vec<usize> = Vec::new();
        let push = |s: usize, tracked: &mut Vec<usize>| {
            if !tracked.contains(&s) {
                tracked.push(s);
            }
        };
        let reg_set: std::collections::HashSet<usize> = raws.iter().map(|r| r.dst).collect();
        for r in &raws {
            push(r.dst, &mut tracked);
        }
        let mut input_leaves: Vec<usize> = Vec::new();
        for r in &raws {
            for &(s, _) in &r.form.leaves {
                push(s, &mut tracked);
                if !reg_set.contains(&s) && !input_leaves.contains(&s) {
                    input_leaves.push(s);
                }
            }
            if let Some(rst) = &r.reset {
                push(rst.signal, &mut tracked);
                if !input_leaves.contains(&rst.signal) {
                    input_leaves.push(rst.signal);
                }
            }
        }
        let mut plane_base: HashMap<usize, usize> = HashMap::new();
        let mut widths: HashMap<usize, u32> = HashMap::new();
        let mut total_planes = 0usize;
        for &s in &tracked {
            plane_base.insert(s, total_planes);
            widths.insert(s, width_of(s));
            total_planes += width_of(s) as usize;
        }

        // 3. Transpose each column-major matrix into per-output-bit XOR rows.
        let regs: Vec<Reg> = raws
            .iter()
            .map(|r| {
                let w = r.width as usize;
                let mut rows: Vec<(Vec<usize>, bool)> = (0..w).map(|_| (Vec::new(), false)).collect();
                let mut b = 0usize;
                for &(sig, sw) in &r.form.leaves {
                    for bit in 0..sw {
                        let col = r.form.columns[b];
                        for (i, row) in rows.iter_mut().enumerate() {
                            if (col >> i) & 1 != 0 {
                                row.0.push(plane_base[&sig] + bit as usize);
                            }
                        }
                        b += 1;
                    }
                }
                for (i, row) in rows.iter_mut().enumerate() {
                    row.1 = (r.form.constant >> i) & 1 != 0;
                }
                let reset = r.reset.as_ref().map(|rst| {
                    let active_low = matches!(rst.polarity, ResetPolarity::ActiveLow);
                    let rv: Vec<bool> = (0..w)
                        .map(|i| (rst.value.get(i / 32).copied().unwrap_or(0) >> (i % 32)) & 1 != 0)
                        .collect();
                    (plane_base[&rst.signal], active_low, rv)
                });
                Reg { dst: r.dst, width: r.width, rows, reset }
            })
            .collect();

        // 4. Emit C: all next-states as locals (feedback-safe), then all commits.
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

        // 5. Compile + dlopen.
        let stamp = {
            let mut h = 0xcbf29ce484222325u64;
            for byte in c.as_bytes() {
                h = (h ^ *byte as u64).wrapping_mul(0x100000001b3);
            }
            format!("{h:x}")
        };
        let ext = if cfg!(target_os = "macos") { "dylib" } else { "so" };
        let dir = std::env::temp_dir();
        let cpath = dir.join(format!("rrtl_linaot_{stamp}.c"));
        let libpath = dir.join(format!("librrtl_linaot_{stamp}.{ext}"));
        std::fs::write(&cpath, &c).map_err(|e| lin_err(format!("write C: {e}")))?;
        let cc = std::env::var("CC").unwrap_or_else(|_| "clang".into());
        let out = std::process::Command::new(&cc)
            .args(["-O3", "-shared", "-fPIC", "-o"])
            .arg(&libpath)
            .arg(&cpath)
            .output()
            .map_err(|e| lin_err(format!("spawn {cc}: {e}")))?;
        if !out.status.success() {
            return Err(lin_err(format!("{cc} -O3 failed: {}", String::from_utf8_lossy(&out.stderr))));
        }
        let lib = unsafe { libloading::Library::new(&libpath) }.map_err(|e| lin_err(format!("dlopen: {e}")))?;
        let tick_fn = unsafe {
            let sym: libloading::Symbol<extern "C" fn(*mut u64, i64, i64)> =
                lib.get(b"tick_lin").map_err(|e| lin_err(format!("sym tick_lin: {e}")))?;
            *sym
        };

        let words = lanes.max(1).div_ceil(64);
        let mut linear_dsts: Vec<usize> = raws.iter().map(|r| r.dst).collect();
        linear_dsts.sort_unstable();
        // Registers start at 0 (matching the scalar/SIMD oracle's zero-initialized
        // state); reset is applied when the caller drives the reset signal. Do NOT
        // pre-seed reset values — other cones read a register's pre-reset value, so
        // seeding would diverge the first cycle.
        Ok(Self {
            _lib: lib,
            tick_fn,
            state: vec![0u64; total_planes * words],
            plane_base,
            widths,
            total_planes,
            words,
            lanes: words * 64,
            linear_dsts,
            input_leaves,
        })
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }
    /// The register signals (program.signals indices) this backend computes.
    pub fn linear_signals(&self) -> &[usize] {
        &self.linear_dsts
    }
    /// The non-register leaves (program.signals indices) the caller must drive.
    pub fn input_leaves(&self) -> &[usize] {
        &self.input_leaves
    }

    pub fn set_signal(&mut self, sig: usize, lane: usize, value: u128) {
        let Some(&base) = self.plane_base.get(&sig) else { return };
        let (k, b) = (lane / 64, lane % 64);
        for p in 0..self.widths[&sig] as usize {
            let slot = k * self.total_planes + base + p;
            if (value >> p) & 1 == 1 {
                self.state[slot] |= 1u64 << b;
            } else {
                self.state[slot] &= !(1u64 << b);
            }
        }
    }
    pub fn get_signal(&self, sig: usize, lane: usize) -> u128 {
        let Some(&base) = self.plane_base.get(&sig) else { return 0 };
        let (k, b) = (lane / 64, lane % 64);
        let mut v = 0u128;
        for p in 0..self.widths[&sig] as usize {
            if (self.state[k * self.total_planes + base + p] >> b) & 1 == 1 {
                v |= 1u128 << p;
            }
        }
        v
    }
    pub fn tick(&mut self) {
        self.tick_many(1);
    }
    pub fn tick_many(&mut self, n: usize) {
        (self.tick_fn)(self.state.as_mut_ptr(), self.words as i64, n as i64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lower_to_packed_program, SimdCpuSimulator};
    use rrtl_sv_frontend::import_sv;

    // crc32 is 100% GF(2)-linear; the matrix AOT must match the SIMD CPU oracle.
    #[test]
    fn linear_aot_matches_simd_cpu_crc32() {
        let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv/crc32.sv")).unwrap();
        let imported = import_sv(&src, Some("crc32")).unwrap();
        let compiled = rrtl_core::compile(&imported.design).unwrap();
        let program = lower_to_packed_program(&compiled, "crc32").unwrap();
        let h = |n: &str| compiled.find_module("crc32").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
        let pos = |n: &str| program.signals.iter().position(|s| s.name == n || s.name.ends_with(&format!(".{n}"))).unwrap();
        let lanes = 100;

        let mut lin = LinearAot::compile(&program, lanes).unwrap();
        let mut cpu = SimdCpuSimulator::new(program.clone(), lanes).unwrap();
        cpu.set_signal(h("clk"), &vec![1u128; lanes]).unwrap();
        let crc_dst = *lin
            .linear_signals()
            .iter()
            .find(|&&d| program.signals[d].name.rsplit('.').next().unwrap().starts_with("crc"))
            .unwrap();

        for cyc in 0..30u64 {
            for (name, m) in [("rst", 1u128), ("din", 0xff)] {
                let vals: Vec<u128> = (0..lanes)
                    .map(|l| if name == "rst" { (cyc < 1) as u128 } else { (cyc.wrapping_mul(2654435761).wrapping_add(l as u64) as u128) & m })
                    .collect();
                for (l, v) in vals.iter().enumerate() {
                    lin.set_signal(pos(name), l, *v);
                }
                cpu.set_signal(h(name), &vals).unwrap();
            }
            lin.tick();
            cpu.tick().unwrap();
            let cv = cpu.get_signal(h("crc")).unwrap();
            for l in 0..lanes {
                assert_eq!(lin.get_signal(crc_dst, l), cv[l], "crc@lane{l} cyc{cyc}");
            }
        }
    }
}
