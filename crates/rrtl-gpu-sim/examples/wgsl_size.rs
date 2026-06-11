//! WGSL shader size vs design size (no GPU compile) — quantifies how the
//! generated kernel grows, to explain GPU `construct` (shader-compile) hangs.

use rrtl_core::{compile, lit_u, uint, Design};
use rrtl_gpu_sim::{gpu_shader_stats, GpuBatchOptions};
use rrtl_sim_ir::lower_to_packed_program;

fn build_wide(width: usize, depth: usize) -> Design {
    let mut design = Design::new();
    let mut m = design.module("Wide");
    let clk = m.input("clk", uint(1));
    let din = m.input("din", uint(32));
    for lane in 0..width {
        let acc = m.reg(format!("acc{lane}"), uint(32));
        m.clock(acc, clk);
        let mut prev = acc;
        for stage in 0..depth {
            let w = m.wire(format!("w{lane}_{stage}"), uint(32));
            m.assign(w, prev * lit_u(0x9e37_79b9, 32) + din);
            prev = w;
        }
        m.next(acc, prev + acc);
        let o = m.output(format!("o{lane}"), uint(32));
        m.assign(o, acc);
    }
    design
}

fn main() {
    println!(
        "{:>5} {:>5} {:>8} {:>12} {:>12} {:>9}",
        "W", "D", "signals", "wgsl_KB", "wgsl_KB(reuse)", "tmp_slots"
    );
    for &(w, d) in &[(8usize, 2usize), (8, 4), (16, 4), (32, 4), (64, 8)] {
        let design = build_wide(w, d);
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Wide").unwrap();
        let base = gpu_shader_stats(&program, GpuBatchOptions::default()).unwrap();
        let reuse = gpu_shader_stats(
            &program,
            GpuBatchOptions {
                reuse_temporaries: true,
                ..GpuBatchOptions::default()
            },
        )
        .unwrap();
        println!(
            "{:>5} {:>5} {:>8} {:>12.1} {:>12.1} {:>9}",
            w,
            d,
            program.signals.len(),
            base.wgsl_bytes as f64 / 1024.0,
            reuse.wgsl_bytes as f64 / 1024.0,
            base.optimized_temp_slots,
        );
    }
}
