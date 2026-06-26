//! Throughput benchmark: how fast does the compiled GPU backend simulate crc32
//! in BATCH on the real M3 GPU, gate-tree vs BMMA/linear kernel, across a lane
//! sweep — and what's the batch moat vs one fast CPU core? Reports aggregate
//! lane·cycles/sec (the metric that matters for regression/fuzzing fleets).
//! Build: cargo run --release -p rrtl-sim-ir --example gpu_throughput
use std::ffi::{c_void, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::time::Instant;

use rrtl_sim_ir::gpu_codegen::{emit_kernel, emit_kernel_linear, Flavor};
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
use rrtl_sv_frontend::import_sv;

type Id = *mut c_void;
#[link(name = "OpenCL", kind = "framework")]
extern "C" {
    fn clGetPlatformIDs(n: u32, p: *mut Id, np: *mut u32) -> c_int;
    fn clGetDeviceIDs(plat: Id, ty: u64, n: u32, d: *mut Id, nd: *mut u32) -> c_int;
    fn clCreateContext(props: *const isize, n: u32, d: *const Id, cb: *const c_void, ud: *mut c_void, e: *mut c_int) -> Id;
    fn clCreateCommandQueue(ctx: Id, dev: Id, props: u64, e: *mut c_int) -> Id;
    fn clCreateProgramWithSource(ctx: Id, n: u32, s: *const *const c_char, l: *const usize, e: *mut c_int) -> Id;
    fn clBuildProgram(p: Id, n: u32, d: *const Id, opt: *const c_char, cb: *const c_void, ud: *mut c_void) -> c_int;
    fn clCreateKernel(p: Id, name: *const c_char, e: *mut c_int) -> Id;
    fn clCreateBuffer(ctx: Id, flags: u64, size: usize, host: *mut c_void, e: *mut c_int) -> Id;
    fn clEnqueueWriteBuffer(q: Id, b: Id, block: u32, off: usize, size: usize, p: *const c_void, n: u32, wl: *const Id, ev: *mut Id) -> c_int;
    fn clSetKernelArg(k: Id, i: u32, size: usize, val: *const c_void) -> c_int;
    fn clEnqueueNDRangeKernel(q: Id, k: Id, dim: u32, off: *const usize, g: *const usize, l: *const usize, n: u32, wl: *const Id, ev: *mut Id) -> c_int;
    fn clFinish(q: Id) -> c_int;
}
const ALL: u64 = 0xFFFF_FFFF;
const RW: u64 = 1;
const TRUE: u32 = 1;
fn ck(e: c_int, what: &str) {
    assert_eq!(e, 0, "OpenCL {what} failed: {e}");
}

struct Gpu {
    q: Id,
}
unsafe fn build(ctx: Id, dev: Id, src: &str) -> Id {
    let cs = CString::new(src).unwrap();
    let sp = cs.as_ptr();
    let mut e = 0;
    let prog = clCreateProgramWithSource(ctx, 1, &sp, ptr::null(), &mut e);
    ck(clBuildProgram(prog, 1, &dev, ptr::null(), ptr::null(), ptr::null_mut()), "build");
    let kn = CString::new("tick").unwrap();
    let k = clCreateKernel(prog, kn.as_ptr(), &mut e);
    ck(e, "kernel");
    k
}

/// Time `ncyc` cycles on `lanes` lanes (NDRange→Finish only); return seconds.
unsafe fn timed(g: &Gpu, kern: Id, buf: Id, lanes: usize, ncyc: i64) -> f64 {
    let nl = lanes as i64;
    ck(clSetKernelArg(kern, 0, 8, &buf as *const _ as *const c_void), "arg0");
    ck(clSetKernelArg(kern, 1, 8, &nl as *const _ as *const c_void), "arg1");
    ck(clSetKernelArg(kern, 2, 8, &ncyc as *const _ as *const c_void), "arg2");
    let gl = lanes;
    let t = Instant::now();
    ck(clEnqueueNDRangeKernel(g.q, kern, 1, ptr::null(), &gl, ptr::null(), 0, ptr::null(), ptr::null_mut()), "ndrange");
    ck(clFinish(g.q), "finish");
    t.elapsed().as_secs_f64()
}

fn cpu_crc32_throughput(iters: u64) -> f64 {
    // one fast CPU core: the same byte-parallel CRC-32 the design computes.
    let din = 0xABu32;
    let mut crc = 0xFFFF_FFFFu32;
    let t = Instant::now();
    for _ in 0..iters {
        let mut c = crc;
        for i in 0..8 {
            let fb = ((c >> 31) ^ (din >> i)) & 1;
            c = (c << 1) ^ if fb == 1 { 0x04C1_1DB7 } else { 0 };
        }
        crc = c;
    }
    std::hint::black_box(crc);
    iters as f64 / t.elapsed().as_secs_f64()
}

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv");
    let src = std::fs::read_to_string(format!("{base}/crc32.sv")).unwrap();
    let imported = import_sv(&src, Some("crc32")).unwrap();
    let compiled = rrtl_core::compile(&imported.design).unwrap();
    let program = lower_to_packed_program(&compiled, "crc32").unwrap();
    let machine = lower_to_machine_program(&program);
    let gate = emit_kernel(&machine, Flavor::OpenCl).unwrap();
    let bmma = emit_kernel_linear(&program, Flavor::OpenCl).unwrap();
    let nsig = program.signals.len();
    let idx = |name: &str| program.signals.iter().position(|s| s.name.ends_with(name)).unwrap();
    let (crc_reg, c_reg, din) = (idx("crc__sv_reg"), idx(".c"), idx("din"));

    // CPU baseline (one core)
    let cpu = cpu_crc32_throughput(50_000_000);
    println!("CPU 1 core (native crc32): {:.1} Mcyc/s", cpu / 1e6);
    println!();
    println!("M3 GPU aggregate throughput (Mlane·cyc/s), crc32:");
    println!("  {:>10}  {:>14}  {:>14}  {:>10}", "lanes", "gate-tree", "BMMA/linear", "vs CPU");

    unsafe {
        let mut plat: Id = ptr::null_mut();
        ck(clGetPlatformIDs(1, &mut plat, ptr::null_mut()), "platforms");
        let mut dev: Id = ptr::null_mut();
        ck(clGetDeviceIDs(plat, ALL, 1, &mut dev, ptr::null_mut()), "devices");
        let mut e = 0;
        let ctx = clCreateContext(ptr::null(), 1, &dev, ptr::null(), ptr::null_mut(), &mut e);
        let q = clCreateCommandQueue(ctx, dev, 0, &mut e);
        let g = Gpu { q };
        let k_gate = build(ctx, dev, &gate);
        let k_bmma = build(ctx, dev, &bmma);

        let ncyc = 2000i64;
        for &lanes in &[16_384usize, 65_536, 262_144, 1_048_576] {
            // fresh buffer per size, init all lanes
            let mut state = vec![0u64; nsig * lanes];
            for l in 0..lanes {
                state[crc_reg * lanes + l] = 0xFFFF_FFFF;
                state[c_reg * lanes + l] = 0;
                state[din * lanes + l] = 0xAB;
            }
            let bytes = state.len() * 8;
            let buf = clCreateBuffer(ctx, RW, bytes, ptr::null_mut(), &mut e);
            ck(clEnqueueWriteBuffer(q, buf, TRUE, 0, bytes, state.as_ptr() as *const c_void, 0, ptr::null(), ptr::null_mut()), "write");

            // warm-up (JIT/driver), then time
            timed(&g, k_gate, buf, lanes, 50);
            timed(&g, k_bmma, buf, lanes, 50);
            let work = lanes as f64 * ncyc as f64;
            let tg = timed(&g, k_gate, buf, lanes, ncyc);
            let tb = timed(&g, k_bmma, buf, lanes, ncyc);
            let (gate_tp, bmma_tp) = (work / tg, work / tb);
            println!(
                "  {:>10}  {:>11.0} M  {:>11.0} M  {:>8.0}x",
                lanes, gate_tp / 1e6, bmma_tp / 1e6, bmma_tp / cpu
            );
        }
    }
    println!("\n(vs CPU = BMMA aggregate / one CPU core; the batch moat for regression/fuzz fleets)");
}
