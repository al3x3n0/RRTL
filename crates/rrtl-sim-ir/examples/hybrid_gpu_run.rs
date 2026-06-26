//! Selective per-cone offload on a MIXED design (mixed.sv: linear CRC + non-linear
//! MAC + counter). Emit BOTH the full gate-tree kernel AND the HYBRID kernel
//! (CRC cone → BMMA matrix, acc/count → gate-tree, CRC's gate instrs pruned by
//! liveness), run both on the real M3 GPU, and check all three outputs bit-exact.
//! Build: cargo run --release -p rrtl-sim-ir --example hybrid_gpu_run
use std::ffi::{c_void, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;

use rrtl_sim_ir::gpu_codegen::{emit_kernel, emit_kernel_hybrid, Flavor};
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
    fn clEnqueueReadBuffer(q: Id, b: Id, block: u32, off: usize, size: usize, p: *mut c_void, n: u32, wl: *const Id, ev: *mut Id) -> c_int;
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

/// Run `kernel_src` for `lanes`×`ncyc` with `init` (slot→value) on every lane;
/// return (lane0 values at `reads`, all-lanes-identical-across-reads).
fn gpu_run(kernel_src: &str, nslots: usize, lanes: usize, ncyc: i64, init: &[(usize, u64)], reads: &[usize]) -> (Vec<u64>, bool) {
    unsafe {
        let mut plat: Id = ptr::null_mut();
        ck(clGetPlatformIDs(1, &mut plat, ptr::null_mut()), "platforms");
        let mut dev: Id = ptr::null_mut();
        ck(clGetDeviceIDs(plat, ALL, 1, &mut dev, ptr::null_mut()), "devices");
        let mut e = 0;
        let ctx = clCreateContext(ptr::null(), 1, &dev, ptr::null(), ptr::null_mut(), &mut e);
        let q = clCreateCommandQueue(ctx, dev, 0, &mut e);
        let cs = CString::new(kernel_src).unwrap();
        let sp = cs.as_ptr();
        let prog = clCreateProgramWithSource(ctx, 1, &sp, ptr::null(), &mut e);
        ck(clBuildProgram(prog, 1, &dev, ptr::null(), ptr::null(), ptr::null_mut()), "build");
        let kn = CString::new("tick").unwrap();
        let kern = clCreateKernel(prog, kn.as_ptr(), &mut e);
        ck(e, "kernel");
        let mut state = vec![0u64; nslots * lanes];
        for l in 0..lanes {
            for &(slot, val) in init {
                state[slot * lanes + l] = val;
            }
        }
        let bytes = state.len() * 8;
        let buf = clCreateBuffer(ctx, RW, bytes, ptr::null_mut(), &mut e);
        ck(clEnqueueWriteBuffer(q, buf, TRUE, 0, bytes, state.as_ptr() as *const c_void, 0, ptr::null(), ptr::null_mut()), "write");
        let nl = lanes as i64;
        ck(clSetKernelArg(kern, 0, 8, &buf as *const _ as *const c_void), "arg0");
        ck(clSetKernelArg(kern, 1, 8, &nl as *const _ as *const c_void), "arg1");
        ck(clSetKernelArg(kern, 2, 8, &ncyc as *const _ as *const c_void), "arg2");
        let g = lanes;
        ck(clEnqueueNDRangeKernel(q, kern, 1, ptr::null(), &g, ptr::null(), 0, ptr::null(), ptr::null_mut()), "ndrange");
        ck(clFinish(q), "finish");
        ck(clEnqueueReadBuffer(q, buf, TRUE, 0, bytes, state.as_mut_ptr() as *mut c_void, 0, ptr::null(), ptr::null_mut()), "read");
        let vals: Vec<u64> = reads.iter().map(|&s| state[s * lanes]).collect();
        let same = reads.iter().all(|&s| (0..lanes).all(|l| state[s * lanes + l] == state[s * lanes]));
        (vals, same)
    }
}

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv");
    let src = std::fs::read_to_string(format!("{base}/mixed.sv")).unwrap();
    let imported = import_sv(&src, Some("mixed")).unwrap();
    let compiled = rrtl_core::compile(&imported.design).unwrap();
    let program = lower_to_packed_program(&compiled, "mixed").unwrap();
    let machine = lower_to_machine_program(&program);

    let gate = emit_kernel(&machine, Flavor::OpenCl).unwrap();
    let hybrid = emit_kernel_hybrid(&machine, &program, Flavor::OpenCl).unwrap();
    let nsig = program.signals.len();
    let idx = |name: &str| program.signals.iter().position(|s| s.name.ends_with(name)).unwrap();
    let (crc, acc, count) = (idx(".crc"), idx(".acc"), idx(".count"));
    let (din, a, b) = (idx("din"), idx(".a"), idx(".b"));

    // the liveness prune line the hybrid kernel reports in its header comment
    let prune = hybrid.lines().find(|l| l.contains("pruned by liveness")).unwrap_or("").trim();
    println!("hybrid kernel: {}", prune.trim_start_matches("// "));

    let lanes = 1024usize;
    let ncyc = 64i64;
    let init = [
        (crc, 0xFFFF_FFFFu64), (acc, 0), (count, 0),
        (din, 0xAB), (a, 0x1234), (b, 0x00CD),
    ];
    let reads = [crc, acc, count];
    let (g, gsame) = gpu_run(&gate, nsig, lanes, ncyc, &init, &reads);
    let (h, hsame) = gpu_run(&hybrid, nsig, lanes, ncyc, &init, &reads);

    println!("mixed.sv on GPU ({lanes} lanes x {ncyc} cyc):");
    println!("  gate-tree : crc=0x{:08x} acc={} count={} (lanes same: {})", g[0], g[1], g[2], gsame);
    println!("  HYBRID    : crc=0x{:08x} acc={} count={} (lanes same: {})", h[0], h[1], h[2], hsame);
    let ok = gsame && hsame && g == h;
    println!(
        "  =>  [{}]",
        if ok { "BIT-EXACT: hybrid (CRC on BMMA, acc/count gate) == full gate kernel on real GPU" } else { "FAIL" }
    );
}
