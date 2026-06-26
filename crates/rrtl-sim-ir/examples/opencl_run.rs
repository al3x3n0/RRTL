//! Run the emitted OpenCL sim kernel on the real GPU (Apple's OpenCL.framework
//! here) for K lanes x N cycles, read the state back, and check it bit-exact
//! against an in-process reference (linearize::eval_expr on the same cones).
//! Proves the compiled-GPU backend EXECUTES correctly, not just compiles.
//! Build: cargo run --release -p rrtl-sim-ir --example opencl_run
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::raw::{c_char, c_int};
use std::ptr;

use rrtl_sim_ir::gpu_codegen::{emit_kernel, Flavor};
use rrtl_sim_ir::{linearize, lower_to_machine_program, lower_to_packed_program, PackedOp};
use rrtl_sv_frontend::import_sv;

// ---- minimal OpenCL 1.2 FFI (Apple framework) ----
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
const CL_DEVICE_TYPE_ALL: u64 = 0xFFFF_FFFF;
const CL_MEM_READ_WRITE: u64 = 1;
const CL_TRUE: u32 = 1;
fn ck(e: c_int, what: &str) {
    assert_eq!(e, 0, "OpenCL {what} failed: {e}");
}

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv");
    let src = std::fs::read_to_string(format!("{base}/crc32.sv")).unwrap();
    let imported = import_sv(&src, Some("crc32")).unwrap();
    let compiled = rrtl_core::compile(&imported.design).unwrap();
    let program = lower_to_packed_program(&compiled, "crc32").unwrap();
    let machine = lower_to_machine_program(&program);
    let kernel_src = emit_kernel(&machine, Flavor::OpenCl).unwrap();
    let nsig = program.signals.len();
    // slot = signal index in the packed program (what the kernel addresses).
    let idx = |name: &str| program.signals.iter().position(|s| s.name.ends_with(name)).unwrap();
    let crc_reg = idx("crc__sv_reg");
    let c_reg = idx(".c");
    let din = idx("din");

    let lanes = 1024usize;
    let ncyc = 200i64;
    let din_val = 0xABu64;

    // ---- in-process reference via the cone interpreter ----
    let comb_def = linearize::comb_defs(&program);
    let mut crc_next = None;
    let mut c_next = None;
    for packet in &program.streams.tick_next {
        for op in &packet.ops {
            if let PackedOp::CaptureReg { dst, next, .. } = op {
                if *dst == crc_reg { crc_next = Some(next); }
                if *dst == c_reg { c_next = Some(next); }
            }
        }
    }
    let (crc_next, c_next) = (crc_next.unwrap(), c_next.unwrap());
    let (mut rcrc, mut rc) = (0xFFFF_FFFFu128, 0u128);
    for _ in 0..ncyc {
        let mut lv: HashMap<usize, u128> = HashMap::new();
        lv.insert(crc_reg, rcrc);
        lv.insert(c_reg, rc);
        lv.insert(din, din_val as u128);
        let nc = linearize::eval_expr(c_next, &comb_def, &lv);
        let ncrc = linearize::eval_expr(crc_next, &comb_def, &lv);
        rc = nc;
        rcrc = ncrc;
    }
    let reference = rcrc as u64;

    // ---- run on the GPU ----
    unsafe {
        let mut plat: Id = ptr::null_mut();
        ck(clGetPlatformIDs(1, &mut plat, ptr::null_mut()), "GetPlatformIDs");
        let mut dev: Id = ptr::null_mut();
        ck(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &mut dev, ptr::null_mut()), "GetDeviceIDs");
        let mut e = 0;
        let ctx = clCreateContext(ptr::null(), 1, &dev, ptr::null(), ptr::null_mut(), &mut e);
        ck(e, "CreateContext");
        let q = clCreateCommandQueue(ctx, dev, 0, &mut e);
        ck(e, "CreateCommandQueue");
        let csrc = std::ffi::CString::new(kernel_src.clone()).unwrap();
        let sptr = csrc.as_ptr();
        let prog = clCreateProgramWithSource(ctx, 1, &sptr, ptr::null(), &mut e);
        ck(e, "CreateProgramWithSource");
        ck(clBuildProgram(prog, 1, &dev, ptr::null(), ptr::null(), ptr::null_mut()), "BuildProgram");
        let kname = std::ffi::CString::new("tick").unwrap();
        let kern = clCreateKernel(prog, kname.as_ptr(), &mut e);
        ck(e, "CreateKernel");

        // lane-major state: st[slot*nl + lane]
        let mut state = vec![0u64; nsig * lanes];
        for l in 0..lanes {
            state[crc_reg * lanes + l] = 0xFFFF_FFFF;
            state[c_reg * lanes + l] = 0;
            state[din * lanes + l] = din_val;
        }
        let bytes = state.len() * 8;
        let buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, ptr::null_mut(), &mut e);
        ck(e, "CreateBuffer");
        ck(clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, bytes, state.as_ptr() as *const c_void, 0, ptr::null(), ptr::null_mut()), "Write");
        let nl = lanes as i64;
        ck(clSetKernelArg(kern, 0, 8, &buf as *const _ as *const c_void), "Arg0");
        ck(clSetKernelArg(kern, 1, 8, &nl as *const _ as *const c_void), "Arg1");
        ck(clSetKernelArg(kern, 2, 8, &ncyc as *const _ as *const c_void), "Arg2");
        let global = lanes;
        ck(clEnqueueNDRangeKernel(q, kern, 1, ptr::null(), &global, ptr::null(), 0, ptr::null(), ptr::null_mut()), "NDRange");
        ck(clFinish(q), "Finish");
        ck(clEnqueueReadBuffer(q, buf, CL_TRUE, 0, bytes, state.as_mut_ptr() as *mut c_void, 0, ptr::null(), ptr::null_mut()), "Read");

        // all lanes identical (same input) + match the reference
        let gpu0 = state[crc_reg * lanes];
        let all_same = (0..lanes).all(|l| state[crc_reg * lanes + l] == gpu0);
        println!("crc32 on GPU ({lanes} lanes x {ncyc} cyc):");
        println!("  GPU lane0 crc = 0x{gpu0:08x} | reference = 0x{reference:08x}");
        println!(
            "  all {lanes} lanes identical: {} | GPU == reference: {}  =>  [{}]",
            all_same, gpu0 == reference,
            if all_same && gpu0 == reference { "BIT-EXACT on real GPU" } else { "FAIL" }
        );
    }
}
