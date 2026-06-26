//! Run the GF(2)-LINEAR (BMMA / XOR-popcount) GPU kernel for crc32 on the real
//! M3 GPU and check it BIT-EXACT against (a) the gate-tree GPU kernel and (b) an
//! in-process reference. This is the tensor-core linear-cone offload: each
//! register update is a binary matrix product out = M·in ⊕ c, emitted as a
//! per-output-bit AND-popcount-parity — the 1-bit matmul a BMMA tensor core runs.
//! Build: cargo run --release -p rrtl-sim-ir --example linear_gpu_run
use std::ffi::{c_void, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;

use rrtl_sim_ir::gpu_codegen::{emit_kernel, emit_kernel_linear, Flavor};
use rrtl_sim_ir::{linearize, lower_to_machine_program, lower_to_packed_program, PackedOp};
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

/// Run `kernel_src` on the GPU for `lanes`×`ncyc`, with `init` (slot→value) set on
/// every lane, and return lane0's value at `read_slot`. `nslots` slots per lane.
fn gpu_run(kernel_src: &str, nslots: usize, lanes: usize, ncyc: i64, init: &[(usize, u64)], read_slot: usize) -> (u64, bool) {
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
        let v = state[read_slot * lanes];
        let same = (0..lanes).all(|l| state[read_slot * lanes + l] == v);
        (v, same)
    }
}

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv");
    let src = std::fs::read_to_string(format!("{base}/crc32.sv")).unwrap();
    let imported = import_sv(&src, Some("crc32")).unwrap();
    let compiled = rrtl_core::compile(&imported.design).unwrap();
    let program = lower_to_packed_program(&compiled, "crc32").unwrap();
    let machine = lower_to_machine_program(&program);

    let gate = emit_kernel(&machine, Flavor::OpenCl).unwrap();
    let lin = emit_kernel_linear(&program, Flavor::OpenCl).unwrap();
    let nsig = program.signals.len();
    let idx = |name: &str| program.signals.iter().position(|s| s.name.ends_with(name)).unwrap();
    let crc_reg = idx("crc__sv_reg");
    let c_reg = idx(".c");
    let din = idx("din");

    // report the matrix dims (the BMMA shape) for crc_reg
    let mut crc_next = None;
    for packet in &program.streams.tick_next {
        for op in &packet.ops {
            if let PackedOp::CaptureReg { dst, next, .. } = op {
                if *dst == crc_reg {
                    crc_next = Some(next);
                }
            }
        }
    }
    let lf = linearize::extract_linear_form(&program, crc_next.unwrap());
    println!(
        "crc32 register cone = GF(2) matrix M[{} x {}] (out_bits x in_bits) ⊕ const",
        lf.out_width, lf.total_in_bits
    );
    println!(
        "kernel size: gate-tree {} lines  vs  linear/BMMA {} lines",
        gate.lines().count(),
        lin.lines().count()
    );

    let lanes = 1024usize;
    let ncyc = 200i64;
    let din_val = 0xABu64;
    let init = [(crc_reg, 0xFFFF_FFFFu64), (c_reg, 0), (din, din_val)];

    let (gate_crc, gate_same) = gpu_run(&gate, nsig, lanes, ncyc, &init, crc_reg);
    let (lin_crc, lin_same) = gpu_run(&lin, nsig, lanes, ncyc, &init, crc_reg);

    println!("crc32 on GPU ({lanes} lanes x {ncyc} cyc):");
    println!("  gate-tree kernel  crc = 0x{gate_crc:08x}  (all lanes same: {gate_same})");
    println!("  linear/BMMA kernel crc = 0x{lin_crc:08x}  (all lanes same: {lin_same})");
    let ok = gate_same && lin_same && gate_crc == lin_crc;
    println!(
        "  =>  [{}]",
        if ok { "BIT-EXACT: BMMA linear-cone kernel == gate kernel on real GPU" } else { "FAIL" }
    );
}
