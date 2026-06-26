//! GPU CODEGEN (compiled OpenCL kernel, not the interpreter) throughput on the
//! W×D mul-add datapath — the apples-to-apples GPU number vs the CPU vector JIT
//! (extsim_compare) and N Verilator processes (bench/extsim). Runs K lanes × N
//! cycles in one compiled kernel and times the device execution.
//! Build: cargo run --release -p rrtl-sim-ir --example opencl_datapath -- [W D lanes cycles]
use std::ffi::c_void;
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::time::Instant;

use rrtl_core::{compile, lit_u, uint, Design};
use rrtl_sim_ir::gpu_codegen::{emit_kernel, Flavor};
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};

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

fn build_datapath(width: usize, depth: usize) -> Design {
    let c: u128 = 0x9e37_79b9;
    let mut design = Design::new();
    {
        let mut m = design.module("dp");
        let clk = m.input("clk", uint(1));
        let din = m.input("din", uint(32));
        for lane in 0..width {
            let acc = m.reg(format!("acc{lane}"), uint(32));
            m.clock(acc, clk);
            let mut prev = acc.value();
            for stage in 0..depth {
                let w = m.wire(format!("w{lane}_{stage}"), uint(32));
                m.assign(w, prev.clone() * lit_u(c, 32) + din.value());
                prev = w.value();
            }
            m.next(acc, prev + acc.value());
            let o = m.output(format!("o{lane}"), uint(32));
            m.assign(o, acc.value());
        }
    }
    design
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| a.get(i).and_then(|x| x.parse().ok()).unwrap_or(d);
    let (width, depth, lanes, cycles) = (p(1, 16), p(2, 8), p(3, 16384), p(4, 200));

    let design = build_datapath(width, depth);
    let compiled = compile(&design).unwrap();
    let program = lower_to_packed_program(&compiled, "dp").unwrap();
    let machine = lower_to_machine_program(&program);
    let kernel_src = emit_kernel(&machine, Flavor::OpenCl).unwrap();
    let nsig = program.signals.len();
    let din_idx = program.signals.iter().position(|s| s.name.ends_with("din")).unwrap();
    println!("GPU codegen datapath: W={width} D={depth} lanes={lanes} cycles={cycles} ({nsig} signals)");

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

        let mut state = vec![0u64; nsig * lanes];
        for l in 0..lanes {
            state[din_idx * lanes + l] = (l as u64).wrapping_mul(0x9e3779b1) & 0xffff_ffff;
        }
        let bytes = state.len() * 8;
        let buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, ptr::null_mut(), &mut e);
        ck(e, "CreateBuffer");
        ck(clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, bytes, state.as_ptr() as *const c_void, 0, ptr::null(), ptr::null_mut()), "Write");
        let nl = lanes as i64;
        let global = lanes;
        let run = |ncyc: i64| {
            ck(clSetKernelArg(kern, 0, 8, &buf as *const _ as *const c_void), "Arg0");
            ck(clSetKernelArg(kern, 1, 8, &nl as *const _ as *const c_void), "Arg1");
            ck(clSetKernelArg(kern, 2, 8, &ncyc as *const _ as *const c_void), "Arg2");
            ck(clEnqueueNDRangeKernel(q, kern, 1, ptr::null(), &global, ptr::null(), 0, ptr::null(), ptr::null_mut()), "NDRange");
            ck(clFinish(q), "Finish");
        };
        run(50); // warm (compile/upload settle)
        let t = Instant::now();
        run(cycles as i64);
        let secs = t.elapsed().as_secs_f64();
        let mlc = (lanes * cycles) as f64 / secs / 1e6;
        println!("  GPU codegen throughput: {mlc:.1} M-lane-cycles/s  ({:.3} M-cyc/s per lane)", cycles as f64 / secs / 1e6);
        println!("  (compare: CPU vector-JIT ~474 M-lane-cyc/s @1024 lanes; Verilator 8 procs ~33 M-inst-cyc/s; GPU interp ~22.8)");
    }
}
