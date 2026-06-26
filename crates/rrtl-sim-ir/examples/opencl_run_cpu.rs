//! Run cpu.sv (a RISC-V execute unit WITH a register-file memory) on the real
//! GPU and check it bit-exact against the Cranelift JIT — validates the GPU
//! codegen's MEMORY support end to end.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example opencl_run_cpu
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
mod cl {
    use std::ffi::c_void;
    use std::os::raw::{c_char, c_int};
    pub type Id = *mut c_void;
    #[link(name = "OpenCL", kind = "framework")]
    extern "C" {
        pub fn clGetPlatformIDs(n: u32, p: *mut Id, np: *mut u32) -> c_int;
        pub fn clGetDeviceIDs(plat: Id, ty: u64, n: u32, d: *mut Id, nd: *mut u32) -> c_int;
        pub fn clCreateContext(props: *const isize, n: u32, d: *const Id, cb: *const c_void, ud: *mut c_void, e: *mut c_int) -> Id;
        pub fn clCreateCommandQueue(ctx: Id, dev: Id, props: u64, e: *mut c_int) -> Id;
        pub fn clCreateProgramWithSource(ctx: Id, n: u32, s: *const *const c_char, l: *const usize, e: *mut c_int) -> Id;
        pub fn clBuildProgram(p: Id, n: u32, d: *const Id, opt: *const c_char, cb: *const c_void, ud: *mut c_void) -> c_int;
        pub fn clCreateKernel(p: Id, name: *const c_char, e: *mut c_int) -> Id;
        pub fn clCreateBuffer(ctx: Id, flags: u64, size: usize, host: *mut c_void, e: *mut c_int) -> Id;
        pub fn clEnqueueWriteBuffer(q: Id, b: Id, block: u32, off: usize, size: usize, p: *const c_void, n: u32, wl: *const Id, ev: *mut Id) -> c_int;
        pub fn clEnqueueReadBuffer(q: Id, b: Id, block: u32, off: usize, size: usize, p: *mut c_void, n: u32, wl: *const Id, ev: *mut Id) -> c_int;
        pub fn clSetKernelArg(k: Id, i: u32, size: usize, val: *const c_void) -> c_int;
        pub fn clEnqueueNDRangeKernel(q: Id, k: Id, dim: u32, off: *const usize, g: *const usize, l: *const usize, n: u32, wl: *const Id, ev: *mut Id) -> c_int;
        pub fn clFinish(q: Id) -> c_int;
    }
    pub const ALL: u64 = 0xFFFF_FFFF;
    pub const RW: u64 = 1;
    pub const TRUE: u32 = 1;
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::gpu_codegen::{emit_kernel, state_slots, Flavor};
    use rrtl_sim_ir::{jit::JitSimulator, lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;
    use std::ffi::CString;
    use std::os::raw::c_void;
    use std::ptr;

    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv");
    let src = std::fs::read_to_string(format!("{base}/cpu.sv")).unwrap();
    let imported = import_sv(&src, Some("cpu")).unwrap();
    let compiled = rrtl_core::compile(&imported.design).unwrap();
    let program = lower_to_packed_program(&compiled, "cpu").unwrap();
    let machine = lower_to_machine_program(&program);
    let kernel_src = emit_kernel(&machine, Flavor::OpenCl).unwrap();
    let slots = state_slots(&machine);
    let slot = |name: &str| program.signals.iter().position(|s| s.name.ends_with(name)).unwrap();
    let (i_clk, i_rst, i_instr) = (slot("clk"), slot("rst"), slot("instr"));
    let (o_pc, o_x10) = (slot(".pc"), slot(".x10"));

    let lanes = 256usize;
    let ncyc = 500i64;
    let instr = 0x0050_0513u64; // addi x10, x0, 5

    // ---- JIT reference (from zero state, rst=0, held instr) ----
    let mut jit = JitSimulator::compile(&machine).unwrap();
    for _ in 0..ncyc {
        jit.set_signal(i_clk, 1);
        jit.set_signal(i_rst, 0);
        jit.set_signal(i_instr, instr);
        jit.tick();
    }
    let (ref_pc, ref_x10) = (jit.get_signal(o_pc), jit.get_signal(o_x10));

    // ---- GPU ----
    use cl::*;
    let cked = |e, w: &str| assert_eq!(e, 0, "OpenCL {w}: {e}");
    unsafe {
        let (mut plat, mut dev): (Id, Id) = (ptr::null_mut(), ptr::null_mut());
        cked(clGetPlatformIDs(1, &mut plat, ptr::null_mut()), "platforms");
        cked(clGetDeviceIDs(plat, ALL, 1, &mut dev, ptr::null_mut()), "devices");
        let mut e = 0;
        let ctx = clCreateContext(ptr::null(), 1, &dev, ptr::null(), ptr::null_mut(), &mut e);
        let q = clCreateCommandQueue(ctx, dev, 0, &mut e);
        let cs = CString::new(kernel_src).unwrap();
        let sp = cs.as_ptr();
        let prog = clCreateProgramWithSource(ctx, 1, &sp, ptr::null(), &mut e);
        cked(clBuildProgram(prog, 1, &dev, ptr::null(), ptr::null(), ptr::null_mut()), "build");
        let kn = CString::new("tick").unwrap();
        let kern = clCreateKernel(prog, kn.as_ptr(), &mut e);
        cked(e, "kernel");

        let mut state = vec![0u64; slots * lanes];
        for l in 0..lanes {
            state[i_clk * lanes + l] = 1;
            state[i_rst * lanes + l] = 0;
            state[i_instr * lanes + l] = instr;
        }
        let bytes = state.len() * 8;
        let buf = clCreateBuffer(ctx, RW, bytes, ptr::null_mut(), &mut e);
        cked(clEnqueueWriteBuffer(q, buf, TRUE, 0, bytes, state.as_ptr() as *const c_void, 0, ptr::null(), ptr::null_mut()), "write");
        let nl = lanes as i64;
        cked(clSetKernelArg(kern, 0, 8, &buf as *const _ as *const c_void), "arg0");
        cked(clSetKernelArg(kern, 1, 8, &nl as *const _ as *const c_void), "arg1");
        cked(clSetKernelArg(kern, 2, 8, &ncyc as *const _ as *const c_void), "arg2");
        let g = lanes;
        cked(clEnqueueNDRangeKernel(q, kern, 1, ptr::null(), &g, ptr::null(), 0, ptr::null(), ptr::null_mut()), "ndrange");
        cked(clFinish(q), "finish");
        cked(clEnqueueReadBuffer(q, buf, TRUE, 0, bytes, state.as_mut_ptr() as *mut c_void, 0, ptr::null(), ptr::null_mut()), "read");

        let gpu_pc = state[o_pc * lanes];
        let gpu_x10 = state[o_x10 * lanes];
        let all_same = (0..lanes).all(|l| state[o_pc * lanes + l] == gpu_pc && state[o_x10 * lanes + l] == gpu_x10);
        println!("cpu.sv (with regfile memory) on GPU, {lanes} lanes x {ncyc} cyc:");
        println!("  GPU  pc=0x{gpu_pc:x} x10={gpu_x10}");
        println!("  JIT  pc=0x{ref_pc:x} x10={ref_x10}");
        println!(
            "  all lanes identical: {all_same} | GPU == JIT: {}  =>  [{}]",
            gpu_pc == ref_pc && gpu_x10 == ref_x10,
            if all_same && gpu_pc == ref_pc && gpu_x10 == ref_x10 { "BIT-EXACT (memory works on GPU)" } else { "FAIL" }
        );
    }
}
