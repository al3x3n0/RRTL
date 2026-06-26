//! Emit lane-parallel GPU sim kernels (CUDA + OpenCL) for a design from the
//! packed IR, and write them out so the OpenCL one can be build-validated on a
//! real runtime (bench/sv/check_opencl.sh uses Apple's OpenCL.framework here;
//! CUDA needs nvcc/NVIDIA, so it is emit-only).
//! Build: cargo run --release -p rrtl-sim-ir --example gpu_emit -- [design top]
use rrtl_sim_ir::gpu_codegen::{emit_kernel, Flavor};
use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
use rrtl_sv_frontend::import_sv;

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/sv");
    let args: Vec<String> = std::env::args().collect();
    let (file, top) = (
        args.get(1).cloned().unwrap_or_else(|| "crc32.sv".into()),
        args.get(2).cloned().unwrap_or_else(|| "crc32".into()),
    );
    let src = std::fs::read_to_string(format!("{base}/{file}")).expect("read design");
    let imported = import_sv(&src, Some(&top)).expect("import_sv");
    let compiled = rrtl_core::compile(&imported.design).expect("compile");
    let program = lower_to_packed_program(&compiled, &top).expect("lower packed");
    let machine = lower_to_machine_program(&program);

    let ocl = match emit_kernel(&machine, Flavor::OpenCl) {
        Ok(k) => k,
        Err(e) => {
            println!("{top}: GPU codegen skipped — {}", e.diagnostics[0].message);
            return;
        }
    };
    let cuda = emit_kernel(&machine, Flavor::Cuda).expect("emit cuda");
    let dir = std::env::temp_dir();
    std::fs::write(dir.join(format!("{top}.cl")), &ocl).unwrap();
    std::fs::write(dir.join(format!("{top}.cu")), &cuda).unwrap();
    println!(
        "{top}: emitted OpenCL kernel ({} lines) -> {}/{top}.cl, CUDA kernel ({} lines) -> {}/{top}.cu",
        ocl.lines().count(), dir.display(), cuda.lines().count(), dir.display()
    );
    println!("--- OpenCL kernel (head) ---");
    for l in ocl.lines().take(8) {
        println!("  {l}");
    }
}
