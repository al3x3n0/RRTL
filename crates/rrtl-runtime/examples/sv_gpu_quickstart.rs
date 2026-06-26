//! One-call SystemVerilog → batch GPU simulation via `SvSim`. Drives per-lane
//! stimulus and cross-checks the GPU backend against the deterministic CPU
//! interpreter backend (both built from the same SV source with name-based I/O).
//!
//! Usage: cargo run --release -p rrtl-runtime --example sv_gpu_quickstart -- [lanes steps]

use std::time::Instant;
use rrtl_runtime::sv_gpu::SvSim;

const SV: &str = r#"
module Dut(
  input  logic        clk, rst, en,
  input  logic [15:0] a, b,
  output logic [15:0] sum,
  output logic [31:0] acc
);
  assign sum = a + b;
  logic [31:0] acc_r;
  assign acc = acc_r;
  always_ff @(posedge clk) begin
    if (rst)      acc_r <= 32'd0;
    else if (en)  acc_r <= acc_r + {16'd0, sum};
  end
endmodule
"#;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (lanes, steps) = (p(1, 1024), p(2, 32));

    let mut cpu = SvSim::cpu(SV, Some("Dut"), lanes).unwrap();
    println!("top=`{}` lanes={lanes}  ports:", cpu.top());
    let mut ports: Vec<_> = cpu.ports().map(|p| format!("{}:{}[{}b]", p.name, if p.dir == rrtl_runtime::sv_gpu::PortDir::Input {"in"} else {"out"}, p.width)).collect();
    ports.sort();
    println!("  {}", ports.join("  "));

    let mut gpu = match SvSim::gpu(SV, Some("Dut"), lanes) {
        Ok(g) => g,
        Err(e) => { println!("GPU unavailable ({e:?}); CPU backend is usable."); return; }
    };

    // Deterministic per-lane stimulus.
    let a: Vec<u128> = (0..lanes as u128).map(|l| (l.wrapping_mul(40503) + 1) & 0xffff).collect();
    let b: Vec<u128> = (0..lanes as u128).map(|l| (l.wrapping_mul(2654435761) + 7) & 0xffff).collect();
    let mut lcg: u128 = 0xdead_beef;
    let mut rng = || { lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1); lcg >> 96 };

    let t0 = Instant::now();
    let mut ok = true;
    for cycle in 0..steps as u32 {
        let rst = (cycle == 0) as u128;
        let en = (rng() & 1) as u128;
        for s in [&mut cpu, &mut gpu] {
            s.set_all("clk", 1).unwrap();
            s.set_all("rst", rst).unwrap();
            s.set_all("en", en).unwrap();
            s.set("a", &a).unwrap();
            s.set("b", &b).unwrap();
            s.tick(1);
        }
        gpu.synchronize();
        for port in ["sum", "acc"] {
            if cpu.get(port).unwrap() != gpu.get(port).unwrap() { ok = false; }
        }
    }
    println!("CPU vs GPU over {steps} cycles ({lanes} lanes): {}  [{:.1} ms]",
        if ok { "OK" } else { "MISMATCH" }, t0.elapsed().as_secs_f64() * 1e3);
    // show a couple of lane results
    let acc = gpu.get("acc").unwrap();
    println!("  acc[0..4] = {:?}", &acc[..4.min(lanes)]);
}
