//! One-call bringup of SystemVerilog source onto the batch interpreter, on either
//! the GPU (`SvSim::gpu`) or the host CPU interpreter (`SvSim::cpu`). This lives in
//! the runtime (orchestration) layer because it ties the SV frontend to the GPU
//! backend, and the frontend must not depend on the backend.
//!
//! ```no_run
//! use rrtl_runtime::sv_gpu::SvSim;
//! let mut sim = SvSim::gpu("module Inc(input logic [7:0] a, output logic [7:0] y); assign y = a + 8'd1; endmodule", None, 1024).unwrap();
//! sim.set_all("a", 41).unwrap();
//! sim.tick(1);
//! sim.synchronize();
//! assert!(sim.get("y").unwrap().iter().all(|&v| v == 42));
//! ```

use std::collections::HashMap;

use rrtl_core::compile;
use rrtl_gpu_sim::interp::{InterpGpuSimulator, InterpProgram, InterpRunner};
use rrtl_ir::{Diagnostic, ErrorReport, SignalKind};
use rrtl_sim_ir::lower_to_packed_program;
use rrtl_sv_frontend::import_sv;

fn err(msg: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![Diagnostic::new("E_SV_SIM", msg)])
}

fn limbs_of(width: u32) -> usize {
    (((width + 31) / 32).max(1)) as usize
}

fn mask_u128(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

/// Direction of a top-level port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortDir {
    Input,
    Output,
}

/// A resolved top-level port: name, direction, width, and storage offset.
#[derive(Clone, Debug)]
pub struct Port {
    pub name: String,
    pub dir: PortDir,
    pub width: u32,
    pub offset: usize,
}

enum Backend {
    Cpu(InterpRunner),
    Gpu(Box<InterpGpuSimulator>),
}

impl Backend {
    fn set_signal(&mut self, offset: usize, vals: &[u32]) {
        match self {
            Backend::Cpu(r) => r.set_signal(offset, vals),
            Backend::Gpu(g) => g.set_signal(offset, vals),
        }
    }
    fn get_signal(&self, offset: usize) -> Vec<u32> {
        match self {
            Backend::Cpu(r) => r.get_signal(offset),
            Backend::Gpu(g) => g.get_signal(offset),
        }
    }
    fn tick_many(&mut self, steps: usize) {
        match self {
            Backend::Cpu(r) => r.tick_many(steps),
            Backend::Gpu(g) => g.tick_many(steps),
        }
    }
    fn synchronize(&self) {
        if let Backend::Gpu(g) = self {
            g.synchronize();
        }
    }
}

/// A simulator built directly from SystemVerilog source, with name-based I/O over
/// `lanes` independent batch lanes.
pub struct SvSim {
    backend: Backend,
    ports: HashMap<String, Port>,
    top: String,
    lanes: usize,
}

fn lower_sv(
    sv: &str,
    top: Option<&str>,
) -> Result<(InterpProgram, HashMap<String, Port>, String), ErrorReport> {
    let imported = import_sv(sv, top)?;
    let top_name = imported.top_name.clone();
    let compiled = compile(&imported.design)?;
    let program = lower_to_packed_program(&compiled, &top_name)?;
    let encoded = InterpProgram::encode_design(&program)?;
    let module = compiled
        .find_module(&top_name)
        .ok_or_else(|| err(format!("top module `{top_name}` not found after compile")))?;

    let mut ports = HashMap::new();
    for s in &module.signals {
        let dir = match s.kind {
            SignalKind::Input => PortDir::Input,
            SignalKind::Output => PortDir::Output,
            _ => continue,
        };
        let idx = program
            .signal_index(s.handle)
            .ok_or_else(|| err(format!("port `{}` missing from packed program", s.name)))?;
        let offset = program.signals[idx].layout.offset;
        ports.insert(
            s.name.clone(),
            Port {
                name: s.name.clone(),
                dir,
                width: s.width,
                offset,
            },
        );
    }
    Ok((encoded, ports, top_name))
}

impl SvSim {
    /// Build a GPU-backed simulator from SV source.
    pub fn gpu(sv: &str, top: Option<&str>, lanes: usize) -> Result<Self, ErrorReport> {
        let (encoded, ports, top) = lower_sv(sv, top)?;
        let gpu = InterpGpuSimulator::new(&encoded, lanes)?;
        Ok(Self {
            backend: Backend::Gpu(Box::new(gpu)),
            ports,
            top,
            lanes,
        })
    }

    /// Build a host-CPU interpreter simulator from SV source (deterministic; useful
    /// for tests and headless/CI runs where no GPU is available).
    pub fn cpu(sv: &str, top: Option<&str>, lanes: usize) -> Result<Self, ErrorReport> {
        let (encoded, ports, top) = lower_sv(sv, top)?;
        Ok(Self {
            backend: Backend::Cpu(InterpRunner::new(encoded, lanes)),
            ports,
            top,
            lanes,
        })
    }

    pub fn top(&self) -> &str {
        &self.top
    }
    pub fn lanes(&self) -> usize {
        self.lanes
    }
    pub fn port(&self, name: &str) -> Option<&Port> {
        self.ports.get(name)
    }
    pub fn ports(&self) -> impl Iterator<Item = &Port> {
        self.ports.values()
    }

    /// Set an input port, one value per lane (`lane_values.len()` must be `lanes`).
    pub fn set(&mut self, name: &str, lane_values: &[u128]) -> Result<(), ErrorReport> {
        let port = self
            .ports
            .get(name)
            .ok_or_else(|| err(format!("no port `{name}`")))?
            .clone();
        if port.dir != PortDir::Input {
            return Err(err(format!("port `{name}` is not an input")));
        }
        if lane_values.len() != self.lanes {
            return Err(err(format!(
                "port `{name}`: expected {} lane values, got {}",
                self.lanes,
                lane_values.len()
            )));
        }
        for l in 0..limbs_of(port.width) {
            let limb: Vec<u32> = lane_values.iter().map(|&v| (v >> (32 * l)) as u32).collect();
            self.backend.set_signal(port.offset + l, &limb);
        }
        Ok(())
    }

    /// Set an input port to the same value on every lane.
    pub fn set_all(&mut self, name: &str, value: u128) -> Result<(), ErrorReport> {
        let vals = vec![value; self.lanes];
        self.set(name, &vals)
    }

    /// Read a port's value on every lane (width-masked).
    pub fn get(&self, name: &str) -> Result<Vec<u128>, ErrorReport> {
        let port = self
            .ports
            .get(name)
            .ok_or_else(|| err(format!("no port `{name}`")))?;
        let mut out = vec![0u128; self.lanes];
        for l in 0..limbs_of(port.width) {
            for (lane, w) in self.backend.get_signal(port.offset + l).into_iter().enumerate() {
                out[lane] |= (w as u128) << (32 * l);
            }
        }
        let m = mask_u128(port.width);
        for o in &mut out {
            *o &= m;
        }
        Ok(out)
    }

    /// Advance `steps` clock cycles.
    pub fn tick(&mut self, steps: usize) {
        self.backend.tick_many(steps);
    }

    /// Block until queued GPU work completes (a no-op on the CPU backend).
    pub fn synchronize(&self) {
        self.backend.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rrtl_core::Simulator;

    const COUNTER: &str = r#"
        module Counter(
          input  logic       clk, rst, en,
          input  logic [7:0] step,
          output logic [7:0] q
        );
          logic [7:0] q_r;
          assign q = q_r;
          always_ff @(posedge clk) begin
            if (rst)     q_r <= 8'd0;
            else if (en) q_r <= q_r + step;
          end
        endmodule
    "#;

    #[test]
    fn cpu_backend_matches_gold_simulator() {
        let imported = import_sv(COUNTER, Some("Counter")).unwrap();
        let mut gold = Simulator::new(&imported.design, "Counter").unwrap();
        let handle = |n: &str| {
            compile(&imported.design)
                .unwrap()
                .find_module("Counter")
                .unwrap()
                .signals
                .iter()
                .find(|s| s.name == n)
                .unwrap()
                .handle
        };

        let mut sim = SvSim::cpu(COUNTER, Some("Counter"), 1).unwrap();
        assert_eq!(sim.lanes(), 1);
        assert_eq!(sim.port("q").unwrap().dir, PortDir::Output);
        assert_eq!(sim.port("step").unwrap().width, 8);

        let mut lcg: u32 = 1;
        let mut rng = || {
            lcg = lcg.wrapping_mul(1664525).wrapping_add(1013904223);
            lcg
        };
        for cycle in 0..50u32 {
            let (rst, en, step) = ((cycle == 0) as u128, (rng() & 1) as u128, (rng() & 0xff) as u128);
            for (n, v) in [("rst", rst), ("en", en), ("step", step)] {
                gold.set(handle(n), v);
            }
            // clk is a tick convention for the interpreter.
            sim.set_all("clk", 1).unwrap();
            sim.set_all("rst", rst).unwrap();
            sim.set_all("en", en).unwrap();
            sim.set_all("step", step).unwrap();
            gold.tick();
            sim.tick(1);
            assert_eq!(gold.get(handle("q")), sim.get("q").unwrap()[0], "cycle {cycle}");
        }
    }

    #[test]
    fn rejects_setting_output_port() {
        let mut sim = SvSim::cpu(COUNTER, Some("Counter"), 1).unwrap();
        assert!(sim.set_all("q", 0).is_err());
        assert!(sim.get("nope").is_err());
    }

    #[test]
    fn lanes_are_independent() {
        let mut sim = SvSim::cpu(COUNTER, Some("Counter"), 4).unwrap();
        sim.set_all("clk", 1).unwrap();
        sim.set_all("rst", 1).unwrap();
        sim.set_all("en", 0).unwrap();
        sim.set_all("step", 0).unwrap();
        sim.tick(1);
        // distinct per-lane step, enable on, no reset
        sim.set_all("rst", 0).unwrap();
        sim.set_all("en", 1).unwrap();
        sim.set("step", &[1, 2, 3, 4]).unwrap();
        sim.tick(3);
        let q = sim.get("q").unwrap();
        assert_eq!(q, vec![3, 6, 9, 12]);
    }
}
