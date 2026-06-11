//! Heterogeneous CPU‖GPU lane-split simulator.
//!
//! Runs the *same whole design* over a disjoint split of the stimulus lanes:
//! lanes `0..gpu_lanes` on the GPU interpreter, `gpu_lanes..total` on the CPU
//! SIMD engine. Because lanes are independent there is no per-cycle
//! communication — the GPU's `tick_many` submits asynchronously and returns, the
//! CPU simulates its lanes on the calling thread *concurrently*, and we only
//! synchronize the GPU before reading. Wall-clock ≈ max(cpu_time, gpu_time), so
//! throughput is additive (`cpu_rate + gpu_rate`) when the split is balanced.
//!
//! Results are per-lane identical to running all lanes on one engine (validated
//! by the lane_split_check example's differential oracle).

use rrtl_ir::{ErrorReport, Signal};
use rrtl_sim_ir::{PackedProgram, SimdCpuSimulator};

use crate::interp::{InterpGpuSimulator, InterpProgram};

/// CPU‖GPU lane-split executor. Lanes `0..gpu_lanes` run on the GPU, the rest on
/// the CPU SIMD engine.
pub struct LaneSplitSimulator {
    program: PackedProgram,
    gpu: Option<InterpGpuSimulator>,
    cpu: Option<SimdCpuSimulator>,
    gpu_lanes: usize,
    cpu_lanes: usize,
}

impl LaneSplitSimulator {
    /// Builds a split running `gpu_lanes` of `total_lanes` on the GPU and the
    /// remainder on the CPU. Either side may be zero (all-CPU or all-GPU).
    pub fn new(
        program: PackedProgram,
        total_lanes: usize,
        gpu_lanes: usize,
    ) -> Result<Self, ErrorReport> {
        let gpu_lanes = gpu_lanes.min(total_lanes);
        let cpu_lanes = total_lanes - gpu_lanes;
        let gpu = if gpu_lanes > 0 {
            let encoded = InterpProgram::encode_design(&program)?;
            Some(InterpGpuSimulator::new(&encoded, gpu_lanes)?)
        } else {
            None
        };
        let cpu = if cpu_lanes > 0 {
            Some(SimdCpuSimulator::new(program.clone(), cpu_lanes)?)
        } else {
            None
        };
        Ok(Self {
            program,
            gpu,
            cpu,
            gpu_lanes,
            cpu_lanes,
        })
    }

    pub fn total_lanes(&self) -> usize {
        self.gpu_lanes + self.cpu_lanes
    }

    pub fn gpu_lanes(&self) -> usize {
        self.gpu_lanes
    }

    fn offset(&self, signal: Signal) -> Result<usize, ErrorReport> {
        let index = self.program.signal_index(signal).ok_or_else(|| {
            ErrorReport::new(vec![rrtl_ir::Diagnostic::new(
                "E_LANE_SPLIT_SIGNAL",
                "signal is not present in the program",
            )])
        })?;
        Ok(self.program.signals[index].layout.offset)
    }

    /// Sets a signal across all lanes. `lane_values` is indexed by global lane:
    /// `0..gpu_lanes` go to the GPU, the rest to the CPU.
    pub fn set_signal(&mut self, signal: Signal, lane_values: &[u128]) -> Result<(), ErrorReport> {
        let offset = self.offset(signal)?;
        if let Some(gpu) = &self.gpu {
            let gpu_vals: Vec<u32> = lane_values[..self.gpu_lanes]
                .iter()
                .map(|&v| v as u32)
                .collect();
            gpu.set_signal(offset, &gpu_vals);
        }
        if let Some(cpu) = &mut self.cpu {
            cpu.set_signal(signal, &lane_values[self.gpu_lanes..])?;
        }
        Ok(())
    }

    /// Advances `steps` cycles. The GPU runs asynchronously while the CPU
    /// simulates its lanes on this thread; we join by synchronizing the GPU.
    pub fn tick_many(&mut self, steps: usize) -> Result<(), ErrorReport> {
        if let Some(gpu) = &self.gpu {
            gpu.tick_many(steps); // async: returns before the GPU is done
        }
        if let Some(cpu) = &mut self.cpu {
            cpu.tick_many(steps)?; // runs concurrently with the GPU
        }
        if let Some(gpu) = &self.gpu {
            gpu.synchronize();
        }
        Ok(())
    }

    /// Reads a signal across all lanes (GPU lanes first, then CPU lanes).
    pub fn get_signal(&self, signal: Signal) -> Result<Vec<u128>, ErrorReport> {
        let offset = self.offset(signal)?;
        let mut out = Vec::with_capacity(self.total_lanes());
        if let Some(gpu) = &self.gpu {
            out.extend(gpu.get_signal(offset).into_iter().map(|v| v as u128));
        }
        if let Some(cpu) = &self.cpu {
            out.extend(cpu.get_signal(signal)?);
        }
        Ok(out)
    }
}
