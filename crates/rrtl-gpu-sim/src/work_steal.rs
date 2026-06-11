//! Work-stealing CPU+GPU batch executor.
//!
//! A batch workload runs the same design over `total_lanes` independent stimulus
//! lanes for `steps` cycles. This executor splits the lanes into *tiles* drawn
//! from a shared atomic cursor: the GPU and CPU each loop grabbing the next tile
//! (the GPU a large one, the CPU a small one), simulating it from reset, and
//! recording its outputs. The faster device grabs more tiles, so the split
//! self-balances across the crossover, thermal drift, and estimation error — no
//! static proportional guess. The GPU's submissions are async, so the CPU's
//! tiles run on the calling thread concurrently with GPU tiles.
//!
//! Results are per-lane identical to running every lane on the CPU (validated by
//! the work_steal_check example).

use std::sync::atomic::{AtomicUsize, Ordering};

use rrtl_ir::{ErrorReport, Signal};
use rrtl_sim_ir::{PackedProgram, PackedSimulatorStorage, SimdCpuSimulator};

use crate::interp::{InterpGpuSimulator, InterpProgram};

/// One tile's outputs: `[output_index][local_lane]` for the lane range
/// `start..start+lanes`.
struct TileOut {
    start: usize,
    lanes: usize,
    outputs: Vec<Vec<u128>>,
}

/// How many tiles each device processed in the last `run`.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkStealStats {
    pub gpu_tiles: usize,
    pub cpu_tiles: usize,
    pub gpu_lanes: usize,
    pub cpu_lanes: usize,
}

pub struct WorkStealingBatch {
    program: PackedProgram,
    gpu: InterpGpuSimulator,
    cpu: SimdCpuSimulator,
    cpu_initial: PackedSimulatorStorage,
    gpu_tile: usize,
    cpu_tile: usize,
    total_lanes: usize,
    inputs: Vec<(Signal, Vec<u128>)>,
    stats: WorkStealStats,
}

fn offset_of(program: &PackedProgram, signal: Signal) -> usize {
    let index = program.signal_index(signal).expect("signal in program");
    program.signals[index].layout.offset
}

impl WorkStealingBatch {
    pub fn new(
        program: PackedProgram,
        total_lanes: usize,
        gpu_tile: usize,
        cpu_tile: usize,
    ) -> Result<Self, ErrorReport> {
        let encoded = InterpProgram::encode_design(&program)?;
        let gpu = InterpGpuSimulator::new(&encoded, gpu_tile)?;
        let cpu = SimdCpuSimulator::new(program.clone(), cpu_tile)?;
        let cpu_initial = cpu.snapshot_storage();
        Ok(Self {
            program,
            gpu,
            cpu,
            cpu_initial,
            gpu_tile,
            cpu_tile,
            total_lanes,
            inputs: Vec::new(),
            stats: WorkStealStats::default(),
        })
    }

    /// Sets a per-lane input stimulus (length `total_lanes`) applied to each tile.
    pub fn set_input(&mut self, signal: Signal, lane_values: Vec<u128>) {
        assert_eq!(lane_values.len(), self.total_lanes, "input must cover all lanes");
        self.inputs.retain(|(s, _)| *s != signal);
        self.inputs.push((signal, lane_values));
    }

    pub fn stats(&self) -> WorkStealStats {
        self.stats
    }

    /// Runs the batch and returns each output signal's values across all lanes.
    pub fn run(&mut self, steps: usize, outputs: &[Signal]) -> Vec<Vec<u128>> {
        let cursor = AtomicUsize::new(0);
        let program = &self.program;
        let inputs = &self.inputs;
        let total = self.total_lanes;
        let gpu_tile = self.gpu_tile;
        let cpu_tile = self.cpu_tile;
        let gpu = &self.gpu;
        let cpu = &mut self.cpu;
        let cpu_initial = &self.cpu_initial;

        let (gpu_tiles, cpu_tiles) = std::thread::scope(|scope| {
            let gpu_handle = scope.spawn(|| {
                let mut tiles = Vec::new();
                loop {
                    let start = cursor.fetch_add(gpu_tile, Ordering::SeqCst);
                    if start >= total {
                        break;
                    }
                    let lanes = gpu_tile.min(total - start);
                    gpu.reset();
                    for (signal, values) in inputs {
                        let mut tile_in = vec![0u32; gpu_tile];
                        for (i, slot) in tile_in.iter_mut().enumerate().take(lanes) {
                            *slot = values[start + i] as u32;
                        }
                        gpu.set_signal(offset_of(program, *signal), &tile_in);
                    }
                    gpu.tick_many(steps);
                    gpu.synchronize();
                    let outs = outputs
                        .iter()
                        .map(|out| {
                            let full = gpu.get_signal(offset_of(program, *out));
                            full[..lanes].iter().map(|&v| v as u128).collect()
                        })
                        .collect();
                    tiles.push(TileOut { start, lanes, outputs: outs });
                }
                tiles
            });

            // CPU tiles on this thread, concurrent with the GPU thread.
            let mut cpu_tiles = Vec::new();
            loop {
                let start = cursor.fetch_add(cpu_tile, Ordering::SeqCst);
                if start >= total {
                    break;
                }
                let lanes = cpu_tile.min(total - start);
                cpu.restore_storage(cpu_initial).unwrap();
                for (signal, values) in inputs {
                    let mut tile_in = vec![0u128; cpu_tile];
                    tile_in[..lanes].copy_from_slice(&values[start..start + lanes]);
                    cpu.set_signal(*signal, &tile_in).unwrap();
                }
                cpu.tick_many(steps).unwrap();
                let outs = outputs
                    .iter()
                    .map(|out| cpu.get_signal(*out).unwrap()[..lanes].to_vec())
                    .collect();
                cpu_tiles.push(TileOut { start, lanes, outputs: outs });
            }

            let gpu_tiles = gpu_handle.join().unwrap();
            (gpu_tiles, cpu_tiles)
        });

        // Stats + merge.
        self.stats = WorkStealStats {
            gpu_tiles: gpu_tiles.len(),
            cpu_tiles: cpu_tiles.len(),
            gpu_lanes: gpu_tiles.iter().map(|t| t.lanes).sum(),
            cpu_lanes: cpu_tiles.iter().map(|t| t.lanes).sum(),
        };

        let mut result = vec![vec![0u128; total]; outputs.len()];
        for tile in gpu_tiles.iter().chain(cpu_tiles.iter()) {
            for (oi, values) in tile.outputs.iter().enumerate() {
                result[oi][tile.start..tile.start + tile.lanes].copy_from_slice(values);
            }
        }
        result
    }
}
