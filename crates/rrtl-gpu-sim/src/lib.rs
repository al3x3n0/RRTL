use std::collections::{HashMap, HashSet};
use std::time::Instant;

use rrtl_core::{compile, CompiledDesign, Design};
use rrtl_ir::{Diagnostic, ErrorReport, Signal, Width};
use rrtl_sim_ir::{
    analyze_machine_program, final_limb_mask, limbs, lower_to_machine_program,
    lower_to_packed_program, optimize_machine_program, PackedBlock, PackedEffect, PackedInstr,
    PackedInstrKind, PackedMachineAnalysis, PackedMachineProgram, PackedProgram, PackedReset,
    PackedScheduleOptions, PackedSimulatorStorage, PackedValueId,
};
use serde::{Deserialize, Serialize};

pub mod interp;
pub mod lane_split;

pub const WORKGROUP_SIZE: u32 = 64;
const TRACE_REPLAY_HEADER_WORDS: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuBatchOptions {
    pub schedule: PackedScheduleOptions,
    pub memory_layout: GpuMemoryLayout,
    pub workgroup_size: u32,
    pub reuse_temporaries: bool,
}

impl Default for GpuBatchOptions {
    fn default() -> Self {
        Self {
            schedule: PackedScheduleOptions {
                max_packet_width: Some(16),
                max_memory_reads_per_packet: None,
                liveness_priority: false,
            },
            memory_layout: GpuMemoryLayout::LaneMajor,
            workgroup_size: WORKGROUP_SIZE,
            reuse_temporaries: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuMemoryLayout {
    #[default]
    LaneMajor,
    WordMajor,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuAutotuneMatchMode {
    #[default]
    Exact,
    Nearest,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuAutotuneRecommendation {
    pub case: String,
    pub lanes: usize,
    pub steps: usize,
    pub schedule_cap: Option<usize>,
    pub memory_read_cap: Option<usize>,
    pub liveness_priority: bool,
    pub reuse_temporaries: bool,
    pub memory_layout: String,
    pub workgroup_size: u32,
    pub autotune_metric: String,
    pub autotune_metric_ns: Option<u128>,
    pub packed_ns: u128,
    pub gpu_tick_ns: Option<u128>,
    pub gpu_tick_many_ns: Option<u128>,
}

impl GpuAutotuneRecommendation {
    pub fn to_gpu_batch_options(&self) -> Result<GpuBatchOptions, ErrorReport> {
        validate_workgroup_size(self.workgroup_size)?;
        Ok(GpuBatchOptions {
            schedule: PackedScheduleOptions {
                max_packet_width: self.schedule_cap,
                max_memory_reads_per_packet: self.memory_read_cap,
                liveness_priority: self.liveness_priority,
            },
            memory_layout: parse_gpu_memory_layout(&self.memory_layout)?,
            workgroup_size: self.workgroup_size,
            reuse_temporaries: self.reuse_temporaries,
        })
    }
}

pub fn load_gpu_autotune_recommendations(
    reader: impl std::io::Read,
) -> serde_json::Result<Vec<GpuAutotuneRecommendation>> {
    serde_json::from_reader(reader)
}

pub fn load_gpu_autotune_recommendations_report(
    reader: impl std::io::Read,
) -> Result<Vec<GpuAutotuneRecommendation>, ErrorReport> {
    load_gpu_autotune_recommendations(reader).map_err(|err| {
        error(
            "E_GPU_AUTOTUNE_JSON",
            format!("failed to load GPU autotune recommendations: {err}"),
        )
    })
}

pub fn find_gpu_autotune_recommendation<'a>(
    items: &'a [GpuAutotuneRecommendation],
    case: &str,
    lanes: usize,
    steps: usize,
) -> Option<&'a GpuAutotuneRecommendation> {
    find_gpu_autotune_recommendation_with_mode(
        items,
        case,
        lanes,
        steps,
        GpuAutotuneMatchMode::Exact,
    )
}

pub fn find_gpu_autotune_recommendation_with_mode<'a>(
    items: &'a [GpuAutotuneRecommendation],
    case: &str,
    lanes: usize,
    steps: usize,
    mode: GpuAutotuneMatchMode,
) -> Option<&'a GpuAutotuneRecommendation> {
    match mode {
        GpuAutotuneMatchMode::Exact => items
            .iter()
            .find(|item| item.case == case && item.lanes == lanes && item.steps == steps),
        GpuAutotuneMatchMode::Nearest => items
            .iter()
            .filter(|item| item.case == case)
            .min_by_key(|item| autotune_match_key(item, lanes, steps)),
    }
}

pub fn gpu_batch_options_from_autotune_recommendations(
    items: &[GpuAutotuneRecommendation],
    case: &str,
    lanes: usize,
    steps: usize,
) -> Result<GpuBatchOptions, ErrorReport> {
    gpu_batch_options_from_autotune_recommendations_with_mode(
        items,
        case,
        lanes,
        steps,
        GpuAutotuneMatchMode::Exact,
    )
}

pub fn gpu_batch_options_from_autotune_recommendations_with_mode(
    items: &[GpuAutotuneRecommendation],
    case: &str,
    lanes: usize,
    steps: usize,
    mode: GpuAutotuneMatchMode,
) -> Result<GpuBatchOptions, ErrorReport> {
    let recommendation = find_gpu_autotune_recommendation_with_mode(
        items, case, lanes, steps, mode,
    )
    .ok_or_else(|| {
        error(
            "E_GPU_AUTOTUNE_RECOMMENDATION",
            format!("no GPU autotune recommendation for case `{case}` lanes={lanes} steps={steps}"),
        )
    })?;
    recommendation.to_gpu_batch_options()
}

fn autotune_match_key(
    recommendation: &GpuAutotuneRecommendation,
    lanes: usize,
    steps: usize,
) -> (usize, usize, u128, usize, usize) {
    (
        recommendation.lanes.abs_diff(lanes),
        recommendation.steps.abs_diff(steps),
        recommendation.autotune_metric_ns.unwrap_or(u128::MAX),
        recommendation.lanes,
        recommendation.steps,
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuShaderStats {
    pub unoptimized: PackedMachineAnalysis,
    pub optimized: PackedMachineAnalysis,
    pub unoptimized_packets: GpuShaderPacketStats,
    pub optimized_packets: GpuShaderPacketStats,
    pub unoptimized_memory: GpuShaderMemoryStats,
    pub optimized_memory: GpuShaderMemoryStats,
    pub wgsl_bytes: usize,
    pub optimized_temp_slots: usize,
    pub optimized_value_vars: usize,
    pub schedule: PackedScheduleOptions,
    pub memory_layout: GpuMemoryLayout,
    pub workgroup_size: u32,
    pub reuse_temporaries: bool,
    pub total_memory_words_per_lane: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuShaderPacketStats {
    pub async_reset_comb: usize,
    pub comb: usize,
    pub tick_next: usize,
    pub tick_commit: usize,
    pub total: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuShaderMemoryStats {
    pub async_reset_comb: GpuShaderMemoryStreamStats,
    pub comb: GpuShaderMemoryStreamStats,
    pub tick_next: GpuShaderMemoryStreamStats,
    pub tick_commit: GpuShaderMemoryStreamStats,
    pub total_reads: usize,
    pub total_writes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuShaderMemoryStreamStats {
    pub reads: usize,
    pub writes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuRegionAnalysis {
    pub streams: GpuRegionStreamsAnalysis,
    pub total: GpuRegionBlockAnalysis,
    pub recommendation: GpuRegionRecommendation,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuRegionStreamsAnalysis {
    pub async_reset_comb: GpuRegionBlockAnalysis,
    pub comb: GpuRegionBlockAnalysis,
    pub tick_next: GpuRegionBlockAnalysis,
    pub tick_commit: GpuRegionBlockAnalysis,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuRegionBlockAnalysis {
    pub packets: usize,
    pub pure_compute_regions: usize,
    pub pure_compute_packets: usize,
    pub memory_hostile_packets: usize,
    pub instr_count: usize,
    pub pure_compute_instrs: usize,
    pub wide_instrs: usize,
    pub memory_reads: usize,
    pub memory_writes: usize,
    pub max_region_packets: usize,
    pub max_region_instrs: usize,
    pub estimated_launch_work_units: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GpuRegionRecommendation {
    ComputeCandidate,
    MixedCandidate,
    MemoryBlocked,
    TooSmall,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuTraceReplayOptions {
    pub max_mismatches: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuTraceReplayPlan {
    pub steps: usize,
    pub inputs: Vec<GpuTraceInputOp>,
    pub checks: Vec<GpuTraceCheckOp>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuTraceInputOp {
    pub step: usize,
    pub signal: Signal,
    pub limb: usize,
    pub values: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuTraceCheckOp {
    pub step: usize,
    pub check_index: usize,
    pub signal: Signal,
    pub limb: usize,
    pub expected: Vec<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuTraceReplayTiming {
    pub upload_ns: u128,
    pub dispatch_ns: u128,
    pub readback_ns: u128,
    #[serde(default)]
    pub count_readback_ns: u128,
    #[serde(default)]
    pub full_readback_ns: u128,
    #[serde(default)]
    pub full_readback_words: usize,
    #[serde(default)]
    pub single_submit_ns: u128,
    #[serde(default)]
    pub single_submit_count_readback_ns: u128,
    #[serde(default)]
    pub single_submit_used: bool,
    pub total_ns: u128,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuTraceReplayReport {
    pub mismatch_count: usize,
    pub mismatches: Vec<GpuTraceMismatch>,
    pub timing: GpuTraceReplayTiming,
}

pub struct PreparedGpuTraceReplay {
    _trace_data_buffer: wgpu::Buffer,
    _params_buffer: wgpu::Buffer,
    zero_results_buffer: wgpu::Buffer,
    trace_results_buffer: wgpu::Buffer,
    trace_count_readback: wgpu::Buffer,
    trace_results_readback: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    zero_results: Vec<u32>,
    steps: u32,
    input_ops: u32,
    check_ops: u32,
    max_mismatches: usize,
    trace_data_words: usize,
    trace_data_uncompressed_words: usize,
    uniform_input_ops: usize,
    uniform_check_ops: usize,
    template_layout: bool,
    template_input_ops: usize,
    template_check_ops: usize,
    metadata_saved_words: usize,
    fixed_template: bool,
    value_metadata_saved_words: usize,
    value_stride_words: usize,
}

impl PreparedGpuTraceReplay {
    pub fn trace_data_words(&self) -> usize {
        self.trace_data_words
    }

    pub fn result_words(&self) -> usize {
        self.zero_results.len()
    }

    pub fn trace_data_bytes(&self) -> usize {
        self.trace_data_words * std::mem::size_of::<u32>()
    }

    pub fn trace_data_uncompressed_words(&self) -> usize {
        self.trace_data_uncompressed_words
    }

    pub fn trace_data_uncompressed_bytes(&self) -> usize {
        self.trace_data_uncompressed_words * std::mem::size_of::<u32>()
    }

    pub fn trace_data_compression_ratio_x100(&self) -> usize {
        if self.trace_data_uncompressed_words == 0 {
            100
        } else {
            self.trace_data_words * 100 / self.trace_data_uncompressed_words
        }
    }

    pub fn uniform_input_ops(&self) -> usize {
        self.uniform_input_ops
    }

    pub fn uniform_check_ops(&self) -> usize {
        self.uniform_check_ops
    }

    pub fn trace_layout(&self) -> &'static str {
        if self.template_layout {
            "templated"
        } else {
            "step-indexed"
        }
    }

    pub fn template_input_ops(&self) -> usize {
        self.template_input_ops
    }

    pub fn template_check_ops(&self) -> usize {
        self.template_check_ops
    }

    pub fn metadata_saved_words(&self) -> usize {
        self.metadata_saved_words
    }

    pub fn fixed_template(&self) -> bool {
        self.fixed_template
    }

    pub fn value_metadata_saved_words(&self) -> usize {
        self.value_metadata_saved_words
    }

    pub fn value_stride_words(&self) -> usize {
        self.value_stride_words
    }
}

pub struct PreparedGpuStorageSnapshot {
    values_buffer: wgpu::Buffer,
    memories_buffer: wgpu::Buffer,
    value_words: usize,
    memory_words: usize,
}

impl PreparedGpuStorageSnapshot {
    pub fn value_words(&self) -> usize {
        self.value_words
    }

    pub fn memory_words(&self) -> usize {
        self.memory_words
    }

    pub fn storage_bytes(&self) -> usize {
        (self.value_words + self.memory_words) * std::mem::size_of::<u32>()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuTraceMismatch {
    pub step: usize,
    pub check_index: usize,
    pub lane: usize,
    pub limb: usize,
    pub expected: u32,
    pub actual: u32,
}

pub struct GpuBatchSimulator {
    lanes: usize,
    program: PackedProgram,
    memory_layout: GpuMemoryLayout,
    workgroup_size: u32,
    values: Vec<u32>,
    memories: Vec<u32>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    replay_bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    eval_pipeline: wgpu::ComputePipeline,
    tick_pipeline: wgpu::ComputePipeline,
    tick_many_pipeline: wgpu::ComputePipeline,
    replay_pipeline: wgpu::ComputePipeline,
    params_buffer: wgpu::Buffer,
    values_buffer: wgpu::Buffer,
    memories_buffer: wgpu::Buffer,
    values_readback: wgpu::Buffer,
    memories_readback: wgpu::Buffer,
}

struct PackedGpuTraceReplay {
    trace_data: Vec<u32>,
    uncompressed_words: usize,
    input_ops: usize,
    check_ops: usize,
    uniform_input_ops: usize,
    uniform_check_ops: usize,
    template_layout: bool,
    template_input_ops: usize,
    template_check_ops: usize,
    metadata_saved_words: usize,
    fixed_template: bool,
    value_metadata_saved_words: usize,
    value_stride_words: usize,
}

impl GpuBatchSimulator {
    pub fn new(design: &Design, module_name: &str, lanes: usize) -> Result<Self, ErrorReport> {
        Self::new_with_options(design, module_name, lanes, GpuBatchOptions::default())
    }

    pub fn new_with_options(
        design: &Design,
        module_name: &str,
        lanes: usize,
        options: GpuBatchOptions,
    ) -> Result<Self, ErrorReport> {
        let compiled = compile(design)?;
        Self::new_from_compiled_with_options(&compiled, module_name, lanes, options)
    }

    pub fn new_from_compiled(
        design: &CompiledDesign,
        module_name: &str,
        lanes: usize,
    ) -> Result<Self, ErrorReport> {
        Self::new_from_compiled_with_options(design, module_name, lanes, GpuBatchOptions::default())
    }

    pub fn new_from_compiled_with_options(
        design: &CompiledDesign,
        module_name: &str,
        lanes: usize,
        options: GpuBatchOptions,
    ) -> Result<Self, ErrorReport> {
        if lanes == 0 {
            return Err(error(
                "E_GPU_LANES",
                "GPU simulator requires at least one lane",
            ));
        }
        validate_workgroup_size(options.workgroup_size)?;
        let program = lower_to_packed_program(design, module_name)?;
        pollster::block_on(Self::new_async(program, lanes, options))
    }

    pub fn new_with_autotune_recommendations(
        design: &Design,
        module_name: &str,
        recommendation_case: &str,
        lanes: usize,
        steps: usize,
        recommendations: &[GpuAutotuneRecommendation],
    ) -> Result<Self, ErrorReport> {
        let options = gpu_batch_options_from_autotune_recommendations(
            recommendations,
            recommendation_case,
            lanes,
            steps,
        )?;
        Self::new_with_options(design, module_name, lanes, options)
    }

    pub fn new_with_autotune_recommendations_with_mode(
        design: &Design,
        module_name: &str,
        recommendation_case: &str,
        lanes: usize,
        steps: usize,
        recommendations: &[GpuAutotuneRecommendation],
        mode: GpuAutotuneMatchMode,
    ) -> Result<Self, ErrorReport> {
        let options = gpu_batch_options_from_autotune_recommendations_with_mode(
            recommendations,
            recommendation_case,
            lanes,
            steps,
            mode,
        )?;
        Self::new_with_options(design, module_name, lanes, options)
    }

    pub fn new_with_autotune_recommendation_reader(
        design: &Design,
        module_name: &str,
        recommendation_case: &str,
        lanes: usize,
        steps: usize,
        reader: impl std::io::Read,
    ) -> Result<Self, ErrorReport> {
        let recommendations = load_gpu_autotune_recommendations_report(reader)?;
        Self::new_with_autotune_recommendations(
            design,
            module_name,
            recommendation_case,
            lanes,
            steps,
            &recommendations,
        )
    }

    pub fn new_with_autotune_recommendation_reader_with_mode(
        design: &Design,
        module_name: &str,
        recommendation_case: &str,
        lanes: usize,
        steps: usize,
        reader: impl std::io::Read,
        mode: GpuAutotuneMatchMode,
    ) -> Result<Self, ErrorReport> {
        let recommendations = load_gpu_autotune_recommendations_report(reader)?;
        Self::new_with_autotune_recommendations_with_mode(
            design,
            module_name,
            recommendation_case,
            lanes,
            steps,
            &recommendations,
            mode,
        )
    }

    async fn new_async(
        program: PackedProgram,
        lanes: usize,
        options: GpuBatchOptions,
    ) -> Result<Self, ErrorReport> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| error("E_GPU_ADAPTER", "no suitable GPU adapter found"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rrtl-gpu-sim-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .map_err(|err| {
                error(
                    "E_GPU_DEVICE",
                    format!("failed to create GPU device: {err}"),
                )
            })?;

        let value_words = program.total_signal_words * lanes;
        let memory_words = program.total_memory_words_per_lane * lanes;
        let values = vec![0; value_words];
        let memories = vec![0; memory_words];

        let values_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rrtl-gpu-sim-values"),
            size: buffer_size_nonzero(value_words),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let memories_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rrtl-gpu-sim-memories"),
            size: buffer_size_nonzero(memory_words),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let values_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rrtl-gpu-sim-values-readback"),
            size: buffer_size_nonzero(value_words),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let memories_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rrtl-gpu-sim-memories-readback"),
            size: buffer_size_nonzero(memory_words),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&values_buffer, 0, bytemuck::cast_slice(&values));
        if !memories.is_empty() {
            queue.write_buffer(&memories_buffer, 0, bytemuck::cast_slice(&memories));
        }

        let params = [
            lanes as u32,
            program.total_signal_words as u32,
            program.total_memory_words_per_lane as u32,
            0,
            0,
            0,
            0,
            0,
        ];
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rrtl-gpu-sim-params"),
            size: buffer_size(params.len()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rrtl-gpu-sim-bind-layout"),
            entries: &[
                storage_entry(0),
                storage_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let replay_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rrtl-gpu-sim-replay-bind-layout"),
                entries: &[
                    storage_entry(0),
                    storage_entry(1),
                    storage_entry(3),
                    storage_entry(4),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rrtl-gpu-sim-bind-group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: values_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: memories_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let wgsl = generate_wgsl(&program, options)?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rrtl-gpu-sim-shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rrtl-gpu-sim-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let replay_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rrtl-gpu-sim-replay-pipeline-layout"),
                bind_group_layouts: &[&replay_bind_group_layout],
                push_constant_ranges: &[],
            });
        let eval_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rrtl-gpu-sim-eval"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "eval_comb",
        });
        let tick_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rrtl-gpu-sim-tick"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "tick",
        });
        let tick_many_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rrtl-gpu-sim-tick-many"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "tick_many",
        });
        let replay_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rrtl-gpu-sim-replay-trace"),
            layout: Some(&replay_pipeline_layout),
            module: &shader,
            entry_point: "replay_trace",
        });

        Ok(Self {
            lanes,
            program,
            memory_layout: options.memory_layout,
            workgroup_size: options.workgroup_size,
            values,
            memories,
            device,
            queue,
            replay_bind_group_layout,
            bind_group,
            eval_pipeline,
            tick_pipeline,
            tick_many_pipeline,
            replay_pipeline,
            params_buffer,
            values_buffer,
            memories_buffer,
            values_readback,
            memories_readback,
        })
    }

    pub fn set_input(&mut self, signal: Signal, lane_values: &[u32]) -> Result<(), ErrorReport> {
        let index = self.signal_index(signal)?;
        if self.program.signals[index].layout.width > 32 {
            return Err(error(
                "E_GPU_WIDE_SIGNAL",
                "use set_input_limbs for signals wider than 32 bits",
            ));
        }
        let lane_limbs = lane_values
            .iter()
            .copied()
            .map(|value| vec![value])
            .collect::<Vec<_>>();
        self.set_signal_limbs(index, &lane_limbs)
    }

    pub fn set_input_limbs(
        &mut self,
        signal: Signal,
        lane_values: &[Vec<u32>],
    ) -> Result<(), ErrorReport> {
        let index = self.signal_index(signal)?;
        self.set_signal_limbs(index, lane_values)
    }

    pub fn get_signal(&mut self, signal: Signal) -> Result<Vec<u32>, ErrorReport> {
        let index = self.signal_index(signal)?;
        if self.program.signals[index].layout.width > 32 {
            return Err(error(
                "E_GPU_WIDE_SIGNAL",
                "use get_signal_limbs for signals wider than 32 bits",
            ));
        }
        Ok(self
            .get_signal_limbs(signal)?
            .into_iter()
            .map(|limbs| limbs.first().copied().unwrap_or(0))
            .collect())
    }

    pub fn get_signal_limbs(&mut self, signal: Signal) -> Result<Vec<Vec<u32>>, ErrorReport> {
        self.sync_values_from_gpu()?;
        let index = self.signal_index(signal)?;
        let layout = self.program.signals[index].layout;
        let mut out = Vec::with_capacity(self.lanes);
        for lane in 0..self.lanes {
            let mut lane_values = Vec::with_capacity(layout.limbs);
            for limb in 0..layout.limbs {
                lane_values.push(self.values[self.value_index(layout.offset, limb, lane)]);
            }
            if let Some(last) = lane_values.last_mut() {
                *last &= final_limb_mask(layout.width);
            }
            out.push(lane_values);
        }
        Ok(out)
    }

    pub fn set_memory(
        &mut self,
        memory: Signal,
        lane_words: &[Vec<u32>],
    ) -> Result<(), ErrorReport> {
        let index = self.memory_index(memory)?;
        if self.program.memories[index].data_layout.width > 32 {
            return Err(error(
                "E_GPU_WIDE_MEMORY",
                "use set_memory_limbs for memories wider than 32 bits",
            ));
        }
        let lane_limbs = lane_words
            .iter()
            .map(|words| {
                words
                    .iter()
                    .copied()
                    .map(|word| vec![word])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        self.set_memory_limbs(memory, &lane_limbs)
    }

    pub fn set_memory_limbs(
        &mut self,
        memory: Signal,
        lane_words: &[Vec<Vec<u32>>],
    ) -> Result<(), ErrorReport> {
        let index = self.memory_index(memory)?;
        self.set_memory_index_limbs(index, lane_words)
    }

    pub fn get_memory(&mut self, memory: Signal) -> Result<Vec<Vec<u32>>, ErrorReport> {
        let index = self.memory_index(memory)?;
        if self.program.memories[index].data_layout.width > 32 {
            return Err(error(
                "E_GPU_WIDE_MEMORY",
                "use get_memory_limbs for memories wider than 32 bits",
            ));
        }
        Ok(self
            .get_memory_limbs(memory)?
            .into_iter()
            .map(|lane| {
                lane.into_iter()
                    .map(|word| word.first().copied().unwrap_or(0))
                    .collect()
            })
            .collect())
    }

    pub fn get_memory_limbs(&mut self, memory: Signal) -> Result<Vec<Vec<Vec<u32>>>, ErrorReport> {
        self.sync_memories_from_gpu()?;
        let index = self.memory_index(memory)?;
        let memory = self.program.memories[index].clone();
        let mut out = Vec::with_capacity(self.lanes);
        for lane in 0..self.lanes {
            let mut lane_words = Vec::with_capacity(memory.depth);
            for addr in 0..memory.depth {
                let mut word = Vec::with_capacity(memory.data_layout.limbs);
                for limb in 0..memory.data_layout.limbs {
                    word.push(self.memories[self.memory_word_index(&memory, lane, addr, limb)]);
                }
                if let Some(last) = word.last_mut() {
                    *last &= final_limb_mask(memory.data_layout.width);
                }
                lane_words.push(word);
            }
            out.push(lane_words);
        }
        Ok(out)
    }

    pub fn eval_combinational(&mut self) -> Result<(), ErrorReport> {
        self.dispatch(&self.eval_pipeline);
        self.sync_values_from_gpu()
    }

    pub fn eval_combinational_many(&mut self, iterations: usize) -> Result<(), ErrorReport> {
        if iterations == 0 {
            return Ok(());
        }
        self.dispatch_pipeline_repeated(&self.eval_pipeline, iterations);
        self.sync_values_from_gpu()
    }

    pub fn tick(&mut self) -> Result<(), ErrorReport> {
        self.dispatch(&self.eval_pipeline);
        self.dispatch(&self.tick_pipeline);
        self.dispatch(&self.eval_pipeline);
        self.sync_values_from_gpu()
    }

    pub fn tick_many(&mut self, steps: usize) -> Result<(), ErrorReport> {
        if steps == 0 {
            return Ok(());
        }
        let steps = u32::try_from(steps).map_err(|_| {
            error(
                "E_GPU_STEPS",
                "tick_many step count exceeds the GPU shader parameter range",
            )
        })?;
        self.write_params(steps);
        self.dispatch(&self.tick_many_pipeline);
        self.write_params(0);
        self.sync_values_from_gpu()
    }

    pub fn replay_lane_trace(
        &mut self,
        plan: &GpuTraceReplayPlan,
        options: GpuTraceReplayOptions,
    ) -> Result<GpuTraceReplayReport, ErrorReport> {
        let total_start = Instant::now();
        let prepare_start = Instant::now();
        let prepared = self.prepare_lane_trace_replay(plan, options.clone())?;
        let prepare_ns = prepare_start.elapsed().as_nanos();
        let mut report = self.replay_prepared_lane_trace(&prepared)?;
        report.timing.upload_ns += prepare_ns;
        report.timing.total_ns = total_start.elapsed().as_nanos();
        Ok(report)
    }

    pub fn prepare_lane_trace_replay(
        &mut self,
        plan: &GpuTraceReplayPlan,
        options: GpuTraceReplayOptions,
    ) -> Result<PreparedGpuTraceReplay, ErrorReport> {
        let max_mismatches = options.max_mismatches.max(1);
        let steps = u32::try_from(plan.steps).map_err(|_| {
            error(
                "E_GPU_TRACE_STEPS",
                "trace replay step count exceeds the GPU shader parameter range",
            )
        })?;
        let packed = self.pack_trace_replay(plan)?;
        let input_ops = u32::try_from(packed.input_ops).map_err(|_| {
            error(
                "E_GPU_TRACE_INPUTS",
                "trace replay input op count exceeds the GPU shader parameter range",
            )
        })?;
        let check_ops = u32::try_from(packed.check_ops).map_err(|_| {
            error(
                "E_GPU_TRACE_CHECKS",
                "trace replay check op count exceeds the GPU shader parameter range",
            )
        })?;
        let result_words = 1 + max_mismatches * 6;
        let zero_results = vec![0u32; result_words];
        let params = [
            self.lanes as u32,
            self.program.total_signal_words as u32,
            self.program.total_memory_words_per_lane as u32,
            steps,
            input_ops,
            check_ops,
            max_mismatches as u32,
            0,
        ];

        let trace_data_buffer =
            self.storage_buffer_with_data("rrtl-gpu-trace-data-prepared", &packed.trace_data);
        let params_buffer =
            self.copy_src_buffer_with_data("rrtl-gpu-trace-params-prepared", &params);
        let zero_results_buffer =
            self.copy_src_buffer_with_data("rrtl-gpu-trace-results-zero-prepared", &zero_results);
        let trace_results_buffer =
            self.storage_buffer_with_data("rrtl-gpu-trace-results-prepared", &zero_results);
        let trace_count_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rrtl-gpu-trace-count-readback-prepared"),
            size: buffer_size_nonzero(1),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let trace_results_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rrtl-gpu-trace-results-readback-prepared"),
            size: buffer_size_nonzero(result_words),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rrtl-gpu-trace-bind-group-prepared"),
            layout: &self.replay_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.values_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.memories_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: trace_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: trace_results_buffer.as_entire_binding(),
                },
            ],
        });
        Ok(PreparedGpuTraceReplay {
            _trace_data_buffer: trace_data_buffer,
            _params_buffer: params_buffer,
            zero_results_buffer,
            trace_results_buffer,
            trace_count_readback,
            trace_results_readback,
            bind_group,
            zero_results,
            steps,
            input_ops,
            check_ops,
            max_mismatches,
            trace_data_words: packed.trace_data.len(),
            trace_data_uncompressed_words: packed.uncompressed_words,
            uniform_input_ops: packed.uniform_input_ops,
            uniform_check_ops: packed.uniform_check_ops,
            template_layout: packed.template_layout,
            template_input_ops: packed.template_input_ops,
            template_check_ops: packed.template_check_ops,
            metadata_saved_words: packed.metadata_saved_words,
            fixed_template: packed.fixed_template,
            value_metadata_saved_words: packed.value_metadata_saved_words,
            value_stride_words: packed.value_stride_words,
        })
    }

    pub fn replay_prepared_lane_trace(
        &mut self,
        prepared: &PreparedGpuTraceReplay,
    ) -> Result<GpuTraceReplayReport, ErrorReport> {
        let total_start = Instant::now();
        let upload_start = Instant::now();
        let params = [
            self.lanes as u32,
            self.program.total_signal_words as u32,
            self.program.total_memory_words_per_lane as u32,
            prepared.steps,
            prepared.input_ops,
            prepared.check_ops,
            prepared.max_mismatches as u32,
            0,
        ];
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));
        self.queue.write_buffer(
            &prepared.trace_results_buffer,
            0,
            bytemuck::cast_slice(&prepared.zero_results),
        );
        let upload_ns = upload_start.elapsed().as_nanos();

        let dispatch_start = Instant::now();
        self.dispatch_with_bind_group(&self.replay_pipeline, &prepared.bind_group);
        let dispatch_ns = dispatch_start.elapsed().as_nanos();

        let count_readback_start = Instant::now();
        let mismatch_count = self.read_trace_result_count(
            &prepared.trace_results_buffer,
            &prepared.trace_count_readback,
        )? as usize;
        let count_readback_ns = count_readback_start.elapsed().as_nanos();
        self.write_params(0);
        if mismatch_count == 0 {
            return Ok(GpuTraceReplayReport {
                mismatch_count,
                mismatches: Vec::new(),
                timing: GpuTraceReplayTiming {
                    upload_ns,
                    dispatch_ns,
                    readback_ns: count_readback_ns,
                    count_readback_ns,
                    full_readback_ns: 0,
                    full_readback_words: 0,
                    single_submit_ns: 0,
                    single_submit_count_readback_ns: 0,
                    single_submit_used: false,
                    total_ns: total_start.elapsed().as_nanos(),
                },
            });
        }

        let full_readback_start = Instant::now();
        let result_words = self.read_trace_results(
            &prepared.trace_results_buffer,
            &prepared.trace_results_readback,
            prepared.result_words(),
        )?;
        let full_readback_ns = full_readback_start.elapsed().as_nanos();
        let readback_ns = count_readback_ns + full_readback_ns;
        let stored = mismatch_count.min(prepared.max_mismatches);
        let mut mismatches = Vec::with_capacity(stored);
        for index in 0..stored {
            let base = 1 + index * 6;
            if base + 5 >= result_words.len() {
                break;
            }
            mismatches.push(GpuTraceMismatch {
                step: result_words[base] as usize,
                check_index: result_words[base + 1] as usize,
                lane: result_words[base + 2] as usize,
                limb: result_words[base + 3] as usize,
                expected: result_words[base + 4],
                actual: result_words[base + 5],
            });
        }

        Ok(GpuTraceReplayReport {
            mismatch_count,
            mismatches,
            timing: GpuTraceReplayTiming {
                upload_ns,
                dispatch_ns,
                readback_ns,
                count_readback_ns,
                full_readback_ns,
                full_readback_words: result_words.len(),
                single_submit_ns: 0,
                single_submit_count_readback_ns: 0,
                single_submit_used: false,
                total_ns: total_start.elapsed().as_nanos(),
            },
        })
    }

    pub fn replay_prepared_lane_trace_from_snapshot(
        &mut self,
        prepared: &PreparedGpuTraceReplay,
        snapshot: &PreparedGpuStorageSnapshot,
    ) -> Result<GpuTraceReplayReport, ErrorReport> {
        let expected_values = self.program.total_signal_words * self.lanes;
        if snapshot.value_words != expected_values {
            return Err(error(
                "E_GPU_STORAGE_VALUES",
                format!(
                    "expected {expected_values} packed signal words, got {}",
                    snapshot.value_words
                ),
            ));
        }
        let expected_memories = self.program.total_memory_words_per_lane * self.lanes;
        if snapshot.memory_words != expected_memories {
            return Err(error(
                "E_GPU_STORAGE_MEMORIES",
                format!(
                    "expected {expected_memories} packed memory words, got {}",
                    snapshot.memory_words
                ),
            ));
        }

        let total_start = Instant::now();
        let submit_start = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rrtl-gpu-sim-single-submit-replay"),
            });
        if snapshot.value_words > 0 {
            encoder.copy_buffer_to_buffer(
                &snapshot.values_buffer,
                0,
                &self.values_buffer,
                0,
                buffer_size(snapshot.value_words),
            );
        }
        if snapshot.memory_words > 0 {
            encoder.copy_buffer_to_buffer(
                &snapshot.memories_buffer,
                0,
                &self.memories_buffer,
                0,
                buffer_size(snapshot.memory_words),
            );
        }
        encoder.copy_buffer_to_buffer(
            &prepared._params_buffer,
            0,
            &self.params_buffer,
            0,
            buffer_size(8),
        );
        encoder.copy_buffer_to_buffer(
            &prepared.zero_results_buffer,
            0,
            &prepared.trace_results_buffer,
            0,
            buffer_size(prepared.result_words()),
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rrtl-gpu-sim-single-submit-replay-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.replay_pipeline);
            pass.set_bind_group(0, &prepared.bind_group, &[]);
            pass.dispatch_workgroups(div_ceil(self.lanes as u32, self.workgroup_size), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &prepared.trace_results_buffer,
            0,
            &prepared.trace_count_readback,
            0,
            buffer_size(1),
        );
        self.queue.submit(Some(encoder.finish()));
        let mismatch_count = self.map_trace_result_count(&prepared.trace_count_readback)? as usize;
        let single_submit_count_readback_ns = submit_start.elapsed().as_nanos();
        let single_submit_ns = single_submit_count_readback_ns;
        if mismatch_count == 0 {
            return Ok(GpuTraceReplayReport {
                mismatch_count,
                mismatches: Vec::new(),
                timing: GpuTraceReplayTiming {
                    upload_ns: 0,
                    dispatch_ns: single_submit_ns,
                    readback_ns: single_submit_count_readback_ns,
                    count_readback_ns: single_submit_count_readback_ns,
                    full_readback_ns: 0,
                    full_readback_words: 0,
                    single_submit_ns,
                    single_submit_count_readback_ns,
                    single_submit_used: true,
                    total_ns: total_start.elapsed().as_nanos(),
                },
            });
        }

        let full_readback_start = Instant::now();
        let result_words = self.read_trace_results(
            &prepared.trace_results_buffer,
            &prepared.trace_results_readback,
            prepared.result_words(),
        )?;
        let full_readback_ns = full_readback_start.elapsed().as_nanos();
        let readback_ns = single_submit_count_readback_ns + full_readback_ns;
        let stored = mismatch_count.min(prepared.max_mismatches);
        let mut mismatches = Vec::with_capacity(stored);
        for index in 0..stored {
            let base = 1 + index * 6;
            if base + 5 >= result_words.len() {
                break;
            }
            mismatches.push(GpuTraceMismatch {
                step: result_words[base] as usize,
                check_index: result_words[base + 1] as usize,
                lane: result_words[base + 2] as usize,
                limb: result_words[base + 3] as usize,
                expected: result_words[base + 4],
                actual: result_words[base + 5],
            });
        }
        Ok(GpuTraceReplayReport {
            mismatch_count,
            mismatches,
            timing: GpuTraceReplayTiming {
                upload_ns: 0,
                dispatch_ns: single_submit_ns,
                readback_ns,
                count_readback_ns: single_submit_count_readback_ns,
                full_readback_ns,
                full_readback_words: result_words.len(),
                single_submit_ns,
                single_submit_count_readback_ns,
                single_submit_used: true,
                total_ns: total_start.elapsed().as_nanos(),
            },
        })
    }

    pub fn program(&self) -> &PackedProgram {
        &self.program
    }

    pub fn snapshot_storage(&mut self) -> Result<PackedSimulatorStorage, ErrorReport> {
        self.sync_values_from_gpu()?;
        self.sync_memories_from_gpu()?;
        Ok(PackedSimulatorStorage {
            values: self.values.clone(),
            memories: self.snapshot_memories_lane_major(),
        })
    }

    pub fn restore_storage(&mut self, storage: &PackedSimulatorStorage) -> Result<(), ErrorReport> {
        let expected_values = self.program.total_signal_words * self.lanes;
        if storage.values.len() != expected_values {
            return Err(error(
                "E_GPU_STORAGE_VALUES",
                format!(
                    "expected {expected_values} packed signal words, got {}",
                    storage.values.len()
                ),
            ));
        }
        let expected_memories = self.program.total_memory_words_per_lane * self.lanes;
        if storage.memories.len() != expected_memories {
            return Err(error(
                "E_GPU_STORAGE_MEMORIES",
                format!(
                    "expected {expected_memories} packed memory words, got {}",
                    storage.memories.len()
                ),
            ));
        }

        self.values.clone_from(&storage.values);
        self.memories = self.storage_memories_from_lane_major(&storage.memories);
        if !self.values.is_empty() {
            self.queue
                .write_buffer(&self.values_buffer, 0, bytemuck::cast_slice(&self.values));
        }
        if !self.memories.is_empty() {
            self.queue.write_buffer(
                &self.memories_buffer,
                0,
                bytemuck::cast_slice(&self.memories),
            );
        }
        Ok(())
    }

    pub fn prepare_storage_snapshot(&self) -> PreparedGpuStorageSnapshot {
        let value_words = self.program.total_signal_words * self.lanes;
        let memory_words = self.program.total_memory_words_per_lane * self.lanes;
        let values_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rrtl-gpu-sim-values-snapshot"),
            size: buffer_size_nonzero(value_words),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let memories_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rrtl-gpu-sim-memories-snapshot"),
            size: buffer_size_nonzero(memory_words),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rrtl-gpu-sim-prepare-storage-snapshot"),
            });
        if value_words > 0 {
            encoder.copy_buffer_to_buffer(
                &self.values_buffer,
                0,
                &values_buffer,
                0,
                buffer_size(value_words),
            );
        }
        if memory_words > 0 {
            encoder.copy_buffer_to_buffer(
                &self.memories_buffer,
                0,
                &memories_buffer,
                0,
                buffer_size(memory_words),
            );
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        PreparedGpuStorageSnapshot {
            values_buffer,
            memories_buffer,
            value_words,
            memory_words,
        }
    }

    pub fn restore_prepared_storage(
        &mut self,
        snapshot: &PreparedGpuStorageSnapshot,
    ) -> Result<(), ErrorReport> {
        let expected_values = self.program.total_signal_words * self.lanes;
        if snapshot.value_words != expected_values {
            return Err(error(
                "E_GPU_STORAGE_VALUES",
                format!(
                    "expected {expected_values} packed signal words, got {}",
                    snapshot.value_words
                ),
            ));
        }
        let expected_memories = self.program.total_memory_words_per_lane * self.lanes;
        if snapshot.memory_words != expected_memories {
            return Err(error(
                "E_GPU_STORAGE_MEMORIES",
                format!(
                    "expected {expected_memories} packed memory words, got {}",
                    snapshot.memory_words
                ),
            ));
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rrtl-gpu-sim-restore-storage-snapshot"),
            });
        if snapshot.value_words > 0 {
            encoder.copy_buffer_to_buffer(
                &snapshot.values_buffer,
                0,
                &self.values_buffer,
                0,
                buffer_size(snapshot.value_words),
            );
        }
        if snapshot.memory_words > 0 {
            encoder.copy_buffer_to_buffer(
                &snapshot.memories_buffer,
                0,
                &self.memories_buffer,
                0,
                buffer_size(snapshot.memory_words),
            );
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    fn set_signal_limbs(
        &mut self,
        index: usize,
        lane_values: &[Vec<u32>],
    ) -> Result<(), ErrorReport> {
        if lane_values.len() != self.lanes {
            return Err(error(
                "E_GPU_LANE_VALUES",
                format!(
                    "expected {} lane values, got {}",
                    self.lanes,
                    lane_values.len()
                ),
            ));
        }
        let layout = self.program.signals[index].layout;
        for (lane, value) in lane_values.iter().enumerate() {
            if value.len() != layout.limbs {
                return Err(error(
                    "E_GPU_LANE_VALUES",
                    format!("expected {} limbs, got {}", layout.limbs, value.len()),
                ));
            }
            for (limb, limb_value) in value.iter().copied().enumerate() {
                let mut stored = limb_value;
                if limb + 1 == layout.limbs {
                    stored &= final_limb_mask(layout.width);
                }
                let value_index = self.value_index(layout.offset, limb, lane);
                self.values[value_index] = stored;
            }
        }
        for limb in 0..layout.limbs {
            let start = self.value_index(layout.offset, limb, 0);
            self.queue.write_buffer(
                &self.values_buffer,
                byte_offset(start),
                bytemuck::cast_slice(&self.values[start..start + self.lanes]),
            );
        }
        Ok(())
    }

    fn set_memory_index_limbs(
        &mut self,
        index: usize,
        lane_words: &[Vec<Vec<u32>>],
    ) -> Result<(), ErrorReport> {
        if lane_words.len() != self.lanes {
            return Err(error(
                "E_GPU_MEMORY_VALUES",
                format!(
                    "expected {} lanes of memory values, got {}",
                    self.lanes,
                    lane_words.len()
                ),
            ));
        }
        let memory = self.program.memories[index].clone();
        for (lane, words) in lane_words.iter().enumerate() {
            if words.len() != memory.depth {
                return Err(error(
                    "E_GPU_MEMORY_VALUES",
                    format!(
                        "expected {} memory words, got {}",
                        memory.depth,
                        words.len()
                    ),
                ));
            }
            for (addr, word) in words.iter().enumerate() {
                if word.len() != memory.data_layout.limbs {
                    return Err(error(
                        "E_GPU_MEMORY_VALUES",
                        format!(
                            "expected {} limbs per memory word, got {}",
                            memory.data_layout.limbs,
                            word.len()
                        ),
                    ));
                }
                for (limb, limb_value) in word.iter().copied().enumerate() {
                    let mut stored = limb_value;
                    if limb + 1 == memory.data_layout.limbs {
                        stored &= final_limb_mask(memory.data_layout.width);
                    }
                    let memory_index = self.memory_word_index(&memory, lane, addr, limb);
                    self.memories[memory_index] = stored;
                }
            }
        }
        if !self.memories.is_empty() {
            self.queue.write_buffer(
                &self.memories_buffer,
                0,
                bytemuck::cast_slice(&self.memories),
            );
        }
        Ok(())
    }

    fn signal_index(&self, signal: Signal) -> Result<usize, ErrorReport> {
        self.program.signal_index(signal).ok_or_else(|| {
            error(
                "E_GPU_SIGNAL",
                format!("signal {:?} is not part of this GPU module", signal.id),
            )
        })
    }

    fn memory_index(&self, memory: Signal) -> Result<usize, ErrorReport> {
        self.program
            .memories
            .iter()
            .position(|packed| packed.source == memory)
            .ok_or_else(|| {
                error(
                    "E_GPU_MEMORY",
                    format!("memory {:?} is not part of this GPU module", memory.id),
                )
            })
    }

    fn value_index(&self, offset: usize, limb: usize, lane: usize) -> usize {
        (offset + limb) * self.lanes + lane
    }

    fn memory_word_index(
        &self,
        memory: &rrtl_sim_ir::PackedMemory,
        lane: usize,
        addr: usize,
        limb: usize,
    ) -> usize {
        let word = memory.offset + addr * memory.data_layout.limbs + limb;
        match self.memory_layout {
            GpuMemoryLayout::LaneMajor => lane * self.program.total_memory_words_per_lane + word,
            GpuMemoryLayout::WordMajor => word * self.lanes + lane,
        }
    }

    fn lane_major_memory_word_index(
        &self,
        memory: &rrtl_sim_ir::PackedMemory,
        lane: usize,
        addr: usize,
        limb: usize,
    ) -> usize {
        lane * self.program.total_memory_words_per_lane
            + memory.offset
            + addr * memory.data_layout.limbs
            + limb
    }

    fn snapshot_memories_lane_major(&self) -> Vec<u32> {
        let mut out = vec![0; self.program.total_memory_words_per_lane * self.lanes];
        for memory in &self.program.memories {
            for lane in 0..self.lanes {
                for addr in 0..memory.depth {
                    for limb in 0..memory.data_layout.limbs {
                        let src = self.memory_word_index(memory, lane, addr, limb);
                        let dst = self.lane_major_memory_word_index(memory, lane, addr, limb);
                        out[dst] = self.memories[src];
                    }
                }
            }
        }
        out
    }

    fn storage_memories_from_lane_major(&self, lane_major: &[u32]) -> Vec<u32> {
        let mut out = vec![0; self.program.total_memory_words_per_lane * self.lanes];
        for memory in &self.program.memories {
            for lane in 0..self.lanes {
                for addr in 0..memory.depth {
                    for limb in 0..memory.data_layout.limbs {
                        let src = self.lane_major_memory_word_index(memory, lane, addr, limb);
                        let dst = self.memory_word_index(memory, lane, addr, limb);
                        out[dst] = lane_major[src];
                    }
                }
            }
        }
        out
    }

    fn pack_trace_replay(
        &self,
        plan: &GpuTraceReplayPlan,
    ) -> Result<PackedGpuTraceReplay, ErrorReport> {
        pack_trace_replay_for_program(&self.program, self.lanes, plan)
    }

    fn storage_buffer_with_data(&self, label: &'static str, words: &[u32]) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: buffer_size_nonzero(words.len()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if !words.is_empty() {
            self.queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(words));
        }
        buffer
    }

    fn copy_src_buffer_with_data(&self, label: &'static str, words: &[u32]) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: buffer_size_nonzero(words.len()),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !words.is_empty() {
            self.queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(words));
        }
        buffer
    }

    fn dispatch(&self, pipeline: &wgpu::ComputePipeline) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rrtl-gpu-sim-dispatch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rrtl-gpu-sim-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(div_ceil(self.lanes as u32, self.workgroup_size), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    fn dispatch_with_bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rrtl-gpu-sim-replay-dispatch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rrtl-gpu-sim-replay-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(div_ceil(self.lanes as u32, self.workgroup_size), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    fn dispatch_pipeline_repeated(&self, pipeline: &wgpu::ComputePipeline, repeat: usize) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rrtl-gpu-sim-batched-dispatch"),
            });
        for _ in 0..repeat {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rrtl-gpu-sim-batched-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(div_ceil(self.lanes as u32, self.workgroup_size), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    fn write_params(&self, steps: u32) {
        let params = [
            self.lanes as u32,
            self.program.total_signal_words as u32,
            self.program.total_memory_words_per_lane as u32,
            steps,
            0,
            0,
            0,
            0,
        ];
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));
    }

    fn read_trace_results(
        &self,
        source: &wgpu::Buffer,
        readback: &wgpu::Buffer,
        result_words: usize,
    ) -> Result<Vec<u32>, ErrorReport> {
        if result_words == 0 {
            return Ok(Vec::new());
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rrtl-gpu-trace-results-readback"),
            });
        encoder.copy_buffer_to_buffer(source, 0, readback, 0, buffer_size(result_words));
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, u32>(&data)[..result_words].to_vec();
        drop(data);
        readback.unmap();
        Ok(result)
    }

    fn read_trace_result_count(
        &self,
        source: &wgpu::Buffer,
        readback: &wgpu::Buffer,
    ) -> Result<u32, ErrorReport> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rrtl-gpu-trace-count-readback"),
            });
        encoder.copy_buffer_to_buffer(source, 0, readback, 0, buffer_size(1));
        self.queue.submit(Some(encoder.finish()));

        self.map_trace_result_count(readback)
    }

    fn map_trace_result_count(&self, readback: &wgpu::Buffer) -> Result<u32, ErrorReport> {
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, u32>(&data)[0];
        drop(data);
        readback.unmap();
        Ok(result)
    }

    fn sync_values_from_gpu(&mut self) -> Result<(), ErrorReport> {
        let value_words = self.program.total_signal_words * self.lanes;
        if value_words == 0 {
            return Ok(());
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rrtl-gpu-sim-readback"),
            });
        encoder.copy_buffer_to_buffer(
            &self.values_buffer,
            0,
            &self.values_readback,
            0,
            buffer_size(value_words),
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = self.values_readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        self.values
            .copy_from_slice(bytemuck::cast_slice::<u8, u32>(&data)[..value_words].as_ref());
        drop(data);
        self.values_readback.unmap();
        Ok(())
    }

    fn sync_memories_from_gpu(&mut self) -> Result<(), ErrorReport> {
        let memory_words = self.program.total_memory_words_per_lane * self.lanes;
        if memory_words == 0 {
            return Ok(());
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rrtl-gpu-sim-memory-readback"),
            });
        encoder.copy_buffer_to_buffer(
            &self.memories_buffer,
            0,
            &self.memories_readback,
            0,
            buffer_size(memory_words),
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = self.memories_readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        self.memories
            .copy_from_slice(bytemuck::cast_slice::<u8, u32>(&data)[..memory_words].as_ref());
        drop(data);
        self.memories_readback.unmap();
        Ok(())
    }
}

fn pack_trace_replay_for_program(
    program: &PackedProgram,
    lanes: usize,
    plan: &GpuTraceReplayPlan,
) -> Result<PackedGpuTraceReplay, ErrorReport> {
    let mut input_buckets = vec![Vec::<usize>::new(); plan.steps];
    let mut input_meta_by_op = Vec::with_capacity(plan.inputs.len());
    for (op_index, op) in plan.inputs.iter().enumerate() {
        if op.step >= plan.steps {
            return Err(error(
                "E_GPU_TRACE_STEP",
                format!(
                    "trace input step {} is outside replay range {}",
                    op.step, plan.steps
                ),
            ));
        }
        input_buckets[op.step].push(op_index);
        if op.values.len() != lanes {
            return Err(error(
                "E_GPU_TRACE_LANES",
                format!(
                    "trace input has {} lane values, expected {}",
                    op.values.len(),
                    lanes
                ),
            ));
        }
        let signal_index = program.signal_index(op.signal).ok_or_else(|| {
            error(
                "E_GPU_SIGNAL",
                format!("signal {:?} is not part of this GPU module", op.signal.id),
            )
        })?;
        let layout = program.signals[signal_index].layout;
        if op.limb >= layout.limbs {
            return Err(error(
                "E_GPU_TRACE_LIMB",
                format!(
                    "trace input limb {} is outside signal limb count {}",
                    op.limb, layout.limbs
                ),
            ));
        }
        input_meta_by_op.push((
            layout.offset as u32,
            op.limb as u32,
            limb_mask(layout.width, op.limb),
        ));
    }

    let mut check_buckets = vec![Vec::<usize>::new(); plan.steps];
    let mut check_meta_by_op = Vec::with_capacity(plan.checks.len());
    for (op_index, op) in plan.checks.iter().enumerate() {
        if op.step >= plan.steps {
            return Err(error(
                "E_GPU_TRACE_STEP",
                format!(
                    "trace check step {} is outside replay range {}",
                    op.step, plan.steps
                ),
            ));
        }
        check_buckets[op.step].push(op_index);
        if op.expected.len() != lanes {
            return Err(error(
                "E_GPU_TRACE_LANES",
                format!(
                    "trace check has {} lane values, expected {}",
                    op.expected.len(),
                    lanes
                ),
            ));
        }
        let signal_index = program.signal_index(op.signal).ok_or_else(|| {
            error(
                "E_GPU_SIGNAL",
                format!("signal {:?} is not part of this GPU module", op.signal.id),
            )
        })?;
        let layout = program.signals[signal_index].layout;
        if op.limb >= layout.limbs {
            return Err(error(
                "E_GPU_TRACE_LIMB",
                format!(
                    "trace check limb {} is outside signal limb count {}",
                    op.limb, layout.limbs
                ),
            ));
        }
        check_meta_by_op.push((
            op.check_index as u32,
            layout.offset as u32,
            op.limb as u32,
            limb_mask(layout.width, op.limb),
        ));
    }

    let input_template = trace_input_template(&input_buckets, &input_meta_by_op);
    let check_template = trace_check_template(&check_buckets, &check_meta_by_op);
    if plan.steps > 1 && input_template.is_some() && check_template.is_some() {
        return pack_templated_trace_replay(
            plan,
            lanes,
            &input_buckets,
            &input_meta_by_op,
            &check_buckets,
            &check_meta_by_op,
            input_template.unwrap(),
            check_template.unwrap(),
        );
    }

    pack_step_indexed_trace_replay(
        plan,
        lanes,
        &input_buckets,
        &input_meta_by_op,
        &check_buckets,
        &check_meta_by_op,
    )
}

fn pack_step_indexed_trace_replay(
    plan: &GpuTraceReplayPlan,
    lanes: usize,
    input_buckets: &[Vec<usize>],
    input_meta_by_op: &[(u32, u32, u32)],
    check_buckets: &[Vec<usize>],
    check_meta_by_op: &[(u32, u32, u32, u32)],
) -> Result<PackedGpuTraceReplay, ErrorReport> {
    let mut input_step_offsets = Vec::with_capacity(plan.steps + 1);
    let mut input_meta = Vec::with_capacity(plan.inputs.len() * 5);
    let mut inputs = Vec::with_capacity(plan.inputs.len() * lanes);
    let mut uniform_input_ops = 0usize;
    input_step_offsets.push(0u32);
    for bucket in input_buckets {
        for &op_index in bucket {
            let (offset, limb, mask) = input_meta_by_op[op_index];
            let (mode, value_offset, stored) =
                pack_trace_values(&plan.inputs[op_index].values, mask, &mut inputs);
            uniform_input_ops += usize::from(mode == 1);
            input_meta.extend([offset, limb, mode, value_offset, 0]);
            let _ = stored;
        }
        input_step_offsets.push((input_meta.len() / 5) as u32);
    }

    let mut check_step_offsets = Vec::with_capacity(plan.steps + 1);
    let mut check_meta = Vec::with_capacity(plan.checks.len() * 5);
    let mut checks = Vec::with_capacity(plan.checks.len() * lanes);
    let mut uniform_check_ops = 0usize;
    check_step_offsets.push(0u32);
    for bucket in check_buckets {
        for &op_index in bucket {
            let (check_index, offset, limb, mask) = check_meta_by_op[op_index];
            let (mode, value_offset, stored) =
                pack_trace_values(&plan.checks[op_index].expected, mask, &mut checks);
            uniform_check_ops += usize::from(mode == 1);
            check_meta.extend([check_index, offset, limb, mode, value_offset]);
            let _ = stored;
        }
        check_step_offsets.push((check_meta.len() / 5) as u32);
    }

    let trace_header: [u32; TRACE_REPLAY_HEADER_WORDS] = [
        0,
        inputs.len() as u32,
        checks.len() as u32,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ];
    let uncompressed_words = input_step_offsets.len()
        + check_step_offsets.len()
        + trace_header.len()
        + plan.inputs.len() * 5
        + plan.inputs.len() * lanes
        + plan.checks.len() * 5
        + plan.checks.len() * lanes;
    let mut trace_data = Vec::with_capacity(
        input_step_offsets.len()
            + check_step_offsets.len()
            + trace_header.len()
            + input_meta.len()
            + inputs.len()
            + check_meta.len()
            + checks.len(),
    );
    trace_data.extend(input_step_offsets);
    trace_data.extend(check_step_offsets);
    trace_data.extend(trace_header);
    trace_data.extend(input_meta);
    trace_data.extend(inputs);
    trace_data.extend(check_meta);
    trace_data.extend(checks);

    Ok(PackedGpuTraceReplay {
        trace_data,
        uncompressed_words,
        input_ops: plan.inputs.len(),
        check_ops: plan.checks.len(),
        uniform_input_ops,
        uniform_check_ops,
        template_layout: false,
        template_input_ops: 0,
        template_check_ops: 0,
        metadata_saved_words: 0,
        fixed_template: false,
        value_metadata_saved_words: 0,
        value_stride_words: 0,
    })
}

fn pack_templated_trace_replay(
    plan: &GpuTraceReplayPlan,
    lanes: usize,
    input_buckets: &[Vec<usize>],
    input_meta_by_op: &[(u32, u32, u32)],
    check_buckets: &[Vec<usize>],
    check_meta_by_op: &[(u32, u32, u32, u32)],
    input_template: Vec<(u32, u32)>,
    check_template: Vec<(u32, u32, u32)>,
) -> Result<PackedGpuTraceReplay, ErrorReport> {
    let mut input_template_meta = Vec::with_capacity(input_template.len() * 2);
    for (offset, limb) in &input_template {
        input_template_meta.extend([*offset, *limb]);
    }
    let mut check_template_meta = Vec::with_capacity(check_template.len() * 3);
    for (check_index, offset, limb) in &check_template {
        check_template_meta.extend([*check_index, *offset, *limb]);
    }

    let input_modes = stable_template_input_modes(plan, lanes, input_buckets, input_meta_by_op);
    let check_modes = stable_template_check_modes(plan, lanes, check_buckets, check_meta_by_op);
    let fixed_template = input_modes.is_some() && check_modes.is_some();
    let mut input_value_meta = Vec::new();
    let mut check_value_meta = Vec::new();
    let mut inputs = Vec::with_capacity(plan.inputs.len() * lanes);
    let mut checks = Vec::with_capacity(plan.checks.len() * lanes);
    let mut uniform_input_ops = 0usize;
    let mut uniform_check_ops = 0usize;
    let mut input_stride_words = 0usize;
    let mut check_stride_words = 0usize;
    let mut input_dense_slots = 0usize;
    let mut check_dense_slots = 0usize;
    if let (Some(input_modes), Some(check_modes)) = (input_modes.as_ref(), check_modes.as_ref()) {
        let input_order = dense_first_template_order(input_modes);
        let check_order = dense_first_template_order(check_modes);
        input_dense_slots = input_order
            .iter()
            .filter(|slot| input_modes[**slot] == 0)
            .count();
        check_dense_slots = check_order
            .iter()
            .filter(|slot| check_modes[**slot] == 0)
            .count();
        input_template_meta.clear();
        for slot in &input_order {
            let (offset, limb) = input_template[*slot];
            input_template_meta.extend([offset, limb]);
        }
        check_template_meta.clear();
        for slot in &check_order {
            let (check_index, offset, limb) = check_template[*slot];
            check_template_meta.extend([check_index, offset, limb]);
        }
        input_stride_words = input_modes
            .iter()
            .map(|mode| if *mode == 1 { 1 } else { lanes })
            .sum();
        check_stride_words = check_modes
            .iter()
            .map(|mode| if *mode == 1 { 1 } else { lanes })
            .sum();
        for bucket in input_buckets {
            for &slot in &input_order {
                let op_index = bucket[slot];
                let (_, _, mask) = input_meta_by_op[op_index];
                let stored = pack_trace_values_fixed_mode(
                    &plan.inputs[op_index].values,
                    mask,
                    input_modes[slot],
                    &mut inputs,
                );
                uniform_input_ops += usize::from(input_modes[slot] == 1);
                let _ = stored;
            }
        }
        for bucket in check_buckets {
            for &slot in &check_order {
                let op_index = bucket[slot];
                let (_, _, _, mask) = check_meta_by_op[op_index];
                let stored = pack_trace_values_fixed_mode(
                    &plan.checks[op_index].expected,
                    mask,
                    check_modes[slot],
                    &mut checks,
                );
                uniform_check_ops += usize::from(check_modes[slot] == 1);
                let _ = stored;
            }
        }
    } else {
        input_value_meta = Vec::with_capacity(plan.inputs.len() * 2);
        for bucket in input_buckets {
            for &op_index in bucket {
                let (_, _, mask) = input_meta_by_op[op_index];
                let (mode, value_offset, _) =
                    pack_trace_values(&plan.inputs[op_index].values, mask, &mut inputs);
                uniform_input_ops += usize::from(mode == 1);
                input_value_meta.extend([mode, value_offset]);
            }
        }
        check_value_meta = Vec::with_capacity(plan.checks.len() * 2);
        for bucket in check_buckets {
            for &op_index in bucket {
                let (_, _, _, mask) = check_meta_by_op[op_index];
                let (mode, value_offset, _) =
                    pack_trace_values(&plan.checks[op_index].expected, mask, &mut checks);
                uniform_check_ops += usize::from(mode == 1);
                check_value_meta.extend([mode, value_offset]);
            }
        }
    }

    let stored_input_metadata_words = input_template_meta.len() + input_value_meta.len();
    let stored_check_metadata_words = check_template_meta.len() + check_value_meta.len();
    let metadata_saved_words = plan
        .inputs
        .len()
        .saturating_mul(5)
        .saturating_sub(stored_input_metadata_words)
        + plan
            .checks
            .len()
            .saturating_mul(5)
            .saturating_sub(stored_check_metadata_words);
    let value_metadata_saved_words = if fixed_template {
        plan.inputs.len().saturating_mul(2) + plan.checks.len().saturating_mul(2)
    } else {
        0
    };
    let value_stride_words = input_stride_words + check_stride_words;
    let trace_header: [u32; TRACE_REPLAY_HEADER_WORDS] = [
        if fixed_template { 2 } else { 1 },
        inputs.len() as u32,
        checks.len() as u32,
        input_template.len() as u32,
        check_template.len() as u32,
        metadata_saved_words as u32,
        input_stride_words as u32,
        check_stride_words as u32,
        input_dense_slots as u32,
        check_dense_slots as u32,
    ];
    let uncompressed_words = plan.steps.saturating_add(1).saturating_mul(2)
        + trace_header.len()
        + plan.inputs.len() * 5
        + plan.inputs.len() * lanes
        + plan.checks.len() * 5
        + plan.checks.len() * lanes;
    let mut input_step_offsets = vec![0u32; plan.steps + 1];
    let mut check_step_offsets = vec![0u32; plan.steps + 1];
    for step in 0..=plan.steps {
        input_step_offsets[step] = (step * input_template.len()) as u32;
        check_step_offsets[step] = (step * check_template.len()) as u32;
    }

    let mut trace_data = Vec::with_capacity(
        input_step_offsets.len()
            + check_step_offsets.len()
            + trace_header.len()
            + input_template_meta.len()
            + input_value_meta.len()
            + inputs.len()
            + check_template_meta.len()
            + check_value_meta.len()
            + checks.len(),
    );
    trace_data.extend(input_step_offsets);
    trace_data.extend(check_step_offsets);
    trace_data.extend(trace_header);
    trace_data.extend(input_template_meta);
    trace_data.extend(input_value_meta);
    trace_data.extend(inputs);
    trace_data.extend(check_template_meta);
    trace_data.extend(check_value_meta);
    trace_data.extend(checks);

    Ok(PackedGpuTraceReplay {
        trace_data,
        uncompressed_words,
        input_ops: plan.inputs.len(),
        check_ops: plan.checks.len(),
        uniform_input_ops,
        uniform_check_ops,
        template_layout: true,
        template_input_ops: input_template.len(),
        template_check_ops: check_template.len(),
        metadata_saved_words,
        fixed_template,
        value_metadata_saved_words,
        value_stride_words,
    })
}

fn dense_first_template_order(modes: &[u32]) -> Vec<usize> {
    let mut order = Vec::with_capacity(modes.len());
    order.extend(
        modes
            .iter()
            .enumerate()
            .filter_map(|(slot, mode)| (*mode == 0).then_some(slot)),
    );
    order.extend(
        modes
            .iter()
            .enumerate()
            .filter_map(|(slot, mode)| (*mode == 1).then_some(slot)),
    );
    order
}

fn trace_input_template(
    buckets: &[Vec<usize>],
    meta_by_op: &[(u32, u32, u32)],
) -> Option<Vec<(u32, u32)>> {
    let first = buckets.first()?;
    let template = first
        .iter()
        .map(|op_index| {
            let (offset, limb, _) = meta_by_op[*op_index];
            (offset, limb)
        })
        .collect::<Vec<_>>();
    if buckets.iter().all(|bucket| {
        bucket.len() == template.len()
            && bucket.iter().zip(&template).all(|(op_index, expected)| {
                let (offset, limb, _) = meta_by_op[*op_index];
                (offset, limb) == *expected
            })
    }) {
        Some(template)
    } else {
        None
    }
}

fn trace_check_template(
    buckets: &[Vec<usize>],
    meta_by_op: &[(u32, u32, u32, u32)],
) -> Option<Vec<(u32, u32, u32)>> {
    let first = buckets.first()?;
    let template = first
        .iter()
        .map(|op_index| {
            let (check_index, offset, limb, _) = meta_by_op[*op_index];
            (check_index, offset, limb)
        })
        .collect::<Vec<_>>();
    if buckets.iter().all(|bucket| {
        bucket.len() == template.len()
            && bucket.iter().zip(&template).all(|(op_index, expected)| {
                let (check_index, offset, limb, _) = meta_by_op[*op_index];
                (check_index, offset, limb) == *expected
            })
    }) {
        Some(template)
    } else {
        None
    }
}

fn stable_template_input_modes(
    plan: &GpuTraceReplayPlan,
    lanes: usize,
    buckets: &[Vec<usize>],
    meta_by_op: &[(u32, u32, u32)],
) -> Option<Vec<u32>> {
    let first = buckets.first()?;
    let mut modes = Vec::with_capacity(first.len());
    for &op_index in first {
        let (_, _, mask) = meta_by_op[op_index];
        modes.push(trace_value_mode(&plan.inputs[op_index].values, mask));
    }
    if buckets.iter().all(|bucket| {
        bucket.len() == modes.len()
            && bucket.iter().zip(&modes).all(|(op_index, mode)| {
                let (_, _, mask) = meta_by_op[*op_index];
                plan.inputs[*op_index].values.len() == lanes
                    && trace_value_mode(&plan.inputs[*op_index].values, mask) == *mode
            })
    }) {
        Some(modes)
    } else {
        None
    }
}

fn stable_template_check_modes(
    plan: &GpuTraceReplayPlan,
    lanes: usize,
    buckets: &[Vec<usize>],
    meta_by_op: &[(u32, u32, u32, u32)],
) -> Option<Vec<u32>> {
    let first = buckets.first()?;
    let mut modes = Vec::with_capacity(first.len());
    for &op_index in first {
        let (_, _, _, mask) = meta_by_op[op_index];
        modes.push(trace_value_mode(&plan.checks[op_index].expected, mask));
    }
    if buckets.iter().all(|bucket| {
        bucket.len() == modes.len()
            && bucket.iter().zip(&modes).all(|(op_index, mode)| {
                let (_, _, _, mask) = meta_by_op[*op_index];
                plan.checks[*op_index].expected.len() == lanes
                    && trace_value_mode(&plan.checks[*op_index].expected, mask) == *mode
            })
    }) {
        Some(modes)
    } else {
        None
    }
}

fn trace_value_mode(values: &[u32], mask: u32) -> u32 {
    let first = values.first().copied().unwrap_or(0) & mask;
    if values.iter().copied().all(|value| (value & mask) == first) {
        1
    } else {
        0
    }
}

fn pack_trace_values(values: &[u32], mask: u32, out: &mut Vec<u32>) -> (u32, u32, usize) {
    let value_offset = out.len() as u32;
    let masked = values
        .iter()
        .copied()
        .map(|value| value & mask)
        .collect::<Vec<_>>();
    let uniform = masked
        .first()
        .map(|first| masked.iter().all(|value| value == first))
        .unwrap_or(true);
    if uniform {
        out.push(masked.first().copied().unwrap_or(0));
        (1, value_offset, 1)
    } else {
        let stored = masked.len();
        out.extend(masked);
        (0, value_offset, stored)
    }
}

fn pack_trace_values_fixed_mode(values: &[u32], mask: u32, mode: u32, out: &mut Vec<u32>) -> usize {
    if mode == 1 {
        out.push(values.first().copied().unwrap_or(0) & mask);
        1
    } else {
        let start_len = out.len();
        out.extend(values.iter().copied().map(|value| value & mask));
        out.len() - start_len
    }
}

pub fn gpu_shader_stats(
    program: &PackedProgram,
    options: GpuBatchOptions,
) -> Result<GpuShaderStats, ErrorReport> {
    validate_workgroup_size(options.workgroup_size)?;
    let machine = lower_to_machine_program(program);
    let unoptimized = analyze_machine_program(&machine)?;
    let machine = optimize_machine_program(&machine, options.schedule)?;
    let optimized = analyze_machine_program(&machine)?;
    let wgsl = generate_wgsl_from_machine(
        &machine,
        options.memory_layout,
        options.workgroup_size,
        options.reuse_temporaries,
    );
    let temp_stats = shader_temp_stats(&machine);
    let unoptimized_packets = shader_packet_stats(&unoptimized);
    let optimized_packets = shader_packet_stats(&optimized);
    let unoptimized_memory = shader_memory_stats(&lower_to_machine_program(program));
    let optimized_memory = shader_memory_stats(&machine);
    Ok(GpuShaderStats {
        unoptimized,
        optimized,
        unoptimized_packets,
        optimized_packets,
        unoptimized_memory,
        optimized_memory,
        wgsl_bytes: wgsl.len(),
        optimized_temp_slots: if options.reuse_temporaries {
            temp_stats.temp_slots
        } else {
            temp_stats.value_vars
        },
        optimized_value_vars: temp_stats.value_vars,
        schedule: options.schedule,
        memory_layout: options.memory_layout,
        workgroup_size: options.workgroup_size,
        reuse_temporaries: options.reuse_temporaries,
        total_memory_words_per_lane: program.total_memory_words_per_lane,
    })
}

pub fn analyze_gpu_regions(program: &PackedProgram) -> Result<GpuRegionAnalysis, ErrorReport> {
    let machine = lower_to_machine_program(program);
    analyze_machine_program(&machine)?;
    let async_reset_comb = analyze_gpu_region_block(&machine.streams.async_reset_comb);
    let comb = analyze_gpu_region_block(&machine.streams.comb);
    let tick_next = analyze_gpu_region_block(&machine.streams.tick_next);
    let tick_commit = analyze_gpu_region_block(&machine.streams.tick_commit);
    let total = merge_gpu_region_blocks([async_reset_comb, comb, tick_next, tick_commit]);
    let recommendation = gpu_region_recommendation(&total);
    Ok(GpuRegionAnalysis {
        streams: GpuRegionStreamsAnalysis {
            async_reset_comb,
            comb,
            tick_next,
            tick_commit,
        },
        total,
        recommendation,
        reasons: gpu_region_reasons(&total, recommendation),
    })
}

fn analyze_gpu_region_block(block: &PackedBlock) -> GpuRegionBlockAnalysis {
    let mut report = GpuRegionBlockAnalysis::default();
    let mut current_region_packets = 0usize;
    let mut current_region_instrs = 0usize;
    for packet in &block.packets {
        report.packets += 1;
        let memory_reads = packet
            .instrs
            .iter()
            .filter(|instr| matches!(instr.kind, PackedInstrKind::MemRead { .. }))
            .count();
        let memory_writes = packet
            .effects
            .iter()
            .filter(|effect| matches!(effect, PackedEffect::MemoryWrite { .. }))
            .count();
        let is_memory_hostile = memory_reads > 0 || memory_writes > 0;
        report.memory_reads += memory_reads;
        report.memory_writes += memory_writes;
        report.instr_count += packet.instrs.len();
        report.wide_instrs += packet
            .instrs
            .iter()
            .filter(|instr| instr.ty.width > 32)
            .count();

        if is_memory_hostile {
            report.memory_hostile_packets += 1;
            finish_gpu_region(
                &mut report,
                &mut current_region_packets,
                &mut current_region_instrs,
            );
        } else {
            if current_region_packets == 0 {
                report.pure_compute_regions += 1;
            }
            current_region_packets += 1;
            current_region_instrs += packet.instrs.len();
            report.pure_compute_packets += 1;
            report.pure_compute_instrs += packet.instrs.len();
        }
    }
    finish_gpu_region(
        &mut report,
        &mut current_region_packets,
        &mut current_region_instrs,
    );
    report.estimated_launch_work_units = report.pure_compute_instrs.max(report.max_region_instrs);
    report
}

fn finish_gpu_region(
    report: &mut GpuRegionBlockAnalysis,
    current_region_packets: &mut usize,
    current_region_instrs: &mut usize,
) {
    report.max_region_packets = report.max_region_packets.max(*current_region_packets);
    report.max_region_instrs = report.max_region_instrs.max(*current_region_instrs);
    *current_region_packets = 0;
    *current_region_instrs = 0;
}

fn merge_gpu_region_blocks(blocks: [GpuRegionBlockAnalysis; 4]) -> GpuRegionBlockAnalysis {
    let mut total = GpuRegionBlockAnalysis::default();
    for block in blocks {
        total.packets += block.packets;
        total.pure_compute_regions += block.pure_compute_regions;
        total.pure_compute_packets += block.pure_compute_packets;
        total.memory_hostile_packets += block.memory_hostile_packets;
        total.instr_count += block.instr_count;
        total.pure_compute_instrs += block.pure_compute_instrs;
        total.wide_instrs += block.wide_instrs;
        total.memory_reads += block.memory_reads;
        total.memory_writes += block.memory_writes;
        total.max_region_packets = total.max_region_packets.max(block.max_region_packets);
        total.max_region_instrs = total.max_region_instrs.max(block.max_region_instrs);
        total.estimated_launch_work_units += block.estimated_launch_work_units;
    }
    total
}

fn gpu_region_recommendation(report: &GpuRegionBlockAnalysis) -> GpuRegionRecommendation {
    if report.memory_reads > 0 || report.memory_writes > 0 {
        if report.pure_compute_instrs
            >= report
                .instr_count
                .saturating_sub(report.pure_compute_instrs)
        {
            GpuRegionRecommendation::MixedCandidate
        } else {
            GpuRegionRecommendation::MemoryBlocked
        }
    } else if report.instr_count == 0 || report.pure_compute_instrs < 64 {
        GpuRegionRecommendation::TooSmall
    } else {
        GpuRegionRecommendation::ComputeCandidate
    }
}

fn gpu_region_reasons(
    report: &GpuRegionBlockAnalysis,
    recommendation: GpuRegionRecommendation,
) -> Vec<String> {
    let mut reasons = Vec::new();
    match recommendation {
        GpuRegionRecommendation::ComputeCandidate => {
            reasons.push("all substantial work is in memory-free compute regions".to_string());
        }
        GpuRegionRecommendation::MixedCandidate => {
            reasons.push("compute regions are substantial but memory packets remain".to_string());
        }
        GpuRegionRecommendation::MemoryBlocked => {
            reasons.push("memory packets dominate the current GPU replay shape".to_string());
        }
        GpuRegionRecommendation::TooSmall => {
            reasons
                .push("compute regions are too small to amortize GPU launch overhead".to_string());
        }
    }
    if report.memory_reads > 0 || report.memory_writes > 0 {
        reasons.push(format!(
            "{} memory reads and {} memory writes are present",
            report.memory_reads, report.memory_writes
        ));
    }
    if report.wide_instrs > 0 {
        reasons.push(format!(
            "{} wide instructions require multi-limb shader code",
            report.wide_instrs
        ));
    }
    reasons.push(format!(
        "{} pure compute regions, max region {} packets / {} instructions",
        report.pure_compute_regions, report.max_region_packets, report.max_region_instrs
    ));
    reasons
}

fn validate_workgroup_size(workgroup_size: u32) -> Result<(), ErrorReport> {
    if workgroup_size == 0 {
        return Err(error(
            "E_GPU_WORKGROUP_SIZE",
            "GPU workgroup size must be greater than zero",
        ));
    }
    Ok(())
}

fn parse_gpu_memory_layout(layout: &str) -> Result<GpuMemoryLayout, ErrorReport> {
    match layout {
        "lane_major" => Ok(GpuMemoryLayout::LaneMajor),
        "word_major" => Ok(GpuMemoryLayout::WordMajor),
        other => Err(error(
            "E_GPU_AUTOTUNE_LAYOUT",
            format!("unknown GPU autotune memory layout `{other}`"),
        )),
    }
}

fn shader_packet_stats(analysis: &PackedMachineAnalysis) -> GpuShaderPacketStats {
    let async_reset_comb = analysis.async_reset_comb.packets.len();
    let comb = analysis.comb.packets.len();
    let tick_next = analysis.tick_next.packets.len();
    let tick_commit = analysis.tick_commit.packets.len();
    GpuShaderPacketStats {
        async_reset_comb,
        comb,
        tick_next,
        tick_commit,
        total: async_reset_comb + comb + tick_next + tick_commit,
    }
}

fn shader_memory_stats(machine: &PackedMachineProgram) -> GpuShaderMemoryStats {
    let async_reset_comb = shader_memory_stream_stats(&machine.streams.async_reset_comb);
    let comb = shader_memory_stream_stats(&machine.streams.comb);
    let tick_next = shader_memory_stream_stats(&machine.streams.tick_next);
    let tick_commit = shader_memory_stream_stats(&machine.streams.tick_commit);
    GpuShaderMemoryStats {
        async_reset_comb,
        comb,
        tick_next,
        tick_commit,
        total_reads: async_reset_comb.reads + comb.reads + tick_next.reads + tick_commit.reads,
        total_writes: async_reset_comb.writes + comb.writes + tick_next.writes + tick_commit.writes,
    }
}

fn shader_memory_stream_stats(block: &PackedBlock) -> GpuShaderMemoryStreamStats {
    let mut stats = GpuShaderMemoryStreamStats::default();
    for packet in &block.packets {
        stats.reads += packet
            .instrs
            .iter()
            .filter(|instr| matches!(instr.kind, PackedInstrKind::MemRead { .. }))
            .count();
        stats.writes += packet
            .effects
            .iter()
            .filter(|effect| matches!(effect, PackedEffect::MemoryWrite { .. }))
            .count();
    }
    stats
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GpuTempStats {
    temp_slots: usize,
    value_vars: usize,
}

fn shader_temp_stats(machine: &PackedMachineProgram) -> GpuTempStats {
    [
        &machine.streams.async_reset_comb,
        &machine.streams.comb,
        &machine.streams.tick_next,
        &machine.streams.tick_commit,
    ]
    .into_iter()
    .map(build_temp_slot_plan)
    .fold(GpuTempStats::default(), |mut acc, plan| {
        acc.temp_slots += plan.slot_count;
        acc.value_vars += plan.value_vars;
        acc
    })
}

fn generate_wgsl(program: &PackedProgram, options: GpuBatchOptions) -> Result<String, ErrorReport> {
    validate_workgroup_size(options.workgroup_size)?;
    let machine = lower_to_machine_program(program);
    let machine = optimize_machine_program(&machine, options.schedule)?;
    Ok(generate_wgsl_from_machine(
        &machine,
        options.memory_layout,
        options.workgroup_size,
        options.reuse_temporaries,
    ))
}

fn generate_wgsl_from_machine(
    machine: &PackedMachineProgram,
    memory_layout: GpuMemoryLayout,
    workgroup_size: u32,
    reuse_temporaries: bool,
) -> String {
    let mut out = String::new();
    out.push_str(
        "struct Params { lanes: u32, signal_words: u32, memory_words_per_lane: u32, steps: u32, trace_input_ops: u32, trace_check_ops: u32, max_mismatches: u32, reserved: u32 };\n",
    );
    out.push_str("@group(0) @binding(0) var<storage, read_write> values: array<u32>;\n");
    out.push_str("@group(0) @binding(1) var<storage, read_write> memories: array<u32>;\n");
    out.push_str("@group(0) @binding(2) var<uniform> params: Params;\n");
    out.push_str("@group(0) @binding(3) var<storage, read_write> trace_data: array<u32>;\n");
    out.push_str(
        "@group(0) @binding(4) var<storage, read_write> trace_results: array<atomic<u32>>;\n",
    );
    out.push_str("fn value_idx(offset: u32, limb: u32, lane: u32) -> u32 { return (offset + limb) * params.lanes + lane; }\n");
    out.push_str("fn load_value(offset: u32, limb: u32, lane: u32) -> u32 { return values[value_idx(offset, limb, lane)]; }\n");
    out.push_str("fn store_value(offset: u32, limb: u32, lane: u32, value: u32) { values[value_idx(offset, limb, lane)] = value; }\n");
    out.push_str("fn add_word(a: u32, b: u32, carry: u32) -> u32 { return a + b + carry; }\n");
    out.push_str("fn add_carry(a: u32, b: u32, carry: u32) -> u32 { let s = a + b; let c1 = select(0u, 1u, s < a); let s2 = s + carry; let c2 = select(0u, 1u, s2 < s); return select(0u, 1u, (c1 | c2) != 0u); }\n");
    out.push_str("fn sub_word(a: u32, b: u32, borrow: u32) -> u32 { return a - b - borrow; }\n");
    out.push_str("fn sub_borrow(a: u32, b: u32, borrow: u32) -> u32 { let d = a - b; let b1 = select(0u, 1u, a < b); let d2 = d - borrow; let b2 = select(0u, 1u, d < borrow); return select(0u, 1u, (b1 | b2) != 0u); }\n");
    out.push_str("fn record_trace_mismatch(step: u32, check_index: u32, lane: u32, limb: u32, expected: u32, actual: u32) { let slot = atomicAdd(&trace_results[0], 1u); if (slot < params.max_mismatches) { let base = 1u + slot * 6u; atomicStore(&trace_results[base + 0u], step); atomicStore(&trace_results[base + 1u], check_index); atomicStore(&trace_results[base + 2u], lane); atomicStore(&trace_results[base + 3u], limb); atomicStore(&trace_results[base + 4u], expected); atomicStore(&trace_results[base + 5u], actual); } }\n");

    out.push_str(&format!(
        "@compute @workgroup_size({workgroup_size})\nfn eval_comb(@builtin(global_invocation_id) gid: vec3<u32>) {{\n"
    ));
    out.push_str("  let lane = gid.x;\n  if (lane >= params.lanes) { return; }\n");
    emit_comb_sequence(
        machine,
        &mut out,
        "  ",
        "e",
        memory_layout,
        reuse_temporaries,
    );
    out.push_str("}\n");

    out.push_str(&format!(
        "@compute @workgroup_size({workgroup_size})\nfn tick(@builtin(global_invocation_id) gid: vec3<u32>) {{\n"
    ));
    out.push_str("  let lane = gid.x;\n  if (lane >= params.lanes) { return; }\n");
    emit_tick_body(
        machine,
        &mut out,
        "  ",
        "t",
        memory_layout,
        reuse_temporaries,
    );
    out.push_str("}\n");

    out.push_str(&format!(
        "@compute @workgroup_size({workgroup_size})\nfn tick_many(@builtin(global_invocation_id) gid: vec3<u32>) {{\n"
    ));
    out.push_str("  let lane = gid.x;\n  if (lane >= params.lanes) { return; }\n");
    out.push_str("  var step = 0u;\n");
    out.push_str("  loop {\n");
    out.push_str("    if (step >= params.steps) { break; }\n");
    emit_comb_sequence(
        machine,
        &mut out,
        "    ",
        "tm_pre",
        memory_layout,
        reuse_temporaries,
    );
    emit_tick_body(
        machine,
        &mut out,
        "    ",
        "tm_tick",
        memory_layout,
        reuse_temporaries,
    );
    emit_comb_sequence(
        machine,
        &mut out,
        "    ",
        "tm_post",
        memory_layout,
        reuse_temporaries,
    );
    out.push_str("    step = step + 1u;\n");
    out.push_str("  }\n");
    out.push_str("}\n");

    out.push_str(&format!(
        "@compute @workgroup_size({workgroup_size})\nfn replay_trace(@builtin(global_invocation_id) gid: vec3<u32>) {{\n"
    ));
    out.push_str("  let lane = gid.x;\n  if (lane >= params.lanes) { return; }\n");
    out.push_str("  let trace_input_offsets_base = 0u;\n");
    out.push_str("  let trace_check_offsets_base = params.steps + 1u;\n");
    out.push_str("  let trace_header_base = trace_check_offsets_base + params.steps + 1u;\n");
    out.push_str("  if (trace_data[trace_header_base + 0u] == 2u) {\n");
    out.push_str("    let template_input_ops = trace_data[trace_header_base + 3u];\n");
    out.push_str("    let template_check_ops = trace_data[trace_header_base + 4u];\n");
    out.push_str("    let fixed_input_stride = trace_data[trace_header_base + 6u];\n");
    out.push_str("    let fixed_check_stride = trace_data[trace_header_base + 7u];\n");
    out.push_str("    let input_dense_slots = trace_data[trace_header_base + 8u];\n");
    out.push_str("    let check_dense_slots = trace_data[trace_header_base + 9u];\n");
    out.push_str("    let trace_input_template_base = trace_header_base + 10u;\n");
    out.push_str(
        "    let trace_input_base = trace_input_template_base + template_input_ops * 2u;\n",
    );
    out.push_str("    let trace_check_template_base = trace_input_base + fixed_input_stride * params.steps;\n");
    out.push_str(
        "    let trace_check_base = trace_check_template_base + template_check_ops * 3u;\n",
    );
    out.push_str("    var step = 0u;\n");
    out.push_str("    loop {\n");
    out.push_str("      if (step >= params.steps) { break; }\n");
    out.push_str("      var input_op = 0u;\n");
    out.push_str("      loop {\n");
    out.push_str("        if (input_op >= input_dense_slots) { break; }\n");
    out.push_str("        let trace_template = trace_input_template_base + input_op * 2u;\n");
    out.push_str("        let trace_value_offset = step * fixed_input_stride + input_op * params.lanes + lane;\n");
    out.push_str("        let trace_value = trace_data[trace_input_base + trace_value_offset];\n");
    out.push_str("        store_value(trace_data[trace_template + 0u], trace_data[trace_template + 1u], lane, trace_value);\n");
    out.push_str("        input_op = input_op + 1u;\n");
    out.push_str("      }\n");
    out.push_str("      input_op = input_dense_slots;\n");
    out.push_str("      loop {\n");
    out.push_str("        if (input_op >= template_input_ops) { break; }\n");
    out.push_str("        let trace_template = trace_input_template_base + input_op * 2u;\n");
    out.push_str("        let uniform_op = input_op - input_dense_slots;\n");
    out.push_str("        let trace_value_offset = step * fixed_input_stride + input_dense_slots * params.lanes + uniform_op;\n");
    out.push_str("        let trace_value = trace_data[trace_input_base + trace_value_offset];\n");
    out.push_str("        store_value(trace_data[trace_template + 0u], trace_data[trace_template + 1u], lane, trace_value);\n");
    out.push_str("        input_op = input_op + 1u;\n");
    out.push_str("      }\n");
    emit_comb_sequence(
        machine,
        &mut out,
        "      ",
        "rt_fix_comb",
        memory_layout,
        reuse_temporaries,
    );
    out.push_str("      var check_op = 0u;\n");
    out.push_str("      loop {\n");
    out.push_str("        if (check_op >= check_dense_slots) { break; }\n");
    out.push_str("        let trace_template = trace_check_template_base + check_op * 3u;\n");
    out.push_str("        let trace_value_offset = step * fixed_check_stride + check_op * params.lanes + lane;\n");
    out.push_str("        let expected = trace_data[trace_check_base + trace_value_offset];\n");
    out.push_str("        let actual = load_value(trace_data[trace_template + 1u], trace_data[trace_template + 2u], lane); if (actual != expected) { record_trace_mismatch(step, trace_data[trace_template + 0u], lane, trace_data[trace_template + 2u], expected, actual); }\n");
    out.push_str("        check_op = check_op + 1u;\n");
    out.push_str("      }\n");
    out.push_str("      check_op = check_dense_slots;\n");
    out.push_str("      loop {\n");
    out.push_str("        if (check_op >= template_check_ops) { break; }\n");
    out.push_str("        let trace_template = trace_check_template_base + check_op * 3u;\n");
    out.push_str("        let uniform_op = check_op - check_dense_slots;\n");
    out.push_str("        let trace_value_offset = step * fixed_check_stride + check_dense_slots * params.lanes + uniform_op;\n");
    out.push_str("        let expected = trace_data[trace_check_base + trace_value_offset];\n");
    out.push_str("        let actual = load_value(trace_data[trace_template + 1u], trace_data[trace_template + 2u], lane); if (actual != expected) { record_trace_mismatch(step, trace_data[trace_template + 0u], lane, trace_data[trace_template + 2u], expected, actual); }\n");
    out.push_str("        check_op = check_op + 1u;\n");
    out.push_str("      }\n");
    emit_tick_body(
        machine,
        &mut out,
        "      ",
        "rt_fix_tick",
        memory_layout,
        reuse_temporaries,
    );
    out.push_str("      step = step + 1u;\n");
    out.push_str("    }\n");
    out.push_str("    return;\n");
    out.push_str("  }\n");
    out.push_str("  if (trace_data[trace_header_base + 0u] == 1u) {\n");
    out.push_str("    let template_input_ops = trace_data[trace_header_base + 3u];\n");
    out.push_str("    let template_check_ops = trace_data[trace_header_base + 4u];\n");
    out.push_str("    let trace_input_template_base = trace_header_base + 10u;\n");
    out.push_str("    let trace_input_value_meta_base = trace_input_template_base + template_input_ops * 2u;\n");
    out.push_str(
        "    let trace_input_base = trace_input_value_meta_base + params.trace_input_ops * 2u;\n",
    );
    out.push_str("    let trace_check_template_base = trace_input_base + trace_data[trace_header_base + 1u];\n");
    out.push_str("    let trace_check_value_meta_base = trace_check_template_base + template_check_ops * 3u;\n");
    out.push_str(
        "    let trace_check_base = trace_check_value_meta_base + params.trace_check_ops * 2u;\n",
    );
    out.push_str("    var step = 0u;\n");
    out.push_str("    loop {\n");
    out.push_str("      if (step >= params.steps) { break; }\n");
    out.push_str("      var input_op = 0u;\n");
    out.push_str("      loop {\n");
    out.push_str("        if (input_op >= template_input_ops) { break; }\n");
    out.push_str("        let trace_template = trace_input_template_base + input_op * 2u;\n");
    out.push_str("        let trace_value_meta = trace_input_value_meta_base + (step * template_input_ops + input_op) * 2u;\n");
    out.push_str("        let trace_value_offset = trace_data[trace_value_meta + 1u];\n");
    out.push_str("        let trace_value_mode = trace_data[trace_value_meta + 0u];\n");
    out.push_str("        var trace_value = trace_data[trace_input_base + trace_value_offset]; if (trace_value_mode == 0u) { trace_value = trace_data[trace_input_base + trace_value_offset + lane]; }\n");
    out.push_str("        store_value(trace_data[trace_template + 0u], trace_data[trace_template + 1u], lane, trace_value);\n");
    out.push_str("        input_op = input_op + 1u;\n");
    out.push_str("      }\n");
    emit_comb_sequence(
        machine,
        &mut out,
        "      ",
        "rt_dyn_comb",
        memory_layout,
        reuse_temporaries,
    );
    out.push_str("      var check_op = 0u;\n");
    out.push_str("      loop {\n");
    out.push_str("        if (check_op >= template_check_ops) { break; }\n");
    out.push_str("        let trace_template = trace_check_template_base + check_op * 3u;\n");
    out.push_str("        let trace_value_meta = trace_check_value_meta_base + (step * template_check_ops + check_op) * 2u;\n");
    out.push_str("        let trace_value_offset = trace_data[trace_value_meta + 1u];\n");
    out.push_str("        let trace_value_mode = trace_data[trace_value_meta + 0u];\n");
    out.push_str("        var expected = trace_data[trace_check_base + trace_value_offset]; if (trace_value_mode == 0u) { expected = trace_data[trace_check_base + trace_value_offset + lane]; }\n");
    out.push_str("        let actual = load_value(trace_data[trace_template + 1u], trace_data[trace_template + 2u], lane); if (actual != expected) { record_trace_mismatch(step, trace_data[trace_template + 0u], lane, trace_data[trace_template + 2u], expected, actual); }\n");
    out.push_str("        check_op = check_op + 1u;\n");
    out.push_str("      }\n");
    emit_tick_body(
        machine,
        &mut out,
        "      ",
        "rt_dyn_tick",
        memory_layout,
        reuse_temporaries,
    );
    out.push_str("      step = step + 1u;\n");
    out.push_str("    }\n");
    out.push_str("    return;\n");
    out.push_str("  }\n");
    out.push_str("  let trace_input_meta_base = trace_header_base + 10u;\n");
    out.push_str("  let trace_input_base = trace_input_meta_base + params.trace_input_ops * 5u;\n");
    out.push_str(
        "  let trace_check_meta_base = trace_input_base + trace_data[trace_header_base + 1u];\n",
    );
    out.push_str("  let trace_check_base = trace_check_meta_base + params.trace_check_ops * 5u;\n");
    out.push_str("  var step = 0u;\n");
    out.push_str("  loop {\n");
    out.push_str("    if (step >= params.steps) { break; }\n");
    out.push_str("    var input_op = trace_data[trace_input_offsets_base + step];\n");
    out.push_str("    let input_end = trace_data[trace_input_offsets_base + step + 1u];\n");
    out.push_str("    loop {\n");
    out.push_str("      if (input_op >= input_end) { break; }\n");
    out.push_str("      let trace_meta = trace_input_meta_base + input_op * 5u;\n");
    out.push_str("      let trace_value_offset = trace_data[trace_meta + 3u]; var trace_value = trace_data[trace_input_base + trace_value_offset]; if (trace_data[trace_meta + 2u] == 0u) { trace_value = trace_data[trace_input_base + trace_value_offset + lane]; }\n");
    out.push_str("      store_value(trace_data[trace_meta + 0u], trace_data[trace_meta + 1u], lane, trace_value);\n");
    out.push_str("      input_op = input_op + 1u;\n");
    out.push_str("    }\n");
    emit_comb_sequence(
        machine,
        &mut out,
        "    ",
        "rt_idx_comb",
        memory_layout,
        reuse_temporaries,
    );
    out.push_str("    var check_op = trace_data[trace_check_offsets_base + step];\n");
    out.push_str("    let check_end = trace_data[trace_check_offsets_base + step + 1u];\n");
    out.push_str("    loop {\n");
    out.push_str("      if (check_op >= check_end) { break; }\n");
    out.push_str("      let trace_meta = trace_check_meta_base + check_op * 5u;\n");
    out.push_str("      let trace_value_offset = trace_data[trace_meta + 4u]; var expected = trace_data[trace_check_base + trace_value_offset]; if (trace_data[trace_meta + 3u] == 0u) { expected = trace_data[trace_check_base + trace_value_offset + lane]; }\n");
    out.push_str("      let actual = load_value(trace_data[trace_meta + 1u], trace_data[trace_meta + 2u], lane); if (actual != expected) { record_trace_mismatch(step, trace_data[trace_meta + 0u], lane, trace_data[trace_meta + 2u], expected, actual); }\n");
    out.push_str("      check_op = check_op + 1u;\n");
    out.push_str("    }\n");
    emit_tick_body(
        machine,
        &mut out,
        "    ",
        "rt_idx_tick",
        memory_layout,
        reuse_temporaries,
    );
    out.push_str("    step = step + 1u;\n");
    out.push_str("  }\n");
    out.push_str("}\n");
    out
}

fn emit_comb_sequence(
    machine: &PackedMachineProgram,
    out: &mut String,
    indent: &str,
    scope: &str,
    memory_layout: GpuMemoryLayout,
    reuse_temporaries: bool,
) {
    let repeats = (machine.streams.comb.packets.len()
        + machine.streams.async_reset_comb.packets.len())
    .max(1);
    for repeat in 0..repeats {
        emit_machine_block(
            machine,
            &machine.streams.comb,
            out,
            indent,
            &format!("{scope}_c{repeat}"),
            memory_layout,
            reuse_temporaries,
        );
        emit_machine_block(
            machine,
            &machine.streams.async_reset_comb,
            out,
            indent,
            &format!("{scope}_a{repeat}"),
            memory_layout,
            reuse_temporaries,
        );
    }
}

fn emit_tick_body(
    machine: &PackedMachineProgram,
    out: &mut String,
    indent: &str,
    scope: &str,
    memory_layout: GpuMemoryLayout,
    reuse_temporaries: bool,
) {
    let next_scope = format!("{scope}_next");
    let captured = emit_tick_capture_block(
        machine,
        out,
        indent,
        &next_scope,
        memory_layout,
        reuse_temporaries,
    );
    emit_machine_block(
        machine,
        &machine.streams.tick_commit,
        out,
        indent,
        &format!("{scope}_commit"),
        memory_layout,
        reuse_temporaries,
    );
    for dst in captured {
        let signal = &machine.source.signals[dst];
        for limb in 0..signal.layout.limbs {
            out.push_str(&format!(
                "{indent}store_value({}u, {limb}u, lane, r{next_scope}_{dst}_{limb});\n",
                signal.layout.offset
            ));
        }
    }
}

fn emit_machine_block(
    machine: &PackedMachineProgram,
    block: &PackedBlock,
    out: &mut String,
    indent: &str,
    scope: &str,
    memory_layout: GpuMemoryLayout,
    reuse_temporaries: bool,
) {
    if reuse_temporaries {
        emit_reusable_machine_block(machine, block, out, indent, scope, memory_layout);
        return;
    }
    let types = block_value_types(block);
    for packet in &block.packets {
        for instr in &packet.instrs {
            for limb in 0..limbs(instr.ty.width) {
                let expr = instr_limb(machine, &types, instr, limb, scope, memory_layout, None);
                out.push_str(&format!(
                    "{indent}let {}: u32 = ({expr}) & {}u;\n",
                    value_name(instr.dst, limb, scope),
                    limb_mask(instr.ty.width, limb)
                ));
            }
        }
        for effect in &packet.effects {
            emit_machine_effect(
                machine,
                &types,
                effect,
                out,
                indent,
                scope,
                memory_layout,
                None,
            );
        }
    }
}

fn emit_tick_capture_block(
    machine: &PackedMachineProgram,
    out: &mut String,
    indent: &str,
    scope: &str,
    memory_layout: GpuMemoryLayout,
    reuse_temporaries: bool,
) -> Vec<usize> {
    if reuse_temporaries {
        return emit_reusable_tick_capture_block(machine, out, indent, scope, memory_layout);
    }
    let block = &machine.streams.tick_next;
    let types = block_value_types(block);
    let mut captured = Vec::new();
    for packet in &block.packets {
        for instr in &packet.instrs {
            for limb in 0..limbs(instr.ty.width) {
                let expr = instr_limb(machine, &types, instr, limb, scope, memory_layout, None);
                out.push_str(&format!(
                    "{indent}let {}: u32 = ({expr}) & {}u;\n",
                    value_name(instr.dst, limb, scope),
                    limb_mask(instr.ty.width, limb)
                ));
            }
        }
        for effect in &packet.effects {
            if let PackedEffect::CaptureReg { dst, value, reset } = effect {
                let signal = &machine.source.signals[*dst];
                for limb in 0..signal.layout.limbs {
                    let value_expr = machine_value_limb(&types, *value, limb, scope, None);
                    let expr = if let Some(reset) = reset {
                        let condition = reset_condition(&machine.source, reset);
                        let reset_value = reset.value.get(limb).copied().unwrap_or(0);
                        format!("select(({value_expr}), {reset_value}u, {condition})")
                    } else {
                        value_expr
                    };
                    out.push_str(&format!(
                        "{indent}var r{scope}_{dst}_{limb}: u32 = ({expr}) & {}u;\n",
                        limb_mask(signal.layout.width, limb)
                    ));
                }
                captured.push(*dst);
            }
        }
    }
    captured
}

type ValueSlotMap = HashMap<(PackedValueId, usize), usize>;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct TempSlotPlan {
    value_slots: ValueSlotMap,
    slot_count: usize,
    value_vars: usize,
}

fn emit_reusable_machine_block(
    machine: &PackedMachineProgram,
    block: &PackedBlock,
    out: &mut String,
    indent: &str,
    scope: &str,
    memory_layout: GpuMemoryLayout,
) {
    let types = block_value_types(block);
    let plan = build_temp_slot_plan(block);
    emit_temp_slot_decls(out, indent, scope, plan.slot_count);
    for packet in &block.packets {
        emit_reusable_packet(
            machine,
            &types,
            &plan.value_slots,
            packet,
            out,
            indent,
            scope,
            memory_layout,
            None,
        );
    }
}

fn emit_reusable_tick_capture_block(
    machine: &PackedMachineProgram,
    out: &mut String,
    indent: &str,
    scope: &str,
    memory_layout: GpuMemoryLayout,
) -> Vec<usize> {
    let block = &machine.streams.tick_next;
    let types = block_value_types(block);
    let plan = build_temp_slot_plan(block);
    let mut captured = Vec::new();
    let mut declared_captures = HashSet::new();
    emit_temp_slot_decls(out, indent, scope, plan.slot_count);
    for packet in &block.packets {
        for effect in &packet.effects {
            if let PackedEffect::CaptureReg { dst, .. } = effect {
                if !declared_captures.insert(*dst) {
                    continue;
                }
                let signal = &machine.source.signals[*dst];
                for limb in 0..signal.layout.limbs {
                    out.push_str(&format!("{indent}var r{scope}_{dst}_{limb}: u32 = 0u;\n"));
                }
            }
        }
    }
    for packet in &block.packets {
        emit_reusable_packet(
            machine,
            &types,
            &plan.value_slots,
            packet,
            out,
            indent,
            scope,
            memory_layout,
            Some(&mut captured),
        );
    }
    captured
}

fn emit_reusable_packet(
    machine: &PackedMachineProgram,
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    slots: &ValueSlotMap,
    packet: &rrtl_sim_ir::PackedMachinePacket,
    out: &mut String,
    indent: &str,
    scope: &str,
    memory_layout: GpuMemoryLayout,
    mut captured: Option<&mut Vec<usize>>,
) {
    out.push_str(&format!("{indent}{{\n"));
    let inner = format!("{indent}  ");
    for instr in &packet.instrs {
        for limb in 0..limbs(instr.ty.width) {
            let expr = instr_limb(
                machine,
                types,
                instr,
                limb,
                scope,
                memory_layout,
                Some(slots),
            );
            out.push_str(&format!(
                "{inner}let {}: u32 = ({expr}) & {}u;\n",
                packet_value_name(instr.dst, limb, scope),
                limb_mask(instr.ty.width, limb)
            ));
        }
    }
    for instr in &packet.instrs {
        for limb in 0..limbs(instr.ty.width) {
            let slot = slots[&(instr.dst, limb)];
            out.push_str(&format!(
                "{inner}{} = {};\n",
                slot_name(scope, slot),
                packet_value_name(instr.dst, limb, scope)
            ));
        }
    }
    for effect in &packet.effects {
        if let Some(captured) = captured.as_deref_mut() {
            if let PackedEffect::CaptureReg { dst, value, reset } = effect {
                emit_tick_capture_effect(
                    machine,
                    types,
                    slots,
                    *dst,
                    *value,
                    reset.as_ref(),
                    out,
                    &inner,
                    scope,
                );
                captured.push(*dst);
                continue;
            }
        }
        emit_machine_effect(
            machine,
            types,
            effect,
            out,
            &inner,
            scope,
            memory_layout,
            Some(slots),
        );
    }
    out.push_str(&format!("{indent}}}\n"));
}

fn emit_tick_capture_effect(
    machine: &PackedMachineProgram,
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    slots: &ValueSlotMap,
    dst: usize,
    value: PackedValueId,
    reset: Option<&PackedReset>,
    out: &mut String,
    indent: &str,
    scope: &str,
) {
    let signal = &machine.source.signals[dst];
    for limb in 0..signal.layout.limbs {
        let value_expr = machine_value_limb(types, value, limb, scope, Some(slots));
        let expr = if let Some(reset) = reset {
            let condition = reset_condition(&machine.source, reset);
            let reset_value = reset.value.get(limb).copied().unwrap_or(0);
            format!("select(({value_expr}), {reset_value}u, {condition})")
        } else {
            value_expr
        };
        out.push_str(&format!(
            "{indent}r{scope}_{dst}_{limb} = ({expr}) & {}u;\n",
            limb_mask(signal.layout.width, limb)
        ));
    }
}

fn emit_temp_slot_decls(out: &mut String, indent: &str, scope: &str, slot_count: usize) {
    for slot in 0..slot_count {
        out.push_str(&format!(
            "{indent}var {}: u32 = 0u;\n",
            slot_name(scope, slot)
        ));
    }
}

fn build_temp_slot_plan(block: &PackedBlock) -> TempSlotPlan {
    let mut plan = TempSlotPlan::default();
    let mut last_use = block_last_uses(block);
    let mut slot_values: Vec<Option<(PackedValueId, usize)>> = Vec::new();
    let mut free_slots = Vec::new();

    for (packet_index, packet) in block.packets.iter().enumerate() {
        for (slot, value) in slot_values.iter_mut().enumerate() {
            let Some((active_value, _)) = value else {
                continue;
            };
            if last_use.get(active_value).copied().unwrap_or(packet_index) < packet_index {
                *value = None;
                free_slots.push(slot);
            }
        }
        free_slots.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));
        free_slots.dedup();

        for instr in &packet.instrs {
            last_use.entry(instr.dst).or_insert(packet_index);
            for limb in 0..limbs(instr.ty.width) {
                plan.value_vars += 1;
                let slot = free_slots.pop().unwrap_or_else(|| {
                    slot_values.push(None);
                    slot_values.len() - 1
                });
                slot_values[slot] = Some((instr.dst, limb));
                plan.value_slots.insert((instr.dst, limb), slot);
            }
        }
    }
    plan.slot_count = slot_values.len();
    plan
}

fn block_last_uses(block: &PackedBlock) -> HashMap<PackedValueId, usize> {
    let mut uses = HashMap::new();
    for (packet_index, packet) in block.packets.iter().enumerate() {
        for instr in &packet.instrs {
            for value in instr_value_deps(&instr.kind) {
                uses.entry(value)
                    .and_modify(|last: &mut usize| *last = (*last).max(packet_index))
                    .or_insert(packet_index);
            }
        }
        for effect in &packet.effects {
            for value in effect_value_deps(effect) {
                uses.entry(value)
                    .and_modify(|last: &mut usize| *last = (*last).max(packet_index))
                    .or_insert(packet_index);
            }
        }
    }
    uses
}

fn instr_value_deps(kind: &PackedInstrKind) -> Vec<PackedValueId> {
    match kind {
        PackedInstrKind::Lit(_) | PackedInstrKind::Signal(_) => Vec::new(),
        PackedInstrKind::Not(value)
        | PackedInstrKind::Zext(value)
        | PackedInstrKind::Sext(value)
        | PackedInstrKind::Trunc(value)
        | PackedInstrKind::Cast(value)
        | PackedInstrKind::Slice { value, .. } => vec![*value],
        PackedInstrKind::And(lhs, rhs)
        | PackedInstrKind::Or(lhs, rhs)
        | PackedInstrKind::Xor(lhs, rhs)
        | PackedInstrKind::Add(lhs, rhs)
        | PackedInstrKind::Sub(lhs, rhs)
        | PackedInstrKind::Mul(lhs, rhs)
        | PackedInstrKind::Eq(lhs, rhs)
        | PackedInstrKind::Ne(lhs, rhs) => vec![*lhs, *rhs],
        PackedInstrKind::Lt { lhs, rhs, .. } => vec![*lhs, *rhs],
        PackedInstrKind::Mux {
            cond,
            then_value,
            else_value,
        } => vec![*cond, *then_value, *else_value],
        PackedInstrKind::Concat(values) => values.clone(),
        PackedInstrKind::MemRead { addr, .. } => vec![*addr],
    }
}

fn effect_value_deps(effect: &PackedEffect) -> Vec<PackedValueId> {
    match effect {
        PackedEffect::StoreSignal { value, .. } | PackedEffect::CaptureReg { value, .. } => {
            vec![*value]
        }
        PackedEffect::MemoryWrite {
            enable, addr, data, ..
        } => vec![*enable, *addr, *data],
    }
}

fn emit_machine_effect(
    machine: &PackedMachineProgram,
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    effect: &PackedEffect,
    out: &mut String,
    indent: &str,
    scope: &str,
    memory_layout: GpuMemoryLayout,
    slots: Option<&ValueSlotMap>,
) {
    match effect {
        PackedEffect::StoreSignal { dst, value } => {
            let signal = &machine.source.signals[*dst];
            for limb in 0..signal.layout.limbs {
                let expr = machine_value_limb(types, *value, limb, scope, slots);
                out.push_str(&format!(
                    "{indent}store_value({}u, {limb}u, lane, ({expr}) & {}u);\n",
                    signal.layout.offset,
                    limb_mask(signal.layout.width, limb)
                ));
            }
        }
        PackedEffect::CaptureReg { dst, value, reset } => {
            let Some(reset) = reset else {
                return;
            };
            if reset.kind != rrtl_ir::ResetKind::Async {
                return;
            }
            let signal = &machine.source.signals[*dst];
            let condition = reset_condition(&machine.source, reset);
            out.push_str(&format!("{indent}if ({condition}) {{\n"));
            for limb in 0..signal.layout.limbs {
                let expr = machine_value_limb(types, *value, limb, scope, slots);
                out.push_str(&format!(
                    "{indent}  store_value({}u, {limb}u, lane, ({expr}) & {}u);\n",
                    signal.layout.offset,
                    limb_mask(signal.layout.width, limb)
                ));
            }
            out.push_str(&format!("{indent}}}\n"));
        }
        PackedEffect::MemoryWrite {
            memory,
            enable,
            addr,
            data,
        } => {
            let mem = &machine.source.memories[*memory];
            let enable = machine_value_bool(types, *enable, scope, slots);
            let addr0 = machine_value_limb(types, *addr, 0, scope, slots);
            let condition = if let Some(in_range) =
                machine_addr_in_range(types, *addr, mem.depth, scope, slots)
            {
                format!("({enable}) && ({in_range})")
            } else {
                enable
            };
            out.push_str(&format!("{indent}if ({condition}) {{\n"));
            for limb in 0..mem.data_layout.limbs {
                let data = machine_value_limb(types, *data, limb, scope, slots);
                let index = memory_index_expr(mem, memory_layout, &addr0, limb);
                out.push_str(&format!(
                    "{indent}  memories[{index}] = ({data}) & {}u;\n",
                    limb_mask(mem.data_layout.width, limb)
                ));
            }
            out.push_str(&format!("{indent}}}\n"));
        }
    }
}

fn instr_limb(
    machine: &PackedMachineProgram,
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    instr: &PackedInstr,
    limb: usize,
    scope: &str,
    memory_layout: GpuMemoryLayout,
    slots: Option<&ValueSlotMap>,
) -> String {
    match &instr.kind {
        PackedInstrKind::Lit(values) => format!("{}u", values.get(limb).copied().unwrap_or(0)),
        PackedInstrKind::Signal(signal) => {
            let layout = machine.source.signals[*signal].layout;
            if limb >= layout.limbs {
                "0u".to_string()
            } else {
                format!("load_value({}u, {limb}u, lane)", layout.offset)
            }
        }
        PackedInstrKind::Not(value) => {
            format!(
                "~({})",
                machine_value_limb(types, *value, limb, scope, slots)
            )
        }
        PackedInstrKind::And(lhs, rhs) => {
            machine_binary_limb(types, *lhs, *rhs, limb, "&", scope, slots)
        }
        PackedInstrKind::Or(lhs, rhs) => {
            machine_binary_limb(types, *lhs, *rhs, limb, "|", scope, slots)
        }
        PackedInstrKind::Xor(lhs, rhs) => {
            machine_binary_limb(types, *lhs, *rhs, limb, "^", scope, slots)
        }
        PackedInstrKind::Add(lhs, rhs) => format!(
            "add_word(({}), ({}), ({}))",
            machine_value_limb(types, *lhs, limb, scope, slots),
            machine_value_limb(types, *rhs, limb, scope, slots),
            machine_add_carry(types, *lhs, *rhs, limb, scope, slots)
        ),
        PackedInstrKind::Sub(lhs, rhs) => format!(
            "sub_word(({}), ({}), ({}))",
            machine_value_limb(types, *lhs, limb, scope, slots),
            machine_value_limb(types, *rhs, limb, scope, slots),
            machine_sub_borrow(types, *lhs, *rhs, limb, scope, slots)
        ),
        PackedInstrKind::Mul(lhs, rhs) => {
            if limb == 0 {
                format!(
                    "({} * {})",
                    machine_value_limb(types, *lhs, 0, scope, slots),
                    machine_value_limb(types, *rhs, 0, scope, slots)
                )
            } else {
                "0u".to_string()
            }
        }
        PackedInstrKind::Eq(lhs, rhs) => {
            if limb == 0 {
                format!(
                    "select(0u, 1u, {})",
                    machine_eq_bool(types, *lhs, *rhs, scope, slots)
                )
            } else {
                "0u".to_string()
            }
        }
        PackedInstrKind::Ne(lhs, rhs) => {
            if limb == 0 {
                format!(
                    "select(0u, 1u, !({}))",
                    machine_eq_bool(types, *lhs, *rhs, scope, slots)
                )
            } else {
                "0u".to_string()
            }
        }
        PackedInstrKind::Lt { lhs, rhs, signed } => {
            if limb == 0 {
                format!(
                    "select(0u, 1u, {})",
                    machine_lt_bool(types, *lhs, *rhs, *signed, scope, slots)
                )
            } else {
                "0u".to_string()
            }
        }
        PackedInstrKind::Mux {
            cond,
            then_value,
            else_value,
        } => format!(
            "select(({}), ({}), {})",
            machine_value_limb(types, *else_value, limb, scope, slots),
            machine_value_limb(types, *then_value, limb, scope, slots),
            machine_value_bool(types, *cond, scope, slots)
        ),
        PackedInstrKind::Slice { value, lsb } => {
            machine_slice_bits(types, *value, *lsb + limb as u32 * 32, 32, scope, slots)
        }
        PackedInstrKind::Zext(value)
        | PackedInstrKind::Trunc(value)
        | PackedInstrKind::Cast(value) => machine_value_limb(types, *value, limb, scope, slots),
        PackedInstrKind::Sext(value) => {
            machine_sext_limb(types, *value, instr.ty.width, limb, scope, slots)
        }
        PackedInstrKind::Concat(values) => machine_concat_limb(types, values, limb, scope, slots),
        PackedInstrKind::MemRead { memory, addr } => machine_mem_read_limb(
            machine,
            types,
            *memory,
            *addr,
            limb,
            scope,
            memory_layout,
            slots,
        ),
    }
}

fn block_value_types(block: &PackedBlock) -> HashMap<PackedValueId, rrtl_ir::BitType> {
    block
        .packets
        .iter()
        .flat_map(|packet| packet.instrs.iter().map(|instr| (instr.dst, instr.ty)))
        .collect()
}

fn value_name(value: PackedValueId, limb: usize, scope: &str) -> String {
    format!("v{scope}_{}_{}", value.0, limb)
}

fn packet_value_name(value: PackedValueId, limb: usize, scope: &str) -> String {
    format!("p{scope}_{}_{}", value.0, limb)
}

fn slot_name(scope: &str, slot: usize) -> String {
    format!("s{scope}_{slot}")
}

fn machine_value_limb(
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    value: PackedValueId,
    limb: usize,
    scope: &str,
    slots: Option<&ValueSlotMap>,
) -> String {
    let Some(ty) = types.get(&value).copied() else {
        return "0u".to_string();
    };
    if limb >= limbs(ty.width) {
        "0u".to_string()
    } else if let Some(slot) = slots.and_then(|slots| slots.get(&(value, limb)).copied()) {
        slot_name(scope, slot)
    } else {
        value_name(value, limb, scope)
    }
}

fn machine_binary_limb(
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    lhs: PackedValueId,
    rhs: PackedValueId,
    limb: usize,
    op: &str,
    scope: &str,
    slots: Option<&ValueSlotMap>,
) -> String {
    format!(
        "(({}) {op} ({}))",
        machine_value_limb(types, lhs, limb, scope, slots),
        machine_value_limb(types, rhs, limb, scope, slots)
    )
}

fn machine_add_carry(
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    lhs: PackedValueId,
    rhs: PackedValueId,
    limb: usize,
    scope: &str,
    slots: Option<&ValueSlotMap>,
) -> String {
    if limb == 0 {
        "0u".to_string()
    } else {
        format!(
            "add_carry(({}), ({}), ({}))",
            machine_value_limb(types, lhs, limb - 1, scope, slots),
            machine_value_limb(types, rhs, limb - 1, scope, slots),
            machine_add_carry(types, lhs, rhs, limb - 1, scope, slots)
        )
    }
}

fn machine_sub_borrow(
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    lhs: PackedValueId,
    rhs: PackedValueId,
    limb: usize,
    scope: &str,
    slots: Option<&ValueSlotMap>,
) -> String {
    if limb == 0 {
        "0u".to_string()
    } else {
        format!(
            "sub_borrow(({}), ({}), ({}))",
            machine_value_limb(types, lhs, limb - 1, scope, slots),
            machine_value_limb(types, rhs, limb - 1, scope, slots),
            machine_sub_borrow(types, lhs, rhs, limb - 1, scope, slots)
        )
    }
}

fn machine_eq_bool(
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    lhs: PackedValueId,
    rhs: PackedValueId,
    scope: &str,
    slots: Option<&ValueSlotMap>,
) -> String {
    let width = types
        .get(&lhs)
        .into_iter()
        .chain(types.get(&rhs))
        .map(|ty| ty.width)
        .max()
        .unwrap_or(1);
    (0..limbs(width))
        .map(|limb| {
            format!(
                "(({}) & {}u) == (({}) & {}u)",
                machine_value_limb(types, lhs, limb, scope, slots),
                machine_limb_mask(types, lhs, limb),
                machine_value_limb(types, rhs, limb, scope, slots),
                machine_limb_mask(types, rhs, limb)
            )
        })
        .collect::<Vec<_>>()
        .join(" && ")
}

fn machine_lt_bool(
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    lhs: PackedValueId,
    rhs: PackedValueId,
    signed: bool,
    scope: &str,
    slots: Option<&ValueSlotMap>,
) -> String {
    let unsigned = machine_unsigned_lt_bool(types, lhs, rhs, scope, slots);
    if !signed {
        return unsigned;
    }
    let width = types.get(&lhs).map(|ty| ty.width).unwrap_or(1);
    let sign_limb = (width as usize - 1) / 32;
    let sign_bit = (width - 1) % 32;
    let lhs_sign = format!(
        "(({}) & {}u) != 0u",
        machine_value_limb(types, lhs, sign_limb, scope, slots),
        1u32 << sign_bit
    );
    let rhs_sign = format!(
        "(({}) & {}u) != 0u",
        machine_value_limb(types, rhs, sign_limb, scope, slots),
        1u32 << sign_bit
    );
    format!("select(({unsigned}), ({lhs_sign}), ({lhs_sign}) != ({rhs_sign}))")
}

fn machine_unsigned_lt_bool(
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    lhs: PackedValueId,
    rhs: PackedValueId,
    scope: &str,
    slots: Option<&ValueSlotMap>,
) -> String {
    let width = types
        .get(&lhs)
        .into_iter()
        .chain(types.get(&rhs))
        .map(|ty| ty.width)
        .max()
        .unwrap_or(1);
    let mut expr = "false".to_string();
    for limb in 0..limbs(width) {
        let i = limbs(width) - 1 - limb;
        let lhs_limb = format!(
            "(({}) & {}u)",
            machine_value_limb(types, lhs, i, scope, slots),
            machine_limb_mask(types, lhs, i)
        );
        let rhs_limb = format!(
            "(({}) & {}u)",
            machine_value_limb(types, rhs, i, scope, slots),
            machine_limb_mask(types, rhs, i)
        );
        expr =
            format!("select(({expr}), ({lhs_limb}) < ({rhs_limb}), ({lhs_limb}) != ({rhs_limb}))");
    }
    expr
}

fn machine_slice_bits(
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    value: PackedValueId,
    lsb: Width,
    width: Width,
    scope: &str,
    slots: Option<&ValueSlotMap>,
) -> String {
    let value_width = types.get(&value).map(|ty| ty.width).unwrap_or(0);
    if width == 0 || lsb >= value_width {
        return "0u".to_string();
    }
    let low_limb = (lsb / 32) as usize;
    let shift = lsb % 32;
    let low = machine_value_limb(types, value, low_limb, scope, slots);
    let high = if shift == 0 {
        "0u".to_string()
    } else {
        machine_value_limb(types, value, low_limb + 1, scope, slots)
    };
    let combined = if shift == 0 {
        low
    } else {
        format!("((({low}) >> {shift}u) | (({high}) << {}u))", 32 - shift)
    };
    format!("(({combined}) & {}u)", low_mask(width.min(32)))
}

fn machine_sext_limb(
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    value: PackedValueId,
    output_width: Width,
    limb: usize,
    scope: &str,
    slots: Option<&ValueSlotMap>,
) -> String {
    let input_width = types.get(&value).map(|ty| ty.width).unwrap_or(output_width);
    let sign_limb = (input_width as usize - 1) / 32;
    let sign_bit = (input_width - 1) % 32;
    let sign = format!(
        "(({}) & {}u) != 0u",
        machine_value_limb(types, value, sign_limb, scope, slots),
        1u32 << sign_bit
    );
    if limb < sign_limb {
        machine_value_limb(types, value, limb, scope, slots)
    } else if limb == sign_limb {
        let mask = final_limb_mask(input_width);
        let word = machine_value_limb(types, value, limb, scope, slots);
        format!("select(({word}) & {mask}u, ({word}) | (~{mask}u), {sign})")
    } else {
        format!("select(0u, 0xffffffffu, {sign})")
    }
}

fn machine_concat_limb(
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    values: &[PackedValueId],
    limb: usize,
    scope: &str,
    slots: Option<&ValueSlotMap>,
) -> String {
    let dst_start = limb as u32 * 32;
    let dst_end = dst_start + 32;
    let mut part_start = 0;
    let mut terms = Vec::new();
    for value in values.iter().rev().copied() {
        let width = types.get(&value).map(|ty| ty.width).unwrap_or(0);
        let part_end = part_start + width;
        let overlap_start = dst_start.max(part_start);
        let overlap_end = dst_end.min(part_end);
        if overlap_start < overlap_end {
            let bits = machine_slice_bits(
                types,
                value,
                overlap_start - part_start,
                overlap_end - overlap_start,
                scope,
                slots,
            );
            terms.push(format!("(({bits}) << {}u)", overlap_start - dst_start));
        }
        part_start = part_end;
    }
    if terms.is_empty() {
        "0u".to_string()
    } else {
        terms.join(" | ")
    }
}

fn machine_mem_read_limb(
    machine: &PackedMachineProgram,
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    memory: usize,
    addr: PackedValueId,
    limb: usize,
    scope: &str,
    memory_layout: GpuMemoryLayout,
    slots: Option<&ValueSlotMap>,
) -> String {
    let mem = &machine.source.memories[memory];
    if limb >= mem.data_layout.limbs {
        return "0u".to_string();
    }
    let addr0 = machine_value_limb(types, addr, 0, scope, slots);
    let index = memory_index_expr(mem, memory_layout, &addr0, limb);
    let load = format!("memories[{index}]");
    if let Some(in_range) = machine_addr_in_range(types, addr, mem.depth, scope, slots) {
        format!("select(0u, {load}, {in_range})")
    } else {
        load
    }
}

fn memory_index_expr(
    memory: &rrtl_sim_ir::PackedMemory,
    memory_layout: GpuMemoryLayout,
    addr: &str,
    limb: usize,
) -> String {
    let word = memory_word_expr(memory.offset, memory.data_layout.limbs, addr, limb);
    match memory_layout {
        GpuMemoryLayout::LaneMajor => {
            format!("lane * params.memory_words_per_lane + {word}")
        }
        GpuMemoryLayout::WordMajor => format!("({word}) * params.lanes + lane"),
    }
}

fn memory_word_expr(offset: usize, limbs: usize, addr: &str, limb: usize) -> String {
    let mut terms = Vec::new();
    if offset != 0 {
        terms.push(format!("{offset}u"));
    }
    if limbs == 1 {
        terms.push(format!("({addr})"));
    } else {
        terms.push(format!("({addr}) * {limbs}u"));
    }
    if limb != 0 {
        terms.push(format!("{limb}u"));
    }
    terms.join(" + ")
}

fn machine_addr_in_range(
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    addr: PackedValueId,
    depth: usize,
    scope: &str,
    slots: Option<&ValueSlotMap>,
) -> Option<String> {
    let width = types.get(&addr).map(|ty| ty.width).unwrap_or(1);
    if addr_width_exactly_covers_depth(width, depth) {
        return None;
    }
    let mut checks = vec![format!(
        "({}) < {}u",
        machine_value_limb(types, addr, 0, scope, slots),
        depth
    )];
    for limb in 1..limbs(width) {
        checks.push(format!(
            "({}) == 0u",
            machine_value_limb(types, addr, limb, scope, slots)
        ));
    }
    Some(checks.join(" && "))
}

fn addr_width_exactly_covers_depth(width: Width, depth: usize) -> bool {
    depth.is_power_of_two() && width < usize::BITS && (1usize << width) == depth
}

fn machine_value_bool(
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    value: PackedValueId,
    scope: &str,
    slots: Option<&ValueSlotMap>,
) -> String {
    format!(
        "(({}) & 1u) != 0u",
        machine_value_limb(types, value, 0, scope, slots)
    )
}

fn machine_limb_mask(
    types: &HashMap<PackedValueId, rrtl_ir::BitType>,
    value: PackedValueId,
    limb: usize,
) -> u32 {
    types
        .get(&value)
        .map(|ty| limb_mask(ty.width, limb))
        .unwrap_or(0)
}

fn reset_condition(program: &PackedProgram, reset: &PackedReset) -> String {
    let value = format!(
        "load_value({}u, 0u, lane)",
        program.signals[reset.signal].layout.offset
    );
    match reset.polarity {
        rrtl_ir::ResetPolarity::ActiveHigh => format!("({value}) != 0u"),
        rrtl_ir::ResetPolarity::ActiveLow => format!("({value}) == 0u"),
    }
}

fn limb_mask(width: Width, limb: usize) -> u32 {
    if limb + 1 == limbs(width) {
        final_limb_mask(width)
    } else if limb < limbs(width) {
        u32::MAX
    } else {
        0
    }
}

fn low_mask(width: Width) -> u32 {
    if width >= 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    (value + divisor - 1) / divisor
}

fn buffer_size(words: usize) -> u64 {
    (words * std::mem::size_of::<u32>()) as u64
}

fn buffer_size_nonzero(words: usize) -> u64 {
    buffer_size(words.max(1))
}

fn byte_offset(words: usize) -> u64 {
    buffer_size(words)
}

fn error(code: &'static str, message: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![Diagnostic::new(code, message)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use rrtl_core::{concat, lit_u, mux, sint, uint, Simulator};
    use rrtl_sim_ir::PackedSimulator;

    fn alu_design() -> (Design, Signal, Signal, Signal, Signal) {
        let mut design = Design::new();
        let (a, b, sum, eq);
        {
            let mut m = design.module("Alu");
            a = m.input("a", uint(8));
            b = m.input("b", uint(8));
            sum = m.output("sum", uint(8));
            eq = m.output("eq", uint(1));
            m.assign(sum, a + b);
            m.assign(eq, a.value().eq_expr(b));
        }
        (design, a, b, sum, eq)
    }

    #[test]
    fn generates_wgsl_from_packed_ir() {
        let (design, ..) = alu_design();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Alu").unwrap();
        let wgsl = generate_wgsl(&program, GpuBatchOptions::default()).unwrap();
        assert!(wgsl.contains("fn eval_comb"));
        assert!(wgsl.contains("fn tick"));
        assert!(wgsl.contains("fn tick_many"));
        assert!(wgsl.contains("params.steps"));
        assert!(wgsl.contains("store_value"));
        assert!(wgsl.contains("let ve"));
    }

    #[test]
    fn generated_replay_wgsl_uses_step_ranges() {
        let (design, ..) = alu_design();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Alu").unwrap();
        let wgsl = generate_wgsl(&program, GpuBatchOptions::default()).unwrap();

        assert!(wgsl.contains("if (trace_data[trace_header_base + 0u] == 2u)"));
        assert!(wgsl.contains("if (trace_data[trace_header_base + 0u] == 1u)"));
        assert!(wgsl.contains("trace_input_offsets_base"));
        assert!(wgsl.contains("trace_check_offsets_base"));
        assert!(wgsl.contains("let fixed_input_stride = trace_data[trace_header_base + 6u]"));
        assert!(wgsl.contains("let fixed_check_stride = trace_data[trace_header_base + 7u]"));
        assert!(wgsl.contains("let input_dense_slots = trace_data[trace_header_base + 8u]"));
        assert!(wgsl.contains("let check_dense_slots = trace_data[trace_header_base + 9u]"));
        assert!(wgsl.contains("input_op * params.lanes + lane"));
        assert!(wgsl.contains("check_op * params.lanes + lane"));
        assert!(wgsl.contains("var input_op = trace_data[trace_input_offsets_base + step]"));
        assert!(wgsl.contains("let input_end = trace_data[trace_input_offsets_base + step + 1u]"));
        assert!(wgsl.contains("var check_op = trace_data[trace_check_offsets_base + step]"));
        assert!(wgsl.contains("let check_end = trace_data[trace_check_offsets_base + step + 1u]"));
        assert!(!wgsl.contains("prior_slot"));
        assert!(!wgsl.contains("fixed_input_slot"));
        assert!(!wgsl.contains("fixed_check_slot"));
        assert!(!wgsl.contains("fixed_template"));
        let fixed_branch = wgsl
            .split("if (trace_data[trace_header_base + 0u] == 2u)")
            .nth(1)
            .and_then(|tail| {
                tail.split("if (trace_data[trace_header_base + 0u] == 1u)")
                    .next()
            })
            .unwrap();
        assert!(!fixed_branch.contains("trace_value_mode"));
        assert!(!fixed_branch.contains("trace_input_template_offset_base"));
        assert!(!fixed_branch.contains("trace_check_template_offset_base"));
        assert!(!wgsl.contains("if (input_op >= params.trace_input_ops)"));
        assert!(!wgsl.contains("if (check_op >= params.trace_check_ops)"));
    }

    #[test]
    fn trace_replay_packer_groups_ops_by_step() {
        let (design, a, b, sum, eq) = alu_design();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Alu").unwrap();
        let a_layout = program.signals[program.signal_index(a).unwrap()].layout;
        let b_layout = program.signals[program.signal_index(b).unwrap()].layout;
        let sum_layout = program.signals[program.signal_index(sum).unwrap()].layout;
        let eq_layout = program.signals[program.signal_index(eq).unwrap()].layout;
        let plan = GpuTraceReplayPlan {
            steps: 3,
            inputs: vec![
                GpuTraceInputOp {
                    step: 2,
                    signal: b,
                    limb: 0,
                    values: vec![0x1ff, 0xff],
                },
                GpuTraceInputOp {
                    step: 0,
                    signal: a,
                    limb: 0,
                    values: vec![0x11, 0x12],
                },
                GpuTraceInputOp {
                    step: 2,
                    signal: a,
                    limb: 0,
                    values: vec![0x21, 0x22],
                },
            ],
            checks: vec![
                GpuTraceCheckOp {
                    step: 1,
                    check_index: 7,
                    signal: sum,
                    limb: 0,
                    expected: vec![0x33, 0x34],
                },
                GpuTraceCheckOp {
                    step: 0,
                    check_index: 8,
                    signal: eq,
                    limb: 0,
                    expected: vec![0x3, 0x0],
                },
            ],
        };

        let packed = pack_trace_replay_for_program(&program, 2, &plan).unwrap();
        assert_eq!(packed.input_ops, 3);
        assert_eq!(packed.check_ops, 2);

        let input_offsets = &packed.trace_data[0..4];
        let check_offsets = &packed.trace_data[4..8];
        let trace_header = &packed.trace_data[8..18];
        assert_eq!(input_offsets, &[0, 1, 1, 3]);
        assert_eq!(check_offsets, &[0, 1, 2, 2]);
        assert_eq!(trace_header, &[0, 5, 4, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(packed.uniform_input_ops, 1);
        assert_eq!(packed.uniform_check_ops, 0);
        assert_eq!(packed.uncompressed_words, 53);
        assert_eq!(packed.trace_data.len(), 52);
        assert!(!packed.template_layout);

        let input_meta = &packed.trace_data[18..33];
        assert_eq!(
            input_meta,
            &[
                a_layout.offset as u32,
                0,
                0,
                0,
                0,
                b_layout.offset as u32,
                0,
                1,
                2,
                0,
                a_layout.offset as u32,
                0,
                0,
                3,
                0,
            ]
        );
        let input_values = &packed.trace_data[33..38];
        assert_eq!(input_values, &[0x11, 0x12, 0xff, 0x21, 0x22]);

        let check_meta = &packed.trace_data[38..48];
        assert_eq!(
            check_meta,
            &[
                8,
                eq_layout.offset as u32,
                0,
                0,
                0,
                7,
                sum_layout.offset as u32,
                0,
                0,
                2,
            ]
        );
        let check_values = &packed.trace_data[48..52];
        assert_eq!(check_values, &[1, 0, 0x33, 0x34]);
    }

    #[test]
    fn trace_replay_packer_uses_template_for_stable_step_signatures() {
        let (design, a, b, sum, eq) = alu_design();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Alu").unwrap();
        let a_layout = program.signals[program.signal_index(a).unwrap()].layout;
        let b_layout = program.signals[program.signal_index(b).unwrap()].layout;
        let sum_layout = program.signals[program.signal_index(sum).unwrap()].layout;
        let eq_layout = program.signals[program.signal_index(eq).unwrap()].layout;
        let plan = GpuTraceReplayPlan {
            steps: 2,
            inputs: vec![
                GpuTraceInputOp {
                    step: 0,
                    signal: a,
                    limb: 0,
                    values: vec![0x10, 0x10],
                },
                GpuTraceInputOp {
                    step: 0,
                    signal: b,
                    limb: 0,
                    values: vec![0x01, 0x02],
                },
                GpuTraceInputOp {
                    step: 1,
                    signal: a,
                    limb: 0,
                    values: vec![0x20, 0x20],
                },
                GpuTraceInputOp {
                    step: 1,
                    signal: b,
                    limb: 0,
                    values: vec![0x03, 0x04],
                },
            ],
            checks: vec![
                GpuTraceCheckOp {
                    step: 0,
                    check_index: 0,
                    signal: sum,
                    limb: 0,
                    expected: vec![0x11, 0x12],
                },
                GpuTraceCheckOp {
                    step: 0,
                    check_index: 1,
                    signal: eq,
                    limb: 0,
                    expected: vec![0, 0],
                },
                GpuTraceCheckOp {
                    step: 1,
                    check_index: 0,
                    signal: sum,
                    limb: 0,
                    expected: vec![0x23, 0x24],
                },
                GpuTraceCheckOp {
                    step: 1,
                    check_index: 1,
                    signal: eq,
                    limb: 0,
                    expected: vec![0, 0],
                },
            ],
        };

        let packed = pack_trace_replay_for_program(&program, 2, &plan).unwrap();
        assert!(packed.template_layout);
        assert_eq!(packed.template_input_ops, 2);
        assert_eq!(packed.template_check_ops, 2);
        assert_eq!(packed.uniform_input_ops, 2);
        assert_eq!(packed.uniform_check_ops, 2);
        assert!(packed.metadata_saved_words > 0);

        let input_offsets = &packed.trace_data[0..3];
        let check_offsets = &packed.trace_data[3..6];
        let trace_header = &packed.trace_data[6..16];
        assert_eq!(input_offsets, &[0, 2, 4]);
        assert_eq!(check_offsets, &[0, 2, 4]);
        assert_eq!(trace_header[0], 2);
        assert_eq!(trace_header[3], 2);
        assert_eq!(trace_header[4], 2);
        assert_eq!(trace_header[6], 3);
        assert_eq!(trace_header[7], 3);
        assert_eq!(trace_header[8], 1);
        assert_eq!(trace_header[9], 1);
        assert!(packed.fixed_template);
        assert!(packed.value_metadata_saved_words > 0);
        assert!(packed.value_stride_words > 0);

        let input_template = &packed.trace_data[16..20];
        assert_eq!(
            input_template,
            &[b_layout.offset as u32, 0, a_layout.offset as u32, 0]
        );
        let input_values = &packed.trace_data[20..26];
        assert_eq!(input_values, &[0x01, 0x02, 0x10, 0x03, 0x04, 0x20]);
        let check_template_base = 16 + 4 + trace_header[1] as usize;
        let check_template = &packed.trace_data[check_template_base..check_template_base + 6];
        assert_eq!(
            check_template,
            &[
                0,
                sum_layout.offset as u32,
                0,
                1,
                eq_layout.offset as u32,
                0,
            ]
        );
        let check_values = &packed.trace_data[check_template_base + 6..check_template_base + 12];
        assert_eq!(check_values, &[0x11, 0x12, 0, 0x23, 0x24, 0]);
    }

    #[test]
    fn trace_replay_packer_keeps_dynamic_template_stride_headers_zero() {
        let (design, a, b, sum, eq) = alu_design();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Alu").unwrap();
        let plan = GpuTraceReplayPlan {
            steps: 2,
            inputs: vec![
                GpuTraceInputOp {
                    step: 0,
                    signal: a,
                    limb: 0,
                    values: vec![0x10, 0x10],
                },
                GpuTraceInputOp {
                    step: 0,
                    signal: b,
                    limb: 0,
                    values: vec![0x01, 0x02],
                },
                GpuTraceInputOp {
                    step: 1,
                    signal: a,
                    limb: 0,
                    values: vec![0x20, 0x21],
                },
                GpuTraceInputOp {
                    step: 1,
                    signal: b,
                    limb: 0,
                    values: vec![0x03, 0x04],
                },
            ],
            checks: vec![
                GpuTraceCheckOp {
                    step: 0,
                    check_index: 0,
                    signal: sum,
                    limb: 0,
                    expected: vec![0x11, 0x12],
                },
                GpuTraceCheckOp {
                    step: 0,
                    check_index: 1,
                    signal: eq,
                    limb: 0,
                    expected: vec![0, 0],
                },
                GpuTraceCheckOp {
                    step: 1,
                    check_index: 0,
                    signal: sum,
                    limb: 0,
                    expected: vec![0x23, 0x25],
                },
                GpuTraceCheckOp {
                    step: 1,
                    check_index: 1,
                    signal: eq,
                    limb: 0,
                    expected: vec![0, 1],
                },
            ],
        };

        let packed = pack_trace_replay_for_program(&program, 2, &plan).unwrap();
        assert!(packed.template_layout);
        assert!(!packed.fixed_template);
        assert_eq!(packed.value_stride_words, 0);

        let trace_header = &packed.trace_data[6..16];
        assert_eq!(trace_header[0], 1);
        assert_eq!(trace_header[6], 0);
        assert_eq!(trace_header[7], 0);
        assert_eq!(trace_header[8], 0);
        assert_eq!(trace_header[9], 0);
    }

    #[test]
    fn gpu_region_analysis_reports_compute_candidate() {
        let mut design = Design::new();
        {
            let mut m = design.module("GpuComputeRegion");
            let a = m.input("a", uint(16));
            let b = m.input("b", uint(16));
            let y = m.output("y", uint(16));
            let mut expr = a.value();
            for _ in 0..80 {
                expr = ((expr + b.value()) ^ (a.value() | b.value())).trunc(16);
            }
            m.assign(y, expr);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "GpuComputeRegion").unwrap();
        let analysis = analyze_gpu_regions(&program).unwrap();
        assert_eq!(
            analysis.recommendation,
            GpuRegionRecommendation::ComputeCandidate
        );
        assert!(analysis.total.pure_compute_regions > 0);
        assert!(analysis.total.pure_compute_instrs >= 64);
        assert_eq!(analysis.total.memory_reads, 0);
        assert_eq!(analysis.total.memory_writes, 0);
    }

    #[test]
    fn gpu_region_analysis_reports_wide_and_memory_pressure() {
        let mut design = Design::new();
        {
            let mut m = design.module("GpuMemoryRegion");
            let clk = m.input("clk", uint(1));
            let we = m.input("we", uint(1));
            let addr = m.input("addr", uint(2));
            let data = m.input("data", uint(40));
            let mem = m.mem("mem", 2, uint(40), 4);
            let y = m.output("y", uint(40));
            let read = m.mem_read(mem, addr);
            m.assign(y, (read + data).trunc(40));
            m.mem_write(mem, clk, we, addr, data);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "GpuMemoryRegion").unwrap();
        let analysis = analyze_gpu_regions(&program).unwrap();
        assert!(matches!(
            analysis.recommendation,
            GpuRegionRecommendation::MemoryBlocked | GpuRegionRecommendation::MixedCandidate
        ));
        assert!(analysis.total.memory_reads > 0);
        assert!(analysis.total.memory_writes > 0);
        assert!(analysis.total.wide_instrs > 0);
    }

    #[test]
    fn default_gpu_options_use_width_cap_sixteen() {
        let options = GpuBatchOptions::default();
        assert_eq!(options.schedule.max_packet_width, Some(16));
        assert!(!options.schedule.liveness_priority);
        assert_eq!(options.memory_layout, GpuMemoryLayout::LaneMajor);
        assert_eq!(options.workgroup_size, WORKGROUP_SIZE);
        assert!(!options.reuse_temporaries);
    }

    fn sample_recommendation() -> GpuAutotuneRecommendation {
        GpuAutotuneRecommendation {
            case: "counter".to_string(),
            lanes: 64,
            steps: 16,
            schedule_cap: Some(8),
            memory_read_cap: Some(2),
            liveness_priority: true,
            reuse_temporaries: true,
            memory_layout: "word_major".to_string(),
            workgroup_size: 128,
            autotune_metric: "gpu_tick_many".to_string(),
            autotune_metric_ns: Some(100),
            packed_ns: 200,
            gpu_tick_ns: Some(150),
            gpu_tick_many_ns: Some(100),
        }
    }

    #[test]
    fn loads_gpu_autotune_recommendations_from_json() {
        let json = br#"[
            {
                "case": "counter",
                "lanes": 64,
                "steps": 16,
                "schedule_cap": 8,
                "memory_read_cap": 2,
                "liveness_priority": true,
                "reuse_temporaries": true,
                "memory_layout": "word_major",
                "workgroup_size": 128,
                "autotune_metric": "gpu_tick_many",
                "autotune_metric_ns": 100,
                "packed_ns": 200,
                "gpu_tick_ns": 150,
                "gpu_tick_many_ns": 100
            }
        ]"#;

        let recommendations = load_gpu_autotune_recommendations(&json[..]).unwrap();

        assert_eq!(recommendations, vec![sample_recommendation()]);
    }

    #[test]
    fn finds_gpu_autotune_recommendation_by_exact_key() {
        let mut other = sample_recommendation();
        other.case = "wide_datapath".to_string();
        other.steps = 1;
        let recommendations = vec![other, sample_recommendation()];

        let found = find_gpu_autotune_recommendation(&recommendations, "counter", 64, 16);
        let missing = find_gpu_autotune_recommendation(&recommendations, "counter", 64, 1);

        assert_eq!(found, Some(&recommendations[1]));
        assert!(missing.is_none());
    }

    #[test]
    fn exact_match_mode_preserves_exact_lookup_behavior() {
        let recommendations = vec![sample_recommendation()];

        let found = find_gpu_autotune_recommendation_with_mode(
            &recommendations,
            "counter",
            64,
            16,
            GpuAutotuneMatchMode::Exact,
        );
        let missing = find_gpu_autotune_recommendation_with_mode(
            &recommendations,
            "counter",
            64,
            1,
            GpuAutotuneMatchMode::Exact,
        );

        assert_eq!(found, Some(&recommendations[0]));
        assert!(missing.is_none());
    }

    #[test]
    fn nearest_match_uses_same_case_only() {
        let mut wrong_case_exact = sample_recommendation();
        wrong_case_exact.case = "wide_datapath".to_string();
        wrong_case_exact.lanes = 128;
        wrong_case_exact.steps = 32;
        let mut same_case_far = sample_recommendation();
        same_case_far.lanes = 4096;
        same_case_far.steps = 512;
        let recommendations = vec![wrong_case_exact, same_case_far];

        let found = find_gpu_autotune_recommendation_with_mode(
            &recommendations,
            "counter",
            128,
            32,
            GpuAutotuneMatchMode::Nearest,
        );

        assert_eq!(found, Some(&recommendations[1]));
    }

    #[test]
    fn nearest_match_prefers_exact_lanes_and_steps() {
        let mut exact = sample_recommendation();
        exact.autotune_metric_ns = Some(10_000);
        let mut near = sample_recommendation();
        near.lanes = 65;
        near.steps = 15;
        near.autotune_metric_ns = Some(1);
        let recommendations = vec![near, exact];

        let found = find_gpu_autotune_recommendation_with_mode(
            &recommendations,
            "counter",
            64,
            16,
            GpuAutotuneMatchMode::Nearest,
        );

        assert_eq!(found, Some(&recommendations[1]));
    }

    #[test]
    fn nearest_match_uses_lane_distance_before_step_distance() {
        let mut closer_lanes = sample_recommendation();
        closer_lanes.lanes = 96;
        closer_lanes.steps = 100;
        let mut closer_steps = sample_recommendation();
        closer_steps.lanes = 128;
        closer_steps.steps = 16;
        let recommendations = vec![closer_steps, closer_lanes];

        let found = find_gpu_autotune_recommendation_with_mode(
            &recommendations,
            "counter",
            100,
            20,
            GpuAutotuneMatchMode::Nearest,
        );

        assert_eq!(found, Some(&recommendations[1]));
    }

    #[test]
    fn nearest_match_tiebreaks_by_metric_lanes_and_steps() {
        let mut slower = sample_recommendation();
        slower.lanes = 96;
        slower.steps = 18;
        slower.autotune_metric_ns = Some(50);
        let mut faster = sample_recommendation();
        faster.lanes = 104;
        faster.steps = 22;
        faster.autotune_metric_ns = Some(10);
        let recommendations = vec![slower, faster];

        let found = find_gpu_autotune_recommendation_with_mode(
            &recommendations,
            "counter",
            100,
            20,
            GpuAutotuneMatchMode::Nearest,
        );

        assert_eq!(found, Some(&recommendations[1]));

        let mut high_lanes = sample_recommendation();
        high_lanes.lanes = 104;
        high_lanes.steps = 18;
        high_lanes.autotune_metric_ns = Some(10);
        let mut low_lanes_high_steps = sample_recommendation();
        low_lanes_high_steps.lanes = 96;
        low_lanes_high_steps.steps = 22;
        low_lanes_high_steps.autotune_metric_ns = Some(10);
        let recommendations = vec![high_lanes, low_lanes_high_steps];

        let found = find_gpu_autotune_recommendation_with_mode(
            &recommendations,
            "counter",
            100,
            20,
            GpuAutotuneMatchMode::Nearest,
        );

        assert_eq!(found, Some(&recommendations[1]));

        let mut low_steps = sample_recommendation();
        low_steps.lanes = 96;
        low_steps.steps = 18;
        low_steps.autotune_metric_ns = Some(10);
        let mut high_steps = sample_recommendation();
        high_steps.lanes = 96;
        high_steps.steps = 22;
        high_steps.autotune_metric_ns = Some(10);
        let recommendations = vec![high_steps, low_steps];

        let found = find_gpu_autotune_recommendation_with_mode(
            &recommendations,
            "counter",
            100,
            20,
            GpuAutotuneMatchMode::Nearest,
        );

        assert_eq!(found, Some(&recommendations[1]));
    }

    #[test]
    fn selects_gpu_batch_options_from_matching_autotune_recommendation() {
        let mut other = sample_recommendation();
        other.case = "wide_datapath".to_string();
        other.schedule_cap = Some(4);
        let recommendations = vec![other, sample_recommendation()];

        let options =
            gpu_batch_options_from_autotune_recommendations(&recommendations, "counter", 64, 16)
                .unwrap();

        assert_eq!(options.schedule.max_packet_width, Some(8));
        assert_eq!(options.schedule.max_memory_reads_per_packet, Some(2));
        assert_eq!(options.memory_layout, GpuMemoryLayout::WordMajor);
        assert_eq!(options.workgroup_size, 128);
        assert!(options.reuse_temporaries);
    }

    #[test]
    fn missing_autotune_recommendation_reports_exact_key_error() {
        let recommendations = vec![sample_recommendation()];

        let err =
            gpu_batch_options_from_autotune_recommendations(&recommendations, "counter", 64, 1)
                .unwrap_err();

        assert_eq!(err.diagnostics[0].code, "E_GPU_AUTOTUNE_RECOMMENDATION");
    }

    #[test]
    fn nearest_options_use_nearest_matching_recommendation() {
        let mut nearest = sample_recommendation();
        nearest.lanes = 128;
        nearest.steps = 32;
        nearest.schedule_cap = Some(4);
        let recommendations = vec![nearest];

        let options = gpu_batch_options_from_autotune_recommendations_with_mode(
            &recommendations,
            "counter",
            96,
            16,
            GpuAutotuneMatchMode::Nearest,
        )
        .unwrap();

        assert_eq!(options.schedule.max_packet_width, Some(4));
    }

    #[test]
    fn report_loader_wraps_autotune_json_parse_errors() {
        let err = load_gpu_autotune_recommendations_report(br#"{"case":"counter"}"#.as_slice())
            .unwrap_err();

        assert_eq!(err.diagnostics[0].code, "E_GPU_AUTOTUNE_JSON");
    }

    #[test]
    fn converts_gpu_autotune_recommendation_to_batch_options() {
        let options = sample_recommendation().to_gpu_batch_options().unwrap();

        assert_eq!(options.schedule.max_packet_width, Some(8));
        assert_eq!(options.schedule.max_memory_reads_per_packet, Some(2));
        assert!(options.schedule.liveness_priority);
        assert_eq!(options.memory_layout, GpuMemoryLayout::WordMajor);
        assert_eq!(options.workgroup_size, 128);
        assert!(options.reuse_temporaries);
    }

    #[test]
    fn rejects_gpu_autotune_recommendation_with_unknown_layout() {
        let mut recommendation = sample_recommendation();
        recommendation.memory_layout = "blocked".to_string();

        let err = recommendation.to_gpu_batch_options().unwrap_err();

        assert_eq!(err.diagnostics[0].code, "E_GPU_AUTOTUNE_LAYOUT");
    }

    #[test]
    fn rejects_gpu_autotune_recommendation_with_zero_workgroup_size() {
        let mut recommendation = sample_recommendation();
        recommendation.workgroup_size = 0;

        let err = recommendation.to_gpu_batch_options().unwrap_err();

        assert_eq!(err.diagnostics[0].code, "E_GPU_WORKGROUP_SIZE");
    }

    #[test]
    fn autotune_constructor_uses_recommendation_before_normal_setup() {
        let mut design = Design::new();
        {
            let mut m = design.module("PadUser");
            m.inout("pad", uint(1));
        }
        let recommendations = vec![sample_recommendation()];

        let err = expect_gpu_error(GpuBatchSimulator::new_with_autotune_recommendations(
            &design,
            "PadUser",
            "counter",
            64,
            16,
            &recommendations,
        ));

        assert!(err.diagnostics.iter().any(|d| d.code == "E_SIM_IR_INOUT"));
    }

    #[test]
    fn autotune_constructor_missing_recommendation_fails_before_gpu_setup() {
        let (design, ..) = alu_design();
        let recommendations = vec![sample_recommendation()];

        let err = expect_gpu_error(GpuBatchSimulator::new_with_autotune_recommendations(
            &design,
            "Alu",
            "counter",
            64,
            1,
            &recommendations,
        ));

        assert_eq!(err.diagnostics[0].code, "E_GPU_AUTOTUNE_RECOMMENDATION");
    }

    #[test]
    fn autotune_constructor_nearest_requires_same_case_before_gpu_setup() {
        let (design, ..) = alu_design();
        let mut recommendation = sample_recommendation();
        recommendation.case = "wide_datapath".to_string();
        let recommendations = vec![recommendation];

        let err = expect_gpu_error(
            GpuBatchSimulator::new_with_autotune_recommendations_with_mode(
                &design,
                "Alu",
                "counter",
                64,
                16,
                &recommendations,
                GpuAutotuneMatchMode::Nearest,
            ),
        );

        assert_eq!(err.diagnostics[0].code, "E_GPU_AUTOTUNE_RECOMMENDATION");
    }

    #[test]
    fn autotune_constructor_nearest_same_case_reaches_normal_setup() {
        let mut design = Design::new();
        {
            let mut m = design.module("PadUser");
            m.inout("pad", uint(1));
        }
        let mut recommendation = sample_recommendation();
        recommendation.lanes = 128;
        recommendation.steps = 32;
        let recommendations = vec![recommendation];

        let err = expect_gpu_error(
            GpuBatchSimulator::new_with_autotune_recommendations_with_mode(
                &design,
                "PadUser",
                "counter",
                64,
                16,
                &recommendations,
                GpuAutotuneMatchMode::Nearest,
            ),
        );

        assert!(err.diagnostics.iter().any(|d| d.code == "E_SIM_IR_INOUT"));
    }

    #[test]
    fn autotune_reader_constructor_nearest_reports_json_errors() {
        let (design, ..) = alu_design();

        let err = expect_gpu_error(
            GpuBatchSimulator::new_with_autotune_recommendation_reader_with_mode(
                &design,
                "Alu",
                "counter",
                64,
                16,
                br#"{"case":"counter"}"#.as_slice(),
                GpuAutotuneMatchMode::Nearest,
            ),
        );

        assert_eq!(err.diagnostics[0].code, "E_GPU_AUTOTUNE_JSON");
    }

    #[test]
    fn autotune_reader_constructor_reports_json_errors() {
        let (design, ..) = alu_design();

        let err = expect_gpu_error(GpuBatchSimulator::new_with_autotune_recommendation_reader(
            &design,
            "Alu",
            "counter",
            64,
            16,
            br#"{"case":"counter"}"#.as_slice(),
        ));

        assert_eq!(err.diagnostics[0].code, "E_GPU_AUTOTUNE_JSON");
    }

    #[test]
    fn shader_stats_report_schedule_and_wgsl_size() {
        let (design, ..) = alu_design();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Alu").unwrap();
        let stats = gpu_shader_stats(&program, GpuBatchOptions::default()).unwrap();
        assert_eq!(stats.schedule.max_packet_width, Some(16));
        assert!(!stats.schedule.liveness_priority);
        assert!(stats.wgsl_bytes > 0);
        assert_eq!(stats.optimized.instr_count, stats.unoptimized.instr_count);
        assert_eq!(stats.optimized.effect_count, stats.unoptimized.effect_count);
        assert!(stats.optimized.avg_live_values_x100 > 0);
        assert_eq!(
            stats.unoptimized_packets.total,
            stats.unoptimized_packets.async_reset_comb
                + stats.unoptimized_packets.comb
                + stats.unoptimized_packets.tick_next
                + stats.unoptimized_packets.tick_commit
        );
        assert_eq!(
            stats.optimized_packets.comb,
            stats.optimized.comb.packets.len()
        );
        assert_eq!(
            stats.optimized_packets.tick_next,
            stats.optimized.tick_next.packets.len()
        );
        assert_eq!(stats.memory_layout, GpuMemoryLayout::LaneMajor);
        assert_eq!(stats.workgroup_size, WORKGROUP_SIZE);
        assert!(!stats.reuse_temporaries);
        assert_eq!(stats.optimized_temp_slots, stats.optimized_value_vars);
        assert!(stats.optimized_value_vars > 0);
        assert_eq!(stats.total_memory_words_per_lane, 0);
        assert_eq!(stats.optimized_memory.total_reads, 0);
        assert_eq!(stats.optimized_memory.total_writes, 0);
    }

    #[test]
    fn temp_slot_plan_reuses_slots_across_packet_boundaries() {
        let block = PackedBlock {
            packets: vec![
                rrtl_sim_ir::PackedMachinePacket {
                    instrs: vec![PackedInstr {
                        dst: PackedValueId(0),
                        ty: uint(8),
                        kind: PackedInstrKind::Lit(vec![1]),
                    }],
                    effects: Vec::new(),
                },
                rrtl_sim_ir::PackedMachinePacket {
                    instrs: vec![PackedInstr {
                        dst: PackedValueId(1),
                        ty: uint(8),
                        kind: PackedInstrKind::Not(PackedValueId(0)),
                    }],
                    effects: Vec::new(),
                },
                rrtl_sim_ir::PackedMachinePacket {
                    instrs: vec![PackedInstr {
                        dst: PackedValueId(2),
                        ty: uint(8),
                        kind: PackedInstrKind::Not(PackedValueId(1)),
                    }],
                    effects: vec![PackedEffect::StoreSignal {
                        dst: 0,
                        value: PackedValueId(2),
                    }],
                },
            ],
        };

        let plan = build_temp_slot_plan(&block);

        assert_eq!(plan.value_vars, 3);
        assert_eq!(plan.slot_count, 2);
        assert_eq!(
            plan.value_slots[&(PackedValueId(2), 0)],
            plan.value_slots[&(PackedValueId(0), 0)]
        );
    }

    #[test]
    fn temp_slot_plan_does_not_clobber_current_packet_uses() {
        let block = PackedBlock {
            packets: vec![
                rrtl_sim_ir::PackedMachinePacket {
                    instrs: vec![
                        PackedInstr {
                            dst: PackedValueId(0),
                            ty: uint(8),
                            kind: PackedInstrKind::Lit(vec![1]),
                        },
                        PackedInstr {
                            dst: PackedValueId(1),
                            ty: uint(8),
                            kind: PackedInstrKind::Lit(vec![2]),
                        },
                    ],
                    effects: Vec::new(),
                },
                rrtl_sim_ir::PackedMachinePacket {
                    instrs: vec![PackedInstr {
                        dst: PackedValueId(2),
                        ty: uint(8),
                        kind: PackedInstrKind::Add(PackedValueId(0), PackedValueId(1)),
                    }],
                    effects: vec![PackedEffect::StoreSignal {
                        dst: 0,
                        value: PackedValueId(2),
                    }],
                },
            ],
        };

        let plan = build_temp_slot_plan(&block);
        let packet_slots = [
            plan.value_slots[&(PackedValueId(0), 0)],
            plan.value_slots[&(PackedValueId(1), 0)],
            plan.value_slots[&(PackedValueId(2), 0)],
        ];
        let unique = packet_slots
            .into_iter()
            .collect::<std::collections::HashSet<_>>();

        assert_eq!(plan.value_vars, 3);
        assert_eq!(plan.slot_count, 3);
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn shader_stats_report_liveness_priority_schedule() {
        let (design, ..) = alu_design();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Alu").unwrap();
        let stats = gpu_shader_stats(
            &program,
            GpuBatchOptions {
                schedule: PackedScheduleOptions {
                    max_packet_width: Some(8),
                    max_memory_reads_per_packet: None,
                    liveness_priority: true,
                },
                ..GpuBatchOptions::default()
            },
        )
        .unwrap();

        assert!(stats.schedule.liveness_priority);
        assert!(stats.optimized.avg_live_values_x100 > 0);
    }

    #[test]
    fn shader_stats_report_reusable_temporaries() {
        let mut design = Design::new();
        {
            let mut m = design.module("TempStats");
            let a = m.input("a", uint(8));
            let b = m.input("b", uint(8));
            let c = m.input("c", uint(8));
            let y = m.output("y", uint(8));
            let x0 = a + b;
            let x1 = b + c;
            let x2 = a ^ c;
            m.assign(y, (x0 ^ x1) + x2);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "TempStats").unwrap();
        let stats = gpu_shader_stats(
            &program,
            GpuBatchOptions {
                reuse_temporaries: true,
                ..GpuBatchOptions::default()
            },
        )
        .unwrap();

        assert!(stats.reuse_temporaries);
        assert!(stats.optimized_temp_slots < stats.optimized_value_vars);
    }

    #[test]
    fn reusable_temporaries_emit_slot_vars() {
        let (design, ..) = alu_design();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Alu").unwrap();
        let default_wgsl = generate_wgsl(&program, GpuBatchOptions::default()).unwrap();
        let reusable_wgsl = generate_wgsl(
            &program,
            GpuBatchOptions {
                reuse_temporaries: true,
                ..GpuBatchOptions::default()
            },
        )
        .unwrap();

        assert!(default_wgsl.contains("let ve_c0"));
        assert!(reusable_wgsl.contains("var se_c0_"));
        assert!(reusable_wgsl.contains("let pe_c0"));
        assert!(!reusable_wgsl.contains("let ve_c0"));
    }

    #[test]
    fn generated_wgsl_uses_configured_workgroup_size() {
        let (design, ..) = alu_design();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Alu").unwrap();
        let wgsl = generate_wgsl(
            &program,
            GpuBatchOptions {
                workgroup_size: 128,
                ..GpuBatchOptions::default()
            },
        )
        .unwrap();
        assert!(wgsl.contains("@workgroup_size(128)"));
        assert!(!wgsl.contains("@workgroup_size(64)"));

        let stats = gpu_shader_stats(
            &program,
            GpuBatchOptions {
                workgroup_size: 128,
                ..GpuBatchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(stats.workgroup_size, 128);
    }

    #[test]
    fn shader_stats_respect_explicit_width_cap() {
        let mut design = Design::new();
        {
            let mut m = design.module("WideStats");
            let a0 = m.input("a0", uint(4));
            let a1 = m.input("a1", uint(4));
            let a2 = m.input("a2", uint(4));
            let a3 = m.input("a3", uint(4));
            let a4 = m.input("a4", uint(4));
            let y = m.output("y", uint(20));
            m.assign(
                y,
                concat([a0.value(), a1.value(), a2.value(), a3.value(), a4.value()]),
            );
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "WideStats").unwrap();
        let stats = gpu_shader_stats(
            &program,
            GpuBatchOptions {
                schedule: PackedScheduleOptions {
                    max_packet_width: Some(2),
                    max_memory_reads_per_packet: None,
                    liveness_priority: false,
                },
                memory_layout: GpuMemoryLayout::LaneMajor,
                workgroup_size: WORKGROUP_SIZE,
                reuse_temporaries: false,
            },
        )
        .unwrap();
        assert_eq!(stats.schedule.max_packet_width, Some(2));
        assert!(stats.unoptimized.max_packet_width > 2);
        assert!(stats.optimized.max_packet_width <= 2);
        assert!(stats.optimized_packets.total >= stats.unoptimized_packets.total);
    }

    #[test]
    fn shader_stats_report_memory_accesses() {
        let mut design = Design::new();
        {
            let mut m = design.module("MemStats");
            let clk = m.input("clk", uint(1));
            let we = m.input("we", uint(1));
            let addr = m.input("addr", uint(2));
            let data = m.input("data", uint(8));
            let mem = m.mem("mem", 2, uint(8), 4);
            let read = m.output("read", uint(8));
            let read_expr = m.mem_read(mem, addr);
            m.assign(read, read_expr);
            m.mem_write(mem, clk, we, addr, data);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "MemStats").unwrap();
        let stats = gpu_shader_stats(
            &program,
            GpuBatchOptions {
                memory_layout: GpuMemoryLayout::WordMajor,
                ..GpuBatchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(stats.memory_layout, GpuMemoryLayout::WordMajor);
        assert_eq!(stats.total_memory_words_per_lane, 4);
        assert_eq!(stats.optimized_memory.total_reads, 1);
        assert_eq!(stats.optimized_memory.total_writes, 1);
        assert_eq!(stats.optimized_memory.comb.reads, 1);
        assert_eq!(stats.optimized_memory.tick_commit.writes, 1);
    }

    #[test]
    fn shader_stats_report_memory_read_cap() {
        let mut design = Design::new();
        {
            let mut m = design.module("MemCapStats");
            let addr = m.input("addr", uint(2));
            let mem0 = m.mem("mem0", 2, uint(8), 4);
            let mem1 = m.mem("mem1", 2, uint(8), 4);
            let read0 = m.mem_read(mem0, addr);
            let read1 = m.mem_read(mem1, addr);
            let y = m.output("y", uint(8));
            m.assign(y, read0 ^ read1);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "MemCapStats").unwrap();
        let uncapped = gpu_shader_stats(&program, GpuBatchOptions::default()).unwrap();
        let capped = gpu_shader_stats(
            &program,
            GpuBatchOptions {
                schedule: PackedScheduleOptions {
                    max_packet_width: Some(16),
                    max_memory_reads_per_packet: Some(1),
                    liveness_priority: false,
                },
                ..GpuBatchOptions::default()
            },
        )
        .unwrap();

        assert!(uncapped.optimized.max_packet_memory_reads > 1);
        assert_eq!(capped.schedule.max_memory_reads_per_packet, Some(1));
        assert_eq!(capped.optimized.max_packet_memory_reads, 1);
    }

    #[test]
    fn wgsl_omits_exact_depth_memory_range_checks() {
        let mut design = Design::new();
        {
            let mut m = design.module("ExactDepthMem");
            let clk = m.input("clk", uint(1));
            let we = m.input("we", uint(1));
            let addr = m.input("addr", uint(2));
            let data = m.input("data", uint(8));
            let mem = m.mem("mem", 2, uint(8), 4);
            let read = m.output("read", uint(8));
            let read_expr = m.mem_read(mem, addr);
            m.assign(read, read_expr);
            m.mem_write(mem, clk, we, addr, data);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "ExactDepthMem").unwrap();
        let wgsl = generate_wgsl(&program, GpuBatchOptions::default()).unwrap();
        assert!(!wgsl.contains("fn memory_idx"));
        assert!(!wgsl.contains("< 4u"));
        assert!(!wgsl.contains("select(0u, memories"));
    }

    #[test]
    fn rejects_extern_and_inout_via_sim_ir_diagnostics() {
        let mut design = Design::new();
        {
            let mut m = design.module("PadUser");
            m.inout("pad", uint(1));
        }
        let err = expect_gpu_error(GpuBatchSimulator::new(&design, "PadUser", 4));
        assert!(err.diagnostics.iter().any(|d| d.code == "E_SIM_IR_INOUT"));
    }

    #[test]
    fn gpu_matches_cpu_for_combinational_lanes_when_adapter_exists() {
        let (design, a, b, sum, eq) = alu_design();
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Alu").unwrap();
        let mut packed = PackedSimulator::new(program, 4).unwrap();
        let Ok(mut gpu) = GpuBatchSimulator::new(&design, "Alu", 4) else {
            return;
        };
        packed.set_signal(a, &[1, 2, 3, 7]).unwrap();
        packed.set_signal(b, &[4, 2, 8, 7]).unwrap();
        gpu.set_input(a, &[1, 2, 3, 7]).unwrap();
        gpu.set_input(b, &[4, 2, 8, 7]).unwrap();
        gpu.eval_combinational().unwrap();

        let mut expected_sum = Vec::new();
        let mut expected_eq = Vec::new();
        for (a_value, b_value) in [(1, 4), (2, 2), (3, 8), (7, 7)] {
            let mut cpu = Simulator::new(&design, "Alu").unwrap();
            cpu.set(a, a_value);
            cpu.set(b, b_value);
            expected_sum.push(cpu.get(sum) as u32);
            expected_eq.push(cpu.get(eq) as u32);
        }

        assert_eq!(gpu.get_signal(sum).unwrap(), expected_sum);
        assert_eq!(gpu.get_signal(eq).unwrap(), expected_eq);
        assert_eq!(
            packed.get_signal(sum).unwrap(),
            expected_sum
                .iter()
                .map(|value| *value as u128)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            packed.get_signal(eq).unwrap(),
            expected_eq
                .iter()
                .map(|value| *value as u128)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            gpu.get_signal(sum).unwrap(),
            packed
                .get_signal(sum)
                .unwrap()
                .into_iter()
                .map(|value| value as u32)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn gpu_reusable_temporaries_match_packed_for_combinational_fanout_when_adapter_exists() {
        let mut design = Design::new();
        let (a, b, c, y);
        {
            let mut m = design.module("ReuseComb");
            a = m.input("a", uint(8));
            b = m.input("b", uint(8));
            c = m.input("c", uint(8));
            y = m.output("y", uint(8));
            let x0 = (a + b) ^ c;
            let x1 = (b + c) ^ a;
            m.assign(y, x0 + x1);
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "ReuseComb").unwrap();
        let mut packed = PackedSimulator::new(program, 4).unwrap();
        let Ok(mut gpu) = GpuBatchSimulator::new_with_options(
            &design,
            "ReuseComb",
            4,
            GpuBatchOptions {
                reuse_temporaries: true,
                ..GpuBatchOptions::default()
            },
        ) else {
            return;
        };
        for (signal, values) in [
            (a, vec![1, 2, 3, 4]),
            (b, vec![5, 6, 7, 8]),
            (c, vec![9, 10, 11, 12]),
        ] {
            packed
                .set_signal(
                    signal,
                    &values
                        .iter()
                        .map(|value| *value as u128)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            gpu.set_input(signal, &values).unwrap();
        }
        gpu.eval_combinational().unwrap();

        assert_eq!(
            gpu.get_signal(y).unwrap(),
            packed
                .get_signal(y)
                .unwrap()
                .into_iter()
                .map(|value| value as u32)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn prepared_gpu_trace_replay_reuses_buffers_when_adapter_exists() {
        let (design, a, b, sum, eq) = alu_design();
        let Ok(mut gpu) = GpuBatchSimulator::new(&design, "Alu", 2) else {
            return;
        };
        let plan = GpuTraceReplayPlan {
            steps: 2,
            inputs: vec![
                GpuTraceInputOp {
                    step: 0,
                    signal: a,
                    limb: 0,
                    values: vec![1, 5],
                },
                GpuTraceInputOp {
                    step: 0,
                    signal: b,
                    limb: 0,
                    values: vec![2, 6],
                },
                GpuTraceInputOp {
                    step: 1,
                    signal: a,
                    limb: 0,
                    values: vec![3, 7],
                },
                GpuTraceInputOp {
                    step: 1,
                    signal: b,
                    limb: 0,
                    values: vec![4, 8],
                },
            ],
            checks: vec![
                GpuTraceCheckOp {
                    step: 0,
                    check_index: 0,
                    signal: sum,
                    limb: 0,
                    expected: vec![3, 11],
                },
                GpuTraceCheckOp {
                    step: 0,
                    check_index: 1,
                    signal: eq,
                    limb: 0,
                    expected: vec![0, 0],
                },
                GpuTraceCheckOp {
                    step: 1,
                    check_index: 2,
                    signal: sum,
                    limb: 0,
                    expected: vec![7, 15],
                },
                GpuTraceCheckOp {
                    step: 1,
                    check_index: 3,
                    signal: eq,
                    limb: 0,
                    expected: vec![0, 0],
                },
            ],
        };
        let prepared = gpu
            .prepare_lane_trace_replay(&plan, GpuTraceReplayOptions { max_mismatches: 4 })
            .unwrap();
        assert!(prepared.fixed_template());
        assert!(prepared.trace_data_words() > 0);
        assert_eq!(prepared.result_words(), 25);

        let first = gpu.replay_prepared_lane_trace(&prepared).unwrap();
        gpu.set_input(a, &[0, 0]).unwrap();
        gpu.set_input(b, &[0, 0]).unwrap();
        let second = gpu.replay_prepared_lane_trace(&prepared).unwrap();

        assert_eq!(first.mismatch_count, 0);
        assert_eq!(second.mismatch_count, 0);
        assert!(second.timing.dispatch_ns > 0);
        assert!(second.timing.count_readback_ns > 0);
        assert_eq!(second.timing.full_readback_words, 0);
        assert_eq!(second.timing.full_readback_ns, 0);
    }

    #[test]
    fn single_submit_gpu_trace_replay_restores_snapshot_when_adapter_exists() {
        let (design, a, b, sum, eq) = alu_design();
        let Ok(mut gpu) = GpuBatchSimulator::new(&design, "Alu", 2) else {
            return;
        };
        let plan = GpuTraceReplayPlan {
            steps: 1,
            inputs: vec![
                GpuTraceInputOp {
                    step: 0,
                    signal: a,
                    limb: 0,
                    values: vec![1, 5],
                },
                GpuTraceInputOp {
                    step: 0,
                    signal: b,
                    limb: 0,
                    values: vec![2, 6],
                },
            ],
            checks: vec![
                GpuTraceCheckOp {
                    step: 0,
                    check_index: 0,
                    signal: sum,
                    limb: 0,
                    expected: vec![3, 11],
                },
                GpuTraceCheckOp {
                    step: 0,
                    check_index: 1,
                    signal: eq,
                    limb: 0,
                    expected: vec![0, 0],
                },
            ],
        };
        let prepared = gpu
            .prepare_lane_trace_replay(&plan, GpuTraceReplayOptions { max_mismatches: 4 })
            .unwrap();
        let snapshot = gpu.prepare_storage_snapshot();
        gpu.set_input(a, &[99, 99]).unwrap();
        gpu.set_input(b, &[99, 99]).unwrap();

        let report = gpu
            .replay_prepared_lane_trace_from_snapshot(&prepared, &snapshot)
            .unwrap();

        assert_eq!(report.mismatch_count, 0);
        assert!(report.timing.single_submit_used);
        assert!(report.timing.single_submit_ns > 0);
        assert!(report.timing.single_submit_count_readback_ns > 0);
        assert_eq!(report.timing.full_readback_words, 0);
    }

    #[test]
    fn prepared_gpu_trace_replay_reads_full_payload_on_mismatch_when_adapter_exists() {
        let (design, a, b, sum, _eq) = alu_design();
        let Ok(mut gpu) = GpuBatchSimulator::new(&design, "Alu", 2) else {
            return;
        };
        let plan = GpuTraceReplayPlan {
            steps: 1,
            inputs: vec![
                GpuTraceInputOp {
                    step: 0,
                    signal: a,
                    limb: 0,
                    values: vec![1, 5],
                },
                GpuTraceInputOp {
                    step: 0,
                    signal: b,
                    limb: 0,
                    values: vec![2, 6],
                },
            ],
            checks: vec![GpuTraceCheckOp {
                step: 0,
                check_index: 3,
                signal: sum,
                limb: 0,
                expected: vec![99, 11],
            }],
        };
        let prepared = gpu
            .prepare_lane_trace_replay(&plan, GpuTraceReplayOptions { max_mismatches: 4 })
            .unwrap();

        let report = gpu.replay_prepared_lane_trace(&prepared).unwrap();

        assert_eq!(report.mismatch_count, 1);
        assert_eq!(report.mismatches.len(), 1);
        assert_eq!(report.mismatches[0].check_index, 3);
        assert_eq!(report.mismatches[0].lane, 0);
        assert_eq!(report.mismatches[0].expected, 99);
        assert_eq!(report.mismatches[0].actual, 3);
        assert!(report.timing.count_readback_ns > 0);
        assert!(report.timing.full_readback_ns > 0);
        assert_eq!(report.timing.full_readback_words, prepared.result_words());
        assert!(report.timing.readback_ns >= report.timing.count_readback_ns);
    }

    #[test]
    fn single_submit_gpu_trace_replay_reads_full_payload_on_mismatch_when_adapter_exists() {
        let (design, a, b, sum, _eq) = alu_design();
        let Ok(mut gpu) = GpuBatchSimulator::new(&design, "Alu", 2) else {
            return;
        };
        let plan = GpuTraceReplayPlan {
            steps: 1,
            inputs: vec![
                GpuTraceInputOp {
                    step: 0,
                    signal: a,
                    limb: 0,
                    values: vec![1, 5],
                },
                GpuTraceInputOp {
                    step: 0,
                    signal: b,
                    limb: 0,
                    values: vec![2, 6],
                },
            ],
            checks: vec![GpuTraceCheckOp {
                step: 0,
                check_index: 3,
                signal: sum,
                limb: 0,
                expected: vec![99, 11],
            }],
        };
        let prepared = gpu
            .prepare_lane_trace_replay(&plan, GpuTraceReplayOptions { max_mismatches: 4 })
            .unwrap();
        let snapshot = gpu.prepare_storage_snapshot();

        let report = gpu
            .replay_prepared_lane_trace_from_snapshot(&prepared, &snapshot)
            .unwrap();

        assert_eq!(report.mismatch_count, 1);
        assert_eq!(report.mismatches[0].actual, 3);
        assert!(report.timing.single_submit_used);
        assert!(report.timing.full_readback_ns > 0);
        assert_eq!(report.timing.full_readback_words, prepared.result_words());
    }

    #[test]
    fn prepared_gpu_storage_snapshot_restores_counter_when_adapter_exists() {
        let mut design = Design::new();
        let (en, count);
        {
            let mut m = design.module("PreparedSnapshotCounter");
            let clk = m.input("clk", uint(1));
            en = m.input("en", uint(1));
            count = m.reg("count", uint(8));
            m.clock(count, clk);
            m.next(count, mux(en, count + lit_u(1, 8), count));
        }
        let Ok(mut gpu) = GpuBatchSimulator::new(&design, "PreparedSnapshotCounter", 3) else {
            return;
        };
        gpu.set_input(en, &[1, 0, 1]).unwrap();
        gpu.tick().unwrap();
        let snapshot = gpu.prepare_storage_snapshot();
        assert!(snapshot.value_words() > 0);
        gpu.tick_many(3).unwrap();
        assert_ne!(gpu.get_signal(count).unwrap(), [1, 0, 1]);

        gpu.restore_prepared_storage(&snapshot).unwrap();

        assert_eq!(gpu.get_signal(count).unwrap(), [1, 0, 1]);
    }

    #[test]
    fn gpu_ticks_counter_with_reset_when_adapter_exists() {
        let mut design = Design::new();
        let (rst, en, count);
        {
            let mut m = design.module("Counter");
            let clk = m.input("clk", uint(1));
            rst = m.input("rst", uint(1));
            en = m.input("en", uint(1));
            count = m.reg("count", uint(8));
            m.clock(count, clk);
            m.reset(count, rst, 0);
            m.next(count, mux(en, count + lit_u(1, 8), count));
        }

        let Ok(mut gpu) = GpuBatchSimulator::new(&design, "Counter", 4) else {
            return;
        };
        gpu.set_input(rst, &[1, 0, 0, 0]).unwrap();
        gpu.set_input(en, &[1, 0, 1, 1]).unwrap();
        gpu.tick().unwrap();
        gpu.set_input(rst, &[0, 0, 0, 0]).unwrap();
        gpu.tick().unwrap();
        assert_eq!(gpu.get_signal(count).unwrap(), [1, 0, 2, 2]);
    }

    #[test]
    fn gpu_reusable_temporaries_match_counter_ticks_when_adapter_exists() {
        let mut design = Design::new();
        let (en, count);
        {
            let mut m = design.module("ReuseCounter");
            let clk = m.input("clk", uint(1));
            en = m.input("en", uint(1));
            count = m.reg("count", uint(8));
            m.clock(count, clk);
            m.next(count, mux(en, count + lit_u(1, 8), count));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "ReuseCounter").unwrap();
        let mut packed = PackedSimulator::new(program, 4).unwrap();
        let Ok(mut gpu) = GpuBatchSimulator::new_with_options(
            &design,
            "ReuseCounter",
            4,
            GpuBatchOptions {
                reuse_temporaries: true,
                ..GpuBatchOptions::default()
            },
        ) else {
            return;
        };
        packed.set_signal(en, &[1, 0, 1, 1]).unwrap();
        gpu.set_input(en, &[1, 0, 1, 1]).unwrap();
        packed.tick();
        packed.tick();
        gpu.tick_many(2).unwrap();

        assert_eq!(
            gpu.get_signal(count).unwrap(),
            packed
                .get_signal(count)
                .unwrap()
                .into_iter()
                .map(|value| value as u32)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn gpu_tick_many_matches_repeated_ticks_when_adapter_exists() {
        let mut design = Design::new();
        let (en, count);
        {
            let mut m = design.module("CounterMany");
            let clk = m.input("clk", uint(1));
            en = m.input("en", uint(1));
            count = m.reg("count", uint(8));
            m.clock(count, clk);
            m.next(count, mux(en, count + lit_u(1, 8), count));
        }

        let Ok(mut one_by_one) = GpuBatchSimulator::new(&design, "CounterMany", 4) else {
            return;
        };
        let Ok(mut batched) = GpuBatchSimulator::new(&design, "CounterMany", 4) else {
            return;
        };
        one_by_one.set_input(en, &[1, 0, 1, 1]).unwrap();
        batched.set_input(en, &[1, 0, 1, 1]).unwrap();
        one_by_one.tick().unwrap();
        one_by_one.tick().unwrap();
        batched.tick_many(2).unwrap();
        assert_eq!(
            batched.get_signal(count).unwrap(),
            one_by_one.get_signal(count).unwrap()
        );
    }

    #[test]
    fn gpu_tick_many_zero_steps_keeps_state_when_adapter_exists() {
        let mut design = Design::new();
        let (en, count);
        {
            let mut m = design.module("CounterZero");
            let clk = m.input("clk", uint(1));
            en = m.input("en", uint(1));
            count = m.reg("count", uint(8));
            m.clock(count, clk);
            m.next(count, mux(en, count + lit_u(1, 8), count));
        }

        let Ok(mut gpu) = GpuBatchSimulator::new(&design, "CounterZero", 2) else {
            return;
        };
        gpu.set_input(en, &[1, 1]).unwrap();
        gpu.tick_many(0).unwrap();
        assert_eq!(gpu.get_signal(count).unwrap(), [0, 0]);
    }

    #[test]
    fn gpu_handles_wide_signed_concat_when_adapter_exists() {
        let mut design = Design::new();
        let (a, b, lt, wide);
        {
            let mut m = design.module("Wide");
            a = m.input("a", sint(40));
            b = m.input("b", sint(40));
            lt = m.output("lt", uint(1));
            wide = m.output("wide", uint(48));
            m.assign(lt, a.value().lt_expr(b));
            m.assign(
                wide,
                concat([a.value().as_uint().slice(0, 24), lit_u(0xabcd, 24)]),
            );
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Wide").unwrap();
        let mut packed = PackedSimulator::new(program, 2).unwrap();
        let Ok(mut gpu) = GpuBatchSimulator::new(&design, "Wide", 2) else {
            return;
        };
        let a_values = [vec![0xffff_fffe, 0xff], vec![3, 0]];
        let b_values = [vec![1, 0], vec![2, 0]];
        packed.set_signal_limbs(a, &a_values).unwrap();
        packed.set_signal_limbs(b, &b_values).unwrap();
        packed.eval_combinational();
        gpu.set_input_limbs(a, &a_values).unwrap();
        gpu.set_input_limbs(b, &b_values).unwrap();
        gpu.eval_combinational().unwrap();
        assert_eq!(gpu.get_signal(lt).unwrap(), [1, 0]);
        assert_eq!(packed.get_signal(lt).unwrap(), [1, 0]);
        assert_eq!(
            gpu.get_signal_limbs(wide).unwrap(),
            packed.get_signal_limbs(wide).unwrap()
        );
        assert_eq!(
            packed.get_signal_limbs(wide).unwrap(),
            [vec![0xfe00_abcd, 0xffff], vec![0x0300_abcd, 0x0000]]
        );
    }

    #[test]
    fn gpu_eval_combinational_many_matches_repeated_eval_when_adapter_exists() {
        let (design, a, b, sum, eq) = alu_design();
        let Ok(mut one_by_one) = GpuBatchSimulator::new(&design, "Alu", 4) else {
            return;
        };
        let Ok(mut batched) = GpuBatchSimulator::new(&design, "Alu", 4) else {
            return;
        };
        one_by_one.set_input(a, &[1, 2, 3, 7]).unwrap();
        one_by_one.set_input(b, &[4, 2, 8, 7]).unwrap();
        batched.set_input(a, &[1, 2, 3, 7]).unwrap();
        batched.set_input(b, &[4, 2, 8, 7]).unwrap();
        for _ in 0..3 {
            one_by_one.eval_combinational().unwrap();
        }
        batched.eval_combinational_many(3).unwrap();
        assert_eq!(
            batched.get_signal(sum).unwrap(),
            one_by_one.get_signal(sum).unwrap()
        );
        assert_eq!(
            batched.get_signal(eq).unwrap(),
            one_by_one.get_signal(eq).unwrap()
        );
    }

    #[test]
    fn gpu_flattens_child_module_when_adapter_exists() {
        let mut design = Design::new();
        {
            let mut m = design.module("Child");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.assign(y, a + lit_u(1, 8));
        }
        let (a, y);
        {
            let mut m = design.module("Top");
            a = m.input("a", uint(8));
            y = m.output("y", uint(8));
            m.instance("u_child", "Child", [("a", a), ("y", y)]);
        }
        let Ok(mut gpu) = GpuBatchSimulator::new(&design, "Top", 3) else {
            return;
        };
        gpu.set_input(a, &[1, 2, 255]).unwrap();
        gpu.eval_combinational().unwrap();
        assert_eq!(gpu.get_signal(y).unwrap(), [2, 3, 0]);
    }

    #[test]
    fn gpu_simulates_simple_memory_when_adapter_exists() {
        let mut design = Design::new();
        let (we, addr, data, read);
        {
            let mut m = design.module("MemTop");
            let clk = m.input("clk", uint(1));
            we = m.input("we", uint(1));
            addr = m.input("addr", uint(2));
            data = m.input("data", uint(8));
            let mem = m.mem("mem", 2, uint(8), 4);
            read = m.output("read", uint(8));
            let read_expr = m.mem_read(mem, addr);
            m.assign(read, read_expr);
            m.mem_write(mem, clk, we, addr, data);
        }
        let Ok(mut gpu) = GpuBatchSimulator::new(&design, "MemTop", 2) else {
            return;
        };
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "MemTop").unwrap();
        let mut packed = PackedSimulator::new(program, 2).unwrap();
        packed.set_signal(we, &[1, 1]).unwrap();
        packed.set_signal(addr, &[1, 2]).unwrap();
        packed.set_signal(data, &[9, 7]).unwrap();
        gpu.set_input(we, &[1, 1]).unwrap();
        gpu.set_input(addr, &[1, 2]).unwrap();
        gpu.set_input(data, &[9, 7]).unwrap();
        packed.tick();
        gpu.tick().unwrap();
        packed.set_signal(we, &[0, 0]).unwrap();
        gpu.set_input(we, &[0, 0]).unwrap();
        packed.eval_combinational();
        gpu.eval_combinational().unwrap();
        assert_eq!(gpu.get_signal(read).unwrap(), [9, 7]);
        assert_eq!(packed.get_signal(read).unwrap(), [9, 7]);
    }

    #[test]
    fn gpu_reusable_temporaries_match_memory_write_when_adapter_exists() {
        let mut design = Design::new();
        let (we, addr, data, read);
        {
            let mut m = design.module("ReuseMem");
            let clk = m.input("clk", uint(1));
            we = m.input("we", uint(1));
            addr = m.input("addr", uint(2));
            data = m.input("data", uint(8));
            let mem = m.mem("mem", 2, uint(8), 4);
            read = m.output("read", uint(8));
            let read_expr = m.mem_read(mem, addr);
            m.assign(read, read_expr);
            m.mem_write(mem, clk, we, addr, data + lit_u(1, 8));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "ReuseMem").unwrap();
        let mut packed = PackedSimulator::new(program, 3).unwrap();
        let Ok(mut gpu) = GpuBatchSimulator::new_with_options(
            &design,
            "ReuseMem",
            3,
            GpuBatchOptions {
                reuse_temporaries: true,
                ..GpuBatchOptions::default()
            },
        ) else {
            return;
        };
        packed.set_signal(we, &[1, 1, 1]).unwrap();
        gpu.set_input(we, &[1, 1, 1]).unwrap();
        packed.set_signal(addr, &[0, 1, 2]).unwrap();
        packed.set_signal(data, &[9, 11, 13]).unwrap();
        gpu.set_input(addr, &[0, 1, 2]).unwrap();
        gpu.set_input(data, &[9, 11, 13]).unwrap();
        packed.tick();
        gpu.tick().unwrap();
        packed.set_signal(we, &[0, 0, 0]).unwrap();
        gpu.set_input(we, &[0, 0, 0]).unwrap();
        packed.eval_combinational();
        gpu.eval_combinational().unwrap();

        assert_eq!(
            gpu.get_signal(read).unwrap(),
            packed
                .get_signal(read)
                .unwrap()
                .into_iter()
                .map(|value| value as u32)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn gpu_memory_set_and_read_apis_work_when_adapter_exists() {
        let mut design = Design::new();
        let mem;
        {
            let mut m = design.module("MemApi");
            mem = m.mem("mem", 2, uint(8), 4);
        }
        let Ok(mut gpu) = GpuBatchSimulator::new(&design, "MemApi", 2) else {
            return;
        };
        gpu.set_memory(mem, &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]])
            .unwrap();
        assert_eq!(
            gpu.get_memory(mem).unwrap(),
            [vec![1, 2, 3, 4], vec![5, 6, 7, 8]]
        );
    }

    #[test]
    fn prepared_gpu_storage_snapshot_restores_word_major_memory_when_adapter_exists() {
        let mut design = Design::new();
        let mem;
        {
            let mut m = design.module("PreparedSnapshotMem");
            mem = m.mem("mem", 2, uint(8), 4);
        }
        let Ok(mut gpu) = GpuBatchSimulator::new_with_options(
            &design,
            "PreparedSnapshotMem",
            2,
            GpuBatchOptions {
                memory_layout: GpuMemoryLayout::WordMajor,
                ..GpuBatchOptions::default()
            },
        ) else {
            return;
        };
        gpu.set_memory(mem, &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]])
            .unwrap();
        let snapshot = gpu.prepare_storage_snapshot();
        assert!(snapshot.memory_words() > 0);
        assert!(snapshot.storage_bytes() >= snapshot.memory_words() * std::mem::size_of::<u32>());
        gpu.set_memory(mem, &[vec![9, 9, 9, 9], vec![8, 8, 8, 8]])
            .unwrap();
        assert_ne!(
            gpu.get_memory(mem).unwrap(),
            [vec![1, 2, 3, 4], vec![5, 6, 7, 8]]
        );

        gpu.restore_prepared_storage(&snapshot).unwrap();

        assert_eq!(
            gpu.get_memory(mem).unwrap(),
            [vec![1, 2, 3, 4], vec![5, 6, 7, 8]]
        );
    }

    #[test]
    fn gpu_word_major_memory_matches_lane_major_when_adapter_exists() {
        let mut design = Design::new();
        let (we, addr, data, read);
        {
            let mut m = design.module("MemLayout");
            let clk = m.input("clk", uint(1));
            we = m.input("we", uint(1));
            addr = m.input("addr", uint(2));
            data = m.input("data", uint(8));
            let mem = m.mem("mem", 2, uint(8), 4);
            read = m.output("read", uint(8));
            let read_expr = m.mem_read(mem, addr);
            m.assign(read, read_expr);
            m.mem_write(mem, clk, we, addr, data);
        }
        let options = |memory_layout| GpuBatchOptions {
            memory_layout,
            ..GpuBatchOptions::default()
        };
        let Ok(mut lane_major) = GpuBatchSimulator::new_with_options(
            &design,
            "MemLayout",
            3,
            options(GpuMemoryLayout::LaneMajor),
        ) else {
            return;
        };
        let Ok(mut word_major) = GpuBatchSimulator::new_with_options(
            &design,
            "MemLayout",
            3,
            options(GpuMemoryLayout::WordMajor),
        ) else {
            return;
        };
        for gpu in [&mut lane_major, &mut word_major] {
            gpu.set_input(we, &[1, 1, 1]).unwrap();
            gpu.set_input(addr, &[0, 1, 2]).unwrap();
            gpu.set_input(data, &[9, 7, 5]).unwrap();
            gpu.tick_many(1).unwrap();
            gpu.set_input(we, &[0, 0, 0]).unwrap();
            gpu.eval_combinational().unwrap();
        }
        assert_eq!(
            word_major.get_signal(read).unwrap(),
            lane_major.get_signal(read).unwrap()
        );
    }

    #[test]
    fn gpu_snapshot_restores_counter_state_when_adapter_exists() {
        let mut design = Design::new();
        let (en, count);
        {
            let mut m = design.module("GpuSnapshotCounter");
            let clk = m.input("clk", uint(1));
            en = m.input("en", uint(1));
            count = m.reg("count", uint(8));
            m.clock(count, clk);
            m.next(count, mux(en, count + lit_u(1, 8), count));
        }
        let Ok(mut gpu) = GpuBatchSimulator::new(&design, "GpuSnapshotCounter", 2) else {
            return;
        };
        gpu.set_input(en, &[1, 0]).unwrap();
        gpu.tick().unwrap();
        let snapshot = gpu.snapshot_storage().unwrap();
        gpu.set_input(en, &[1, 1]).unwrap();
        gpu.tick().unwrap();
        assert_eq!(gpu.get_signal(count).unwrap(), [2, 1]);

        gpu.restore_storage(&snapshot).unwrap();
        assert_eq!(gpu.get_signal(count).unwrap(), [1, 0]);

        let mut bad = snapshot.clone();
        bad.memories.push(1);
        let err = gpu.restore_storage(&bad).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_GPU_STORAGE_MEMORIES");
    }

    #[test]
    fn gpu_snapshot_memory_uses_lane_major_storage_when_adapter_exists() {
        let mut design = Design::new();
        let (mem, addr, read);
        {
            let mut m = design.module("GpuSnapshotMem");
            addr = m.input("addr", uint(2));
            mem = m.mem("mem", 2, uint(8), 4);
            read = m.output("read", uint(8));
            let read_expr = m.mem_read(mem, addr);
            m.assign(read, read_expr);
        }
        let options = |memory_layout| GpuBatchOptions {
            memory_layout,
            ..GpuBatchOptions::default()
        };
        let Ok(mut word_major) = GpuBatchSimulator::new_with_options(
            &design,
            "GpuSnapshotMem",
            2,
            options(GpuMemoryLayout::WordMajor),
        ) else {
            return;
        };
        let Ok(mut lane_major) = GpuBatchSimulator::new_with_options(
            &design,
            "GpuSnapshotMem",
            2,
            options(GpuMemoryLayout::LaneMajor),
        ) else {
            return;
        };

        word_major
            .set_memory(mem, &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]])
            .unwrap();
        let snapshot = word_major.snapshot_storage().unwrap();
        lane_major.restore_storage(&snapshot).unwrap();
        lane_major.set_input(addr, &[2, 1]).unwrap();
        lane_major.eval_combinational().unwrap();

        assert_eq!(lane_major.get_signal(read).unwrap(), [3, 6]);
    }

    #[test]
    fn gpu_tick_many_handles_memory_write_when_adapter_exists() {
        let mut design = Design::new();
        let (we, addr, data, read);
        {
            let mut m = design.module("MemMany");
            let clk = m.input("clk", uint(1));
            we = m.input("we", uint(1));
            addr = m.input("addr", uint(2));
            data = m.input("data", uint(8));
            let mem = m.mem("mem", 2, uint(8), 4);
            read = m.output("read", uint(8));
            let read_expr = m.mem_read(mem, addr);
            m.assign(read, read_expr);
            m.mem_write(mem, clk, we, addr, data);
        }
        let Ok(mut gpu) = GpuBatchSimulator::new(&design, "MemMany", 2) else {
            return;
        };
        gpu.set_input(we, &[1, 1]).unwrap();
        gpu.set_input(addr, &[1, 2]).unwrap();
        gpu.set_input(data, &[9, 7]).unwrap();
        gpu.tick_many(1).unwrap();
        gpu.set_input(we, &[0, 0]).unwrap();
        gpu.eval_combinational().unwrap();
        assert_eq!(gpu.get_signal(read).unwrap(), [9, 7]);
    }

    fn expect_gpu_error(result: Result<GpuBatchSimulator, ErrorReport>) -> ErrorReport {
        match result {
            Ok(_) => panic!("expected GPU simulator construction to fail"),
            Err(err) => err,
        }
    }
}
