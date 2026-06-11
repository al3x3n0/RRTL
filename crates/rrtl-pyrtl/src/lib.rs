use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::io::Read;
use std::time::Instant;

use rrtl_core::Simulator;
use rrtl_core::{compile, Design};
use rrtl_gpu_sim::{
    analyze_gpu_regions, gpu_shader_stats, GpuBatchOptions, GpuBatchSimulator, GpuMemoryLayout,
    GpuRegionAnalysis, GpuRegionRecommendation, GpuTraceCheckOp, GpuTraceInputOp,
    GpuTraceReplayOptions, GpuTraceReplayPlan, GpuTraceReplayReport, GpuTraceReplayTiming,
    PreparedGpuStorageSnapshot, PreparedGpuTraceReplay,
};
use rrtl_ir::{
    uint, Assignment, BitType, Expr, InitialMemoryValue, InitialRegisterValue, MemoryWrite, Module,
    ModuleId, Signal, SignalId, SignalInfo, SignalKind,
};
use rrtl_sim_ir::{
    analyze_backend_affinity, analyze_simd_suitability,
    build_replay_autotune_candidate_set_with_affinity, final_limb_mask, limbs,
    lower_to_packed_program, replay_trace_autotune_with_initial_state,
    BackendAffinityRecommendation, BackendAffinityReport, EncodedTraceReplayPlan,
    EncodedTraceReplayWorkload, PackedProgram, PackedSimulator, ReplayAutotuneOptions,
    ReplayAutotunePrunedCandidateReport, ReplayAutotuneReport, ReplayCheckMode, ReplayLaneMode,
    ReplayOptions, ReplaySimdStats, ReplayTimingReport, SimBackend, SimBackendInstance,
    SimBackendKind, SimBackendOptions, SimdSuitabilityRecommendation, SimdSuitabilityReport,
    SingleLaneMachineSimulator, ThreadedReplayInitialState, ThreadedReplayOptions,
    ThreadedReplayReport, ThreadedReplayRunner, ThreadedReplayWorkerOptions,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyrtlExport {
    pub schema: String,
    pub top_name: String,
    pub clock_name: String,
    pub wires: Vec<PyrtlWire>,
    pub memories: Vec<PyrtlMemory>,
    pub nets: Vec<PyrtlNet>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyrtlWire {
    pub name: String,
    pub kind: PyrtlWireKind,
    pub bitwidth: u32,
    #[serde(default)]
    pub value: Option<i128>,
    #[serde(default)]
    pub reset_value: Option<i128>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PyrtlWireKind {
    Input,
    Output,
    Wire,
    Register,
    Const,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyrtlMemory {
    pub name: String,
    pub id: usize,
    pub kind: PyrtlMemoryKind,
    pub bitwidth: u32,
    pub addrwidth: u32,
    #[serde(default)]
    pub asynchronous: bool,
    #[serde(default)]
    pub initial: Vec<PyrtlMemoryInitial>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PyrtlMemoryKind {
    Mem,
    Rom,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyrtlMemoryInitial {
    pub addr: usize,
    pub value: i128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyrtlNet {
    pub index: usize,
    pub op: String,
    #[serde(default)]
    pub op_param: serde_json::Value,
    pub args: Vec<String>,
    pub dests: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyrtlTrace {
    pub schema: String,
    pub steps: Vec<PyrtlTraceStep>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyrtlTraceStep {
    pub inputs: HashMap<String, u128>,
    pub outputs: HashMap<String, u128>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyrtlLaneTrace {
    pub schema: String,
    pub lanes: usize,
    pub steps: Vec<PyrtlLaneTraceStep>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyrtlLaneTraceStep {
    pub inputs: HashMap<String, Vec<u128>>,
    pub outputs: HashMap<String, Vec<u128>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceMismatch {
    pub step: usize,
    pub signal: String,
    pub expected: u128,
    pub actual: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchTraceOptions {
    pub repeat: usize,
    pub warmup: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchTraceReport {
    pub schema: String,
    pub steps: usize,
    pub repeat: usize,
    pub warmup: usize,
    pub import_ns: u128,
    pub setup_ns: u128,
    pub replay_ns_samples: Vec<u128>,
    pub replay_ns_best: u128,
    pub replay_ns_median: u128,
    pub mismatch_count: usize,
    pub mismatches: Vec<TraceMismatch>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchPackedTraceOptions {
    pub repeat: usize,
    pub warmup: usize,
    pub lanes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchPackedTraceReport {
    pub schema: String,
    pub steps: usize,
    pub repeat: usize,
    pub warmup: usize,
    pub lanes: usize,
    pub import_ns: u128,
    pub setup_ns: u128,
    pub replay_ns_samples: Vec<u128>,
    pub replay_ns_best: u128,
    pub replay_ns_median: u128,
    pub mismatch_count: usize,
    pub mismatches: Vec<TraceMismatch>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchSingleTraceOptions {
    pub repeat: usize,
    pub warmup: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchSingleTraceReport {
    pub schema: String,
    pub steps: usize,
    pub repeat: usize,
    pub warmup: usize,
    pub import_ns: u128,
    pub setup_ns: u128,
    pub replay_ns_samples: Vec<u128>,
    pub replay_ns_best: u128,
    pub replay_ns_median: u128,
    pub mismatch_count: usize,
    pub mismatches: Vec<TraceMismatch>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PyrtlBenchBackendKind {
    Scalar,
    PackedCpu,
    SimdCpu,
    JitCpu,
}

impl PyrtlBenchBackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::PackedCpu => "packed-cpu",
            Self::SimdCpu => "simd-cpu",
            Self::JitCpu => "jit-cpu",
        }
    }

    fn sim_kind(self) -> SimBackendKind {
        match self {
            Self::Scalar => SimBackendKind::Scalar,
            Self::PackedCpu => SimBackendKind::PackedCpu,
            Self::SimdCpu => SimBackendKind::SimdCpu,
            Self::JitCpu => SimBackendKind::JitCpu,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchBackendsTraceOptions {
    pub repeat: usize,
    pub warmup: usize,
    pub lanes: usize,
    pub backends: Vec<PyrtlBenchBackendKind>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchBackendsTraceReport {
    pub schema: String,
    pub steps: usize,
    pub repeat: usize,
    pub warmup: usize,
    pub import_ns: u128,
    pub setup_ns: u128,
    pub simd_suitability: SimdSuitabilityReport,
    pub backends: Vec<BenchBackendTraceReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileReplayOptions {
    pub repeat: usize,
    pub warmup: usize,
    pub lanes: usize,
}

impl Default for ProfileReplayOptions {
    fn default() -> Self {
        Self {
            repeat: 1,
            warmup: 0,
            lanes: 1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeProfile {
    pub schema: String,
    pub recommended_runtime_backend: Option<String>,
    #[serde(default)]
    pub recommended_runtime_source: Option<String>,
    #[serde(default)]
    pub selected_backend: Option<RuntimeProfileSelectedBackend>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeProfileSelectedBackend {
    #[serde(default)]
    pub selected_threaded_layout: Option<RuntimeProfileThreadedLayout>,
    #[serde(default)]
    pub selected_gpu_options: Option<RuntimeProfileGpuOptions>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeProfileThreadedLayout {
    pub workers: Vec<RuntimeProfileWorker>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeProfileWorker {
    pub backend: String,
    pub lanes: usize,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeProfileGpuOptions {
    pub workgroup_size: Option<u32>,
    pub memory_layout: Option<GpuMemoryLayout>,
    #[serde(default)]
    pub reuse_temporaries: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileReplayReport {
    pub schema: String,
    pub selected_backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_source: Option<String>,
    pub steps: usize,
    pub lanes: usize,
    pub repeat: usize,
    pub warmup: usize,
    pub import_ns: u128,
    pub setup_ns: u128,
    pub runner_setup_ns: u128,
    #[serde(default)]
    pub setup_ns_total: u128,
    pub replay_ns_samples: Vec<u128>,
    #[serde(default)]
    pub first_replay_ns: u128,
    pub replay_ns_best: u128,
    pub replay_ns_median: u128,
    #[serde(default)]
    pub hot_replay_ns_best: u128,
    #[serde(default)]
    pub hot_replay_ns_median: u128,
    pub setup_to_replay_ratio: f64,
    #[serde(default)]
    pub setup_to_hot_ratio: f64,
    #[serde(default)]
    pub hot_replay_speedup: f64,
    pub replay_ns_per_step: f64,
    pub replay_ns_per_lane_step: f64,
    pub mismatch_count: usize,
    pub mismatches: Vec<TraceMismatch>,
    pub replay: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchThreadedTraceOptions {
    pub repeat: usize,
    pub warmup: usize,
    pub max_workers: usize,
    pub workers: Vec<ThreadedReplayWorkerOptions>,
    pub autotune: bool,
    pub autotune_prune: bool,
    pub plan_first: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner_calibration: Option<PlannerCalibration>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchThreadedTraceReport {
    pub schema: String,
    pub steps: usize,
    pub lanes: usize,
    pub repeat: usize,
    pub warmup: usize,
    pub import_ns: u128,
    pub setup_ns: u128,
    pub simd_suitability: SimdSuitabilityReport,
    pub backend_affinity: BackendAffinityReport,
    pub replay_workload: EncodedTraceReplayWorkload,
    pub selected_threaded_layout: ThreadedReplayOptions,
    pub selected_reason: String,
    pub replay_ns_samples: Vec<u128>,
    pub replay_ns_best: u128,
    pub replay_ns_median: u128,
    pub replay: ThreadedReplayReport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub autotune: Option<ReplayAutotuneReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub autotune_pruned_candidates: Vec<ReplayAutotunePrunedCandidateReport>,
    pub mismatch_count: usize,
    pub mismatches: Vec<TraceMismatch>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanBackendsOptions {
    pub max_workers: usize,
    pub autotune_prune: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner_calibration: Option<PlannerCalibration>,
}

impl Default for PlanBackendsOptions {
    fn default() -> Self {
        Self {
            max_workers: std::thread::available_parallelism().map_or(1, usize::from),
            autotune_prune: true,
            planner_calibration: None,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlannerCalibration {
    pub schema: String,
    #[serde(default)]
    pub summary: PlannerCalibrationSummary,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlannerCalibrationSummary {
    #[serde(default)]
    pub backend_preferences: Vec<PlannerCalibrationPreference>,
    #[serde(default)]
    pub hot_backend_preferences: Vec<PlannerCalibrationPreference>,
    #[serde(default)]
    pub threaded_layout_preferences: Vec<PlannerCalibrationPreference>,
    #[serde(default)]
    pub gpu_option_preferences: Vec<PlannerCalibrationPreference>,
    #[serde(default)]
    pub profitability_backend_preferences: Vec<PlannerCalibrationPreference>,
    #[serde(default)]
    pub profitability_penalties: Vec<PlannerCalibrationPreference>,
    #[serde(default)]
    pub profitability_feature_preferences: Vec<PlannerCalibrationPreference>,
    #[serde(default)]
    pub profitability_feature_penalties: Vec<PlannerCalibrationPreference>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlannerCalibrationPreference {
    pub signature: String,
    #[serde(default)]
    pub score: f64,
    #[serde(default)]
    pub count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendPlanReport {
    pub schema: String,
    pub steps: usize,
    pub lanes: usize,
    pub max_workers: usize,
    pub profitability: BackendProfitabilityReport,
    #[serde(default)]
    pub profitability_features: BackendProfitabilityFeatures,
    pub backend_candidates: Vec<BackendProfitabilityCandidate>,
    pub selected_runtime_backend: String,
    pub selected_runtime_reason: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pruned_runtime_candidates: Vec<BackendProfitabilityCandidate>,
    pub simd_suitability: SimdSuitabilityReport,
    pub backend_affinity: BackendAffinityReport,
    pub gpu_region_analysis: GpuRegionAnalysis,
    pub shader_stats: BenchGpuShaderStats,
    pub replay_workload: EncodedTraceReplayWorkload,
    pub selected_threaded_layout: ThreadedReplayOptions,
    pub selected_reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner_calibration_schema: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner_calibration_threaded_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner_calibration_gpu_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner_calibration_hot_backend_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner_calibration_hot_backend_preference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner_calibration_hot_backend_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_gpu_options: Option<GpuBatchOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_gpu_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recommended_gpu_options: Vec<PlannedGpuOption>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pruned_gpu_options: Vec<PlannedGpuOption>,
    pub recommended_threaded_layouts: Vec<ThreadedReplayOptions>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pruned_threaded_layouts: Vec<ReplayAutotunePrunedCandidateReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlannedGpuOption {
    pub options: GpuBatchOptions,
    pub shader_stats: BenchGpuShaderStats,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendProfitabilityReport {
    pub op_profile: BackendProfitabilityOpProfile,
    pub simd_coverage_score_x100: usize,
    pub gpu_suitability_score_x100: usize,
    pub threading_score_x100: usize,
    pub recommended_backend: String,
    pub recommended_reason: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BackendProfitabilityOpProfile {
    pub instr_count: usize,
    pub one_limb_ops: usize,
    pub two_limb_ops: usize,
    pub native_two_limb_ops: usize,
    pub two_limb_mul_ops: usize,
    pub memory_ops: usize,
    pub wide_fallback_ops: usize,
    pub pure_compute_packets: usize,
    pub memory_hostile_packets: usize,
    pub estimated_lane_work_units: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BackendProfitabilityFeatures {
    pub lanes: usize,
    pub steps: usize,
    pub lane_steps: usize,
    #[serde(default)]
    pub gpu_lane_steps: usize,
    pub estimated_lane_work_units: usize,
    pub instr_count: usize,
    pub simd_coverage_score_x100: usize,
    pub native_simd_score_x100: usize,
    pub fallback_ratio_x100: usize,
    pub memory_op_ratio_x100: usize,
    pub pure_compute_packets: usize,
    pub memory_hostile_packets: usize,
    pub wide_fallback_ops: usize,
    pub gpu_suitability_score_x100: usize,
    pub threading_score_x100: usize,
    #[serde(default)]
    pub gpu_single_submit_available: bool,
    #[serde(default)]
    pub gpu_count_readback_only: bool,
    #[serde(default)]
    pub gpu_full_readback_penalty: usize,
    #[serde(default)]
    pub gpu_prepared_trace_bytes: usize,
    #[serde(default)]
    pub gpu_trace_uncompressed_bytes: usize,
    #[serde(default)]
    pub gpu_trace_compression_ratio_x100: usize,
    #[serde(default)]
    pub gpu_trace_uniform_input_ops: usize,
    #[serde(default)]
    pub gpu_trace_uniform_check_ops: usize,
    #[serde(default)]
    pub gpu_trace_template_layout: bool,
    #[serde(default)]
    pub gpu_trace_template_input_ops: usize,
    #[serde(default)]
    pub gpu_trace_template_check_ops: usize,
    #[serde(default)]
    pub gpu_trace_metadata_saved_words: usize,
    #[serde(default)]
    pub gpu_trace_fixed_template: bool,
    #[serde(default)]
    pub gpu_trace_value_metadata_saved_words: usize,
    #[serde(default)]
    pub gpu_trace_value_stride_words: usize,
    pub feature_buckets: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendProfitabilityCandidate {
    pub backend: String,
    pub rank: usize,
    pub score: isize,
    pub estimated_setup_cost: usize,
    pub estimated_per_lane_step_cost: usize,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchGpuTraceOptions {
    pub repeat: usize,
    pub warmup: usize,
    pub workgroup_size: u32,
    pub memory_layout: GpuMemoryLayout,
    pub reuse_temporaries: bool,
    pub fused: bool,
    pub max_mismatches: usize,
    pub plan_first: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner_calibration: Option<PlannerCalibration>,
}

impl Default for BenchGpuTraceOptions {
    fn default() -> Self {
        Self {
            repeat: 1,
            warmup: 0,
            workgroup_size: 128,
            memory_layout: GpuMemoryLayout::LaneMajor,
            reuse_temporaries: false,
            fused: true,
            max_mismatches: 16,
            plan_first: false,
            planner_calibration: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchGpuTraceReport {
    pub schema: String,
    pub steps: usize,
    pub lanes: usize,
    pub repeat: usize,
    pub warmup: usize,
    pub import_ns: u128,
    pub setup_ns: u128,
    #[serde(default)]
    pub prepared_runner_setup_ns: u128,
    #[serde(default)]
    pub prepared_snapshot_setup_ns: u128,
    #[serde(default)]
    pub prepared_trace_bytes: usize,
    #[serde(default)]
    pub prepared_trace_uncompressed_bytes: usize,
    #[serde(default)]
    pub prepared_trace_compression_ratio_x100: usize,
    #[serde(default)]
    pub prepared_trace_uniform_input_ops: usize,
    #[serde(default)]
    pub prepared_trace_uniform_check_ops: usize,
    #[serde(default)]
    pub prepared_trace_layout: String,
    #[serde(default)]
    pub prepared_trace_template_input_ops: usize,
    #[serde(default)]
    pub prepared_trace_template_check_ops: usize,
    #[serde(default)]
    pub prepared_trace_metadata_saved_words: usize,
    #[serde(default)]
    pub prepared_trace_fixed_template: bool,
    #[serde(default)]
    pub prepared_trace_value_metadata_saved_words: usize,
    #[serde(default)]
    pub prepared_trace_value_stride_words: usize,
    #[serde(default)]
    pub hot_restore_ns_best: u128,
    #[serde(default)]
    pub hot_restore_ns_median: u128,
    #[serde(default)]
    pub hot_gpu_replay_ns_best: u128,
    #[serde(default)]
    pub hot_gpu_replay_ns_median: u128,
    #[serde(default)]
    pub gpu_single_submit_profitable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_planner_calibration_reason: Option<String>,
    pub available: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub backend_affinity: BackendAffinityReport,
    pub gpu_region_analysis: GpuRegionAnalysis,
    pub shader_stats: BenchGpuShaderStats,
    pub gpu_replay_mode: String,
    pub gpu_timing: GpuTraceReplayTiming,
    pub replay_ns_samples: Vec<u128>,
    pub replay_ns_best: u128,
    pub replay_ns_median: u128,
    pub mismatch_count: usize,
    pub mismatches: Vec<TraceMismatch>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchGpuOptionsReport {
    pub schema: String,
    pub steps: usize,
    pub lanes: usize,
    pub repeat: usize,
    pub warmup: usize,
    pub candidates: Vec<BenchGpuOptionCandidateReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_candidate_index: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchGpuCombinedReport {
    pub schema: String,
    pub static_trace: BenchGpuTraceReport,
    pub option_sweep: BenchGpuOptionsReport,
    pub measured_trace: BenchGpuTraceReport,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchGpuOptionCandidateReport {
    pub candidate_index: usize,
    pub planned: PlannedGpuOption,
    pub available: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub replay_ns_samples: Vec<u128>,
    pub replay_ns_best: u128,
    pub replay_ns_median: u128,
    pub mismatch_count: usize,
    pub report: BenchGpuTraceReport,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchGpuShaderStats {
    pub wgsl_bytes: usize,
    pub optimized_temp_slots: usize,
    pub optimized_value_vars: usize,
    pub unoptimized_packets_total: usize,
    pub optimized_packets_total: usize,
    pub unoptimized_memory_reads: usize,
    pub unoptimized_memory_writes: usize,
    pub optimized_memory_reads: usize,
    pub optimized_memory_writes: usize,
    pub workgroup_size: u32,
    pub memory_layout: GpuMemoryLayout,
    pub reuse_temporaries: bool,
    pub total_memory_words_per_lane: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchBackendTraceReport {
    pub backend: String,
    pub lanes: usize,
    pub available: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub replay_ns_samples: Vec<u128>,
    pub replay_ns_best: u128,
    pub replay_ns_median: u128,
    pub replay_timing: ReplayTimingReport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub simd_stats: Option<ReplaySimdStats>,
    pub mismatch_count: usize,
    pub mismatches: Vec<TraceMismatch>,
}

#[derive(Clone, Debug)]
pub struct ImportedDesign {
    pub design: Design,
    pub top_name: String,
    pub clock: Option<Signal>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportError {
    message: String,
}

impl ImportError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    fn with_net(self, net: &PyrtlNet) -> Self {
        Self::new(format!(
            "net {} op `{}` args {:?} dests {:?}: {}",
            net.index, net.op, net.args, net.dests, self.message
        ))
    }
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}

impl std::error::Error for ImportError {}

impl From<serde_json::Error> for ImportError {
    fn from(value: serde_json::Error) -> Self {
        Self::new(value.to_string())
    }
}

impl From<std::io::Error> for ImportError {
    fn from(value: std::io::Error) -> Self {
        Self::new(value.to_string())
    }
}

pub fn read_export(mut reader: impl Read) -> Result<PyrtlExport, ImportError> {
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    Ok(serde_json::from_str(&text)?)
}

pub fn read_trace(mut reader: impl Read) -> Result<PyrtlTrace, ImportError> {
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    Ok(serde_json::from_str(&text)?)
}

pub fn read_lane_trace(mut reader: impl Read) -> Result<PyrtlLaneTrace, ImportError> {
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    Ok(serde_json::from_str(&text)?)
}

pub fn import_export(export: &PyrtlExport) -> Result<ImportedDesign, ImportError> {
    if export.schema != "rrtl-pyrtl-block-v1" {
        return Err(ImportError::new(format!(
            "unsupported PyRTL export schema `{}`",
            export.schema
        )));
    }

    let module_id = ModuleId(0);
    let mut signals = Vec::new();
    let mut signal_by_name = HashMap::new();
    let mut consts = HashMap::new();
    let mut initial_register_values = Vec::new();

    for wire in &export.wires {
        if wire.bitwidth == 0 {
            return Err(ImportError::new(format!(
                "wire `{}` has zero bitwidth",
                wire.name
            )));
        }
        if wire.kind == PyrtlWireKind::Const {
            consts.insert(
                wire.name.clone(),
                Expr::Lit {
                    value: wire.value.unwrap_or(0),
                    ty: uint(wire.bitwidth),
                },
            );
            continue;
        }

        let kind = match wire.kind {
            PyrtlWireKind::Input => SignalKind::Input,
            PyrtlWireKind::Output => SignalKind::Output,
            PyrtlWireKind::Wire => SignalKind::Wire,
            PyrtlWireKind::Register => SignalKind::Reg {
                clock: None,
                reset: None,
                next: None,
            },
            PyrtlWireKind::Const => unreachable!(),
        };
        let signal = push_signal(
            &mut signals,
            module_id,
            &wire.name,
            uint(wire.bitwidth),
            kind,
        )?;
        signal_by_name.insert(wire.name.clone(), signal);
        if wire.kind == PyrtlWireKind::Register {
            initial_register_values.push(InitialRegisterValue {
                signal,
                value: wire.reset_value.unwrap_or(0),
            });
        }
    }

    let needs_clock = export
        .wires
        .iter()
        .any(|wire| wire.kind == PyrtlWireKind::Register)
        || export.nets.iter().any(|net| net.op == "@");
    let clock = if needs_clock {
        Some(ensure_clock(
            &mut signals,
            &mut signal_by_name,
            module_id,
            &export.clock_name,
        )?)
    } else {
        None
    };

    let mut memory_by_name = HashMap::new();
    let mut memory_by_id = HashMap::new();
    let mut initial_memory_values = Vec::new();
    for memory in &export.memories {
        if memory.bitwidth == 0 || memory.addrwidth == 0 {
            return Err(ImportError::new(format!(
                "memory `{}` has an invalid shape",
                memory.name
            )));
        }
        let depth = checked_depth(memory.addrwidth)?;
        if memory_by_name.contains_key(&memory.name) {
            return Err(ImportError::new(format!(
                "duplicate memory name `{}`",
                memory.name
            )));
        }
        if memory_by_id.contains_key(&memory.id) {
            return Err(ImportError::new(format!(
                "duplicate memory id {} for `{}`",
                memory.id, memory.name
            )));
        }
        let signal = push_signal(
            &mut signals,
            module_id,
            &memory.name,
            uint(memory.bitwidth),
            SignalKind::Mem {
                addr_width: memory.addrwidth,
                data_width: memory.bitwidth,
                depth,
            },
        )?;
        memory_by_name.insert(memory.name.clone(), signal);
        memory_by_id.insert(memory.id, signal);
        for initial in &memory.initial {
            initial_memory_values.push(InitialMemoryValue {
                mem: signal,
                addr: initial.addr,
                value: initial.value,
            });
        }
    }

    let mut assignments = Vec::new();
    let mut memory_writes = Vec::new();
    let mut driven_registers = HashSet::new();
    for net in &export.nets {
        let result: Result<(), ImportError> = (|| match net.op.as_str() {
            "@" => {
                let clock =
                    clock.ok_or_else(|| ImportError::new("memory write requires a clock"))?;
                let mem = memory_from_param(net, &memory_by_name, &memory_by_id)?;
                let addr = expr_for(&net.args, 0, &signal_by_name, &consts)?;
                let data = expr_for(&net.args, 1, &signal_by_name, &consts)?;
                let enable = expr_for(&net.args, 2, &signal_by_name, &consts)?;
                memory_writes.push(MemoryWrite {
                    mem,
                    clock,
                    enable: fit_expr(enable, uint(1), &signals)?,
                    addr: fit_expr(addr, uint(memory_addr_width(&signals, mem)?), &signals)?,
                    data: fit_expr(data, signal_ty(&signals, mem)?, &signals)?,
                });
                Ok(())
            }
            "r" => {
                let clock = clock.ok_or_else(|| ImportError::new("register requires a clock"))?;
                let dst = signal_for(&net.dests, 0, &signal_by_name)?;
                let next = expr_for(&net.args, 0, &signal_by_name, &consts)?;
                let dst_ty = signal_ty(&signals, dst)?;
                let next = fit_expr(next, dst_ty, &signals)?;
                set_register_next(&mut signals, dst, clock, next)?;
                driven_registers.insert(dst);
                Ok(())
            }
            _ => {
                let dst = signal_for(&net.dests, 0, &signal_by_name)?;
                let dst_ty = signal_ty(&signals, dst)?;
                let expr = lower_expr(
                    net,
                    dst_ty,
                    &signals,
                    &signal_by_name,
                    &consts,
                    &memory_by_name,
                    &memory_by_id,
                )?;
                assignments.push(Assignment {
                    dst,
                    expr: fit_expr(expr, dst_ty, &signals)?,
                });
                Ok(())
            }
        })();
        result.map_err(|err| err.with_net(net))?;
    }

    for signal in &mut signals {
        if let SignalKind::Reg {
            clock: reg_clock,
            next,
            ..
        } = &mut signal.kind
        {
            if !driven_registers.contains(&signal.handle) {
                return Err(ImportError::new(format!(
                    "register `{}` has no PyRTL next-value net",
                    signal.name
                )));
            }
            *reg_clock = clock;
            if next.is_none() {
                *next = Some(Expr::Signal(signal.handle));
            }
        }
    }

    let module = Module {
        id: module_id,
        name: export.top_name.clone(),
        is_external: false,
        signals,
        assignments,
        memory_writes,
        initial_register_values,
        initial_memory_values,
        assertions: Vec::new(),
        cover_points: Vec::new(),
        instances: Vec::new(),
        state_types: Vec::new(),
        state_signals: Vec::new(),
        bundle_types: Vec::new(),
        bundle_signals: Vec::new(),
        interface_types: Vec::new(),
        interface_signals: Vec::new(),
        builder_diagnostics: Vec::new(),
    };
    let design = Design::from_ir(rrtl_ir::Design {
        modules: vec![module],
    });
    compile(&design).map_err(|err| ImportError::new(format!("{err}")))?;

    Ok(ImportedDesign {
        design,
        top_name: export.top_name.clone(),
        clock,
    })
}

fn push_signal(
    signals: &mut Vec<SignalInfo>,
    module: ModuleId,
    name: &str,
    ty: BitType,
    kind: SignalKind,
) -> Result<Signal, ImportError> {
    if signals.iter().any(|signal| signal.name == name) {
        return Err(ImportError::new(format!("duplicate signal `{name}`")));
    }
    let signal = Signal {
        module,
        id: SignalId(signals.len()),
    };
    signals.push(SignalInfo {
        handle: signal,
        name: name.to_string(),
        width: ty.width,
        ty,
        kind,
    });
    Ok(signal)
}

fn ensure_clock(
    signals: &mut Vec<SignalInfo>,
    signal_by_name: &mut HashMap<String, Signal>,
    module: ModuleId,
    clock_name: &str,
) -> Result<Signal, ImportError> {
    if let Some(signal) = signal_by_name.get(clock_name).copied() {
        let info = signals
            .get(signal.id.0)
            .ok_or_else(|| ImportError::new("clock signal index is invalid"))?;
        if !matches!(info.kind, SignalKind::Input) || info.ty != uint(1) {
            return Err(ImportError::new(format!(
                "clock `{clock_name}` must be a 1-bit input"
            )));
        }
        return Ok(signal);
    }

    let signal = push_signal(signals, module, clock_name, uint(1), SignalKind::Input)?;
    signal_by_name.insert(clock_name.to_string(), signal);
    Ok(signal)
}

fn checked_depth(addrwidth: u32) -> Result<usize, ImportError> {
    if addrwidth >= usize::BITS {
        return Err(ImportError::new(format!(
            "address width {addrwidth} is too large for this platform"
        )));
    }
    Ok(1usize << addrwidth)
}

fn lower_expr(
    net: &PyrtlNet,
    dst_ty: BitType,
    signals: &[SignalInfo],
    signal_by_name: &HashMap<String, Signal>,
    consts: &HashMap<String, Expr>,
    memory_by_name: &HashMap<String, Signal>,
    memory_by_id: &HashMap<usize, Signal>,
) -> Result<Expr, ImportError> {
    let arg = |index| expr_for(&net.args, index, signal_by_name, consts);
    Ok(match net.op.as_str() {
        "w" => arg(0)?,
        "~" => Expr::Not(Box::new(fit_expr(arg(0)?, dst_ty, signals)?)),
        "&" => bin(arg(0)?, arg(1)?, dst_ty, signals, Expr::And)?,
        "|" => bin(arg(0)?, arg(1)?, dst_ty, signals, Expr::Or)?,
        "^" => bin(arg(0)?, arg(1)?, dst_ty, signals, Expr::Xor)?,
        "n" => Expr::Not(Box::new(bin(arg(0)?, arg(1)?, dst_ty, signals, Expr::And)?)),
        "+" => bin(arg(0)?, arg(1)?, dst_ty, signals, Expr::Add)?,
        "-" => bin(arg(0)?, arg(1)?, dst_ty, signals, Expr::Sub)?,
        "*" => bin(arg(0)?, arg(1)?, dst_ty, signals, Expr::Mul)?,
        "=" => compare(arg(0)?, arg(1)?, signals, Expr::Eq)?,
        "<" => compare(arg(0)?, arg(1)?, signals, Expr::Lt)?,
        ">" => compare(arg(1)?, arg(0)?, signals, Expr::Lt)?,
        "x" => Expr::Mux {
            cond: Box::new(fit_expr(arg(0)?, uint(1), signals)?),
            else_expr: Box::new(fit_expr(arg(1)?, dst_ty, signals)?),
            then_expr: Box::new(fit_expr(arg(2)?, dst_ty, signals)?),
        },
        "c" => Expr::Concat(
            net.args
                .iter()
                .map(|name| expr_by_name(name, signal_by_name, consts))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        "s" => lower_select(net, arg(0)?)?,
        "m" => {
            let mem = memory_from_param(net, memory_by_name, memory_by_id)?;
            let addr = arg(0)?;
            Expr::MemRead {
                mem,
                addr: Box::new(fit_expr(
                    addr,
                    uint(memory_addr_width(signals, mem)?),
                    signals,
                )?),
            }
        }
        other => {
            return Err(ImportError::new(format!(
                "unsupported PyRTL op `{other}` in net {}",
                net.index
            )))
        }
    })
}

fn lower_select(net: &PyrtlNet, expr: Expr) -> Result<Expr, ImportError> {
    let Some(items) = net.op_param.as_array() else {
        return Err(ImportError::new("select net is missing bit indices"));
    };
    let mut bits = Vec::new();
    for item in items {
        let bit = item
            .as_u64()
            .ok_or_else(|| ImportError::new("select bit index is not an integer"))?;
        bits.push(bit as u32);
    }
    if bits.is_empty() {
        return Err(ImportError::new("empty select is not supported"));
    }
    let contiguous = bits
        .windows(2)
        .all(|window| window[1] == window[0].saturating_add(1));
    if contiguous {
        return Ok(Expr::Slice {
            expr: Box::new(expr),
            lsb: bits[0],
            width: bits.len() as u32,
        });
    }

    Ok(Expr::Concat(
        bits.into_iter()
            .rev()
            .map(|bit| Expr::Slice {
                expr: Box::new(expr.clone()),
                lsb: bit,
                width: 1,
            })
            .collect(),
    ))
}

fn bin(
    lhs: Expr,
    rhs: Expr,
    ty: BitType,
    signals: &[SignalInfo],
    make: impl FnOnce(Box<Expr>, Box<Expr>) -> Expr,
) -> Result<Expr, ImportError> {
    Ok(make(
        Box::new(fit_expr(lhs, ty, signals)?),
        Box::new(fit_expr(rhs, ty, signals)?),
    ))
}

fn compare(
    lhs: Expr,
    rhs: Expr,
    signals: &[SignalInfo],
    make: impl FnOnce(Box<Expr>, Box<Expr>) -> Expr,
) -> Result<Expr, ImportError> {
    let width = expr_width(&lhs, signals)?.max(expr_width(&rhs, signals)?);
    let ty = uint(width);
    Ok(make(
        Box::new(fit_expr(lhs, ty, signals)?),
        Box::new(fit_expr(rhs, ty, signals)?),
    ))
}

fn fit_expr(expr: Expr, ty: BitType, signals: &[SignalInfo]) -> Result<Expr, ImportError> {
    if let Expr::Lit { value, .. } = expr {
        return Ok(Expr::Lit { value, ty });
    }
    let width = expr_width(&expr, signals)?;
    if width == ty.width {
        Ok(expr)
    } else if width < ty.width {
        Ok(Expr::Zext {
            expr: Box::new(expr),
            width: ty.width,
        })
    } else {
        Ok(Expr::Trunc {
            expr: Box::new(expr),
            width: ty.width,
        })
    }
}

fn expr_for(
    names: &[String],
    index: usize,
    signal_by_name: &HashMap<String, Signal>,
    consts: &HashMap<String, Expr>,
) -> Result<Expr, ImportError> {
    let name = names
        .get(index)
        .ok_or_else(|| ImportError::new(format!("missing net argument {index}")))?;
    expr_by_name(name, signal_by_name, consts)
}

fn expr_by_name(
    name: &str,
    signal_by_name: &HashMap<String, Signal>,
    consts: &HashMap<String, Expr>,
) -> Result<Expr, ImportError> {
    if let Some(expr) = consts.get(name) {
        return Ok(expr.clone());
    }
    signal_by_name
        .get(name)
        .copied()
        .map(Expr::Signal)
        .ok_or_else(|| ImportError::new(format!("unknown wire `{name}`")))
}

fn signal_for(
    names: &[String],
    index: usize,
    signal_by_name: &HashMap<String, Signal>,
) -> Result<Signal, ImportError> {
    let name = names
        .get(index)
        .ok_or_else(|| ImportError::new(format!("missing net destination {index}")))?;
    signal_by_name
        .get(name)
        .copied()
        .ok_or_else(|| ImportError::new(format!("unknown destination wire `{name}`")))
}

fn signal_ty(signals: &[SignalInfo], signal: Signal) -> Result<BitType, ImportError> {
    signals
        .get(signal.id.0)
        .map(|info| info.ty)
        .ok_or_else(|| ImportError::new("signal index is invalid"))
}

fn expr_width(expr: &Expr, signals: &[SignalInfo]) -> Result<u32, ImportError> {
    match expr {
        Expr::Lit { ty, .. } => Ok(ty.width),
        Expr::Signal(signal) => signal_ty(signals, *signal).map(|ty| ty.width),
        Expr::Slice { width, .. }
        | Expr::Zext { width, .. }
        | Expr::Sext { width, .. }
        | Expr::Trunc { width, .. } => Ok(*width),
        Expr::Concat(parts) => parts
            .iter()
            .try_fold(0, |total, part| Ok(total + expr_width(part, signals)?)),
        Expr::Eq(_, _) | Expr::Ne(_, _) | Expr::Lt(_, _) => Ok(1),
        Expr::Not(inner) | Expr::Cast { expr: inner, .. } => expr_width(inner, signals),
        Expr::And(lhs, _)
        | Expr::Or(lhs, _)
        | Expr::Xor(lhs, _)
        | Expr::Add(lhs, _)
        | Expr::Sub(lhs, _)
        | Expr::Mul(lhs, _) => expr_width(lhs, signals),
        Expr::Mux { then_expr, .. } => expr_width(then_expr, signals),
        Expr::MemRead { mem, .. } => signal_ty(signals, *mem).map(|ty| ty.width),
    }
}

fn memory_addr_width(signals: &[SignalInfo], mem: Signal) -> Result<u32, ImportError> {
    let Some(info) = signals.get(mem.id.0) else {
        return Err(ImportError::new("memory signal index is invalid"));
    };
    let SignalKind::Mem { addr_width, .. } = info.kind else {
        return Err(ImportError::new(format!("`{}` is not a memory", info.name)));
    };
    Ok(addr_width)
}

fn memory_from_param(
    net: &PyrtlNet,
    memory_by_name: &HashMap<String, Signal>,
    memory_by_id: &HashMap<usize, Signal>,
) -> Result<Signal, ImportError> {
    if let Some(name) = net.op_param.get("memory").and_then(|value| value.as_str()) {
        if let Some(signal) = memory_by_name.get(name).copied() {
            return Ok(signal);
        }
    }
    if let Some(id) = net
        .op_param
        .get("memory_id")
        .and_then(|value| value.as_u64())
        .and_then(|value| usize::try_from(value).ok())
    {
        if let Some(signal) = memory_by_id.get(&id).copied() {
            return Ok(signal);
        }
    }
    Err(ImportError::new(format!(
        "memory op in net {} references an unknown memory",
        net.index
    )))
}

fn set_register_next(
    signals: &mut [SignalInfo],
    signal: Signal,
    clock: Signal,
    next_expr: Expr,
) -> Result<(), ImportError> {
    let Some(info) = signals.get_mut(signal.id.0) else {
        return Err(ImportError::new("register signal index is invalid"));
    };
    let SignalKind::Reg {
        clock: reg_clock,
        next,
        ..
    } = &mut info.kind
    else {
        return Err(ImportError::new(format!(
            "`{}` is not a register",
            info.name
        )));
    };
    *reg_clock = Some(clock);
    *next = Some(next_expr);
    Ok(())
}

pub fn emit_systemverilog(export: &PyrtlExport) -> Result<String, ImportError> {
    let imported = import_export(export)?;
    rrtl_sv::emit(&imported.design).map_err(|err| ImportError::new(format!("{err}")))
}

pub fn emit_compiled_json(export: &PyrtlExport) -> Result<String, ImportError> {
    let imported = import_export(export)?;
    imported
        .design
        .compile()
        .map_err(|err| ImportError::new(format!("{err}")))?
        .to_json_pretty()
        .map_err(|err| ImportError::new(err.to_string()))
}

pub fn compare_trace(
    export: &PyrtlExport,
    trace: &PyrtlTrace,
) -> Result<Vec<TraceMismatch>, ImportError> {
    if trace.schema != "rrtl-pyrtl-trace-v1" {
        return Err(ImportError::new(format!(
            "unsupported PyRTL trace schema `{}`",
            trace.schema
        )));
    }
    let imported = import_export(export)?;
    let plan = TraceReplayPlan::new(&imported, trace)?;
    let mut sim = plan.simulator()?;
    Ok(plan.replay(&mut sim))
}

pub fn bench_trace(
    export: &PyrtlExport,
    trace: &PyrtlTrace,
    options: BenchTraceOptions,
) -> Result<BenchTraceReport, ImportError> {
    if options.repeat == 0 {
        return Err(ImportError::new("bench repeat must be greater than zero"));
    }
    if trace.schema != "rrtl-pyrtl-trace-v1" {
        return Err(ImportError::new(format!(
            "unsupported PyRTL trace schema `{}`",
            trace.schema
        )));
    }

    let import_start = Instant::now();
    let imported = import_export(export)?;
    let import_ns = import_start.elapsed().as_nanos();

    let setup_start = Instant::now();
    let plan = TraceReplayPlan::new(&imported, trace)?;
    let setup_ns = setup_start.elapsed().as_nanos();

    for _ in 0..options.warmup {
        let mut sim = plan.simulator()?;
        let mismatches = plan.replay(&mut sim);
        if !mismatches.is_empty() {
            return Ok(empty_bench_trace_report(
                trace, &options, import_ns, setup_ns, mismatches,
            ));
        }
    }

    let mut samples = Vec::with_capacity(options.repeat);
    for _ in 0..options.repeat {
        let mut sim = plan.simulator()?;
        let start = Instant::now();
        let mismatches = plan.replay(&mut sim);
        let elapsed = start.elapsed().as_nanos();
        if !mismatches.is_empty() {
            return Ok(empty_bench_trace_report(
                trace, &options, import_ns, setup_ns, mismatches,
            ));
        }
        samples.push(elapsed);
    }

    let mut sorted = samples.clone();
    sorted.sort_unstable();
    let best = sorted.first().copied().unwrap_or(0);
    let median = sorted[sorted.len() / 2];
    Ok(BenchTraceReport {
        schema: "rrtl-pyrtl-bench-trace-v1".to_string(),
        steps: trace.steps.len(),
        repeat: options.repeat,
        warmup: options.warmup,
        import_ns,
        setup_ns,
        replay_ns_samples: samples,
        replay_ns_best: best,
        replay_ns_median: median,
        mismatch_count: 0,
        mismatches: Vec::new(),
    })
}

pub fn bench_packed_trace(
    export: &PyrtlExport,
    trace: &PyrtlTrace,
    options: BenchPackedTraceOptions,
) -> Result<BenchPackedTraceReport, ImportError> {
    if options.repeat == 0 {
        return Err(ImportError::new("bench repeat must be greater than zero"));
    }
    if options.lanes == 0 {
        return Err(ImportError::new(
            "packed bench lanes must be greater than zero",
        ));
    }
    if trace.schema != "rrtl-pyrtl-trace-v1" {
        return Err(ImportError::new(format!(
            "unsupported PyRTL trace schema `{}`",
            trace.schema
        )));
    }

    let import_start = Instant::now();
    let imported = import_export(export)?;
    let import_ns = import_start.elapsed().as_nanos();

    let setup_start = Instant::now();
    let plan = TraceReplayPlan::new(&imported, trace)?;
    let compiled = compile(&imported.design).map_err(|err| ImportError::new(format!("{err}")))?;
    let program = lower_to_packed_program(&compiled, &imported.top_name)
        .map_err(|err| ImportError::new(format!("{err}")))?;
    let setup_ns = setup_start.elapsed().as_nanos();

    for _ in 0..options.warmup {
        let mut sim = packed_simulator(&imported, &program, options.lanes)?;
        let mismatches = plan.replay_packed(&mut sim, options.lanes)?;
        if !mismatches.is_empty() {
            return Ok(empty_bench_packed_trace_report(
                trace, &options, import_ns, setup_ns, mismatches,
            ));
        }
    }

    let mut samples = Vec::with_capacity(options.repeat);
    for _ in 0..options.repeat {
        let mut sim = packed_simulator(&imported, &program, options.lanes)?;
        let start = Instant::now();
        let mismatches = plan.replay_packed(&mut sim, options.lanes)?;
        let elapsed = start.elapsed().as_nanos();
        if !mismatches.is_empty() {
            return Ok(empty_bench_packed_trace_report(
                trace, &options, import_ns, setup_ns, mismatches,
            ));
        }
        samples.push(elapsed);
    }

    let mut sorted = samples.clone();
    sorted.sort_unstable();
    let best = sorted.first().copied().unwrap_or(0);
    let median = sorted[sorted.len() / 2];
    Ok(BenchPackedTraceReport {
        schema: "rrtl-pyrtl-bench-packed-trace-v1".to_string(),
        steps: trace.steps.len(),
        repeat: options.repeat,
        warmup: options.warmup,
        lanes: options.lanes,
        import_ns,
        setup_ns,
        replay_ns_samples: samples,
        replay_ns_best: best,
        replay_ns_median: median,
        mismatch_count: 0,
        mismatches: Vec::new(),
    })
}

pub fn bench_single_trace(
    export: &PyrtlExport,
    trace: &PyrtlTrace,
    options: BenchSingleTraceOptions,
) -> Result<BenchSingleTraceReport, ImportError> {
    if options.repeat == 0 {
        return Err(ImportError::new("bench repeat must be greater than zero"));
    }
    if trace.schema != "rrtl-pyrtl-trace-v1" {
        return Err(ImportError::new(format!(
            "unsupported PyRTL trace schema `{}`",
            trace.schema
        )));
    }

    let import_start = Instant::now();
    let imported = import_export(export)?;
    let import_ns = import_start.elapsed().as_nanos();

    let setup_start = Instant::now();
    let plan = TraceReplayPlan::new(&imported, trace)?;
    let compiled = compile(&imported.design).map_err(|err| ImportError::new(format!("{err}")))?;
    let program = lower_to_packed_program(&compiled, &imported.top_name)
        .map_err(|err| ImportError::new(format!("{err}")))?;
    let setup_ns = setup_start.elapsed().as_nanos();

    for _ in 0..options.warmup {
        let mut sim = single_lane_simulator(&imported, &program)?;
        let mismatches = plan.replay_single(&mut sim)?;
        if !mismatches.is_empty() {
            return Ok(empty_bench_single_trace_report(
                trace, &options, import_ns, setup_ns, mismatches,
            ));
        }
    }

    let mut samples = Vec::with_capacity(options.repeat);
    for _ in 0..options.repeat {
        let mut sim = single_lane_simulator(&imported, &program)?;
        let start = Instant::now();
        let mismatches = plan.replay_single(&mut sim)?;
        let elapsed = start.elapsed().as_nanos();
        if !mismatches.is_empty() {
            return Ok(empty_bench_single_trace_report(
                trace, &options, import_ns, setup_ns, mismatches,
            ));
        }
        samples.push(elapsed);
    }

    let mut sorted = samples.clone();
    sorted.sort_unstable();
    let best = sorted.first().copied().unwrap_or(0);
    let median = sorted[sorted.len() / 2];
    Ok(BenchSingleTraceReport {
        schema: "rrtl-pyrtl-bench-single-trace-v1".to_string(),
        steps: trace.steps.len(),
        repeat: options.repeat,
        warmup: options.warmup,
        import_ns,
        setup_ns,
        replay_ns_samples: samples,
        replay_ns_best: best,
        replay_ns_median: median,
        mismatch_count: 0,
        mismatches: Vec::new(),
    })
}

pub fn bench_backends_trace(
    export: &PyrtlExport,
    trace: &PyrtlTrace,
    options: BenchBackendsTraceOptions,
) -> Result<BenchBackendsTraceReport, ImportError> {
    if options.repeat == 0 {
        return Err(ImportError::new("bench repeat must be greater than zero"));
    }
    if options.lanes == 0 {
        return Err(ImportError::new(
            "backend bench lanes must be greater than zero",
        ));
    }
    if options.backends.is_empty() {
        return Err(ImportError::new(
            "backend bench requires at least one backend",
        ));
    }
    if trace.schema != "rrtl-pyrtl-trace-v1" {
        return Err(ImportError::new(format!(
            "unsupported PyRTL trace schema `{}`",
            trace.schema
        )));
    }

    let import_start = Instant::now();
    let imported = import_export(export)?;
    let import_ns = import_start.elapsed().as_nanos();

    let setup_start = Instant::now();
    let plan = TraceReplayPlan::new(&imported, trace)?;
    let compiled = compile(&imported.design).map_err(|err| ImportError::new(format!("{err}")))?;
    let program = lower_to_packed_program(&compiled, &imported.top_name)
        .map_err(|err| ImportError::new(format!("{err}")))?;
    let simd_suitability =
        analyze_simd_suitability(&program).map_err(|err| ImportError::new(format!("{err}")))?;
    let encoded_plan = plan.encoded(&program)?;
    let setup_ns = setup_start.elapsed().as_nanos();

    let mut backend_reports = Vec::with_capacity(options.backends.len());
    for backend in &options.backends {
        backend_reports.push(bench_backend_trace(
            &imported,
            &plan,
            &encoded_plan,
            &program,
            *backend,
            &options,
        )?);
    }

    Ok(BenchBackendsTraceReport {
        schema: "rrtl-pyrtl-bench-backends-trace-v1".to_string(),
        steps: trace.steps.len(),
        repeat: options.repeat,
        warmup: options.warmup,
        import_ns,
        setup_ns,
        simd_suitability,
        backends: backend_reports,
    })
}

pub fn bench_threaded_trace(
    export: &PyrtlExport,
    trace: &PyrtlLaneTrace,
    options: BenchThreadedTraceOptions,
) -> Result<BenchThreadedTraceReport, ImportError> {
    if options.repeat == 0 {
        return Err(ImportError::new("bench repeat must be greater than zero"));
    }
    if trace.schema != "rrtl-pyrtl-lane-trace-v1" {
        return Err(ImportError::new(format!(
            "unsupported PyRTL lane trace schema `{}`",
            trace.schema
        )));
    }
    if trace.lanes == 0 {
        return Err(ImportError::new(
            "lane trace lanes must be greater than zero",
        ));
    }
    if options.max_workers == 0 {
        return Err(ImportError::new("max workers must be greater than zero"));
    }
    if !options.autotune && options.workers.is_empty() && !options.plan_first {
        return Err(ImportError::new(
            "threaded bench requires --autotune or at least one --worker",
        ));
    }

    let import_start = Instant::now();
    let imported = import_export(export)?;
    let import_ns = import_start.elapsed().as_nanos();

    let setup_start = Instant::now();
    let plan = LaneTraceReplayPlan::new(&imported, trace)?;
    let compiled = compile(&imported.design).map_err(|err| ImportError::new(format!("{err}")))?;
    let program = lower_to_packed_program(&compiled, &imported.top_name)
        .map_err(|err| ImportError::new(format!("{err}")))?;
    let simd_suitability =
        analyze_simd_suitability(&program).map_err(|err| ImportError::new(format!("{err}")))?;
    let backend_affinity =
        analyze_backend_affinity(&program).map_err(|err| ImportError::new(format!("{err}")))?;
    let encoded_plan = plan.encoded(&program)?;
    let replay_workload = encoded_plan.workload();
    let initial_state = threaded_initial_state(&imported)?;

    let explicit_layout = if options.workers.is_empty() {
        None
    } else {
        Some(ThreadedReplayOptions {
            workers: options.workers.clone(),
            max_mismatches: 16,
        })
    };
    let planned_selection = if options.plan_first && explicit_layout.is_none() && !options.autotune
    {
        let gpu_region_analysis =
            analyze_gpu_regions(&program).map_err(|err| ImportError::new(format!("{err}")))?;
        let gpu_selected_by_region = !gpu_backend_blocked(&gpu_region_analysis);
        let gpu_trace_compression =
            estimate_gpu_trace_compression(&plan.gpu_replay_plan(&program)?, trace.lanes);
        let (_profitability, backend_candidates, _profitability_features) =
            plan_backend_profitability(
                trace.lanes,
                trace.steps.len(),
                &simd_suitability,
                &backend_affinity,
                &gpu_region_analysis,
                &replay_workload,
                gpu_selected_by_region,
                Some(&gpu_trace_compression),
                options.planner_calibration.as_ref(),
            );
        let candidate_set = backend_plan_candidate_set(
            trace.lanes,
            options.max_workers,
            options.autotune_prune,
            &simd_suitability,
            &backend_affinity,
        );
        Some(select_threaded_layout_from_profitability(
            trace.lanes,
            options.max_workers,
            &backend_candidates,
            candidate_set.candidates,
            &simd_suitability,
            &backend_affinity,
            &replay_workload,
            options.planner_calibration.as_ref(),
        ))
    } else {
        None
    };

    let autotune_report = if options.autotune {
        Some(
            replay_trace_autotune_with_initial_state(
                &program,
                &encoded_plan,
                trace.lanes,
                ReplayAutotuneOptions {
                    warmup_steps: trace.steps.len().clamp(1, 16),
                    max_workers: options.max_workers,
                    candidates: explicit_layout.clone().into_iter().collect(),
                    simd_suitability: if explicit_layout.is_none() && options.autotune_prune {
                        Some(simd_suitability.clone())
                    } else {
                        None
                    },
                    backend_affinity: if explicit_layout.is_none() && options.autotune_prune {
                        Some(backend_affinity.clone())
                    } else {
                        None
                    },
                },
                &initial_state,
            )
            .map_err(|err| ImportError::new(format!("{err}")))?,
        )
    } else {
        None
    };
    let (selected_layout, selected_reason) = if let Some((layout, reason)) = planned_selection {
        (layout, reason)
    } else if let Some(report) = &autotune_report {
        let layout = report
            .candidates
            .get(report.selected_candidate)
            .map(|candidate| candidate.layout.clone())
            .ok_or_else(|| ImportError::new("autotune selected candidate is missing"))?;
        (layout, "autotune-selected".to_string())
    } else {
        let layout = explicit_layout
            .clone()
            .ok_or_else(|| ImportError::new("missing threaded worker layout"))?;
        (layout, "explicit-worker-layout".to_string())
    };
    let mut runner = ThreadedReplayRunner::new_with_initial_state(
        &program,
        &encoded_plan,
        &selected_layout,
        &initial_state,
    )
    .map_err(|err| ImportError::new(format!("{err}")))?;
    let setup_ns = setup_start.elapsed().as_nanos();

    for _ in 0..options.warmup {
        let replay = runner
            .replay()
            .map_err(|err| ImportError::new(format!("{err}")))?;
        let mismatches = plan.threaded_mismatches(&replay);
        if !mismatches.is_empty() {
            return Ok(empty_bench_threaded_trace_report(
                trace,
                &options,
                import_ns,
                setup_ns,
                replay,
                None,
                mismatches,
                simd_suitability.clone(),
                backend_affinity.clone(),
                replay_workload.clone(),
                selected_layout.clone(),
                selected_reason.clone(),
            ));
        }
    }

    let mut samples = Vec::with_capacity(options.repeat);
    let mut last_replay = None;
    for _ in 0..options.repeat {
        let start = Instant::now();
        let replay = runner
            .replay()
            .map_err(|err| ImportError::new(format!("{err}")))?;
        let elapsed = start.elapsed().as_nanos();
        let mismatches = plan.threaded_mismatches(&replay);
        if !mismatches.is_empty() {
            return Ok(empty_bench_threaded_trace_report(
                trace,
                &options,
                import_ns,
                setup_ns,
                replay,
                autotune_report.clone(),
                mismatches,
                simd_suitability.clone(),
                backend_affinity.clone(),
                replay_workload.clone(),
                selected_layout.clone(),
                selected_reason.clone(),
            ));
        }
        samples.push(elapsed);
        last_replay = Some(replay);
    }

    let mut sorted = samples.clone();
    sorted.sort_unstable();
    let best = sorted.first().copied().unwrap_or(0);
    let median = sorted[sorted.len() / 2];
    let autotune_pruned_candidates = autotune_report
        .as_ref()
        .map(|report| report.pruned_candidates.clone())
        .unwrap_or_default();
    Ok(BenchThreadedTraceReport {
        schema: "rrtl-pyrtl-bench-threaded-trace-v1".to_string(),
        steps: trace.steps.len(),
        lanes: trace.lanes,
        repeat: options.repeat,
        warmup: options.warmup,
        import_ns,
        setup_ns,
        simd_suitability,
        backend_affinity,
        replay_workload,
        selected_threaded_layout: selected_layout,
        selected_reason,
        replay_ns_samples: samples,
        replay_ns_best: best,
        replay_ns_median: median,
        replay: last_replay.ok_or_else(|| ImportError::new("missing threaded replay sample"))?,
        autotune: autotune_report,
        autotune_pruned_candidates,
        mismatch_count: 0,
        mismatches: Vec::new(),
    })
}

pub fn profile_replay(
    export: &PyrtlExport,
    trace: &PyrtlTrace,
    lane_trace: &PyrtlLaneTrace,
    profile: &RuntimeProfile,
    options: ProfileReplayOptions,
) -> Result<ProfileReplayReport, ImportError> {
    if options.repeat == 0 {
        return Err(ImportError::new(
            "profile replay repeat must be greater than zero",
        ));
    }
    if options.lanes == 0 {
        return Err(ImportError::new(
            "profile replay lanes must be greater than zero",
        ));
    }
    validate_runtime_profile(profile)?;
    let backend = profile
        .recommended_runtime_backend
        .as_deref()
        .ok_or_else(|| ImportError::new("runtime profile does not select a backend"))?;
    if backend == "rrtl_threaded_autotune_trace" {
        profile_replay_threaded(export, lane_trace, profile, &options)
    } else if backend == "rrtl_gpu_measured_trace" {
        profile_replay_gpu(export, lane_trace, profile, &options)
    } else if let Some(kind) = backend.strip_prefix("rrtl_backend:") {
        profile_replay_backend(
            export,
            trace,
            profile,
            &options,
            parse_profile_backend(kind)?,
        )
    } else {
        Err(ImportError::new(format!(
            "runtime profile backend `{backend}` is not replayable"
        )))
    }
}

pub fn validate_runtime_profile(profile: &RuntimeProfile) -> Result<(), ImportError> {
    if profile.schema != "rrtl-pyrtl-runtime-profile-v1" {
        return Err(ImportError::new(format!(
            "unsupported runtime profile schema `{}`",
            profile.schema
        )));
    }
    let backend = profile
        .recommended_runtime_backend
        .as_deref()
        .ok_or_else(|| ImportError::new("runtime profile does not select a backend"))?;
    if backend.is_empty() {
        return Err(ImportError::new(
            "runtime profile does not select a backend",
        ));
    }
    if profile.recommended_runtime_source.as_deref() == Some("no-valid-measurements") {
        return Err(ImportError::new(
            "runtime profile has no valid measurements",
        ));
    }
    if profile.selected_backend.is_none() {
        return Err(ImportError::new(
            "runtime profile missing selected backend details",
        ));
    }
    Ok(())
}

fn profile_replay_threaded(
    export: &PyrtlExport,
    trace: &PyrtlLaneTrace,
    profile: &RuntimeProfile,
    options: &ProfileReplayOptions,
) -> Result<ProfileReplayReport, ImportError> {
    if trace.schema != "rrtl-pyrtl-lane-trace-v1" {
        return Err(ImportError::new(format!(
            "unsupported PyRTL lane trace schema `{}`",
            trace.schema
        )));
    }
    let selected = profile
        .selected_backend
        .as_ref()
        .ok_or_else(|| ImportError::new("runtime profile missing selected backend details"))?;
    let layout = selected
        .selected_threaded_layout
        .as_ref()
        .ok_or_else(|| ImportError::new("runtime profile threaded selection has no layout"))?;
    let workers = runtime_profile_workers(&layout.workers)?;
    if workers.is_empty() {
        return Err(ImportError::new(
            "runtime profile threaded selection has no workers",
        ));
    }

    let import_start = Instant::now();
    let imported = import_export(export)?;
    let import_ns = import_start.elapsed().as_nanos();
    let setup_start = Instant::now();
    let plan = LaneTraceReplayPlan::new(&imported, trace)?;
    let compiled = compile(&imported.design).map_err(|err| ImportError::new(format!("{err}")))?;
    let program = lower_to_packed_program(&compiled, &imported.top_name)
        .map_err(|err| ImportError::new(format!("{err}")))?;
    let encoded_plan = plan.encoded(&program)?;
    let initial_state = threaded_initial_state(&imported)?;
    let setup_ns = setup_start.elapsed().as_nanos();
    let replay_options = ThreadedReplayOptions {
        workers,
        max_mismatches: 16,
    };
    let runner_start = Instant::now();
    let mut runner = ThreadedReplayRunner::new_with_initial_state(
        &program,
        &encoded_plan,
        &replay_options,
        &initial_state,
    )
    .map_err(|err| ImportError::new(format!("{err}")))?;
    let runner_setup_ns = runner_start.elapsed().as_nanos();

    let mut last_replay = None;
    let mut samples = Vec::with_capacity(options.repeat);
    for _ in 0..options.warmup {
        let replay = runner
            .replay()
            .map_err(|err| ImportError::new(format!("{err}")))?;
        let mismatches = plan.threaded_mismatches(&replay);
        if !mismatches.is_empty() {
            return profile_replay_mismatch_report(
                profile,
                trace.steps.len(),
                trace.lanes,
                options,
                import_ns,
                setup_ns,
                runner_setup_ns,
                serde_json::to_value(replay).map_err(|err| ImportError::new(format!("{err}")))?,
                mismatches,
            );
        }
    }
    for _ in 0..options.repeat {
        let start = Instant::now();
        let replay = runner
            .replay()
            .map_err(|err| ImportError::new(format!("{err}")))?;
        let elapsed = start.elapsed().as_nanos();
        let mismatches = plan.threaded_mismatches(&replay);
        if !mismatches.is_empty() {
            return profile_replay_mismatch_report(
                profile,
                trace.steps.len(),
                trace.lanes,
                options,
                import_ns,
                setup_ns,
                runner_setup_ns,
                serde_json::to_value(replay).map_err(|err| ImportError::new(format!("{err}")))?,
                mismatches,
            );
        }
        samples.push(elapsed);
        last_replay = Some(replay);
    }
    profile_replay_success_report(
        profile,
        trace.steps.len(),
        trace.lanes,
        options,
        import_ns,
        setup_ns,
        runner_setup_ns,
        samples,
        serde_json::to_value(
            last_replay.ok_or_else(|| ImportError::new("missing threaded replay sample"))?,
        )
        .map_err(|err| ImportError::new(format!("{err}")))?,
    )
}

fn profile_replay_gpu(
    export: &PyrtlExport,
    trace: &PyrtlLaneTrace,
    profile: &RuntimeProfile,
    options: &ProfileReplayOptions,
) -> Result<ProfileReplayReport, ImportError> {
    if trace.schema != "rrtl-pyrtl-lane-trace-v1" {
        return Err(ImportError::new(format!(
            "unsupported PyRTL lane trace schema `{}`",
            trace.schema
        )));
    }
    let selected = profile
        .selected_backend
        .as_ref()
        .ok_or_else(|| ImportError::new("runtime profile missing selected backend details"))?;
    let gpu_profile = selected
        .selected_gpu_options
        .ok_or_else(|| ImportError::new("runtime profile GPU selection has no options"))?;
    let gpu_options = GpuBatchOptions {
        workgroup_size: gpu_profile.workgroup_size.unwrap_or(128),
        memory_layout: gpu_profile
            .memory_layout
            .unwrap_or(GpuMemoryLayout::LaneMajor),
        reuse_temporaries: gpu_profile.reuse_temporaries,
        ..GpuBatchOptions::default()
    };
    let context = prepare_gpu_bench_context(export, trace)?;
    let runner_start = Instant::now();
    let gpu_replay_plan = context.plan.gpu_replay_plan(&context.program)?;
    let mut sim = gpu_simulator(
        &context.imported,
        &context.compiled,
        &context.program,
        trace.lanes,
        gpu_options,
    )?;
    let prepared_replay = sim
        .prepare_lane_trace_replay(
            &gpu_replay_plan,
            GpuTraceReplayOptions { max_mismatches: 16 },
        )
        .map_err(|err| ImportError::new(format!("{err}")))?;
    let snapshot = sim.prepare_storage_snapshot();
    let runner_setup_ns = runner_start.elapsed().as_nanos();
    let replay_options = BenchGpuTraceOptions {
        repeat: options.repeat,
        warmup: options.warmup,
        workgroup_size: gpu_options.workgroup_size,
        memory_layout: gpu_options.memory_layout,
        reuse_temporaries: gpu_options.reuse_temporaries,
        fused: true,
        max_mismatches: 16,
        plan_first: false,
        planner_calibration: None,
    };

    let mut last_replay = None;
    let mut samples = Vec::with_capacity(options.repeat);
    for _ in 0..options.warmup {
        let (mismatches, timing) = replay_gpu_trace_once(
            &context.plan,
            Some(&gpu_replay_plan),
            Some(&prepared_replay),
            Some(&snapshot),
            &mut sim,
            &replay_options,
        )?;
        if !mismatches.is_empty() {
            return profile_replay_mismatch_report(
                profile,
                trace.steps.len(),
                trace.lanes,
                options,
                context.import_ns,
                context.setup_ns,
                runner_setup_ns,
                serde_json::to_value(timing).map_err(|err| ImportError::new(format!("{err}")))?,
                mismatches,
            );
        }
    }
    for _ in 0..options.repeat {
        let (mismatches, timing) = replay_gpu_trace_once(
            &context.plan,
            Some(&gpu_replay_plan),
            Some(&prepared_replay),
            Some(&snapshot),
            &mut sim,
            &replay_options,
        )?;
        if !mismatches.is_empty() {
            return profile_replay_mismatch_report(
                profile,
                trace.steps.len(),
                trace.lanes,
                options,
                context.import_ns,
                context.setup_ns,
                runner_setup_ns,
                serde_json::to_value(timing).map_err(|err| ImportError::new(format!("{err}")))?,
                mismatches,
            );
        }
        samples.push(timing.total_ns);
        last_replay = Some(timing);
    }
    profile_replay_success_report(
        profile,
        trace.steps.len(),
        trace.lanes,
        options,
        context.import_ns,
        context.setup_ns,
        runner_setup_ns,
        samples,
        serde_json::to_value(
            last_replay.ok_or_else(|| ImportError::new("missing GPU replay sample"))?,
        )
        .map_err(|err| ImportError::new(format!("{err}")))?,
    )
}

fn profile_replay_backend(
    export: &PyrtlExport,
    trace: &PyrtlTrace,
    profile: &RuntimeProfile,
    options: &ProfileReplayOptions,
    backend: PyrtlBenchBackendKind,
) -> Result<ProfileReplayReport, ImportError> {
    if trace.schema != "rrtl-pyrtl-trace-v1" {
        return Err(ImportError::new(format!(
            "unsupported PyRTL trace schema `{}`",
            trace.schema
        )));
    }
    let lanes = if backend == PyrtlBenchBackendKind::Scalar {
        1
    } else {
        options.lanes
    };
    let import_start = Instant::now();
    let imported = import_export(export)?;
    let import_ns = import_start.elapsed().as_nanos();
    let setup_start = Instant::now();
    let plan = TraceReplayPlan::new(&imported, trace)?;
    let compiled = compile(&imported.design).map_err(|err| ImportError::new(format!("{err}")))?;
    let program = lower_to_packed_program(&compiled, &imported.top_name)
        .map_err(|err| ImportError::new(format!("{err}")))?;
    let encoded_plan = plan.encoded(&program)?;
    let setup_ns = setup_start.elapsed().as_nanos();
    let runner_start = Instant::now();
    let mut sim = backend_simulator(&imported, &program, backend, lanes)?;
    let snapshot = sim.snapshot_storage();
    let runner_setup_ns = runner_start.elapsed().as_nanos();

    let mut last_replay = None;
    let mut samples = Vec::with_capacity(options.repeat);
    for _ in 0..options.warmup {
        sim.restore_storage(&snapshot)
            .map_err(|err| ImportError::new(format!("{err}")))?;
        let replay = sim
            .replay_trace(&encoded_plan, backend_replay_options())
            .map_err(|err| ImportError::new(format!("{err}")))?;
        let mismatches = plan.backend_mismatches(&replay);
        if !mismatches.is_empty() {
            return profile_replay_mismatch_report(
                profile,
                trace.steps.len(),
                lanes,
                options,
                import_ns,
                setup_ns,
                runner_setup_ns,
                serde_json::to_value(replay).map_err(|err| ImportError::new(format!("{err}")))?,
                mismatches,
            );
        }
    }
    for _ in 0..options.repeat {
        sim.restore_storage(&snapshot)
            .map_err(|err| ImportError::new(format!("{err}")))?;
        let start = Instant::now();
        let replay = sim
            .replay_trace(&encoded_plan, backend_replay_options())
            .map_err(|err| ImportError::new(format!("{err}")))?;
        let elapsed = start.elapsed().as_nanos();
        let mismatches = plan.backend_mismatches(&replay);
        if !mismatches.is_empty() {
            return profile_replay_mismatch_report(
                profile,
                trace.steps.len(),
                lanes,
                options,
                import_ns,
                setup_ns,
                runner_setup_ns,
                serde_json::to_value(replay).map_err(|err| ImportError::new(format!("{err}")))?,
                mismatches,
            );
        }
        samples.push(elapsed);
        last_replay = Some(replay);
    }
    profile_replay_success_report(
        profile,
        trace.steps.len(),
        lanes,
        options,
        import_ns,
        setup_ns,
        runner_setup_ns,
        samples,
        serde_json::to_value(
            last_replay.ok_or_else(|| ImportError::new("missing backend replay sample"))?,
        )
        .map_err(|err| ImportError::new(format!("{err}")))?,
    )
}

fn profile_replay_success_report(
    profile: &RuntimeProfile,
    steps: usize,
    lanes: usize,
    options: &ProfileReplayOptions,
    import_ns: u128,
    setup_ns: u128,
    runner_setup_ns: u128,
    samples: Vec<u128>,
    replay: serde_json::Value,
) -> Result<ProfileReplayReport, ImportError> {
    let (best, median) = replay_sample_best_median(&samples)?;
    let setup_ns_total = import_ns + setup_ns + runner_setup_ns;
    let first_replay_ns = samples.first().copied().unwrap_or(0);
    let setup_to_hot_ratio = setup_to_replay_ratio(setup_ns_total, best);
    Ok(ProfileReplayReport {
        schema: "rrtl-pyrtl-profile-replay-hot-v1".to_string(),
        selected_backend: profile
            .recommended_runtime_backend
            .clone()
            .unwrap_or_default(),
        selected_source: profile.recommended_runtime_source.clone(),
        steps,
        lanes,
        repeat: options.repeat,
        warmup: options.warmup,
        import_ns,
        setup_ns,
        runner_setup_ns,
        setup_ns_total,
        replay_ns_samples: samples,
        first_replay_ns,
        replay_ns_best: best,
        replay_ns_median: median,
        hot_replay_ns_best: best,
        hot_replay_ns_median: median,
        setup_to_replay_ratio: setup_to_hot_ratio,
        setup_to_hot_ratio,
        hot_replay_speedup: replay_speedup(first_replay_ns, best),
        replay_ns_per_step: ns_per_unit(best, steps),
        replay_ns_per_lane_step: ns_per_unit(best, steps.saturating_mul(lanes)),
        mismatch_count: 0,
        mismatches: Vec::new(),
        replay,
    })
}

fn profile_replay_mismatch_report(
    profile: &RuntimeProfile,
    steps: usize,
    lanes: usize,
    options: &ProfileReplayOptions,
    import_ns: u128,
    setup_ns: u128,
    runner_setup_ns: u128,
    replay: serde_json::Value,
    mismatches: Vec<TraceMismatch>,
) -> Result<ProfileReplayReport, ImportError> {
    let mismatch_count = mismatches.len();
    let setup_ns_total = import_ns + setup_ns + runner_setup_ns;
    Ok(ProfileReplayReport {
        schema: "rrtl-pyrtl-profile-replay-hot-v1".to_string(),
        selected_backend: profile
            .recommended_runtime_backend
            .clone()
            .unwrap_or_default(),
        selected_source: profile.recommended_runtime_source.clone(),
        steps,
        lanes,
        repeat: options.repeat,
        warmup: options.warmup,
        import_ns,
        setup_ns,
        runner_setup_ns,
        setup_ns_total,
        replay_ns_samples: Vec::new(),
        first_replay_ns: 0,
        replay_ns_best: 0,
        replay_ns_median: 0,
        hot_replay_ns_best: 0,
        hot_replay_ns_median: 0,
        setup_to_replay_ratio: 0.0,
        setup_to_hot_ratio: 0.0,
        hot_replay_speedup: 0.0,
        replay_ns_per_step: 0.0,
        replay_ns_per_lane_step: 0.0,
        mismatch_count,
        mismatches,
        replay,
    })
}

fn replay_sample_best_median(samples: &[u128]) -> Result<(u128, u128), ImportError> {
    if samples.is_empty() {
        return Err(ImportError::new("missing profile replay samples"));
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    Ok((sorted[0], sorted[sorted.len() / 2]))
}

fn setup_to_replay_ratio(setup_ns: u128, replay_ns: u128) -> f64 {
    if replay_ns == 0 {
        0.0
    } else {
        setup_ns as f64 / replay_ns as f64
    }
}

fn replay_speedup(numerator_ns: u128, denominator_ns: u128) -> f64 {
    if numerator_ns == 0 || denominator_ns == 0 {
        0.0
    } else {
        numerator_ns as f64 / denominator_ns as f64
    }
}

fn ns_per_unit(ns: u128, units: usize) -> f64 {
    if units == 0 {
        0.0
    } else {
        ns as f64 / units as f64
    }
}

fn parse_profile_backend(value: &str) -> Result<PyrtlBenchBackendKind, ImportError> {
    match value {
        "scalar" => Ok(PyrtlBenchBackendKind::Scalar),
        "packed-cpu" | "packed" => Ok(PyrtlBenchBackendKind::PackedCpu),
        "simd-cpu" | "simd" => Ok(PyrtlBenchBackendKind::SimdCpu),
        "jit-cpu" | "jit" => Ok(PyrtlBenchBackendKind::JitCpu),
        other => Err(ImportError::new(format!(
            "unknown runtime profile backend `{other}`"
        ))),
    }
}

fn runtime_profile_workers(
    workers: &[RuntimeProfileWorker],
) -> Result<Vec<ThreadedReplayWorkerOptions>, ImportError> {
    workers
        .iter()
        .map(|worker| {
            if worker.lanes == 0 {
                return Err(ImportError::new(
                    "runtime profile worker lanes must be greater than zero",
                ));
            }
            Ok(ThreadedReplayWorkerOptions {
                backend: parse_profile_worker_backend(&worker.backend)?,
                lanes: worker.lanes,
            })
        })
        .collect()
}

fn parse_profile_worker_backend(value: &str) -> Result<SimBackendKind, ImportError> {
    match value {
        "scalar" => Ok(SimBackendKind::Scalar),
        "packed-cpu" | "packed" => Ok(SimBackendKind::PackedCpu),
        "simd-cpu" | "simd" => Ok(SimBackendKind::SimdCpu),
        "jit-cpu" | "jit" => Ok(SimBackendKind::JitCpu),
        other => Err(ImportError::new(format!(
            "unknown runtime profile worker backend `{other}`"
        ))),
    }
}

pub fn plan_backends(
    export: &PyrtlExport,
    trace: &PyrtlLaneTrace,
    options: PlanBackendsOptions,
) -> Result<BackendPlanReport, ImportError> {
    if trace.schema != "rrtl-pyrtl-lane-trace-v1" {
        return Err(ImportError::new(format!(
            "unsupported PyRTL lane trace schema `{}`",
            trace.schema
        )));
    }
    if trace.lanes == 0 {
        return Err(ImportError::new(
            "lane trace lanes must be greater than zero",
        ));
    }
    if options.max_workers == 0 {
        return Err(ImportError::new("max workers must be greater than zero"));
    }

    let imported = import_export(export)?;
    let plan = LaneTraceReplayPlan::new(&imported, trace)?;
    let compiled = compile(&imported.design).map_err(|err| ImportError::new(format!("{err}")))?;
    let program = lower_to_packed_program(&compiled, &imported.top_name)
        .map_err(|err| ImportError::new(format!("{err}")))?;
    let simd_suitability =
        analyze_simd_suitability(&program).map_err(|err| ImportError::new(format!("{err}")))?;
    let backend_affinity =
        analyze_backend_affinity(&program).map_err(|err| ImportError::new(format!("{err}")))?;
    let gpu_region_analysis =
        analyze_gpu_regions(&program).map_err(|err| ImportError::new(format!("{err}")))?;
    let gpu_trace_compression =
        estimate_gpu_trace_compression(&plan.gpu_replay_plan(&program)?, trace.lanes);
    let shader_stats = bench_gpu_shader_stats(
        gpu_shader_stats(&program, GpuBatchOptions::default())
            .map_err(|err| ImportError::new(format!("{err}")))?,
    );
    let replay_workload = plan.encoded(&program)?.workload();
    let candidate_set = backend_plan_candidate_set(
        trace.lanes,
        options.max_workers,
        options.autotune_prune,
        &simd_suitability,
        &backend_affinity,
    );
    let recommended_threaded_layouts = hybrid_threaded_layout_candidates(
        trace.lanes,
        options.max_workers,
        candidate_set.candidates.clone(),
        &simd_suitability,
        &backend_affinity,
        &replay_workload,
        options.planner_calibration.as_ref(),
    );
    let (selected_threaded_layout, selected_reason) = select_threaded_layout(
        trace.lanes,
        recommended_threaded_layouts.clone(),
        options.planner_calibration.as_ref(),
    );
    let hot_backend_preference =
        calibration_hot_backend_preference(options.planner_calibration.as_ref());
    let gpu_option_plan = plan_gpu_options(
        &program,
        &gpu_region_analysis,
        options.planner_calibration.as_ref(),
        hot_backend_preference.as_ref(),
    )?;
    let hot_backend_reason = hot_backend_calibration_reason(
        hot_backend_preference.as_ref(),
        gpu_option_plan.selected_gpu_options.is_some(),
        &gpu_region_analysis,
    );
    let (profitability, backend_candidates, profitability_features) = plan_backend_profitability(
        trace.lanes,
        trace.steps.len(),
        &simd_suitability,
        &backend_affinity,
        &gpu_region_analysis,
        &replay_workload,
        gpu_option_plan.selected_gpu_options.is_some(),
        Some(&gpu_trace_compression),
        options.planner_calibration.as_ref(),
    );
    let selected_runtime = selected_runtime_candidate(&backend_candidates);
    let selected_runtime_backend = selected_runtime
        .map(|candidate| candidate.backend.clone())
        .unwrap_or_else(|| "scalar".to_string());
    let selected_runtime_reason = selected_runtime_reason(selected_runtime);
    let pruned_runtime_candidates = backend_candidates
        .iter()
        .filter(|candidate| candidate.backend != selected_runtime_backend)
        .cloned()
        .collect::<Vec<_>>();

    Ok(BackendPlanReport {
        schema: "rrtl-pyrtl-backend-plan-v1".to_string(),
        steps: trace.steps.len(),
        lanes: trace.lanes,
        max_workers: options.max_workers,
        profitability,
        profitability_features,
        backend_candidates,
        selected_runtime_backend,
        selected_runtime_reason,
        pruned_runtime_candidates,
        simd_suitability,
        backend_affinity,
        gpu_region_analysis,
        shader_stats,
        replay_workload,
        selected_threaded_layout,
        selected_reason,
        planner_calibration_schema: options
            .planner_calibration
            .as_ref()
            .map(|calibration| calibration.schema.clone()),
        planner_calibration_threaded_reason: options
            .planner_calibration
            .as_ref()
            .map(|_| "threaded-layout-calibration-applied".to_string()),
        planner_calibration_gpu_reason: gpu_option_plan.calibration_reason,
        planner_calibration_hot_backend_reason: hot_backend_reason,
        planner_calibration_hot_backend_preference: hot_backend_preference
            .as_ref()
            .map(|preference| preference.signature.clone()),
        planner_calibration_hot_backend_score: hot_backend_preference
            .as_ref()
            .map(|preference| preference.score),
        selected_gpu_options: gpu_option_plan.selected_gpu_options,
        selected_gpu_reason: gpu_option_plan.selected_gpu_reason,
        recommended_gpu_options: gpu_option_plan.recommended_gpu_options,
        pruned_gpu_options: gpu_option_plan.pruned_gpu_options,
        recommended_threaded_layouts,
        pruned_threaded_layouts: candidate_set.pruned_candidates,
    })
}

pub fn bench_gpu_trace(
    export: &PyrtlExport,
    trace: &PyrtlLaneTrace,
    options: BenchGpuTraceOptions,
) -> Result<BenchGpuTraceReport, ImportError> {
    if options.repeat == 0 {
        return Err(ImportError::new("bench repeat must be greater than zero"));
    }
    if trace.schema != "rrtl-pyrtl-lane-trace-v1" {
        return Err(ImportError::new(format!(
            "unsupported PyRTL lane trace schema `{}`",
            trace.schema
        )));
    }
    if trace.lanes == 0 {
        return Err(ImportError::new(
            "lane trace lanes must be greater than zero",
        ));
    }

    let context = prepare_gpu_bench_context(export, trace)?;
    bench_gpu_trace_from_context(&context, trace, &options)
}

fn bench_gpu_trace_from_context(
    context: &PreparedGpuBenchContext,
    trace: &PyrtlLaneTrace,
    options: &BenchGpuTraceOptions,
) -> Result<BenchGpuTraceReport, ImportError> {
    let requested_gpu_options = GpuBatchOptions {
        workgroup_size: options.workgroup_size,
        memory_layout: options.memory_layout,
        reuse_temporaries: options.reuse_temporaries,
        ..GpuBatchOptions::default()
    };
    let option_selection_start = Instant::now();
    let gpu_options = if options.plan_first {
        let gpu_option_plan = plan_gpu_options(
            &context.program,
            &context.gpu_region_analysis,
            options.planner_calibration.as_ref(),
            None,
        )?;
        let (_profitability, backend_candidates, _profitability_features) =
            plan_backend_profitability(
                trace.lanes,
                trace.steps.len(),
                &context.simd_suitability,
                &context.backend_affinity,
                &context.gpu_region_analysis,
                &context.replay_workload,
                gpu_option_plan.selected_gpu_options.is_some(),
                Some(&context.gpu_trace_compression),
                options.planner_calibration.as_ref(),
            );
        if !profitability_allows_gpu(&backend_candidates) {
            let shader_stats = bench_gpu_shader_stats(
                gpu_shader_stats(&context.program, requested_gpu_options)
                    .map_err(|err| ImportError::new(format!("{err}")))?,
            );
            let setup_ns = context.setup_ns + option_selection_start.elapsed().as_nanos();
            return Ok(BenchGpuTraceReport {
                schema: "rrtl-pyrtl-bench-gpu-trace-v1".to_string(),
                steps: trace.steps.len(),
                lanes: trace.lanes,
                repeat: options.repeat,
                warmup: options.warmup,
                import_ns: context.import_ns,
                setup_ns,
                prepared_runner_setup_ns: 0,
                prepared_snapshot_setup_ns: 0,
                prepared_trace_bytes: 0,
                prepared_trace_uncompressed_bytes: 0,
                prepared_trace_compression_ratio_x100: 100,
                prepared_trace_uniform_input_ops: 0,
                prepared_trace_uniform_check_ops: 0,
                prepared_trace_layout: String::new(),
                prepared_trace_template_input_ops: 0,
                prepared_trace_template_check_ops: 0,
                prepared_trace_metadata_saved_words: 0,
                prepared_trace_fixed_template: false,
                prepared_trace_value_metadata_saved_words: 0,
                prepared_trace_value_stride_words: 0,
                hot_restore_ns_best: 0,
                hot_restore_ns_median: 0,
                hot_gpu_replay_ns_best: 0,
                hot_gpu_replay_ns_median: 0,
                gpu_single_submit_profitable: false,
                gpu_planner_calibration_reason: None,
                available: false,
                error: Some("gpu-not-selected-by-profitability".to_string()),
                backend_affinity: context.backend_affinity.clone(),
                gpu_region_analysis: context.gpu_region_analysis.clone(),
                shader_stats,
                gpu_replay_mode: gpu_replay_mode(options).to_string(),
                gpu_timing: GpuTraceReplayTiming::default(),
                replay_ns_samples: Vec::new(),
                replay_ns_best: 0,
                replay_ns_median: 0,
                mismatch_count: 0,
                mismatches: Vec::new(),
            });
        }
        match gpu_option_plan.selected_gpu_options {
            Some(plan_options) => plan_options,
            None => {
                let shader_stats = bench_gpu_shader_stats(
                    gpu_shader_stats(&context.program, requested_gpu_options)
                        .map_err(|err| ImportError::new(format!("{err}")))?,
                );
                let setup_ns = context.setup_ns + option_selection_start.elapsed().as_nanos();
                return Ok(BenchGpuTraceReport {
                    schema: "rrtl-pyrtl-bench-gpu-trace-v1".to_string(),
                    steps: trace.steps.len(),
                    lanes: trace.lanes,
                    repeat: options.repeat,
                    warmup: options.warmup,
                    import_ns: context.import_ns,
                    setup_ns,
                    prepared_runner_setup_ns: 0,
                    prepared_snapshot_setup_ns: 0,
                    prepared_trace_bytes: 0,
                    prepared_trace_uncompressed_bytes: 0,
                    prepared_trace_compression_ratio_x100: 100,
                    prepared_trace_uniform_input_ops: 0,
                    prepared_trace_uniform_check_ops: 0,
                    prepared_trace_layout: String::new(),
                    prepared_trace_template_input_ops: 0,
                    prepared_trace_template_check_ops: 0,
                    prepared_trace_metadata_saved_words: 0,
                    prepared_trace_fixed_template: false,
                    prepared_trace_value_metadata_saved_words: 0,
                    prepared_trace_value_stride_words: 0,
                    hot_restore_ns_best: 0,
                    hot_restore_ns_median: 0,
                    hot_gpu_replay_ns_best: 0,
                    hot_gpu_replay_ns_median: 0,
                    gpu_single_submit_profitable: false,
                    gpu_planner_calibration_reason: None,
                    available: false,
                    error: Some("gpu-not-selected-by-plan".to_string()),
                    backend_affinity: context.backend_affinity.clone(),
                    gpu_region_analysis: context.gpu_region_analysis.clone(),
                    shader_stats,
                    gpu_replay_mode: gpu_replay_mode(options).to_string(),
                    gpu_timing: GpuTraceReplayTiming::default(),
                    replay_ns_samples: Vec::new(),
                    replay_ns_best: 0,
                    replay_ns_median: 0,
                    mismatch_count: 0,
                    mismatches: Vec::new(),
                });
            }
        }
    } else {
        requested_gpu_options
    };
    let setup_ns_base = context.setup_ns + option_selection_start.elapsed().as_nanos();
    bench_gpu_trace_with_options(context, trace, options, gpu_options, setup_ns_base)
}

pub fn bench_gpu_options(
    export: &PyrtlExport,
    trace: &PyrtlLaneTrace,
    options: BenchGpuTraceOptions,
) -> Result<BenchGpuOptionsReport, ImportError> {
    if options.repeat == 0 {
        return Err(ImportError::new("bench repeat must be greater than zero"));
    }
    if trace.schema != "rrtl-pyrtl-lane-trace-v1" {
        return Err(ImportError::new(format!(
            "unsupported PyRTL lane trace schema `{}`",
            trace.schema
        )));
    }
    if trace.lanes == 0 {
        return Err(ImportError::new(
            "lane trace lanes must be greater than zero",
        ));
    }

    let context = prepare_gpu_bench_context(export, trace)?;
    bench_gpu_options_from_context(&context, trace, &options)
}

fn bench_gpu_options_from_context(
    context: &PreparedGpuBenchContext,
    trace: &PyrtlLaneTrace,
    options: &BenchGpuTraceOptions,
) -> Result<BenchGpuOptionsReport, ImportError> {
    let option_selection_start = Instant::now();
    let planned_options = plan_gpu_options(
        &context.program,
        &context.gpu_region_analysis,
        options.planner_calibration.as_ref(),
        None,
    )?
    .recommended_gpu_options;
    let setup_ns_base = context.setup_ns + option_selection_start.elapsed().as_nanos();

    let mut candidates = Vec::with_capacity(planned_options.len());
    for (candidate_index, planned) in planned_options.into_iter().enumerate() {
        let report = bench_gpu_trace_with_options(
            &context,
            trace,
            &BenchGpuTraceOptions {
                workgroup_size: planned.options.workgroup_size,
                memory_layout: planned.options.memory_layout,
                reuse_temporaries: planned.options.reuse_temporaries,
                plan_first: false,
                ..options.clone()
            },
            planned.options,
            setup_ns_base,
        )?;
        candidates.push(BenchGpuOptionCandidateReport {
            candidate_index,
            planned,
            available: report.available,
            error: report.error.clone(),
            replay_ns_samples: report.replay_ns_samples.clone(),
            replay_ns_best: report.replay_ns_best,
            replay_ns_median: report.replay_ns_median,
            mismatch_count: report.mismatch_count,
            report,
        });
    }
    let selected_candidate_index = candidates
        .iter()
        .filter(|candidate| {
            candidate.available && candidate.mismatch_count == 0 && candidate.replay_ns_best > 0
        })
        .min_by_key(|candidate| {
            (
                candidate.replay_ns_best,
                candidate.replay_ns_median,
                candidate.candidate_index,
            )
        })
        .map(|candidate| candidate.candidate_index);

    Ok(BenchGpuOptionsReport {
        schema: "rrtl-pyrtl-bench-gpu-options-v1".to_string(),
        steps: trace.steps.len(),
        lanes: trace.lanes,
        repeat: options.repeat,
        warmup: options.warmup,
        candidates,
        selected_candidate_index,
    })
}

pub fn bench_gpu_combined(
    export: &PyrtlExport,
    trace: &PyrtlLaneTrace,
    options: BenchGpuTraceOptions,
) -> Result<BenchGpuCombinedReport, ImportError> {
    if options.repeat == 0 {
        return Err(ImportError::new("bench repeat must be greater than zero"));
    }
    if trace.schema != "rrtl-pyrtl-lane-trace-v1" {
        return Err(ImportError::new(format!(
            "unsupported PyRTL lane trace schema `{}`",
            trace.schema
        )));
    }
    if trace.lanes == 0 {
        return Err(ImportError::new(
            "lane trace lanes must be greater than zero",
        ));
    }

    let context = prepare_gpu_bench_context(export, trace)?;
    let static_trace = bench_gpu_trace_from_context(
        &context,
        trace,
        &BenchGpuTraceOptions {
            plan_first: true,
            ..options.clone()
        },
    )?;
    let option_sweep = bench_gpu_options_from_context(&context, trace, &options)?;
    let measured_trace =
        bench_gpu_measured_trace_from_context(&context, trace, &options, &option_sweep)?;

    Ok(BenchGpuCombinedReport {
        schema: "rrtl-pyrtl-bench-gpu-combined-v1".to_string(),
        static_trace,
        option_sweep,
        measured_trace,
    })
}

fn bench_gpu_measured_trace_from_context(
    context: &PreparedGpuBenchContext,
    trace: &PyrtlLaneTrace,
    options: &BenchGpuTraceOptions,
    sweep: &BenchGpuOptionsReport,
) -> Result<BenchGpuTraceReport, ImportError> {
    let Some(selected_index) = sweep.selected_candidate_index else {
        return unavailable_measured_gpu_trace(
            context,
            trace,
            options,
            "gpu-measured-option-not-selected",
            GpuBatchOptions::default(),
        );
    };
    let Some(candidate) = sweep.candidates.get(selected_index) else {
        return unavailable_measured_gpu_trace(
            context,
            trace,
            options,
            "gpu-measured-option-not-selected",
            GpuBatchOptions::default(),
        );
    };
    if !candidate.available || candidate.mismatch_count != 0 || candidate.replay_ns_best == 0 {
        return unavailable_measured_gpu_trace(
            context,
            trace,
            options,
            "gpu-measured-option-invalid",
            candidate.planned.options,
        );
    }
    let measured_options = BenchGpuTraceOptions {
        workgroup_size: candidate.planned.options.workgroup_size,
        memory_layout: candidate.planned.options.memory_layout,
        reuse_temporaries: candidate.planned.options.reuse_temporaries,
        plan_first: false,
        ..options.clone()
    };
    bench_gpu_trace_with_options(
        context,
        trace,
        &measured_options,
        candidate.planned.options,
        context.setup_ns,
    )
}

fn unavailable_measured_gpu_trace(
    context: &PreparedGpuBenchContext,
    trace: &PyrtlLaneTrace,
    options: &BenchGpuTraceOptions,
    error: &str,
    gpu_options: GpuBatchOptions,
) -> Result<BenchGpuTraceReport, ImportError> {
    let shader_stats = bench_gpu_shader_stats(
        gpu_shader_stats(&context.program, gpu_options)
            .map_err(|err| ImportError::new(format!("{err}")))?,
    );
    Ok(BenchGpuTraceReport {
        schema: "rrtl-pyrtl-bench-gpu-trace-v1".to_string(),
        steps: trace.steps.len(),
        lanes: trace.lanes,
        repeat: options.repeat,
        warmup: options.warmup,
        import_ns: context.import_ns,
        setup_ns: context.setup_ns,
        prepared_runner_setup_ns: 0,
        prepared_snapshot_setup_ns: 0,
        prepared_trace_bytes: 0,
        prepared_trace_uncompressed_bytes: 0,
        prepared_trace_compression_ratio_x100: 100,
        prepared_trace_uniform_input_ops: 0,
        prepared_trace_uniform_check_ops: 0,
        prepared_trace_layout: String::new(),
        prepared_trace_template_input_ops: 0,
        prepared_trace_template_check_ops: 0,
        prepared_trace_metadata_saved_words: 0,
        prepared_trace_fixed_template: false,
        prepared_trace_value_metadata_saved_words: 0,
        prepared_trace_value_stride_words: 0,
        hot_restore_ns_best: 0,
        hot_restore_ns_median: 0,
        hot_gpu_replay_ns_best: 0,
        hot_gpu_replay_ns_median: 0,
        gpu_single_submit_profitable: false,
        gpu_planner_calibration_reason: None,
        available: false,
        error: Some(error.to_string()),
        backend_affinity: context.backend_affinity.clone(),
        gpu_region_analysis: context.gpu_region_analysis.clone(),
        shader_stats,
        gpu_replay_mode: gpu_replay_mode(options).to_string(),
        gpu_timing: GpuTraceReplayTiming::default(),
        replay_ns_samples: Vec::new(),
        replay_ns_best: 0,
        replay_ns_median: 0,
        mismatch_count: 0,
        mismatches: Vec::new(),
    })
}

struct PreparedGpuBenchContext {
    imported: ImportedDesign,
    compiled: rrtl_core::CompiledDesign,
    program: PackedProgram,
    plan: LaneTraceReplayPlan,
    simd_suitability: SimdSuitabilityReport,
    backend_affinity: BackendAffinityReport,
    gpu_region_analysis: GpuRegionAnalysis,
    replay_workload: EncodedTraceReplayWorkload,
    gpu_trace_compression: GpuTraceCompressionEstimate,
    import_ns: u128,
    setup_ns: u128,
}

#[derive(Clone, Copy, Debug, Default)]
struct GpuTraceCompressionEstimate {
    compressed_bytes: usize,
    uncompressed_bytes: usize,
    compression_ratio_x100: usize,
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

fn estimate_gpu_trace_compression(
    plan: &GpuTraceReplayPlan,
    lanes: usize,
) -> GpuTraceCompressionEstimate {
    let uniform_input_ops = plan
        .inputs
        .iter()
        .filter(|op| values_are_uniform(&op.values))
        .count();
    let uniform_check_ops = plan
        .checks
        .iter()
        .filter(|op| values_are_uniform(&op.expected))
        .count();
    let dense_input_ops = plan.inputs.len().saturating_sub(uniform_input_ops);
    let dense_check_ops = plan.checks.len().saturating_sub(uniform_check_ops);
    let input_value_words = uniform_input_ops + dense_input_ops.saturating_mul(lanes);
    let check_value_words = uniform_check_ops + dense_check_ops.saturating_mul(lanes);
    let uncompressed_words = plan.steps.saturating_add(1).saturating_mul(2)
        + 10
        + plan.inputs.len().saturating_mul(5)
        + plan.inputs.len().saturating_mul(lanes)
        + plan.checks.len().saturating_mul(5)
        + plan.checks.len().saturating_mul(lanes);
    let (template_layout, template_input_ops, template_check_ops) =
        estimate_gpu_trace_template_shape(plan);
    let (fixed_template, value_stride_words) = if template_layout {
        estimate_gpu_trace_fixed_template(plan, lanes)
    } else {
        (false, 0)
    };
    let step_indexed_metadata_words =
        plan.inputs.len().saturating_mul(5) + plan.checks.len().saturating_mul(5);
    let templated_metadata_words = template_input_ops.saturating_mul(2)
        + template_check_ops.saturating_mul(3)
        + if fixed_template {
            0
        } else {
            plan.inputs.len().saturating_mul(2) + plan.checks.len().saturating_mul(2)
        };
    let metadata_saved_words = if template_layout {
        step_indexed_metadata_words.saturating_sub(templated_metadata_words)
    } else {
        0
    };
    let value_metadata_saved_words = if fixed_template {
        plan.inputs.len().saturating_mul(2) + plan.checks.len().saturating_mul(2)
    } else {
        0
    };
    let compressed_words = plan.steps.saturating_add(1).saturating_mul(2)
        + 10
        + if template_layout {
            templated_metadata_words
        } else {
            step_indexed_metadata_words
        }
        + input_value_words
        + check_value_words;
    let compression_ratio_x100 = if uncompressed_words == 0 {
        100
    } else {
        compressed_words * 100 / uncompressed_words
    };
    GpuTraceCompressionEstimate {
        compressed_bytes: compressed_words * std::mem::size_of::<u32>(),
        uncompressed_bytes: uncompressed_words * std::mem::size_of::<u32>(),
        compression_ratio_x100,
        uniform_input_ops,
        uniform_check_ops,
        template_layout,
        template_input_ops,
        template_check_ops,
        metadata_saved_words,
        fixed_template,
        value_metadata_saved_words,
        value_stride_words,
    }
}

fn estimate_gpu_trace_fixed_template(plan: &GpuTraceReplayPlan, lanes: usize) -> (bool, usize) {
    if plan.steps <= 1 {
        return (false, 0);
    }
    let mut input_buckets = vec![Vec::new(); plan.steps];
    for op in &plan.inputs {
        input_buckets[op.step].push(if values_are_uniform(&op.values) { 1 } else { 0 });
    }
    let mut check_buckets = vec![Vec::new(); plan.steps];
    for op in &plan.checks {
        check_buckets[op.step].push(if values_are_uniform(&op.expected) {
            1
        } else {
            0
        });
    }
    let Some(first_inputs) = input_buckets.first() else {
        return (false, 0);
    };
    let Some(first_checks) = check_buckets.first() else {
        return (false, 0);
    };
    if !input_buckets.iter().all(|bucket| bucket == first_inputs)
        || !check_buckets.iter().all(|bucket| bucket == first_checks)
    {
        return (false, 0);
    }
    let input_stride = first_inputs
        .iter()
        .map(|mode| if *mode == 1 { 1 } else { lanes })
        .sum::<usize>();
    let check_stride = first_checks
        .iter()
        .map(|mode| if *mode == 1 { 1 } else { lanes })
        .sum::<usize>();
    (true, input_stride + check_stride)
}

fn estimate_gpu_trace_template_shape(plan: &GpuTraceReplayPlan) -> (bool, usize, usize) {
    if plan.steps <= 1 {
        return (false, 0, 0);
    }
    let mut input_buckets = vec![Vec::new(); plan.steps];
    for op in &plan.inputs {
        input_buckets[op.step].push((op.signal, op.limb));
    }
    let mut check_buckets = vec![Vec::new(); plan.steps];
    for op in &plan.checks {
        check_buckets[op.step].push((op.check_index, op.signal, op.limb));
    }
    let Some(first_inputs) = input_buckets.first() else {
        return (false, 0, 0);
    };
    let Some(first_checks) = check_buckets.first() else {
        return (false, 0, 0);
    };
    let stable_inputs = input_buckets.iter().all(|bucket| bucket == first_inputs);
    let stable_checks = check_buckets.iter().all(|bucket| bucket == first_checks);
    if stable_inputs && stable_checks {
        (true, first_inputs.len(), first_checks.len())
    } else {
        (false, 0, 0)
    }
}

fn values_are_uniform(values: &[u32]) -> bool {
    values
        .first()
        .map(|first| values.iter().all(|value| value == first))
        .unwrap_or(true)
}

fn prepare_gpu_bench_context(
    export: &PyrtlExport,
    trace: &PyrtlLaneTrace,
) -> Result<PreparedGpuBenchContext, ImportError> {
    let import_start = Instant::now();
    let imported = import_export(export)?;
    let import_ns = import_start.elapsed().as_nanos();

    let setup_start = Instant::now();
    let plan = LaneTraceReplayPlan::new(&imported, trace)?;
    let compiled = compile(&imported.design).map_err(|err| ImportError::new(format!("{err}")))?;
    let program = lower_to_packed_program(&compiled, &imported.top_name)
        .map_err(|err| ImportError::new(format!("{err}")))?;
    let simd_suitability =
        analyze_simd_suitability(&program).map_err(|err| ImportError::new(format!("{err}")))?;
    let backend_affinity =
        analyze_backend_affinity(&program).map_err(|err| ImportError::new(format!("{err}")))?;
    let gpu_region_analysis =
        analyze_gpu_regions(&program).map_err(|err| ImportError::new(format!("{err}")))?;
    let replay_workload = plan.encoded(&program)?.workload();
    let gpu_trace_compression =
        estimate_gpu_trace_compression(&plan.gpu_replay_plan(&program)?, trace.lanes);
    let setup_ns = setup_start.elapsed().as_nanos();

    Ok(PreparedGpuBenchContext {
        imported,
        compiled,
        program,
        plan,
        simd_suitability,
        backend_affinity,
        gpu_region_analysis,
        replay_workload,
        gpu_trace_compression,
        import_ns,
        setup_ns,
    })
}

fn bench_gpu_trace_with_options(
    context: &PreparedGpuBenchContext,
    trace: &PyrtlLaneTrace,
    options: &BenchGpuTraceOptions,
    gpu_options: GpuBatchOptions,
    setup_ns_base: u128,
) -> Result<BenchGpuTraceReport, ImportError> {
    let option_setup_start = Instant::now();
    let gpu_replay_plan = if options.fused {
        Some(context.plan.gpu_replay_plan(&context.program)?)
    } else {
        None
    };
    let shader_stats = bench_gpu_shader_stats(
        gpu_shader_stats(&context.program, gpu_options)
            .map_err(|err| ImportError::new(format!("{err}")))?,
    );

    let mut sim = match gpu_simulator(
        &context.imported,
        &context.compiled,
        &context.program,
        trace.lanes,
        gpu_options,
    ) {
        Ok(sim) => sim,
        Err(err) => {
            let setup_ns = setup_ns_base + option_setup_start.elapsed().as_nanos();
            return Ok(BenchGpuTraceReport {
                schema: "rrtl-pyrtl-bench-gpu-trace-v1".to_string(),
                steps: trace.steps.len(),
                lanes: trace.lanes,
                repeat: options.repeat,
                warmup: options.warmup,
                import_ns: context.import_ns,
                setup_ns,
                prepared_runner_setup_ns: 0,
                prepared_snapshot_setup_ns: 0,
                prepared_trace_bytes: 0,
                prepared_trace_uncompressed_bytes: 0,
                prepared_trace_compression_ratio_x100: 100,
                prepared_trace_uniform_input_ops: 0,
                prepared_trace_uniform_check_ops: 0,
                prepared_trace_layout: String::new(),
                prepared_trace_template_input_ops: 0,
                prepared_trace_template_check_ops: 0,
                prepared_trace_metadata_saved_words: 0,
                prepared_trace_fixed_template: false,
                prepared_trace_value_metadata_saved_words: 0,
                prepared_trace_value_stride_words: 0,
                hot_restore_ns_best: 0,
                hot_restore_ns_median: 0,
                hot_gpu_replay_ns_best: 0,
                hot_gpu_replay_ns_median: 0,
                gpu_single_submit_profitable: false,
                gpu_planner_calibration_reason: None,
                available: false,
                error: Some(err.to_string()),
                backend_affinity: context.backend_affinity.clone(),
                gpu_region_analysis: context.gpu_region_analysis.clone(),
                shader_stats,
                gpu_replay_mode: gpu_replay_mode(options).to_string(),
                gpu_timing: GpuTraceReplayTiming::default(),
                replay_ns_samples: Vec::new(),
                replay_ns_best: 0,
                replay_ns_median: 0,
                mismatch_count: 0,
                mismatches: Vec::new(),
            });
        }
    };
    let prepared_setup_start = Instant::now();
    let prepared_replay = match gpu_replay_plan.as_ref() {
        Some(plan) => Some(
            sim.prepare_lane_trace_replay(
                plan,
                GpuTraceReplayOptions {
                    max_mismatches: options.max_mismatches,
                },
            )
            .map_err(|err| ImportError::new(format!("{err}")))?,
        ),
        None => None,
    };
    let prepared_runner_setup_ns = prepared_setup_start.elapsed().as_nanos();
    let prepared_trace_bytes = prepared_replay
        .as_ref()
        .map(|prepared| prepared.trace_data_bytes())
        .unwrap_or(0);
    let prepared_trace_uncompressed_bytes = prepared_replay
        .as_ref()
        .map(|prepared| prepared.trace_data_uncompressed_bytes())
        .unwrap_or(0);
    let prepared_trace_compression_ratio_x100 = prepared_replay
        .as_ref()
        .map(|prepared| prepared.trace_data_compression_ratio_x100())
        .unwrap_or(100);
    let prepared_trace_uniform_input_ops = prepared_replay
        .as_ref()
        .map(|prepared| prepared.uniform_input_ops())
        .unwrap_or(0);
    let prepared_trace_uniform_check_ops = prepared_replay
        .as_ref()
        .map(|prepared| prepared.uniform_check_ops())
        .unwrap_or(0);
    let prepared_trace_layout = prepared_replay
        .as_ref()
        .map(|prepared| prepared.trace_layout().to_string())
        .unwrap_or_default();
    let prepared_trace_template_input_ops = prepared_replay
        .as_ref()
        .map(|prepared| prepared.template_input_ops())
        .unwrap_or(0);
    let prepared_trace_template_check_ops = prepared_replay
        .as_ref()
        .map(|prepared| prepared.template_check_ops())
        .unwrap_or(0);
    let prepared_trace_metadata_saved_words = prepared_replay
        .as_ref()
        .map(|prepared| prepared.metadata_saved_words())
        .unwrap_or(0);
    let prepared_trace_fixed_template = prepared_replay
        .as_ref()
        .map(|prepared| prepared.fixed_template())
        .unwrap_or(false);
    let prepared_trace_value_metadata_saved_words = prepared_replay
        .as_ref()
        .map(|prepared| prepared.value_metadata_saved_words())
        .unwrap_or(0);
    let prepared_trace_value_stride_words = prepared_replay
        .as_ref()
        .map(|prepared| prepared.value_stride_words())
        .unwrap_or(0);
    let snapshot_setup_start = Instant::now();
    let snapshot = sim.prepare_storage_snapshot();
    let prepared_snapshot_setup_ns = snapshot_setup_start.elapsed().as_nanos();
    let setup_ns = setup_ns_base + option_setup_start.elapsed().as_nanos();

    for _ in 0..options.warmup {
        if prepared_replay.is_none() {
            sim.restore_prepared_storage(&snapshot)
                .map_err(|err| ImportError::new(format!("{err}")))?;
        }
        let (mismatches, _) = replay_gpu_trace_once(
            &context.plan,
            gpu_replay_plan.as_ref(),
            prepared_replay.as_ref(),
            prepared_replay.as_ref().map(|_| &snapshot),
            &mut sim,
            options,
        )?;
        if !mismatches.is_empty() {
            return Ok(empty_bench_gpu_trace_report(
                trace,
                options,
                context.import_ns,
                setup_ns,
                context.backend_affinity.clone(),
                context.gpu_region_analysis.clone(),
                shader_stats.clone(),
                mismatches,
            ));
        }
    }

    let mut samples = Vec::with_capacity(options.repeat);
    let mut restore_samples = Vec::with_capacity(options.repeat);
    let mut gpu_timing = GpuTraceReplayTiming::default();
    for _ in 0..options.repeat {
        if prepared_replay.is_some() {
            restore_samples.push(0);
        } else {
            let restore_start = Instant::now();
            sim.restore_prepared_storage(&snapshot)
                .map_err(|err| ImportError::new(format!("{err}")))?;
            restore_samples.push(restore_start.elapsed().as_nanos());
        }
        let start = Instant::now();
        let (mismatches, timing) = replay_gpu_trace_once(
            &context.plan,
            gpu_replay_plan.as_ref(),
            prepared_replay.as_ref(),
            prepared_replay.as_ref().map(|_| &snapshot),
            &mut sim,
            options,
        )?;
        let elapsed = start.elapsed().as_nanos();
        if !mismatches.is_empty() {
            return Ok(empty_bench_gpu_trace_report(
                trace,
                options,
                context.import_ns,
                setup_ns,
                context.backend_affinity.clone(),
                context.gpu_region_analysis.clone(),
                shader_stats.clone(),
                mismatches,
            ));
        }
        gpu_timing = timing;
        samples.push(if options.fused {
            gpu_timing.total_ns
        } else {
            elapsed
        });
    }

    let mut sorted = samples.clone();
    sorted.sort_unstable();
    let best = sorted.first().copied().unwrap_or(0);
    let median = sorted[sorted.len() / 2];
    let (hot_restore_ns_best, hot_restore_ns_median) = replay_sample_best_median(&restore_samples)?;
    let (hot_gpu_replay_ns_best, hot_gpu_replay_ns_median) =
        hot_gpu_replay_ns(best, median, &gpu_timing);
    let gpu_single_submit_profitable = gpu_single_submit_profitable(
        trace.lanes,
        trace.steps.len(),
        &context.gpu_region_analysis,
        prepared_trace_bytes,
        &gpu_timing,
    );
    let gpu_planner_calibration_reason =
        gpu_single_submit_profitable.then(|| "single-submit-hot-gpu-profitable".to_string());
    Ok(BenchGpuTraceReport {
        schema: "rrtl-pyrtl-bench-gpu-trace-v1".to_string(),
        steps: trace.steps.len(),
        lanes: trace.lanes,
        repeat: options.repeat,
        warmup: options.warmup,
        import_ns: context.import_ns,
        setup_ns,
        prepared_runner_setup_ns,
        prepared_snapshot_setup_ns,
        prepared_trace_bytes,
        prepared_trace_uncompressed_bytes,
        prepared_trace_compression_ratio_x100,
        prepared_trace_uniform_input_ops,
        prepared_trace_uniform_check_ops,
        prepared_trace_layout,
        prepared_trace_template_input_ops,
        prepared_trace_template_check_ops,
        prepared_trace_metadata_saved_words,
        prepared_trace_fixed_template,
        prepared_trace_value_metadata_saved_words,
        prepared_trace_value_stride_words,
        hot_restore_ns_best,
        hot_restore_ns_median,
        hot_gpu_replay_ns_best,
        hot_gpu_replay_ns_median,
        gpu_single_submit_profitable,
        gpu_planner_calibration_reason,
        available: true,
        error: None,
        backend_affinity: context.backend_affinity.clone(),
        gpu_region_analysis: context.gpu_region_analysis.clone(),
        shader_stats,
        gpu_replay_mode: gpu_replay_mode(options).to_string(),
        gpu_timing,
        replay_ns_samples: samples,
        replay_ns_best: best,
        replay_ns_median: median,
        mismatch_count: 0,
        mismatches: Vec::new(),
    })
}

fn bench_gpu_shader_stats(stats: rrtl_gpu_sim::GpuShaderStats) -> BenchGpuShaderStats {
    BenchGpuShaderStats {
        wgsl_bytes: stats.wgsl_bytes,
        optimized_temp_slots: stats.optimized_temp_slots,
        optimized_value_vars: stats.optimized_value_vars,
        unoptimized_packets_total: stats.unoptimized_packets.total,
        optimized_packets_total: stats.optimized_packets.total,
        unoptimized_memory_reads: stats.unoptimized_memory.total_reads,
        unoptimized_memory_writes: stats.unoptimized_memory.total_writes,
        optimized_memory_reads: stats.optimized_memory.total_reads,
        optimized_memory_writes: stats.optimized_memory.total_writes,
        workgroup_size: stats.workgroup_size,
        memory_layout: stats.memory_layout,
        reuse_temporaries: stats.reuse_temporaries,
        total_memory_words_per_lane: stats.total_memory_words_per_lane,
    }
}

fn hot_gpu_replay_ns(best: u128, median: u128, timing: &GpuTraceReplayTiming) -> (u128, u128) {
    if timing.single_submit_used && timing.single_submit_ns > 0 {
        (timing.single_submit_ns, timing.single_submit_ns)
    } else {
        (best, median)
    }
}

fn gpu_single_submit_profitable(
    lanes: usize,
    steps: usize,
    analysis: &GpuRegionAnalysis,
    prepared_trace_bytes: usize,
    timing: &GpuTraceReplayTiming,
) -> bool {
    timing.single_submit_used
        && timing.single_submit_ns > 0
        && timing.full_readback_words == 0
        && prepared_trace_bytes > 0
        && lanes.saturating_mul(steps) >= 512
        && !gpu_backend_blocked(analysis)
}

fn empty_bench_trace_report(
    trace: &PyrtlTrace,
    options: &BenchTraceOptions,
    import_ns: u128,
    setup_ns: u128,
    mismatches: Vec<TraceMismatch>,
) -> BenchTraceReport {
    let mismatch_count = mismatches.len();
    BenchTraceReport {
        schema: "rrtl-pyrtl-bench-trace-v1".to_string(),
        steps: trace.steps.len(),
        repeat: options.repeat,
        warmup: options.warmup,
        import_ns,
        setup_ns,
        replay_ns_samples: Vec::new(),
        replay_ns_best: 0,
        replay_ns_median: 0,
        mismatch_count,
        mismatches,
    }
}

fn empty_bench_packed_trace_report(
    trace: &PyrtlTrace,
    options: &BenchPackedTraceOptions,
    import_ns: u128,
    setup_ns: u128,
    mismatches: Vec<TraceMismatch>,
) -> BenchPackedTraceReport {
    let mismatch_count = mismatches.len();
    BenchPackedTraceReport {
        schema: "rrtl-pyrtl-bench-packed-trace-v1".to_string(),
        steps: trace.steps.len(),
        repeat: options.repeat,
        warmup: options.warmup,
        lanes: options.lanes,
        import_ns,
        setup_ns,
        replay_ns_samples: Vec::new(),
        replay_ns_best: 0,
        replay_ns_median: 0,
        mismatch_count,
        mismatches,
    }
}

fn empty_bench_single_trace_report(
    trace: &PyrtlTrace,
    options: &BenchSingleTraceOptions,
    import_ns: u128,
    setup_ns: u128,
    mismatches: Vec<TraceMismatch>,
) -> BenchSingleTraceReport {
    let mismatch_count = mismatches.len();
    BenchSingleTraceReport {
        schema: "rrtl-pyrtl-bench-single-trace-v1".to_string(),
        steps: trace.steps.len(),
        repeat: options.repeat,
        warmup: options.warmup,
        import_ns,
        setup_ns,
        replay_ns_samples: Vec::new(),
        replay_ns_best: 0,
        replay_ns_median: 0,
        mismatch_count,
        mismatches,
    }
}

fn empty_bench_threaded_trace_report(
    trace: &PyrtlLaneTrace,
    options: &BenchThreadedTraceOptions,
    import_ns: u128,
    setup_ns: u128,
    replay: ThreadedReplayReport,
    autotune: Option<ReplayAutotuneReport>,
    mismatches: Vec<TraceMismatch>,
    simd_suitability: SimdSuitabilityReport,
    backend_affinity: BackendAffinityReport,
    replay_workload: EncodedTraceReplayWorkload,
    selected_threaded_layout: ThreadedReplayOptions,
    selected_reason: String,
) -> BenchThreadedTraceReport {
    let mismatch_count = replay.replay.mismatch_count;
    let autotune_pruned_candidates = autotune
        .as_ref()
        .map(|report| report.pruned_candidates.clone())
        .unwrap_or_default();
    BenchThreadedTraceReport {
        schema: "rrtl-pyrtl-bench-threaded-trace-v1".to_string(),
        steps: trace.steps.len(),
        lanes: trace.lanes,
        repeat: options.repeat,
        warmup: options.warmup,
        import_ns,
        setup_ns,
        simd_suitability,
        backend_affinity,
        replay_workload,
        selected_threaded_layout,
        selected_reason,
        replay_ns_samples: Vec::new(),
        replay_ns_best: 0,
        replay_ns_median: 0,
        replay,
        autotune,
        autotune_pruned_candidates,
        mismatch_count,
        mismatches,
    }
}

fn backend_plan_candidate_set(
    lanes: usize,
    max_workers: usize,
    autotune_prune: bool,
    simd_suitability: &SimdSuitabilityReport,
    backend_affinity: &BackendAffinityReport,
) -> rrtl_sim_ir::ReplayAutotuneCandidateSet {
    build_replay_autotune_candidate_set_with_affinity(
        lanes,
        max_workers,
        if autotune_prune {
            Some(simd_suitability)
        } else {
            None
        },
        if autotune_prune {
            Some(backend_affinity)
        } else {
            None
        },
    )
}

fn hybrid_threaded_layout_candidates(
    lanes: usize,
    max_workers: usize,
    mut candidates: Vec<ThreadedReplayOptions>,
    simd_suitability: &SimdSuitabilityReport,
    backend_affinity: &BackendAffinityReport,
    replay_workload: &EncodedTraceReplayWorkload,
    calibration: Option<&PlannerCalibration>,
) -> Vec<ThreadedReplayOptions> {
    candidates.extend(feature_threaded_layout_candidates(
        lanes,
        max_workers,
        simd_suitability,
        backend_affinity,
        replay_workload,
    ));
    dedupe_threaded_layouts(&mut candidates);
    let preference = calibration_threaded_preferences(calibration);
    candidates.sort_by(|left, right| {
        threaded_layout_score(
            right,
            &preference,
            simd_suitability,
            backend_affinity,
            replay_workload,
        )
        .cmp(&threaded_layout_score(
            left,
            &preference,
            simd_suitability,
            backend_affinity,
            replay_workload,
        ))
        .then_with(|| threaded_layout_signature(left).cmp(&threaded_layout_signature(right)))
    });
    candidates
}

fn select_threaded_layout(
    lanes: usize,
    candidates: Vec<ThreadedReplayOptions>,
    calibration: Option<&PlannerCalibration>,
) -> (ThreadedReplayOptions, String) {
    let preference = calibration_threaded_preferences(calibration);
    if let Some(layout) = candidates.into_iter().next() {
        let signature = threaded_layout_signature(&layout);
        let reason = if preference.contains_key(&signature) {
            "calibrated-hybrid-threaded-layout"
        } else if signature.is_empty() {
            "scalar-fallback-no-candidates"
        } else if layout
            .workers
            .iter()
            .any(|worker| worker.backend == SimBackendKind::Scalar)
            && layout
                .workers
                .iter()
                .any(|worker| worker.backend == SimBackendKind::SimdCpu)
        {
            "hybrid-scalar-simd-layout"
        } else if layout
            .workers
            .iter()
            .any(|worker| worker.backend == SimBackendKind::PackedCpu)
            && layout
                .workers
                .iter()
                .any(|worker| worker.backend == SimBackendKind::SimdCpu)
        {
            "hybrid-packed-simd-layout"
        } else {
            "hybrid-threaded-layout"
        };
        (layout, reason.to_string())
    } else {
        (
            ThreadedReplayOptions {
                workers: vec![ThreadedReplayWorkerOptions {
                    backend: SimBackendKind::Scalar,
                    lanes,
                }],
                max_mismatches: 16,
            },
            "scalar-fallback-no-candidates".to_string(),
        )
    }
}

fn feature_threaded_layout_candidates(
    lanes: usize,
    max_workers: usize,
    simd_suitability: &SimdSuitabilityReport,
    backend_affinity: &BackendAffinityReport,
    replay_workload: &EncodedTraceReplayWorkload,
) -> Vec<ThreadedReplayOptions> {
    let mut layouts = Vec::new();
    let max_workers = max_workers.max(1).min(lanes.max(1));
    let simd_coverage = if simd_suitability.total.instr_count == 0 {
        0
    } else {
        simd_suitability.total.fast_instrs * 100 / simd_suitability.total.instr_count
    };
    if replay_workload.estimated_lane_work_units < 4096 || lanes <= 2 || max_workers == 1 {
        layouts.push(single_backend_layout(SimBackendKind::Scalar, lanes));
    }
    if simd_coverage >= 70
        || matches!(
            simd_suitability.recommendation,
            SimdSuitabilityRecommendation::SimdCandidate
        )
    {
        layouts.push(single_backend_layout(SimBackendKind::SimdCpu, lanes));
    }
    if max_workers >= 2
        && (matches!(
            backend_affinity.recommendation,
            BackendAffinityRecommendation::MixedScalarSimdCandidate
        ) || simd_suitability.fallback_ratio_x100 >= 25)
    {
        layouts.push(split_backend_layout(
            SimBackendKind::Scalar,
            SimBackendKind::SimdCpu,
            lanes,
        ));
    }
    if max_workers >= 2 && simd_coverage >= 50 && simd_suitability.fallback_ratio_x100 <= 25 {
        layouts.push(split_backend_layout(
            SimBackendKind::PackedCpu,
            SimBackendKind::SimdCpu,
            lanes,
        ));
    }
    layouts.push(single_backend_layout(SimBackendKind::Scalar, lanes));
    layouts
}

fn single_backend_layout(backend: SimBackendKind, lanes: usize) -> ThreadedReplayOptions {
    ThreadedReplayOptions {
        workers: vec![ThreadedReplayWorkerOptions { backend, lanes }],
        max_mismatches: 16,
    }
}

fn split_backend_layout(
    first: SimBackendKind,
    second: SimBackendKind,
    lanes: usize,
) -> ThreadedReplayOptions {
    if lanes <= 1 {
        return single_backend_layout(second, lanes);
    }
    let first_lanes = (lanes / 4).max(1);
    let second_lanes = lanes.saturating_sub(first_lanes).max(1);
    ThreadedReplayOptions {
        workers: vec![
            ThreadedReplayWorkerOptions {
                backend: first,
                lanes: first_lanes,
            },
            ThreadedReplayWorkerOptions {
                backend: second,
                lanes: second_lanes,
            },
        ],
        max_mismatches: 16,
    }
}

fn dedupe_threaded_layouts(candidates: &mut Vec<ThreadedReplayOptions>) {
    let mut seen = BTreeMap::new();
    candidates.retain(|candidate| {
        let signature = threaded_layout_signature(candidate);
        if seen.contains_key(&signature) {
            false
        } else {
            seen.insert(signature, ());
            true
        }
    });
}

fn threaded_layout_score(
    layout: &ThreadedReplayOptions,
    preference: &BTreeMap<String, f64>,
    simd_suitability: &SimdSuitabilityReport,
    backend_affinity: &BackendAffinityReport,
    replay_workload: &EncodedTraceReplayWorkload,
) -> isize {
    let total_lanes = layout
        .workers
        .iter()
        .map(|worker| worker.lanes)
        .sum::<usize>()
        .max(1);
    let mut score = 0isize;
    for worker in &layout.workers {
        let backend_score = match worker.backend {
            SimBackendKind::Scalar => scalar_layout_score(backend_affinity, simd_suitability),
            SimBackendKind::PackedCpu => packed_layout_score(backend_affinity, simd_suitability),
            SimBackendKind::SimdCpu => simd_layout_score(simd_suitability),
            SimBackendKind::JitCpu => 40,
        };
        score += backend_score * worker.lanes as isize / total_lanes as isize;
    }
    score -= (layout.workers.len().saturating_sub(1) * 8) as isize;
    if replay_workload.estimated_lane_work_units < 4096 && layout.workers.len() > 1 {
        score -= 80;
    }
    score += (calibration_threaded_score(preference, layout) * 10.0)
        .round()
        .clamp(0.0, 300.0) as isize;
    score
}

fn scalar_layout_score(
    backend_affinity: &BackendAffinityReport,
    simd_suitability: &SimdSuitabilityReport,
) -> isize {
    let mut score = 80;
    if matches!(
        backend_affinity.recommendation,
        BackendAffinityRecommendation::ScalarPreferred
    ) {
        score += 80;
    }
    score + (simd_suitability.fallback_ratio_x100 / 2) as isize
}

fn packed_layout_score(
    backend_affinity: &BackendAffinityReport,
    simd_suitability: &SimdSuitabilityReport,
) -> isize {
    let mut score = 90;
    if matches!(
        backend_affinity.recommendation,
        BackendAffinityRecommendation::PackedCpuCandidate
            | BackendAffinityRecommendation::MixedScalarSimdCandidate
    ) {
        score += 50;
    }
    score - (simd_suitability.fallback_ratio_x100 / 3) as isize
}

fn simd_layout_score(simd_suitability: &SimdSuitabilityReport) -> isize {
    let coverage = if simd_suitability.total.instr_count == 0 {
        0
    } else {
        simd_suitability.total.fast_instrs * 100 / simd_suitability.total.instr_count
    };
    let mut score = 80 + coverage as isize;
    if matches!(
        simd_suitability.recommendation,
        SimdSuitabilityRecommendation::SimdCandidate
    ) {
        score += 60;
    }
    score - (simd_suitability.fallback_ratio_x100 / 2) as isize
}

fn calibration_threaded_preferences(
    calibration: Option<&PlannerCalibration>,
) -> BTreeMap<String, f64> {
    calibration
        .filter(|calibration| calibration.schema == "rrtl-pyrtl-planner-calibration-v1")
        .map(|calibration| {
            calibration
                .summary
                .threaded_layout_preferences
                .iter()
                .map(|preference| (preference.signature.clone(), preference.score))
                .collect()
        })
        .unwrap_or_default()
}

fn calibration_gpu_preferences(calibration: Option<&PlannerCalibration>) -> BTreeMap<String, f64> {
    calibration
        .filter(|calibration| calibration.schema == "rrtl-pyrtl-planner-calibration-v1")
        .map(|calibration| {
            calibration
                .summary
                .gpu_option_preferences
                .iter()
                .map(|preference| (preference.signature.clone(), preference.score))
                .collect()
        })
        .unwrap_or_default()
}

fn calibration_profitability_backend_preferences(
    calibration: Option<&PlannerCalibration>,
) -> BTreeMap<String, f64> {
    calibration
        .filter(|calibration| calibration.schema == "rrtl-pyrtl-planner-calibration-v1")
        .map(|calibration| {
            calibration
                .summary
                .profitability_backend_preferences
                .iter()
                .map(|preference| (preference.signature.clone(), preference.score))
                .collect()
        })
        .unwrap_or_default()
}

fn calibration_profitability_penalties(
    calibration: Option<&PlannerCalibration>,
) -> BTreeMap<String, f64> {
    calibration
        .filter(|calibration| calibration.schema == "rrtl-pyrtl-planner-calibration-v1")
        .map(|calibration| {
            calibration
                .summary
                .profitability_penalties
                .iter()
                .map(|preference| (preference.signature.clone(), preference.score))
                .collect()
        })
        .unwrap_or_default()
}

fn calibration_profitability_feature_preferences(
    calibration: Option<&PlannerCalibration>,
) -> BTreeMap<String, f64> {
    calibration
        .filter(|calibration| calibration.schema == "rrtl-pyrtl-planner-calibration-v1")
        .map(|calibration| {
            calibration
                .summary
                .profitability_feature_preferences
                .iter()
                .map(|preference| (preference.signature.clone(), preference.score))
                .collect()
        })
        .unwrap_or_default()
}

fn calibration_profitability_feature_penalties(
    calibration: Option<&PlannerCalibration>,
) -> BTreeMap<String, f64> {
    calibration
        .filter(|calibration| calibration.schema == "rrtl-pyrtl-planner-calibration-v1")
        .map(|calibration| {
            calibration
                .summary
                .profitability_feature_penalties
                .iter()
                .map(|preference| (preference.signature.clone(), preference.score))
                .collect()
        })
        .unwrap_or_default()
}

fn calibration_hot_backend_preference(
    calibration: Option<&PlannerCalibration>,
) -> Option<PlannerCalibrationPreference> {
    calibration
        .filter(|calibration| calibration.schema == "rrtl-pyrtl-planner-calibration-v1")
        .and_then(|calibration| {
            calibration
                .summary
                .hot_backend_preferences
                .iter()
                .filter(|preference| preference.score > 0.0)
                .max_by(|left, right| {
                    left.score
                        .total_cmp(&right.score)
                        .then_with(|| right.signature.cmp(&left.signature))
                })
                .cloned()
        })
}

fn hot_backend_calibration_reason(
    preference: Option<&PlannerCalibrationPreference>,
    gpu_selected: bool,
    gpu_region_analysis: &GpuRegionAnalysis,
) -> Option<String> {
    let preference = preference?;
    if is_hot_gpu_backend(&preference.signature) {
        return Some(
            if gpu_selected {
                "hot-backend-gpu-preferred"
            } else if gpu_backend_blocked(gpu_region_analysis) {
                "hot-backend-gpu-blocked"
            } else {
                "hot-backend-gpu-unselected"
            }
            .to_string(),
        );
    }
    if is_hot_threaded_backend(&preference.signature) {
        Some("hot-backend-threaded-preferred".to_string())
    } else {
        Some("hot-backend-direct-backend-preferred".to_string())
    }
}

fn is_hot_gpu_backend(signature: &str) -> bool {
    signature == "rrtl_gpu_measured_trace" || signature == "rrtl_gpu_trace"
}

fn is_hot_threaded_backend(signature: &str) -> bool {
    signature == "rrtl_threaded_autotune_trace" || signature == "rrtl_threaded_trace"
}

fn gpu_backend_blocked(analysis: &GpuRegionAnalysis) -> bool {
    matches!(
        analysis.recommendation,
        GpuRegionRecommendation::MemoryBlocked | GpuRegionRecommendation::TooSmall
    )
}

fn hot_backend_prefers_cpu(preference: Option<&PlannerCalibrationPreference>) -> bool {
    preference.is_some_and(|preference| !is_hot_gpu_backend(&preference.signature))
}

fn calibration_threaded_score(
    preference: &BTreeMap<String, f64>,
    layout: &ThreadedReplayOptions,
) -> f64 {
    preference
        .get(&threaded_layout_signature(layout))
        .copied()
        .unwrap_or(0.0)
}

fn calibration_gpu_score(preference: &BTreeMap<String, f64>, candidate: &PlannedGpuOption) -> f64 {
    preference
        .get(&gpu_option_signature(candidate.options))
        .copied()
        .unwrap_or(0.0)
}

fn threaded_layout_signature(layout: &ThreadedReplayOptions) -> String {
    layout
        .workers
        .iter()
        .map(|worker| format!("{}:{}", worker.backend.as_str(), worker.lanes))
        .collect::<Vec<_>>()
        .join(",")
}

fn gpu_option_signature(options: GpuBatchOptions) -> String {
    format!(
        "workgroup={},memory={},reuse={}",
        options.workgroup_size,
        gpu_memory_layout_signature(options.memory_layout),
        options.reuse_temporaries
    )
}

fn gpu_memory_layout_signature(layout: GpuMemoryLayout) -> &'static str {
    match layout {
        GpuMemoryLayout::LaneMajor => "lane-major",
        GpuMemoryLayout::WordMajor => "word-major",
    }
}

struct GpuOptionPlan {
    selected_gpu_options: Option<GpuBatchOptions>,
    selected_gpu_reason: Option<String>,
    calibration_reason: Option<String>,
    recommended_gpu_options: Vec<PlannedGpuOption>,
    pruned_gpu_options: Vec<PlannedGpuOption>,
}

fn plan_gpu_options(
    program: &PackedProgram,
    analysis: &GpuRegionAnalysis,
    calibration: Option<&PlannerCalibration>,
    hot_backend_preference: Option<&PlannerCalibrationPreference>,
) -> Result<GpuOptionPlan, ImportError> {
    let candidates = gpu_option_candidates(program)?;
    let reason = match analysis.recommendation {
        GpuRegionRecommendation::ComputeCandidate => "gpu-compute-candidate",
        GpuRegionRecommendation::MixedCandidate => "gpu-mixed-candidate",
        GpuRegionRecommendation::MemoryBlocked => {
            return Ok(GpuOptionPlan {
                selected_gpu_options: None,
                selected_gpu_reason: Some("gpu-memory-blocked".to_string()),
                calibration_reason: None,
                recommended_gpu_options: Vec::new(),
                pruned_gpu_options: candidates
                    .into_iter()
                    .map(|mut candidate| {
                        candidate.reason = "gpu-memory-blocked".to_string();
                        candidate
                    })
                    .collect(),
            });
        }
        GpuRegionRecommendation::TooSmall => {
            return Ok(GpuOptionPlan {
                selected_gpu_options: None,
                selected_gpu_reason: Some("gpu-too-small".to_string()),
                calibration_reason: None,
                recommended_gpu_options: Vec::new(),
                pruned_gpu_options: candidates
                    .into_iter()
                    .map(|mut candidate| {
                        candidate.reason = "gpu-too-small".to_string();
                        candidate
                    })
                    .collect(),
            });
        }
    };

    let mut recommended_gpu_options = candidates;
    let preference = calibration_gpu_preferences(calibration);
    recommended_gpu_options.sort_by(|left, right| {
        calibration_gpu_score(&preference, right)
            .total_cmp(&calibration_gpu_score(&preference, left))
            .then_with(|| gpu_option_score(left).cmp(&gpu_option_score(right)))
    });
    if hot_backend_prefers_cpu(hot_backend_preference) {
        return Ok(GpuOptionPlan {
            selected_gpu_options: None,
            selected_gpu_reason: Some("hot-backend-prefers-cpu".to_string()),
            calibration_reason: None,
            recommended_gpu_options,
            pruned_gpu_options: Vec::new(),
        });
    }
    let selected_gpu_options = recommended_gpu_options
        .first()
        .map(|candidate| candidate.options);
    let calibration_reason = selected_gpu_options
        .filter(|options| preference.contains_key(&gpu_option_signature(*options)))
        .map(|_| "gpu-option-calibration-applied".to_string());
    Ok(GpuOptionPlan {
        selected_gpu_options,
        selected_gpu_reason: Some(reason.to_string()),
        calibration_reason,
        recommended_gpu_options,
        pruned_gpu_options: Vec::new(),
    })
}

fn gpu_option_candidates(program: &PackedProgram) -> Result<Vec<PlannedGpuOption>, ImportError> {
    let options = [
        (
            "lane-major-workgroup-64",
            GpuBatchOptions {
                workgroup_size: 64,
                memory_layout: GpuMemoryLayout::LaneMajor,
                ..GpuBatchOptions::default()
            },
        ),
        (
            "lane-major-workgroup-128",
            GpuBatchOptions {
                workgroup_size: 128,
                memory_layout: GpuMemoryLayout::LaneMajor,
                ..GpuBatchOptions::default()
            },
        ),
        (
            "lane-major-workgroup-256",
            GpuBatchOptions {
                workgroup_size: 256,
                memory_layout: GpuMemoryLayout::LaneMajor,
                ..GpuBatchOptions::default()
            },
        ),
        (
            "word-major-workgroup-128",
            GpuBatchOptions {
                workgroup_size: 128,
                memory_layout: GpuMemoryLayout::WordMajor,
                ..GpuBatchOptions::default()
            },
        ),
        (
            "reusable-temporaries-lane-major-workgroup-128",
            GpuBatchOptions {
                workgroup_size: 128,
                memory_layout: GpuMemoryLayout::LaneMajor,
                reuse_temporaries: true,
                ..GpuBatchOptions::default()
            },
        ),
    ];
    options
        .into_iter()
        .map(|(reason, options)| {
            let shader_stats = bench_gpu_shader_stats(
                gpu_shader_stats(program, options)
                    .map_err(|err| ImportError::new(format!("{err}")))?,
            );
            Ok(PlannedGpuOption {
                options,
                shader_stats,
                reason: reason.to_string(),
            })
        })
        .collect()
}

fn gpu_option_score(candidate: &PlannedGpuOption) -> (usize, usize, usize, usize) {
    let stats = &candidate.shader_stats;
    let memory_ops = stats.optimized_memory_reads + stats.optimized_memory_writes;
    (
        stats.optimized_packets_total,
        memory_ops,
        stats.wgsl_bytes,
        usize::from(!is_default_like_gpu_option(candidate.options)),
    )
}

fn selected_runtime_candidate(
    candidates: &[BackendProfitabilityCandidate],
) -> Option<&BackendProfitabilityCandidate> {
    candidates.iter().min_by_key(|candidate| candidate.rank)
}

fn selected_runtime_reason(candidate: Option<&BackendProfitabilityCandidate>) -> String {
    candidate
        .and_then(|candidate| candidate.reasons.first())
        .cloned()
        .unwrap_or_else(|| "no-candidates".to_string())
}

fn direct_backend_kind_for_profitability(backend: &str) -> Option<SimBackendKind> {
    match backend {
        "scalar" => Some(SimBackendKind::Scalar),
        "packed-cpu" => Some(SimBackendKind::PackedCpu),
        "simd-cpu" => Some(SimBackendKind::SimdCpu),
        _ => None,
    }
}

fn select_threaded_layout_from_profitability(
    lanes: usize,
    max_workers: usize,
    candidates: &[BackendProfitabilityCandidate],
    planned_candidates: Vec<ThreadedReplayOptions>,
    simd_suitability: &SimdSuitabilityReport,
    backend_affinity: &BackendAffinityReport,
    replay_workload: &EncodedTraceReplayWorkload,
    calibration: Option<&PlannerCalibration>,
) -> (ThreadedReplayOptions, String) {
    let selected = selected_runtime_candidate(candidates);
    if selected.is_some_and(|candidate| candidate.backend == "threaded-mixed") {
        let recommended = hybrid_threaded_layout_candidates(
            lanes,
            max_workers,
            planned_candidates,
            simd_suitability,
            backend_affinity,
            replay_workload,
            calibration,
        );
        let (layout, reason) = select_threaded_layout(lanes, recommended, calibration);
        return (layout, format!("profitability-threaded-mixed-{reason}"));
    }
    if let Some(kind) =
        selected.and_then(|candidate| direct_backend_kind_for_profitability(&candidate.backend))
    {
        return (
            ThreadedReplayOptions {
                workers: vec![ThreadedReplayWorkerOptions {
                    backend: kind,
                    lanes,
                }],
                max_mismatches: 16,
            },
            format!(
                "profitability-direct-{}",
                selected
                    .map(|candidate| candidate.backend.as_str())
                    .unwrap_or("scalar")
            ),
        );
    }
    let cpu_candidate = candidates.iter().find_map(|candidate| {
        direct_backend_kind_for_profitability(&candidate.backend).map(|kind| (candidate, kind))
    });
    if let Some((candidate, kind)) = cpu_candidate {
        return (
            ThreadedReplayOptions {
                workers: vec![ThreadedReplayWorkerOptions {
                    backend: kind,
                    lanes,
                }],
                max_mismatches: 16,
            },
            format!("profitability-{}-cpu-fallback", candidate.backend),
        );
    }
    let recommended = hybrid_threaded_layout_candidates(
        lanes,
        max_workers,
        planned_candidates,
        simd_suitability,
        backend_affinity,
        replay_workload,
        calibration,
    );
    select_threaded_layout(lanes, recommended, calibration)
}

fn profitability_allows_gpu(candidates: &[BackendProfitabilityCandidate]) -> bool {
    let Some(gpu) = candidates
        .iter()
        .find(|candidate| candidate.backend == "gpu-fused")
    else {
        return false;
    };
    if gpu
        .reasons
        .iter()
        .any(|reason| reason == "gpu-launch-not-amortized" || reason == "memory-hostile")
    {
        return false;
    }
    gpu.rank == 1 || gpu.score >= 180
}

fn plan_backend_profitability(
    lanes: usize,
    steps: usize,
    simd_suitability: &SimdSuitabilityReport,
    backend_affinity: &BackendAffinityReport,
    gpu_region_analysis: &GpuRegionAnalysis,
    replay_workload: &EncodedTraceReplayWorkload,
    gpu_selected_by_region: bool,
    gpu_trace_compression: Option<&GpuTraceCompressionEstimate>,
    calibration: Option<&PlannerCalibration>,
) -> (
    BackendProfitabilityReport,
    Vec<BackendProfitabilityCandidate>,
    BackendProfitabilityFeatures,
) {
    let op_profile =
        backend_profitability_op_profile(simd_suitability, gpu_region_analysis, replay_workload);
    let fast_ops = simd_suitability.total.fast_instrs;
    let simd_coverage_score_x100 = if simd_suitability.total.instr_count == 0 {
        0
    } else {
        fast_ops * 100 / simd_suitability.total.instr_count
    };
    let native_simd_ops = simd_suitability.total.fast_path_profile.one_limb_ops
        + op_profile.native_two_limb_ops
        + simd_suitability.total.fast_path_profile.two_limb_mux_ops
        + simd_suitability
            .total
            .fast_path_profile
            .two_limb_memory_reads;
    let native_simd_score_x100 = if fast_ops == 0 {
        0
    } else {
        native_simd_ops * 100 / fast_ops
    };
    let gpu_suitability_score_x100 = gpu_profitability_score_x100(
        lanes,
        steps,
        gpu_region_analysis,
        replay_workload.estimated_lane_work_units,
        gpu_selected_by_region,
    );
    let threading_score_x100 =
        threading_profitability_score_x100(lanes, steps, replay_workload, backend_affinity);
    let profitability_features = backend_profitability_features(
        lanes,
        steps,
        simd_suitability,
        &op_profile,
        simd_coverage_score_x100,
        native_simd_score_x100,
        gpu_suitability_score_x100,
        threading_score_x100,
        gpu_trace_compression,
    );
    let lane_steps = lanes.saturating_mul(steps).max(1);
    let mut candidates = vec![
        BackendProfitabilityCandidate {
            backend: "scalar".to_string(),
            rank: 0,
            score: scalar_profitability_score(backend_affinity),
            estimated_setup_cost: 1,
            estimated_per_lane_step_cost: per_lane_step_cost(
                backend_affinity.total.estimated_scalar_cost,
                lane_steps,
                20,
            ),
            reasons: scalar_profitability_reasons(backend_affinity),
        },
        BackendProfitabilityCandidate {
            backend: "packed-cpu".to_string(),
            rank: 0,
            score: packed_profitability_score(backend_affinity),
            estimated_setup_cost: 3,
            estimated_per_lane_step_cost: per_lane_step_cost(
                backend_affinity.total.estimated_packed_cpu_cost,
                lane_steps,
                10,
            ),
            reasons: packed_profitability_reasons(backend_affinity),
        },
        BackendProfitabilityCandidate {
            backend: "simd-cpu".to_string(),
            rank: 0,
            score: simd_profitability_score(
                simd_suitability,
                simd_coverage_score_x100,
                native_simd_score_x100,
            ),
            estimated_setup_cost: 5,
            estimated_per_lane_step_cost: per_lane_step_cost(
                backend_affinity.total.estimated_simd_cpu_cost,
                lane_steps,
                5,
            ),
            reasons: simd_profitability_reasons(
                simd_suitability,
                simd_coverage_score_x100,
                native_simd_score_x100,
            ),
        },
        BackendProfitabilityCandidate {
            backend: "threaded-mixed".to_string(),
            rank: 0,
            score: threaded_profitability_score(
                threading_score_x100,
                lanes,
                steps,
                backend_affinity,
            ),
            estimated_setup_cost: 12,
            estimated_per_lane_step_cost: per_lane_step_cost(
                backend_affinity
                    .total
                    .estimated_simd_cpu_cost
                    .min(backend_affinity.total.estimated_packed_cpu_cost),
                lane_steps,
                4,
            ),
            reasons: threaded_profitability_reasons(
                threading_score_x100,
                lanes,
                steps,
                backend_affinity,
            ),
        },
        BackendProfitabilityCandidate {
            backend: "gpu-fused".to_string(),
            rank: 0,
            score: gpu_profitability_score(
                lanes,
                steps,
                gpu_suitability_score_x100,
                gpu_region_analysis,
                gpu_trace_compression,
            ),
            estimated_setup_cost: 80,
            estimated_per_lane_step_cost: per_lane_step_cost(
                gpu_region_analysis.total.estimated_launch_work_units,
                lane_steps,
                2,
            ),
            reasons: gpu_profitability_reasons(
                gpu_suitability_score_x100,
                gpu_region_analysis,
                replay_workload.estimated_lane_work_units,
            ),
        },
    ];
    apply_profitability_calibration(&mut candidates, &profitability_features, calibration);
    candidates.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.estimated_setup_cost.cmp(&right.estimated_setup_cost))
            .then_with(|| left.backend.cmp(&right.backend))
    });
    for (index, candidate) in candidates.iter_mut().enumerate() {
        candidate.rank = index + 1;
    }
    let winner = candidates
        .first()
        .cloned()
        .unwrap_or(BackendProfitabilityCandidate {
            backend: "scalar".to_string(),
            rank: 1,
            score: 0,
            estimated_setup_cost: 1,
            estimated_per_lane_step_cost: 1,
            reasons: vec!["no-candidates".to_string()],
        });
    (
        BackendProfitabilityReport {
            op_profile,
            simd_coverage_score_x100,
            gpu_suitability_score_x100,
            threading_score_x100,
            recommended_backend: winner.backend.clone(),
            recommended_reason: winner
                .reasons
                .first()
                .cloned()
                .unwrap_or_else(|| "highest-static-score".to_string()),
        },
        candidates,
        profitability_features,
    )
}

fn apply_profitability_calibration(
    candidates: &mut [BackendProfitabilityCandidate],
    features: &BackendProfitabilityFeatures,
    calibration: Option<&PlannerCalibration>,
) {
    let backend_preferences = calibration_profitability_backend_preferences(calibration);
    let penalties = calibration_profitability_penalties(calibration);
    let feature_preferences = calibration_profitability_feature_preferences(calibration);
    let feature_penalties = calibration_profitability_feature_penalties(calibration);
    if backend_preferences.is_empty()
        && penalties.is_empty()
        && feature_preferences.is_empty()
        && feature_penalties.is_empty()
    {
        return;
    }
    for candidate in candidates {
        if let Some(score) = backend_preferences
            .get(&candidate.backend)
            .filter(|_| !profitability_gpu_hard_blocked(candidate))
        {
            candidate.score += calibrated_score_delta(*score);
            candidate
                .reasons
                .push(format!("profitability-calibrated-backend:{:.2}", score));
        }
        for reason in &candidate.reasons.clone() {
            if let Some(penalty) = penalties.get(reason) {
                candidate.score -= calibrated_score_delta(*penalty).abs();
                candidate.reasons.push(format!(
                    "profitability-calibrated-penalty:{reason}:{penalty:.2}"
                ));
            }
        }
        for feature in &features.feature_buckets {
            let signature = profitability_feature_signature(&candidate.backend, feature);
            if let Some(score) = feature_preferences
                .get(&signature)
                .filter(|_| !profitability_gpu_hard_blocked(candidate))
            {
                candidate.score += calibrated_feature_score_delta(*score);
                candidate.reasons.push(format!(
                    "profitability-calibrated-feature:{feature}:{score:.2}"
                ));
            }
            if let Some(penalty) = feature_penalties.get(&signature) {
                candidate.score -= calibrated_feature_score_delta(*penalty).abs();
                candidate.reasons.push(format!(
                    "profitability-calibrated-feature-penalty:{feature}:{penalty:.2}"
                ));
            }
        }
        if profitability_gpu_hard_blocked(candidate) {
            candidate.score = candidate.score.min(0);
        }
    }
}

fn profitability_gpu_hard_blocked(candidate: &BackendProfitabilityCandidate) -> bool {
    candidate.backend == "gpu-fused"
        && candidate
            .reasons
            .iter()
            .any(|reason| reason == "memory-hostile" || reason == "gpu-launch-not-amortized")
}

fn calibrated_score_delta(score: f64) -> isize {
    if !score.is_finite() || score <= 0.0 {
        0
    } else {
        (score * 10.0).round().clamp(0.0, 250.0) as isize
    }
}

fn calibrated_feature_score_delta(score: f64) -> isize {
    if !score.is_finite() || score <= 0.0 {
        0
    } else {
        (score * 4.0).round().clamp(0.0, 80.0) as isize
    }
}

fn profitability_feature_signature(backend: &str, feature: &str) -> String {
    format!("{backend}|{feature}")
}

fn backend_profitability_op_profile(
    simd_suitability: &SimdSuitabilityReport,
    gpu_region_analysis: &GpuRegionAnalysis,
    replay_workload: &EncodedTraceReplayWorkload,
) -> BackendProfitabilityOpProfile {
    let fast_profile = simd_suitability.total.fast_path_profile;
    let native_two_limb_ops = native_two_limb_static_ops(fast_profile);
    BackendProfitabilityOpProfile {
        instr_count: simd_suitability.total.instr_count,
        one_limb_ops: fast_profile.one_limb_ops,
        two_limb_ops: fast_profile.two_limb_ops
            + fast_profile.two_limb_memory_reads
            + fast_profile.two_limb_mux_ops,
        native_two_limb_ops,
        two_limb_mul_ops: fast_profile.two_limb_mul_ops,
        memory_ops: simd_suitability.total.memory_read_instrs
            + simd_suitability.total.memory_write_effects,
        wide_fallback_ops: simd_suitability.total.wide_instrs,
        pure_compute_packets: gpu_region_analysis.total.pure_compute_packets,
        memory_hostile_packets: gpu_region_analysis.total.memory_hostile_packets,
        estimated_lane_work_units: replay_workload.estimated_lane_work_units,
    }
}

fn backend_profitability_features(
    lanes: usize,
    steps: usize,
    simd_suitability: &SimdSuitabilityReport,
    op_profile: &BackendProfitabilityOpProfile,
    simd_coverage_score_x100: usize,
    native_simd_score_x100: usize,
    gpu_suitability_score_x100: usize,
    threading_score_x100: usize,
    gpu_trace_compression: Option<&GpuTraceCompressionEstimate>,
) -> BackendProfitabilityFeatures {
    let lane_steps = lanes.saturating_mul(steps);
    let memory_op_ratio_x100 = if op_profile.instr_count == 0 {
        0
    } else {
        op_profile.memory_ops * 100 / op_profile.instr_count
    };
    let mut features = BackendProfitabilityFeatures {
        lanes,
        steps,
        lane_steps,
        gpu_lane_steps: lane_steps,
        estimated_lane_work_units: op_profile.estimated_lane_work_units,
        instr_count: op_profile.instr_count,
        simd_coverage_score_x100,
        native_simd_score_x100,
        fallback_ratio_x100: simd_suitability.fallback_ratio_x100,
        memory_op_ratio_x100,
        pure_compute_packets: op_profile.pure_compute_packets,
        memory_hostile_packets: op_profile.memory_hostile_packets,
        wide_fallback_ops: op_profile.wide_fallback_ops,
        gpu_suitability_score_x100,
        threading_score_x100,
        gpu_single_submit_available: gpu_suitability_score_x100 >= 70
            && lane_steps >= 4096
            && op_profile.memory_hostile_packets == 0,
        gpu_count_readback_only: op_profile.memory_hostile_packets == 0,
        gpu_full_readback_penalty: op_profile.memory_hostile_packets,
        gpu_prepared_trace_bytes: gpu_trace_compression
            .map(|estimate| estimate.compressed_bytes)
            .unwrap_or_else(|| {
                lane_steps.saturating_mul(
                    op_profile
                        .memory_ops
                        .saturating_add(op_profile.instr_count)
                        .max(1),
                )
            }),
        gpu_trace_uncompressed_bytes: gpu_trace_compression
            .map(|estimate| estimate.uncompressed_bytes)
            .unwrap_or(0),
        gpu_trace_compression_ratio_x100: gpu_trace_compression
            .map(|estimate| estimate.compression_ratio_x100)
            .unwrap_or(100),
        gpu_trace_uniform_input_ops: gpu_trace_compression
            .map(|estimate| estimate.uniform_input_ops)
            .unwrap_or(0),
        gpu_trace_uniform_check_ops: gpu_trace_compression
            .map(|estimate| estimate.uniform_check_ops)
            .unwrap_or(0),
        gpu_trace_template_layout: gpu_trace_compression
            .map(|estimate| estimate.template_layout)
            .unwrap_or(false),
        gpu_trace_template_input_ops: gpu_trace_compression
            .map(|estimate| estimate.template_input_ops)
            .unwrap_or(0),
        gpu_trace_template_check_ops: gpu_trace_compression
            .map(|estimate| estimate.template_check_ops)
            .unwrap_or(0),
        gpu_trace_metadata_saved_words: gpu_trace_compression
            .map(|estimate| estimate.metadata_saved_words)
            .unwrap_or(0),
        gpu_trace_fixed_template: gpu_trace_compression
            .map(|estimate| estimate.fixed_template)
            .unwrap_or(false),
        gpu_trace_value_metadata_saved_words: gpu_trace_compression
            .map(|estimate| estimate.value_metadata_saved_words)
            .unwrap_or(0),
        gpu_trace_value_stride_words: gpu_trace_compression
            .map(|estimate| estimate.value_stride_words)
            .unwrap_or(0),
        feature_buckets: Vec::new(),
    };
    features.feature_buckets = profitability_feature_buckets(&features);
    features
}

fn profitability_feature_buckets(features: &BackendProfitabilityFeatures) -> Vec<String> {
    let mut buckets = Vec::new();
    push_threshold_bucket(&mut buckets, "lanes", features.lanes, &[4, 16, 64, 256]);
    push_threshold_bucket(&mut buckets, "steps", features.steps, &[16, 64, 256, 1024]);
    push_threshold_bucket(
        &mut buckets,
        "lane_steps",
        features.lane_steps,
        &[256, 4096, 65_536, 1_048_576],
    );
    push_threshold_bucket(
        &mut buckets,
        "lane_work",
        features.estimated_lane_work_units,
        &[4096, 65_536, 1_048_576],
    );
    push_threshold_bucket(
        &mut buckets,
        "gpu_lane_steps",
        features.gpu_lane_steps,
        &[4096, 65_536, 1_048_576],
    );
    push_threshold_bucket(
        &mut buckets,
        "instr",
        features.instr_count,
        &[32, 128, 512, 2048],
    );
    push_threshold_bucket(
        &mut buckets,
        "simd_coverage",
        features.simd_coverage_score_x100,
        &[50, 70, 90],
    );
    push_threshold_bucket(
        &mut buckets,
        "native_simd",
        features.native_simd_score_x100,
        &[50, 70, 90],
    );
    push_threshold_bucket(
        &mut buckets,
        "fallback",
        features.fallback_ratio_x100,
        &[10, 25, 50],
    );
    push_threshold_bucket(
        &mut buckets,
        "memory_ops",
        features.memory_op_ratio_x100,
        &[10, 25, 50],
    );
    push_threshold_bucket(
        &mut buckets,
        "gpu_pure_packets",
        features.pure_compute_packets,
        &[4, 16, 64],
    );
    push_threshold_bucket(
        &mut buckets,
        "gpu_memory_hostile",
        features.memory_hostile_packets,
        &[1, 4, 16],
    );
    push_threshold_bucket(
        &mut buckets,
        "wide_fallback",
        features.wide_fallback_ops,
        &[1, 16, 64],
    );
    push_threshold_bucket(
        &mut buckets,
        "gpu_suitability",
        features.gpu_suitability_score_x100,
        &[50, 70, 90],
    );
    push_threshold_bucket(
        &mut buckets,
        "threading",
        features.threading_score_x100,
        &[50, 70, 90],
    );
    if features.gpu_single_submit_available {
        buckets.push("gpu_single_submit_available".to_string());
    }
    if features.gpu_count_readback_only {
        buckets.push("gpu_count_readback_only".to_string());
    }
    if features.gpu_full_readback_penalty > 0 {
        buckets.push("gpu_full_readback_penalty".to_string());
    }
    push_threshold_bucket(
        &mut buckets,
        "gpu_prepared_trace_bytes",
        features.gpu_prepared_trace_bytes,
        &[4096, 65_536, 1_048_576],
    );
    push_threshold_bucket(
        &mut buckets,
        "gpu_trace_uniform_input_ops",
        features.gpu_trace_uniform_input_ops,
        &[1, 16, 64],
    );
    push_threshold_bucket(
        &mut buckets,
        "gpu_trace_uniform_check_ops",
        features.gpu_trace_uniform_check_ops,
        &[1, 16, 64],
    );
    if features.gpu_trace_compression_ratio_x100 <= 75 {
        buckets.push("gpu_trace_compression<=75".to_string());
    }
    if features.gpu_trace_compression_ratio_x100 <= 50 {
        buckets.push("gpu_trace_compression<=50".to_string());
    }
    if features.gpu_trace_template_layout {
        buckets.push("gpu_trace_template_layout".to_string());
    }
    push_threshold_bucket(
        &mut buckets,
        "gpu_trace_metadata_saved_words",
        features.gpu_trace_metadata_saved_words,
        &[16, 128, 1024],
    );
    if features.gpu_trace_fixed_template {
        buckets.push("gpu_trace_fixed_template".to_string());
    }
    push_threshold_bucket(
        &mut buckets,
        "gpu_trace_value_metadata_saved_words",
        features.gpu_trace_value_metadata_saved_words,
        &[16, 128, 1024],
    );
    buckets
}

fn push_threshold_bucket(
    buckets: &mut Vec<String>,
    name: &str,
    value: usize,
    thresholds: &[usize],
) {
    for threshold in thresholds {
        if value >= *threshold {
            buckets.push(format!("{name}>={threshold}"));
        }
    }
}

fn native_two_limb_static_ops(profile: rrtl_sim_ir::SimdFastPathProfile) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        profile.two_limb_ops + profile.two_limb_memory_reads + profile.two_limb_mux_ops
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = profile;
        0
    }
}

fn per_lane_step_cost(cost: usize, lane_steps: usize, minimum: usize) -> usize {
    (cost / lane_steps).max(minimum)
}

fn scalar_profitability_score(backend_affinity: &BackendAffinityReport) -> isize {
    match backend_affinity.recommendation {
        BackendAffinityRecommendation::ScalarPreferred => 120,
        BackendAffinityRecommendation::PackedCpuCandidate => 60,
        BackendAffinityRecommendation::SimdCpuCandidate => 30,
        BackendAffinityRecommendation::MixedScalarSimdCandidate => 50,
        BackendAffinityRecommendation::GpuBlocked => 70,
    }
}

fn packed_profitability_score(backend_affinity: &BackendAffinityReport) -> isize {
    let base = match backend_affinity.recommendation {
        BackendAffinityRecommendation::PackedCpuCandidate => 110,
        BackendAffinityRecommendation::ScalarPreferred => 65,
        BackendAffinityRecommendation::SimdCpuCandidate => 70,
        BackendAffinityRecommendation::MixedScalarSimdCandidate => 80,
        BackendAffinityRecommendation::GpuBlocked => 85,
    };
    base + backend_affinity.total.max_packet_width.min(32) as isize
}

fn simd_profitability_score(
    simd_suitability: &SimdSuitabilityReport,
    simd_coverage_score_x100: usize,
    native_simd_score_x100: usize,
) -> isize {
    let base = match simd_suitability.recommendation {
        SimdSuitabilityRecommendation::SimdCandidate => 135,
        SimdSuitabilityRecommendation::MixedCandidate => 95,
        SimdSuitabilityRecommendation::ScalarPreferred => 35,
        SimdSuitabilityRecommendation::GpuCandidateBlocked => 90,
    };
    base + (simd_coverage_score_x100 / 2) as isize + (native_simd_score_x100 / 4) as isize
        - (simd_suitability.fallback_ratio_x100 / 2) as isize
}

fn threaded_profitability_score(
    threading_score_x100: usize,
    lanes: usize,
    steps: usize,
    backend_affinity: &BackendAffinityReport,
) -> isize {
    let base = match backend_affinity.recommendation {
        BackendAffinityRecommendation::MixedScalarSimdCandidate => 130,
        BackendAffinityRecommendation::SimdCpuCandidate => 100,
        BackendAffinityRecommendation::PackedCpuCandidate => 75,
        BackendAffinityRecommendation::ScalarPreferred => 30,
        BackendAffinityRecommendation::GpuBlocked => 90,
    };
    let setup_penalty = if lanes.saturating_mul(steps) < 256 {
        50
    } else {
        0
    };
    base + (threading_score_x100 / 2) as isize - setup_penalty
}

fn gpu_profitability_score(
    lanes: usize,
    steps: usize,
    gpu_suitability_score_x100: usize,
    gpu_region_analysis: &GpuRegionAnalysis,
    gpu_trace_compression: Option<&GpuTraceCompressionEstimate>,
) -> isize {
    let base = match gpu_region_analysis.recommendation {
        GpuRegionRecommendation::ComputeCandidate => 120,
        GpuRegionRecommendation::MixedCandidate => 75,
        GpuRegionRecommendation::MemoryBlocked => 10,
        GpuRegionRecommendation::TooSmall => 5,
    };
    let setup_penalty = if gpu_suitability_score_x100 < 50 {
        80
    } else {
        0
    };
    let lane_steps = lanes.saturating_mul(steps);
    let single_submit_bonus = if gpu_suitability_score_x100 >= 70
        && lane_steps >= 4096
        && gpu_region_analysis.total.memory_hostile_packets == 0
    {
        (lane_steps / 4096).min(80) as isize
    } else {
        0
    };
    let compression_bonus = gpu_trace_compression
        .filter(|_| gpu_region_analysis.total.memory_hostile_packets == 0)
        .map(|estimate| {
            if estimate.compression_ratio_x100 < 100 {
                ((100 - estimate.compression_ratio_x100) / 2).min(40) as isize
            } else {
                0
            }
        })
        .unwrap_or(0);
    let template_bonus = gpu_trace_compression
        .filter(|estimate| {
            estimate.template_layout && gpu_region_analysis.total.memory_hostile_packets == 0
        })
        .map(|estimate| (estimate.metadata_saved_words / 64).min(30) as isize)
        .unwrap_or(0);
    let fixed_template_bonus = gpu_trace_compression
        .filter(|estimate| {
            estimate.fixed_template && gpu_region_analysis.total.memory_hostile_packets == 0
        })
        .map(|estimate| (estimate.value_metadata_saved_words / 64).min(20) as isize)
        .unwrap_or(0);
    base + gpu_suitability_score_x100 as isize
        + single_submit_bonus
        + compression_bonus
        + template_bonus
        + fixed_template_bonus
        - (gpu_region_analysis.total.memory_hostile_packets * 8).min(120) as isize
        - setup_penalty
}

fn gpu_profitability_score_x100(
    lanes: usize,
    steps: usize,
    analysis: &GpuRegionAnalysis,
    lane_work_units: usize,
    gpu_selected_by_region: bool,
) -> usize {
    if !gpu_selected_by_region || gpu_backend_blocked(analysis) {
        return 0;
    }
    let lane_steps = lanes.saturating_mul(steps);
    if lane_steps < 512 || lane_work_units < 4096 {
        return 20;
    }
    let total_packets = analysis.total.packets.max(1);
    let compute_ratio = analysis.total.pure_compute_packets * 100 / total_packets;
    let launch_units = analysis.total.estimated_launch_work_units;
    let amortization = (lane_work_units / launch_units.max(1)).min(100);
    compute_ratio.saturating_add(amortization).min(100)
}

fn threading_profitability_score_x100(
    lanes: usize,
    steps: usize,
    replay_workload: &EncodedTraceReplayWorkload,
    backend_affinity: &BackendAffinityReport,
) -> usize {
    if lanes < 2 || steps == 0 {
        return 0;
    }
    let lane_steps = lanes.saturating_mul(steps);
    let work_score = (replay_workload.estimated_lane_work_units / 64).min(70);
    let parallel_score = (lane_steps / 8).min(30);
    let mixed_bonus = usize::from(matches!(
        backend_affinity.recommendation,
        BackendAffinityRecommendation::MixedScalarSimdCandidate
            | BackendAffinityRecommendation::SimdCpuCandidate
    )) * 20;
    (work_score + parallel_score + mixed_bonus).min(100)
}

fn scalar_profitability_reasons(backend_affinity: &BackendAffinityReport) -> Vec<String> {
    let mut reasons = Vec::new();
    if matches!(
        backend_affinity.recommendation,
        BackendAffinityRecommendation::ScalarPreferred
    ) {
        reasons.push("scalar-preferred".to_string());
    }
    if backend_affinity.total.wide_fallback_instrs > 0 {
        reasons.push("wide-fallback-pressure".to_string());
    }
    if reasons.is_empty() {
        reasons.push("low-setup-cost".to_string());
    }
    reasons
}

fn packed_profitability_reasons(backend_affinity: &BackendAffinityReport) -> Vec<String> {
    let mut reasons = Vec::new();
    if backend_affinity.total.max_packet_width > 1 {
        reasons.push("packet-parallelism".to_string());
    }
    if matches!(
        backend_affinity.recommendation,
        BackendAffinityRecommendation::PackedCpuCandidate
    ) {
        reasons.push("packed-cpu-candidate".to_string());
    }
    if reasons.is_empty() {
        reasons.push("moderate-setup-cost".to_string());
    }
    reasons
}

fn simd_profitability_reasons(
    simd_suitability: &SimdSuitabilityReport,
    simd_coverage_score_x100: usize,
    native_simd_score_x100: usize,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if simd_coverage_score_x100 >= 70 {
        reasons.push("high-simd-coverage".to_string());
    }
    if native_simd_score_x100 >= 50 {
        reasons.push("native-simd-coverage".to_string());
    }
    if simd_suitability.fallback_ratio_x100 > 25 {
        reasons.push("fallback-pressure".to_string());
    }
    if reasons.is_empty() {
        reasons.push("simd-static-score".to_string());
    }
    reasons
}

fn threaded_profitability_reasons(
    threading_score_x100: usize,
    lanes: usize,
    steps: usize,
    backend_affinity: &BackendAffinityReport,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if lanes.saturating_mul(steps) >= 256 {
        reasons.push("threaded-lane-parallel".to_string());
    }
    if matches!(
        backend_affinity.recommendation,
        BackendAffinityRecommendation::MixedScalarSimdCandidate
    ) {
        reasons.push("mixed-scalar-simd-candidate".to_string());
    }
    if threading_score_x100 < 25 {
        reasons.push("threading-setup-not-amortized".to_string());
    }
    if reasons.is_empty() {
        reasons.push("threaded-static-score".to_string());
    }
    reasons
}

fn gpu_profitability_reasons(
    gpu_suitability_score_x100: usize,
    analysis: &GpuRegionAnalysis,
    lane_work_units: usize,
) -> Vec<String> {
    let mut reasons = Vec::new();
    match analysis.recommendation {
        GpuRegionRecommendation::ComputeCandidate => {
            reasons.push("gpu-compute-candidate".to_string())
        }
        GpuRegionRecommendation::MixedCandidate => reasons.push("gpu-mixed-candidate".to_string()),
        GpuRegionRecommendation::MemoryBlocked => reasons.push("memory-hostile".to_string()),
        GpuRegionRecommendation::TooSmall => reasons.push("gpu-launch-not-amortized".to_string()),
    }
    if analysis.total.memory_hostile_packets > 0 {
        reasons.push("memory-hostile-packets".to_string());
    }
    if gpu_suitability_score_x100 < 50 || lane_work_units < 4096 {
        reasons.push("gpu-launch-not-amortized".to_string());
    }
    reasons
}

fn is_default_like_gpu_option(options: GpuBatchOptions) -> bool {
    options.memory_layout == GpuMemoryLayout::LaneMajor
        && options.workgroup_size == 64
        && !options.reuse_temporaries
}

fn empty_bench_gpu_trace_report(
    trace: &PyrtlLaneTrace,
    options: &BenchGpuTraceOptions,
    import_ns: u128,
    setup_ns: u128,
    backend_affinity: BackendAffinityReport,
    gpu_region_analysis: GpuRegionAnalysis,
    shader_stats: BenchGpuShaderStats,
    mismatches: Vec<TraceMismatch>,
) -> BenchGpuTraceReport {
    let mismatch_count = mismatches.len();
    BenchGpuTraceReport {
        schema: "rrtl-pyrtl-bench-gpu-trace-v1".to_string(),
        steps: trace.steps.len(),
        lanes: trace.lanes,
        repeat: options.repeat,
        warmup: options.warmup,
        import_ns,
        setup_ns,
        prepared_runner_setup_ns: 0,
        prepared_snapshot_setup_ns: 0,
        prepared_trace_bytes: 0,
        prepared_trace_uncompressed_bytes: 0,
        prepared_trace_compression_ratio_x100: 100,
        prepared_trace_uniform_input_ops: 0,
        prepared_trace_uniform_check_ops: 0,
        prepared_trace_layout: String::new(),
        prepared_trace_template_input_ops: 0,
        prepared_trace_template_check_ops: 0,
        prepared_trace_metadata_saved_words: 0,
        prepared_trace_fixed_template: false,
        prepared_trace_value_metadata_saved_words: 0,
        prepared_trace_value_stride_words: 0,
        hot_restore_ns_best: 0,
        hot_restore_ns_median: 0,
        hot_gpu_replay_ns_best: 0,
        hot_gpu_replay_ns_median: 0,
        gpu_single_submit_profitable: false,
        gpu_planner_calibration_reason: None,
        available: true,
        error: None,
        backend_affinity,
        gpu_region_analysis,
        shader_stats,
        gpu_replay_mode: gpu_replay_mode(options).to_string(),
        gpu_timing: GpuTraceReplayTiming::default(),
        replay_ns_samples: Vec::new(),
        replay_ns_best: 0,
        replay_ns_median: 0,
        mismatch_count,
        mismatches,
    }
}

fn gpu_replay_mode(options: &BenchGpuTraceOptions) -> &'static str {
    if options.fused {
        "fused-kernel"
    } else {
        "host-loop"
    }
}

fn replay_gpu_trace_once(
    plan: &LaneTraceReplayPlan,
    gpu_plan: Option<&GpuTraceReplayPlan>,
    prepared_replay: Option<&PreparedGpuTraceReplay>,
    prepared_snapshot: Option<&PreparedGpuStorageSnapshot>,
    sim: &mut GpuBatchSimulator,
    options: &BenchGpuTraceOptions,
) -> Result<(Vec<TraceMismatch>, GpuTraceReplayTiming), ImportError> {
    if options.fused {
        let report = if let Some(prepared) = prepared_replay {
            if let Some(snapshot) = prepared_snapshot {
                sim.replay_prepared_lane_trace_from_snapshot(prepared, snapshot)
                    .map_err(|err| ImportError::new(format!("{err}")))?
            } else {
                sim.replay_prepared_lane_trace(prepared)
                    .map_err(|err| ImportError::new(format!("{err}")))?
            }
        } else {
            let gpu_plan =
                gpu_plan.ok_or_else(|| ImportError::new("missing fused GPU trace replay plan"))?;
            sim.replay_lane_trace(
                gpu_plan,
                GpuTraceReplayOptions {
                    max_mismatches: options.max_mismatches,
                },
            )
            .map_err(|err| ImportError::new(format!("{err}")))?
        };
        let timing = report.timing.clone();
        Ok((plan.gpu_replay_mismatches(&report), timing))
    } else {
        Ok((plan.replay_gpu(sim)?, GpuTraceReplayTiming::default()))
    }
}

fn bench_backend_trace(
    imported: &ImportedDesign,
    plan: &TraceReplayPlan,
    encoded_plan: &EncodedTraceReplayPlan,
    program: &PackedProgram,
    backend: PyrtlBenchBackendKind,
    options: &BenchBackendsTraceOptions,
) -> Result<BenchBackendTraceReport, ImportError> {
    let lanes = if backend == PyrtlBenchBackendKind::Scalar {
        1
    } else {
        options.lanes
    };

    for _ in 0..options.warmup {
        let mut sim = match backend_simulator(imported, program, backend, lanes) {
            Ok(sim) => sim,
            Err(err) => {
                return Ok(unavailable_backend_trace_report(backend, lanes, err));
            }
        };
        let replay = sim
            .replay_trace(encoded_plan, backend_replay_options())
            .map_err(|err| ImportError::new(format!("{err}")))?;
        let mismatches = plan.backend_mismatches(&replay);
        if !mismatches.is_empty() {
            return Ok(empty_backend_trace_report(
                backend,
                lanes,
                replay.mismatch_count,
                mismatches,
                replay.timing,
                replay.simd_stats,
            ));
        }
    }

    let mut samples = Vec::with_capacity(options.repeat);
    let mut replay_timing = ReplayTimingReport::default();
    let mut simd_stats = None;
    for _ in 0..options.repeat {
        let mut sim = match backend_simulator(imported, program, backend, lanes) {
            Ok(sim) => sim,
            Err(err) => {
                return Ok(unavailable_backend_trace_report(backend, lanes, err));
            }
        };
        let start = Instant::now();
        let replay = sim
            .replay_trace(encoded_plan, backend_replay_options())
            .map_err(|err| ImportError::new(format!("{err}")))?;
        let elapsed = start.elapsed().as_nanos();
        let mismatches = plan.backend_mismatches(&replay);
        if !mismatches.is_empty() {
            return Ok(empty_backend_trace_report(
                backend,
                lanes,
                replay.mismatch_count,
                mismatches,
                replay.timing,
                replay.simd_stats,
            ));
        }
        replay_timing = replay.timing;
        simd_stats = replay.simd_stats;
        samples.push(elapsed);
    }

    let mut sorted = samples.clone();
    sorted.sort_unstable();
    let best = sorted.first().copied().unwrap_or(0);
    let median = sorted[sorted.len() / 2];
    Ok(BenchBackendTraceReport {
        backend: backend.as_str().to_string(),
        lanes,
        available: true,
        error: None,
        replay_ns_samples: samples,
        replay_ns_best: best,
        replay_ns_median: median,
        replay_timing,
        simd_stats,
        mismatch_count: 0,
        mismatches: Vec::new(),
    })
}

fn empty_backend_trace_report(
    backend: PyrtlBenchBackendKind,
    lanes: usize,
    mismatch_count: usize,
    mismatches: Vec<TraceMismatch>,
    replay_timing: ReplayTimingReport,
    simd_stats: Option<ReplaySimdStats>,
) -> BenchBackendTraceReport {
    BenchBackendTraceReport {
        backend: backend.as_str().to_string(),
        lanes,
        available: true,
        error: None,
        replay_ns_samples: Vec::new(),
        replay_ns_best: 0,
        replay_ns_median: 0,
        replay_timing,
        simd_stats,
        mismatch_count,
        mismatches,
    }
}

fn unavailable_backend_trace_report(
    backend: PyrtlBenchBackendKind,
    lanes: usize,
    err: ImportError,
) -> BenchBackendTraceReport {
    BenchBackendTraceReport {
        backend: backend.as_str().to_string(),
        lanes,
        available: false,
        error: Some(err.to_string()),
        replay_ns_samples: Vec::new(),
        replay_ns_best: 0,
        replay_ns_median: 0,
        replay_timing: ReplayTimingReport::default(),
        simd_stats: None,
        mismatch_count: 0,
        mismatches: Vec::new(),
    }
}

fn backend_replay_options() -> ReplayOptions {
    ReplayOptions {
        lane_mode: ReplayLaneMode::Replicated,
        check_mode: ReplayCheckMode::Lane0Fast,
        max_mismatches: 16,
    }
}

fn backend_simulator(
    imported: &ImportedDesign,
    program: &PackedProgram,
    backend: PyrtlBenchBackendKind,
    lanes: usize,
) -> Result<SimBackendInstance, ImportError> {
    let options = SimBackendOptions {
        kind: backend.sim_kind(),
        lanes,
    };
    let mut sim = SimBackendInstance::new(program.clone(), options)
        .map_err(|err| ImportError::new(format!("{err}")))?;
    initialize_backend_simulator(imported, &mut sim)?;
    Ok(sim)
}

fn initialize_backend_simulator(
    imported: &ImportedDesign,
    sim: &mut SimBackendInstance,
) -> Result<(), ImportError> {
    let Some(module) = imported.design.ir().find_module(&imported.top_name) else {
        return Err(ImportError::new(format!(
            "module `{}` does not exist",
            imported.top_name
        )));
    };

    for initial in &module.initial_register_values {
        sim.set_signals_replicated(&[(initial.signal, initial.value as u128)])
            .map_err(|err| ImportError::new(format!("{err}")))?;
    }

    for signal in &module.signals {
        let SignalKind::Mem { depth, .. } = signal.kind else {
            continue;
        };
        let mut words = vec![0u128; depth];
        let mut has_initial = false;
        for initial in module
            .initial_memory_values
            .iter()
            .filter(|initial| initial.mem == signal.handle)
        {
            if initial.addr < words.len() {
                words[initial.addr] = initial.value as u128;
                has_initial = true;
            }
        }
        if has_initial {
            sim.set_memory_replicated(signal.handle, &words)
                .map_err(|err| ImportError::new(format!("{err}")))?;
        }
    }

    sim.eval_combinational()
        .map_err(|err| ImportError::new(format!("{err}")))?;
    Ok(())
}

fn threaded_initial_state(
    imported: &ImportedDesign,
) -> Result<ThreadedReplayInitialState, ImportError> {
    let Some(module) = imported.design.ir().find_module(&imported.top_name) else {
        return Err(ImportError::new(format!(
            "module `{}` does not exist",
            imported.top_name
        )));
    };

    let signals = module
        .initial_register_values
        .iter()
        .map(|initial| (initial.signal, initial.value as u128))
        .collect::<Vec<_>>();

    let mut memories = Vec::new();
    for signal in &module.signals {
        let SignalKind::Mem { depth, .. } = signal.kind else {
            continue;
        };
        let mut words = vec![0u128; depth];
        let mut has_initial = false;
        for initial in module
            .initial_memory_values
            .iter()
            .filter(|initial| initial.mem == signal.handle)
        {
            if initial.addr < words.len() {
                words[initial.addr] = initial.value as u128;
                has_initial = true;
            }
        }
        if has_initial {
            memories.push((signal.handle, words));
        }
    }

    Ok(ThreadedReplayInitialState { signals, memories })
}

fn packed_simulator(
    imported: &ImportedDesign,
    program: &PackedProgram,
    lanes: usize,
) -> Result<PackedSimulator, ImportError> {
    let mut sim = PackedSimulator::new(program.clone(), lanes)
        .map_err(|err| ImportError::new(format!("{err}")))?;
    initialize_packed_simulator(imported, &mut sim, lanes)?;
    Ok(sim)
}

fn initialize_packed_simulator(
    imported: &ImportedDesign,
    sim: &mut PackedSimulator,
    lanes: usize,
) -> Result<(), ImportError> {
    let Some(module) = imported.design.ir().find_module(&imported.top_name) else {
        return Err(ImportError::new(format!(
            "module `{}` does not exist",
            imported.top_name
        )));
    };

    for initial in &module.initial_register_values {
        sim.set_signal(initial.signal, &vec![initial.value as u128; lanes])
            .map_err(|err| ImportError::new(format!("{err}")))?;
    }

    for signal in &module.signals {
        let SignalKind::Mem { depth, .. } = signal.kind else {
            continue;
        };
        let mut words = vec![0u128; depth];
        let mut has_initial = false;
        for initial in module
            .initial_memory_values
            .iter()
            .filter(|initial| initial.mem == signal.handle)
        {
            if initial.addr < words.len() {
                words[initial.addr] = initial.value as u128;
                has_initial = true;
            }
        }
        if has_initial {
            sim.set_memory(signal.handle, &vec![words; lanes])
                .map_err(|err| ImportError::new(format!("{err}")))?;
        }
    }

    sim.eval_combinational();
    Ok(())
}

fn gpu_simulator(
    imported: &ImportedDesign,
    compiled: &rrtl_core::CompiledDesign,
    program: &PackedProgram,
    lanes: usize,
    options: GpuBatchOptions,
) -> Result<GpuBatchSimulator, ImportError> {
    let mut sim = GpuBatchSimulator::new_from_compiled_with_options(
        compiled,
        &imported.top_name,
        lanes,
        options,
    )
    .map_err(|err| ImportError::new(format!("{err}")))?;
    initialize_gpu_simulator(imported, program, &mut sim, lanes)?;
    Ok(sim)
}

fn initialize_gpu_simulator(
    imported: &ImportedDesign,
    program: &PackedProgram,
    sim: &mut GpuBatchSimulator,
    lanes: usize,
) -> Result<(), ImportError> {
    let Some(module) = imported.design.ir().find_module(&imported.top_name) else {
        return Err(ImportError::new(format!(
            "module `{}` does not exist",
            imported.top_name
        )));
    };

    for initial in &module.initial_register_values {
        let width = packed_signal_width(program, initial.signal)?;
        let limbs = u128_to_gpu_limbs(initial.value as u128, width);
        sim.set_input_limbs(initial.signal, &vec![limbs; lanes])
            .map_err(|err| ImportError::new(format!("{err}")))?;
    }

    for signal in &module.signals {
        let SignalKind::Mem { depth, .. } = signal.kind else {
            continue;
        };
        let memory = program
            .memories
            .iter()
            .find(|memory| memory.source == signal.handle)
            .ok_or_else(|| {
                ImportError::new(format!(
                    "memory signal {:?} is not part of the packed GPU program",
                    signal.handle.id
                ))
            })?;
        let mut words = vec![vec![0u32; memory.data_layout.limbs]; depth];
        let mut has_initial = false;
        for initial in module
            .initial_memory_values
            .iter()
            .filter(|initial| initial.mem == signal.handle)
        {
            if initial.addr < words.len() {
                words[initial.addr] =
                    u128_to_gpu_limbs(initial.value as u128, memory.data_layout.width);
                has_initial = true;
            }
        }
        if has_initial {
            sim.set_memory_limbs(signal.handle, &vec![words; lanes])
                .map_err(|err| ImportError::new(format!("{err}")))?;
        }
    }

    sim.eval_combinational()
        .map_err(|err| ImportError::new(format!("{err}")))?;
    Ok(())
}

fn packed_signal_width(program: &PackedProgram, signal: Signal) -> Result<u32, ImportError> {
    let index = program.signal_index(signal).ok_or_else(|| {
        ImportError::new(format!(
            "signal {:?} is not part of the packed GPU program",
            signal.id
        ))
    })?;
    Ok(program.signals[index].layout.width)
}

fn u128_to_gpu_limbs(value: u128, width: u32) -> Vec<u32> {
    let limb_count = limbs(width).max(1);
    let mut out = Vec::with_capacity(limb_count);
    for limb in 0..limb_count {
        out.push(((value >> (limb * 32)) & 0xffff_ffff) as u32);
    }
    if let Some(last) = out.last_mut() {
        *last &= final_limb_mask(width);
    }
    out
}

fn gpu_limbs_to_u128(limbs_value: &[u32], width: u32) -> u128 {
    let mut out = 0u128;
    for (limb, value) in limbs_value.iter().copied().take(4).enumerate() {
        out |= u128::from(value) << (limb * 32);
    }
    out & mask_u128_local(width)
}

fn mask_u128_local(width: u32) -> u128 {
    if width == 0 {
        0
    } else if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

fn single_lane_simulator(
    imported: &ImportedDesign,
    program: &PackedProgram,
) -> Result<SingleLaneMachineSimulator, ImportError> {
    let mut sim = SingleLaneMachineSimulator::new(program.clone())
        .map_err(|err| ImportError::new(format!("{err}")))?;
    initialize_single_lane_simulator(imported, &mut sim)?;
    Ok(sim)
}

fn initialize_single_lane_simulator(
    imported: &ImportedDesign,
    sim: &mut SingleLaneMachineSimulator,
) -> Result<(), ImportError> {
    let Some(module) = imported.design.ir().find_module(&imported.top_name) else {
        return Err(ImportError::new(format!(
            "module `{}` does not exist",
            imported.top_name
        )));
    };

    for initial in &module.initial_register_values {
        sim.set_signal(initial.signal, initial.value as u128)
            .map_err(|err| ImportError::new(format!("{err}")))?;
    }

    for signal in &module.signals {
        let SignalKind::Mem { depth, .. } = signal.kind else {
            continue;
        };
        let mut words = vec![0u128; depth];
        let mut has_initial = false;
        for initial in module
            .initial_memory_values
            .iter()
            .filter(|initial| initial.mem == signal.handle)
        {
            if initial.addr < words.len() {
                words[initial.addr] = initial.value as u128;
                has_initial = true;
            }
        }
        if has_initial {
            sim.set_memory(signal.handle, &words)
                .map_err(|err| ImportError::new(format!("{err}")))?;
        }
    }

    sim.eval_combinational();
    Ok(())
}

struct TraceReplayPlan<'a> {
    imported: &'a ImportedDesign,
    steps: Vec<TraceReplayStep>,
    check_names: Vec<String>,
}

struct TraceReplayStep {
    inputs: Vec<(Signal, u128)>,
    outputs: Vec<TraceReplayOutput>,
}

struct TraceReplayOutput {
    check_index: usize,
    name: String,
    signal: Signal,
    expected: u128,
}

struct LaneTraceReplayPlan {
    lanes: usize,
    steps: Vec<LaneTraceReplayStep>,
    check_names: Vec<String>,
}

struct LaneTraceReplayStep {
    inputs: Vec<(Signal, Vec<u128>)>,
    outputs: Vec<LaneTraceReplayOutput>,
}

struct LaneTraceReplayOutput {
    check_index: usize,
    signal: Signal,
    expected: Vec<u128>,
}

impl LaneTraceReplayPlan {
    fn new(imported: &ImportedDesign, trace: &PyrtlLaneTrace) -> Result<Self, ImportError> {
        if trace.lanes == 0 {
            return Err(ImportError::new(
                "lane trace lanes must be greater than zero",
            ));
        }
        let mut steps = Vec::with_capacity(trace.steps.len());
        let mut check_names = Vec::new();
        for step in &trace.steps {
            let mut inputs = Vec::with_capacity(step.inputs.len());
            for (name, values) in &step.inputs {
                if values.len() != trace.lanes {
                    return Err(ImportError::new(format!(
                        "lane trace input `{name}` has {} lanes, expected {}",
                        values.len(),
                        trace.lanes
                    )));
                }
                let Some(signal) = imported.design.find_signal(&imported.top_name, name) else {
                    return Err(ImportError::new(format!(
                        "lane trace input `{name}` is unknown"
                    )));
                };
                inputs.push((signal, values.clone()));
            }

            let mut outputs = Vec::with_capacity(step.outputs.len());
            for (name, expected) in &step.outputs {
                if expected.len() != trace.lanes {
                    return Err(ImportError::new(format!(
                        "lane trace output `{name}` has {} lanes, expected {}",
                        expected.len(),
                        trace.lanes
                    )));
                }
                let Some(signal) = imported.design.find_signal(&imported.top_name, name) else {
                    return Err(ImportError::new(format!(
                        "lane trace output `{name}` is unknown"
                    )));
                };
                let check_index = check_names.len();
                check_names.push(name.clone());
                outputs.push(LaneTraceReplayOutput {
                    check_index,
                    signal,
                    expected: expected.clone(),
                });
            }
            steps.push(LaneTraceReplayStep { inputs, outputs });
        }
        Ok(Self {
            lanes: trace.lanes,
            steps,
            check_names,
        })
    }

    fn encoded(&self, program: &PackedProgram) -> Result<EncodedTraceReplayPlan, ImportError> {
        let steps = self.steps.iter().map(|step| {
            let inputs = step.inputs.clone();
            let outputs = step
                .outputs
                .iter()
                .map(|output| (output.check_index, output.signal, output.expected.clone()))
                .collect::<Vec<_>>();
            (inputs, outputs)
        });
        EncodedTraceReplayPlan::from_independent_lane_steps(program, self.lanes, steps)
            .map_err(|err| ImportError::new(format!("{err}")))
    }

    fn threaded_mismatches(&self, report: &ThreadedReplayReport) -> Vec<TraceMismatch> {
        report
            .replay
            .mismatches
            .iter()
            .map(|mismatch| TraceMismatch {
                step: mismatch.step,
                signal: self
                    .check_names
                    .get(mismatch.check_index)
                    .cloned()
                    .unwrap_or_else(|| format!("check#{}", mismatch.check_index)),
                expected: mismatch.expected,
                actual: mismatch.actual,
                lane: mismatch.lane,
            })
            .collect()
    }

    fn gpu_replay_plan(&self, program: &PackedProgram) -> Result<GpuTraceReplayPlan, ImportError> {
        let mut inputs = Vec::new();
        let mut checks = Vec::new();
        for (step_index, step) in self.steps.iter().enumerate() {
            for (signal, values) in &step.inputs {
                let width = packed_signal_width(program, *signal)?;
                let limb_count = limbs(width).max(1);
                for limb in 0..limb_count {
                    inputs.push(GpuTraceInputOp {
                        step: step_index,
                        signal: *signal,
                        limb,
                        values: values
                            .iter()
                            .copied()
                            .map(|value| u128_to_gpu_limbs(value, width)[limb])
                            .collect(),
                    });
                }
            }
            for output in &step.outputs {
                let width = packed_signal_width(program, output.signal)?;
                let limb_count = limbs(width).max(1);
                for limb in 0..limb_count {
                    checks.push(GpuTraceCheckOp {
                        step: step_index,
                        check_index: output.check_index,
                        signal: output.signal,
                        limb,
                        expected: output
                            .expected
                            .iter()
                            .copied()
                            .map(|value| u128_to_gpu_limbs(value, width)[limb])
                            .collect(),
                    });
                }
            }
        }
        Ok(GpuTraceReplayPlan {
            steps: self.steps.len(),
            inputs,
            checks,
        })
    }

    fn gpu_replay_mismatches(&self, report: &GpuTraceReplayReport) -> Vec<TraceMismatch> {
        report
            .mismatches
            .iter()
            .map(|mismatch| {
                let name = self
                    .check_names
                    .get(mismatch.check_index)
                    .cloned()
                    .unwrap_or_else(|| format!("check#{}", mismatch.check_index));
                let signal = if mismatch.limb == 0 {
                    name
                } else {
                    format!("{name}[limb{}]", mismatch.limb)
                };
                TraceMismatch {
                    step: mismatch.step,
                    signal,
                    expected: u128::from(mismatch.expected),
                    actual: u128::from(mismatch.actual),
                    lane: Some(mismatch.lane),
                }
            })
            .collect()
    }

    fn replay_gpu(&self, sim: &mut GpuBatchSimulator) -> Result<Vec<TraceMismatch>, ImportError> {
        let mut mismatches = Vec::new();
        for (step_index, step) in self.steps.iter().enumerate() {
            for (signal, values) in &step.inputs {
                let width = packed_signal_width(sim.program(), *signal)?;
                let lane_values = values
                    .iter()
                    .copied()
                    .map(|value| u128_to_gpu_limbs(value, width))
                    .collect::<Vec<_>>();
                sim.set_input_limbs(*signal, &lane_values)
                    .map_err(|err| ImportError::new(format!("{err}")))?;
            }
            sim.eval_combinational()
                .map_err(|err| ImportError::new(format!("{err}")))?;
            for output in &step.outputs {
                let width = packed_signal_width(sim.program(), output.signal)?;
                let actual = sim
                    .get_signal_limbs(output.signal)
                    .map_err(|err| ImportError::new(format!("{err}")))?
                    .into_iter()
                    .map(|limbs| gpu_limbs_to_u128(&limbs, width))
                    .collect::<Vec<_>>();
                for (lane, (actual, expected)) in actual
                    .iter()
                    .copied()
                    .zip(output.expected.iter().copied())
                    .enumerate()
                {
                    let expected = expected & mask_u128_local(width);
                    if actual != expected {
                        mismatches.push(TraceMismatch {
                            step: step_index,
                            signal: self
                                .check_names
                                .get(output.check_index)
                                .cloned()
                                .unwrap_or_else(|| format!("check#{}", output.check_index)),
                            expected,
                            actual,
                            lane: Some(lane),
                        });
                    }
                }
            }
            sim.tick()
                .map_err(|err| ImportError::new(format!("{err}")))?;
        }
        Ok(mismatches)
    }
}

impl<'a> TraceReplayPlan<'a> {
    fn new(imported: &'a ImportedDesign, trace: &PyrtlTrace) -> Result<Self, ImportError> {
        let mut steps = Vec::with_capacity(trace.steps.len());
        let mut check_names = Vec::new();
        for step in &trace.steps {
            let mut inputs = Vec::with_capacity(step.inputs.len());
            for (name, value) in &step.inputs {
                let Some(signal) = imported.design.find_signal(&imported.top_name, name) else {
                    return Err(ImportError::new(format!("trace input `{name}` is unknown")));
                };
                inputs.push((signal, *value));
            }

            let mut outputs = Vec::with_capacity(step.outputs.len());
            for (name, expected) in &step.outputs {
                let Some(signal) = imported.design.find_signal(&imported.top_name, name) else {
                    return Err(ImportError::new(format!(
                        "trace output `{name}` is unknown"
                    )));
                };
                let check_index = check_names.len();
                check_names.push(name.clone());
                outputs.push(TraceReplayOutput {
                    check_index,
                    name: name.clone(),
                    signal,
                    expected: *expected,
                });
            }
            steps.push(TraceReplayStep { inputs, outputs });
        }
        Ok(Self {
            imported,
            steps,
            check_names,
        })
    }

    fn encoded(&self, program: &PackedProgram) -> Result<EncodedTraceReplayPlan, ImportError> {
        let steps = self.steps.iter().map(|step| {
            let inputs = step.inputs.clone();
            let outputs = step
                .outputs
                .iter()
                .map(|output| (output.check_index, output.signal, output.expected))
                .collect::<Vec<_>>();
            (inputs, outputs)
        });
        EncodedTraceReplayPlan::from_signal_steps(program, steps)
            .map_err(|err| ImportError::new(format!("{err}")))
    }

    fn backend_mismatches(&self, report: &rrtl_sim_ir::ReplayReport) -> Vec<TraceMismatch> {
        report
            .mismatches
            .iter()
            .map(|mismatch| TraceMismatch {
                step: mismatch.step,
                signal: self
                    .check_names
                    .get(mismatch.check_index)
                    .cloned()
                    .unwrap_or_else(|| format!("check#{}", mismatch.check_index)),
                expected: mismatch.expected,
                actual: mismatch.actual,
                lane: mismatch.lane,
            })
            .collect()
    }

    fn simulator(&self) -> Result<Simulator<'a>, ImportError> {
        Simulator::new(&self.imported.design, &self.imported.top_name)
            .map_err(|err| ImportError::new(format!("{err}")))
    }

    fn replay(&self, sim: &mut Simulator) -> Vec<TraceMismatch> {
        let mut mismatches = Vec::new();
        for (step_index, step) in self.steps.iter().enumerate() {
            for (signal, value) in &step.inputs {
                sim.set(*signal, *value);
            }
            for output in &step.outputs {
                let actual = sim.get(output.signal);
                if actual != output.expected {
                    mismatches.push(TraceMismatch {
                        step: step_index,
                        signal: output.name.clone(),
                        expected: output.expected,
                        actual,
                        lane: None,
                    });
                }
            }
            sim.tick();
        }
        mismatches
    }

    fn replay_packed(
        &self,
        sim: &mut PackedSimulator,
        lanes: usize,
    ) -> Result<Vec<TraceMismatch>, ImportError> {
        let mut mismatches = Vec::new();
        for (step_index, step) in self.steps.iter().enumerate() {
            sim.set_signals_replicated(&step.inputs)
                .map_err(|err| ImportError::new(format!("{err}")))?;
            for output in &step.outputs {
                for lane in 0..lanes {
                    let actual = sim
                        .get_signal_lane(output.signal, lane)
                        .map_err(|err| ImportError::new(format!("{err}")))?;
                    if actual != output.expected {
                        mismatches.push(TraceMismatch {
                            step: step_index,
                            signal: output.name.clone(),
                            expected: output.expected,
                            actual,
                            lane: Some(lane),
                        });
                    }
                }
            }
            sim.tick();
        }
        Ok(mismatches)
    }

    fn replay_single(
        &self,
        sim: &mut SingleLaneMachineSimulator,
    ) -> Result<Vec<TraceMismatch>, ImportError> {
        let mut mismatches = Vec::new();
        for (step_index, step) in self.steps.iter().enumerate() {
            sim.set_signals(&step.inputs)
                .map_err(|err| ImportError::new(format!("{err}")))?;
            for output in &step.outputs {
                let actual = sim
                    .get_signal(output.signal)
                    .map_err(|err| ImportError::new(format!("{err}")))?;
                if actual != output.expected {
                    mismatches.push(TraceMismatch {
                        step: step_index,
                        signal: output.name.clone(),
                        expected: output.expected,
                        actual,
                        lane: None,
                    });
                }
            }
            sim.tick();
        }
        Ok(mismatches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rrtl_gpu_sim::{GpuRegionBlockAnalysis, GpuRegionStreamsAnalysis};
    use rrtl_sim_ir::{
        BackendAffinityBlockReport, BackendAffinityStreamsReport, SimdFastPathProfile,
        SimdSuitabilityBlockReport, SimdSuitabilityStreamsReport,
    };

    #[test]
    fn profitability_prefers_simd_for_high_coverage_native_work() {
        let simd = simd_report(
            SimdSuitabilityRecommendation::SimdCandidate,
            100,
            100,
            SimdFastPathProfile {
                one_limb_ops: 80,
                two_limb_ops: 20,
                ..SimdFastPathProfile::default()
            },
            0,
        );
        let affinity = affinity_report(BackendAffinityRecommendation::SimdCpuCandidate, 100, 40);
        let gpu = gpu_report(GpuRegionRecommendation::TooSmall, 4, 0, 20);
        let workload = workload(4, 8, 512);

        let (report, candidates, _features) =
            plan_backend_profitability(4, 8, &simd, &affinity, &gpu, &workload, false, None, None);

        assert_eq!(report.recommended_backend, "simd-cpu");
        assert_eq!(candidates[0].backend, "simd-cpu");
        assert!(candidates[0]
            .reasons
            .contains(&"high-simd-coverage".to_string()));
    }

    #[test]
    fn profitability_promotes_threaded_for_large_mixed_replay() {
        let simd = simd_report(
            SimdSuitabilityRecommendation::MixedCandidate,
            120,
            70,
            SimdFastPathProfile {
                one_limb_ops: 30,
                two_limb_ops: 20,
                ..SimdFastPathProfile::default()
            },
            20,
        );
        let affinity = affinity_report(
            BackendAffinityRecommendation::MixedScalarSimdCandidate,
            140,
            80,
        );
        let gpu = gpu_report(GpuRegionRecommendation::MixedCandidate, 10, 2, 80);
        let workload = workload(64, 64, 50_000);

        let (report, candidates, _features) = plan_backend_profitability(
            64, 64, &simd, &affinity, &gpu, &workload, false, None, None,
        );

        assert_eq!(report.recommended_backend, "threaded-mixed");
        assert_eq!(candidates[0].backend, "threaded-mixed");
        assert!(candidates[0]
            .reasons
            .contains(&"threaded-lane-parallel".to_string()));
    }

    #[test]
    fn profitability_marks_small_gpu_work_as_launch_not_amortized() {
        let simd = simd_report(
            SimdSuitabilityRecommendation::ScalarPreferred,
            8,
            2,
            SimdFastPathProfile::default(),
            80,
        );
        let affinity = affinity_report(BackendAffinityRecommendation::ScalarPreferred, 8, 2);
        let gpu = gpu_report(GpuRegionRecommendation::ComputeCandidate, 4, 0, 20);
        let workload = workload(2, 2, 128);

        let (_report, candidates, _features) =
            plan_backend_profitability(2, 2, &simd, &affinity, &gpu, &workload, true, None, None);
        let gpu_candidate = candidates
            .iter()
            .find(|candidate| candidate.backend == "gpu-fused")
            .unwrap();

        assert!(gpu_candidate
            .reasons
            .contains(&"gpu-launch-not-amortized".to_string()));
        assert_ne!(candidates[0].backend, "gpu-fused");
    }

    #[test]
    fn profitability_demotes_memory_hostile_gpu() {
        let simd = simd_report(
            SimdSuitabilityRecommendation::SimdCandidate,
            40,
            35,
            SimdFastPathProfile {
                one_limb_ops: 30,
                ..SimdFastPathProfile::default()
            },
            5,
        );
        let affinity = affinity_report(BackendAffinityRecommendation::GpuBlocked, 40, 35);
        let gpu = gpu_report(GpuRegionRecommendation::MemoryBlocked, 4, 12, 40);
        let workload = workload(64, 64, 50_000);

        let (_report, candidates, _features) = plan_backend_profitability(
            64, 64, &simd, &affinity, &gpu, &workload, false, None, None,
        );
        let gpu_candidate = candidates
            .iter()
            .find(|candidate| candidate.backend == "gpu-fused")
            .unwrap();

        assert!(gpu_candidate
            .reasons
            .contains(&"memory-hostile".to_string()));
        assert_ne!(candidates[0].backend, "gpu-fused");
    }

    #[test]
    fn profitability_calibration_can_reorder_close_candidates() {
        let simd = simd_report(
            SimdSuitabilityRecommendation::SimdCandidate,
            100,
            60,
            SimdFastPathProfile {
                one_limb_ops: 40,
                ..SimdFastPathProfile::default()
            },
            10,
        );
        let affinity = affinity_report(BackendAffinityRecommendation::SimdCpuCandidate, 100, 60);
        let gpu = gpu_report(GpuRegionRecommendation::TooSmall, 4, 0, 20);
        let workload = workload(8, 8, 1_000);
        let calibration = profitability_calibration(&[("packed-cpu", 50.0)], &[]);

        let (_report, candidates, _features) = plan_backend_profitability(
            8,
            8,
            &simd,
            &affinity,
            &gpu,
            &workload,
            false,
            None,
            Some(&calibration),
        );

        assert_eq!(candidates[0].backend, "packed-cpu");
        assert!(candidates[0]
            .reasons
            .iter()
            .any(|reason| reason.starts_with("profitability-calibrated-backend:")));
    }

    #[test]
    fn profitability_calibration_cannot_override_gpu_memory_hard_gate() {
        let simd = simd_report(
            SimdSuitabilityRecommendation::MixedCandidate,
            40,
            25,
            SimdFastPathProfile {
                one_limb_ops: 20,
                ..SimdFastPathProfile::default()
            },
            20,
        );
        let affinity = affinity_report(BackendAffinityRecommendation::GpuBlocked, 40, 25);
        let gpu = gpu_report(GpuRegionRecommendation::MemoryBlocked, 4, 12, 40);
        let workload = workload(64, 64, 50_000);
        let calibration = profitability_calibration(&[("gpu-fused", 100.0)], &[]);

        let (_report, candidates, _features) = plan_backend_profitability(
            64,
            64,
            &simd,
            &affinity,
            &gpu,
            &workload,
            false,
            None,
            Some(&calibration),
        );

        assert!(!profitability_allows_gpu(&candidates));
        assert_ne!(candidates[0].backend, "gpu-fused");
        let gpu_candidate = candidates
            .iter()
            .find(|candidate| candidate.backend == "gpu-fused")
            .unwrap();
        assert!(gpu_candidate
            .reasons
            .contains(&"memory-hostile".to_string()));
    }

    #[test]
    fn profitability_feature_calibration_can_boost_simd() {
        let simd = simd_report(
            SimdSuitabilityRecommendation::MixedCandidate,
            100,
            65,
            SimdFastPathProfile {
                one_limb_ops: 60,
                ..SimdFastPathProfile::default()
            },
            10,
        );
        let affinity = affinity_report(
            BackendAffinityRecommendation::MixedScalarSimdCandidate,
            100,
            65,
        );
        let gpu = gpu_report(GpuRegionRecommendation::TooSmall, 4, 0, 20);
        let workload = workload(16, 16, 8_000);
        let calibration =
            profitability_feature_calibration(&[("simd-cpu|simd_coverage>=50", 50.0)], &[]);

        let (_report, candidates, features) = plan_backend_profitability(
            16,
            16,
            &simd,
            &affinity,
            &gpu,
            &workload,
            false,
            None,
            Some(&calibration),
        );

        assert!(features
            .feature_buckets
            .contains(&"simd_coverage>=50".to_string()));
        let simd_candidate = candidates
            .iter()
            .find(|candidate| candidate.backend == "simd-cpu")
            .unwrap();
        assert!(simd_candidate
            .reasons
            .iter()
            .any(|reason| reason.starts_with("profitability-calibrated-feature:")));
    }

    #[test]
    fn profitability_feature_calibration_cannot_override_gpu_hard_gate() {
        let simd = simd_report(
            SimdSuitabilityRecommendation::MixedCandidate,
            40,
            25,
            SimdFastPathProfile {
                one_limb_ops: 20,
                ..SimdFastPathProfile::default()
            },
            20,
        );
        let affinity = affinity_report(BackendAffinityRecommendation::GpuBlocked, 40, 25);
        let gpu = gpu_report(GpuRegionRecommendation::MemoryBlocked, 4, 12, 40);
        let workload = workload(64, 64, 50_000);
        let calibration =
            profitability_feature_calibration(&[("gpu-fused|lane_work>=4096", 100.0)], &[]);

        let (_report, candidates, _features) = plan_backend_profitability(
            64,
            64,
            &simd,
            &affinity,
            &gpu,
            &workload,
            false,
            None,
            Some(&calibration),
        );

        assert!(!profitability_allows_gpu(&candidates));
        assert_ne!(candidates[0].backend, "gpu-fused");
    }

    #[test]
    fn profitability_promotes_single_submit_gpu_for_large_compute_trace() {
        let simd = simd_report(
            SimdSuitabilityRecommendation::MixedCandidate,
            160,
            95,
            SimdFastPathProfile {
                one_limb_ops: 80,
                ..SimdFastPathProfile::default()
            },
            20,
        );
        let affinity = affinity_report(BackendAffinityRecommendation::GpuBlocked, 160, 95);
        let gpu = gpu_report(GpuRegionRecommendation::ComputeCandidate, 80, 0, 512);
        let workload = workload(256, 256, 4_000_000);

        let (report, candidates, features) = plan_backend_profitability(
            256, 256, &simd, &affinity, &gpu, &workload, true, None, None,
        );

        assert_eq!(report.recommended_backend, "gpu-fused");
        assert_eq!(candidates[0].backend, "gpu-fused");
        assert!(features.gpu_single_submit_available);
        assert!(features.gpu_count_readback_only);
        assert!(features
            .feature_buckets
            .contains(&"gpu_single_submit_available".to_string()));
        assert!(features
            .feature_buckets
            .contains(&"gpu_count_readback_only".to_string()));
    }

    #[test]
    fn profitability_single_submit_gpu_keeps_memory_hard_gate() {
        let simd = simd_report(
            SimdSuitabilityRecommendation::MixedCandidate,
            160,
            95,
            SimdFastPathProfile {
                one_limb_ops: 80,
                ..SimdFastPathProfile::default()
            },
            20,
        );
        let affinity = affinity_report(BackendAffinityRecommendation::GpuBlocked, 160, 95);
        let gpu = gpu_report(GpuRegionRecommendation::MemoryBlocked, 80, 8, 512);
        let workload = workload(256, 256, 4_000_000);

        let (_report, candidates, features) = plan_backend_profitability(
            256, 256, &simd, &affinity, &gpu, &workload, true, None, None,
        );

        assert!(!profitability_allows_gpu(&candidates));
        assert_ne!(candidates[0].backend, "gpu-fused");
        assert!(!features.gpu_single_submit_available);
        assert!(features.gpu_full_readback_penalty > 0);
        assert!(features
            .feature_buckets
            .contains(&"gpu_full_readback_penalty".to_string()));
    }

    #[test]
    fn profitability_features_include_gpu_trace_compression() {
        let simd = simd_report(
            SimdSuitabilityRecommendation::MixedCandidate,
            160,
            95,
            SimdFastPathProfile {
                one_limb_ops: 80,
                ..SimdFastPathProfile::default()
            },
            20,
        );
        let affinity = affinity_report(BackendAffinityRecommendation::GpuBlocked, 160, 95);
        let gpu = gpu_report(GpuRegionRecommendation::ComputeCandidate, 80, 0, 512);
        let workload = workload(128, 128, 1_000_000);
        let compression = GpuTraceCompressionEstimate {
            compressed_bytes: 32_768,
            uncompressed_bytes: 131_072,
            compression_ratio_x100: 25,
            uniform_input_ops: 32,
            uniform_check_ops: 16,
            template_layout: true,
            template_input_ops: 2,
            template_check_ops: 1,
            metadata_saved_words: 256,
            fixed_template: true,
            value_metadata_saved_words: 128,
            value_stride_words: 6,
        };

        let (_report, candidates, features) = plan_backend_profitability(
            128,
            128,
            &simd,
            &affinity,
            &gpu,
            &workload,
            true,
            Some(&compression),
            None,
        );

        let gpu_candidate = candidates
            .iter()
            .find(|candidate| candidate.backend == "gpu-fused")
            .unwrap();
        assert!(gpu_candidate.score > 0);
        assert_eq!(features.gpu_prepared_trace_bytes, 32_768);
        assert_eq!(features.gpu_trace_uncompressed_bytes, 131_072);
        assert_eq!(features.gpu_trace_compression_ratio_x100, 25);
        assert!(features.gpu_trace_template_layout);
        assert_eq!(features.gpu_trace_template_input_ops, 2);
        assert_eq!(features.gpu_trace_template_check_ops, 1);
        assert_eq!(features.gpu_trace_metadata_saved_words, 256);
        assert!(features.gpu_trace_fixed_template);
        assert_eq!(features.gpu_trace_value_metadata_saved_words, 128);
        assert_eq!(features.gpu_trace_value_stride_words, 6);
        assert!(features
            .feature_buckets
            .contains(&"gpu_trace_uniform_input_ops>=16".to_string()));
        assert!(features
            .feature_buckets
            .contains(&"gpu_trace_uniform_check_ops>=16".to_string()));
        assert!(features
            .feature_buckets
            .contains(&"gpu_trace_compression<=50".to_string()));
        assert!(features
            .feature_buckets
            .contains(&"gpu_trace_template_layout".to_string()));
        assert!(features
            .feature_buckets
            .contains(&"gpu_trace_metadata_saved_words>=128".to_string()));
        assert!(features
            .feature_buckets
            .contains(&"gpu_trace_fixed_template".to_string()));
        assert!(features
            .feature_buckets
            .contains(&"gpu_trace_value_metadata_saved_words>=128".to_string()));
    }

    #[test]
    fn hybrid_threaded_layout_prefers_all_simd_for_high_coverage() {
        let simd = simd_report(
            SimdSuitabilityRecommendation::SimdCandidate,
            100,
            95,
            SimdFastPathProfile {
                one_limb_ops: 90,
                ..SimdFastPathProfile::default()
            },
            0,
        );
        let affinity = affinity_report(BackendAffinityRecommendation::SimdCpuCandidate, 100, 95);
        let workload = workload(8, 64, 20_000);

        let candidates =
            hybrid_threaded_layout_candidates(8, 4, Vec::new(), &simd, &affinity, &workload, None);

        assert_eq!(candidates[0].workers.len(), 1);
        assert_eq!(candidates[0].workers[0].backend, SimBackendKind::SimdCpu);
        assert_eq!(candidates[0].workers[0].lanes, 8);
    }

    #[test]
    fn hybrid_threaded_layout_includes_scalar_simd_for_fallback_heavy_work() {
        let simd = simd_report(
            SimdSuitabilityRecommendation::MixedCandidate,
            100,
            60,
            SimdFastPathProfile {
                one_limb_ops: 40,
                ..SimdFastPathProfile::default()
            },
            40,
        );
        let affinity = affinity_report(
            BackendAffinityRecommendation::MixedScalarSimdCandidate,
            100,
            60,
        );
        let workload = workload(16, 64, 30_000);

        let candidates =
            hybrid_threaded_layout_candidates(16, 4, Vec::new(), &simd, &affinity, &workload, None);

        assert!(candidates.iter().any(|layout| {
            layout
                .workers
                .iter()
                .any(|worker| worker.backend == SimBackendKind::Scalar)
                && layout
                    .workers
                    .iter()
                    .any(|worker| worker.backend == SimBackendKind::SimdCpu)
        }));
    }

    #[test]
    fn hybrid_threaded_layout_includes_packed_simd_for_low_fallback_work() {
        let simd = simd_report(
            SimdSuitabilityRecommendation::MixedCandidate,
            100,
            65,
            SimdFastPathProfile {
                one_limb_ops: 50,
                ..SimdFastPathProfile::default()
            },
            10,
        );
        let affinity = affinity_report(BackendAffinityRecommendation::PackedCpuCandidate, 100, 65);
        let workload = workload(16, 64, 30_000);

        let candidates =
            hybrid_threaded_layout_candidates(16, 4, Vec::new(), &simd, &affinity, &workload, None);

        assert!(candidates.iter().any(|layout| {
            layout
                .workers
                .iter()
                .any(|worker| worker.backend == SimBackendKind::PackedCpu)
                && layout
                    .workers
                    .iter()
                    .any(|worker| worker.backend == SimBackendKind::SimdCpu)
        }));
    }

    #[test]
    fn hybrid_threaded_layout_keeps_tiny_work_single_worker() {
        let simd = simd_report(
            SimdSuitabilityRecommendation::MixedCandidate,
            20,
            12,
            SimdFastPathProfile {
                one_limb_ops: 8,
                ..SimdFastPathProfile::default()
            },
            20,
        );
        let affinity = affinity_report(
            BackendAffinityRecommendation::MixedScalarSimdCandidate,
            20,
            12,
        );
        let workload = workload(2, 4, 128);

        let candidates =
            hybrid_threaded_layout_candidates(2, 4, Vec::new(), &simd, &affinity, &workload, None);

        assert_eq!(candidates[0].workers.len(), 1);
    }

    fn simd_report(
        recommendation: SimdSuitabilityRecommendation,
        instr_count: usize,
        fast_instrs: usize,
        fast_path_profile: SimdFastPathProfile,
        fallback_ratio_x100: usize,
    ) -> SimdSuitabilityReport {
        let total = SimdSuitabilityBlockReport {
            instr_count,
            fast_instrs,
            fallback_instrs: instr_count.saturating_sub(fast_instrs),
            fast_path_profile,
            wide_instrs: instr_count.saturating_sub(fast_instrs),
            max_packet_width: 8,
            ..SimdSuitabilityBlockReport::default()
        };
        SimdSuitabilityReport {
            streams: SimdSuitabilityStreamsReport {
                async_reset_comb: SimdSuitabilityBlockReport::default(),
                comb: total,
                tick_next: SimdSuitabilityBlockReport::default(),
                tick_commit: SimdSuitabilityBlockReport::default(),
            },
            total,
            recommendation,
            score_x100: fast_instrs,
            estimated_fast_cost: fast_instrs,
            estimated_fallback_cost: instr_count.saturating_sub(fast_instrs),
            estimated_materialization_cost: 0,
            fallback_ratio_x100,
        }
    }

    fn affinity_report(
        recommendation: BackendAffinityRecommendation,
        instr_count: usize,
        simd_fast_instrs: usize,
    ) -> BackendAffinityReport {
        let total = BackendAffinityBlockReport {
            instr_count,
            scalar_fast_one_limb_instrs: instr_count.saturating_sub(simd_fast_instrs),
            simd_fast_instrs,
            wide_fallback_instrs: instr_count.saturating_sub(simd_fast_instrs),
            max_packet_width: 8,
            estimated_scalar_cost: instr_count * 20,
            estimated_packed_cpu_cost: instr_count * 10,
            estimated_simd_cpu_cost: instr_count * 5,
            ..BackendAffinityBlockReport::default()
        };
        BackendAffinityReport {
            streams: BackendAffinityStreamsReport {
                async_reset_comb: BackendAffinityBlockReport::default(),
                comb: total,
                tick_next: BackendAffinityBlockReport::default(),
                tick_commit: BackendAffinityBlockReport::default(),
            },
            total,
            recommendation,
            reasons: Vec::new(),
        }
    }

    fn gpu_report(
        recommendation: GpuRegionRecommendation,
        pure_compute_packets: usize,
        memory_hostile_packets: usize,
        launch_work_units: usize,
    ) -> GpuRegionAnalysis {
        let total = GpuRegionBlockAnalysis {
            packets: pure_compute_packets + memory_hostile_packets,
            pure_compute_packets,
            memory_hostile_packets,
            estimated_launch_work_units: launch_work_units,
            ..GpuRegionBlockAnalysis::default()
        };
        GpuRegionAnalysis {
            streams: GpuRegionStreamsAnalysis {
                async_reset_comb: GpuRegionBlockAnalysis::default(),
                comb: total,
                tick_next: GpuRegionBlockAnalysis::default(),
                tick_commit: GpuRegionBlockAnalysis::default(),
            },
            total,
            recommendation,
            reasons: Vec::new(),
        }
    }

    fn workload(
        lanes: usize,
        steps: usize,
        estimated_lane_work_units: usize,
    ) -> EncodedTraceReplayWorkload {
        EncodedTraceReplayWorkload {
            lanes,
            steps,
            estimated_lane_work_units,
            ..EncodedTraceReplayWorkload::default()
        }
    }

    fn profitability_calibration(
        backend_preferences: &[(&str, f64)],
        penalties: &[(&str, f64)],
    ) -> PlannerCalibration {
        PlannerCalibration {
            schema: "rrtl-pyrtl-planner-calibration-v1".to_string(),
            summary: PlannerCalibrationSummary {
                profitability_backend_preferences: backend_preferences
                    .iter()
                    .map(|(signature, score)| PlannerCalibrationPreference {
                        signature: (*signature).to_string(),
                        score: *score,
                        count: 1,
                    })
                    .collect(),
                profitability_penalties: penalties
                    .iter()
                    .map(|(signature, score)| PlannerCalibrationPreference {
                        signature: (*signature).to_string(),
                        score: *score,
                        count: 1,
                    })
                    .collect(),
                ..PlannerCalibrationSummary::default()
            },
        }
    }

    fn profitability_feature_calibration(
        feature_preferences: &[(&str, f64)],
        feature_penalties: &[(&str, f64)],
    ) -> PlannerCalibration {
        PlannerCalibration {
            schema: "rrtl-pyrtl-planner-calibration-v1".to_string(),
            summary: PlannerCalibrationSummary {
                profitability_feature_preferences: feature_preferences
                    .iter()
                    .map(|(signature, score)| PlannerCalibrationPreference {
                        signature: (*signature).to_string(),
                        score: *score,
                        count: 1,
                    })
                    .collect(),
                profitability_feature_penalties: feature_penalties
                    .iter()
                    .map(|(signature, score)| PlannerCalibrationPreference {
                        signature: (*signature).to_string(),
                        score: *score,
                        count: 1,
                    })
                    .collect(),
                ..PlannerCalibrationSummary::default()
            },
        }
    }
}
