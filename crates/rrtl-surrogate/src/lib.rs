use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MANIFEST_SCHEMA: &str = "rrtl-surrogate-manifest-v1";
pub const DATASET_SCHEMA: &str = "rrtl-surrogate-dataset-v1";
pub const VALIDATION_SCHEMA: &str = "rrtl-surrogate-validation-v1";
pub const GEMM_RESULT_SCHEMA: &str = "rrtl-surrogate-gemm-result-v1";
pub const GEMM_BATCH_SCHEMA: &str = "rrtl-surrogate-gemm-batch-v1";
pub const GEMM_BATCH_RESULT_SCHEMA: &str = "rrtl-surrogate-gemm-batch-result-v1";
pub const GEMM_POLICY_REPORT_SCHEMA: &str = "rrtl-surrogate-gemm-policy-report-v1";
pub const GEMM_FAST_RUN_SCHEMA: &str = "rrtl-surrogate-gemm-fast-run-v1";
pub const GEMM_RUNTIME_PLAN_SCHEMA: &str = "rrtl-surrogate-runtime-plan-v1";
pub const EVENT_CORPUS_SCHEMA: &str = "rrtl-surrogate-instrumentation-corpus-v1";
pub const EVENT_SCHEMA: &str = "rrtl-surrogate-instrumentation-event-v1";
pub const EVENT_EMITTER_CONFIG_SCHEMA: &str = "rrtl-surrogate-event-emitter-config-v1";
pub const RRTL_INSTRUMENTATION_TRACE_SCHEMA: &str = "rrtl-instrumentation-trace-v1";
pub const RRTL_INSTRUMENTATION_TRACE_INSPECTION_SCHEMA: &str =
    "rrtl-instrumentation-trace-inspection-v1";
pub const EVENT_INSPECTION_SCHEMA: &str = "rrtl-surrogate-event-inspection-v1";
pub const EVENT_VALIDATION_SCHEMA: &str = "rrtl-surrogate-event-validation-v1";
pub const EVENT_SHADOW_SCHEMA: &str = "rrtl-surrogate-event-shadow-v1";
pub const EVENT_POLICY_REPORT_SCHEMA: &str = "rrtl-surrogate-event-policy-report-v1";
pub const EVENT_FAST_RUN_SCHEMA: &str = "rrtl-surrogate-event-fast-run-v1";
pub const EVENT_RUNTIME_PLAN_SCHEMA: &str = "rrtl-surrogate-event-runtime-plan-v1";
pub const MODEL_FAST_PLAN_SCHEMA: &str = "rrtl-surrogate-model-fast-plan-v1";
pub const MODEL_FAST_REPORT_SCHEMA: &str = "rrtl-surrogate-model-fast-report-v1";
pub const MODEL_FAST_GOLDEN_SCHEMA: &str = "rrtl-surrogate-model-fast-golden-v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    message: String,
}

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SurrogateManifest {
    pub schema: String,
    pub surrogate_id: String,
    pub surrogate_class: SurrogateClass,
    pub model_family: ModelFamily,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<TaskSpec>,
    pub source: SourceSpec,
    pub artifact: ArtifactSpec,
    pub domain: DomainSpec,
    pub validation: ValidationSpec,
    pub policy: PolicySpec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurrogateClass {
    TransactionKernel,
    EventPredictor,
    MetricPredictor,
    PolicyPredictor,
    Telemetry,
    Stateless,
    StatefulCycle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelFamily {
    GnnTransformer,
    MockGemm,
    RuleBaseline,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskSpec {
    pub prediction_target: String,
    pub input_window_cycles: usize,
    pub horizon_cycles: usize,
    #[serde(default)]
    pub signal_features: Vec<String>,
    #[serde(default)]
    pub program_features: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<LabelSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelSpec {
    pub name: String,
    pub kind: String,
    pub positive_value: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceSpec {
    pub top_name: String,
    pub export_schema: String,
    pub source_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactSpec {
    pub format: ArtifactFormat,
    pub path: String,
    pub sha256: String,
    #[serde(default)]
    pub input_tensors: Vec<String>,
    #[serde(default)]
    pub output_tensors: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opset: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArtifactFormat {
    MockGemm,
    MockEventPredictor,
    Onnx,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainSpec {
    pub rows: usize,
    pub cols: usize,
    pub k_min: usize,
    pub k_max: usize,
    pub data_width: u32,
    pub acc_width: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationSpec {
    pub max_abs_error: i128,
    pub max_mean_abs_error: f64,
    pub max_latency_error_cycles: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicySpec {
    pub mode: PolicyMode,
    pub fallback: FallbackPolicy,
    #[serde(default)]
    pub provenance_tag: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyMode {
    ApproximateWithTolerance,
    ShadowCompare,
    TelemetryOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackPolicy {
    FailClosed,
    ExactFallback,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ManifestValidationReport {
    pub schema: String,
    pub ok: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub surrogate_id: String,
    pub artifact_hash: String,
    pub source_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SurrogateDataset {
    pub schema: String,
    pub source_hash: String,
    pub top_name: String,
    pub graph: GraphDataset,
    pub trace: TraceDataset,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphDataset {
    pub node_features: Vec<GraphNodeFeature>,
    pub edge_index: Vec<[usize; 2]>,
    pub edge_features: Vec<GraphEdgeFeature>,
    pub metadata: GraphMetadata,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphNodeFeature {
    pub name: String,
    pub kind: String,
    pub bitwidth: u32,
    pub value: Option<i128>,
    pub reset_value: Option<i128>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphEdgeFeature {
    pub net_index: usize,
    pub op: String,
    pub role: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphMetadata {
    pub export_schema: String,
    pub net_count: usize,
    pub memory_count: usize,
    pub input_count: usize,
    pub output_count: usize,
    pub register_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceDataset {
    pub schema: String,
    pub steps: Vec<TraceDatasetStep>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceDatasetStep {
    pub inputs: BTreeMap<String, u128>,
    pub outputs: BTreeMap<String, u128>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmTransaction {
    pub schema: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane: Option<usize>,
    pub rows: usize,
    pub cols: usize,
    pub k: usize,
    pub a: Vec<Vec<i64>>,
    pub w: Vec<Vec<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_c: Option<Vec<Vec<i128>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_latency_cycles: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmBatch {
    pub schema: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_hash: Option<String>,
    pub transactions: Vec<GemmTransaction>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstrumentationEventCorpus {
    pub schema: String,
    pub source_hash: String,
    pub top_name: String,
    pub events: Vec<InstrumentationEvent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstrumentationEvent {
    pub schema: String,
    pub sample_id: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane: Option<usize>,
    pub target: String,
    pub window_cycles: usize,
    pub horizon_cycles: usize,
    pub program: BTreeMap<String, i64>,
    pub signals: Vec<BTreeMap<String, i64>>,
    pub label: BTreeMap<String, i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventEmitterConfig {
    pub schema: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<String>,
    pub top_name: String,
    pub source_hash: String,
    pub target: String,
    pub window_cycles: usize,
    pub horizon_cycles: usize,
    pub signal_features: Vec<EventFeatureMapping>,
    #[serde(default)]
    pub program_features: Vec<EventFeatureMapping>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane: Option<EventFeatureMapping>,
    pub label: EventFeatureMapping,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventFeatureMapping {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<EventTraceSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constant: Option<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventTraceSource {
    Inputs,
    Outputs,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RrtlInstrumentationTrace {
    pub schema: String,
    pub top_name: String,
    pub source_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub program_id: Option<String>,
    pub steps: Vec<RrtlInstrumentationStep>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RrtlInstrumentationStep {
    pub cycle: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane: Option<usize>,
    #[serde(default)]
    pub signals: BTreeMap<String, i64>,
    #[serde(default)]
    pub program: BTreeMap<String, i64>,
    #[serde(default)]
    pub labels: BTreeMap<String, i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RrtlInstrumentationTraceInspectionReport {
    pub schema: String,
    pub ok: bool,
    pub top_name: String,
    pub source_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub program_id: Option<String>,
    pub steps: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cycle_min: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cycle_max: Option<usize>,
    pub cycle_monotonic: bool,
    pub lanes: Vec<usize>,
    pub lane_count: usize,
    pub steps_with_lane: usize,
    pub signal_fields: Vec<String>,
    pub program_fields: Vec<String>,
    pub label_fields: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compatibility: Option<RrtlInstrumentationTraceCompatibilityReport>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RrtlInstrumentationTraceCompatibilityReport {
    pub target: String,
    pub window_cycles: usize,
    pub horizon_cycles: usize,
    pub emittable_samples: usize,
    pub missing_fields: Vec<RrtlInstrumentationMissingField>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RrtlInstrumentationMissingField {
    pub sample_id: usize,
    pub step_index: usize,
    pub source: String,
    pub field: String,
    pub feature: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmRunResult {
    pub schema: String,
    pub surrogate_id: String,
    pub ok: bool,
    pub c: Vec<Vec<i128>>,
    pub telemetry: GemmTelemetry,
    pub metrics: Option<GemmValidationMetrics>,
    pub provenance: Provenance,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmBatchRunReport {
    pub schema: String,
    pub ok: bool,
    pub count: usize,
    pub total_lanes: usize,
    pub metrics: GemmBatchMetrics,
    pub lanes: Vec<GemmBatchLaneSummary>,
    pub results: Vec<GemmBatchItemResult>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GemmBatchMetrics {
    pub max_abs_error: i128,
    pub max_mean_abs_error: f64,
    pub max_latency_error_cycles: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmBatchLaneSummary {
    pub lane: usize,
    pub count: usize,
    pub ok: usize,
    pub failed: usize,
    pub metrics: GemmBatchMetrics,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmBatchItemResult {
    pub index: usize,
    pub lane: usize,
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<GemmRunResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmPolicyReport {
    pub schema: String,
    pub ok: bool,
    pub count: usize,
    pub used_surrogate: usize,
    pub exact_fallbacks: usize,
    pub fail_closed: usize,
    pub results: Vec<GemmPolicyItemResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmPolicyItemResult {
    pub index: usize,
    pub lane: usize,
    pub decision: GemmPolicyDecision,
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surrogate_result: Option<GemmRunResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_result: Option<GemmRunResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmFastRunReport {
    pub schema: String,
    pub ok: bool,
    pub count: usize,
    pub total_lanes: usize,
    pub surrogate_replacements: usize,
    pub exact_fallbacks: usize,
    pub fail_closed: usize,
    pub shadow_compared: usize,
    pub shadow_passed: usize,
    pub shadow_failed: usize,
    pub workers: Vec<GemmFastRunWorkerSummary>,
    pub lanes: Vec<GemmFastRunLaneSummary>,
    pub results: Vec<GemmFastRunItem>,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GemmFastRunOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_sample_stride: Option<usize>,
    #[serde(default)]
    pub shadow_sample_offset: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmFastRunWorkerSummary {
    pub worker_id: String,
    pub start_lane: usize,
    pub lanes: usize,
    pub assigned_items: usize,
    pub surrogate_replacements: usize,
    pub exact_fallbacks: usize,
    pub fail_closed: usize,
    pub shadow_compared: usize,
    pub shadow_failed: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmFastRunLaneSummary {
    pub lane: usize,
    pub count: usize,
    pub surrogate_replacements: usize,
    pub exact_fallbacks: usize,
    pub fail_closed: usize,
    pub shadow_compared: usize,
    pub shadow_failed: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmFastRunItem {
    pub index: usize,
    pub lane: usize,
    pub decision: GemmPolicyDecision,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_result: Option<GemmRuntimeSourceResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<GemmRunResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_ok: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_max_abs_error: Option<i128>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_latency_error_cycles: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_first_divergence: Option<GemmDivergence>,
    #[serde(default)]
    pub shadow_sampled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmPolicyDecision {
    SurrogateUsed,
    ExactFallback,
    FailClosed,
    ShadowCompare,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmRuntimePlan {
    pub schema: String,
    pub ok: bool,
    pub total_lanes: usize,
    pub workers: Vec<GemmRuntimeWorkerSummary>,
    pub items: Vec<GemmRuntimePlanItem>,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmRuntimeWorkerSummary {
    pub worker_id: String,
    pub start_lane: usize,
    pub lanes: usize,
    pub assigned_items: usize,
    pub used_surrogate: usize,
    pub exact_fallbacks: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GemmRuntimeWorkerSpec {
    pub worker_id: String,
    pub start_lane: usize,
    pub lanes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmRuntimePlanItem {
    pub index: usize,
    pub lane: usize,
    pub worker_id: String,
    pub decision: GemmPolicyDecision,
    pub provenance: Provenance,
    pub source_result: GemmRuntimeSourceResult,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_ok: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_max_abs_error: Option<i128>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_latency_error_cycles: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_first_divergence: Option<GemmDivergence>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmRuntimeSourceResult {
    Surrogate,
    Exact,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventPolicyReport {
    pub schema: String,
    pub ok: bool,
    pub count: usize,
    pub used_surrogate: usize,
    pub exact_fallbacks: usize,
    pub fail_closed: usize,
    pub results: Vec<EventPolicyItemResult>,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventPolicyItemResult {
    pub index: usize,
    pub sample_id: usize,
    pub lane: usize,
    pub target: String,
    pub decision: EventPolicyDecision,
    pub ok: bool,
    pub predicted: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<i64>,
    pub provenance: Provenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventFastRunReport {
    pub schema: String,
    pub ok: bool,
    pub count: usize,
    pub total_lanes: usize,
    pub surrogate_replacements: usize,
    pub exact_fallbacks: usize,
    pub fail_closed: usize,
    pub shadow_compared: usize,
    pub shadow_passed: usize,
    pub shadow_failed: usize,
    pub workers: Vec<EventFastRunWorkerSummary>,
    pub lanes: Vec<EventFastRunLaneSummary>,
    pub results: Vec<EventFastRunItem>,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EventFastRunOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_sample_stride: Option<usize>,
    #[serde(default)]
    pub shadow_sample_offset: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventFastRunWorkerSummary {
    pub worker_id: String,
    pub start_lane: usize,
    pub lanes: usize,
    pub assigned_items: usize,
    pub surrogate_replacements: usize,
    pub exact_fallbacks: usize,
    pub fail_closed: usize,
    pub shadow_compared: usize,
    pub shadow_failed: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventFastRunLaneSummary {
    pub lane: usize,
    pub count: usize,
    pub surrogate_replacements: usize,
    pub exact_fallbacks: usize,
    pub fail_closed: usize,
    pub shadow_compared: usize,
    pub shadow_failed: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventFastRunItem {
    pub index: usize,
    pub sample_id: usize,
    pub lane: usize,
    pub target: String,
    pub decision: EventPolicyDecision,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_result: Option<GemmRuntimeSourceResult>,
    pub predicted: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<i64>,
    pub provenance: Provenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_ok: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_error: Option<String>,
    #[serde(default)]
    pub shadow_sampled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFastPlan {
    pub schema: String,
    #[serde(default)]
    pub ops: Vec<ModelFastOp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thresholds: Option<ModelFastThresholds>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelFastThresholds {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_op_coverage: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_item_coverage: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_fallback_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_shadow_sample_ratio: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFastOp {
    pub op_id: String,
    pub op_kind: String,
    pub name: String,
    pub fast_report_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub golden_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fast_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFastReport {
    pub schema: String,
    pub ok: bool,
    pub op_count: usize,
    pub totals: ModelFastTotals,
    pub coverage: ModelFastCoverage,
    pub timing: ModelFastTimingSummary,
    pub ops: Vec<ModelFastOpReport>,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelFastCoverage {
    pub op_coverage: f64,
    pub item_coverage: f64,
    pub fallback_ratio: f64,
    pub shadow_sample_ratio: f64,
    pub accepted: bool,
    pub reject_reasons: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelFastTotals {
    pub items: usize,
    pub surrogate_replacements: usize,
    pub exact_fallbacks: usize,
    pub fail_closed: usize,
    pub shadow_compared: usize,
    pub shadow_passed: usize,
    pub shadow_failed: usize,
    pub shadow_sampled: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFastOpReport {
    pub op_id: String,
    pub op_kind: String,
    pub name: String,
    pub fast_report_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub ok: bool,
    pub totals: ModelFastTotals,
    pub provenance: Vec<ModelFastProvenanceSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub golden: Option<ModelFastGoldenComparison>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing: Option<ModelFastOpTiming>,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelFastTimingSummary {
    pub timed_ops: usize,
    pub exact_ns: u64,
    pub fast_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speedup: Option<f64>,
    pub missing_timing_ops: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFastOpTiming {
    pub exact_ns: u64,
    pub fast_ns: u64,
    pub speedup: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFastGoldenComparison {
    pub golden_compared: bool,
    pub golden_ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub golden_error: Option<String>,
    #[serde(default)]
    pub tensor_compared: bool,
    #[serde(default)]
    pub tensor_count: usize,
    #[serde(default)]
    pub max_abs_error: i128,
    #[serde(default)]
    pub tensor_errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFastGolden {
    pub schema: String,
    pub op_id: String,
    pub op_kind: String,
    pub expected: ModelFastGoldenExpected,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_tensors: Option<BTreeMap<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_tensors: Option<BTreeMap<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_abs_error: Option<i128>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFastGoldenExpected {
    pub items: usize,
    pub surrogate_replacements: usize,
    pub exact_fallbacks: usize,
    pub fail_closed: usize,
    pub shadow_sampled: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFastProvenanceSummary {
    pub tag: String,
    pub exact: bool,
    pub surrogate_id: String,
    pub model_family: ModelFamily,
    pub artifact_format: ArtifactFormat,
    pub artifact_hash: String,
    pub source_hash: String,
    pub policy: PolicyMode,
    pub count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventPolicyDecision {
    SurrogateUsed,
    ExactFallback,
    FailClosed,
    ShadowCompare,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventRuntimePlan {
    pub schema: String,
    pub ok: bool,
    pub total_lanes: usize,
    pub workers: Vec<EventRuntimeWorkerSummary>,
    pub items: Vec<EventRuntimePlanItem>,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventRuntimeWorkerSummary {
    pub worker_id: String,
    pub start_lane: usize,
    pub lanes: usize,
    pub assigned_items: usize,
    pub used_surrogate: usize,
    pub exact_fallbacks: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventRuntimePlanItem {
    pub index: usize,
    pub sample_id: usize,
    pub lane: usize,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub target: String,
    pub worker_id: String,
    pub decision: EventPolicyDecision,
    pub provenance: Provenance,
    pub source_result: GemmRuntimeSourceResult,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_ok: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmTelemetry {
    pub latency_cycles: u64,
    pub active_cycles: u64,
    pub utilization: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmValidationMetrics {
    pub max_abs_error: i128,
    pub mean_abs_error: f64,
    pub latency_error_cycles: u64,
    pub first_divergence: Option<GemmDivergence>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GemmDivergence {
    pub row: usize,
    pub col: usize,
    pub expected: i128,
    pub actual: i128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Provenance {
    pub tag: String,
    pub exact: bool,
    pub surrogate_id: String,
    pub model_family: ModelFamily,
    pub artifact_format: ArtifactFormat,
    pub artifact_hash: String,
    pub source_hash: String,
    pub policy: PolicyMode,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SurrogateValidationReport {
    pub schema: String,
    pub ok: bool,
    pub manifest: ManifestValidationReport,
    pub dataset: DatasetSummary,
    pub run: Option<GemmRunResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetSummary {
    pub source_hash: String,
    pub top_name: String,
    pub nodes: usize,
    pub edges: usize,
    pub steps: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventValidationReport {
    pub schema: String,
    pub ok: bool,
    pub errors: Vec<String>,
    pub manifest: ManifestValidationReport,
    pub corpus: EventCorpusSummary,
    pub metrics: EventValidationMetrics,
    pub first_mismatch: Option<EventMismatch>,
    pub provenance: Provenance,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventShadowReport {
    pub schema: String,
    pub ok: bool,
    pub errors: Vec<String>,
    pub manifest: ManifestValidationReport,
    pub corpus: EventCorpusSummary,
    pub total_lanes: usize,
    pub metrics: EventValidationMetrics,
    pub lanes: Vec<EventLaneShadowSummary>,
    pub first_mismatch: Option<EventMismatch>,
    pub results: Vec<EventShadowSample>,
    pub provenance: Provenance,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventLaneShadowSummary {
    pub lane: usize,
    pub samples: usize,
    pub metrics: EventValidationMetrics,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventShadowSample {
    pub sample_id: usize,
    pub lane: usize,
    pub expected: i64,
    pub predicted: i64,
    pub ok: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventCorpusSummary {
    pub source_hash: String,
    pub top_name: String,
    pub target: String,
    pub samples: usize,
    pub window_cycles: usize,
    pub horizon_cycles: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventInspectionReport {
    pub schema: String,
    pub ok: bool,
    pub errors: Vec<String>,
    pub corpus: EventCorpusSummary,
    pub targets: Vec<String>,
    pub signal_features: Vec<String>,
    pub program_features: Vec<String>,
    pub labels: Vec<String>,
    pub positive_labels: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EventValidationMetrics {
    pub accuracy: f64,
    pub false_positive: usize,
    pub false_negative: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventMismatch {
    pub sample_id: usize,
    pub expected: i64,
    pub predicted: i64,
}

pub fn read_manifest(mut reader: impl Read) -> Result<SurrogateManifest, Error> {
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|err| Error::new(format!("failed to read manifest: {err}")))?;
    serde_json::from_str(&text)
        .map_err(|err| Error::new(format!("failed to parse manifest: {err}")))
}

pub fn read_event_corpus(mut reader: impl Read) -> Result<InstrumentationEventCorpus, Error> {
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|err| Error::new(format!("failed to read event corpus: {err}")))?;
    serde_json::from_str(&text)
        .map_err(|err| Error::new(format!("failed to parse event corpus: {err}")))
}

pub fn read_event_emitter_config(mut reader: impl Read) -> Result<EventEmitterConfig, Error> {
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|err| Error::new(format!("failed to read event emitter config: {err}")))?;
    serde_json::from_str(&text)
        .map_err(|err| Error::new(format!("failed to parse event emitter config: {err}")))
}

pub fn read_rrtl_instrumentation_trace(
    mut reader: impl Read,
) -> Result<RrtlInstrumentationTrace, Error> {
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|err| Error::new(format!("failed to read RRTL instrumentation trace: {err}")))?;
    serde_json::from_str(&text)
        .map_err(|err| Error::new(format!("failed to parse RRTL instrumentation trace: {err}")))
}

pub fn emit_event_corpus(
    trace: &serde_json::Value,
    config: &EventEmitterConfig,
) -> Result<InstrumentationEventCorpus, Error> {
    validate_event_emitter_config(config)?;
    let steps = required_array(trace, "steps")?;
    let required_len = config.window_cycles + config.horizon_cycles;
    let sample_count = if steps.len() < required_len {
        0
    } else {
        steps.len() - required_len + 1
    };
    let mut events = Vec::with_capacity(sample_count);

    for sample_id in 0..sample_count {
        let mut signals = Vec::with_capacity(config.window_cycles);
        for offset in 0..config.window_cycles {
            let step = &steps[sample_id + offset];
            let mut features = BTreeMap::new();
            for mapping in &config.signal_features {
                features.insert(mapping.name.clone(), resolve_feature(step, mapping)?);
            }
            signals.push(features);
        }

        let label_step = &steps[sample_id + config.window_cycles + config.horizon_cycles - 1];
        let mut label = BTreeMap::new();
        label.insert(
            config.label.name.clone(),
            resolve_feature(label_step, &config.label)?,
        );

        let mut program = BTreeMap::new();
        let program_step = &steps[sample_id];
        for mapping in &config.program_features {
            program.insert(
                mapping.name.clone(),
                resolve_feature(program_step, mapping)?,
            );
        }
        let lane = config
            .lane
            .as_ref()
            .map(|mapping| resolve_lane(program_step, mapping))
            .transpose()?;

        events.push(InstrumentationEvent {
            schema: EVENT_SCHEMA.to_string(),
            sample_id,
            lane,
            target: config.target.clone(),
            window_cycles: config.window_cycles,
            horizon_cycles: config.horizon_cycles,
            program,
            signals,
            label,
        });
    }

    Ok(InstrumentationEventCorpus {
        schema: EVENT_CORPUS_SCHEMA.to_string(),
        source_hash: config.source_hash.clone(),
        top_name: config.top_name.clone(),
        events,
    })
}

pub fn emit_instrumented_event_corpus(
    trace: &RrtlInstrumentationTrace,
    config: &EventEmitterConfig,
) -> Result<InstrumentationEventCorpus, Error> {
    if trace.schema != RRTL_INSTRUMENTATION_TRACE_SCHEMA {
        return Err(Error::new(format!(
            "unsupported RRTL instrumentation trace schema `{}`",
            trace.schema
        )));
    }
    let normalized = normalize_rrtl_instrumentation_trace(trace);
    emit_event_corpus(&normalized, config)
}

pub fn normalize_rrtl_instrumentation_trace(trace: &RrtlInstrumentationTrace) -> serde_json::Value {
    let steps: Vec<serde_json::Value> = trace
        .steps
        .iter()
        .map(|step| {
            let mut inputs = serde_json::Map::new();
            let mut outputs = serde_json::Map::new();
            inputs.insert("cycle".to_string(), serde_json::json!(step.cycle));
            outputs.insert("cycle".to_string(), serde_json::json!(step.cycle));
            if let Some(lane) = step.lane {
                inputs.insert("lane".to_string(), serde_json::json!(lane));
                outputs.insert("lane".to_string(), serde_json::json!(lane));
            }
            for (name, value) in &step.signals {
                inputs.insert(name.clone(), serde_json::json!(value));
                outputs.insert(name.clone(), serde_json::json!(value));
            }
            for (name, value) in &step.program {
                inputs.insert(name.clone(), serde_json::json!(value));
            }
            for (name, value) in &step.labels {
                outputs.insert(name.clone(), serde_json::json!(value));
            }
            serde_json::json!({
                "inputs": inputs,
                "outputs": outputs,
            })
        })
        .collect();

    serde_json::json!({
        "schema": "rrtl-pyrtl-trace-v1",
        "top_name": trace.top_name,
        "source_hash": trace.source_hash,
        "program_id": trace.program_id,
        "steps": steps,
    })
}

pub fn inspect_rrtl_instrumentation_trace(
    trace: &RrtlInstrumentationTrace,
    config: Option<&EventEmitterConfig>,
) -> RrtlInstrumentationTraceInspectionReport {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    if trace.schema != RRTL_INSTRUMENTATION_TRACE_SCHEMA {
        errors.push(format!(
            "unsupported RRTL instrumentation trace schema `{}`",
            trace.schema
        ));
    }
    if trace.top_name.trim().is_empty() {
        errors.push("RRTL instrumentation trace top_name must not be empty".to_string());
    }
    if trace.source_hash.trim().is_empty() {
        errors.push("RRTL instrumentation trace source_hash must not be empty".to_string());
    }
    if trace.steps.is_empty() {
        errors.push("RRTL instrumentation trace must contain at least one step".to_string());
    }

    let mut lanes = BTreeSet::new();
    let mut steps_with_lane = 0usize;
    let mut signal_fields = BTreeSet::new();
    let mut program_fields = BTreeSet::new();
    let mut label_fields = BTreeSet::new();
    let mut cycle_min = None::<usize>;
    let mut cycle_max = None::<usize>;
    let mut previous_cycle = None::<usize>;
    let mut cycle_monotonic = true;
    for step in &trace.steps {
        cycle_min = Some(cycle_min.map_or(step.cycle, |value| value.min(step.cycle)));
        cycle_max = Some(cycle_max.map_or(step.cycle, |value| value.max(step.cycle)));
        if previous_cycle.is_some_and(|previous| step.cycle < previous) {
            cycle_monotonic = false;
        }
        previous_cycle = Some(step.cycle);
        if let Some(lane) = step.lane {
            lanes.insert(lane);
            steps_with_lane += 1;
        }
        signal_fields.extend(step.signals.keys().cloned());
        program_fields.extend(step.program.keys().cloned());
        label_fields.extend(step.labels.keys().cloned());
    }
    if !cycle_monotonic {
        warnings.push("RRTL instrumentation trace cycles are not monotonic".to_string());
    }
    if label_fields.is_empty() {
        warnings.push("RRTL instrumentation trace has no label fields".to_string());
    }

    let compatibility = config
        .map(|config| inspect_rrtl_instrumentation_trace_compatibility(trace, config, &mut errors));

    RrtlInstrumentationTraceInspectionReport {
        schema: RRTL_INSTRUMENTATION_TRACE_INSPECTION_SCHEMA.to_string(),
        ok: errors.is_empty(),
        top_name: trace.top_name.clone(),
        source_hash: trace.source_hash.clone(),
        program_id: trace.program_id.clone(),
        steps: trace.steps.len(),
        cycle_min,
        cycle_max,
        cycle_monotonic,
        lanes: lanes.iter().copied().collect(),
        lane_count: lanes.len(),
        steps_with_lane,
        signal_fields: signal_fields.into_iter().collect(),
        program_fields: program_fields.into_iter().collect(),
        label_fields: label_fields.into_iter().collect(),
        compatibility,
        warnings,
        errors,
    }
}

fn inspect_rrtl_instrumentation_trace_compatibility(
    trace: &RrtlInstrumentationTrace,
    config: &EventEmitterConfig,
    errors: &mut Vec<String>,
) -> RrtlInstrumentationTraceCompatibilityReport {
    if let Err(err) = validate_event_emitter_config(config) {
        errors.push(err.to_string());
    }
    let required_len = config.window_cycles.saturating_add(config.horizon_cycles);
    let emittable_samples = if required_len == 0 || trace.steps.len() < required_len {
        0
    } else {
        trace.steps.len() - required_len + 1
    };
    if emittable_samples == 0 {
        errors.push(format!(
            "RRTL instrumentation trace has {} steps, fewer than required window+horizon {}",
            trace.steps.len(),
            required_len
        ));
    }

    let normalized = normalize_rrtl_instrumentation_trace(trace);
    let steps = normalized["steps"].as_array().cloned().unwrap_or_default();
    let mut missing_fields = Vec::new();
    for sample_id in 0..emittable_samples {
        for offset in 0..config.window_cycles {
            let step_index = sample_id + offset;
            if let Some(step) = steps.get(step_index) {
                for mapping in &config.signal_features {
                    collect_missing_instrumentation_mapping(
                        sample_id,
                        step_index,
                        step,
                        mapping,
                        &mut missing_fields,
                    );
                }
            }
        }
        let program_step_index = sample_id;
        if let Some(step) = steps.get(program_step_index) {
            for mapping in &config.program_features {
                collect_missing_instrumentation_mapping(
                    sample_id,
                    program_step_index,
                    step,
                    mapping,
                    &mut missing_fields,
                );
            }
            if let Some(mapping) = &config.lane {
                collect_missing_instrumentation_mapping(
                    sample_id,
                    program_step_index,
                    step,
                    mapping,
                    &mut missing_fields,
                );
            }
        }
        let label_step_index = sample_id + config.window_cycles + config.horizon_cycles - 1;
        if let Some(step) = steps.get(label_step_index) {
            collect_missing_instrumentation_mapping(
                sample_id,
                label_step_index,
                step,
                &config.label,
                &mut missing_fields,
            );
        }
    }
    for missing in &missing_fields {
        errors.push(format!(
            "sample {} step {} missing {} field `{}` for feature `{}`",
            missing.sample_id, missing.step_index, missing.source, missing.field, missing.feature
        ));
    }

    RrtlInstrumentationTraceCompatibilityReport {
        target: config.target.clone(),
        window_cycles: config.window_cycles,
        horizon_cycles: config.horizon_cycles,
        emittable_samples,
        missing_fields,
    }
}

fn collect_missing_instrumentation_mapping(
    sample_id: usize,
    step_index: usize,
    step: &serde_json::Value,
    mapping: &EventFeatureMapping,
    missing_fields: &mut Vec<RrtlInstrumentationMissingField>,
) {
    if mapping.constant.is_some() {
        return;
    }
    let Some(source) = mapping.source else {
        return;
    };
    let Some(field) = mapping.field.as_ref() else {
        return;
    };
    let source_name = match source {
        EventTraceSource::Inputs => "inputs",
        EventTraceSource::Outputs => "outputs",
    };
    let has_field = step
        .get(source_name)
        .and_then(|value| value.as_object())
        .is_some_and(|object| object.contains_key(field));
    if !has_field {
        missing_fields.push(RrtlInstrumentationMissingField {
            sample_id,
            step_index,
            source: source_name.to_string(),
            field: field.clone(),
            feature: mapping.name.clone(),
        });
    }
}

pub fn inspect_event_corpus(corpus: &InstrumentationEventCorpus) -> EventInspectionReport {
    let mut errors = Vec::new();
    if corpus.schema != EVENT_CORPUS_SCHEMA {
        errors.push(format!(
            "unsupported event corpus schema `{}`",
            corpus.schema
        ));
    }
    if corpus.top_name.trim().is_empty() {
        errors.push("event corpus top_name must not be empty".to_string());
    }
    if corpus.source_hash.trim().is_empty() {
        errors.push("event corpus source_hash must not be empty".to_string());
    }
    if corpus.events.is_empty() {
        errors.push("event corpus must contain at least one event".to_string());
    }

    let mut targets = BTreeSet::new();
    let mut signal_features = BTreeSet::new();
    let mut program_features = BTreeSet::new();
    let mut labels = BTreeSet::new();
    let mut positive_labels = BTreeMap::new();
    let mut window_cycles = 0usize;
    let mut horizon_cycles = 0usize;

    for (index, event) in corpus.events.iter().enumerate() {
        if index == 0 {
            window_cycles = event.window_cycles;
            horizon_cycles = event.horizon_cycles;
        } else {
            if event.window_cycles != window_cycles {
                errors.push(format!(
                    "event {} window {} does not match first event window {}",
                    event.sample_id, event.window_cycles, window_cycles
                ));
            }
            if event.horizon_cycles != horizon_cycles {
                errors.push(format!(
                    "event {} horizon {} does not match first event horizon {}",
                    event.sample_id, event.horizon_cycles, horizon_cycles
                ));
            }
        }
        if event.schema != EVENT_SCHEMA {
            errors.push(format!(
                "event {} has unsupported schema `{}`",
                event.sample_id, event.schema
            ));
        }
        if event.target.trim().is_empty() {
            errors.push(format!(
                "event {} target must not be empty",
                event.sample_id
            ));
        }
        if event.signals.len() != event.window_cycles {
            errors.push(format!(
                "event {} has {} signal steps, expected {}",
                event.sample_id,
                event.signals.len(),
                event.window_cycles
            ));
        }
        targets.insert(event.target.clone());
        for step in &event.signals {
            for feature in step.keys() {
                signal_features.insert(feature.clone());
            }
        }
        for feature in event.program.keys() {
            program_features.insert(feature.clone());
        }
        for (label, value) in &event.label {
            labels.insert(label.clone());
            if *value == 1 {
                *positive_labels.entry(label.clone()).or_insert(0) += 1;
            } else if *value != 0 {
                errors.push(format!(
                    "event {} label `{}` must be binary, got {}",
                    event.sample_id, label, value
                ));
            }
        }
        if event.label.is_empty() {
            errors.push(format!("event {} has no labels", event.sample_id));
        }
    }

    EventInspectionReport {
        schema: EVENT_INSPECTION_SCHEMA.to_string(),
        ok: errors.is_empty(),
        errors,
        corpus: EventCorpusSummary {
            source_hash: corpus.source_hash.clone(),
            top_name: corpus.top_name.clone(),
            target: if targets.len() == 1 {
                targets.iter().next().cloned().unwrap_or_default()
            } else {
                String::new()
            },
            samples: corpus.events.len(),
            window_cycles,
            horizon_cycles,
        },
        targets: targets.into_iter().collect(),
        signal_features: signal_features.into_iter().collect(),
        program_features: program_features.into_iter().collect(),
        labels: labels.into_iter().collect(),
        positive_labels,
    }
}

pub fn read_gemm_transaction(mut reader: impl Read) -> Result<GemmTransaction, Error> {
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|err| Error::new(format!("failed to read transaction: {err}")))?;
    serde_json::from_str(&text)
        .map_err(|err| Error::new(format!("failed to parse transaction: {err}")))
}

pub fn read_gemm_batch(mut reader: impl Read) -> Result<GemmBatch, Error> {
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|err| Error::new(format!("failed to read GEMM batch: {err}")))?;
    serde_json::from_str(&text)
        .map_err(|err| Error::new(format!("failed to parse GEMM batch: {err}")))
}

pub fn read_gemm_policy_report(mut reader: impl Read) -> Result<GemmPolicyReport, Error> {
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|err| Error::new(format!("failed to read GEMM policy report: {err}")))?;
    serde_json::from_str(&text)
        .map_err(|err| Error::new(format!("failed to parse GEMM policy report: {err}")))
}

pub fn read_gemm_fast_run_report(mut reader: impl Read) -> Result<GemmFastRunReport, Error> {
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|err| Error::new(format!("failed to read GEMM FAST report: {err}")))?;
    serde_json::from_str(&text)
        .map_err(|err| Error::new(format!("failed to parse GEMM FAST report: {err}")))
}

pub fn read_event_policy_report(mut reader: impl Read) -> Result<EventPolicyReport, Error> {
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|err| Error::new(format!("failed to read event policy report: {err}")))?;
    serde_json::from_str(&text)
        .map_err(|err| Error::new(format!("failed to parse event policy report: {err}")))
}

pub fn read_event_fast_run_report(mut reader: impl Read) -> Result<EventFastRunReport, Error> {
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|err| Error::new(format!("failed to read event FAST report: {err}")))?;
    serde_json::from_str(&text)
        .map_err(|err| Error::new(format!("failed to parse event FAST report: {err}")))
}

pub fn read_event_runtime_plan(mut reader: impl Read) -> Result<EventRuntimePlan, Error> {
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|err| Error::new(format!("failed to read event runtime plan: {err}")))?;
    serde_json::from_str(&text)
        .map_err(|err| Error::new(format!("failed to parse event runtime plan: {err}")))
}

pub fn read_model_fast_plan(mut reader: impl Read) -> Result<ModelFastPlan, Error> {
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|err| Error::new(format!("failed to read model FAST plan: {err}")))?;
    serde_json::from_str(&text)
        .map_err(|err| Error::new(format!("failed to parse model FAST plan: {err}")))
}

pub fn read_model_fast_golden(mut reader: impl Read) -> Result<ModelFastGolden, Error> {
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|err| Error::new(format!("failed to read model FAST golden: {err}")))?;
    serde_json::from_str(&text)
        .map_err(|err| Error::new(format!("failed to parse model FAST golden: {err}")))
}

pub fn infer_model_fast_op_kind(mut reader: impl Read) -> Result<String, Error> {
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|err| Error::new(format!("failed to read FAST report: {err}")))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|err| Error::new(format!("failed to parse FAST report: {err}")))?;
    let schema = value
        .get("schema")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::new("FAST report missing string schema"))?;
    match schema {
        GEMM_FAST_RUN_SCHEMA => Ok("gemm".to_string()),
        EVENT_FAST_RUN_SCHEMA => Ok("event".to_string()),
        other => Err(Error::new(format!(
            "unsupported model FAST op report schema `{other}`"
        ))),
    }
}

pub fn validate_manifest_path(
    manifest: &SurrogateManifest,
    manifest_path: impl AsRef<Path>,
) -> ManifestValidationReport {
    let base = manifest_path
        .as_ref()
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    validate_manifest(manifest, &base)
}

pub fn validate_manifest(
    manifest: &SurrogateManifest,
    base_dir: impl AsRef<Path>,
) -> ManifestValidationReport {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    if manifest.schema != MANIFEST_SCHEMA {
        errors.push(format!("unsupported manifest schema `{}`", manifest.schema));
    }
    if manifest.surrogate_id.trim().is_empty() {
        errors.push("surrogate_id must not be empty".to_string());
    }
    match manifest.surrogate_class {
        SurrogateClass::TransactionKernel => validate_transaction_kernel_manifest(manifest, &mut errors),
        SurrogateClass::EventPredictor => validate_event_predictor_manifest(manifest, &mut errors),
        other => warnings.push(format!(
            "manifest-only validation for surrogate class {:?}; runtime execution is not implemented",
            other
        )),
    }
    if manifest.model_family == ModelFamily::GnnTransformer
        && manifest.artifact.format != ArtifactFormat::Onnx
    {
        errors.push("gnn-transformer manifests must use an onnx artifact".to_string());
    }
    if manifest.artifact.format == ArtifactFormat::Onnx
        && manifest.surrogate_class == SurrogateClass::TransactionKernel
    {
        require_tensor_names(
            "input",
            &manifest.artifact.input_tensors,
            &["gemm_descriptor", "a_tensor", "w_tensor"],
            &mut errors,
        );
        require_tensor_names(
            "output",
            &manifest.artifact.output_tensors,
            &["c_tensor", "telemetry"],
            &mut errors,
        );
    }
    if manifest.artifact.format == ArtifactFormat::Onnx
        && manifest.surrogate_class == SurrogateClass::EventPredictor
    {
        require_tensor_names(
            "input",
            &manifest.artifact.input_tensors,
            &["signal_window", "program_context"],
            &mut errors,
        );
        require_tensor_names(
            "output",
            &manifest.artifact.output_tensors,
            &["event_probability", "predicted_event"],
            &mut errors,
        );
    }
    if manifest.validation.max_abs_error < 0 {
        errors.push("validation.max_abs_error must be non-negative".to_string());
    }
    if manifest.validation.max_mean_abs_error < 0.0 {
        errors.push("validation.max_mean_abs_error must be non-negative".to_string());
    }

    let artifact_path = resolve_artifact_path(&base_dir, &manifest.artifact.path);
    let artifact_hash = match fs::read(&artifact_path) {
        Ok(bytes) => {
            let hash = sha256_hex(&bytes);
            if !manifest.artifact.sha256.eq_ignore_ascii_case(&hash) {
                errors.push(format!(
                    "artifact sha256 mismatch for `{}`: manifest {}, actual {}",
                    artifact_path.display(),
                    manifest.artifact.sha256,
                    hash
                ));
            }
            hash
        }
        Err(err) => {
            errors.push(format!(
                "artifact `{}` is not readable: {err}",
                artifact_path.display()
            ));
            String::new()
        }
    };

    ManifestValidationReport {
        schema: "rrtl-surrogate-manifest-validation-v1".to_string(),
        ok: errors.is_empty(),
        errors,
        warnings,
        surrogate_id: manifest.surrogate_id.clone(),
        artifact_hash,
        source_hash: manifest.source.source_hash.clone(),
    }
}

fn validate_transaction_kernel_manifest(manifest: &SurrogateManifest, errors: &mut Vec<String>) {
    if manifest.domain.rows == 0 || manifest.domain.cols == 0 {
        errors.push("domain rows and cols must be greater than zero".to_string());
    }
    if manifest.domain.k_min == 0 || manifest.domain.k_max < manifest.domain.k_min {
        errors.push("domain must satisfy 0 < k_min <= k_max".to_string());
    }
}

fn validate_event_predictor_manifest(manifest: &SurrogateManifest, errors: &mut Vec<String>) {
    let Some(task) = &manifest.task else {
        errors.push("event_predictor manifests require a task section".to_string());
        return;
    };
    if task.prediction_target.trim().is_empty() {
        errors.push("task.prediction_target must not be empty".to_string());
    }
    if task.input_window_cycles == 0 {
        errors.push("task.input_window_cycles must be greater than zero".to_string());
    }
    if task.signal_features.is_empty() {
        errors.push("task.signal_features must not be empty".to_string());
    }
    if task.label.is_none() {
        errors.push("event_predictor manifests require task.label".to_string());
    }
    if manifest.artifact.format == ArtifactFormat::MockGemm {
        errors.push("event_predictor manifests cannot use mock-gemm artifacts".to_string());
    }
}

pub fn export_dataset(
    export: &serde_json::Value,
    trace: &serde_json::Value,
) -> Result<SurrogateDataset, Error> {
    let export_schema = required_str(export, "schema")?.to_string();
    let top_name = required_str(export, "top_name")?.to_string();
    let wires = required_array(export, "wires")?;
    let memories = required_array(export, "memories")?;
    let nets = required_array(export, "nets")?;
    let trace_schema = required_str(trace, "schema")?.to_string();
    let trace_steps = required_array(trace, "steps")?;
    let source_hash = canonical_json_hash(export)?;

    let mut node_features = Vec::with_capacity(wires.len() + memories.len());
    let mut index_by_name = BTreeMap::new();
    let mut input_count = 0usize;
    let mut output_count = 0usize;
    let mut register_count = 0usize;
    for wire in wires {
        let name = required_str(wire, "name")?.to_string();
        let kind = required_str(wire, "kind")?.to_string();
        let bitwidth = required_u64(wire, "bitwidth")? as u32;
        if kind == "input" {
            input_count += 1;
        } else if kind == "output" {
            output_count += 1;
        } else if kind == "register" {
            register_count += 1;
        }
        let node_index = node_features.len();
        index_by_name.insert(name.clone(), node_index);
        node_features.push(GraphNodeFeature {
            name,
            kind,
            bitwidth,
            value: optional_i128(wire, "value")?,
            reset_value: optional_i128(wire, "reset_value")?,
        });
    }

    for memory in memories {
        let name = required_str(memory, "name")?.to_string();
        let bitwidth = required_u64(memory, "bitwidth")? as u32;
        let kind = format!("memory:{}", required_str(memory, "kind")?);
        let node_index = node_features.len();
        index_by_name.insert(name.clone(), node_index);
        node_features.push(GraphNodeFeature {
            name,
            kind,
            bitwidth,
            value: None,
            reset_value: None,
        });
    }

    let mut edge_index = Vec::new();
    let mut edge_features = Vec::new();
    for net in nets {
        let net_index = required_u64(net, "index")? as usize;
        let op = required_str(net, "op")?.to_string();
        let args = required_array(net, "args")?;
        let dests = required_array(net, "dests")?;
        for arg in args {
            let arg_name = arg
                .as_str()
                .ok_or_else(|| Error::new("net args must be strings"))?;
            let Some(&arg_index) = index_by_name.get(arg_name) else {
                continue;
            };
            for dest in dests {
                let dest_name = dest
                    .as_str()
                    .ok_or_else(|| Error::new("net dests must be strings"))?;
                let Some(&dest_index) = index_by_name.get(dest_name) else {
                    continue;
                };
                edge_index.push([arg_index, dest_index]);
                edge_features.push(GraphEdgeFeature {
                    net_index,
                    op: op.clone(),
                    role: "arg-to-dest".to_string(),
                });
            }
        }
    }

    let mut steps = Vec::with_capacity(trace_steps.len());
    for step in trace_steps {
        steps.push(TraceDatasetStep {
            inputs: value_object_to_u128_map(required_object(step, "inputs")?)?,
            outputs: value_object_to_u128_map(required_object(step, "outputs")?)?,
        });
    }

    Ok(SurrogateDataset {
        schema: DATASET_SCHEMA.to_string(),
        source_hash,
        top_name,
        graph: GraphDataset {
            node_features,
            edge_index,
            edge_features,
            metadata: GraphMetadata {
                export_schema,
                net_count: nets.len(),
                memory_count: memories.len(),
                input_count,
                output_count,
                register_count,
            },
        },
        trace: TraceDataset {
            schema: trace_schema,
            steps,
        },
    })
}

pub fn validate_surrogate(
    manifest: &SurrogateManifest,
    manifest_path: impl AsRef<Path>,
    export: &serde_json::Value,
    trace: &serde_json::Value,
) -> Result<SurrogateValidationReport, Error> {
    let mut manifest_report = validate_manifest_path(manifest, &manifest_path);
    let dataset = export_dataset(export, trace)?;
    let dataset_summary = DatasetSummary {
        source_hash: dataset.source_hash.clone(),
        top_name: dataset.top_name.clone(),
        nodes: dataset.graph.node_features.len(),
        edges: dataset.graph.edge_index.len(),
        steps: dataset.trace.steps.len(),
    };
    if dataset.source_hash != manifest.source.source_hash {
        manifest_report.errors.push(format!(
            "source hash mismatch: manifest {}, actual {}",
            manifest.source.source_hash, dataset.source_hash
        ));
    }
    if dataset.top_name != manifest.source.top_name {
        manifest_report.errors.push(format!(
            "top name mismatch: manifest `{}`, actual `{}`",
            manifest.source.top_name, dataset.top_name
        ));
    }
    if dataset.graph.metadata.export_schema != manifest.source.export_schema {
        manifest_report.errors.push(format!(
            "export schema mismatch: manifest `{}`, actual `{}`",
            manifest.source.export_schema, dataset.graph.metadata.export_schema
        ));
    }
    manifest_report.ok = manifest_report.errors.is_empty();
    let ok = manifest_report.ok;
    Ok(SurrogateValidationReport {
        schema: VALIDATION_SCHEMA.to_string(),
        ok,
        manifest: manifest_report,
        dataset: dataset_summary,
        run: None,
    })
}

pub fn run_gemm_transaction(
    manifest: &SurrogateManifest,
    manifest_path: impl AsRef<Path>,
    transaction: &GemmTransaction,
) -> Result<GemmRunResult, Error> {
    let manifest_report = validate_manifest_path(manifest, &manifest_path);
    if !manifest_report.ok {
        return Err(Error::new(format!(
            "manifest validation failed: {}",
            manifest_report.errors.join("; ")
        )));
    }
    validate_transaction_domain(manifest, transaction)?;

    let base_dir = manifest_path
        .as_ref()
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let artifact_path = resolve_artifact_path(&base_dir, &manifest.artifact.path);
    let default_latency_cycles =
        (transaction.k + manifest.domain.rows + manifest.domain.cols - 1) as u64;
    let (c, telemetry) = match manifest.artifact.format {
        ArtifactFormat::MockGemm => (
            mock_gemm(transaction),
            GemmTelemetry {
                latency_cycles: default_latency_cycles,
                active_cycles: transaction.k as u64,
                utilization: if default_latency_cycles == 0 {
                    0.0
                } else {
                    transaction.k as f64 / default_latency_cycles as f64
                },
            },
        ),
        ArtifactFormat::Onnx => run_onnx_gemm(manifest, transaction, &artifact_path)?,
        ArtifactFormat::MockEventPredictor => {
            return Err(Error::new(
                "mock-event-predictor artifacts cannot run GEMM transactions",
            ));
        }
    };
    let metrics = transaction.expected_c.as_ref().map(|expected| {
        gemm_metrics(
            &c,
            expected,
            telemetry.latency_cycles,
            transaction.expected_latency_cycles,
        )
    });
    let metrics_ok = metrics.as_ref().map_or(true, |metrics| {
        metrics.max_abs_error <= manifest.validation.max_abs_error
            && metrics.mean_abs_error <= manifest.validation.max_mean_abs_error
            && metrics.latency_error_cycles <= manifest.validation.max_latency_error_cycles
    });
    Ok(GemmRunResult {
        schema: GEMM_RESULT_SCHEMA.to_string(),
        surrogate_id: manifest.surrogate_id.clone(),
        ok: metrics_ok,
        c,
        telemetry,
        metrics,
        provenance: Provenance {
            tag: if manifest.policy.provenance_tag.is_empty() {
                "approximate".to_string()
            } else {
                manifest.policy.provenance_tag.clone()
            },
            exact: false,
            surrogate_id: manifest.surrogate_id.clone(),
            model_family: manifest.model_family,
            artifact_format: manifest.artifact.format,
            artifact_hash: manifest.artifact.sha256.clone(),
            source_hash: manifest.source.source_hash.clone(),
            policy: manifest.policy.mode,
        },
    })
}

pub fn run_gemm_batch(
    manifest: &SurrogateManifest,
    manifest_path: impl AsRef<Path>,
    batch: &GemmBatch,
) -> GemmBatchRunReport {
    let mut results = Vec::with_capacity(batch.transactions.len());
    let mut metrics = GemmBatchMetrics::default();
    let mut lanes = BTreeMap::<usize, GemmBatchLaneAccumulator>::new();

    for (index, transaction) in batch.transactions.iter().enumerate() {
        let lane = transaction.lane.unwrap_or(index);
        let lane_summary = lanes.entry(lane).or_default();
        lane_summary.count += 1;
        let item = if batch.schema != GEMM_BATCH_SCHEMA {
            lane_summary.failed += 1;
            GemmBatchItemResult {
                index,
                lane,
                ok: false,
                result: None,
                error: Some(format!("unsupported GEMM batch schema `{}`", batch.schema)),
            }
        } else {
            match run_gemm_transaction(manifest, manifest_path.as_ref(), transaction) {
                Ok(result) => {
                    if result.ok {
                        lane_summary.ok += 1;
                    } else {
                        lane_summary.failed += 1;
                    }
                    if let Some(item_metrics) = result.metrics.as_ref() {
                        metrics.include_gemm_metrics(item_metrics);
                        lane_summary.metrics.include_gemm_metrics(item_metrics);
                    }
                    GemmBatchItemResult {
                        index,
                        lane,
                        ok: result.ok,
                        result: Some(result),
                        error: None,
                    }
                }
                Err(err) => {
                    lane_summary.failed += 1;
                    GemmBatchItemResult {
                        index,
                        lane,
                        ok: false,
                        result: None,
                        error: Some(err.to_string()),
                    }
                }
            }
        };
        results.push(item);
    }

    let lanes = lanes
        .into_iter()
        .map(|(lane, summary)| GemmBatchLaneSummary {
            lane,
            count: summary.count,
            ok: summary.ok,
            failed: summary.failed,
            metrics: summary.metrics,
        })
        .collect::<Vec<_>>();
    let ok = !results.is_empty() && results.iter().all(|item| item.ok);
    GemmBatchRunReport {
        schema: GEMM_BATCH_RESULT_SCHEMA.to_string(),
        ok,
        count: results.len(),
        total_lanes: lanes.len(),
        metrics,
        lanes,
        results,
    }
}

pub fn policy_gemm_batch(
    manifest: &SurrogateManifest,
    manifest_path: impl AsRef<Path>,
    batch: &GemmBatch,
) -> GemmPolicyReport {
    let mut results = Vec::with_capacity(batch.transactions.len());
    for (index, transaction) in batch.transactions.iter().enumerate() {
        let lane = transaction.lane.unwrap_or(index);
        let surrogate = if batch.schema == GEMM_BATCH_SCHEMA {
            run_gemm_transaction(manifest, manifest_path.as_ref(), transaction)
        } else {
            Err(Error::new(format!(
                "unsupported GEMM batch schema `{}`",
                batch.schema
            )))
        };
        let exact = exact_gemm_result(manifest, transaction);
        results.push(policy_gemm_item(
            manifest.policy.mode,
            manifest.policy.fallback,
            index,
            lane,
            surrogate,
            exact,
        ));
    }

    let used_surrogate = results
        .iter()
        .filter(|item| item.decision == GemmPolicyDecision::SurrogateUsed)
        .count();
    let exact_fallbacks = results
        .iter()
        .filter(|item| {
            matches!(
                item.decision,
                GemmPolicyDecision::ExactFallback | GemmPolicyDecision::ShadowCompare
            )
        })
        .count();
    let fail_closed = results
        .iter()
        .filter(|item| item.decision == GemmPolicyDecision::FailClosed)
        .count();
    GemmPolicyReport {
        schema: GEMM_POLICY_REPORT_SCHEMA.to_string(),
        ok: !results.is_empty() && fail_closed == 0 && results.iter().all(|item| item.ok),
        count: results.len(),
        used_surrogate,
        exact_fallbacks,
        fail_closed,
        results,
    }
}

pub fn run_fast_gemm_batch(
    manifest: &SurrogateManifest,
    manifest_path: impl AsRef<Path>,
    batch: &GemmBatch,
    runtime_plan: &GemmRuntimePlan,
) -> GemmFastRunReport {
    run_fast_gemm_batch_with_options(
        manifest,
        manifest_path,
        batch,
        runtime_plan,
        GemmFastRunOptions::default(),
    )
}

pub fn run_fast_gemm_batch_with_options(
    manifest: &SurrogateManifest,
    manifest_path: impl AsRef<Path>,
    batch: &GemmBatch,
    runtime_plan: &GemmRuntimePlan,
    options: GemmFastRunOptions,
) -> GemmFastRunReport {
    let policy = policy_gemm_batch(manifest, manifest_path, batch);
    let errors = validate_fast_gemm_plan(&policy, runtime_plan, &options);
    let mut results = Vec::with_capacity(policy.results.len());

    for item in &policy.results {
        results.push(fast_gemm_item(item, runtime_plan, &options));
    }

    let surrogate_replacements = results
        .iter()
        .filter(|item| item.source_result == Some(GemmRuntimeSourceResult::Surrogate))
        .count();
    let exact_fallbacks = results
        .iter()
        .filter(|item| item.source_result == Some(GemmRuntimeSourceResult::Exact))
        .count();
    let fail_closed = results
        .iter()
        .filter(|item| item.decision == GemmPolicyDecision::FailClosed)
        .count();
    let shadow_compared = results
        .iter()
        .filter(|item| item.shadow_ok.is_some())
        .count();
    let shadow_passed = results
        .iter()
        .filter(|item| item.shadow_ok == Some(true))
        .count();
    let shadow_failed = results
        .iter()
        .filter(|item| item.shadow_ok == Some(false))
        .count();
    let workers = fast_gemm_worker_summaries(runtime_plan, &results);
    let lanes = fast_gemm_lane_summaries(&results);

    GemmFastRunReport {
        schema: GEMM_FAST_RUN_SCHEMA.to_string(),
        ok: errors.is_empty()
            && fail_closed == 0
            && results.iter().all(|item| item.result.is_some()),
        count: results.len(),
        total_lanes: runtime_plan.total_lanes,
        surrogate_replacements,
        exact_fallbacks,
        fail_closed,
        shadow_compared,
        shadow_passed,
        shadow_failed,
        workers,
        lanes,
        results,
        errors,
    }
}

fn validate_fast_gemm_plan(
    policy: &GemmPolicyReport,
    runtime_plan: &GemmRuntimePlan,
    options: &GemmFastRunOptions,
) -> Vec<String> {
    let mut errors = Vec::new();
    if runtime_plan.schema != GEMM_RUNTIME_PLAN_SCHEMA {
        errors.push(format!(
            "unsupported GEMM runtime plan schema `{}`",
            runtime_plan.schema
        ));
    }
    if !runtime_plan.ok {
        errors.push("GEMM runtime plan is not ok".to_string());
    }
    errors.extend(runtime_plan.errors.iter().cloned());
    if policy.schema != GEMM_POLICY_REPORT_SCHEMA {
        errors.push(format!(
            "unsupported GEMM policy report schema `{}`",
            policy.schema
        ));
    }
    if policy.count != policy.results.len() {
        errors.push("GEMM policy report count does not match results".to_string());
    }
    if options.shadow_sample_stride == Some(0) {
        errors.push("shadow sample stride must be greater than zero".to_string());
    }

    let mut seen_policy_items = BTreeSet::new();
    for policy_item in &policy.results {
        if !seen_policy_items.insert(policy_item.index) {
            errors.push(format!(
                "policy item {} appears more than once",
                policy_item.index
            ));
        }
    }
    let mut seen_plan_items = BTreeSet::new();
    for plan_item in &runtime_plan.items {
        if !seen_plan_items.insert(plan_item.index) {
            errors.push(format!(
                "runtime plan item {} appears more than once",
                plan_item.index
            ));
        }
    }

    for policy_item in &policy.results {
        if policy_item.lane >= runtime_plan.total_lanes {
            errors.push(format!(
                "policy item {} lane {} is outside runtime plan lanes {}",
                policy_item.index, policy_item.lane, runtime_plan.total_lanes
            ));
        }
        if policy_item.decision == GemmPolicyDecision::FailClosed {
            continue;
        }
        let Some(plan_item) = runtime_plan
            .items
            .iter()
            .find(|item| item.index == policy_item.index)
        else {
            errors.push(format!(
                "runtime plan missing non-fail-closed policy item {}",
                policy_item.index
            ));
            continue;
        };
        if plan_item.lane != policy_item.lane {
            errors.push(format!(
                "runtime plan item {} lane {} does not match policy lane {}",
                policy_item.index, plan_item.lane, policy_item.lane
            ));
        }
        if plan_item.decision != policy_item.decision {
            errors.push(format!(
                "runtime plan item {} decision {:?} does not match policy decision {:?}",
                policy_item.index, plan_item.decision, policy_item.decision
            ));
        }
        if plan_item.source_result != fast_source_result_for_decision(policy_item.decision) {
            errors.push(format!(
                "runtime plan item {} source result {:?} does not match policy decision {:?}",
                policy_item.index, plan_item.source_result, policy_item.decision
            ));
        }
    }
    for plan_item in &runtime_plan.items {
        let Some(policy_item) = policy
            .results
            .iter()
            .find(|item| item.index == plan_item.index)
        else {
            errors.push(format!(
                "runtime plan contains extra item {} with no policy result",
                plan_item.index
            ));
            continue;
        };
        if policy_item.decision == GemmPolicyDecision::FailClosed {
            errors.push(format!(
                "runtime plan contains fail-closed policy item {}",
                plan_item.index
            ));
        }
    }

    errors
}

fn fast_gemm_item(
    item: &GemmPolicyItemResult,
    runtime_plan: &GemmRuntimePlan,
    options: &GemmFastRunOptions,
) -> GemmFastRunItem {
    let plan_item = runtime_plan
        .items
        .iter()
        .find(|plan_item| plan_item.index == item.index);
    let (source_result, result, error) = match item.decision {
        GemmPolicyDecision::SurrogateUsed => (
            Some(GemmRuntimeSourceResult::Surrogate),
            item.surrogate_result.clone(),
            item.error.clone(),
        ),
        GemmPolicyDecision::ExactFallback | GemmPolicyDecision::ShadowCompare => (
            Some(GemmRuntimeSourceResult::Exact),
            item.exact_result.clone(),
            item.error.clone(),
        ),
        GemmPolicyDecision::FailClosed => (None, None, item.error.clone()),
    };
    let provenance = result.as_ref().map(|result| result.provenance.clone());
    GemmFastRunItem {
        index: item.index,
        lane: item.lane,
        decision: item.decision,
        source_result,
        result,
        provenance,
        shadow_ok: plan_item.and_then(|item| item.shadow_ok),
        shadow_error: plan_item.and_then(|item| item.shadow_error.clone()),
        shadow_max_abs_error: plan_item.and_then(|item| item.shadow_max_abs_error),
        shadow_latency_error_cycles: plan_item.and_then(|item| item.shadow_latency_error_cycles),
        shadow_first_divergence: plan_item.and_then(|item| item.shadow_first_divergence.clone()),
        shadow_sampled: gemm_fast_shadow_sampled(item.index, options),
        error,
    }
}

fn gemm_fast_shadow_sampled(index: usize, options: &GemmFastRunOptions) -> bool {
    let Some(stride) = options.shadow_sample_stride else {
        return false;
    };
    stride != 0
        && index >= options.shadow_sample_offset
        && (index - options.shadow_sample_offset) % stride == 0
}

fn fast_gemm_worker_summaries(
    runtime_plan: &GemmRuntimePlan,
    results: &[GemmFastRunItem],
) -> Vec<GemmFastRunWorkerSummary> {
    runtime_plan
        .workers
        .iter()
        .map(|worker| {
            let worker_results = results
                .iter()
                .filter(|item| {
                    item.lane >= worker.start_lane
                        && item.lane < worker.start_lane.saturating_add(worker.lanes)
                })
                .collect::<Vec<_>>();
            GemmFastRunWorkerSummary {
                worker_id: worker.worker_id.clone(),
                start_lane: worker.start_lane,
                lanes: worker.lanes,
                assigned_items: worker_results.len(),
                surrogate_replacements: worker_results
                    .iter()
                    .filter(|item| item.source_result == Some(GemmRuntimeSourceResult::Surrogate))
                    .count(),
                exact_fallbacks: worker_results
                    .iter()
                    .filter(|item| item.source_result == Some(GemmRuntimeSourceResult::Exact))
                    .count(),
                fail_closed: worker_results
                    .iter()
                    .filter(|item| item.decision == GemmPolicyDecision::FailClosed)
                    .count(),
                shadow_compared: worker_results
                    .iter()
                    .filter(|item| item.shadow_ok.is_some())
                    .count(),
                shadow_failed: worker_results
                    .iter()
                    .filter(|item| item.shadow_ok == Some(false))
                    .count(),
            }
        })
        .collect()
}

fn fast_gemm_lane_summaries(results: &[GemmFastRunItem]) -> Vec<GemmFastRunLaneSummary> {
    let mut lanes = BTreeMap::<usize, Vec<&GemmFastRunItem>>::new();
    for item in results {
        lanes.entry(item.lane).or_default().push(item);
    }
    lanes
        .into_iter()
        .map(|(lane, items)| GemmFastRunLaneSummary {
            lane,
            count: items.len(),
            surrogate_replacements: items
                .iter()
                .filter(|item| item.source_result == Some(GemmRuntimeSourceResult::Surrogate))
                .count(),
            exact_fallbacks: items
                .iter()
                .filter(|item| item.source_result == Some(GemmRuntimeSourceResult::Exact))
                .count(),
            fail_closed: items
                .iter()
                .filter(|item| item.decision == GemmPolicyDecision::FailClosed)
                .count(),
            shadow_compared: items.iter().filter(|item| item.shadow_ok.is_some()).count(),
            shadow_failed: items
                .iter()
                .filter(|item| item.shadow_ok == Some(false))
                .count(),
        })
        .collect()
}

fn fast_source_result_for_decision(decision: GemmPolicyDecision) -> GemmRuntimeSourceResult {
    match decision {
        GemmPolicyDecision::SurrogateUsed => GemmRuntimeSourceResult::Surrogate,
        GemmPolicyDecision::ExactFallback | GemmPolicyDecision::ShadowCompare => {
            GemmRuntimeSourceResult::Exact
        }
        GemmPolicyDecision::FailClosed => GemmRuntimeSourceResult::Exact,
    }
}

fn policy_gemm_item(
    mode: PolicyMode,
    fallback: FallbackPolicy,
    index: usize,
    lane: usize,
    surrogate: Result<GemmRunResult, Error>,
    exact: Result<GemmRunResult, Error>,
) -> GemmPolicyItemResult {
    match mode {
        PolicyMode::ApproximateWithTolerance => match surrogate {
            Ok(result) if result.ok => GemmPolicyItemResult {
                index,
                lane,
                decision: GemmPolicyDecision::SurrogateUsed,
                ok: true,
                surrogate_result: Some(result),
                exact_result: None,
                error: None,
            },
            Ok(result) => fallback_policy_item(index, lane, fallback, Some(result), None, exact),
            Err(err) => fallback_policy_item(index, lane, fallback, None, Some(err), exact),
        },
        PolicyMode::ShadowCompare => match exact {
            Ok(exact_result) => GemmPolicyItemResult {
                index,
                lane,
                decision: GemmPolicyDecision::ShadowCompare,
                ok: true,
                surrogate_result: surrogate.ok(),
                exact_result: Some(exact_result),
                error: None,
            },
            Err(err) => GemmPolicyItemResult {
                index,
                lane,
                decision: GemmPolicyDecision::FailClosed,
                ok: false,
                surrogate_result: surrogate.ok(),
                exact_result: None,
                error: Some(err.to_string()),
            },
        },
        PolicyMode::TelemetryOnly => match exact {
            Ok(exact_result) => GemmPolicyItemResult {
                index,
                lane,
                decision: GemmPolicyDecision::ExactFallback,
                ok: true,
                surrogate_result: surrogate.ok(),
                exact_result: Some(exact_result),
                error: None,
            },
            Err(err) => GemmPolicyItemResult {
                index,
                lane,
                decision: GemmPolicyDecision::FailClosed,
                ok: false,
                surrogate_result: surrogate.ok(),
                exact_result: None,
                error: Some(err.to_string()),
            },
        },
    }
}

fn fallback_policy_item(
    index: usize,
    lane: usize,
    fallback: FallbackPolicy,
    surrogate_result: Option<GemmRunResult>,
    surrogate_error: Option<Error>,
    exact: Result<GemmRunResult, Error>,
) -> GemmPolicyItemResult {
    match fallback {
        FallbackPolicy::ExactFallback => match exact {
            Ok(exact_result) => GemmPolicyItemResult {
                index,
                lane,
                decision: GemmPolicyDecision::ExactFallback,
                ok: true,
                surrogate_result,
                exact_result: Some(exact_result),
                error: surrogate_error.map(|err| err.to_string()),
            },
            Err(err) => GemmPolicyItemResult {
                index,
                lane,
                decision: GemmPolicyDecision::FailClosed,
                ok: false,
                surrogate_result,
                exact_result: None,
                error: Some(match surrogate_error {
                    Some(surrogate_error) => {
                        format!("{}; exact fallback failed: {}", surrogate_error, err)
                    }
                    None => err.to_string(),
                }),
            },
        },
        FallbackPolicy::FailClosed => GemmPolicyItemResult {
            index,
            lane,
            decision: GemmPolicyDecision::FailClosed,
            ok: false,
            surrogate_result,
            exact_result: None,
            error: Some(
                surrogate_error
                    .map(|err| err.to_string())
                    .unwrap_or_else(|| "surrogate result failed validation tolerance".to_string()),
            ),
        },
    }
}

pub fn plan_runtime_gemm(policy: &GemmPolicyReport, total_lanes: usize) -> GemmRuntimePlan {
    let workers = if total_lanes == 0 {
        Vec::new()
    } else {
        vec![GemmRuntimeWorkerSpec {
            worker_id: "worker0".to_string(),
            start_lane: 0,
            lanes: total_lanes,
        }]
    };
    plan_runtime_gemm_for_workers(policy, &workers)
}

pub fn plan_runtime_gemm_for_workers(
    policy: &GemmPolicyReport,
    workers: &[GemmRuntimeWorkerSpec],
) -> GemmRuntimePlan {
    let mut errors = Vec::new();
    let mut items = Vec::new();
    let total_lanes = workers
        .iter()
        .map(|worker| worker.start_lane.saturating_add(worker.lanes))
        .max()
        .unwrap_or(0);

    if policy.schema != GEMM_POLICY_REPORT_SCHEMA {
        errors.push(format!(
            "unsupported GEMM policy report schema `{}`",
            policy.schema
        ));
    }
    if policy.results.is_empty() {
        errors.push("GEMM policy report must contain at least one result".to_string());
    }
    if policy.fail_closed > 0 {
        errors.push(format!(
            "GEMM policy report contains {} fail-closed item(s)",
            policy.fail_closed
        ));
    }
    if workers.is_empty() {
        errors.push("runtime topology must contain at least one worker".to_string());
    }
    if let Err(err) = validate_runtime_worker_specs(workers) {
        errors.push(err);
    }

    for item in &policy.results {
        match runtime_plan_item(item, workers, total_lanes) {
            Ok(plan_item) => items.push(plan_item),
            Err(err) => errors.push(err),
        }
    }

    let worker_summaries = workers
        .iter()
        .map(|worker| {
            let worker_items = items
                .iter()
                .filter(|item| item.worker_id == worker.worker_id)
                .collect::<Vec<_>>();
            GemmRuntimeWorkerSummary {
                worker_id: worker.worker_id.clone(),
                start_lane: worker.start_lane,
                lanes: worker.lanes,
                assigned_items: worker_items.len(),
                used_surrogate: worker_items
                    .iter()
                    .filter(|item| item.source_result == GemmRuntimeSourceResult::Surrogate)
                    .count(),
                exact_fallbacks: worker_items
                    .iter()
                    .filter(|item| item.source_result == GemmRuntimeSourceResult::Exact)
                    .count(),
            }
        })
        .collect();

    GemmRuntimePlan {
        schema: GEMM_RUNTIME_PLAN_SCHEMA.to_string(),
        ok: errors.is_empty(),
        total_lanes,
        workers: worker_summaries,
        items,
        errors,
    }
}

fn runtime_plan_item(
    item: &GemmPolicyItemResult,
    workers: &[GemmRuntimeWorkerSpec],
    total_lanes: usize,
) -> Result<GemmRuntimePlanItem, String> {
    if !item.ok {
        return Err(format!("policy item {} is not ok", item.index));
    }
    if item.decision == GemmPolicyDecision::FailClosed {
        return Err(format!("policy item {} is fail-closed", item.index));
    }
    if total_lanes == 0 || item.lane >= total_lanes {
        return Err(format!(
            "policy item {} lane {} is outside runtime topology lanes {}",
            item.index, item.lane, total_lanes
        ));
    }
    let worker = worker_for_lane(workers, item.lane).ok_or_else(|| {
        format!(
            "policy item {} lane {} is not covered by runtime workers",
            item.index, item.lane
        )
    })?;

    let (source_result, result) = match item.decision {
        GemmPolicyDecision::SurrogateUsed => (
            GemmRuntimeSourceResult::Surrogate,
            item.surrogate_result
                .as_ref()
                .ok_or_else(|| format!("policy item {} missing surrogate result", item.index))?,
        ),
        GemmPolicyDecision::ExactFallback | GemmPolicyDecision::ShadowCompare => (
            GemmRuntimeSourceResult::Exact,
            item.exact_result
                .as_ref()
                .ok_or_else(|| format!("policy item {} missing exact result", item.index))?,
        ),
        GemmPolicyDecision::FailClosed => unreachable!(),
    };

    match source_result {
        GemmRuntimeSourceResult::Surrogate if result.provenance.exact => {
            return Err(format!(
                "policy item {} surrogate result has exact provenance",
                item.index
            ));
        }
        GemmRuntimeSourceResult::Exact if !result.provenance.exact => {
            return Err(format!(
                "policy item {} exact fallback result lacks exact provenance",
                item.index
            ));
        }
        _ => {}
    }

    Ok(GemmRuntimePlanItem {
        index: item.index,
        lane: item.lane,
        worker_id: worker.worker_id.clone(),
        decision: item.decision,
        provenance: result.provenance.clone(),
        source_result,
        shadow_ok: gemm_shadow_ok(item),
        shadow_error: gemm_shadow_error(item),
        shadow_max_abs_error: gemm_shadow_metrics(item).map(|metrics| metrics.max_abs_error),
        shadow_latency_error_cycles: gemm_shadow_metrics(item)
            .map(|metrics| metrics.latency_error_cycles),
        shadow_first_divergence: gemm_shadow_metrics(item)
            .and_then(|metrics| metrics.first_divergence),
    })
}

fn gemm_shadow_metrics(item: &GemmPolicyItemResult) -> Option<GemmValidationMetrics> {
    if item.decision != GemmPolicyDecision::ShadowCompare {
        return None;
    }
    let surrogate = item.surrogate_result.as_ref()?;
    let exact = item.exact_result.as_ref()?;
    Some(gemm_metrics(
        &surrogate.c,
        &exact.c,
        surrogate.telemetry.latency_cycles,
        Some(exact.telemetry.latency_cycles),
    ))
}

fn gemm_shadow_ok(item: &GemmPolicyItemResult) -> Option<bool> {
    if item.decision != GemmPolicyDecision::ShadowCompare {
        return None;
    }
    Some(
        gemm_shadow_metrics(item).is_some_and(|metrics| {
            metrics.max_abs_error == 0
                && metrics.latency_error_cycles == 0
                && metrics.first_divergence.is_none()
        }) && item.error.is_none(),
    )
}

fn gemm_shadow_error(item: &GemmPolicyItemResult) -> Option<String> {
    if item.decision != GemmPolicyDecision::ShadowCompare {
        return None;
    }
    if let Some(err) = &item.error {
        return Some(err.clone());
    }
    if item.surrogate_result.is_none() {
        return Some("shadow comparison missing surrogate result".to_string());
    }
    if item.exact_result.is_none() {
        return Some("shadow comparison missing exact result".to_string());
    }
    gemm_shadow_metrics(item).and_then(|metrics| {
        if metrics.max_abs_error == 0 && metrics.latency_error_cycles == 0 {
            None
        } else {
            Some(format!(
                "shadow comparison diverged: max_abs_error={}, latency_error_cycles={}",
                metrics.max_abs_error, metrics.latency_error_cycles
            ))
        }
    })
}

fn validate_runtime_worker_specs(workers: &[GemmRuntimeWorkerSpec]) -> Result<(), String> {
    let mut expected_start = 0;
    let mut seen = BTreeSet::new();
    for worker in workers {
        if worker.worker_id.trim().is_empty() {
            return Err("runtime worker_id must not be empty".to_string());
        }
        if !seen.insert(worker.worker_id.as_str()) {
            return Err(format!(
                "runtime worker `{}` appears more than once",
                worker.worker_id
            ));
        }
        if worker.lanes == 0 {
            return Err(format!(
                "runtime worker `{}` lanes must be greater than zero",
                worker.worker_id
            ));
        }
        if worker.start_lane != expected_start {
            return Err(format!(
                "runtime worker `{}` starts at lane {}, expected contiguous start lane {}",
                worker.worker_id, worker.start_lane, expected_start
            ));
        }
        expected_start += worker.lanes;
    }
    Ok(())
}

fn worker_for_lane<'a>(
    workers: &'a [GemmRuntimeWorkerSpec],
    lane: usize,
) -> Option<&'a GemmRuntimeWorkerSpec> {
    workers.iter().find(|worker| {
        lane >= worker.start_lane && lane < worker.start_lane.saturating_add(worker.lanes)
    })
}

pub fn policy_event_corpus(
    manifest: &SurrogateManifest,
    manifest_path: impl AsRef<Path>,
    corpus: &InstrumentationEventCorpus,
) -> EventPolicyReport {
    let manifest_report = validate_manifest_path(manifest, &manifest_path);
    let shape_errors = validate_event_corpus_shape(manifest, corpus);
    let mut errors = Vec::new();
    if !manifest_report.ok {
        errors.extend(manifest_report.errors.clone());
    }
    errors.extend(shape_errors);

    let provenance = event_provenance(manifest, false);
    let exact_provenance = event_provenance(manifest, true);
    let mut results = Vec::new();

    if errors.is_empty() {
        match event_predictions(manifest, manifest_path.as_ref(), corpus) {
            Ok(predictions) => {
                for (index, (event, prediction)) in
                    corpus.events.iter().zip(predictions.iter()).enumerate()
                {
                    results.push(policy_event_item(
                        index,
                        event,
                        prediction.lane,
                        prediction.predicted,
                        Some(prediction.expected),
                        &provenance,
                        &exact_provenance,
                        manifest.policy.mode,
                        manifest.policy.fallback,
                    ));
                }
            }
            Err(err) => errors.push(err.to_string()),
        }
    }

    let used_surrogate = results
        .iter()
        .filter(|item| item.decision == EventPolicyDecision::SurrogateUsed)
        .count();
    let exact_fallbacks = results
        .iter()
        .filter(|item| {
            matches!(
                item.decision,
                EventPolicyDecision::ExactFallback | EventPolicyDecision::ShadowCompare
            )
        })
        .count();
    let fail_closed = results
        .iter()
        .filter(|item| item.decision == EventPolicyDecision::FailClosed)
        .count();
    EventPolicyReport {
        schema: EVENT_POLICY_REPORT_SCHEMA.to_string(),
        ok: errors.is_empty() && fail_closed == 0,
        count: results.len(),
        used_surrogate,
        exact_fallbacks,
        fail_closed,
        results,
        errors,
    }
}

pub fn run_fast_event_corpus(
    manifest: &SurrogateManifest,
    manifest_path: impl AsRef<Path>,
    corpus: &InstrumentationEventCorpus,
    runtime_plan: &EventRuntimePlan,
) -> EventFastRunReport {
    run_fast_event_corpus_with_options(
        manifest,
        manifest_path,
        corpus,
        runtime_plan,
        EventFastRunOptions::default(),
    )
}

pub fn run_fast_event_corpus_with_options(
    manifest: &SurrogateManifest,
    manifest_path: impl AsRef<Path>,
    corpus: &InstrumentationEventCorpus,
    runtime_plan: &EventRuntimePlan,
    options: EventFastRunOptions,
) -> EventFastRunReport {
    let policy = policy_event_corpus(manifest, manifest_path, corpus);
    let errors = validate_fast_event_plan(&policy, runtime_plan, &options);
    let mut results = Vec::with_capacity(policy.results.len());

    for item in &policy.results {
        results.push(fast_event_item(item, runtime_plan, &options));
    }

    let surrogate_replacements = results
        .iter()
        .filter(|item| item.source_result == Some(GemmRuntimeSourceResult::Surrogate))
        .count();
    let exact_fallbacks = results
        .iter()
        .filter(|item| item.source_result == Some(GemmRuntimeSourceResult::Exact))
        .count();
    let fail_closed = results
        .iter()
        .filter(|item| item.decision == EventPolicyDecision::FailClosed)
        .count();
    let shadow_compared = results
        .iter()
        .filter(|item| item.shadow_ok.is_some())
        .count();
    let shadow_passed = results
        .iter()
        .filter(|item| item.shadow_ok == Some(true))
        .count();
    let shadow_failed = results
        .iter()
        .filter(|item| item.shadow_ok == Some(false))
        .count();
    let workers = fast_event_worker_summaries(runtime_plan, &results);
    let lanes = fast_event_lane_summaries(&results);

    EventFastRunReport {
        schema: EVENT_FAST_RUN_SCHEMA.to_string(),
        ok: errors.is_empty() && fail_closed == 0,
        count: results.len(),
        total_lanes: runtime_plan.total_lanes,
        surrogate_replacements,
        exact_fallbacks,
        fail_closed,
        shadow_compared,
        shadow_passed,
        shadow_failed,
        workers,
        lanes,
        results,
        errors,
    }
}

fn validate_fast_event_plan(
    policy: &EventPolicyReport,
    runtime_plan: &EventRuntimePlan,
    options: &EventFastRunOptions,
) -> Vec<String> {
    let mut errors = Vec::new();
    if runtime_plan.schema != EVENT_RUNTIME_PLAN_SCHEMA {
        errors.push(format!(
            "unsupported event runtime plan schema `{}`",
            runtime_plan.schema
        ));
    }
    if !runtime_plan.ok {
        errors.push("event runtime plan is not ok".to_string());
    }
    errors.extend(runtime_plan.errors.iter().cloned());
    if policy.schema != EVENT_POLICY_REPORT_SCHEMA {
        errors.push(format!(
            "unsupported event policy report schema `{}`",
            policy.schema
        ));
    }
    if policy.count != policy.results.len() {
        errors.push("event policy report count does not match results".to_string());
    }
    if options.shadow_sample_stride == Some(0) {
        errors.push("shadow sample stride must be greater than zero".to_string());
    }

    let mut seen_policy_items = BTreeSet::new();
    for policy_item in &policy.results {
        if !seen_policy_items.insert(policy_item.index) {
            errors.push(format!(
                "event policy item {} appears more than once",
                policy_item.index
            ));
        }
    }
    let mut seen_plan_items = BTreeSet::new();
    for plan_item in &runtime_plan.items {
        if !seen_plan_items.insert(plan_item.index) {
            errors.push(format!(
                "event runtime plan item {} appears more than once",
                plan_item.index
            ));
        }
    }

    for policy_item in &policy.results {
        if policy_item.lane >= runtime_plan.total_lanes {
            errors.push(format!(
                "event policy item {} lane {} is outside runtime plan lanes {}",
                policy_item.index, policy_item.lane, runtime_plan.total_lanes
            ));
        }
        if policy_item.decision == EventPolicyDecision::FailClosed {
            continue;
        }
        let Some(plan_item) = runtime_plan
            .items
            .iter()
            .find(|item| item.index == policy_item.index)
        else {
            errors.push(format!(
                "event runtime plan missing non-fail-closed policy item {}",
                policy_item.index
            ));
            continue;
        };
        if plan_item.sample_id != policy_item.sample_id {
            errors.push(format!(
                "event runtime plan item {} sample_id {} does not match policy sample_id {}",
                policy_item.index, plan_item.sample_id, policy_item.sample_id
            ));
        }
        if plan_item.lane != policy_item.lane {
            errors.push(format!(
                "event runtime plan item {} lane {} does not match policy lane {}",
                policy_item.index, plan_item.lane, policy_item.lane
            ));
        }
        if plan_item.decision != policy_item.decision {
            errors.push(format!(
                "event runtime plan item {} decision {:?} does not match policy decision {:?}",
                policy_item.index, plan_item.decision, policy_item.decision
            ));
        }
        if plan_item.source_result != fast_source_result_for_event_decision(policy_item.decision) {
            errors.push(format!(
                "event runtime plan item {} source result {:?} does not match policy decision {:?}",
                policy_item.index, plan_item.source_result, policy_item.decision
            ));
        }
        if plan_item.predicted != Some(policy_item.predicted) {
            errors.push(format!(
                "event runtime plan item {} predicted {:?} does not match policy predicted {}",
                policy_item.index, plan_item.predicted, policy_item.predicted
            ));
        }
        if plan_item.expected != policy_item.expected {
            errors.push(format!(
                "event runtime plan item {} expected {:?} does not match policy expected {:?}",
                policy_item.index, plan_item.expected, policy_item.expected
            ));
        }
    }
    for plan_item in &runtime_plan.items {
        let Some(policy_item) = policy
            .results
            .iter()
            .find(|item| item.index == plan_item.index)
        else {
            errors.push(format!(
                "event runtime plan contains extra item {} with no policy result",
                plan_item.index
            ));
            continue;
        };
        if policy_item.decision == EventPolicyDecision::FailClosed {
            errors.push(format!(
                "event runtime plan contains fail-closed policy item {}",
                plan_item.index
            ));
        }
    }

    errors
}

fn fast_event_item(
    item: &EventPolicyItemResult,
    runtime_plan: &EventRuntimePlan,
    options: &EventFastRunOptions,
) -> EventFastRunItem {
    let plan_item = runtime_plan
        .items
        .iter()
        .find(|plan_item| plan_item.index == item.index);
    let source_result = match item.decision {
        EventPolicyDecision::SurrogateUsed => Some(GemmRuntimeSourceResult::Surrogate),
        EventPolicyDecision::ExactFallback | EventPolicyDecision::ShadowCompare => {
            Some(GemmRuntimeSourceResult::Exact)
        }
        EventPolicyDecision::FailClosed => None,
    };
    EventFastRunItem {
        index: item.index,
        sample_id: item.sample_id,
        lane: item.lane,
        target: item.target.clone(),
        decision: item.decision,
        source_result,
        predicted: item.predicted,
        expected: item.expected,
        provenance: item.provenance.clone(),
        shadow_ok: plan_item.and_then(|item| item.shadow_ok),
        shadow_error: plan_item.and_then(|item| item.shadow_error.clone()),
        shadow_sampled: event_fast_shadow_sampled(item.index, options),
        error: item.error.clone(),
    }
}

fn event_fast_shadow_sampled(index: usize, options: &EventFastRunOptions) -> bool {
    let Some(stride) = options.shadow_sample_stride else {
        return false;
    };
    stride != 0
        && index >= options.shadow_sample_offset
        && (index - options.shadow_sample_offset) % stride == 0
}

fn fast_event_worker_summaries(
    runtime_plan: &EventRuntimePlan,
    results: &[EventFastRunItem],
) -> Vec<EventFastRunWorkerSummary> {
    runtime_plan
        .workers
        .iter()
        .map(|worker| {
            let worker_results = results
                .iter()
                .filter(|item| {
                    item.lane >= worker.start_lane
                        && item.lane < worker.start_lane.saturating_add(worker.lanes)
                })
                .collect::<Vec<_>>();
            EventFastRunWorkerSummary {
                worker_id: worker.worker_id.clone(),
                start_lane: worker.start_lane,
                lanes: worker.lanes,
                assigned_items: worker_results.len(),
                surrogate_replacements: worker_results
                    .iter()
                    .filter(|item| item.source_result == Some(GemmRuntimeSourceResult::Surrogate))
                    .count(),
                exact_fallbacks: worker_results
                    .iter()
                    .filter(|item| item.source_result == Some(GemmRuntimeSourceResult::Exact))
                    .count(),
                fail_closed: worker_results
                    .iter()
                    .filter(|item| item.decision == EventPolicyDecision::FailClosed)
                    .count(),
                shadow_compared: worker_results
                    .iter()
                    .filter(|item| item.shadow_ok.is_some())
                    .count(),
                shadow_failed: worker_results
                    .iter()
                    .filter(|item| item.shadow_ok == Some(false))
                    .count(),
            }
        })
        .collect()
}

fn fast_event_lane_summaries(results: &[EventFastRunItem]) -> Vec<EventFastRunLaneSummary> {
    let mut lanes = BTreeMap::<usize, Vec<&EventFastRunItem>>::new();
    for item in results {
        lanes.entry(item.lane).or_default().push(item);
    }
    lanes
        .into_iter()
        .map(|(lane, items)| EventFastRunLaneSummary {
            lane,
            count: items.len(),
            surrogate_replacements: items
                .iter()
                .filter(|item| item.source_result == Some(GemmRuntimeSourceResult::Surrogate))
                .count(),
            exact_fallbacks: items
                .iter()
                .filter(|item| item.source_result == Some(GemmRuntimeSourceResult::Exact))
                .count(),
            fail_closed: items
                .iter()
                .filter(|item| item.decision == EventPolicyDecision::FailClosed)
                .count(),
            shadow_compared: items.iter().filter(|item| item.shadow_ok.is_some()).count(),
            shadow_failed: items
                .iter()
                .filter(|item| item.shadow_ok == Some(false))
                .count(),
        })
        .collect()
}

fn fast_source_result_for_event_decision(decision: EventPolicyDecision) -> GemmRuntimeSourceResult {
    match decision {
        EventPolicyDecision::SurrogateUsed => GemmRuntimeSourceResult::Surrogate,
        EventPolicyDecision::ExactFallback | EventPolicyDecision::ShadowCompare => {
            GemmRuntimeSourceResult::Exact
        }
        EventPolicyDecision::FailClosed => GemmRuntimeSourceResult::Exact,
    }
}

pub fn run_model_fast_plan(plan: &ModelFastPlan, base_dir: impl AsRef<Path>) -> ModelFastReport {
    let mut errors = Vec::new();
    if plan.schema != MODEL_FAST_PLAN_SCHEMA {
        errors.push(format!(
            "unsupported model FAST plan schema `{}`",
            plan.schema
        ));
    }
    if plan.ops.is_empty() {
        errors.push("model FAST plan must contain at least one op".to_string());
    }

    let mut ops = Vec::with_capacity(plan.ops.len());
    let mut totals = ModelFastTotals::default();
    for op in &plan.ops {
        let op_report = run_model_fast_op(op, base_dir.as_ref());
        totals.include(&op_report.totals);
        if !op_report.ok {
            errors.extend(
                op_report
                    .errors
                    .iter()
                    .map(|err| format!("op `{}`: {err}", op.op_id)),
            );
        }
        ops.push(op_report);
    }
    let mut coverage = ModelFastCoverage::from_totals(&totals, &ops);
    coverage.apply_thresholds(plan.thresholds.as_ref());
    errors.extend(coverage.reject_reasons.iter().cloned());
    coverage.accepted = errors.is_empty() && ops.iter().all(|op| op.ok) && coverage.accepted;
    let timing = ModelFastTimingSummary::from_ops(&ops);

    ModelFastReport {
        schema: MODEL_FAST_REPORT_SCHEMA.to_string(),
        ok: coverage.accepted,
        op_count: ops.len(),
        totals,
        coverage,
        timing,
        ops,
        errors,
    }
}

fn run_model_fast_op(op: &ModelFastOp, base_dir: &Path) -> ModelFastOpReport {
    let report_path = resolve_model_fast_report_path(base_dir, &op.fast_report_path);
    let mut report = ModelFastOpReport {
        op_id: op.op_id.clone(),
        op_kind: op.op_kind.clone(),
        name: op.name.clone(),
        fast_report_path: op.fast_report_path.clone(),
        source_hash: op.source_hash.clone(),
        description: op.description.clone(),
        ok: false,
        totals: ModelFastTotals::default(),
        provenance: Vec::new(),
        golden: None,
        timing: None,
        errors: Vec::new(),
    };

    match op.op_kind.as_str() {
        "gemm" => match fs::File::open(&report_path)
            .map_err(|err| Error::new(format!("failed to open FAST report: {err}")))
            .and_then(read_gemm_fast_run_report)
        {
            Ok(fast) => include_gemm_fast_report(&mut report, fast),
            Err(err) => report.errors.push(err.to_string()),
        },
        "event" => match fs::File::open(&report_path)
            .map_err(|err| Error::new(format!("failed to open FAST report: {err}")))
            .and_then(read_event_fast_run_report)
        {
            Ok(fast) => include_event_fast_report(&mut report, fast),
            Err(err) => report.errors.push(err.to_string()),
        },
        other => report
            .errors
            .push(format!("unsupported model FAST op kind `{other}`")),
    }

    if report.errors.is_empty() {
        compare_model_fast_golden(&mut report, op, base_dir);
    }
    attach_model_fast_timing(&mut report, op);
    report.ok = report.errors.is_empty();
    report
}

fn attach_model_fast_timing(op_report: &mut ModelFastOpReport, op: &ModelFastOp) {
    match (op.exact_ns, op.fast_ns) {
        (Some(0), _) => op_report
            .errors
            .push("model FAST exact_ns must be greater than zero".to_string()),
        (_, Some(0)) => op_report
            .errors
            .push("model FAST fast_ns must be greater than zero".to_string()),
        (Some(exact_ns), Some(fast_ns)) => {
            op_report.timing = Some(ModelFastOpTiming {
                exact_ns,
                fast_ns,
                speedup: exact_ns as f64 / fast_ns as f64,
            });
        }
        _ => {}
    }
}

fn compare_model_fast_golden(op_report: &mut ModelFastOpReport, op: &ModelFastOp, base_dir: &Path) {
    let Some(golden_path) = op.golden_path.as_ref() else {
        return;
    };
    let path = resolve_model_fast_report_path(base_dir, golden_path);
    match fs::File::open(&path)
        .map_err(|err| Error::new(format!("failed to open model FAST golden: {err}")))
        .and_then(read_model_fast_golden)
    {
        Ok(golden) => {
            let tensor = compare_model_fast_golden_tensors(&golden);
            let mut mismatches = model_fast_golden_mismatches(op_report, &golden);
            mismatches.extend(tensor.errors.iter().cloned());
            let golden_ok = mismatches.is_empty();
            let golden_error = if golden_ok {
                None
            } else {
                Some(mismatches.join("; "))
            };
            if let Some(err) = golden_error.as_ref() {
                op_report
                    .errors
                    .push(format!("model FAST golden mismatch: {err}"));
            }
            op_report.golden = Some(ModelFastGoldenComparison {
                golden_compared: true,
                golden_ok,
                golden_error,
                tensor_compared: tensor.compared,
                tensor_count: tensor.count,
                max_abs_error: tensor.max_abs_error,
                tensor_errors: tensor.errors,
            });
        }
        Err(err) => {
            let message = err.to_string();
            op_report
                .errors
                .push(format!("model FAST golden comparison failed: {message}"));
            op_report.golden = Some(ModelFastGoldenComparison {
                golden_compared: true,
                golden_ok: false,
                golden_error: Some(message),
                tensor_compared: false,
                tensor_count: 0,
                max_abs_error: 0,
                tensor_errors: Vec::new(),
            });
        }
    }
}

fn model_fast_golden_mismatches(
    op_report: &ModelFastOpReport,
    golden: &ModelFastGolden,
) -> Vec<String> {
    let mut mismatches = Vec::new();
    if golden.schema != MODEL_FAST_GOLDEN_SCHEMA {
        mismatches.push(format!(
            "unsupported model FAST golden schema `{}`",
            golden.schema
        ));
    }
    if golden.op_id != op_report.op_id {
        mismatches.push(format!(
            "golden op_id `{}` does not match op `{}`",
            golden.op_id, op_report.op_id
        ));
    }
    if golden.op_kind != op_report.op_kind {
        mismatches.push(format!(
            "golden op_kind `{}` does not match op kind `{}`",
            golden.op_kind, op_report.op_kind
        ));
    }
    compare_model_fast_golden_counter(
        &mut mismatches,
        "items",
        golden.expected.items,
        op_report.totals.items,
    );
    compare_model_fast_golden_counter(
        &mut mismatches,
        "surrogate_replacements",
        golden.expected.surrogate_replacements,
        op_report.totals.surrogate_replacements,
    );
    compare_model_fast_golden_counter(
        &mut mismatches,
        "exact_fallbacks",
        golden.expected.exact_fallbacks,
        op_report.totals.exact_fallbacks,
    );
    compare_model_fast_golden_counter(
        &mut mismatches,
        "fail_closed",
        golden.expected.fail_closed,
        op_report.totals.fail_closed,
    );
    compare_model_fast_golden_counter(
        &mut mismatches,
        "shadow_sampled",
        golden.expected.shadow_sampled,
        op_report.totals.shadow_sampled,
    );
    mismatches
}

fn compare_model_fast_golden_counter(
    mismatches: &mut Vec<String>,
    name: &str,
    expected: usize,
    actual: usize,
) {
    if expected != actual {
        mismatches.push(format!(
            "{name} expected {expected} but actual was {actual}"
        ));
    }
}

#[derive(Clone, Debug, Default)]
struct ModelFastTensorComparison {
    compared: bool,
    count: usize,
    max_abs_error: i128,
    errors: Vec<String>,
}

#[derive(Clone, Debug)]
struct NormalizedTensor {
    shape: Vec<usize>,
    values: Vec<i128>,
}

fn compare_model_fast_golden_tensors(golden: &ModelFastGolden) -> ModelFastTensorComparison {
    let mut comparison = ModelFastTensorComparison::default();
    let expected = golden.expected_tensors.as_ref();
    let actual = golden.actual_tensors.as_ref();
    if expected.is_none() && actual.is_none() {
        return comparison;
    }
    comparison.compared = true;
    let allowed_error = golden.max_abs_error.unwrap_or(0);
    if allowed_error < 0 {
        comparison
            .errors
            .push("tensor max_abs_error must be non-negative".to_string());
    }
    let Some(expected) = expected else {
        comparison
            .errors
            .push("actual_tensors provided without expected_tensors".to_string());
        return comparison;
    };
    let Some(actual) = actual else {
        comparison
            .errors
            .push("expected_tensors provided without actual_tensors".to_string());
        comparison.count = expected.len();
        return comparison;
    };
    comparison.count = expected.len();
    for name in expected.keys() {
        if !actual.contains_key(name) {
            comparison
                .errors
                .push(format!("tensor `{name}` missing from actual_tensors"));
        }
    }
    for name in actual.keys() {
        if !expected.contains_key(name) {
            comparison
                .errors
                .push(format!("unexpected actual tensor `{name}`"));
        }
    }
    for (name, expected_value) in expected {
        let Some(actual_value) = actual.get(name) else {
            continue;
        };
        let expected_tensor = normalize_model_fast_tensor(expected_value);
        let actual_tensor = normalize_model_fast_tensor(actual_value);
        match (expected_tensor, actual_tensor) {
            (Ok(expected_tensor), Ok(actual_tensor)) => {
                if expected_tensor.shape != actual_tensor.shape {
                    comparison.errors.push(format!(
                        "tensor `{name}` shape {:?} does not match actual shape {:?}",
                        expected_tensor.shape, actual_tensor.shape
                    ));
                    continue;
                }
                for (index, (expected_item, actual_item)) in expected_tensor
                    .values
                    .iter()
                    .zip(actual_tensor.values.iter())
                    .enumerate()
                {
                    let error = (*expected_item - *actual_item).abs();
                    comparison.max_abs_error = comparison.max_abs_error.max(error);
                    if error > allowed_error {
                        comparison.errors.push(format!(
                            "tensor `{name}` element {index} abs_error {error} exceeds allowed {allowed_error}"
                        ));
                    }
                }
            }
            (Err(err), _) => comparison
                .errors
                .push(format!("tensor `{name}` expected parse error: {err}")),
            (_, Err(err)) => comparison
                .errors
                .push(format!("tensor `{name}` actual parse error: {err}")),
        }
    }
    comparison
}

fn normalize_model_fast_tensor(value: &serde_json::Value) -> Result<NormalizedTensor, String> {
    let mut values = Vec::new();
    let shape = normalize_model_fast_tensor_value(value, &mut values)?;
    Ok(NormalizedTensor { shape, values })
}

fn normalize_model_fast_tensor_value(
    value: &serde_json::Value,
    values: &mut Vec<i128>,
) -> Result<Vec<usize>, String> {
    match value {
        serde_json::Value::Number(number) => {
            let item = number
                .as_i64()
                .map(i128::from)
                .or_else(|| number.as_u64().map(i128::from))
                .ok_or_else(|| "tensor values must be integers".to_string())?;
            values.push(item);
            Ok(Vec::new())
        }
        serde_json::Value::Array(items) => {
            let mut child_shape = None;
            for item in items {
                let shape = normalize_model_fast_tensor_value(item, values)?;
                if let Some(expected) = child_shape.as_ref() {
                    if expected != &shape {
                        return Err("tensor arrays must be rectangular".to_string());
                    }
                } else {
                    child_shape = Some(shape);
                }
            }
            let mut shape = vec![items.len()];
            if let Some(child_shape) = child_shape {
                shape.extend(child_shape);
            }
            Ok(shape)
        }
        _ => Err("tensor values must be integers or arrays".to_string()),
    }
}

fn include_gemm_fast_report(op_report: &mut ModelFastOpReport, fast: GemmFastRunReport) {
    if fast.schema != GEMM_FAST_RUN_SCHEMA {
        op_report.errors.push(format!(
            "unsupported GEMM FAST report schema `{}`",
            fast.schema
        ));
    }
    if !fast.ok {
        op_report
            .errors
            .push("GEMM FAST report is not ok".to_string());
    }
    op_report.errors.extend(fast.errors);
    op_report.totals = ModelFastTotals {
        items: fast.count,
        surrogate_replacements: fast.surrogate_replacements,
        exact_fallbacks: fast.exact_fallbacks,
        fail_closed: fast.fail_closed,
        shadow_compared: fast.shadow_compared,
        shadow_passed: fast.shadow_passed,
        shadow_failed: fast.shadow_failed,
        shadow_sampled: fast
            .results
            .iter()
            .filter(|item| item.shadow_sampled)
            .count(),
    };
    op_report.provenance = model_fast_provenance_summary(
        fast.results
            .iter()
            .filter_map(|item| item.provenance.as_ref())
            .cloned(),
    );
}

fn include_event_fast_report(op_report: &mut ModelFastOpReport, fast: EventFastRunReport) {
    if fast.schema != EVENT_FAST_RUN_SCHEMA {
        op_report.errors.push(format!(
            "unsupported event FAST report schema `{}`",
            fast.schema
        ));
    }
    if !fast.ok {
        op_report
            .errors
            .push("event FAST report is not ok".to_string());
    }
    op_report.errors.extend(fast.errors);
    op_report.totals = ModelFastTotals {
        items: fast.count,
        surrogate_replacements: fast.surrogate_replacements,
        exact_fallbacks: fast.exact_fallbacks,
        fail_closed: fast.fail_closed,
        shadow_compared: fast.shadow_compared,
        shadow_passed: fast.shadow_passed,
        shadow_failed: fast.shadow_failed,
        shadow_sampled: fast
            .results
            .iter()
            .filter(|item| item.shadow_sampled)
            .count(),
    };
    op_report.provenance =
        model_fast_provenance_summary(fast.results.iter().map(|item| item.provenance.clone()));
}

fn model_fast_provenance_summary(
    provenance: impl IntoIterator<Item = Provenance>,
) -> Vec<ModelFastProvenanceSummary> {
    let mut summaries = BTreeMap::<String, ModelFastProvenanceSummary>::new();
    for item in provenance {
        let key = format!(
            "{}|{}|{:?}|{:?}|{}|{}|{:?}",
            item.tag,
            item.exact,
            item.model_family,
            item.artifact_format,
            item.artifact_hash,
            item.source_hash,
            item.policy
        );
        summaries
            .entry(key)
            .and_modify(|summary| summary.count += 1)
            .or_insert(ModelFastProvenanceSummary {
                tag: item.tag,
                exact: item.exact,
                surrogate_id: item.surrogate_id,
                model_family: item.model_family,
                artifact_format: item.artifact_format,
                artifact_hash: item.artifact_hash,
                source_hash: item.source_hash,
                policy: item.policy,
                count: 1,
            });
    }
    summaries.into_values().collect()
}

fn resolve_model_fast_report_path(base_dir: &Path, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    }
}

impl ModelFastTotals {
    fn include(&mut self, other: &ModelFastTotals) {
        self.items += other.items;
        self.surrogate_replacements += other.surrogate_replacements;
        self.exact_fallbacks += other.exact_fallbacks;
        self.fail_closed += other.fail_closed;
        self.shadow_compared += other.shadow_compared;
        self.shadow_passed += other.shadow_passed;
        self.shadow_failed += other.shadow_failed;
        self.shadow_sampled += other.shadow_sampled;
    }
}

impl ModelFastTimingSummary {
    fn from_ops(ops: &[ModelFastOpReport]) -> Self {
        let mut summary = Self::default();
        for op in ops {
            if let Some(timing) = &op.timing {
                summary.timed_ops += 1;
                summary.exact_ns += timing.exact_ns;
                summary.fast_ns += timing.fast_ns;
            } else {
                summary.missing_timing_ops.push(op.op_id.clone());
            }
        }
        if summary.fast_ns > 0 {
            summary.speedup = Some(summary.exact_ns as f64 / summary.fast_ns as f64);
        }
        summary
    }
}

impl ModelFastCoverage {
    fn from_totals(totals: &ModelFastTotals, ops: &[ModelFastOpReport]) -> Self {
        let op_coverage = ratio(ops.iter().filter(|op| op.ok).count(), ops.len());
        Self {
            op_coverage,
            item_coverage: ratio(totals.surrogate_replacements, totals.items),
            fallback_ratio: ratio(totals.exact_fallbacks, totals.items),
            shadow_sample_ratio: ratio(totals.shadow_sampled, totals.items),
            accepted: true,
            reject_reasons: Vec::new(),
        }
    }

    fn apply_thresholds(&mut self, thresholds: Option<&ModelFastThresholds>) {
        let Some(thresholds) = thresholds else {
            return;
        };
        self.reject_if_below("op_coverage", self.op_coverage, thresholds.min_op_coverage);
        self.reject_if_below(
            "item_coverage",
            self.item_coverage,
            thresholds.min_item_coverage,
        );
        self.reject_if_above(
            "fallback_ratio",
            self.fallback_ratio,
            thresholds.max_fallback_ratio,
        );
        self.reject_if_below(
            "shadow_sample_ratio",
            self.shadow_sample_ratio,
            thresholds.min_shadow_sample_ratio,
        );
        self.accepted = self.reject_reasons.is_empty();
    }

    fn reject_if_below(&mut self, name: &str, actual: f64, minimum: Option<f64>) {
        if let Some(minimum) = minimum {
            if actual < minimum {
                self.reject_reasons.push(format!(
                    "{name} {actual:.6} is below required minimum {minimum:.6}"
                ));
            }
        }
    }

    fn reject_if_above(&mut self, name: &str, actual: f64, maximum: Option<f64>) {
        if let Some(maximum) = maximum {
            if actual > maximum {
                self.reject_reasons.push(format!(
                    "{name} {actual:.6} is above allowed maximum {maximum:.6}"
                ));
            }
        }
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn policy_event_item(
    index: usize,
    event: &InstrumentationEvent,
    lane: usize,
    predicted: i64,
    expected: Option<i64>,
    provenance: &Provenance,
    exact_provenance: &Provenance,
    mode: PolicyMode,
    fallback: FallbackPolicy,
) -> EventPolicyItemResult {
    match mode {
        PolicyMode::ShadowCompare => EventPolicyItemResult {
            index,
            sample_id: event.sample_id,
            lane,
            target: event.target.clone(),
            decision: EventPolicyDecision::ShadowCompare,
            ok: true,
            predicted,
            expected,
            provenance: exact_provenance.clone(),
            error: None,
        },
        PolicyMode::TelemetryOnly => EventPolicyItemResult {
            index,
            sample_id: event.sample_id,
            lane,
            target: event.target.clone(),
            decision: EventPolicyDecision::ExactFallback,
            ok: true,
            predicted,
            expected,
            provenance: exact_provenance.clone(),
            error: None,
        },
        PolicyMode::ApproximateWithTolerance if expected == Some(predicted) => {
            EventPolicyItemResult {
                index,
                sample_id: event.sample_id,
                lane,
                target: event.target.clone(),
                decision: EventPolicyDecision::SurrogateUsed,
                ok: true,
                predicted,
                expected,
                provenance: provenance.clone(),
                error: None,
            }
        }
        PolicyMode::ApproximateWithTolerance => match fallback {
            FallbackPolicy::ExactFallback => EventPolicyItemResult {
                index,
                sample_id: event.sample_id,
                lane,
                target: event.target.clone(),
                decision: EventPolicyDecision::ExactFallback,
                ok: true,
                predicted,
                expected,
                provenance: exact_provenance.clone(),
                error: None,
            },
            FallbackPolicy::FailClosed => EventPolicyItemResult {
                index,
                sample_id: event.sample_id,
                lane,
                target: event.target.clone(),
                decision: EventPolicyDecision::FailClosed,
                ok: false,
                predicted,
                expected,
                provenance: provenance.clone(),
                error: Some("event prediction failed policy comparison".to_string()),
            },
        },
    }
}

pub fn plan_runtime_events(policy: &EventPolicyReport, total_lanes: usize) -> EventRuntimePlan {
    let workers = if total_lanes == 0 {
        Vec::new()
    } else {
        vec![GemmRuntimeWorkerSpec {
            worker_id: "worker0".to_string(),
            start_lane: 0,
            lanes: total_lanes,
        }]
    };
    plan_runtime_events_for_workers(policy, &workers)
}

pub fn plan_runtime_events_for_workers(
    policy: &EventPolicyReport,
    workers: &[GemmRuntimeWorkerSpec],
) -> EventRuntimePlan {
    let mut errors = Vec::new();
    let mut items = Vec::new();
    let total_lanes = workers
        .iter()
        .map(|worker| worker.start_lane.saturating_add(worker.lanes))
        .max()
        .unwrap_or(0);

    if policy.schema != EVENT_POLICY_REPORT_SCHEMA {
        errors.push(format!(
            "unsupported event policy report schema `{}`",
            policy.schema
        ));
    }
    if policy.results.is_empty() {
        errors.push("event policy report must contain at least one result".to_string());
    }
    if policy.fail_closed > 0 {
        errors.push(format!(
            "event policy report contains {} fail-closed item(s)",
            policy.fail_closed
        ));
    }
    if workers.is_empty() {
        errors.push("runtime topology must contain at least one worker".to_string());
    }
    if let Err(err) = validate_runtime_worker_specs(workers) {
        errors.push(err);
    }

    for item in &policy.results {
        match runtime_event_plan_item(item, workers, total_lanes) {
            Ok(plan_item) => items.push(plan_item),
            Err(err) => errors.push(err),
        }
    }

    let worker_summaries = workers
        .iter()
        .map(|worker| {
            let worker_items = items
                .iter()
                .filter(|item| item.worker_id == worker.worker_id)
                .collect::<Vec<_>>();
            EventRuntimeWorkerSummary {
                worker_id: worker.worker_id.clone(),
                start_lane: worker.start_lane,
                lanes: worker.lanes,
                assigned_items: worker_items.len(),
                used_surrogate: worker_items
                    .iter()
                    .filter(|item| item.source_result == GemmRuntimeSourceResult::Surrogate)
                    .count(),
                exact_fallbacks: worker_items
                    .iter()
                    .filter(|item| item.source_result == GemmRuntimeSourceResult::Exact)
                    .count(),
            }
        })
        .collect();

    EventRuntimePlan {
        schema: EVENT_RUNTIME_PLAN_SCHEMA.to_string(),
        ok: errors.is_empty(),
        total_lanes,
        workers: worker_summaries,
        items,
        errors,
    }
}

fn runtime_event_plan_item(
    item: &EventPolicyItemResult,
    workers: &[GemmRuntimeWorkerSpec],
    total_lanes: usize,
) -> Result<EventRuntimePlanItem, String> {
    if !item.ok {
        return Err(format!("event policy item {} is not ok", item.index));
    }
    if item.decision == EventPolicyDecision::FailClosed {
        return Err(format!("event policy item {} is fail-closed", item.index));
    }
    if total_lanes == 0 || item.lane >= total_lanes {
        return Err(format!(
            "event policy item {} lane {} is outside runtime topology lanes {}",
            item.index, item.lane, total_lanes
        ));
    }
    let worker = worker_for_lane(workers, item.lane).ok_or_else(|| {
        format!(
            "event policy item {} lane {} is not covered by runtime workers",
            item.index, item.lane
        )
    })?;
    let source_result = match item.decision {
        EventPolicyDecision::SurrogateUsed => GemmRuntimeSourceResult::Surrogate,
        EventPolicyDecision::ExactFallback | EventPolicyDecision::ShadowCompare => {
            GemmRuntimeSourceResult::Exact
        }
        EventPolicyDecision::FailClosed => unreachable!(),
    };
    match source_result {
        GemmRuntimeSourceResult::Surrogate if item.provenance.exact => {
            return Err(format!(
                "event policy item {} surrogate result has exact provenance",
                item.index
            ));
        }
        GemmRuntimeSourceResult::Exact if !item.provenance.exact => {
            return Err(format!(
                "event policy item {} exact fallback result lacks exact provenance",
                item.index
            ));
        }
        _ => {}
    }
    Ok(EventRuntimePlanItem {
        index: item.index,
        sample_id: item.sample_id,
        lane: item.lane,
        target: item.target.clone(),
        worker_id: worker.worker_id.clone(),
        decision: item.decision,
        provenance: item.provenance.clone(),
        source_result,
        predicted: Some(item.predicted),
        expected: item.expected,
        shadow_ok: if item.decision == EventPolicyDecision::ShadowCompare {
            Some(item.expected == Some(item.predicted))
        } else {
            None
        },
        shadow_error: if item.decision == EventPolicyDecision::ShadowCompare
            && item.expected != Some(item.predicted)
        {
            Some("event shadow prediction diverged from exact label".to_string())
        } else {
            item.error.clone()
        },
    })
}

fn event_provenance(manifest: &SurrogateManifest, exact: bool) -> Provenance {
    Provenance {
        tag: if exact {
            "exact".to_string()
        } else if manifest.policy.provenance_tag.is_empty() {
            "instrumentation_prediction".to_string()
        } else {
            manifest.policy.provenance_tag.clone()
        },
        exact,
        surrogate_id: manifest.surrogate_id.clone(),
        model_family: manifest.model_family,
        artifact_format: manifest.artifact.format,
        artifact_hash: manifest.artifact.sha256.clone(),
        source_hash: manifest.source.source_hash.clone(),
        policy: manifest.policy.mode,
    }
}

impl GemmBatchMetrics {
    fn include_gemm_metrics(&mut self, metrics: &GemmValidationMetrics) {
        self.max_abs_error = self.max_abs_error.max(metrics.max_abs_error);
        self.max_mean_abs_error = self.max_mean_abs_error.max(metrics.mean_abs_error);
        self.max_latency_error_cycles = self
            .max_latency_error_cycles
            .max(metrics.latency_error_cycles);
    }
}

#[derive(Default)]
struct GemmBatchLaneAccumulator {
    count: usize,
    ok: usize,
    failed: usize,
    metrics: GemmBatchMetrics,
}

pub fn validate_event_corpus(
    manifest: &SurrogateManifest,
    manifest_path: impl AsRef<Path>,
    corpus: &InstrumentationEventCorpus,
) -> EventValidationReport {
    let mut manifest_report = validate_manifest_path(manifest, &manifest_path);
    let mut errors = validate_event_corpus_shape(manifest, corpus);
    let task = manifest.task.as_ref();
    let target = task
        .map(|task| task.prediction_target.clone())
        .unwrap_or_default();
    let window_cycles = task.map_or(0, |task| task.input_window_cycles);
    let horizon_cycles = task.map_or(0, |task| task.horizon_cycles);
    let mut metrics = EventValidationMetrics::default();
    let mut first_mismatch = None;

    if manifest_report.ok && errors.is_empty() {
        match event_predictions(manifest, manifest_path.as_ref(), corpus) {
            Ok(predictions) => {
                apply_event_validation_metrics(&predictions, &mut metrics, &mut first_mismatch);
            }
            Err(err) => errors.push(err.to_string()),
        }
    }

    manifest_report.ok = manifest_report.errors.is_empty();
    let ok = manifest_report.ok && errors.is_empty() && first_mismatch.is_none();
    EventValidationReport {
        schema: EVENT_VALIDATION_SCHEMA.to_string(),
        ok,
        errors,
        manifest: manifest_report,
        corpus: EventCorpusSummary {
            source_hash: corpus.source_hash.clone(),
            top_name: corpus.top_name.clone(),
            target,
            samples: corpus.events.len(),
            window_cycles,
            horizon_cycles,
        },
        metrics,
        first_mismatch,
        provenance: Provenance {
            tag: if manifest.policy.provenance_tag.is_empty() {
                "instrumentation_prediction".to_string()
            } else {
                manifest.policy.provenance_tag.clone()
            },
            exact: false,
            surrogate_id: manifest.surrogate_id.clone(),
            model_family: manifest.model_family,
            artifact_format: manifest.artifact.format,
            artifact_hash: manifest.artifact.sha256.clone(),
            source_hash: manifest.source.source_hash.clone(),
            policy: manifest.policy.mode,
        },
    }
}

pub fn shadow_event_corpus(
    manifest: &SurrogateManifest,
    manifest_path: impl AsRef<Path>,
    corpus: &InstrumentationEventCorpus,
) -> EventShadowReport {
    let mut manifest_report = validate_manifest_path(manifest, &manifest_path);
    let mut errors = validate_event_corpus_shape(manifest, corpus);
    let task = manifest.task.as_ref();
    let target = task
        .map(|task| task.prediction_target.clone())
        .unwrap_or_default();
    let window_cycles = task.map_or(0, |task| task.input_window_cycles);
    let horizon_cycles = task.map_or(0, |task| task.horizon_cycles);
    let mut metrics = EventValidationMetrics::default();
    let mut first_mismatch = None;
    let mut results = Vec::new();
    let mut lanes = BTreeMap::<usize, EventLaneShadowAccumulator>::new();

    if manifest_report.ok && errors.is_empty() {
        match event_predictions(manifest, manifest_path.as_ref(), corpus) {
            Ok(predictions) => {
                apply_event_shadow_metrics(
                    &predictions,
                    &mut metrics,
                    &mut first_mismatch,
                    &mut results,
                    &mut lanes,
                );
            }
            Err(err) => errors.push(err.to_string()),
        }
    }

    manifest_report.ok = manifest_report.errors.is_empty();
    let ok = manifest_report.ok && errors.is_empty() && first_mismatch.is_none();
    let lanes = lanes
        .into_iter()
        .map(|(lane, summary)| EventLaneShadowSummary {
            lane,
            samples: summary.samples,
            metrics: summary.metrics(),
        })
        .collect::<Vec<_>>();
    let total_lanes = lanes.len();
    EventShadowReport {
        schema: EVENT_SHADOW_SCHEMA.to_string(),
        ok,
        errors,
        manifest: manifest_report,
        corpus: EventCorpusSummary {
            source_hash: corpus.source_hash.clone(),
            top_name: corpus.top_name.clone(),
            target,
            samples: corpus.events.len(),
            window_cycles,
            horizon_cycles,
        },
        total_lanes,
        metrics,
        lanes,
        first_mismatch,
        results,
        provenance: Provenance {
            tag: if manifest.policy.provenance_tag.is_empty() {
                "instrumentation_prediction".to_string()
            } else {
                manifest.policy.provenance_tag.clone()
            },
            exact: false,
            surrogate_id: manifest.surrogate_id.clone(),
            model_family: manifest.model_family,
            artifact_format: manifest.artifact.format,
            artifact_hash: manifest.artifact.sha256.clone(),
            source_hash: manifest.source.source_hash.clone(),
            policy: manifest.policy.mode,
        },
    }
}

#[derive(Default)]
struct EventLaneShadowAccumulator {
    samples: usize,
    correct: usize,
    false_positive: usize,
    false_negative: usize,
}

impl EventLaneShadowAccumulator {
    fn metrics(&self) -> EventValidationMetrics {
        EventValidationMetrics {
            accuracy: if self.samples == 0 {
                0.0
            } else {
                self.correct as f64 / self.samples as f64
            },
            false_positive: self.false_positive,
            false_negative: self.false_negative,
        }
    }
}

pub fn render_validation_markdown(report: &SurrogateValidationReport) -> String {
    let mut lines = vec![
        "# RRTL Surrogate Validation".to_string(),
        String::new(),
        format!("- OK: {}", report.ok),
        format!("- Surrogate: `{}`", report.manifest.surrogate_id),
        format!("- Source hash: `{}`", report.dataset.source_hash),
        format!("- Nodes: {}", report.dataset.nodes),
        format!("- Edges: {}", report.dataset.edges),
        format!("- Trace steps: {}", report.dataset.steps),
    ];
    if !report.manifest.errors.is_empty() {
        lines.push(String::new());
        lines.push("## Errors".to_string());
        for err in &report.manifest.errors {
            lines.push(format!("- {err}"));
        }
    }
    if !report.manifest.warnings.is_empty() {
        lines.push(String::new());
        lines.push("## Warnings".to_string());
        for warning in &report.manifest.warnings {
            lines.push(format!("- {warning}"));
        }
    }
    lines.push(String::new());
    lines.join("\n")
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn canonical_json_hash(value: &serde_json::Value) -> Result<String, Error> {
    let bytes = serde_json::to_vec(value)
        .map_err(|err| Error::new(format!("failed to serialize canonical json: {err}")))?;
    Ok(sha256_hex(&bytes))
}

fn resolve_artifact_path(base_dir: impl AsRef<Path>, artifact: &str) -> PathBuf {
    let path = PathBuf::from(artifact);
    if path.is_absolute() {
        path
    } else {
        base_dir.as_ref().join(path)
    }
}

fn require_tensor_names(
    kind: &str,
    actual: &[String],
    required: &[&str],
    errors: &mut Vec<String>,
) {
    let actual = actual.iter().map(String::as_str).collect::<BTreeSet<_>>();
    for required in required {
        if !actual.contains(required) {
            errors.push(format!(
                "onnx artifact missing required {kind} tensor `{required}`"
            ));
        }
    }
}

fn validate_event_emitter_config(config: &EventEmitterConfig) -> Result<(), Error> {
    let mut errors = Vec::new();
    if config.schema != EVENT_EMITTER_CONFIG_SCHEMA {
        errors.push(format!(
            "unsupported event emitter config schema `{}`",
            config.schema
        ));
    }
    if let Some(input_schema) = &config.input_schema {
        if input_schema != "rrtl-pyrtl-trace-v1"
            && input_schema != RRTL_INSTRUMENTATION_TRACE_SCHEMA
        {
            errors.push(format!(
                "unsupported event emitter config input_schema `{input_schema}`"
            ));
        }
    }
    if config.top_name.trim().is_empty() {
        errors.push("event emitter config top_name must not be empty".to_string());
    }
    if config.source_hash.trim().is_empty() {
        errors.push("event emitter config source_hash must not be empty".to_string());
    }
    if config.target.trim().is_empty() {
        errors.push("event emitter config target must not be empty".to_string());
    }
    if config.window_cycles == 0 {
        errors.push("event emitter config window_cycles must be greater than zero".to_string());
    }
    if config.horizon_cycles == 0 {
        errors.push("event emitter config horizon_cycles must be greater than zero".to_string());
    }
    if config.signal_features.is_empty() {
        errors.push("event emitter config signal_features must not be empty".to_string());
    }
    for mapping in config
        .signal_features
        .iter()
        .chain(config.program_features.iter())
        .chain(config.lane.iter())
        .chain(std::iter::once(&config.label))
    {
        validate_event_feature_mapping(mapping, &mut errors);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(Error::new(errors.join("; ")))
    }
}

fn validate_event_feature_mapping(mapping: &EventFeatureMapping, errors: &mut Vec<String>) {
    if mapping.name.trim().is_empty() {
        errors.push("event feature mapping name must not be empty".to_string());
    }
    match (mapping.constant, mapping.source, mapping.field.as_ref()) {
        (Some(_), None, None) => {}
        (None, Some(_), Some(field)) if !field.trim().is_empty() => {}
        (Some(_), Some(_), _) | (Some(_), _, Some(_)) => errors.push(format!(
            "event feature `{}` must use either constant or source+field, not both",
            mapping.name
        )),
        _ => errors.push(format!(
            "event feature `{}` must define constant or source+field",
            mapping.name
        )),
    }
}

fn resolve_feature(step: &serde_json::Value, mapping: &EventFeatureMapping) -> Result<i64, Error> {
    if let Some(value) = mapping.constant {
        return Ok(value);
    }
    let source = mapping
        .source
        .ok_or_else(|| Error::new(format!("event feature `{}` missing source", mapping.name)))?;
    let field = mapping
        .field
        .as_ref()
        .ok_or_else(|| Error::new(format!("event feature `{}` missing field", mapping.name)))?;
    let object_name = match source {
        EventTraceSource::Inputs => "inputs",
        EventTraceSource::Outputs => "outputs",
    };
    let value = required_object(step, object_name)?
        .get(field)
        .ok_or_else(|| Error::new(format!("trace step missing {object_name} field `{field}`")))?;
    value_to_i64(value)
}

fn resolve_lane(step: &serde_json::Value, mapping: &EventFeatureMapping) -> Result<usize, Error> {
    let value = resolve_feature(step, mapping)?;
    usize::try_from(value).map_err(|_| {
        Error::new(format!(
            "event lane feature `{}` must be a non-negative integer, got {}",
            mapping.name, value
        ))
    })
}

fn validate_transaction_domain(
    manifest: &SurrogateManifest,
    transaction: &GemmTransaction,
) -> Result<(), Error> {
    if transaction.rows != manifest.domain.rows || transaction.cols != manifest.domain.cols {
        return Err(Error::new(format!(
            "transaction shape {}x{} is outside manifest domain {}x{}",
            transaction.rows, transaction.cols, manifest.domain.rows, manifest.domain.cols
        )));
    }
    if transaction.k < manifest.domain.k_min || transaction.k > manifest.domain.k_max {
        return Err(Error::new(format!(
            "transaction k={} is outside manifest domain {}..={}",
            transaction.k, manifest.domain.k_min, manifest.domain.k_max
        )));
    }
    if transaction.a.len() != transaction.rows {
        return Err(Error::new("transaction A row count does not match rows"));
    }
    if transaction.w.len() != transaction.k {
        return Err(Error::new("transaction W row count does not match k"));
    }
    for (row, values) in transaction.a.iter().enumerate() {
        if values.len() != transaction.k {
            return Err(Error::new(format!(
                "transaction A row {row} has {}, expected {}",
                values.len(),
                transaction.k
            )));
        }
    }
    for (row, values) in transaction.w.iter().enumerate() {
        if values.len() != transaction.cols {
            return Err(Error::new(format!(
                "transaction W row {row} has {}, expected {}",
                values.len(),
                transaction.cols
            )));
        }
    }
    Ok(())
}

fn validate_transaction_shape(transaction: &GemmTransaction) -> Result<(), Error> {
    if transaction.a.len() != transaction.rows {
        return Err(Error::new("transaction A row count does not match rows"));
    }
    if transaction.w.len() != transaction.k {
        return Err(Error::new("transaction W row count does not match k"));
    }
    for (row, values) in transaction.a.iter().enumerate() {
        if values.len() != transaction.k {
            return Err(Error::new(format!(
                "transaction A row {row} has {}, expected {}",
                values.len(),
                transaction.k
            )));
        }
    }
    for (row, values) in transaction.w.iter().enumerate() {
        if values.len() != transaction.cols {
            return Err(Error::new(format!(
                "transaction W row {row} has {}, expected {}",
                values.len(),
                transaction.cols
            )));
        }
    }
    Ok(())
}

fn exact_gemm_result(
    manifest: &SurrogateManifest,
    transaction: &GemmTransaction,
) -> Result<GemmRunResult, Error> {
    validate_transaction_shape(transaction)?;
    let c = mock_gemm(transaction);
    let latency_cycles = (transaction.k + transaction.rows + transaction.cols - 1) as u64;
    let telemetry = GemmTelemetry {
        latency_cycles,
        active_cycles: transaction.k as u64,
        utilization: if latency_cycles == 0 {
            0.0
        } else {
            transaction.k as f64 / latency_cycles as f64
        },
    };
    let metrics = transaction.expected_c.as_ref().map(|expected| {
        gemm_metrics(
            &c,
            expected,
            telemetry.latency_cycles,
            transaction.expected_latency_cycles,
        )
    });
    Ok(GemmRunResult {
        schema: GEMM_RESULT_SCHEMA.to_string(),
        surrogate_id: manifest.surrogate_id.clone(),
        ok: true,
        c,
        telemetry,
        metrics,
        provenance: Provenance {
            tag: "exact".to_string(),
            exact: true,
            surrogate_id: manifest.surrogate_id.clone(),
            model_family: manifest.model_family,
            artifact_format: manifest.artifact.format,
            artifact_hash: manifest.artifact.sha256.clone(),
            source_hash: manifest.source.source_hash.clone(),
            policy: manifest.policy.mode,
        },
    })
}

fn mock_gemm(transaction: &GemmTransaction) -> Vec<Vec<i128>> {
    let mut c = vec![vec![0i128; transaction.cols]; transaction.rows];
    for row in 0..transaction.rows {
        for col in 0..transaction.cols {
            let mut acc = 0i128;
            for k in 0..transaction.k {
                acc += transaction.a[row][k] as i128 * transaction.w[k][col] as i128;
            }
            c[row][col] = acc;
        }
    }
    c
}

fn validate_event_corpus_shape(
    manifest: &SurrogateManifest,
    corpus: &InstrumentationEventCorpus,
) -> Vec<String> {
    let mut errors = Vec::new();
    if manifest.surrogate_class != SurrogateClass::EventPredictor {
        errors.push(format!(
            "validate-events requires an event_predictor manifest; got {:?}",
            manifest.surrogate_class
        ));
    }
    if !matches!(
        manifest.artifact.format,
        ArtifactFormat::MockEventPredictor | ArtifactFormat::Onnx
    ) {
        errors.push(format!(
            "validate-events requires a mock-event-predictor or onnx artifact; got {:?}",
            manifest.artifact.format
        ));
    }
    if corpus.schema != EVENT_CORPUS_SCHEMA {
        errors.push(format!(
            "unsupported event corpus schema `{}`",
            corpus.schema
        ));
    }
    if corpus.source_hash != manifest.source.source_hash {
        errors.push(format!(
            "source hash mismatch: manifest {}, actual {}",
            manifest.source.source_hash, corpus.source_hash
        ));
    }
    if corpus.top_name != manifest.source.top_name {
        errors.push(format!(
            "top name mismatch: manifest `{}`, actual `{}`",
            manifest.source.top_name, corpus.top_name
        ));
    }
    if manifest.source.export_schema != EVENT_CORPUS_SCHEMA {
        errors.push(format!(
            "export schema mismatch: manifest `{}`, expected `{}`",
            manifest.source.export_schema, EVENT_CORPUS_SCHEMA
        ));
    }
    if corpus.events.is_empty() {
        errors.push("event corpus must contain at least one event".to_string());
    }

    let Some(task) = &manifest.task else {
        errors.push("event_predictor manifests require a task section".to_string());
        return errors;
    };
    let label_name = task
        .label
        .as_ref()
        .map(|label| label.name.as_str())
        .unwrap_or("");
    for event in &corpus.events {
        if event.schema != EVENT_SCHEMA {
            errors.push(format!(
                "event {} has unsupported schema `{}`",
                event.sample_id, event.schema
            ));
        }
        if event.target != task.prediction_target {
            errors.push(format!(
                "event {} target `{}` does not match manifest target `{}`",
                event.sample_id, event.target, task.prediction_target
            ));
        }
        if event.window_cycles != task.input_window_cycles {
            errors.push(format!(
                "event {} window {} does not match manifest window {}",
                event.sample_id, event.window_cycles, task.input_window_cycles
            ));
        }
        if event.horizon_cycles != task.horizon_cycles {
            errors.push(format!(
                "event {} horizon {} does not match manifest horizon {}",
                event.sample_id, event.horizon_cycles, task.horizon_cycles
            ));
        }
        if event.signals.len() != task.input_window_cycles {
            errors.push(format!(
                "event {} has {} signal steps, expected {}",
                event.sample_id,
                event.signals.len(),
                task.input_window_cycles
            ));
        }
        for feature in &task.program_features {
            if !event.program.contains_key(feature) {
                errors.push(format!(
                    "event {} missing program feature `{}`",
                    event.sample_id, feature
                ));
            }
        }
        for (step_index, step) in event.signals.iter().enumerate() {
            for feature in &task.signal_features {
                if !step.contains_key(feature) {
                    errors.push(format!(
                        "event {} signal step {} missing feature `{}`",
                        event.sample_id, step_index, feature
                    ));
                }
            }
        }
        match event.label.get(label_name).copied() {
            Some(0 | 1) => {}
            Some(value) => errors.push(format!(
                "event {} label `{}` must be binary, got {}",
                event.sample_id, label_name, value
            )),
            None => errors.push(format!(
                "event {} missing label `{}`",
                event.sample_id, label_name
            )),
        }
    }
    errors
}

#[derive(Clone, Debug)]
struct EventPrediction {
    sample_id: usize,
    lane: usize,
    expected: i64,
    predicted: i64,
}

fn event_predictions(
    manifest: &SurrogateManifest,
    manifest_path: &Path,
    corpus: &InstrumentationEventCorpus,
) -> Result<Vec<EventPrediction>, Error> {
    let task = manifest
        .task
        .as_ref()
        .ok_or_else(|| Error::new("event_predictor manifests require a task section"))?;
    let label_name = task
        .label
        .as_ref()
        .map(|label| label.name.as_str())
        .unwrap_or("");
    let base_dir = manifest_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let artifact_path = resolve_artifact_path(&base_dir, &manifest.artifact.path);

    let mock_artifact = if manifest.artifact.format == ArtifactFormat::MockEventPredictor {
        read_mock_event_artifact(&artifact_path).ok()
    } else {
        None
    };

    corpus
        .events
        .iter()
        .map(|event| {
            let expected = *event.label.get(label_name).unwrap_or(&0);
            let predicted = match manifest.artifact.format {
                ArtifactFormat::MockEventPredictor => mock_event_prediction(
                    manifest,
                    mock_artifact.as_ref(),
                    event,
                )?,
                ArtifactFormat::Onnx => run_onnx_event_predictor(manifest, event, &artifact_path)?,
                other => {
                    return Err(Error::new(format!(
                        "event prediction supports mock-event-predictor or onnx artifacts; got {:?}",
                        other
                    )));
                }
            };
            Ok(EventPrediction {
                sample_id: event.sample_id,
                lane: event.lane.unwrap_or(0),
                expected,
                predicted,
            })
        })
        .collect()
}

fn apply_event_validation_metrics(
    predictions: &[EventPrediction],
    metrics: &mut EventValidationMetrics,
    first_mismatch: &mut Option<EventMismatch>,
) {
    let mut correct = 0usize;
    for prediction in predictions {
        if prediction.predicted == prediction.expected {
            correct += 1;
        } else {
            record_event_mismatch(prediction, metrics, first_mismatch, None);
        }
    }
    metrics.accuracy = if predictions.is_empty() {
        0.0
    } else {
        correct as f64 / predictions.len() as f64
    };
}

fn apply_event_shadow_metrics(
    predictions: &[EventPrediction],
    metrics: &mut EventValidationMetrics,
    first_mismatch: &mut Option<EventMismatch>,
    results: &mut Vec<EventShadowSample>,
    lanes: &mut BTreeMap<usize, EventLaneShadowAccumulator>,
) {
    let mut correct = 0usize;
    for prediction in predictions {
        let sample_ok = prediction.predicted == prediction.expected;
        let lane_metrics = lanes.entry(prediction.lane).or_default();
        lane_metrics.samples += 1;
        if sample_ok {
            correct += 1;
            lane_metrics.correct += 1;
        } else {
            record_event_mismatch(prediction, metrics, first_mismatch, Some(lane_metrics));
        }
        results.push(EventShadowSample {
            sample_id: prediction.sample_id,
            lane: prediction.lane,
            expected: prediction.expected,
            predicted: prediction.predicted,
            ok: sample_ok,
        });
    }
    metrics.accuracy = if predictions.is_empty() {
        0.0
    } else {
        correct as f64 / predictions.len() as f64
    };
}

fn record_event_mismatch(
    prediction: &EventPrediction,
    metrics: &mut EventValidationMetrics,
    first_mismatch: &mut Option<EventMismatch>,
    lane_metrics: Option<&mut EventLaneShadowAccumulator>,
) {
    if first_mismatch.is_none() {
        *first_mismatch = Some(EventMismatch {
            sample_id: prediction.sample_id,
            expected: prediction.expected,
            predicted: prediction.predicted,
        });
    }
    if prediction.predicted == 1 && prediction.expected == 0 {
        metrics.false_positive += 1;
        if let Some(lane_metrics) = lane_metrics {
            lane_metrics.false_positive += 1;
        }
    } else if prediction.predicted == 0 && prediction.expected == 1 {
        metrics.false_negative += 1;
        if let Some(lane_metrics) = lane_metrics {
            lane_metrics.false_negative += 1;
        }
    }
}

fn mock_event_prediction(
    manifest: &SurrogateManifest,
    artifact: Option<&MockEventArtifact>,
    event: &InstrumentationEvent,
) -> Result<i64, Error> {
    if let Some(artifact) = artifact {
        return evaluate_mock_event_rule(&artifact.mock_rule, event);
    }
    let task = manifest
        .task
        .as_ref()
        .ok_or_else(|| Error::new("event_predictor manifests require a task section"))?;
    match task.prediction_target.as_str() {
        "cache_miss" => Ok(mock_cache_miss_event(event)),
        "stall_event" => Ok(mock_stall_event(event)),
        other => Err(Error::new(format!(
            "mock-event-predictor does not support event target `{other}`"
        ))),
    }
}

#[derive(Clone, Debug, Deserialize)]
struct MockEventArtifact {
    #[allow(dead_code)]
    schema: String,
    #[allow(dead_code)]
    prediction_target: String,
    mock_rule: MockEventRule,
}

#[derive(Clone, Debug, Deserialize)]
struct MockEventRule {
    kind: String,
    threshold: i64,
    terms: Vec<MockEventRuleTerm>,
    #[serde(default)]
    bonuses: Vec<MockEventRuleBonus>,
}

#[derive(Clone, Debug, Deserialize)]
struct MockEventRuleTerm {
    source: String,
    feature: String,
    #[serde(default = "default_rule_reduction")]
    reduction: String,
    #[serde(default = "default_rule_weight")]
    weight: i64,
}

#[derive(Clone, Debug, Deserialize)]
struct MockEventRuleBonus {
    source: String,
    feature: String,
    #[serde(default = "default_rule_reduction")]
    reduction: String,
    #[serde(default = "default_rule_op")]
    op: String,
    value: i64,
    bonus: i64,
}

fn default_rule_reduction() -> String {
    "sum".to_string()
}

fn default_rule_weight() -> i64 {
    1
}

fn default_rule_op() -> String {
    ">=".to_string()
}

fn read_mock_event_artifact(path: &Path) -> Result<MockEventArtifact, Error> {
    let text = fs::read_to_string(path).map_err(|err| {
        Error::new(format!(
            "failed to read mock-event-predictor artifact `{}`: {err}",
            path.display()
        ))
    })?;
    serde_json::from_str(&text).map_err(|err| {
        Error::new(format!(
            "failed to parse mock-event-predictor artifact `{}`: {err}",
            path.display()
        ))
    })
}

fn evaluate_mock_event_rule(
    rule: &MockEventRule,
    event: &InstrumentationEvent,
) -> Result<i64, Error> {
    if rule.kind != "linear_threshold" {
        return Err(Error::new(format!(
            "unsupported mock-event-predictor rule kind `{}`",
            rule.kind
        )));
    }
    let mut score = 0i64;
    for term in &rule.terms {
        score += mock_event_rule_term_value(term, event)? * term.weight;
    }
    for bonus in &rule.bonuses {
        let value = mock_event_rule_bonus_value(bonus, event)?;
        if compare_mock_rule_value(value, &bonus.op, bonus.value)? {
            score += bonus.bonus;
        }
    }
    Ok(if score >= rule.threshold { 1 } else { 0 })
}

fn mock_event_rule_term_value(
    term: &MockEventRuleTerm,
    event: &InstrumentationEvent,
) -> Result<i64, Error> {
    mock_event_rule_value(&term.source, &term.feature, &term.reduction, event)
}

fn mock_event_rule_bonus_value(
    bonus: &MockEventRuleBonus,
    event: &InstrumentationEvent,
) -> Result<i64, Error> {
    mock_event_rule_value(&bonus.source, &bonus.feature, &bonus.reduction, event)
}

fn mock_event_rule_value(
    source: &str,
    feature: &str,
    reduction: &str,
    event: &InstrumentationEvent,
) -> Result<i64, Error> {
    if source == "program" {
        return Ok(*event.program.get(feature).unwrap_or(&0));
    }
    if source != "signal" {
        return Err(Error::new(format!(
            "unsupported mock-event-predictor rule source `{source}`"
        )));
    }
    let values = event
        .signals
        .iter()
        .map(|step| *step.get(feature).unwrap_or(&0))
        .collect::<Vec<_>>();
    match reduction {
        "sum" => Ok(values.iter().sum()),
        "max" => Ok(values.into_iter().max().unwrap_or(0)),
        "min" => Ok(values.into_iter().min().unwrap_or(0)),
        "last" => Ok(values.last().copied().unwrap_or(0)),
        "unique_count" => Ok(values.into_iter().collect::<BTreeSet<_>>().len() as i64),
        other => Err(Error::new(format!(
            "unsupported mock-event-predictor rule reduction `{other}`"
        ))),
    }
}

fn compare_mock_rule_value(actual: i64, op: &str, expected: i64) -> Result<bool, Error> {
    match op {
        ">=" => Ok(actual >= expected),
        ">" => Ok(actual > expected),
        "<=" => Ok(actual <= expected),
        "<" => Ok(actual < expected),
        "==" | "=" => Ok(actual == expected),
        "!=" => Ok(actual != expected),
        other => Err(Error::new(format!(
            "unsupported mock-event-predictor rule comparison `{other}`"
        ))),
    }
}

fn mock_cache_miss_event(event: &InstrumentationEvent) -> i64 {
    let loads = event
        .signals
        .iter()
        .map(|step| *step.get("load").unwrap_or(&0))
        .sum::<i64>();
    let tag_motion = event
        .signals
        .iter()
        .map(|step| *step.get("tag_delta").unwrap_or(&0))
        .sum::<i64>();
    let pending_peak = event
        .signals
        .iter()
        .map(|step| *step.get("pending_misses").unwrap_or(&0))
        .max()
        .unwrap_or(0);
    let set_span = event
        .signals
        .iter()
        .filter_map(|step| step.get("cache_set").copied())
        .collect::<BTreeSet<_>>()
        .len() as i64;
    let working_set_log2 = *event.program.get("working_set_log2").unwrap_or(&0);
    let stride = *event.program.get("stride").unwrap_or(&0);
    let mut pressure = loads * 2 + tag_motion * 3 + pending_peak + set_span;
    if working_set_log2 >= 14 {
        pressure += 4;
    }
    if stride >= 64 {
        pressure += 3;
    }
    if pressure >= 14 {
        1
    } else {
        0
    }
}

fn mock_stall_event(event: &InstrumentationEvent) -> i64 {
    let stores = event
        .signals
        .iter()
        .map(|step| *step.get("store").unwrap_or(&0))
        .sum::<i64>();
    let loads = event
        .signals
        .iter()
        .map(|step| *step.get("load").unwrap_or(&0))
        .sum::<i64>();
    let pending_peak = event
        .signals
        .iter()
        .map(|step| *step.get("pending_misses").unwrap_or(&0))
        .max()
        .unwrap_or(0);
    let store_buffer_peak = event
        .signals
        .iter()
        .map(|step| *step.get("store_buffer_occupancy").unwrap_or(&0))
        .max()
        .unwrap_or(0);
    let mixed_memory_ops = if stores > 0 && loads > 0 { 2 } else { 0 };
    let pressure = pending_peak * 2 + store_buffer_peak + stores + mixed_memory_ops;
    if pressure >= 12 {
        1
    } else {
        0
    }
}

#[cfg(feature = "onnx-ort")]
fn run_onnx_gemm(
    manifest: &SurrogateManifest,
    transaction: &GemmTransaction,
    artifact_path: &Path,
) -> Result<(Vec<Vec<i128>>, GemmTelemetry), Error> {
    use ort::{session::Session, value::Tensor};

    let mut session = Session::builder()
        .map_err(|err| Error::new(format!("failed to create ONNX session builder: {err}")))?
        .commit_from_file(artifact_path)
        .map_err(|err| {
            Error::new(format!(
                "failed to load ONNX artifact `{}`: {err}",
                artifact_path.display()
            ))
        })?;
    validate_onnx_session_io(&session, manifest)?;

    let descriptor = vec![
        transaction.rows as f32,
        transaction.cols as f32,
        transaction.k as f32,
        manifest.domain.data_width as f32,
        manifest.domain.acc_width as f32,
        0.0,
    ];
    let a = transaction
        .a
        .iter()
        .flat_map(|row| row.iter().map(|value| *value as f32))
        .collect::<Vec<_>>();
    let w = transaction
        .w
        .iter()
        .flat_map(|row| row.iter().map(|value| *value as f32))
        .collect::<Vec<_>>();

    let outputs = session
        .run(ort::inputs! {
            "gemm_descriptor" => Tensor::from_array(([6usize], descriptor))
                .map_err(|err| Error::new(format!("failed to build gemm_descriptor tensor: {err}")))?,
            "a_tensor" => Tensor::from_array(([transaction.rows, transaction.k], a))
                .map_err(|err| Error::new(format!("failed to build a_tensor: {err}")))?,
            "w_tensor" => Tensor::from_array(([transaction.k, transaction.cols], w))
                .map_err(|err| Error::new(format!("failed to build w_tensor: {err}")))?,
        })
        .map_err(|err| Error::new(format!("ONNX GEMM inference failed: {err}")))?;

    let (_, c_values) = outputs
        .get("c_tensor")
        .ok_or_else(|| Error::new("ONNX output `c_tensor` is missing"))?
        .try_extract_tensor::<f32>()
        .map_err(|err| Error::new(format!("failed to extract c_tensor: {err}")))?;
    let (_, telemetry_values) = outputs
        .get("telemetry")
        .ok_or_else(|| Error::new("ONNX output `telemetry` is missing"))?
        .try_extract_tensor::<f32>()
        .map_err(|err| Error::new(format!("failed to extract telemetry: {err}")))?;

    if c_values.len() != transaction.rows * transaction.cols {
        return Err(Error::new(format!(
            "ONNX c_tensor has {} values, expected {}",
            c_values.len(),
            transaction.rows * transaction.cols
        )));
    }
    if telemetry_values.len() < 3 {
        return Err(Error::new(format!(
            "ONNX telemetry has {} values, expected at least 3",
            telemetry_values.len()
        )));
    }

    let c = c_values
        .chunks(transaction.cols)
        .map(|row| row.iter().map(|value| value.round() as i128).collect())
        .collect::<Vec<Vec<i128>>>();
    let telemetry = GemmTelemetry {
        latency_cycles: telemetry_values[0].round().max(0.0) as u64,
        active_cycles: telemetry_values[1].round().max(0.0) as u64,
        utilization: telemetry_values[2] as f64,
    };
    Ok((c, telemetry))
}

#[cfg(not(feature = "onnx-ort"))]
fn run_onnx_gemm(
    _manifest: &SurrogateManifest,
    _transaction: &GemmTransaction,
    _artifact_path: &Path,
) -> Result<(Vec<Vec<i128>>, GemmTelemetry), Error> {
    Err(Error::new(
        "ONNX artifact execution requires building with the `onnx-ort` feature",
    ))
}

#[cfg(feature = "onnx-ort")]
fn run_onnx_event_predictor(
    manifest: &SurrogateManifest,
    event: &InstrumentationEvent,
    artifact_path: &Path,
) -> Result<i64, Error> {
    use ort::{session::Session, value::Tensor};

    let task = manifest
        .task
        .as_ref()
        .ok_or_else(|| Error::new("event_predictor manifests require a task section"))?;
    let mut session = Session::builder()
        .map_err(|err| Error::new(format!("failed to create ONNX session builder: {err}")))?
        .commit_from_file(artifact_path)
        .map_err(|err| {
            Error::new(format!(
                "failed to load ONNX artifact `{}`: {err}",
                artifact_path.display()
            ))
        })?;
    validate_onnx_session_io(&session, manifest)?;

    let signal_features = task.signal_features.len();
    let program_features = task.program_features.len();
    let signal_window = event
        .signals
        .iter()
        .flat_map(|step| {
            task.signal_features
                .iter()
                .map(|feature| *step.get(feature).unwrap_or(&0) as f32)
        })
        .collect::<Vec<_>>();
    let program_context = task
        .program_features
        .iter()
        .map(|feature| *event.program.get(feature).unwrap_or(&0) as f32)
        .collect::<Vec<_>>();

    let outputs = session
        .run(ort::inputs! {
            "signal_window" => Tensor::from_array(([1usize, event.window_cycles, signal_features], signal_window))
                .map_err(|err| Error::new(format!("failed to build signal_window tensor: {err}")))?,
            "program_context" => Tensor::from_array(([1usize, program_features], program_context))
                .map_err(|err| Error::new(format!("failed to build program_context tensor: {err}")))?,
        })
        .map_err(|err| {
            Error::new(format!(
                "ONNX event predictor inference failed for sample {}: {err}",
                event.sample_id
            ))
        })?;

    let (_, predicted_values) = outputs
        .get("predicted_event")
        .ok_or_else(|| Error::new("ONNX output `predicted_event` is missing"))?
        .try_extract_tensor::<f32>()
        .map_err(|err| Error::new(format!("failed to extract predicted_event: {err}")))?;
    let predicted = predicted_values
        .first()
        .copied()
        .ok_or_else(|| Error::new("ONNX predicted_event output is empty"))?
        .round() as i64;
    if matches!(predicted, 0 | 1) {
        Ok(predicted)
    } else {
        Err(Error::new(format!(
            "ONNX predicted_event must round to binary 0/1, got {predicted}"
        )))
    }
}

#[cfg(not(feature = "onnx-ort"))]
fn run_onnx_event_predictor(
    _manifest: &SurrogateManifest,
    _event: &InstrumentationEvent,
    _artifact_path: &Path,
) -> Result<i64, Error> {
    Err(Error::new(
        "ONNX event predictor execution requires building with the `onnx-ort` feature",
    ))
}

#[cfg(feature = "onnx-ort")]
fn validate_onnx_session_io(
    session: &ort::session::Session,
    manifest: &SurrogateManifest,
) -> Result<(), Error> {
    let session_inputs = session
        .inputs()
        .iter()
        .map(|input| input.name())
        .collect::<BTreeSet<_>>();
    for name in &manifest.artifact.input_tensors {
        if !session_inputs.contains(name.as_str()) {
            return Err(Error::new(format!(
                "ONNX model is missing manifest input tensor `{name}`"
            )));
        }
    }
    let session_outputs = session
        .outputs()
        .iter()
        .map(|output| output.name())
        .collect::<BTreeSet<_>>();
    for name in &manifest.artifact.output_tensors {
        if !session_outputs.contains(name.as_str()) {
            return Err(Error::new(format!(
                "ONNX model is missing manifest output tensor `{name}`"
            )));
        }
    }
    Ok(())
}

fn gemm_metrics(
    actual: &[Vec<i128>],
    expected: &[Vec<i128>],
    latency_cycles: u64,
    expected_latency_cycles: Option<u64>,
) -> GemmValidationMetrics {
    let mut count = 0usize;
    let mut sum_abs_error = 0i128;
    let mut max_abs_error = 0i128;
    let mut first_divergence = None;
    for (row, actual_row) in actual.iter().enumerate() {
        for (col, actual_value) in actual_row.iter().enumerate() {
            let expected_value = expected
                .get(row)
                .and_then(|values| values.get(col))
                .copied()
                .unwrap_or_default();
            let error = (*actual_value - expected_value).abs();
            if error > max_abs_error {
                max_abs_error = error;
            }
            if error != 0 && first_divergence.is_none() {
                first_divergence = Some(GemmDivergence {
                    row,
                    col,
                    expected: expected_value,
                    actual: *actual_value,
                });
            }
            sum_abs_error += error;
            count += 1;
        }
    }
    let latency_error_cycles = expected_latency_cycles
        .map(|expected| latency_cycles.abs_diff(expected))
        .unwrap_or(0);
    GemmValidationMetrics {
        max_abs_error,
        mean_abs_error: if count == 0 {
            0.0
        } else {
            sum_abs_error as f64 / count as f64
        },
        latency_error_cycles,
        first_divergence,
    }
}

fn value_object_to_u128_map(
    object: &serde_json::Map<String, serde_json::Value>,
) -> Result<BTreeMap<String, u128>, Error> {
    let mut result = BTreeMap::new();
    for (name, value) in object {
        result.insert(name.clone(), value_to_u128(value)?);
    }
    Ok(result)
}

fn value_to_u128(value: &serde_json::Value) -> Result<u128, Error> {
    if let Some(value) = value.as_u64() {
        return Ok(value as u128);
    }
    if let Some(value) = value.as_i64() {
        if value < 0 {
            return Err(Error::new("negative trace values are not supported"));
        }
        return Ok(value as u128);
    }
    Err(Error::new("trace values must be integers"))
}

fn value_to_i64(value: &serde_json::Value) -> Result<i64, Error> {
    if let Some(value) = value.as_i64() {
        return Ok(value);
    }
    if let Some(value) = value.as_u64() {
        return i64::try_from(value)
            .map_err(|_| Error::new(format!("trace value {value} does not fit in i64")));
    }
    Err(Error::new("trace values must be integers"))
}

fn required_array<'a>(
    value: &'a serde_json::Value,
    key: &str,
) -> Result<&'a Vec<serde_json::Value>, Error> {
    value
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::new(format!("field `{key}` must be an array")))
}

fn required_object<'a>(
    value: &'a serde_json::Value,
    key: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, Error> {
    value
        .get(key)
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| Error::new(format!("field `{key}` must be an object")))
}

fn required_str<'a>(value: &'a serde_json::Value, key: &str) -> Result<&'a str, Error> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::new(format!("field `{key}` must be a string")))
}

fn required_u64(value: &serde_json::Value, key: &str) -> Result<u64, Error> {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| Error::new(format!("field `{key}` must be an unsigned integer")))
}

fn optional_i128(value: &serde_json::Value, key: &str) -> Result<Option<i128>, Error> {
    let Some(value) = value.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    if let Some(value) = value.as_i64() {
        return Ok(Some(value as i128));
    }
    if let Some(value) = value.as_u64() {
        return Ok(Some(value as i128));
    }
    Err(Error::new(format!("field `{key}` must be an integer")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn export_json() -> serde_json::Value {
        serde_json::json!({
            "schema": "rrtl-pyrtl-block-v1",
            "top_name": "Top",
            "clock_name": "clk",
            "wires": [
                {"name": "a", "kind": "input", "bitwidth": 8},
                {"name": "b", "kind": "input", "bitwidth": 8},
                {"name": "sum", "kind": "output", "bitwidth": 9}
            ],
            "memories": [],
            "nets": [
                {"index": 0, "op": "+", "op_param": null, "args": ["a", "b"], "dests": ["sum"]}
            ]
        })
    }

    fn trace_json() -> serde_json::Value {
        serde_json::json!({
            "schema": "rrtl-pyrtl-trace-v1",
            "steps": [
                {"inputs": {"a": 1, "b": 2}, "outputs": {"sum": 3}},
                {"inputs": {"a": 3, "b": 4}, "outputs": {"sum": 7}}
            ]
        })
    }

    fn event_emitter_config() -> EventEmitterConfig {
        EventEmitterConfig {
            schema: EVENT_EMITTER_CONFIG_SCHEMA.to_string(),
            input_schema: None,
            top_name: "InstrumentedCache".to_string(),
            source_hash: "trace-source".to_string(),
            target: "cache_miss".to_string(),
            window_cycles: 2,
            horizon_cycles: 1,
            signal_features: vec![
                EventFeatureMapping {
                    name: "load".to_string(),
                    source: Some(EventTraceSource::Inputs),
                    field: Some("load".to_string()),
                    constant: None,
                },
                EventFeatureMapping {
                    name: "pending_misses".to_string(),
                    source: Some(EventTraceSource::Outputs),
                    field: Some("pending".to_string()),
                    constant: None,
                },
            ],
            program_features: vec![EventFeatureMapping {
                name: "pc".to_string(),
                source: None,
                field: None,
                constant: Some(4096),
            }],
            lane: None,
            label: EventFeatureMapping {
                name: "cache_miss".to_string(),
                source: Some(EventTraceSource::Outputs),
                field: Some("miss".to_string()),
                constant: None,
            },
        }
    }

    fn event_trace_json() -> serde_json::Value {
        serde_json::json!({
            "schema": "rrtl-pyrtl-trace-v1",
            "steps": [
                {"inputs": {"load": 1}, "outputs": {"pending": 0, "miss": 0}},
                {"inputs": {"load": 1}, "outputs": {"pending": 1, "miss": 0}},
                {"inputs": {"load": 0}, "outputs": {"pending": 2, "miss": 1}},
                {"inputs": {"load": 1}, "outputs": {"pending": 3, "miss": 0}}
            ]
        })
    }

    fn instrumentation_trace_fixture() -> RrtlInstrumentationTrace {
        RrtlInstrumentationTrace {
            schema: RRTL_INSTRUMENTATION_TRACE_SCHEMA.to_string(),
            top_name: "InstrumentedCache".to_string(),
            source_hash: "instrumented-source".to_string(),
            program_id: Some("prog-a".to_string()),
            steps: vec![
                RrtlInstrumentationStep {
                    cycle: 0,
                    lane: Some(0),
                    signals: BTreeMap::from([("load".to_string(), 1), ("pending".to_string(), 0)]),
                    program: BTreeMap::from([("pc".to_string(), 4096)]),
                    labels: BTreeMap::from([("miss".to_string(), 0)]),
                },
                RrtlInstrumentationStep {
                    cycle: 1,
                    lane: Some(0),
                    signals: BTreeMap::from([("load".to_string(), 1), ("pending".to_string(), 1)]),
                    program: BTreeMap::from([("pc".to_string(), 4100)]),
                    labels: BTreeMap::from([("miss".to_string(), 0)]),
                },
                RrtlInstrumentationStep {
                    cycle: 2,
                    lane: Some(1),
                    signals: BTreeMap::from([("load".to_string(), 0), ("pending".to_string(), 2)]),
                    program: BTreeMap::from([("pc".to_string(), 4104)]),
                    labels: BTreeMap::from([("miss".to_string(), 1)]),
                },
            ],
        }
    }

    fn mock_manifest(
        artifact_path: String,
        artifact_hash: String,
        source_hash: String,
    ) -> SurrogateManifest {
        SurrogateManifest {
            schema: MANIFEST_SCHEMA.to_string(),
            surrogate_id: "gemm_mock".to_string(),
            surrogate_class: SurrogateClass::TransactionKernel,
            model_family: ModelFamily::MockGemm,
            task: None,
            source: SourceSpec {
                top_name: "Top".to_string(),
                export_schema: "rrtl-pyrtl-block-v1".to_string(),
                source_hash,
            },
            artifact: ArtifactSpec {
                format: ArtifactFormat::MockGemm,
                path: artifact_path,
                sha256: artifact_hash,
                input_tensors: vec![],
                output_tensors: vec![],
                opset: None,
            },
            domain: DomainSpec {
                rows: 2,
                cols: 2,
                k_min: 1,
                k_max: 4,
                data_width: 8,
                acc_width: 32,
            },
            validation: ValidationSpec {
                max_abs_error: 0,
                max_mean_abs_error: 0.0,
                max_latency_error_cycles: 0,
            },
            policy: PolicySpec {
                mode: PolicyMode::ApproximateWithTolerance,
                fallback: FallbackPolicy::FailClosed,
                provenance_tag: "approximate".to_string(),
            },
        }
    }

    fn valid_gemm_transaction(lane: Option<usize>, left: i64) -> GemmTransaction {
        let expected_left = i128::from(left);
        GemmTransaction {
            schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
            lane,
            rows: 2,
            cols: 2,
            k: 2,
            a: vec![vec![left, 2], vec![3, 4]],
            w: vec![vec![5, 6], vec![7, 8]],
            expected_c: Some(vec![
                vec![expected_left * 5 + 14, expected_left * 6 + 16],
                vec![43, 50],
            ]),
            expected_latency_cycles: Some(5),
        }
    }

    fn mock_gemm_result(exact: bool) -> GemmRunResult {
        GemmRunResult {
            schema: GEMM_RESULT_SCHEMA.to_string(),
            surrogate_id: "mock".to_string(),
            ok: true,
            c: vec![vec![1]],
            telemetry: GemmTelemetry {
                latency_cycles: 1,
                active_cycles: 1,
                utilization: 1.0,
            },
            metrics: None,
            provenance: Provenance {
                tag: if exact {
                    "exact".to_string()
                } else {
                    "approximate".to_string()
                },
                exact,
                surrogate_id: "mock".to_string(),
                model_family: ModelFamily::MockGemm,
                artifact_format: ArtifactFormat::MockGemm,
                artifact_hash: "hash".to_string(),
                source_hash: "source".to_string(),
                policy: PolicyMode::ApproximateWithTolerance,
            },
        }
    }

    fn event_manifest(
        artifact_path: String,
        artifact_hash: String,
        source_hash: String,
    ) -> SurrogateManifest {
        SurrogateManifest {
            schema: MANIFEST_SCHEMA.to_string(),
            surrogate_id: "cache_miss_event_predictor".to_string(),
            surrogate_class: SurrogateClass::EventPredictor,
            model_family: ModelFamily::RuleBaseline,
            task: Some(TaskSpec {
                prediction_target: "cache_miss".to_string(),
                input_window_cycles: 2,
                horizon_cycles: 1,
                signal_features: vec![
                    "cycle_delta".to_string(),
                    "load".to_string(),
                    "store".to_string(),
                    "addr_low".to_string(),
                    "cache_set".to_string(),
                    "tag_delta".to_string(),
                    "pending_misses".to_string(),
                    "store_buffer_occupancy".to_string(),
                ],
                program_features: vec![
                    "pc".to_string(),
                    "opcode_id".to_string(),
                    "stride".to_string(),
                    "working_set_log2".to_string(),
                ],
                label: Some(LabelSpec {
                    name: "cache_miss".to_string(),
                    kind: "binary".to_string(),
                    positive_value: 1,
                }),
            }),
            source: SourceSpec {
                top_name: "InstrumentedCache".to_string(),
                export_schema: EVENT_CORPUS_SCHEMA.to_string(),
                source_hash,
            },
            artifact: ArtifactSpec {
                format: ArtifactFormat::MockEventPredictor,
                path: artifact_path,
                sha256: artifact_hash,
                input_tensors: vec!["signal_window".to_string(), "program_context".to_string()],
                output_tensors: vec![
                    "event_probability".to_string(),
                    "predicted_event".to_string(),
                ],
                opset: None,
            },
            domain: DomainSpec {
                rows: 1,
                cols: 1,
                k_min: 1,
                k_max: 2,
                data_width: 64,
                acc_width: 64,
            },
            validation: ValidationSpec {
                max_abs_error: 0,
                max_mean_abs_error: 0.0,
                max_latency_error_cycles: 0,
            },
            policy: PolicySpec {
                mode: PolicyMode::TelemetryOnly,
                fallback: FallbackPolicy::FailClosed,
                provenance_tag: "instrumentation_prediction".to_string(),
            },
        }
    }

    fn onnx_event_manifest(
        artifact_path: String,
        artifact_hash: String,
        source_hash: String,
    ) -> SurrogateManifest {
        let mut manifest = event_manifest(artifact_path, artifact_hash, source_hash);
        manifest.model_family = ModelFamily::GnnTransformer;
        manifest.artifact.format = ArtifactFormat::Onnx;
        manifest.artifact.opset = Some(17);
        manifest
    }

    fn event_corpus(label: i64) -> InstrumentationEventCorpus {
        let mut program = BTreeMap::new();
        program.insert("pc".to_string(), 4096);
        program.insert("opcode_id".to_string(), 1);
        program.insert("stride".to_string(), 64);
        program.insert("working_set_log2".to_string(), 14);
        let mut first = BTreeMap::new();
        first.insert("cycle_delta".to_string(), 0);
        first.insert("load".to_string(), 1);
        first.insert("store".to_string(), 0);
        first.insert("addr_low".to_string(), 16);
        first.insert("cache_set".to_string(), 1);
        first.insert("tag_delta".to_string(), 1);
        first.insert("pending_misses".to_string(), 2);
        first.insert("store_buffer_occupancy".to_string(), 0);
        let mut second = BTreeMap::new();
        second.insert("cycle_delta".to_string(), 1);
        second.insert("load".to_string(), 1);
        second.insert("store".to_string(), 0);
        second.insert("addr_low".to_string(), 80);
        second.insert("cache_set".to_string(), 2);
        second.insert("tag_delta".to_string(), 1);
        second.insert("pending_misses".to_string(), 1);
        second.insert("store_buffer_occupancy".to_string(), 0);
        let mut labels = BTreeMap::new();
        labels.insert("cache_miss".to_string(), label);
        InstrumentationEventCorpus {
            schema: EVENT_CORPUS_SCHEMA.to_string(),
            source_hash: "event-source".to_string(),
            top_name: "InstrumentedCache".to_string(),
            events: vec![InstrumentationEvent {
                schema: EVENT_SCHEMA.to_string(),
                sample_id: 7,
                lane: None,
                target: "cache_miss".to_string(),
                window_cycles: 2,
                horizon_cycles: 1,
                program,
                signals: vec![first, second],
                label: labels,
            }],
        }
    }

    fn stall_event_manifest(
        artifact_path: String,
        artifact_hash: String,
        source_hash: String,
    ) -> SurrogateManifest {
        let mut manifest = event_manifest(artifact_path, artifact_hash, source_hash);
        manifest.surrogate_id = "stall_event_predictor".to_string();
        if let Some(task) = manifest.task.as_mut() {
            task.prediction_target = "stall_event".to_string();
            task.signal_features = vec![
                "cycle_delta".to_string(),
                "load".to_string(),
                "store".to_string(),
                "pending_misses".to_string(),
                "store_buffer_occupancy".to_string(),
            ];
            task.program_features = vec!["pc".to_string(), "opcode_id".to_string()];
            task.label = Some(LabelSpec {
                name: "stall_event".to_string(),
                kind: "binary".to_string(),
                positive_value: 1,
            });
        }
        manifest
    }

    fn stall_event_corpus(label: i64) -> InstrumentationEventCorpus {
        let mut corpus = event_corpus(1);
        corpus.events[0].target = "stall_event".to_string();
        corpus.events[0].label = BTreeMap::from([("stall_event".to_string(), label)]);
        corpus.events[0].program.remove("stride");
        corpus.events[0].program.remove("working_set_log2");
        corpus.events[0].signals[0].insert("store".to_string(), 1);
        corpus.events[0].signals[0].insert("pending_misses".to_string(), 3);
        corpus.events[0].signals[0].insert("store_buffer_occupancy".to_string(), 5);
        corpus.events[0].signals[1].insert("load".to_string(), 1);
        corpus.events[0].signals[1].insert("pending_misses".to_string(), 2);
        corpus.events[0].signals[1].insert("store_buffer_occupancy".to_string(), 4);
        corpus
    }

    fn mock_event_provenance(exact: bool) -> Provenance {
        Provenance {
            tag: if exact {
                "exact".to_string()
            } else {
                "instrumentation_prediction".to_string()
            },
            exact,
            surrogate_id: "cache_miss_event_predictor".to_string(),
            model_family: ModelFamily::RuleBaseline,
            artifact_format: ArtifactFormat::MockEventPredictor,
            artifact_hash: "hash".to_string(),
            source_hash: "event-source".to_string(),
            policy: PolicyMode::ApproximateWithTolerance,
        }
    }

    fn model_fast_gemm_report() -> GemmFastRunReport {
        let result = mock_gemm_result(false);
        GemmFastRunReport {
            schema: GEMM_FAST_RUN_SCHEMA.to_string(),
            ok: true,
            count: 2,
            total_lanes: 2,
            surrogate_replacements: 1,
            exact_fallbacks: 1,
            fail_closed: 0,
            shadow_compared: 1,
            shadow_passed: 1,
            shadow_failed: 0,
            workers: Vec::new(),
            lanes: Vec::new(),
            results: vec![
                GemmFastRunItem {
                    index: 0,
                    lane: 0,
                    decision: GemmPolicyDecision::SurrogateUsed,
                    source_result: Some(GemmRuntimeSourceResult::Surrogate),
                    result: Some(result.clone()),
                    provenance: Some(result.provenance.clone()),
                    shadow_ok: None,
                    shadow_error: None,
                    shadow_max_abs_error: None,
                    shadow_latency_error_cycles: None,
                    shadow_first_divergence: None,
                    shadow_sampled: false,
                    error: None,
                },
                GemmFastRunItem {
                    index: 1,
                    lane: 1,
                    decision: GemmPolicyDecision::ShadowCompare,
                    source_result: Some(GemmRuntimeSourceResult::Exact),
                    result: Some(mock_gemm_result(true)),
                    provenance: Some(mock_gemm_result(true).provenance),
                    shadow_ok: Some(true),
                    shadow_error: None,
                    shadow_max_abs_error: Some(0),
                    shadow_latency_error_cycles: Some(0),
                    shadow_first_divergence: None,
                    shadow_sampled: true,
                    error: None,
                },
            ],
            errors: Vec::new(),
        }
    }

    fn model_fast_event_report() -> EventFastRunReport {
        EventFastRunReport {
            schema: EVENT_FAST_RUN_SCHEMA.to_string(),
            ok: true,
            count: 3,
            total_lanes: 2,
            surrogate_replacements: 2,
            exact_fallbacks: 1,
            fail_closed: 0,
            shadow_compared: 1,
            shadow_passed: 0,
            shadow_failed: 1,
            workers: Vec::new(),
            lanes: Vec::new(),
            results: vec![
                EventFastRunItem {
                    index: 0,
                    sample_id: 10,
                    lane: 0,
                    target: "cache_miss".to_string(),
                    decision: EventPolicyDecision::SurrogateUsed,
                    source_result: Some(GemmRuntimeSourceResult::Surrogate),
                    predicted: 1,
                    expected: Some(1),
                    provenance: mock_event_provenance(false),
                    shadow_ok: None,
                    shadow_error: None,
                    shadow_sampled: false,
                    error: None,
                },
                EventFastRunItem {
                    index: 1,
                    sample_id: 11,
                    lane: 1,
                    target: "cache_miss".to_string(),
                    decision: EventPolicyDecision::SurrogateUsed,
                    source_result: Some(GemmRuntimeSourceResult::Surrogate),
                    predicted: 1,
                    expected: Some(1),
                    provenance: mock_event_provenance(false),
                    shadow_ok: None,
                    shadow_error: None,
                    shadow_sampled: true,
                    error: None,
                },
                EventFastRunItem {
                    index: 2,
                    sample_id: 12,
                    lane: 0,
                    target: "cache_miss".to_string(),
                    decision: EventPolicyDecision::ShadowCompare,
                    source_result: Some(GemmRuntimeSourceResult::Exact),
                    predicted: 1,
                    expected: Some(0),
                    provenance: mock_event_provenance(true),
                    shadow_ok: Some(false),
                    shadow_error: Some("event shadow prediction diverged".to_string()),
                    shadow_sampled: false,
                    error: None,
                },
            ],
            errors: Vec::new(),
        }
    }

    fn model_fast_golden(
        op_id: &str,
        op_kind: &str,
        expected: ModelFastGoldenExpected,
    ) -> ModelFastGolden {
        ModelFastGolden {
            schema: MODEL_FAST_GOLDEN_SCHEMA.to_string(),
            op_id: op_id.to_string(),
            op_kind: op_kind.to_string(),
            expected,
            expected_tensors: None,
            actual_tensors: None,
            max_abs_error: None,
        }
    }

    #[test]
    fn exports_graph_and_trace_dataset() {
        let dataset = export_dataset(&export_json(), &trace_json()).unwrap();
        assert_eq!(dataset.schema, DATASET_SCHEMA);
        assert_eq!(dataset.top_name, "Top");
        assert_eq!(dataset.graph.node_features.len(), 3);
        assert_eq!(dataset.graph.edge_index.len(), 2);
        assert_eq!(dataset.trace.steps.len(), 2);
        assert_eq!(dataset.graph.metadata.input_count, 2);
        assert_eq!(dataset.graph.metadata.output_count, 1);
    }

    #[test]
    fn validates_manifest_hashes() {
        let dir = std::env::temp_dir().join(format!("rrtl-surrogate-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let artifact = dir.join("artifact.mock");
        fs::write(&artifact, b"mock").unwrap();
        let hash = sha256_hex(b"mock");
        let manifest = mock_manifest(
            "artifact.mock".to_string(),
            hash.clone(),
            canonical_json_hash(&export_json()).unwrap(),
        );
        let report = validate_manifest(&manifest, &dir);
        assert!(report.ok, "{report:?}");
        assert_eq!(report.artifact_hash, hash);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn reports_hash_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-mismatch-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let manifest = mock_manifest(
            "artifact.mock".to_string(),
            "bad".to_string(),
            canonical_json_hash(&export_json()).unwrap(),
        );
        let report = validate_manifest(&manifest, &dir);
        assert!(!report.ok);
        assert!(report
            .errors
            .iter()
            .any(|err| err.contains("sha256 mismatch")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn validates_event_predictor_manifest() {
        let dir =
            std::env::temp_dir().join(format!("rrtl-surrogate-test-event-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cache_miss_rule.json"), b"rule").unwrap();
        let manifest = SurrogateManifest {
            schema: MANIFEST_SCHEMA.to_string(),
            surrogate_id: "cache_miss_event_predictor".to_string(),
            surrogate_class: SurrogateClass::EventPredictor,
            model_family: ModelFamily::RuleBaseline,
            task: Some(TaskSpec {
                prediction_target: "cache_miss".to_string(),
                input_window_cycles: 8,
                horizon_cycles: 1,
                signal_features: vec!["load".to_string(), "cache_set".to_string()],
                program_features: vec!["pc".to_string(), "opcode_id".to_string()],
                label: Some(LabelSpec {
                    name: "cache_miss".to_string(),
                    kind: "binary".to_string(),
                    positive_value: 1,
                }),
            }),
            source: SourceSpec {
                top_name: "InstrumentedCache".to_string(),
                export_schema: "rrtl-surrogate-instrumentation-corpus-v1".to_string(),
                source_hash: "source".to_string(),
            },
            artifact: ArtifactSpec {
                format: ArtifactFormat::MockEventPredictor,
                path: "cache_miss_rule.json".to_string(),
                sha256: sha256_hex(b"rule"),
                input_tensors: vec!["signal_window".to_string(), "program_context".to_string()],
                output_tensors: vec![
                    "event_probability".to_string(),
                    "predicted_event".to_string(),
                ],
                opset: None,
            },
            domain: DomainSpec {
                rows: 1,
                cols: 1,
                k_min: 1,
                k_max: 8,
                data_width: 64,
                acc_width: 64,
            },
            validation: ValidationSpec {
                max_abs_error: 0,
                max_mean_abs_error: 0.0,
                max_latency_error_cycles: 0,
            },
            policy: PolicySpec {
                mode: PolicyMode::TelemetryOnly,
                fallback: FallbackPolicy::FailClosed,
                provenance_tag: "instrumentation_prediction".to_string(),
            },
        };

        let report = validate_manifest(&manifest, &dir);

        assert!(report.ok, "{report:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn validates_onnx_event_predictor_manifest() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-onnx-manifest-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("event.onnx"), b"onnx").unwrap();
        let manifest = onnx_event_manifest(
            "event.onnx".to_string(),
            sha256_hex(b"onnx"),
            "event-source".to_string(),
        );

        let report = validate_manifest(&manifest, &dir);

        assert!(report.ok, "{report:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn validates_cache_miss_event_corpus() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-run-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cache_miss_rule.json"), b"rule").unwrap();
        let manifest = event_manifest(
            "cache_miss_rule.json".to_string(),
            sha256_hex(b"rule"),
            "event-source".to_string(),
        );

        let report = validate_event_corpus(&manifest, dir.join("manifest.json"), &event_corpus(1));

        assert!(report.ok, "{report:?}");
        assert_eq!(report.schema, EVENT_VALIDATION_SCHEMA);
        assert_eq!(report.corpus.samples, 1);
        assert_eq!(report.metrics.accuracy, 1.0);
        assert!(report.first_mismatch.is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn validates_stall_event_corpus_with_generic_target() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-stall-event-run-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("stall_event_rule.json"), b"rule").unwrap();
        let manifest = stall_event_manifest(
            "stall_event_rule.json".to_string(),
            sha256_hex(b"rule"),
            "event-source".to_string(),
        );

        let report =
            validate_event_corpus(&manifest, dir.join("manifest.json"), &stall_event_corpus(1));

        assert!(report.ok, "{report:?}");
        assert_eq!(report.corpus.target, "stall_event");
        assert_eq!(report.metrics.accuracy, 1.0);
        assert!(report.first_mismatch.is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn event_validation_reports_generic_target_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-stall-event-mismatch-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("stall_event_rule.json"), b"rule").unwrap();
        let manifest = stall_event_manifest(
            "stall_event_rule.json".to_string(),
            sha256_hex(b"rule"),
            "event-source".to_string(),
        );

        let report = validate_event_corpus(&manifest, dir.join("manifest.json"), &event_corpus(1));

        assert!(!report.ok);
        assert!(report.errors.iter().any(|err| {
            err.contains("target `cache_miss` does not match manifest target `stall_event`")
        }));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn validates_custom_event_corpus_with_json_mock_rule() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-custom-event-rule-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let artifact = serde_json::json!({
            "schema": "rrtl-surrogate-rule-artifact-v1",
            "prediction_target": "custom_event",
            "mock_rule": {
                "kind": "linear_threshold",
                "threshold": 2,
                "terms": [
                    {
                        "source": "signal",
                        "feature": "load",
                        "reduction": "sum",
                        "weight": 1
                    }
                ]
            }
        })
        .to_string();
        fs::write(dir.join("custom_event_rule.json"), artifact.as_bytes()).unwrap();
        let mut manifest = event_manifest(
            "custom_event_rule.json".to_string(),
            sha256_hex(artifact.as_bytes()),
            "event-source".to_string(),
        );
        manifest.surrogate_id = "custom_event_predictor".to_string();
        if let Some(task) = manifest.task.as_mut() {
            task.prediction_target = "custom_event".to_string();
            task.signal_features = vec!["load".to_string()];
            task.program_features = vec!["pc".to_string()];
            task.label = Some(LabelSpec {
                name: "custom_event".to_string(),
                kind: "binary".to_string(),
                positive_value: 1,
            });
        }
        let mut corpus = event_corpus(1);
        corpus.events[0].target = "custom_event".to_string();
        corpus.events[0].label = BTreeMap::from([("custom_event".to_string(), 1)]);
        corpus.events[0].program = BTreeMap::from([("pc".to_string(), 4096)]);

        let report = validate_event_corpus(&manifest, dir.join("manifest.json"), &corpus);

        assert!(report.ok, "{report:?}");
        assert_eq!(report.corpus.target, "custom_event");
        assert_eq!(report.metrics.accuracy, 1.0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(not(feature = "onnx-ort"))]
    #[test]
    fn event_validation_reports_onnx_feature_required() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-onnx-required-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("event.onnx"), b"onnx").unwrap();
        let manifest = onnx_event_manifest(
            "event.onnx".to_string(),
            sha256_hex(b"onnx"),
            "event-source".to_string(),
        );

        let report = validate_event_corpus(&manifest, dir.join("manifest.json"), &event_corpus(1));

        assert!(!report.ok);
        assert!(report.manifest.ok, "{report:?}");
        assert!(report.errors.iter().any(|err| {
            err.contains(
                "ONNX event predictor execution requires building with the `onnx-ort` feature",
            )
        }));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn inspects_event_corpus_features_and_labels() {
        let report = inspect_event_corpus(&event_corpus(1));

        assert!(report.ok, "{report:?}");
        assert_eq!(report.schema, EVENT_INSPECTION_SCHEMA);
        assert_eq!(report.corpus.samples, 1);
        assert_eq!(report.corpus.target, "cache_miss");
        assert!(report.targets.contains(&"cache_miss".to_string()));
        assert!(report.signal_features.contains(&"tag_delta".to_string()));
        assert!(report
            .program_features
            .contains(&"working_set_log2".to_string()));
        assert_eq!(report.positive_labels.get("cache_miss"), Some(&1));
    }

    #[test]
    fn emits_event_corpus_from_trace_windows() {
        let corpus = emit_event_corpus(&event_trace_json(), &event_emitter_config()).unwrap();

        assert_eq!(corpus.schema, EVENT_CORPUS_SCHEMA);
        assert_eq!(corpus.top_name, "InstrumentedCache");
        assert_eq!(corpus.source_hash, "trace-source");
        assert_eq!(corpus.events.len(), 2);
        assert_eq!(corpus.events[0].sample_id, 0);
        assert_eq!(corpus.events[0].signals.len(), 2);
        assert_eq!(corpus.events[0].signals[0]["load"], 1);
        assert_eq!(corpus.events[0].signals[1]["pending_misses"], 1);
        assert_eq!(corpus.events[0].program["pc"], 4096);
        assert_eq!(corpus.events[0].label["cache_miss"], 1);
        assert_eq!(corpus.events[0].lane, None);
    }

    #[test]
    fn emits_event_corpus_with_lane_mapping() {
        let mut config = event_emitter_config();
        config.lane = Some(EventFeatureMapping {
            name: "lane".to_string(),
            source: Some(EventTraceSource::Inputs),
            field: Some("lane".to_string()),
            constant: None,
        });
        let trace = serde_json::json!({
            "schema": "rrtl-pyrtl-trace-v1",
            "steps": [
                {"inputs": {"load": 1, "lane": 0}, "outputs": {"pending": 0, "miss": 0}},
                {"inputs": {"load": 1, "lane": 0}, "outputs": {"pending": 1, "miss": 0}},
                {"inputs": {"load": 0, "lane": 1}, "outputs": {"pending": 2, "miss": 1}},
                {"inputs": {"load": 1, "lane": 1}, "outputs": {"pending": 3, "miss": 0}},
                {"inputs": {"load": 1, "lane": 1}, "outputs": {"pending": 1, "miss": 0}}
            ]
        });

        let corpus = emit_event_corpus(&trace, &config).unwrap();

        assert_eq!(corpus.events.len(), 3);
        assert_eq!(corpus.events[0].lane, Some(0));
        assert_eq!(corpus.events[1].lane, Some(0));
        assert_eq!(corpus.events[2].lane, Some(1));
    }

    #[test]
    fn emits_event_corpus_from_rrtl_instrumentation_trace() {
        let mut config = event_emitter_config();
        config.input_schema = Some(RRTL_INSTRUMENTATION_TRACE_SCHEMA.to_string());
        config.lane = Some(EventFeatureMapping {
            name: "lane".to_string(),
            source: Some(EventTraceSource::Inputs),
            field: Some("lane".to_string()),
            constant: None,
        });
        config.program_features = vec![EventFeatureMapping {
            name: "pc".to_string(),
            source: Some(EventTraceSource::Inputs),
            field: Some("pc".to_string()),
            constant: None,
        }];
        let trace = RrtlInstrumentationTrace {
            schema: RRTL_INSTRUMENTATION_TRACE_SCHEMA.to_string(),
            top_name: "InstrumentedCache".to_string(),
            source_hash: "instrumented-source".to_string(),
            program_id: Some("prog-a".to_string()),
            steps: vec![
                RrtlInstrumentationStep {
                    cycle: 0,
                    lane: Some(0),
                    signals: BTreeMap::from([("load".to_string(), 1), ("pending".to_string(), 0)]),
                    program: BTreeMap::from([("pc".to_string(), 4096)]),
                    labels: BTreeMap::from([("miss".to_string(), 0)]),
                },
                RrtlInstrumentationStep {
                    cycle: 1,
                    lane: Some(0),
                    signals: BTreeMap::from([("load".to_string(), 1), ("pending".to_string(), 1)]),
                    program: BTreeMap::from([("pc".to_string(), 4100)]),
                    labels: BTreeMap::from([("miss".to_string(), 0)]),
                },
                RrtlInstrumentationStep {
                    cycle: 2,
                    lane: Some(1),
                    signals: BTreeMap::from([("load".to_string(), 0), ("pending".to_string(), 2)]),
                    program: BTreeMap::from([("pc".to_string(), 4104)]),
                    labels: BTreeMap::from([("miss".to_string(), 1)]),
                },
            ],
        };

        let corpus = emit_instrumented_event_corpus(&trace, &config).unwrap();

        assert_eq!(corpus.schema, EVENT_CORPUS_SCHEMA);
        assert_eq!(corpus.events.len(), 1);
        assert_eq!(corpus.events[0].sample_id, 0);
        assert_eq!(corpus.events[0].lane, Some(0));
        assert_eq!(corpus.events[0].target, "cache_miss");
        assert_eq!(corpus.events[0].signals[0]["load"], 1);
        assert_eq!(corpus.events[0].signals[1]["pending_misses"], 1);
        assert_eq!(corpus.events[0].program["pc"], 4096);
        assert_eq!(corpus.events[0].label["cache_miss"], 1);
    }

    #[test]
    fn instrumentation_trace_inspection_reports_config_compatibility() {
        let mut config = event_emitter_config();
        config.input_schema = Some(RRTL_INSTRUMENTATION_TRACE_SCHEMA.to_string());
        config.lane = Some(EventFeatureMapping {
            name: "lane".to_string(),
            source: Some(EventTraceSource::Inputs),
            field: Some("lane".to_string()),
            constant: None,
        });
        config.program_features = vec![EventFeatureMapping {
            name: "pc".to_string(),
            source: Some(EventTraceSource::Inputs),
            field: Some("pc".to_string()),
            constant: None,
        }];
        let report =
            inspect_rrtl_instrumentation_trace(&instrumentation_trace_fixture(), Some(&config));

        assert!(report.ok, "{report:?}");
        assert_eq!(report.schema, RRTL_INSTRUMENTATION_TRACE_INSPECTION_SCHEMA);
        assert_eq!(report.steps, 3);
        assert_eq!(report.cycle_min, Some(0));
        assert_eq!(report.cycle_max, Some(2));
        assert!(report.cycle_monotonic);
        assert_eq!(report.lanes, vec![0, 1]);
        assert_eq!(report.steps_with_lane, 3);
        assert!(report.signal_fields.contains(&"load".to_string()));
        assert!(report.program_fields.contains(&"pc".to_string()));
        assert!(report.label_fields.contains(&"miss".to_string()));
        let compatibility = report.compatibility.unwrap();
        assert_eq!(compatibility.target, "cache_miss");
        assert_eq!(compatibility.emittable_samples, 1);
        assert!(compatibility.missing_fields.is_empty());
    }

    #[test]
    fn instrumentation_trace_inspection_reports_missing_mapped_label() {
        let mut trace = instrumentation_trace_fixture();
        trace.steps[2].labels.clear();
        let report = inspect_rrtl_instrumentation_trace(&trace, Some(&event_emitter_config()));

        assert!(!report.ok);
        let compatibility = report.compatibility.unwrap();
        assert_eq!(compatibility.missing_fields.len(), 1);
        assert_eq!(compatibility.missing_fields[0].field, "miss");
        assert!(report
            .errors
            .iter()
            .any(|err| err.contains("missing outputs field `miss`")));
    }

    #[test]
    fn instrumentation_trace_inspection_reports_non_monotonic_cycles() {
        let mut trace = instrumentation_trace_fixture();
        trace.steps[2].cycle = 0;
        let report = inspect_rrtl_instrumentation_trace(&trace, None);

        assert!(report.ok, "{report:?}");
        assert!(!report.cycle_monotonic);
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("cycles are not monotonic")));
    }

    #[test]
    fn emit_event_corpus_reports_missing_mapped_field() {
        let mut trace = event_trace_json();
        trace["steps"][0]["inputs"]
            .as_object_mut()
            .unwrap()
            .remove("load");

        let err = emit_event_corpus(&trace, &event_emitter_config())
            .unwrap_err()
            .to_string();

        assert!(err.contains("trace step missing inputs field `load`"));
    }

    #[test]
    fn emit_event_corpus_skips_incomplete_tail_windows() {
        let trace = serde_json::json!({
            "schema": "rrtl-pyrtl-trace-v1",
            "steps": [
                {"inputs": {"load": 1}, "outputs": {"pending": 0, "miss": 0}},
                {"inputs": {"load": 1}, "outputs": {"pending": 1, "miss": 0}}
            ]
        });

        let corpus = emit_event_corpus(&trace, &event_emitter_config()).unwrap();

        assert!(corpus.events.is_empty());
    }

    #[test]
    fn inspect_event_corpus_reports_mixed_window() {
        let mut corpus = event_corpus(1);
        corpus.events.push(corpus.events[0].clone());
        corpus.events[1].sample_id = 8;
        corpus.events[1].window_cycles = 3;

        let report = inspect_event_corpus(&corpus);

        assert!(!report.ok);
        assert!(report
            .errors
            .iter()
            .any(|err| err.contains("window 3 does not match first event window 2")));
    }

    #[test]
    fn event_validation_reports_first_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-mismatch-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cache_miss_rule.json"), b"rule").unwrap();
        let manifest = event_manifest(
            "cache_miss_rule.json".to_string(),
            sha256_hex(b"rule"),
            "event-source".to_string(),
        );

        let report = validate_event_corpus(&manifest, dir.join("manifest.json"), &event_corpus(0));

        assert!(!report.ok);
        assert_eq!(report.metrics.accuracy, 0.0);
        assert_eq!(report.metrics.false_positive, 1);
        let mismatch = report.first_mismatch.unwrap();
        assert_eq!(mismatch.sample_id, 7);
        assert_eq!(mismatch.expected, 0);
        assert_eq!(mismatch.predicted, 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn shadow_event_corpus_reports_per_sample_results() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-shadow-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cache_miss_rule.json"), b"rule").unwrap();
        let manifest = event_manifest(
            "cache_miss_rule.json".to_string(),
            sha256_hex(b"rule"),
            "event-source".to_string(),
        );

        let report = shadow_event_corpus(&manifest, dir.join("manifest.json"), &event_corpus(1));

        assert!(report.ok, "{report:?}");
        assert_eq!(report.schema, EVENT_SHADOW_SCHEMA);
        assert_eq!(report.metrics.accuracy, 1.0);
        assert_eq!(report.total_lanes, 1);
        assert_eq!(report.lanes.len(), 1);
        assert_eq!(report.lanes[0].lane, 0);
        assert_eq!(report.lanes[0].samples, 1);
        assert_eq!(report.lanes[0].metrics.accuracy, 1.0);
        assert_eq!(report.results.len(), 1);
        assert_eq!(report.results[0].sample_id, 7);
        assert_eq!(report.results[0].lane, 0);
        assert_eq!(report.results[0].expected, 1);
        assert_eq!(report.results[0].predicted, 1);
        assert!(report.results[0].ok);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn policy_event_corpus_uses_surrogate_when_prediction_matches() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-policy-use-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cache_miss_rule.json"), b"rule").unwrap();
        let mut manifest = event_manifest(
            "cache_miss_rule.json".to_string(),
            sha256_hex(b"rule"),
            "event-source".to_string(),
        );
        manifest.policy.mode = PolicyMode::ApproximateWithTolerance;

        let report = policy_event_corpus(&manifest, dir.join("manifest.json"), &event_corpus(1));

        assert!(report.ok, "{report:?}");
        assert_eq!(report.schema, EVENT_POLICY_REPORT_SCHEMA);
        assert_eq!(report.used_surrogate, 1);
        assert_eq!(report.exact_fallbacks, 0);
        assert_eq!(
            report.results[0].decision,
            EventPolicyDecision::SurrogateUsed
        );
        assert!(!report.results[0].provenance.exact);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn policy_event_corpus_fail_closed_blocks_runtime_plan() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-policy-fail-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cache_miss_rule.json"), b"rule").unwrap();
        let mut manifest = event_manifest(
            "cache_miss_rule.json".to_string(),
            sha256_hex(b"rule"),
            "event-source".to_string(),
        );
        manifest.policy.mode = PolicyMode::ApproximateWithTolerance;
        manifest.policy.fallback = FallbackPolicy::FailClosed;
        let report = policy_event_corpus(&manifest, dir.join("manifest.json"), &event_corpus(0));

        let plan = plan_runtime_events(&report, 1);

        assert!(!report.ok);
        assert!(!plan.ok);
        assert!(plan.errors.iter().any(|err| err.contains("fail-closed")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(not(feature = "onnx-ort"))]
    #[test]
    fn policy_event_corpus_reports_onnx_feature_required() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-policy-onnx-required-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("event.onnx"), b"onnx").unwrap();
        let mut manifest = onnx_event_manifest(
            "event.onnx".to_string(),
            sha256_hex(b"onnx"),
            "event-source".to_string(),
        );
        manifest.policy.mode = PolicyMode::ApproximateWithTolerance;

        let report = policy_event_corpus(&manifest, dir.join("manifest.json"), &event_corpus(1));

        assert!(!report.ok);
        assert_eq!(report.count, 0);
        assert!(report.errors.iter().any(|err| {
            err.contains(
                "ONNX event predictor execution requires building with the `onnx-ort` feature",
            )
        }));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn runtime_plan_events_maps_items_to_worker_specs() {
        let policy = EventPolicyReport {
            schema: EVENT_POLICY_REPORT_SCHEMA.to_string(),
            ok: true,
            count: 2,
            used_surrogate: 2,
            exact_fallbacks: 0,
            fail_closed: 0,
            results: vec![
                EventPolicyItemResult {
                    index: 0,
                    sample_id: 10,
                    lane: 0,
                    target: "cache_miss".to_string(),
                    decision: EventPolicyDecision::SurrogateUsed,
                    ok: true,
                    predicted: 1,
                    expected: Some(1),
                    provenance: mock_event_provenance(false),
                    error: None,
                },
                EventPolicyItemResult {
                    index: 1,
                    sample_id: 11,
                    lane: 2,
                    target: "cache_miss".to_string(),
                    decision: EventPolicyDecision::SurrogateUsed,
                    ok: true,
                    predicted: 1,
                    expected: Some(1),
                    provenance: mock_event_provenance(false),
                    error: None,
                },
            ],
            errors: Vec::new(),
        };
        let workers = vec![
            GemmRuntimeWorkerSpec {
                worker_id: "cpu-a".to_string(),
                start_lane: 0,
                lanes: 1,
            },
            GemmRuntimeWorkerSpec {
                worker_id: "cpu-b".to_string(),
                start_lane: 1,
                lanes: 2,
            },
        ];

        let plan = plan_runtime_events_for_workers(&policy, &workers);

        assert!(plan.ok, "{plan:?}");
        assert_eq!(plan.schema, EVENT_RUNTIME_PLAN_SCHEMA);
        assert_eq!(plan.total_lanes, 3);
        assert_eq!(plan.items[0].worker_id, "cpu-a");
        assert_eq!(plan.items[1].worker_id, "cpu-b");
        assert_eq!(plan.workers[0].assigned_items, 1);
        assert_eq!(plan.workers[1].assigned_items, 1);
    }

    #[test]
    fn runtime_plan_events_preserves_shadow_metadata() {
        let policy = EventPolicyReport {
            schema: EVENT_POLICY_REPORT_SCHEMA.to_string(),
            ok: true,
            count: 1,
            used_surrogate: 0,
            exact_fallbacks: 1,
            fail_closed: 0,
            results: vec![EventPolicyItemResult {
                index: 0,
                sample_id: 10,
                lane: 0,
                target: "cache_miss".to_string(),
                decision: EventPolicyDecision::ShadowCompare,
                ok: true,
                predicted: 1,
                expected: Some(0),
                provenance: mock_event_provenance(true),
                error: None,
            }],
            errors: Vec::new(),
        };

        let plan = plan_runtime_events(&policy, 1);

        assert!(plan.ok, "{plan:?}");
        assert_eq!(plan.items[0].source_result, GemmRuntimeSourceResult::Exact);
        assert_eq!(plan.items[0].predicted, Some(1));
        assert_eq!(plan.items[0].expected, Some(0));
        assert_eq!(plan.items[0].shadow_ok, Some(false));
        assert!(plan.items[0].shadow_error.is_some());
    }

    #[test]
    fn fast_events_reports_worker_and_lane_summaries() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-fast-summaries-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cache_miss_rule.json"), b"rule").unwrap();
        let mut manifest = event_manifest(
            "cache_miss_rule.json".to_string(),
            sha256_hex(b"rule"),
            "event-source".to_string(),
        );
        manifest.policy.mode = PolicyMode::ApproximateWithTolerance;
        let mut corpus = event_corpus(1);
        corpus.events[0].lane = Some(0);
        let mut second = corpus.events[0].clone();
        second.sample_id = 8;
        second.lane = Some(1);
        let mut third = corpus.events[0].clone();
        third.sample_id = 9;
        third.lane = Some(2);
        corpus.events.push(second);
        corpus.events.push(third);
        let policy = policy_event_corpus(&manifest, dir.join("manifest.json"), &corpus);
        let workers = vec![
            GemmRuntimeWorkerSpec {
                worker_id: "cpu-a".to_string(),
                start_lane: 0,
                lanes: 2,
            },
            GemmRuntimeWorkerSpec {
                worker_id: "cpu-b".to_string(),
                start_lane: 2,
                lanes: 1,
            },
        ];
        let plan = plan_runtime_events_for_workers(&policy, &workers);

        let report = run_fast_event_corpus(&manifest, dir.join("manifest.json"), &corpus, &plan);

        assert!(report.ok, "{report:?}");
        assert_eq!(report.schema, EVENT_FAST_RUN_SCHEMA);
        assert_eq!(report.surrogate_replacements, 3);
        assert_eq!(report.exact_fallbacks, 0);
        assert_eq!(report.workers.len(), 2);
        assert_eq!(report.workers[0].worker_id, "cpu-a");
        assert_eq!(report.workers[0].assigned_items, 2);
        assert_eq!(report.workers[0].surrogate_replacements, 2);
        assert_eq!(report.workers[1].worker_id, "cpu-b");
        assert_eq!(report.workers[1].assigned_items, 1);
        assert_eq!(report.workers[1].surrogate_replacements, 1);
        assert_eq!(report.lanes.len(), 3);
        assert_eq!(report.lanes[0].lane, 0);
        assert_eq!(report.lanes[1].lane, 1);
        assert_eq!(report.lanes[2].lane, 2);
        assert_eq!(report.results[0].sample_id, 7);
        assert_eq!(report.results[1].sample_id, 8);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fast_events_reports_fail_closed_without_runtime_item() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-fast-fail-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cache_miss_rule.json"), b"rule").unwrap();
        let mut manifest = event_manifest(
            "cache_miss_rule.json".to_string(),
            sha256_hex(b"rule"),
            "event-source".to_string(),
        );
        manifest.policy.mode = PolicyMode::ApproximateWithTolerance;
        manifest.policy.fallback = FallbackPolicy::FailClosed;
        let plan = EventRuntimePlan {
            schema: EVENT_RUNTIME_PLAN_SCHEMA.to_string(),
            ok: true,
            total_lanes: 1,
            workers: vec![EventRuntimeWorkerSummary {
                worker_id: "worker0".to_string(),
                start_lane: 0,
                lanes: 1,
                assigned_items: 0,
                used_surrogate: 0,
                exact_fallbacks: 0,
            }],
            items: Vec::new(),
            errors: Vec::new(),
        };

        let report = run_fast_event_corpus(
            &manifest,
            dir.join("manifest.json"),
            &event_corpus(0),
            &plan,
        );

        assert!(!report.ok);
        assert_eq!(report.fail_closed, 1);
        assert_eq!(report.results[0].source_result, None);
        assert_eq!(report.workers[0].assigned_items, 1);
        assert_eq!(report.workers[0].fail_closed, 1);
        assert_eq!(report.lanes[0].fail_closed, 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fast_events_rejects_runtime_plan_extra_and_missing_items() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-fast-plan-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cache_miss_rule.json"), b"rule").unwrap();
        let mut manifest = event_manifest(
            "cache_miss_rule.json".to_string(),
            sha256_hex(b"rule"),
            "event-source".to_string(),
        );
        manifest.policy.mode = PolicyMode::ApproximateWithTolerance;
        let corpus = event_corpus(1);
        let policy = policy_event_corpus(&manifest, dir.join("manifest.json"), &corpus);
        let mut extra_plan = plan_runtime_events(&policy, 1);
        let mut extra_item = extra_plan.items[0].clone();
        extra_item.index = 99;
        extra_plan.items.push(extra_item);
        let mut missing_plan = plan_runtime_events(&policy, 1);
        missing_plan.items.clear();

        let extra_report =
            run_fast_event_corpus(&manifest, dir.join("manifest.json"), &corpus, &extra_plan);
        let missing_report =
            run_fast_event_corpus(&manifest, dir.join("manifest.json"), &corpus, &missing_plan);

        assert!(!extra_report.ok);
        assert!(extra_report
            .errors
            .iter()
            .any(|err| err.contains("extra item 99")));
        assert!(!missing_report.ok);
        assert!(missing_report
            .errors
            .iter()
            .any(|err| err.contains("missing non-fail-closed policy item 0")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fast_events_preserves_shadow_and_marks_samples() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-fast-shadow-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cache_miss_rule.json"), b"rule").unwrap();
        let mut manifest = event_manifest(
            "cache_miss_rule.json".to_string(),
            sha256_hex(b"rule"),
            "event-source".to_string(),
        );
        manifest.policy.mode = PolicyMode::ShadowCompare;
        let mut corpus = event_corpus(1);
        let mut second = corpus.events[0].clone();
        second.sample_id = 8;
        corpus.events.push(second);
        let policy = policy_event_corpus(&manifest, dir.join("manifest.json"), &corpus);
        let plan = plan_runtime_events(&policy, 1);

        let report = run_fast_event_corpus_with_options(
            &manifest,
            dir.join("manifest.json"),
            &corpus,
            &plan,
            EventFastRunOptions {
                shadow_sample_stride: Some(2),
                shadow_sample_offset: 1,
            },
        );

        assert!(report.ok, "{report:?}");
        assert_eq!(report.exact_fallbacks, 2);
        assert_eq!(report.shadow_compared, 2);
        assert_eq!(report.shadow_passed, 2);
        assert_eq!(report.results[0].shadow_ok, Some(true));
        assert_eq!(report.results[1].shadow_ok, Some(true));
        assert_eq!(
            report
                .results
                .iter()
                .map(|item| item.shadow_sampled)
                .collect::<Vec<_>>(),
            vec![false, true]
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_fast_plan_aggregates_gemm_and_event_reports() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-model-fast-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("gemm_fast.json"),
            serde_json::to_string_pretty(&model_fast_gemm_report()).unwrap(),
        )
        .unwrap();
        fs::write(
            dir.join("event_fast.json"),
            serde_json::to_string_pretty(&model_fast_event_report()).unwrap(),
        )
        .unwrap();
        let plan = ModelFastPlan {
            schema: MODEL_FAST_PLAN_SCHEMA.to_string(),
            ops: vec![
                ModelFastOp {
                    op_id: "op0".to_string(),
                    op_kind: "gemm".to_string(),
                    name: "gemm tile".to_string(),
                    fast_report_path: "gemm_fast.json".to_string(),
                    golden_path: None,
                    exact_ns: None,
                    fast_ns: None,
                    source_hash: Some("gemm-source".to_string()),
                    description: None,
                },
                ModelFastOp {
                    op_id: "op1".to_string(),
                    op_kind: "event".to_string(),
                    name: "cache miss".to_string(),
                    fast_report_path: "event_fast.json".to_string(),
                    golden_path: None,
                    exact_ns: None,
                    fast_ns: None,
                    source_hash: Some("event-source".to_string()),
                    description: Some("instrumentation".to_string()),
                },
            ],
            thresholds: None,
        };

        let report = run_model_fast_plan(&plan, &dir);

        assert!(report.ok, "{report:?}");
        assert_eq!(report.schema, MODEL_FAST_REPORT_SCHEMA);
        assert_eq!(report.op_count, 2);
        assert_eq!(report.ops[0].op_id, "op0");
        assert_eq!(report.ops[1].op_id, "op1");
        assert_eq!(report.totals.items, 5);
        assert_eq!(report.totals.surrogate_replacements, 3);
        assert_eq!(report.totals.exact_fallbacks, 2);
        assert_eq!(report.totals.shadow_compared, 2);
        assert_eq!(report.totals.shadow_passed, 1);
        assert_eq!(report.totals.shadow_failed, 1);
        assert_eq!(report.totals.shadow_sampled, 2);
        assert_eq!(report.coverage.op_coverage, 1.0);
        assert_eq!(report.coverage.item_coverage, 0.6);
        assert_eq!(report.coverage.fallback_ratio, 0.4);
        assert_eq!(report.coverage.shadow_sample_ratio, 0.4);
        assert!(report.coverage.accepted);
        assert!(report.coverage.reject_reasons.is_empty());
        assert_eq!(report.timing.timed_ops, 0);
        assert_eq!(report.timing.exact_ns, 0);
        assert_eq!(report.timing.fast_ns, 0);
        assert_eq!(report.timing.speedup, None);
        assert_eq!(report.timing.missing_timing_ops, vec!["op0", "op1"]);
        assert_eq!(report.ops[0].provenance.len(), 2);
        assert_eq!(report.ops[1].provenance.len(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_fast_plan_rejects_threshold_violations() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-model-fast-thresholds-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("gemm_fast.json"),
            serde_json::to_string_pretty(&model_fast_gemm_report()).unwrap(),
        )
        .unwrap();
        fs::write(
            dir.join("event_fast.json"),
            serde_json::to_string_pretty(&model_fast_event_report()).unwrap(),
        )
        .unwrap();
        let plan = ModelFastPlan {
            schema: MODEL_FAST_PLAN_SCHEMA.to_string(),
            ops: vec![
                ModelFastOp {
                    op_id: "op0".to_string(),
                    op_kind: "gemm".to_string(),
                    name: "gemm tile".to_string(),
                    fast_report_path: "gemm_fast.json".to_string(),
                    golden_path: None,
                    exact_ns: None,
                    fast_ns: None,
                    source_hash: None,
                    description: None,
                },
                ModelFastOp {
                    op_id: "op1".to_string(),
                    op_kind: "event".to_string(),
                    name: "cache miss".to_string(),
                    fast_report_path: "event_fast.json".to_string(),
                    golden_path: None,
                    exact_ns: None,
                    fast_ns: None,
                    source_hash: None,
                    description: None,
                },
            ],
            thresholds: Some(ModelFastThresholds {
                min_op_coverage: Some(1.0),
                min_item_coverage: Some(0.7),
                max_fallback_ratio: Some(0.3),
                min_shadow_sample_ratio: Some(0.5),
            }),
        };

        let report = run_model_fast_plan(&plan, &dir);

        assert!(!report.ok);
        assert!(!report.coverage.accepted);
        assert_eq!(report.coverage.op_coverage, 1.0);
        assert_eq!(report.coverage.item_coverage, 0.6);
        assert_eq!(report.coverage.fallback_ratio, 0.4);
        assert_eq!(report.coverage.shadow_sample_ratio, 0.4);
        assert_eq!(report.coverage.reject_reasons.len(), 3);
        assert!(report
            .coverage
            .reject_reasons
            .iter()
            .any(|err| err.contains("item_coverage")));
        assert!(report
            .coverage
            .reject_reasons
            .iter()
            .any(|err| err.contains("fallback_ratio")));
        assert!(report
            .coverage
            .reject_reasons
            .iter()
            .any(|err| err.contains("shadow_sample_ratio")));
        assert_eq!(report.errors, report.coverage.reject_reasons);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_fast_plan_reports_speedup_metrics() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-model-fast-timing-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("gemm_fast.json"),
            serde_json::to_string_pretty(&model_fast_gemm_report()).unwrap(),
        )
        .unwrap();
        fs::write(
            dir.join("event_fast.json"),
            serde_json::to_string_pretty(&model_fast_event_report()).unwrap(),
        )
        .unwrap();
        let plan = ModelFastPlan {
            schema: MODEL_FAST_PLAN_SCHEMA.to_string(),
            ops: vec![
                ModelFastOp {
                    op_id: "op0".to_string(),
                    op_kind: "gemm".to_string(),
                    name: "gemm tile".to_string(),
                    fast_report_path: "gemm_fast.json".to_string(),
                    golden_path: None,
                    exact_ns: Some(1000),
                    fast_ns: Some(250),
                    source_hash: None,
                    description: None,
                },
                ModelFastOp {
                    op_id: "op1".to_string(),
                    op_kind: "event".to_string(),
                    name: "cache miss".to_string(),
                    fast_report_path: "event_fast.json".to_string(),
                    golden_path: None,
                    exact_ns: Some(600),
                    fast_ns: Some(300),
                    source_hash: None,
                    description: None,
                },
            ],
            thresholds: None,
        };

        let report = run_model_fast_plan(&plan, &dir);

        assert!(report.ok, "{report:?}");
        let op0_timing = report.ops[0].timing.as_ref().unwrap();
        assert_eq!(op0_timing.exact_ns, 1000);
        assert_eq!(op0_timing.fast_ns, 250);
        assert_eq!(op0_timing.speedup, 4.0);
        assert_eq!(report.timing.timed_ops, 2);
        assert_eq!(report.timing.exact_ns, 1600);
        assert_eq!(report.timing.fast_ns, 550);
        assert_eq!(report.timing.speedup, Some(1600.0 / 550.0));
        assert!(report.timing.missing_timing_ops.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_fast_plan_reports_partial_missing_timing() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-model-fast-partial-timing-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("gemm_fast.json"),
            serde_json::to_string_pretty(&model_fast_gemm_report()).unwrap(),
        )
        .unwrap();
        fs::write(
            dir.join("event_fast.json"),
            serde_json::to_string_pretty(&model_fast_event_report()).unwrap(),
        )
        .unwrap();
        let plan = ModelFastPlan {
            schema: MODEL_FAST_PLAN_SCHEMA.to_string(),
            ops: vec![
                ModelFastOp {
                    op_id: "op0".to_string(),
                    op_kind: "gemm".to_string(),
                    name: "gemm tile".to_string(),
                    fast_report_path: "gemm_fast.json".to_string(),
                    golden_path: None,
                    exact_ns: Some(1000),
                    fast_ns: Some(500),
                    source_hash: None,
                    description: None,
                },
                ModelFastOp {
                    op_id: "op1".to_string(),
                    op_kind: "event".to_string(),
                    name: "cache miss".to_string(),
                    fast_report_path: "event_fast.json".to_string(),
                    golden_path: None,
                    exact_ns: None,
                    fast_ns: None,
                    source_hash: None,
                    description: None,
                },
            ],
            thresholds: None,
        };

        let report = run_model_fast_plan(&plan, &dir);

        assert!(report.ok, "{report:?}");
        assert_eq!(report.timing.timed_ops, 1);
        assert_eq!(report.timing.exact_ns, 1000);
        assert_eq!(report.timing.fast_ns, 500);
        assert_eq!(report.timing.speedup, Some(2.0));
        assert_eq!(report.timing.missing_timing_ops, vec!["op1"]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_fast_plan_rejects_zero_timing() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-model-fast-zero-timing-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("gemm_fast.json"),
            serde_json::to_string_pretty(&model_fast_gemm_report()).unwrap(),
        )
        .unwrap();
        let plan = ModelFastPlan {
            schema: MODEL_FAST_PLAN_SCHEMA.to_string(),
            ops: vec![ModelFastOp {
                op_id: "op0".to_string(),
                op_kind: "gemm".to_string(),
                name: "gemm".to_string(),
                fast_report_path: "gemm_fast.json".to_string(),
                golden_path: None,
                exact_ns: Some(1000),
                fast_ns: Some(0),
                source_hash: None,
                description: None,
            }],
            thresholds: None,
        };

        let report = run_model_fast_plan(&plan, &dir);

        assert!(!report.ok);
        assert!(report.ops[0].timing.is_none());
        assert!(report
            .errors
            .iter()
            .any(|err| err.contains("model FAST fast_ns must be greater than zero")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_fast_plan_compares_golden_counters() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-model-fast-golden-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("gemm_fast.json"),
            serde_json::to_string_pretty(&model_fast_gemm_report()).unwrap(),
        )
        .unwrap();
        fs::write(
            dir.join("gemm_golden.json"),
            serde_json::to_string_pretty(&model_fast_golden(
                "op0",
                "gemm",
                ModelFastGoldenExpected {
                    items: 2,
                    surrogate_replacements: 1,
                    exact_fallbacks: 1,
                    fail_closed: 0,
                    shadow_sampled: 1,
                },
            ))
            .unwrap(),
        )
        .unwrap();
        let plan = ModelFastPlan {
            schema: MODEL_FAST_PLAN_SCHEMA.to_string(),
            ops: vec![ModelFastOp {
                op_id: "op0".to_string(),
                op_kind: "gemm".to_string(),
                name: "gemm".to_string(),
                fast_report_path: "gemm_fast.json".to_string(),
                golden_path: Some("gemm_golden.json".to_string()),
                exact_ns: None,
                fast_ns: None,
                source_hash: None,
                description: None,
            }],
            thresholds: None,
        };

        let report = run_model_fast_plan(&plan, &dir);

        assert!(report.ok, "{report:?}");
        let golden = report.ops[0].golden.as_ref().unwrap();
        assert!(golden.golden_compared);
        assert!(golden.golden_ok);
        assert_eq!(golden.golden_error, None);
        assert!(!golden.tensor_compared);
        assert_eq!(golden.tensor_count, 0);
        assert_eq!(golden.max_abs_error, 0);
        assert!(golden.tensor_errors.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_fast_plan_compares_exact_golden_tensors() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-model-fast-golden-tensor-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("gemm_fast.json"),
            serde_json::to_string_pretty(&model_fast_gemm_report()).unwrap(),
        )
        .unwrap();
        let mut golden = model_fast_golden(
            "op0",
            "gemm",
            ModelFastGoldenExpected {
                items: 2,
                surrogate_replacements: 1,
                exact_fallbacks: 1,
                fail_closed: 0,
                shadow_sampled: 1,
            },
        );
        golden.expected_tensors = Some(BTreeMap::from([(
            "c".to_string(),
            serde_json::json!([[1, 2], [3, 4]]),
        )]));
        golden.actual_tensors = Some(BTreeMap::from([(
            "c".to_string(),
            serde_json::json!([[1, 2], [3, 4]]),
        )]));
        fs::write(
            dir.join("gemm_golden.json"),
            serde_json::to_string_pretty(&golden).unwrap(),
        )
        .unwrap();
        let plan = ModelFastPlan {
            schema: MODEL_FAST_PLAN_SCHEMA.to_string(),
            ops: vec![ModelFastOp {
                op_id: "op0".to_string(),
                op_kind: "gemm".to_string(),
                name: "gemm".to_string(),
                fast_report_path: "gemm_fast.json".to_string(),
                golden_path: Some("gemm_golden.json".to_string()),
                exact_ns: None,
                fast_ns: None,
                source_hash: None,
                description: None,
            }],
            thresholds: None,
        };

        let report = run_model_fast_plan(&plan, &dir);

        assert!(report.ok, "{report:?}");
        let golden = report.ops[0].golden.as_ref().unwrap();
        assert!(golden.golden_ok);
        assert!(golden.tensor_compared);
        assert_eq!(golden.tensor_count, 1);
        assert_eq!(golden.max_abs_error, 0);
        assert!(golden.tensor_errors.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_fast_plan_accepts_tensor_tolerance() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-model-fast-golden-tensor-tolerance-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("gemm_fast.json"),
            serde_json::to_string_pretty(&model_fast_gemm_report()).unwrap(),
        )
        .unwrap();
        let mut golden = model_fast_golden(
            "op0",
            "gemm",
            ModelFastGoldenExpected {
                items: 2,
                surrogate_replacements: 1,
                exact_fallbacks: 1,
                fail_closed: 0,
                shadow_sampled: 1,
            },
        );
        golden.expected_tensors = Some(BTreeMap::from([(
            "c".to_string(),
            serde_json::json!([[10, 20]]),
        )]));
        golden.actual_tensors = Some(BTreeMap::from([(
            "c".to_string(),
            serde_json::json!([[12, 18]]),
        )]));
        golden.max_abs_error = Some(2);
        fs::write(
            dir.join("gemm_golden.json"),
            serde_json::to_string_pretty(&golden).unwrap(),
        )
        .unwrap();
        let plan = ModelFastPlan {
            schema: MODEL_FAST_PLAN_SCHEMA.to_string(),
            ops: vec![ModelFastOp {
                op_id: "op0".to_string(),
                op_kind: "gemm".to_string(),
                name: "gemm".to_string(),
                fast_report_path: "gemm_fast.json".to_string(),
                golden_path: Some("gemm_golden.json".to_string()),
                exact_ns: None,
                fast_ns: None,
                source_hash: None,
                description: None,
            }],
            thresholds: None,
        };

        let report = run_model_fast_plan(&plan, &dir);

        assert!(report.ok, "{report:?}");
        let golden = report.ops[0].golden.as_ref().unwrap();
        assert!(golden.tensor_compared);
        assert_eq!(golden.max_abs_error, 2);
        assert!(golden.tensor_errors.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_fast_plan_rejects_tensor_value_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-model-fast-golden-tensor-mismatch-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("gemm_fast.json"),
            serde_json::to_string_pretty(&model_fast_gemm_report()).unwrap(),
        )
        .unwrap();
        let mut golden = model_fast_golden(
            "op0",
            "gemm",
            ModelFastGoldenExpected {
                items: 2,
                surrogate_replacements: 1,
                exact_fallbacks: 1,
                fail_closed: 0,
                shadow_sampled: 1,
            },
        );
        golden.expected_tensors = Some(BTreeMap::from([(
            "c".to_string(),
            serde_json::json!([[10]]),
        )]));
        golden.actual_tensors = Some(BTreeMap::from([(
            "c".to_string(),
            serde_json::json!([[13]]),
        )]));
        golden.max_abs_error = Some(2);
        fs::write(
            dir.join("gemm_golden.json"),
            serde_json::to_string_pretty(&golden).unwrap(),
        )
        .unwrap();
        let plan = ModelFastPlan {
            schema: MODEL_FAST_PLAN_SCHEMA.to_string(),
            ops: vec![ModelFastOp {
                op_id: "op0".to_string(),
                op_kind: "gemm".to_string(),
                name: "gemm".to_string(),
                fast_report_path: "gemm_fast.json".to_string(),
                golden_path: Some("gemm_golden.json".to_string()),
                exact_ns: None,
                fast_ns: None,
                source_hash: None,
                description: None,
            }],
            thresholds: None,
        };

        let report = run_model_fast_plan(&plan, &dir);

        assert!(!report.ok);
        let golden = report.ops[0].golden.as_ref().unwrap();
        assert!(golden.tensor_compared);
        assert_eq!(golden.max_abs_error, 3);
        assert!(golden
            .tensor_errors
            .iter()
            .any(|err| err.contains("abs_error 3 exceeds allowed 2")));
        assert!(golden
            .golden_error
            .as_ref()
            .unwrap()
            .contains("abs_error 3 exceeds allowed 2"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_fast_plan_rejects_tensor_shape_and_name_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-model-fast-golden-tensor-shape-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("gemm_fast.json"),
            serde_json::to_string_pretty(&model_fast_gemm_report()).unwrap(),
        )
        .unwrap();
        let mut golden = model_fast_golden(
            "op0",
            "gemm",
            ModelFastGoldenExpected {
                items: 2,
                surrogate_replacements: 1,
                exact_fallbacks: 1,
                fail_closed: 0,
                shadow_sampled: 1,
            },
        );
        golden.expected_tensors = Some(BTreeMap::from([
            ("c".to_string(), serde_json::json!([[1, 2]])),
            ("missing".to_string(), serde_json::json!([1])),
        ]));
        golden.actual_tensors = Some(BTreeMap::from([
            ("c".to_string(), serde_json::json!([[1], [2]])),
            ("extra".to_string(), serde_json::json!([1])),
        ]));
        fs::write(
            dir.join("gemm_golden.json"),
            serde_json::to_string_pretty(&golden).unwrap(),
        )
        .unwrap();
        let plan = ModelFastPlan {
            schema: MODEL_FAST_PLAN_SCHEMA.to_string(),
            ops: vec![ModelFastOp {
                op_id: "op0".to_string(),
                op_kind: "gemm".to_string(),
                name: "gemm".to_string(),
                fast_report_path: "gemm_fast.json".to_string(),
                golden_path: Some("gemm_golden.json".to_string()),
                exact_ns: None,
                fast_ns: None,
                source_hash: None,
                description: None,
            }],
            thresholds: None,
        };

        let report = run_model_fast_plan(&plan, &dir);

        assert!(!report.ok);
        let golden = report.ops[0].golden.as_ref().unwrap();
        assert!(golden.tensor_compared);
        assert_eq!(golden.tensor_count, 2);
        assert!(golden
            .tensor_errors
            .iter()
            .any(|err| err.contains("tensor `missing` missing from actual_tensors")));
        assert!(golden
            .tensor_errors
            .iter()
            .any(|err| err.contains("unexpected actual tensor `extra`")));
        assert!(golden
            .tensor_errors
            .iter()
            .any(|err| err.contains("tensor `c` shape [1, 2] does not match actual shape [2, 1]")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_fast_plan_rejects_golden_counter_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-model-fast-golden-mismatch-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("gemm_fast.json"),
            serde_json::to_string_pretty(&model_fast_gemm_report()).unwrap(),
        )
        .unwrap();
        fs::write(
            dir.join("gemm_golden.json"),
            serde_json::to_string_pretty(&model_fast_golden(
                "op0",
                "gemm",
                ModelFastGoldenExpected {
                    items: 3,
                    surrogate_replacements: 1,
                    exact_fallbacks: 1,
                    fail_closed: 0,
                    shadow_sampled: 0,
                },
            ))
            .unwrap(),
        )
        .unwrap();
        let plan = ModelFastPlan {
            schema: MODEL_FAST_PLAN_SCHEMA.to_string(),
            ops: vec![ModelFastOp {
                op_id: "op0".to_string(),
                op_kind: "gemm".to_string(),
                name: "gemm".to_string(),
                fast_report_path: "gemm_fast.json".to_string(),
                golden_path: Some("gemm_golden.json".to_string()),
                exact_ns: None,
                fast_ns: None,
                source_hash: None,
                description: None,
            }],
            thresholds: None,
        };

        let report = run_model_fast_plan(&plan, &dir);

        assert!(!report.ok);
        let golden = report.ops[0].golden.as_ref().unwrap();
        assert!(golden.golden_compared);
        assert!(!golden.golden_ok);
        assert!(golden
            .golden_error
            .as_ref()
            .unwrap()
            .contains("items expected 3 but actual was 2"));
        assert!(report
            .errors
            .iter()
            .any(|err| err.contains("shadow_sampled expected 0 but actual was 1")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_fast_plan_rejects_golden_identity_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-model-fast-golden-identity-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("gemm_fast.json"),
            serde_json::to_string_pretty(&model_fast_gemm_report()).unwrap(),
        )
        .unwrap();
        let mut golden = model_fast_golden(
            "other",
            "event",
            ModelFastGoldenExpected {
                items: 2,
                surrogate_replacements: 1,
                exact_fallbacks: 1,
                fail_closed: 0,
                shadow_sampled: 1,
            },
        );
        golden.schema = "bad".to_string();
        fs::write(
            dir.join("gemm_golden.json"),
            serde_json::to_string_pretty(&golden).unwrap(),
        )
        .unwrap();
        let plan = ModelFastPlan {
            schema: MODEL_FAST_PLAN_SCHEMA.to_string(),
            ops: vec![ModelFastOp {
                op_id: "op0".to_string(),
                op_kind: "gemm".to_string(),
                name: "gemm".to_string(),
                fast_report_path: "gemm_fast.json".to_string(),
                golden_path: Some("gemm_golden.json".to_string()),
                exact_ns: None,
                fast_ns: None,
                source_hash: None,
                description: None,
            }],
            thresholds: None,
        };

        let report = run_model_fast_plan(&plan, &dir);

        assert!(!report.ok);
        let err = report.ops[0]
            .golden
            .as_ref()
            .unwrap()
            .golden_error
            .as_ref()
            .unwrap();
        assert!(err.contains("unsupported model FAST golden schema `bad`"));
        assert!(err.contains("golden op_id `other` does not match op `op0`"));
        assert!(err.contains("golden op_kind `event` does not match op kind `gemm`"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_fast_plan_reports_missing_and_unsupported_ops() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-model-fast-errors-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let plan = ModelFastPlan {
            schema: MODEL_FAST_PLAN_SCHEMA.to_string(),
            ops: vec![
                ModelFastOp {
                    op_id: "missing".to_string(),
                    op_kind: "gemm".to_string(),
                    name: "missing".to_string(),
                    fast_report_path: "missing.json".to_string(),
                    golden_path: None,
                    exact_ns: None,
                    fast_ns: None,
                    source_hash: None,
                    description: None,
                },
                ModelFastOp {
                    op_id: "bad-kind".to_string(),
                    op_kind: "attention".to_string(),
                    name: "attention".to_string(),
                    fast_report_path: "unused.json".to_string(),
                    golden_path: None,
                    exact_ns: None,
                    fast_ns: None,
                    source_hash: None,
                    description: None,
                },
            ],
            thresholds: Some(ModelFastThresholds {
                min_op_coverage: Some(1.0),
                min_item_coverage: None,
                max_fallback_ratio: None,
                min_shadow_sample_ratio: None,
            }),
        };

        let report = run_model_fast_plan(&plan, &dir);

        assert!(!report.ok);
        assert_eq!(report.op_count, 2);
        assert_eq!(report.coverage.op_coverage, 0.0);
        assert!(!report.coverage.accepted);
        assert!(report
            .coverage
            .reject_reasons
            .iter()
            .any(|err| err.contains("op_coverage")));
        assert!(report
            .errors
            .iter()
            .any(|err| err.contains("failed to open FAST report")));
        assert!(report
            .errors
            .iter()
            .any(|err| err.contains("unsupported model FAST op kind `attention`")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_fast_plan_reports_schema_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-model-fast-schema-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let mut gemm = model_fast_gemm_report();
        gemm.schema = "bad".to_string();
        fs::write(
            dir.join("gemm_fast.json"),
            serde_json::to_string_pretty(&gemm).unwrap(),
        )
        .unwrap();
        let plan = ModelFastPlan {
            schema: MODEL_FAST_PLAN_SCHEMA.to_string(),
            ops: vec![ModelFastOp {
                op_id: "op0".to_string(),
                op_kind: "gemm".to_string(),
                name: "gemm".to_string(),
                fast_report_path: "gemm_fast.json".to_string(),
                golden_path: None,
                exact_ns: None,
                fast_ns: None,
                source_hash: None,
                description: None,
            }],
            thresholds: None,
        };

        let report = run_model_fast_plan(&plan, &dir);

        assert!(!report.ok);
        assert!(report
            .errors
            .iter()
            .any(|err| err.contains("unsupported GEMM FAST report schema `bad`")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_fast_op_kind_infers_known_fast_report_schemas() {
        let gemm = serde_json::json!({ "schema": GEMM_FAST_RUN_SCHEMA });
        let event = serde_json::json!({ "schema": EVENT_FAST_RUN_SCHEMA });

        assert_eq!(
            infer_model_fast_op_kind(gemm.to_string().as_bytes()).unwrap(),
            "gemm"
        );
        assert_eq!(
            infer_model_fast_op_kind(event.to_string().as_bytes()).unwrap(),
            "event"
        );
    }

    #[test]
    fn model_fast_op_kind_rejects_unknown_schema() {
        let report = serde_json::json!({ "schema": "other" });

        let err = infer_model_fast_op_kind(report.to_string().as_bytes()).unwrap_err();

        assert!(err
            .to_string()
            .contains("unsupported model FAST op report schema `other`"));
    }

    #[test]
    fn shadow_event_corpus_reports_mismatch_and_ordering() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-shadow-mismatch-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cache_miss_rule.json"), b"rule").unwrap();
        let manifest = event_manifest(
            "cache_miss_rule.json".to_string(),
            sha256_hex(b"rule"),
            "event-source".to_string(),
        );
        let mut corpus = event_corpus(0);
        corpus.events[0].lane = Some(0);
        corpus.events.push(event_corpus(1).events.remove(0));
        corpus.events[1].sample_id = 8;
        corpus.events[1].lane = Some(1);

        let report = shadow_event_corpus(&manifest, dir.join("manifest.json"), &corpus);

        assert!(!report.ok);
        assert_eq!(report.metrics.accuracy, 0.5);
        assert_eq!(report.metrics.false_positive, 1);
        assert_eq!(report.total_lanes, 2);
        assert_eq!(report.lanes.len(), 2);
        assert_eq!(report.lanes[0].lane, 0);
        assert_eq!(report.lanes[0].samples, 1);
        assert_eq!(report.lanes[0].metrics.accuracy, 0.0);
        assert_eq!(report.lanes[0].metrics.false_positive, 1);
        assert_eq!(report.lanes[1].lane, 1);
        assert_eq!(report.lanes[1].samples, 1);
        assert_eq!(report.lanes[1].metrics.accuracy, 1.0);
        assert_eq!(report.results.len(), 2);
        assert_eq!(report.results[0].sample_id, 7);
        assert_eq!(report.results[0].lane, 0);
        assert_eq!(report.results[1].sample_id, 8);
        assert_eq!(report.results[1].lane, 1);
        assert!(!report.results[0].ok);
        assert!(report.results[1].ok);
        assert_eq!(report.first_mismatch.unwrap().sample_id, 7);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn shadow_event_corpus_surfaces_manifest_hash_failure() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-shadow-hash-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cache_miss_rule.json"), b"rule").unwrap();
        let manifest = event_manifest(
            "cache_miss_rule.json".to_string(),
            "bad".to_string(),
            "event-source".to_string(),
        );

        let report = shadow_event_corpus(&manifest, dir.join("manifest.json"), &event_corpus(1));

        assert!(!report.ok);
        assert!(report.results.is_empty());
        assert!(report
            .manifest
            .errors
            .iter()
            .any(|err| err.contains("sha256 mismatch")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn event_validation_reports_source_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-source-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cache_miss_rule.json"), b"rule").unwrap();
        let manifest = event_manifest(
            "cache_miss_rule.json".to_string(),
            sha256_hex(b"rule"),
            "wrong-source".to_string(),
        );

        let report = validate_event_corpus(&manifest, dir.join("manifest.json"), &event_corpus(1));

        assert!(!report.ok);
        assert!(report
            .errors
            .iter()
            .any(|err| err.contains("source hash mismatch")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn event_validation_reports_missing_signal_feature() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-signal-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cache_miss_rule.json"), b"rule").unwrap();
        let manifest = event_manifest(
            "cache_miss_rule.json".to_string(),
            sha256_hex(b"rule"),
            "event-source".to_string(),
        );
        let mut corpus = event_corpus(1);
        corpus.events[0].signals[0].remove("tag_delta");

        let report = validate_event_corpus(&manifest, dir.join("manifest.json"), &corpus);

        assert!(!report.ok);
        assert!(report
            .errors
            .iter()
            .any(|err| err.contains("missing feature `tag_delta`")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn event_validation_reports_mixed_window() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-window-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cache_miss_rule.json"), b"rule").unwrap();
        let manifest = event_manifest(
            "cache_miss_rule.json".to_string(),
            sha256_hex(b"rule"),
            "event-source".to_string(),
        );
        let mut corpus = event_corpus(1);
        corpus.events[0].window_cycles = 3;

        let report = validate_event_corpus(&manifest, dir.join("manifest.json"), &corpus);

        assert!(!report.ok);
        assert!(report
            .errors
            .iter()
            .any(|err| err.contains("window 3 does not match manifest window 2")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn event_validation_reports_artifact_hash_failure() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-event-hash-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cache_miss_rule.json"), b"rule").unwrap();
        let manifest = event_manifest(
            "cache_miss_rule.json".to_string(),
            "bad".to_string(),
            "event-source".to_string(),
        );

        let report = validate_event_corpus(&manifest, dir.join("manifest.json"), &event_corpus(1));

        assert!(!report.ok);
        assert!(report
            .manifest
            .errors
            .iter()
            .any(|err| err.contains("sha256 mismatch")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mock_gemm_reports_metrics_and_provenance() {
        let dir =
            std::env::temp_dir().join(format!("rrtl-surrogate-test-gemm-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        let transaction = GemmTransaction {
            schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
            lane: None,
            rows: 2,
            cols: 2,
            k: 2,
            a: vec![vec![1, 2], vec![3, 4]],
            w: vec![vec![5, 6], vec![7, 8]],
            expected_c: Some(vec![vec![19, 22], vec![43, 50]]),
            expected_latency_cycles: Some(5),
        };
        let result =
            run_gemm_transaction(&manifest, dir.join("manifest.json"), &transaction).unwrap();
        assert!(result.ok);
        assert_eq!(result.c, vec![vec![19, 22], vec![43, 50]]);
        assert!(!result.provenance.exact);
        assert_eq!(result.metrics.unwrap().max_abs_error, 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mock_gemm_batch_preserves_order_and_default_lanes() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-batch-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        let first = GemmTransaction {
            schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
            lane: None,
            rows: 2,
            cols: 2,
            k: 2,
            a: vec![vec![1, 2], vec![3, 4]],
            w: vec![vec![5, 6], vec![7, 8]],
            expected_c: Some(vec![vec![19, 22], vec![43, 50]]),
            expected_latency_cycles: Some(5),
        };
        let second = GemmTransaction {
            schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
            lane: None,
            rows: 2,
            cols: 2,
            k: 2,
            a: vec![vec![2, 0], vec![1, 1]],
            w: vec![vec![3, 4], vec![5, 6]],
            expected_c: Some(vec![vec![6, 8], vec![8, 10]]),
            expected_latency_cycles: Some(5),
        };
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: Some("batch-source".to_string()),
            transactions: vec![first, second],
        };

        let report = run_gemm_batch(&manifest, dir.join("manifest.json"), &batch);

        assert!(report.ok, "{report:?}");
        assert_eq!(report.schema, GEMM_BATCH_RESULT_SCHEMA);
        assert_eq!(report.count, 2);
        assert_eq!(report.total_lanes, 2);
        assert_eq!(report.results[0].index, 0);
        assert_eq!(report.results[0].lane, 0);
        assert_eq!(report.results[1].index, 1);
        assert_eq!(report.results[1].lane, 1);
        assert_eq!(report.lanes[0].count, 1);
        assert_eq!(report.lanes[1].count, 1);
        assert_eq!(report.metrics.max_abs_error, 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mock_gemm_batch_aggregates_repeated_lanes_and_failures() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-batch-failure-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        let valid = GemmTransaction {
            schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
            lane: Some(7),
            rows: 2,
            cols: 2,
            k: 2,
            a: vec![vec![1, 2], vec![3, 4]],
            w: vec![vec![5, 6], vec![7, 8]],
            expected_c: Some(vec![vec![19, 22], vec![43, 50]]),
            expected_latency_cycles: Some(5),
        };
        let invalid = GemmTransaction {
            schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
            lane: Some(7),
            rows: 3,
            cols: 2,
            k: 2,
            a: vec![vec![1, 2], vec![3, 4], vec![5, 6]],
            w: vec![vec![5, 6], vec![7, 8]],
            expected_c: None,
            expected_latency_cycles: None,
        };
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: None,
            transactions: vec![valid, invalid],
        };

        let report = run_gemm_batch(&manifest, dir.join("manifest.json"), &batch);

        assert!(!report.ok);
        assert_eq!(report.total_lanes, 1);
        assert_eq!(report.lanes[0].lane, 7);
        assert_eq!(report.lanes[0].count, 2);
        assert_eq!(report.lanes[0].ok, 1);
        assert_eq!(report.lanes[0].failed, 1);
        assert!(report.results[0].ok);
        assert!(!report.results[1].ok);
        assert!(report.results[1]
            .error
            .as_ref()
            .unwrap()
            .contains("outside manifest domain"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn policy_gemm_batch_uses_surrogate_when_in_tolerance() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-policy-use-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: None,
            transactions: vec![GemmTransaction {
                schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
                lane: Some(3),
                rows: 2,
                cols: 2,
                k: 2,
                a: vec![vec![1, 2], vec![3, 4]],
                w: vec![vec![5, 6], vec![7, 8]],
                expected_c: Some(vec![vec![19, 22], vec![43, 50]]),
                expected_latency_cycles: Some(5),
            }],
        };

        let report = policy_gemm_batch(&manifest, dir.join("manifest.json"), &batch);

        assert!(report.ok, "{report:?}");
        assert_eq!(report.used_surrogate, 1);
        assert_eq!(report.exact_fallbacks, 0);
        assert_eq!(report.fail_closed, 0);
        assert_eq!(
            report.results[0].decision,
            GemmPolicyDecision::SurrogateUsed
        );
        assert!(
            !report.results[0]
                .surrogate_result
                .as_ref()
                .unwrap()
                .provenance
                .exact
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fast_gemm_replaces_with_surrogate_for_approximate_policy() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-fast-use-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: None,
            transactions: vec![GemmTransaction {
                schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
                lane: Some(1),
                rows: 2,
                cols: 2,
                k: 2,
                a: vec![vec![1, 2], vec![3, 4]],
                w: vec![vec![5, 6], vec![7, 8]],
                expected_c: Some(vec![vec![19, 22], vec![43, 50]]),
                expected_latency_cycles: Some(5),
            }],
        };
        let policy = policy_gemm_batch(&manifest, dir.join("manifest.json"), &batch);
        let plan = plan_runtime_gemm(&policy, 2);

        let report = run_fast_gemm_batch(&manifest, dir.join("manifest.json"), &batch, &plan);

        assert!(report.ok, "{report:?}");
        assert_eq!(report.schema, GEMM_FAST_RUN_SCHEMA);
        assert_eq!(report.total_lanes, 2);
        assert_eq!(report.surrogate_replacements, 1);
        assert_eq!(report.exact_fallbacks, 0);
        assert_eq!(
            report.results[0].source_result,
            Some(GemmRuntimeSourceResult::Surrogate)
        );
        assert!(!report.results[0].provenance.as_ref().unwrap().exact);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fast_gemm_uses_exact_for_shadow_policy_and_preserves_shadow() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-fast-shadow-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let mut manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        manifest.policy.mode = PolicyMode::ShadowCompare;
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: None,
            transactions: vec![GemmTransaction {
                schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
                lane: Some(0),
                rows: 2,
                cols: 2,
                k: 2,
                a: vec![vec![1, 2], vec![3, 4]],
                w: vec![vec![5, 6], vec![7, 8]],
                expected_c: Some(vec![vec![19, 22], vec![43, 50]]),
                expected_latency_cycles: Some(5),
            }],
        };
        let policy = policy_gemm_batch(&manifest, dir.join("manifest.json"), &batch);
        let plan = plan_runtime_gemm(&policy, 1);

        let report = run_fast_gemm_batch(&manifest, dir.join("manifest.json"), &batch, &plan);

        assert!(report.ok, "{report:?}");
        assert_eq!(report.surrogate_replacements, 0);
        assert_eq!(report.exact_fallbacks, 1);
        assert_eq!(report.shadow_compared, 1);
        assert_eq!(report.shadow_passed, 1);
        assert_eq!(
            report.results[0].source_result,
            Some(GemmRuntimeSourceResult::Exact)
        );
        assert!(report.results[0].provenance.as_ref().unwrap().exact);
        assert_eq!(report.results[0].shadow_ok, Some(true));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fast_gemm_uses_exact_fallback_for_out_of_domain_transaction() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-fast-fallback-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let mut manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        manifest.policy.fallback = FallbackPolicy::ExactFallback;
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: None,
            transactions: vec![GemmTransaction {
                schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
                lane: Some(0),
                rows: 3,
                cols: 2,
                k: 2,
                a: vec![vec![1, 2], vec![3, 4], vec![5, 6]],
                w: vec![vec![5, 6], vec![7, 8]],
                expected_c: Some(vec![vec![19, 22], vec![43, 50], vec![67, 78]]),
                expected_latency_cycles: Some(6),
            }],
        };
        let policy = policy_gemm_batch(&manifest, dir.join("manifest.json"), &batch);
        let plan = plan_runtime_gemm(&policy, 1);

        let report = run_fast_gemm_batch(&manifest, dir.join("manifest.json"), &batch, &plan);

        assert!(report.ok, "{report:?}");
        assert_eq!(report.surrogate_replacements, 0);
        assert_eq!(report.exact_fallbacks, 1);
        assert_eq!(
            report.results[0].source_result,
            Some(GemmRuntimeSourceResult::Exact)
        );
        assert!(report.results[0].provenance.as_ref().unwrap().exact);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fast_gemm_reports_fail_closed_without_result() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-fast-fail-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: None,
            transactions: vec![GemmTransaction {
                schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
                lane: Some(0),
                rows: 3,
                cols: 2,
                k: 2,
                a: vec![vec![1, 2], vec![3, 4], vec![5, 6]],
                w: vec![vec![5, 6], vec![7, 8]],
                expected_c: None,
                expected_latency_cycles: None,
            }],
        };
        let plan = GemmRuntimePlan {
            schema: GEMM_RUNTIME_PLAN_SCHEMA.to_string(),
            ok: true,
            total_lanes: 1,
            workers: vec![GemmRuntimeWorkerSummary {
                worker_id: "worker0".to_string(),
                start_lane: 0,
                lanes: 1,
                assigned_items: 0,
                used_surrogate: 0,
                exact_fallbacks: 0,
            }],
            items: Vec::new(),
            errors: Vec::new(),
        };

        let report = run_fast_gemm_batch(&manifest, dir.join("manifest.json"), &batch, &plan);

        assert!(!report.ok);
        assert_eq!(report.fail_closed, 1);
        assert_eq!(report.results[0].source_result, None);
        assert!(report.results[0].result.is_none());
        assert!(report.results[0].error.is_some());
        assert_eq!(report.workers[0].assigned_items, 1);
        assert_eq!(report.workers[0].fail_closed, 1);
        assert_eq!(report.lanes[0].lane, 0);
        assert_eq!(report.lanes[0].fail_closed, 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fast_gemm_reports_worker_and_lane_summaries() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-fast-summaries-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: None,
            transactions: vec![
                valid_gemm_transaction(Some(0), 1),
                valid_gemm_transaction(Some(1), 2),
                valid_gemm_transaction(Some(2), 3),
            ],
        };
        let policy = policy_gemm_batch(&manifest, dir.join("manifest.json"), &batch);
        let workers = vec![
            GemmRuntimeWorkerSpec {
                worker_id: "cpu-a".to_string(),
                start_lane: 0,
                lanes: 2,
            },
            GemmRuntimeWorkerSpec {
                worker_id: "cpu-b".to_string(),
                start_lane: 2,
                lanes: 1,
            },
        ];
        let plan = plan_runtime_gemm_for_workers(&policy, &workers);

        let report = run_fast_gemm_batch(&manifest, dir.join("manifest.json"), &batch, &plan);

        assert!(report.ok, "{report:?}");
        assert_eq!(report.surrogate_replacements, 3);
        assert_eq!(report.exact_fallbacks, 0);
        assert_eq!(report.workers.len(), 2);
        assert_eq!(report.workers[0].worker_id, "cpu-a");
        assert_eq!(report.workers[0].assigned_items, 2);
        assert_eq!(report.workers[0].surrogate_replacements, 2);
        assert_eq!(report.workers[1].worker_id, "cpu-b");
        assert_eq!(report.workers[1].assigned_items, 1);
        assert_eq!(report.workers[1].surrogate_replacements, 1);
        assert_eq!(report.lanes.len(), 3);
        assert_eq!(report.lanes[0].lane, 0);
        assert_eq!(report.lanes[1].lane, 1);
        assert_eq!(report.lanes[2].lane, 2);
        assert!(report
            .lanes
            .iter()
            .all(|lane| lane.count == 1 && lane.surrogate_replacements == 1));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fast_gemm_rejects_runtime_plan_extra_item() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-fast-extra-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: None,
            transactions: vec![valid_gemm_transaction(Some(0), 1)],
        };
        let policy = policy_gemm_batch(&manifest, dir.join("manifest.json"), &batch);
        let mut plan = plan_runtime_gemm(&policy, 1);
        let mut extra = plan.items[0].clone();
        extra.index = 99;
        plan.items.push(extra);

        let report = run_fast_gemm_batch(&manifest, dir.join("manifest.json"), &batch, &plan);

        assert!(!report.ok);
        assert!(report
            .errors
            .iter()
            .any(|err| err.contains("extra item 99")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fast_gemm_rejects_runtime_plan_missing_item() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-fast-missing-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: None,
            transactions: vec![valid_gemm_transaction(Some(0), 1)],
        };
        let policy = policy_gemm_batch(&manifest, dir.join("manifest.json"), &batch);
        let mut plan = plan_runtime_gemm(&policy, 1);
        plan.items.clear();

        let report = run_fast_gemm_batch(&manifest, dir.join("manifest.json"), &batch, &plan);

        assert!(!report.ok);
        assert!(report
            .errors
            .iter()
            .any(|err| err.contains("missing non-fail-closed policy item 0")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fast_gemm_marks_deterministic_shadow_samples() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-fast-samples-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: None,
            transactions: vec![
                valid_gemm_transaction(Some(0), 1),
                valid_gemm_transaction(Some(1), 2),
                valid_gemm_transaction(Some(2), 3),
                valid_gemm_transaction(Some(3), 4),
            ],
        };
        let policy = policy_gemm_batch(&manifest, dir.join("manifest.json"), &batch);
        let plan = plan_runtime_gemm(&policy, 4);

        let report = run_fast_gemm_batch_with_options(
            &manifest,
            dir.join("manifest.json"),
            &batch,
            &plan,
            GemmFastRunOptions {
                shadow_sample_stride: Some(2),
                shadow_sample_offset: 1,
            },
        );

        assert!(report.ok, "{report:?}");
        assert_eq!(
            report
                .results
                .iter()
                .map(|item| item.shadow_sampled)
                .collect::<Vec<_>>(),
            vec![false, true, false, true]
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn policy_gemm_batch_fail_closed_rejects_out_of_domain() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-policy-fail-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: None,
            transactions: vec![GemmTransaction {
                schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
                lane: Some(0),
                rows: 3,
                cols: 2,
                k: 2,
                a: vec![vec![1, 2], vec![3, 4], vec![5, 6]],
                w: vec![vec![5, 6], vec![7, 8]],
                expected_c: None,
                expected_latency_cycles: None,
            }],
        };

        let report = policy_gemm_batch(&manifest, dir.join("manifest.json"), &batch);

        assert!(!report.ok);
        assert_eq!(report.fail_closed, 1);
        assert_eq!(report.results[0].decision, GemmPolicyDecision::FailClosed);
        assert!(report.results[0]
            .error
            .as_ref()
            .unwrap()
            .contains("outside manifest domain"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn policy_gemm_batch_exact_fallback_handles_out_of_domain() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-policy-fallback-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let mut manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        manifest.policy.fallback = FallbackPolicy::ExactFallback;
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: None,
            transactions: vec![GemmTransaction {
                schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
                lane: Some(1),
                rows: 3,
                cols: 2,
                k: 2,
                a: vec![vec![1, 2], vec![3, 4], vec![5, 6]],
                w: vec![vec![5, 6], vec![7, 8]],
                expected_c: Some(vec![vec![19, 22], vec![43, 50], vec![67, 78]]),
                expected_latency_cycles: Some(6),
            }],
        };

        let report = policy_gemm_batch(&manifest, dir.join("manifest.json"), &batch);

        assert!(report.ok, "{report:?}");
        assert_eq!(report.exact_fallbacks, 1);
        assert_eq!(
            report.results[0].decision,
            GemmPolicyDecision::ExactFallback
        );
        assert!(
            report.results[0]
                .exact_result
                .as_ref()
                .unwrap()
                .provenance
                .exact
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn policy_gemm_batch_shadow_compare_keeps_exact_decision() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-policy-shadow-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let mut manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        manifest.policy.mode = PolicyMode::ShadowCompare;
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: None,
            transactions: vec![GemmTransaction {
                schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
                lane: None,
                rows: 2,
                cols: 2,
                k: 2,
                a: vec![vec![1, 2], vec![3, 4]],
                w: vec![vec![5, 6], vec![7, 8]],
                expected_c: Some(vec![vec![19, 22], vec![43, 50]]),
                expected_latency_cycles: Some(5),
            }],
        };

        let report = policy_gemm_batch(&manifest, dir.join("manifest.json"), &batch);

        assert!(report.ok, "{report:?}");
        assert_eq!(report.exact_fallbacks, 1);
        assert_eq!(
            report.results[0].decision,
            GemmPolicyDecision::ShadowCompare
        );
        assert!(report.results[0].surrogate_result.is_some());
        assert!(report.results[0].exact_result.is_some());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn runtime_plan_gemm_preserves_shadow_metadata() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-runtime-shadow-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let mut manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        manifest.policy.mode = PolicyMode::ShadowCompare;
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: None,
            transactions: vec![GemmTransaction {
                schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
                lane: Some(0),
                rows: 2,
                cols: 2,
                k: 2,
                a: vec![vec![1, 2], vec![3, 4]],
                w: vec![vec![5, 6], vec![7, 8]],
                expected_c: Some(vec![vec![19, 22], vec![43, 50]]),
                expected_latency_cycles: Some(3),
            }],
        };
        let policy = policy_gemm_batch(&manifest, dir.join("manifest.json"), &batch);

        let plan = plan_runtime_gemm(&policy, 1);

        assert!(plan.ok, "{plan:?}");
        assert_eq!(plan.items[0].decision, GemmPolicyDecision::ShadowCompare);
        assert_eq!(plan.items[0].source_result, GemmRuntimeSourceResult::Exact);
        assert_eq!(plan.items[0].shadow_ok, Some(true));
        assert_eq!(plan.items[0].shadow_max_abs_error, Some(0));
        assert_eq!(plan.items[0].shadow_latency_error_cycles, Some(0));
        assert_eq!(plan.items[0].shadow_error, None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn policy_gemm_batch_telemetry_only_never_replaces() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-policy-telemetry-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let mut manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        manifest.policy.mode = PolicyMode::TelemetryOnly;
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: None,
            transactions: vec![GemmTransaction {
                schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
                lane: Some(4),
                rows: 2,
                cols: 2,
                k: 2,
                a: vec![vec![1, 2], vec![3, 4]],
                w: vec![vec![5, 6], vec![7, 8]],
                expected_c: Some(vec![vec![19, 22], vec![43, 50]]),
                expected_latency_cycles: Some(5),
            }],
        };

        let report = policy_gemm_batch(&manifest, dir.join("manifest.json"), &batch);

        assert!(report.ok, "{report:?}");
        assert_eq!(report.used_surrogate, 0);
        assert_eq!(report.exact_fallbacks, 1);
        assert_eq!(
            report.results[0].decision,
            GemmPolicyDecision::ExactFallback
        );
        assert!(report.results[0].surrogate_result.is_some());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn runtime_plan_gemm_maps_valid_policy_report() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-runtime-plan-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: None,
            transactions: vec![GemmTransaction {
                schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
                lane: Some(1),
                rows: 2,
                cols: 2,
                k: 2,
                a: vec![vec![1, 2], vec![3, 4]],
                w: vec![vec![5, 6], vec![7, 8]],
                expected_c: Some(vec![vec![19, 22], vec![43, 50]]),
                expected_latency_cycles: Some(5),
            }],
        };
        let policy = policy_gemm_batch(&manifest, dir.join("manifest.json"), &batch);

        let plan = plan_runtime_gemm(&policy, 2);

        assert!(plan.ok, "{plan:?}");
        assert_eq!(plan.schema, GEMM_RUNTIME_PLAN_SCHEMA);
        assert_eq!(plan.total_lanes, 2);
        assert_eq!(plan.workers.len(), 1);
        assert_eq!(plan.workers[0].worker_id, "worker0");
        assert_eq!(plan.workers[0].assigned_items, 1);
        assert_eq!(plan.workers[0].used_surrogate, 1);
        assert_eq!(plan.items[0].index, 0);
        assert_eq!(plan.items[0].lane, 1);
        assert_eq!(plan.items[0].worker_id, "worker0");
        assert_eq!(
            plan.items[0].source_result,
            GemmRuntimeSourceResult::Surrogate
        );
        assert!(!plan.items[0].provenance.exact);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn runtime_plan_gemm_accepts_exact_fallback_provenance() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-surrogate-test-gemm-runtime-plan-exact-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let mut manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        manifest.policy.mode = PolicyMode::TelemetryOnly;
        let batch = GemmBatch {
            schema: GEMM_BATCH_SCHEMA.to_string(),
            source_hash: None,
            transactions: vec![GemmTransaction {
                schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
                lane: Some(0),
                rows: 2,
                cols: 2,
                k: 2,
                a: vec![vec![1, 2], vec![3, 4]],
                w: vec![vec![5, 6], vec![7, 8]],
                expected_c: Some(vec![vec![19, 22], vec![43, 50]]),
                expected_latency_cycles: Some(5),
            }],
        };
        let policy = policy_gemm_batch(&manifest, dir.join("manifest.json"), &batch);

        let plan = plan_runtime_gemm(&policy, 1);

        assert!(plan.ok, "{plan:?}");
        assert_eq!(plan.workers[0].used_surrogate, 0);
        assert_eq!(plan.workers[0].exact_fallbacks, 1);
        assert_eq!(plan.items[0].source_result, GemmRuntimeSourceResult::Exact);
        assert!(plan.items[0].provenance.exact);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn runtime_plan_gemm_maps_items_to_worker_specs() {
        let policy = GemmPolicyReport {
            schema: GEMM_POLICY_REPORT_SCHEMA.to_string(),
            ok: true,
            count: 2,
            used_surrogate: 2,
            exact_fallbacks: 0,
            fail_closed: 0,
            results: vec![
                GemmPolicyItemResult {
                    index: 0,
                    lane: 1,
                    decision: GemmPolicyDecision::SurrogateUsed,
                    ok: true,
                    surrogate_result: Some(mock_gemm_result(false)),
                    exact_result: None,
                    error: None,
                },
                GemmPolicyItemResult {
                    index: 1,
                    lane: 3,
                    decision: GemmPolicyDecision::SurrogateUsed,
                    ok: true,
                    surrogate_result: Some(mock_gemm_result(false)),
                    exact_result: None,
                    error: None,
                },
            ],
        };
        let workers = vec![
            GemmRuntimeWorkerSpec {
                worker_id: "cpu-a".to_string(),
                start_lane: 0,
                lanes: 2,
            },
            GemmRuntimeWorkerSpec {
                worker_id: "cpu-b".to_string(),
                start_lane: 2,
                lanes: 3,
            },
        ];

        let plan = plan_runtime_gemm_for_workers(&policy, &workers);

        assert!(plan.ok, "{plan:?}");
        assert_eq!(plan.total_lanes, 5);
        assert_eq!(plan.workers[0].worker_id, "cpu-a");
        assert_eq!(plan.workers[0].assigned_items, 1);
        assert_eq!(plan.workers[1].worker_id, "cpu-b");
        assert_eq!(plan.workers[1].assigned_items, 1);
        assert_eq!(plan.items[0].worker_id, "cpu-a");
        assert_eq!(plan.items[1].worker_id, "cpu-b");
    }

    #[test]
    fn runtime_plan_gemm_rejects_non_contiguous_worker_specs() {
        let policy = GemmPolicyReport {
            schema: GEMM_POLICY_REPORT_SCHEMA.to_string(),
            ok: true,
            count: 1,
            used_surrogate: 1,
            exact_fallbacks: 0,
            fail_closed: 0,
            results: vec![GemmPolicyItemResult {
                index: 0,
                lane: 0,
                decision: GemmPolicyDecision::SurrogateUsed,
                ok: true,
                surrogate_result: Some(mock_gemm_result(false)),
                exact_result: None,
                error: None,
            }],
        };
        let workers = vec![GemmRuntimeWorkerSpec {
            worker_id: "cpu-a".to_string(),
            start_lane: 1,
            lanes: 2,
        }];

        let plan = plan_runtime_gemm_for_workers(&policy, &workers);

        assert!(!plan.ok);
        assert!(plan.errors.iter().any(|err| err.contains("contiguous")));
    }

    #[test]
    fn runtime_plan_gemm_rejects_fail_closed_policy_report() {
        let policy = GemmPolicyReport {
            schema: GEMM_POLICY_REPORT_SCHEMA.to_string(),
            ok: false,
            count: 1,
            used_surrogate: 0,
            exact_fallbacks: 0,
            fail_closed: 1,
            results: vec![GemmPolicyItemResult {
                index: 0,
                lane: 0,
                decision: GemmPolicyDecision::FailClosed,
                ok: false,
                surrogate_result: None,
                exact_result: None,
                error: Some("failed".to_string()),
            }],
        };

        let plan = plan_runtime_gemm(&policy, 1);

        assert!(!plan.ok);
        assert!(plan.errors.iter().any(|err| err.contains("fail-closed")));
    }

    #[test]
    fn runtime_plan_gemm_rejects_lane_outside_topology() {
        let mut provenance = mock_manifest(
            "artifact.mock".to_string(),
            "hash".to_string(),
            "source".to_string(),
        )
        .policy;
        provenance.provenance_tag = "approximate".to_string();
        let policy = GemmPolicyReport {
            schema: GEMM_POLICY_REPORT_SCHEMA.to_string(),
            ok: true,
            count: 1,
            used_surrogate: 1,
            exact_fallbacks: 0,
            fail_closed: 0,
            results: vec![GemmPolicyItemResult {
                index: 0,
                lane: 3,
                decision: GemmPolicyDecision::SurrogateUsed,
                ok: true,
                surrogate_result: Some(GemmRunResult {
                    schema: GEMM_RESULT_SCHEMA.to_string(),
                    surrogate_id: "mock".to_string(),
                    ok: true,
                    c: vec![vec![1]],
                    telemetry: GemmTelemetry {
                        latency_cycles: 1,
                        active_cycles: 1,
                        utilization: 1.0,
                    },
                    metrics: None,
                    provenance: Provenance {
                        tag: "approximate".to_string(),
                        exact: false,
                        surrogate_id: "mock".to_string(),
                        model_family: ModelFamily::MockGemm,
                        artifact_format: ArtifactFormat::MockGemm,
                        artifact_hash: "hash".to_string(),
                        source_hash: "source".to_string(),
                        policy: PolicyMode::ApproximateWithTolerance,
                    },
                }),
                exact_result: None,
                error: None,
            }],
        };

        let plan = plan_runtime_gemm(&policy, 2);

        assert!(!plan.ok);
        assert!(plan
            .errors
            .iter()
            .any(|err| err.contains("outside runtime topology")));
    }

    #[test]
    fn mock_gemm_detects_first_divergence() {
        let metrics = gemm_metrics(&[vec![1, 2]], &[vec![1, 9]], 3, Some(4));
        assert_eq!(metrics.max_abs_error, 7);
        assert_eq!(metrics.latency_error_cycles, 1);
        let divergence = metrics.first_divergence.unwrap();
        assert_eq!(divergence.col, 1);
        assert_eq!(divergence.expected, 9);
        assert_eq!(divergence.actual, 2);
    }

    #[test]
    fn validate_surrogate_reports_source_mismatch() {
        let dir =
            std::env::temp_dir().join(format!("rrtl-surrogate-test-source-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            "wrong-source".to_string(),
        );
        let report = validate_surrogate(
            &manifest,
            dir.join("manifest.json"),
            &export_json(),
            &trace_json(),
        )
        .unwrap();
        assert!(!report.ok);
        assert!(report
            .manifest
            .errors
            .iter()
            .any(|err| err.contains("source hash mismatch")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_out_of_domain_transaction() {
        let dir =
            std::env::temp_dir().join(format!("rrtl-surrogate-test-domain-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("artifact.mock"), b"mock").unwrap();
        let manifest = mock_manifest(
            "artifact.mock".to_string(),
            sha256_hex(b"mock"),
            canonical_json_hash(&export_json()).unwrap(),
        );
        let transaction = GemmTransaction {
            schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
            lane: None,
            rows: 3,
            cols: 2,
            k: 2,
            a: vec![vec![1, 2], vec![3, 4], vec![5, 6]],
            w: vec![vec![5, 6], vec![7, 8]],
            expected_c: None,
            expected_latency_cycles: None,
        };
        let err = run_gemm_transaction(&manifest, dir.join("manifest.json"), &transaction)
            .unwrap_err()
            .to_string();
        assert!(err.contains("outside manifest domain"));
        let _ = fs::remove_dir_all(&dir);
    }
}
