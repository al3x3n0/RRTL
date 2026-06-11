use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs},
    path::PathBuf,
    process::{Child, Command, ExitStatus, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use rrtl_core::{compile, CompiledDesign, CompiledInstance, CompiledModule, Design, PortDirection};
use rrtl_gpu_sim::{GpuBatchOptions, GpuBatchSimulator};
use rrtl_ir::{BitType, Diagnostic, ErrorReport, Expr, Signal, SignalInfo, SignalKind, Width};
use rrtl_sim_ir::{
    final_limb_mask, limbs, lower_to_packed_program, JitCpuSimulator, PackedProgram,
    PackedSimulator, PackedSimulatorStorage, PackedSliceGroup, PartitionedSimulator,
    SimdCpuSimulator, SingleLaneMachineSimulator,
};
use rrtl_surrogate::{
    EventPolicyDecision, EventRuntimePlan, GemmPolicyDecision, GemmRuntimePlan,
    GemmRuntimeSourceResult, EVENT_RUNTIME_PLAN_SCHEMA, GEMM_RUNTIME_PLAN_SCHEMA,
};
use serde::{Deserialize, Serialize};

/// Backend kind used by a runtime worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeBackend {
    /// Execute one lane with the scalar machine interpreter.
    ScalarCpu,
    /// Interpret the packed simulation machine on the host CPU.
    PackedCpu,
    /// Execute the packed simulation machine with the host SIMD CPU backend.
    SimdCpu,
    /// Execute the packed simulation machine with the CPU JIT backend.
    JitCpu,
    /// Execute the packed simulation machine with the `rrtl-gpu-sim` wgpu backend.
    Gpu(GpuBatchOptions),
}

/// One worker participating in a distributed runtime topology.
///
/// `node` is placement metadata. The current runtime executes workers in-process;
/// keeping placement explicit makes the shard plan stable for future remote
/// transports and external schedulers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeWorker {
    pub id: String,
    pub node: String,
    pub backend: RuntimeBackend,
    pub lanes: usize,
}

impl RuntimeWorker {
    pub fn local_cpu(id: impl Into<String>, lanes: usize) -> Self {
        Self {
            id: id.into(),
            node: "localhost".to_string(),
            backend: RuntimeBackend::PackedCpu,
            lanes,
        }
    }

    pub fn local_scalar_cpu(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            node: "localhost".to_string(),
            backend: RuntimeBackend::ScalarCpu,
            lanes: 1,
        }
    }

    pub fn local_simd_cpu(id: impl Into<String>, lanes: usize) -> Self {
        Self {
            id: id.into(),
            node: "localhost".to_string(),
            backend: RuntimeBackend::SimdCpu,
            lanes,
        }
    }

    pub fn local_jit_cpu(id: impl Into<String>, lanes: usize) -> Self {
        Self {
            id: id.into(),
            node: "localhost".to_string(),
            backend: RuntimeBackend::JitCpu,
            lanes,
        }
    }

    pub fn local_gpu(id: impl Into<String>, lanes: usize, options: GpuBatchOptions) -> Self {
        Self {
            id: id.into(),
            node: "localhost".to_string(),
            backend: RuntimeBackend::Gpu(options),
            lanes,
        }
    }

    pub fn on_node(mut self, node: impl Into<String>) -> Self {
        self.node = node.into();
        self
    }
}

/// Runtime topology for lane-parallel simulations.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeTopology {
    workers: Vec<RuntimeWorker>,
}

impl RuntimeTopology {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn local_cpu(lanes: usize) -> Self {
        Self {
            workers: vec![RuntimeWorker::local_cpu("cpu0", lanes)],
        }
    }

    pub fn local_heterogeneous(
        cpu_lanes: usize,
        gpu_lanes: usize,
        gpu_options: GpuBatchOptions,
    ) -> Self {
        let mut topology = Self::new();
        if cpu_lanes > 0 {
            topology.push(RuntimeWorker::local_cpu("cpu0", cpu_lanes));
        }
        if gpu_lanes > 0 {
            topology.push(RuntimeWorker::local_gpu("gpu0", gpu_lanes, gpu_options));
        }
        topology
    }

    pub fn push(&mut self, worker: RuntimeWorker) {
        self.workers.push(worker);
    }

    pub fn workers(&self) -> &[RuntimeWorker] {
        &self.workers
    }

    pub fn total_lanes(&self) -> usize {
        self.workers.iter().map(|worker| worker.lanes).sum()
    }

    pub fn attach_gemm_runtime_plan(
        &self,
        plan: &GemmRuntimePlan,
    ) -> Result<RuntimeSurrogateAttachment, ErrorReport> {
        attach_gemm_runtime_plan_to_topology(self, plan)
    }

    pub fn attach_event_runtime_plan(
        &self,
        plan: &EventRuntimePlan,
    ) -> Result<RuntimeSurrogateAttachment, ErrorReport> {
        attach_event_runtime_plan_to_topology(self, plan)
    }
}

/// Public shard plan emitted by [`DistributedRuntime`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeShardInfo {
    pub worker_id: String,
    pub node: String,
    pub backend: RuntimeBackend,
    pub start_lane: usize,
    pub lanes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSurrogateAttachment {
    pub format_version: u32,
    pub plan_schema: String,
    pub total_lanes: usize,
    pub workers: Vec<RuntimeSurrogateWorkerAttachment>,
    pub items: Vec<RuntimeSurrogateItemAttachment>,
    pub health: RuntimeSurrogateHealth,
}

impl RuntimeSurrogateAttachment {
    pub fn write_json(&self, writer: &mut impl Write) -> Result<(), ErrorReport> {
        self.validate_format_version()?;
        serde_json::to_writer_pretty(&mut *writer, self)
            .map_err(surrogate_attachment_json_error)?;
        writer
            .write_all(b"\n")
            .map_err(surrogate_attachment_io_error)?;
        writer.flush().map_err(surrogate_attachment_io_error)
    }

    pub fn read_json(reader: &mut impl Read) -> Result<Self, ErrorReport> {
        let attachment: Self =
            serde_json::from_reader(reader).map_err(surrogate_attachment_json_error)?;
        attachment.validate_format_version()?;
        Ok(attachment)
    }

    fn validate_format_version(&self) -> Result<(), ErrorReport> {
        if self.format_version != RUNTIME_SURROGATE_ATTACHMENT_FORMAT_VERSION {
            return Err(error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_VERSION",
                format!(
                    "unsupported runtime surrogate attachment format version {}, expected {}",
                    self.format_version, RUNTIME_SURROGATE_ATTACHMENT_FORMAT_VERSION
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSurrogateWorkerAttachment {
    pub worker_id: String,
    pub node: String,
    pub backend: RuntimeBackend,
    pub start_lane: usize,
    pub lanes: usize,
    pub assigned_items: usize,
    pub used_surrogate: usize,
    pub exact_fallbacks: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSurrogateItemAttachment {
    pub index: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_id: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    pub lane: usize,
    pub worker_id: String,
    pub decision: GemmPolicyDecision,
    pub source_result: GemmRuntimeSourceResult,
    pub provenance_tag: String,
    pub exact: bool,
    pub surrogate_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_ok: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_predicted: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_expected: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_max_abs_error: Option<i128>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_latency_error_cycles: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_first_divergence: Option<RuntimeSurrogateGemmDivergence>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSurrogateGemmDivergence {
    pub row: usize,
    pub col: usize,
    pub expected: i128,
    pub actual: i128,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSurrogateHealth {
    pub ready: bool,
    pub worker_count: usize,
    pub item_count: usize,
    pub used_surrogate: usize,
    pub exact_fallbacks: usize,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSurrogateExecutionReport {
    pub ready: bool,
    pub executed: bool,
    pub plan_schema: String,
    pub total_lanes: usize,
    pub attached_items: usize,
    pub surrogate_eligible_items: usize,
    pub exact_fallback_items: usize,
    pub invalid_items: usize,
    pub shadow_compared_items: usize,
    pub shadow_passed_items: usize,
    pub shadow_failed_items: usize,
    pub shadow_unavailable_items: usize,
    pub latest_action_kind: Option<RuntimeSurrogateActionKind>,
    pub action_count: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub event_workers: Vec<RuntimeSurrogateEventWorkerReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub event_lanes: Vec<RuntimeSurrogateEventLaneReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub event_items: Vec<RuntimeSurrogateEventItemReport>,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSurrogateEventWorkerReport {
    pub worker_id: String,
    pub start_lane: usize,
    pub lanes: usize,
    pub assigned_items: usize,
    pub surrogate_replacements: usize,
    pub exact_fallbacks: usize,
    pub shadow_compared: usize,
    pub shadow_failed: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSurrogateEventLaneReport {
    pub lane: usize,
    pub count: usize,
    pub surrogate_replacements: usize,
    pub exact_fallbacks: usize,
    pub shadow_compared: usize,
    pub shadow_failed: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSurrogateEventItemReport {
    pub index: usize,
    pub sample_id: Option<usize>,
    pub lane: usize,
    pub worker_id: String,
    pub target: Option<String>,
    pub decision: GemmPolicyDecision,
    pub source_result: GemmRuntimeSourceResult,
    pub predicted: Option<i64>,
    pub expected: Option<i64>,
    pub shadow_ok: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_error: Option<String>,
    pub provenance_tag: String,
    pub exact: bool,
    pub surrogate_id: String,
}

impl RuntimeSurrogateExecutionReport {
    pub fn inspect_attachment(attachment: &RuntimeSurrogateAttachment) -> Self {
        Self::from_attachment(attachment, None, 0)
    }

    fn from_attachment(
        attachment: &RuntimeSurrogateAttachment,
        latest_action_kind: Option<RuntimeSurrogateActionKind>,
        action_count: u64,
    ) -> Self {
        let attached_items = attachment.items.len();
        let surrogate_eligible_items = attachment
            .items
            .iter()
            .filter(|item| item.source_result == GemmRuntimeSourceResult::Surrogate)
            .count();
        let exact_fallback_items = attachment
            .items
            .iter()
            .filter(|item| item.source_result == GemmRuntimeSourceResult::Exact)
            .count();
        let mut diagnostics = attachment.health.diagnostics.clone();
        if !attachment.health.ready {
            diagnostics.push("surrogate attachment health is not ready".to_string());
        }
        if attachment.total_lanes == 0 {
            diagnostics.push("surrogate attachment has zero runtime lanes".to_string());
        }
        let invalid_items =
            attached_items.saturating_sub(surrogate_eligible_items + exact_fallback_items);
        let shadow_compared_items = attachment
            .items
            .iter()
            .filter(|item| item.shadow_ok.is_some())
            .count();
        let shadow_passed_items = attachment
            .items
            .iter()
            .filter(|item| item.shadow_ok == Some(true))
            .count();
        let shadow_failed_items = attachment
            .items
            .iter()
            .filter(|item| item.shadow_ok == Some(false))
            .count();
        let shadow_unavailable_items = attachment
            .items
            .iter()
            .filter(|item| {
                item.decision == GemmPolicyDecision::ShadowCompare && item.shadow_ok.is_none()
            })
            .count();
        let ready = attachment.health.ready && diagnostics.is_empty();
        let event_items = if attachment.plan_schema == EVENT_RUNTIME_PLAN_SCHEMA {
            runtime_surrogate_event_items(attachment)
        } else {
            Vec::new()
        };
        let event_workers = if attachment.plan_schema == EVENT_RUNTIME_PLAN_SCHEMA {
            runtime_surrogate_event_workers(attachment)
        } else {
            Vec::new()
        };
        let event_lanes = if attachment.plan_schema == EVENT_RUNTIME_PLAN_SCHEMA {
            runtime_surrogate_event_lanes(attachment)
        } else {
            Vec::new()
        };
        Self {
            ready,
            executed: ready && attached_items > 0,
            plan_schema: attachment.plan_schema.clone(),
            total_lanes: attachment.total_lanes,
            attached_items,
            surrogate_eligible_items,
            exact_fallback_items,
            invalid_items,
            shadow_compared_items,
            shadow_passed_items,
            shadow_failed_items,
            shadow_unavailable_items,
            latest_action_kind,
            action_count,
            event_workers,
            event_lanes,
            event_items,
            diagnostics,
        }
    }
}

fn runtime_surrogate_event_items(
    attachment: &RuntimeSurrogateAttachment,
) -> Vec<RuntimeSurrogateEventItemReport> {
    attachment
        .items
        .iter()
        .map(|item| RuntimeSurrogateEventItemReport {
            index: item.index,
            sample_id: item.sample_id,
            lane: item.lane,
            worker_id: item.worker_id.clone(),
            target: item.target.clone(),
            decision: item.decision,
            source_result: item.source_result,
            predicted: item.shadow_predicted,
            expected: item.shadow_expected,
            shadow_ok: item.shadow_ok,
            shadow_error: item.shadow_error.clone(),
            provenance_tag: item.provenance_tag.clone(),
            exact: item.exact,
            surrogate_id: item.surrogate_id.clone(),
        })
        .collect()
}

fn runtime_surrogate_event_workers(
    attachment: &RuntimeSurrogateAttachment,
) -> Vec<RuntimeSurrogateEventWorkerReport> {
    attachment
        .workers
        .iter()
        .map(|worker| {
            let items = attachment
                .items
                .iter()
                .filter(|item| item.worker_id == worker.worker_id)
                .collect::<Vec<_>>();
            RuntimeSurrogateEventWorkerReport {
                worker_id: worker.worker_id.clone(),
                start_lane: worker.start_lane,
                lanes: worker.lanes,
                assigned_items: items.len(),
                surrogate_replacements: items
                    .iter()
                    .filter(|item| item.source_result == GemmRuntimeSourceResult::Surrogate)
                    .count(),
                exact_fallbacks: items
                    .iter()
                    .filter(|item| item.source_result == GemmRuntimeSourceResult::Exact)
                    .count(),
                shadow_compared: items.iter().filter(|item| item.shadow_ok.is_some()).count(),
                shadow_failed: items
                    .iter()
                    .filter(|item| item.shadow_ok == Some(false))
                    .count(),
            }
        })
        .collect()
}

fn runtime_surrogate_event_lanes(
    attachment: &RuntimeSurrogateAttachment,
) -> Vec<RuntimeSurrogateEventLaneReport> {
    let mut lanes = attachment
        .items
        .iter()
        .map(|item| item.lane)
        .collect::<Vec<_>>();
    lanes.sort_unstable();
    lanes.dedup();
    lanes
        .into_iter()
        .map(|lane| {
            let items = attachment
                .items
                .iter()
                .filter(|item| item.lane == lane)
                .collect::<Vec<_>>();
            RuntimeSurrogateEventLaneReport {
                lane,
                count: items.len(),
                surrogate_replacements: items
                    .iter()
                    .filter(|item| item.source_result == GemmRuntimeSourceResult::Surrogate)
                    .count(),
                exact_fallbacks: items
                    .iter()
                    .filter(|item| item.source_result == GemmRuntimeSourceResult::Exact)
                    .count(),
                shadow_compared: items.iter().filter(|item| item.shadow_ok.is_some()).count(),
                shadow_failed: items
                    .iter()
                    .filter(|item| item.shadow_ok == Some(false))
                    .count(),
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeSurrogateActionKind {
    Inspect,
    EvalCombinational,
    Tick,
    TickMany,
    PartitionEvalCombinational,
    PartitionTick,
    PartitionTickMany,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    pub total_lanes: usize,
    pub program_top: String,
    pub signal_words_per_lane: usize,
    pub memory_words_per_lane: usize,
    pub shards: Vec<RuntimeShardSnapshot>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeShardSnapshot {
    pub shard: RuntimeShardInfo,
    pub values: Vec<u32>,
    pub memories: Vec<u32>,
}

pub const RUNTIME_CHECKPOINT_FORMAT_VERSION: u32 = 1;
pub const RUNTIME_TELEMETRY_FORMAT_VERSION: u32 = 1;
pub const RUNTIME_PARTITION_PLAN_FORMAT_VERSION: u32 = 1;
pub const RUNTIME_PARTITION_PLACEMENT_FORMAT_VERSION: u32 = 1;
pub const RUNTIME_PARTITION_COMMUNICATION_FORMAT_VERSION: u32 = 1;
pub const RUNTIME_PARTITION_BUNDLE_FORMAT_VERSION: u32 = 1;
pub const RUNTIME_PARTITION_LAUNCH_FORMAT_VERSION: u32 = 1;
pub const RUNTIME_PARTITION_SESSION_TELEMETRY_FORMAT_VERSION: u32 = 1;
pub const RUNTIME_SURROGATE_ATTACHMENT_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeTcpEndpoint {
    pub worker_id: String,
    pub addr: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCheckpoint {
    pub format_version: u32,
    pub module_name: String,
    pub topology: RuntimeTopology,
    pub tcp_endpoints: Vec<RuntimeTcpEndpoint>,
    pub snapshot: RuntimeSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TcpRuntimeSupervisorTelemetry {
    pub format_version: u32,
    pub module_name: String,
    pub topology: RuntimeTopology,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surrogate_attachment: Option<RuntimeSurrogateAttachment>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surrogate_execution: Option<RuntimeSurrogateExecutionReport>,
    pub endpoints: Vec<RuntimeTcpEndpoint>,
    pub processes: Vec<TcpRuntimeWorkerProcessTelemetry>,
    pub runtime_stats: RuntimeStats,
    pub runtime_health: Option<RuntimeHealthReport>,
    pub runtime_health_error: Option<ErrorReport>,
    pub latest_checkpoint: Option<TcpRuntimeSupervisorCheckpointTelemetry>,
    pub last_recovery: Option<TcpRuntimeSupervisorRecoveryReport>,
}

impl TcpRuntimeSupervisorTelemetry {
    pub fn write_json(&self, writer: &mut impl Write) -> Result<(), ErrorReport> {
        self.validate_format_version()?;
        serde_json::to_writer_pretty(&mut *writer, self).map_err(telemetry_json_error)?;
        writer.write_all(b"\n").map_err(telemetry_io_error)?;
        writer.flush().map_err(telemetry_io_error)
    }

    pub fn read_json(reader: &mut impl Read) -> Result<Self, ErrorReport> {
        let telemetry: Self = serde_json::from_reader(reader).map_err(telemetry_json_error)?;
        telemetry.validate_format_version()?;
        Ok(telemetry)
    }

    fn validate_format_version(&self) -> Result<(), ErrorReport> {
        if self.format_version != RUNTIME_TELEMETRY_FORMAT_VERSION {
            return Err(error(
                "E_RUNTIME_TELEMETRY_VERSION",
                format!(
                    "unsupported runtime telemetry format version {}, expected {}",
                    self.format_version, RUNTIME_TELEMETRY_FORMAT_VERSION
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TcpRuntimeWorkerProcessTelemetry {
    pub worker_id: String,
    pub endpoint: String,
    pub running: bool,
    pub exit: Option<TcpRuntimeWorkerProcessExit>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TcpRuntimeSupervisorCheckpointTelemetry {
    pub event: Option<RuntimeCheckpointEvent>,
    pub checkpoint: RuntimeCheckpoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionConfig {
    pub target_partitions: usize,
}

impl Default for RuntimePartitionConfig {
    fn default() -> Self {
        Self {
            target_partitions: 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionPlan {
    pub format_version: u32,
    pub module_name: String,
    pub partitions: Vec<RuntimePartition>,
    pub boundary_signals: Vec<RuntimePartitionBoundarySignal>,
    pub total_cost: RuntimePartitionCost,
    #[serde(default)]
    pub diagnostics: Vec<RuntimePartitionDiagnostic>,
    #[serde(default)]
    pub recommendations: Vec<RuntimePartitionRecommendation>,
}

impl RuntimePartitionPlan {
    pub fn write_json(&self, writer: &mut impl Write) -> Result<(), ErrorReport> {
        self.validate_format_version()?;
        serde_json::to_writer_pretty(&mut *writer, self).map_err(partition_plan_json_error)?;
        writer.write_all(b"\n").map_err(partition_plan_io_error)?;
        writer.flush().map_err(partition_plan_io_error)
    }

    pub fn read_json(reader: &mut impl Read) -> Result<Self, ErrorReport> {
        let plan: Self = serde_json::from_reader(reader).map_err(partition_plan_json_error)?;
        plan.validate_format_version()?;
        Ok(plan)
    }

    fn validate_format_version(&self) -> Result<(), ErrorReport> {
        if self.format_version != RUNTIME_PARTITION_PLAN_FORMAT_VERSION {
            return Err(error(
                "E_RUNTIME_PARTITION_PLAN_VERSION",
                format!(
                    "unsupported runtime partition plan format version {}, expected {}",
                    self.format_version, RUNTIME_PARTITION_PLAN_FORMAT_VERSION
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartition {
    pub id: String,
    pub module_name: String,
    pub instance_path: Vec<String>,
    pub external: bool,
    pub signals: Vec<RuntimePartitionSignalRef>,
    pub cost: RuntimePartitionCost,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionSignalRef {
    pub module_name: String,
    pub signal_name: String,
    pub signal: Signal,
    pub width: Width,
    pub kind: RuntimePartitionSignalKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimePartitionSignalKind {
    Input,
    Output,
    Inout,
    Wire,
    Register,
    Memory,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionBoundarySignal {
    pub signal: RuntimePartitionSignalRef,
    pub instance_path: Vec<String>,
    pub port_name: String,
    pub producer_partition: Option<String>,
    pub consumer_partitions: Vec<String>,
    pub width: Width,
    pub cost: RuntimePartitionCost,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionCost {
    pub compute_ops: u64,
    pub state_bits: u64,
    pub memory_bits: u64,
    pub boundary_bits: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionDiagnostic {
    pub code: RuntimePartitionDiagnosticCode,
    pub severity: RuntimePartitionDiagnosticSeverity,
    pub message: String,
    pub partition_id: Option<String>,
    pub boundary_index: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimePartitionDiagnosticSeverity {
    Info,
    Warning,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimePartitionDiagnosticCode {
    TargetUnderfilled,
    PartitionImbalance,
    HighBoundaryCost,
    ExternalModuleOpaque,
    EmptyPartition,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionRecommendation {
    pub code: RuntimePartitionRecommendationCode,
    pub message: String,
    pub partition_id: Option<String>,
    pub boundary_index: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimePartitionRecommendationCode {
    IncreaseHierarchy,
    SplitHeavyPartition,
    CoLocateBoundary,
    InspectExternalModule,
    ReduceTargetPartitions,
}

/// Legality of a partition boundary under the v0 one-barrier-per-cycle BSP
/// schedule. A boundary is safe to cut only if the value crossing it is stable
/// for the entire simulated cycle: a register output, or a top-level input set
/// before evaluation. A combinational crossing would create an intra-cycle
/// cross-partition data dependence, which the v0 schedule cannot satisfy with a
/// single barrier — those partitions are merged instead (see
/// [`legal_partition_merge_groups`]). v1 lifts this restriction by replicating
/// the combinational fan-in cone into the consumer partition (RepCut-style).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryLegality {
    /// The crossing value is stable for the whole cycle (register output or
    /// top-level input). Legal to cut with a single barrier per cycle.
    RegisterStable,
    /// The crossing value is combinational; cutting here would require
    /// intra-cycle communication. Illegal in v0.
    CombinationalUnsafe,
}

/// Classifies a single partition boundary for the v0 register/IO-only cut rule.
///
/// Conservative by construction: only `Register` and `Input` crossings are
/// treated as stable. `Output`/`Inout`/`Memory`/`Wire` are treated as
/// combinational because a register driver behind them is not provable from the
/// boundary alone; marking them unsafe can only over-merge (reducing
/// parallelism), never produce an incorrect schedule.
pub fn classify_partition_boundary(boundary: &RuntimePartitionBoundarySignal) -> BoundaryLegality {
    match boundary.signal.kind {
        RuntimePartitionSignalKind::Register | RuntimePartitionSignalKind::Input => {
            BoundaryLegality::RegisterStable
        }
        RuntimePartitionSignalKind::Output
        | RuntimePartitionSignalKind::Inout
        | RuntimePartitionSignalKind::Memory
        | RuntimePartitionSignalKind::Wire => BoundaryLegality::CombinationalUnsafe,
    }
}

/// A set of original plan partitions that must be co-scheduled (run on the same
/// worker in the same superstep) so that no combinational boundary crosses a
/// group edge. The unit the v0 BSP executor schedules.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegalPartitionGroup {
    /// Synthetic id: sorted member ids joined by `+`.
    pub id: String,
    /// Original partition ids in this group (sorted).
    pub members: Vec<String>,
    /// Field-wise sum of the member partitions' costs.
    pub cost: RuntimePartitionCost,
}

/// The v0-legal coarsening of a [`RuntimePartitionPlan`]: partitions joined by
/// combinational boundaries are merged into [`LegalPartitionGroup`]s, leaving
/// only register-stable boundaries crossing between groups. Those
/// `exchange_boundaries` are exactly the values moved at the per-cycle barrier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegalPartitionPlan {
    pub module_name: String,
    pub groups: Vec<LegalPartitionGroup>,
    /// Register-stable boundaries crossing group edges — the per-cycle exchange
    /// set. Every entry is guaranteed [`BoundaryLegality::RegisterStable`].
    pub exchange_boundaries: Vec<RuntimePartitionBoundarySignal>,
}

/// Union-find over partition ids: groups partitions joined by any
/// [`BoundaryLegality::CombinationalUnsafe`] boundary. Returns every group
/// (including singletons) as sorted member-id lists, themselves sorted for
/// deterministic output. Shared by [`legal_partition_merge_groups`] and
/// [`legalize_partition_plan`].
fn partition_merge_components(plan: &RuntimePartitionPlan) -> Vec<Vec<String>> {
    let index: HashMap<&str, usize> = plan
        .partitions
        .iter()
        .enumerate()
        .map(|(i, partition)| (partition.id.as_str(), i))
        .collect();
    let mut parent: Vec<usize> = (0..plan.partitions.len()).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path halving
            x = parent[x];
        }
        x
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            // Keep the smaller root for determinism.
            parent[ra.max(rb)] = ra.min(rb);
        }
    }

    for boundary in &plan.boundary_signals {
        if classify_partition_boundary(boundary) != BoundaryLegality::CombinationalUnsafe {
            continue;
        }
        let Some(producer) = boundary
            .producer_partition
            .as_deref()
            .and_then(|id| index.get(id).copied())
        else {
            continue;
        };
        for consumer in &boundary.consumer_partitions {
            if let Some(&consumer) = index.get(consumer.as_str()) {
                if consumer != producer {
                    union(&mut parent, producer, consumer);
                }
            }
        }
    }

    let mut groups: HashMap<usize, Vec<String>> = HashMap::new();
    for (i, partition) in plan.partitions.iter().enumerate() {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(partition.id.clone());
    }
    let mut result: Vec<Vec<String>> = groups
        .into_values()
        .map(|mut group| {
            group.sort();
            group
        })
        .collect();
    result.sort();
    result
}

/// Computes the groups of partitions that must be merged so that every remaining
/// cross-partition boundary in `plan` is register-stable under the v0 schedule.
///
/// Returns one sorted group per merged set containing more than one partition
/// id; already-legal singleton partitions are omitted. See
/// [`legalize_partition_plan`] for the full coarsened plan including singletons
/// and the resulting exchange-boundary set.
pub fn legal_partition_merge_groups(plan: &RuntimePartitionPlan) -> Vec<Vec<String>> {
    partition_merge_components(plan)
        .into_iter()
        .filter(|group| group.len() > 1)
        .collect()
}

fn add_partition_cost(acc: &mut RuntimePartitionCost, cost: &RuntimePartitionCost) {
    acc.compute_ops += cost.compute_ops;
    acc.state_bits += cost.state_bits;
    acc.memory_bits += cost.memory_bits;
    acc.boundary_bits += cost.boundary_bits;
}

/// Coarsens `plan` into its v0-legal form: merges combinationally-joined
/// partitions into [`LegalPartitionGroup`]s and retains only the register-stable
/// boundaries that cross between groups as the per-cycle exchange set.
///
/// Boundaries internal to a group (both endpoints in the same group) are
/// dropped — they are evaluated locally. Top-level boundaries (no in-partition
/// producer) are not inter-group exchanges and are excluded. The resulting
/// `exchange_boundaries` are guaranteed register-stable by construction.
pub fn legalize_partition_plan(plan: &RuntimePartitionPlan) -> LegalPartitionPlan {
    let components = partition_merge_components(plan);

    // Map each original partition id -> its group index.
    let mut group_of: HashMap<&str, usize> = HashMap::new();
    for (g, members) in components.iter().enumerate() {
        for member in members {
            group_of.insert(member.as_str(), g);
        }
    }

    let cost_of: HashMap<&str, &RuntimePartitionCost> = plan
        .partitions
        .iter()
        .map(|partition| (partition.id.as_str(), &partition.cost))
        .collect();

    let groups: Vec<LegalPartitionGroup> = components
        .iter()
        .map(|members| {
            let mut cost = RuntimePartitionCost::default();
            for member in members {
                if let Some(member_cost) = cost_of.get(member.as_str()) {
                    add_partition_cost(&mut cost, member_cost);
                }
            }
            LegalPartitionGroup {
                id: members.join("+"),
                members: members.clone(),
                cost,
            }
        })
        .collect();

    let exchange_boundaries: Vec<RuntimePartitionBoundarySignal> = plan
        .boundary_signals
        .iter()
        .filter(|boundary| {
            // Must have an in-partition producer (top-level ports are external).
            let Some(producer_group) = boundary
                .producer_partition
                .as_deref()
                .and_then(|id| group_of.get(id).copied())
            else {
                return false;
            };
            // Keep only boundaries that actually cross a group edge.
            boundary
                .consumer_partitions
                .iter()
                .any(|consumer| group_of.get(consumer.as_str()).copied() != Some(producer_group))
        })
        .cloned()
        .collect();

    // By construction every cross-group boundary is register-stable: any
    // combinational boundary would have merged its endpoints into one group.
    debug_assert!(exchange_boundaries
        .iter()
        .all(|b| classify_partition_boundary(b) == BoundaryLegality::RegisterStable));

    LegalPartitionPlan {
        module_name: plan.module_name.clone(),
        groups,
        exchange_boundaries,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionPlacementPlan {
    pub format_version: u32,
    pub module_name: String,
    pub topology: RuntimeTopology,
    pub assignments: Vec<RuntimePartitionAssignment>,
    pub worker_summaries: Vec<RuntimePartitionWorkerSummary>,
    pub total_cost: RuntimePartitionCost,
    pub diagnostics: Vec<RuntimePartitionPlacementDiagnostic>,
    pub recommendations: Vec<RuntimePartitionPlacementRecommendation>,
}

impl RuntimePartitionPlacementPlan {
    pub fn write_json(&self, writer: &mut impl Write) -> Result<(), ErrorReport> {
        self.validate_format_version()?;
        serde_json::to_writer_pretty(&mut *writer, self).map_err(partition_placement_json_error)?;
        writer
            .write_all(b"\n")
            .map_err(partition_placement_io_error)?;
        writer.flush().map_err(partition_placement_io_error)
    }

    pub fn read_json(reader: &mut impl Read) -> Result<Self, ErrorReport> {
        let plan: Self = serde_json::from_reader(reader).map_err(partition_placement_json_error)?;
        plan.validate_format_version()?;
        Ok(plan)
    }

    fn validate_format_version(&self) -> Result<(), ErrorReport> {
        if self.format_version != RUNTIME_PARTITION_PLACEMENT_FORMAT_VERSION {
            return Err(error(
                "E_RUNTIME_PARTITION_PLACEMENT_VERSION",
                format!(
                    "unsupported runtime partition placement format version {}, expected {}",
                    self.format_version, RUNTIME_PARTITION_PLACEMENT_FORMAT_VERSION
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionAssignment {
    pub partition_id: String,
    pub worker_id: String,
    pub backend: RuntimeBackend,
    pub node: String,
    pub instance_path: Vec<String>,
    pub cost: RuntimePartitionCost,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionWorkerSummary {
    pub worker_id: String,
    pub backend: RuntimeBackend,
    pub node: String,
    pub partitions: Vec<String>,
    pub cost: RuntimePartitionCost,
    pub inbound_boundary_bits: u64,
    pub outbound_boundary_bits: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionPlacementDiagnostic {
    pub code: RuntimePartitionPlacementDiagnosticCode,
    pub severity: RuntimePartitionDiagnosticSeverity,
    pub message: String,
    pub worker_id: Option<String>,
    pub partition_id: Option<String>,
    pub boundary_index: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimePartitionPlacementDiagnosticCode {
    UnderProvisionedWorkers,
    WorkerImbalance,
    HighCrossWorkerBoundaryTraffic,
    CpuFallback,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionPlacementRecommendation {
    pub code: RuntimePartitionPlacementRecommendationCode,
    pub message: String,
    pub worker_id: Option<String>,
    pub partition_id: Option<String>,
    pub boundary_index: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimePartitionPlacementRecommendationCode {
    AddWorkers,
    MovePartition,
    CoLocatePartitions,
    AddCpuWorker,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionCommunicationPlan {
    pub format_version: u32,
    pub module_name: String,
    pub routes: Vec<RuntimePartitionRoute>,
    pub worker_summaries: Vec<RuntimePartitionCommunicationWorkerSummary>,
    pub total_cross_worker_boundary_bits: u64,
    pub diagnostics: Vec<RuntimePartitionCommunicationDiagnostic>,
    pub recommendations: Vec<RuntimePartitionCommunicationRecommendation>,
}

impl RuntimePartitionCommunicationPlan {
    pub fn write_json(&self, writer: &mut impl Write) -> Result<(), ErrorReport> {
        self.validate_format_version()?;
        serde_json::to_writer_pretty(&mut *writer, self)
            .map_err(partition_communication_json_error)?;
        writer
            .write_all(b"\n")
            .map_err(partition_communication_io_error)?;
        writer.flush().map_err(partition_communication_io_error)
    }

    pub fn read_json(reader: &mut impl Read) -> Result<Self, ErrorReport> {
        let plan: Self =
            serde_json::from_reader(reader).map_err(partition_communication_json_error)?;
        plan.validate_format_version()?;
        Ok(plan)
    }

    fn validate_format_version(&self) -> Result<(), ErrorReport> {
        if self.format_version != RUNTIME_PARTITION_COMMUNICATION_FORMAT_VERSION {
            return Err(error(
                "E_RUNTIME_PARTITION_COMMUNICATION_VERSION",
                format!(
                    "unsupported runtime partition communication format version {}, expected {}",
                    self.format_version, RUNTIME_PARTITION_COMMUNICATION_FORMAT_VERSION
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionBundle {
    pub format_version: u32,
    pub module_name: String,
    pub partition_plan: RuntimePartitionPlan,
    pub placement_plan: RuntimePartitionPlacementPlan,
    pub communication_plan: RuntimePartitionCommunicationPlan,
    #[serde(default)]
    pub diagnostics: Vec<RuntimePartitionBundleDiagnostic>,
    #[serde(default)]
    pub recommendations: Vec<RuntimePartitionBundleRecommendation>,
}

impl RuntimePartitionBundle {
    pub fn from_plans(
        partition_plan: RuntimePartitionPlan,
        placement_plan: RuntimePartitionPlacementPlan,
        communication_plan: RuntimePartitionCommunicationPlan,
    ) -> Result<Self, ErrorReport> {
        partition_plan.validate_format_version()?;
        placement_plan.validate_format_version()?;
        communication_plan.validate_format_version()?;
        validate_partition_bundle_modules(&partition_plan, &placement_plan, &communication_plan)?;
        validate_partition_bundle_routes(&placement_plan, &communication_plan)?;

        let diagnostics = runtime_partition_bundle_diagnostics(
            &partition_plan,
            &placement_plan,
            &communication_plan,
        );
        let recommendations = runtime_partition_bundle_recommendations(
            &partition_plan,
            &placement_plan,
            &communication_plan,
        );
        let module_name = partition_plan.module_name.clone();

        Ok(Self {
            format_version: RUNTIME_PARTITION_BUNDLE_FORMAT_VERSION,
            module_name,
            partition_plan,
            placement_plan,
            communication_plan,
            diagnostics,
            recommendations,
        })
    }

    pub fn write_json(&self, writer: &mut impl Write) -> Result<(), ErrorReport> {
        self.validate_format_version()?;
        self.partition_plan.validate_format_version()?;
        self.placement_plan.validate_format_version()?;
        self.communication_plan.validate_format_version()?;
        validate_partition_bundle_modules(
            &self.partition_plan,
            &self.placement_plan,
            &self.communication_plan,
        )?;
        validate_partition_bundle_routes(&self.placement_plan, &self.communication_plan)?;
        serde_json::to_writer_pretty(&mut *writer, self).map_err(partition_bundle_json_error)?;
        writer.write_all(b"\n").map_err(partition_bundle_io_error)?;
        writer.flush().map_err(partition_bundle_io_error)
    }

    pub fn read_json(reader: &mut impl Read) -> Result<Self, ErrorReport> {
        let bundle: Self = serde_json::from_reader(reader).map_err(partition_bundle_json_error)?;
        bundle.validate_format_version()?;
        bundle.partition_plan.validate_format_version()?;
        bundle.placement_plan.validate_format_version()?;
        bundle.communication_plan.validate_format_version()?;
        validate_partition_bundle_modules(
            &bundle.partition_plan,
            &bundle.placement_plan,
            &bundle.communication_plan,
        )?;
        validate_partition_bundle_routes(&bundle.placement_plan, &bundle.communication_plan)?;
        Ok(bundle)
    }

    fn validate_format_version(&self) -> Result<(), ErrorReport> {
        if self.format_version != RUNTIME_PARTITION_BUNDLE_FORMAT_VERSION {
            return Err(error(
                "E_RUNTIME_PARTITION_BUNDLE_VERSION",
                format!(
                    "unsupported runtime partition bundle format version {}, expected {}",
                    self.format_version, RUNTIME_PARTITION_BUNDLE_FORMAT_VERSION
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionBundleDiagnostic {
    pub source: RuntimePartitionBundleSource,
    pub code: String,
    pub severity: RuntimePartitionDiagnosticSeverity,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionBundleRecommendation {
    pub source: RuntimePartitionBundleSource,
    pub code: String,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimePartitionBundleSource {
    Partition,
    Placement,
    Communication,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionLaunchPlan {
    pub format_version: u32,
    pub module_name: String,
    pub workers: Vec<RuntimePartitionWorkerLaunch>,
    pub routes: Vec<RuntimePartitionLaunchRoute>,
    #[serde(default)]
    pub diagnostics: Vec<RuntimePartitionLaunchDiagnostic>,
}

impl RuntimePartitionLaunchPlan {
    pub fn write_json(&self, writer: &mut impl Write) -> Result<(), ErrorReport> {
        self.validate_format_version()?;
        serde_json::to_writer_pretty(&mut *writer, self).map_err(partition_launch_json_error)?;
        writer.write_all(b"\n").map_err(partition_launch_io_error)?;
        writer.flush().map_err(partition_launch_io_error)
    }

    pub fn read_json(reader: &mut impl Read) -> Result<Self, ErrorReport> {
        let plan: Self = serde_json::from_reader(reader).map_err(partition_launch_json_error)?;
        plan.validate_format_version()?;
        Ok(plan)
    }

    fn validate_format_version(&self) -> Result<(), ErrorReport> {
        if self.format_version != RUNTIME_PARTITION_LAUNCH_FORMAT_VERSION {
            return Err(error(
                "E_RUNTIME_PARTITION_LAUNCH_VERSION",
                format!(
                    "unsupported runtime partition launch format version {}, expected {}",
                    self.format_version, RUNTIME_PARTITION_LAUNCH_FORMAT_VERSION
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionDeploymentReport {
    pub module_name: String,
    pub workers: Vec<RuntimePartitionWorkerHealth>,
    pub diagnostics: Vec<RuntimePartitionLaunchDiagnostic>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionSessionHealth {
    pub module_name: String,
    pub ready: bool,
    pub worker_count: usize,
    pub initialized_worker_count: usize,
    pub partition_count: usize,
    pub route_count: usize,
    pub diagnostics: Vec<RuntimePartitionLaunchDiagnostic>,
    pub workers: Vec<RuntimePartitionWorkerHealth>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionSessionTelemetry {
    pub format_version: u32,
    pub module_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surrogate_attachment: Option<RuntimeSurrogateAttachment>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surrogate_execution: Option<RuntimeSurrogateExecutionReport>,
    pub launch: RuntimePartitionLaunchPlan,
    pub deployment: RuntimePartitionDeploymentReport,
    pub health: RuntimePartitionSessionHealth,
    pub action_summary: RuntimePartitionSessionActionSummary,
    pub route_mailboxes: Vec<RuntimePartitionRouteMailboxReport>,
    pub actions: Vec<RuntimePartitionSessionActionReport>,
}

impl RuntimePartitionSessionTelemetry {
    pub fn write_json(&self, writer: &mut impl Write) -> Result<(), ErrorReport> {
        self.validate_format_version()?;
        serde_json::to_writer_pretty(&mut *writer, self)
            .map_err(partition_session_telemetry_json_error)?;
        writer
            .write_all(b"\n")
            .map_err(partition_session_telemetry_io_error)?;
        writer.flush().map_err(partition_session_telemetry_io_error)
    }

    pub fn read_json(reader: &mut impl Read) -> Result<Self, ErrorReport> {
        let telemetry: Self =
            serde_json::from_reader(reader).map_err(partition_session_telemetry_json_error)?;
        telemetry.validate_format_version()?;
        Ok(telemetry)
    }

    fn validate_format_version(&self) -> Result<(), ErrorReport> {
        if self.format_version != RUNTIME_PARTITION_SESSION_TELEMETRY_FORMAT_VERSION {
            return Err(error(
                "E_RUNTIME_PARTITION_SESSION_TELEMETRY_VERSION",
                format!(
                    "unsupported runtime partition session telemetry format version {}, expected {}",
                    self.format_version, RUNTIME_PARTITION_SESSION_TELEMETRY_FORMAT_VERSION
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionSessionActionReport {
    pub module_name: String,
    pub kind: RuntimePartitionWorkerActionKind,
    pub workers: Vec<RuntimePartitionWorkerActionReport>,
    pub operations: RuntimeOperationStats,
    pub diagnostics: Vec<RuntimePartitionLaunchDiagnostic>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionSessionActionSummary {
    pub module_name: String,
    pub action_count: usize,
    pub latest_action_kind: Option<RuntimePartitionWorkerActionKind>,
    pub operations: RuntimeOperationStats,
    pub worker_count: usize,
    pub partition_count: usize,
    pub route_count: usize,
    pub diagnostics: Vec<RuntimePartitionLaunchDiagnostic>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionSessionRunScript {
    pub actions: Vec<RuntimePartitionWorkerActionKind>,
    pub every_actions: usize,
    pub emit_initial: bool,
    pub emit_final: bool,
}

impl RuntimePartitionSessionRunScript {
    pub fn every_actions(
        actions: Vec<RuntimePartitionWorkerActionKind>,
        every_actions: usize,
    ) -> Self {
        Self {
            actions,
            every_actions,
            emit_initial: false,
            emit_final: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionSessionRunEvent {
    pub completed_actions: usize,
    pub reason: RuntimePartitionSessionRunEventReason,
    pub latest_action_kind: Option<RuntimePartitionWorkerActionKind>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimePartitionSessionRunEventReason {
    Initial,
    Cadence,
    Final,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionSessionRunReport {
    pub completed_actions: usize,
    pub telemetry_emitted: usize,
    pub action_summary: RuntimePartitionSessionActionSummary,
    pub diagnostics: Vec<RuntimePartitionLaunchDiagnostic>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimePartitionSession {
    launch: RuntimePartitionLaunchPlan,
    deployment: RuntimePartitionDeploymentReport,
    route_mailboxes: Vec<RuntimePartitionRouteMailboxReport>,
    actions: Vec<RuntimePartitionSessionActionReport>,
    surrogate_attachment: Option<RuntimeSurrogateAttachment>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionWorkerLaunch {
    pub worker_id: String,
    pub backend: RuntimeBackend,
    pub node: String,
    pub partitions: Vec<RuntimePartitionLaunchPartition>,
    pub outbound_routes: Vec<usize>,
    pub inbound_routes: Vec<usize>,
    pub outbound_route_specs: Vec<RuntimePartitionLaunchRoute>,
    pub inbound_route_specs: Vec<RuntimePartitionLaunchRoute>,
    pub cost: RuntimePartitionCost,
    pub outbound_bits: u64,
    pub inbound_bits: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionLaunchPartition {
    pub partition_id: String,
    pub module_name: String,
    pub instance_path: Vec<String>,
    pub external: bool,
    pub cost: RuntimePartitionCost,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionLaunchRoute {
    pub route_index: usize,
    pub boundary_index: usize,
    pub signal: RuntimePartitionSignalRef,
    pub port_name: String,
    pub instance_path: Vec<String>,
    pub producer_partition: String,
    pub producer_worker: String,
    pub consumer_partition: String,
    pub consumer_worker: String,
    pub width: Width,
    pub bits_per_transfer: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionLaunchDiagnostic {
    pub source: RuntimePartitionLaunchDiagnosticSource,
    pub code: String,
    pub severity: RuntimePartitionDiagnosticSeverity,
    pub message: String,
    pub worker_id: Option<String>,
    pub partition_id: Option<String>,
    pub route_index: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimePartitionLaunchDiagnosticSource {
    BundlePartition,
    BundlePlacement,
    BundleCommunication,
    Launch,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionRoute {
    pub boundary_index: usize,
    pub signal: RuntimePartitionSignalRef,
    pub port_name: String,
    pub instance_path: Vec<String>,
    pub producer_partition: String,
    pub producer_worker: String,
    pub consumer_partition: String,
    pub consumer_worker: String,
    pub width: Width,
    pub bits_per_transfer: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionCommunicationWorkerSummary {
    pub worker_id: String,
    pub outbound_routes: Vec<usize>,
    pub inbound_routes: Vec<usize>,
    pub outbound_bits: u64,
    pub inbound_bits: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionCommunicationDiagnostic {
    pub code: RuntimePartitionCommunicationDiagnosticCode,
    pub severity: RuntimePartitionDiagnosticSeverity,
    pub message: String,
    pub worker_id: Option<String>,
    pub partition_id: Option<String>,
    pub boundary_index: Option<usize>,
    pub route_index: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimePartitionCommunicationDiagnosticCode {
    HighRouteWidth,
    HighTotalCrossWorkerTraffic,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionCommunicationRecommendation {
    pub code: RuntimePartitionCommunicationRecommendationCode,
    pub message: String,
    pub worker_id: Option<String>,
    pub partition_id: Option<String>,
    pub boundary_index: Option<usize>,
    pub route_index: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimePartitionCommunicationRecommendationCode {
    CoLocateRoute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCheckpointCadence {
    pub every_steps: usize,
    pub include_initial: bool,
    pub include_final: bool,
}

impl RuntimeCheckpointCadence {
    pub fn every_steps(every_steps: usize) -> Self {
        Self {
            every_steps,
            include_initial: false,
            include_final: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCheckpointEvent {
    pub completed_steps: usize,
    pub reason: RuntimeCheckpointReason,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeCheckpointReason {
    Initial,
    Cadence,
    Final,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCheckpointRunReport {
    pub requested_steps: usize,
    pub completed_steps: usize,
    pub checkpoints_emitted: usize,
}

impl RuntimeCheckpoint {
    pub fn tcp_endpoint_map(&self) -> Result<HashMap<String, SocketAddr>, ErrorReport> {
        let mut endpoints = HashMap::new();
        for endpoint in &self.tcp_endpoints {
            let addr = endpoint.addr.parse::<SocketAddr>().map_err(|err| {
                error(
                    "E_RUNTIME_CHECKPOINT_ENDPOINT",
                    format!(
                        "checkpoint TCP endpoint for worker `{}` has invalid address `{}`: {err}",
                        endpoint.worker_id, endpoint.addr
                    ),
                )
            })?;
            if endpoints.insert(endpoint.worker_id.clone(), addr).is_some() {
                return Err(error(
                    "E_RUNTIME_CHECKPOINT_ENDPOINT",
                    format!(
                        "checkpoint has duplicate TCP endpoint for worker `{}`",
                        endpoint.worker_id
                    ),
                ));
            }
        }
        Ok(endpoints)
    }

    pub fn write_json(&self, writer: &mut impl Write) -> Result<(), ErrorReport> {
        self.validate_format_version()?;
        serde_json::to_writer_pretty(&mut *writer, self).map_err(checkpoint_json_error)?;
        writer.write_all(b"\n").map_err(checkpoint_io_error)?;
        writer.flush().map_err(checkpoint_io_error)
    }

    pub fn read_json(reader: &mut impl Read) -> Result<Self, ErrorReport> {
        let checkpoint: Self = serde_json::from_reader(reader).map_err(checkpoint_json_error)?;
        checkpoint.validate_format_version()?;
        Ok(checkpoint)
    }

    fn validate_format_version(&self) -> Result<(), ErrorReport> {
        if self.format_version != RUNTIME_CHECKPOINT_FORMAT_VERSION {
            return Err(error(
                "E_RUNTIME_CHECKPOINT_VERSION",
                format!(
                    "unsupported runtime checkpoint format version {}, expected {}",
                    self.format_version, RUNTIME_CHECKPOINT_FORMAT_VERSION
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeShardInit {
    pub design: CompiledDesign,
    pub module_name: String,
    pub shard: RuntimeShardInfo,
    pub backend: RuntimeBackend,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionWorkerInit {
    pub worker_id: String,
    pub launch: RuntimePartitionWorkerLaunch,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionWorkerAction {
    pub worker_id: String,
    pub kind: RuntimePartitionWorkerActionKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimePartitionWorkerActionKind {
    EvalCombinational,
    Tick,
    TickMany(usize),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionWorkerActionReport {
    pub worker_id: String,
    pub kind: RuntimePartitionWorkerActionKind,
    pub partition_count: usize,
    pub outbound_route_count: usize,
    pub inbound_route_count: usize,
    pub outbound_bits: u64,
    pub inbound_bits: u64,
    pub operations: RuntimeOperationStats,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionRoutePayload {
    pub route_index: usize,
    pub width: Width,
    pub limbs: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionRouteMailboxReport {
    pub worker_id: String,
    pub outbound_payload_count: usize,
    pub inbound_payload_count: usize,
    pub outbound_routes: Vec<usize>,
    pub inbound_routes: Vec<usize>,
    pub diagnostics: Vec<RuntimePartitionLaunchDiagnostic>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionRouteTransferReport {
    pub route_index: usize,
    pub producer_worker: String,
    pub consumer_worker: String,
    pub width: Width,
    pub limb_count: usize,
    pub producer_mailbox: RuntimePartitionRouteMailboxReport,
    pub consumer_mailbox: RuntimePartitionRouteMailboxReport,
    pub diagnostics: Vec<RuntimePartitionLaunchDiagnostic>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeWorkerRequest {
    InitShard(RuntimeShardInit),
    InitPartitionWorker(RuntimePartitionWorkerInit),
    RunPartitionWorkerAction(RuntimePartitionWorkerAction),
    StorePartitionRouteOutbound {
        worker_id: String,
        payload: RuntimePartitionRoutePayload,
    },
    DeliverPartitionRouteInbound {
        worker_id: String,
        payload: RuntimePartitionRoutePayload,
    },
    PartitionRouteMailbox {
        worker_id: String,
    },
    ClearPartitionRouteMailboxes {
        worker_id: String,
    },
    Health {
        worker_id: String,
    },
    PartitionWorkerHealth {
        worker_id: String,
    },
    SetInput {
        worker_id: String,
        signal: Signal,
        lane_values: Vec<u128>,
    },
    SetInputLimbs {
        worker_id: String,
        signal: Signal,
        lane_values: Vec<Vec<u32>>,
    },
    SetMemoryLimbs {
        worker_id: String,
        memory: Signal,
        lane_words: Vec<Vec<Vec<u32>>>,
    },
    EvalCombinational {
        worker_id: String,
    },
    Tick {
        worker_id: String,
    },
    TickMany {
        worker_id: String,
        steps: usize,
    },
    GetSignalLimbs {
        worker_id: String,
        signal: Signal,
    },
    GetMemoryLimbs {
        worker_id: String,
        memory: Signal,
    },
    Snapshot {
        worker_id: String,
    },
    RestoreSnapshot {
        worker_id: String,
        snapshot: RuntimeShardSnapshot,
    },
    Stats {
        worker_id: String,
    },
    ResetStats {
        worker_id: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeWorkerResponse {
    Ack,
    Health(RuntimeWorkerHealth),
    PartitionWorkerHealth(RuntimePartitionWorkerHealth),
    PartitionWorkerAction(RuntimePartitionWorkerActionReport),
    PartitionRouteMailbox(RuntimePartitionRouteMailboxReport),
    SignalLimbs(Vec<Vec<u32>>),
    MemoryLimbs(Vec<Vec<Vec<u32>>>),
    Snapshot(RuntimeShardSnapshot),
    Stats(RuntimeShardStats),
}

pub trait RuntimeWorkerService {
    fn handle(
        &mut self,
        request: RuntimeWorkerRequest,
    ) -> Result<RuntimeWorkerResponse, ErrorReport>;
}

pub trait RuntimeShardClient {
    fn worker_id(&self) -> &str;
    fn request(
        &mut self,
        request: RuntimeWorkerRequest,
    ) -> Result<RuntimeWorkerResponse, ErrorReport>;

    fn health(&mut self) -> Result<RuntimeWorkerHealth, ErrorReport> {
        let worker_id = self.worker_id().to_string();
        match self.request(RuntimeWorkerRequest::Health { worker_id })? {
            RuntimeWorkerResponse::Health(health) => Ok(health),
            response => Err(unexpected_worker_response("Health", response)),
        }
    }
}

#[derive(Default)]
pub struct LocalRuntimeWorkerService {
    shards: HashMap<String, RuntimeShard>,
    partition_workers: HashMap<String, RuntimePartitionWorkerState>,
}

struct RuntimePartitionWorkerState {
    launch: RuntimePartitionWorkerLaunch,
    operations: RuntimeOperationStats,
    outbound_route_payloads: HashMap<usize, RuntimePartitionRoutePayload>,
    inbound_route_payloads: HashMap<usize, RuntimePartitionRoutePayload>,
}

impl LocalRuntimeWorkerService {
    pub fn new() -> Self {
        Self::default()
    }

    fn init_shard(&mut self, init: RuntimeShardInit) -> Result<RuntimeWorkerResponse, ErrorReport> {
        if init.shard.lanes == 0 {
            return Err(error(
                "E_RUNTIME_WORKER_INIT",
                "worker shard must own at least one lane",
            ));
        }
        if self.shards.contains_key(&init.shard.worker_id) {
            return Err(error(
                "E_RUNTIME_WORKER_INIT",
                format!(
                    "worker shard `{}` is already initialized",
                    init.shard.worker_id
                ),
            ));
        }

        let program = lower_to_packed_program(&init.design, &init.module_name)?;
        let engine = match init.backend {
            RuntimeBackend::ScalarCpu => {
                if init.shard.lanes != 1 {
                    return Err(error(
                        "E_RUNTIME_BACKEND_LANES",
                        "scalar CPU backend supports exactly one lane",
                    ));
                }
                RuntimeEngine::ScalarCpu(SingleLaneMachineSimulator::new(program)?)
            }
            RuntimeBackend::PackedCpu => {
                RuntimeEngine::PackedCpu(PackedSimulator::new(program, init.shard.lanes)?)
            }
            RuntimeBackend::SimdCpu => {
                RuntimeEngine::SimdCpu(SimdCpuSimulator::new(program, init.shard.lanes)?)
            }
            RuntimeBackend::JitCpu => {
                RuntimeEngine::JitCpu(JitCpuSimulator::new(program, init.shard.lanes)?)
            }
            RuntimeBackend::Gpu(options) => {
                RuntimeEngine::Gpu(GpuBatchSimulator::new_from_compiled_with_options(
                    &init.design,
                    &init.module_name,
                    init.shard.lanes,
                    options,
                )?)
            }
        };

        self.shards.insert(
            init.shard.worker_id.clone(),
            RuntimeShard {
                info: init.shard,
                engine,
                operations: RuntimeOperationStats::default(),
            },
        );
        Ok(RuntimeWorkerResponse::Ack)
    }

    fn init_partition_worker(
        &mut self,
        init: RuntimePartitionWorkerInit,
    ) -> Result<RuntimeWorkerResponse, ErrorReport> {
        if init.worker_id != init.launch.worker_id {
            return Err(error(
                "E_RUNTIME_PARTITION_WORKER_INIT",
                format!(
                    "partition worker init id `{}` does not match launch worker `{}`",
                    init.worker_id, init.launch.worker_id
                ),
            ));
        }
        if self.partition_workers.contains_key(&init.worker_id) {
            return Err(error(
                "E_RUNTIME_PARTITION_WORKER_INIT",
                format!(
                    "partition worker `{}` is already initialized",
                    init.worker_id
                ),
            ));
        }

        self.partition_workers.insert(
            init.worker_id,
            RuntimePartitionWorkerState {
                launch: init.launch,
                operations: RuntimeOperationStats::default(),
                outbound_route_payloads: HashMap::new(),
                inbound_route_payloads: HashMap::new(),
            },
        );
        Ok(RuntimeWorkerResponse::Ack)
    }

    fn partition_worker_health(&self, worker_id: String) -> RuntimePartitionWorkerHealth {
        match self.partition_workers.get(&worker_id) {
            Some(state) => RuntimePartitionWorkerHealth {
                worker_id,
                initialized: true,
                backend: Some(state.launch.backend),
                node: Some(state.launch.node.clone()),
                partitions: state.launch.partitions.clone(),
                outbound_routes: state.launch.outbound_routes.clone(),
                inbound_routes: state.launch.inbound_routes.clone(),
                outbound_bits: state.launch.outbound_bits,
                inbound_bits: state.launch.inbound_bits,
                diagnostics: Vec::new(),
            },
            None => RuntimePartitionWorkerHealth {
                worker_id,
                initialized: false,
                backend: None,
                node: None,
                partitions: Vec::new(),
                outbound_routes: Vec::new(),
                inbound_routes: Vec::new(),
                outbound_bits: 0,
                inbound_bits: 0,
                diagnostics: Vec::new(),
            },
        }
    }

    fn run_partition_worker_action(
        &mut self,
        action: RuntimePartitionWorkerAction,
    ) -> Result<RuntimeWorkerResponse, ErrorReport> {
        let state = self
            .partition_workers
            .get_mut(&action.worker_id)
            .ok_or_else(|| {
                error(
                    "E_RUNTIME_PARTITION_WORKER_ACTION",
                    format!("partition worker `{}` is not initialized", action.worker_id),
                )
            })?;
        record_partition_worker_action(&mut state.operations, action.kind);
        Ok(RuntimeWorkerResponse::PartitionWorkerAction(
            RuntimePartitionWorkerActionReport {
                worker_id: action.worker_id,
                kind: action.kind,
                partition_count: state.launch.partitions.len(),
                outbound_route_count: state.launch.outbound_routes.len(),
                inbound_route_count: state.launch.inbound_routes.len(),
                outbound_bits: state.launch.outbound_bits,
                inbound_bits: state.launch.inbound_bits,
                operations: state.operations,
            },
        ))
    }

    fn store_partition_route_outbound(
        &mut self,
        worker_id: String,
        payload: RuntimePartitionRoutePayload,
    ) -> Result<RuntimeWorkerResponse, ErrorReport> {
        let state = self.partition_worker_state_mut(&worker_id)?;
        validate_partition_route_payload(
            &worker_id,
            &payload,
            &state.launch.outbound_route_specs,
            "outbound",
        )?;
        state
            .outbound_route_payloads
            .insert(payload.route_index, payload);
        Ok(RuntimeWorkerResponse::PartitionRouteMailbox(
            partition_route_mailbox_report(&worker_id, state),
        ))
    }

    fn deliver_partition_route_inbound(
        &mut self,
        worker_id: String,
        payload: RuntimePartitionRoutePayload,
    ) -> Result<RuntimeWorkerResponse, ErrorReport> {
        let state = self.partition_worker_state_mut(&worker_id)?;
        validate_partition_route_payload(
            &worker_id,
            &payload,
            &state.launch.inbound_route_specs,
            "inbound",
        )?;
        state
            .inbound_route_payloads
            .insert(payload.route_index, payload);
        Ok(RuntimeWorkerResponse::PartitionRouteMailbox(
            partition_route_mailbox_report(&worker_id, state),
        ))
    }

    fn partition_route_mailbox(
        &self,
        worker_id: String,
    ) -> Result<RuntimeWorkerResponse, ErrorReport> {
        let state = self.partition_worker_state(&worker_id)?;
        Ok(RuntimeWorkerResponse::PartitionRouteMailbox(
            partition_route_mailbox_report(&worker_id, state),
        ))
    }

    fn clear_partition_route_mailboxes(
        &mut self,
        worker_id: String,
    ) -> Result<RuntimeWorkerResponse, ErrorReport> {
        let state = self.partition_worker_state_mut(&worker_id)?;
        state.outbound_route_payloads.clear();
        state.inbound_route_payloads.clear();
        Ok(RuntimeWorkerResponse::PartitionRouteMailbox(
            partition_route_mailbox_report(&worker_id, state),
        ))
    }

    fn partition_worker_state(
        &self,
        worker_id: &str,
    ) -> Result<&RuntimePartitionWorkerState, ErrorReport> {
        self.partition_workers.get(worker_id).ok_or_else(|| {
            error(
                "E_RUNTIME_PARTITION_WORKER_MAILBOX",
                format!("partition worker `{worker_id}` is not initialized"),
            )
        })
    }

    fn partition_worker_state_mut(
        &mut self,
        worker_id: &str,
    ) -> Result<&mut RuntimePartitionWorkerState, ErrorReport> {
        self.partition_workers.get_mut(worker_id).ok_or_else(|| {
            error(
                "E_RUNTIME_PARTITION_WORKER_MAILBOX",
                format!("partition worker `{worker_id}` is not initialized"),
            )
        })
    }

    fn shard_mut(&mut self, worker_id: &str) -> Result<&mut RuntimeShard, ErrorReport> {
        self.shards.get_mut(worker_id).ok_or_else(|| {
            error(
                "E_RUNTIME_WORKER_SHARD",
                format!("worker shard `{worker_id}` is not initialized"),
            )
        })
    }
}

impl RuntimeWorkerService for LocalRuntimeWorkerService {
    fn handle(
        &mut self,
        request: RuntimeWorkerRequest,
    ) -> Result<RuntimeWorkerResponse, ErrorReport> {
        match request {
            RuntimeWorkerRequest::InitShard(init) => self.init_shard(init),
            RuntimeWorkerRequest::InitPartitionWorker(init) => self.init_partition_worker(init),
            RuntimeWorkerRequest::RunPartitionWorkerAction(action) => {
                self.run_partition_worker_action(action)
            }
            RuntimeWorkerRequest::StorePartitionRouteOutbound { worker_id, payload } => {
                self.store_partition_route_outbound(worker_id, payload)
            }
            RuntimeWorkerRequest::DeliverPartitionRouteInbound { worker_id, payload } => {
                self.deliver_partition_route_inbound(worker_id, payload)
            }
            RuntimeWorkerRequest::PartitionRouteMailbox { worker_id } => {
                self.partition_route_mailbox(worker_id)
            }
            RuntimeWorkerRequest::ClearPartitionRouteMailboxes { worker_id } => {
                self.clear_partition_route_mailboxes(worker_id)
            }
            RuntimeWorkerRequest::Health { worker_id } => {
                let health = self
                    .shards
                    .get(&worker_id)
                    .map(|shard| RuntimeWorkerHealth {
                        worker_id: worker_id.clone(),
                        initialized: true,
                        shard: Some(shard.info.clone()),
                        operations: Some(shard.operations),
                    })
                    .unwrap_or(RuntimeWorkerHealth {
                        worker_id,
                        initialized: false,
                        shard: None,
                        operations: None,
                    });
                Ok(RuntimeWorkerResponse::Health(health))
            }
            RuntimeWorkerRequest::PartitionWorkerHealth { worker_id } => {
                Ok(RuntimeWorkerResponse::PartitionWorkerHealth(
                    self.partition_worker_health(worker_id),
                ))
            }
            RuntimeWorkerRequest::SetInput {
                worker_id,
                signal,
                lane_values,
            } => {
                self.shard_mut(&worker_id)?
                    .engine
                    .set_signal(signal, &lane_values)?;
                Ok(RuntimeWorkerResponse::Ack)
            }
            RuntimeWorkerRequest::SetInputLimbs {
                worker_id,
                signal,
                lane_values,
            } => {
                self.shard_mut(&worker_id)?
                    .engine
                    .set_signal_limbs(signal, &lane_values)?;
                Ok(RuntimeWorkerResponse::Ack)
            }
            RuntimeWorkerRequest::SetMemoryLimbs {
                worker_id,
                memory,
                lane_words,
            } => {
                self.shard_mut(&worker_id)?
                    .engine
                    .set_memory_limbs(memory, &lane_words)?;
                Ok(RuntimeWorkerResponse::Ack)
            }
            RuntimeWorkerRequest::EvalCombinational { worker_id } => {
                self.shard_mut(&worker_id)?
                    .execute(RuntimeShardAction::EvalCombinational)?;
                Ok(RuntimeWorkerResponse::Ack)
            }
            RuntimeWorkerRequest::Tick { worker_id } => {
                self.shard_mut(&worker_id)?
                    .execute(RuntimeShardAction::Tick)?;
                Ok(RuntimeWorkerResponse::Ack)
            }
            RuntimeWorkerRequest::TickMany { worker_id, steps } => {
                self.shard_mut(&worker_id)?
                    .execute(RuntimeShardAction::TickMany(steps))?;
                Ok(RuntimeWorkerResponse::Ack)
            }
            RuntimeWorkerRequest::GetSignalLimbs { worker_id, signal } => {
                let values = self
                    .shard_mut(&worker_id)?
                    .engine
                    .get_signal_limbs(signal)?;
                Ok(RuntimeWorkerResponse::SignalLimbs(values))
            }
            RuntimeWorkerRequest::GetMemoryLimbs { worker_id, memory } => {
                let values = self
                    .shard_mut(&worker_id)?
                    .engine
                    .get_memory_limbs(memory)?;
                Ok(RuntimeWorkerResponse::MemoryLimbs(values))
            }
            RuntimeWorkerRequest::Snapshot { worker_id } => {
                let shard = self.shard_mut(&worker_id)?;
                let storage = shard.engine.snapshot_storage()?;
                Ok(RuntimeWorkerResponse::Snapshot(RuntimeShardSnapshot {
                    shard: shard.info.clone(),
                    values: storage.values,
                    memories: storage.memories,
                }))
            }
            RuntimeWorkerRequest::RestoreSnapshot {
                worker_id,
                snapshot,
            } => {
                let shard = self.shard_mut(&worker_id)?;
                if snapshot.shard.start_lane != shard.info.start_lane
                    || snapshot.shard.lanes != shard.info.lanes
                {
                    return Err(error(
                        "E_RUNTIME_WORKER_SNAPSHOT",
                        "snapshot shard lane range does not match initialized worker shard",
                    ));
                }
                shard.engine.restore_storage(&PackedSimulatorStorage {
                    values: snapshot.values,
                    memories: snapshot.memories,
                })?;
                Ok(RuntimeWorkerResponse::Ack)
            }
            RuntimeWorkerRequest::Stats { worker_id } => {
                let shard = self.shard_mut(&worker_id)?;
                Ok(RuntimeWorkerResponse::Stats(RuntimeShardStats {
                    shard: shard.info.clone(),
                    operations: shard.operations,
                }))
            }
            RuntimeWorkerRequest::ResetStats { worker_id } => {
                self.shard_mut(&worker_id)?.operations = RuntimeOperationStats::default();
                Ok(RuntimeWorkerResponse::Ack)
            }
        }
    }
}

pub struct LoopbackRuntimeShardClient {
    worker_id: String,
    service: Arc<Mutex<LocalRuntimeWorkerService>>,
}

impl LoopbackRuntimeShardClient {
    pub fn new(
        worker_id: impl Into<String>,
        service: Arc<Mutex<LocalRuntimeWorkerService>>,
    ) -> Self {
        Self {
            worker_id: worker_id.into(),
            service,
        }
    }
}

impl RuntimeShardClient for LoopbackRuntimeShardClient {
    fn worker_id(&self) -> &str {
        &self.worker_id
    }

    fn request(
        &mut self,
        request: RuntimeWorkerRequest,
    ) -> Result<RuntimeWorkerResponse, ErrorReport> {
        self.service
            .lock()
            .map_err(|_| {
                error(
                    "E_RUNTIME_WORKER_LOCK",
                    "runtime worker service lock is poisoned",
                )
            })?
            .handle(request)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TcpRuntimeTransportConfig {
    pub bind_addr: SocketAddr,
    pub connect_timeout: Duration,
    pub read_timeout: Option<Duration>,
    pub write_timeout: Option<Duration>,
}

impl Default for TcpRuntimeTransportConfig {
    fn default() -> Self {
        Self {
            bind_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
            connect_timeout: Duration::from_secs(5),
            read_timeout: Some(Duration::from_secs(30)),
            write_timeout: Some(Duration::from_secs(30)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcpRuntimeWorkerProcessConfig {
    pub executable: PathBuf,
    pub bind_addr: SocketAddr,
    pub read_timeout: Option<Duration>,
    pub write_timeout: Option<Duration>,
    pub max_connections: Option<usize>,
}

impl TcpRuntimeWorkerProcessConfig {
    pub fn new(executable: impl Into<PathBuf>) -> Self {
        let transport = TcpRuntimeTransportConfig::default();
        Self {
            executable: executable.into(),
            bind_addr: transport.bind_addr,
            read_timeout: transport.read_timeout,
            write_timeout: transport.write_timeout,
            max_connections: None,
        }
    }
}

pub struct TcpRuntimeWorkerProcess {
    worker_id: String,
    endpoint: SocketAddr,
    child: Child,
    exit: Option<TcpRuntimeWorkerProcessExit>,
}

impl TcpRuntimeWorkerProcess {
    pub fn spawn(
        worker_id: impl Into<String>,
        config: &TcpRuntimeWorkerProcessConfig,
    ) -> Result<Self, ErrorReport> {
        let worker_id = worker_id.into();
        let mut child = worker_process_command(config)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(worker_process_io_error)?;
        let stdout = child.stdout.take().ok_or_else(|| {
            error(
                "E_RUNTIME_WORKER_PROCESS",
                "worker process stdout was not captured",
            )
        })?;
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        let bytes = reader
            .read_line(&mut line)
            .map_err(worker_process_io_error)?;
        if bytes == 0 {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error(
                "E_RUNTIME_WORKER_PROCESS",
                format!("worker process `{worker_id}` exited before reporting an endpoint"),
            ));
        }
        let endpoint = parse_worker_startup_addr(line.trim_end_matches('\n'))?;
        Ok(Self {
            worker_id,
            endpoint,
            child,
            exit: None,
        })
    }

    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    pub fn endpoint(&self) -> SocketAddr {
        self.endpoint
    }

    pub fn kill(&mut self) -> Result<(), ErrorReport> {
        if self.exit.is_some() {
            return Ok(());
        }
        if let Some(status) = self.child.try_wait().map_err(worker_process_io_error)? {
            self.exit = Some(worker_process_exit(self.worker_id.clone(), status)?);
        } else {
            self.child.kill().map_err(worker_process_io_error)?;
        }
        Ok(())
    }

    pub fn wait(&mut self) -> Result<TcpRuntimeWorkerProcessExit, ErrorReport> {
        if let Some(exit) = &self.exit {
            return Ok(exit.clone());
        }
        let exit = worker_process_exit(
            self.worker_id.clone(),
            self.child.wait().map_err(worker_process_io_error)?,
        )?;
        self.exit = Some(exit.clone());
        Ok(exit)
    }

    pub fn health(&mut self) -> Result<TcpRuntimeWorkerProcessHealth, ErrorReport> {
        if self.exit.is_none() {
            if let Some(status) = self.child.try_wait().map_err(worker_process_io_error)? {
                self.exit = Some(worker_process_exit(self.worker_id.clone(), status)?);
            }
        }
        Ok(TcpRuntimeWorkerProcessHealth {
            worker_id: self.worker_id.clone(),
            endpoint: self.endpoint,
            running: self.exit.is_none(),
            exit: self.exit.clone(),
        })
    }
}

impl Drop for TcpRuntimeWorkerProcess {
    fn drop(&mut self) {
        if self.exit.is_some() {
            return;
        }
        let _ = self.kill();
        let _ = self.child.wait();
    }
}

pub struct TcpRuntimeWorkerProcessSet {
    endpoints: HashMap<String, SocketAddr>,
    processes: Vec<TcpRuntimeWorkerProcess>,
}

impl TcpRuntimeWorkerProcessSet {
    pub fn spawn(
        topology: &RuntimeTopology,
        config: &TcpRuntimeWorkerProcessConfig,
    ) -> Result<Self, ErrorReport> {
        let mut endpoints = HashMap::new();
        let mut processes = Vec::with_capacity(topology.workers().len());
        for worker in topology.workers() {
            let process = TcpRuntimeWorkerProcess::spawn(worker.id.clone(), config)?;
            endpoints.insert(worker.id.clone(), process.endpoint());
            processes.push(process);
        }
        Ok(Self {
            endpoints,
            processes,
        })
    }

    pub fn endpoints(&self) -> &HashMap<String, SocketAddr> {
        &self.endpoints
    }

    pub fn processes(&self) -> &[TcpRuntimeWorkerProcess] {
        &self.processes
    }

    pub fn processes_mut(&mut self) -> &mut [TcpRuntimeWorkerProcess] {
        &mut self.processes
    }

    pub fn into_endpoints_and_processes(
        self,
    ) -> (HashMap<String, SocketAddr>, Vec<TcpRuntimeWorkerProcess>) {
        (self.endpoints, self.processes)
    }

    pub fn restart_all(
        &mut self,
        topology: &RuntimeTopology,
        config: &TcpRuntimeWorkerProcessConfig,
    ) -> Result<Vec<TcpRuntimeWorkerProcessExit>, ErrorReport> {
        let fresh = Self::spawn(topology, config)?;
        let mut old = std::mem::replace(self, fresh);
        old.kill_all()?;
        old.wait_all()
    }

    pub fn kill_all(&mut self) -> Result<(), ErrorReport> {
        for process in &mut self.processes {
            process.kill()?;
        }
        Ok(())
    }

    pub fn wait_all(&mut self) -> Result<Vec<TcpRuntimeWorkerProcessExit>, ErrorReport> {
        let mut exits = Vec::with_capacity(self.processes.len());
        for process in &mut self.processes {
            exits.push(process.wait()?);
        }
        Ok(exits)
    }

    pub fn health(&mut self) -> Result<Vec<TcpRuntimeWorkerProcessHealth>, ErrorReport> {
        let mut health = Vec::with_capacity(self.processes.len());
        for process in &mut self.processes {
            health.push(process.health()?);
        }
        Ok(health)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TcpRuntimeWorkerProcessExit {
    pub worker_id: String,
    pub success: bool,
    pub code: Option<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TcpRuntimeWorkerProcessHealth {
    pub worker_id: String,
    pub endpoint: SocketAddr,
    pub running: bool,
    pub exit: Option<TcpRuntimeWorkerProcessExit>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcpRuntimeSupervisorConfig {
    pub process_config: TcpRuntimeWorkerProcessConfig,
    pub runtime_options: DistributedRuntimeOptions,
    pub transport: TcpRuntimeTransportConfig,
}

impl TcpRuntimeSupervisorConfig {
    pub fn new(process_config: TcpRuntimeWorkerProcessConfig) -> Self {
        Self {
            process_config,
            runtime_options: DistributedRuntimeOptions::default(),
            transport: TcpRuntimeTransportConfig::default(),
        }
    }
}

pub struct TcpRuntimeSupervisor {
    design: Design,
    module_name: String,
    topology: RuntimeTopology,
    config: TcpRuntimeSupervisorConfig,
    runtime: DistributedRuntime,
    processes: TcpRuntimeWorkerProcessSet,
    latest_checkpoint: Option<RuntimeCheckpoint>,
    latest_checkpoint_event: Option<RuntimeCheckpointEvent>,
    last_recovery: Option<TcpRuntimeSupervisorRecoveryReport>,
}

impl TcpRuntimeSupervisor {
    pub fn spawn(
        design: &Design,
        module_name: impl Into<String>,
        topology: RuntimeTopology,
        config: TcpRuntimeSupervisorConfig,
    ) -> Result<Self, ErrorReport> {
        let module_name = module_name.into();
        let processes = TcpRuntimeWorkerProcessSet::spawn(&topology, &config.process_config)?;
        let runtime = DistributedRuntime::new_tcp_workers_with_config(
            design,
            &module_name,
            topology.clone(),
            config.runtime_options,
            processes.endpoints().clone(),
            config.transport,
        )?;
        Ok(Self {
            design: design.clone(),
            module_name,
            topology,
            config,
            runtime,
            processes,
            latest_checkpoint: None,
            latest_checkpoint_event: None,
            last_recovery: None,
        })
    }

    pub fn runtime(&self) -> &DistributedRuntime {
        &self.runtime
    }

    pub fn runtime_mut(&mut self) -> &mut DistributedRuntime {
        &mut self.runtime
    }

    pub fn module_name(&self) -> &str {
        &self.module_name
    }

    pub fn processes(&self) -> &TcpRuntimeWorkerProcessSet {
        &self.processes
    }

    pub fn processes_mut(&mut self) -> &mut TcpRuntimeWorkerProcessSet {
        &mut self.processes
    }

    pub fn endpoints(&self) -> &HashMap<String, SocketAddr> {
        self.processes.endpoints()
    }

    pub fn latest_checkpoint(&self) -> Option<&RuntimeCheckpoint> {
        self.latest_checkpoint.as_ref()
    }

    pub fn latest_checkpoint_event(&self) -> Option<RuntimeCheckpointEvent> {
        self.latest_checkpoint_event
    }

    pub fn last_recovery(&self) -> Option<&TcpRuntimeSupervisorRecoveryReport> {
        self.last_recovery.as_ref()
    }

    pub fn checkpoint(&mut self) -> Result<&RuntimeCheckpoint, ErrorReport> {
        let checkpoint = self
            .runtime
            .checkpoint_with_tcp_endpoints(self.processes.endpoints())?;
        self.latest_checkpoint = Some(checkpoint);
        self.latest_checkpoint_event = None;
        Ok(self.latest_checkpoint.as_ref().unwrap())
    }

    pub fn tick_many_with_checkpoints<F>(
        &mut self,
        steps: usize,
        cadence: RuntimeCheckpointCadence,
        mut on_checkpoint: F,
    ) -> Result<RuntimeCheckpointRunReport, ErrorReport>
    where
        F: FnMut(RuntimeCheckpointEvent, &RuntimeCheckpoint) -> Result<(), ErrorReport>,
    {
        let mut latest = None;
        let result = self.runtime.tick_many_with_tcp_checkpoints(
            steps,
            cadence,
            self.processes.endpoints(),
            |event, checkpoint| {
                latest = Some((event, checkpoint.clone()));
                on_checkpoint(event, checkpoint)
            },
        );
        if let Some((event, checkpoint)) = latest {
            self.latest_checkpoint = Some(checkpoint);
            self.latest_checkpoint_event = Some(event);
        }
        let report = result?;
        Ok(report)
    }

    pub fn health(&mut self) -> Result<TcpRuntimeSupervisorHealth, ErrorReport> {
        let processes = self.processes.health()?;
        let (runtime, runtime_error) = match self.runtime.health() {
            Ok(health) => (Some(health), None),
            Err(err) => (None, Some(err)),
        };
        Ok(TcpRuntimeSupervisorHealth {
            processes,
            runtime,
            runtime_error,
        })
    }

    pub fn telemetry(&mut self) -> Result<TcpRuntimeSupervisorTelemetry, ErrorReport> {
        let processes = self
            .processes
            .health()?
            .into_iter()
            .map(TcpRuntimeWorkerProcessTelemetry::from)
            .collect();
        let runtime_stats = self.runtime.stats();
        let (runtime_health, runtime_health_error) = match self.runtime.health() {
            Ok(health) => (Some(health), None),
            Err(err) => (None, Some(err)),
        };

        Ok(TcpRuntimeSupervisorTelemetry {
            format_version: RUNTIME_TELEMETRY_FORMAT_VERSION,
            module_name: self.module_name.clone(),
            topology: self.topology.clone(),
            surrogate_attachment: self.runtime.surrogate_attachment().cloned(),
            surrogate_execution: runtime_stats.surrogate_execution.clone(),
            endpoints: tcp_endpoint_list(self.processes.endpoints()),
            processes,
            runtime_stats,
            runtime_health,
            runtime_health_error,
            latest_checkpoint: self.latest_checkpoint.clone().map(|checkpoint| {
                TcpRuntimeSupervisorCheckpointTelemetry {
                    event: self.latest_checkpoint_event,
                    checkpoint,
                }
            }),
            last_recovery: self.last_recovery.clone(),
        })
    }

    pub fn recover_from_latest_checkpoint(
        &mut self,
    ) -> Result<TcpRuntimeSupervisorRecoveryReport, ErrorReport> {
        let checkpoint = self.latest_checkpoint.clone().ok_or_else(|| {
            error(
                "E_RUNTIME_SUPERVISOR_CHECKPOINT",
                "runtime supervisor has no checkpoint to recover from",
            )
        })?;

        let fresh = TcpRuntimeWorkerProcessSet::spawn(&self.topology, &self.config.process_config)?;
        let endpoints = fresh.endpoints().clone();
        let runtime_recovery = self
            .runtime
            .recover_tcp_workers_from_checkpoint_with_config(
                &self.design,
                &checkpoint,
                endpoints,
                self.config.transport,
            )?;
        let mut old_processes = std::mem::replace(&mut self.processes, fresh);
        old_processes.kill_all()?;
        let restarted_workers = old_processes.wait_all()?;
        self.latest_checkpoint = Some(
            self.runtime
                .checkpoint_with_tcp_endpoints(self.processes.endpoints())?,
        );
        let report = TcpRuntimeSupervisorRecoveryReport {
            restarted_workers,
            runtime_recovery,
        };
        self.last_recovery = Some(report.clone());
        Ok(report)
    }

    pub fn shutdown(&mut self) -> Result<Vec<TcpRuntimeWorkerProcessExit>, ErrorReport> {
        self.processes.kill_all()?;
        self.processes.wait_all()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcpRuntimeSupervisorHealth {
    pub processes: Vec<TcpRuntimeWorkerProcessHealth>,
    pub runtime: Option<RuntimeHealthReport>,
    pub runtime_error: Option<ErrorReport>,
}

impl From<TcpRuntimeWorkerProcessHealth> for TcpRuntimeWorkerProcessTelemetry {
    fn from(health: TcpRuntimeWorkerProcessHealth) -> Self {
        Self {
            worker_id: health.worker_id,
            endpoint: health.endpoint.to_string(),
            running: health.running,
            exit: health.exit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TcpRuntimeSupervisorRecoveryReport {
    pub restarted_workers: Vec<TcpRuntimeWorkerProcessExit>,
    pub runtime_recovery: RuntimeRecoveryReport,
}

pub struct TcpRuntimeWorkerServer {
    listener: TcpListener,
    service: LocalRuntimeWorkerService,
    config: TcpRuntimeTransportConfig,
}

impl TcpRuntimeWorkerServer {
    pub fn bind<A: ToSocketAddrs>(addr: A) -> Result<Self, ErrorReport> {
        let listener = TcpListener::bind(addr).map_err(tcp_io_error)?;
        let mut config = TcpRuntimeTransportConfig::default();
        config.bind_addr = listener.local_addr().map_err(tcp_io_error)?;
        Ok(Self {
            listener,
            service: LocalRuntimeWorkerService::new(),
            config,
        })
    }

    pub fn bind_with_config(config: TcpRuntimeTransportConfig) -> Result<Self, ErrorReport> {
        let listener = TcpListener::bind(config.bind_addr).map_err(tcp_io_error)?;
        Ok(Self {
            listener,
            service: LocalRuntimeWorkerService::new(),
            config,
        })
    }

    pub fn local_addr(&self) -> Result<SocketAddr, ErrorReport> {
        self.listener.local_addr().map_err(tcp_io_error)
    }

    pub fn serve_once(&mut self) -> Result<(), ErrorReport> {
        let (stream, _) = self.listener.accept().map_err(tcp_io_error)?;
        configure_tcp_stream(&stream, self.config)?;
        self.serve_stream(stream)
    }

    pub fn serve_connections(&mut self, connections: usize) -> Result<(), ErrorReport> {
        for _ in 0..connections {
            self.serve_once()?;
        }
        Ok(())
    }

    pub fn serve(&mut self) -> Result<(), ErrorReport> {
        loop {
            self.serve_once()?;
        }
    }

    fn serve_stream(&mut self, mut stream: TcpStream) -> Result<(), ErrorReport> {
        let reader_stream = stream.try_clone().map_err(tcp_io_error)?;
        let mut reader = BufReader::new(reader_stream);
        loop {
            let mut line = String::new();
            let bytes = reader.read_line(&mut line).map_err(tcp_io_error)?;
            if bytes == 0 {
                return Ok(());
            }
            let report = if line.ends_with('\n') {
                match serde_json::from_str::<RuntimeWorkerRequest>(line.trim_end_matches('\n')) {
                    Ok(request) => match self.service.handle(request) {
                        Ok(response) => RuntimeWorkerWireResponse::Ok(response),
                        Err(report) => RuntimeWorkerWireResponse::Err(report),
                    },
                    Err(err) => RuntimeWorkerWireResponse::Err(tcp_json_error(err)),
                }
            } else {
                RuntimeWorkerWireResponse::Err(error(
                    "E_RUNTIME_TCP_EOF",
                    "TCP worker request ended before newline frame delimiter",
                ))
            };
            write_json_line(&mut stream, &report)?;
        }
    }
}

pub struct TcpRuntimeShardClient {
    worker_id: String,
    reader: BufReader<TcpStream>,
    writer: TcpStream,
}

impl TcpRuntimeShardClient {
    pub fn connect<A: ToSocketAddrs>(
        worker_id: impl Into<String>,
        addr: A,
    ) -> Result<Self, ErrorReport> {
        Self::connect_with_config(worker_id, addr, TcpRuntimeTransportConfig::default())
    }

    pub fn connect_with_config<A: ToSocketAddrs>(
        worker_id: impl Into<String>,
        addr: A,
        config: TcpRuntimeTransportConfig,
    ) -> Result<Self, ErrorReport> {
        let addr = first_socket_addr(addr)?;
        let stream =
            TcpStream::connect_timeout(&addr, config.connect_timeout).map_err(tcp_io_error)?;
        configure_tcp_stream(&stream, config)?;
        let reader = BufReader::new(stream.try_clone().map_err(tcp_io_error)?);
        Ok(Self {
            worker_id: worker_id.into(),
            reader,
            writer: stream,
        })
    }

    pub fn health(&mut self) -> Result<RuntimeWorkerHealth, ErrorReport> {
        <Self as RuntimeShardClient>::health(self)
    }
}

impl RuntimeShardClient for TcpRuntimeShardClient {
    fn worker_id(&self) -> &str {
        &self.worker_id
    }

    fn request(
        &mut self,
        request: RuntimeWorkerRequest,
    ) -> Result<RuntimeWorkerResponse, ErrorReport> {
        write_json_line(&mut self.writer, &request)?;
        match read_worker_wire_response(&mut self.reader)? {
            RuntimeWorkerWireResponse::Ok(response) => Ok(response),
            RuntimeWorkerWireResponse::Err(report) => Err(report),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum RuntimeWorkerWireResponse {
    Ok(RuntimeWorkerResponse),
    Err(ErrorReport),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistributedRuntimeOptions {
    pub execution_mode: RuntimeExecutionMode,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeExecutionMode {
    #[default]
    Serial,
    Parallel,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeStats {
    pub execution_mode: RuntimeExecutionMode,
    pub total_lanes: usize,
    pub operations: RuntimeOperationStats,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surrogate_execution: Option<RuntimeSurrogateExecutionReport>,
    pub shards: Vec<RuntimeShardStats>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeOperationStats {
    pub eval_combinational: RuntimeOperationCounters,
    pub tick: RuntimeOperationCounters,
    pub tick_many: RuntimeOperationCounters,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeOperationCounters {
    pub calls: u64,
    pub total_ns: u128,
    pub last_ns: u128,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeShardStats {
    pub shard: RuntimeShardInfo,
    pub operations: RuntimeOperationStats,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeWorkerHealth {
    pub worker_id: String,
    pub initialized: bool,
    pub shard: Option<RuntimeShardInfo>,
    pub operations: Option<RuntimeOperationStats>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePartitionWorkerHealth {
    pub worker_id: String,
    pub initialized: bool,
    pub backend: Option<RuntimeBackend>,
    pub node: Option<String>,
    pub partitions: Vec<RuntimePartitionLaunchPartition>,
    pub outbound_routes: Vec<usize>,
    pub inbound_routes: Vec<usize>,
    pub outbound_bits: u64,
    pub inbound_bits: u64,
    pub diagnostics: Vec<RuntimePartitionLaunchDiagnostic>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeHealthReport {
    pub total_lanes: usize,
    pub shards: Vec<RuntimeShardHealth>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeShardHealth {
    pub shard: RuntimeShardInfo,
    pub status: RuntimeShardHealthStatus,
    pub operations: Option<RuntimeOperationStats>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeShardHealthStatus {
    Healthy,
    Uninitialized,
    Unreachable,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeRecoveryReport {
    pub recovered_workers: Vec<String>,
}

/// Converts a runtime partition plan into the `PackedSliceGroup`s consumed by
/// the simulation IR's [`PartitionedSimulator`]. Each partition becomes one
/// group owning its instance-path context; the slicer's longest-prefix
/// ownership then assigns every signal — including signals of unsplit child
/// instances — to the most-specific partition.
pub fn slice_groups_for_partition_plan(plan: &RuntimePartitionPlan) -> Vec<PackedSliceGroup> {
    plan.partitions
        .iter()
        .map(|partition| PackedSliceGroup {
            id: partition.id.clone(),
            owned_paths: vec![partition.instance_path.join(".")],
        })
        .collect()
}

/// Builds a [`PartitionedSimulator`] that runs `module_name` (in `design`) split
/// across the partitions of `plan`, with `lanes` independent stimulus lanes.
/// Groups at the same combinational depth execute concurrently, and the result
/// is bit-identical to the whole-design packed simulator.
pub fn build_partitioned_simulator(
    design: &Design,
    module_name: &str,
    plan: &RuntimePartitionPlan,
    lanes: usize,
) -> Result<PartitionedSimulator, ErrorReport> {
    let compiled = compile(design)?;
    let program = lower_to_packed_program(&compiled, module_name)?;
    let groups = slice_groups_for_partition_plan(plan);
    PartitionedSimulator::new(&program, &groups, lanes)
}

pub fn plan_runtime_partitions(
    design: &Design,
    module_name: &str,
    config: RuntimePartitionConfig,
) -> Result<RuntimePartitionPlan, ErrorReport> {
    if config.target_partitions == 0 {
        return Err(error(
            "E_RUNTIME_PARTITION_CONFIG",
            "runtime partition planner target_partitions must be greater than zero",
        ));
    }

    let compiled = compile(design)?;
    let top = compiled.find_module(module_name).ok_or_else(|| {
        error(
            "E_RUNTIME_PARTITION_MODULE",
            format!("module `{module_name}` does not exist"),
        )
    })?;

    let regions = partition_regions(&compiled, top, &config)?;
    let partitions = regions
        .iter()
        .map(|region| build_partition(&compiled, region))
        .collect::<Result<Vec<_>, _>>()?;
    let boundary_signals = partition_boundaries(&compiled, top, &regions);
    let total_cost = total_partition_cost(&partitions, &boundary_signals);
    let analysis = analyze_partition_plan(&partitions, &boundary_signals, &total_cost, &config);

    Ok(RuntimePartitionPlan {
        format_version: RUNTIME_PARTITION_PLAN_FORMAT_VERSION,
        module_name: module_name.to_string(),
        partitions,
        boundary_signals,
        total_cost,
        diagnostics: analysis.diagnostics,
        recommendations: analysis.recommendations,
    })
}

pub fn plan_runtime_partition_placement(
    plan: &RuntimePartitionPlan,
    topology: &RuntimeTopology,
) -> Result<RuntimePartitionPlacementPlan, ErrorReport> {
    plan.validate_format_version()?;
    validate_partition_placement_topology(topology)?;

    let mut workers = topology
        .workers()
        .iter()
        .map(PlacementWorkerLoad::from_worker)
        .collect::<Vec<_>>();
    let has_cpu_worker = workers.iter().any(|worker| worker.is_cpu());
    let has_gpu_worker = workers.iter().any(|worker| worker.is_gpu());
    let mut assignments = Vec::with_capacity(plan.partitions.len());
    let mut diagnostics = Vec::new();
    let mut recommendations = Vec::new();

    let mut ranked_partitions = plan.partitions.iter().collect::<Vec<_>>();
    ranked_partitions.sort_by(|left, right| {
        partition_split_score(right.cost)
            .cmp(&partition_split_score(left.cost))
            .then_with(|| left.id.cmp(&right.id))
    });

    for partition in ranked_partitions {
        let cpu_fallback = has_cpu_worker && has_gpu_worker && partition_gpu_hostile(partition);
        let Some(worker_index) = best_placement_worker(&workers, partition, has_cpu_worker) else {
            return Err(error(
                "E_RUNTIME_PARTITION_PLACEMENT_UNPLACED",
                format!(
                    "partition `{}` has no compatible runtime worker",
                    partition.id
                ),
            ));
        };
        let worker = &mut workers[worker_index];
        worker.cost += partition.cost;
        worker.partitions.push(partition.id.clone());
        assignments.push(RuntimePartitionAssignment {
            partition_id: partition.id.clone(),
            worker_id: worker.worker_id.clone(),
            backend: worker.backend.clone(),
            node: worker.node.clone(),
            instance_path: partition.instance_path.clone(),
            cost: partition.cost,
        });

        if cpu_fallback {
            diagnostics.push(RuntimePartitionPlacementDiagnostic {
                code: RuntimePartitionPlacementDiagnosticCode::CpuFallback,
                severity: RuntimePartitionDiagnosticSeverity::Info,
                message: format!(
                    "partition `{}` uses CPU placement because external or memory-heavy partitions are GPU-hostile in v1",
                    partition.id
                ),
                worker_id: Some(worker.worker_id.clone()),
                partition_id: Some(partition.id.clone()),
                boundary_index: None,
            });
        }
    }

    assignments.sort_by(|left, right| left.partition_id.cmp(&right.partition_id));
    account_placement_boundary_traffic(
        plan,
        &assignments,
        &mut workers,
        &mut diagnostics,
        &mut recommendations,
    );
    analyze_placement_worker_balance(
        plan,
        topology,
        &workers,
        &mut diagnostics,
        &mut recommendations,
    );
    let worker_summaries = workers
        .into_iter()
        .map(PlacementWorkerLoad::into_summary)
        .collect::<Vec<_>>();

    Ok(RuntimePartitionPlacementPlan {
        format_version: RUNTIME_PARTITION_PLACEMENT_FORMAT_VERSION,
        module_name: plan.module_name.clone(),
        topology: topology.clone(),
        assignments,
        worker_summaries,
        total_cost: plan.total_cost,
        diagnostics,
        recommendations,
    })
}

pub fn plan_runtime_partition_communication(
    plan: &RuntimePartitionPlan,
    placement: &RuntimePartitionPlacementPlan,
) -> Result<RuntimePartitionCommunicationPlan, ErrorReport> {
    plan.validate_format_version()?;
    placement.validate_format_version()?;
    if plan.module_name != placement.module_name {
        return Err(error(
            "E_RUNTIME_PARTITION_COMMUNICATION_MODULE",
            format!(
                "partition plan module `{}` does not match placement module `{}`",
                plan.module_name, placement.module_name
            ),
        ));
    }

    let assignment_by_partition = placement
        .assignments
        .iter()
        .map(|assignment| (assignment.partition_id.as_str(), assignment))
        .collect::<HashMap<_, _>>();
    let mut routes = Vec::new();
    for (boundary_index, boundary) in plan.boundary_signals.iter().enumerate() {
        let Some(producer_partition) = boundary.producer_partition.as_deref() else {
            continue;
        };
        let producer =
            communication_assignment(&assignment_by_partition, producer_partition, boundary_index)?;
        for consumer_partition in &boundary.consumer_partitions {
            let consumer = communication_assignment(
                &assignment_by_partition,
                consumer_partition,
                boundary_index,
            )?;
            if producer.worker_id == consumer.worker_id {
                continue;
            }
            routes.push(RuntimePartitionRoute {
                boundary_index,
                signal: boundary.signal.clone(),
                port_name: boundary.port_name.clone(),
                instance_path: boundary.instance_path.clone(),
                producer_partition: producer_partition.to_string(),
                producer_worker: producer.worker_id.clone(),
                consumer_partition: consumer_partition.clone(),
                consumer_worker: consumer.worker_id.clone(),
                width: boundary.width,
                bits_per_transfer: u64::from(boundary.width),
            });
        }
    }
    routes.sort_by(|left, right| {
        left.boundary_index
            .cmp(&right.boundary_index)
            .then_with(|| left.producer_worker.cmp(&right.producer_worker))
            .then_with(|| left.consumer_worker.cmp(&right.consumer_worker))
            .then_with(|| left.consumer_partition.cmp(&right.consumer_partition))
    });

    let worker_summaries = communication_worker_summaries(placement, &routes);
    let total_cross_worker_boundary_bits = routes
        .iter()
        .map(|route| route.bits_per_transfer)
        .sum::<u64>();
    let mut diagnostics = Vec::new();
    let mut recommendations = Vec::new();
    analyze_partition_communication(
        plan,
        &routes,
        total_cross_worker_boundary_bits,
        &mut diagnostics,
        &mut recommendations,
    );

    Ok(RuntimePartitionCommunicationPlan {
        format_version: RUNTIME_PARTITION_COMMUNICATION_FORMAT_VERSION,
        module_name: plan.module_name.clone(),
        routes,
        worker_summaries,
        total_cross_worker_boundary_bits,
        diagnostics,
        recommendations,
    })
}

pub fn plan_runtime_partition_bundle(
    design: &Design,
    module_name: &str,
    config: RuntimePartitionConfig,
    topology: &RuntimeTopology,
) -> Result<RuntimePartitionBundle, ErrorReport> {
    let partition_plan = plan_runtime_partitions(design, module_name, config)?;
    let placement_plan = plan_runtime_partition_placement(&partition_plan, topology)?;
    let communication_plan =
        plan_runtime_partition_communication(&partition_plan, &placement_plan)?;
    RuntimePartitionBundle::from_plans(partition_plan, placement_plan, communication_plan)
}

pub fn plan_runtime_partition_launch(
    bundle: &RuntimePartitionBundle,
) -> Result<RuntimePartitionLaunchPlan, ErrorReport> {
    validate_partition_launch_bundle(bundle)?;

    let partition_by_id = bundle
        .partition_plan
        .partitions
        .iter()
        .map(|partition| (partition.id.as_str(), partition))
        .collect::<HashMap<_, _>>();
    let assignment_by_partition = bundle
        .placement_plan
        .assignments
        .iter()
        .map(|assignment| (assignment.partition_id.as_str(), assignment))
        .collect::<HashMap<_, _>>();

    let mut workers = bundle
        .placement_plan
        .topology
        .workers()
        .iter()
        .map(|worker| RuntimePartitionWorkerLaunch {
            worker_id: worker.id.clone(),
            backend: worker.backend,
            node: worker.node.clone(),
            partitions: Vec::new(),
            outbound_routes: Vec::new(),
            inbound_routes: Vec::new(),
            outbound_route_specs: Vec::new(),
            inbound_route_specs: Vec::new(),
            cost: RuntimePartitionCost::default(),
            outbound_bits: 0,
            inbound_bits: 0,
        })
        .collect::<Vec<_>>();
    workers.sort_by(|left, right| left.worker_id.cmp(&right.worker_id));
    let worker_index_by_id = workers
        .iter()
        .enumerate()
        .map(|(index, worker)| (worker.worker_id.clone(), index))
        .collect::<HashMap<_, _>>();

    let mut assignments = bundle.placement_plan.assignments.iter().collect::<Vec<_>>();
    assignments.sort_by(|left, right| left.partition_id.cmp(&right.partition_id));
    for assignment in assignments {
        let partition = partition_by_id
            .get(assignment.partition_id.as_str())
            .copied()
            .ok_or_else(|| {
                error(
                    "E_RUNTIME_PARTITION_LAUNCH_PARTITION",
                    format!(
                        "placement assignment references partition `{}` that is not in the partition plan",
                        assignment.partition_id
                    ),
                )
            })?;
        let worker_index = *worker_index_by_id
            .get(&assignment.worker_id)
            .ok_or_else(|| {
                error(
                    "E_RUNTIME_PARTITION_LAUNCH_WORKER",
                    format!(
                        "placement assignment for partition `{}` references worker `{}` that is not in the topology",
                        assignment.partition_id, assignment.worker_id
                    ),
                )
            })?;
        let worker = &mut workers[worker_index];
        worker.cost += assignment.cost;
        worker.partitions.push(RuntimePartitionLaunchPartition {
            partition_id: assignment.partition_id.clone(),
            module_name: partition.module_name.clone(),
            instance_path: partition.instance_path.clone(),
            external: partition.external,
            cost: partition.cost,
        });
    }

    let mut routes = Vec::with_capacity(bundle.communication_plan.routes.len());
    for (route_index, route) in bundle.communication_plan.routes.iter().enumerate() {
        let producer = assignment_by_partition
            .get(route.producer_partition.as_str())
            .copied()
            .ok_or_else(|| {
                error(
                    "E_RUNTIME_PARTITION_LAUNCH_ASSIGNMENT",
                    format!(
                        "communication route {route_index} references producer partition `{}` with no placement assignment",
                        route.producer_partition
                    ),
                )
            })?;
        let consumer = assignment_by_partition
            .get(route.consumer_partition.as_str())
            .copied()
            .ok_or_else(|| {
                error(
                    "E_RUNTIME_PARTITION_LAUNCH_ASSIGNMENT",
                    format!(
                        "communication route {route_index} references consumer partition `{}` with no placement assignment",
                        route.consumer_partition
                    ),
                )
            })?;
        if producer.worker_id != route.producer_worker
            || consumer.worker_id != route.consumer_worker
        {
            return Err(error(
                "E_RUNTIME_PARTITION_LAUNCH_ROUTE",
                format!(
                    "communication route {route_index} worker mapping is stale for producer `{}` or consumer `{}`",
                    route.producer_partition, route.consumer_partition
                ),
            ));
        }

        let producer_worker_index = *worker_index_by_id
            .get(&route.producer_worker)
            .ok_or_else(|| {
                error(
                    "E_RUNTIME_PARTITION_LAUNCH_WORKER",
                    format!(
                        "communication route {route_index} references producer worker `{}` that is not in the topology",
                        route.producer_worker
                    ),
                )
            })?;
        let consumer_worker_index = *worker_index_by_id
            .get(&route.consumer_worker)
            .ok_or_else(|| {
                error(
                    "E_RUNTIME_PARTITION_LAUNCH_WORKER",
                    format!(
                        "communication route {route_index} references consumer worker `{}` that is not in the topology",
                        route.consumer_worker
                    ),
                )
            })?;

        workers[producer_worker_index]
            .outbound_routes
            .push(route_index);
        workers[producer_worker_index].outbound_bits += route.bits_per_transfer;
        workers[consumer_worker_index]
            .inbound_routes
            .push(route_index);
        workers[consumer_worker_index].inbound_bits += route.bits_per_transfer;
        routes.push(RuntimePartitionLaunchRoute {
            route_index,
            boundary_index: route.boundary_index,
            signal: route.signal.clone(),
            port_name: route.port_name.clone(),
            instance_path: route.instance_path.clone(),
            producer_partition: route.producer_partition.clone(),
            producer_worker: route.producer_worker.clone(),
            consumer_partition: route.consumer_partition.clone(),
            consumer_worker: route.consumer_worker.clone(),
            width: route.width,
            bits_per_transfer: route.bits_per_transfer,
        });
    }

    for worker in &mut workers {
        worker.outbound_route_specs = worker
            .outbound_routes
            .iter()
            .map(|route_index| routes[*route_index].clone())
            .collect();
        worker.inbound_route_specs = worker
            .inbound_routes
            .iter()
            .map(|route_index| routes[*route_index].clone())
            .collect();
    }

    let diagnostics = runtime_partition_launch_diagnostics(bundle, &workers);

    Ok(RuntimePartitionLaunchPlan {
        format_version: RUNTIME_PARTITION_LAUNCH_FORMAT_VERSION,
        module_name: bundle.module_name.clone(),
        workers,
        routes,
        diagnostics,
    })
}

pub fn deploy_runtime_partition_launch_local(
    launch: &RuntimePartitionLaunchPlan,
    service: &mut LocalRuntimeWorkerService,
) -> Result<RuntimePartitionDeploymentReport, ErrorReport> {
    validate_partition_deployment_launch(launch)?;
    let mut workers = Vec::with_capacity(launch.workers.len());
    for worker in &launch.workers {
        expect_ack(service.handle(RuntimeWorkerRequest::InitPartitionWorker(
            RuntimePartitionWorkerInit {
                worker_id: worker.worker_id.clone(),
                launch: worker.clone(),
            },
        ))?)?;
        let health = expect_partition_worker_health(
            worker.worker_id.as_str(),
            service.handle(RuntimeWorkerRequest::PartitionWorkerHealth {
                worker_id: worker.worker_id.clone(),
            })?,
        )?;
        validate_partition_deployment_health(worker, &health)?;
        workers.push(health);
    }
    Ok(RuntimePartitionDeploymentReport {
        module_name: launch.module_name.clone(),
        workers,
        diagnostics: launch.diagnostics.clone(),
    })
}

pub fn deploy_runtime_partition_launch_loopback(
    launch: &RuntimePartitionLaunchPlan,
    service: Arc<Mutex<LocalRuntimeWorkerService>>,
) -> Result<RuntimePartitionDeploymentReport, ErrorReport> {
    validate_partition_deployment_launch(launch)?;
    let mut workers = Vec::with_capacity(launch.workers.len());
    for worker in &launch.workers {
        let mut client = LoopbackRuntimeShardClient::new(worker.worker_id.clone(), service.clone());
        workers.push(deploy_runtime_partition_worker(worker, &mut client)?);
    }
    Ok(RuntimePartitionDeploymentReport {
        module_name: launch.module_name.clone(),
        workers,
        diagnostics: launch.diagnostics.clone(),
    })
}

pub fn deploy_runtime_partition_launch_tcp(
    launch: &RuntimePartitionLaunchPlan,
    endpoints: &HashMap<String, SocketAddr>,
    transport: TcpRuntimeTransportConfig,
) -> Result<RuntimePartitionDeploymentReport, ErrorReport> {
    validate_partition_deployment_launch(launch)?;
    let mut workers = Vec::with_capacity(launch.workers.len());
    for worker in &launch.workers {
        let endpoint = *endpoints.get(&worker.worker_id).ok_or_else(|| {
            error(
                "E_RUNTIME_PARTITION_DEPLOY_ENDPOINT",
                format!(
                    "missing TCP endpoint for partition worker `{}`",
                    worker.worker_id
                ),
            )
        })?;
        let mut client = TcpRuntimeShardClient::connect_with_config(
            worker.worker_id.clone(),
            endpoint,
            transport,
        )?;
        workers.push(deploy_runtime_partition_worker(worker, &mut client)?);
    }
    Ok(RuntimePartitionDeploymentReport {
        module_name: launch.module_name.clone(),
        workers,
        diagnostics: launch.diagnostics.clone(),
    })
}

impl RuntimePartitionSession {
    pub fn deploy_local(
        launch: RuntimePartitionLaunchPlan,
        service: &mut LocalRuntimeWorkerService,
    ) -> Result<Self, ErrorReport> {
        let deployment = deploy_runtime_partition_launch_local(&launch, service)?;
        let route_mailboxes = runtime_partition_initial_mailboxes(&launch, &deployment);
        Ok(Self {
            launch,
            deployment,
            route_mailboxes,
            actions: Vec::new(),
            surrogate_attachment: None,
        })
    }

    pub fn deploy_loopback(
        launch: RuntimePartitionLaunchPlan,
        service: Arc<Mutex<LocalRuntimeWorkerService>>,
    ) -> Result<Self, ErrorReport> {
        let deployment = deploy_runtime_partition_launch_loopback(&launch, service)?;
        let route_mailboxes = runtime_partition_initial_mailboxes(&launch, &deployment);
        Ok(Self {
            launch,
            deployment,
            route_mailboxes,
            actions: Vec::new(),
            surrogate_attachment: None,
        })
    }

    pub fn deploy_tcp(
        launch: RuntimePartitionLaunchPlan,
        endpoints: &HashMap<String, SocketAddr>,
        transport: TcpRuntimeTransportConfig,
    ) -> Result<Self, ErrorReport> {
        let deployment = deploy_runtime_partition_launch_tcp(&launch, endpoints, transport)?;
        let route_mailboxes = runtime_partition_initial_mailboxes(&launch, &deployment);
        Ok(Self {
            launch,
            deployment,
            route_mailboxes,
            actions: Vec::new(),
            surrogate_attachment: None,
        })
    }

    pub fn launch(&self) -> &RuntimePartitionLaunchPlan {
        &self.launch
    }

    pub fn deployment(&self) -> &RuntimePartitionDeploymentReport {
        &self.deployment
    }

    pub fn route_mailboxes(&self) -> &[RuntimePartitionRouteMailboxReport] {
        &self.route_mailboxes
    }

    pub fn actions(&self) -> &[RuntimePartitionSessionActionReport] {
        &self.actions
    }

    pub fn latest_action(&self) -> Option<&RuntimePartitionSessionActionReport> {
        self.actions.last()
    }

    pub fn clear_actions(&mut self) {
        self.actions.clear();
    }

    pub fn attach_surrogate_plan(
        &mut self,
        topology: &RuntimeTopology,
        plan: &GemmRuntimePlan,
    ) -> Result<(), ErrorReport> {
        self.surrogate_attachment = Some(topology.attach_gemm_runtime_plan(plan)?);
        Ok(())
    }

    pub fn attach_event_surrogate_plan(
        &mut self,
        topology: &RuntimeTopology,
        plan: &EventRuntimePlan,
    ) -> Result<(), ErrorReport> {
        self.surrogate_attachment = Some(topology.attach_event_runtime_plan(plan)?);
        Ok(())
    }

    pub fn surrogate_attachment(&self) -> Option<&RuntimeSurrogateAttachment> {
        self.surrogate_attachment.as_ref()
    }

    pub fn surrogate_execution(&self) -> Option<RuntimeSurrogateExecutionReport> {
        self.surrogate_attachment.as_ref().map(|attachment| {
            RuntimeSurrogateExecutionReport::from_attachment(
                attachment,
                self.latest_action()
                    .map(|action| partition_surrogate_action_kind(action.kind)),
                self.actions.len() as u64,
            )
        })
    }

    pub fn health(&self) -> RuntimePartitionSessionHealth {
        runtime_partition_session_health(&self.launch, &self.deployment)
    }

    pub fn action_summary(&self) -> RuntimePartitionSessionActionSummary {
        runtime_partition_session_action_summary(&self.launch, &self.deployment, &self.actions)
    }

    pub fn telemetry(&self) -> RuntimePartitionSessionTelemetry {
        RuntimePartitionSessionTelemetry {
            format_version: RUNTIME_PARTITION_SESSION_TELEMETRY_FORMAT_VERSION,
            module_name: self.launch.module_name.clone(),
            surrogate_attachment: self.surrogate_attachment.clone(),
            surrogate_execution: self.surrogate_execution(),
            launch: self.launch.clone(),
            deployment: self.deployment.clone(),
            health: self.health(),
            action_summary: self.action_summary(),
            route_mailboxes: self.route_mailboxes.clone(),
            actions: self.actions.clone(),
        }
    }

    pub fn run_local_action(
        &mut self,
        kind: RuntimePartitionWorkerActionKind,
        service: &mut LocalRuntimeWorkerService,
    ) -> Result<RuntimePartitionSessionActionReport, ErrorReport> {
        validate_partition_session_ready(self)?;
        let mut workers = Vec::with_capacity(self.launch.workers.len());
        for worker in &self.launch.workers {
            let report = expect_partition_worker_action(
                worker.worker_id.as_str(),
                kind,
                service.handle(RuntimeWorkerRequest::RunPartitionWorkerAction(
                    RuntimePartitionWorkerAction {
                        worker_id: worker.worker_id.clone(),
                        kind,
                    },
                ))?,
            )?;
            workers.push(report);
        }
        Ok(self.record_action_report(kind, workers))
    }

    pub fn run_loopback_action(
        &mut self,
        kind: RuntimePartitionWorkerActionKind,
        service: Arc<Mutex<LocalRuntimeWorkerService>>,
    ) -> Result<RuntimePartitionSessionActionReport, ErrorReport> {
        validate_partition_session_ready(self)?;
        let mut workers = Vec::with_capacity(self.launch.workers.len());
        for worker in &self.launch.workers {
            let mut client =
                LoopbackRuntimeShardClient::new(worker.worker_id.clone(), service.clone());
            workers.push(run_partition_worker_action(
                worker.worker_id.as_str(),
                kind,
                &mut client,
            )?);
        }
        Ok(self.record_action_report(kind, workers))
    }

    pub fn run_tcp_action(
        &mut self,
        kind: RuntimePartitionWorkerActionKind,
        endpoints: &HashMap<String, SocketAddr>,
        transport: TcpRuntimeTransportConfig,
    ) -> Result<RuntimePartitionSessionActionReport, ErrorReport> {
        validate_partition_session_ready(self)?;
        let mut workers = Vec::with_capacity(self.launch.workers.len());
        for worker in &self.launch.workers {
            let endpoint = *endpoints.get(&worker.worker_id).ok_or_else(|| {
                error(
                    "E_RUNTIME_PARTITION_DEPLOY_ENDPOINT",
                    format!(
                        "missing TCP endpoint for partition worker `{}`",
                        worker.worker_id
                    ),
                )
            })?;
            let mut client = TcpRuntimeShardClient::connect_with_config(
                worker.worker_id.clone(),
                endpoint,
                transport,
            )?;
            workers.push(run_partition_worker_action(
                worker.worker_id.as_str(),
                kind,
                &mut client,
            )?);
        }
        Ok(self.record_action_report(kind, workers))
    }

    pub fn eval_combinational_local(
        &mut self,
        service: &mut LocalRuntimeWorkerService,
    ) -> Result<RuntimePartitionSessionActionReport, ErrorReport> {
        self.run_local_action(RuntimePartitionWorkerActionKind::EvalCombinational, service)
    }

    pub fn tick_local(
        &mut self,
        service: &mut LocalRuntimeWorkerService,
    ) -> Result<RuntimePartitionSessionActionReport, ErrorReport> {
        self.run_local_action(RuntimePartitionWorkerActionKind::Tick, service)
    }

    pub fn tick_many_local(
        &mut self,
        steps: usize,
        service: &mut LocalRuntimeWorkerService,
    ) -> Result<RuntimePartitionSessionActionReport, ErrorReport> {
        self.run_local_action(RuntimePartitionWorkerActionKind::TickMany(steps), service)
    }

    pub fn eval_combinational_loopback(
        &mut self,
        service: Arc<Mutex<LocalRuntimeWorkerService>>,
    ) -> Result<RuntimePartitionSessionActionReport, ErrorReport> {
        self.run_loopback_action(RuntimePartitionWorkerActionKind::EvalCombinational, service)
    }

    pub fn tick_loopback(
        &mut self,
        service: Arc<Mutex<LocalRuntimeWorkerService>>,
    ) -> Result<RuntimePartitionSessionActionReport, ErrorReport> {
        self.run_loopback_action(RuntimePartitionWorkerActionKind::Tick, service)
    }

    pub fn tick_many_loopback(
        &mut self,
        steps: usize,
        service: Arc<Mutex<LocalRuntimeWorkerService>>,
    ) -> Result<RuntimePartitionSessionActionReport, ErrorReport> {
        self.run_loopback_action(RuntimePartitionWorkerActionKind::TickMany(steps), service)
    }

    pub fn eval_combinational_tcp(
        &mut self,
        endpoints: &HashMap<String, SocketAddr>,
        transport: TcpRuntimeTransportConfig,
    ) -> Result<RuntimePartitionSessionActionReport, ErrorReport> {
        self.run_tcp_action(
            RuntimePartitionWorkerActionKind::EvalCombinational,
            endpoints,
            transport,
        )
    }

    pub fn tick_tcp(
        &mut self,
        endpoints: &HashMap<String, SocketAddr>,
        transport: TcpRuntimeTransportConfig,
    ) -> Result<RuntimePartitionSessionActionReport, ErrorReport> {
        self.run_tcp_action(RuntimePartitionWorkerActionKind::Tick, endpoints, transport)
    }

    pub fn tick_many_tcp(
        &mut self,
        steps: usize,
        endpoints: &HashMap<String, SocketAddr>,
        transport: TcpRuntimeTransportConfig,
    ) -> Result<RuntimePartitionSessionActionReport, ErrorReport> {
        self.run_tcp_action(
            RuntimePartitionWorkerActionKind::TickMany(steps),
            endpoints,
            transport,
        )
    }

    pub fn run_local_script<F>(
        &mut self,
        script: &RuntimePartitionSessionRunScript,
        service: &mut LocalRuntimeWorkerService,
        on_telemetry: F,
    ) -> Result<RuntimePartitionSessionRunReport, ErrorReport>
    where
        F: FnMut(
            RuntimePartitionSessionRunEvent,
            &RuntimePartitionSessionTelemetry,
        ) -> Result<(), ErrorReport>,
    {
        self.run_script_with_telemetry(
            script,
            |session, kind| session.run_local_action(kind, service),
            on_telemetry,
        )
    }

    pub fn run_loopback_script<F>(
        &mut self,
        script: &RuntimePartitionSessionRunScript,
        service: Arc<Mutex<LocalRuntimeWorkerService>>,
        on_telemetry: F,
    ) -> Result<RuntimePartitionSessionRunReport, ErrorReport>
    where
        F: FnMut(
            RuntimePartitionSessionRunEvent,
            &RuntimePartitionSessionTelemetry,
        ) -> Result<(), ErrorReport>,
    {
        self.run_script_with_telemetry(
            script,
            |session, kind| session.run_loopback_action(kind, service.clone()),
            on_telemetry,
        )
    }

    pub fn run_tcp_script<F>(
        &mut self,
        script: &RuntimePartitionSessionRunScript,
        endpoints: &HashMap<String, SocketAddr>,
        transport: TcpRuntimeTransportConfig,
        on_telemetry: F,
    ) -> Result<RuntimePartitionSessionRunReport, ErrorReport>
    where
        F: FnMut(
            RuntimePartitionSessionRunEvent,
            &RuntimePartitionSessionTelemetry,
        ) -> Result<(), ErrorReport>,
    {
        self.run_script_with_telemetry(
            script,
            |session, kind| session.run_tcp_action(kind, endpoints, transport),
            on_telemetry,
        )
    }

    pub fn publish_route_outbound_local(
        &mut self,
        worker_id: &str,
        payload: RuntimePartitionRoutePayload,
        service: &mut LocalRuntimeWorkerService,
    ) -> Result<RuntimePartitionRouteMailboxReport, ErrorReport> {
        let report = expect_partition_route_mailbox(
            worker_id,
            service.handle(RuntimeWorkerRequest::StorePartitionRouteOutbound {
                worker_id: worker_id.to_string(),
                payload,
            })?,
        )?;
        self.record_route_mailbox(report.clone());
        Ok(report)
    }

    pub fn deliver_route_inbound_local(
        &mut self,
        worker_id: &str,
        payload: RuntimePartitionRoutePayload,
        service: &mut LocalRuntimeWorkerService,
    ) -> Result<RuntimePartitionRouteMailboxReport, ErrorReport> {
        let report = expect_partition_route_mailbox(
            worker_id,
            service.handle(RuntimeWorkerRequest::DeliverPartitionRouteInbound {
                worker_id: worker_id.to_string(),
                payload,
            })?,
        )?;
        self.record_route_mailbox(report.clone());
        Ok(report)
    }

    pub fn collect_route_mailboxes_local(
        &mut self,
        service: &mut LocalRuntimeWorkerService,
    ) -> Result<Vec<RuntimePartitionRouteMailboxReport>, ErrorReport> {
        let worker_ids = self
            .launch
            .workers
            .iter()
            .map(|worker| worker.worker_id.clone())
            .collect::<Vec<_>>();
        let mut reports = Vec::with_capacity(worker_ids.len());
        for worker_id in worker_ids {
            let report = expect_partition_route_mailbox(
                worker_id.as_str(),
                service.handle(RuntimeWorkerRequest::PartitionRouteMailbox {
                    worker_id: worker_id.clone(),
                })?,
            )?;
            self.record_route_mailbox(report.clone());
            reports.push(report);
        }
        Ok(reports)
    }

    pub fn clear_route_mailboxes_local(
        &mut self,
        service: &mut LocalRuntimeWorkerService,
    ) -> Result<Vec<RuntimePartitionRouteMailboxReport>, ErrorReport> {
        let worker_ids = self
            .launch
            .workers
            .iter()
            .map(|worker| worker.worker_id.clone())
            .collect::<Vec<_>>();
        let mut reports = Vec::with_capacity(worker_ids.len());
        for worker_id in worker_ids {
            let report = expect_partition_route_mailbox(
                worker_id.as_str(),
                service.handle(RuntimeWorkerRequest::ClearPartitionRouteMailboxes {
                    worker_id: worker_id.clone(),
                })?,
            )?;
            self.record_route_mailbox(report.clone());
            reports.push(report);
        }
        Ok(reports)
    }

    pub fn publish_route_outbound_loopback(
        &mut self,
        worker_id: &str,
        payload: RuntimePartitionRoutePayload,
        service: Arc<Mutex<LocalRuntimeWorkerService>>,
    ) -> Result<RuntimePartitionRouteMailboxReport, ErrorReport> {
        let mut client = LoopbackRuntimeShardClient::new(worker_id.to_string(), service);
        self.publish_route_outbound_client(worker_id, payload, &mut client)
    }

    pub fn deliver_route_inbound_loopback(
        &mut self,
        worker_id: &str,
        payload: RuntimePartitionRoutePayload,
        service: Arc<Mutex<LocalRuntimeWorkerService>>,
    ) -> Result<RuntimePartitionRouteMailboxReport, ErrorReport> {
        let mut client = LoopbackRuntimeShardClient::new(worker_id.to_string(), service);
        self.deliver_route_inbound_client(worker_id, payload, &mut client)
    }

    pub fn collect_route_mailboxes_loopback(
        &mut self,
        service: Arc<Mutex<LocalRuntimeWorkerService>>,
    ) -> Result<Vec<RuntimePartitionRouteMailboxReport>, ErrorReport> {
        let worker_ids = self
            .launch
            .workers
            .iter()
            .map(|worker| worker.worker_id.clone())
            .collect::<Vec<_>>();
        let mut reports = Vec::with_capacity(worker_ids.len());
        for worker_id in worker_ids {
            let mut client = LoopbackRuntimeShardClient::new(worker_id.clone(), service.clone());
            reports.push(self.collect_route_mailbox_client(worker_id.as_str(), &mut client)?);
        }
        Ok(reports)
    }

    pub fn clear_route_mailboxes_loopback(
        &mut self,
        service: Arc<Mutex<LocalRuntimeWorkerService>>,
    ) -> Result<Vec<RuntimePartitionRouteMailboxReport>, ErrorReport> {
        let worker_ids = self
            .launch
            .workers
            .iter()
            .map(|worker| worker.worker_id.clone())
            .collect::<Vec<_>>();
        let mut reports = Vec::with_capacity(worker_ids.len());
        for worker_id in worker_ids {
            let mut client = LoopbackRuntimeShardClient::new(worker_id.clone(), service.clone());
            reports.push(self.clear_route_mailbox_client(worker_id.as_str(), &mut client)?);
        }
        Ok(reports)
    }

    pub fn publish_route_outbound_tcp(
        &mut self,
        worker_id: &str,
        payload: RuntimePartitionRoutePayload,
        endpoint: SocketAddr,
        transport: TcpRuntimeTransportConfig,
    ) -> Result<RuntimePartitionRouteMailboxReport, ErrorReport> {
        let mut client =
            TcpRuntimeShardClient::connect_with_config(worker_id.to_string(), endpoint, transport)?;
        self.publish_route_outbound_client(worker_id, payload, &mut client)
    }

    pub fn deliver_route_inbound_tcp(
        &mut self,
        worker_id: &str,
        payload: RuntimePartitionRoutePayload,
        endpoint: SocketAddr,
        transport: TcpRuntimeTransportConfig,
    ) -> Result<RuntimePartitionRouteMailboxReport, ErrorReport> {
        let mut client =
            TcpRuntimeShardClient::connect_with_config(worker_id.to_string(), endpoint, transport)?;
        self.deliver_route_inbound_client(worker_id, payload, &mut client)
    }

    pub fn collect_route_mailboxes_tcp(
        &mut self,
        endpoints: &HashMap<String, SocketAddr>,
        transport: TcpRuntimeTransportConfig,
    ) -> Result<Vec<RuntimePartitionRouteMailboxReport>, ErrorReport> {
        let worker_ids = self
            .launch
            .workers
            .iter()
            .map(|worker| worker.worker_id.clone())
            .collect::<Vec<_>>();
        let mut reports = Vec::with_capacity(worker_ids.len());
        for worker_id in worker_ids {
            let endpoint = partition_session_worker_endpoint(&worker_id, endpoints)?;
            let mut client =
                TcpRuntimeShardClient::connect_with_config(worker_id.clone(), endpoint, transport)?;
            reports.push(self.collect_route_mailbox_client(worker_id.as_str(), &mut client)?);
        }
        Ok(reports)
    }

    pub fn clear_route_mailboxes_tcp(
        &mut self,
        endpoints: &HashMap<String, SocketAddr>,
        transport: TcpRuntimeTransportConfig,
    ) -> Result<Vec<RuntimePartitionRouteMailboxReport>, ErrorReport> {
        let worker_ids = self
            .launch
            .workers
            .iter()
            .map(|worker| worker.worker_id.clone())
            .collect::<Vec<_>>();
        let mut reports = Vec::with_capacity(worker_ids.len());
        for worker_id in worker_ids {
            let endpoint = partition_session_worker_endpoint(&worker_id, endpoints)?;
            let mut client =
                TcpRuntimeShardClient::connect_with_config(worker_id.clone(), endpoint, transport)?;
            reports.push(self.clear_route_mailbox_client(worker_id.as_str(), &mut client)?);
        }
        Ok(reports)
    }

    pub fn transfer_route_local(
        &mut self,
        route_index: usize,
        limbs: Vec<u32>,
        service: &mut LocalRuntimeWorkerService,
    ) -> Result<RuntimePartitionRouteTransferReport, ErrorReport> {
        let route = self.transfer_route(route_index)?;
        let payload = RuntimePartitionRoutePayload {
            route_index,
            width: route.width,
            limbs,
        };
        let producer_mailbox = self.publish_route_outbound_local(
            route.producer_worker.as_str(),
            payload.clone(),
            service,
        )?;
        let consumer_mailbox =
            self.deliver_route_inbound_local(route.consumer_worker.as_str(), payload, service)?;
        Ok(runtime_partition_route_transfer_report(
            &route,
            producer_mailbox,
            consumer_mailbox,
            &self.deployment,
        ))
    }

    pub fn transfer_route_loopback(
        &mut self,
        route_index: usize,
        limbs: Vec<u32>,
        service: Arc<Mutex<LocalRuntimeWorkerService>>,
    ) -> Result<RuntimePartitionRouteTransferReport, ErrorReport> {
        let route = self.transfer_route(route_index)?;
        let payload = RuntimePartitionRoutePayload {
            route_index,
            width: route.width,
            limbs,
        };
        let producer_mailbox = self.publish_route_outbound_loopback(
            route.producer_worker.as_str(),
            payload.clone(),
            service.clone(),
        )?;
        let consumer_mailbox =
            self.deliver_route_inbound_loopback(route.consumer_worker.as_str(), payload, service)?;
        Ok(runtime_partition_route_transfer_report(
            &route,
            producer_mailbox,
            consumer_mailbox,
            &self.deployment,
        ))
    }

    pub fn transfer_route_tcp(
        &mut self,
        route_index: usize,
        limbs: Vec<u32>,
        endpoints: &HashMap<String, SocketAddr>,
        transport: TcpRuntimeTransportConfig,
    ) -> Result<RuntimePartitionRouteTransferReport, ErrorReport> {
        let route = self.transfer_route(route_index)?;
        let producer_endpoint =
            partition_session_worker_endpoint(route.producer_worker.as_str(), endpoints)?;
        let consumer_endpoint =
            partition_session_worker_endpoint(route.consumer_worker.as_str(), endpoints)?;
        let payload = RuntimePartitionRoutePayload {
            route_index,
            width: route.width,
            limbs,
        };
        let producer_mailbox = self.publish_route_outbound_tcp(
            route.producer_worker.as_str(),
            payload.clone(),
            producer_endpoint,
            transport,
        )?;
        let consumer_mailbox = self.deliver_route_inbound_tcp(
            route.consumer_worker.as_str(),
            payload,
            consumer_endpoint,
            transport,
        )?;
        Ok(runtime_partition_route_transfer_report(
            &route,
            producer_mailbox,
            consumer_mailbox,
            &self.deployment,
        ))
    }

    fn record_action_report(
        &mut self,
        kind: RuntimePartitionWorkerActionKind,
        workers: Vec<RuntimePartitionWorkerActionReport>,
    ) -> RuntimePartitionSessionActionReport {
        let report = runtime_partition_session_action_report(self, kind, workers);
        self.actions.push(report.clone());
        report
    }

    fn publish_route_outbound_client(
        &mut self,
        worker_id: &str,
        payload: RuntimePartitionRoutePayload,
        client: &mut impl RuntimeShardClient,
    ) -> Result<RuntimePartitionRouteMailboxReport, ErrorReport> {
        let report = expect_partition_route_mailbox(
            worker_id,
            client.request(RuntimeWorkerRequest::StorePartitionRouteOutbound {
                worker_id: worker_id.to_string(),
                payload,
            })?,
        )?;
        self.record_route_mailbox(report.clone());
        Ok(report)
    }

    fn deliver_route_inbound_client(
        &mut self,
        worker_id: &str,
        payload: RuntimePartitionRoutePayload,
        client: &mut impl RuntimeShardClient,
    ) -> Result<RuntimePartitionRouteMailboxReport, ErrorReport> {
        let report = expect_partition_route_mailbox(
            worker_id,
            client.request(RuntimeWorkerRequest::DeliverPartitionRouteInbound {
                worker_id: worker_id.to_string(),
                payload,
            })?,
        )?;
        self.record_route_mailbox(report.clone());
        Ok(report)
    }

    fn collect_route_mailbox_client(
        &mut self,
        worker_id: &str,
        client: &mut impl RuntimeShardClient,
    ) -> Result<RuntimePartitionRouteMailboxReport, ErrorReport> {
        let report = expect_partition_route_mailbox(
            worker_id,
            client.request(RuntimeWorkerRequest::PartitionRouteMailbox {
                worker_id: worker_id.to_string(),
            })?,
        )?;
        self.record_route_mailbox(report.clone());
        Ok(report)
    }

    fn clear_route_mailbox_client(
        &mut self,
        worker_id: &str,
        client: &mut impl RuntimeShardClient,
    ) -> Result<RuntimePartitionRouteMailboxReport, ErrorReport> {
        let report = expect_partition_route_mailbox(
            worker_id,
            client.request(RuntimeWorkerRequest::ClearPartitionRouteMailboxes {
                worker_id: worker_id.to_string(),
            })?,
        )?;
        self.record_route_mailbox(report.clone());
        Ok(report)
    }

    fn record_route_mailbox(&mut self, report: RuntimePartitionRouteMailboxReport) {
        if let Some(existing) = self
            .route_mailboxes
            .iter_mut()
            .find(|mailbox| mailbox.worker_id == report.worker_id)
        {
            *existing = report;
        } else {
            self.route_mailboxes.push(report);
            self.route_mailboxes
                .sort_by(|left, right| left.worker_id.cmp(&right.worker_id));
        }
    }

    fn transfer_route(
        &self,
        route_index: usize,
    ) -> Result<RuntimePartitionLaunchRoute, ErrorReport> {
        let route = self
            .launch
            .routes
            .iter()
            .find(|route| route.route_index == route_index)
            .ok_or_else(|| {
                error(
                    "E_RUNTIME_PARTITION_ROUTE_TRANSFER",
                    format!(
                        "runtime partition session `{}` has no route {route_index}",
                        self.launch.module_name
                    ),
                )
            })?;
        if route.producer_worker == route.consumer_worker {
            return Err(error(
                "E_RUNTIME_PARTITION_ROUTE_TRANSFER",
                format!(
                    "runtime partition route {route_index} stays on worker `{}` and cannot be transferred across workers",
                    route.producer_worker
                ),
            ));
        }
        Ok(route.clone())
    }

    fn run_script_with_telemetry<A, T>(
        &mut self,
        script: &RuntimePartitionSessionRunScript,
        mut run_action: A,
        mut on_telemetry: T,
    ) -> Result<RuntimePartitionSessionRunReport, ErrorReport>
    where
        A: FnMut(
            &mut RuntimePartitionSession,
            RuntimePartitionWorkerActionKind,
        ) -> Result<RuntimePartitionSessionActionReport, ErrorReport>,
        T: FnMut(
            RuntimePartitionSessionRunEvent,
            &RuntimePartitionSessionTelemetry,
        ) -> Result<(), ErrorReport>,
    {
        validate_partition_session_run_script(script)?;

        let mut completed_actions = 0;
        let mut telemetry_emitted = 0;
        let mut last_telemetry_action = None;

        if script.emit_initial {
            self.emit_run_script_telemetry(
                RuntimePartitionSessionRunEventReason::Initial,
                completed_actions,
                &mut on_telemetry,
            )?;
            telemetry_emitted += 1;
            last_telemetry_action = Some(completed_actions);
        }

        for kind in &script.actions {
            run_action(self, *kind)?;
            completed_actions += 1;

            if completed_actions % script.every_actions == 0 {
                self.emit_run_script_telemetry(
                    RuntimePartitionSessionRunEventReason::Cadence,
                    completed_actions,
                    &mut on_telemetry,
                )?;
                telemetry_emitted += 1;
                last_telemetry_action = Some(completed_actions);
            }
        }

        if script.emit_final && last_telemetry_action != Some(completed_actions) {
            self.emit_run_script_telemetry(
                RuntimePartitionSessionRunEventReason::Final,
                completed_actions,
                &mut on_telemetry,
            )?;
            telemetry_emitted += 1;
        }

        Ok(RuntimePartitionSessionRunReport {
            completed_actions,
            telemetry_emitted,
            action_summary: self.action_summary(),
            diagnostics: self.deployment.diagnostics.clone(),
        })
    }

    fn emit_run_script_telemetry<T>(
        &self,
        reason: RuntimePartitionSessionRunEventReason,
        completed_actions: usize,
        on_telemetry: &mut T,
    ) -> Result<(), ErrorReport>
    where
        T: FnMut(
            RuntimePartitionSessionRunEvent,
            &RuntimePartitionSessionTelemetry,
        ) -> Result<(), ErrorReport>,
    {
        let event = RuntimePartitionSessionRunEvent {
            completed_actions,
            reason,
            latest_action_kind: self.latest_action().map(|action| action.kind),
        };
        let telemetry = self.telemetry();
        on_telemetry(event, &telemetry)
    }
}

fn runtime_partition_session_health(
    launch: &RuntimePartitionLaunchPlan,
    deployment: &RuntimePartitionDeploymentReport,
) -> RuntimePartitionSessionHealth {
    let initialized_worker_count = deployment
        .workers
        .iter()
        .filter(|worker| worker.initialized)
        .count();
    let partition_count = deployment
        .workers
        .iter()
        .map(|worker| worker.partitions.len())
        .sum::<usize>();
    RuntimePartitionSessionHealth {
        module_name: launch.module_name.clone(),
        ready: initialized_worker_count == launch.workers.len(),
        worker_count: launch.workers.len(),
        initialized_worker_count,
        partition_count,
        route_count: launch.routes.len(),
        diagnostics: deployment.diagnostics.clone(),
        workers: deployment.workers.clone(),
    }
}

fn runtime_partition_initial_mailboxes(
    launch: &RuntimePartitionLaunchPlan,
    deployment: &RuntimePartitionDeploymentReport,
) -> Vec<RuntimePartitionRouteMailboxReport> {
    launch
        .workers
        .iter()
        .map(|worker| RuntimePartitionRouteMailboxReport {
            worker_id: worker.worker_id.clone(),
            outbound_payload_count: 0,
            inbound_payload_count: 0,
            outbound_routes: worker.outbound_routes.clone(),
            inbound_routes: worker.inbound_routes.clone(),
            diagnostics: deployment.diagnostics.clone(),
        })
        .collect()
}

fn runtime_partition_route_transfer_report(
    route: &RuntimePartitionLaunchRoute,
    producer_mailbox: RuntimePartitionRouteMailboxReport,
    consumer_mailbox: RuntimePartitionRouteMailboxReport,
    deployment: &RuntimePartitionDeploymentReport,
) -> RuntimePartitionRouteTransferReport {
    RuntimePartitionRouteTransferReport {
        route_index: route.route_index,
        producer_worker: route.producer_worker.clone(),
        consumer_worker: route.consumer_worker.clone(),
        width: route.width,
        limb_count: route_width_limbs(route.width),
        producer_mailbox,
        consumer_mailbox,
        diagnostics: deployment.diagnostics.clone(),
    }
}

fn runtime_partition_session_action_summary(
    launch: &RuntimePartitionLaunchPlan,
    deployment: &RuntimePartitionDeploymentReport,
    actions: &[RuntimePartitionSessionActionReport],
) -> RuntimePartitionSessionActionSummary {
    let mut operations = RuntimeOperationStats::default();
    for action in actions {
        operations += action.operations;
    }
    RuntimePartitionSessionActionSummary {
        module_name: launch.module_name.clone(),
        action_count: actions.len(),
        latest_action_kind: actions.last().map(|action| action.kind),
        operations,
        worker_count: launch.workers.len(),
        partition_count: deployment
            .workers
            .iter()
            .map(|worker| worker.partitions.len())
            .sum(),
        route_count: launch.routes.len(),
        diagnostics: deployment.diagnostics.clone(),
    }
}

fn validate_partition_session_run_script(
    script: &RuntimePartitionSessionRunScript,
) -> Result<(), ErrorReport> {
    if script.every_actions == 0 {
        return Err(error(
            "E_RUNTIME_PARTITION_SESSION_SCRIPT",
            "runtime partition session script every_actions must be greater than zero",
        ));
    }
    Ok(())
}

fn validate_partition_session_ready(session: &RuntimePartitionSession) -> Result<(), ErrorReport> {
    let health = session.health();
    if !health.ready {
        return Err(error(
            "E_RUNTIME_PARTITION_SESSION_ACTION",
            format!(
                "runtime partition session `{}` is not ready for structural actions",
                health.module_name
            ),
        ));
    }
    Ok(())
}

fn run_partition_worker_action(
    worker_id: &str,
    kind: RuntimePartitionWorkerActionKind,
    client: &mut impl RuntimeShardClient,
) -> Result<RuntimePartitionWorkerActionReport, ErrorReport> {
    expect_partition_worker_action(
        worker_id,
        kind,
        client.request(RuntimeWorkerRequest::RunPartitionWorkerAction(
            RuntimePartitionWorkerAction {
                worker_id: worker_id.to_string(),
                kind,
            },
        ))?,
    )
}

fn expect_partition_worker_action(
    expected_worker_id: &str,
    expected_kind: RuntimePartitionWorkerActionKind,
    response: RuntimeWorkerResponse,
) -> Result<RuntimePartitionWorkerActionReport, ErrorReport> {
    let RuntimeWorkerResponse::PartitionWorkerAction(report) = response else {
        return Err(unexpected_worker_response(
            "PartitionWorkerAction",
            response,
        ));
    };
    if report.worker_id != expected_worker_id || report.kind != expected_kind {
        return Err(error(
            "E_RUNTIME_PARTITION_SESSION_ACTION",
            format!(
                "partition worker action report for `{}`/{:?} does not match expected `{expected_worker_id}`/{:?}",
                report.worker_id, report.kind, expected_kind
            ),
        ));
    }
    Ok(report)
}

fn expect_partition_route_mailbox(
    expected_worker_id: &str,
    response: RuntimeWorkerResponse,
) -> Result<RuntimePartitionRouteMailboxReport, ErrorReport> {
    let RuntimeWorkerResponse::PartitionRouteMailbox(report) = response else {
        return Err(unexpected_worker_response(
            "PartitionRouteMailbox",
            response,
        ));
    };
    if report.worker_id != expected_worker_id {
        return Err(error(
            "E_RUNTIME_PARTITION_ROUTE_MAILBOX",
            format!(
                "partition route mailbox report for `{}` does not match expected `{expected_worker_id}`",
                report.worker_id
            ),
        ));
    }
    Ok(report)
}

fn partition_session_worker_endpoint(
    worker_id: &str,
    endpoints: &HashMap<String, SocketAddr>,
) -> Result<SocketAddr, ErrorReport> {
    endpoints.get(worker_id).copied().ok_or_else(|| {
        error(
            "E_RUNTIME_PARTITION_DEPLOY_ENDPOINT",
            format!("missing TCP endpoint for partition worker `{worker_id}`"),
        )
    })
}

fn runtime_partition_session_action_report(
    session: &RuntimePartitionSession,
    kind: RuntimePartitionWorkerActionKind,
    workers: Vec<RuntimePartitionWorkerActionReport>,
) -> RuntimePartitionSessionActionReport {
    let mut operations = RuntimeOperationStats::default();
    for worker in &workers {
        operations.eval_combinational.calls += worker.operations.eval_combinational.calls;
        operations.eval_combinational.total_ns += worker.operations.eval_combinational.total_ns;
        operations.eval_combinational.last_ns = worker.operations.eval_combinational.last_ns;
        operations.tick.calls += worker.operations.tick.calls;
        operations.tick.total_ns += worker.operations.tick.total_ns;
        operations.tick.last_ns = worker.operations.tick.last_ns;
        operations.tick_many.calls += worker.operations.tick_many.calls;
        operations.tick_many.total_ns += worker.operations.tick_many.total_ns;
        operations.tick_many.last_ns = worker.operations.tick_many.last_ns;
    }
    RuntimePartitionSessionActionReport {
        module_name: session.launch.module_name.clone(),
        kind,
        workers,
        operations,
        diagnostics: session.deployment.diagnostics.clone(),
    }
}

fn deploy_runtime_partition_worker(
    worker: &RuntimePartitionWorkerLaunch,
    client: &mut impl RuntimeShardClient,
) -> Result<RuntimePartitionWorkerHealth, ErrorReport> {
    expect_ack(client.request(RuntimeWorkerRequest::InitPartitionWorker(
        RuntimePartitionWorkerInit {
            worker_id: worker.worker_id.clone(),
            launch: worker.clone(),
        },
    ))?)?;
    let health = expect_partition_worker_health(
        worker.worker_id.as_str(),
        client.request(RuntimeWorkerRequest::PartitionWorkerHealth {
            worker_id: worker.worker_id.clone(),
        })?,
    )?;
    validate_partition_deployment_health(worker, &health)?;
    Ok(health)
}

fn validate_partition_deployment_launch(
    launch: &RuntimePartitionLaunchPlan,
) -> Result<(), ErrorReport> {
    launch.validate_format_version()?;
    let mut worker_ids = HashMap::new();
    for worker in &launch.workers {
        if worker_ids.insert(worker.worker_id.as_str(), ()).is_some() {
            return Err(error(
                "E_RUNTIME_PARTITION_DEPLOY_WORKER",
                format!(
                    "partition launch has duplicate worker `{}`",
                    worker.worker_id
                ),
            ));
        }
    }
    Ok(())
}

fn expect_partition_worker_health(
    expected_worker_id: &str,
    response: RuntimeWorkerResponse,
) -> Result<RuntimePartitionWorkerHealth, ErrorReport> {
    let RuntimeWorkerResponse::PartitionWorkerHealth(health) = response else {
        return Err(unexpected_worker_response(
            "PartitionWorkerHealth",
            response,
        ));
    };
    if health.worker_id != expected_worker_id {
        return Err(error(
            "E_RUNTIME_PARTITION_DEPLOY_WORKER",
            format!(
                "partition worker health for `{}` was returned for expected worker `{expected_worker_id}`",
                health.worker_id
            ),
        ));
    }
    Ok(health)
}

fn validate_partition_deployment_health(
    worker: &RuntimePartitionWorkerLaunch,
    health: &RuntimePartitionWorkerHealth,
) -> Result<(), ErrorReport> {
    if !health.initialized {
        return Err(error(
            "E_RUNTIME_PARTITION_DEPLOY_HEALTH",
            format!(
                "partition worker `{}` did not report initialized after deployment",
                worker.worker_id
            ),
        ));
    }
    if health.backend != Some(worker.backend)
        || health.node.as_deref() != Some(worker.node.as_str())
        || health.partitions != worker.partitions
        || health.outbound_routes != worker.outbound_routes
        || health.inbound_routes != worker.inbound_routes
        || health.outbound_bits != worker.outbound_bits
        || health.inbound_bits != worker.inbound_bits
    {
        return Err(error(
            "E_RUNTIME_PARTITION_DEPLOY_HEALTH",
            format!(
                "partition worker `{}` health does not match launch payload",
                worker.worker_id
            ),
        ));
    }
    Ok(())
}

fn validate_partition_launch_bundle(bundle: &RuntimePartitionBundle) -> Result<(), ErrorReport> {
    bundle.validate_format_version()?;
    bundle.partition_plan.validate_format_version()?;
    bundle.placement_plan.validate_format_version()?;
    bundle.communication_plan.validate_format_version()?;
    validate_partition_bundle_modules(
        &bundle.partition_plan,
        &bundle.placement_plan,
        &bundle.communication_plan,
    )?;
    Ok(())
}

fn runtime_partition_launch_diagnostics(
    bundle: &RuntimePartitionBundle,
    workers: &[RuntimePartitionWorkerLaunch],
) -> Vec<RuntimePartitionLaunchDiagnostic> {
    let mut diagnostics = bundle
        .diagnostics
        .iter()
        .map(|diagnostic| RuntimePartitionLaunchDiagnostic {
            source: match diagnostic.source {
                RuntimePartitionBundleSource::Partition => {
                    RuntimePartitionLaunchDiagnosticSource::BundlePartition
                }
                RuntimePartitionBundleSource::Placement => {
                    RuntimePartitionLaunchDiagnosticSource::BundlePlacement
                }
                RuntimePartitionBundleSource::Communication => {
                    RuntimePartitionLaunchDiagnosticSource::BundleCommunication
                }
            },
            code: diagnostic.code.clone(),
            severity: diagnostic.severity,
            message: diagnostic.message.clone(),
            worker_id: None,
            partition_id: None,
            route_index: None,
        })
        .collect::<Vec<_>>();

    diagnostics.extend(workers.iter().filter_map(|worker| {
        if worker.partitions.is_empty() {
            Some(RuntimePartitionLaunchDiagnostic {
                source: RuntimePartitionLaunchDiagnosticSource::Launch,
                code: "EmptyWorkerLaunch".to_string(),
                severity: RuntimePartitionDiagnosticSeverity::Info,
                message: format!(
                    "worker `{}` has no structural partitions assigned in this launch plan",
                    worker.worker_id
                ),
                worker_id: Some(worker.worker_id.clone()),
                partition_id: None,
                route_index: None,
            })
        } else {
            None
        }
    }));

    diagnostics
}

fn validate_partition_bundle_modules(
    partition_plan: &RuntimePartitionPlan,
    placement_plan: &RuntimePartitionPlacementPlan,
    communication_plan: &RuntimePartitionCommunicationPlan,
) -> Result<(), ErrorReport> {
    if partition_plan.module_name != placement_plan.module_name {
        return Err(error(
            "E_RUNTIME_PARTITION_BUNDLE_MODULE",
            format!(
                "partition plan module `{}` does not match placement module `{}`",
                partition_plan.module_name, placement_plan.module_name
            ),
        ));
    }
    if partition_plan.module_name != communication_plan.module_name {
        return Err(error(
            "E_RUNTIME_PARTITION_BUNDLE_MODULE",
            format!(
                "partition plan module `{}` does not match communication module `{}`",
                partition_plan.module_name, communication_plan.module_name
            ),
        ));
    }
    Ok(())
}

fn validate_partition_bundle_routes(
    placement_plan: &RuntimePartitionPlacementPlan,
    communication_plan: &RuntimePartitionCommunicationPlan,
) -> Result<(), ErrorReport> {
    let assignment_by_partition = placement_plan
        .assignments
        .iter()
        .map(|assignment| (assignment.partition_id.as_str(), assignment))
        .collect::<HashMap<_, _>>();

    for route in &communication_plan.routes {
        let producer = assignment_by_partition
            .get(route.producer_partition.as_str())
            .ok_or_else(|| {
                error(
                    "E_RUNTIME_PARTITION_BUNDLE_ASSIGNMENT",
                    format!(
                        "communication route {} references producer partition `{}` with no placement assignment",
                        route.boundary_index, route.producer_partition
                    ),
                )
            })?;
        let consumer = assignment_by_partition
            .get(route.consumer_partition.as_str())
            .ok_or_else(|| {
                error(
                    "E_RUNTIME_PARTITION_BUNDLE_ASSIGNMENT",
                    format!(
                        "communication route {} references consumer partition `{}` with no placement assignment",
                        route.boundary_index, route.consumer_partition
                    ),
                )
            })?;
        if producer.worker_id != route.producer_worker
            || consumer.worker_id != route.consumer_worker
        {
            return Err(error(
                "E_RUNTIME_PARTITION_BUNDLE_ROUTE",
                format!(
                    "communication route {} worker mapping is stale for producer `{}` or consumer `{}`",
                    route.boundary_index, route.producer_partition, route.consumer_partition
                ),
            ));
        }
    }

    Ok(())
}

fn runtime_partition_bundle_diagnostics(
    partition_plan: &RuntimePartitionPlan,
    placement_plan: &RuntimePartitionPlacementPlan,
    communication_plan: &RuntimePartitionCommunicationPlan,
) -> Vec<RuntimePartitionBundleDiagnostic> {
    let mut diagnostics = Vec::new();
    diagnostics.extend(partition_plan.diagnostics.iter().map(|diagnostic| {
        RuntimePartitionBundleDiagnostic {
            source: RuntimePartitionBundleSource::Partition,
            code: format!("{:?}", diagnostic.code),
            severity: diagnostic.severity,
            message: diagnostic.message.clone(),
        }
    }));
    diagnostics.extend(placement_plan.diagnostics.iter().map(|diagnostic| {
        RuntimePartitionBundleDiagnostic {
            source: RuntimePartitionBundleSource::Placement,
            code: format!("{:?}", diagnostic.code),
            severity: diagnostic.severity,
            message: diagnostic.message.clone(),
        }
    }));
    diagnostics.extend(communication_plan.diagnostics.iter().map(|diagnostic| {
        RuntimePartitionBundleDiagnostic {
            source: RuntimePartitionBundleSource::Communication,
            code: format!("{:?}", diagnostic.code),
            severity: diagnostic.severity,
            message: diagnostic.message.clone(),
        }
    }));
    diagnostics
}

fn runtime_partition_bundle_recommendations(
    partition_plan: &RuntimePartitionPlan,
    placement_plan: &RuntimePartitionPlacementPlan,
    communication_plan: &RuntimePartitionCommunicationPlan,
) -> Vec<RuntimePartitionBundleRecommendation> {
    let mut recommendations = Vec::new();
    recommendations.extend(partition_plan.recommendations.iter().map(|recommendation| {
        RuntimePartitionBundleRecommendation {
            source: RuntimePartitionBundleSource::Partition,
            code: format!("{:?}", recommendation.code),
            message: recommendation.message.clone(),
        }
    }));
    recommendations.extend(placement_plan.recommendations.iter().map(|recommendation| {
        RuntimePartitionBundleRecommendation {
            source: RuntimePartitionBundleSource::Placement,
            code: format!("{:?}", recommendation.code),
            message: recommendation.message.clone(),
        }
    }));
    recommendations.extend(
        communication_plan
            .recommendations
            .iter()
            .map(|recommendation| RuntimePartitionBundleRecommendation {
                source: RuntimePartitionBundleSource::Communication,
                code: format!("{:?}", recommendation.code),
                message: recommendation.message.clone(),
            }),
    );
    recommendations
}

fn communication_assignment<'a>(
    assignments: &HashMap<&str, &'a RuntimePartitionAssignment>,
    partition_id: &str,
    boundary_index: usize,
) -> Result<&'a RuntimePartitionAssignment, ErrorReport> {
    assignments.get(partition_id).copied().ok_or_else(|| {
        error(
            "E_RUNTIME_PARTITION_COMMUNICATION_ASSIGNMENT",
            format!(
                "partition `{partition_id}` referenced by boundary {boundary_index} has no placement assignment"
            ),
        )
    })
}

fn communication_worker_summaries(
    placement: &RuntimePartitionPlacementPlan,
    routes: &[RuntimePartitionRoute],
) -> Vec<RuntimePartitionCommunicationWorkerSummary> {
    let mut summaries = placement
        .worker_summaries
        .iter()
        .map(|worker| RuntimePartitionCommunicationWorkerSummary {
            worker_id: worker.worker_id.clone(),
            outbound_routes: Vec::new(),
            inbound_routes: Vec::new(),
            outbound_bits: 0,
            inbound_bits: 0,
        })
        .collect::<Vec<_>>();
    let worker_index_by_id = summaries
        .iter()
        .enumerate()
        .map(|(index, summary)| (summary.worker_id.clone(), index))
        .collect::<HashMap<_, _>>();

    for (route_index, route) in routes.iter().enumerate() {
        if let Some(index) = worker_index_by_id.get(&route.producer_worker) {
            summaries[*index].outbound_routes.push(route_index);
            summaries[*index].outbound_bits += route.bits_per_transfer;
        }
        if let Some(index) = worker_index_by_id.get(&route.consumer_worker) {
            summaries[*index].inbound_routes.push(route_index);
            summaries[*index].inbound_bits += route.bits_per_transfer;
        }
    }
    summaries
}

fn analyze_partition_communication(
    plan: &RuntimePartitionPlan,
    routes: &[RuntimePartitionRoute],
    total_cross_worker_boundary_bits: u64,
    diagnostics: &mut Vec<RuntimePartitionCommunicationDiagnostic>,
    recommendations: &mut Vec<RuntimePartitionCommunicationRecommendation>,
) {
    let local_bits = plan
        .total_cost
        .state_bits
        .saturating_add(plan.total_cost.memory_bits);
    let total_traffic_is_high = total_cross_worker_boundary_bits > 0
        && local_bits > 0
        && total_cross_worker_boundary_bits > local_bits;
    if total_traffic_is_high {
        diagnostics.push(RuntimePartitionCommunicationDiagnostic {
            code: RuntimePartitionCommunicationDiagnosticCode::HighTotalCrossWorkerTraffic,
            severity: RuntimePartitionDiagnosticSeverity::Warning,
            message: format!(
                "communication schedule moves {total_cross_worker_boundary_bits} cross-worker bits per transfer, exceeding local state plus memory bits"
            ),
            worker_id: None,
            partition_id: None,
            boundary_index: None,
            route_index: None,
        });
    }

    for (route_index, route) in routes.iter().enumerate() {
        if route.width < 1024 && !total_traffic_is_high {
            continue;
        }
        diagnostics.push(RuntimePartitionCommunicationDiagnostic {
            code: RuntimePartitionCommunicationDiagnosticCode::HighRouteWidth,
            severity: RuntimePartitionDiagnosticSeverity::Warning,
            message: format!(
                "route `{}` -> `{}` for `{}` moves {} bits per transfer",
                route.producer_worker,
                route.consumer_worker,
                route.instance_path.join("."),
                route.bits_per_transfer
            ),
            worker_id: Some(route.producer_worker.clone()),
            partition_id: Some(route.producer_partition.clone()),
            boundary_index: Some(route.boundary_index),
            route_index: Some(route_index),
        });
        recommendations.push(RuntimePartitionCommunicationRecommendation {
            code: RuntimePartitionCommunicationRecommendationCode::CoLocateRoute,
            message: format!(
                "co-locate partitions `{}` and `{}` or cut `{}` on a narrower interface",
                route.producer_partition,
                route.consumer_partition,
                route.instance_path.join(".")
            ),
            worker_id: Some(route.producer_worker.clone()),
            partition_id: Some(route.producer_partition.clone()),
            boundary_index: Some(route.boundary_index),
            route_index: Some(route_index),
        });
    }
}

#[derive(Clone, Debug)]
struct PlacementWorkerLoad {
    worker_id: String,
    backend: RuntimeBackend,
    node: String,
    partitions: Vec<String>,
    cost: RuntimePartitionCost,
    inbound_boundary_bits: u64,
    outbound_boundary_bits: u64,
}

impl PlacementWorkerLoad {
    fn from_worker(worker: &RuntimeWorker) -> Self {
        Self {
            worker_id: worker.id.clone(),
            backend: worker.backend.clone(),
            node: worker.node.clone(),
            partitions: Vec::new(),
            cost: RuntimePartitionCost::default(),
            inbound_boundary_bits: 0,
            outbound_boundary_bits: 0,
        }
    }

    fn is_cpu(&self) -> bool {
        !matches!(self.backend, RuntimeBackend::Gpu(_))
    }

    fn is_gpu(&self) -> bool {
        matches!(self.backend, RuntimeBackend::Gpu(_))
    }

    fn into_summary(self) -> RuntimePartitionWorkerSummary {
        RuntimePartitionWorkerSummary {
            worker_id: self.worker_id,
            backend: self.backend,
            node: self.node,
            partitions: self.partitions,
            cost: self.cost,
            inbound_boundary_bits: self.inbound_boundary_bits,
            outbound_boundary_bits: self.outbound_boundary_bits,
        }
    }
}

fn attach_gemm_runtime_plan_to_topology(
    topology: &RuntimeTopology,
    plan: &GemmRuntimePlan,
) -> Result<RuntimeSurrogateAttachment, ErrorReport> {
    let mut start_lane = 0;
    let mut shards = Vec::with_capacity(topology.workers.len());
    for worker in &topology.workers {
        shards.push(RuntimeShardInfo {
            worker_id: worker.id.clone(),
            node: worker.node.clone(),
            backend: worker.backend,
            start_lane,
            lanes: worker.lanes,
        });
        start_lane += worker.lanes;
    }
    attach_gemm_runtime_plan_to_shard_infos(&shards, plan)
}

fn attach_event_runtime_plan_to_topology(
    topology: &RuntimeTopology,
    plan: &EventRuntimePlan,
) -> Result<RuntimeSurrogateAttachment, ErrorReport> {
    let mut start_lane = 0;
    let mut shards = Vec::with_capacity(topology.workers.len());
    for worker in &topology.workers {
        shards.push(RuntimeShardInfo {
            worker_id: worker.id.clone(),
            node: worker.node.clone(),
            backend: worker.backend,
            start_lane,
            lanes: worker.lanes,
        });
        start_lane += worker.lanes;
    }
    attach_event_runtime_plan_to_shard_infos(&shards, plan)
}

fn attach_gemm_runtime_plan_to_shards(
    shards: &[RuntimeShard],
    plan: &GemmRuntimePlan,
) -> Result<RuntimeSurrogateAttachment, ErrorReport> {
    let infos = shards
        .iter()
        .map(|shard| shard.info.clone())
        .collect::<Vec<_>>();
    attach_gemm_runtime_plan_to_shard_infos(&infos, plan)
}

fn attach_gemm_runtime_plan_to_shard_infos(
    shards: &[RuntimeShardInfo],
    plan: &GemmRuntimePlan,
) -> Result<RuntimeSurrogateAttachment, ErrorReport> {
    if plan.schema != GEMM_RUNTIME_PLAN_SCHEMA {
        return Err(error(
            "E_RUNTIME_SURROGATE_ATTACHMENT_SCHEMA",
            format!(
                "unsupported GEMM runtime plan schema `{}`, expected `{}`",
                plan.schema, GEMM_RUNTIME_PLAN_SCHEMA
            ),
        ));
    }
    if !plan.ok {
        return Err(error(
            "E_RUNTIME_SURROGATE_ATTACHMENT_PLAN",
            "cannot attach surrogate runtime plan that is not ok",
        ));
    }
    if !plan.errors.is_empty() {
        return Err(error(
            "E_RUNTIME_SURROGATE_ATTACHMENT_PLAN",
            "cannot attach surrogate runtime plan with validation errors",
        ));
    }

    let total_lanes = shards.iter().map(|shard| shard.lanes).sum::<usize>();
    if plan.total_lanes != total_lanes {
        return Err(error(
            "E_RUNTIME_SURROGATE_ATTACHMENT_LANES",
            format!(
                "surrogate plan total lanes {} does not match runtime total lanes {}",
                plan.total_lanes, total_lanes
            ),
        ));
    }

    let mut shard_by_worker = HashMap::new();
    for shard in shards {
        if shard.lanes == 0 {
            return Err(error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_WORKER",
                format!("runtime worker `{}` owns zero lanes", shard.worker_id),
            ));
        }
        if shard_by_worker
            .insert(shard.worker_id.clone(), shard)
            .is_some()
        {
            return Err(error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_WORKER",
                format!(
                    "runtime worker `{}` appears more than once",
                    shard.worker_id
                ),
            ));
        }
    }

    let mut plan_worker_by_id = HashMap::new();
    for worker in &plan.workers {
        let shard = shard_by_worker.get(&worker.worker_id).ok_or_else(|| {
            error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_WORKER",
                format!(
                    "surrogate plan references unknown runtime worker `{}`",
                    worker.worker_id
                ),
            )
        })?;
        if worker.start_lane != shard.start_lane || worker.lanes != shard.lanes {
            return Err(error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_WORKER",
                format!(
                    "surrogate worker `{}` maps lanes {}..{}, runtime maps lanes {}..{}",
                    worker.worker_id,
                    worker.start_lane,
                    worker.start_lane + worker.lanes,
                    shard.start_lane,
                    shard.start_lane + shard.lanes
                ),
            ));
        }
        if plan_worker_by_id
            .insert(worker.worker_id.clone(), worker)
            .is_some()
        {
            return Err(error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_WORKER",
                format!(
                    "surrogate plan worker `{}` appears more than once",
                    worker.worker_id
                ),
            ));
        }
    }
    if plan_worker_by_id.len() != shard_by_worker.len() {
        return Err(error(
            "E_RUNTIME_SURROGATE_ATTACHMENT_WORKER",
            "surrogate plan worker summaries must match runtime workers",
        ));
    }

    let mut counts = HashMap::<String, (usize, usize, usize)>::new();
    let mut items = Vec::with_capacity(plan.items.len());
    for item in &plan.items {
        let shard = shard_by_worker.get(&item.worker_id).ok_or_else(|| {
            error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_ITEM",
                format!(
                    "surrogate plan item {} references unknown runtime worker `{}`",
                    item.index, item.worker_id
                ),
            )
        })?;
        let end_lane = shard.start_lane + shard.lanes;
        if item.lane < shard.start_lane || item.lane >= end_lane {
            return Err(error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_ITEM",
                format!(
                    "surrogate plan item {} lane {} is outside worker `{}` lane range {}..{}",
                    item.index, item.lane, item.worker_id, shard.start_lane, end_lane
                ),
            ));
        }

        let entry = counts.entry(item.worker_id.clone()).or_default();
        entry.0 += 1;
        match item.source_result {
            GemmRuntimeSourceResult::Surrogate => entry.1 += 1,
            GemmRuntimeSourceResult::Exact => entry.2 += 1,
        }

        items.push(RuntimeSurrogateItemAttachment {
            index: item.index,
            sample_id: None,
            target: None,
            lane: item.lane,
            worker_id: item.worker_id.clone(),
            decision: item.decision,
            source_result: item.source_result,
            provenance_tag: item.provenance.tag.clone(),
            exact: item.provenance.exact,
            surrogate_id: item.provenance.surrogate_id.clone(),
            shadow_ok: item.shadow_ok,
            shadow_error: item.shadow_error.clone(),
            shadow_predicted: None,
            shadow_expected: None,
            shadow_max_abs_error: item.shadow_max_abs_error,
            shadow_latency_error_cycles: item.shadow_latency_error_cycles,
            shadow_first_divergence: item
                .shadow_first_divergence
                .as_ref()
                .map(runtime_gemm_divergence),
        });
    }

    let mut workers = Vec::with_capacity(shards.len());
    let mut used_surrogate = 0;
    let mut exact_fallbacks = 0;
    for shard in shards {
        let planned = plan_worker_by_id.get(&shard.worker_id).ok_or_else(|| {
            error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_WORKER",
                format!(
                    "surrogate plan missing runtime worker `{}`",
                    shard.worker_id
                ),
            )
        })?;
        let (assigned, surrogate, exact) =
            counts.get(&shard.worker_id).copied().unwrap_or_default();
        if planned.assigned_items != assigned
            || planned.used_surrogate != surrogate
            || planned.exact_fallbacks != exact
        {
            return Err(error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_WORKER",
                format!(
                    "surrogate worker `{}` summary counts do not match attached items",
                    shard.worker_id
                ),
            ));
        }
        used_surrogate += surrogate;
        exact_fallbacks += exact;
        workers.push(RuntimeSurrogateWorkerAttachment {
            worker_id: shard.worker_id.clone(),
            node: shard.node.clone(),
            backend: shard.backend,
            start_lane: shard.start_lane,
            lanes: shard.lanes,
            assigned_items: assigned,
            used_surrogate: surrogate,
            exact_fallbacks: exact,
        });
    }

    Ok(RuntimeSurrogateAttachment {
        format_version: RUNTIME_SURROGATE_ATTACHMENT_FORMAT_VERSION,
        plan_schema: plan.schema.clone(),
        total_lanes,
        workers,
        items,
        health: RuntimeSurrogateHealth {
            ready: true,
            worker_count: shards.len(),
            item_count: plan.items.len(),
            used_surrogate,
            exact_fallbacks,
            diagnostics: Vec::new(),
        },
    })
}

fn attach_event_runtime_plan_to_shard_infos(
    shards: &[RuntimeShardInfo],
    plan: &EventRuntimePlan,
) -> Result<RuntimeSurrogateAttachment, ErrorReport> {
    if plan.schema != EVENT_RUNTIME_PLAN_SCHEMA {
        return Err(error(
            "E_RUNTIME_SURROGATE_ATTACHMENT_SCHEMA",
            format!(
                "unsupported event runtime plan schema `{}`, expected `{}`",
                plan.schema, EVENT_RUNTIME_PLAN_SCHEMA
            ),
        ));
    }
    if !plan.ok || !plan.errors.is_empty() {
        return Err(error(
            "E_RUNTIME_SURROGATE_ATTACHMENT_PLAN",
            "cannot attach event runtime plan that is not ok",
        ));
    }

    let total_lanes = shards.iter().map(|shard| shard.lanes).sum::<usize>();
    if plan.total_lanes != total_lanes {
        return Err(error(
            "E_RUNTIME_SURROGATE_ATTACHMENT_LANES",
            format!(
                "event plan total lanes {} does not match runtime total lanes {}",
                plan.total_lanes, total_lanes
            ),
        ));
    }

    let shard_by_worker = runtime_shards_by_worker(shards)?;
    if plan.workers.len() != shard_by_worker.len() {
        return Err(error(
            "E_RUNTIME_SURROGATE_ATTACHMENT_WORKER",
            "event plan worker summaries must match runtime workers",
        ));
    }
    let mut plan_worker_by_id = HashMap::new();
    for worker in &plan.workers {
        let shard = shard_by_worker.get(&worker.worker_id).ok_or_else(|| {
            error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_WORKER",
                format!(
                    "event plan references unknown runtime worker `{}`",
                    worker.worker_id
                ),
            )
        })?;
        if worker.start_lane != shard.start_lane || worker.lanes != shard.lanes {
            return Err(error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_WORKER",
                format!(
                    "event worker `{}` maps lanes {}..{}, runtime maps lanes {}..{}",
                    worker.worker_id,
                    worker.start_lane,
                    worker.start_lane + worker.lanes,
                    shard.start_lane,
                    shard.start_lane + shard.lanes
                ),
            ));
        }
        if plan_worker_by_id
            .insert(worker.worker_id.clone(), worker)
            .is_some()
        {
            return Err(error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_WORKER",
                format!(
                    "event plan worker `{}` appears more than once",
                    worker.worker_id
                ),
            ));
        }
    }

    let mut counts = HashMap::<String, (usize, usize, usize)>::new();
    let mut items = Vec::with_capacity(plan.items.len());
    for item in &plan.items {
        let shard = shard_by_worker.get(&item.worker_id).ok_or_else(|| {
            error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_ITEM",
                format!(
                    "event plan item {} references unknown runtime worker `{}`",
                    item.index, item.worker_id
                ),
            )
        })?;
        let end_lane = shard.start_lane + shard.lanes;
        if item.lane < shard.start_lane || item.lane >= end_lane {
            return Err(error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_ITEM",
                format!(
                    "event plan item {} lane {} is outside worker `{}` lane range {}..{}",
                    item.index, item.lane, item.worker_id, shard.start_lane, end_lane
                ),
            ));
        }
        match item.source_result {
            GemmRuntimeSourceResult::Surrogate if item.provenance.exact => {
                return Err(error(
                    "E_RUNTIME_SURROGATE_ATTACHMENT_ITEM",
                    format!(
                        "event plan item {} surrogate provenance is exact",
                        item.index
                    ),
                ));
            }
            GemmRuntimeSourceResult::Exact if !item.provenance.exact => {
                return Err(error(
                    "E_RUNTIME_SURROGATE_ATTACHMENT_ITEM",
                    format!(
                        "event plan item {} exact provenance is approximate",
                        item.index
                    ),
                ));
            }
            _ => {}
        }

        let entry = counts.entry(item.worker_id.clone()).or_default();
        entry.0 += 1;
        match item.source_result {
            GemmRuntimeSourceResult::Surrogate => entry.1 += 1,
            GemmRuntimeSourceResult::Exact => entry.2 += 1,
        }
        items.push(RuntimeSurrogateItemAttachment {
            index: item.index,
            sample_id: Some(item.sample_id),
            target: if item.target.is_empty() {
                None
            } else {
                Some(item.target.clone())
            },
            lane: item.lane,
            worker_id: item.worker_id.clone(),
            decision: event_decision_to_gemm(item.decision),
            source_result: item.source_result,
            provenance_tag: item.provenance.tag.clone(),
            exact: item.provenance.exact,
            surrogate_id: item.provenance.surrogate_id.clone(),
            shadow_ok: item.shadow_ok,
            shadow_error: item.shadow_error.clone(),
            shadow_predicted: item.predicted,
            shadow_expected: item.expected,
            shadow_max_abs_error: None,
            shadow_latency_error_cycles: None,
            shadow_first_divergence: None,
        });
    }

    let mut workers = Vec::with_capacity(shards.len());
    let mut used_surrogate = 0;
    let mut exact_fallbacks = 0;
    for shard in shards {
        let planned = plan_worker_by_id.get(&shard.worker_id).ok_or_else(|| {
            error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_WORKER",
                format!("event plan missing runtime worker `{}`", shard.worker_id),
            )
        })?;
        let (assigned, surrogate, exact) =
            counts.get(&shard.worker_id).copied().unwrap_or_default();
        if planned.assigned_items != assigned
            || planned.used_surrogate != surrogate
            || planned.exact_fallbacks != exact
        {
            return Err(error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_WORKER",
                format!(
                    "event worker `{}` summary counts do not match attached items",
                    shard.worker_id
                ),
            ));
        }
        used_surrogate += surrogate;
        exact_fallbacks += exact;
        workers.push(RuntimeSurrogateWorkerAttachment {
            worker_id: shard.worker_id.clone(),
            node: shard.node.clone(),
            backend: shard.backend,
            start_lane: shard.start_lane,
            lanes: shard.lanes,
            assigned_items: assigned,
            used_surrogate: surrogate,
            exact_fallbacks: exact,
        });
    }

    Ok(RuntimeSurrogateAttachment {
        format_version: RUNTIME_SURROGATE_ATTACHMENT_FORMAT_VERSION,
        plan_schema: plan.schema.clone(),
        total_lanes,
        workers,
        items,
        health: RuntimeSurrogateHealth {
            ready: true,
            worker_count: shards.len(),
            item_count: plan.items.len(),
            used_surrogate,
            exact_fallbacks,
            diagnostics: Vec::new(),
        },
    })
}

fn runtime_gemm_divergence(
    divergence: &rrtl_surrogate::GemmDivergence,
) -> RuntimeSurrogateGemmDivergence {
    RuntimeSurrogateGemmDivergence {
        row: divergence.row,
        col: divergence.col,
        expected: divergence.expected,
        actual: divergence.actual,
    }
}

fn runtime_shards_by_worker<'a>(
    shards: &'a [RuntimeShardInfo],
) -> Result<HashMap<String, &'a RuntimeShardInfo>, ErrorReport> {
    let mut shard_by_worker = HashMap::new();
    for shard in shards {
        if shard.lanes == 0 {
            return Err(error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_WORKER",
                format!("runtime worker `{}` owns zero lanes", shard.worker_id),
            ));
        }
        if shard_by_worker
            .insert(shard.worker_id.clone(), shard)
            .is_some()
        {
            return Err(error(
                "E_RUNTIME_SURROGATE_ATTACHMENT_WORKER",
                format!(
                    "runtime worker `{}` appears more than once",
                    shard.worker_id
                ),
            ));
        }
    }
    Ok(shard_by_worker)
}

fn event_decision_to_gemm(decision: EventPolicyDecision) -> GemmPolicyDecision {
    match decision {
        EventPolicyDecision::SurrogateUsed => GemmPolicyDecision::SurrogateUsed,
        EventPolicyDecision::ExactFallback => GemmPolicyDecision::ExactFallback,
        EventPolicyDecision::FailClosed => GemmPolicyDecision::FailClosed,
        EventPolicyDecision::ShadowCompare => GemmPolicyDecision::ShadowCompare,
    }
}

fn validate_partition_placement_topology(topology: &RuntimeTopology) -> Result<(), ErrorReport> {
    if topology.workers().is_empty() {
        return Err(error(
            "E_RUNTIME_PARTITION_PLACEMENT_TOPOLOGY",
            "runtime partition placement requires at least one worker",
        ));
    }
    if topology.workers().iter().any(|worker| worker.lanes == 0) {
        return Err(error(
            "E_RUNTIME_PARTITION_PLACEMENT_TOPOLOGY",
            "runtime partition placement workers must have at least one lane",
        ));
    }
    let mut worker_ids = HashMap::new();
    for worker in topology.workers() {
        if worker_ids.insert(worker.id.as_str(), ()).is_some() {
            return Err(error(
                "E_RUNTIME_PARTITION_PLACEMENT_TOPOLOGY",
                format!(
                    "runtime partition placement has duplicate worker `{}`",
                    worker.id
                ),
            ));
        }
    }
    Ok(())
}

fn best_placement_worker(
    workers: &[PlacementWorkerLoad],
    partition: &RuntimePartition,
    has_cpu_worker: bool,
) -> Option<usize> {
    workers
        .iter()
        .enumerate()
        .filter(|(_, worker)| placement_worker_compatible(worker, partition, has_cpu_worker))
        .min_by(|(left_index, left), (right_index, right)| {
            partition_split_score(left.cost)
                .cmp(&partition_split_score(right.cost))
                .then_with(|| left.partitions.len().cmp(&right.partitions.len()))
                .then_with(|| left.worker_id.cmp(&right.worker_id))
                .then_with(|| left_index.cmp(right_index))
        })
        .map(|(index, _)| index)
}

fn placement_worker_compatible(
    worker: &PlacementWorkerLoad,
    partition: &RuntimePartition,
    has_cpu_worker: bool,
) -> bool {
    if has_cpu_worker && worker.is_gpu() && partition_gpu_hostile(partition) {
        return false;
    }
    true
}

fn partition_gpu_hostile(partition: &RuntimePartition) -> bool {
    partition.external || partition.cost.memory_bits > 0
}

fn account_placement_boundary_traffic(
    plan: &RuntimePartitionPlan,
    assignments: &[RuntimePartitionAssignment],
    workers: &mut [PlacementWorkerLoad],
    diagnostics: &mut Vec<RuntimePartitionPlacementDiagnostic>,
    recommendations: &mut Vec<RuntimePartitionPlacementRecommendation>,
) {
    let worker_by_partition = assignments
        .iter()
        .map(|assignment| {
            (
                assignment.partition_id.as_str(),
                assignment.worker_id.as_str(),
            )
        })
        .collect::<HashMap<_, _>>();
    let worker_index_by_id = workers
        .iter()
        .enumerate()
        .map(|(index, worker)| (worker.worker_id.clone(), index))
        .collect::<HashMap<_, _>>();
    let mut cross_boundary_bits = vec![0u64; plan.boundary_signals.len()];

    for (index, boundary) in plan.boundary_signals.iter().enumerate() {
        let Some(producer_partition) = boundary.producer_partition.as_deref() else {
            continue;
        };
        let Some(producer_worker) = worker_by_partition.get(producer_partition).copied() else {
            continue;
        };
        for consumer_partition in &boundary.consumer_partitions {
            let Some(consumer_worker) = worker_by_partition
                .get(consumer_partition.as_str())
                .copied()
            else {
                continue;
            };
            if producer_worker == consumer_worker {
                continue;
            }
            let bits = u64::from(boundary.width);
            cross_boundary_bits[index] += bits;
            if let Some(worker_index) = worker_index_by_id.get(producer_worker) {
                workers[*worker_index].outbound_boundary_bits += bits;
            }
            if let Some(worker_index) = worker_index_by_id.get(consumer_worker) {
                workers[*worker_index].inbound_boundary_bits += bits;
            }
        }
    }

    let total_cross_boundary_bits = cross_boundary_bits.iter().sum::<u64>();
    let local_bits = plan
        .total_cost
        .state_bits
        .saturating_add(plan.total_cost.memory_bits);
    let total_cross_boundary_is_high =
        total_cross_boundary_bits > 0 && local_bits > 0 && total_cross_boundary_bits > local_bits;
    for (index, bits) in cross_boundary_bits.into_iter().enumerate() {
        if bits == 0 {
            continue;
        }
        let boundary = &plan.boundary_signals[index];
        if boundary.width < 1024 && !total_cross_boundary_is_high {
            continue;
        }
        diagnostics.push(RuntimePartitionPlacementDiagnostic {
            code: RuntimePartitionPlacementDiagnosticCode::HighCrossWorkerBoundaryTraffic,
            severity: RuntimePartitionDiagnosticSeverity::Warning,
            message: format!(
                "boundary `{}` on `{}` moves {} bits across workers",
                boundary.port_name,
                boundary.instance_path.join("."),
                bits
            ),
            worker_id: boundary
                .producer_partition
                .as_deref()
                .and_then(|partition| worker_by_partition.get(partition).copied())
                .map(str::to_string),
            partition_id: boundary.producer_partition.clone(),
            boundary_index: Some(index),
        });
        recommendations.push(RuntimePartitionPlacementRecommendation {
            code: RuntimePartitionPlacementRecommendationCode::CoLocatePartitions,
            message: format!(
                "consider placing partitions connected by `{}` on the same worker or node",
                boundary.instance_path.join(".")
            ),
            worker_id: None,
            partition_id: boundary.producer_partition.clone(),
            boundary_index: Some(index),
        });
    }
}

fn analyze_placement_worker_balance(
    plan: &RuntimePartitionPlan,
    topology: &RuntimeTopology,
    workers: &[PlacementWorkerLoad],
    diagnostics: &mut Vec<RuntimePartitionPlacementDiagnostic>,
    recommendations: &mut Vec<RuntimePartitionPlacementRecommendation>,
) {
    if plan.partitions.len() > topology.workers().len() {
        diagnostics.push(RuntimePartitionPlacementDiagnostic {
            code: RuntimePartitionPlacementDiagnosticCode::UnderProvisionedWorkers,
            severity: RuntimePartitionDiagnosticSeverity::Info,
            message: format!(
                "placement maps {} partitions onto {} workers",
                plan.partitions.len(),
                topology.workers().len()
            ),
            worker_id: None,
            partition_id: None,
            boundary_index: None,
        });
        recommendations.push(RuntimePartitionPlacementRecommendation {
            code: RuntimePartitionPlacementRecommendationCode::AddWorkers,
            message: "add workers or reduce target partitions to lower per-worker multiplexing"
                .to_string(),
            worker_id: None,
            partition_id: None,
            boundary_index: None,
        });
    }

    if workers.len() < 2 {
        return;
    }
    let total_score = workers
        .iter()
        .map(|worker| partition_split_score(worker.cost))
        .sum::<u64>();
    if total_score == 0 {
        return;
    }
    let Some(heavy) = workers
        .iter()
        .max_by_key(|worker| partition_split_score(worker.cost))
    else {
        return;
    };
    if partition_split_score(heavy.cost).saturating_mul(workers.len() as u64)
        <= total_score.saturating_mul(2)
    {
        return;
    }
    diagnostics.push(RuntimePartitionPlacementDiagnostic {
        code: RuntimePartitionPlacementDiagnosticCode::WorkerImbalance,
        severity: RuntimePartitionDiagnosticSeverity::Warning,
        message: format!(
            "worker `{}` has more than 2x average placed cost",
            heavy.worker_id
        ),
        worker_id: Some(heavy.worker_id.clone()),
        partition_id: None,
        boundary_index: None,
    });
    recommendations.push(RuntimePartitionPlacementRecommendation {
        code: RuntimePartitionPlacementRecommendationCode::MovePartition,
        message: format!(
            "move a partition away from worker `{}` or split its heaviest assigned partition",
            heavy.worker_id
        ),
        worker_id: Some(heavy.worker_id.clone()),
        partition_id: None,
        boundary_index: None,
    });
}

#[derive(Clone, Debug)]
struct PartitionRegion {
    id: String,
    module_name: String,
    instance_path: Vec<String>,
    cut_child_instances: Vec<String>,
}

fn partition_regions(
    design: &CompiledDesign,
    top: &CompiledModule,
    config: &RuntimePartitionConfig,
) -> Result<Vec<PartitionRegion>, ErrorReport> {
    if top.instances.is_empty() || config.target_partitions == 1 {
        return Ok(vec![PartitionRegion {
            id: "p0".to_string(),
            module_name: top.name.clone(),
            instance_path: vec![top.name.clone()],
            cut_child_instances: Vec::new(),
        }]);
    }

    validate_partition_instance_modules(design, top)?;

    let mut regions = vec![PartitionRegion {
        id: String::new(),
        module_name: top.name.clone(),
        instance_path: vec![top.name.clone()],
        cut_child_instances: Vec::new(),
    }];
    let mut guard = 0usize;
    while regions.len() < config.target_partitions {
        guard += 1;
        if guard > config.target_partitions.saturating_mul(8).saturating_add(8) {
            break;
        }
        let Some(index) = best_partition_split_candidate(design, &regions) else {
            break;
        };
        if !split_partition_region(design, &mut regions, index, config.target_partitions)? {
            break;
        }
    }

    regions.sort_by(|left, right| {
        left.instance_path
            .cmp(&right.instance_path)
            .then_with(|| left.module_name.cmp(&right.module_name))
    });
    for (index, region) in regions.iter_mut().enumerate() {
        region.id = format!("p{index}");
    }
    Ok(regions)
}

fn validate_partition_instance_modules(
    design: &CompiledDesign,
    module: &CompiledModule,
) -> Result<(), ErrorReport> {
    for instance in &module.instances {
        let child = design.find_module(&instance.module).ok_or_else(|| {
            error(
                "E_RUNTIME_PARTITION_MODULE",
                format!(
                    "instance `{}` references missing module `{}`",
                    instance.name, instance.module
                ),
            )
        })?;
        validate_partition_instance_modules(design, child)?;
    }
    Ok(())
}

fn best_partition_split_candidate(
    design: &CompiledDesign,
    regions: &[PartitionRegion],
) -> Option<usize> {
    regions
        .iter()
        .enumerate()
        .filter_map(|(index, region)| {
            let module = design.find_module(&region.module_name)?;
            if module.is_external || uncut_child_instances(module, region).is_empty() {
                return None;
            }
            Some((index, region_partition_cost(design, region)))
        })
        .max_by(|(left_index, left_cost), (right_index, right_cost)| {
            partition_split_score(*left_cost)
                .cmp(&partition_split_score(*right_cost))
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
}

fn partition_split_score(cost: RuntimePartitionCost) -> u64 {
    cost.compute_ops
        .saturating_add(cost.state_bits)
        .saturating_add(cost.memory_bits)
}

fn split_partition_region(
    design: &CompiledDesign,
    regions: &mut Vec<PartitionRegion>,
    index: usize,
    target_partitions: usize,
) -> Result<bool, ErrorReport> {
    let region = regions[index].clone();
    let module = design.find_module(&region.module_name).ok_or_else(|| {
        error(
            "E_RUNTIME_PARTITION_MODULE",
            format!("module `{}` does not exist", region.module_name),
        )
    })?;
    let uncut = uncut_child_instances(module, &region);
    if uncut.is_empty() {
        return Ok(false);
    }

    let remaining_slots = target_partitions.saturating_sub(regions.len());
    let local_work = module_has_local_partition_work(module);
    let selected_count = partition_split_child_count(local_work, uncut.len(), remaining_slots);
    if selected_count == 0 {
        return Ok(false);
    }

    let selected_names = select_partition_split_children(design, &uncut, selected_count);
    let mut selected_in_module_order = uncut
        .iter()
        .filter(|instance| selected_names.iter().any(|name| name == &instance.name))
        .collect::<Vec<_>>();
    selected_in_module_order.sort_by_key(|instance| {
        module
            .instances
            .iter()
            .position(|candidate| candidate.name == instance.name)
            .unwrap_or(usize::MAX)
    });

    let mut updated_region = region.clone();
    for name in &selected_names {
        if !updated_region.cut_child_instances.contains(name) {
            updated_region.cut_child_instances.push(name.clone());
        }
    }
    let keep_parent = partition_region_has_remaining_work(design, module, &updated_region);

    let mut replacement = Vec::new();
    if keep_parent {
        replacement.push(updated_region);
    }
    for instance in selected_in_module_order {
        let child_module = design.find_module(&instance.module).ok_or_else(|| {
            error(
                "E_RUNTIME_PARTITION_MODULE",
                format!(
                    "instance `{}` references missing module `{}`",
                    instance.name, instance.module
                ),
            )
        })?;
        let mut child_path = region.instance_path.clone();
        child_path.push(instance.name.clone());
        replacement.push(PartitionRegion {
            id: String::new(),
            module_name: child_module.name.clone(),
            instance_path: child_path,
            cut_child_instances: Vec::new(),
        });
    }

    regions.splice(index..=index, replacement);
    Ok(true)
}

fn partition_split_child_count(
    local_work: bool,
    uncut_count: usize,
    remaining_slots: usize,
) -> usize {
    if uncut_count == 0 {
        return 0;
    }
    if !local_work && uncut_count <= remaining_slots.saturating_add(1) {
        return uncut_count;
    }
    remaining_slots.min(uncut_count)
}

fn select_partition_split_children(
    design: &CompiledDesign,
    uncut: &[&CompiledInstance],
    count: usize,
) -> Vec<String> {
    let mut ranked = uncut
        .iter()
        .enumerate()
        .map(|(index, instance)| {
            let cost = design
                .find_module(&instance.module)
                .map(|module| module_partition_cost(design, module, true))
                .unwrap_or_default();
            (index, instance.name.clone(), partition_split_score(cost))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.2.cmp(&left.2).then_with(|| left.0.cmp(&right.0)));
    ranked
        .into_iter()
        .take(count)
        .map(|(_, name, _)| name)
        .collect()
}

fn uncut_child_instances<'a>(
    module: &'a CompiledModule,
    region: &PartitionRegion,
) -> Vec<&'a CompiledInstance> {
    module
        .instances
        .iter()
        .filter(|instance| !region.cut_child_instances.contains(&instance.name))
        .collect()
}

fn partition_region_has_remaining_work(
    design: &CompiledDesign,
    module: &CompiledModule,
    region: &PartitionRegion,
) -> bool {
    module_has_local_partition_work(module)
        || module.instances.iter().any(|instance| {
            !region.cut_child_instances.contains(&instance.name)
                && design.find_module(&instance.module).is_some()
        })
}

fn module_has_local_partition_work(module: &CompiledModule) -> bool {
    !module.assignments.is_empty()
        || !module.registers.is_empty()
        || !module.memory_writes.is_empty()
        || !module.assertions.is_empty()
        || !module.cover_points.is_empty()
        || module.instances.is_empty()
}

fn build_partition(
    design: &CompiledDesign,
    region: &PartitionRegion,
) -> Result<RuntimePartition, ErrorReport> {
    let module = design.find_module(&region.module_name).ok_or_else(|| {
        error(
            "E_RUNTIME_PARTITION_MODULE",
            format!("module `{}` does not exist", region.module_name),
        )
    })?;
    let mut signals = module
        .signals
        .iter()
        .map(|signal| partition_signal_ref(module, signal))
        .collect::<Vec<_>>();
    signals.sort_by(|left, right| {
        left.module_name
            .cmp(&right.module_name)
            .then_with(|| left.signal.id.0.cmp(&right.signal.id.0))
    });

    Ok(RuntimePartition {
        id: region.id.clone(),
        module_name: region.module_name.clone(),
        instance_path: region.instance_path.clone(),
        external: module.is_external,
        signals,
        cost: region_partition_cost(design, region),
    })
}

fn partition_boundaries(
    design: &CompiledDesign,
    top: &CompiledModule,
    regions: &[PartitionRegion],
) -> Vec<RuntimePartitionBoundarySignal> {
    let partition_by_path = regions
        .iter()
        .map(|region| (region.instance_path.clone(), region.id.clone()))
        .collect::<HashMap<_, _>>();
    let mut boundaries = Vec::new();
    for region in regions {
        if region.instance_path.len() <= 1 {
            continue;
        }
        let parent_path = &region.instance_path[..region.instance_path.len() - 1];
        let Some(parent_module) = module_at_instance_path(design, top, parent_path) else {
            continue;
        };
        let Some(instance_name) = region.instance_path.last() else {
            continue;
        };
        let Some(instance) = parent_module
            .instances
            .iter()
            .find(|instance| &instance.name == instance_name)
        else {
            continue;
        };
        let parent_partition = partition_by_path
            .get(parent_path)
            .map(|partition| partition.as_str());
        let child_partition = region.id.as_str();
        emit_partition_boundaries_for_instance(
            parent_module,
            instance,
            &region.instance_path,
            parent_partition,
            child_partition,
            &mut boundaries,
        );
    }
    boundaries.sort_by(|left, right| {
        left.instance_path
            .cmp(&right.instance_path)
            .then_with(|| left.port_name.cmp(&right.port_name))
            .then_with(|| left.signal.signal_name.cmp(&right.signal.signal_name))
    });
    boundaries
}

fn emit_partition_boundaries_for_instance(
    parent_module: &CompiledModule,
    instance: &CompiledInstance,
    instance_path: &[String],
    parent_partition: Option<&str>,
    child_partition: &str,
    boundaries: &mut Vec<RuntimePartitionBoundarySignal>,
) {
    for connection in &instance.connections {
        let Some(signal) = parent_module.signal(connection.signal) else {
            continue;
        };
        let mut consumers = Vec::new();
        let producer = match connection.direction {
            PortDirection::Input => {
                consumers.push(child_partition.to_string());
                parent_partition.map(|partition| partition.to_string())
            }
            PortDirection::Output => {
                if let Some(partition) = parent_partition {
                    consumers.push(partition.to_string());
                }
                Some(child_partition.to_string())
            }
            PortDirection::Inout => {
                consumers.push(child_partition.to_string());
                if let Some(partition) = parent_partition {
                    consumers.push(partition.to_string());
                }
                None
            }
        };
        consumers.sort();
        consumers.dedup();
        let consumer_count = consumers.len().max(1) as u64;
        boundaries.push(RuntimePartitionBoundarySignal {
            signal: partition_signal_ref(parent_module, signal),
            instance_path: instance_path.to_vec(),
            port_name: connection.port.clone(),
            producer_partition: producer,
            consumer_partitions: consumers,
            width: connection.width,
            cost: RuntimePartitionCost {
                boundary_bits: u64::from(connection.width) * consumer_count,
                ..RuntimePartitionCost::default()
            },
        });
    }
}

fn module_at_instance_path<'a>(
    design: &'a CompiledDesign,
    top: &'a CompiledModule,
    instance_path: &[String],
) -> Option<&'a CompiledModule> {
    if instance_path.first()? != &top.name {
        return None;
    }
    let mut module = top;
    for instance_name in &instance_path[1..] {
        let instance = module
            .instances
            .iter()
            .find(|instance| &instance.name == instance_name)?;
        module = design.find_module(&instance.module)?;
    }
    Some(module)
}

fn partition_signal_ref(module: &CompiledModule, signal: &SignalInfo) -> RuntimePartitionSignalRef {
    RuntimePartitionSignalRef {
        module_name: module.name.clone(),
        signal_name: signal.name.clone(),
        signal: signal.handle,
        width: signal.width,
        kind: partition_signal_kind(&signal.kind),
    }
}

fn partition_signal_kind(kind: &SignalKind) -> RuntimePartitionSignalKind {
    match kind {
        SignalKind::Input => RuntimePartitionSignalKind::Input,
        SignalKind::Output => RuntimePartitionSignalKind::Output,
        SignalKind::Inout => RuntimePartitionSignalKind::Inout,
        SignalKind::Wire => RuntimePartitionSignalKind::Wire,
        SignalKind::Reg { .. } => RuntimePartitionSignalKind::Register,
        SignalKind::Mem { .. } => RuntimePartitionSignalKind::Memory,
    }
}

fn module_partition_cost(
    design: &CompiledDesign,
    module: &CompiledModule,
    include_instances: bool,
) -> RuntimePartitionCost {
    let mut cost = RuntimePartitionCost::default();
    for assignment in &module.assignments {
        cost.compute_ops += expr_compute_ops(&assignment.expr.expr);
    }
    for register in &module.registers {
        cost.compute_ops += 1 + expr_compute_ops(&register.next.expr);
        if let Some(signal) = module.signal(register.signal) {
            cost.state_bits += u64::from(signal.width);
        }
    }
    for write in &module.memory_writes {
        cost.compute_ops += 1
            + expr_compute_ops(&write.enable.expr)
            + expr_compute_ops(&write.addr.expr)
            + expr_compute_ops(&write.data.expr);
    }
    for assertion in &module.assertions {
        cost.compute_ops += expr_compute_ops(&assertion.condition.expr);
        if let Some(enable) = &assertion.enable {
            cost.compute_ops += expr_compute_ops(&enable.expr);
        }
    }
    for cover in &module.cover_points {
        cost.compute_ops += expr_compute_ops(&cover.condition.expr);
        if let Some(enable) = &cover.enable {
            cost.compute_ops += expr_compute_ops(&enable.expr);
        }
    }
    for signal in &module.signals {
        if let SignalKind::Mem {
            data_width, depth, ..
        } = &signal.kind
        {
            cost.memory_bits += u64::from(*data_width) * *depth as u64;
        }
    }
    if include_instances {
        for instance in &module.instances {
            if let Some(child) = design.find_module(&instance.module) {
                cost += module_partition_cost(design, child, true);
            }
        }
    }
    cost
}

fn region_partition_cost(
    design: &CompiledDesign,
    region: &PartitionRegion,
) -> RuntimePartitionCost {
    let Some(module) = design.find_module(&region.module_name) else {
        return RuntimePartitionCost::default();
    };
    let mut cost = module_partition_cost(design, module, false);
    for instance in &module.instances {
        if region.cut_child_instances.contains(&instance.name) {
            continue;
        }
        if let Some(child) = design.find_module(&instance.module) {
            cost += module_partition_cost(design, child, true);
        }
    }
    cost
}

fn total_partition_cost(
    partitions: &[RuntimePartition],
    boundaries: &[RuntimePartitionBoundarySignal],
) -> RuntimePartitionCost {
    let mut cost = RuntimePartitionCost::default();
    for partition in partitions {
        cost += partition.cost;
    }
    for boundary in boundaries {
        cost += boundary.cost;
    }
    cost
}

#[derive(Clone, Debug, Default)]
struct RuntimePartitionAnalysis {
    diagnostics: Vec<RuntimePartitionDiagnostic>,
    recommendations: Vec<RuntimePartitionRecommendation>,
}

fn analyze_partition_plan(
    partitions: &[RuntimePartition],
    boundaries: &[RuntimePartitionBoundarySignal],
    total_cost: &RuntimePartitionCost,
    config: &RuntimePartitionConfig,
) -> RuntimePartitionAnalysis {
    let mut analysis = RuntimePartitionAnalysis::default();
    analyze_target_underfill(partitions, config, &mut analysis);
    analyze_partition_imbalance(partitions, &mut analysis);
    analyze_boundary_cost(boundaries, total_cost, &mut analysis);
    analyze_external_partitions(partitions, &mut analysis);
    analyze_empty_partitions(partitions, &mut analysis);
    analysis
}

fn analyze_target_underfill(
    partitions: &[RuntimePartition],
    config: &RuntimePartitionConfig,
    analysis: &mut RuntimePartitionAnalysis,
) {
    if partitions.len() >= config.target_partitions {
        return;
    }

    analysis.diagnostics.push(RuntimePartitionDiagnostic {
        code: RuntimePartitionDiagnosticCode::TargetUnderfilled,
        severity: RuntimePartitionDiagnosticSeverity::Info,
        message: format!(
            "runtime partition planner emitted {} partitions for target {}",
            partitions.len(),
            config.target_partitions
        ),
        partition_id: None,
        boundary_index: None,
    });
    analysis
        .recommendations
        .push(RuntimePartitionRecommendation {
            code: RuntimePartitionRecommendationCode::IncreaseHierarchy,
            message: "add deeper hierarchy cuts or enable recursive partition planning before scaling this design across more workers".to_string(),
            partition_id: None,
            boundary_index: None,
        });
}

fn analyze_partition_imbalance(
    partitions: &[RuntimePartition],
    analysis: &mut RuntimePartitionAnalysis,
) {
    if partitions.len() < 2 {
        return;
    }

    let total_compute = partitions
        .iter()
        .map(|partition| partition.cost.compute_ops)
        .sum::<u64>();
    if total_compute == 0 {
        return;
    }
    let Some(heavy) = partitions
        .iter()
        .max_by_key(|partition| partition.cost.compute_ops)
    else {
        return;
    };
    let partition_count = partitions.len() as u64;
    if heavy.cost.compute_ops.saturating_mul(partition_count) <= total_compute.saturating_mul(2) {
        return;
    }

    analysis.diagnostics.push(RuntimePartitionDiagnostic {
        code: RuntimePartitionDiagnosticCode::PartitionImbalance,
        severity: RuntimePartitionDiagnosticSeverity::Warning,
        message: format!(
            "partition `{}` has {} compute ops, more than 2x the average",
            heavy.id, heavy.cost.compute_ops
        ),
        partition_id: Some(heavy.id.clone()),
        boundary_index: None,
    });
    analysis
        .recommendations
        .push(RuntimePartitionRecommendation {
            code: RuntimePartitionRecommendationCode::SplitHeavyPartition,
            message: format!(
                "split partition `{}` with deeper hierarchy or a finer structural cut",
                heavy.id
            ),
            partition_id: Some(heavy.id.clone()),
            boundary_index: None,
        });
}

fn analyze_boundary_cost(
    boundaries: &[RuntimePartitionBoundarySignal],
    total_cost: &RuntimePartitionCost,
    analysis: &mut RuntimePartitionAnalysis,
) {
    let local_bits = total_cost.state_bits.saturating_add(total_cost.memory_bits);
    let total_boundary_is_high = local_bits > 0 && total_cost.boundary_bits > local_bits;
    for (index, boundary) in boundaries.iter().enumerate() {
        if boundary.width < 1024 && !total_boundary_is_high {
            continue;
        }

        analysis.diagnostics.push(RuntimePartitionDiagnostic {
            code: RuntimePartitionDiagnosticCode::HighBoundaryCost,
            severity: RuntimePartitionDiagnosticSeverity::Warning,
            message: format!(
                "boundary `{}` on instance `{}` moves {} bits",
                boundary.port_name,
                boundary.instance_path.join("."),
                boundary.cost.boundary_bits
            ),
            partition_id: boundary.producer_partition.clone(),
            boundary_index: Some(index),
        });
        analysis
            .recommendations
            .push(RuntimePartitionRecommendation {
                code: RuntimePartitionRecommendationCode::CoLocateBoundary,
                message: format!(
                "consider co-locating `{}` with adjacent logic or cutting on a narrower interface",
                boundary.instance_path.join(".")
            ),
                partition_id: boundary.producer_partition.clone(),
                boundary_index: Some(index),
            });
    }
}

fn analyze_external_partitions(
    partitions: &[RuntimePartition],
    analysis: &mut RuntimePartitionAnalysis,
) {
    for partition in partitions.iter().filter(|partition| partition.external) {
        analysis.diagnostics.push(RuntimePartitionDiagnostic {
            code: RuntimePartitionDiagnosticCode::ExternalModuleOpaque,
            severity: RuntimePartitionDiagnosticSeverity::Info,
            message: format!(
                "partition `{}` targets external module `{}`; internal compute cost is opaque",
                partition.id, partition.module_name
            ),
            partition_id: Some(partition.id.clone()),
            boundary_index: None,
        });
        analysis
            .recommendations
            .push(RuntimePartitionRecommendation {
                code: RuntimePartitionRecommendationCode::InspectExternalModule,
                message: format!(
                    "provide a cost model or implementation metadata for external module `{}`",
                    partition.module_name
                ),
                partition_id: Some(partition.id.clone()),
                boundary_index: None,
            });
    }
}

fn analyze_empty_partitions(
    partitions: &[RuntimePartition],
    analysis: &mut RuntimePartitionAnalysis,
) {
    for partition in partitions.iter().filter(|partition| {
        !partition.external
            && partition.cost.compute_ops == 0
            && partition.cost.state_bits == 0
            && partition.cost.memory_bits == 0
    }) {
        analysis.diagnostics.push(RuntimePartitionDiagnostic {
            code: RuntimePartitionDiagnosticCode::EmptyPartition,
            severity: RuntimePartitionDiagnosticSeverity::Warning,
            message: format!(
                "partition `{}` has no local compute, register state, or memory state",
                partition.id
            ),
            partition_id: Some(partition.id.clone()),
            boundary_index: None,
        });
        analysis
            .recommendations
            .push(RuntimePartitionRecommendation {
                code: RuntimePartitionRecommendationCode::ReduceTargetPartitions,
                message: format!(
                    "merge partition `{}` or reduce the requested partition count",
                    partition.id
                ),
                partition_id: Some(partition.id.clone()),
                boundary_index: None,
            });
    }
}

fn expr_compute_ops(expr: &Expr) -> u64 {
    match expr {
        Expr::Lit { .. } | Expr::Signal(_) => 1,
        Expr::Not(expr)
        | Expr::Slice { expr, .. }
        | Expr::Zext { expr, .. }
        | Expr::Sext { expr, .. }
        | Expr::Trunc { expr, .. }
        | Expr::Cast { expr, .. } => 1 + expr_compute_ops(expr),
        Expr::And(left, right)
        | Expr::Or(left, right)
        | Expr::Xor(left, right)
        | Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Eq(left, right)
        | Expr::Ne(left, right)
        | Expr::Lt(left, right) => 1 + expr_compute_ops(left) + expr_compute_ops(right),
        Expr::Mux {
            cond,
            then_expr,
            else_expr,
        } => 1 + expr_compute_ops(cond) + expr_compute_ops(then_expr) + expr_compute_ops(else_expr),
        Expr::Concat(exprs) => 1 + exprs.iter().map(expr_compute_ops).sum::<u64>(),
        Expr::MemRead { addr, .. } => 2 + expr_compute_ops(addr),
    }
}

impl std::ops::AddAssign for RuntimePartitionCost {
    fn add_assign(&mut self, rhs: Self) {
        self.compute_ops += rhs.compute_ops;
        self.state_bits += rhs.state_bits;
        self.memory_bits += rhs.memory_bits;
        self.boundary_bits += rhs.boundary_bits;
    }
}

impl std::ops::Add for RuntimePartitionCost {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeAutotuneCandidate {
    pub name: String,
    pub topology: RuntimeTopology,
    pub options: DistributedRuntimeOptions,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeAutotuneConfig {
    pub warmup_steps: usize,
    pub measure_steps: usize,
    pub stimulus: Option<RuntimeAutotuneStimulus>,
}

impl Default for RuntimeAutotuneConfig {
    fn default() -> Self {
        Self {
            warmup_steps: 1,
            measure_steps: 16,
            stimulus: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeAutotuneStimulus {
    pub setup: RuntimeStimulusSetup,
    pub steps: Vec<RuntimeStimulusStep>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeStimulusSetup {
    pub inputs: Vec<RuntimeSignalValue>,
    pub input_limbs: Vec<RuntimeSignalLimbs>,
    pub memories: Vec<RuntimeMemoryValue>,
    pub memory_limbs: Vec<RuntimeMemoryLimbs>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeStimulusStep {
    pub inputs: Vec<RuntimeSignalValue>,
    pub input_limbs: Vec<RuntimeSignalLimbs>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSignalValue {
    pub signal: Signal,
    pub lane_values: Vec<u128>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSignalLimbs {
    pub signal: Signal,
    pub lane_values: Vec<Vec<u32>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeMemoryValue {
    pub memory: Signal,
    pub lane_words: Vec<Vec<u128>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeMemoryLimbs {
    pub memory: Signal,
    pub lane_words: Vec<Vec<Vec<u32>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeAutotuneCandidateReport {
    pub name: String,
    pub topology: RuntimeTopology,
    pub options: DistributedRuntimeOptions,
    pub stats: Option<RuntimeStats>,
    pub diagnostics: Vec<Diagnostic>,
    pub score_ns: Option<u128>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeAutotuneReport {
    pub candidates: Vec<RuntimeAutotuneCandidateReport>,
    pub best_index: usize,
}

impl RuntimeOperationStats {
    fn record(&mut self, action: RuntimeShardAction, elapsed_ns: u128) {
        match action {
            RuntimeShardAction::EvalCombinational => {
                self.eval_combinational.record(elapsed_ns);
            }
            RuntimeShardAction::Tick => {
                self.tick.record(elapsed_ns);
            }
            RuntimeShardAction::TickMany(_) => {
                self.tick_many.record(elapsed_ns);
            }
        }
    }
}

impl std::ops::AddAssign for RuntimeOperationStats {
    fn add_assign(&mut self, rhs: Self) {
        self.eval_combinational += rhs.eval_combinational;
        self.tick += rhs.tick;
        self.tick_many += rhs.tick_many;
    }
}

fn record_partition_worker_action(
    operations: &mut RuntimeOperationStats,
    action: RuntimePartitionWorkerActionKind,
) {
    match action {
        RuntimePartitionWorkerActionKind::EvalCombinational => {
            operations.eval_combinational.record(0);
        }
        RuntimePartitionWorkerActionKind::Tick => {
            operations.tick.record(0);
        }
        RuntimePartitionWorkerActionKind::TickMany(steps) => {
            if steps > 0 {
                operations.tick_many.record(0);
            }
        }
    }
}

fn partition_route_mailbox_report(
    worker_id: &str,
    state: &RuntimePartitionWorkerState,
) -> RuntimePartitionRouteMailboxReport {
    RuntimePartitionRouteMailboxReport {
        worker_id: worker_id.to_string(),
        outbound_payload_count: state.outbound_route_payloads.len(),
        inbound_payload_count: state.inbound_route_payloads.len(),
        outbound_routes: state.launch.outbound_routes.clone(),
        inbound_routes: state.launch.inbound_routes.clone(),
        diagnostics: Vec::new(),
    }
}

fn validate_partition_route_payload(
    worker_id: &str,
    payload: &RuntimePartitionRoutePayload,
    routes: &[RuntimePartitionLaunchRoute],
    direction: &str,
) -> Result<(), ErrorReport> {
    let route = routes
        .iter()
        .find(|route| route.route_index == payload.route_index)
        .ok_or_else(|| {
            error(
                "E_RUNTIME_PARTITION_ROUTE_MAILBOX",
                format!(
                    "partition worker `{worker_id}` has no {direction} route {}",
                    payload.route_index
                ),
            )
        })?;
    if payload.width != route.width {
        return Err(error(
            "E_RUNTIME_PARTITION_ROUTE_MAILBOX",
            format!(
                "partition worker `{worker_id}` {direction} route {} payload width {} does not match route width {}",
                payload.route_index, payload.width, route.width
            ),
        ));
    }
    let required_limbs = route_width_limbs(route.width);
    if required_limbs > 0 && payload.limbs.is_empty() {
        return Err(error(
            "E_RUNTIME_PARTITION_ROUTE_MAILBOX",
            format!(
                "partition worker `{worker_id}` {direction} route {} payload has no limbs for width {}",
                payload.route_index, payload.width
            ),
        ));
    }
    if payload.limbs.len() > required_limbs {
        return Err(error(
            "E_RUNTIME_PARTITION_ROUTE_MAILBOX",
            format!(
                "partition worker `{worker_id}` {direction} route {} payload has {} limbs, expected at most {}",
                payload.route_index,
                payload.limbs.len(),
                required_limbs
            ),
        ));
    }
    Ok(())
}

fn route_width_limbs(width: Width) -> usize {
    usize::try_from(width.div_ceil(32)).unwrap_or(usize::MAX)
}

impl std::ops::AddAssign for RuntimeOperationCounters {
    fn add_assign(&mut self, rhs: Self) {
        self.calls += rhs.calls;
        self.total_ns += rhs.total_ns;
        self.last_ns += rhs.last_ns;
    }
}

impl RuntimeOperationCounters {
    fn record(&mut self, elapsed_ns: u128) {
        self.calls += 1;
        self.total_ns += elapsed_ns;
        self.last_ns = elapsed_ns;
    }
}

pub fn recommend_runtime_topology(
    design: &Design,
    module_name: &str,
    candidates: Vec<RuntimeAutotuneCandidate>,
    config: RuntimeAutotuneConfig,
) -> Result<RuntimeAutotuneReport, ErrorReport> {
    if candidates.is_empty() {
        return Err(error(
            "E_RUNTIME_AUTOTUNE_CANDIDATES",
            "runtime topology autotune requires at least one candidate",
        ));
    }
    if config.measure_steps == 0 {
        return Err(error(
            "E_RUNTIME_AUTOTUNE_STEPS",
            "runtime topology autotune measure_steps must be greater than zero",
        ));
    }

    let mut reports = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        reports.push(evaluate_runtime_autotune_candidate(
            design,
            module_name,
            candidate,
            &config,
        ));
    }

    let Some(best_index) = reports
        .iter()
        .enumerate()
        .filter_map(|(index, report)| report.score_ns.map(|score| (index, score)))
        .min_by_key(|(index, score)| (*score, *index))
        .map(|(index, _)| index)
    else {
        let mut diagnostics = vec![Diagnostic::new(
            "E_RUNTIME_AUTOTUNE_NO_CANDIDATE",
            "runtime topology autotune found no successful candidate",
        )];
        diagnostics.extend(
            reports
                .iter()
                .flat_map(|report| report.diagnostics.iter().cloned()),
        );
        return Err(ErrorReport::new(diagnostics));
    };

    Ok(RuntimeAutotuneReport {
        candidates: reports,
        best_index,
    })
}

/// Distributed, heterogeneous runtime for huge batched RRTL simulations.
///
/// The runtime partitions global lanes across workers and routes signal I/O to
/// the corresponding shard. Independent lanes do not communicate, so a single
/// tick is embarrassingly parallel across CPU/GPU workers.
pub struct DistributedRuntime {
    total_lanes: usize,
    program: PackedProgram,
    shards: Vec<RuntimeShard>,
    options: DistributedRuntimeOptions,
    operations: RuntimeOperationStats,
    surrogate_attachment: Option<RuntimeSurrogateAttachment>,
    surrogate_execution: Option<RuntimeSurrogateExecutionReport>,
}

impl DistributedRuntime {
    pub fn new(
        design: &Design,
        module_name: &str,
        topology: RuntimeTopology,
    ) -> Result<Self, ErrorReport> {
        Self::new_with_options(
            design,
            module_name,
            topology,
            DistributedRuntimeOptions::default(),
        )
    }

    pub fn new_with_options(
        design: &Design,
        module_name: &str,
        topology: RuntimeTopology,
        options: DistributedRuntimeOptions,
    ) -> Result<Self, ErrorReport> {
        if topology.workers.is_empty() {
            return Err(error(
                "E_RUNTIME_TOPOLOGY",
                "distributed runtime requires at least one worker",
            ));
        }
        if topology.workers.iter().any(|worker| worker.lanes == 0) {
            return Err(error(
                "E_RUNTIME_LANES",
                "runtime workers must own at least one lane",
            ));
        }

        let compiled = compile(design)?;
        let program = lower_to_packed_program(&compiled, module_name)?;
        let mut start_lane = 0;
        let mut shards = Vec::with_capacity(topology.workers.len());
        for worker in topology.workers {
            let lanes = worker.lanes;
            let engine = match worker.backend {
                RuntimeBackend::ScalarCpu => {
                    if lanes != 1 {
                        return Err(error(
                            "E_RUNTIME_BACKEND_LANES",
                            "scalar CPU backend supports exactly one lane",
                        ));
                    }
                    RuntimeEngine::ScalarCpu(SingleLaneMachineSimulator::new(program.clone())?)
                }
                RuntimeBackend::PackedCpu => {
                    RuntimeEngine::PackedCpu(PackedSimulator::new(program.clone(), lanes)?)
                }
                RuntimeBackend::SimdCpu => {
                    RuntimeEngine::SimdCpu(SimdCpuSimulator::new(program.clone(), lanes)?)
                }
                RuntimeBackend::JitCpu => {
                    RuntimeEngine::JitCpu(JitCpuSimulator::new(program.clone(), lanes)?)
                }
                RuntimeBackend::Gpu(options) => RuntimeEngine::Gpu(
                    GpuBatchSimulator::new_with_options(design, module_name, lanes, options)?,
                ),
            };
            shards.push(RuntimeShard {
                info: RuntimeShardInfo {
                    worker_id: worker.id,
                    node: worker.node,
                    backend: worker.backend,
                    start_lane,
                    lanes,
                },
                engine,
                operations: RuntimeOperationStats::default(),
            });
            start_lane += lanes;
        }

        Ok(Self {
            total_lanes: start_lane,
            program,
            shards,
            options,
            operations: RuntimeOperationStats::default(),
            surrogate_attachment: None,
            surrogate_execution: None,
        })
    }

    pub fn new_loopback_workers(
        design: &Design,
        module_name: &str,
        topology: RuntimeTopology,
        options: DistributedRuntimeOptions,
    ) -> Result<Self, ErrorReport> {
        if topology.workers.is_empty() {
            return Err(error(
                "E_RUNTIME_TOPOLOGY",
                "distributed runtime requires at least one worker",
            ));
        }
        if topology.workers.iter().any(|worker| worker.lanes == 0) {
            return Err(error(
                "E_RUNTIME_LANES",
                "runtime workers must own at least one lane",
            ));
        }

        let compiled = compile(design)?;
        let program = lower_to_packed_program(&compiled, module_name)?;
        let service = Arc::new(Mutex::new(LocalRuntimeWorkerService::new()));
        let mut start_lane = 0;
        let mut shards = Vec::with_capacity(topology.workers.len());
        for worker in topology.workers {
            let info = RuntimeShardInfo {
                worker_id: worker.id,
                node: worker.node,
                backend: worker.backend,
                start_lane,
                lanes: worker.lanes,
            };
            let init = RuntimeShardInit {
                design: compiled.clone(),
                module_name: module_name.to_string(),
                shard: info.clone(),
                backend: worker.backend,
            };
            expect_ack(
                service
                    .lock()
                    .map_err(|_| {
                        error(
                            "E_RUNTIME_WORKER_LOCK",
                            "runtime worker service lock is poisoned",
                        )
                    })?
                    .handle(RuntimeWorkerRequest::InitShard(init))?,
            )?;
            shards.push(RuntimeShard {
                engine: RuntimeEngine::Worker(Box::new(LoopbackRuntimeShardClient::new(
                    info.worker_id.clone(),
                    service.clone(),
                ))),
                info,
                operations: RuntimeOperationStats::default(),
            });
            start_lane += worker.lanes;
        }

        Ok(Self {
            total_lanes: start_lane,
            program,
            shards,
            options,
            operations: RuntimeOperationStats::default(),
            surrogate_attachment: None,
            surrogate_execution: None,
        })
    }

    pub fn new_tcp_workers(
        design: &Design,
        module_name: &str,
        topology: RuntimeTopology,
        options: DistributedRuntimeOptions,
        endpoints: HashMap<String, SocketAddr>,
    ) -> Result<Self, ErrorReport> {
        Self::new_tcp_workers_with_config(
            design,
            module_name,
            topology,
            options,
            endpoints,
            TcpRuntimeTransportConfig::default(),
        )
    }

    pub fn new_tcp_workers_with_config(
        design: &Design,
        module_name: &str,
        topology: RuntimeTopology,
        options: DistributedRuntimeOptions,
        endpoints: HashMap<String, SocketAddr>,
        transport: TcpRuntimeTransportConfig,
    ) -> Result<Self, ErrorReport> {
        if topology.workers.is_empty() {
            return Err(error(
                "E_RUNTIME_TOPOLOGY",
                "distributed runtime requires at least one worker",
            ));
        }
        if topology.workers.iter().any(|worker| worker.lanes == 0) {
            return Err(error(
                "E_RUNTIME_LANES",
                "runtime workers must own at least one lane",
            ));
        }

        let compiled = compile(design)?;
        let program = lower_to_packed_program(&compiled, module_name)?;
        let mut start_lane = 0;
        let mut shards = Vec::with_capacity(topology.workers.len());
        for worker in topology.workers {
            let endpoint = *endpoints.get(&worker.id).ok_or_else(|| {
                error(
                    "E_RUNTIME_TCP_ENDPOINT",
                    format!("missing TCP endpoint for worker `{}`", worker.id),
                )
            })?;
            let info = RuntimeShardInfo {
                worker_id: worker.id,
                node: worker.node,
                backend: worker.backend,
                start_lane,
                lanes: worker.lanes,
            };
            let mut client = TcpRuntimeShardClient::connect_with_config(
                info.worker_id.clone(),
                endpoint,
                transport,
            )?;
            expect_ack(
                client.request(RuntimeWorkerRequest::InitShard(RuntimeShardInit {
                    design: compiled.clone(),
                    module_name: module_name.to_string(),
                    shard: info.clone(),
                    backend: worker.backend,
                }))?,
            )?;
            shards.push(RuntimeShard {
                engine: RuntimeEngine::Worker(Box::new(client)),
                info,
                operations: RuntimeOperationStats::default(),
            });
            start_lane += worker.lanes;
        }

        Ok(Self {
            total_lanes: start_lane,
            program,
            shards,
            options,
            operations: RuntimeOperationStats::default(),
            surrogate_attachment: None,
            surrogate_execution: None,
        })
    }

    pub fn local_cpu(
        design: &Design,
        module_name: &str,
        lanes: usize,
    ) -> Result<Self, ErrorReport> {
        Self::new(design, module_name, RuntimeTopology::local_cpu(lanes))
    }

    pub fn total_lanes(&self) -> usize {
        self.total_lanes
    }

    pub fn program(&self) -> &PackedProgram {
        &self.program
    }

    pub fn shard_plan(&self) -> Vec<RuntimeShardInfo> {
        self.shards.iter().map(|shard| shard.info.clone()).collect()
    }

    pub fn attach_surrogate_plan(&mut self, plan: &GemmRuntimePlan) -> Result<(), ErrorReport> {
        let attachment = attach_gemm_runtime_plan_to_shards(&self.shards, plan)?;
        self.surrogate_execution = Some(RuntimeSurrogateExecutionReport::inspect_attachment(
            &attachment,
        ));
        self.surrogate_attachment = Some(attachment);
        Ok(())
    }

    pub fn attach_event_surrogate_plan(
        &mut self,
        plan: &EventRuntimePlan,
    ) -> Result<(), ErrorReport> {
        let infos = self
            .shards
            .iter()
            .map(|shard| shard.info.clone())
            .collect::<Vec<_>>();
        let attachment = attach_event_runtime_plan_to_shard_infos(&infos, plan)?;
        self.surrogate_execution = Some(RuntimeSurrogateExecutionReport::inspect_attachment(
            &attachment,
        ));
        self.surrogate_attachment = Some(attachment);
        Ok(())
    }

    pub fn surrogate_attachment(&self) -> Option<&RuntimeSurrogateAttachment> {
        self.surrogate_attachment.as_ref()
    }

    pub fn surrogate_execution(&self) -> Option<&RuntimeSurrogateExecutionReport> {
        self.surrogate_execution.as_ref()
    }

    pub fn stats(&self) -> RuntimeStats {
        RuntimeStats {
            execution_mode: self.options.execution_mode,
            total_lanes: self.total_lanes,
            operations: self.operations,
            surrogate_execution: self.surrogate_execution.clone(),
            shards: self
                .shards
                .iter()
                .map(|shard| RuntimeShardStats {
                    shard: shard.info.clone(),
                    operations: shard.operations,
                })
                .collect(),
        }
    }

    pub fn health(&mut self) -> Result<RuntimeHealthReport, ErrorReport> {
        let mut shards = Vec::with_capacity(self.shards.len());
        for shard in &mut self.shards {
            shards.push(shard.health()?);
        }
        Ok(RuntimeHealthReport {
            total_lanes: self.total_lanes,
            shards,
        })
    }

    pub fn reset_stats(&mut self) {
        self.operations = RuntimeOperationStats::default();
        self.surrogate_execution = self
            .surrogate_attachment
            .as_ref()
            .map(RuntimeSurrogateExecutionReport::inspect_attachment);
        for shard in &mut self.shards {
            shard.operations = RuntimeOperationStats::default();
        }
    }

    pub fn snapshot(&mut self) -> Result<RuntimeSnapshot, ErrorReport> {
        let mut shards = Vec::with_capacity(self.shards.len());
        for shard in &mut self.shards {
            let storage = shard.engine.snapshot_storage()?;
            shards.push(RuntimeShardSnapshot {
                shard: shard.info.clone(),
                values: storage.values,
                memories: storage.memories,
            });
        }

        Ok(RuntimeSnapshot {
            total_lanes: self.total_lanes,
            program_top: self.program.top.clone(),
            signal_words_per_lane: self.program.total_signal_words,
            memory_words_per_lane: self.program.total_memory_words_per_lane,
            shards,
        })
    }

    pub fn checkpoint(&mut self) -> Result<RuntimeCheckpoint, ErrorReport> {
        self.checkpoint_with_tcp_endpoints(&HashMap::new())
    }

    pub fn checkpoint_with_tcp_endpoints(
        &mut self,
        endpoints: &HashMap<String, SocketAddr>,
    ) -> Result<RuntimeCheckpoint, ErrorReport> {
        let mut tcp_endpoints = Vec::new();
        for shard in &self.shards {
            if let Some(endpoint) = endpoints.get(&shard.info.worker_id) {
                tcp_endpoints.push(RuntimeTcpEndpoint {
                    worker_id: shard.info.worker_id.clone(),
                    addr: endpoint.to_string(),
                });
            }
        }
        for worker_id in endpoints.keys() {
            if !self
                .shards
                .iter()
                .any(|shard| shard.info.worker_id == *worker_id)
            {
                return Err(error(
                    "E_RUNTIME_CHECKPOINT_ENDPOINT",
                    format!("TCP endpoint provided for unknown worker `{worker_id}`"),
                ));
            }
        }

        Ok(RuntimeCheckpoint {
            format_version: RUNTIME_CHECKPOINT_FORMAT_VERSION,
            module_name: self.program.top.clone(),
            topology: self.runtime_topology(),
            tcp_endpoints,
            snapshot: self.snapshot()?,
        })
    }

    pub fn tick_many_with_checkpoints<F>(
        &mut self,
        steps: usize,
        cadence: RuntimeCheckpointCadence,
        on_checkpoint: F,
    ) -> Result<RuntimeCheckpointRunReport, ErrorReport>
    where
        F: FnMut(RuntimeCheckpointEvent, &RuntimeCheckpoint) -> Result<(), ErrorReport>,
    {
        self.tick_many_with_checkpoint_sink(steps, cadence, &HashMap::new(), on_checkpoint)
    }

    pub fn tick_many_with_tcp_checkpoints<F>(
        &mut self,
        steps: usize,
        cadence: RuntimeCheckpointCadence,
        tcp_endpoints: &HashMap<String, SocketAddr>,
        on_checkpoint: F,
    ) -> Result<RuntimeCheckpointRunReport, ErrorReport>
    where
        F: FnMut(RuntimeCheckpointEvent, &RuntimeCheckpoint) -> Result<(), ErrorReport>,
    {
        self.tick_many_with_checkpoint_sink(steps, cadence, tcp_endpoints, on_checkpoint)
    }

    pub fn restore_snapshot(&mut self, snapshot: &RuntimeSnapshot) -> Result<(), ErrorReport> {
        self.validate_snapshot(snapshot)?;
        for (shard, snapshot) in self.shards.iter_mut().zip(&snapshot.shards) {
            shard.engine.restore_storage(&PackedSimulatorStorage {
                values: snapshot.values.clone(),
                memories: snapshot.memories.clone(),
            })?;
        }
        Ok(())
    }

    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &RuntimeCheckpoint,
    ) -> Result<(), ErrorReport> {
        self.validate_checkpoint(checkpoint)?;
        self.restore_snapshot(&checkpoint.snapshot)
    }

    pub fn recover_tcp_workers_from_checkpoint(
        &mut self,
        design: &Design,
        checkpoint: &RuntimeCheckpoint,
        endpoints: HashMap<String, SocketAddr>,
    ) -> Result<RuntimeRecoveryReport, ErrorReport> {
        self.recover_tcp_workers_from_checkpoint_with_config(
            design,
            checkpoint,
            endpoints,
            TcpRuntimeTransportConfig::default(),
        )
    }

    pub fn recover_tcp_workers_from_checkpoint_with_config(
        &mut self,
        design: &Design,
        checkpoint: &RuntimeCheckpoint,
        endpoints: HashMap<String, SocketAddr>,
        transport: TcpRuntimeTransportConfig,
    ) -> Result<RuntimeRecoveryReport, ErrorReport> {
        self.validate_checkpoint(checkpoint)?;
        self.validate_snapshot(&checkpoint.snapshot)?;

        let compiled = compile(design)?;
        let program =
            lower_to_packed_program(&compiled, &checkpoint.module_name).map_err(|report| {
                let mut diagnostics = vec![Diagnostic::new(
                    "E_RUNTIME_CHECKPOINT_PROGRAM",
                    "recovery design cannot be lowered for the checkpoint module",
                )];
                diagnostics.extend(report.diagnostics);
                ErrorReport::new(diagnostics)
            })?;
        if program != self.program {
            return Err(error(
                "E_RUNTIME_CHECKPOINT_PROGRAM",
                "recovery design does not match checkpoint runtime program",
            ));
        }

        for shard in &self.shards {
            if !endpoints.contains_key(&shard.info.worker_id) {
                return Err(error(
                    "E_RUNTIME_TCP_ENDPOINT",
                    format!("missing TCP endpoint for worker `{}`", shard.info.worker_id),
                ));
            }
        }
        for worker_id in endpoints.keys() {
            if !self
                .shards
                .iter()
                .any(|shard| shard.info.worker_id == *worker_id)
            {
                return Err(error(
                    "E_RUNTIME_TCP_ENDPOINT",
                    format!("recovery endpoint provided for unknown worker `{worker_id}`"),
                ));
            }
        }

        let mut recovered = Vec::with_capacity(self.shards.len());
        let mut new_engines = Vec::with_capacity(self.shards.len());
        for (shard, snapshot) in self.shards.iter().zip(&checkpoint.snapshot.shards) {
            let endpoint = endpoints[&shard.info.worker_id];
            let mut client = TcpRuntimeShardClient::connect_with_config(
                shard.info.worker_id.clone(),
                endpoint,
                transport,
            )?;
            expect_ack(
                client.request(RuntimeWorkerRequest::InitShard(RuntimeShardInit {
                    design: compiled.clone(),
                    module_name: checkpoint.module_name.clone(),
                    shard: shard.info.clone(),
                    backend: shard.info.backend,
                }))?,
            )?;
            expect_ack(client.request(RuntimeWorkerRequest::RestoreSnapshot {
                worker_id: shard.info.worker_id.clone(),
                snapshot: snapshot.clone(),
            })?)?;
            recovered.push(shard.info.worker_id.clone());
            new_engines.push(RuntimeEngine::Worker(Box::new(client)));
        }

        for (shard, engine) in self.shards.iter_mut().zip(new_engines) {
            shard.engine = engine;
            shard.operations = RuntimeOperationStats::default();
        }

        Ok(RuntimeRecoveryReport {
            recovered_workers: recovered,
        })
    }

    pub fn set_input(&mut self, signal: Signal, lane_values: &[u128]) -> Result<(), ErrorReport> {
        self.validate_lane_count(lane_values.len())?;
        for shard in &mut self.shards {
            let range = shard.range();
            shard.engine.set_signal(signal, &lane_values[range])?;
        }
        Ok(())
    }

    pub fn set_input_limbs(
        &mut self,
        signal: Signal,
        lane_values: &[Vec<u32>],
    ) -> Result<(), ErrorReport> {
        self.validate_lane_count(lane_values.len())?;
        for shard in &mut self.shards {
            let range = shard.range();
            shard.engine.set_signal_limbs(signal, &lane_values[range])?;
        }
        Ok(())
    }

    pub fn get_signal(&mut self, signal: Signal) -> Result<Vec<u128>, ErrorReport> {
        let mut out = Vec::with_capacity(self.total_lanes);
        for shard in &mut self.shards {
            out.extend(shard.engine.get_signal(signal)?);
        }
        Ok(out)
    }

    pub fn get_signal_limbs(&mut self, signal: Signal) -> Result<Vec<Vec<u32>>, ErrorReport> {
        let mut out = Vec::with_capacity(self.total_lanes);
        for shard in &mut self.shards {
            out.extend(shard.engine.get_signal_limbs(signal)?);
        }
        Ok(out)
    }

    pub fn set_memory(
        &mut self,
        memory: Signal,
        lane_words: &[Vec<u128>],
    ) -> Result<(), ErrorReport> {
        self.validate_lane_count(lane_words.len())?;
        let ty = self.memory_ty(memory)?;
        if ty.width > 128 {
            return Err(error(
                "E_RUNTIME_WIDE_MEMORY",
                "use set_memory_limbs for memories wider than 128 bits",
            ));
        }
        let lane_limbs = lane_words
            .iter()
            .map(|words| {
                words
                    .iter()
                    .copied()
                    .map(|word| encode_u128_limbs(word, ty))
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
        self.validate_lane_count(lane_words.len())?;
        for shard in &mut self.shards {
            let range = shard.range();
            shard.engine.set_memory_limbs(memory, &lane_words[range])?;
        }
        Ok(())
    }

    pub fn get_memory(&mut self, memory: Signal) -> Result<Vec<Vec<u128>>, ErrorReport> {
        let ty = self.memory_ty(memory)?;
        if ty.width > 128 {
            return Err(error(
                "E_RUNTIME_WIDE_MEMORY",
                "use get_memory_limbs for memories wider than 128 bits",
            ));
        }
        Ok(self
            .get_memory_limbs(memory)?
            .iter()
            .map(|lane| {
                lane.iter()
                    .map(|word| decode_u128_limbs(word))
                    .collect::<Vec<_>>()
            })
            .collect())
    }

    pub fn get_memory_limbs(&mut self, memory: Signal) -> Result<Vec<Vec<Vec<u32>>>, ErrorReport> {
        let mut out = Vec::with_capacity(self.total_lanes);
        for shard in &mut self.shards {
            out.extend(shard.engine.get_memory_limbs(memory)?);
        }
        Ok(out)
    }

    pub fn eval_combinational(&mut self) -> Result<(), ErrorReport> {
        self.execute_shards(RuntimeShardAction::EvalCombinational)
    }

    pub fn tick(&mut self) -> Result<(), ErrorReport> {
        self.execute_shards(RuntimeShardAction::Tick)
    }

    pub fn tick_many(&mut self, steps: usize) -> Result<(), ErrorReport> {
        self.execute_shards(RuntimeShardAction::TickMany(steps))
    }

    fn tick_many_with_checkpoint_sink<F>(
        &mut self,
        steps: usize,
        cadence: RuntimeCheckpointCadence,
        tcp_endpoints: &HashMap<String, SocketAddr>,
        mut on_checkpoint: F,
    ) -> Result<RuntimeCheckpointRunReport, ErrorReport>
    where
        F: FnMut(RuntimeCheckpointEvent, &RuntimeCheckpoint) -> Result<(), ErrorReport>,
    {
        if cadence.every_steps == 0 {
            return Err(error(
                "E_RUNTIME_CHECKPOINT_CADENCE",
                "runtime checkpoint cadence every_steps must be greater than zero",
            ));
        }

        let mut completed_steps = 0;
        let mut checkpoints_emitted = 0;
        let mut last_checkpoint_step = None;

        if cadence.include_initial {
            self.emit_checkpoint(
                RuntimeCheckpointEvent {
                    completed_steps,
                    reason: RuntimeCheckpointReason::Initial,
                },
                tcp_endpoints,
                &mut on_checkpoint,
            )?;
            checkpoints_emitted += 1;
            last_checkpoint_step = Some(completed_steps);
        }

        while completed_steps < steps {
            let remaining = steps - completed_steps;
            let steps_to_cadence = cadence.every_steps - (completed_steps % cadence.every_steps);
            let chunk = remaining.min(steps_to_cadence);
            self.tick_many(chunk)?;
            completed_steps += chunk;

            if completed_steps % cadence.every_steps == 0 {
                self.emit_checkpoint(
                    RuntimeCheckpointEvent {
                        completed_steps,
                        reason: RuntimeCheckpointReason::Cadence,
                    },
                    tcp_endpoints,
                    &mut on_checkpoint,
                )?;
                checkpoints_emitted += 1;
                last_checkpoint_step = Some(completed_steps);
            }
        }

        if cadence.include_final && last_checkpoint_step != Some(completed_steps) {
            self.emit_checkpoint(
                RuntimeCheckpointEvent {
                    completed_steps,
                    reason: RuntimeCheckpointReason::Final,
                },
                tcp_endpoints,
                &mut on_checkpoint,
            )?;
            checkpoints_emitted += 1;
        }

        Ok(RuntimeCheckpointRunReport {
            requested_steps: steps,
            completed_steps,
            checkpoints_emitted,
        })
    }

    fn emit_checkpoint<F>(
        &mut self,
        event: RuntimeCheckpointEvent,
        tcp_endpoints: &HashMap<String, SocketAddr>,
        on_checkpoint: &mut F,
    ) -> Result<(), ErrorReport>
    where
        F: FnMut(RuntimeCheckpointEvent, &RuntimeCheckpoint) -> Result<(), ErrorReport>,
    {
        let checkpoint = self.checkpoint_with_tcp_endpoints(tcp_endpoints)?;
        on_checkpoint(event, &checkpoint)
    }

    fn validate_lane_count(&self, got: usize) -> Result<(), ErrorReport> {
        if got != self.total_lanes {
            return Err(error(
                "E_RUNTIME_LANE_VALUES",
                format!("expected {} lane values, got {got}", self.total_lanes),
            ));
        }
        Ok(())
    }

    fn runtime_topology(&self) -> RuntimeTopology {
        RuntimeTopology {
            workers: self
                .shards
                .iter()
                .map(|shard| RuntimeWorker {
                    id: shard.info.worker_id.clone(),
                    node: shard.info.node.clone(),
                    backend: shard.info.backend,
                    lanes: shard.info.lanes,
                })
                .collect(),
        }
    }

    fn validate_checkpoint(&self, checkpoint: &RuntimeCheckpoint) -> Result<(), ErrorReport> {
        checkpoint.validate_format_version()?;
        if checkpoint.module_name != self.program.top {
            return Err(error(
                "E_RUNTIME_CHECKPOINT_PROGRAM",
                format!(
                    "checkpoint module `{}` does not match runtime program `{}`",
                    checkpoint.module_name, self.program.top
                ),
            ));
        }
        if checkpoint.topology != self.runtime_topology() {
            return Err(error(
                "E_RUNTIME_CHECKPOINT_TOPOLOGY",
                "checkpoint topology does not match runtime topology",
            ));
        }
        Ok(())
    }

    fn validate_snapshot(&self, snapshot: &RuntimeSnapshot) -> Result<(), ErrorReport> {
        if snapshot.total_lanes != self.total_lanes {
            return Err(error(
                "E_RUNTIME_SNAPSHOT_TOPOLOGY",
                format!(
                    "snapshot has {} lanes, runtime has {}",
                    snapshot.total_lanes, self.total_lanes
                ),
            ));
        }
        if snapshot.shards.len() != self.shards.len() {
            return Err(error(
                "E_RUNTIME_SNAPSHOT_TOPOLOGY",
                format!(
                    "snapshot has {} shards, runtime has {}",
                    snapshot.shards.len(),
                    self.shards.len()
                ),
            ));
        }
        if snapshot.program_top != self.program.top {
            return Err(error(
                "E_RUNTIME_SNAPSHOT_PROGRAM",
                format!(
                    "snapshot program `{}` does not match runtime program `{}`",
                    snapshot.program_top, self.program.top
                ),
            ));
        }
        if snapshot.signal_words_per_lane != self.program.total_signal_words {
            return Err(error(
                "E_RUNTIME_SNAPSHOT_PROGRAM",
                format!(
                    "snapshot has {} signal words per lane, runtime has {}",
                    snapshot.signal_words_per_lane, self.program.total_signal_words
                ),
            ));
        }
        if snapshot.memory_words_per_lane != self.program.total_memory_words_per_lane {
            return Err(error(
                "E_RUNTIME_SNAPSHOT_PROGRAM",
                format!(
                    "snapshot has {} memory words per lane, runtime has {}",
                    snapshot.memory_words_per_lane, self.program.total_memory_words_per_lane
                ),
            ));
        }

        for (index, (snapshot_shard, runtime_shard)) in
            snapshot.shards.iter().zip(&self.shards).enumerate()
        {
            if snapshot_shard.shard.start_lane != runtime_shard.info.start_lane
                || snapshot_shard.shard.lanes != runtime_shard.info.lanes
            {
                return Err(error(
                    "E_RUNTIME_SNAPSHOT_TOPOLOGY",
                    format!(
                        "snapshot shard {index} covers lanes {}..{}, runtime shard covers lanes {}..{}",
                        snapshot_shard.shard.start_lane,
                        snapshot_shard.shard.start_lane + snapshot_shard.shard.lanes,
                        runtime_shard.info.start_lane,
                        runtime_shard.info.start_lane + runtime_shard.info.lanes
                    ),
                ));
            }

            let expected_values = self.program.total_signal_words * runtime_shard.info.lanes;
            if snapshot_shard.values.len() != expected_values {
                return Err(error(
                    "E_RUNTIME_SNAPSHOT_STORAGE",
                    format!(
                        "snapshot shard {index} has {} signal words, expected {expected_values}",
                        snapshot_shard.values.len()
                    ),
                ));
            }
            let expected_memories =
                self.program.total_memory_words_per_lane * runtime_shard.info.lanes;
            if snapshot_shard.memories.len() != expected_memories {
                return Err(error(
                    "E_RUNTIME_SNAPSHOT_STORAGE",
                    format!(
                        "snapshot shard {index} has {} memory words, expected {expected_memories}",
                        snapshot_shard.memories.len()
                    ),
                ));
            }
        }
        Ok(())
    }

    fn memory_ty(&self, memory: Signal) -> Result<BitType, ErrorReport> {
        self.program
            .memories
            .iter()
            .find(|packed| packed.source == memory)
            .map(|packed| packed.data_layout.ty)
            .ok_or_else(|| {
                error(
                    "E_RUNTIME_MEMORY",
                    format!("memory {:?} is not part of this runtime module", memory.id),
                )
            })
    }

    fn execute_shards(&mut self, action: RuntimeShardAction) -> Result<(), ErrorReport> {
        let started = Instant::now();
        let result = match self.options.execution_mode {
            RuntimeExecutionMode::Serial => {
                let results = self
                    .shards
                    .iter_mut()
                    .map(|shard| shard.execute(action))
                    .collect::<Vec<_>>();
                self.operations.record(action, started.elapsed().as_nanos());
                merge_ordered_results(results)
            }
            RuntimeExecutionMode::Parallel => {
                let results = std::thread::scope(|scope| {
                    let handles = self
                        .shards
                        .iter_mut()
                        .map(|shard| scope.spawn(move || shard.execute(action)))
                        .collect::<Vec<_>>();
                    handles
                        .into_iter()
                        .map(|handle| match handle.join() {
                            Ok(result) => result,
                            Err(payload) => Err(parallel_panic_error(payload)),
                        })
                        .collect::<Vec<_>>()
                });
                self.operations.record(action, started.elapsed().as_nanos());
                merge_ordered_results(results)
            }
        };
        if result.is_ok() {
            self.record_surrogate_execution(action);
        }
        result
    }

    fn record_surrogate_execution(&mut self, action: RuntimeShardAction) {
        let Some(attachment) = self.surrogate_attachment.as_ref() else {
            return;
        };
        let previous_count = self
            .surrogate_execution
            .as_ref()
            .map_or(0, |report| report.action_count);
        self.surrogate_execution = Some(RuntimeSurrogateExecutionReport::from_attachment(
            attachment,
            Some(runtime_surrogate_action_kind(action)),
            previous_count + 1,
        ));
    }
}

fn runtime_surrogate_action_kind(action: RuntimeShardAction) -> RuntimeSurrogateActionKind {
    match action {
        RuntimeShardAction::EvalCombinational => RuntimeSurrogateActionKind::EvalCombinational,
        RuntimeShardAction::Tick => RuntimeSurrogateActionKind::Tick,
        RuntimeShardAction::TickMany(_) => RuntimeSurrogateActionKind::TickMany,
    }
}

fn partition_surrogate_action_kind(
    action: RuntimePartitionWorkerActionKind,
) -> RuntimeSurrogateActionKind {
    match action {
        RuntimePartitionWorkerActionKind::EvalCombinational => {
            RuntimeSurrogateActionKind::PartitionEvalCombinational
        }
        RuntimePartitionWorkerActionKind::Tick => RuntimeSurrogateActionKind::PartitionTick,
        RuntimePartitionWorkerActionKind::TickMany(_) => {
            RuntimeSurrogateActionKind::PartitionTickMany
        }
    }
}

struct RuntimeShard {
    info: RuntimeShardInfo,
    engine: RuntimeEngine,
    operations: RuntimeOperationStats,
}

impl RuntimeShard {
    fn range(&self) -> std::ops::Range<usize> {
        self.info.start_lane..self.info.start_lane + self.info.lanes
    }

    fn health(&mut self) -> Result<RuntimeShardHealth, ErrorReport> {
        self.engine.health(&self.info, self.operations)
    }

    fn execute(&mut self, action: RuntimeShardAction) -> Result<(), ErrorReport> {
        let started = Instant::now();
        let result = self.engine.execute(action);
        self.operations.record(action, started.elapsed().as_nanos());
        result
    }
}

enum RuntimeEngine {
    ScalarCpu(SingleLaneMachineSimulator),
    PackedCpu(PackedSimulator),
    SimdCpu(SimdCpuSimulator),
    JitCpu(JitCpuSimulator),
    Gpu(GpuBatchSimulator),
    Worker(Box<dyn RuntimeShardClient + Send>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuntimeShardAction {
    EvalCombinational,
    Tick,
    TickMany(usize),
}

impl RuntimeEngine {
    fn health(
        &mut self,
        expected: &RuntimeShardInfo,
        operations: RuntimeOperationStats,
    ) -> Result<RuntimeShardHealth, ErrorReport> {
        match self {
            Self::ScalarCpu(_)
            | Self::PackedCpu(_)
            | Self::SimdCpu(_)
            | Self::JitCpu(_)
            | Self::Gpu(_) => Ok(RuntimeShardHealth {
                shard: expected.clone(),
                status: RuntimeShardHealthStatus::Healthy,
                operations: Some(operations),
            }),
            Self::Worker(client) => {
                let health = client.health()?;
                Ok(RuntimeShardHealth {
                    shard: health.shard.unwrap_or_else(|| expected.clone()),
                    status: if health.initialized {
                        RuntimeShardHealthStatus::Healthy
                    } else {
                        RuntimeShardHealthStatus::Uninitialized
                    },
                    operations: health.operations,
                })
            }
        }
    }

    fn snapshot_storage(&mut self) -> Result<PackedSimulatorStorage, ErrorReport> {
        match self {
            Self::ScalarCpu(sim) => Ok(sim.snapshot_storage()),
            Self::PackedCpu(sim) => Ok(sim.snapshot_storage()),
            Self::SimdCpu(sim) => Ok(sim.snapshot_storage()),
            Self::JitCpu(sim) => Ok(sim.snapshot_storage()),
            Self::Gpu(sim) => sim.snapshot_storage(),
            Self::Worker(client) => {
                let worker_id = client.worker_id().to_string();
                match client.request(RuntimeWorkerRequest::Snapshot { worker_id })? {
                    RuntimeWorkerResponse::Snapshot(snapshot) => Ok(PackedSimulatorStorage {
                        values: snapshot.values,
                        memories: snapshot.memories,
                    }),
                    response => Err(unexpected_worker_response("Snapshot", response)),
                }
            }
        }
    }

    fn restore_storage(&mut self, storage: &PackedSimulatorStorage) -> Result<(), ErrorReport> {
        match self {
            Self::ScalarCpu(sim) => sim.restore_storage(storage),
            Self::PackedCpu(sim) => sim.restore_storage(storage),
            Self::SimdCpu(sim) => sim.restore_storage(storage),
            Self::JitCpu(sim) => sim.restore_storage(storage),
            Self::Gpu(sim) => sim.restore_storage(storage),
            Self::Worker(client) => {
                let worker_id = client.worker_id().to_string();
                let stats = match client.request(RuntimeWorkerRequest::Stats {
                    worker_id: worker_id.clone(),
                })? {
                    RuntimeWorkerResponse::Stats(stats) => stats,
                    response => return Err(unexpected_worker_response("Stats", response)),
                };
                expect_ack(client.request(RuntimeWorkerRequest::RestoreSnapshot {
                    worker_id,
                    snapshot: RuntimeShardSnapshot {
                        shard: stats.shard,
                        values: storage.values.clone(),
                        memories: storage.memories.clone(),
                    },
                })?)
            }
        }
    }

    fn set_signal(&mut self, signal: Signal, values: &[u128]) -> Result<(), ErrorReport> {
        match self {
            Self::ScalarCpu(sim) => {
                if values.len() != 1 {
                    return Err(error(
                        "E_RUNTIME_LANE_VALUES",
                        format!("expected one scalar lane value, got {}", values.len()),
                    ));
                }
                sim.set_signal(signal, values[0])
            }
            Self::PackedCpu(sim) => sim.set_signal(signal, values),
            Self::SimdCpu(sim) => sim.set_signal(signal, values),
            Self::JitCpu(sim) => sim.set_signal(signal, values),
            Self::Gpu(sim) => {
                let values = values
                    .iter()
                    .map(|value| {
                        u32::try_from(*value).map_err(|_| {
                            error(
                                "E_RUNTIME_WIDE_SIGNAL",
                                "use set_input_limbs for GPU signal values wider than 32 bits",
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                sim.set_input(signal, &values)
            }
            Self::Worker(client) => {
                let worker_id = client.worker_id().to_string();
                expect_ack(client.request(RuntimeWorkerRequest::SetInput {
                    worker_id,
                    signal,
                    lane_values: values.to_vec(),
                })?)
            }
        }
    }

    fn set_signal_limbs(&mut self, signal: Signal, values: &[Vec<u32>]) -> Result<(), ErrorReport> {
        match self {
            Self::ScalarCpu(sim) => sim.set_signal_limbs(signal, values),
            Self::PackedCpu(sim) => sim.set_signal_limbs(signal, values),
            Self::SimdCpu(sim) => sim.set_signal_limbs(signal, values),
            Self::JitCpu(sim) => sim.set_signal_limbs(signal, values),
            Self::Gpu(sim) => sim.set_input_limbs(signal, values),
            Self::Worker(client) => {
                let worker_id = client.worker_id().to_string();
                expect_ack(client.request(RuntimeWorkerRequest::SetInputLimbs {
                    worker_id,
                    signal,
                    lane_values: values.to_vec(),
                })?)
            }
        }
    }

    fn get_signal(&mut self, signal: Signal) -> Result<Vec<u128>, ErrorReport> {
        match self {
            Self::ScalarCpu(sim) => Ok(vec![sim.get_signal(signal)?]),
            Self::PackedCpu(sim) => sim.get_signal(signal),
            Self::SimdCpu(sim) => sim.get_signal(signal),
            Self::JitCpu(sim) => sim.get_signal(signal),
            Self::Gpu(sim) => Ok(sim
                .get_signal_limbs(signal)?
                .into_iter()
                .map(|limbs| decode_u128_limbs(&limbs))
                .collect()),
            Self::Worker(_) => Ok(self
                .get_signal_limbs(signal)?
                .into_iter()
                .map(|limbs| decode_u128_limbs(&limbs))
                .collect()),
        }
    }

    fn get_signal_limbs(&mut self, signal: Signal) -> Result<Vec<Vec<u32>>, ErrorReport> {
        match self {
            Self::ScalarCpu(sim) => sim.get_signal_limbs(signal),
            Self::PackedCpu(sim) => sim.get_signal_limbs(signal),
            Self::SimdCpu(sim) => sim.get_signal_limbs(signal),
            Self::JitCpu(sim) => sim.get_signal_limbs(signal),
            Self::Gpu(sim) => sim.get_signal_limbs(signal),
            Self::Worker(client) => {
                let worker_id = client.worker_id().to_string();
                match client.request(RuntimeWorkerRequest::GetSignalLimbs { worker_id, signal })? {
                    RuntimeWorkerResponse::SignalLimbs(values) => Ok(values),
                    response => Err(unexpected_worker_response("GetSignalLimbs", response)),
                }
            }
        }
    }

    fn set_memory_limbs(
        &mut self,
        memory: Signal,
        values: &[Vec<Vec<u32>>],
    ) -> Result<(), ErrorReport> {
        match self {
            Self::ScalarCpu(sim) => sim.set_memory_limbs(memory, values),
            Self::PackedCpu(sim) => sim.set_memory_limbs(memory, values),
            Self::SimdCpu(sim) => sim.set_memory_limbs(memory, values),
            Self::JitCpu(sim) => sim.set_memory_limbs(memory, values),
            Self::Gpu(sim) => sim.set_memory_limbs(memory, values),
            Self::Worker(client) => {
                let worker_id = client.worker_id().to_string();
                expect_ack(client.request(RuntimeWorkerRequest::SetMemoryLimbs {
                    worker_id,
                    memory,
                    lane_words: values.to_vec(),
                })?)
            }
        }
    }

    fn get_memory_limbs(&mut self, memory: Signal) -> Result<Vec<Vec<Vec<u32>>>, ErrorReport> {
        match self {
            Self::ScalarCpu(sim) => sim.get_memory_limbs(memory),
            Self::PackedCpu(sim) => sim.get_memory_limbs(memory),
            Self::SimdCpu(sim) => sim.get_memory_limbs(memory),
            Self::JitCpu(sim) => sim.get_memory_limbs(memory),
            Self::Gpu(sim) => sim.get_memory_limbs(memory),
            Self::Worker(client) => {
                let worker_id = client.worker_id().to_string();
                match client.request(RuntimeWorkerRequest::GetMemoryLimbs { worker_id, memory })? {
                    RuntimeWorkerResponse::MemoryLimbs(values) => Ok(values),
                    response => Err(unexpected_worker_response("GetMemoryLimbs", response)),
                }
            }
        }
    }

    fn eval_combinational(&mut self) -> Result<(), ErrorReport> {
        match self {
            Self::ScalarCpu(sim) => {
                sim.eval_combinational();
                Ok(())
            }
            Self::PackedCpu(sim) => {
                sim.eval_combinational();
                Ok(())
            }
            Self::SimdCpu(sim) => {
                sim.eval_combinational()?;
                Ok(())
            }
            Self::JitCpu(sim) => {
                sim.eval_combinational()?;
                Ok(())
            }
            Self::Gpu(sim) => sim.eval_combinational(),
            Self::Worker(client) => {
                let worker_id = client.worker_id().to_string();
                expect_ack(client.request(RuntimeWorkerRequest::EvalCombinational { worker_id })?)
            }
        }
    }

    fn tick(&mut self) -> Result<(), ErrorReport> {
        match self {
            Self::ScalarCpu(sim) => {
                sim.tick();
                Ok(())
            }
            Self::PackedCpu(sim) => {
                sim.tick();
                Ok(())
            }
            Self::SimdCpu(sim) => {
                sim.tick()?;
                Ok(())
            }
            Self::JitCpu(sim) => {
                sim.tick()?;
                Ok(())
            }
            Self::Gpu(sim) => sim.tick(),
            Self::Worker(client) => {
                let worker_id = client.worker_id().to_string();
                expect_ack(client.request(RuntimeWorkerRequest::Tick { worker_id })?)
            }
        }
    }

    fn tick_many(&mut self, steps: usize) -> Result<(), ErrorReport> {
        match self {
            Self::ScalarCpu(sim) => {
                sim.tick_many(steps);
                Ok(())
            }
            Self::PackedCpu(sim) => {
                sim.tick_many(steps);
                Ok(())
            }
            Self::SimdCpu(sim) => {
                sim.tick_many(steps)?;
                Ok(())
            }
            Self::JitCpu(sim) => {
                sim.tick_many(steps)?;
                Ok(())
            }
            Self::Gpu(sim) => sim.tick_many(steps),
            Self::Worker(client) => {
                let worker_id = client.worker_id().to_string();
                expect_ack(client.request(RuntimeWorkerRequest::TickMany { worker_id, steps })?)
            }
        }
    }

    fn execute(&mut self, action: RuntimeShardAction) -> Result<(), ErrorReport> {
        match action {
            RuntimeShardAction::EvalCombinational => self.eval_combinational(),
            RuntimeShardAction::Tick => self.tick(),
            RuntimeShardAction::TickMany(steps) => self.tick_many(steps),
        }
    }
}

fn encode_u128_limbs(value: u128, ty: BitType) -> Vec<u32> {
    let mut encoded = value;
    let count = limbs(ty.width);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(encoded as u32);
        encoded >>= 32;
    }
    if let Some(last) = out.last_mut() {
        *last &= final_limb_mask(ty.width);
    }
    out
}

fn decode_u128_limbs(limbs: &[u32]) -> u128 {
    limbs
        .iter()
        .copied()
        .take(4)
        .enumerate()
        .fold(0u128, |acc, (index, limb)| {
            acc | ((limb as u128) << (index * 32))
        })
}

fn merge_ordered_results(results: Vec<Result<(), ErrorReport>>) -> Result<(), ErrorReport> {
    let diagnostics = results
        .into_iter()
        .filter_map(Result::err)
        .flat_map(|report| report.diagnostics)
        .collect::<Vec<_>>();
    if diagnostics.is_empty() {
        Ok(())
    } else {
        Err(ErrorReport::new(diagnostics))
    }
}

fn evaluate_runtime_autotune_candidate(
    design: &Design,
    module_name: &str,
    candidate: RuntimeAutotuneCandidate,
    config: &RuntimeAutotuneConfig,
) -> RuntimeAutotuneCandidateReport {
    let RuntimeAutotuneCandidate {
        name,
        topology,
        options,
    } = candidate;

    let result: Result<(RuntimeStats, u128), ErrorReport> = (|| {
        let mut runtime =
            DistributedRuntime::new_with_options(design, module_name, topology.clone(), options)?;
        if let Some(stimulus) = config.stimulus.as_ref() {
            apply_stimulus_setup(&mut runtime, &stimulus.setup)?;
        }
        run_autotune_cycles(&mut runtime, config.stimulus.as_ref(), config.warmup_steps)?;
        runtime.reset_stats();
        run_autotune_cycles(&mut runtime, config.stimulus.as_ref(), config.measure_steps)?;
        let stats = runtime.stats();
        let score_ns = autotune_score(&stats, config.stimulus.as_ref());
        Ok((stats, score_ns))
    })();

    match result {
        Ok((stats, score_ns)) => RuntimeAutotuneCandidateReport {
            name,
            topology,
            options,
            stats: Some(stats),
            diagnostics: Vec::new(),
            score_ns: Some(score_ns),
        },
        Err(report) => RuntimeAutotuneCandidateReport {
            name,
            topology,
            options,
            stats: None,
            diagnostics: report.diagnostics,
            score_ns: None,
        },
    }
}

fn apply_stimulus_setup(
    runtime: &mut DistributedRuntime,
    setup: &RuntimeStimulusSetup,
) -> Result<(), ErrorReport> {
    for input in &setup.inputs {
        runtime.set_input(input.signal, &input.lane_values)?;
    }
    for input in &setup.input_limbs {
        runtime.set_input_limbs(input.signal, &input.lane_values)?;
    }
    for memory in &setup.memories {
        runtime.set_memory(memory.memory, &memory.lane_words)?;
    }
    for memory in &setup.memory_limbs {
        runtime.set_memory_limbs(memory.memory, &memory.lane_words)?;
    }
    Ok(())
}

fn apply_stimulus_step(
    runtime: &mut DistributedRuntime,
    step: &RuntimeStimulusStep,
) -> Result<(), ErrorReport> {
    for input in &step.inputs {
        runtime.set_input(input.signal, &input.lane_values)?;
    }
    for input in &step.input_limbs {
        runtime.set_input_limbs(input.signal, &input.lane_values)?;
    }
    Ok(())
}

fn run_autotune_cycles(
    runtime: &mut DistributedRuntime,
    stimulus: Option<&RuntimeAutotuneStimulus>,
    cycles: usize,
) -> Result<(), ErrorReport> {
    if cycles == 0 {
        return Ok(());
    }

    if let Some(stimulus) = stimulus.filter(|stimulus| !stimulus.steps.is_empty()) {
        for cycle in 0..cycles {
            apply_stimulus_step(runtime, &stimulus.steps[cycle % stimulus.steps.len()])?;
            runtime.tick()?;
        }
        Ok(())
    } else {
        runtime.tick_many(cycles)
    }
}

fn autotune_score(stats: &RuntimeStats, stimulus: Option<&RuntimeAutotuneStimulus>) -> u128 {
    if stimulus.is_some_and(|stimulus| !stimulus.steps.is_empty()) {
        stats.operations.tick.total_ns
    } else {
        stats.operations.tick_many.total_ns
    }
}

fn expect_ack(response: RuntimeWorkerResponse) -> Result<(), ErrorReport> {
    match response {
        RuntimeWorkerResponse::Ack => Ok(()),
        response => Err(unexpected_worker_response("Ack", response)),
    }
}

fn unexpected_worker_response(
    expected: &'static str,
    response: RuntimeWorkerResponse,
) -> ErrorReport {
    error(
        "E_RUNTIME_WORKER_RESPONSE",
        format!("expected worker response {expected}, got {response:?}"),
    )
}

fn worker_process_command(config: &TcpRuntimeWorkerProcessConfig) -> Command {
    let mut command = Command::new(&config.executable);
    command.arg("--bind").arg(config.bind_addr.to_string());
    command
        .arg("--read-timeout-ms")
        .arg(timeout_arg(config.read_timeout));
    command
        .arg("--write-timeout-ms")
        .arg(timeout_arg(config.write_timeout));
    if let Some(connections) = config.max_connections {
        command
            .arg("--max-connections")
            .arg(connections.to_string());
    }
    command
}

fn timeout_arg(timeout: Option<Duration>) -> String {
    timeout
        .map(|timeout| timeout.as_millis().to_string())
        .unwrap_or_else(|| "none".to_string())
}

#[derive(Deserialize)]
struct RuntimeWorkerStartupLine {
    addr: String,
}

fn parse_worker_startup_addr(line: &str) -> Result<SocketAddr, ErrorReport> {
    let startup: RuntimeWorkerStartupLine = serde_json::from_str(line).map_err(|err| {
        error(
            "E_RUNTIME_WORKER_PROCESS_STARTUP",
            format!("worker process startup line is not valid JSON: {err}"),
        )
    })?;
    startup.addr.parse::<SocketAddr>().map_err(|err| {
        error(
            "E_RUNTIME_WORKER_PROCESS_STARTUP",
            format!(
                "worker process startup address `{}` is not a socket address: {err}",
                startup.addr
            ),
        )
    })
}

fn worker_process_exit(
    worker_id: String,
    status: ExitStatus,
) -> Result<TcpRuntimeWorkerProcessExit, ErrorReport> {
    Ok(TcpRuntimeWorkerProcessExit {
        worker_id,
        success: status.success(),
        code: status.code(),
    })
}

fn tcp_endpoint_list(endpoints: &HashMap<String, SocketAddr>) -> Vec<RuntimeTcpEndpoint> {
    let mut endpoints = endpoints
        .iter()
        .map(|(worker_id, addr)| RuntimeTcpEndpoint {
            worker_id: worker_id.clone(),
            addr: addr.to_string(),
        })
        .collect::<Vec<_>>();
    endpoints.sort_by(|left, right| left.worker_id.cmp(&right.worker_id));
    endpoints
}

fn write_json_line<T: Serialize>(writer: &mut impl Write, value: &T) -> Result<(), ErrorReport> {
    serde_json::to_writer(&mut *writer, value).map_err(tcp_json_error)?;
    writer.write_all(b"\n").map_err(tcp_io_error)?;
    writer.flush().map_err(tcp_io_error)
}

fn read_worker_wire_response(
    reader: &mut impl BufRead,
) -> Result<RuntimeWorkerWireResponse, ErrorReport> {
    read_json_line(
        reader,
        "TCP worker response ended before newline frame delimiter",
    )
}

fn read_json_line<T: for<'de> Deserialize<'de>>(
    reader: &mut impl BufRead,
    eof_message: &'static str,
) -> Result<T, ErrorReport> {
    let mut line = String::new();
    let bytes = reader.read_line(&mut line).map_err(tcp_io_error)?;
    if bytes == 0 || !line.ends_with('\n') {
        return Err(error("E_RUNTIME_TCP_EOF", eof_message));
    }
    serde_json::from_str(line.trim_end_matches('\n')).map_err(tcp_json_error)
}

fn configure_tcp_stream(
    stream: &TcpStream,
    config: TcpRuntimeTransportConfig,
) -> Result<(), ErrorReport> {
    stream
        .set_read_timeout(config.read_timeout)
        .map_err(tcp_io_error)?;
    stream
        .set_write_timeout(config.write_timeout)
        .map_err(tcp_io_error)?;
    stream.set_nodelay(true).map_err(tcp_io_error)
}

fn first_socket_addr<A: ToSocketAddrs>(addr: A) -> Result<SocketAddr, ErrorReport> {
    addr.to_socket_addrs()
        .map_err(tcp_io_error)?
        .next()
        .ok_or_else(|| error("E_RUNTIME_TCP_ADDR", "no TCP socket address resolved"))
}

fn tcp_io_error(err: std::io::Error) -> ErrorReport {
    let code = match err.kind() {
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock => "E_RUNTIME_TCP_TIMEOUT",
        std::io::ErrorKind::UnexpectedEof => "E_RUNTIME_TCP_EOF",
        _ => "E_RUNTIME_TCP_IO",
    };
    error(code, format!("TCP worker transport I/O error: {err}"))
}

fn tcp_json_error(err: serde_json::Error) -> ErrorReport {
    error(
        "E_RUNTIME_TCP_JSON",
        format!("TCP worker transport JSON error: {err}"),
    )
}

fn worker_process_io_error(err: std::io::Error) -> ErrorReport {
    error(
        "E_RUNTIME_WORKER_PROCESS_IO",
        format!("runtime worker process I/O error: {err}"),
    )
}

fn checkpoint_io_error(err: std::io::Error) -> ErrorReport {
    error(
        "E_RUNTIME_CHECKPOINT_IO",
        format!("runtime checkpoint I/O error: {err}"),
    )
}

fn checkpoint_json_error(err: serde_json::Error) -> ErrorReport {
    error(
        "E_RUNTIME_CHECKPOINT_JSON",
        format!("runtime checkpoint JSON error: {err}"),
    )
}

fn telemetry_io_error(err: std::io::Error) -> ErrorReport {
    error(
        "E_RUNTIME_TELEMETRY_IO",
        format!("runtime telemetry I/O error: {err}"),
    )
}

fn telemetry_json_error(err: serde_json::Error) -> ErrorReport {
    error(
        "E_RUNTIME_TELEMETRY_JSON",
        format!("runtime telemetry JSON error: {err}"),
    )
}

fn partition_plan_io_error(err: std::io::Error) -> ErrorReport {
    error(
        "E_RUNTIME_PARTITION_PLAN_IO",
        format!("runtime partition plan I/O error: {err}"),
    )
}

fn partition_plan_json_error(err: serde_json::Error) -> ErrorReport {
    error(
        "E_RUNTIME_PARTITION_PLAN_JSON",
        format!("runtime partition plan JSON error: {err}"),
    )
}

fn partition_placement_io_error(err: std::io::Error) -> ErrorReport {
    error(
        "E_RUNTIME_PARTITION_PLACEMENT_IO",
        format!("runtime partition placement I/O error: {err}"),
    )
}

fn partition_placement_json_error(err: serde_json::Error) -> ErrorReport {
    error(
        "E_RUNTIME_PARTITION_PLACEMENT_JSON",
        format!("runtime partition placement JSON error: {err}"),
    )
}

fn partition_communication_io_error(err: std::io::Error) -> ErrorReport {
    error(
        "E_RUNTIME_PARTITION_COMMUNICATION_IO",
        format!("runtime partition communication I/O error: {err}"),
    )
}

fn partition_communication_json_error(err: serde_json::Error) -> ErrorReport {
    error(
        "E_RUNTIME_PARTITION_COMMUNICATION_JSON",
        format!("runtime partition communication JSON error: {err}"),
    )
}

fn partition_bundle_io_error(err: std::io::Error) -> ErrorReport {
    error(
        "E_RUNTIME_PARTITION_BUNDLE_IO",
        format!("runtime partition bundle I/O error: {err}"),
    )
}

fn partition_bundle_json_error(err: serde_json::Error) -> ErrorReport {
    error(
        "E_RUNTIME_PARTITION_BUNDLE_JSON",
        format!("runtime partition bundle JSON error: {err}"),
    )
}

fn partition_launch_io_error(err: std::io::Error) -> ErrorReport {
    error(
        "E_RUNTIME_PARTITION_LAUNCH_IO",
        format!("runtime partition launch I/O error: {err}"),
    )
}

fn partition_launch_json_error(err: serde_json::Error) -> ErrorReport {
    error(
        "E_RUNTIME_PARTITION_LAUNCH_JSON",
        format!("runtime partition launch JSON error: {err}"),
    )
}

fn partition_session_telemetry_io_error(err: std::io::Error) -> ErrorReport {
    error(
        "E_RUNTIME_PARTITION_SESSION_TELEMETRY_IO",
        format!("runtime partition session telemetry I/O error: {err}"),
    )
}

fn partition_session_telemetry_json_error(err: serde_json::Error) -> ErrorReport {
    error(
        "E_RUNTIME_PARTITION_SESSION_TELEMETRY_JSON",
        format!("runtime partition session telemetry JSON error: {err}"),
    )
}

fn surrogate_attachment_io_error(err: std::io::Error) -> ErrorReport {
    error(
        "E_RUNTIME_SURROGATE_ATTACHMENT_IO",
        format!("runtime surrogate attachment I/O error: {err}"),
    )
}

fn surrogate_attachment_json_error(err: serde_json::Error) -> ErrorReport {
    error(
        "E_RUNTIME_SURROGATE_ATTACHMENT_JSON",
        format!("runtime surrogate attachment JSON error: {err}"),
    )
}

fn parallel_panic_error(payload: Box<dyn std::any::Any + Send>) -> ErrorReport {
    let message = if let Some(message) = payload.downcast_ref::<&'static str>() {
        format!("runtime shard panicked during parallel execution: {message}")
    } else if let Some(message) = payload.downcast_ref::<String>() {
        format!("runtime shard panicked during parallel execution: {message}")
    } else {
        "runtime shard panicked during parallel execution".to_string()
    };
    error("E_RUNTIME_PARALLEL", message)
}

fn error(code: &'static str, message: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![Diagnostic::new(code, message)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use rrtl_core::{lit_u, mux, uint};
    use rrtl_surrogate::{
        ArtifactFormat, EventRuntimePlan, EventRuntimePlanItem, EventRuntimeWorkerSummary,
        GemmRuntimePlanItem, GemmRuntimeWorkerSummary, ModelFamily, PolicyMode, Provenance,
    };

    fn test_partition(id: &str) -> RuntimePartition {
        RuntimePartition {
            id: id.to_string(),
            module_name: "Top".to_string(),
            instance_path: vec![id.to_string()],
            external: false,
            signals: Vec::new(),
            cost: RuntimePartitionCost::default(),
        }
    }

    fn test_boundary(
        kind: RuntimePartitionSignalKind,
        producer: Option<&str>,
        consumers: &[&str],
    ) -> RuntimePartitionBoundarySignal {
        RuntimePartitionBoundarySignal {
            signal: RuntimePartitionSignalRef {
                module_name: "Top".to_string(),
                signal_name: "sig".to_string(),
                signal: Signal {
                    module: rrtl_ir::ModuleId(0),
                    id: rrtl_ir::SignalId(0),
                },
                width: 8,
                kind,
            },
            instance_path: vec!["Top".to_string()],
            port_name: "p".to_string(),
            producer_partition: producer.map(|p| p.to_string()),
            consumer_partitions: consumers.iter().map(|c| c.to_string()).collect(),
            width: 8,
            cost: RuntimePartitionCost::default(),
        }
    }

    fn test_plan(
        partitions: &[&str],
        boundaries: Vec<RuntimePartitionBoundarySignal>,
    ) -> RuntimePartitionPlan {
        RuntimePartitionPlan {
            format_version: RUNTIME_PARTITION_PLAN_FORMAT_VERSION,
            module_name: "Top".to_string(),
            partitions: partitions.iter().map(|id| test_partition(id)).collect(),
            boundary_signals: boundaries,
            total_cost: RuntimePartitionCost::default(),
            diagnostics: Vec::new(),
            recommendations: Vec::new(),
        }
    }

    #[test]
    fn register_and_input_boundaries_are_legal_to_cut() {
        assert_eq!(
            classify_partition_boundary(&test_boundary(
                RuntimePartitionSignalKind::Register,
                Some("a"),
                &["b"],
            )),
            BoundaryLegality::RegisterStable,
        );
        assert_eq!(
            classify_partition_boundary(&test_boundary(
                RuntimePartitionSignalKind::Input,
                Some("a"),
                &["b"],
            )),
            BoundaryLegality::RegisterStable,
        );
        for kind in [
            RuntimePartitionSignalKind::Wire,
            RuntimePartitionSignalKind::Output,
            RuntimePartitionSignalKind::Inout,
            RuntimePartitionSignalKind::Memory,
        ] {
            assert_eq!(
                classify_partition_boundary(&test_boundary(kind, Some("a"), &["b"])),
                BoundaryLegality::CombinationalUnsafe,
                "{kind:?} must be unsafe in v0",
            );
        }
    }

    #[test]
    fn register_boundaries_require_no_merges() {
        let plan = test_plan(
            &["a", "b", "c"],
            vec![
                test_boundary(RuntimePartitionSignalKind::Register, Some("a"), &["b"]),
                test_boundary(RuntimePartitionSignalKind::Input, Some("b"), &["c"]),
            ],
        );
        assert!(legal_partition_merge_groups(&plan).is_empty());
    }

    #[test]
    fn combinational_boundary_merges_producer_and_consumer() {
        let plan = test_plan(
            &["a", "b", "c"],
            vec![test_boundary(
                RuntimePartitionSignalKind::Wire,
                Some("a"),
                &["b"],
            )],
        );
        // a and b merge; c stays independent (singletons omitted).
        assert_eq!(
            legal_partition_merge_groups(&plan),
            vec![vec!["a".to_string(), "b".to_string()]],
        );
    }

    #[test]
    fn combinational_boundaries_merge_transitively() {
        // a->b (wire) and b->c (wire) must collapse {a,b,c} into one group,
        // while d, reached only by a register edge, stays separate.
        let plan = test_plan(
            &["a", "b", "c", "d"],
            vec![
                test_boundary(RuntimePartitionSignalKind::Wire, Some("a"), &["b"]),
                test_boundary(RuntimePartitionSignalKind::Output, Some("b"), &["c"]),
                test_boundary(RuntimePartitionSignalKind::Register, Some("c"), &["d"]),
            ],
        );
        assert_eq!(
            legal_partition_merge_groups(&plan),
            vec![vec!["a".to_string(), "b".to_string(), "c".to_string()]],
        );
    }

    #[test]
    fn top_level_combinational_boundary_imposes_no_merge() {
        // No in-partition producer (top-level port) → no internal cross-partition
        // edge to resolve, even though the kind is combinational.
        let plan = test_plan(
            &["a", "b"],
            vec![test_boundary(
                RuntimePartitionSignalKind::Wire,
                None,
                &["a", "b"],
            )],
        );
        assert!(legal_partition_merge_groups(&plan).is_empty());
    }

    fn test_partition_cost(id: &str, compute_ops: u64) -> RuntimePartition {
        let mut partition = test_partition(id);
        partition.cost = RuntimePartitionCost {
            compute_ops,
            state_bits: compute_ops, // arbitrary but distinct so sums are checkable
            memory_bits: 0,
            boundary_bits: 0,
        };
        partition
    }

    #[test]
    fn legalize_keeps_independent_register_partitions_separate() {
        let mut plan = test_plan(
            &[],
            vec![test_boundary(
                RuntimePartitionSignalKind::Register,
                Some("a"),
                &["b"],
            )],
        );
        plan.partitions = vec![test_partition_cost("a", 10), test_partition_cost("b", 20)];

        let legal = legalize_partition_plan(&plan);
        assert_eq!(legal.groups.len(), 2);
        assert_eq!(legal.groups[0].members, vec!["a".to_string()]);
        assert_eq!(legal.groups[1].members, vec!["b".to_string()]);
        // The register boundary crosses a group edge → it is in the exchange set.
        assert_eq!(legal.exchange_boundaries.len(), 1);
    }

    #[test]
    fn legalize_merges_combinational_partitions_and_sums_cost() {
        let mut plan = test_plan(
            &[],
            vec![test_boundary(
                RuntimePartitionSignalKind::Wire,
                Some("a"),
                &["b"],
            )],
        );
        plan.partitions = vec![test_partition_cost("a", 10), test_partition_cost("b", 20)];

        let legal = legalize_partition_plan(&plan);
        assert_eq!(legal.groups.len(), 1);
        assert_eq!(
            legal.groups[0].members,
            vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(legal.groups[0].id, "a+b");
        assert_eq!(legal.groups[0].cost.compute_ops, 30);
        assert_eq!(legal.groups[0].cost.state_bits, 30);
        // The only boundary is now internal to the merged group → not exchanged.
        assert!(legal.exchange_boundaries.is_empty());
    }

    #[test]
    fn legalize_drops_internal_but_keeps_cross_group_boundaries() {
        // a~b merge (wire); a->c is a register edge that survives as exchange.
        let mut plan = test_plan(
            &[],
            vec![
                test_boundary(RuntimePartitionSignalKind::Wire, Some("a"), &["b"]),
                test_boundary(RuntimePartitionSignalKind::Register, Some("b"), &["c"]),
            ],
        );
        plan.partitions = vec![
            test_partition_cost("a", 1),
            test_partition_cost("b", 2),
            test_partition_cost("c", 4),
        ];

        let legal = legalize_partition_plan(&plan);
        assert_eq!(legal.groups.len(), 2);
        // Group {a,b} and group {c}.
        assert!(legal.groups.iter().any(|g| g.members == vec!["a", "b"]));
        assert!(legal.groups.iter().any(|g| g.members == vec!["c"]));
        // b->c register boundary crosses the {a,b}|{c} edge → exchanged.
        assert_eq!(legal.exchange_boundaries.len(), 1);
        assert_eq!(
            legal.exchange_boundaries[0].signal.kind,
            RuntimePartitionSignalKind::Register
        );
    }

    #[test]
    fn build_partitioned_simulator_matches_whole_design() {
        use rrtl_core::lit_u;
        let mut design = Design::new();
        {
            let mut m = design.module("Inc");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.assign(y, a + lit_u(1, 8));
        }
        let (clk, din, dout);
        {
            let mut m = design.module("Top");
            clk = m.input("clk", uint(1));
            din = m.input("din", uint(8));
            let w0 = m.wire("w0", uint(8));
            let r = m.reg("r", uint(8));
            dout = m.output("dout", uint(8));
            m.clock(r, clk);
            m.next(r, w0);
            m.instance("u0", "Inc", [("a", din), ("y", w0)]);
            m.instance("u1", "Inc", [("a", r), ("y", dout)]);
        }

        let plan = plan_runtime_partitions(
            &design,
            "Top",
            RuntimePartitionConfig {
                target_partitions: 4,
            },
        )
        .unwrap();
        assert!(
            plan.partitions.len() >= 2,
            "expected the planner to split the hierarchy, got {}",
            plan.partitions.len()
        );

        let lanes = 3;
        let mut partitioned = build_partitioned_simulator(&design, "Top", &plan, lanes).unwrap();

        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Top").unwrap();
        let mut oracle = PackedSimulator::new(program, lanes).unwrap();

        let clk_lanes = vec![1u128; lanes];
        oracle.set_signal(clk, &clk_lanes).unwrap();
        partitioned.set_signal(clk, &clk_lanes).unwrap();

        for cycle in 0..6u128 {
            let din_lanes: Vec<u128> = (0..lanes as u128)
                .map(|lane| (cycle * 29 + lane * 7) & 0xff)
                .collect();
            oracle.set_signal(din, &din_lanes).unwrap();
            partitioned.set_signal(din, &din_lanes).unwrap();
            oracle.tick();
            partitioned.tick();
            assert_eq!(
                partitioned.get_signal(dout).unwrap(),
                oracle.get_signal(dout).unwrap(),
                "dout mismatch at cycle {cycle}",
            );
        }
    }

    fn counter_design() -> (Design, Signal, Signal) {
        let mut design = Design::new();
        let en;
        let out;
        {
            let mut m = design.module("Counter");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            en = m.input("en", uint(1));
            out = m.output("out", uint(8));
            let count = m.reg("count", uint(8));

            m.clock(count, clk);
            m.reset(count, rst, 0);
            m.next(count, mux(en, count + lit_u(1, 8), count));
            m.assign(out, count);
        }
        (design, en, out)
    }

    fn memory_design() -> (Design, Signal, Signal, Signal, Signal, Signal) {
        let mut design = Design::new();
        let (we, addr, data, mem, read);
        {
            let mut m = design.module("MemoryTop");
            let clk = m.input("clk", uint(1));
            we = m.input("we", uint(1));
            addr = m.input("addr", uint(2));
            data = m.input("data", uint(8));
            mem = m.mem("mem", 2, uint(8), 4);
            read = m.output("read", uint(8));
            let read_expr = m.mem_read(mem, addr);
            m.assign(read, read_expr);
            m.mem_write(mem, clk, we, addr, data);
        }
        (design, we, addr, data, mem, read)
    }

    fn snapshot_design() -> (
        Design,
        Signal,
        Signal,
        Signal,
        Signal,
        Signal,
        Signal,
        Signal,
    ) {
        let mut design = Design::new();
        let (en, we, addr, data, mem, count, read);
        {
            let mut m = design.module("SnapshotTop");
            let clk = m.input("clk", uint(1));
            en = m.input("en", uint(1));
            we = m.input("we", uint(1));
            addr = m.input("addr", uint(2));
            data = m.input("data", uint(8));
            count = m.reg("count", uint(8));
            mem = m.mem("mem", 2, uint(8), 4);
            read = m.output("read", uint(8));
            m.clock(count, clk);
            m.next(count, mux(en, count + lit_u(1, 8), count));
            let read_expr = m.mem_read(mem, addr);
            m.assign(read, read_expr);
            m.mem_write(mem, clk, we, addr, data);
        }
        (design, en, we, addr, data, mem, count, read)
    }

    fn wide_input_design() -> (Design, Signal, Signal) {
        let mut design = Design::new();
        let wide;
        let out;
        {
            let mut m = design.module("WideInput");
            wide = m.input("wide", uint(160));
            out = m.output("out", uint(160));
            m.assign(out, wide);
        }
        (design, wide, out)
    }

    fn two_cpu_topology() -> RuntimeTopology {
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("cpu-a", 2).on_node("node-a"));
        topology.push(RuntimeWorker::local_cpu("cpu-b", 3).on_node("node-b"));
        topology
    }

    fn surrogate_provenance(exact: bool) -> Provenance {
        Provenance {
            tag: if exact {
                "exact_fallback".to_string()
            } else {
                "surrogate".to_string()
            },
            exact,
            surrogate_id: "gemm_mock".to_string(),
            model_family: ModelFamily::MockGemm,
            artifact_format: ArtifactFormat::MockGemm,
            artifact_hash: "artifact-hash".to_string(),
            source_hash: "source-hash".to_string(),
            policy: PolicyMode::ApproximateWithTolerance,
        }
    }

    fn single_worker_surrogate_plan() -> GemmRuntimePlan {
        GemmRuntimePlan {
            schema: GEMM_RUNTIME_PLAN_SCHEMA.to_string(),
            ok: true,
            total_lanes: 2,
            workers: vec![GemmRuntimeWorkerSummary {
                worker_id: "worker0".to_string(),
                start_lane: 0,
                lanes: 2,
                assigned_items: 2,
                used_surrogate: 1,
                exact_fallbacks: 1,
            }],
            items: vec![
                GemmRuntimePlanItem {
                    index: 0,
                    lane: 0,
                    worker_id: "worker0".to_string(),
                    decision: GemmPolicyDecision::SurrogateUsed,
                    provenance: surrogate_provenance(false),
                    source_result: GemmRuntimeSourceResult::Surrogate,
                    shadow_ok: None,
                    shadow_error: None,
                    shadow_max_abs_error: None,
                    shadow_latency_error_cycles: None,
                    shadow_first_divergence: None,
                },
                GemmRuntimePlanItem {
                    index: 1,
                    lane: 1,
                    worker_id: "worker0".to_string(),
                    decision: GemmPolicyDecision::ExactFallback,
                    provenance: surrogate_provenance(true),
                    source_result: GemmRuntimeSourceResult::Exact,
                    shadow_ok: None,
                    shadow_error: None,
                    shadow_max_abs_error: None,
                    shadow_latency_error_cycles: None,
                    shadow_first_divergence: None,
                },
            ],
            errors: Vec::new(),
        }
    }

    fn multi_worker_surrogate_plan() -> GemmRuntimePlan {
        GemmRuntimePlan {
            schema: GEMM_RUNTIME_PLAN_SCHEMA.to_string(),
            ok: true,
            total_lanes: 5,
            workers: vec![
                GemmRuntimeWorkerSummary {
                    worker_id: "cpu-a".to_string(),
                    start_lane: 0,
                    lanes: 2,
                    assigned_items: 1,
                    used_surrogate: 1,
                    exact_fallbacks: 0,
                },
                GemmRuntimeWorkerSummary {
                    worker_id: "cpu-b".to_string(),
                    start_lane: 2,
                    lanes: 3,
                    assigned_items: 1,
                    used_surrogate: 0,
                    exact_fallbacks: 1,
                },
            ],
            items: vec![
                GemmRuntimePlanItem {
                    index: 0,
                    lane: 1,
                    worker_id: "cpu-a".to_string(),
                    decision: GemmPolicyDecision::SurrogateUsed,
                    provenance: surrogate_provenance(false),
                    source_result: GemmRuntimeSourceResult::Surrogate,
                    shadow_ok: None,
                    shadow_error: None,
                    shadow_max_abs_error: None,
                    shadow_latency_error_cycles: None,
                    shadow_first_divergence: None,
                },
                GemmRuntimePlanItem {
                    index: 1,
                    lane: 4,
                    worker_id: "cpu-b".to_string(),
                    decision: GemmPolicyDecision::ExactFallback,
                    provenance: surrogate_provenance(true),
                    source_result: GemmRuntimeSourceResult::Exact,
                    shadow_ok: None,
                    shadow_error: None,
                    shadow_max_abs_error: None,
                    shadow_latency_error_cycles: None,
                    shadow_first_divergence: None,
                },
            ],
            errors: Vec::new(),
        }
    }

    fn multi_worker_event_plan() -> EventRuntimePlan {
        EventRuntimePlan {
            schema: EVENT_RUNTIME_PLAN_SCHEMA.to_string(),
            ok: true,
            total_lanes: 5,
            workers: vec![
                EventRuntimeWorkerSummary {
                    worker_id: "cpu-a".to_string(),
                    start_lane: 0,
                    lanes: 2,
                    assigned_items: 1,
                    used_surrogate: 1,
                    exact_fallbacks: 0,
                },
                EventRuntimeWorkerSummary {
                    worker_id: "cpu-b".to_string(),
                    start_lane: 2,
                    lanes: 3,
                    assigned_items: 1,
                    used_surrogate: 0,
                    exact_fallbacks: 1,
                },
            ],
            items: vec![
                EventRuntimePlanItem {
                    index: 0,
                    sample_id: 10,
                    lane: 1,
                    target: "cache_miss".to_string(),
                    worker_id: "cpu-a".to_string(),
                    decision: EventPolicyDecision::SurrogateUsed,
                    provenance: surrogate_provenance(false),
                    source_result: GemmRuntimeSourceResult::Surrogate,
                    predicted: None,
                    expected: None,
                    shadow_ok: None,
                    shadow_error: None,
                },
                EventRuntimePlanItem {
                    index: 1,
                    sample_id: 11,
                    lane: 4,
                    target: "cache_miss".to_string(),
                    worker_id: "cpu-b".to_string(),
                    decision: EventPolicyDecision::ExactFallback,
                    provenance: surrogate_provenance(true),
                    source_result: GemmRuntimeSourceResult::Exact,
                    predicted: None,
                    expected: None,
                    shadow_ok: None,
                    shadow_error: None,
                },
            ],
            errors: Vec::new(),
        }
    }

    fn parallel_options() -> DistributedRuntimeOptions {
        DistributedRuntimeOptions {
            execution_mode: RuntimeExecutionMode::Parallel,
        }
    }

    #[test]
    fn runtime_surrogate_attachment_maps_single_worker_plan() {
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("worker0", 2));

        let attachment = topology
            .attach_gemm_runtime_plan(&single_worker_surrogate_plan())
            .unwrap();

        assert_eq!(
            attachment.format_version,
            RUNTIME_SURROGATE_ATTACHMENT_FORMAT_VERSION
        );
        assert!(attachment.health.ready);
        assert_eq!(attachment.total_lanes, 2);
        assert_eq!(attachment.workers.len(), 1);
        assert_eq!(attachment.workers[0].worker_id, "worker0");
        assert_eq!(attachment.workers[0].used_surrogate, 1);
        assert_eq!(attachment.workers[0].exact_fallbacks, 1);
        assert_eq!(attachment.items.len(), 2);
        assert_eq!(
            attachment.items[0].source_result,
            GemmRuntimeSourceResult::Surrogate
        );
        assert_eq!(
            attachment.items[1].source_result,
            GemmRuntimeSourceResult::Exact
        );
    }

    #[test]
    fn runtime_surrogate_attachment_maps_multi_worker_plan() {
        let attachment = two_cpu_topology()
            .attach_gemm_runtime_plan(&multi_worker_surrogate_plan())
            .unwrap();

        assert_eq!(attachment.total_lanes, 5);
        assert_eq!(attachment.health.worker_count, 2);
        assert_eq!(attachment.workers[0].worker_id, "cpu-a");
        assert_eq!(attachment.workers[0].start_lane, 0);
        assert_eq!(attachment.workers[1].worker_id, "cpu-b");
        assert_eq!(attachment.workers[1].start_lane, 2);
        assert_eq!(attachment.health.used_surrogate, 1);
        assert_eq!(attachment.health.exact_fallbacks, 1);
    }

    #[test]
    fn runtime_surrogate_attachment_maps_event_plan() {
        let attachment = two_cpu_topology()
            .attach_event_runtime_plan(&multi_worker_event_plan())
            .unwrap();

        assert_eq!(attachment.plan_schema, EVENT_RUNTIME_PLAN_SCHEMA);
        assert_eq!(attachment.total_lanes, 5);
        assert_eq!(attachment.health.worker_count, 2);
        assert_eq!(attachment.health.item_count, 2);
        assert_eq!(attachment.health.used_surrogate, 1);
        assert_eq!(attachment.health.exact_fallbacks, 1);
        assert_eq!(
            attachment.items[0].decision,
            GemmPolicyDecision::SurrogateUsed
        );
        assert_eq!(attachment.items[0].sample_id, Some(10));
        assert_eq!(attachment.items[0].target.as_deref(), Some("cache_miss"));
        assert_eq!(
            attachment.items[1].decision,
            GemmPolicyDecision::ExactFallback
        );
    }

    #[test]
    fn runtime_surrogate_attachment_reports_event_shadow_counters() {
        let mut plan = multi_worker_event_plan();
        plan.items[1].decision = EventPolicyDecision::ShadowCompare;
        plan.items[1].predicted = Some(1);
        plan.items[1].expected = Some(0);
        plan.items[1].shadow_ok = Some(false);
        plan.items[1].shadow_error = Some("event shadow prediction diverged".to_string());

        let attachment = two_cpu_topology().attach_event_runtime_plan(&plan).unwrap();
        let report = RuntimeSurrogateExecutionReport::inspect_attachment(&attachment);

        assert_eq!(
            attachment.items[1].decision,
            GemmPolicyDecision::ShadowCompare
        );
        assert_eq!(attachment.items[1].shadow_predicted, Some(1));
        assert_eq!(attachment.items[1].shadow_expected, Some(0));
        assert_eq!(report.shadow_compared_items, 1);
        assert_eq!(report.shadow_passed_items, 0);
        assert_eq!(report.shadow_failed_items, 1);
        assert_eq!(report.shadow_unavailable_items, 0);
        assert_eq!(report.exact_fallback_items, 1);
        assert!(report.ready);
        assert!(report.executed);
        assert_eq!(report.event_workers.len(), 2);
        assert_eq!(report.event_workers[1].shadow_failed, 1);
        assert_eq!(report.event_lanes.len(), 2);
        assert_eq!(report.event_items.len(), 2);
        assert_eq!(report.event_items[1].sample_id, Some(11));
        assert_eq!(report.event_items[1].target.as_deref(), Some("cache_miss"));
        assert_eq!(report.event_items[1].predicted, Some(1));
        assert_eq!(report.event_items[1].expected, Some(0));
        assert_eq!(report.event_items[1].shadow_ok, Some(false));
    }

    #[test]
    fn runtime_surrogate_attachment_reports_gemm_shadow_counters() {
        let mut plan = single_worker_surrogate_plan();
        plan.items[1].decision = GemmPolicyDecision::ShadowCompare;
        plan.items[1].shadow_ok = Some(true);
        plan.items[1].shadow_max_abs_error = Some(0);
        plan.items[1].shadow_latency_error_cycles = Some(0);

        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("worker0", 2));
        let attachment = topology.attach_gemm_runtime_plan(&plan).unwrap();
        let report = RuntimeSurrogateExecutionReport::inspect_attachment(&attachment);

        assert_eq!(
            attachment.items[1].decision,
            GemmPolicyDecision::ShadowCompare
        );
        assert_eq!(attachment.items[1].shadow_max_abs_error, Some(0));
        assert_eq!(report.shadow_compared_items, 1);
        assert_eq!(report.shadow_passed_items, 1);
        assert_eq!(report.shadow_failed_items, 0);
        assert_eq!(report.shadow_unavailable_items, 0);
        assert_eq!(report.exact_fallback_items, 1);
    }

    #[test]
    fn runtime_surrogate_attachment_rejects_unknown_worker() {
        let mut plan = single_worker_surrogate_plan();
        plan.workers[0].worker_id = "missing".to_string();
        plan.items[0].worker_id = "missing".to_string();
        plan.items[1].worker_id = "missing".to_string();

        let err = RuntimeTopology::local_cpu(2)
            .attach_gemm_runtime_plan(&plan)
            .unwrap_err();

        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_SURROGATE_ATTACHMENT_WORKER"
        );
    }

    #[test]
    fn runtime_surrogate_attachment_rejects_lane_outside_worker_range() {
        let mut plan = multi_worker_surrogate_plan();
        plan.items[0].lane = 3;

        let err = two_cpu_topology()
            .attach_gemm_runtime_plan(&plan)
            .unwrap_err();

        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_SURROGATE_ATTACHMENT_ITEM"
        );
    }

    #[test]
    fn runtime_surrogate_attachment_rejects_event_lane_outside_worker_range() {
        let mut plan = multi_worker_event_plan();
        plan.items[0].lane = 3;

        let err = two_cpu_topology()
            .attach_event_runtime_plan(&plan)
            .unwrap_err();

        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_SURROGATE_ATTACHMENT_ITEM"
        );
    }

    #[test]
    fn runtime_surrogate_attachment_rejects_invalid_plan() {
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("worker0", 2));
        let mut plan = single_worker_surrogate_plan();
        plan.ok = false;
        plan.errors.push("fail closed".to_string());

        let err = topology.attach_gemm_runtime_plan(&plan).unwrap_err();

        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_SURROGATE_ATTACHMENT_PLAN"
        );
    }

    #[test]
    fn runtime_surrogate_attachment_serializes_round_trip() {
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("worker0", 2));
        let attachment = topology
            .attach_gemm_runtime_plan(&single_worker_surrogate_plan())
            .unwrap();

        let mut bytes = Vec::new();
        attachment.write_json(&mut bytes).unwrap();
        let decoded = RuntimeSurrogateAttachment::read_json(&mut bytes.as_slice()).unwrap();

        assert_eq!(decoded, attachment);
    }

    #[test]
    fn distributed_runtime_attaches_surrogate_plan() {
        let (design, _, _) = counter_design();
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("worker0", 2));
        let mut runtime = DistributedRuntime::new(&design, "Counter", topology).unwrap();

        runtime
            .attach_surrogate_plan(&single_worker_surrogate_plan())
            .unwrap();

        let attachment = runtime.surrogate_attachment().unwrap();
        assert_eq!(attachment.health.item_count, 2);
        assert_eq!(attachment.health.used_surrogate, 1);
    }

    #[test]
    fn distributed_runtime_records_surrogate_execution_report() {
        let (design, _, _) = counter_design();
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("worker0", 2));
        let mut runtime = DistributedRuntime::new(&design, "Counter", topology).unwrap();

        runtime
            .attach_surrogate_plan(&single_worker_surrogate_plan())
            .unwrap();

        let attached = runtime.surrogate_execution().unwrap();
        assert!(attached.ready);
        assert_eq!(attached.action_count, 0);
        assert_eq!(attached.latest_action_kind, None);
        assert_eq!(attached.surrogate_eligible_items, 1);
        assert_eq!(attached.exact_fallback_items, 1);

        runtime.tick_many(3).unwrap();

        let report = runtime.stats().surrogate_execution.unwrap();
        assert_eq!(report.action_count, 1);
        assert_eq!(
            report.latest_action_kind,
            Some(RuntimeSurrogateActionKind::TickMany)
        );
        assert_eq!(report.attached_items, 2);
        assert_eq!(report.surrogate_eligible_items, 1);
        assert_eq!(report.exact_fallback_items, 1);

        runtime.reset_stats();

        let reset = runtime.surrogate_execution().unwrap();
        assert_eq!(reset.action_count, 0);
        assert_eq!(reset.latest_action_kind, None);
    }

    #[test]
    fn distributed_runtime_records_event_surrogate_execution_report() {
        let (design, _, _) = counter_design();
        let mut runtime = DistributedRuntime::new(&design, "Counter", two_cpu_topology()).unwrap();

        runtime
            .attach_event_surrogate_plan(&multi_worker_event_plan())
            .unwrap();
        runtime.tick().unwrap();

        let report = runtime.stats().surrogate_execution.unwrap();
        assert_eq!(report.plan_schema, EVENT_RUNTIME_PLAN_SCHEMA);
        assert_eq!(report.action_count, 1);
        assert_eq!(
            report.latest_action_kind,
            Some(RuntimeSurrogateActionKind::Tick)
        );
        assert_eq!(report.surrogate_eligible_items, 1);
        assert_eq!(report.exact_fallback_items, 1);
    }

    #[test]
    fn runtime_telemetry_serializes_surrogate_attachment() {
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("worker0", 2));
        let attachment = topology
            .attach_gemm_runtime_plan(&single_worker_surrogate_plan())
            .unwrap();
        let telemetry = TcpRuntimeSupervisorTelemetry {
            format_version: RUNTIME_TELEMETRY_FORMAT_VERSION,
            module_name: "Counter".to_string(),
            topology,
            surrogate_attachment: Some(attachment.clone()),
            surrogate_execution: Some(RuntimeSurrogateExecutionReport::inspect_attachment(
                &attachment,
            )),
            endpoints: Vec::new(),
            processes: Vec::new(),
            runtime_stats: RuntimeStats::default(),
            runtime_health: None,
            runtime_health_error: None,
            latest_checkpoint: None,
            last_recovery: None,
        };

        let mut bytes = Vec::new();
        telemetry.write_json(&mut bytes).unwrap();
        let decoded = TcpRuntimeSupervisorTelemetry::read_json(&mut bytes.as_slice()).unwrap();

        assert_eq!(decoded.surrogate_attachment, Some(attachment));
        assert_eq!(decoded, telemetry);
    }

    #[test]
    fn runtime_partition_session_telemetry_serializes_surrogate_attachment() {
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("worker0", 2));
        let mut session = RuntimePartitionSession::deploy_local(
            recursive_partition_launch(),
            &mut LocalRuntimeWorkerService::new(),
        )
        .unwrap();

        session
            .attach_surrogate_plan(&topology, &single_worker_surrogate_plan())
            .unwrap();
        let telemetry = session.telemetry();
        let attachment = session.surrogate_attachment().cloned();

        let mut bytes = Vec::new();
        telemetry.write_json(&mut bytes).unwrap();
        let decoded = RuntimePartitionSessionTelemetry::read_json(&mut bytes.as_slice()).unwrap();

        assert!(attachment.is_some());
        assert_eq!(decoded.surrogate_attachment, attachment);
        assert_eq!(
            decoded.surrogate_execution,
            Some(RuntimeSurrogateExecutionReport::inspect_attachment(
                decoded.surrogate_attachment.as_ref().unwrap()
            ))
        );
        assert_eq!(decoded, telemetry);
    }

    fn candidate(
        name: &str,
        topology: RuntimeTopology,
        options: DistributedRuntimeOptions,
    ) -> RuntimeAutotuneCandidate {
        RuntimeAutotuneCandidate {
            name: name.to_string(),
            topology,
            options,
        }
    }

    fn two_child_hierarchy_design() -> (Design, Signal, Signal, Signal, Signal) {
        let mut design = Design::new();
        {
            let mut m = design.module("AddOne");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.assign(y, a + lit_u(1, 8));
        }

        let (a0, a1, y0, y1);
        {
            let mut m = design.module("Top");
            a0 = m.input("a0", uint(8));
            a1 = m.input("a1", uint(8));
            y0 = m.output("y0", uint(8));
            y1 = m.output("y1", uint(8));
            m.instance("u0", "AddOne", [("a", a0), ("y", y0)]);
            m.instance("u1", "AddOne", [("a", a1), ("y", y1)]);
        }
        (design, a0, a1, y0, y1)
    }

    fn wide_child_hierarchy_design() -> Design {
        let mut design = Design::new();
        {
            let mut m = design.module("WideChild");
            let a = m.input("a", uint(160));
            let y = m.output("y", uint(160));
            m.assign(y, a);
        }
        {
            let mut m = design.module("WideTop");
            let a = m.input("a", uint(160));
            let y = m.output("y", uint(160));
            m.instance("u_wide", "WideChild", [("a", a), ("y", y)]);
        }
        design
    }

    fn high_boundary_hierarchy_design() -> Design {
        let mut design = Design::new();
        {
            let mut m = design.module("HighBoundaryChild");
            let a = m.input("a", uint(1024));
            let y = m.output("y", uint(1024));
            m.assign(y, a);
        }
        {
            let mut m = design.module("HighBoundaryTop");
            let a = m.input("a", uint(1024));
            let y = m.output("y", uint(1024));
            m.instance("u_wide", "HighBoundaryChild", [("a", a), ("y", y)]);
        }
        design
    }

    fn high_cross_worker_boundary_design() -> Design {
        let mut design = Design::new();
        {
            let mut m = design.module("CrossWorkerWideChild");
            let a = m.input("a", uint(1024));
            let y = m.output("y", uint(1024));
            m.assign(y, a);
        }
        {
            let mut m = design.module("CrossWorkerWideTop");
            let a = m.input("a", uint(1024));
            let y = m.output("y", uint(1024));
            let tap = m.output("tap", uint(1));
            m.assign(tap, lit_u(1, 1));
            m.instance("u_wide", "CrossWorkerWideChild", [("a", a), ("y", y)]);
        }
        design
    }

    fn imbalanced_hierarchy_design() -> Design {
        let mut design = Design::new();
        {
            let mut m = design.module("HeavyChild");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            let mut expr = a + lit_u(1, 8);
            for value in 2..32 {
                expr = expr + lit_u(value, 8);
            }
            m.assign(y, expr);
        }
        {
            let mut m = design.module("LightChild");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.assign(y, a);
        }
        {
            let mut m = design.module("ImbalancedTop");
            let a0 = m.input("a0", uint(8));
            let a1 = m.input("a1", uint(8));
            let a2 = m.input("a2", uint(8));
            let y0 = m.output("y0", uint(8));
            let y1 = m.output("y1", uint(8));
            let y2 = m.output("y2", uint(8));
            m.instance("u_heavy", "HeavyChild", [("a", a0), ("y", y0)]);
            m.instance("u_light0", "LightChild", [("a", a1), ("y", y1)]);
            m.instance("u_light1", "LightChild", [("a", a2), ("y", y2)]);
        }
        design
    }

    fn empty_partition_hierarchy_design() -> Design {
        let mut design = Design::new();
        {
            let mut m = design.module("EmptyChild");
            m.input("a", uint(8));
        }
        {
            let mut m = design.module("EmptyTop");
            let a = m.input("a", uint(8));
            m.instance("u_empty", "EmptyChild", [("a", a)]);
        }
        design
    }

    fn external_partition_hierarchy_design() -> Design {
        let mut design = Design::new();
        {
            let mut ext = design.extern_module("VendorChild");
            ext.input("a", uint(8));
            ext.output("y", uint(8));
        }
        {
            let mut m = design.module("ExternalTop");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.instance("u_vendor", "VendorChild", [("a", a), ("y", y)]);
        }
        design
    }

    fn recursive_partition_hierarchy_design() -> Design {
        let mut design = Design::new();
        {
            let mut m = design.module("RecursiveLeaf");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.assign(y, a + lit_u(1, 8));
        }
        {
            let mut m = design.module("RecursiveMid");
            let a0 = m.input("a0", uint(8));
            let a1 = m.input("a1", uint(8));
            let y0 = m.output("y0", uint(8));
            let y1 = m.output("y1", uint(8));
            let tap = m.output("tap", uint(8));
            m.assign(tap, a0);
            m.instance("u_leaf0", "RecursiveLeaf", [("a", a0), ("y", y0)]);
            m.instance("u_leaf1", "RecursiveLeaf", [("a", a1), ("y", y1)]);
        }
        {
            let mut m = design.module("RecursiveTop");
            let a0 = m.input("a0", uint(8));
            let a1 = m.input("a1", uint(8));
            let y0 = m.output("y0", uint(8));
            let y1 = m.output("y1", uint(8));
            let tap = m.output("tap", uint(8));
            m.instance(
                "u_mid",
                "RecursiveMid",
                [("a0", a0), ("a1", a1), ("y0", y0), ("y1", y1), ("tap", tap)],
            );
        }
        design
    }

    fn recursive_external_hierarchy_design() -> Design {
        let mut design = Design::new();
        {
            let mut ext = design.extern_module("RecursiveVendor");
            ext.input("a", uint(8));
            ext.output("y", uint(8));
        }
        {
            let mut m = design.module("RecursiveExternalMid");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            let tap = m.output("tap", uint(8));
            m.assign(tap, a);
            m.instance("u_vendor", "RecursiveVendor", [("a", a), ("y", y)]);
        }
        {
            let mut m = design.module("RecursiveExternalTop");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            let tap = m.output("tap", uint(8));
            m.instance(
                "u_mid",
                "RecursiveExternalMid",
                [("a", a), ("y", y), ("tap", tap)],
            );
        }
        design
    }

    fn recursive_partition_bundle_parts() -> (
        RuntimePartitionPlan,
        RuntimePartitionPlacementPlan,
        RuntimePartitionCommunicationPlan,
    ) {
        let design = recursive_partition_hierarchy_design();
        let partition_plan = plan_runtime_partitions(
            &design,
            "RecursiveTop",
            RuntimePartitionConfig {
                target_partitions: 3,
            },
        )
        .unwrap();
        let placement_plan =
            plan_runtime_partition_placement(&partition_plan, &two_cpu_topology()).unwrap();
        let communication_plan =
            plan_runtime_partition_communication(&partition_plan, &placement_plan).unwrap();
        (partition_plan, placement_plan, communication_plan)
    }

    fn recursive_partition_launch() -> RuntimePartitionLaunchPlan {
        let (partition_plan, placement_plan, communication_plan) =
            recursive_partition_bundle_parts();
        let bundle =
            RuntimePartitionBundle::from_plans(partition_plan, placement_plan, communication_plan)
                .unwrap();
        plan_runtime_partition_launch(&bundle).unwrap()
    }

    fn runtime_shard_info(
        worker_id: &str,
        backend: RuntimeBackend,
        start_lane: usize,
        lanes: usize,
    ) -> RuntimeShardInfo {
        RuntimeShardInfo {
            worker_id: worker_id.to_string(),
            node: "localhost".to_string(),
            backend,
            start_lane,
            lanes,
        }
    }

    fn init_request(
        design: &Design,
        module_name: &str,
        shard: RuntimeShardInfo,
    ) -> RuntimeWorkerRequest {
        RuntimeWorkerRequest::InitShard(RuntimeShardInit {
            design: compile(design).unwrap(),
            module_name: module_name.to_string(),
            backend: shard.backend,
            shard,
        })
    }

    fn spawn_tcp_worker_server() -> (SocketAddr, std::thread::JoinHandle<Result<(), ErrorReport>>) {
        let mut server =
            TcpRuntimeWorkerServer::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
        let addr = server.local_addr().unwrap();
        let handle = std::thread::spawn(move || server.serve_once());
        (addr, handle)
    }

    fn spawn_tcp_servers_for_topology(
        topology: &RuntimeTopology,
    ) -> (
        HashMap<String, SocketAddr>,
        Vec<std::thread::JoinHandle<Result<(), ErrorReport>>>,
    ) {
        let mut endpoints = HashMap::new();
        let mut handles = Vec::new();
        for worker in topology.workers() {
            let (addr, handle) = spawn_tcp_worker_server();
            endpoints.insert(worker.id.clone(), addr);
            handles.push(handle);
        }
        (endpoints, handles)
    }

    fn spawn_tcp_servers_for_topology_connections(
        topology: &RuntimeTopology,
        connections: usize,
    ) -> (
        HashMap<String, SocketAddr>,
        Vec<std::thread::JoinHandle<Result<(), ErrorReport>>>,
    ) {
        let mut endpoints = HashMap::new();
        let mut handles = Vec::new();
        for worker in topology.workers() {
            let mut server =
                TcpRuntimeWorkerServer::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
            let addr = server.local_addr().unwrap();
            let handle = std::thread::spawn(move || server.serve_connections(connections));
            endpoints.insert(worker.id.clone(), addr);
            handles.push(handle);
        }
        (endpoints, handles)
    }

    fn join_tcp_servers(handles: Vec<std::thread::JoinHandle<Result<(), ErrorReport>>>) {
        for handle in handles {
            handle.join().unwrap().unwrap();
        }
    }

    fn assert_operation_stats_zero(stats: RuntimeOperationStats) {
        assert_eq!(stats.eval_combinational.calls, 0);
        assert_eq!(stats.eval_combinational.total_ns, 0);
        assert_eq!(stats.eval_combinational.last_ns, 0);
        assert_eq!(stats.tick.calls, 0);
        assert_eq!(stats.tick.total_ns, 0);
        assert_eq!(stats.tick.last_ns, 0);
        assert_eq!(stats.tick_many.calls, 0);
        assert_eq!(stats.tick_many.total_ns, 0);
        assert_eq!(stats.tick_many.last_ns, 0);
    }

    #[test]
    fn runtime_partition_planner_reports_single_module_state() {
        let (design, _, _) = counter_design();
        let plan = plan_runtime_partitions(
            &design,
            "Counter",
            RuntimePartitionConfig {
                target_partitions: 4,
            },
        )
        .unwrap();

        assert_eq!(plan.format_version, RUNTIME_PARTITION_PLAN_FORMAT_VERSION);
        assert_eq!(plan.module_name, "Counter");
        assert_eq!(plan.partitions.len(), 1);
        assert!(plan.boundary_signals.is_empty());
        assert!(plan.total_cost.compute_ops > 0);
        assert_eq!(plan.total_cost.state_bits, 8);
        assert!(plan.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == RuntimePartitionDiagnosticCode::TargetUnderfilled
                && diagnostic.severity == RuntimePartitionDiagnosticSeverity::Info
        }));
        assert!(plan.partitions[0]
            .signals
            .iter()
            .any(|signal| signal.signal_name == "count"
                && signal.kind == RuntimePartitionSignalKind::Register));
    }

    #[test]
    fn runtime_partition_planner_reports_child_boundaries() {
        let (design, _, _, _, _) = two_child_hierarchy_design();
        let plan = plan_runtime_partitions(
            &design,
            "Top",
            RuntimePartitionConfig {
                target_partitions: 2,
            },
        )
        .unwrap();

        assert_eq!(plan.partitions.len(), 2);
        assert_eq!(
            plan.partitions
                .iter()
                .map(|partition| partition.instance_path.clone())
                .collect::<Vec<_>>(),
            vec![
                vec!["Top".to_string(), "u0".to_string()],
                vec!["Top".to_string(), "u1".to_string()],
            ]
        );
        assert_eq!(plan.boundary_signals.len(), 4);
        assert_eq!(plan.total_cost.boundary_bits, 32);
        assert!(plan.diagnostics.is_empty());
        assert!(plan.recommendations.is_empty());
        assert!(plan.boundary_signals.iter().any(|boundary| {
            boundary.instance_path == vec!["Top".to_string(), "u0".to_string()]
                && boundary.port_name == "a"
                && boundary.producer_partition.is_none()
                && boundary.consumer_partitions == vec!["p0".to_string()]
        }));
        assert!(plan.boundary_signals.iter().any(|boundary| {
            boundary.instance_path == vec!["Top".to_string(), "u1".to_string()]
                && boundary.port_name == "y"
                && boundary.producer_partition == Some("p1".to_string())
                && boundary.consumer_partitions.is_empty()
        }));
    }

    #[test]
    fn runtime_partition_planner_reports_memory_cost() {
        let (design, _, _, _, _, _) = memory_design();
        let plan = plan_runtime_partitions(
            &design,
            "MemoryTop",
            RuntimePartitionConfig {
                target_partitions: 1,
            },
        )
        .unwrap();

        assert_eq!(plan.partitions.len(), 1);
        assert_eq!(plan.total_cost.memory_bits, 32);
        assert!(plan.partitions[0]
            .signals
            .iter()
            .any(|signal| signal.signal_name == "mem"
                && signal.kind == RuntimePartitionSignalKind::Memory));
    }

    #[test]
    fn runtime_partition_planner_uses_signal_width_for_boundary_cost() {
        let design = wide_child_hierarchy_design();
        let plan = plan_runtime_partitions(
            &design,
            "WideTop",
            RuntimePartitionConfig {
                target_partitions: 2,
            },
        )
        .unwrap();

        let input_boundary = plan
            .boundary_signals
            .iter()
            .find(|boundary| boundary.port_name == "a")
            .unwrap();
        assert_eq!(input_boundary.width, 160);
        assert_eq!(input_boundary.cost.boundary_bits, 160);
    }

    #[test]
    fn runtime_partition_planner_reports_target_underfill() {
        let (design, _, _, _, _) = two_child_hierarchy_design();
        let plan = plan_runtime_partitions(
            &design,
            "Top",
            RuntimePartitionConfig {
                target_partitions: 4,
            },
        )
        .unwrap();

        assert_eq!(plan.partitions.len(), 2);
        assert!(plan.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == RuntimePartitionDiagnosticCode::TargetUnderfilled
                && diagnostic.severity == RuntimePartitionDiagnosticSeverity::Info
        }));
        assert!(plan.recommendations.iter().any(|recommendation| {
            recommendation.code == RuntimePartitionRecommendationCode::IncreaseHierarchy
        }));
    }

    #[test]
    fn runtime_partition_planner_recursively_splits_deep_hierarchy() {
        let design = recursive_partition_hierarchy_design();
        let plan = plan_runtime_partitions(
            &design,
            "RecursiveTop",
            RuntimePartitionConfig {
                target_partitions: 3,
            },
        )
        .unwrap();

        assert_eq!(
            plan.partitions
                .iter()
                .map(|partition| partition.instance_path.clone())
                .collect::<Vec<_>>(),
            vec![
                vec!["RecursiveTop".to_string(), "u_mid".to_string()],
                vec![
                    "RecursiveTop".to_string(),
                    "u_mid".to_string(),
                    "u_leaf0".to_string()
                ],
                vec![
                    "RecursiveTop".to_string(),
                    "u_mid".to_string(),
                    "u_leaf1".to_string()
                ],
            ]
        );
        assert!(plan.diagnostics.is_empty());
        assert!(plan.recommendations.is_empty());
    }

    #[test]
    fn runtime_partition_planner_reports_recursive_boundaries() {
        let design = recursive_partition_hierarchy_design();
        let plan = plan_runtime_partitions(
            &design,
            "RecursiveTop",
            RuntimePartitionConfig {
                target_partitions: 3,
            },
        )
        .unwrap();

        assert!(plan.boundary_signals.iter().any(|boundary| {
            boundary.instance_path == vec!["RecursiveTop".to_string(), "u_mid".to_string()]
                && boundary.port_name == "a0"
                && boundary.consumer_partitions == vec!["p0".to_string()]
        }));
        assert!(plan.boundary_signals.iter().any(|boundary| {
            boundary.instance_path
                == vec![
                    "RecursiveTop".to_string(),
                    "u_mid".to_string(),
                    "u_leaf0".to_string(),
                ]
                && boundary.port_name == "a"
                && boundary.producer_partition == Some("p0".to_string())
                && boundary.consumer_partitions == vec!["p1".to_string()]
        }));
        assert!(plan.boundary_signals.iter().any(|boundary| {
            boundary.instance_path
                == vec![
                    "RecursiveTop".to_string(),
                    "u_mid".to_string(),
                    "u_leaf1".to_string(),
                ]
                && boundary.port_name == "y"
                && boundary.producer_partition == Some("p2".to_string())
                && boundary.consumer_partitions == vec!["p0".to_string()]
        }));
        assert_eq!(plan.total_cost.boundary_bits, 72);
    }

    #[test]
    fn runtime_partition_planner_avoids_recursive_cost_double_counting() {
        let design = recursive_partition_hierarchy_design();
        let plan = plan_runtime_partitions(
            &design,
            "RecursiveTop",
            RuntimePartitionConfig {
                target_partitions: 3,
            },
        )
        .unwrap();

        let mid = plan
            .partitions
            .iter()
            .find(|partition| partition.module_name == "RecursiveMid")
            .unwrap();
        let leaf_costs = plan
            .partitions
            .iter()
            .filter(|partition| partition.module_name == "RecursiveLeaf")
            .map(|partition| partition.cost.compute_ops)
            .collect::<Vec<_>>();
        assert_eq!(mid.cost.compute_ops, 1);
        assert_eq!(leaf_costs, vec![3, 3]);
        assert_eq!(plan.total_cost.compute_ops, 7);
    }

    #[test]
    fn runtime_partition_planner_stops_recursive_split_at_external_module() {
        let design = recursive_external_hierarchy_design();
        let plan = plan_runtime_partitions(
            &design,
            "RecursiveExternalTop",
            RuntimePartitionConfig {
                target_partitions: 3,
            },
        )
        .unwrap();

        assert_eq!(plan.partitions.len(), 2);
        assert!(plan.partitions.iter().any(|partition| {
            partition.external
                && partition.instance_path
                    == vec![
                        "RecursiveExternalTop".to_string(),
                        "u_mid".to_string(),
                        "u_vendor".to_string(),
                    ]
        }));
        assert!(plan.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == RuntimePartitionDiagnosticCode::ExternalModuleOpaque
        }));
        assert!(plan.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == RuntimePartitionDiagnosticCode::TargetUnderfilled
        }));
    }

    #[test]
    fn runtime_partition_planner_reports_compute_imbalance() {
        let design = imbalanced_hierarchy_design();
        let plan = plan_runtime_partitions(
            &design,
            "ImbalancedTop",
            RuntimePartitionConfig {
                target_partitions: 3,
            },
        )
        .unwrap();

        let heavy = plan
            .partitions
            .iter()
            .find(|partition| partition.module_name == "HeavyChild")
            .unwrap();
        assert!(plan.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == RuntimePartitionDiagnosticCode::PartitionImbalance
                && diagnostic.severity == RuntimePartitionDiagnosticSeverity::Warning
                && diagnostic.partition_id.as_deref() == Some(heavy.id.as_str())
        }));
        assert!(plan.recommendations.iter().any(|recommendation| {
            recommendation.code == RuntimePartitionRecommendationCode::SplitHeavyPartition
                && recommendation.partition_id.as_deref() == Some(heavy.id.as_str())
        }));
    }

    #[test]
    fn runtime_partition_planner_reports_high_boundary_cost() {
        let design = high_boundary_hierarchy_design();
        let plan = plan_runtime_partitions(
            &design,
            "HighBoundaryTop",
            RuntimePartitionConfig {
                target_partitions: 2,
            },
        )
        .unwrap();

        assert!(plan
            .boundary_signals
            .iter()
            .any(|boundary| boundary.width == 1024));
        assert!(plan.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == RuntimePartitionDiagnosticCode::HighBoundaryCost
                && diagnostic.severity == RuntimePartitionDiagnosticSeverity::Warning
                && diagnostic.boundary_index.is_some()
        }));
        assert!(plan.recommendations.iter().any(|recommendation| {
            recommendation.code == RuntimePartitionRecommendationCode::CoLocateBoundary
                && recommendation.boundary_index.is_some()
        }));
    }

    #[test]
    fn runtime_partition_planner_reports_empty_partition() {
        let design = empty_partition_hierarchy_design();
        let plan = plan_runtime_partitions(
            &design,
            "EmptyTop",
            RuntimePartitionConfig {
                target_partitions: 1,
            },
        )
        .unwrap();

        assert_eq!(plan.partitions.len(), 1);
        assert!(plan.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == RuntimePartitionDiagnosticCode::EmptyPartition
                && diagnostic.severity == RuntimePartitionDiagnosticSeverity::Warning
                && diagnostic.partition_id.as_deref() == Some("p0")
        }));
        assert!(plan.recommendations.iter().any(|recommendation| {
            recommendation.code == RuntimePartitionRecommendationCode::ReduceTargetPartitions
                && recommendation.partition_id.as_deref() == Some("p0")
        }));
    }

    #[test]
    fn runtime_partition_planner_reports_external_module_opacity() {
        let design = external_partition_hierarchy_design();
        let plan = plan_runtime_partitions(
            &design,
            "ExternalTop",
            RuntimePartitionConfig {
                target_partitions: 2,
            },
        )
        .unwrap();

        assert!(plan.partitions[0].external);
        assert!(plan.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == RuntimePartitionDiagnosticCode::ExternalModuleOpaque
                && diagnostic.severity == RuntimePartitionDiagnosticSeverity::Info
                && diagnostic.partition_id.as_deref() == Some("p0")
        }));
        assert!(plan.recommendations.iter().any(|recommendation| {
            recommendation.code == RuntimePartitionRecommendationCode::InspectExternalModule
                && recommendation.partition_id.as_deref() == Some("p0")
        }));
        assert!(!plan
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == RuntimePartitionDiagnosticCode::EmptyPartition));
    }

    #[test]
    fn runtime_partition_planner_validates_inputs() {
        let (design, _, _) = counter_design();
        let err = plan_runtime_partitions(
            &design,
            "Counter",
            RuntimePartitionConfig {
                target_partitions: 0,
            },
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_PARTITION_CONFIG");

        let err = plan_runtime_partitions(&design, "Missing", RuntimePartitionConfig::default())
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_PARTITION_MODULE");
    }

    #[test]
    fn runtime_partition_plan_json_round_trip_and_version_validation() {
        let (design, _, _) = counter_design();
        let plan =
            plan_runtime_partitions(&design, "Counter", RuntimePartitionConfig::default()).unwrap();

        let mut bytes = Vec::new();
        plan.write_json(&mut bytes).unwrap();
        let decoded = RuntimePartitionPlan::read_json(&mut bytes.as_slice()).unwrap();
        assert_eq!(decoded, plan);

        let mut bad = plan;
        bad.format_version = RUNTIME_PARTITION_PLAN_FORMAT_VERSION + 1;
        let err =
            RuntimePartitionPlan::read_json(&mut serde_json::to_vec(&bad).unwrap().as_slice())
                .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_PARTITION_PLAN_VERSION");
    }

    #[test]
    fn runtime_partition_placement_maps_recursive_plan_to_cpu_workers() {
        let design = recursive_partition_hierarchy_design();
        let partition_plan = plan_runtime_partitions(
            &design,
            "RecursiveTop",
            RuntimePartitionConfig {
                target_partitions: 3,
            },
        )
        .unwrap();
        let placement =
            plan_runtime_partition_placement(&partition_plan, &two_cpu_topology()).unwrap();

        assert_eq!(
            placement
                .assignments
                .iter()
                .map(|assignment| {
                    (
                        assignment.partition_id.as_str(),
                        assignment.worker_id.as_str(),
                    )
                })
                .collect::<Vec<_>>(),
            vec![("p0", "cpu-a"), ("p1", "cpu-a"), ("p2", "cpu-b")]
        );
        assert_eq!(placement.worker_summaries.len(), 2);
        assert_eq!(placement.worker_summaries[0].worker_id, "cpu-a");
        assert_eq!(placement.worker_summaries[0].cost.compute_ops, 4);
        assert_eq!(placement.worker_summaries[1].worker_id, "cpu-b");
        assert_eq!(placement.worker_summaries[1].cost.compute_ops, 3);
        assert!(placement.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == RuntimePartitionPlacementDiagnosticCode::UnderProvisionedWorkers
                && diagnostic.severity == RuntimePartitionDiagnosticSeverity::Info
        }));
    }

    #[test]
    fn runtime_partition_placement_balances_heavy_partitions_first() {
        let design = imbalanced_hierarchy_design();
        let partition_plan = plan_runtime_partitions(
            &design,
            "ImbalancedTop",
            RuntimePartitionConfig {
                target_partitions: 3,
            },
        )
        .unwrap();
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("cpu-a", 2));
        topology.push(RuntimeWorker::local_cpu("cpu-b", 2));
        topology.push(RuntimeWorker::local_cpu("cpu-c", 2));
        let placement = plan_runtime_partition_placement(&partition_plan, &topology).unwrap();

        let heavy = placement
            .assignments
            .iter()
            .find(|assignment| assignment.instance_path.ends_with(&["u_heavy".to_string()]))
            .unwrap();
        assert_eq!(heavy.worker_id, "cpu-a");
        assert!(placement.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == RuntimePartitionPlacementDiagnosticCode::WorkerImbalance
                && diagnostic.severity == RuntimePartitionDiagnosticSeverity::Warning
                && diagnostic.worker_id.as_deref() == Some("cpu-a")
        }));
        assert!(placement.recommendations.iter().any(|recommendation| {
            recommendation.code == RuntimePartitionPlacementRecommendationCode::MovePartition
                && recommendation.worker_id.as_deref() == Some("cpu-a")
        }));
    }

    #[test]
    fn runtime_partition_placement_counts_cross_worker_boundary_traffic() {
        let design = recursive_partition_hierarchy_design();
        let partition_plan = plan_runtime_partitions(
            &design,
            "RecursiveTop",
            RuntimePartitionConfig {
                target_partitions: 3,
            },
        )
        .unwrap();
        let placement =
            plan_runtime_partition_placement(&partition_plan, &two_cpu_topology()).unwrap();

        let cpu_a = placement
            .worker_summaries
            .iter()
            .find(|summary| summary.worker_id == "cpu-a")
            .unwrap();
        let cpu_b = placement
            .worker_summaries
            .iter()
            .find(|summary| summary.worker_id == "cpu-b")
            .unwrap();
        assert_eq!(cpu_a.inbound_boundary_bits, 8);
        assert_eq!(cpu_a.outbound_boundary_bits, 8);
        assert_eq!(cpu_b.inbound_boundary_bits, 8);
        assert_eq!(cpu_b.outbound_boundary_bits, 8);
    }

    #[test]
    fn runtime_partition_placement_reports_high_cross_worker_boundary_traffic() {
        let design = high_cross_worker_boundary_design();
        let partition_plan = plan_runtime_partitions(
            &design,
            "CrossWorkerWideTop",
            RuntimePartitionConfig {
                target_partitions: 2,
            },
        )
        .unwrap();
        let placement =
            plan_runtime_partition_placement(&partition_plan, &two_cpu_topology()).unwrap();

        assert!(placement.worker_summaries.iter().any(|summary| {
            summary.inbound_boundary_bits == 1024 && summary.outbound_boundary_bits == 1024
        }));
        assert!(placement.diagnostics.iter().any(|diagnostic| {
            diagnostic.code
                == RuntimePartitionPlacementDiagnosticCode::HighCrossWorkerBoundaryTraffic
                && diagnostic.severity == RuntimePartitionDiagnosticSeverity::Warning
                && diagnostic.boundary_index.is_some()
        }));
        assert!(placement.recommendations.iter().any(|recommendation| {
            recommendation.code == RuntimePartitionPlacementRecommendationCode::CoLocatePartitions
                && recommendation.boundary_index.is_some()
        }));
    }

    #[test]
    fn runtime_partition_placement_prefers_cpu_for_gpu_hostile_partitions() {
        let (design, _, _, _, _, _) = memory_design();
        let partition_plan =
            plan_runtime_partitions(&design, "MemoryTop", RuntimePartitionConfig::default())
                .unwrap();
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_gpu(
            "gpu0",
            8,
            GpuBatchOptions::default(),
        ));
        topology.push(RuntimeWorker::local_cpu("cpu0", 2));

        let placement = plan_runtime_partition_placement(&partition_plan, &topology).unwrap();

        assert_eq!(placement.assignments[0].worker_id, "cpu0");
        assert!(placement.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == RuntimePartitionPlacementDiagnosticCode::CpuFallback
                && diagnostic.severity == RuntimePartitionDiagnosticSeverity::Info
                && diagnostic.partition_id.as_deref() == Some("p0")
        }));
    }

    #[test]
    fn runtime_partition_placement_validates_topology() {
        let (design, _, _) = counter_design();
        let partition_plan =
            plan_runtime_partitions(&design, "Counter", RuntimePartitionConfig::default()).unwrap();

        let err =
            plan_runtime_partition_placement(&partition_plan, &RuntimeTopology::new()).unwrap_err();
        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_PLACEMENT_TOPOLOGY"
        );

        let mut zero_lane = RuntimeTopology::new();
        zero_lane.push(RuntimeWorker::local_cpu("bad", 0));
        let err = plan_runtime_partition_placement(&partition_plan, &zero_lane).unwrap_err();
        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_PLACEMENT_TOPOLOGY"
        );

        let mut duplicate = RuntimeTopology::new();
        duplicate.push(RuntimeWorker::local_cpu("dup", 1));
        duplicate.push(RuntimeWorker::local_cpu("dup", 1));
        let err = plan_runtime_partition_placement(&partition_plan, &duplicate).unwrap_err();
        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_PLACEMENT_TOPOLOGY"
        );
    }

    #[test]
    fn runtime_partition_placement_json_round_trip_and_version_validation() {
        let (design, _, _) = counter_design();
        let partition_plan =
            plan_runtime_partitions(&design, "Counter", RuntimePartitionConfig::default()).unwrap();
        let placement =
            plan_runtime_partition_placement(&partition_plan, &RuntimeTopology::local_cpu(1))
                .unwrap();

        let mut bytes = Vec::new();
        placement.write_json(&mut bytes).unwrap();
        let decoded = RuntimePartitionPlacementPlan::read_json(&mut bytes.as_slice()).unwrap();
        assert_eq!(decoded, placement);

        let mut bad = placement;
        bad.format_version = RUNTIME_PARTITION_PLACEMENT_FORMAT_VERSION + 1;
        let err = RuntimePartitionPlacementPlan::read_json(
            &mut serde_json::to_vec(&bad).unwrap().as_slice(),
        )
        .unwrap_err();
        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_PLACEMENT_VERSION"
        );
    }

    #[test]
    fn runtime_partition_communication_routes_cross_worker_boundaries() {
        let design = recursive_partition_hierarchy_design();
        let partition_plan = plan_runtime_partitions(
            &design,
            "RecursiveTop",
            RuntimePartitionConfig {
                target_partitions: 3,
            },
        )
        .unwrap();
        let placement =
            plan_runtime_partition_placement(&partition_plan, &two_cpu_topology()).unwrap();
        let communication =
            plan_runtime_partition_communication(&partition_plan, &placement).unwrap();

        assert_eq!(communication.routes.len(), 2);
        assert!(communication.routes.iter().all(|route| {
            route.producer_worker != route.consumer_worker && route.bits_per_transfer == 8
        }));
        assert_eq!(communication.total_cross_worker_boundary_bits, 16);
        let cpu_a = communication
            .worker_summaries
            .iter()
            .find(|summary| summary.worker_id == "cpu-a")
            .unwrap();
        let cpu_b = communication
            .worker_summaries
            .iter()
            .find(|summary| summary.worker_id == "cpu-b")
            .unwrap();
        assert_eq!(cpu_a.outbound_bits, 8);
        assert_eq!(cpu_a.inbound_bits, 8);
        assert_eq!(cpu_b.outbound_bits, 8);
        assert_eq!(cpu_b.inbound_bits, 8);
    }

    #[test]
    fn runtime_partition_communication_omits_same_worker_boundaries() {
        let (design, _, _, _, _) = two_child_hierarchy_design();
        let partition_plan = plan_runtime_partitions(
            &design,
            "Top",
            RuntimePartitionConfig {
                target_partitions: 2,
            },
        )
        .unwrap();
        let placement =
            plan_runtime_partition_placement(&partition_plan, &RuntimeTopology::local_cpu(2))
                .unwrap();
        let communication =
            plan_runtime_partition_communication(&partition_plan, &placement).unwrap();

        assert!(communication.routes.is_empty());
        assert_eq!(communication.total_cross_worker_boundary_bits, 0);
        assert!(communication.diagnostics.is_empty());
        assert!(communication.recommendations.is_empty());
    }

    #[test]
    fn runtime_partition_communication_validates_inputs() {
        let design = recursive_partition_hierarchy_design();
        let partition_plan = plan_runtime_partitions(
            &design,
            "RecursiveTop",
            RuntimePartitionConfig {
                target_partitions: 3,
            },
        )
        .unwrap();
        let mut placement =
            plan_runtime_partition_placement(&partition_plan, &two_cpu_topology()).unwrap();

        let mut module_mismatch = placement.clone();
        module_mismatch.module_name = "Other".to_string();
        let err =
            plan_runtime_partition_communication(&partition_plan, &module_mismatch).unwrap_err();
        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_COMMUNICATION_MODULE"
        );

        placement
            .assignments
            .retain(|assignment| assignment.partition_id != "p2");
        let err = plan_runtime_partition_communication(&partition_plan, &placement).unwrap_err();
        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_COMMUNICATION_ASSIGNMENT"
        );
    }

    #[test]
    fn runtime_partition_communication_reports_high_route_width() {
        let design = high_cross_worker_boundary_design();
        let partition_plan = plan_runtime_partitions(
            &design,
            "CrossWorkerWideTop",
            RuntimePartitionConfig {
                target_partitions: 2,
            },
        )
        .unwrap();
        let placement =
            plan_runtime_partition_placement(&partition_plan, &two_cpu_topology()).unwrap();
        let communication =
            plan_runtime_partition_communication(&partition_plan, &placement).unwrap();

        assert_eq!(communication.routes.len(), 2);
        assert_eq!(communication.total_cross_worker_boundary_bits, 2048);
        assert!(communication.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == RuntimePartitionCommunicationDiagnosticCode::HighRouteWidth
                && diagnostic.severity == RuntimePartitionDiagnosticSeverity::Warning
                && diagnostic.route_index.is_some()
                && diagnostic.boundary_index.is_some()
        }));
        assert!(communication.recommendations.iter().any(|recommendation| {
            recommendation.code == RuntimePartitionCommunicationRecommendationCode::CoLocateRoute
                && recommendation.route_index.is_some()
        }));
    }

    #[test]
    fn runtime_partition_communication_json_round_trip_and_version_validation() {
        let design = recursive_partition_hierarchy_design();
        let partition_plan = plan_runtime_partitions(
            &design,
            "RecursiveTop",
            RuntimePartitionConfig {
                target_partitions: 3,
            },
        )
        .unwrap();
        let placement =
            plan_runtime_partition_placement(&partition_plan, &two_cpu_topology()).unwrap();
        let communication =
            plan_runtime_partition_communication(&partition_plan, &placement).unwrap();

        let mut bytes = Vec::new();
        communication.write_json(&mut bytes).unwrap();
        let decoded = RuntimePartitionCommunicationPlan::read_json(&mut bytes.as_slice()).unwrap();
        assert_eq!(decoded, communication);

        let mut bad = communication;
        bad.format_version = RUNTIME_PARTITION_COMMUNICATION_FORMAT_VERSION + 1;
        let err = RuntimePartitionCommunicationPlan::read_json(
            &mut serde_json::to_vec(&bad).unwrap().as_slice(),
        )
        .unwrap_err();
        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_COMMUNICATION_VERSION"
        );
    }

    #[test]
    fn runtime_partition_bundle_plans_partition_placement_and_communication() {
        let design = recursive_partition_hierarchy_design();
        let bundle = plan_runtime_partition_bundle(
            &design,
            "RecursiveTop",
            RuntimePartitionConfig {
                target_partitions: 3,
            },
            &two_cpu_topology(),
        )
        .unwrap();

        let (partition_plan, placement_plan, communication_plan) =
            recursive_partition_bundle_parts();
        assert_eq!(
            bundle.format_version,
            RUNTIME_PARTITION_BUNDLE_FORMAT_VERSION
        );
        assert_eq!(bundle.module_name, "RecursiveTop");
        assert_eq!(bundle.partition_plan, partition_plan);
        assert_eq!(bundle.placement_plan, placement_plan);
        assert_eq!(bundle.communication_plan, communication_plan);
        assert_eq!(bundle.partition_plan.partitions.len(), 3);
        assert_eq!(bundle.placement_plan.assignments.len(), 3);
        assert_eq!(bundle.communication_plan.routes.len(), 2);
        assert!(bundle.diagnostics.iter().any(|diagnostic| {
            diagnostic.source == RuntimePartitionBundleSource::Placement
                && diagnostic.code == "UnderProvisionedWorkers"
                && diagnostic.severity == RuntimePartitionDiagnosticSeverity::Info
        }));
    }

    #[test]
    fn runtime_partition_bundle_from_plans_validates_module_names() {
        let (partition_plan, mut placement_plan, communication_plan) =
            recursive_partition_bundle_parts();
        placement_plan.module_name = "Other".to_string();

        let err =
            RuntimePartitionBundle::from_plans(partition_plan, placement_plan, communication_plan)
                .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_PARTITION_BUNDLE_MODULE");
    }

    #[test]
    fn runtime_partition_bundle_from_plans_validates_route_assignments() {
        let (partition_plan, mut placement_plan, communication_plan) =
            recursive_partition_bundle_parts();
        placement_plan
            .assignments
            .retain(|assignment| assignment.partition_id != "p2");

        let err =
            RuntimePartitionBundle::from_plans(partition_plan, placement_plan, communication_plan)
                .unwrap_err();
        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_BUNDLE_ASSIGNMENT"
        );
    }

    #[test]
    fn runtime_partition_bundle_json_round_trip_and_version_validation() {
        let (partition_plan, placement_plan, communication_plan) =
            recursive_partition_bundle_parts();
        let bundle =
            RuntimePartitionBundle::from_plans(partition_plan, placement_plan, communication_plan)
                .unwrap();

        let mut bytes = Vec::new();
        bundle.write_json(&mut bytes).unwrap();
        let decoded = RuntimePartitionBundle::read_json(&mut bytes.as_slice()).unwrap();
        assert_eq!(decoded, bundle);

        let mut bad = bundle;
        bad.format_version = RUNTIME_PARTITION_BUNDLE_FORMAT_VERSION + 1;
        let err =
            RuntimePartitionBundle::read_json(&mut serde_json::to_vec(&bad).unwrap().as_slice())
                .unwrap_err();
        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_BUNDLE_VERSION"
        );
    }

    #[test]
    fn runtime_partition_bundle_rejects_embedded_bad_versions() {
        let (partition_plan, placement_plan, communication_plan) =
            recursive_partition_bundle_parts();

        let mut bad_partition = partition_plan.clone();
        bad_partition.format_version = RUNTIME_PARTITION_PLAN_FORMAT_VERSION + 1;
        let err = RuntimePartitionBundle::from_plans(
            bad_partition,
            placement_plan.clone(),
            communication_plan.clone(),
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_PARTITION_PLAN_VERSION");

        let mut bad_placement = placement_plan.clone();
        bad_placement.format_version = RUNTIME_PARTITION_PLACEMENT_FORMAT_VERSION + 1;
        let err = RuntimePartitionBundle::from_plans(
            partition_plan.clone(),
            bad_placement,
            communication_plan.clone(),
        )
        .unwrap_err();
        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_PLACEMENT_VERSION"
        );

        let mut bad_communication = communication_plan;
        bad_communication.format_version = RUNTIME_PARTITION_COMMUNICATION_FORMAT_VERSION + 1;
        let err =
            RuntimePartitionBundle::from_plans(partition_plan, placement_plan, bad_communication)
                .unwrap_err();
        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_COMMUNICATION_VERSION"
        );
    }

    #[test]
    fn runtime_partition_launch_builds_worker_payloads_and_routes() {
        let (partition_plan, placement_plan, communication_plan) =
            recursive_partition_bundle_parts();
        let bundle =
            RuntimePartitionBundle::from_plans(partition_plan, placement_plan, communication_plan)
                .unwrap();
        let launch = plan_runtime_partition_launch(&bundle).unwrap();

        assert_eq!(
            launch.format_version,
            RUNTIME_PARTITION_LAUNCH_FORMAT_VERSION
        );
        assert_eq!(launch.module_name, "RecursiveTop");
        assert_eq!(
            launch
                .workers
                .iter()
                .map(|worker| worker.worker_id.as_str())
                .collect::<Vec<_>>(),
            vec!["cpu-a", "cpu-b"]
        );
        assert_eq!(
            launch
                .workers
                .iter()
                .flat_map(|worker| worker
                    .partitions
                    .iter()
                    .map(|partition| partition.partition_id.as_str()))
                .collect::<Vec<_>>(),
            vec!["p0", "p1", "p2"]
        );
        assert_eq!(launch.routes.len(), 2);
        for route in &launch.routes {
            let producer = launch
                .workers
                .iter()
                .find(|worker| worker.worker_id == route.producer_worker)
                .unwrap();
            let consumer = launch
                .workers
                .iter()
                .find(|worker| worker.worker_id == route.consumer_worker)
                .unwrap();
            assert!(producer.outbound_routes.contains(&route.route_index));
            assert!(consumer.inbound_routes.contains(&route.route_index));
        }
        assert!(launch.diagnostics.iter().any(|diagnostic| {
            diagnostic.source == RuntimePartitionLaunchDiagnosticSource::BundlePlacement
                && diagnostic.code == "UnderProvisionedWorkers"
        }));
    }

    #[test]
    fn runtime_partition_launch_reports_empty_worker_launches() {
        let design = recursive_partition_hierarchy_design();
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("cpu-a", 1));
        topology.push(RuntimeWorker::local_cpu("cpu-b", 1));
        topology.push(RuntimeWorker::local_cpu("cpu-c", 1));
        topology.push(RuntimeWorker::local_cpu("cpu-d", 1));
        let bundle = plan_runtime_partition_bundle(
            &design,
            "RecursiveTop",
            RuntimePartitionConfig {
                target_partitions: 3,
            },
            &topology,
        )
        .unwrap();

        let launch = plan_runtime_partition_launch(&bundle).unwrap();

        assert!(launch.workers.iter().any(|worker| {
            worker.worker_id == "cpu-d"
                && worker.partitions.is_empty()
                && worker.inbound_routes.is_empty()
                && worker.outbound_routes.is_empty()
        }));
        assert!(launch.diagnostics.iter().any(|diagnostic| {
            diagnostic.source == RuntimePartitionLaunchDiagnosticSource::Launch
                && diagnostic.code == "EmptyWorkerLaunch"
                && diagnostic.worker_id.as_deref() == Some("cpu-d")
        }));
    }

    #[test]
    fn runtime_partition_launch_validates_assignment_partition_references() {
        let (partition_plan, placement_plan, communication_plan) =
            recursive_partition_bundle_parts();
        let mut bundle =
            RuntimePartitionBundle::from_plans(partition_plan, placement_plan, communication_plan)
                .unwrap();
        bundle
            .partition_plan
            .partitions
            .retain(|partition| partition.id != "p2");

        let err = plan_runtime_partition_launch(&bundle).unwrap_err();
        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_LAUNCH_PARTITION"
        );
    }

    #[test]
    fn runtime_partition_launch_validates_route_assignments() {
        let (partition_plan, placement_plan, communication_plan) =
            recursive_partition_bundle_parts();
        let mut bundle =
            RuntimePartitionBundle::from_plans(partition_plan, placement_plan, communication_plan)
                .unwrap();
        let producer_partition = bundle.communication_plan.routes[0]
            .producer_partition
            .clone();
        bundle
            .placement_plan
            .assignments
            .retain(|assignment| assignment.partition_id != producer_partition);

        let err = plan_runtime_partition_launch(&bundle).unwrap_err();
        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_LAUNCH_ASSIGNMENT"
        );
    }

    #[test]
    fn runtime_partition_launch_validates_stale_route_worker_mapping() {
        let (partition_plan, placement_plan, communication_plan) =
            recursive_partition_bundle_parts();
        let mut bundle =
            RuntimePartitionBundle::from_plans(partition_plan, placement_plan, communication_plan)
                .unwrap();
        let route = &mut bundle.communication_plan.routes[0];
        route.consumer_worker = if route.consumer_worker == "cpu-a" {
            "cpu-b".to_string()
        } else {
            "cpu-a".to_string()
        };

        let err = plan_runtime_partition_launch(&bundle).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_PARTITION_LAUNCH_ROUTE");
    }

    #[test]
    fn runtime_partition_launch_json_round_trip_and_version_validation() {
        let (partition_plan, placement_plan, communication_plan) =
            recursive_partition_bundle_parts();
        let bundle =
            RuntimePartitionBundle::from_plans(partition_plan, placement_plan, communication_plan)
                .unwrap();
        let launch = plan_runtime_partition_launch(&bundle).unwrap();

        let mut bytes = Vec::new();
        launch.write_json(&mut bytes).unwrap();
        let decoded = RuntimePartitionLaunchPlan::read_json(&mut bytes.as_slice()).unwrap();
        assert_eq!(decoded, launch);

        let mut bad = launch;
        bad.format_version = RUNTIME_PARTITION_LAUNCH_FORMAT_VERSION + 1;
        let err = RuntimePartitionLaunchPlan::read_json(
            &mut serde_json::to_vec(&bad).unwrap().as_slice(),
        )
        .unwrap_err();
        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_LAUNCH_VERSION"
        );
    }

    #[test]
    fn local_worker_service_initializes_partition_worker_and_reports_health() {
        let launch = recursive_partition_launch();
        let worker_launch = launch
            .workers
            .iter()
            .find(|worker| worker.worker_id == "cpu-a")
            .unwrap()
            .clone();
        let mut service = LocalRuntimeWorkerService::new();

        let RuntimeWorkerResponse::PartitionWorkerHealth(health) = service
            .handle(RuntimeWorkerRequest::PartitionWorkerHealth {
                worker_id: "cpu-a".to_string(),
            })
            .unwrap()
        else {
            panic!("expected partition worker health response");
        };
        assert_eq!(health.worker_id, "cpu-a");
        assert!(!health.initialized);
        assert!(health.partitions.is_empty());
        assert_eq!(health.backend, None);

        expect_ack(
            service
                .handle(RuntimeWorkerRequest::InitPartitionWorker(
                    RuntimePartitionWorkerInit {
                        worker_id: "cpu-a".to_string(),
                        launch: worker_launch.clone(),
                    },
                ))
                .unwrap(),
        )
        .unwrap();
        let RuntimeWorkerResponse::PartitionWorkerHealth(health) = service
            .handle(RuntimeWorkerRequest::PartitionWorkerHealth {
                worker_id: "cpu-a".to_string(),
            })
            .unwrap()
        else {
            panic!("expected partition worker health response");
        };
        assert!(health.initialized);
        assert_eq!(health.backend, Some(worker_launch.backend));
        assert_eq!(health.node.as_deref(), Some(worker_launch.node.as_str()));
        assert_eq!(health.partitions, worker_launch.partitions);
        assert_eq!(health.outbound_routes, worker_launch.outbound_routes);
        assert_eq!(health.inbound_routes, worker_launch.inbound_routes);
        assert_eq!(health.outbound_bits, worker_launch.outbound_bits);
        assert_eq!(health.inbound_bits, worker_launch.inbound_bits);
    }

    #[test]
    fn local_worker_service_initializes_empty_partition_worker_launch() {
        let design = recursive_partition_hierarchy_design();
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("cpu-a", 1));
        topology.push(RuntimeWorker::local_cpu("cpu-b", 1));
        topology.push(RuntimeWorker::local_cpu("cpu-c", 1));
        topology.push(RuntimeWorker::local_cpu("cpu-d", 1));
        let bundle = plan_runtime_partition_bundle(
            &design,
            "RecursiveTop",
            RuntimePartitionConfig {
                target_partitions: 3,
            },
            &topology,
        )
        .unwrap();
        let launch = plan_runtime_partition_launch(&bundle).unwrap();
        let worker_launch = launch
            .workers
            .iter()
            .find(|worker| worker.worker_id == "cpu-d")
            .unwrap()
            .clone();
        let mut service = LocalRuntimeWorkerService::new();

        expect_ack(
            service
                .handle(RuntimeWorkerRequest::InitPartitionWorker(
                    RuntimePartitionWorkerInit {
                        worker_id: "cpu-d".to_string(),
                        launch: worker_launch.clone(),
                    },
                ))
                .unwrap(),
        )
        .unwrap();
        let RuntimeWorkerResponse::PartitionWorkerHealth(health) = service
            .handle(RuntimeWorkerRequest::PartitionWorkerHealth {
                worker_id: "cpu-d".to_string(),
            })
            .unwrap()
        else {
            panic!("expected partition worker health response");
        };
        assert!(health.initialized);
        assert!(health.partitions.is_empty());
        assert!(health.inbound_routes.is_empty());
        assert!(health.outbound_routes.is_empty());
    }

    #[test]
    fn local_worker_service_rejects_bad_partition_worker_init() {
        let launch = recursive_partition_launch();
        let worker_launch = launch
            .workers
            .iter()
            .find(|worker| worker.worker_id == "cpu-a")
            .unwrap()
            .clone();
        let mut service = LocalRuntimeWorkerService::new();

        let err = service
            .handle(RuntimeWorkerRequest::InitPartitionWorker(
                RuntimePartitionWorkerInit {
                    worker_id: "wrong".to_string(),
                    launch: worker_launch.clone(),
                },
            ))
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_PARTITION_WORKER_INIT");

        expect_ack(
            service
                .handle(RuntimeWorkerRequest::InitPartitionWorker(
                    RuntimePartitionWorkerInit {
                        worker_id: "cpu-a".to_string(),
                        launch: worker_launch.clone(),
                    },
                ))
                .unwrap(),
        )
        .unwrap();
        let err = service
            .handle(RuntimeWorkerRequest::InitPartitionWorker(
                RuntimePartitionWorkerInit {
                    worker_id: "cpu-a".to_string(),
                    launch: worker_launch,
                },
            ))
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_PARTITION_WORKER_INIT");
    }

    #[test]
    fn local_worker_service_runs_partition_worker_actions() {
        let launch = recursive_partition_launch();
        let worker_launch = launch
            .workers
            .iter()
            .find(|worker| worker.worker_id == "cpu-a")
            .unwrap()
            .clone();
        let mut service = LocalRuntimeWorkerService::new();

        let err = service
            .handle(RuntimeWorkerRequest::RunPartitionWorkerAction(
                RuntimePartitionWorkerAction {
                    worker_id: "cpu-a".to_string(),
                    kind: RuntimePartitionWorkerActionKind::Tick,
                },
            ))
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_PARTITION_WORKER_ACTION");

        expect_ack(
            service
                .handle(RuntimeWorkerRequest::InitPartitionWorker(
                    RuntimePartitionWorkerInit {
                        worker_id: "cpu-a".to_string(),
                        launch: worker_launch.clone(),
                    },
                ))
                .unwrap(),
        )
        .unwrap();

        let RuntimeWorkerResponse::PartitionWorkerAction(report) = service
            .handle(RuntimeWorkerRequest::RunPartitionWorkerAction(
                RuntimePartitionWorkerAction {
                    worker_id: "cpu-a".to_string(),
                    kind: RuntimePartitionWorkerActionKind::EvalCombinational,
                },
            ))
            .unwrap()
        else {
            panic!("expected partition worker action report");
        };
        assert_eq!(report.worker_id, "cpu-a");
        assert_eq!(
            report.kind,
            RuntimePartitionWorkerActionKind::EvalCombinational
        );
        assert_eq!(report.partition_count, worker_launch.partitions.len());
        assert_eq!(
            report.outbound_route_count,
            worker_launch.outbound_routes.len()
        );
        assert_eq!(
            report.inbound_route_count,
            worker_launch.inbound_routes.len()
        );
        assert_eq!(report.outbound_bits, worker_launch.outbound_bits);
        assert_eq!(report.inbound_bits, worker_launch.inbound_bits);
        assert_eq!(report.operations.eval_combinational.calls, 1);
        assert_eq!(report.operations.tick.calls, 0);
        assert_eq!(report.operations.tick_many.calls, 0);

        let RuntimeWorkerResponse::PartitionWorkerAction(report) = service
            .handle(RuntimeWorkerRequest::RunPartitionWorkerAction(
                RuntimePartitionWorkerAction {
                    worker_id: "cpu-a".to_string(),
                    kind: RuntimePartitionWorkerActionKind::Tick,
                },
            ))
            .unwrap()
        else {
            panic!("expected partition worker action report");
        };
        assert_eq!(report.operations.eval_combinational.calls, 1);
        assert_eq!(report.operations.tick.calls, 1);
        assert_eq!(report.operations.tick_many.calls, 0);

        let RuntimeWorkerResponse::PartitionWorkerAction(report) = service
            .handle(RuntimeWorkerRequest::RunPartitionWorkerAction(
                RuntimePartitionWorkerAction {
                    worker_id: "cpu-a".to_string(),
                    kind: RuntimePartitionWorkerActionKind::TickMany(0),
                },
            ))
            .unwrap()
        else {
            panic!("expected partition worker action report");
        };
        assert_eq!(report.operations.tick_many.calls, 0);

        let RuntimeWorkerResponse::PartitionWorkerAction(report) = service
            .handle(RuntimeWorkerRequest::RunPartitionWorkerAction(
                RuntimePartitionWorkerAction {
                    worker_id: "cpu-a".to_string(),
                    kind: RuntimePartitionWorkerActionKind::TickMany(4),
                },
            ))
            .unwrap()
        else {
            panic!("expected partition worker action report");
        };
        assert_eq!(report.operations.eval_combinational.calls, 1);
        assert_eq!(report.operations.tick.calls, 1);
        assert_eq!(report.operations.tick_many.calls, 1);
    }

    #[test]
    fn local_worker_service_manages_partition_route_mailboxes() {
        let launch = recursive_partition_launch();
        let route = launch.routes[0].clone();
        let producer_launch = launch
            .workers
            .iter()
            .find(|worker| worker.worker_id == route.producer_worker)
            .unwrap()
            .clone();
        let consumer_launch = launch
            .workers
            .iter()
            .find(|worker| worker.worker_id == route.consumer_worker)
            .unwrap()
            .clone();
        let payload = RuntimePartitionRoutePayload {
            route_index: route.route_index,
            width: route.width,
            limbs: vec![0x5a],
        };
        let mut service = LocalRuntimeWorkerService::new();
        for worker_launch in [&producer_launch, &consumer_launch] {
            expect_ack(
                service
                    .handle(RuntimeWorkerRequest::InitPartitionWorker(
                        RuntimePartitionWorkerInit {
                            worker_id: worker_launch.worker_id.clone(),
                            launch: worker_launch.clone(),
                        },
                    ))
                    .unwrap(),
            )
            .unwrap();
        }

        let RuntimeWorkerResponse::PartitionRouteMailbox(report) = service
            .handle(RuntimeWorkerRequest::StorePartitionRouteOutbound {
                worker_id: route.producer_worker.clone(),
                payload: payload.clone(),
            })
            .unwrap()
        else {
            panic!("expected partition route mailbox report");
        };
        assert_eq!(report.worker_id, route.producer_worker);
        assert_eq!(report.outbound_payload_count, 1);
        assert_eq!(report.inbound_payload_count, 0);

        let RuntimeWorkerResponse::PartitionRouteMailbox(report) = service
            .handle(RuntimeWorkerRequest::DeliverPartitionRouteInbound {
                worker_id: route.consumer_worker.clone(),
                payload: payload.clone(),
            })
            .unwrap()
        else {
            panic!("expected partition route mailbox report");
        };
        assert_eq!(report.worker_id, route.consumer_worker);
        assert_eq!(report.outbound_payload_count, 0);
        assert_eq!(report.inbound_payload_count, 1);

        let err = service
            .handle(RuntimeWorkerRequest::StorePartitionRouteOutbound {
                worker_id: route.consumer_worker.clone(),
                payload: payload.clone(),
            })
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_PARTITION_ROUTE_MAILBOX");

        let err = service
            .handle(RuntimeWorkerRequest::StorePartitionRouteOutbound {
                worker_id: route.producer_worker.clone(),
                payload: RuntimePartitionRoutePayload {
                    route_index: route.route_index,
                    width: route.width + 1,
                    limbs: vec![0x5a],
                },
            })
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_PARTITION_ROUTE_MAILBOX");

        let err = service
            .handle(RuntimeWorkerRequest::StorePartitionRouteOutbound {
                worker_id: route.producer_worker.clone(),
                payload: RuntimePartitionRoutePayload {
                    route_index: route.route_index,
                    width: route.width,
                    limbs: vec![0x5a, 0x00],
                },
            })
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_PARTITION_ROUTE_MAILBOX");

        let RuntimeWorkerResponse::PartitionRouteMailbox(report) = service
            .handle(RuntimeWorkerRequest::ClearPartitionRouteMailboxes {
                worker_id: route.producer_worker,
            })
            .unwrap()
        else {
            panic!("expected partition route mailbox report");
        };
        assert_eq!(report.outbound_payload_count, 0);
        assert_eq!(report.inbound_payload_count, 0);
    }

    #[test]
    fn local_worker_service_runs_empty_partition_worker_actions() {
        let design = recursive_partition_hierarchy_design();
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("cpu-a", 1));
        topology.push(RuntimeWorker::local_cpu("cpu-b", 1));
        topology.push(RuntimeWorker::local_cpu("cpu-c", 1));
        topology.push(RuntimeWorker::local_cpu("cpu-d", 1));
        let bundle = plan_runtime_partition_bundle(
            &design,
            "RecursiveTop",
            RuntimePartitionConfig {
                target_partitions: 3,
            },
            &topology,
        )
        .unwrap();
        let launch = plan_runtime_partition_launch(&bundle).unwrap();
        let worker_launch = launch
            .workers
            .iter()
            .find(|worker| worker.worker_id == "cpu-d")
            .unwrap()
            .clone();
        let mut service = LocalRuntimeWorkerService::new();
        expect_ack(
            service
                .handle(RuntimeWorkerRequest::InitPartitionWorker(
                    RuntimePartitionWorkerInit {
                        worker_id: "cpu-d".to_string(),
                        launch: worker_launch,
                    },
                ))
                .unwrap(),
        )
        .unwrap();

        let RuntimeWorkerResponse::PartitionWorkerAction(report) = service
            .handle(RuntimeWorkerRequest::RunPartitionWorkerAction(
                RuntimePartitionWorkerAction {
                    worker_id: "cpu-d".to_string(),
                    kind: RuntimePartitionWorkerActionKind::Tick,
                },
            ))
            .unwrap()
        else {
            panic!("expected partition worker action report");
        };
        assert_eq!(report.partition_count, 0);
        assert_eq!(report.outbound_route_count, 0);
        assert_eq!(report.inbound_route_count, 0);
        assert_eq!(report.operations.tick.calls, 1);
    }

    #[test]
    fn runtime_partition_deployment_local_initializes_all_workers() {
        let launch = recursive_partition_launch();
        let mut service = LocalRuntimeWorkerService::new();

        let report = deploy_runtime_partition_launch_local(&launch, &mut service).unwrap();

        assert_eq!(report.module_name, launch.module_name);
        assert_eq!(report.workers.len(), launch.workers.len());
        assert_eq!(report.diagnostics, launch.diagnostics);
        for worker in &launch.workers {
            let health = report
                .workers
                .iter()
                .find(|health| health.worker_id == worker.worker_id)
                .unwrap();
            assert!(health.initialized);
            assert_eq!(health.partitions, worker.partitions);
            assert_eq!(health.outbound_routes, worker.outbound_routes);
            assert_eq!(health.inbound_routes, worker.inbound_routes);
        }
    }

    #[test]
    fn runtime_partition_deployment_loopback_initializes_all_workers() {
        let launch = recursive_partition_launch();
        let service = Arc::new(Mutex::new(LocalRuntimeWorkerService::new()));

        let report = deploy_runtime_partition_launch_loopback(&launch, service).unwrap();

        assert_eq!(report.module_name, "RecursiveTop");
        assert_eq!(
            report
                .workers
                .iter()
                .map(|health| (health.worker_id.as_str(), health.initialized))
                .collect::<Vec<_>>(),
            vec![("cpu-a", true), ("cpu-b", true)]
        );
    }

    #[test]
    fn runtime_partition_deployment_tcp_initializes_all_workers() {
        let launch = recursive_partition_launch();
        let topology = two_cpu_topology();
        let (endpoints, handles) = spawn_tcp_servers_for_topology(&topology);

        let report = deploy_runtime_partition_launch_tcp(
            &launch,
            &endpoints,
            TcpRuntimeTransportConfig::default(),
        )
        .unwrap();

        assert_eq!(report.workers.len(), launch.workers.len());
        assert!(report.workers.iter().all(|health| health.initialized));
        join_tcp_servers(handles);
    }

    #[test]
    fn runtime_partition_deployment_validates_launch_and_endpoints() {
        let launch = recursive_partition_launch();

        let mut duplicate = launch.clone();
        duplicate.workers.push(duplicate.workers[0].clone());
        let err = deploy_runtime_partition_launch_local(
            &duplicate,
            &mut LocalRuntimeWorkerService::new(),
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_PARTITION_DEPLOY_WORKER");

        let mut bad_version = launch.clone();
        bad_version.format_version = RUNTIME_PARTITION_LAUNCH_FORMAT_VERSION + 1;
        let err = deploy_runtime_partition_launch_local(
            &bad_version,
            &mut LocalRuntimeWorkerService::new(),
        )
        .unwrap_err();
        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_LAUNCH_VERSION"
        );

        let err = deploy_runtime_partition_launch_tcp(
            &launch,
            &HashMap::new(),
            TcpRuntimeTransportConfig::default(),
        )
        .unwrap_err();
        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_DEPLOY_ENDPOINT"
        );
    }

    #[test]
    fn runtime_partition_deployment_reports_duplicate_worker_init() {
        let launch = recursive_partition_launch();
        let mut service = LocalRuntimeWorkerService::new();
        deploy_runtime_partition_launch_local(&launch, &mut service).unwrap();

        let err = deploy_runtime_partition_launch_local(&launch, &mut service).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_PARTITION_WORKER_INIT");
    }

    struct UnexpectedPartitionWorkerClient {
        worker_id: String,
    }

    impl RuntimeShardClient for UnexpectedPartitionWorkerClient {
        fn worker_id(&self) -> &str {
            &self.worker_id
        }

        fn request(
            &mut self,
            request: RuntimeWorkerRequest,
        ) -> Result<RuntimeWorkerResponse, ErrorReport> {
            match request {
                RuntimeWorkerRequest::InitPartitionWorker(_) => Ok(RuntimeWorkerResponse::Ack),
                RuntimeWorkerRequest::PartitionWorkerHealth { .. } => {
                    Ok(RuntimeWorkerResponse::Ack)
                }
                _ => unreachable!("unexpected request in partition deployment test"),
            }
        }
    }

    #[test]
    fn runtime_partition_deployment_reports_unexpected_worker_response() {
        let worker = recursive_partition_launch().workers[0].clone();
        let mut client = UnexpectedPartitionWorkerClient {
            worker_id: worker.worker_id.clone(),
        };

        let err = deploy_runtime_partition_worker(&worker, &mut client).unwrap_err();

        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_WORKER_RESPONSE");
    }

    #[test]
    fn runtime_partition_session_local_reports_ready_health() {
        let launch = recursive_partition_launch();
        let mut service = LocalRuntimeWorkerService::new();

        let session = RuntimePartitionSession::deploy_local(launch.clone(), &mut service).unwrap();
        let health = session.health();

        assert_eq!(session.launch(), &launch);
        assert_eq!(session.deployment().workers.len(), launch.workers.len());
        assert!(session.actions().is_empty());
        assert_eq!(session.latest_action(), None);
        assert_eq!(health.module_name, "RecursiveTop");
        assert!(health.ready);
        assert_eq!(health.worker_count, 2);
        assert_eq!(health.initialized_worker_count, 2);
        assert_eq!(health.partition_count, 3);
        assert_eq!(health.route_count, launch.routes.len());
        assert_eq!(health.diagnostics, launch.diagnostics);
        assert!(health.workers.iter().all(|worker| worker.initialized));
    }

    #[test]
    fn runtime_partition_session_loopback_reports_launch_counts() {
        let launch = recursive_partition_launch();
        let service = Arc::new(Mutex::new(LocalRuntimeWorkerService::new()));

        let session = RuntimePartitionSession::deploy_loopback(launch.clone(), service).unwrap();
        let health = session.health();

        assert!(health.ready);
        assert_eq!(health.worker_count, launch.workers.len());
        assert_eq!(health.partition_count, 3);
        assert_eq!(health.route_count, launch.routes.len());
    }

    #[test]
    fn runtime_partition_session_tcp_reports_ready_health() {
        let launch = recursive_partition_launch();
        let topology = two_cpu_topology();
        let (endpoints, handles) = spawn_tcp_servers_for_topology(&topology);

        let session = RuntimePartitionSession::deploy_tcp(
            launch.clone(),
            &endpoints,
            TcpRuntimeTransportConfig::default(),
        )
        .unwrap();
        let health = session.health();

        assert!(health.ready);
        assert_eq!(health.worker_count, launch.workers.len());
        assert_eq!(health.initialized_worker_count, launch.workers.len());
        join_tcp_servers(handles);
    }

    #[test]
    fn runtime_partition_session_telemetry_json_round_trip_and_version_validation() {
        let launch = recursive_partition_launch();
        let mut service = LocalRuntimeWorkerService::new();
        let mut session = RuntimePartitionSession::deploy_local(launch, &mut service).unwrap();
        let action = session.tick_local(&mut service).unwrap();
        let telemetry = session.telemetry();

        assert_eq!(
            telemetry.format_version,
            RUNTIME_PARTITION_SESSION_TELEMETRY_FORMAT_VERSION
        );
        assert_eq!(telemetry.module_name, "RecursiveTop");
        assert!(telemetry.health.ready);
        assert_eq!(telemetry.actions, vec![action]);
        assert_eq!(telemetry.action_summary, session.action_summary());
        assert_eq!(telemetry.action_summary.action_count, 1);
        assert_eq!(
            telemetry.action_summary.latest_action_kind,
            Some(RuntimePartitionWorkerActionKind::Tick)
        );

        let mut bytes = Vec::new();
        telemetry.write_json(&mut bytes).unwrap();
        let decoded = RuntimePartitionSessionTelemetry::read_json(&mut bytes.as_slice()).unwrap();
        assert_eq!(decoded, telemetry);

        let mut bad = telemetry;
        bad.format_version = RUNTIME_PARTITION_SESSION_TELEMETRY_FORMAT_VERSION + 1;
        let err = RuntimePartitionSessionTelemetry::read_json(
            &mut serde_json::to_vec(&bad).unwrap().as_slice(),
        )
        .unwrap_err();
        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_SESSION_TELEMETRY_VERSION"
        );
    }

    #[test]
    fn runtime_partition_session_propagates_deployment_failures() {
        let launch = recursive_partition_launch();
        let mut service = LocalRuntimeWorkerService::new();
        RuntimePartitionSession::deploy_local(launch.clone(), &mut service).unwrap();

        let err = RuntimePartitionSession::deploy_local(launch, &mut service).unwrap_err();

        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_PARTITION_WORKER_INIT");
    }

    #[test]
    fn runtime_partition_session_local_runs_structural_actions() {
        let launch = recursive_partition_launch();
        let mut service = LocalRuntimeWorkerService::new();
        let mut session =
            RuntimePartitionSession::deploy_local(launch.clone(), &mut service).unwrap();
        assert!(session.actions().is_empty());
        let empty_summary = session.action_summary();
        assert_eq!(empty_summary.module_name, "RecursiveTop");
        assert_eq!(empty_summary.action_count, 0);
        assert_eq!(empty_summary.latest_action_kind, None);
        assert_eq!(empty_summary.operations, RuntimeOperationStats::default());
        assert_eq!(empty_summary.worker_count, launch.workers.len());
        assert_eq!(empty_summary.partition_count, 3);
        assert_eq!(empty_summary.route_count, launch.routes.len());

        let eval = session.eval_combinational_local(&mut service).unwrap();
        assert_eq!(
            eval.kind,
            RuntimePartitionWorkerActionKind::EvalCombinational
        );
        assert_eq!(eval.module_name, "RecursiveTop");
        assert_eq!(eval.workers.len(), launch.workers.len());
        assert_eq!(eval.operations.eval_combinational.calls, 2);
        assert_eq!(eval.operations.tick.calls, 0);
        assert_eq!(eval.operations.tick_many.calls, 0);
        assert_eq!(session.actions(), std::slice::from_ref(&eval));
        assert_eq!(session.latest_action(), Some(&eval));

        let tick = session.tick_local(&mut service).unwrap();
        assert_eq!(tick.kind, RuntimePartitionWorkerActionKind::Tick);
        assert_eq!(tick.operations.eval_combinational.calls, 2);
        assert_eq!(tick.operations.tick.calls, 2);
        assert_eq!(tick.operations.tick_many.calls, 0);
        assert_eq!(session.actions().len(), 2);
        assert_eq!(session.latest_action(), Some(&tick));

        let tick_many = session.tick_many_local(4, &mut service).unwrap();
        assert_eq!(
            tick_many.kind,
            RuntimePartitionWorkerActionKind::TickMany(4)
        );
        assert_eq!(tick_many.operations.eval_combinational.calls, 2);
        assert_eq!(tick_many.operations.tick.calls, 2);
        assert_eq!(tick_many.operations.tick_many.calls, 2);
        assert!(tick_many.workers.iter().all(|worker| {
            worker.partition_count > 0
                && worker.outbound_route_count + worker.inbound_route_count > 0
        }));
        assert_eq!(session.actions().len(), 3);
        assert_eq!(session.latest_action(), Some(&tick_many));

        let summary = session.action_summary();
        assert_eq!(summary.module_name, "RecursiveTop");
        assert_eq!(summary.action_count, 3);
        assert_eq!(
            summary.latest_action_kind,
            Some(RuntimePartitionWorkerActionKind::TickMany(4))
        );
        assert_eq!(summary.operations.eval_combinational.calls, 6);
        assert_eq!(summary.operations.tick.calls, 4);
        assert_eq!(summary.operations.tick_many.calls, 2);
        assert_eq!(summary.worker_count, launch.workers.len());
        assert_eq!(summary.partition_count, 3);
        assert_eq!(summary.route_count, launch.routes.len());
    }

    #[test]
    fn runtime_partition_session_loopback_runs_structural_actions() {
        let launch = recursive_partition_launch();
        let service = Arc::new(Mutex::new(LocalRuntimeWorkerService::new()));
        let mut session =
            RuntimePartitionSession::deploy_loopback(launch.clone(), service.clone()).unwrap();

        let report = session.tick_loopback(service).unwrap();

        assert_eq!(report.kind, RuntimePartitionWorkerActionKind::Tick);
        assert_eq!(report.workers.len(), launch.workers.len());
        assert_eq!(report.operations.tick.calls, launch.workers.len() as u64);
        assert_eq!(session.actions(), std::slice::from_ref(&report));
    }

    #[test]
    fn runtime_partition_session_tcp_runs_structural_actions() {
        let launch = recursive_partition_launch();
        let topology = two_cpu_topology();
        let (endpoints, handles) = spawn_tcp_servers_for_topology_connections(&topology, 2);
        let mut session = RuntimePartitionSession::deploy_tcp(
            launch.clone(),
            &endpoints,
            TcpRuntimeTransportConfig::default(),
        )
        .unwrap();

        let report = session
            .tick_many_tcp(3, &endpoints, TcpRuntimeTransportConfig::default())
            .unwrap();

        assert_eq!(report.kind, RuntimePartitionWorkerActionKind::TickMany(3));
        assert_eq!(report.workers.len(), launch.workers.len());
        assert_eq!(
            report.operations.tick_many.calls,
            launch.workers.len() as u64
        );
        assert_eq!(session.latest_action(), Some(&report));
        join_tcp_servers(handles);
    }

    #[test]
    fn runtime_partition_session_action_validates_readiness() {
        let launch = recursive_partition_launch();
        let deployment = RuntimePartitionDeploymentReport {
            module_name: launch.module_name.clone(),
            workers: Vec::new(),
            diagnostics: Vec::new(),
        };
        let mut session = RuntimePartitionSession {
            launch,
            deployment,
            route_mailboxes: Vec::new(),
            actions: Vec::new(),
            surrogate_attachment: None,
        };
        let mut service = LocalRuntimeWorkerService::new();

        let err = session.tick_local(&mut service).unwrap_err();

        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_SESSION_ACTION"
        );
        assert!(session.actions().is_empty());
    }

    #[test]
    fn runtime_partition_session_tick_many_zero_does_not_increment_counters() {
        let launch = recursive_partition_launch();
        let mut service = LocalRuntimeWorkerService::new();
        let mut session =
            RuntimePartitionSession::deploy_local(launch.clone(), &mut service).unwrap();

        let report = session.tick_many_local(0, &mut service).unwrap();

        assert_eq!(report.kind, RuntimePartitionWorkerActionKind::TickMany(0));
        assert_eq!(report.workers.len(), launch.workers.len());
        assert_eq!(report.operations.tick_many.calls, 0);
        assert!(report
            .workers
            .iter()
            .all(|worker| worker.operations.tick_many.calls == 0));
        assert_eq!(session.actions(), std::slice::from_ref(&report));
    }

    #[test]
    fn runtime_partition_session_clear_actions_preserves_health() {
        let launch = recursive_partition_launch();
        let mut service = LocalRuntimeWorkerService::new();
        let mut session = RuntimePartitionSession::deploy_local(launch, &mut service).unwrap();
        session.tick_local(&mut service).unwrap();
        session.tick_many_local(2, &mut service).unwrap();
        assert_eq!(session.actions().len(), 2);
        assert!(session.health().ready);

        session.clear_actions();

        assert!(session.actions().is_empty());
        assert_eq!(session.latest_action(), None);
        assert!(session.health().ready);
        let summary = session.action_summary();
        assert_eq!(summary.action_count, 0);
        assert_eq!(summary.latest_action_kind, None);
        assert_eq!(summary.operations, RuntimeOperationStats::default());
        let telemetry = session.telemetry();
        assert!(telemetry.actions.is_empty());
        assert_eq!(telemetry.action_summary, summary);
    }

    #[test]
    fn runtime_partition_session_local_manages_route_mailboxes() {
        let launch = recursive_partition_launch();
        let route = launch.routes[0].clone();
        let payload = RuntimePartitionRoutePayload {
            route_index: route.route_index,
            width: route.width,
            limbs: vec![0x33],
        };
        let mut service = LocalRuntimeWorkerService::new();
        let mut session =
            RuntimePartitionSession::deploy_local(launch.clone(), &mut service).unwrap();
        assert_eq!(session.route_mailboxes().len(), launch.workers.len());
        assert!(session
            .route_mailboxes()
            .iter()
            .all(
                |mailbox| mailbox.outbound_payload_count == 0 && mailbox.inbound_payload_count == 0
            ));

        let producer = session
            .publish_route_outbound_local(&route.producer_worker, payload.clone(), &mut service)
            .unwrap();
        assert_eq!(producer.outbound_payload_count, 1);
        let consumer = session
            .deliver_route_inbound_local(&route.consumer_worker, payload, &mut service)
            .unwrap();
        assert_eq!(consumer.inbound_payload_count, 1);

        let telemetry = session.telemetry();
        assert!(telemetry.route_mailboxes.iter().any(|mailbox| {
            mailbox.worker_id == route.producer_worker && mailbox.outbound_payload_count == 1
        }));
        assert!(telemetry.route_mailboxes.iter().any(|mailbox| {
            mailbox.worker_id == route.consumer_worker && mailbox.inbound_payload_count == 1
        }));

        let reports = session.collect_route_mailboxes_local(&mut service).unwrap();
        assert_eq!(reports.len(), launch.workers.len());
        let reports = session.clear_route_mailboxes_local(&mut service).unwrap();
        assert_eq!(reports.len(), launch.workers.len());
        assert!(session
            .route_mailboxes()
            .iter()
            .all(
                |mailbox| mailbox.outbound_payload_count == 0 && mailbox.inbound_payload_count == 0
            ));
    }

    #[test]
    fn runtime_partition_session_local_transfers_route_payload() {
        let launch = recursive_partition_launch();
        let route = launch.routes[0].clone();
        let mut service = LocalRuntimeWorkerService::new();
        let mut session =
            RuntimePartitionSession::deploy_local(launch.clone(), &mut service).unwrap();

        let report = session
            .transfer_route_local(route.route_index, vec![0x66], &mut service)
            .unwrap();

        assert_eq!(report.route_index, route.route_index);
        assert_eq!(report.producer_worker, route.producer_worker);
        assert_eq!(report.consumer_worker, route.consumer_worker);
        assert_eq!(report.width, route.width);
        assert_eq!(report.limb_count, 1);
        assert_eq!(report.producer_mailbox.outbound_payload_count, 1);
        assert_eq!(report.consumer_mailbox.inbound_payload_count, 1);
        let telemetry = session.telemetry();
        assert!(telemetry.route_mailboxes.iter().any(|mailbox| {
            mailbox.worker_id == route.producer_worker && mailbox.outbound_payload_count == 1
        }));
        assert!(telemetry.route_mailboxes.iter().any(|mailbox| {
            mailbox.worker_id == route.consumer_worker && mailbox.inbound_payload_count == 1
        }));
    }

    #[test]
    fn runtime_partition_session_transfer_validates_route_lookup() {
        let launch = recursive_partition_launch();
        let mut service = LocalRuntimeWorkerService::new();
        let mut session = RuntimePartitionSession::deploy_local(launch, &mut service).unwrap();

        let err = session
            .transfer_route_local(usize::MAX, vec![0x66], &mut service)
            .unwrap_err();

        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_ROUTE_TRANSFER"
        );
    }

    #[test]
    fn runtime_partition_session_transfer_rejects_no_route_launch() {
        let design = recursive_partition_hierarchy_design();
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("cpu-a", 1));
        let bundle = plan_runtime_partition_bundle(
            &design,
            "RecursiveTop",
            RuntimePartitionConfig {
                target_partitions: 3,
            },
            &topology,
        )
        .unwrap();
        let launch = plan_runtime_partition_launch(&bundle).unwrap();
        assert!(launch.routes.is_empty());
        let mut service = LocalRuntimeWorkerService::new();
        let mut session = RuntimePartitionSession::deploy_local(launch, &mut service).unwrap();

        let err = session
            .transfer_route_local(0, vec![0x66], &mut service)
            .unwrap_err();

        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_ROUTE_TRANSFER"
        );
    }

    #[test]
    fn runtime_partition_session_loopback_manages_route_mailboxes() {
        let launch = recursive_partition_launch();
        let route = launch.routes[0].clone();
        let payload = RuntimePartitionRoutePayload {
            route_index: route.route_index,
            width: route.width,
            limbs: vec![0x44],
        };
        let service = Arc::new(Mutex::new(LocalRuntimeWorkerService::new()));
        let mut session =
            RuntimePartitionSession::deploy_loopback(launch.clone(), service.clone()).unwrap();

        session
            .publish_route_outbound_loopback(
                &route.producer_worker,
                payload.clone(),
                service.clone(),
            )
            .unwrap();
        session
            .deliver_route_inbound_loopback(&route.consumer_worker, payload, service.clone())
            .unwrap();
        let reports = session
            .collect_route_mailboxes_loopback(service.clone())
            .unwrap();

        assert_eq!(reports.len(), launch.workers.len());
        assert!(reports.iter().any(|mailbox| {
            mailbox.worker_id == route.producer_worker && mailbox.outbound_payload_count == 1
        }));
        assert!(reports.iter().any(|mailbox| {
            mailbox.worker_id == route.consumer_worker && mailbox.inbound_payload_count == 1
        }));
        session.clear_route_mailboxes_loopback(service).unwrap();
        assert!(session
            .telemetry()
            .route_mailboxes
            .iter()
            .all(
                |mailbox| mailbox.outbound_payload_count == 0 && mailbox.inbound_payload_count == 0
            ));
    }

    #[test]
    fn runtime_partition_session_loopback_transfers_route_payload() {
        let launch = recursive_partition_launch();
        let route = launch.routes[0].clone();
        let service = Arc::new(Mutex::new(LocalRuntimeWorkerService::new()));
        let mut session =
            RuntimePartitionSession::deploy_loopback(launch.clone(), service.clone()).unwrap();

        let report = session
            .transfer_route_loopback(route.route_index, vec![0x77], service)
            .unwrap();

        assert_eq!(report.route_index, route.route_index);
        assert_eq!(report.producer_mailbox.outbound_payload_count, 1);
        assert_eq!(report.consumer_mailbox.inbound_payload_count, 1);
    }

    #[test]
    fn runtime_partition_session_tcp_manages_route_mailboxes() {
        let launch = recursive_partition_launch();
        let route = launch.routes[0].clone();
        let payload = RuntimePartitionRoutePayload {
            route_index: route.route_index,
            width: route.width,
            limbs: vec![0x55],
        };
        let topology = two_cpu_topology();
        let (endpoints, handles) = spawn_tcp_servers_for_topology_connections(&topology, 3);
        let mut session = RuntimePartitionSession::deploy_tcp(
            launch.clone(),
            &endpoints,
            TcpRuntimeTransportConfig::default(),
        )
        .unwrap();

        session
            .publish_route_outbound_tcp(
                &route.producer_worker,
                payload.clone(),
                endpoints[&route.producer_worker],
                TcpRuntimeTransportConfig::default(),
            )
            .unwrap();
        session
            .deliver_route_inbound_tcp(
                &route.consumer_worker,
                payload,
                endpoints[&route.consumer_worker],
                TcpRuntimeTransportConfig::default(),
            )
            .unwrap();
        let reports = session
            .collect_route_mailboxes_tcp(&endpoints, TcpRuntimeTransportConfig::default())
            .unwrap();

        assert_eq!(reports.len(), launch.workers.len());
        assert!(reports.iter().any(|mailbox| {
            mailbox.worker_id == route.producer_worker && mailbox.outbound_payload_count == 1
        }));
        assert!(reports.iter().any(|mailbox| {
            mailbox.worker_id == route.consumer_worker && mailbox.inbound_payload_count == 1
        }));
        join_tcp_servers(handles);
    }

    #[test]
    fn runtime_partition_session_tcp_transfers_route_payload() {
        let launch = recursive_partition_launch();
        let route = launch.routes[0].clone();
        let topology = two_cpu_topology();
        let (endpoints, handles) = spawn_tcp_servers_for_topology_connections(&topology, 2);
        let mut session = RuntimePartitionSession::deploy_tcp(
            launch.clone(),
            &endpoints,
            TcpRuntimeTransportConfig::default(),
        )
        .unwrap();

        let report = session
            .transfer_route_tcp(
                route.route_index,
                vec![0x88],
                &endpoints,
                TcpRuntimeTransportConfig::default(),
            )
            .unwrap();

        assert_eq!(report.route_index, route.route_index);
        assert_eq!(report.producer_mailbox.outbound_payload_count, 1);
        assert_eq!(report.consumer_mailbox.inbound_payload_count, 1);
        join_tcp_servers(handles);
    }

    #[test]
    fn runtime_partition_session_empty_script_emits_initial_telemetry_once() {
        let launch = recursive_partition_launch();
        let mut service = LocalRuntimeWorkerService::new();
        let mut session = RuntimePartitionSession::deploy_local(launch, &mut service).unwrap();
        let script = RuntimePartitionSessionRunScript {
            actions: Vec::new(),
            every_actions: 1,
            emit_initial: true,
            emit_final: true,
        };
        let mut events = Vec::new();

        let report = session
            .run_local_script(&script, &mut service, |event, telemetry| {
                events.push(event);
                assert_eq!(telemetry.action_summary.action_count, 0);
                assert!(telemetry.actions.is_empty());
                Ok(())
            })
            .unwrap();

        assert_eq!(report.completed_actions, 0);
        assert_eq!(report.telemetry_emitted, 1);
        assert_eq!(report.action_summary.action_count, 0);
        assert_eq!(
            events,
            vec![RuntimePartitionSessionRunEvent {
                completed_actions: 0,
                reason: RuntimePartitionSessionRunEventReason::Initial,
                latest_action_kind: None,
            }]
        );
        assert!(session.actions().is_empty());
    }

    #[test]
    fn runtime_partition_session_local_runs_script_and_emits_cadence_telemetry() {
        let launch = recursive_partition_launch();
        let mut service = LocalRuntimeWorkerService::new();
        let mut session =
            RuntimePartitionSession::deploy_local(launch.clone(), &mut service).unwrap();
        let script = RuntimePartitionSessionRunScript {
            actions: vec![
                RuntimePartitionWorkerActionKind::EvalCombinational,
                RuntimePartitionWorkerActionKind::Tick,
                RuntimePartitionWorkerActionKind::TickMany(2),
            ],
            every_actions: 2,
            emit_initial: true,
            emit_final: true,
        };
        let mut events = Vec::new();

        let report = session
            .run_local_script(&script, &mut service, |event, telemetry| {
                events.push(event);
                assert_eq!(
                    telemetry.action_summary.action_count,
                    event.completed_actions
                );
                Ok(())
            })
            .unwrap();

        assert_eq!(report.completed_actions, 3);
        assert_eq!(report.telemetry_emitted, 3);
        assert_eq!(report.action_summary.action_count, 3);
        assert_eq!(
            report.action_summary.latest_action_kind,
            Some(RuntimePartitionWorkerActionKind::TickMany(2))
        );
        assert_eq!(
            report.action_summary.operations.eval_combinational.calls,
            launch.workers.len() as u64 * 3
        );
        assert_eq!(
            report.action_summary.operations.tick.calls,
            launch.workers.len() as u64 * 2
        );
        assert_eq!(
            report.action_summary.operations.tick_many.calls,
            launch.workers.len() as u64
        );
        assert_eq!(
            events,
            vec![
                RuntimePartitionSessionRunEvent {
                    completed_actions: 0,
                    reason: RuntimePartitionSessionRunEventReason::Initial,
                    latest_action_kind: None,
                },
                RuntimePartitionSessionRunEvent {
                    completed_actions: 2,
                    reason: RuntimePartitionSessionRunEventReason::Cadence,
                    latest_action_kind: Some(RuntimePartitionWorkerActionKind::Tick),
                },
                RuntimePartitionSessionRunEvent {
                    completed_actions: 3,
                    reason: RuntimePartitionSessionRunEventReason::Final,
                    latest_action_kind: Some(RuntimePartitionWorkerActionKind::TickMany(2)),
                },
            ]
        );
        assert_eq!(session.actions().len(), 3);
    }

    #[test]
    fn runtime_partition_session_script_callback_error_preserves_recorded_action() {
        let launch = recursive_partition_launch();
        let mut service = LocalRuntimeWorkerService::new();
        let mut session = RuntimePartitionSession::deploy_local(launch, &mut service).unwrap();
        let script = RuntimePartitionSessionRunScript::every_actions(
            vec![RuntimePartitionWorkerActionKind::Tick],
            1,
        );

        let err = session
            .run_local_script(&script, &mut service, |_, _| {
                Err(error(
                    "E_TEST_PARTITION_SESSION_TELEMETRY",
                    "partition session telemetry sink failed",
                ))
            })
            .unwrap_err();

        assert_eq!(
            err.diagnostics[0].code,
            "E_TEST_PARTITION_SESSION_TELEMETRY"
        );
        assert_eq!(session.actions().len(), 1);
        assert_eq!(
            session.latest_action().unwrap().kind,
            RuntimePartitionWorkerActionKind::Tick
        );
    }

    #[test]
    fn runtime_partition_session_script_validates_cadence() {
        let launch = recursive_partition_launch();
        let mut service = LocalRuntimeWorkerService::new();
        let mut session = RuntimePartitionSession::deploy_local(launch, &mut service).unwrap();
        let script = RuntimePartitionSessionRunScript {
            actions: vec![RuntimePartitionWorkerActionKind::Tick],
            every_actions: 0,
            emit_initial: false,
            emit_final: true,
        };

        let err = session
            .run_local_script(&script, &mut service, |_, _| Ok(()))
            .unwrap_err();

        assert_eq!(
            err.diagnostics[0].code,
            "E_RUNTIME_PARTITION_SESSION_SCRIPT"
        );
        assert!(session.actions().is_empty());
    }

    #[test]
    fn runtime_partition_session_loopback_runs_script() {
        let launch = recursive_partition_launch();
        let service = Arc::new(Mutex::new(LocalRuntimeWorkerService::new()));
        let mut session =
            RuntimePartitionSession::deploy_loopback(launch.clone(), service.clone()).unwrap();
        let script = RuntimePartitionSessionRunScript::every_actions(
            vec![RuntimePartitionWorkerActionKind::Tick],
            1,
        );
        let mut events = Vec::new();

        let report = session
            .run_loopback_script(&script, service, |event, telemetry| {
                events.push(event);
                assert_eq!(telemetry.actions.len(), event.completed_actions);
                Ok(())
            })
            .unwrap();

        assert_eq!(report.completed_actions, 1);
        assert_eq!(report.telemetry_emitted, 1);
        assert_eq!(
            report.action_summary.operations.tick.calls,
            launch.workers.len() as u64
        );
        assert_eq!(
            events,
            vec![RuntimePartitionSessionRunEvent {
                completed_actions: 1,
                reason: RuntimePartitionSessionRunEventReason::Cadence,
                latest_action_kind: Some(RuntimePartitionWorkerActionKind::Tick),
            }]
        );
    }

    #[test]
    fn runtime_partition_session_tcp_runs_script() {
        let launch = recursive_partition_launch();
        let topology = two_cpu_topology();
        let (endpoints, handles) = spawn_tcp_servers_for_topology_connections(&topology, 2);
        let mut session = RuntimePartitionSession::deploy_tcp(
            launch.clone(),
            &endpoints,
            TcpRuntimeTransportConfig::default(),
        )
        .unwrap();
        let script = RuntimePartitionSessionRunScript::every_actions(
            vec![RuntimePartitionWorkerActionKind::TickMany(3)],
            1,
        );
        let mut events = Vec::new();

        let report = session
            .run_tcp_script(
                &script,
                &endpoints,
                TcpRuntimeTransportConfig::default(),
                |event, telemetry| {
                    events.push(event);
                    assert_eq!(
                        telemetry.action_summary.action_count,
                        event.completed_actions
                    );
                    Ok(())
                },
            )
            .unwrap();

        assert_eq!(report.completed_actions, 1);
        assert_eq!(report.telemetry_emitted, 1);
        assert_eq!(
            report.action_summary.operations.tick_many.calls,
            launch.workers.len() as u64
        );
        assert_eq!(
            events,
            vec![RuntimePartitionSessionRunEvent {
                completed_actions: 1,
                reason: RuntimePartitionSessionRunEventReason::Cadence,
                latest_action_kind: Some(RuntimePartitionWorkerActionKind::TickMany(3)),
            }]
        );
        join_tcp_servers(handles);
    }

    #[test]
    fn worker_protocol_serde_round_trips_requests_and_responses() {
        let (design, en, _) = counter_design();
        let shard = runtime_shard_info("cpu0", RuntimeBackend::PackedCpu, 0, 2);
        let partition_worker_launch = recursive_partition_launch()
            .workers
            .into_iter()
            .find(|worker| !worker.outbound_routes.is_empty())
            .unwrap();
        let route_payload = RuntimePartitionRoutePayload {
            route_index: partition_worker_launch.outbound_routes[0],
            width: partition_worker_launch.outbound_route_specs[0].width,
            limbs: vec![1],
        };
        let requests = vec![
            init_request(&design, "Counter", shard.clone()),
            RuntimeWorkerRequest::InitPartitionWorker(RuntimePartitionWorkerInit {
                worker_id: partition_worker_launch.worker_id.clone(),
                launch: partition_worker_launch.clone(),
            }),
            RuntimeWorkerRequest::RunPartitionWorkerAction(RuntimePartitionWorkerAction {
                worker_id: partition_worker_launch.worker_id.clone(),
                kind: RuntimePartitionWorkerActionKind::TickMany(4),
            }),
            RuntimeWorkerRequest::StorePartitionRouteOutbound {
                worker_id: partition_worker_launch.worker_id.clone(),
                payload: route_payload.clone(),
            },
            RuntimeWorkerRequest::PartitionRouteMailbox {
                worker_id: partition_worker_launch.worker_id.clone(),
            },
            RuntimeWorkerRequest::ClearPartitionRouteMailboxes {
                worker_id: partition_worker_launch.worker_id.clone(),
            },
            RuntimeWorkerRequest::Health {
                worker_id: "cpu0".to_string(),
            },
            RuntimeWorkerRequest::PartitionWorkerHealth {
                worker_id: partition_worker_launch.worker_id.clone(),
            },
            RuntimeWorkerRequest::SetInputLimbs {
                worker_id: "cpu0".to_string(),
                signal: en,
                lane_values: vec![vec![1], vec![0]],
            },
            RuntimeWorkerRequest::TickMany {
                worker_id: "cpu0".to_string(),
                steps: 4,
            },
            RuntimeWorkerRequest::Stats {
                worker_id: "cpu0".to_string(),
            },
        ];

        for request in requests {
            let json = serde_json::to_string(&request).unwrap();
            let decoded: RuntimeWorkerRequest = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, request);
        }

        let responses = vec![
            RuntimeWorkerResponse::Ack,
            RuntimeWorkerResponse::Health(RuntimeWorkerHealth {
                worker_id: "cpu0".to_string(),
                initialized: true,
                shard: Some(shard.clone()),
                operations: Some(RuntimeOperationStats::default()),
            }),
            RuntimeWorkerResponse::PartitionWorkerHealth(RuntimePartitionWorkerHealth {
                worker_id: partition_worker_launch.worker_id.clone(),
                initialized: true,
                backend: Some(partition_worker_launch.backend),
                node: Some(partition_worker_launch.node.clone()),
                partitions: partition_worker_launch.partitions.clone(),
                outbound_routes: partition_worker_launch.outbound_routes.clone(),
                inbound_routes: partition_worker_launch.inbound_routes.clone(),
                outbound_bits: partition_worker_launch.outbound_bits,
                inbound_bits: partition_worker_launch.inbound_bits,
                diagnostics: Vec::new(),
            }),
            RuntimeWorkerResponse::PartitionWorkerAction(RuntimePartitionWorkerActionReport {
                worker_id: partition_worker_launch.worker_id.clone(),
                kind: RuntimePartitionWorkerActionKind::TickMany(4),
                partition_count: partition_worker_launch.partitions.len(),
                outbound_route_count: partition_worker_launch.outbound_routes.len(),
                inbound_route_count: partition_worker_launch.inbound_routes.len(),
                outbound_bits: partition_worker_launch.outbound_bits,
                inbound_bits: partition_worker_launch.inbound_bits,
                operations: RuntimeOperationStats::default(),
            }),
            RuntimeWorkerResponse::PartitionRouteMailbox(RuntimePartitionRouteMailboxReport {
                worker_id: partition_worker_launch.worker_id.clone(),
                outbound_payload_count: 1,
                inbound_payload_count: 0,
                outbound_routes: partition_worker_launch.outbound_routes.clone(),
                inbound_routes: partition_worker_launch.inbound_routes.clone(),
                diagnostics: Vec::new(),
            }),
            RuntimeWorkerResponse::SignalLimbs(vec![vec![1], vec![2]]),
            RuntimeWorkerResponse::MemoryLimbs(vec![vec![vec![1]], vec![vec![2]]]),
            RuntimeWorkerResponse::Snapshot(RuntimeShardSnapshot {
                shard: shard.clone(),
                values: vec![1, 2],
                memories: vec![3, 4],
            }),
            RuntimeWorkerResponse::Stats(RuntimeShardStats {
                shard,
                operations: RuntimeOperationStats::default(),
            }),
        ];

        for response in responses {
            let json = serde_json::to_string(&response).unwrap();
            let decoded: RuntimeWorkerResponse = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, response);
        }
    }

    #[test]
    fn tcp_json_line_framing_round_trips_and_reports_errors() {
        let (design, en, _) = counter_design();
        let partition_worker_launch = recursive_partition_launch().workers[0].clone();
        let request = RuntimeWorkerRequest::SetInputLimbs {
            worker_id: "cpu0".to_string(),
            signal: en,
            lane_values: vec![vec![1], vec![0]],
        };
        let mut bytes = Vec::new();
        write_json_line(&mut bytes, &request).unwrap();
        let decoded: RuntimeWorkerRequest =
            read_json_line(&mut BufReader::new(bytes.as_slice()), "request EOF").unwrap();
        assert_eq!(decoded, request);

        let request = RuntimeWorkerRequest::InitPartitionWorker(RuntimePartitionWorkerInit {
            worker_id: partition_worker_launch.worker_id.clone(),
            launch: partition_worker_launch.clone(),
        });
        let mut bytes = Vec::new();
        write_json_line(&mut bytes, &request).unwrap();
        let decoded: RuntimeWorkerRequest =
            read_json_line(&mut BufReader::new(bytes.as_slice()), "request EOF").unwrap();
        assert_eq!(decoded, request);

        let request =
            RuntimeWorkerRequest::RunPartitionWorkerAction(RuntimePartitionWorkerAction {
                worker_id: partition_worker_launch.worker_id.clone(),
                kind: RuntimePartitionWorkerActionKind::Tick,
            });
        let mut bytes = Vec::new();
        write_json_line(&mut bytes, &request).unwrap();
        let decoded: RuntimeWorkerRequest =
            read_json_line(&mut BufReader::new(bytes.as_slice()), "request EOF").unwrap();
        assert_eq!(decoded, request);

        let response =
            RuntimeWorkerWireResponse::Ok(RuntimeWorkerResponse::Stats(RuntimeShardStats {
                shard: runtime_shard_info("cpu0", RuntimeBackend::PackedCpu, 0, 2),
                operations: RuntimeOperationStats::default(),
            }));
        let mut bytes = Vec::new();
        write_json_line(&mut bytes, &response).unwrap();
        let decoded = read_worker_wire_response(&mut BufReader::new(bytes.as_slice())).unwrap();
        assert_eq!(decoded, response);

        let err = read_json_line::<RuntimeWorkerRequest>(
            &mut BufReader::new(b"{not json}\n".as_slice()),
            "request EOF",
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_TCP_JSON");

        let json = serde_json::to_string(&init_request(
            &design,
            "Counter",
            runtime_shard_info("cpu0", RuntimeBackend::PackedCpu, 0, 2),
        ))
        .unwrap();
        let err = read_json_line::<RuntimeWorkerRequest>(
            &mut BufReader::new(json.as_bytes()),
            "request EOF",
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_TCP_EOF");
    }

    #[test]
    fn tcp_worker_service_runs_commands_and_preserves_diagnostics() {
        let (design, en, out) = counter_design();
        let partition_worker_launch = recursive_partition_launch().workers[0].clone();
        let (addr, handle) = spawn_tcp_worker_server();
        {
            let mut client = TcpRuntimeShardClient::connect("cpu0", addr).unwrap();
            let health = client.health().unwrap();
            assert_eq!(health.worker_id, "cpu0");
            assert!(!health.initialized);
            assert_eq!(health.shard, None);
            assert_eq!(health.operations, None);

            let err = client
                .request(RuntimeWorkerRequest::Tick {
                    worker_id: "cpu0".to_string(),
                })
                .unwrap_err();
            assert_eq!(err.diagnostics[0].code, "E_RUNTIME_WORKER_SHARD");

            expect_ack(
                client
                    .request(RuntimeWorkerRequest::InitPartitionWorker(
                        RuntimePartitionWorkerInit {
                            worker_id: partition_worker_launch.worker_id.clone(),
                            launch: partition_worker_launch.clone(),
                        },
                    ))
                    .unwrap(),
            )
            .unwrap();
            let RuntimeWorkerResponse::PartitionWorkerHealth(partition_health) = client
                .request(RuntimeWorkerRequest::PartitionWorkerHealth {
                    worker_id: partition_worker_launch.worker_id.clone(),
                })
                .unwrap()
            else {
                panic!("expected partition worker health response");
            };
            assert!(partition_health.initialized);
            assert_eq!(
                partition_health.partitions,
                partition_worker_launch.partitions
            );

            let RuntimeWorkerResponse::PartitionWorkerAction(action_report) = client
                .request(RuntimeWorkerRequest::RunPartitionWorkerAction(
                    RuntimePartitionWorkerAction {
                        worker_id: partition_worker_launch.worker_id.clone(),
                        kind: RuntimePartitionWorkerActionKind::Tick,
                    },
                ))
                .unwrap()
            else {
                panic!("expected partition worker action report");
            };
            assert_eq!(action_report.worker_id, partition_worker_launch.worker_id);
            assert_eq!(action_report.operations.tick.calls, 1);

            let shard = runtime_shard_info("cpu0", RuntimeBackend::PackedCpu, 0, 2);
            expect_ack(
                client
                    .request(init_request(&design, "Counter", shard.clone()))
                    .unwrap(),
            )
            .unwrap();
            let err = client
                .request(init_request(&design, "Counter", shard))
                .unwrap_err();
            assert_eq!(err.diagnostics[0].code, "E_RUNTIME_WORKER_INIT");

            expect_ack(
                client
                    .request(RuntimeWorkerRequest::SetInput {
                        worker_id: "cpu0".to_string(),
                        signal: en,
                        lane_values: vec![1, 0],
                    })
                    .unwrap(),
            )
            .unwrap();
            expect_ack(
                client
                    .request(RuntimeWorkerRequest::Tick {
                        worker_id: "cpu0".to_string(),
                    })
                    .unwrap(),
            )
            .unwrap();
            let RuntimeWorkerResponse::SignalLimbs(values) = client
                .request(RuntimeWorkerRequest::GetSignalLimbs {
                    worker_id: "cpu0".to_string(),
                    signal: out,
                })
                .unwrap()
            else {
                panic!("expected signal response");
            };
            assert_eq!(values, vec![vec![1], vec![0]]);

            let RuntimeWorkerResponse::Stats(stats) = client
                .request(RuntimeWorkerRequest::Stats {
                    worker_id: "cpu0".to_string(),
                })
                .unwrap()
            else {
                panic!("expected stats response");
            };
            assert_eq!(stats.operations.tick.calls, 1);

            let health = client.health().unwrap();
            assert!(health.initialized);
            assert_eq!(health.operations.unwrap().tick.calls, 1);
        }
        join_tcp_servers(vec![handle]);
    }

    #[test]
    fn tcp_worker_server_serves_bounded_connections() {
        let mut server =
            TcpRuntimeWorkerServer::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
        let addr = server.local_addr().unwrap();
        let handle = std::thread::spawn(move || server.serve_connections(1));

        let client = TcpRuntimeShardClient::connect("cpu0", addr).unwrap();
        drop(client);

        join_tcp_servers(vec![handle]);
    }

    #[test]
    fn local_worker_service_health_reports_uninitialized_and_initialized_shards() {
        let (design, _, _) = counter_design();
        let mut service = LocalRuntimeWorkerService::new();

        let RuntimeWorkerResponse::Health(health) = service
            .handle(RuntimeWorkerRequest::Health {
                worker_id: "cpu0".to_string(),
            })
            .unwrap()
        else {
            panic!("expected health response");
        };
        assert_eq!(health.worker_id, "cpu0");
        assert!(!health.initialized);
        assert_eq!(health.shard, None);
        assert_eq!(health.operations, None);

        let shard = runtime_shard_info("cpu0", RuntimeBackend::PackedCpu, 0, 2);
        expect_ack(
            service
                .handle(init_request(&design, "Counter", shard.clone()))
                .unwrap(),
        )
        .unwrap();
        let RuntimeWorkerResponse::Health(health) = service
            .handle(RuntimeWorkerRequest::Health {
                worker_id: "cpu0".to_string(),
            })
            .unwrap()
        else {
            panic!("expected health response");
        };
        assert!(health.initialized);
        assert_eq!(health.shard, Some(shard));
        assert_eq!(health.operations, Some(RuntimeOperationStats::default()));
    }

    #[test]
    fn local_worker_service_runs_counter_commands_and_stats() {
        let (design, en, out) = counter_design();
        let mut service = LocalRuntimeWorkerService::new();
        let err = service
            .handle(RuntimeWorkerRequest::Tick {
                worker_id: "cpu0".to_string(),
            })
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_WORKER_SHARD");

        let shard = runtime_shard_info("cpu0", RuntimeBackend::PackedCpu, 0, 2);
        expect_ack(
            service
                .handle(init_request(&design, "Counter", shard.clone()))
                .unwrap(),
        )
        .unwrap();
        let err = service
            .handle(init_request(&design, "Counter", shard))
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_WORKER_INIT");

        expect_ack(
            service
                .handle(RuntimeWorkerRequest::SetInput {
                    worker_id: "cpu0".to_string(),
                    signal: en,
                    lane_values: vec![1, 0],
                })
                .unwrap(),
        )
        .unwrap();
        expect_ack(
            service
                .handle(RuntimeWorkerRequest::TickMany {
                    worker_id: "cpu0".to_string(),
                    steps: 2,
                })
                .unwrap(),
        )
        .unwrap();

        let RuntimeWorkerResponse::SignalLimbs(values) = service
            .handle(RuntimeWorkerRequest::GetSignalLimbs {
                worker_id: "cpu0".to_string(),
                signal: out,
            })
            .unwrap()
        else {
            panic!("expected signal response");
        };
        assert_eq!(values, vec![vec![2], vec![0]]);

        let RuntimeWorkerResponse::Stats(stats) = service
            .handle(RuntimeWorkerRequest::Stats {
                worker_id: "cpu0".to_string(),
            })
            .unwrap()
        else {
            panic!("expected stats response");
        };
        assert_eq!(stats.operations.tick_many.calls, 1);

        expect_ack(
            service
                .handle(RuntimeWorkerRequest::ResetStats {
                    worker_id: "cpu0".to_string(),
                })
                .unwrap(),
        )
        .unwrap();
        let RuntimeWorkerResponse::Stats(stats) = service
            .handle(RuntimeWorkerRequest::Stats {
                worker_id: "cpu0".to_string(),
            })
            .unwrap()
        else {
            panic!("expected stats response");
        };
        assert_operation_stats_zero(stats.operations);
    }

    #[test]
    fn local_worker_service_snapshots_and_restores_memory() {
        let (design, _, addr, _, mem, read) = memory_design();
        let mut service = LocalRuntimeWorkerService::new();
        let shard = runtime_shard_info("cpu0", RuntimeBackend::PackedCpu, 0, 2);
        expect_ack(
            service
                .handle(init_request(&design, "MemoryTop", shard))
                .unwrap(),
        )
        .unwrap();

        expect_ack(
            service
                .handle(RuntimeWorkerRequest::SetMemoryLimbs {
                    worker_id: "cpu0".to_string(),
                    memory: mem,
                    lane_words: vec![
                        vec![vec![1], vec![2], vec![3], vec![4]],
                        vec![vec![5], vec![6], vec![7], vec![8]],
                    ],
                })
                .unwrap(),
        )
        .unwrap();
        expect_ack(
            service
                .handle(RuntimeWorkerRequest::SetInput {
                    worker_id: "cpu0".to_string(),
                    signal: addr,
                    lane_values: vec![2, 1],
                })
                .unwrap(),
        )
        .unwrap();
        let RuntimeWorkerResponse::Snapshot(snapshot) = service
            .handle(RuntimeWorkerRequest::Snapshot {
                worker_id: "cpu0".to_string(),
            })
            .unwrap()
        else {
            panic!("expected snapshot response");
        };

        expect_ack(
            service
                .handle(RuntimeWorkerRequest::SetMemoryLimbs {
                    worker_id: "cpu0".to_string(),
                    memory: mem,
                    lane_words: vec![
                        vec![vec![9], vec![9], vec![9], vec![9]],
                        vec![vec![9], vec![9], vec![9], vec![9]],
                    ],
                })
                .unwrap(),
        )
        .unwrap();
        expect_ack(
            service
                .handle(RuntimeWorkerRequest::RestoreSnapshot {
                    worker_id: "cpu0".to_string(),
                    snapshot,
                })
                .unwrap(),
        )
        .unwrap();
        expect_ack(
            service
                .handle(RuntimeWorkerRequest::EvalCombinational {
                    worker_id: "cpu0".to_string(),
                })
                .unwrap(),
        )
        .unwrap();

        let RuntimeWorkerResponse::SignalLimbs(values) = service
            .handle(RuntimeWorkerRequest::GetSignalLimbs {
                worker_id: "cpu0".to_string(),
                signal: read,
            })
            .unwrap()
        else {
            panic!("expected signal response");
        };
        assert_eq!(values, vec![vec![3], vec![6]]);
    }

    #[test]
    fn runtime_stats_start_at_zero() {
        let (design, _, _) = counter_design();
        let runtime = DistributedRuntime::new(&design, "Counter", two_cpu_topology()).unwrap();
        let stats = runtime.stats();

        assert_eq!(stats.execution_mode, RuntimeExecutionMode::Serial);
        assert_eq!(stats.total_lanes, 5);
        assert_operation_stats_zero(stats.operations);
        assert_eq!(stats.shards.len(), 2);
        for shard in stats.shards {
            assert_operation_stats_zero(shard.operations);
        }
    }

    #[test]
    fn local_runtime_health_reports_shards_and_counters() {
        let (design, en, _) = counter_design();
        let mut runtime = DistributedRuntime::new(&design, "Counter", two_cpu_topology()).unwrap();
        runtime.set_input(en, &[1, 1, 1, 1, 1]).unwrap();
        runtime.tick().unwrap();

        let health = runtime.health().unwrap();
        assert_eq!(health.total_lanes, 5);
        assert_eq!(health.shards.len(), 2);
        assert_eq!(health.shards[0].status, RuntimeShardHealthStatus::Healthy);
        assert_eq!(health.shards[0].shard.worker_id, "cpu-a");
        assert_eq!(health.shards[0].operations.unwrap().tick.calls, 1);
        assert_eq!(health.shards[1].status, RuntimeShardHealthStatus::Healthy);
        assert_eq!(health.shards[1].shard.worker_id, "cpu-b");
        assert_eq!(health.shards[1].operations.unwrap().tick.calls, 1);
    }

    #[test]
    fn runtime_autotune_ranks_successful_cpu_candidates() {
        let (design, _, _) = counter_design();
        let candidates = vec![
            candidate(
                "serial",
                RuntimeTopology::local_cpu(5),
                DistributedRuntimeOptions::default(),
            ),
            candidate("parallel", two_cpu_topology(), parallel_options()),
        ];

        let report = recommend_runtime_topology(
            &design,
            "Counter",
            candidates,
            RuntimeAutotuneConfig {
                warmup_steps: 0,
                measure_steps: 2,
                stimulus: None,
            },
        )
        .unwrap();

        assert_eq!(report.candidates.len(), 2);
        assert!(report.best_index < report.candidates.len());
        for candidate in &report.candidates {
            assert!(candidate.score_ns.is_some());
            assert!(candidate.diagnostics.is_empty());
            let stats = candidate.stats.as_ref().unwrap();
            assert_eq!(stats.operations.tick_many.calls, 1);
            assert_eq!(
                stats.operations.tick_many.total_ns,
                candidate.score_ns.unwrap()
            );
        }
    }

    #[test]
    fn runtime_autotune_keeps_failed_candidates_when_one_succeeds() {
        let (design, _, _) = counter_design();
        let mut invalid = RuntimeTopology::new();
        invalid.push(RuntimeWorker::local_cpu("bad", 0));
        let candidates = vec![
            candidate("bad", invalid, DistributedRuntimeOptions::default()),
            candidate(
                "good",
                RuntimeTopology::local_cpu(2),
                DistributedRuntimeOptions::default(),
            ),
        ];

        let report = recommend_runtime_topology(
            &design,
            "Counter",
            candidates,
            RuntimeAutotuneConfig {
                warmup_steps: 0,
                measure_steps: 1,
                stimulus: None,
            },
        )
        .unwrap();

        assert_eq!(report.candidates.len(), 2);
        assert_eq!(report.best_index, 1);
        assert_eq!(report.candidates[0].score_ns, None);
        assert_eq!(report.candidates[0].stats, None);
        assert_eq!(report.candidates[0].diagnostics[0].code, "E_RUNTIME_LANES");
        assert!(report.candidates[1].score_ns.is_some());
    }

    #[test]
    fn runtime_autotune_validates_candidates_and_measure_steps() {
        let (design, _, _) = counter_design();
        let err = recommend_runtime_topology(
            &design,
            "Counter",
            Vec::new(),
            RuntimeAutotuneConfig::default(),
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_AUTOTUNE_CANDIDATES");

        let err = recommend_runtime_topology(
            &design,
            "Counter",
            vec![candidate(
                "serial",
                RuntimeTopology::local_cpu(1),
                DistributedRuntimeOptions::default(),
            )],
            RuntimeAutotuneConfig {
                warmup_steps: 0,
                measure_steps: 0,
                stimulus: None,
            },
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_AUTOTUNE_STEPS");
    }

    #[test]
    fn runtime_autotune_reports_all_candidate_failure() {
        let (design, _, _) = counter_design();
        let mut invalid = RuntimeTopology::new();
        invalid.push(RuntimeWorker::local_cpu("bad", 0));

        let err = recommend_runtime_topology(
            &design,
            "Counter",
            vec![candidate(
                "bad",
                invalid,
                DistributedRuntimeOptions::default(),
            )],
            RuntimeAutotuneConfig {
                warmup_steps: 0,
                measure_steps: 1,
                stimulus: None,
            },
        )
        .unwrap_err();

        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_AUTOTUNE_NO_CANDIDATE");
        assert!(err
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "E_RUNTIME_LANES"));
    }

    #[test]
    fn runtime_autotune_preserves_order_for_reported_ties() {
        let (design, _, _) = counter_design();
        let candidates = vec![
            candidate(
                "first",
                RuntimeTopology::local_cpu(1),
                DistributedRuntimeOptions::default(),
            ),
            candidate(
                "second",
                RuntimeTopology::local_cpu(1),
                DistributedRuntimeOptions::default(),
            ),
        ];

        let report = recommend_runtime_topology(
            &design,
            "Counter",
            candidates,
            RuntimeAutotuneConfig {
                warmup_steps: 0,
                measure_steps: 1,
                stimulus: None,
            },
        )
        .unwrap();

        assert_eq!(report.candidates[0].name, "first");
        assert_eq!(report.candidates[1].name, "second");
        if report.candidates[0].score_ns == report.candidates[1].score_ns {
            assert_eq!(report.best_index, 0);
        }
    }

    #[test]
    fn runtime_autotune_gpu_candidate_smoke_when_adapter_exists() {
        let (design, _, _) = counter_design();
        let mut gpu_topology = RuntimeTopology::new();
        gpu_topology.push(RuntimeWorker::local_gpu(
            "gpu0",
            2,
            GpuBatchOptions::default(),
        ));
        let report = recommend_runtime_topology(
            &design,
            "Counter",
            vec![candidate("gpu", gpu_topology, parallel_options())],
            RuntimeAutotuneConfig {
                warmup_steps: 0,
                measure_steps: 1,
                stimulus: None,
            },
        );

        let Ok(report) = report else {
            return;
        };
        assert_eq!(report.candidates.len(), 1);
        assert_eq!(report.best_index, 0);
        assert!(report.candidates[0].score_ns.is_some());
    }

    #[test]
    fn runtime_autotune_stimulus_setup_applies_inputs_and_memory() {
        let (design, _, addr, _, mem, read) = memory_design();
        let mut runtime = DistributedRuntime::local_cpu(&design, "MemoryTop", 2).unwrap();
        let setup = RuntimeStimulusSetup {
            inputs: vec![RuntimeSignalValue {
                signal: addr,
                lane_values: vec![2, 1],
            }],
            memories: vec![RuntimeMemoryValue {
                memory: mem,
                lane_words: vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
            }],
            ..RuntimeStimulusSetup::default()
        };

        apply_stimulus_setup(&mut runtime, &setup).unwrap();
        runtime.eval_combinational().unwrap();

        assert_eq!(runtime.get_signal(read).unwrap(), vec![3, 6]);
    }

    #[test]
    fn runtime_autotune_stepped_stimulus_counts_ticks() {
        let (design, en, _) = counter_design();
        let report = recommend_runtime_topology(
            &design,
            "Counter",
            vec![candidate(
                "serial",
                RuntimeTopology::local_cpu(3),
                DistributedRuntimeOptions::default(),
            )],
            RuntimeAutotuneConfig {
                warmup_steps: 1,
                measure_steps: 4,
                stimulus: Some(RuntimeAutotuneStimulus {
                    setup: RuntimeStimulusSetup::default(),
                    steps: vec![
                        RuntimeStimulusStep {
                            inputs: vec![RuntimeSignalValue {
                                signal: en,
                                lane_values: vec![1, 1, 1],
                            }],
                            input_limbs: Vec::new(),
                        },
                        RuntimeStimulusStep {
                            inputs: vec![RuntimeSignalValue {
                                signal: en,
                                lane_values: vec![0, 0, 0],
                            }],
                            input_limbs: Vec::new(),
                        },
                    ],
                }),
            },
        )
        .unwrap();

        let stats = report.candidates[0].stats.as_ref().unwrap();
        assert_eq!(stats.operations.tick.calls, 4);
        assert_eq!(stats.operations.tick_many.calls, 0);
        assert_eq!(
            report.candidates[0].score_ns,
            Some(stats.operations.tick.total_ns)
        );
    }

    #[test]
    fn runtime_autotune_limb_stimulus_supports_wide_inputs() {
        let (design, wide, _) = wide_input_design();
        let report = recommend_runtime_topology(
            &design,
            "WideInput",
            vec![candidate(
                "serial",
                RuntimeTopology::local_cpu(2),
                DistributedRuntimeOptions::default(),
            )],
            RuntimeAutotuneConfig {
                warmup_steps: 0,
                measure_steps: 1,
                stimulus: Some(RuntimeAutotuneStimulus {
                    setup: RuntimeStimulusSetup {
                        input_limbs: vec![RuntimeSignalLimbs {
                            signal: wide,
                            lane_values: vec![vec![1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10]],
                        }],
                        ..RuntimeStimulusSetup::default()
                    },
                    steps: Vec::new(),
                }),
            },
        )
        .unwrap();

        assert!(report.candidates[0].diagnostics.is_empty());
        assert!(report.candidates[0].score_ns.is_some());
    }

    #[test]
    fn runtime_autotune_failed_stimulus_candidate_is_kept() {
        let (design, en, _) = counter_design();
        let candidates = vec![
            candidate(
                "bad-lanes",
                RuntimeTopology::local_cpu(1),
                DistributedRuntimeOptions::default(),
            ),
            candidate(
                "good",
                RuntimeTopology::local_cpu(2),
                DistributedRuntimeOptions::default(),
            ),
        ];
        let report = recommend_runtime_topology(
            &design,
            "Counter",
            candidates,
            RuntimeAutotuneConfig {
                warmup_steps: 0,
                measure_steps: 1,
                stimulus: Some(RuntimeAutotuneStimulus {
                    setup: RuntimeStimulusSetup {
                        inputs: vec![RuntimeSignalValue {
                            signal: en,
                            lane_values: vec![1, 0],
                        }],
                        ..RuntimeStimulusSetup::default()
                    },
                    steps: Vec::new(),
                }),
            },
        )
        .unwrap();

        assert_eq!(report.best_index, 1);
        assert_eq!(report.candidates[0].score_ns, None);
        assert_eq!(
            report.candidates[0].diagnostics[0].code,
            "E_RUNTIME_LANE_VALUES"
        );
        assert!(report.candidates[1].score_ns.is_some());
    }

    #[test]
    fn runtime_autotune_stimulus_serde_round_trip() {
        let (_, signal, _) = counter_design();
        let stimulus = RuntimeAutotuneStimulus {
            setup: RuntimeStimulusSetup {
                inputs: vec![RuntimeSignalValue {
                    signal,
                    lane_values: vec![1, 2],
                }],
                memory_limbs: vec![RuntimeMemoryLimbs {
                    memory: signal,
                    lane_words: vec![vec![vec![1, 2]], vec![vec![3, 4]]],
                }],
                ..RuntimeStimulusSetup::default()
            },
            steps: vec![RuntimeStimulusStep {
                inputs: Vec::new(),
                input_limbs: vec![RuntimeSignalLimbs {
                    signal,
                    lane_values: vec![vec![5, 6], vec![7, 8]],
                }],
            }],
        };

        let json = serde_json::to_string(&stimulus).unwrap();
        let decoded: RuntimeAutotuneStimulus = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded, stimulus);
    }

    #[test]
    fn runtime_snapshot_restores_state_into_matching_shards() {
        let (design, en, we, addr, data, mem, count, read) = snapshot_design();
        let mut source =
            DistributedRuntime::new(&design, "SnapshotTop", two_cpu_topology()).unwrap();
        source.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
        source.set_input(we, &[1, 1, 1, 1, 1]).unwrap();
        source.set_input(addr, &[0, 1, 2, 3, 0]).unwrap();
        source.set_input(data, &[9, 11, 13, 15, 17]).unwrap();
        source.tick().unwrap();
        source.set_input(we, &[0, 0, 0, 0, 0]).unwrap();
        source.eval_combinational().unwrap();
        let snapshot = source.snapshot().unwrap();
        let expected_count = source.get_signal(count).unwrap();
        let expected_read = source.get_signal(read).unwrap();
        let expected_memory = source.get_memory(mem).unwrap();

        source.set_input(en, &[1, 1, 1, 1, 1]).unwrap();
        source.tick().unwrap();

        let mut restore_topology = RuntimeTopology::new();
        restore_topology.push(RuntimeWorker::local_cpu("restore-a", 2));
        restore_topology.push(RuntimeWorker::local_cpu("restore-b", 3));
        let mut restored =
            DistributedRuntime::new(&design, "SnapshotTop", restore_topology).unwrap();
        restored.restore_snapshot(&snapshot).unwrap();

        assert_eq!(restored.get_signal(count).unwrap(), expected_count);
        assert_eq!(restored.get_signal(read).unwrap(), expected_read);
        assert_eq!(restored.get_memory(mem).unwrap(), expected_memory);
    }

    #[test]
    fn runtime_snapshot_restore_validates_shape() {
        let (design, en, _, _, _, _, _, _) = snapshot_design();
        let mut runtime =
            DistributedRuntime::new(&design, "SnapshotTop", two_cpu_topology()).unwrap();
        runtime.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
        runtime.tick().unwrap();
        let snapshot = runtime.snapshot().unwrap();

        let mut bad = snapshot.clone();
        bad.total_lanes = 4;
        let err = runtime.restore_snapshot(&bad).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_SNAPSHOT_TOPOLOGY");

        let mut bad = snapshot.clone();
        bad.program_top = "OtherTop".to_string();
        let err = runtime.restore_snapshot(&bad).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_SNAPSHOT_PROGRAM");

        let mut bad = snapshot.clone();
        bad.shards[0].values.pop();
        let err = runtime.restore_snapshot(&bad).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_SNAPSHOT_STORAGE");

        let mut mismatched_topology = RuntimeTopology::new();
        mismatched_topology.push(RuntimeWorker::local_cpu("cpu-a", 1));
        mismatched_topology.push(RuntimeWorker::local_cpu("cpu-b", 4));
        let mut mismatched =
            DistributedRuntime::new(&design, "SnapshotTop", mismatched_topology).unwrap();
        let err = mismatched.restore_snapshot(&snapshot).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_SNAPSHOT_TOPOLOGY");
    }

    #[test]
    fn runtime_snapshot_serde_round_trip() {
        let (design, en, _, _, _, _, _, _) = snapshot_design();
        let mut runtime =
            DistributedRuntime::new(&design, "SnapshotTop", two_cpu_topology()).unwrap();
        runtime.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
        runtime.tick().unwrap();
        let snapshot = runtime.snapshot().unwrap();

        let json = serde_json::to_string(&snapshot).unwrap();
        let decoded: RuntimeSnapshot = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded, snapshot);
    }

    #[test]
    fn runtime_checkpoint_json_round_trip_preserves_topology_and_endpoints() {
        let (design, en, _, _, _, _, _, _) = snapshot_design();
        let topology = two_cpu_topology();
        let mut runtime =
            DistributedRuntime::new(&design, "SnapshotTop", topology.clone()).unwrap();
        runtime.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
        runtime.tick().unwrap();

        let mut endpoints = HashMap::new();
        endpoints.insert("cpu-a".to_string(), "127.0.0.1:10001".parse().unwrap());
        endpoints.insert("cpu-b".to_string(), "127.0.0.1:10002".parse().unwrap());
        let checkpoint = runtime.checkpoint_with_tcp_endpoints(&endpoints).unwrap();

        assert_eq!(checkpoint.format_version, RUNTIME_CHECKPOINT_FORMAT_VERSION);
        assert_eq!(checkpoint.module_name, "SnapshotTop");
        assert_eq!(checkpoint.topology, topology);
        assert_eq!(
            checkpoint.tcp_endpoints,
            vec![
                RuntimeTcpEndpoint {
                    worker_id: "cpu-a".to_string(),
                    addr: "127.0.0.1:10001".to_string(),
                },
                RuntimeTcpEndpoint {
                    worker_id: "cpu-b".to_string(),
                    addr: "127.0.0.1:10002".to_string(),
                },
            ]
        );
        assert_eq!(checkpoint.tcp_endpoint_map().unwrap(), endpoints);

        let mut bytes = Vec::new();
        checkpoint.write_json(&mut bytes).unwrap();
        let decoded = RuntimeCheckpoint::read_json(&mut bytes.as_slice()).unwrap();

        assert_eq!(decoded, checkpoint);
    }

    #[test]
    fn runtime_checkpoint_restores_state_into_matching_runtime() {
        let (design, en, we, addr, data, mem, count, read) = snapshot_design();
        let topology = two_cpu_topology();
        let mut source = DistributedRuntime::new(&design, "SnapshotTop", topology.clone()).unwrap();
        source.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
        source.set_input(we, &[1, 1, 1, 1, 1]).unwrap();
        source.set_input(addr, &[0, 1, 2, 3, 0]).unwrap();
        source.set_input(data, &[9, 11, 13, 15, 17]).unwrap();
        source.tick().unwrap();
        source.set_input(we, &[0, 0, 0, 0, 0]).unwrap();
        source.eval_combinational().unwrap();
        let checkpoint = source.checkpoint().unwrap();
        let expected_count = source.get_signal(count).unwrap();
        let expected_read = source.get_signal(read).unwrap();
        let expected_memory = source.get_memory(mem).unwrap();

        let mut restored = DistributedRuntime::new(&design, "SnapshotTop", topology).unwrap();
        restored.restore_checkpoint(&checkpoint).unwrap();

        assert_eq!(restored.get_signal(count).unwrap(), expected_count);
        assert_eq!(restored.get_signal(read).unwrap(), expected_read);
        assert_eq!(restored.get_memory(mem).unwrap(), expected_memory);
    }

    #[test]
    fn runtime_checkpoint_validates_version_topology_endpoints_and_snapshot() {
        let (design, en, _, _, _, _, _, _) = snapshot_design();
        let topology = two_cpu_topology();
        let mut runtime =
            DistributedRuntime::new(&design, "SnapshotTop", topology.clone()).unwrap();
        runtime.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
        runtime.tick().unwrap();
        let checkpoint = runtime.checkpoint().unwrap();

        let mut bad = checkpoint.clone();
        bad.format_version = RUNTIME_CHECKPOINT_FORMAT_VERSION + 1;
        let err = RuntimeCheckpoint::read_json(&mut serde_json::to_vec(&bad).unwrap().as_slice())
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_CHECKPOINT_VERSION");
        let err = runtime.restore_checkpoint(&bad).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_CHECKPOINT_VERSION");

        let mut bad = checkpoint.clone();
        bad.module_name = "OtherTop".to_string();
        let err = runtime.restore_checkpoint(&bad).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_CHECKPOINT_PROGRAM");

        let mut bad = checkpoint.clone();
        bad.topology.workers[0].id = "other-worker".to_string();
        let err = runtime.restore_checkpoint(&bad).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_CHECKPOINT_TOPOLOGY");

        let mut bad = checkpoint.clone();
        bad.snapshot.shards[0].values.pop();
        let err = runtime.restore_checkpoint(&bad).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_SNAPSHOT_STORAGE");

        let err = runtime
            .checkpoint_with_tcp_endpoints(&HashMap::from([(
                "missing".to_string(),
                "127.0.0.1:1".parse().unwrap(),
            )]))
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_CHECKPOINT_ENDPOINT");

        let mut bad = checkpoint.clone();
        bad.tcp_endpoints = vec![RuntimeTcpEndpoint {
            worker_id: "cpu-a".to_string(),
            addr: "not-a-socket".to_string(),
        }];
        let err = bad.tcp_endpoint_map().unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_CHECKPOINT_ENDPOINT");

        bad.tcp_endpoints = vec![
            RuntimeTcpEndpoint {
                worker_id: "cpu-a".to_string(),
                addr: "127.0.0.1:1".to_string(),
            },
            RuntimeTcpEndpoint {
                worker_id: "cpu-a".to_string(),
                addr: "127.0.0.1:2".to_string(),
            },
        ];
        let err = bad.tcp_endpoint_map().unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_CHECKPOINT_ENDPOINT");
    }

    #[test]
    fn runtime_checkpoint_cadence_emits_expected_boundaries() {
        let (design, en, out) = counter_design();
        let mut runtime = DistributedRuntime::new(&design, "Counter", two_cpu_topology()).unwrap();
        runtime.set_input(en, &[1, 1, 1, 1, 1]).unwrap();
        let mut events = Vec::new();
        let mut checkpoints = Vec::new();

        let report = runtime
            .tick_many_with_checkpoints(
                5,
                RuntimeCheckpointCadence {
                    every_steps: 2,
                    include_initial: true,
                    include_final: true,
                },
                |event, checkpoint| {
                    events.push(event);
                    checkpoints.push(checkpoint.clone());
                    Ok(())
                },
            )
            .unwrap();

        assert_eq!(
            events,
            vec![
                RuntimeCheckpointEvent {
                    completed_steps: 0,
                    reason: RuntimeCheckpointReason::Initial,
                },
                RuntimeCheckpointEvent {
                    completed_steps: 2,
                    reason: RuntimeCheckpointReason::Cadence,
                },
                RuntimeCheckpointEvent {
                    completed_steps: 4,
                    reason: RuntimeCheckpointReason::Cadence,
                },
                RuntimeCheckpointEvent {
                    completed_steps: 5,
                    reason: RuntimeCheckpointReason::Final,
                },
            ]
        );
        assert_eq!(
            report,
            RuntimeCheckpointRunReport {
                requested_steps: 5,
                completed_steps: 5,
                checkpoints_emitted: 4,
            }
        );

        let mut direct = DistributedRuntime::new(&design, "Counter", two_cpu_topology()).unwrap();
        direct.set_input(en, &[1, 1, 1, 1, 1]).unwrap();
        direct.tick_many(5).unwrap();

        let mut restored = DistributedRuntime::new(&design, "Counter", two_cpu_topology()).unwrap();
        restored
            .restore_checkpoint(checkpoints.last().unwrap())
            .unwrap();

        assert_eq!(
            runtime.get_signal(out).unwrap(),
            direct.get_signal(out).unwrap()
        );
        assert_eq!(
            restored.get_signal(out).unwrap(),
            direct.get_signal(out).unwrap()
        );
    }

    #[test]
    fn runtime_checkpoint_cadence_suppresses_duplicate_final() {
        let (design, en, _) = counter_design();
        let mut runtime = DistributedRuntime::new(&design, "Counter", two_cpu_topology()).unwrap();
        runtime.set_input(en, &[1, 1, 1, 1, 1]).unwrap();
        let mut events = Vec::new();

        let report = runtime
            .tick_many_with_checkpoints(
                4,
                RuntimeCheckpointCadence {
                    every_steps: 2,
                    include_initial: false,
                    include_final: true,
                },
                |event, _| {
                    events.push(event);
                    Ok(())
                },
            )
            .unwrap();

        assert_eq!(
            events,
            vec![
                RuntimeCheckpointEvent {
                    completed_steps: 2,
                    reason: RuntimeCheckpointReason::Cadence,
                },
                RuntimeCheckpointEvent {
                    completed_steps: 4,
                    reason: RuntimeCheckpointReason::Cadence,
                },
            ]
        );
        assert_eq!(report.checkpoints_emitted, 2);
    }

    #[test]
    fn runtime_checkpoint_cadence_supports_zero_steps() {
        let (design, _, _) = counter_design();
        let mut runtime = DistributedRuntime::new(&design, "Counter", two_cpu_topology()).unwrap();
        let mut events = Vec::new();

        let report = runtime
            .tick_many_with_checkpoints(
                0,
                RuntimeCheckpointCadence {
                    every_steps: 3,
                    include_initial: true,
                    include_final: true,
                },
                |event, _| {
                    events.push(event);
                    Ok(())
                },
            )
            .unwrap();

        assert_eq!(
            events,
            vec![RuntimeCheckpointEvent {
                completed_steps: 0,
                reason: RuntimeCheckpointReason::Initial,
            }]
        );
        assert_eq!(
            report,
            RuntimeCheckpointRunReport {
                requested_steps: 0,
                completed_steps: 0,
                checkpoints_emitted: 1,
            }
        );

        events.clear();
        runtime
            .tick_many_with_checkpoints(
                0,
                RuntimeCheckpointCadence {
                    every_steps: 3,
                    include_initial: false,
                    include_final: true,
                },
                |event, _| {
                    events.push(event);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(
            events,
            vec![RuntimeCheckpointEvent {
                completed_steps: 0,
                reason: RuntimeCheckpointReason::Final,
            }]
        );
    }

    #[test]
    fn runtime_checkpoint_cadence_includes_tcp_endpoints() {
        let (design, en, _) = counter_design();
        let mut runtime = DistributedRuntime::new(&design, "Counter", two_cpu_topology()).unwrap();
        runtime.set_input(en, &[1, 1, 1, 1, 1]).unwrap();
        let endpoints = HashMap::from([
            ("cpu-a".to_string(), "127.0.0.1:10001".parse().unwrap()),
            ("cpu-b".to_string(), "127.0.0.1:10002".parse().unwrap()),
        ]);
        let mut endpoint_maps = Vec::new();

        runtime
            .tick_many_with_tcp_checkpoints(
                1,
                RuntimeCheckpointCadence::every_steps(1),
                &endpoints,
                |_, checkpoint| {
                    endpoint_maps.push(checkpoint.tcp_endpoint_map().unwrap());
                    Ok(())
                },
            )
            .unwrap();

        assert_eq!(endpoint_maps, vec![endpoints]);
    }

    #[test]
    fn runtime_checkpoint_cadence_validates_every_steps() {
        let (design, _, _) = counter_design();
        let mut runtime = DistributedRuntime::new(&design, "Counter", two_cpu_topology()).unwrap();

        let err = runtime
            .tick_many_with_checkpoints(
                1,
                RuntimeCheckpointCadence {
                    every_steps: 0,
                    include_initial: false,
                    include_final: true,
                },
                |_, _| Ok(()),
            )
            .unwrap_err();

        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_CHECKPOINT_CADENCE");
    }

    #[test]
    fn runtime_checkpoint_cadence_propagates_callback_errors() {
        let (design, en, out) = counter_design();
        let mut runtime = DistributedRuntime::new(&design, "Counter", two_cpu_topology()).unwrap();
        runtime.set_input(en, &[1, 1, 1, 1, 1]).unwrap();
        let mut events = Vec::new();

        let err = runtime
            .tick_many_with_checkpoints(
                5,
                RuntimeCheckpointCadence {
                    every_steps: 2,
                    include_initial: false,
                    include_final: true,
                },
                |event, _| {
                    events.push(event);
                    Err(error("E_TEST_CHECKPOINT_SINK", "checkpoint sink failed"))
                },
            )
            .unwrap_err();

        assert_eq!(err.diagnostics[0].code, "E_TEST_CHECKPOINT_SINK");
        assert_eq!(
            events,
            vec![RuntimeCheckpointEvent {
                completed_steps: 2,
                reason: RuntimeCheckpointReason::Cadence,
            }]
        );
        assert_eq!(runtime.get_signal(out).unwrap(), vec![2, 2, 2, 2, 2]);
    }

    #[test]
    fn loopback_runtime_checkpoint_restore_round_trip() {
        let (design, en, we, addr, data, mem, count, read) = snapshot_design();
        let topology = two_cpu_topology();
        let mut source = DistributedRuntime::new_loopback_workers(
            &design,
            "SnapshotTop",
            topology.clone(),
            DistributedRuntimeOptions::default(),
        )
        .unwrap();
        source.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
        source.set_input(we, &[1, 1, 1, 1, 1]).unwrap();
        source.set_input(addr, &[0, 1, 2, 3, 0]).unwrap();
        source.set_input(data, &[9, 11, 13, 15, 17]).unwrap();
        source.tick().unwrap();
        source.set_input(we, &[0, 0, 0, 0, 0]).unwrap();
        source.eval_combinational().unwrap();
        let checkpoint = source.checkpoint().unwrap();
        let expected_count = source.get_signal(count).unwrap();
        let expected_read = source.get_signal(read).unwrap();
        let expected_memory = source.get_memory(mem).unwrap();

        let mut restored = DistributedRuntime::new_loopback_workers(
            &design,
            "SnapshotTop",
            topology,
            DistributedRuntimeOptions::default(),
        )
        .unwrap();
        restored.restore_checkpoint(&checkpoint).unwrap();

        assert_eq!(restored.get_signal(count).unwrap(), expected_count);
        assert_eq!(restored.get_signal(read).unwrap(), expected_read);
        assert_eq!(restored.get_memory(mem).unwrap(), expected_memory);
    }

    #[test]
    fn loopback_runtime_matches_direct_counter_execution() {
        let (design, en, out) = counter_design();
        let topology = two_cpu_topology();
        let mut direct = DistributedRuntime::new(&design, "Counter", topology.clone()).unwrap();
        let mut loopback = DistributedRuntime::new_loopback_workers(
            &design,
            "Counter",
            topology,
            DistributedRuntimeOptions::default(),
        )
        .unwrap();

        for runtime in [&mut direct, &mut loopback] {
            runtime.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
            runtime.tick().unwrap();
            runtime.set_input(en, &[1, 1, 0, 1, 1]).unwrap();
            runtime.tick_many(2).unwrap();
        }

        assert_eq!(
            loopback.get_signal(out).unwrap(),
            direct.get_signal(out).unwrap()
        );
        assert_eq!(loopback.shard_plan(), direct.shard_plan());
    }

    #[test]
    fn loopback_runtime_matches_direct_memory_execution() {
        let (design, we, addr, data, mem, read) = memory_design();
        let topology = two_cpu_topology();
        let mut direct = DistributedRuntime::new(&design, "MemoryTop", topology.clone()).unwrap();
        let mut loopback = DistributedRuntime::new_loopback_workers(
            &design,
            "MemoryTop",
            topology,
            DistributedRuntimeOptions::default(),
        )
        .unwrap();
        let seeded = vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
            vec![17, 18, 19, 20],
        ];

        for runtime in [&mut direct, &mut loopback] {
            runtime.set_memory(mem, &seeded).unwrap();
            runtime.set_input(we, &[1, 1, 0, 0, 1]).unwrap();
            runtime.set_input(addr, &[0, 1, 2, 3, 0]).unwrap();
            runtime.set_input(data, &[21, 22, 23, 24, 25]).unwrap();
            runtime.tick().unwrap();
            runtime.set_input(we, &[0, 0, 0, 0, 0]).unwrap();
            runtime.eval_combinational().unwrap();
        }

        assert_eq!(
            loopback.get_signal(read).unwrap(),
            direct.get_signal(read).unwrap()
        );
        assert_eq!(
            loopback.get_memory(mem).unwrap(),
            direct.get_memory(mem).unwrap()
        );
    }

    #[test]
    fn parallel_loopback_runtime_counts_operations() {
        let (design, en, out) = counter_design();
        let mut runtime = DistributedRuntime::new_loopback_workers(
            &design,
            "Counter",
            two_cpu_topology(),
            parallel_options(),
        )
        .unwrap();

        runtime.set_input(en, &[1, 1, 1, 1, 1]).unwrap();
        runtime.tick().unwrap();
        assert_eq!(runtime.get_signal(out).unwrap(), vec![1, 1, 1, 1, 1]);

        let stats = runtime.stats();
        assert_eq!(stats.execution_mode, RuntimeExecutionMode::Parallel);
        assert_eq!(stats.operations.tick.calls, 1);
        assert_eq!(stats.shards.len(), 2);
        for shard in stats.shards {
            assert_eq!(shard.operations.tick.calls, 1);
        }

        let health = runtime.health().unwrap();
        assert_eq!(health.total_lanes, 5);
        assert_eq!(health.shards.len(), 2);
        for shard in health.shards {
            assert_eq!(shard.status, RuntimeShardHealthStatus::Healthy);
            assert_eq!(shard.operations.unwrap().tick.calls, 1);
        }
    }

    #[test]
    fn loopback_runtime_snapshot_restore_round_trip() {
        let (design, en, we, addr, data, mem, count, read) = snapshot_design();
        let topology = two_cpu_topology();
        let mut source = DistributedRuntime::new_loopback_workers(
            &design,
            "SnapshotTop",
            topology.clone(),
            DistributedRuntimeOptions::default(),
        )
        .unwrap();
        source.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
        source.set_input(we, &[1, 1, 1, 1, 1]).unwrap();
        source.set_input(addr, &[0, 1, 2, 3, 0]).unwrap();
        source.set_input(data, &[9, 11, 13, 15, 17]).unwrap();
        source.tick().unwrap();
        source.set_input(we, &[0, 0, 0, 0, 0]).unwrap();
        source.eval_combinational().unwrap();
        let snapshot = source.snapshot().unwrap();
        let expected_count = source.get_signal(count).unwrap();
        let expected_read = source.get_signal(read).unwrap();
        let expected_memory = source.get_memory(mem).unwrap();

        let mut restored = DistributedRuntime::new_loopback_workers(
            &design,
            "SnapshotTop",
            topology,
            DistributedRuntimeOptions::default(),
        )
        .unwrap();
        restored.restore_snapshot(&snapshot).unwrap();

        assert_eq!(restored.get_signal(count).unwrap(), expected_count);
        assert_eq!(restored.get_signal(read).unwrap(), expected_read);
        assert_eq!(restored.get_memory(mem).unwrap(), expected_memory);
    }

    #[test]
    fn loopback_gpu_runtime_smoke_when_adapter_exists() {
        let (design, en, out) = counter_design();
        let topology = RuntimeTopology::local_heterogeneous(0, 2, GpuBatchOptions::default());
        let Ok(mut runtime) = DistributedRuntime::new_loopback_workers(
            &design,
            "Counter",
            topology,
            parallel_options(),
        ) else {
            return;
        };

        runtime.set_input(en, &[1, 0]).unwrap();
        runtime.tick().unwrap();
        assert_eq!(runtime.get_signal(out).unwrap(), vec![1, 0]);
    }

    #[test]
    fn tcp_runtime_matches_direct_counter_execution() {
        let (design, en, out) = counter_design();
        let topology = two_cpu_topology();
        let mut direct = DistributedRuntime::new(&design, "Counter", topology.clone()).unwrap();
        let (endpoints, handles) = spawn_tcp_servers_for_topology(&topology);
        let tcp_out = {
            let mut tcp = DistributedRuntime::new_tcp_workers(
                &design,
                "Counter",
                topology,
                DistributedRuntimeOptions::default(),
                endpoints,
            )
            .unwrap();

            for runtime in [&mut direct, &mut tcp] {
                runtime.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
                runtime.tick().unwrap();
                runtime.set_input(en, &[1, 1, 0, 1, 1]).unwrap();
                runtime.tick_many(2).unwrap();
            }
            tcp.get_signal(out).unwrap()
        };
        join_tcp_servers(handles);

        assert_eq!(tcp_out, direct.get_signal(out).unwrap());
    }

    #[test]
    fn tcp_runtime_matches_direct_memory_execution() {
        let (design, we, addr, data, mem, read) = memory_design();
        let topology = two_cpu_topology();
        let mut direct = DistributedRuntime::new(&design, "MemoryTop", topology.clone()).unwrap();
        let (endpoints, handles) = spawn_tcp_servers_for_topology(&topology);
        let (tcp_read, tcp_memory) = {
            let mut tcp = DistributedRuntime::new_tcp_workers(
                &design,
                "MemoryTop",
                topology,
                DistributedRuntimeOptions::default(),
                endpoints,
            )
            .unwrap();
            let seeded = vec![
                vec![1, 2, 3, 4],
                vec![5, 6, 7, 8],
                vec![9, 10, 11, 12],
                vec![13, 14, 15, 16],
                vec![17, 18, 19, 20],
            ];

            for runtime in [&mut direct, &mut tcp] {
                runtime.set_memory(mem, &seeded).unwrap();
                runtime.set_input(we, &[1, 1, 0, 0, 1]).unwrap();
                runtime.set_input(addr, &[0, 1, 2, 3, 0]).unwrap();
                runtime.set_input(data, &[21, 22, 23, 24, 25]).unwrap();
                runtime.tick().unwrap();
                runtime.set_input(we, &[0, 0, 0, 0, 0]).unwrap();
                runtime.eval_combinational().unwrap();
            }
            (tcp.get_signal(read).unwrap(), tcp.get_memory(mem).unwrap())
        };
        join_tcp_servers(handles);

        assert_eq!(tcp_read, direct.get_signal(read).unwrap());
        assert_eq!(tcp_memory, direct.get_memory(mem).unwrap());
    }

    #[test]
    fn parallel_tcp_runtime_counts_operations() {
        let (design, en, out) = counter_design();
        let topology = two_cpu_topology();
        let (endpoints, handles) = spawn_tcp_servers_for_topology(&topology);
        let (stats, health) = {
            let mut runtime = DistributedRuntime::new_tcp_workers(
                &design,
                "Counter",
                topology,
                parallel_options(),
                endpoints,
            )
            .unwrap();

            runtime.set_input(en, &[1, 1, 1, 1, 1]).unwrap();
            runtime.tick().unwrap();
            assert_eq!(runtime.get_signal(out).unwrap(), vec![1, 1, 1, 1, 1]);
            (runtime.stats(), runtime.health().unwrap())
        };
        join_tcp_servers(handles);

        assert_eq!(stats.execution_mode, RuntimeExecutionMode::Parallel);
        assert_eq!(stats.operations.tick.calls, 1);
        assert_eq!(stats.shards.len(), 2);
        for shard in stats.shards {
            assert_eq!(shard.operations.tick.calls, 1);
        }
        assert_eq!(health.total_lanes, 5);
        assert_eq!(health.shards.len(), 2);
        for shard in health.shards {
            assert_eq!(shard.status, RuntimeShardHealthStatus::Healthy);
            assert_eq!(shard.operations.unwrap().tick.calls, 1);
        }
    }

    #[test]
    fn tcp_runtime_snapshot_restore_round_trip() {
        let (design, en, we, addr, data, mem, count, read) = snapshot_design();
        let topology = two_cpu_topology();
        let (source_endpoints, source_handles) = spawn_tcp_servers_for_topology(&topology);
        let (snapshot, expected_count, expected_read, expected_memory) = {
            let mut source = DistributedRuntime::new_tcp_workers(
                &design,
                "SnapshotTop",
                topology.clone(),
                DistributedRuntimeOptions::default(),
                source_endpoints,
            )
            .unwrap();
            source.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
            source.set_input(we, &[1, 1, 1, 1, 1]).unwrap();
            source.set_input(addr, &[0, 1, 2, 3, 0]).unwrap();
            source.set_input(data, &[9, 11, 13, 15, 17]).unwrap();
            source.tick().unwrap();
            source.set_input(we, &[0, 0, 0, 0, 0]).unwrap();
            source.eval_combinational().unwrap();
            (
                source.snapshot().unwrap(),
                source.get_signal(count).unwrap(),
                source.get_signal(read).unwrap(),
                source.get_memory(mem).unwrap(),
            )
        };
        join_tcp_servers(source_handles);

        let (restore_endpoints, restore_handles) = spawn_tcp_servers_for_topology(&topology);
        let (restored_count, restored_read, restored_memory) = {
            let mut restored = DistributedRuntime::new_tcp_workers(
                &design,
                "SnapshotTop",
                topology,
                DistributedRuntimeOptions::default(),
                restore_endpoints,
            )
            .unwrap();
            restored.restore_snapshot(&snapshot).unwrap();
            (
                restored.get_signal(count).unwrap(),
                restored.get_signal(read).unwrap(),
                restored.get_memory(mem).unwrap(),
            )
        };
        join_tcp_servers(restore_handles);

        assert_eq!(restored_count, expected_count);
        assert_eq!(restored_read, expected_read);
        assert_eq!(restored_memory, expected_memory);
    }

    #[test]
    fn tcp_runtime_checkpoint_restore_round_trip() {
        let (design, en, we, addr, data, mem, count, read) = snapshot_design();
        let topology = two_cpu_topology();
        let (source_endpoints, source_handles) = spawn_tcp_servers_for_topology(&topology);
        let (checkpoint, expected_count, expected_read, expected_memory) = {
            let mut source = DistributedRuntime::new_tcp_workers(
                &design,
                "SnapshotTop",
                topology.clone(),
                DistributedRuntimeOptions::default(),
                source_endpoints.clone(),
            )
            .unwrap();
            source.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
            source.set_input(we, &[1, 1, 1, 1, 1]).unwrap();
            source.set_input(addr, &[0, 1, 2, 3, 0]).unwrap();
            source.set_input(data, &[9, 11, 13, 15, 17]).unwrap();
            source.tick().unwrap();
            source.set_input(we, &[0, 0, 0, 0, 0]).unwrap();
            source.eval_combinational().unwrap();
            (
                source
                    .checkpoint_with_tcp_endpoints(&source_endpoints)
                    .unwrap(),
                source.get_signal(count).unwrap(),
                source.get_signal(read).unwrap(),
                source.get_memory(mem).unwrap(),
            )
        };
        join_tcp_servers(source_handles);
        assert_eq!(checkpoint.tcp_endpoint_map().unwrap(), source_endpoints);

        let (restore_endpoints, restore_handles) = spawn_tcp_servers_for_topology(&topology);
        let (restored_count, restored_read, restored_memory) = {
            let mut restored = DistributedRuntime::new_tcp_workers(
                &design,
                "SnapshotTop",
                topology,
                DistributedRuntimeOptions::default(),
                restore_endpoints,
            )
            .unwrap();
            restored.restore_checkpoint(&checkpoint).unwrap();
            (
                restored.get_signal(count).unwrap(),
                restored.get_signal(read).unwrap(),
                restored.get_memory(mem).unwrap(),
            )
        };
        join_tcp_servers(restore_handles);

        assert_eq!(restored_count, expected_count);
        assert_eq!(restored_read, expected_read);
        assert_eq!(restored_memory, expected_memory);
    }

    #[test]
    fn tcp_runtime_recovers_workers_from_checkpoint() {
        let (design, en, we, addr, data, mem, count, read) = snapshot_design();
        let topology = two_cpu_topology();
        let (initial_endpoints, initial_handles) = spawn_tcp_servers_for_topology(&topology);
        let mut runtime = DistributedRuntime::new_tcp_workers(
            &design,
            "SnapshotTop",
            topology.clone(),
            DistributedRuntimeOptions::default(),
            initial_endpoints.clone(),
        )
        .unwrap();
        runtime.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
        runtime.set_input(we, &[1, 1, 1, 1, 1]).unwrap();
        runtime.set_input(addr, &[0, 1, 2, 3, 0]).unwrap();
        runtime.set_input(data, &[9, 11, 13, 15, 17]).unwrap();
        runtime.tick().unwrap();
        runtime.set_input(we, &[0, 0, 0, 0, 0]).unwrap();
        runtime.eval_combinational().unwrap();
        let checkpoint = runtime
            .checkpoint_with_tcp_endpoints(&initial_endpoints)
            .unwrap();
        let expected_count = runtime.get_signal(count).unwrap();
        let expected_read = runtime.get_signal(read).unwrap();
        let expected_memory = runtime.get_memory(mem).unwrap();
        let top_tick_calls = runtime.stats().operations.tick.calls;

        let (recovery_endpoints, recovery_handles) = spawn_tcp_servers_for_topology(&topology);
        let report = runtime
            .recover_tcp_workers_from_checkpoint(&design, &checkpoint, recovery_endpoints)
            .unwrap();
        join_tcp_servers(initial_handles);

        assert_eq!(
            report.recovered_workers,
            vec!["cpu-a".to_string(), "cpu-b".to_string()]
        );
        assert_eq!(runtime.get_signal(count).unwrap(), expected_count);
        assert_eq!(runtime.get_signal(read).unwrap(), expected_read);
        assert_eq!(runtime.get_memory(mem).unwrap(), expected_memory);
        let stats = runtime.stats();
        assert_eq!(stats.operations.tick.calls, top_tick_calls);
        for shard in stats.shards {
            assert_operation_stats_zero(shard.operations);
        }

        let mut direct = DistributedRuntime::new(&design, "SnapshotTop", topology).unwrap();
        direct.restore_checkpoint(&checkpoint).unwrap();
        for recovered in [&mut runtime, &mut direct] {
            recovered.set_input(en, &[1, 1, 1, 1, 1]).unwrap();
            recovered.tick().unwrap();
        }
        assert_eq!(
            runtime.get_signal(count).unwrap(),
            direct.get_signal(count).unwrap()
        );
        assert_eq!(
            runtime.get_signal(read).unwrap(),
            direct.get_signal(read).unwrap()
        );
        assert_eq!(
            runtime.get_memory(mem).unwrap(),
            direct.get_memory(mem).unwrap()
        );

        drop(runtime);
        join_tcp_servers(recovery_handles);
    }

    #[test]
    fn tcp_runtime_recovery_validates_checkpoint_design_and_endpoints() {
        let (design, en, _, _, _, _, _, _) = snapshot_design();
        let topology = two_cpu_topology();
        let (initial_endpoints, initial_handles) = spawn_tcp_servers_for_topology(&topology);
        let mut runtime = DistributedRuntime::new_tcp_workers(
            &design,
            "SnapshotTop",
            topology.clone(),
            DistributedRuntimeOptions::default(),
            initial_endpoints.clone(),
        )
        .unwrap();
        runtime.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
        runtime.tick().unwrap();
        let checkpoint = runtime
            .checkpoint_with_tcp_endpoints(&initial_endpoints)
            .unwrap();

        let mut missing = checkpoint.tcp_endpoint_map().unwrap();
        missing.remove("cpu-b");
        let err = runtime
            .recover_tcp_workers_from_checkpoint(&design, &checkpoint, missing)
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_TCP_ENDPOINT");

        let err = runtime
            .recover_tcp_workers_from_checkpoint(
                &design,
                &checkpoint,
                HashMap::from([
                    ("cpu-a".to_string(), "127.0.0.1:1".parse().unwrap()),
                    ("cpu-b".to_string(), "127.0.0.1:2".parse().unwrap()),
                    ("unknown".to_string(), "127.0.0.1:3".parse().unwrap()),
                ]),
            )
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_TCP_ENDPOINT");

        let (other_design, _, _) = counter_design();
        let err = runtime
            .recover_tcp_workers_from_checkpoint(
                &other_design,
                &checkpoint,
                checkpoint.tcp_endpoint_map().unwrap(),
            )
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_CHECKPOINT_PROGRAM");

        let mut bad = checkpoint.clone();
        bad.snapshot.shards[0].values.pop();
        let err = runtime
            .recover_tcp_workers_from_checkpoint(
                &design,
                &bad,
                checkpoint.tcp_endpoint_map().unwrap(),
            )
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_SNAPSHOT_STORAGE");

        drop(runtime);
        join_tcp_servers(initial_handles);
    }

    #[test]
    fn tcp_gpu_runtime_smoke_when_adapter_exists() {
        let (design, en, out) = counter_design();
        let topology = RuntimeTopology::local_heterogeneous(0, 2, GpuBatchOptions::default());
        let (endpoints, handles) = spawn_tcp_servers_for_topology(&topology);
        let result = {
            let Ok(mut runtime) = DistributedRuntime::new_tcp_workers(
                &design,
                "Counter",
                topology,
                parallel_options(),
                endpoints,
            ) else {
                join_tcp_servers(handles);
                return;
            };
            runtime.set_input(en, &[1, 0]).unwrap();
            runtime.tick().unwrap();
            runtime.get_signal(out).unwrap()
        };
        join_tcp_servers(handles);

        assert_eq!(result, vec![1, 0]);
    }

    #[test]
    fn local_cpu_runtime_ticks_all_lanes() {
        let (design, en, out) = counter_design();
        let mut runtime = DistributedRuntime::local_cpu(&design, "Counter", 4).unwrap();

        runtime.set_input(en, &[1, 0, 1, 1]).unwrap();
        runtime.tick().unwrap();
        assert_eq!(runtime.get_signal(out).unwrap(), vec![1, 0, 1, 1]);

        runtime.set_input(en, &[1, 1, 0, 1]).unwrap();
        runtime.tick_many(2).unwrap();
        assert_eq!(runtime.get_signal(out).unwrap(), vec![3, 2, 1, 3]);
    }

    #[test]
    fn topology_shards_preserve_global_lane_order() {
        let (design, en, out) = counter_design();
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("cpu-a", 2).on_node("node-a"));
        topology.push(RuntimeWorker::local_cpu("cpu-b", 3).on_node("node-b"));
        let mut runtime = DistributedRuntime::new(&design, "Counter", topology).unwrap();

        assert_eq!(runtime.total_lanes(), 5);
        assert_eq!(
            runtime
                .shard_plan()
                .iter()
                .map(|shard| (shard.worker_id.as_str(), shard.start_lane, shard.lanes))
                .collect::<Vec<_>>(),
            vec![("cpu-a", 0, 2), ("cpu-b", 2, 3)]
        );

        runtime.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
        runtime.tick().unwrap();
        assert_eq!(runtime.get_signal(out).unwrap(), vec![1, 0, 1, 0, 1]);
    }

    #[test]
    fn parallel_cpu_shards_match_serial_counter_execution() {
        let (design, en, out) = counter_design();
        let topology = two_cpu_topology();
        let mut serial = DistributedRuntime::new(&design, "Counter", topology.clone()).unwrap();
        let mut parallel =
            DistributedRuntime::new_with_options(&design, "Counter", topology, parallel_options())
                .unwrap();

        for runtime in [&mut serial, &mut parallel] {
            runtime.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
            runtime.tick().unwrap();
            runtime.set_input(en, &[1, 1, 0, 1, 1]).unwrap();
            runtime.tick_many(2).unwrap();
        }

        assert_eq!(
            parallel.get_signal(out).unwrap(),
            serial.get_signal(out).unwrap()
        );
        assert_eq!(parallel.shard_plan(), serial.shard_plan());
    }

    #[test]
    fn serial_runtime_stats_count_operations_and_shards() {
        let (design, en, out) = counter_design();
        let mut runtime = DistributedRuntime::new(&design, "Counter", two_cpu_topology()).unwrap();

        runtime.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
        runtime.tick().unwrap();
        runtime.tick_many(2).unwrap();
        runtime.eval_combinational().unwrap();
        assert_eq!(runtime.get_signal(out).unwrap(), vec![3, 0, 3, 0, 3]);

        let stats = runtime.stats();
        assert_eq!(stats.execution_mode, RuntimeExecutionMode::Serial);
        assert_eq!(stats.operations.tick.calls, 1);
        assert_eq!(stats.operations.tick_many.calls, 1);
        assert_eq!(stats.operations.eval_combinational.calls, 1);
        for shard in stats.shards {
            assert_eq!(shard.operations.tick.calls, 1);
            assert_eq!(shard.operations.tick_many.calls, 1);
            assert_eq!(shard.operations.eval_combinational.calls, 1);
        }
    }

    #[test]
    fn topology_shards_preserve_memory_global_lane_order() {
        let (design, _, addr, _, mem, read) = memory_design();
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("cpu-a", 2).on_node("node-a"));
        topology.push(RuntimeWorker::local_cpu("cpu-b", 3).on_node("node-b"));
        let mut runtime = DistributedRuntime::new(&design, "MemoryTop", topology).unwrap();
        let memory = vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
            vec![17, 18, 19, 20],
        ];

        runtime.set_memory(mem, &memory).unwrap();
        assert_eq!(runtime.get_memory(mem).unwrap(), memory);

        runtime.set_input(addr, &[0, 1, 2, 3, 0]).unwrap();
        runtime.eval_combinational().unwrap();
        assert_eq!(runtime.get_signal(read).unwrap(), vec![1, 6, 11, 16, 17]);
    }

    #[test]
    fn runtime_memory_writes_route_through_shards() {
        let (design, we, addr, data, mem, read) = memory_design();
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("cpu-a", 1));
        topology.push(RuntimeWorker::local_cpu("cpu-b", 2));
        let mut runtime = DistributedRuntime::new(&design, "MemoryTop", topology).unwrap();

        runtime.set_input(we, &[1, 1, 1]).unwrap();
        runtime.set_input(addr, &[0, 1, 2]).unwrap();
        runtime.set_input(data, &[9, 11, 13]).unwrap();
        runtime.tick().unwrap();

        runtime.set_input(we, &[0, 0, 0]).unwrap();
        runtime.eval_combinational().unwrap();
        assert_eq!(runtime.get_signal(read).unwrap(), vec![9, 11, 13]);
        assert_eq!(
            runtime.get_memory(mem).unwrap(),
            vec![vec![9, 0, 0, 0], vec![0, 11, 0, 0], vec![0, 0, 13, 0]]
        );
    }

    #[test]
    fn parallel_cpu_shards_match_serial_memory_execution() {
        let (design, we, addr, data, mem, read) = memory_design();
        let topology = two_cpu_topology();
        let mut serial = DistributedRuntime::new(&design, "MemoryTop", topology.clone()).unwrap();
        let mut parallel = DistributedRuntime::new_with_options(
            &design,
            "MemoryTop",
            topology,
            parallel_options(),
        )
        .unwrap();
        let seeded = vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
            vec![17, 18, 19, 20],
        ];

        for runtime in [&mut serial, &mut parallel] {
            runtime.set_memory(mem, &seeded).unwrap();
            runtime.set_input(we, &[1, 1, 0, 0, 1]).unwrap();
            runtime.set_input(addr, &[0, 1, 2, 3, 0]).unwrap();
            runtime.set_input(data, &[21, 22, 23, 24, 25]).unwrap();
            runtime.tick().unwrap();
            runtime.set_input(we, &[0, 0, 0, 0, 0]).unwrap();
            runtime.eval_combinational().unwrap();
        }

        assert_eq!(
            parallel.get_signal(read).unwrap(),
            serial.get_signal(read).unwrap()
        );
        assert_eq!(
            parallel.get_memory(mem).unwrap(),
            serial.get_memory(mem).unwrap()
        );
    }

    #[test]
    fn parallel_runtime_stats_count_operations_and_report_mode() {
        let (design, we, addr, data, _, _) = memory_design();
        let mut runtime = DistributedRuntime::new_with_options(
            &design,
            "MemoryTop",
            two_cpu_topology(),
            parallel_options(),
        )
        .unwrap();

        runtime.set_input(we, &[1, 1, 0, 0, 1]).unwrap();
        runtime.set_input(addr, &[0, 1, 2, 3, 0]).unwrap();
        runtime.set_input(data, &[21, 22, 23, 24, 25]).unwrap();
        runtime.tick().unwrap();
        runtime.tick_many(0).unwrap();
        runtime.eval_combinational().unwrap();

        let stats = runtime.stats();
        assert_eq!(stats.execution_mode, RuntimeExecutionMode::Parallel);
        assert_eq!(stats.operations.tick.calls, 1);
        assert_eq!(stats.operations.tick_many.calls, 1);
        assert_eq!(stats.operations.eval_combinational.calls, 1);
        assert_eq!(stats.shards.len(), 2);
        for shard in stats.shards {
            assert_eq!(shard.operations.tick.calls, 1);
            assert_eq!(shard.operations.tick_many.calls, 1);
            assert_eq!(shard.operations.eval_combinational.calls, 1);
        }
    }

    #[test]
    fn reset_stats_keeps_simulation_state() {
        let (design, en, out) = counter_design();
        let mut runtime = DistributedRuntime::new_with_options(
            &design,
            "Counter",
            two_cpu_topology(),
            parallel_options(),
        )
        .unwrap();

        runtime.set_input(en, &[1, 1, 1, 1, 1]).unwrap();
        runtime.tick().unwrap();
        assert_eq!(runtime.get_signal(out).unwrap(), vec![1, 1, 1, 1, 1]);
        assert_eq!(runtime.stats().operations.tick.calls, 1);

        runtime.reset_stats();
        let stats = runtime.stats();
        assert_operation_stats_zero(stats.operations);
        for shard in stats.shards {
            assert_operation_stats_zero(shard.operations);
        }
        assert_eq!(runtime.get_signal(out).unwrap(), vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn parallel_tick_many_zero_steps_keeps_state() {
        let (design, en, out) = counter_design();
        let mut runtime = DistributedRuntime::new_with_options(
            &design,
            "Counter",
            two_cpu_topology(),
            parallel_options(),
        )
        .unwrap();

        runtime.set_input(en, &[1, 1, 1, 1, 1]).unwrap();
        runtime.tick_many(0).unwrap();
        assert_eq!(runtime.get_signal(out).unwrap(), vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn runtime_memory_limb_apis_support_wide_words() {
        let mut design = Design::new();
        let mem;
        {
            let mut m = design.module("WideRuntimeMem");
            mem = m.mem("mem", 1, uint(160), 2);
        }
        let mut runtime = DistributedRuntime::local_cpu(&design, "WideRuntimeMem", 2).unwrap();
        let values = vec![
            vec![vec![1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10]],
            vec![vec![11, 12, 13, 14, 15], vec![16, 17, 18, 19, 20]],
        ];

        runtime.set_memory_limbs(mem, &values).unwrap();
        assert_eq!(runtime.get_memory_limbs(mem).unwrap(), values);

        let err = runtime
            .set_memory(mem, &[vec![0, 1], vec![2, 3]])
            .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_WIDE_MEMORY");
        let err = runtime.get_memory(mem).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_WIDE_MEMORY");
    }

    #[test]
    fn gpu_runtime_memory_apis_work_when_adapter_exists() {
        let (design, _, _, _, mem, _) = memory_design();
        let topology = RuntimeTopology::local_heterogeneous(0, 2, GpuBatchOptions::default());
        let Ok(mut runtime) = DistributedRuntime::new(&design, "MemoryTop", topology) else {
            return;
        };

        runtime
            .set_memory(mem, &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]])
            .unwrap();
        assert_eq!(
            runtime.get_memory(mem).unwrap(),
            vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]]
        );
    }

    #[test]
    fn parallel_cpu_gpu_shards_work_when_adapter_exists() {
        let (design, en, out) = counter_design();
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("cpu0", 1));
        topology.push(RuntimeWorker::local_gpu(
            "gpu0",
            2,
            GpuBatchOptions::default(),
        ));
        let Ok(mut runtime) =
            DistributedRuntime::new_with_options(&design, "Counter", topology, parallel_options())
        else {
            return;
        };

        runtime.set_input(en, &[1, 0, 1]).unwrap();
        runtime.tick().unwrap();
        assert_eq!(runtime.get_signal(out).unwrap(), vec![1, 0, 1]);
    }

    #[test]
    fn cpu_gpu_runtime_stats_report_shard_metadata_when_adapter_exists() {
        let (design, en, _) = counter_design();
        let mut topology = RuntimeTopology::new();
        topology.push(RuntimeWorker::local_cpu("cpu0", 1).on_node("host-a"));
        topology.push(
            RuntimeWorker::local_gpu("gpu0", 2, GpuBatchOptions::default()).on_node("host-b"),
        );
        let Ok(mut runtime) =
            DistributedRuntime::new_with_options(&design, "Counter", topology, parallel_options())
        else {
            return;
        };

        runtime.set_input(en, &[1, 0, 1]).unwrap();
        runtime.tick().unwrap();
        let stats = runtime.stats();
        assert_eq!(stats.execution_mode, RuntimeExecutionMode::Parallel);
        assert_eq!(stats.total_lanes, 3);
        assert_eq!(stats.shards.len(), 2);
        assert_eq!(stats.shards[0].shard.worker_id, "cpu0");
        assert_eq!(stats.shards[0].shard.node, "host-a");
        assert_eq!(stats.shards[0].shard.backend, RuntimeBackend::PackedCpu);
        assert_eq!(stats.shards[0].shard.lanes, 1);
        assert_eq!(stats.shards[1].shard.worker_id, "gpu0");
        assert_eq!(stats.shards[1].shard.node, "host-b");
        assert!(matches!(
            stats.shards[1].shard.backend,
            RuntimeBackend::Gpu(_)
        ));
        assert_eq!(stats.shards[1].shard.lanes, 2);
        assert_eq!(stats.shards[0].operations.tick.calls, 1);
        assert_eq!(stats.shards[1].operations.tick.calls, 1);
    }

    #[test]
    fn rejects_empty_topology_and_wrong_lane_count() {
        let (design, en, _) = counter_design();

        assert!(DistributedRuntime::new(&design, "Counter", RuntimeTopology::new()).is_err());

        let mut runtime = DistributedRuntime::local_cpu(&design, "Counter", 2).unwrap();
        let err = runtime.set_input(en, &[1]).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_LANE_VALUES");
    }

    #[test]
    fn parallel_runtime_keeps_lane_count_diagnostics() {
        let (design, en, _) = counter_design();
        let mut runtime = DistributedRuntime::new_with_options(
            &design,
            "Counter",
            two_cpu_topology(),
            parallel_options(),
        )
        .unwrap();

        let err = runtime.set_input(en, &[1, 0, 1, 0]).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_RUNTIME_LANE_VALUES");
    }
}
