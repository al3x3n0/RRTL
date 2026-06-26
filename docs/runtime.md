# Distributed Heterogeneous Runtime


`rrtl-runtime` builds on compiled RRTL, the packed simulation IR, and GPU
simulation to analyze and run AI-scale hardware workloads. It has two related
surfaces:

- Structural partition sessions recursively cut a compiled hierarchy into
  worker payloads, deploy those payloads, run worker-level structural actions,
  and emit JSON telemetry for huge single-design splits.
- Lane-sharded distributed runtimes split huge independent simulation batches
  across CPU, GPU, loopback, TCP, and supervised worker-process transports
  while preserving global lane order.

Structural partition sessions start from a deterministic JSON-ready planning
bundle and launch plan:

```rust
use rrtl::runtime::{
    plan_runtime_partition_bundle, plan_runtime_partition_launch, RuntimePartitionConfig,
    RuntimeTopology, RuntimeWorker,
};

let mut topology = RuntimeTopology::new();
topology.push(RuntimeWorker::local_cpu("cpu0", 1).on_node("host-a"));
topology.push(RuntimeWorker::local_cpu("cpu1", 1).on_node("host-b"));

let bundle = plan_runtime_partition_bundle(
    &design,
    "Top",
    RuntimePartitionConfig {
        target_partitions: 8,
    },
    &topology,
)
.unwrap();
let launch = plan_runtime_partition_launch(&bundle).unwrap();
println!(
    "partitions={} workers={} routes={} diagnostics={}",
    bundle.partition_plan.partitions.len(),
    launch.workers.len(),
    launch.routes.len(),
    launch.diagnostics.len(),
);
```

The launch plan can then be deployed through local, loopback, or TCP worker
services. Scripted partition-session runs execute an ordered sequence of
structural actions and can write telemetry snapshots at action-count cadence:

```rust
use std::{
    fs::File,
    sync::{Arc, Mutex},
};

use rrtl::runtime::{
    LocalRuntimeWorkerService, RuntimePartitionSession, RuntimePartitionSessionRunScript,
    RuntimePartitionWorkerActionKind,
};

let service = Arc::new(Mutex::new(LocalRuntimeWorkerService::new()));
let mut session = RuntimePartitionSession::deploy_loopback(launch, service.clone()).unwrap();
let script = RuntimePartitionSessionRunScript::every_actions(
    vec![
        RuntimePartitionWorkerActionKind::EvalCombinational,
        RuntimePartitionWorkerActionKind::TickMany(1024),
    ],
    1,
);

let report = session
    .run_loopback_script(&script, service, |event, telemetry| {
        let path = format!("partition-session-action-{}.json", event.completed_actions);
        let mut file = File::create(path).unwrap();
        telemetry.write_json(&mut file)
    })
    .unwrap();
println!(
    "actions={} telemetry={} workers={}",
    report.completed_actions,
    report.telemetry_emitted,
    report.action_summary.worker_count,
);
```

Current structural actions are orchestration-level worker actions. They record
deployment health, per-worker action counters, aggregate summaries, diagnostics,
and telemetry, but they do not yet model cross-partition signal-value transfer
semantics for a full partitioned RTL simulation step.

The runtime can also partition large independent lane batches across workers:

```rust
use rrtl::runtime::{
    DistributedRuntime, DistributedRuntimeOptions, RuntimeExecutionMode,
    RuntimeTopology, RuntimeWorker,
};
use rrtl_gpu_sim::GpuBatchOptions;

let mut topology = RuntimeTopology::new();
topology.push(RuntimeWorker::local_cpu("cpu0", 4096).on_node("host-a"));
topology.push(RuntimeWorker::local_gpu(
    "gpu0",
    65536,
    GpuBatchOptions::default(),
).on_node("host-a"));

let mut runtime = DistributedRuntime::new_with_options(
    &design,
    "Counter",
    topology.clone(),
    DistributedRuntimeOptions {
        execution_mode: RuntimeExecutionMode::Parallel,
    },
)
.unwrap();
runtime.set_input(en, &vec![1; runtime.total_lanes()]).unwrap();
runtime.tick_many(1024).unwrap();
let counts = runtime.get_signal(out).unwrap();
let stats = runtime.stats();
runtime.reset_stats();
```

Health checks are explicit, non-mutating calls that report shard reachability,
initialization state, and per-shard operation counters:

```rust
let health = runtime.health().unwrap();
for shard in health.shards {
    assert!(matches!(shard.status, RuntimeShardHealthStatus::Healthy));
}
```

The runtime executes workers in-process, with serial execution by default and
opt-in parallel shard dispatch through `DistributedRuntimeOptions`. Its shard
plan and stats snapshots preserve worker IDs, node placement, backend choice,
global start lane, lane count, operation counters, and per-shard timings so
future remote transports can reuse the same scheduling contract.

Long simulations can checkpoint exact packed shard state and restore it into a
runtime with the same module shape and lane partition:

```rust
let snapshot = runtime.snapshot().unwrap();

let mut resumed = DistributedRuntime::new_with_options(
    &design,
    "Counter",
    topology.clone(),
    DistributedRuntimeOptions {
        execution_mode: RuntimeExecutionMode::Parallel,
    },
)
.unwrap();
resumed.restore_snapshot(&snapshot).unwrap();
```

For durable restarts, save a versioned checkpoint manifest that includes the
runtime topology, optional TCP worker endpoints, and the snapshot state:

```rust
let checkpoint = runtime.checkpoint_with_tcp_endpoints(&endpoints).unwrap();
let mut file = std::fs::File::create("runtime-checkpoint.json").unwrap();
checkpoint.write_json(&mut file).unwrap();

let mut file = std::fs::File::open("runtime-checkpoint.json").unwrap();
let checkpoint = RuntimeCheckpoint::read_json(&mut file).unwrap();
let endpoints = checkpoint.tcp_endpoint_map().unwrap();
let mut resumed = DistributedRuntime::new_tcp_workers(
    &design,
    &checkpoint.module_name,
    checkpoint.topology.clone(),
    DistributedRuntimeOptions::default(),
    endpoints,
)
.unwrap();
resumed.restore_checkpoint(&checkpoint).unwrap();
```

For long runs, checkpoint cadence helpers can emit durable recovery points while
the runtime advances in `tick_many` chunks. The callback owns storage policy,
so it can write JSON, upload manifests, or apply retention rules:

```rust
let cadence = RuntimeCheckpointCadence {
    every_steps: 10_000,
    include_initial: true,
    include_final: true,
};
let report = runtime
    .tick_many_with_tcp_checkpoints(1_000_000, cadence, &endpoints, |event, checkpoint| {
        let path = format!("checkpoint-step-{}.json", event.completed_steps);
        let mut file = std::fs::File::create(path).unwrap();
        checkpoint.write_json(&mut file)
    })
    .unwrap();
assert_eq!(report.completed_steps, 1_000_000);
```

If TCP workers are restarted, recover an existing runtime by reconnecting to the
new endpoints and restoring every shard from the checkpoint. Recovery is
explicit and checkpoint-bound; a failed `tick` or `tick_many` is not replayed
automatically:

```rust
let new_endpoints = checkpoint.tcp_endpoint_map().unwrap();
let report = runtime
    .recover_tcp_workers_from_checkpoint(&design, &checkpoint, new_endpoints)
    .unwrap();
assert_eq!(report.recovered_workers.len(), checkpoint.topology.workers().len());
```

The same runtime API can also execute through the transport-neutral worker
protocol using a loopback worker service. This keeps execution in-process while
exercising the request/response boundary intended for future remote adapters:

```rust
let mut runtime = DistributedRuntime::new_loopback_workers(
    &design,
    "Counter",
    topology,
    DistributedRuntimeOptions {
        execution_mode: RuntimeExecutionMode::Parallel,
    },
)
.unwrap();
runtime.tick_many(1024).unwrap();
```

The worker protocol also has a synchronous TCP transport with JSON-line framing.
Start one server per shard endpoint, then connect the runtime to those already
listening workers:

```sh
cargo run -p rrtl-runtime --bin rrtl-runtime-worker -- --bind 127.0.0.1:0
# stdout: {"addr":"127.0.0.1:54321"}
```

```rust
use std::collections::HashMap;

let mut endpoints = HashMap::new();
endpoints.insert("cpu0".to_string(), "127.0.0.1:54321".parse().unwrap());
let mut runtime = DistributedRuntime::new_tcp_workers(
    &design,
    "Counter",
    RuntimeTopology::local_cpu(4096),
    DistributedRuntimeOptions::default(),
    endpoints,
)
.unwrap();
```

Use `--once` or `--max-connections <n>` when a worker process should exit
after serving a bounded number of runtime connections.

For local process orchestration, spawn one worker subprocess per topology worker
and use the collected endpoint map directly:

```rust
use rrtl::runtime::{
    TcpRuntimeWorkerProcessConfig, TcpRuntimeWorkerProcessSet,
};

let mut process_config =
    TcpRuntimeWorkerProcessConfig::new("target/debug/rrtl-runtime-worker");
process_config.max_connections = Some(1);
let mut workers = TcpRuntimeWorkerProcessSet::spawn(&topology, &process_config).unwrap();
let process_health = workers.health().unwrap();

let mut runtime = DistributedRuntime::new_tcp_workers(
    &design,
    "Counter",
    topology,
    DistributedRuntimeOptions::default(),
    workers.endpoints().clone(),
)
.unwrap();
runtime.tick_many(1024).unwrap();
drop(runtime);
workers.wait_all().unwrap();
```

For local TCP workers that should be restarted from the latest checkpoint under
caller control, use the supervisor. Recovery is explicit: health reports process
and runtime state, and `recover_from_latest_checkpoint` restarts every worker
process before restoring all shards:

```rust
use rrtl::runtime::{
    RuntimeCheckpointCadence, TcpRuntimeSupervisor, TcpRuntimeSupervisorConfig,
    TcpRuntimeWorkerProcessConfig,
};

let supervisor_config = TcpRuntimeSupervisorConfig::new(
    TcpRuntimeWorkerProcessConfig::new("target/debug/rrtl-runtime-worker"),
);
let mut supervisor =
    TcpRuntimeSupervisor::spawn(&design, "Counter", topology, supervisor_config).unwrap();

let cadence = RuntimeCheckpointCadence::every_steps(10_000);
supervisor
    .tick_many_with_checkpoints(100_000, cadence, |event, checkpoint| {
        let path = format!("supervisor-step-{}.json", event.completed_steps);
        let mut file = std::fs::File::create(path).unwrap();
        checkpoint.write_json(&mut file)
    })
    .unwrap();

let health = supervisor.health().unwrap();
if health.runtime_error.is_some() || health.processes.iter().any(|worker| !worker.running) {
    supervisor.recover_from_latest_checkpoint().unwrap();
}
```

Supervisor telemetry reports are JSON-ready and include process health, runtime
stats, best-effort runtime health, current endpoints, recovery metadata, and the
latest full checkpoint snapshot when one has been stored:

```rust
let telemetry = supervisor.telemetry().unwrap();
let mut file = std::fs::File::create("runtime-telemetry.json").unwrap();
telemetry.write_json(&mut file).unwrap();
```

Callers can benchmark explicit topology candidates and select the fastest
runtime configuration without mutating an existing simulation:

```rust
use rrtl::runtime::{
    recommend_runtime_topology, DistributedRuntimeOptions, RuntimeAutotuneCandidate,
    RuntimeAutotuneConfig, RuntimeExecutionMode, RuntimeTopology, RuntimeWorker,
};
use rrtl_gpu_sim::GpuBatchOptions;

let mut mixed_topology = RuntimeTopology::new();
mixed_topology.push(RuntimeWorker::local_cpu("cpu0", 4096));
mixed_topology.push(RuntimeWorker::local_gpu(
    "gpu0",
    65536,
    GpuBatchOptions::default(),
));

let report = recommend_runtime_topology(
    &design,
    "Counter",
    vec![
        RuntimeAutotuneCandidate {
            name: "serial-cpu".to_string(),
            topology: RuntimeTopology::local_cpu(4096),
            options: DistributedRuntimeOptions::default(),
        },
        RuntimeAutotuneCandidate {
            name: "parallel-mixed".to_string(),
            topology: mixed_topology,
            options: DistributedRuntimeOptions {
                execution_mode: RuntimeExecutionMode::Parallel,
            },
        },
    ],
    RuntimeAutotuneConfig::default(),
)
.unwrap();
let best = &report.candidates[report.best_index];
```

Autotune can also use declarative stimulus. Setup values are applied before
warmup; nonempty step vectors are rotated during warmup and measurement, with
one `tick()` per measured cycle:

```rust
use rrtl::runtime::{
    RuntimeAutotuneStimulus, RuntimeSignalValue, RuntimeStimulusSetup,
    RuntimeStimulusStep,
};

let report = recommend_runtime_topology(
    &design,
    "Counter",
    candidates,
    RuntimeAutotuneConfig {
        warmup_steps: 8,
        measure_steps: 256,
        stimulus: Some(RuntimeAutotuneStimulus {
            setup: RuntimeStimulusSetup::default(),
            steps: vec![
                RuntimeStimulusStep {
                    inputs: vec![RuntimeSignalValue {
                        signal: en,
                        lane_values: vec![1; lanes],
                    }],
                    input_limbs: Vec::new(),
                },
                RuntimeStimulusStep {
                    inputs: vec![RuntimeSignalValue {
                        signal: en,
                        lane_values: vec![0; lanes],
                    }],
                    input_limbs: Vec::new(),
                },
            ],
        }),
    },
)
.unwrap();
```
