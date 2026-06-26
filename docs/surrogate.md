# RTL-Derived Surrogates

The surrogate scaffold ingests instrumentation event corpora and validates
generic event predictors. `cache_miss` remains the default built-in profile, and
`stall_event` is available as a second profile to exercise the same path with a
different target, feature set, and label. Lanes are batching/report metadata:
they are preserved in event corpora and shadow reports, but are not model input
tensors.

Generate a synthetic lane-aware corpus, train the rule artifact, then validate
and run it through the Rust CLI:

```sh
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss generate \
  --out target/cache_miss/events.json --samples 32 --window 8 --seed 1 --lanes 4
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss train \
  --corpus target/cache_miss/events.json --out-dir target/cache_miss/model
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss validate \
  --manifest target/cache_miss/model/manifest.json --corpus target/cache_miss/events.json
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss shadow \
  --manifest target/cache_miss/model/manifest.json --corpus target/cache_miss/events.json
cargo run -q -p rrtl-pyrtl --bin pyrtl2rrtl -- surrogate policy-events \
  target/cache_miss/model/manifest.json target/cache_miss/events.json \
  --out target/cache_miss/policy.json
cargo run -q -p rrtl-pyrtl --bin pyrtl2rrtl -- surrogate plan-runtime-events \
  target/cache_miss/policy.json --worker cpu-a:2 --worker cpu-b:2 \
  --out target/cache_miss/runtime_plan.json
cargo run -q -p rrtl-pyrtl --bin pyrtl2rrtl -- surrogate run-fast-events \
  target/cache_miss/model/manifest.json target/cache_miss/events.json \
  target/cache_miss/runtime_plan.json --shadow-sample-stride 8 \
  --out target/cache_miss/fast_run.json
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss export-tensors \
  --corpus target/cache_miss/events.json --out target/cache_miss/tensor_bundle.json
PYTHONPATH=python python -m rrtl_surrogate.gnn_transformer event-bundle \
  target/cache_miss/tensor_bundle.json --out target/cache_miss/trainer_manifest.json
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss model-fast-golden \
  --corpus target/cache_miss/events.json --fast-run target/cache_miss/fast_run.json \
  --op-id cache0 --out target/cache_miss/cache0_golden.json
cargo run -q -p rrtl-pyrtl --bin pyrtl2rrtl -- surrogate attach-runtime-events \
  target/cache_miss/runtime_plan.json --worker cpu-a:2 --worker cpu-b:2 \
  --out target/cache_miss/runtime_attachment.json
cargo run -q -p rrtl-pyrtl --bin pyrtl2rrtl -- surrogate inspect-runtime-attachment \
  target/cache_miss/runtime_attachment.json \
  --out target/cache_miss/runtime_execution.json
```

Use `--target` to select a non-default event profile. The generated corpus,
tensor bundle, trainer manifest, surrogate manifest, policy reports, and FAST
golden all carry the selected target:

```sh
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss catalog-use-cases \
  --out target/surrogate_use_cases/catalog.json
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss generate \
  --target stall_event --out target/stall_event/events.json \
  --samples 32 --window 8 --seed 1 --lanes 4
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss train \
  --target stall_event --corpus target/stall_event/events.json \
  --out-dir target/stall_event/model
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss export-tensors \
  --target stall_event --corpus target/stall_event/events.json \
  --out target/stall_event/tensor_bundle.json
```

Use `--profile <profile.json>` to load a declarative event profile without
editing Python or Rust code. Profiles use
`rrtl-surrogate-event-profile-v1` and define the target, feature lists, label,
and a simple `linear_threshold` mock rule for rule-baseline artifacts:

```sh
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss catalog-use-cases \
  --profile examples/surrogate/profiles/stall_event.json \
  --out target/stall_event_profile/catalog.json
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss generate \
  --profile examples/surrogate/profiles/stall_event.json \
  --out target/stall_event_profile/events.json --samples 32 --window 8 --seed 1
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss train \
  --profile examples/surrogate/profiles/stall_event.json \
  --corpus target/stall_event_profile/events.json \
  --out-dir target/stall_event_profile/model
```

For the learned GNN/Transformer-style path, export an ONNX event predictor with
`train-learned` and run the same validation, policy, runtime-planning, and FAST
commands against the learned manifest. ONNX execution requires building the Rust
surrogate crate with the `onnx-ort` feature; without that feature, ONNX manifests
still validate structurally but execution reports a feature-required error.

```sh
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss train-learned \
  --corpus target/cache_miss/events.json --out-dir target/cache_miss/learned-model \
  --epochs 4
cargo run -q -p rrtl-pyrtl --features onnx-ort --bin pyrtl2rrtl -- surrogate validate-events \
  target/cache_miss/learned-model/manifest.json target/cache_miss/events.json \
  --out target/cache_miss/learned_validate.json
cargo run -q -p rrtl-pyrtl --features onnx-ort --bin pyrtl2rrtl -- surrogate shadow-events \
  target/cache_miss/learned-model/manifest.json target/cache_miss/events.json \
  --out target/cache_miss/learned_shadow.json
cargo run -q -p rrtl-pyrtl --features onnx-ort --bin pyrtl2rrtl -- surrogate policy-events \
  target/cache_miss/learned-model/manifest.json target/cache_miss/events.json \
  --out target/cache_miss/learned_policy.json
cargo run -q -p rrtl-pyrtl --features onnx-ort --bin pyrtl2rrtl -- surrogate plan-runtime-events \
  target/cache_miss/learned_policy.json --worker cpu-a:2 --worker cpu-b:2 \
  --out target/cache_miss/learned_runtime_plan.json
cargo run -q -p rrtl-pyrtl --features onnx-ort --bin pyrtl2rrtl -- surrogate run-fast-events \
  target/cache_miss/learned-model/manifest.json target/cache_miss/events.json \
  target/cache_miss/learned_runtime_plan.json --shadow-sample-stride 8 \
  --out target/cache_miss/learned_fast_run.json
```

To model the future instrumentation path, emit a corpus from a trace plus
feature mapping config:

```sh
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss emit \
  --trace examples/surrogate/cache_miss_trace.json \
  --config examples/surrogate/cache_miss_emitter_config.json \
  --out target/cache_miss/emitted_events.json
cargo run -q -p rrtl-pyrtl --bin pyrtl2rrtl -- surrogate inspect-events \
  target/cache_miss/emitted_events.json
```

RRTL-native instrumentation can feed the same corpus contract through the
normalized `rrtl-instrumentation-trace-v1` shape:

```sh
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss inspect-instrumentation \
  --trace examples/surrogate/cache_miss_instrumentation_trace.json \
  --config examples/surrogate/cache_miss_emitter_config.json \
  --out target/cache_miss/instrumentation_inspection.json
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss match-instrumentation-use-case \
  --trace examples/surrogate/cache_miss_instrumentation_trace.json \
  --config examples/surrogate/cache_miss_emitter_config.json \
  --target cache_miss \
  --out target/cache_miss/instrumentation_use_case_match.json
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss emit-instrumented \
  --trace examples/surrogate/cache_miss_instrumentation_trace.json \
  --config examples/surrogate/cache_miss_emitter_config.json \
  --out target/cache_miss/instrumented_events.json
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss instrumented-flow \
  --trace examples/surrogate/cache_miss_instrumentation_trace.json \
  --config examples/surrogate/cache_miss_emitter_config.json \
  --out-dir target/cache_miss/instrumented-flow
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss inspect-flow \
  --bundle target/cache_miss/instrumented-flow/flow_bundle.json
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss quality-gate \
  --bundle target/cache_miss/instrumented-flow/flow_bundle.json \
  --out target/cache_miss/instrumented-flow/quality_gate.json
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss package-runtime-handoff \
  --bundle target/cache_miss/instrumented-flow/flow_bundle.json \
  --out target/cache_miss/instrumented-flow/runtime_handoff.json
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss inspect-runtime-telemetry \
  --bundle target/cache_miss/instrumented-flow/flow_bundle.json \
  --telemetry target/cache_miss/runtime_telemetry.json
PYTHONPATH=python python -m rrtl_surrogate.train_cache_miss inspect-runtime-handoff-telemetry \
  --handoff target/cache_miss/instrumented-flow/runtime_handoff.json \
  --telemetry target/cache_miss/runtime_telemetry.json
```

`instrumented-flow` also writes `flow_bundle.json`, a compact
`rrtl-surrogate-flow-bundle-v1` handoff report with artifact paths, readiness
bits, counters, the selected use-case contract, warnings, and errors for
orchestration.
`catalog-use-cases` writes an `rrtl-surrogate-use-case-catalog-v1` report
derived from event profiles, so instrumentation and FAST orchestration can
discover targets such as `cache_miss`, `stall_event`, or project-local profile
files before selecting a surrogate.
`match-instrumentation-use-case` writes an
`rrtl-surrogate-instrumentation-use-case-match-v1` preflight report that checks
the emitter config target, feature names, label, and instrumentation
compatibility against the selected profile before training or FAST orchestration.
`quality-gate` writes an `rrtl-surrogate-quality-gate-v1` report that checks
validation accuracy, shadow accuracy, fail-closed/shadow-failed counters,
model-FAST status, and runtime readiness against strict default thresholds or a
project-local thresholds JSON file.
`package-runtime-handoff` writes an `rrtl-surrogate-runtime-handoff-v1` manifest
that references the accepted manifest, runtime plan, runtime attachment,
runtime execution, quality gate, and model-FAST report for FAST/runtime
orchestration.
`inspect-runtime-telemetry` writes an
`rrtl-surrogate-runtime-telemetry-gate-v1` report that fails closed when runtime
telemetry is missing `surrogate_execution`, is not ready, or drifts from the
bundle/runtime execution counters.
`inspect-runtime-handoff-telemetry` writes an
`rrtl-surrogate-runtime-handoff-telemetry-gate-v1` report with the same runtime
checks against the accepted `runtime_handoff.json` package, so FAST/runtime
orchestration can validate live telemetry without reading the full flow bundle.

`shadow` exits nonzero when predictions diverge, while still writing a JSON
report with aggregate metrics, `total_lanes`, per-lane summaries, per-sample
lane IDs, and provenance.
`policy-events` applies the manifest fallback policy to each predicted event,
`plan-runtime-events` maps accepted samples onto runtime workers, and
`run-fast-events` emits an execution-facing report with per-worker/per-lane
prediction, fallback, fail-closed, and shadow counters. Optional
`--shadow-sample-stride/--shadow-sample-offset` marks deterministic event
samples for exact follow-up without changing the policy result.
`attach-runtime-events` validates the plan into the same runtime telemetry
attachment shape used by transaction-kernel surrogates.
`inspect-runtime-attachment` emits the execution-facing readiness report that
the distributed runtime also publishes in telemetry. For event plans it includes
per-worker, per-lane, and per-item `sample_id`/`target`/prediction provenance
summaries. This is an overlay report: surrogate attachments are counted and
surfaced to fast-sim orchestration, while the exact RTL simulator remains the
state owner.
When the manifest policy is `shadow_compare`, runtime plans preserve compact
shadow metadata such as predicted/expected event values or GEMM mismatch
counters. Attachments and execution reports count shadow-compared,
shadow-passed, and shadow-failed items, but shadow plans still attach as exact
fallbacks and do not replace RTL state.

For transaction-kernel surrogates, the GEMM path also supports batched policy
evaluation and a runtime-ready handoff plan:

```sh
cargo run -q -p rrtl-pyrtl --bin pyrtl2rrtl -- surrogate run-gemm-batch \
  target/gemm/model/manifest.json target/gemm/batch.json --out target/gemm/batch_result.json
cargo run -q -p rrtl-pyrtl --bin pyrtl2rrtl -- surrogate policy-gemm-batch \
  target/gemm/model/manifest.json target/gemm/batch.json --out target/gemm/policy.json
cargo run -q -p rrtl-pyrtl --bin pyrtl2rrtl -- surrogate plan-runtime-gemm \
  target/gemm/policy.json --topology lanes:4 --out target/gemm/runtime_plan.json
cargo run -q -p rrtl-pyrtl --bin pyrtl2rrtl -- surrogate plan-runtime-gemm \
  target/gemm/policy.json --worker cpu-a:2 --worker cpu-b:2 \
  --out target/gemm/runtime_plan_workers.json
cargo run -q -p rrtl-pyrtl --bin pyrtl2rrtl -- surrogate run-fast-gemm \
  target/gemm/model/manifest.json target/gemm/batch.json \
  target/gemm/runtime_plan_workers.json --shadow-sample-stride 8 \
  --out target/gemm/fast_run.json
cargo run -q -p rrtl-pyrtl --bin pyrtl2rrtl -- surrogate attach-runtime-gemm \
  target/gemm/runtime_plan_workers.json --worker cpu-a:2 --worker cpu-b:2 \
  --out target/gemm/runtime_attachment.json
```

`run-fast-gemm` is the first opt-in replacement report for transaction-kernel
surrogates. It recomputes the manifest policy for the batch, validates the
runtime plan as the lane/topology contract, and emits per-item provenance for
surrogate replacements, exact fallbacks, shadow comparisons, and fail-closed
items. FAST reports also summarize per-worker and per-lane replacement,
fallback, fail-closed, and shadow counters. Optional
`--shadow-sample-stride/--shadow-sample-offset` marks deterministic sample
items for exact shadow follow-up without changing the policy result. Event
predictors remain report-only in this slice. `plan-runtime-gemm` preserves item
order and summarizes worker assignment and exact/approximate provenance
coverage. Use repeatable `--worker id:lanes` to emit worker IDs and contiguous
lane ranges that match a runtime topology. `attach-runtime-gemm` validates that
plan through `rrtl-runtime` and writes the telemetry attachment report; exact
cycle-simulation state remains authoritative.

To aggregate several per-op FAST reports into a local model-level R4 scaffold,
generate a model FAST plan that points at the existing op reports:

```sh
cargo run -q -p rrtl-pyrtl --bin pyrtl2rrtl -- surrogate plan-model-fast \
  --op gemm0:../gemm-batch-smoke/fast_run.json:"GEMM tile" \
  --op cache0:../cache_miss/fast_run.json:"Cache miss predictor" \
  --golden gemm0:target/model-fast-smoke/gemm0_golden.json \
  --golden cache0:../cache_miss/cache0_golden.json \
  --timing gemm0:1000000:125000 \
  --thresholds target/model-fast-smoke/thresholds.json \
  --out target/model-fast-smoke/model_fast_plan.json
```

The generated plan has this shape:

```json
{
  "schema": "rrtl-surrogate-model-fast-plan-v1",
  "thresholds": {
    "min_op_coverage": 1.0,
    "min_item_coverage": 0.5,
    "max_fallback_ratio": 0.5,
    "min_shadow_sample_ratio": 0.1
  },
  "ops": [
    {
      "op_id": "gemm0",
      "op_kind": "gemm",
      "name": "GEMM tile",
      "fast_report_path": "../gemm-batch-smoke/fast_run.json",
      "golden_path": "target/model-fast-smoke/gemm0_golden.json",
      "exact_ns": 1000000,
      "fast_ns": 125000
    },
    {
      "op_id": "cache0",
      "op_kind": "event",
      "name": "Cache miss predictor",
      "fast_report_path": "../cache_miss/fast_run.json"
    }
  ]
}
```

Golden comparison files use the minimal counter schema below. `run-model-fast`
compares these expected counters against the per-op FAST totals and rejects the
model report on mismatch:

```json
{
  "schema": "rrtl-surrogate-model-fast-golden-v1",
  "op_id": "gemm0",
  "op_kind": "gemm",
  "expected": {
    "items": 2,
    "surrogate_replacements": 1,
    "exact_fallbacks": 1,
    "fail_closed": 0,
    "shadow_sampled": 1
  }
}
```

Golden files can also carry optional small integer tensor payloads for
use-case-specific surrogate checks, such as cache-miss vectors or GEMM tile
results. When both `expected_tensors` and `actual_tensors` are present,
`run-model-fast` requires matching tensor names and shapes, then compares each
element with `max_abs_error` tolerance. Omit these fields to keep counter-only
goldens:

```json
{
  "schema": "rrtl-surrogate-model-fast-golden-v1",
  "op_id": "gemm0",
  "op_kind": "gemm",
  "expected": {
    "items": 2,
    "surrogate_replacements": 1,
    "exact_fallbacks": 1,
    "fail_closed": 0,
    "shadow_sampled": 1
  },
  "expected_tensors": {
    "c_tile": [[1, 2], [3, 4]]
  },
  "actual_tensors": {
    "c_tile": [[1, 2], [3, 5]]
  },
  "max_abs_error": 1
}
```

```sh
cargo run -q -p rrtl-pyrtl --bin pyrtl2rrtl -- surrogate run-model-fast \
  target/model-fast-smoke/model_fast_plan.json \
  --out target/model-fast-smoke/model_fast_report.json
```

`run-model-fast` preserves op order and totals replacement, fallback,
fail-closed, shadow, sampled-shadow, and provenance counters across the
referenced FAST reports. Optional thresholds reject a model FAST report when op
coverage, item coverage, fallback ratio, or sampled-shadow ratio misses the
plan's acceptance gates. Optional timing fields report per-op speedup and an
aggregate timing summary over ops that provide both exact and FAST nanosecond
measurements; missing timing is reported but does not reject the model report.
It is an aggregation scaffold only; XNN/TIS-specific op metadata and end-to-end
execution will layer on top of this report shape.
