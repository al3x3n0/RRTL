# RRTL

RRTL is a modern Rust stack for hardware modeling and design in the AI era. It
combines typed RTL construction, strict compilation, deterministic simulation,
SystemVerilog output, accelerated backends, distributed runtime tooling, and
machine-readable artifacts for AI-scale design and verification workflows.

```rust
use rrtl_core::{lit_u, mux, uint, Design};

let mut design = Design::new();
{
    let mut m = design.module("Counter");
    let clk = m.input("clk", uint(1));
    let rst = m.input("rst", uint(1));
    let en = m.input("en", uint(1));
    let out = m.output("out", uint(8));
    let count = m.reg("count", uint(8));

    m.clock(count, clk);
    m.reset(count, rst, 0);
    m.next(count, mux(en, count + lit_u(1, 8), count));
    m.assign(out, count);
}

design.validate().unwrap();
let compiled = design.compile().unwrap();
let sv = rrtl_sv::emit(&design).unwrap();
println!("{sv}");
```

## Crates

- `rrtl-ir`: core IR, identifiers, expressions, modules, and diagnostics.
- `rrtl-core`: builder API, validation passes, and cycle simulation.
- `rrtl-sim-ir`: experimental packed simulation IR and VLIW-style scheduling
  layer for accelerated backends.
- `rrtl-gpu-sim`: experimental batched GPU simulation research backend.
- `rrtl-runtime`: distributed and heterogeneous lane runtime that shards packed
  simulations across CPU and GPU workers.
- `rrtl-macros`: procedural macros for concise structured RTL declarations.
- `rrtl-sv`: SystemVerilog backend.
- `rrtl-pyrtl`: PyRTL block JSON importer and `pyrtl2rrtl` CLI.

## Current capabilities

- Strict explicit bit types for signals, literals, expressions, assignments,
  register next values, memory ports, and instance connections.
- Signed and unsigned types through `uint(width)` and `sint(width)`, with
  explicit `lit_u`, `lit_s`, `zext`, `sext`, `trunc`, `as_uint`, and `as_sint`.
- Enum-like state encodings through `state_type`, `state_reg`, `state_next`,
  `state_next_hold`, and `StateReg::is`.
- Structured bundles for ports and wires through `bundle_type`, `field`,
  `nested`, `input_bundle`, `output_bundle`, `wire_bundle`, field/path lookup,
  aggregate bundle assignment, and exact-match bundle instance connections. The
  `rrtl::bundle!` macro can generate the same `BundleType` values from a
  compact Rust DSL.
- Typed interfaces for reusable multi-port contracts through `interface_type`,
  `iface_input`, `iface_output`, `ModuleBuilder::interface`, scalar/bundle port
  lookup, and exact-match interface instance connections. The
  `rrtl::interface!` macro can generate the same `InterfaceType` values from a
  compact Rust DSL.
- Ready/valid channel contract helpers through `rv_source`, `rv_sink`,
  `rv_scalar`, `rv_bundle`, builder channel constructors, `ReadyValidRef`, and
  forward register-slice, one-entry skid-buffer, and register-backed FIFO
  helpers, plus memory-backed FIFO generation.
- Continuous assignments, registers with synchronous or asynchronous,
  active-high or active-low reset behavior, simple memories with combinational
  reads and synchronous writes, and module instances. The macro DSL covers
  common declarations, logic statements, state transitions, memory access, and
  instance, ready/valid, and external-module boilerplate.
- External module declarations for vendor primitives and blackbox IP
  instantiation, including scalar bidirectional `inout` ports for pad-level
  connections.
- Deterministic cycle simulation for a validated design.
- Experimental packed simulation IR and `wgpu` batched GPU simulation for
  accelerated multi-lane simulation research.
- Distributed heterogeneous runtime topology for routing huge independent lane
  batches across packed CPU and GPU workers while preserving global lane order.
- Hierarchy-aware runtime partition planning, launch/deployment payloads,
  structural worker actions, scripted partition-session runs, and JSON
  telemetry for huge single-design split orchestration.
- VCD waveform tracing for simulator sessions through explicit `VcdTrace`
  samples.
- Design assertions and cover points for simulation verification and
  SystemVerilog assertion/cover emission.
- A compiler pipeline via `rrtl_core::compile` / `Design::compile` that
  validates and normalizes authoring IR before backend emission.
- SystemVerilog emission for the supported RTL subset, including
  typed enum emission for state helpers and `rrtl_sv::emit_compiled` for
  already-compiled designs.
- Pretty JSON serialization for compiled designs through
  `CompiledDesign::to_json_pretty()`.
- PyRTL migration bridge for importing elaborated PyRTL `Block` netlists into
  native RRTL IR, including co-simulation-friendly register and memory initial
  values.

## PyRTL Import

Existing PyRTL source does not need to be rewritten. Elaborate the design as a
normal PyRTL `Block`, export it with the Python helper, then import or emit it
with the Rust CLI:

```sh
PYTHONPATH=python python -m rrtl_pyrtl my_design:build --top-name MyTop --out /tmp/mytop.pyrtl.json
cargo run -p rrtl-pyrtl --bin pyrtl2rrtl -- check /tmp/mytop.pyrtl.json
cargo run -p rrtl-pyrtl --bin pyrtl2rrtl -- sv /tmp/mytop.pyrtl.json
cargo run -p rrtl-pyrtl --bin pyrtl2rrtl -- compare /tmp/mytop.pyrtl.json /tmp/mytop.trace.json
```

The bridge targets PyRTL's normalized `LogicNet` block representation rather
than Python source. PyRTL's implicit sequential domain is imported as an
explicit 1-bit `clk` input by default, with register reset values and ROM
contents preserved as simulation/SystemVerilog initial values.
The Python helper also exposes `simulate_trace(input_vectors)` for creating
trace JSON that the `compare` command replays against the imported RRTL design.

## RTL-Derived Surrogate Ingestion

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

For larger migrations, run a corpus manifest and keep the per-design artifacts:

```json
{
  "targets": [
    {
      "name": "counter",
      "target": "my_designs.counter:build",
      "top_name": "Counter",
      "inputs": [{ "en": 1 }, { "en": 0 }]
    }
  ]
}
```

Use `python/rrtl_pyrtl/corpus_template.json` as a starting point for private
manifests. Keep the filled manifest next to the private PyRTL sources, add those
sources to `PYTHONPATH`, and validate the manifest before running the full
bridge:

```sh
PYTHONPATH=python:/path/to/private/sources \
  python -m rrtl_pyrtl.corpus --validate-only /path/to/corpus.json
```

For a large existing source tree, generate a draft manifest with the static
discovery scanner first. Discovery parses Python source without importing or
executing private modules, then writes both a draft JSON manifest and a Markdown
candidate report:

```sh
PYTHONPATH=python python -m rrtl_pyrtl.discover /path/to/private/sources \
  --package-root /path/to/private/sources \
  --out target/rrtl-pyrtl-corpus/discovered.json \
  --report target/rrtl-pyrtl-corpus/discovered.md
```

Review the report, edit the draft manifest as needed, then validate it with
`rrtl_pyrtl.corpus --validate-only`.

To generate discovery artifacts and run manifest validation in one safe step,
use the prepare wrapper. It still does not execute PyRTL builders:

```sh
PYTHONPATH=python:/path/to/private/sources \
  python -m rrtl_pyrtl.prepare_corpus /path/to/private/sources \
  --package-root /path/to/private/sources \
  --out-dir target/rrtl-pyrtl-corpus/private
```

This writes `discovered.json`, `discovered.md`, `validation.json`, and
`validation.md` under the output directory. It also writes `next_commands.md`
with exact commands for re-running prepare, validating the generated manifest,
and running full corpus triage.

For CI or repeated private-tree baselining, use the report-only gate. It runs
discovery, validation, full corpus triage, and high-level gate reporting in one
command. Target import or compare failures are reported in the artifacts but do
not make the gate exit nonzero; manifest/tooling failures still do.

```sh
PYTHONPATH=python \
  python -m rrtl_pyrtl.gate /path/to/private/sources \
  --package-root /path/to/private/sources \
  --out-dir target/rrtl-pyrtl-corpus/private-gate
```

The gate writes `gate.json`, `gate.md`, `summary.json`, `summary.md`,
`validation.json`, and `validation.md`. Pass `--manifest /path/to/corpus.json`
to skip discovery and run against a reviewed manifest.

```sh
PYTHONPATH=python python -m rrtl_pyrtl.corpus corpus.json \
  --emit-sv \
  --emit-json \
  --summary-json target/rrtl-pyrtl-corpus/summary.json \
  --summary-md target/rrtl-pyrtl-corpus/summary.md
```

Each target is exported, checked, optionally emitted as SV/compiled JSON, and
compared against a PyRTL trace when `inputs` are provided. Failures include the
target name and the importer reports the PyRTL net index/op/args/dests for
unsupported constructs. Corpus runs continue after failures by default so the
summary can show the full compatibility surface; pass `--fail-fast` when a local
debugging run should stop at the first failing target. `--validate-only` checks
manifest shape, duplicate target names, target importability, and input-vector
shape without invoking design builders.

A checked-in smoke corpus is available for validating the toolchain itself:

```sh
PYTHONPATH=python python -m rrtl_pyrtl.corpus \
  python/rrtl_pyrtl/corpus_smoke.json \
  --emit-sv \
  --emit-json \
  --summary-json target/rrtl-pyrtl-corpus/summary.json \
  --summary-md target/rrtl-pyrtl-corpus/summary.md
```

The summary JSON uses stable fields for triage automation: `name`, `target`,
`phase`, `ok`, `error`, `bucket`, and `outputs`. It also includes aggregate
totals, phase counts, and failure bucket counts. The Markdown summary is the
human-readable triage report for reviewing large private manifests.

## PyRTL Benchmarking

The bridge includes a synthetic systolic-style PyRTL benchmark for establishing
a first throughput baseline against PyRTL `FastSimulation`. It builds a
repo-local signed MAC array, exports it through the PyRTL bridge, checks RRTL
scalar, packed, single-lane machine, and backend-selected trace replay for
bit-exactness, and writes timing artifacts:

```sh
PYTHONPATH=python python -m rrtl_pyrtl.bench \
  --out-dir target/rrtl-pyrtl-bench/smoke \
  --rows 2 \
  --cols 2 \
  --steps 32 \
  --repeat 2 \
  --packed-lanes 1
```

For an XNN-like single-lane baseline, use a larger array and longer stream:

```sh
PYTHONPATH=python python -m rrtl_pyrtl.bench \
  --out-dir target/rrtl-pyrtl-bench/xnn-r1 \
  --rows 8 \
  --cols 8 \
  --steps 512 \
  --repeat 5 \
  --warmup 1
```

The benchmark writes `bench.json`, `bench.md`, the exported PyRTL JSON, and the
trace JSON. Timing is report-only for now; correctness mismatches fail. The
single-lane machine row is a scalar CPU baseline and correctness oracle. The
backend-selected report exposes the shared scalar, packed CPU, SIMD CPU, and JIT
CPU backend surface; SIMD currently shares the packed CPU execution path while
the dedicated SIMD kernels are filled in, and JIT reports unavailable until
Cranelift codegen is implemented behind the `jit` feature. This is the runtime
architecture hook for the intended Apple Silicon/NVGPU path, not the final
performance endpoint. The
underlying Rust replay timers are also available directly:

```sh
cargo run -p rrtl-pyrtl --bin pyrtl2rrtl -- bench-trace \
  target/rrtl-pyrtl-bench/smoke/systolic_mac.pyrtl.json \
  target/rrtl-pyrtl-bench/smoke/systolic_mac.trace.json \
  --repeat 5 \
  --warmup 1

cargo run -p rrtl-pyrtl --bin pyrtl2rrtl -- bench-packed-trace \
  target/rrtl-pyrtl-bench/smoke/systolic_mac.pyrtl.json \
  target/rrtl-pyrtl-bench/smoke/systolic_mac.trace.json \
  --repeat 5 \
  --warmup 1 \
  --lanes 1

cargo run -p rrtl-pyrtl --bin pyrtl2rrtl -- bench-single-trace \
  target/rrtl-pyrtl-bench/smoke/systolic_mac.pyrtl.json \
  target/rrtl-pyrtl-bench/smoke/systolic_mac.trace.json \
  --repeat 5 \
  --warmup 1

cargo run -p rrtl-pyrtl --bin pyrtl2rrtl -- bench-backends \
  target/rrtl-pyrtl-bench/smoke/systolic_mac.pyrtl.json \
  target/rrtl-pyrtl-bench/smoke/systolic_mac.trace.json \
  --backend scalar,packed-cpu,simd-cpu,jit-cpu \
  --repeat 5 \
  --warmup 1 \
  --lanes 4
```

## VCD Waveform Tracing

Simulator sessions can be sampled into deterministic VCD output for waveform
viewers:

```rust
use rrtl_core::{Simulator, VcdTrace};

let mut sim = Simulator::new(&design, "Counter").unwrap();
let mut trace = VcdTrace::new(&design, "Counter").unwrap();

trace.sample(&sim, 0).unwrap();
sim.tick();
trace.sample(&sim, 1).unwrap();

let vcd = trace.finish();
println!("{vcd}");
```

Tracing records scalar signals across the simulated instance hierarchy,
including flattened bundle/interface fields. Memory words can be included
explicitly when debugging register files and FIFOs:

```rust
use rrtl_core::{VcdMemoryTrace, VcdTraceOptions};

let mut trace = VcdTrace::with_options(
    &design,
    "Counter",
    VcdTraceOptions {
        memories: VcdMemoryTrace::All,
    },
)
.unwrap();
```

## Experimental GPU Simulation

`rrtl-gpu-sim` is a research crate for running many independent lanes of the
same design on a GPU through `wgpu`. It consumes the `rrtl-sim-ir` packed
simulation IR, which flattens hierarchy, lays out scalar state as `u32` limbs,
and schedules combinational and tick work into packet streams before shader
generation:

```rust
use rrtl_gpu_sim::GpuBatchSimulator;

let mut gpu = GpuBatchSimulator::new(&design, "Counter", 1024).unwrap();
gpu.set_input(en, &vec![1; 1024]).unwrap();
gpu.tick().unwrap();
let counts = gpu.get_signal(count).unwrap();
```

Signals wider than 32 bits use explicit limb APIs:

```rust
gpu.set_input_limbs(wide_in, &[vec![0xffff_fffe, 0xff]; 1024]).unwrap();
let wide_values = gpu.get_signal_limbs(wide_out).unwrap();
```

The current experimental GPU path supports flattened non-external hierarchy,
signed and unsigned scalar values represented as little-endian `u32` limbs,
concat/slice/cast/extension expressions, registers with sync or async resets,
and per-lane memories with combinational reads and synchronous writes. It still
rejects external modules, inout ports, assertions, and cover points for GPU
execution with explicit diagnostics. Bundle and interface signals work through
their flattened scalar leaves.

The packed IR also has a CPU-side `PackedSimulator` for differential testing
and backend development. It lowers expressions into a lower-level
`PackedMachineProgram` with reusable SSA-like values and packeted effects
before interpretation or GPU shader emission. A lightweight benchmark example
compares scalar CPU lanes, packed interpretation, and GPU batch execution:

```sh
cargo run -p rrtl-gpu-sim --example batch_bench --release
```

The same benchmark can sweep scheduler and shader layout knobs, rank each
configuration, and export the best recommendation for each benchmark
`(case, lanes, steps)` tuple:

```sh
cargo run -p rrtl-gpu-sim --example batch_bench --release -- \
  --quick \
  --autotune \
  --autotune-metric gpu_tick_many \
  --format human

cargo run -p rrtl-gpu-sim --example batch_bench --release -- \
  --quick \
  --autotune \
  --recommend-config \
  --output /tmp/rrtl-gpu-recommend.json
```

On machines without a usable GPU adapter, GPU timing fields are unavailable and
autotune ranking falls back to packed CPU timing. The recommendation JSON still
captures the selected scheduling cap, memory read cap, liveness priority,
temporary reuse, memory layout, workgroup size, and timing fields.

Runtime code can load the recommendation file and construct a simulator with
the matching options:

```rust
use std::fs::File;

use rrtl_gpu_sim::{
    load_gpu_autotune_recommendations_report, GpuBatchSimulator,
};

let recommendations =
    load_gpu_autotune_recommendations_report(File::open("/tmp/rrtl-gpu-recommend.json").unwrap())
        .unwrap();
let mut gpu = GpuBatchSimulator::new_with_autotune_recommendations(
    &design,
    "CounterBench",
    "counter",
    64,
    16,
    &recommendations,
)
.unwrap();
```

Recommendation lookup is exact on `(case, lanes, steps)`. JSON parse failures
use `E_GPU_AUTOTUNE_JSON`, missing exact matches use
`E_GPU_AUTOTUNE_RECOMMENDATION`, and invalid memory layout values use
`E_GPU_AUTOTUNE_LAYOUT`.

## Distributed Heterogeneous Runtime

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

## Bundle Macro

Manual bundle construction remains the canonical runtime API:

```rust
use rrtl::{bundle_type, field, nested, uint};

let req_ty = bundle_type(
    "Req",
    [
        field("valid", uint(1)),
        field("addr", uint(8)),
        nested("meta", bundle_type("ReqMeta", [field("tag", uint(4))])),
    ],
);
```

The procedural macro removes the repetitive constructor calls while producing
the same value:

```rust
use rrtl::bundle;

let req_ty = bundle! {
    Req {
        valid: uint(1),
        addr: uint(8),
        meta: ReqMeta {
            tag: uint(4),
        },
    }
};
```

The first macro slice supports nested bundle blocks and `uint(width)` /
`sint(width)` leaf fields.

Bundles can be assigned leaf-for-leaf when their paths and types match:

```rust
let req = m.input_bundle("req", req_ty.clone());
let pipe = m.wire_bundle("pipe", req_ty.clone());
let resp = m.output_bundle("resp", req_ty);

m.assign_bundle(&pipe, &req);
m.assign_bundle_when(&resp, accepted, &pipe);
```

## Interface Macro

Manual interface construction remains available:

```rust
use rrtl::{iface_input, iface_output, interface_type, uint};

let bus_ty = interface_type(
    "Bus",
    [
        iface_input("req", req_ty),
        iface_output("resp", resp_ty),
        iface_input("ready", uint(1)),
    ],
);
```

The companion macro supports scalar ports and inline bundle ports:

```rust
use rrtl::interface;

let bus_ty = interface! {
    Bus {
        input req: Req {
            valid: uint(1),
            addr: uint(8),
            meta: ReqMeta {
                tag: uint(4),
            },
        },
        output resp: Resp {
            valid: uint(1),
            data: uint(8),
        },
        input ready: uint(1),
    }
};
```

Interface handles expose scalar and bundle leaves directly:

```rust
let bus = m.interface("bus", bus_ty);
let req_valid = bus.field("req", "valid").unwrap();
let req_tag = bus.path("req", ["meta", "tag"]).unwrap();
let ready = bus.signal("ready").unwrap();
```

## Signals Macro

Manual module signal declarations remain available:

```rust
let clk = m.input("clk", uint(1));
let rst = m.input("rst", uint(1));
let en = m.input("en", uint(1));
let out = m.output("out", uint(8));
let count = m.reg("count", uint(8));
```

The `signals!` macro keeps binding explicit while removing repeated builder
calls:

```rust
use rrtl::signals;

let (clk, rst, en, out, count) = signals! {
    m {
        input clk: uint(1),
        input rst: uint(1),
        input en: uint(1),
        output out: uint(8),
        reg count: uint(8),
    }
};
```

It supports scalar `input`, `output`, `wire`, and `reg` declarations, plus
`input_bundle`, `output_bundle`, `wire_bundle`, inline `interface`, `mem`,
`rv_sink`, and `rv_source` declarations.

## Logic Macro

Manual module body statements remain available:

```rust
m.clock(count, clk);
m.reset(count, rst, 0);
m.next(count, mux(en, count + lit_u(1, 8), count));
m.assign(out, count);
```

The `logic!` macro groups common builder statements while keeping expressions
as normal Rust:

```rust
use rrtl::logic;

logic! {
    m {
        clock count: clk;
        reset count: rst = 0;
        next count = mux(en, count + lit_u(1, 8), count);
        assign out = count;
    }
}
```

It supports `assign`, `clock`, `reset`, `reset_low`, `async_reset`,
`async_reset_low`, `next`, `state_next_hold`, `state_next`, `mem_write`,
`assign_bundle`, `assign_bundle_when`, `assert`, `assert_when`,
`assert_clocked`, `assert_msg`, `cover`, `cover_when`, `cover_clocked`, and
`cover_msg`.

## Design Assertions And Covers

Assertions encode simulation and generated-HDL invariants. Cover points count
interesting states or events without failing simulation:

```rust
m.assert("count_small", count.value().lt_expr(lit_u(10, 8)));
m.assert_when("accepted_small", accepted, count.value().lt_expr(lit_u(10, 8)));
m.assert_clocked("count_checked", clk, count.value().lt_expr(lit_u(10, 8)));
m.cover("done_seen", done);
m.cover_when("accepted_done", accepted, done);
m.cover_clocked("done_clocked", clk, done);

let mut sim = Simulator::new(&design, "Counter").unwrap();
sim.check_assertions().unwrap();
sim.tick_checked().unwrap();
assert_eq!(sim.cover_hits("Counter.done_seen"), 1);
```

The `logic!` macro has matching assertion and cover forms:

```rust
logic! {
    m {
        assert count_small: count.value().lt_expr(lit_u(10, 8));
        assert_when accepted_small: accepted, count.value().lt_expr(lit_u(10, 8));
        assert_clocked count_checked: clk, count.value().lt_expr(lit_u(10, 8));
        assert_msg count_message: count.value().lt_expr(lit_u(10, 8)), "count too large";
        cover done_seen: done;
        cover_when accepted_done: accepted, done;
        cover_clocked done_clocked: clk, done;
        cover_msg done_message: done, "done state reached";
    }
}
```

## Ready/Valid Macro

Manual ready/valid channels and helpers remain available:

```rust
let input = m.rv_sink("in", rv_scalar(uint(8)));
let output = m.rv_source("out", rv_scalar(uint(8)));

// Choose one passthrough/buffering helper for a pair of endpoints.
m.rv_connect(&input, &output);
// m.rv_register_slice("slice", &input, &output, clk, rst);
```

The macro form keeps channel handles explicit while removing repeated payload
and helper boilerplate:

```rust
use rrtl::{ready_valid, signals};

let (clk, rst, input, output) = signals! {
    m {
        input clk: uint(1),
        input rst: uint(1),
        rv_sink input: scalar uint(8),
        rv_source output: scalar uint(8),
    }
};

ready_valid! {
    m {
        // Choose one passthrough/buffering helper for a pair of endpoints.
        connect input => output;
        // register_slice slice: input => output, clk, rst;
        // skid_buffer skid: input => output, clk, rst;
        // fifo fifo: input => output, clk, rst, depth 3;
        // mem_fifo mem_fifo: input => output, clk, rst, depth 4;
    }
}
```

Ready/valid declarations also support bundle payloads:

```rust
rv_sink input: bundle Payload {
    data: uint(8),
    last: uint(1),
},
```

## Memory Macro

Manual memory construction remains available:

```rust
let regs = m.mem("regs", 2, uint(8), 4);
m.mem_write(regs, clk, we, waddr, wdata);
m.assign(rdata, m.mem_read(regs, raddr));
```

The macro form keeps reads usable as normal expressions:

```rust
use rrtl::{logic, signals};

let (clk, we, waddr, wdata, raddr, rdata, regs) = signals! {
    m {
        input clk: uint(1),
        input we: uint(1),
        input waddr: uint(2),
        input wdata: uint(8),
        input raddr: uint(2),
        output rdata: uint(8),
        mem regs: addr(2), data uint(8), depth 4,
    }
};

logic! {
    m {
        mem_write regs: clk, we, waddr, wdata;
        assign rdata = rrtl::mem_read!(m, regs, raddr);
    }
}
```

## Instances Macro

Manual instance construction remains available:

```rust
m.instance("u_child", "Child", [("a", a), ("y", y)]);
m.instance_bundles("u_responder", "Responder", [("req", req), ("resp", resp)]);
m.instance_interfaces("u_bus", "BusResponder", [("bus", bus)]);
```

The `instances!` macro removes repeated string tuples for scalar, bundle, and
interface connections:

```rust
use rrtl::instances;

instances! {
    m {
        instance u_child: Child {
            a: a,
            y: y,
        },
        instance_bundles u_responder: Responder {
            req: req,
            resp: resp,
        },
        instance_interfaces u_bus: BusResponder {
            bus: bus,
        },
    }
}
```

## External Module Macro

Manual external module declarations remain available:

```rust
let mut ext = design.extern_module("SB_PLL40_CORE");
ext.input("REFERENCECLK", uint(1));
ext.input("RESETB", uint(1));
ext.output("PLLOUTCORE", uint(1));
```

The `extern_module!` macro declares vendor and blackbox module ports without
repeated string calls:

```rust
use rrtl::extern_module;

extern_module! {
    design SB_PLL40_CORE {
        input REFERENCECLK: uint(1),
        input RESETB: uint(1),
        output PLLOUTCORE: uint(1),
    }
}

extern_module! {
    design IOBUF {
        inout PAD: uint(1),
        input I: uint(1),
        input T: uint(1),
        output O: uint(1),
    }
}
```

## State Macro

Manual state type construction remains available:

```rust
let states = state_type(
    "ControllerState",
    uint(2),
    [("Idle", 0), ("Run", 1), ("Done", 2)],
);
```

The `state!` macro removes the string tuple boilerplate:

```rust
use rrtl::state;

let states = state! {
    ControllerState: uint(2) {
        Idle = 0,
        Run = 1,
        Done = 2,
    }
};
```

State transitions can be grouped in `logic!`:

```rust
logic! {
    m {
        state_next_hold flow {
            start.value() => Run,
            done.value() => Done,
        };
        assign busy = flow.is("Run");
    }
}
```

## Current limitations

- Memory reads outside declared depth simulate as zero, and writes outside
  declared depth are ignored.
- Bidirectional electrical behavior and internal tri-state simulation are not
  modeled yet.
