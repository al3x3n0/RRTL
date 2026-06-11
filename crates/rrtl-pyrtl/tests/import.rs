use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use rrtl_core::Simulator;
use rrtl_gpu_sim::GpuMemoryLayout;
use rrtl_pyrtl::{
    bench_backends_trace, bench_gpu_combined, bench_gpu_options, bench_gpu_trace,
    bench_packed_trace, bench_single_trace, bench_threaded_trace, bench_trace, compare_trace,
    import_export, plan_backends, profile_replay, BenchBackendsTraceOptions, BenchGpuTraceOptions,
    BenchPackedTraceOptions, BenchSingleTraceOptions, BenchThreadedTraceOptions, BenchTraceOptions,
    PlanBackendsOptions, PlannerCalibration, PlannerCalibrationPreference,
    PlannerCalibrationSummary, ProfileReplayOptions, PyrtlBenchBackendKind, PyrtlExport,
    PyrtlLaneTrace, PyrtlTrace, RuntimeProfile, RuntimeProfileSelectedBackend,
    RuntimeProfileThreadedLayout, RuntimeProfileWorker,
};
use rrtl_sim_ir::{BackendAffinityRecommendation, SimBackendKind, ThreadedReplayWorkerOptions};

fn load_export(text: &str) -> PyrtlExport {
    serde_json::from_str(text).unwrap()
}

fn planner_calibration(threaded: Vec<&str>, gpu: Vec<&str>) -> PlannerCalibration {
    PlannerCalibration {
        schema: "rrtl-pyrtl-planner-calibration-v1".to_string(),
        summary: PlannerCalibrationSummary {
            backend_preferences: Vec::new(),
            hot_backend_preferences: Vec::new(),
            threaded_layout_preferences: threaded
                .into_iter()
                .enumerate()
                .map(|(index, signature)| PlannerCalibrationPreference {
                    signature: signature.to_string(),
                    score: 100.0 - index as f64,
                    count: 1,
                })
                .collect(),
            gpu_option_preferences: gpu
                .into_iter()
                .enumerate()
                .map(|(index, signature)| PlannerCalibrationPreference {
                    signature: signature.to_string(),
                    score: 100.0 - index as f64,
                    count: 1,
                })
                .collect(),
            profitability_backend_preferences: Vec::new(),
            profitability_penalties: Vec::new(),
            profitability_feature_preferences: Vec::new(),
            profitability_feature_penalties: Vec::new(),
        },
    }
}

fn planner_hot_backend_calibration(signature: &str) -> PlannerCalibration {
    PlannerCalibration {
        schema: "rrtl-pyrtl-planner-calibration-v1".to_string(),
        summary: PlannerCalibrationSummary {
            backend_preferences: Vec::new(),
            hot_backend_preferences: vec![PlannerCalibrationPreference {
                signature: signature.to_string(),
                score: 100.0,
                count: 1,
            }],
            threaded_layout_preferences: Vec::new(),
            gpu_option_preferences: Vec::new(),
            profitability_backend_preferences: Vec::new(),
            profitability_penalties: Vec::new(),
            profitability_feature_preferences: Vec::new(),
            profitability_feature_penalties: Vec::new(),
        },
    }
}

fn threaded_signature(layout: &rrtl_sim_ir::ThreadedReplayOptions) -> String {
    layout
        .workers
        .iter()
        .map(|worker| format!("{}:{}", worker.backend.as_str(), worker.lanes))
        .collect::<Vec<_>>()
        .join(",")
}

#[test]
fn imports_comb_register_and_memory_semantics() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 4},
            {"name": "b", "kind": "input", "bitwidth": 4},
            {"name": "sel", "kind": "input", "bitwidth": 1},
            {"name": "addr", "kind": "input", "bitwidth": 2},
            {"name": "sum", "kind": "wire", "bitwidth": 5},
            {"name": "prod", "kind": "wire", "bitwidth": 8},
            {"name": "choice", "kind": "output", "bitwidth": 5},
            {"name": "prod_out", "kind": "output", "bitwidth": 8},
            {"name": "q", "kind": "register", "bitwidth": 4, "reset_value": 3},
            {"name": "q_out", "kind": "output", "bitwidth": 4},
            {"name": "rd", "kind": "output", "bitwidth": 8}
          ],
          "memories": [
            {
              "name": "rom",
              "id": 0,
              "kind": "rom",
              "bitwidth": 8,
              "addrwidth": 2,
              "initial": [
                {"addr": 0, "value": 9},
                {"addr": 1, "value": 10}
              ]
            }
          ],
          "nets": [
            {"index": 0, "op": "+", "op_param": null, "args": ["a", "b"], "dests": ["sum"]},
            {"index": 1, "op": "*", "op_param": null, "args": ["a", "b"], "dests": ["prod"]},
            {"index": 2, "op": "x", "op_param": null, "args": ["sel", "sum", "a"], "dests": ["choice"]},
            {"index": 3, "op": "w", "op_param": null, "args": ["prod"], "dests": ["prod_out"]},
            {"index": 4, "op": "r", "op_param": null, "args": ["a"], "dests": ["q"]},
            {"index": 5, "op": "w", "op_param": null, "args": ["q"], "dests": ["q_out"]},
            {"index": 6, "op": "m", "op_param": {"memory_id": 0, "memory": "rom"}, "args": ["addr"], "dests": ["rd"]}
          ]
        }
        "#,
    );

    let imported = import_export(&export).unwrap();
    let design = imported.design;
    let a = design.find_signal("Top", "a").unwrap();
    let b = design.find_signal("Top", "b").unwrap();
    let sel = design.find_signal("Top", "sel").unwrap();
    let addr = design.find_signal("Top", "addr").unwrap();
    let choice = design.find_signal("Top", "choice").unwrap();
    let prod_out = design.find_signal("Top", "prod_out").unwrap();
    let q_out = design.find_signal("Top", "q_out").unwrap();
    let rd = design.find_signal("Top", "rd").unwrap();

    let mut sim = Simulator::new(&design, "Top").unwrap();
    sim.set(a, 7);
    sim.set(b, 4);
    sim.set(sel, 0);
    sim.set(addr, 1);
    assert_eq!(sim.get(choice), 11);
    assert_eq!(sim.get(prod_out), 28);
    assert_eq!(sim.get(q_out), 3);
    assert_eq!(sim.get(rd), 10);

    sim.set(sel, 1);
    assert_eq!(sim.get(choice), 7);

    sim.tick();
    assert_eq!(sim.get(q_out), 7);
}

#[test]
fn compares_pyrtl_trace_against_imported_design() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 4},
            {"name": "q", "kind": "register", "bitwidth": 4, "reset_value": 1},
            {"name": "out", "kind": "output", "bitwidth": 4}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "r", "op_param": null, "args": ["a"], "dests": ["q"]},
            {"index": 1, "op": "w", "op_param": null, "args": ["q"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-trace-v1",
          "steps": [
            {"inputs": {"a": 5}, "outputs": {"out": 1}},
            {"inputs": {"a": 9}, "outputs": {"out": 5}},
            {"inputs": {"a": 2}, "outputs": {"out": 9}}
          ]
        }
        "#,
    )
    .unwrap();

    let mismatches = compare_trace(&export, &trace).unwrap();
    assert!(mismatches.is_empty());
}

#[test]
fn bench_trace_reports_replay_timing_samples() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 4},
            {"name": "q", "kind": "register", "bitwidth": 4, "reset_value": 1},
            {"name": "out", "kind": "output", "bitwidth": 4}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "r", "op_param": null, "args": ["a"], "dests": ["q"]},
            {"index": 1, "op": "w", "op_param": null, "args": ["q"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-trace-v1",
          "steps": [
            {"inputs": {"a": 5}, "outputs": {"out": 1}},
            {"inputs": {"a": 9}, "outputs": {"out": 5}},
            {"inputs": {"a": 2}, "outputs": {"out": 9}}
          ]
        }
        "#,
    )
    .unwrap();

    let report = bench_trace(
        &export,
        &trace,
        BenchTraceOptions {
            repeat: 2,
            warmup: 1,
        },
    )
    .unwrap();

    assert_eq!(report.schema, "rrtl-pyrtl-bench-trace-v1");
    assert_eq!(report.steps, 3);
    assert_eq!(report.repeat, 2);
    assert_eq!(report.warmup, 1);
    assert_eq!(report.mismatch_count, 0);
    assert!(report.mismatches.is_empty());
    assert_eq!(report.replay_ns_samples.len(), 2);
    assert!(report.replay_ns_best > 0);
    assert!(report.replay_ns_median >= report.replay_ns_best);
}

#[test]
fn bench_trace_reports_mismatch_without_samples() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 4},
            {"name": "out", "kind": "output", "bitwidth": 4}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "w", "op_param": null, "args": ["a"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-trace-v1",
          "steps": [
            {"inputs": {"a": 5}, "outputs": {"out": 6}}
          ]
        }
        "#,
    )
    .unwrap();

    let report = bench_trace(
        &export,
        &trace,
        BenchTraceOptions {
            repeat: 2,
            warmup: 0,
        },
    )
    .unwrap();

    assert_eq!(report.mismatch_count, 1);
    assert_eq!(report.mismatches[0].signal, "out");
    assert_eq!(report.mismatches[0].expected, 6);
    assert_eq!(report.mismatches[0].actual, 5);
    assert!(report.replay_ns_samples.is_empty());
}

#[test]
fn bench_packed_trace_reports_replay_timing_samples() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 4},
            {"name": "q", "kind": "register", "bitwidth": 4, "reset_value": 1},
            {"name": "out", "kind": "output", "bitwidth": 4}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "r", "op_param": null, "args": ["a"], "dests": ["q"]},
            {"index": 1, "op": "w", "op_param": null, "args": ["q"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-trace-v1",
          "steps": [
            {"inputs": {"a": 5}, "outputs": {"out": 1}},
            {"inputs": {"a": 9}, "outputs": {"out": 5}},
            {"inputs": {"a": 2}, "outputs": {"out": 9}}
          ]
        }
        "#,
    )
    .unwrap();

    let report = bench_packed_trace(
        &export,
        &trace,
        BenchPackedTraceOptions {
            repeat: 2,
            warmup: 1,
            lanes: 2,
        },
    )
    .unwrap();

    assert_eq!(report.schema, "rrtl-pyrtl-bench-packed-trace-v1");
    assert_eq!(report.steps, 3);
    assert_eq!(report.repeat, 2);
    assert_eq!(report.warmup, 1);
    assert_eq!(report.lanes, 2);
    assert_eq!(report.mismatch_count, 0);
    assert!(report.mismatches.is_empty());
    assert_eq!(report.replay_ns_samples.len(), 2);
    assert!(report.replay_ns_best > 0);
    assert!(report.replay_ns_median >= report.replay_ns_best);
}

#[test]
fn bench_packed_trace_reports_lane_mismatch_without_samples() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 4},
            {"name": "out", "kind": "output", "bitwidth": 4}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "w", "op_param": null, "args": ["a"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-trace-v1",
          "steps": [
            {"inputs": {"a": 5}, "outputs": {"out": 6}}
          ]
        }
        "#,
    )
    .unwrap();

    let report = bench_packed_trace(
        &export,
        &trace,
        BenchPackedTraceOptions {
            repeat: 2,
            warmup: 0,
            lanes: 2,
        },
    )
    .unwrap();

    assert_eq!(report.mismatch_count, 2);
    assert_eq!(report.mismatches[0].signal, "out");
    assert_eq!(report.mismatches[0].expected, 6);
    assert_eq!(report.mismatches[0].actual, 5);
    assert_eq!(report.mismatches[0].lane, Some(0));
    assert_eq!(report.mismatches[1].lane, Some(1));
    assert!(report.replay_ns_samples.is_empty());
}

#[test]
fn bench_single_trace_reports_replay_timing_samples() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 4},
            {"name": "q", "kind": "register", "bitwidth": 4, "reset_value": 1},
            {"name": "out", "kind": "output", "bitwidth": 4}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "r", "op_param": null, "args": ["a"], "dests": ["q"]},
            {"index": 1, "op": "w", "op_param": null, "args": ["q"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-trace-v1",
          "steps": [
            {"inputs": {"a": 5}, "outputs": {"out": 1}},
            {"inputs": {"a": 9}, "outputs": {"out": 5}},
            {"inputs": {"a": 2}, "outputs": {"out": 9}}
          ]
        }
        "#,
    )
    .unwrap();

    let report = bench_single_trace(
        &export,
        &trace,
        BenchSingleTraceOptions {
            repeat: 2,
            warmup: 1,
        },
    )
    .unwrap();

    assert_eq!(report.schema, "rrtl-pyrtl-bench-single-trace-v1");
    assert_eq!(report.steps, 3);
    assert_eq!(report.repeat, 2);
    assert_eq!(report.warmup, 1);
    assert_eq!(report.mismatch_count, 0);
    assert!(report.mismatches.is_empty());
    assert_eq!(report.replay_ns_samples.len(), 2);
    assert!(report.replay_ns_best > 0);
    assert!(report.replay_ns_median >= report.replay_ns_best);
}

#[test]
fn bench_single_trace_reports_mismatch_without_samples() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 4},
            {"name": "out", "kind": "output", "bitwidth": 4}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "w", "op_param": null, "args": ["a"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-trace-v1",
          "steps": [
            {"inputs": {"a": 5}, "outputs": {"out": 6}}
          ]
        }
        "#,
    )
    .unwrap();

    let report = bench_single_trace(
        &export,
        &trace,
        BenchSingleTraceOptions {
            repeat: 2,
            warmup: 0,
        },
    )
    .unwrap();

    assert_eq!(report.mismatch_count, 1);
    assert_eq!(report.mismatches[0].signal, "out");
    assert_eq!(report.mismatches[0].expected, 6);
    assert_eq!(report.mismatches[0].actual, 5);
    assert_eq!(report.mismatches[0].lane, None);
    assert!(report.replay_ns_samples.is_empty());
}

#[test]
fn bench_backends_trace_reports_selected_backend_samples() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 4},
            {"name": "q", "kind": "register", "bitwidth": 4, "reset_value": 1},
            {"name": "out", "kind": "output", "bitwidth": 4}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "r", "op_param": null, "args": ["a"], "dests": ["q"]},
            {"index": 1, "op": "w", "op_param": null, "args": ["q"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-trace-v1",
          "steps": [
            {"inputs": {"a": 5}, "outputs": {"out": 1}},
            {"inputs": {"a": 9}, "outputs": {"out": 5}},
            {"inputs": {"a": 2}, "outputs": {"out": 9}}
          ]
        }
        "#,
    )
    .unwrap();

    let report = bench_backends_trace(
        &export,
        &trace,
        BenchBackendsTraceOptions {
            repeat: 2,
            warmup: 1,
            lanes: 2,
            backends: vec![
                PyrtlBenchBackendKind::Scalar,
                PyrtlBenchBackendKind::PackedCpu,
                PyrtlBenchBackendKind::SimdCpu,
            ],
        },
    )
    .unwrap();

    assert_eq!(report.schema, "rrtl-pyrtl-bench-backends-trace-v1");
    assert_eq!(report.backends.len(), 3);
    assert_eq!(report.backends[0].backend, "scalar");
    assert_eq!(report.backends[0].lanes, 1);
    assert!(report.backends[0].available);
    assert_eq!(report.backends[0].replay_ns_samples.len(), 2);
    assert_eq!(report.backends[0].mismatch_count, 0);
    assert_eq!(report.backends[1].backend, "packed-cpu");
    assert_eq!(report.backends[1].lanes, 2);
    assert!(report.backends[1].available);
    assert_eq!(report.backends[1].replay_ns_samples.len(), 2);
    assert_eq!(report.backends[1].mismatch_count, 0);
    assert_eq!(report.backends[2].backend, "simd-cpu");
    assert_eq!(report.backends[2].lanes, 2);
    assert!(report.backends[2].available);
    assert_eq!(report.backends[2].replay_ns_samples.len(), 2);
    assert_eq!(report.backends[2].mismatch_count, 0);
}

#[test]
fn bench_backends_trace_reports_unavailable_jit_backend() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 4},
            {"name": "out", "kind": "output", "bitwidth": 4}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "w", "op_param": null, "args": ["a"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-trace-v1",
          "steps": [
            {"inputs": {"a": 5}, "outputs": {"out": 5}}
          ]
        }
        "#,
    )
    .unwrap();

    let report = bench_backends_trace(
        &export,
        &trace,
        BenchBackendsTraceOptions {
            repeat: 1,
            warmup: 0,
            lanes: 2,
            backends: vec![PyrtlBenchBackendKind::JitCpu],
        },
    )
    .unwrap();

    assert_eq!(report.backends.len(), 1);
    assert_eq!(report.backends[0].backend, "jit-cpu");
    assert!(!report.backends[0].available);
    assert!(report.backends[0]
        .error
        .as_deref()
        .unwrap()
        .contains("JIT CPU backend requires"));
    assert!(report.backends[0].replay_ns_samples.is_empty());
}

#[test]
fn bench_threaded_trace_replays_independent_lanes_with_initial_state() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 4},
            {"name": "q", "kind": "register", "bitwidth": 4, "reset_value": 1},
            {"name": "sum", "kind": "wire", "bitwidth": 4},
            {"name": "out", "kind": "output", "bitwidth": 4}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "r", "op_param": null, "args": ["a"], "dests": ["q"]},
            {"index": 1, "op": "+", "op_param": null, "args": ["q", "a"], "dests": ["sum"]},
            {"index": 2, "op": "w", "op_param": null, "args": ["sum"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlLaneTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-lane-trace-v1",
          "lanes": 3,
          "steps": [
            {"inputs": {"a": [1, 2, 3]}, "outputs": {"out": [2, 3, 4]}},
            {"inputs": {"a": [4, 5, 6]}, "outputs": {"out": [5, 7, 9]}}
          ]
        }
        "#,
    )
    .unwrap();

    let report = bench_threaded_trace(
        &export,
        &trace,
        BenchThreadedTraceOptions {
            repeat: 1,
            warmup: 0,
            max_workers: 2,
            workers: vec![
                ThreadedReplayWorkerOptions {
                    backend: SimBackendKind::Scalar,
                    lanes: 1,
                },
                ThreadedReplayWorkerOptions {
                    backend: SimBackendKind::PackedCpu,
                    lanes: 2,
                },
            ],
            autotune: false,
            autotune_prune: true,
            plan_first: false,
            planner_calibration: None,
        },
    )
    .unwrap();
    assert_eq!(report.schema, "rrtl-pyrtl-bench-threaded-trace-v1");
    assert_eq!(report.lanes, 3);
    assert_eq!(report.mismatch_count, 0);
    assert_eq!(report.replay.total_lanes, 3);
    assert!(report.autotune.is_none());
}

#[test]
fn bench_threaded_trace_prunes_automatic_autotune_candidates() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 40},
            {"name": "b", "kind": "input", "bitwidth": 40},
            {"name": "out", "kind": "output", "bitwidth": 40}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "+", "op_param": null, "args": ["a", "b"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlLaneTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-lane-trace-v1",
          "lanes": 2,
          "steps": [
            {"inputs": {"a": [1, 3], "b": [2, 4]}, "outputs": {"out": [3, 7]}}
          ]
        }
        "#,
    )
    .unwrap();

    let pruned = bench_threaded_trace(
        &export,
        &trace,
        BenchThreadedTraceOptions {
            repeat: 1,
            warmup: 0,
            max_workers: 2,
            workers: Vec::new(),
            autotune: true,
            autotune_prune: true,
            plan_first: false,
            planner_calibration: None,
        },
    )
    .unwrap();
    assert_eq!(pruned.mismatch_count, 0);
    assert_eq!(
        pruned.backend_affinity.recommendation,
        BackendAffinityRecommendation::SimdCpuCandidate
    );
    assert_eq!(pruned.autotune.as_ref().unwrap().candidates.len(), 5);
    assert!(pruned.autotune_pruned_candidates.is_empty());

    let unpruned = bench_threaded_trace(
        &export,
        &trace,
        BenchThreadedTraceOptions {
            repeat: 1,
            warmup: 0,
            max_workers: 2,
            workers: Vec::new(),
            autotune: true,
            autotune_prune: false,
            plan_first: false,
            planner_calibration: None,
        },
    )
    .unwrap();
    assert_eq!(unpruned.autotune.as_ref().unwrap().candidates.len(), 5);
    assert!(unpruned.autotune_pruned_candidates.is_empty());
}

#[test]
fn plan_backends_reports_static_recommendations_without_replay() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 8},
            {"name": "b", "kind": "input", "bitwidth": 8},
            {"name": "out", "kind": "output", "bitwidth": 8}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "+", "op_param": null, "args": ["a", "b"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlLaneTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-lane-trace-v1",
          "lanes": 4,
          "steps": [
            {"inputs": {"a": [1, 3, 5, 7], "b": [2, 4, 6, 8]}, "outputs": {"out": [3, 7, 11, 15]}}
          ]
        }
        "#,
    )
    .unwrap();

    let report = plan_backends(
        &export,
        &trace,
        PlanBackendsOptions {
            max_workers: 2,
            autotune_prune: true,
            planner_calibration: None,
        },
    )
    .unwrap();

    assert_eq!(report.schema, "rrtl-pyrtl-backend-plan-v1");
    assert_eq!(report.steps, 1);
    assert_eq!(report.lanes, 4);
    assert_eq!(report.max_workers, 2);
    assert_eq!(report.replay_workload.steps, 1);
    assert!(report.replay_workload.estimated_lane_work_units > 0);
    assert!(report.gpu_region_analysis.total.instr_count > 0);
    assert!(report.shader_stats.wgsl_bytes > 0);
    assert!(!report.backend_candidates.is_empty());
    assert_eq!(
        report.selected_runtime_backend,
        report.backend_candidates[0].backend
    );
    assert_eq!(
        report.selected_runtime_reason,
        report.backend_candidates[0].reasons[0]
    );
    assert_eq!(
        report.pruned_runtime_candidates.len(),
        report.backend_candidates.len() - 1
    );
    assert!(!report.recommended_threaded_layouts.is_empty());
    assert_eq!(
        report.selected_threaded_layout,
        report.recommended_threaded_layouts[0]
    );
    assert!(report.selected_reason.ends_with("threaded-layout"));
}

#[test]
fn plan_backends_calibration_reorders_threaded_candidates() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 8},
            {"name": "b", "kind": "input", "bitwidth": 8},
            {"name": "out", "kind": "output", "bitwidth": 8}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "+", "op_param": null, "args": ["a", "b"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlLaneTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-lane-trace-v1",
          "lanes": 4,
          "steps": [
            {"inputs": {"a": [1, 3, 5, 7], "b": [2, 4, 6, 8]}, "outputs": {"out": [3, 7, 11, 15]}}
          ]
        }
        "#,
    )
    .unwrap();

    let baseline = plan_backends(
        &export,
        &trace,
        PlanBackendsOptions {
            max_workers: 2,
            autotune_prune: true,
            planner_calibration: None,
        },
    )
    .unwrap();
    assert!(baseline.recommended_threaded_layouts.len() > 1);
    let preferred_signature = threaded_signature(&baseline.recommended_threaded_layouts[1]);
    let calibrated = plan_backends(
        &export,
        &trace,
        PlanBackendsOptions {
            max_workers: 2,
            autotune_prune: true,
            planner_calibration: Some(planner_calibration(vec![&preferred_signature], vec![])),
        },
    )
    .unwrap();

    assert_eq!(
        threaded_signature(&calibrated.selected_threaded_layout),
        preferred_signature
    );
    assert_eq!(
        calibrated.selected_reason,
        "calibrated-hybrid-threaded-layout"
    );
    assert_eq!(
        calibrated.planner_calibration_schema.as_deref(),
        Some("rrtl-pyrtl-planner-calibration-v1")
    );
}

#[test]
fn plan_backends_selects_gpu_for_compute_heavy_design() {
    let mut wires = vec![
        r#"{"name":"a","kind":"input","bitwidth":8}"#.to_string(),
        r#"{"name":"b","kind":"input","bitwidth":8}"#.to_string(),
        r#"{"name":"out","kind":"output","bitwidth":8}"#.to_string(),
    ];
    let mut nets = Vec::new();
    let mut prev = "a".to_string();
    for index in 0..80 {
        let dst = format!("w{index}");
        wires.push(format!(r#"{{"name":"{dst}","kind":"wire","bitwidth":8}}"#));
        nets.push(format!(
            r#"{{"index":{index},"op":"+","op_param":null,"args":["{prev}","b"],"dests":["{dst}"]}}"#
        ));
        prev = dst;
    }
    nets.push(format!(
        r#"{{"index":80,"op":"w","op_param":null,"args":["{prev}"],"dests":["out"]}}"#
    ));
    let export = load_export(&format!(
        r#"{{
          "schema":"rrtl-pyrtl-block-v1",
          "top_name":"Top",
          "clock_name":"clk",
          "wires":[{}],
          "memories":[],
          "nets":[{}]
        }}"#,
        wires.join(","),
        nets.join(",")
    ));
    let trace: PyrtlLaneTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-lane-trace-v1",
          "lanes": 4,
          "steps": [
            {"inputs": {"a": [1, 3, 5, 7], "b": [2, 4, 6, 8]}, "outputs": {"out": [161, 67, 229, 135]}}
          ]
        }
        "#,
    )
    .unwrap();

    let report = plan_backends(
        &export,
        &trace,
        PlanBackendsOptions {
            max_workers: 2,
            autotune_prune: true,
            planner_calibration: None,
        },
    )
    .unwrap();
    assert!(report.selected_gpu_options.is_some());
    assert_eq!(
        report.selected_gpu_reason.as_deref(),
        Some("gpu-compute-candidate")
    );
    assert!(report.recommended_gpu_options.len() >= 5);
    assert!(report.pruned_gpu_options.is_empty());
    assert_eq!(
        report.selected_gpu_options,
        report
            .recommended_gpu_options
            .first()
            .map(|item| item.options)
    );
    assert!(report
        .recommended_gpu_options
        .iter()
        .all(|item| item.shader_stats.wgsl_bytes > 0));

    let hot_gpu = plan_backends(
        &export,
        &trace,
        PlanBackendsOptions {
            max_workers: 2,
            autotune_prune: true,
            planner_calibration: Some(planner_hot_backend_calibration("rrtl_gpu_measured_trace")),
        },
    )
    .unwrap();
    assert!(hot_gpu.selected_gpu_options.is_some());
    assert_eq!(
        hot_gpu
            .planner_calibration_hot_backend_preference
            .as_deref(),
        Some("rrtl_gpu_measured_trace")
    );
    assert_eq!(
        hot_gpu.planner_calibration_hot_backend_reason.as_deref(),
        Some("hot-backend-gpu-preferred")
    );
    assert_eq!(hot_gpu.planner_calibration_hot_backend_score, Some(100.0));

    let hot_cpu = plan_backends(
        &export,
        &trace,
        PlanBackendsOptions {
            max_workers: 2,
            autotune_prune: true,
            planner_calibration: Some(planner_hot_backend_calibration("rrtl_backend:simd-cpu")),
        },
    )
    .unwrap();
    assert!(hot_cpu.selected_gpu_options.is_none());
    assert_eq!(
        hot_cpu.selected_gpu_reason.as_deref(),
        Some("hot-backend-prefers-cpu")
    );
    assert_eq!(
        hot_cpu.planner_calibration_hot_backend_reason.as_deref(),
        Some("hot-backend-direct-backend-preferred")
    );

    let gpu = bench_gpu_trace(
        &export,
        &trace,
        BenchGpuTraceOptions {
            repeat: 1,
            warmup: 0,
            workgroup_size: 256,
            memory_layout: GpuMemoryLayout::WordMajor,
            reuse_temporaries: false,
            fused: true,
            max_mismatches: 16,
            plan_first: true,
            planner_calibration: None,
        },
    )
    .unwrap();
    let selected = report.selected_gpu_options.unwrap();
    if gpu.available {
        assert_eq!(gpu.shader_stats.workgroup_size, selected.workgroup_size);
        assert_eq!(gpu.shader_stats.memory_layout, selected.memory_layout);
        assert_eq!(
            gpu.shader_stats.reuse_temporaries,
            selected.reuse_temporaries
        );
    } else {
        assert_eq!(
            gpu.error.as_deref(),
            Some("gpu-not-selected-by-profitability")
        );
    }
}

#[test]
fn plan_backends_calibration_reorders_gpu_options_but_respects_blocked_gpu() {
    let mut wires = vec![
        r#"{"name":"a","kind":"input","bitwidth":8}"#.to_string(),
        r#"{"name":"b","kind":"input","bitwidth":8}"#.to_string(),
        r#"{"name":"out","kind":"output","bitwidth":8}"#.to_string(),
    ];
    let mut nets = Vec::new();
    let mut prev = "a".to_string();
    for index in 0..80 {
        let dst = format!("w{index}");
        wires.push(format!(r#"{{"name":"{dst}","kind":"wire","bitwidth":8}}"#));
        nets.push(format!(
            r#"{{"index":{index},"op":"+","op_param":null,"args":["{prev}","b"],"dests":["{dst}"]}}"#
        ));
        prev = dst;
    }
    nets.push(format!(
        r#"{{"index":80,"op":"w","op_param":null,"args":["{prev}"],"dests":["out"]}}"#
    ));
    let export = load_export(&format!(
        r#"{{
          "schema":"rrtl-pyrtl-block-v1",
          "top_name":"Top",
          "clock_name":"clk",
          "wires":[{}],
          "memories":[],
          "nets":[{}]
        }}"#,
        wires.join(","),
        nets.join(",")
    ));
    let trace: PyrtlLaneTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-lane-trace-v1",
          "lanes": 4,
          "steps": [
            {"inputs": {"a": [1, 3, 5, 7], "b": [2, 4, 6, 8]}, "outputs": {"out": [161, 67, 229, 135]}}
          ]
        }
        "#,
    )
    .unwrap();
    let calibrated = plan_backends(
        &export,
        &trace,
        PlanBackendsOptions {
            max_workers: 2,
            autotune_prune: true,
            planner_calibration: Some(planner_calibration(
                vec![],
                vec!["workgroup=128,memory=word-major,reuse=false"],
            )),
        },
    )
    .unwrap();
    let selected = calibrated.selected_gpu_options.unwrap();
    assert_eq!(selected.workgroup_size, 128);
    assert_eq!(selected.memory_layout, GpuMemoryLayout::WordMajor);
    assert_eq!(
        calibrated.planner_calibration_gpu_reason.as_deref(),
        Some("gpu-option-calibration-applied")
    );

    let memory_export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "MemoryTop",
          "clock_name": "clk",
          "wires": [
            {"name": "we", "kind": "input", "bitwidth": 1},
            {"name": "addr", "kind": "input", "bitwidth": 2},
            {"name": "data", "kind": "input", "bitwidth": 40},
            {"name": "out", "kind": "output", "bitwidth": 40}
          ],
          "memories": [
            {"name": "mem", "id": 0, "kind": "mem", "bitwidth": 40, "addrwidth": 2}
          ],
          "nets": [
            {"index": 0, "op": "m", "op_param": {"memory_id": 0}, "args": ["addr"], "dests": ["out"]},
            {"index": 1, "op": "@", "op_param": {"memory_id": 0}, "args": ["addr", "data", "we"], "dests": []}
          ]
        }
        "#,
    );
    let memory_trace: PyrtlLaneTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-lane-trace-v1",
          "lanes": 2,
          "steps": [
            {"inputs": {"we": [0, 0], "addr": [0, 1], "data": [3, 7]}, "outputs": {"out": [0, 0]}}
          ]
        }
        "#,
    )
    .unwrap();
    let blocked = plan_backends(
        &memory_export,
        &memory_trace,
        PlanBackendsOptions {
            max_workers: 2,
            autotune_prune: true,
            planner_calibration: Some(planner_calibration(
                vec![],
                vec!["workgroup=128,memory=word-major,reuse=false"],
            )),
        },
    )
    .unwrap();
    assert!(blocked.selected_gpu_options.is_none());
    assert_eq!(
        blocked.selected_gpu_reason.as_deref(),
        Some("gpu-memory-blocked")
    );

    let hot_gpu_blocked = plan_backends(
        &memory_export,
        &memory_trace,
        PlanBackendsOptions {
            max_workers: 2,
            autotune_prune: true,
            planner_calibration: Some(planner_hot_backend_calibration("rrtl_gpu_measured_trace")),
        },
    )
    .unwrap();
    assert!(hot_gpu_blocked.selected_gpu_options.is_none());
    assert_eq!(
        hot_gpu_blocked
            .planner_calibration_hot_backend_reason
            .as_deref(),
        Some("hot-backend-gpu-blocked")
    );
    assert_eq!(
        hot_gpu_blocked
            .planner_calibration_hot_backend_preference
            .as_deref(),
        Some("rrtl_gpu_measured_trace")
    );
}

#[test]
fn bench_gpu_options_sweeps_planned_gpu_candidates() {
    let mut wires = vec![
        r#"{"name":"clk","kind":"input","bitwidth":1}"#.to_string(),
        r#"{"name":"a","kind":"input","bitwidth":8}"#.to_string(),
        r#"{"name":"b","kind":"input","bitwidth":8}"#.to_string(),
        r#"{"name":"out","kind":"output","bitwidth":8}"#.to_string(),
    ];
    let mut nets = Vec::new();
    let mut prev = "a".to_string();
    for index in 0..80 {
        let dst = format!("w{index}");
        wires.push(format!(r#"{{"name":"{dst}","kind":"wire","bitwidth":8}}"#));
        nets.push(format!(
            r#"{{"index":{index},"op":"+","op_param":null,"args":["{prev}","b"],"dests":["{dst}"]}}"#
        ));
        prev = dst;
    }
    nets.push(format!(
        r#"{{"index":80,"op":"w","op_param":null,"args":["{prev}"],"dests":["out"]}}"#
    ));
    let export = load_export(&format!(
        r#"{{
          "schema":"rrtl-pyrtl-block-v1",
          "top_name":"Top",
          "clock_name":"clk",
          "wires":[{}],
          "memories":[],
          "nets":[{}]
        }}"#,
        wires.join(","),
        nets.join(",")
    ));
    let trace: PyrtlLaneTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-lane-trace-v1",
          "lanes": 4,
          "steps": [
            {"inputs": {"a": [1, 3, 5, 7], "b": [2, 4, 6, 8]}, "outputs": {"out": [161, 67, 229, 135]}}
          ]
        }
        "#,
    )
    .unwrap();

    let planned = plan_backends(
        &export,
        &trace,
        PlanBackendsOptions {
            max_workers: 2,
            autotune_prune: true,
            planner_calibration: None,
        },
    )
    .unwrap();
    let sweep = bench_gpu_options(
        &export,
        &trace,
        BenchGpuTraceOptions {
            repeat: 1,
            warmup: 0,
            workgroup_size: 64,
            memory_layout: GpuMemoryLayout::LaneMajor,
            reuse_temporaries: false,
            fused: true,
            max_mismatches: 16,
            plan_first: false,
            planner_calibration: None,
        },
    )
    .unwrap();

    assert_eq!(sweep.schema, "rrtl-pyrtl-bench-gpu-options-v1");
    assert_eq!(
        sweep.candidates.len(),
        planned.recommended_gpu_options.len()
    );
    assert_eq!(sweep.lanes, 4);
    assert!(sweep
        .candidates
        .iter()
        .zip(planned.recommended_gpu_options.iter())
        .all(|(actual, planned)| actual.planned.options == planned.options));
    if let Some(first) = sweep.candidates.first() {
        assert!(sweep
            .candidates
            .iter()
            .all(|candidate| candidate.report.import_ns == first.report.import_ns));
    }
    if let Some(selected) = sweep.selected_candidate_index {
        let candidate = &sweep.candidates[selected];
        assert!(candidate.available);
        assert_eq!(candidate.mismatch_count, 0);
        assert!(candidate.replay_ns_best > 0);
    } else {
        assert!(sweep
            .candidates
            .iter()
            .all(|candidate| !candidate.available || candidate.replay_ns_best == 0));
    }
}

#[test]
fn bench_gpu_combined_reports_static_sweep_and_measured_sections() {
    let mut wires = vec![
        r#"{"name":"clk","kind":"input","bitwidth":1}"#.to_string(),
        r#"{"name":"a","kind":"input","bitwidth":8}"#.to_string(),
        r#"{"name":"b","kind":"input","bitwidth":8}"#.to_string(),
        r#"{"name":"out","kind":"output","bitwidth":8}"#.to_string(),
    ];
    let mut nets = Vec::new();
    let mut prev = "a".to_string();
    for index in 0..80 {
        let dst = format!("w{index}");
        wires.push(format!(r#"{{"name":"{dst}","kind":"wire","bitwidth":8}}"#));
        nets.push(format!(
            r#"{{"index":{index},"op":"+","op_param":null,"args":["{prev}","b"],"dests":["{dst}"]}}"#
        ));
        prev = dst;
    }
    nets.push(format!(
        r#"{{"index":80,"op":"w","op_param":null,"args":["{prev}"],"dests":["out"]}}"#
    ));
    let export = load_export(&format!(
        r#"{{
          "schema":"rrtl-pyrtl-block-v1",
          "top_name":"Top",
          "clock_name":"clk",
          "wires":[{}],
          "memories":[],
          "nets":[{}]
        }}"#,
        wires.join(","),
        nets.join(",")
    ));
    let trace: PyrtlLaneTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-lane-trace-v1",
          "lanes": 4,
          "steps": [
            {"inputs": {"a": [1, 3, 5, 7], "b": [2, 4, 6, 8]}, "outputs": {"out": [161, 67, 229, 135]}}
          ]
        }
        "#,
    )
    .unwrap();

    let report = bench_gpu_combined(
        &export,
        &trace,
        BenchGpuTraceOptions {
            repeat: 1,
            warmup: 0,
            workgroup_size: 64,
            memory_layout: GpuMemoryLayout::LaneMajor,
            reuse_temporaries: false,
            fused: true,
            max_mismatches: 16,
            plan_first: false,
            planner_calibration: None,
        },
    )
    .unwrap();

    assert_eq!(report.schema, "rrtl-pyrtl-bench-gpu-combined-v1");
    assert_eq!(report.static_trace.schema, "rrtl-pyrtl-bench-gpu-trace-v1");
    assert_eq!(
        report.option_sweep.schema,
        "rrtl-pyrtl-bench-gpu-options-v1"
    );
    assert_eq!(
        report.measured_trace.schema,
        "rrtl-pyrtl-bench-gpu-trace-v1"
    );
    assert!(!report.option_sweep.candidates.is_empty());
    if let Some(selected) = report.option_sweep.selected_candidate_index {
        let candidate = &report.option_sweep.candidates[selected];
        if candidate.available && candidate.mismatch_count == 0 && candidate.replay_ns_best > 0 {
            assert_eq!(
                report.measured_trace.shader_stats.workgroup_size,
                candidate.planned.options.workgroup_size
            );
            assert_eq!(
                report.measured_trace.shader_stats.memory_layout,
                candidate.planned.options.memory_layout
            );
            assert_eq!(
                report.measured_trace.shader_stats.reuse_temporaries,
                candidate.planned.options.reuse_temporaries
            );
            assert!(report.measured_trace.prepared_trace_bytes > 0);
            assert!(report.measured_trace.prepared_snapshot_setup_ns > 0);
            assert!(
                report.measured_trace.hot_restore_ns_median
                    >= report.measured_trace.hot_restore_ns_best
            );
            assert!(report.measured_trace.gpu_timing.count_readback_ns > 0);
            assert_eq!(report.measured_trace.gpu_timing.full_readback_words, 0);
            assert!(report.measured_trace.gpu_timing.single_submit_used);
            assert!(report.measured_trace.gpu_timing.single_submit_ns > 0);
        } else {
            assert!(!report.measured_trace.available);
        }
    } else {
        assert!(!report.measured_trace.available);
    }
}

#[test]
fn bench_threaded_trace_plan_first_uses_selected_layout() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 8},
            {"name": "b", "kind": "input", "bitwidth": 8},
            {"name": "out", "kind": "output", "bitwidth": 8}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "+", "op_param": null, "args": ["a", "b"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlLaneTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-lane-trace-v1",
          "lanes": 4,
          "steps": [
            {"inputs": {"a": [1, 3, 5, 7], "b": [2, 4, 6, 8]}, "outputs": {"out": [3, 7, 11, 15]}}
          ]
        }
        "#,
    )
    .unwrap();

    let planned = plan_backends(
        &export,
        &trace,
        PlanBackendsOptions {
            max_workers: 2,
            autotune_prune: true,
            planner_calibration: None,
        },
    )
    .unwrap();
    let report = bench_threaded_trace(
        &export,
        &trace,
        BenchThreadedTraceOptions {
            repeat: 1,
            warmup: 0,
            max_workers: 2,
            workers: Vec::new(),
            autotune: false,
            autotune_prune: true,
            plan_first: true,
            planner_calibration: None,
        },
    )
    .unwrap();

    assert_eq!(report.mismatch_count, 0);
    assert!(report.autotune.is_none());
    if planned.selected_runtime_backend == "threaded-mixed" {
        assert_eq!(
            report.selected_threaded_layout,
            planned.selected_threaded_layout
        );
        assert!(report
            .selected_reason
            .starts_with("profitability-threaded-mixed"));
    } else {
        assert_eq!(report.selected_threaded_layout.workers.len(), 1);
        assert_eq!(
            report.selected_threaded_layout.workers[0].lanes,
            trace.lanes
        );
        assert!(report.selected_reason.starts_with("profitability-direct-"));
    }
}

#[test]
fn bench_gpu_trace_reports_shape_and_availability() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 8},
            {"name": "b", "kind": "input", "bitwidth": 8},
            {"name": "out", "kind": "output", "bitwidth": 8}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "+", "op_param": null, "args": ["a", "b"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlLaneTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-lane-trace-v1",
          "lanes": 2,
          "steps": [
            {"inputs": {"a": [1, 3], "b": [2, 4]}, "outputs": {"out": [3, 7]}},
            {"inputs": {"a": [5, 7], "b": [8, 9]}, "outputs": {"out": [13, 16]}}
          ]
        }
        "#,
    )
    .unwrap();

    let report = bench_gpu_trace(
        &export,
        &trace,
        BenchGpuTraceOptions {
            repeat: 1,
            warmup: 0,
            workgroup_size: 64,
            memory_layout: GpuMemoryLayout::LaneMajor,
            reuse_temporaries: false,
            fused: true,
            max_mismatches: 16,
            plan_first: false,
            planner_calibration: None,
        },
    )
    .unwrap();
    assert_eq!(report.schema, "rrtl-pyrtl-bench-gpu-trace-v1");
    assert_eq!(report.lanes, 2);
    assert_eq!(report.gpu_replay_mode, "fused-kernel");
    assert!(report.shader_stats.wgsl_bytes > 0);
    assert!(report.gpu_region_analysis.total.instr_count > 0);
    if report.available {
        assert_eq!(report.error, None);
        assert_eq!(report.mismatch_count, 0);
        assert_eq!(report.replay_ns_samples.len(), 1);
        assert!(report.prepared_trace_bytes > 0);
        assert!(report.prepared_snapshot_setup_ns > 0);
        assert!(report.hot_restore_ns_median >= report.hot_restore_ns_best);
        assert!(report.gpu_timing.count_readback_ns > 0);
        assert_eq!(report.gpu_timing.full_readback_ns, 0);
        assert_eq!(report.gpu_timing.full_readback_words, 0);
        assert!(report.gpu_timing.single_submit_used);
        assert!(report.gpu_timing.single_submit_ns > 0);
    } else {
        assert!(report.error.is_some());
        assert!(report.replay_ns_samples.is_empty());
        assert_eq!(report.prepared_snapshot_setup_ns, 0);
        assert_eq!(report.hot_restore_ns_best, 0);
        assert_eq!(report.hot_restore_ns_median, 0);
    }
}

#[test]
fn bench_gpu_trace_plan_first_reports_unselected_memory_hostile_design() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "we", "kind": "input", "bitwidth": 1},
            {"name": "addr", "kind": "input", "bitwidth": 2},
            {"name": "data", "kind": "input", "bitwidth": 40},
            {"name": "out", "kind": "output", "bitwidth": 40}
          ],
          "memories": [
            {"name": "mem", "id": 0, "kind": "mem", "bitwidth": 40, "addrwidth": 2}
          ],
          "nets": [
            {"index": 0, "op": "m", "op_param": {"memory_id": 0}, "args": ["addr"], "dests": ["out"]},
            {"index": 1, "op": "@", "op_param": {"memory_id": 0}, "args": ["addr", "data", "we"], "dests": []}
          ]
        }
        "#,
    );
    let trace: PyrtlLaneTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-lane-trace-v1",
          "lanes": 2,
          "steps": [
            {"inputs": {"we": [0, 0], "addr": [0, 1], "data": [3, 7]}, "outputs": {"out": [0, 0]}}
          ]
        }
        "#,
    )
    .unwrap();

    let plan = plan_backends(
        &export,
        &trace,
        PlanBackendsOptions {
            max_workers: 2,
            autotune_prune: true,
            planner_calibration: None,
        },
    )
    .unwrap();
    assert!(plan.selected_gpu_options.is_none());
    assert!(plan.recommended_gpu_options.is_empty());
    assert!(!plan.pruned_gpu_options.is_empty());
    assert_eq!(
        plan.selected_gpu_reason.as_deref(),
        Some("gpu-memory-blocked")
    );

    let report = bench_gpu_trace(
        &export,
        &trace,
        BenchGpuTraceOptions {
            repeat: 1,
            warmup: 0,
            workgroup_size: 64,
            memory_layout: GpuMemoryLayout::LaneMajor,
            reuse_temporaries: false,
            fused: true,
            max_mismatches: 16,
            plan_first: true,
            planner_calibration: None,
        },
    )
    .unwrap();
    assert!(!report.available);
    assert_eq!(
        report.error.as_deref(),
        Some("gpu-not-selected-by-profitability")
    );
    assert!(report.replay_ns_samples.is_empty());
}

#[test]
fn bench_threaded_trace_reports_global_lane_mismatch() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 4},
            {"name": "out", "kind": "output", "bitwidth": 4}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "w", "op_param": null, "args": ["a"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlLaneTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-lane-trace-v1",
          "lanes": 2,
          "steps": [
            {"inputs": {"a": [7, 8]}, "outputs": {"out": [7, 9]}}
          ]
        }
        "#,
    )
    .unwrap();

    let report = bench_threaded_trace(
        &export,
        &trace,
        BenchThreadedTraceOptions {
            repeat: 1,
            warmup: 0,
            max_workers: 1,
            workers: vec![ThreadedReplayWorkerOptions {
                backend: SimBackendKind::PackedCpu,
                lanes: 2,
            }],
            autotune: false,
            autotune_prune: true,
            plan_first: false,
            planner_calibration: None,
        },
    )
    .unwrap();
    assert_eq!(report.mismatch_count, 1);
    assert_eq!(report.mismatches[0].signal, "out");
    assert_eq!(report.mismatches[0].lane, Some(1));
}

#[test]
fn bench_packed_trace_cli_rejects_zero_lanes() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let export_path = std::env::temp_dir().join(format!(
        "rrtl-pyrtl-cli-{unique}-{}.json",
        std::process::id()
    ));
    fs::write(
        &export_path,
        r#"{"schema":"rrtl-pyrtl-block-v1","top_name":"Top","clock_name":"clk","wires":[],"memories":[],"nets":[]}"#,
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_pyrtl2rrtl"))
        .arg("bench-packed-trace")
        .arg(&export_path)
        .arg("missing.trace.json")
        .arg("--lanes")
        .arg("0")
        .output()
        .unwrap();
    let _ = fs::remove_file(&export_path);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--lanes must be greater than zero"));
}

#[test]
fn bench_profile_replay_cli_runs_measured_cpu_profile() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let base = std::env::temp_dir().join(format!(
        "rrtl-pyrtl-profile-replay-{unique}-{}",
        std::process::id()
    ));
    let export_path = base.with_extension("export.json");
    let trace_path = base.with_extension("trace.json");
    let lane_trace_path = base.with_extension("lane_trace.json");
    let profile_path = base.with_extension("runtime_profile.json");
    fs::write(
        &export_path,
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 4},
            {"name": "q", "kind": "register", "bitwidth": 4, "reset_value": 1},
            {"name": "out", "kind": "output", "bitwidth": 4}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "r", "op_param": null, "args": ["a"], "dests": ["q"]},
            {"index": 1, "op": "w", "op_param": null, "args": ["q"], "dests": ["out"]}
          ]
        }
        "#,
    )
    .unwrap();
    fs::write(
        &trace_path,
        r#"
        {
          "schema": "rrtl-pyrtl-trace-v1",
          "steps": [
            {"inputs": {"a": 5}, "outputs": {"out": 1}},
            {"inputs": {"a": 9}, "outputs": {"out": 5}}
          ]
        }
        "#,
    )
    .unwrap();
    fs::write(
        &lane_trace_path,
        r#"
        {
          "schema": "rrtl-pyrtl-lane-trace-v1",
          "lanes": 2,
          "steps": [
            {"inputs": {"a": [5, 6]}, "outputs": {"out": [1, 1]}},
            {"inputs": {"a": [9, 10]}, "outputs": {"out": [5, 6]}}
          ]
        }
        "#,
    )
    .unwrap();
    fs::write(
        &profile_path,
        r#"
        {
          "schema": "rrtl-pyrtl-runtime-profile-v1",
          "recommended_runtime_backend": "rrtl_backend:simd-cpu",
          "recommended_runtime_source": "measured-cpu",
          "selected_backend": {"backend": "rrtl_backend:simd-cpu"}
        }
        "#,
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_pyrtl2rrtl"))
        .arg("bench-profile-replay")
        .arg(&export_path)
        .arg(&trace_path)
        .arg(&lane_trace_path)
        .arg(&profile_path)
        .arg("--repeat")
        .arg("1")
        .arg("--warmup")
        .arg("0")
        .arg("--lanes")
        .arg("2")
        .output()
        .unwrap();
    let _ = fs::remove_file(&export_path);
    let _ = fs::remove_file(&trace_path);
    let _ = fs::remove_file(&lane_trace_path);
    let _ = fs::remove_file(&profile_path);

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["schema"], "rrtl-pyrtl-profile-replay-hot-v1");
    assert_eq!(report["selected_backend"], "rrtl_backend:simd-cpu");
    assert_eq!(report["mismatch_count"], 0);
    assert!(report["replay_ns_best"].as_u64().unwrap() > 0);
    assert!(report["setup_ns_total"].as_u64().unwrap() > 0);
    assert!(report["first_replay_ns"].as_u64().unwrap() > 0);
    assert_eq!(report["hot_replay_ns_best"], report["replay_ns_best"]);
    assert!(report["replay_ns_per_lane_step"].as_f64().unwrap() > 0.0);
}

#[test]
fn profile_replay_api_runs_measured_cpu_profile_hot_loop() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 4},
            {"name": "q", "kind": "register", "bitwidth": 4, "reset_value": 1},
            {"name": "out", "kind": "output", "bitwidth": 4}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "r", "op_param": null, "args": ["a"], "dests": ["q"]},
            {"index": 1, "op": "w", "op_param": null, "args": ["q"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-trace-v1",
          "steps": [
            {"inputs": {"a": 5}, "outputs": {"out": 1}},
            {"inputs": {"a": 9}, "outputs": {"out": 5}}
          ]
        }
        "#,
    )
    .unwrap();
    let lane_trace: PyrtlLaneTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-lane-trace-v1",
          "lanes": 2,
          "steps": [
            {"inputs": {"a": [5, 6]}, "outputs": {"out": [1, 1]}},
            {"inputs": {"a": [9, 10]}, "outputs": {"out": [5, 6]}}
          ]
        }
        "#,
    )
    .unwrap();
    let profile = RuntimeProfile {
        schema: "rrtl-pyrtl-runtime-profile-v1".to_string(),
        recommended_runtime_backend: Some("rrtl_backend:simd-cpu".to_string()),
        recommended_runtime_source: Some("measured-cpu".to_string()),
        selected_backend: Some(RuntimeProfileSelectedBackend {
            selected_threaded_layout: None,
            selected_gpu_options: None,
        }),
    };

    let report = profile_replay(
        &export,
        &trace,
        &lane_trace,
        &profile,
        ProfileReplayOptions {
            repeat: 1,
            warmup: 0,
            lanes: 2,
        },
    )
    .unwrap();

    assert_eq!(report.schema, "rrtl-pyrtl-profile-replay-hot-v1");
    assert_eq!(report.selected_backend, "rrtl_backend:simd-cpu");
    assert_eq!(report.mismatch_count, 0);
    assert!(report.replay_ns_best > 0);
    assert!(report.setup_ns_total >= report.import_ns + report.setup_ns + report.runner_setup_ns);
    assert!(report.first_replay_ns > 0);
    assert_eq!(report.hot_replay_ns_best, report.replay_ns_best);
    assert_eq!(report.hot_replay_ns_median, report.replay_ns_median);
    assert!(report.replay_ns_per_step > 0.0);
}

#[test]
fn profile_replay_api_runs_threaded_selection_hot_loop() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 4},
            {"name": "q", "kind": "register", "bitwidth": 4, "reset_value": 1},
            {"name": "out", "kind": "output", "bitwidth": 4}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "r", "op_param": null, "args": ["a"], "dests": ["q"]},
            {"index": 1, "op": "w", "op_param": null, "args": ["q"], "dests": ["out"]}
          ]
        }
        "#,
    );
    let trace: PyrtlTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-trace-v1",
          "steps": [
            {"inputs": {"a": 5}, "outputs": {"out": 1}},
            {"inputs": {"a": 9}, "outputs": {"out": 5}}
          ]
        }
        "#,
    )
    .unwrap();
    let lane_trace: PyrtlLaneTrace = serde_json::from_str(
        r#"
        {
          "schema": "rrtl-pyrtl-lane-trace-v1",
          "lanes": 2,
          "steps": [
            {"inputs": {"a": [5, 6]}, "outputs": {"out": [1, 1]}},
            {"inputs": {"a": [9, 10]}, "outputs": {"out": [5, 6]}}
          ]
        }
        "#,
    )
    .unwrap();
    let profile = RuntimeProfile {
        schema: "rrtl-pyrtl-runtime-profile-v1".to_string(),
        recommended_runtime_backend: Some("rrtl_threaded_autotune_trace".to_string()),
        recommended_runtime_source: Some("measured-cpu".to_string()),
        selected_backend: Some(RuntimeProfileSelectedBackend {
            selected_threaded_layout: Some(RuntimeProfileThreadedLayout {
                workers: vec![RuntimeProfileWorker {
                    backend: "simd-cpu".to_string(),
                    lanes: 2,
                }],
            }),
            selected_gpu_options: None,
        }),
    };

    let report = profile_replay(
        &export,
        &trace,
        &lane_trace,
        &profile,
        ProfileReplayOptions {
            repeat: 1,
            warmup: 0,
            lanes: 2,
        },
    )
    .unwrap();

    assert_eq!(report.schema, "rrtl-pyrtl-profile-replay-hot-v1");
    assert_eq!(report.selected_backend, "rrtl_threaded_autotune_trace");
    assert_eq!(report.mismatch_count, 0);
    assert!(report.replay_ns_best > 0);
    assert!(report.setup_ns_total >= report.import_ns + report.setup_ns + report.runner_setup_ns);
    assert!(report.first_replay_ns > 0);
    assert_eq!(report.hot_replay_ns_best, report.replay_ns_best);
    assert!(report.replay_ns_per_lane_step > 0.0);
}

#[test]
fn imports_rewritten_ops_and_noncontiguous_selects() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 4},
            {"name": "b", "kind": "input", "bitwidth": 4},
            {"name": "nand_out", "kind": "output", "bitwidth": 4},
            {"name": "gt_out", "kind": "output", "bitwidth": 1},
            {"name": "pick", "kind": "output", "bitwidth": 2}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "n", "op_param": null, "args": ["a", "b"], "dests": ["nand_out"]},
            {"index": 1, "op": ">", "op_param": null, "args": ["a", "b"], "dests": ["gt_out"]},
            {"index": 2, "op": "s", "op_param": [0, 2], "args": ["a"], "dests": ["pick"]}
          ]
        }
        "#,
    );

    let imported = import_export(&export).unwrap();
    let design = imported.design;
    let a = design.find_signal("Top", "a").unwrap();
    let b = design.find_signal("Top", "b").unwrap();
    let nand_out = design.find_signal("Top", "nand_out").unwrap();
    let gt_out = design.find_signal("Top", "gt_out").unwrap();
    let pick = design.find_signal("Top", "pick").unwrap();
    let mut sim = Simulator::new(&design, "Top").unwrap();
    sim.set(a, 0b1101);
    sim.set(b, 0b0111);
    assert_eq!(sim.get(nand_out), 0b1010);
    assert_eq!(sim.get(gt_out), 1);
    assert_eq!(sim.get(pick), 0b11);
}

#[test]
fn unsupported_op_reports_net_context() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 1},
            {"name": "out", "kind": "output", "bitwidth": 1}
          ],
          "memories": [],
          "nets": [
            {"index": 42, "op": "?", "op_param": null, "args": ["a"], "dests": ["out"]}
          ]
        }
        "#,
    );

    let err = import_export(&export).unwrap_err().to_string();
    assert!(err.contains("net 42 op `?`"), "{err}");
    assert!(err.contains("args [\"a\"]"), "{err}");
    assert!(err.contains("dests [\"out\"]"), "{err}");
    assert!(err.contains("unsupported PyRTL op"), "{err}");
}

#[test]
fn emits_systemverilog_for_imported_initial_values() {
    let export = load_export(
        r#"
        {
          "schema": "rrtl-pyrtl-block-v1",
          "top_name": "Top",
          "clock_name": "clk",
          "wires": [
            {"name": "a", "kind": "input", "bitwidth": 4},
            {"name": "q", "kind": "register", "bitwidth": 4, "reset_value": 2},
            {"name": "out", "kind": "output", "bitwidth": 4}
          ],
          "memories": [],
          "nets": [
            {"index": 0, "op": "r", "op_param": null, "args": ["a"], "dests": ["q"]},
            {"index": 1, "op": "w", "op_param": null, "args": ["q"], "dests": ["out"]}
          ]
        }
        "#,
    );

    let sv = rrtl_pyrtl::emit_systemverilog(&export).unwrap();
    assert!(sv.contains("module Top(a, out, clk);") || sv.contains("module Top(a, q, out, clk);"));
    assert!(sv.contains("initial begin"));
    assert!(sv.contains("q = 4'd2;"));
    assert!(sv.contains("always_ff @(posedge clk)"));
}
