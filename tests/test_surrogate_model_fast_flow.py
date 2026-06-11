import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import rrtl_surrogate.train_cache_miss as train_cache_miss


ROOT = Path(__file__).resolve().parents[1]


def pyrtl2rrtl_command():
    override = os.environ.get("PYRTL2RRTL")
    if override:
        return override.split()
    return ["cargo", "run", "-q", "-p", "rrtl-pyrtl", "--bin", "pyrtl2rrtl", "--"]


def run_checked(command):
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


class TestSurrogateModelFastFlow(unittest.TestCase):
    def test_cache_miss_event_flow_reaches_model_fast_golden_acceptance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            corpus_path = work / "events.json"
            model_dir = work / "model"
            policy_path = work / "policy.json"
            runtime_plan_path = work / "runtime_plan.json"
            fast_run_path = work / "fast_run.json"
            tensor_bundle_path = work / "tensor_bundle.json"
            golden_path = work / "cache0_golden.json"
            model_fast_plan_path = work / "model_fast_plan.json"
            model_fast_report_path = work / "model_fast_report.json"

            corpus = train_cache_miss.generate_corpus(
                samples=8,
                window=4,
                seed=9,
                lanes=2,
            )
            corpus_path.write_text(
                json.dumps(corpus, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            train_cache_miss.train_and_export(
                corpus_path=corpus_path,
                out_dir=model_dir,
                seed=9,
            )
            manifest_path = model_dir / "manifest.json"
            cli = pyrtl2rrtl_command()

            run_checked(
                [
                    *cli,
                    "surrogate",
                    "policy-events",
                    str(manifest_path),
                    str(corpus_path),
                    "--out",
                    str(policy_path),
                ]
            )
            run_checked(
                [
                    *cli,
                    "surrogate",
                    "plan-runtime-events",
                    str(policy_path),
                    "--worker",
                    "cpu-a:1",
                    "--worker",
                    "cpu-b:1",
                    "--out",
                    str(runtime_plan_path),
                ]
            )
            run_checked(
                [
                    *cli,
                    "surrogate",
                    "run-fast-events",
                    str(manifest_path),
                    str(corpus_path),
                    str(runtime_plan_path),
                    "--shadow-sample-stride",
                    "4",
                    "--out",
                    str(fast_run_path),
                ]
            )
            train_cache_miss.write_tensor_bundle(
                corpus_path=corpus_path,
                out_path=tensor_bundle_path,
            )
            train_cache_miss.write_model_fast_golden(
                corpus_path=corpus_path,
                fast_run_path=fast_run_path,
                op_id="cache0",
                out_path=golden_path,
            )
            run_checked(
                [
                    *cli,
                    "surrogate",
                    "plan-model-fast",
                    "--op",
                    f"cache0:{fast_run_path}:Cache miss predictor",
                    "--golden",
                    f"cache0:{golden_path}",
                    "--out",
                    str(model_fast_plan_path),
                ]
            )
            run_checked(
                [
                    *cli,
                    "surrogate",
                    "run-model-fast",
                    str(model_fast_plan_path),
                    "--out",
                    str(model_fast_report_path),
                ]
            )

            fast_run = json.loads(fast_run_path.read_text(encoding="utf-8"))
            tensor_bundle = json.loads(tensor_bundle_path.read_text(encoding="utf-8"))
            report = json.loads(model_fast_report_path.read_text(encoding="utf-8"))

        self.assertTrue(report["ok"], report)
        self.assertEqual(
            len(tensor_bundle["inputs"]["signal_window"]),
            fast_run["count"],
        )
        self.assertEqual(report["op_count"], 1)
        self.assertEqual(report["coverage"]["op_coverage"], 1.0)
        self.assertEqual(report["totals"]["items"], fast_run["count"])
        self.assertEqual(
            report["totals"]["surrogate_replacements"],
            fast_run["surrogate_replacements"],
        )
        self.assertEqual(report["totals"]["exact_fallbacks"], fast_run["exact_fallbacks"])
        self.assertEqual(report["totals"]["fail_closed"], fast_run["fail_closed"])
        self.assertEqual(
            report["totals"]["shadow_sampled"],
            sum(1 for item in fast_run["results"] if item["shadow_sampled"]),
        )

        op = report["ops"][0]
        self.assertEqual(op["op_kind"], "event")
        self.assertTrue(op["ok"], op)
        self.assertEqual(op["totals"], report["totals"])
        self.assertEqual(op["totals"]["items"], 8)
        self.assertEqual(op["totals"]["exact_fallbacks"], 8)
        self.assertEqual(op["totals"]["surrogate_replacements"], 0)
        golden = op["golden"]
        self.assertTrue(golden["golden_compared"])
        self.assertTrue(golden["golden_ok"], golden)
        self.assertTrue(golden["tensor_compared"])
        self.assertEqual(golden["tensor_count"], 1)
        self.assertEqual(golden["max_abs_error"], 0)
        self.assertEqual(golden["tensor_errors"], [])

    def test_instrumented_event_flow_reaches_runtime_attachment_and_model_fast(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            fake = work / "fake_pyrtl2rrtl.py"
            out_dir = work / "flow"
            fake.write_text(_fake_pyrtl2rrtl_script(), encoding="utf-8")

            summary = train_cache_miss.run_instrumented_flow(
                trace_path=ROOT / "examples/surrogate/cache_miss_instrumentation_trace.json",
                config_path=ROOT / "examples/surrogate/cache_miss_emitter_config.json",
                out_dir=out_dir,
                pyrtl2rrtl=["python", str(fake)],
            )

            corpus = json.loads(Path(summary["outputs"]["corpus"]).read_text(encoding="utf-8"))
            tensor_bundle = json.loads(
                Path(summary["outputs"]["tensor_bundle"]).read_text(encoding="utf-8")
            )
            fast_run = json.loads(Path(summary["outputs"]["fast_run"]).read_text(encoding="utf-8"))
            model_fast = json.loads(
                Path(summary["outputs"]["model_fast_report"]).read_text(encoding="utf-8")
            )
            runtime_execution = json.loads(
                Path(summary["outputs"]["runtime_execution"]).read_text(encoding="utf-8")
            )
            quality_gate = json.loads(
                Path(summary["outputs"]["quality_gate"]).read_text(encoding="utf-8")
            )
            runtime_handoff = json.loads(
                Path(summary["outputs"]["runtime_handoff"]).read_text(encoding="utf-8")
            )
            instrumentation_inspection = json.loads(
                Path(summary["outputs"]["instrumentation_inspection"]).read_text(encoding="utf-8")
            )
            instrumentation_match = json.loads(
                Path(summary["outputs"]["instrumentation_use_case_match"]).read_text(
                    encoding="utf-8"
                )
            )
            flow_bundle = json.loads(
                Path(summary["outputs"]["flow_bundle"]).read_text(encoding="utf-8")
            )

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["model"], "rule")
        self.assertEqual(summary["samples"], 2)
        self.assertTrue(flow_bundle["ok"], flow_bundle)
        self.assertEqual(flow_bundle["schema"], "rrtl-surrogate-flow-bundle-v1")
        self.assertEqual(flow_bundle["readiness"]["instrumentation_use_case_match"], True)
        self.assertEqual(flow_bundle["readiness"]["instrumentation_compatible"], True)
        self.assertEqual(flow_bundle["readiness"]["runtime_ready"], True)
        self.assertEqual(flow_bundle["readiness"]["model_fast_ok"], True)
        self.assertEqual(flow_bundle["readiness"]["quality_gate"], True)
        self.assertEqual(flow_bundle["counters"]["emittable_samples"], 2)
        self.assertEqual(flow_bundle["counters"]["exact_fallbacks"], 2)
        self.assertEqual(flow_bundle["use_case"]["target"], "cache_miss")
        self.assertEqual(flow_bundle["use_case"]["surrogate_class"], "event_predictor")
        self.assertEqual(flow_bundle["use_case"]["supported_model_modes"], ["rule", "learned"])
        self.assertEqual(flow_bundle["instrumentation_match"]["ok"], True)
        self.assertEqual(flow_bundle["quality_gate"]["ok"], True)
        self.assertIn("instrumentation_use_case_match", flow_bundle["artifacts"])
        self.assertIn("quality_gate", flow_bundle["artifacts"])
        self.assertIn("runtime_handoff", flow_bundle["artifacts"])
        self.assertIn("runtime_execution", flow_bundle["artifacts"])
        self.assertTrue(instrumentation_inspection["ok"], instrumentation_inspection)
        self.assertEqual(instrumentation_inspection["compatibility"]["emittable_samples"], 2)
        self.assertTrue(instrumentation_match["ok"], instrumentation_match)
        self.assertEqual(instrumentation_match["use_case"]["target"], "cache_miss")
        self.assertTrue(quality_gate["ok"], quality_gate)
        self.assertTrue(quality_gate["checks"]["validation_accuracy"]["ok"])
        self.assertTrue(runtime_handoff["ok"], runtime_handoff)
        self.assertEqual(runtime_handoff["schema"], train_cache_miss.RUNTIME_HANDOFF_SCHEMA)
        self.assertEqual(runtime_handoff["target"], "cache_miss")
        self.assertEqual({event["target"] for event in corpus["events"]}, {"cache_miss"})
        self.assertEqual([event["sample_id"] for event in corpus["events"]], [0, 1])
        self.assertEqual([event["lane"] for event in corpus["events"]], [0, 1])
        self.assertEqual(len(tensor_bundle["inputs"]["signal_window"]), fast_run["count"])
        self.assertTrue(model_fast["ok"], model_fast)
        self.assertEqual(model_fast["op_count"], 1)
        self.assertEqual(model_fast["coverage"]["op_coverage"], 1.0)
        self.assertTrue(runtime_execution["ready"], runtime_execution)
        self.assertEqual(runtime_execution["event_items"][0]["sample_id"], 0)
        self.assertEqual(runtime_execution["event_items"][0]["target"], "cache_miss")

def _fake_pyrtl2rrtl_script():
    return r'''
import json
import sys
from pathlib import Path

SIGNAL_FEATURES = [
    "cycle_delta",
    "load",
    "store",
    "addr_low",
    "cache_set",
    "tag_delta",
    "pending_misses",
    "store_buffer_occupancy",
]
PROGRAM = {"opcode_id": 1, "pc": 4096, "stride": 64, "working_set_log2": 14}

def out_path():
    return Path(sys.argv[sys.argv.index("--out") + 1])

def write(data):
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if "--out" in sys.argv:
        out_path().write_text(text, encoding="utf-8")
    else:
        print(text)

def corpus():
    signal0 = {
        "cycle_delta": 0,
        "load": 1,
        "store": 0,
        "addr_low": 16,
        "cache_set": 1,
        "tag_delta": 1,
        "pending_misses": 2,
        "store_buffer_occupancy": 0,
    }
    signal1 = {
        "cycle_delta": 1,
        "load": 1,
        "store": 0,
        "addr_low": 80,
        "cache_set": 2,
        "tag_delta": 1,
        "pending_misses": 1,
        "store_buffer_occupancy": 0,
    }
    return {
        "schema": "rrtl-surrogate-instrumentation-corpus-v1",
        "source_hash": "fake-instrumented-flow",
        "top_name": "InstrumentedCache",
        "events": [
            {
                "schema": "rrtl-surrogate-instrumentation-event-v1",
                "sample_id": 0,
                "lane": 0,
                "target": "cache_miss",
                "window_cycles": 2,
                "horizon_cycles": 1,
                "program": PROGRAM,
                "signals": [signal0, signal1],
                "label": {"cache_miss": 1},
            },
            {
                "schema": "rrtl-surrogate-instrumentation-event-v1",
                "sample_id": 1,
                "lane": 1,
                "target": "cache_miss",
                "window_cycles": 2,
                "horizon_cycles": 1,
                "program": PROGRAM,
                "signals": [signal1, signal0],
                "label": {"cache_miss": 0},
            },
        ],
    }

cmd = sys.argv[sys.argv.index("surrogate") + 1]
if cmd == "emit-instrumented-events":
    write(corpus())
elif cmd == "inspect-instrumentation":
    write({
        "schema": "rrtl-instrumentation-trace-inspection-v1",
        "ok": True,
        "top_name": "InstrumentedCache",
        "source_hash": "fake-instrumented-flow",
        "steps": 4,
        "cycle_min": 0,
        "cycle_max": 3,
        "cycle_monotonic": True,
        "lanes": [0, 1],
        "lane_count": 2,
        "steps_with_lane": 4,
        "signal_fields": SIGNAL_FEATURES,
        "program_fields": sorted(PROGRAM),
        "label_fields": ["miss"],
        "compatibility": {
            "target": "cache_miss",
            "window_cycles": 2,
            "horizon_cycles": 1,
            "emittable_samples": 2,
            "missing_fields": [],
        },
        "warnings": [],
        "errors": [],
    })
elif cmd == "inspect-events":
    write({
        "schema": "rrtl-surrogate-event-inspection-v1",
        "ok": True,
        "corpus": {"samples": 2, "target": "cache_miss"},
        "targets": ["cache_miss"],
        "signal_features": SIGNAL_FEATURES,
        "program_features": sorted(PROGRAM),
        "positive_labels": {"cache_miss": 1},
        "errors": [],
    })
elif cmd == "validate-events":
    write({"schema": "rrtl-surrogate-event-validation-v1", "ok": True, "metrics": {"accuracy": 1.0}, "errors": []})
elif cmd == "shadow-events":
    write({"schema": "rrtl-surrogate-event-shadow-v1", "ok": True, "metrics": {"accuracy": 1.0}, "errors": []})
elif cmd == "policy-events":
    write({
        "schema": "rrtl-surrogate-event-policy-report-v1",
        "ok": True,
        "results": [
            {"sample_id": 0, "lane": 0, "target": "cache_miss", "decision": "exact_fallback", "predicted": 1, "expected": 1},
            {"sample_id": 1, "lane": 1, "target": "cache_miss", "decision": "exact_fallback", "predicted": 0, "expected": 0},
        ],
        "errors": [],
    })
elif cmd == "plan-runtime-events":
    write({
        "schema": "rrtl-surrogate-event-runtime-plan-v1",
        "ok": True,
        "workers": [{"worker_id": "cpu-a", "lanes": 1}, {"worker_id": "cpu-b", "lanes": 1}],
        "items": [{"sample_id": 0, "lane": 0}, {"sample_id": 1, "lane": 1}],
        "errors": [],
    })
elif cmd == "run-fast-events":
    write({
        "schema": "rrtl-surrogate-event-fast-run-v1",
        "ok": True,
        "count": 2,
        "total_lanes": 2,
        "surrogate_replacements": 0,
        "exact_fallbacks": 2,
        "fail_closed": 0,
        "shadow_compared": 1,
        "shadow_passed": 1,
        "shadow_failed": 0,
        "workers": [],
        "lanes": [],
        "results": [
            {"index": 0, "sample_id": 0, "lane": 0, "target": "cache_miss", "decision": "exact_fallback", "source_result": "exact", "predicted": 1, "expected": 1, "provenance": {"tag": "instrumentation_prediction", "exact": True}, "shadow_sampled": True},
            {"index": 1, "sample_id": 1, "lane": 1, "target": "cache_miss", "decision": "exact_fallback", "source_result": "exact", "predicted": 0, "expected": 0, "provenance": {"tag": "instrumentation_prediction", "exact": True}, "shadow_sampled": False},
        ],
        "errors": [],
    })
elif cmd == "plan-model-fast":
    write({"schema": "rrtl-surrogate-model-fast-plan-v1", "ops": [{"op_id": "cache0"}]})
elif cmd == "run-model-fast":
    write({
        "schema": "rrtl-surrogate-model-fast-report-v1",
        "ok": True,
        "op_count": 1,
        "coverage": {"op_coverage": 1.0},
        "totals": {"items": 2, "surrogate_replacements": 0, "exact_fallbacks": 2, "fail_closed": 0, "shadow_sampled": 1},
        "ops": [{"op_id": "cache0", "op_kind": "event", "ok": True}],
        "errors": [],
    })
elif cmd == "attach-runtime-events":
    write({"schema": "rrtl-runtime-surrogate-attachment-v1", "attached_items": 2})
elif cmd == "inspect-runtime-attachment":
    write({
        "schema": "rrtl-runtime-surrogate-execution-report-v1",
        "ready": True,
        "executed": True,
        "attached_items": 2,
        "event_items": [
            {"index": 0, "sample_id": 0, "lane": 0, "target": "cache_miss", "predicted": 1},
            {"index": 1, "sample_id": 1, "lane": 1, "target": "cache_miss", "predicted": 0},
        ],
        "event_workers": [],
        "event_lanes": [],
        "errors": [],
    })
else:
    raise SystemExit(f"unsupported command {cmd}")
'''


if __name__ == "__main__":
    unittest.main()
