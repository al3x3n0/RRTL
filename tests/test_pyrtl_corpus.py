import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rrtl_pyrtl import corpus


class CorpusReportTests(unittest.TestCase):
    def test_validate_manifest_accepts_importable_callable(self):
        manifest = {
            "targets": [
                {
                    "name": "counter",
                    "target": "rrtl_pyrtl.examples:counter",
                    "top_name": "Counter",
                    "clock_name": "clk",
                    "reset_working_block": True,
                    "inputs": [{"en": 1}, {"en": 0}],
                }
            ]
        }

        self.assertEqual(corpus.validate_manifest(manifest), [])

    def test_validate_manifest_reports_invalid_top_level_shape(self):
        findings = corpus.validate_manifest({"not_targets": []})

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["error"], "manifest must be a list or an object with a 'targets' list")

    def test_validate_manifest_reports_non_object_target_entry(self):
        findings = corpus.validate_manifest(["not an object"])

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["name"], "target-0")
        self.assertEqual(findings[0]["error"], "manifest entry is not an object")

    def test_validate_manifest_reports_malformed_target_string(self):
        findings = corpus.validate_manifest({"targets": [{"name": "bad", "target": "not_a_target"}]})

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["error"], "target must be in module:function form")

    def test_validate_manifest_reports_missing_target_module_or_function(self):
        missing_module = corpus.validate_manifest(
            {"targets": [{"name": "missing", "target": "no_such_module:build"}]}
        )
        missing_function = corpus.validate_manifest(
            {"targets": [{"name": "missing_func", "target": "rrtl_pyrtl.examples:no_such_func"}]}
        )

        self.assertIn("target import failed", missing_module[0]["error"])
        self.assertIn("target import failed", missing_function[0]["error"])

    def test_validate_manifest_reports_duplicate_resolved_names(self):
        findings = corpus.validate_manifest(
            {
                "targets": [
                    {"name": "dup", "target": "rrtl_pyrtl.examples:counter"},
                    {"name": "dup", "target": "rrtl_pyrtl.examples:alu"},
                ]
            }
        )

        self.assertEqual(len(findings), 1)
        self.assertIn("duplicate target name `dup`", findings[0]["error"])

    def test_validate_manifest_reports_invalid_inputs_shape(self):
        findings = corpus.validate_manifest(
            {
                "targets": [
                    {
                        "name": "bad_inputs",
                        "target": "rrtl_pyrtl.examples:counter",
                        "inputs": [{"en": "1"}, ["not", "a", "dict"]],
                    }
                ]
            }
        )

        errors = [finding["error"] for finding in findings]
        self.assertIn("inputs[0]['en'] must be an integer", errors)
        self.assertIn("inputs[1] must be a dictionary", errors)

    def test_validate_manifest_accepts_profile_fields_and_rejects_bad_values(self):
        valid = corpus.validate_manifest(
            {
                "targets": [
                    {
                        "name": "profiled",
                        "target": "rrtl_pyrtl.examples:counter",
                        "inputs": [{"en": 1}],
                        "benchmark_profile": True,
                        "packed_lanes": 2,
                        "repeat": 1,
                        "warmup": 0,
                        "profile_replay_hot_repeat": 4,
                    }
                ]
            }
        )
        invalid = corpus.validate_manifest(
            {
                "targets": [
                    {
                        "name": "bad_profile",
                        "target": "rrtl_pyrtl.examples:counter",
                        "benchmark_profile": "yes",
                        "packed_lanes": 0,
                        "repeat": 0,
                        "warmup": -1,
                        "profile_replay_hot_repeat": -1,
                    }
                ]
            }
        )

        self.assertEqual(valid, [])
        errors = [finding["error"] for finding in invalid]
        self.assertIn("benchmark_profile must be a boolean", errors)
        self.assertIn("packed_lanes must be greater than zero", errors)
        self.assertIn("repeat must be greater than zero", errors)
        self.assertIn("warmup must be non-negative", errors)
        self.assertIn("profile_replay_hot_repeat must be non-negative", errors)

    def test_build_summary_groups_phases_and_failure_buckets(self):
        results = [
            {
                "name": "ok",
                "target": "x:ok",
                "phase": "done",
                "ok": True,
                "error": None,
                "bucket": None,
                "outputs": {},
            },
            {
                "name": "profiled",
                "target": "x:profiled",
                "phase": "done",
                "ok": True,
                "error": None,
                "bucket": None,
                "outputs": {"runtime_profile": "profile.json"},
                "runtime_profile": {"recommended_runtime_backend": "rrtl_threaded_autotune_trace"},
            },
            corpus._failure(
                "bad_op",
                "x:bad_op",
                "check",
                "bad_op check failed: unsupported PyRTL op `?` in net 7",
            ),
            corpus._failure(
                "bad_trace",
                "x:bad_trace",
                "compare",
                "trace mismatch at step 2",
            ),
        ]

        summary = corpus.build_summary(results)

        self.assertEqual(summary["summary"]["total"], 4)
        self.assertEqual(summary["summary"]["passed"], 2)
        self.assertEqual(summary["summary"]["failed"], 2)
        self.assertEqual(summary["summary"]["phase_counts"]["done"], 2)
        self.assertEqual(summary["summary"]["phase_counts"]["check"], 1)
        self.assertEqual(summary["summary"]["failure_buckets"]["unsupported_op"], 1)
        self.assertEqual(summary["summary"]["failure_buckets"]["trace_mismatch"], 1)
        self.assertEqual(summary["summary"]["profiled"], 1)
        self.assertEqual(
            summary["summary"]["recommended_backends"]["rrtl_threaded_autotune_trace"],
            1,
        )

    def test_render_summary_markdown_includes_totals_and_failures(self):
        summary = corpus.build_summary(
            [
                corpus._failure(
                    "bad|name",
                    "x:bad",
                    "check",
                    "net 1 | unsupported PyRTL op `?`",
                )
            ]
        )

        text = corpus.render_summary_markdown(summary)

        self.assertIn("# PyRTL-to-RRTL Corpus Triage", text)
        self.assertIn("- Total: 1", text)
        self.assertIn("- Profiled: 0", text)
        self.assertIn("| unsupported_op | 1 |", text)
        self.assertIn("No profiled recommendations.", text)
        self.assertIn("bad\\|name", text)
        self.assertIn("net 1 \\| unsupported PyRTL op", text)

    def test_render_validation_report_groups_errors(self):
        findings = [
            {
                "name": "a",
                "target": "pkg:a",
                "error": "target import failed: missing module",
            },
            {
                "name": "b",
                "target": "pkg:b",
                "error": "target import failed: missing module",
            },
        ]

        text = corpus.render_validation_report(findings)

        self.assertIn("- Status: failed", text)
        self.assertIn("| target import failed: missing module | 2 |", text)
        self.assertIn("## Findings", text)

    def test_run_manifest_continues_after_failure_by_default(self):
        manifest = {
            "targets": [
                "not an object",
                {"name": "ok", "target": "rrtl_pyrtl.examples:counter", "inputs": [{"en": 1}]},
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(corpus, "_run_cmd", return_value=""):
                results = corpus.run_manifest(
                    manifest,
                    out_dir=Path(tmpdir),
                    pyrtl2rrtl=["pyrtl2rrtl"],
                )

        self.assertEqual([result["name"] for result in results], ["target-0", "ok"])
        self.assertFalse(results[0]["ok"])
        self.assertEqual(results[0]["bucket"], "manifest")
        self.assertTrue(results[1]["ok"])
        self.assertIn("trace", results[1]["outputs"])

    def test_run_manifest_fail_fast_stops_after_first_failure(self):
        manifest = {
            "targets": [
                "not an object",
                {"name": "skipped", "target": "rrtl_pyrtl.examples:counter"},
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            results = corpus.run_manifest(
                manifest,
                out_dir=Path(tmpdir),
                pyrtl2rrtl=["pyrtl2rrtl"],
                fail_fast=True,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "target-0")

    def test_run_manifest_classifies_build_failure(self):
        manifest = {"targets": [{"name": "missing", "target": "no_such_module:build"}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            results = corpus.run_manifest(
                manifest,
                out_dir=Path(tmpdir),
                pyrtl2rrtl=["pyrtl2rrtl"],
            )

        self.assertEqual(results[0]["phase"], "build")
        self.assertEqual(results[0]["bucket"], "build")
        self.assertFalse(results[0]["ok"])

    def test_run_manifest_profiles_target_and_replays_runtime_profile(self):
        manifest = {
            "targets": [
                {
                    "name": "profiled",
                    "target": "rrtl_pyrtl.examples:counter",
                    "inputs": [{"en": 1}, {"en": 0}],
                    "benchmark_profile": True,
                    "packed_lanes": 2,
                    "repeat": 1,
                    "warmup": 0,
                    "profile_replay_hot_repeat": 3,
                }
            ]
        }
        calls = []

        def fake_run(cmd, target_name, phase):
            calls.append((cmd, phase))
            command = cmd[1] if cmd[0] == "pyrtl2rrtl" else cmd[0]
            if command in {"check", "compare"}:
                return ""
            if command == "plan-backends":
                return json.dumps(
                    {
                        "schema": "rrtl-pyrtl-backend-plan-v1",
                        "selected_threaded_layout": {
                            "workers": [{"backend": "simd-cpu", "lanes": 2}]
                        },
                        "selected_gpu_options": {"workgroup_size": 64, "memory_layout": "lane-major"},
                    }
                )
            if command == "bench-threaded-trace" and "--plan-first" in cmd:
                return json.dumps(
                    {
                        "schema": "rrtl-pyrtl-bench-threaded-trace-v1",
                        "replay_ns_best": 60,
                        "replay_ns_median": 62,
                        "mismatch_count": 0,
                    }
                )
            if command == "bench-threaded-trace" and "--autotune" in cmd:
                return json.dumps(
                    {
                        "schema": "rrtl-pyrtl-bench-threaded-trace-v1",
                        "selected_threaded_layout": {
                            "workers": [
                                {"backend": "scalar", "lanes": 1},
                                {"backend": "simd-cpu", "lanes": 1},
                            ]
                        },
                        "selected_reason": "autotune-selected",
                        "autotune": {"selected_candidate": 1, "candidates": [{}, {}]},
                        "replay_ns_best": 40,
                        "replay_ns_median": 42,
                        "mismatch_count": 0,
                    }
                )
            if command == "bench-gpu-combined":
                return json.dumps(
                    {
                        "schema": "rrtl-pyrtl-bench-gpu-combined-v1",
                        "static_trace": {
                            "available": False,
                            "replay_ns_best": 0,
                            "replay_ns_median": 0,
                            "mismatch_count": 0,
                        },
                        "option_sweep": {"candidates": [], "selected_candidate_index": None},
                        "measured_trace": {
                            "available": False,
                            "replay_ns_best": 0,
                            "replay_ns_median": 0,
                            "mismatch_count": 0,
                        },
                    }
                )
            if command == "bench-profile-replay":
                profile_path = next(
                    (str(part) for part in cmd if str(part).endswith(".runtime_profile.json")),
                    "",
                )
                if "backend-simd-cpu" in profile_path:
                    best = 30
                    lane_step = 15.0
                    selected_backend = "rrtl_backend:simd-cpu"
                elif "autotuned-threaded" in profile_path:
                    best = 40
                    lane_step = 20.0
                    selected_backend = "rrtl_threaded_autotune_trace"
                else:
                    best = 80
                    lane_step = 40.0
                    selected_backend = "rrtl_threaded_autotune_trace"
                return json.dumps(
                    {
                        "schema": "rrtl-pyrtl-profile-replay-hot-v1",
                        "selected_backend": selected_backend,
                        "selected_source": "hot-profile-sweep",
                        "repeat": 3,
                        "replay_ns_best": best,
                        "replay_ns_median": best + 2,
                        "replay_ns_per_step": float(best),
                        "replay_ns_per_lane_step": lane_step,
                        "setup_to_replay_ratio": 2.0,
                        "mismatch_count": 0,
                    }
                )
            raise AssertionError(f"unexpected command: {cmd}")

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(corpus, "_run_cmd", side_effect=fake_run):
                results = corpus.run_manifest(
                    manifest,
                    out_dir=Path(tmpdir),
                    pyrtl2rrtl=["pyrtl2rrtl"],
                )

            result = results[0]
            runtime_profile = json.loads(
                Path(result["outputs"]["runtime_profile"]).read_text(encoding="utf-8")
            )

        self.assertTrue(result["ok"])
        self.assertIn("lane_trace", result["outputs"])
        self.assertIn("backend_plan", result["outputs"])
        self.assertIn("profile_replay", result["outputs"])
        self.assertIn("hot_profile_sweep", result["outputs"])
        self.assertIn("hot_profile_replay", result["outputs"])
        self.assertEqual(
            runtime_profile["recommended_runtime_backend"],
            "rrtl_backend:simd-cpu",
        )
        self.assertEqual(runtime_profile["recommended_runtime_source"], "hot-profile-sweep")
        self.assertEqual(runtime_profile["hot_profile_sweep"]["selected_candidate_name"], "backend-simd-cpu")
        commands = [call[0] for call in calls]
        profile_replay_commands = [cmd for cmd in commands if "bench-profile-replay" in cmd]
        self.assertEqual(len(profile_replay_commands), 7)
        self.assertTrue(any("backend-simd-cpu" in str(part) for cmd in profile_replay_commands for part in cmd))
        self.assertIn("3", profile_replay_commands[-1])

    def test_run_manifest_reports_profile_replay_failure_phase(self):
        manifest = {
            "targets": [
                {
                    "name": "profiled",
                    "target": "rrtl_pyrtl.examples:counter",
                    "inputs": [{"en": 1}],
                    "benchmark_profile": True,
                }
            ]
        }
        def fake_run(cmd, target_name, phase):
            command = cmd[1] if cmd[0] == "pyrtl2rrtl" else cmd[0]
            if command in {"check", "compare"}:
                return ""
            if command == "plan-backends":
                return json.dumps({"schema": "rrtl-pyrtl-backend-plan-v1"})
            if command == "bench-threaded-trace":
                return json.dumps({"replay_ns_best": 10, "replay_ns_median": 11, "mismatch_count": 0})
            if command == "bench-gpu-combined":
                return json.dumps(
                    {
                        "static_trace": {"available": False, "replay_ns_best": 0, "mismatch_count": 0},
                        "option_sweep": {},
                        "measured_trace": {"available": False, "replay_ns_best": 0, "mismatch_count": 0},
                    }
                )
            if command == "bench-profile-replay":
                raise RuntimeError("profiled profile_replay failed: profile rejected")
            raise AssertionError(f"unexpected command: {cmd}")

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(corpus, "_run_cmd", side_effect=fake_run):
                results = corpus.run_manifest(
                    manifest,
                    out_dir=Path(tmpdir),
                    pyrtl2rrtl=["pyrtl2rrtl"],
                )

        self.assertFalse(results[0]["ok"])
        self.assertEqual(results[0]["phase"], "profile_replay")
        self.assertEqual(results[0]["bucket"], "profile_replay")

    def test_main_writes_planner_feedback_for_profiled_targets(self):
        manifest = {
            "targets": [
                {
                    "name": "profiled",
                    "target": "rrtl_pyrtl.examples:counter",
                    "inputs": [{"en": 1}, {"en": 0}],
                    "benchmark_profile": True,
                    "packed_lanes": 2,
                    "repeat": 1,
                    "warmup": 0,
                }
            ]
        }
        commands = []

        def fake_run(cmd, target_name, phase):
            commands.append(cmd)
            command = cmd[1] if cmd[0] == "pyrtl2rrtl" else cmd[0]
            if command in {"check", "compare"}:
                return ""
            if command == "plan-backends":
                return json.dumps(
                    {
                        "schema": "rrtl-pyrtl-backend-plan-v1",
                        "selected_threaded_layout": {
                            "workers": [{"backend": "simd-cpu", "lanes": 2}]
                        },
                    }
                )
            if command == "bench-threaded-trace" and "--plan-first" in cmd:
                return json.dumps(
                    {
                        "schema": "rrtl-pyrtl-bench-threaded-trace-v1",
                        "replay_ns_best": 80,
                        "replay_ns_median": 82,
                        "mismatch_count": 0,
                    }
                )
            if command == "bench-threaded-trace" and "--autotune" in cmd:
                return json.dumps(
                    {
                        "schema": "rrtl-pyrtl-bench-threaded-trace-v1",
                        "selected_threaded_layout": {
                            "workers": [{"backend": "simd-cpu", "lanes": 2}]
                        },
                        "replay_ns_best": 40,
                        "replay_ns_median": 41,
                        "mismatch_count": 0,
                    }
                )
            if command == "bench-gpu-combined":
                return json.dumps(
                    {
                        "schema": "rrtl-pyrtl-bench-gpu-combined-v1",
                        "static_trace": {
                            "available": False,
                            "replay_ns_best": 0,
                            "replay_ns_median": 0,
                            "mismatch_count": 0,
                        },
                        "option_sweep": {"candidates": []},
                        "measured_trace": {
                            "available": False,
                            "replay_ns_best": 0,
                            "replay_ns_median": 0,
                            "mismatch_count": 0,
                        },
                    }
                )
            if command == "bench-profile-replay":
                profile_path = next(
                    (str(part) for part in cmd if str(part).endswith(".runtime_profile.json")),
                    "",
                )
                lane_step = 8.0 if "autotuned-threaded" in profile_path else 30.0
                best = 32 if "autotuned-threaded" in profile_path else 90
                return json.dumps(
                    {
                        "schema": "rrtl-pyrtl-profile-replay-hot-v1",
                        "selected_backend": "rrtl_threaded_autotune_trace",
                        "selected_source": "measured-cpu",
                        "repeat": 5,
                        "replay_ns_best": best,
                        "replay_ns_median": best + 1,
                        "replay_ns_per_step": 20.0,
                        "replay_ns_per_lane_step": lane_step,
                        "setup_to_replay_ratio": 2.0,
                        "mismatch_count": 0,
                    }
                )
            raise AssertionError(f"unexpected command: {cmd}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "corpus.json"
            calibration_path = tmp / "seed_calibration.json"
            summary_json = tmp / "summary.json"
            summary_md = tmp / "summary.md"
            out_dir = tmp / "out"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            calibration_path.write_text(
                json.dumps(
                    {
                        "schema": "rrtl-pyrtl-planner-calibration-v1",
                        "summary": {
                            "profiles": 0,
                            "backend_preferences": [],
                            "hot_backend_preferences": [],
                            "threaded_layout_preferences": [],
                            "gpu_option_preferences": [],
                            "profitability_backend_preferences": [],
                            "profitability_penalties": [],
                            "profitability_feature_preferences": [],
                            "profitability_feature_penalties": [],
                        },
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(corpus, "_run_cmd", side_effect=fake_run):
                code = corpus.main(
                    [
                        str(manifest_path),
                        "--out-dir",
                        str(out_dir),
                        "--pyrtl2rrtl",
                        "pyrtl2rrtl",
                        "--summary-json",
                        str(summary_json),
                        "--summary-md",
                        str(summary_md),
                        "--profile-replay-hot-repeat",
                        "5",
                        "--planner-calibration",
                        str(calibration_path),
                    ]
                )

            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            feedback = json.loads((out_dir / "planner_feedback.json").read_text(encoding="utf-8"))
            calibration = json.loads(
                (out_dir / "planner_calibration.json").read_text(encoding="utf-8")
            )
            feedback_md = (out_dir / "planner_feedback.md").read_text(encoding="utf-8")
            calibration_md = (out_dir / "planner_calibration.md").read_text(encoding="utf-8")
            summary_text = summary_md.read_text(encoding="utf-8")

        self.assertEqual(code, 0)
        self.assertEqual(feedback["summary"]["profiles"], 1)
        self.assertEqual(feedback["summary"]["hot_profile_replay"]["valid_profiles"], 1)
        self.assertEqual(calibration["summary"]["profiles"], 1)
        self.assertEqual(
            calibration["summary"]["hot_backend_preferences"][0]["signature"],
            "rrtl_threaded_autotune_trace",
        )
        self.assertEqual(summary["summary"]["planner_feedback"]["plan_misses"], 1)
        self.assertIn("planner_calibration", summary["summary"])
        self.assertEqual(
            summary["summary"]["planner_calibrations"][str(calibration_path)],
            1,
        )
        self.assertEqual(summary["summary"]["planner_feedback"]["hot_profile_replay"]["valid_profiles"], 1)
        self.assertIn("## Planner Feedback", summary_text)
        self.assertIn("## Planner Calibration", summary_text)
        self.assertIn("## Calibration Inputs", summary_text)
        self.assertIn("Static profitability hit rate", summary_text)
        self.assertTrue(
            any(
                "plan-backends" in cmd
                and "--planner-calibration" in cmd
                and str(calibration_path) in cmd
                for cmd in commands
            )
        )
        self.assertTrue(
            any(
                "bench-threaded-trace" in cmd
                and "--plan-first" in cmd
                and "--planner-calibration" in cmd
                and str(calibration_path) in cmd
                for cmd in commands
            )
        )
        self.assertIn("Hot Profile Replay", feedback_md)
        self.assertIn("Plan misses: 1", feedback_md)
        self.assertIn("# RRTL Planner Calibration", calibration_md)

    def test_main_writes_json_and_markdown_summaries(self):
        manifest = {
            "targets": [
                {"name": "counter", "target": "rrtl_pyrtl.examples:counter", "inputs": [{"en": 1}]}
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "corpus.json"
            summary_json = tmp / "summary.json"
            summary_md = tmp / "summary.md"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with mock.patch.object(corpus, "_run_cmd", return_value=""):
                code = corpus.main(
                    [
                        str(manifest_path),
                        "--out-dir",
                        str(tmp / "out"),
                        "--pyrtl2rrtl",
                        "pyrtl2rrtl",
                        "--summary-json",
                        str(summary_json),
                        "--summary-md",
                        str(summary_md),
                    ]
                )

            self.assertEqual(code, 0)
            data = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["passed"], 1)
            self.assertEqual(data["results"][0]["phase"], "done")
            self.assertIn("All targets passed.", summary_md.read_text(encoding="utf-8"))

    def test_main_validate_only_success_and_failure_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            valid_path = tmp / "valid.json"
            invalid_path = tmp / "invalid.json"
            valid_path.write_text(
                json.dumps({"targets": [{"name": "counter", "target": "rrtl_pyrtl.examples:counter"}]}),
                encoding="utf-8",
            )
            invalid_path.write_text(json.dumps({"targets": ["not an object"]}), encoding="utf-8")

            self.assertEqual(corpus.main(["--validate-only", str(valid_path)]), 0)
            self.assertEqual(corpus.main(["--validate-only", str(invalid_path)]), 1)


if __name__ == "__main__":
    unittest.main()
