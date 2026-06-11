import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rrtl_pyrtl import bench


class BenchTests(unittest.TestCase):
    def test_render_benchmark_markdown_includes_simd_fast_path_stats(self):
        report = {
            "config": {
                "rows": 1,
                "cols": 1,
                "steps": 1,
                "data_width": 8,
                "acc_width": 16,
                "repeat": 1,
                "warmup": 0,
                "packed_lanes": 2,
            },
            "pyrtl_fast": {"run_ns_best": 1, "run_ns_median": 1},
            "rrtl_trace": {"replay_ns_best": 1, "replay_ns_median": 1, "mismatch_count": 0},
            "rrtl_packed_trace": {
                "replay_ns_best": 1,
                "replay_ns_median": 1,
                "mismatch_count": 0,
            },
            "rrtl_single_trace": {
                "replay_ns_best": 1,
                "replay_ns_median": 1,
                "mismatch_count": 0,
            },
            "speedup_best": 1.0,
            "speedup_median": 1.0,
            "packed_speedup_best": 1.0,
            "packed_speedup_median": 1.0,
            "single_speedup_best": 1.0,
            "single_speedup_median": 1.0,
            "rrtl_backends": {
                "simd_suitability": {
                    "recommendation": "mixed-candidate",
                    "score_x100": 9000,
                    "fallback_ratio_x100": 0,
                    "estimated_fast_cost": 27,
                    "estimated_fallback_cost": 0,
                    "estimated_materialization_cost": 0,
                    "total": {
                        "fast_instrs": 12,
                        "fallback_instrs": 0,
                        "lane_materializations_per_lane": 0,
                        "fast_path_profile": {
                            "one_limb_ops": 3,
                            "two_limb_ops": 4,
                            "two_limb_mul_ops": 1,
                            "two_limb_memory_reads": 2,
                            "two_limb_mux_ops": 1,
                            "memory_write_effects": 1,
                        },
                        "fallback_reasons": {},
                    },
                },
                "backends": [
                    {
                        "backend": "simd-cpu",
                        "replay_ns_best": 1,
                        "replay_ns_median": 1,
                        "mismatch_count": 0,
                        "replay_timing": {"eval_ns": 10},
                        "simd_stats": {
                            "fast_instrs": 12,
                            "fallback_instrs": 0,
                            "lane_materializations": 0,
                            "fast_paths": {
                                "one_limb_ops": 3,
                                "two_limb_ops": 4,
                                "native_two_limb_ops": 2,
                                "two_limb_mul_ops": 1,
                                "two_limb_memory_reads": 2,
                                "two_limb_mux_ops": 1,
                                "memory_write_effects": 1,
                            },
                            "fallback_reasons": {},
                        },
                    },
                    {
                        "backend": "packed-cpu",
                        "replay_ns_best": 1,
                        "replay_ns_median": 1,
                        "mismatch_count": 0,
                        "replay_timing": {"eval_ns": 11},
                        "simd_stats": {"fast_instrs": 0, "fallback_reasons": {}},
                    },
                ]
            }
        }

        markdown = bench.render_benchmark_markdown(report)

        self.assertIn("1-limb fast", markdown)
        self.assertIn("Native 2-limb", markdown)
        self.assertIn("Mem write effects", markdown)
        self.assertIn("| simd-cpu | 0 | 10 | 0 | 0 | 12 | 0 | 0 | 3 | 4 | 2 | 1 | 2 | 1 | 1 |", markdown)
        self.assertIn("| packed-cpu | 0 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |", markdown)
        self.assertIn(
            "Predicted fast paths: one_limb=3, two_limb=4, two_limb_mul=1, mem_read=2, mux=1, mem_write=1",
            markdown,
        )

    def test_build_systolic_mac_and_input_vectors(self):
        block = bench.build_systolic_mac(rows=2, cols=3, data_width=4, acc_width=16)
        vectors = bench.input_vectors(rows=2, cols=3, data_width=4, steps=5)
        lane_vectors = bench.lane_input_vectors(
            rows=2, cols=3, data_width=4, steps=5, lanes=2
        )

        wire_names = {wire.name for wire in block.wirevector_set}
        self.assertIn("act_0", wire_names)
        self.assertIn("weight_2", wire_names)
        self.assertIn("out_r1_c2", wire_names)
        self.assertEqual(len(vectors), 5)
        self.assertEqual(len(lane_vectors), 2)
        self.assertEqual(len(lane_vectors[0]), 5)
        self.assertNotEqual(lane_vectors[0][0], lane_vectors[1][0])
        self.assertEqual(set(vectors[0]), {"act_0", "act_1", "weight_0", "weight_1", "weight_2"})

    def test_run_benchmark_writes_reports(self):
        fake_rrtl = {
            "schema": "rrtl-pyrtl-bench-trace-v1",
            "steps": 4,
            "repeat": 2,
            "warmup": 1,
            "import_ns": 10,
            "setup_ns": 20,
            "replay_ns_samples": [100, 120],
            "replay_ns_best": 100,
            "replay_ns_median": 120,
            "mismatch_count": 0,
        }
        fake_rrtl_packed = {
            "schema": "rrtl-pyrtl-bench-packed-trace-v1",
            "steps": 4,
            "repeat": 2,
            "warmup": 1,
            "lanes": 2,
            "import_ns": 10,
            "setup_ns": 30,
            "replay_ns_samples": [80, 90],
            "replay_ns_best": 80,
            "replay_ns_median": 90,
            "mismatch_count": 0,
        }
        fake_rrtl_single = {
            "schema": "rrtl-pyrtl-bench-single-trace-v1",
            "steps": 4,
            "repeat": 2,
            "warmup": 1,
            "import_ns": 10,
            "setup_ns": 25,
            "replay_ns_samples": [70, 75],
            "replay_ns_best": 70,
            "replay_ns_median": 75,
            "mismatch_count": 0,
        }
        fake_rrtl_backends = {
            "schema": "rrtl-pyrtl-bench-backends-trace-v1",
            "steps": 4,
            "repeat": 2,
            "warmup": 1,
            "import_ns": 10,
            "setup_ns": 40,
            "backends": [
                {
                    "backend": "simd-cpu",
                    "lanes": 2,
                    "available": True,
                    "replay_ns_samples": [60, 65],
                    "replay_ns_best": 60,
                    "replay_ns_median": 65,
                    "mismatch_count": 0,
                    "mismatches": [],
                }
            ],
        }
        fake_rrtl_threaded = {
            "schema": "rrtl-pyrtl-bench-threaded-trace-v1",
            "steps": 4,
            "lanes": 2,
            "repeat": 2,
            "warmup": 1,
            "import_ns": 10,
            "setup_ns": 45,
            "selected_threaded_layout": {
                "workers": [{"backend": "simd-cpu", "lanes": 2}],
                "max_mismatches": 16,
            },
            "selected_reason": "first-planned-candidate",
            "replay_ns_samples": [55, 58],
            "replay_ns_best": 55,
            "replay_ns_median": 58,
            "replay": {"total_lanes": 2, "workers": [], "replay": {}, "lane_cycles_per_sec": 1.0},
            "replay_workload": {
                "steps": 4,
                "lanes": 2,
                "input_ops": 8,
                "check_ops": 4,
                "one_limb_input_ops": 2,
                "one_limb_check_ops": 1,
                "one_limb_input_batches": 3,
                "one_limb_check_batches": 2,
                "generic_input_ops": 1,
                "generic_check_ops": 0,
                "estimated_lane_work_units": 42,
            },
            "mismatch_count": 0,
            "mismatches": [],
        }
        fake_rrtl_threaded_autotune = {
            "schema": "rrtl-pyrtl-bench-threaded-trace-v1",
            "steps": 4,
            "lanes": 2,
            "repeat": 2,
            "warmup": 1,
            "import_ns": 10,
            "setup_ns": 47,
            "selected_threaded_layout": {
                "workers": [{"backend": "scalar", "lanes": 1}, {"backend": "simd-cpu", "lanes": 1}],
                "max_mismatches": 16,
            },
            "selected_reason": "autotune-selected",
            "replay_ns_samples": [50, 53],
            "replay_ns_best": 50,
            "replay_ns_median": 53,
            "replay": {"total_lanes": 2, "workers": [], "replay": {}, "lane_cycles_per_sec": 1.2},
            "replay_workload": {
                "steps": 4,
                "lanes": 2,
                "estimated_lane_work_units": 42,
            },
            "autotune": {
                "selected_candidate": 1,
                "candidates": [
                    {"candidate_index": 0, "available": True, "lane_cycles_per_sec": 1.0},
                    {"candidate_index": 1, "available": True, "lane_cycles_per_sec": 1.2},
                ],
                "pruned_candidates": [{"candidate_index": 2}],
            },
            "autotune_pruned_candidates": [{"candidate_index": 2}],
            "mismatch_count": 0,
            "mismatches": [],
        }
        fake_backend_plan = {
            "schema": "rrtl-pyrtl-backend-plan-v1",
            "steps": 4,
            "lanes": 2,
            "max_workers": 2,
            "profitability": {
                "op_profile": {
                    "instr_count": 12,
                    "one_limb_ops": 8,
                    "two_limb_ops": 2,
                    "native_two_limb_ops": 2,
                    "two_limb_mul_ops": 0,
                    "memory_ops": 0,
                    "wide_fallback_ops": 0,
                    "pure_compute_packets": 4,
                    "memory_hostile_packets": 0,
                    "estimated_lane_work_units": 42,
                },
                "simd_coverage_score_x100": 90,
                "gpu_suitability_score_x100": 20,
                "threading_score_x100": 30,
                "recommended_backend": "simd-cpu",
                "recommended_reason": "high-simd-coverage",
            },
            "backend_candidates": [
                {
                    "backend": "simd-cpu",
                    "rank": 1,
                    "score": 190,
                    "estimated_setup_cost": 5,
                    "estimated_per_lane_step_cost": 5,
                    "reasons": ["high-simd-coverage"],
                },
                {
                    "backend": "gpu-fused",
                    "rank": 2,
                    "score": 120,
                    "estimated_setup_cost": 80,
                    "estimated_per_lane_step_cost": 2,
                    "reasons": ["gpu-launch-not-amortized"],
                },
            ],
            "selected_runtime_backend": "simd-cpu",
            "selected_runtime_reason": "high-simd-coverage",
            "pruned_runtime_candidates": [
                {
                    "backend": "gpu-fused",
                    "rank": 2,
                    "score": 120,
                    "estimated_setup_cost": 80,
                    "estimated_per_lane_step_cost": 2,
                    "reasons": ["gpu-launch-not-amortized"],
                }
            ],
            "simd_suitability": {"recommendation": "simd-candidate"},
            "backend_affinity": {"recommendation": "simd-cpu-candidate"},
            "gpu_region_analysis": {"recommendation": "compute-candidate"},
            "shader_stats": {"wgsl_bytes": 128},
            "replay_workload": {
                "steps": 4,
                "lanes": 2,
                "estimated_lane_work_units": 42,
            },
            "selected_threaded_layout": {
                "workers": [{"backend": "simd-cpu", "lanes": 2}],
                "max_mismatches": 16,
            },
            "selected_reason": "first-planned-candidate",
            "selected_gpu_options": {"workgroup_size": 64, "memory_layout": "lane-major"},
            "selected_gpu_reason": "gpu-compute-candidate",
            "recommended_threaded_layouts": [],
            "pruned_threaded_layouts": [{"candidate_index": 1}],
            "recommended_gpu_options": [
                {
                    "options": {"workgroup_size": 64, "memory_layout": "lane-major"},
                    "shader_stats": {"wgsl_bytes": 128, "optimized_packets_total": 4},
                    "reason": "lane-major-workgroup-64",
                }
            ],
            "pruned_gpu_options": [],
        }
        fake_rrtl_gpu = {
            "schema": "rrtl-pyrtl-bench-gpu-trace-v1",
            "steps": 4,
            "lanes": 2,
            "repeat": 2,
            "warmup": 1,
            "import_ns": 10,
            "setup_ns": 50,
            "prepared_runner_setup_ns": 0,
            "prepared_snapshot_setup_ns": 0,
            "prepared_trace_bytes": 0,
            "prepared_trace_uncompressed_bytes": 0,
            "prepared_trace_compression_ratio_x100": 100,
            "prepared_trace_uniform_input_ops": 0,
            "prepared_trace_uniform_check_ops": 0,
            "prepared_trace_layout": "",
            "prepared_trace_template_input_ops": 0,
            "prepared_trace_template_check_ops": 0,
            "prepared_trace_metadata_saved_words": 0,
            "prepared_trace_fixed_template": False,
            "prepared_trace_value_metadata_saved_words": 0,
            "prepared_trace_value_stride_words": 0,
            "hot_restore_ns_best": 0,
            "hot_restore_ns_median": 0,
            "hot_gpu_replay_ns_best": 0,
            "hot_gpu_replay_ns_median": 0,
            "gpu_single_submit_profitable": False,
            "gpu_planner_calibration_reason": None,
            "available": False,
            "error": "gpu-not-selected-by-plan",
            "backend_affinity": {},
            "gpu_region_analysis": {"recommendation": "compute-candidate"},
            "shader_stats": {"wgsl_bytes": 128},
            "gpu_replay_mode": "fused-kernel",
            "gpu_timing": {
                "count_readback_ns": 0,
                "full_readback_ns": 0,
                "full_readback_words": 0,
                "single_submit_used": False,
                "single_submit_ns": 0,
                "single_submit_count_readback_ns": 0,
            },
            "replay_ns_samples": [],
            "replay_ns_best": 0,
            "replay_ns_median": 0,
            "mismatch_count": 0,
            "mismatches": [],
        }
        fake_rrtl_gpu_options = {
            "schema": "rrtl-pyrtl-bench-gpu-options-v1",
            "steps": 4,
            "lanes": 2,
            "repeat": 2,
            "warmup": 1,
            "selected_candidate_index": 0,
            "candidates": [
                {
                    "candidate_index": 0,
                    "planned": {
                        "options": {
                            "workgroup_size": 64,
                            "memory_layout": "lane-major",
                            "reuse_temporaries": False,
                        },
                        "shader_stats": {"wgsl_bytes": 128},
                        "reason": "lane-major-workgroup-64",
                    },
                    "available": True,
                    "replay_ns_samples": [45, 50],
                    "replay_ns_best": 45,
                    "replay_ns_median": 50,
                    "mismatch_count": 0,
                    "report": {
                        "schema": "rrtl-pyrtl-bench-gpu-trace-v1",
                        "available": True,
                        "replay_ns_best": 45,
                        "replay_ns_median": 50,
                        "mismatch_count": 0,
                    },
                }
            ],
        }
        fake_rrtl_gpu_measured = {
            "schema": "rrtl-pyrtl-bench-gpu-trace-v1",
            "steps": 4,
            "lanes": 2,
            "repeat": 2,
            "warmup": 1,
            "import_ns": 10,
            "setup_ns": 52,
            "prepared_runner_setup_ns": 7,
            "prepared_snapshot_setup_ns": 9,
            "prepared_trace_bytes": 256,
            "prepared_trace_uncompressed_bytes": 512,
            "prepared_trace_compression_ratio_x100": 50,
            "prepared_trace_uniform_input_ops": 12,
            "prepared_trace_uniform_check_ops": 8,
            "prepared_trace_layout": "templated",
            "prepared_trace_template_input_ops": 2,
            "prepared_trace_template_check_ops": 1,
            "prepared_trace_metadata_saved_words": 96,
            "prepared_trace_fixed_template": True,
            "prepared_trace_value_metadata_saved_words": 40,
            "prepared_trace_value_stride_words": 6,
            "hot_restore_ns_best": 11,
            "hot_restore_ns_median": 13,
            "hot_gpu_replay_ns_best": 17,
            "hot_gpu_replay_ns_median": 17,
            "gpu_single_submit_profitable": True,
            "gpu_planner_calibration_reason": "single-submit-hot-gpu-profitable",
            "available": True,
            "error": None,
            "backend_affinity": {},
            "gpu_region_analysis": {"recommendation": "compute-candidate"},
            "shader_stats": {"wgsl_bytes": 128},
            "gpu_replay_mode": "fused-kernel",
            "gpu_timing": {
                "count_readback_ns": 5,
                "full_readback_ns": 0,
                "full_readback_words": 0,
                "single_submit_used": True,
                "single_submit_ns": 17,
                "single_submit_count_readback_ns": 5,
            },
            "replay_ns_samples": [45, 50],
            "replay_ns_best": 45,
            "replay_ns_median": 50,
            "mismatch_count": 0,
            "mismatches": [],
        }
        fake_rrtl_gpu_combined = {
            "schema": "rrtl-pyrtl-bench-gpu-combined-v1",
            "static_trace": fake_rrtl_gpu,
            "option_sweep": fake_rrtl_gpu_options,
            "measured_trace": fake_rrtl_gpu_measured,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "bench"
            scalar_completed = mock.Mock(returncode=0, stdout=json.dumps(fake_rrtl), stderr="")
            packed_completed = mock.Mock(
                returncode=0, stdout=json.dumps(fake_rrtl_packed), stderr=""
            )
            single_completed = mock.Mock(
                returncode=0, stdout=json.dumps(fake_rrtl_single), stderr=""
            )
            backends_completed = mock.Mock(
                returncode=0, stdout=json.dumps(fake_rrtl_backends), stderr=""
            )
            threaded_completed = mock.Mock(
                returncode=0, stdout=json.dumps(fake_rrtl_threaded), stderr=""
            )
            threaded_autotune_completed = mock.Mock(
                returncode=0, stdout=json.dumps(fake_rrtl_threaded_autotune), stderr=""
            )
            plan_completed = mock.Mock(
                returncode=0, stdout=json.dumps(fake_backend_plan), stderr=""
            )
            gpu_combined_completed = mock.Mock(
                returncode=0, stdout=json.dumps(fake_rrtl_gpu_combined), stderr=""
            )
            normal_completed = [
                scalar_completed,
                packed_completed,
                single_completed,
                backends_completed,
                plan_completed,
                threaded_completed,
                threaded_autotune_completed,
                gpu_combined_completed,
            ]

            def fake_run(cmd, text=True, capture_output=True):
                del text, capture_output
                if normal_completed:
                    return normal_completed.pop(0)
                profile_path = next(
                    (str(part) for part in cmd if str(part).endswith(".runtime_profile.json")),
                    "",
                )
                best_by_candidate = {
                    "backend-scalar": (90, 45.0, "rrtl_backend:scalar"),
                    "backend-packed-cpu": (80, 40.0, "rrtl_backend:packed-cpu"),
                    "backend-simd-cpu": (70, 35.0, "rrtl_backend:simd-cpu"),
                    "backend-jit-cpu": (120, 60.0, "rrtl_backend:jit-cpu"),
                    "planned-threaded": (55, 27.5, "rrtl_threaded_autotune_trace"),
                    "autotuned-threaded": (50, 25.0, "rrtl_threaded_autotune_trace"),
                    "measured-gpu": (44, 5.5, "rrtl_gpu_measured_trace"),
                }
                best, lane_step, selected_backend = next(
                    (
                        value
                        for candidate, value in best_by_candidate.items()
                        if candidate in profile_path
                    ),
                    (200, 100.0, "unknown"),
                )
                return mock.Mock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "schema": "rrtl-pyrtl-profile-replay-hot-v1",
                            "selected_backend": selected_backend,
                            "selected_source": "hot-profile-sweep",
                            "repeat": 5,
                            "first_replay_ns": best + 10,
                            "replay_ns_best": best,
                            "replay_ns_median": best + 2,
                            "hot_replay_ns_best": best,
                            "hot_replay_ns_median": best + 2,
                            "setup_ns_total": best * 5,
                            "setup_to_replay_ratio": 3.5,
                            "setup_to_hot_ratio": 5.0,
                            "hot_replay_speedup": (best + 10) / best,
                            "replay_ns_per_lane_step": lane_step,
                            "mismatch_count": 0,
                        }
                    ),
                    stderr="",
                )

            with mock.patch(
                "rrtl_pyrtl.bench.subprocess.run",
                side_effect=fake_run,
            ) as run:
                report = bench.run_benchmark(
                    out_dir=out_dir,
                    rows=1,
                    cols=1,
                    steps=4,
                    data_width=4,
                    acc_width=16,
                    repeat=2,
                    warmup=1,
                    packed_lanes=2,
                    profile_replay_hot_repeat=5,
                    pyrtl2rrtl=["pyrtl2rrtl"],
                )

            bench_json = json.loads((out_dir / "bench.json").read_text(encoding="utf-8"))
            bench_md = (out_dir / "bench.md").read_text(encoding="utf-8")
            runtime_profile = json.loads(
                (out_dir / "runtime_profile.json").read_text(encoding="utf-8")
            )

        self.assertEqual(report["schema"], "rrtl-pyrtl-bench-v1")
        self.assertEqual(report["config"]["rows"], 1)
        self.assertEqual(report["config"]["packed_lanes"], 2)
        self.assertEqual(report["rrtl_trace"]["replay_ns_best"], 100)
        self.assertEqual(report["rrtl_packed_trace"]["replay_ns_best"], 80)
        self.assertEqual(report["rrtl_single_trace"]["replay_ns_best"], 70)
        self.assertEqual(report["rrtl_backends"]["backends"][0]["backend"], "simd-cpu")
        self.assertEqual(report["rrtl_backend_plan"]["schema"], "rrtl-pyrtl-backend-plan-v1")
        self.assertEqual(report["rrtl_threaded_trace"]["replay_ns_best"], 55)
        self.assertEqual(report["rrtl_threaded_autotune_trace"]["replay_ns_best"], 50)
        self.assertEqual(
            report["rrtl_threaded_autotune_trace"]["selected_reason"],
            "autotune-selected",
        )
        self.assertEqual(
            report["threaded_autotune_speedup_best"],
            report["pyrtl_fast"]["run_ns_best"] / 50,
        )
        self.assertEqual(report["rrtl_gpu_trace"]["schema"], "rrtl-pyrtl-bench-gpu-trace-v1")
        self.assertEqual(
            report["rrtl_gpu_combined"]["schema"],
            "rrtl-pyrtl-bench-gpu-combined-v1",
        )
        self.assertEqual(
            report["rrtl_gpu_option_sweep"]["schema"],
            "rrtl-pyrtl-bench-gpu-options-v1",
        )
        self.assertEqual(report["rrtl_gpu_option_sweep"]["selected_candidate_index"], 0)
        self.assertEqual(report["rrtl_gpu_measured_trace"]["replay_ns_best"], 45)
        self.assertEqual(report["rrtl_gpu_measured_trace"]["selected_gpu_option_index"], 0)
        self.assertEqual(report["rrtl_gpu_measured_trace"]["prepared_runner_setup_ns"], 7)
        self.assertEqual(report["rrtl_gpu_measured_trace"]["prepared_snapshot_setup_ns"], 9)
        self.assertEqual(report["rrtl_gpu_measured_trace"]["prepared_trace_bytes"], 256)
        self.assertEqual(report["rrtl_gpu_measured_trace"]["prepared_trace_uncompressed_bytes"], 512)
        self.assertEqual(report["rrtl_gpu_measured_trace"]["prepared_trace_compression_ratio_x100"], 50)
        self.assertEqual(report["rrtl_gpu_measured_trace"]["prepared_trace_uniform_input_ops"], 12)
        self.assertEqual(report["rrtl_gpu_measured_trace"]["prepared_trace_uniform_check_ops"], 8)
        self.assertEqual(report["rrtl_gpu_measured_trace"]["prepared_trace_layout"], "templated")
        self.assertEqual(report["rrtl_gpu_measured_trace"]["prepared_trace_template_input_ops"], 2)
        self.assertEqual(report["rrtl_gpu_measured_trace"]["prepared_trace_template_check_ops"], 1)
        self.assertEqual(report["rrtl_gpu_measured_trace"]["prepared_trace_metadata_saved_words"], 96)
        self.assertTrue(report["rrtl_gpu_measured_trace"]["prepared_trace_fixed_template"])
        self.assertEqual(
            report["rrtl_gpu_measured_trace"]["prepared_trace_value_metadata_saved_words"], 40
        )
        self.assertEqual(report["rrtl_gpu_measured_trace"]["prepared_trace_value_stride_words"], 6)
        self.assertEqual(report["rrtl_gpu_measured_trace"]["hot_restore_ns_best"], 11)
        self.assertEqual(report["rrtl_gpu_measured_trace"]["hot_restore_ns_median"], 13)
        self.assertEqual(report["rrtl_gpu_measured_trace"]["hot_gpu_replay_ns_best"], 17)
        self.assertTrue(report["rrtl_gpu_measured_trace"]["gpu_single_submit_profitable"])
        self.assertEqual(
            report["rrtl_gpu_measured_trace"]["gpu_planner_calibration_reason"],
            "single-submit-hot-gpu-profitable",
        )
        self.assertEqual(
            report["rrtl_gpu_measured_trace"]["gpu_timing"]["count_readback_ns"], 5
        )
        self.assertEqual(
            report["rrtl_gpu_measured_trace"]["gpu_timing"]["full_readback_words"], 0
        )
        self.assertTrue(
            report["rrtl_gpu_measured_trace"]["gpu_timing"]["single_submit_used"]
        )
        self.assertEqual(
            report["rrtl_gpu_measured_trace"]["gpu_timing"]["single_submit_ns"], 17
        )
        self.assertEqual(report["profile_replay_hot"]["replay_ns_best"], 44)
        self.assertEqual(report["profile_replay_hot"]["first_replay_ns"], 54)
        self.assertEqual(report["profile_replay_hot"]["hot_replay_ns_best"], 44)
        self.assertEqual(report["profile_replay_hot"]["setup_ns_total"], 220)
        self.assertEqual(report["profile_replay_hot"]["setup_to_hot_ratio"], 5.0)
        self.assertEqual(report["profile_replay_hot_sweep"]["valid_candidate_count"], 7)
        self.assertEqual(report["profile_replay_hot_sweep"]["selected_candidate_name"], "measured-gpu")
        self.assertEqual(
            bench_json["outputs"]["hot_profile_replay"],
            str(out_dir / "hot_profile_replay.json"),
        )
        self.assertEqual(
            bench_json["outputs"]["hot_profile_sweep"],
            str(out_dir / "hot_profile_sweep.json"),
        )
        self.assertEqual(
            report["gpu_measured_speedup_best"],
            report["pyrtl_fast"]["run_ns_best"] / 45,
        )
        self.assertEqual(report["gpu_speedup_best"], 0.0)
        self.assertEqual(
            report["backend_plan_evaluation"]["fastest_rrtl_available"]["name"],
            "rrtl_threaded_trace",
        )
        self.assertEqual(
            report["backend_plan_evaluation"]["fastest_overall_backend"]["name"],
            "rrtl_gpu_measured_trace",
        )
        self.assertEqual(
            report["backend_plan_evaluation"]["recommended_runtime_backend"],
            "rrtl_gpu_measured_trace",
        )
        self.assertEqual(
            report["backend_plan_evaluation"]["static_profitability_backend"],
            "simd-cpu",
        )
        self.assertEqual(
            report["backend_plan_evaluation"]["static_profitability_measured_backend"],
            "rrtl_backend:simd-cpu",
        )
        self.assertFalse(report["backend_plan_evaluation"]["static_profitability_hit"])
        self.assertEqual(
            report["backend_plan_evaluation"]["static_profitability_miss_reason"],
            "static-profitability-slower",
        )
        self.assertEqual(
            report["backend_plan_evaluation"]["recommended_runtime_source"],
            "measured-gpu",
        )
        self.assertAlmostEqual(
            report["backend_plan_evaluation"]["static_plan_vs_recommended_speedup"],
            55 / 45,
        )
        self.assertEqual(
            report["backend_plan_evaluation"]["recommended_backend_reason"],
            "measured-gpu-beats-static-plan",
        )
        self.assertEqual(report["backend_plan_evaluation"]["planned_cpu_rank"], 3)
        self.assertIsNone(report["backend_plan_evaluation"]["planned_gpu_rank"])
        self.assertTrue(report["backend_plan_evaluation"]["planned_gpu_selected"])
        self.assertTrue(report["backend_plan_evaluation"]["plan_hit"])
        self.assertEqual(report["backend_plan_evaluation"]["miss_reason"], "none")
        self.assertEqual(report["backend_plan_evaluation"]["planned_gpu_option_rank"], 1)
        self.assertEqual(report["backend_plan_evaluation"]["planned_gpu_option_best_ns"], 45)
        self.assertEqual(
            report["backend_plan_evaluation"]["measured_gpu_option_best_index"], 0
        )
        self.assertEqual(report["backend_plan_evaluation"]["measured_gpu_option_best_ns"], 45)
        self.assertTrue(report["backend_plan_evaluation"]["gpu_option_plan_hit"])
        self.assertEqual(
            report["backend_plan_evaluation"]["gpu_option_miss_reason"], "none"
        )
        self.assertGreater(report["pyrtl_fast"]["run_ns_best"], 0)
        self.assertEqual(bench_json["schema"], "rrtl-pyrtl-bench-v1")
        self.assertEqual(
            bench_json["backend_plan_evaluation"]["schema"],
            "rrtl-pyrtl-backend-plan-evaluation-v1",
        )
        self.assertEqual(
            bench_json["outputs"]["lane_trace"],
            str(out_dir / "systolic_mac.lane_trace.json"),
        )
        self.assertEqual(
            bench_json["outputs"]["backend_plan"],
            str(out_dir / "backend_plan.json"),
        )
        self.assertEqual(
            bench_json["outputs"]["runtime_profile"],
            str(out_dir / "runtime_profile.json"),
        )
        self.assertEqual(
            report["runtime_profile"],
            runtime_profile,
        )
        self.assertEqual(runtime_profile["schema"], "rrtl-pyrtl-runtime-profile-v1")
        self.assertEqual(
            runtime_profile["static_backend_profitability"]["recommended_backend"],
            "simd-cpu",
        )
        self.assertEqual(
            runtime_profile["static_backend_candidates"][0]["backend"],
            "simd-cpu",
        )
        self.assertEqual(
            runtime_profile["recommended_runtime_backend"],
            "rrtl_gpu_measured_trace",
        )
        self.assertEqual(runtime_profile["recommended_runtime_source"], "hot-profile-sweep")
        self.assertEqual(runtime_profile["hot_profile_sweep"]["selected_candidate_name"], "measured-gpu")
        self.assertEqual(
            runtime_profile["selected_backend"]["backend"],
            "rrtl_gpu_measured_trace",
        )
        self.assertEqual(runtime_profile["selected_backend"]["selected_gpu_option_index"], 0)
        self.assertEqual(
            runtime_profile["selected_backend"]["selected_gpu_options"]["workgroup_size"],
            64,
        )
        self.assertEqual(
            runtime_profile["threaded_autotune"]["selected_threaded_layout"]["workers"][0]["backend"],
            "scalar",
        )
        self.assertEqual(runtime_profile["threaded_autotune"]["autotune_candidate_count"], 2)
        self.assertIn(
            "rrtl_gpu_measured_trace",
            {backend["name"] for backend in runtime_profile["measured_backends"]},
        )
        self.assertIn("# PyRTL-to-RRTL Synthetic Benchmark", bench_md)
        self.assertIn("## Backend Plan", bench_md)
        self.assertIn("Selected workers: simd-cpu:2", bench_md)
        self.assertIn("Selected GPU workgroup size: 64", bench_md)
        self.assertIn("Recommended GPU options: 1", bench_md)
        self.assertIn("Static profitability backend: simd-cpu", bench_md)
        self.assertIn("Selected runtime backend: simd-cpu", bench_md)
        self.assertIn("Pruned runtime candidates: 1", bench_md)
        self.assertIn("Static top candidate: simd-cpu score=190", bench_md)
        self.assertIn("native_two_limb=2", bench_md)
        self.assertIn("## GPU Replay", bench_md)
        self.assertIn("Prepared runner setup ns: 0", bench_md)
        self.assertIn("Prepared snapshot setup ns: 0", bench_md)
        self.assertIn("Full readback words: 0", bench_md)
        self.assertIn("Single submit used: False", bench_md)
        self.assertIn("Prepared trace compression ratio x100: 100", bench_md)
        self.assertIn("Prepared trace metadata saved words: 0", bench_md)
        self.assertIn("Prepared trace fixed template: False", bench_md)
        self.assertIn("Hot GPU replay best ns: 0", bench_md)
        self.assertIn("GPU single-submit profitable: False", bench_md)
        self.assertIn("## GPU Option Sweep", bench_md)
        self.assertIn("Selected candidate: 0", bench_md)
        self.assertIn("Selected reason: lane-major-workgroup-64", bench_md)
        self.assertIn("Selected best ns: 45", bench_md)
        self.assertIn("## GPU Measured Replay", bench_md)
        self.assertIn("Prepared runner setup ns: 7", bench_md)
        self.assertIn("Prepared snapshot setup ns: 9", bench_md)
        self.assertIn("Prepared trace bytes: 256", bench_md)
        self.assertIn("Prepared trace uncompressed bytes: 512", bench_md)
        self.assertIn("Prepared trace compression ratio x100: 50", bench_md)
        self.assertIn("Prepared trace uniform input ops: 12", bench_md)
        self.assertIn("Prepared trace uniform check ops: 8", bench_md)
        self.assertIn("Prepared trace layout: templated", bench_md)
        self.assertIn("Prepared trace template input ops: 2", bench_md)
        self.assertIn("Prepared trace template check ops: 1", bench_md)
        self.assertIn("Prepared trace metadata saved words: 96", bench_md)
        self.assertIn("Prepared trace fixed template: True", bench_md)
        self.assertIn("Prepared trace value metadata saved words: 40", bench_md)
        self.assertIn("Prepared trace value stride words: 6", bench_md)
        self.assertIn("Hot restore best ns: 11", bench_md)
        self.assertIn("Hot restore median ns: 13", bench_md)
        self.assertIn("Hot GPU replay best ns: 17", bench_md)
        self.assertIn("Hot GPU replay median ns: 17", bench_md)
        self.assertIn("GPU single-submit profitable: True", bench_md)
        self.assertIn("GPU planner calibration reason: single-submit-hot-gpu-profitable", bench_md)
        self.assertIn("Count readback ns: 5", bench_md)
        self.assertIn("Full readback ns: 0", bench_md)
        self.assertIn("Full readback words: 0", bench_md)
        self.assertIn("Single submit used: True", bench_md)
        self.assertIn("Single submit ns: 17", bench_md)
        self.assertIn("Single submit count readback ns: 5", bench_md)
        self.assertIn("Differs from static GPU option: False", bench_md)
        self.assertIn("## Threaded Autotune Replay", bench_md)
        self.assertIn("Selected workers: scalar:1,simd-cpu:1", bench_md)
        self.assertIn("Autotune selected candidate: 1", bench_md)
        self.assertIn("Autotune candidates: 2", bench_md)
        self.assertIn("RRTL GPU mismatches: 0", bench_md)
        self.assertIn("RRTL packed trace replay", bench_md)
        self.assertIn("RRTL single-lane machine trace replay", bench_md)
        self.assertIn("RRTL threaded independent-lane replay", bench_md)
        self.assertIn("RRTL backend simd-cpu", bench_md)
        self.assertIn("## Plan Evaluation", bench_md)
        self.assertIn("Fastest measured RRTL backend: rrtl_threaded_trace", bench_md)
        self.assertIn("Fastest overall backend: rrtl_gpu_measured_trace", bench_md)
        self.assertIn("Recommended runtime backend: rrtl_gpu_measured_trace", bench_md)
        self.assertIn("Recommended runtime source: measured-gpu", bench_md)
        self.assertIn("Recommended reason: measured-gpu-beats-static-plan", bench_md)
        self.assertIn("Static profitability hit: False", bench_md)
        self.assertIn("Static profitability miss reason: static-profitability-slower", bench_md)
        self.assertIn("## Runtime Profile", bench_md)
        self.assertIn("Recommended backend: rrtl_gpu_measured_trace", bench_md)
        self.assertIn("Profile GPU option: workgroup=64", bench_md)
        self.assertIn("## Hot Profile Replay", bench_md)
        self.assertIn("First replay ns: 54", bench_md)
        self.assertIn("Best ns: 44", bench_md)
        self.assertIn("Setup to hot ratio: 5.00x", bench_md)
        self.assertIn("Hot replay speedup:", bench_md)
        self.assertIn("## Hot Profile Sweep", bench_md)
        self.assertIn("Selected candidate: measured-gpu", bench_md)
        self.assertIn("Planned CPU rank: 3", bench_md)
        self.assertIn("Planned GPU rank: none", bench_md)
        self.assertIn("Plan hit: True", bench_md)
        self.assertIn("Planned GPU option rank: 1", bench_md)
        self.assertIn("Measured best GPU option: 0", bench_md)
        self.assertIn("GPU option plan hit: True", bench_md)
        self.assertIn("RRTL packed mismatches: 0", bench_md)
        self.assertIn("RRTL single-lane machine mismatches: 0", bench_md)
        self.assertIn("RRTL threaded independent-lane mismatches: 0", bench_md)
        self.assertIn("RRTL threaded autotune mismatches: 0", bench_md)
        self.assertIn("RRTL backend simd-cpu mismatches: 0", bench_md)
        self.assertIn("## Replay Workload", bench_md)
        self.assertIn("Estimated lane work units: 42", bench_md)
        self.assertEqual(run.call_count, 15)
        self.assertIn("bench-trace", run.call_args_list[0].args[0])
        self.assertIn("bench-packed-trace", run.call_args_list[1].args[0])
        self.assertIn("bench-single-trace", run.call_args_list[2].args[0])
        self.assertIn("bench-backends", run.call_args_list[3].args[0])
        self.assertIn("plan-backends", run.call_args_list[4].args[0])
        self.assertIn("bench-threaded-trace", run.call_args_list[5].args[0])
        self.assertIn("--plan-first", run.call_args_list[5].args[0])
        autotune_cmd = run.call_args_list[6].args[0]
        self.assertIn("bench-threaded-trace", autotune_cmd)
        self.assertIn("--autotune", autotune_cmd)
        self.assertNotIn("--plan-first", autotune_cmd)
        gpu_cmd = run.call_args_list[7].args[0]
        self.assertIn("bench-gpu-combined", gpu_cmd)
        self.assertNotIn("--plan-first", gpu_cmd)
        hot_cmds = [call.args[0] for call in run.call_args_list[8:]]
        self.assertEqual(len(hot_cmds), 7)
        self.assertTrue(all("bench-profile-replay" in cmd for cmd in hot_cmds))
        self.assertTrue(any("measured-gpu" in str(part) for cmd in hot_cmds for part in cmd))
        self.assertTrue(all("5" in cmd for cmd in hot_cmds))
        self.assertIn("--lanes", run.call_args_list[1].args[0])
        self.assertIn("2", run.call_args_list[1].args[0])

    def test_run_bench_gpu_trace_with_options_passes_reuse_temporaries(self):
        completed = mock.Mock(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema": "rrtl-pyrtl-bench-gpu-trace-v1",
                    "available": True,
                    "replay_ns_best": 10,
                    "replay_ns_median": 10,
                    "mismatch_count": 0,
                }
            ),
            stderr="",
        )
        with mock.patch("rrtl_pyrtl.bench.subprocess.run", return_value=completed) as run:
            report = bench._run_bench_gpu_trace_with_options(
                ["pyrtl2rrtl"],
                Path("export.json"),
                Path("lane_trace.json"),
                {
                    "workgroup_size": 128,
                    "memory_layout": "lane-major",
                    "reuse_temporaries": True,
                },
                repeat=2,
                warmup=1,
            )

        cmd = run.call_args.args[0]
        self.assertEqual(report["replay_ns_best"], 10)
        self.assertIn("--workgroup-size", cmd)
        self.assertIn("128", cmd)
        self.assertIn("--memory-layout", cmd)
        self.assertIn("lane-major", cmd)
        self.assertIn("--reuse-temporaries", cmd)
        self.assertNotIn("--plan-first", cmd)

    def test_run_bench_gpu_measured_trace_skips_without_valid_winner(self):
        sweep = {
            "schema": "rrtl-pyrtl-bench-gpu-options-v1",
            "steps": 4,
            "lanes": 2,
            "repeat": 2,
            "warmup": 1,
            "selected_candidate_index": None,
            "candidates": [],
        }
        with mock.patch("rrtl_pyrtl.bench.subprocess.run") as run:
            report = bench._run_bench_gpu_measured_trace(
                ["pyrtl2rrtl"],
                Path("export.json"),
                Path("lane_trace.json"),
                sweep,
                repeat=2,
                warmup=1,
            )

        run.assert_not_called()
        self.assertFalse(report["available"])
        self.assertEqual(report["error"], "gpu-measured-option-not-selected")
        self.assertEqual(report["replay_ns_best"], 0)

    def test_load_runtime_profile_validates_schema_and_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runtime_profile.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "rrtl-pyrtl-runtime-profile-v1",
                        "recommended_runtime_backend": "rrtl_gpu_measured_trace",
                        "recommended_runtime_source": "measured-gpu",
                        "selected_backend": {
                            "backend": "rrtl_gpu_measured_trace",
                            "selected_gpu_options": {"workgroup_size": 64},
                        },
                    }
                ),
                encoding="utf-8",
            )

            profile = bench.load_runtime_profile(path)

            self.assertEqual(profile["recommended_runtime_backend"], "rrtl_gpu_measured_trace")

            path.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported runtime profile schema"):
                bench.load_runtime_profile(path)

            path.write_text(
                json.dumps(
                    {
                        "schema": "rrtl-pyrtl-runtime-profile-v1",
                        "recommended_runtime_source": "no-valid-measurements",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not select a backend"):
                bench.load_runtime_profile(path)

    def test_run_profile_selected_replay_uses_native_hot_replay(self):
        completed = mock.Mock(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema": "rrtl-pyrtl-profile-replay-hot-v1",
                    "selected_backend": "rrtl_threaded_autotune_trace",
                    "replay_ns_best": 10,
                    "replay_ns_median": 11,
                    "mismatch_count": 0,
                }
            ),
            stderr="",
        )
        profile = {
            "schema": "rrtl-pyrtl-runtime-profile-v1",
            "recommended_runtime_backend": "rrtl_threaded_autotune_trace",
            "recommended_runtime_source": "measured-cpu",
            "selected_backend": {
                "backend": "rrtl_threaded_autotune_trace",
                "selected_threaded_layout": {
                    "workers": [
                        {"backend": "scalar", "lanes": 1},
                        {"backend": "simd-cpu", "lanes": 3},
                    ]
                },
            },
        }
        with mock.patch("rrtl_pyrtl.bench.subprocess.run", return_value=completed) as run:
            report = bench._run_profile_selected_replay(
                profile,
                ["pyrtl2rrtl"],
                Path("export.json"),
                Path("trace.json"),
                Path("lane_trace.json"),
                runtime_profile_path=Path("runtime_profile.json"),
                repeat=2,
                warmup=1,
                lanes=4,
            )

        cmd = run.call_args.args[0]
        self.assertEqual(report["replay_ns_best"], 10)
        self.assertIn("bench-profile-replay", cmd)
        self.assertIn("runtime_profile.json", cmd)
        self.assertIn("--repeat", cmd)
        self.assertIn("2", cmd)
        self.assertIn("--lanes", cmd)
        self.assertIn("4", cmd)

    def test_run_profile_selected_replay_dispatches_gpu_options(self):
        completed = mock.Mock(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema": "rrtl-pyrtl-bench-gpu-trace-v1",
                    "available": True,
                    "replay_ns_best": 10,
                    "replay_ns_median": 11,
                    "mismatch_count": 0,
                }
            ),
            stderr="",
        )
        profile = {
            "schema": "rrtl-pyrtl-runtime-profile-v1",
            "recommended_runtime_backend": "rrtl_gpu_measured_trace",
            "recommended_runtime_source": "measured-gpu",
            "selected_backend": {
                "backend": "rrtl_gpu_measured_trace",
                "selected_gpu_option_index": 2,
                "selected_gpu_options": {
                    "workgroup_size": 128,
                    "memory_layout": "word-major",
                    "reuse_temporaries": True,
                },
            },
        }
        with mock.patch("rrtl_pyrtl.bench.subprocess.run", return_value=completed) as run:
            report = bench._run_profile_selected_replay(
                profile,
                ["pyrtl2rrtl"],
                Path("export.json"),
                Path("trace.json"),
                Path("lane_trace.json"),
                runtime_profile_path=Path("runtime_profile.json"),
                repeat=2,
                warmup=1,
                lanes=4,
            )

        cmd = run.call_args.args[0]
        self.assertEqual(report["replay_ns_best"], 10)
        self.assertIn("bench-profile-replay", cmd)
        self.assertIn("runtime_profile.json", cmd)

    def test_run_profile_selected_replay_dispatches_measured_cpu_backend(self):
        completed = mock.Mock(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema": "rrtl-pyrtl-bench-backends-trace-v1",
                    "backends": [
                        {
                            "backend": "simd-cpu",
                            "replay_ns_best": 10,
                            "replay_ns_median": 11,
                            "mismatch_count": 0,
                        }
                    ],
                }
            ),
            stderr="",
        )
        profile = {
            "schema": "rrtl-pyrtl-runtime-profile-v1",
            "recommended_runtime_backend": "rrtl_backend:simd-cpu",
            "recommended_runtime_source": "measured-cpu",
            "selected_backend": {"backend": "rrtl_backend:simd-cpu"},
        }
        with mock.patch("rrtl_pyrtl.bench.subprocess.run", return_value=completed) as run:
            report = bench._run_profile_selected_replay(
                profile,
                ["pyrtl2rrtl"],
                Path("export.json"),
                Path("trace.json"),
                Path("lane_trace.json"),
                runtime_profile_path=Path("runtime_profile.json"),
                repeat=2,
                warmup=1,
                lanes=4,
            )

        cmd = run.call_args.args[0]
        self.assertEqual(report["backends"][0]["backend"], "simd-cpu")
        self.assertIn("bench-profile-replay", cmd)
        self.assertIn("runtime_profile.json", cmd)

    def test_hot_profile_sweep_ignores_invalid_candidates_and_ranks_lane_step(self):
        profile = {
            "schema": "rrtl-pyrtl-runtime-profile-v1",
            "recommended_runtime_backend": "rrtl_backend:simd-cpu",
            "recommended_runtime_source": "measured-cpu",
            "selected_backend": {"backend": "rrtl_backend:simd-cpu"},
        }
        backend_plan = {
            "selected_threaded_layout": {
                "workers": [{"backend": "simd-cpu", "lanes": 2}]
            }
        }
        threaded_autotune = {
            "selected_threaded_layout": {
                "workers": [{"backend": "scalar", "lanes": 1}, {"backend": "simd-cpu", "lanes": 1}]
            },
            "replay_ns_best": 30,
            "mismatch_count": 0,
        }

        def fake_runner(cmd, phase):
            self.assertEqual(phase, "profile_replay")
            profile_path = next(str(part) for part in cmd if str(part).endswith(".runtime_profile.json"))
            if "backend-simd-cpu" in profile_path:
                return json.dumps(
                    {
                        "schema": "rrtl-pyrtl-profile-replay-hot-v1",
                        "replay_ns_best": 10,
                        "replay_ns_median": 10,
                        "replay_ns_per_lane_step": 9.0,
                        "mismatch_count": 1,
                    }
                )
            lane_step = 3.0 if "autotuned-threaded" in profile_path else 7.0
            backend = "rrtl_threaded_autotune_trace" if "threaded" in profile_path else "rrtl_backend:scalar"
            return json.dumps(
                {
                    "schema": "rrtl-pyrtl-profile-replay-hot-v1",
                    "selected_backend": backend,
                    "replay_ns_best": int(lane_step * 10),
                    "replay_ns_median": int(lane_step * 10),
                    "replay_ns_per_lane_step": lane_step,
                    "mismatch_count": 0,
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            sweep = bench.run_hot_profile_sweep(
                runtime_profile=profile,
                backend_plan=backend_plan,
                rrtl_threaded=None,
                rrtl_threaded_autotune=threaded_autotune,
                rrtl_gpu_measured=None,
                pyrtl2rrtl=["pyrtl2rrtl"],
                export_path=Path("export.json"),
                trace_path=Path("trace.json"),
                lane_trace_path=Path("lane_trace.json"),
                out_dir=Path(tmpdir),
                profile_prefix="unit_sweep",
                repeat=3,
                warmup=0,
                lanes=2,
                command_runner=fake_runner,
            )

        self.assertEqual(sweep["selected_candidate_name"], "autotuned-threaded")
        self.assertEqual(sweep["selected_backend"], "rrtl_threaded_autotune_trace")
        self.assertLess(sweep["valid_candidate_count"], sweep["candidate_count"])
        invalid = next(candidate for candidate in sweep["candidates"] if candidate["candidate_name"] == "backend-simd-cpu")
        self.assertFalse(invalid["valid"])

    def test_evaluate_backend_plan_reports_slower_planned_gpu_option(self):
        evaluation = bench.evaluate_backend_plan(
            rrtl={"replay_ns_best": 100, "replay_ns_median": 100, "mismatch_count": 0},
            rrtl_packed={"replay_ns_best": 90, "replay_ns_median": 90, "mismatch_count": 0},
            rrtl_single={"replay_ns_best": 80, "replay_ns_median": 80, "mismatch_count": 0},
            rrtl_backends={"backends": []},
            rrtl_threaded={
                "replay_ns_best": 70,
                "replay_ns_median": 70,
                "mismatch_count": 0,
            },
            rrtl_gpu={"replay_ns_best": 60, "replay_ns_median": 60, "mismatch_count": 0},
            backend_plan={
                "selected_threaded_layout": {"workers": []},
                "selected_gpu_options": {
                    "workgroup_size": 64,
                    "memory_layout": "lane-major",
                    "reuse_temporaries": False,
                },
            },
            rrtl_gpu_option_sweep={
                "candidates": [
                    {
                        "candidate_index": 0,
                        "planned": {
                            "options": {
                                "workgroup_size": 64,
                                "memory_layout": "lane-major",
                                "reuse_temporaries": False,
                            }
                        },
                        "available": True,
                        "replay_ns_best": 50,
                        "replay_ns_median": 55,
                        "mismatch_count": 0,
                    },
                    {
                        "candidate_index": 1,
                        "planned": {
                            "options": {
                                "workgroup_size": 128,
                                "memory_layout": "lane-major",
                                "reuse_temporaries": False,
                            }
                        },
                        "available": True,
                        "replay_ns_best": 40,
                        "replay_ns_median": 45,
                        "mismatch_count": 0,
                    },
                ]
            },
        )

        self.assertEqual(evaluation["planned_gpu_option_rank"], 2)
        self.assertEqual(evaluation["planned_gpu_option_best_ns"], 50)
        self.assertEqual(evaluation["measured_gpu_option_best_index"], 1)
        self.assertEqual(evaluation["measured_gpu_option_best_ns"], 40)
        self.assertFalse(evaluation["gpu_option_plan_hit"])
        self.assertEqual(
            evaluation["gpu_option_miss_reason"], "planned-gpu-option-slower"
        )

    def test_evaluate_backend_plan_reports_absent_or_invalid_gpu_option(self):
        common = {
            "rrtl": {"replay_ns_best": 100, "replay_ns_median": 100, "mismatch_count": 0},
            "rrtl_packed": {
                "replay_ns_best": 90,
                "replay_ns_median": 90,
                "mismatch_count": 0,
            },
            "rrtl_single": {
                "replay_ns_best": 80,
                "replay_ns_median": 80,
                "mismatch_count": 0,
            },
            "rrtl_backends": {"backends": []},
            "rrtl_threaded": None,
            "rrtl_gpu": None,
        }
        missing = bench.evaluate_backend_plan(
            **common,
            backend_plan={
                "selected_gpu_options": {
                    "workgroup_size": 256,
                    "memory_layout": "lane-major",
                }
            },
            rrtl_gpu_option_sweep={
                "candidates": [
                    {
                        "candidate_index": 0,
                        "planned": {
                            "options": {
                                "workgroup_size": 64,
                                "memory_layout": "lane-major",
                            }
                        },
                        "available": True,
                        "replay_ns_best": 10,
                        "replay_ns_median": 10,
                        "mismatch_count": 0,
                    }
                ]
            },
        )
        invalid = bench.evaluate_backend_plan(
            **common,
            backend_plan={
                "selected_gpu_options": {
                    "workgroup_size": 64,
                    "memory_layout": "lane-major",
                }
            },
            rrtl_gpu_option_sweep={
                "candidates": [
                    {
                        "candidate_index": 0,
                        "planned": {
                            "options": {
                                "workgroup_size": 64,
                                "memory_layout": "lane-major",
                            }
                        },
                        "available": False,
                        "replay_ns_best": 0,
                        "replay_ns_median": 0,
                        "mismatch_count": 0,
                    }
                ]
            },
        )

        self.assertEqual(
            missing["gpu_option_miss_reason"], "planned-option-not-in-sweep"
        )
        self.assertIsNone(missing["planned_gpu_option_rank"])
        self.assertEqual(invalid["gpu_option_miss_reason"], "no-valid-gpu-options")
        self.assertIsNone(invalid["measured_gpu_option_best_index"])

    def test_evaluate_backend_plan_recommends_static_plan_when_fastest(self):
        evaluation = bench.evaluate_backend_plan(
            rrtl={"replay_ns_best": 100, "replay_ns_median": 100, "mismatch_count": 0},
            rrtl_packed={"replay_ns_best": 90, "replay_ns_median": 90, "mismatch_count": 0},
            rrtl_single={"replay_ns_best": 80, "replay_ns_median": 80, "mismatch_count": 0},
            rrtl_backends={"backends": []},
            rrtl_threaded={
                "replay_ns_best": 40,
                "replay_ns_median": 42,
                "mismatch_count": 0,
            },
            rrtl_gpu={"replay_ns_best": 60, "replay_ns_median": 65, "mismatch_count": 0},
            rrtl_gpu_measured={
                "available": True,
                "replay_ns_best": 50,
                "replay_ns_median": 55,
                "mismatch_count": 0,
            },
            backend_plan={"selected_threaded_layout": {"workers": []}},
        )

        self.assertEqual(evaluation["fastest_overall_backend"]["name"], "rrtl_threaded_trace")
        self.assertEqual(evaluation["recommended_runtime_backend"], "rrtl_threaded_trace")
        self.assertEqual(evaluation["recommended_runtime_source"], "static-plan")
        self.assertEqual(evaluation["recommended_backend_reason"], "static-plan-fastest")
        self.assertEqual(evaluation["static_plan_vs_recommended_speedup"], 1.0)

    def test_evaluate_backend_plan_recommends_measured_cpu_when_fastest(self):
        evaluation = bench.evaluate_backend_plan(
            rrtl={"replay_ns_best": 100, "replay_ns_median": 100, "mismatch_count": 0},
            rrtl_packed={"replay_ns_best": 90, "replay_ns_median": 90, "mismatch_count": 0},
            rrtl_single={"replay_ns_best": 80, "replay_ns_median": 80, "mismatch_count": 0},
            rrtl_backends={
                "backends": [
                    {
                        "backend": "simd-cpu",
                        "replay_ns_best": 35,
                        "replay_ns_median": 36,
                        "mismatch_count": 0,
                    }
                ]
            },
            rrtl_threaded={
                "replay_ns_best": 60,
                "replay_ns_median": 62,
                "mismatch_count": 0,
            },
            rrtl_gpu=None,
            rrtl_gpu_measured={
                "available": True,
                "replay_ns_best": 45,
                "replay_ns_median": 47,
                "mismatch_count": 0,
            },
            backend_plan={"selected_threaded_layout": {"workers": []}},
        )

        self.assertEqual(evaluation["fastest_overall_backend"]["name"], "rrtl_backend:simd-cpu")
        self.assertEqual(evaluation["recommended_runtime_backend"], "rrtl_backend:simd-cpu")
        self.assertEqual(evaluation["recommended_runtime_source"], "measured-cpu")
        self.assertEqual(
            evaluation["recommended_backend_reason"],
            "measured-cpu-beats-static-plan",
        )
        self.assertAlmostEqual(
            evaluation["static_plan_vs_recommended_speedup"],
            60 / 35,
        )

    def test_evaluate_backend_plan_recommends_threaded_autotune_when_fastest(self):
        evaluation = bench.evaluate_backend_plan(
            rrtl={"replay_ns_best": 100, "replay_ns_median": 100, "mismatch_count": 0},
            rrtl_packed={"replay_ns_best": 90, "replay_ns_median": 90, "mismatch_count": 0},
            rrtl_single={"replay_ns_best": 80, "replay_ns_median": 80, "mismatch_count": 0},
            rrtl_backends={"backends": []},
            rrtl_threaded={
                "replay_ns_best": 60,
                "replay_ns_median": 62,
                "mismatch_count": 0,
            },
            rrtl_threaded_autotune={
                "replay_ns_best": 35,
                "replay_ns_median": 36,
                "mismatch_count": 0,
            },
            rrtl_gpu=None,
            rrtl_gpu_measured={
                "available": True,
                "replay_ns_best": 45,
                "replay_ns_median": 47,
                "mismatch_count": 0,
            },
            backend_plan={"selected_threaded_layout": {"workers": []}},
        )

        self.assertEqual(
            evaluation["fastest_overall_backend"]["name"],
            "rrtl_threaded_autotune_trace",
        )
        self.assertEqual(
            evaluation["recommended_runtime_backend"],
            "rrtl_threaded_autotune_trace",
        )
        self.assertEqual(evaluation["recommended_runtime_source"], "measured-cpu")
        self.assertEqual(
            evaluation["recommended_backend_reason"],
            "measured-cpu-beats-static-plan",
        )
        self.assertAlmostEqual(
            evaluation["static_plan_vs_recommended_speedup"],
            60 / 35,
        )

    def test_evaluate_backend_plan_reports_no_valid_recommendation(self):
        evaluation = bench.evaluate_backend_plan(
            rrtl={"replay_ns_best": 0, "replay_ns_median": 0, "mismatch_count": 0},
            rrtl_packed={"replay_ns_best": 0, "replay_ns_median": 0, "mismatch_count": 1},
            rrtl_single={
                "available": False,
                "replay_ns_best": 10,
                "replay_ns_median": 10,
                "mismatch_count": 0,
            },
            rrtl_backends={"backends": []},
            rrtl_threaded={
                "replay_ns_best": 0,
                "replay_ns_median": 0,
                "mismatch_count": 0,
            },
            rrtl_gpu=None,
            rrtl_gpu_measured={
                "available": False,
                "replay_ns_best": 0,
                "replay_ns_median": 0,
                "mismatch_count": 0,
            },
            backend_plan={"selected_threaded_layout": {"workers": []}},
        )

        self.assertIsNone(evaluation["fastest_overall_backend"])
        self.assertIsNone(evaluation["recommended_runtime_backend"])
        self.assertEqual(
            evaluation["recommended_runtime_source"],
            "no-valid-measurements",
        )
        self.assertEqual(evaluation["recommended_backend_reason"], "no-valid-measurements")
        self.assertEqual(evaluation["static_plan_vs_recommended_speedup"], 0.0)

    def test_build_runtime_profile_records_measured_gpu_winner(self):
        evaluation = bench.evaluate_backend_plan(
            rrtl={"replay_ns_best": 100, "replay_ns_median": 100, "mismatch_count": 0},
            rrtl_packed={"replay_ns_best": 90, "replay_ns_median": 90, "mismatch_count": 0},
            rrtl_single={"replay_ns_best": 80, "replay_ns_median": 80, "mismatch_count": 0},
            rrtl_backends={"backends": []},
            rrtl_threaded={
                "replay_ns_best": 60,
                "replay_ns_median": 62,
                "mismatch_count": 0,
            },
            rrtl_gpu=None,
            rrtl_gpu_measured={
                "available": True,
                "replay_ns_best": 35,
                "replay_ns_median": 36,
                "mismatch_count": 0,
            },
            backend_plan={"selected_threaded_layout": {"workers": []}},
        )
        profile = bench.build_runtime_profile(
            {
                "config": {"steps": 4, "packed_lanes": 2},
                "backend_plan_evaluation": evaluation,
                "rrtl_gpu_measured_trace": {
                    "available": True,
                    "selected_gpu_option_index": 2,
                    "selected_gpu_options": {
                        "workgroup_size": 128,
                        "memory_layout": "word-major",
                        "reuse_temporaries": True,
                    },
                    "gpu_replay_mode": "fused-kernel",
                    "replay_ns_best": 35,
                    "replay_ns_median": 36,
                    "mismatch_count": 0,
                },
            }
        )

        self.assertEqual(profile["schema"], "rrtl-pyrtl-runtime-profile-v1")
        self.assertEqual(profile["recommended_runtime_backend"], "rrtl_gpu_measured_trace")
        self.assertEqual(profile["recommended_runtime_source"], "measured-gpu")
        self.assertEqual(profile["selected_backend"]["backend"], "rrtl_gpu_measured_trace")
        self.assertEqual(profile["selected_backend"]["selected_gpu_option_index"], 2)
        self.assertTrue(
            profile["selected_backend"]["selected_gpu_options"]["reuse_temporaries"]
        )

    def test_build_runtime_profile_records_threaded_autotune_winner(self):
        evaluation = bench.evaluate_backend_plan(
            rrtl={"replay_ns_best": 100, "replay_ns_median": 100, "mismatch_count": 0},
            rrtl_packed={"replay_ns_best": 90, "replay_ns_median": 90, "mismatch_count": 0},
            rrtl_single={"replay_ns_best": 80, "replay_ns_median": 80, "mismatch_count": 0},
            rrtl_backends={"backends": []},
            rrtl_threaded={
                "replay_ns_best": 60,
                "replay_ns_median": 62,
                "mismatch_count": 0,
            },
            rrtl_threaded_autotune={
                "replay_ns_best": 35,
                "replay_ns_median": 36,
                "mismatch_count": 0,
            },
            rrtl_gpu=None,
            backend_plan={"selected_threaded_layout": {"workers": []}},
        )
        profile = bench.build_runtime_profile(
            {
                "config": {"steps": 4, "packed_lanes": 2},
                "backend_plan_evaluation": evaluation,
                "rrtl_threaded_autotune_trace": {
                    "selected_threaded_layout": {
                        "workers": [
                            {"backend": "scalar", "lanes": 1},
                            {"backend": "simd-cpu", "lanes": 1},
                        ],
                        "max_mismatches": 16,
                    },
                    "selected_reason": "autotune-selected",
                    "autotune": {
                        "selected_candidate": 1,
                        "candidates": [{"candidate_index": 0}, {"candidate_index": 1}],
                    },
                    "autotune_pruned_candidates": [{"candidate_index": 2}],
                    "replay_ns_best": 35,
                    "replay_ns_median": 36,
                    "mismatch_count": 0,
                },
            }
        )

        self.assertEqual(
            profile["recommended_runtime_backend"],
            "rrtl_threaded_autotune_trace",
        )
        self.assertEqual(profile["recommended_runtime_source"], "measured-cpu")
        self.assertEqual(profile["selected_backend"]["backend"], "rrtl_threaded_autotune_trace")
        self.assertEqual(profile["selected_backend"]["autotune_selected_candidate"], 1)
        self.assertEqual(profile["selected_backend"]["autotune_pruned_candidate_count"], 1)
        self.assertEqual(
            profile["selected_backend"]["selected_threaded_layout"]["workers"][1]["backend"],
            "simd-cpu",
        )

    def test_build_runtime_profile_handles_no_valid_backend(self):
        evaluation = bench.evaluate_backend_plan(
            rrtl={"replay_ns_best": 0, "replay_ns_median": 0, "mismatch_count": 0},
            rrtl_packed={"replay_ns_best": 0, "replay_ns_median": 0, "mismatch_count": 1},
            rrtl_single={
                "available": False,
                "replay_ns_best": 10,
                "replay_ns_median": 10,
                "mismatch_count": 0,
            },
            rrtl_backends={"backends": []},
            rrtl_threaded=None,
            rrtl_gpu=None,
            backend_plan={"selected_threaded_layout": {"workers": []}},
        )
        profile = bench.build_runtime_profile(
            {
                "config": {"steps": 4},
                "backend_plan_evaluation": evaluation,
            }
        )

        self.assertIsNone(profile["recommended_runtime_backend"])
        self.assertEqual(profile["recommended_runtime_source"], "no-valid-measurements")
        self.assertEqual(profile["measured_backends"], [])
        self.assertNotIn("selected_backend", profile)

    def test_build_planner_feedback_aggregates_profile_misses_and_gpu_options(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            (out_dir / "hit.runtime_profile.json").write_text(
                json.dumps(
                    {
                        "schema": "rrtl-pyrtl-runtime-profile-v1",
                        "recommended_runtime_backend": "rrtl_threaded_trace",
                        "recommended_runtime_source": "static-plan",
                        "static_plan_vs_recommended_speedup": 1.0,
                        "recommended_backend_reason": "static-plan-fastest",
                        "planner_feedback": {
                            "plan_hit": True,
                            "miss_reason": "none",
                            "gpu_option_plan_hit": True,
                            "gpu_option_miss_reason": "none",
                            "static_profitability_backend": "threaded-mixed",
                            "static_profitability_rank": 1,
                            "static_profitability_hit": True,
                            "static_profitability_miss_reason": "none",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "hit.backend_plan.json").write_text("{}", encoding="utf-8")
            (out_dir / "cpu_miss.runtime_profile.json").write_text(
                json.dumps(
                    {
                        "schema": "rrtl-pyrtl-runtime-profile-v1",
                        "recommended_runtime_backend": "rrtl_threaded_autotune_trace",
                        "recommended_runtime_source": "measured-cpu",
                        "static_plan_vs_recommended_speedup": 1.8,
                        "recommended_backend_reason": "measured-cpu-beats-static-plan",
                        "selected_backend": {
                            "backend": "rrtl_threaded_autotune_trace",
                            "selected_threaded_layout": {
                                "workers": [{"backend": "simd-cpu", "lanes": 2}]
                            },
                        },
                        "planner_feedback": {
                            "plan_hit": False,
                            "miss_reason": "planned-cpu-slower",
                            "planned_cpu_rank": 3,
                            "gpu_option_plan_hit": False,
                            "gpu_option_miss_reason": "gpu-not-planned",
                            "static_profitability_backend": "scalar",
                            "static_profitability_rank": 3,
                            "static_profitability_hit": False,
                            "static_profitability_miss_reason": "static-profitability-slower",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "cpu_miss.backend_plan.json").write_text(
                json.dumps(
                    {
                        "schema": "rrtl-pyrtl-backend-plan-v1",
                        "profitability_features": {
                            "lanes": 2,
                            "steps": 2,
                            "feature_buckets": [
                                "simd_coverage>=70",
                                "native_simd>=50",
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "gpu_miss.runtime_profile.json").write_text(
                json.dumps(
                    {
                        "schema": "rrtl-pyrtl-runtime-profile-v1",
                        "recommended_runtime_backend": "rrtl_gpu_measured_trace",
                        "recommended_runtime_source": "measured-gpu",
                        "static_plan_vs_recommended_speedup": 2.4,
                        "recommended_backend_reason": "measured-gpu-beats-static-plan",
                        "selected_backend": {
                            "backend": "rrtl_gpu_measured_trace",
                            "selected_gpu_options": {
                                "workgroup_size": 128,
                                "memory_layout": "word-major",
                                "reuse_temporaries": False,
                            },
                        },
                        "planner_feedback": {
                            "plan_hit": False,
                            "miss_reason": "planned-gpu-slower",
                            "planned_gpu_rank": 2,
                            "planned_gpu_selected": True,
                            "gpu_option_plan_hit": False,
                            "gpu_option_miss_reason": "planned-gpu-option-slower",
                            "static_profitability_backend": "gpu-fused",
                            "static_profitability_rank": 2,
                            "static_profitability_hit": False,
                            "static_profitability_miss_reason": "static-profitability-slower",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "cpu_miss.hot_profile_replay.json").write_text(
                json.dumps(
                    {
                        "schema": "rrtl-pyrtl-profile-replay-hot-v1",
                        "selected_backend": "rrtl_backend:simd-cpu",
                        "selected_source": "measured-cpu",
                        "repeat": 9,
                        "first_replay_ns": 120,
                        "replay_ns_best": 90,
                        "replay_ns_median": 100,
                        "hot_replay_ns_best": 90,
                        "hot_replay_ns_median": 100,
                        "replay_ns_per_step": 45.0,
                        "replay_ns_per_lane_step": 22.5,
                        "setup_ns_total": 360,
                        "setup_to_replay_ratio": 4.0,
                        "setup_to_hot_ratio": 4.0,
                        "hot_replay_speedup": 120 / 90,
                        "mismatch_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "gpu_miss.hot_profile_replay.json").write_text(
                json.dumps(
                    {
                        "schema": "rrtl-pyrtl-profile-replay-hot-v1",
                        "selected_backend": "rrtl_gpu_measured_trace",
                        "selected_source": "measured-gpu",
                        "repeat": 9,
                        "first_replay_ns": 90,
                        "replay_ns_best": 60,
                        "replay_ns_median": 70,
                        "hot_replay_ns_best": 60,
                        "hot_replay_ns_median": 70,
                        "replay_ns_per_step": 30.0,
                        "replay_ns_per_lane_step": 15.0,
                        "setup_ns_total": 360,
                        "setup_to_replay_ratio": 6.0,
                        "setup_to_hot_ratio": 6.0,
                        "hot_replay_speedup": 1.5,
                        "mismatch_count": 0,
                    }
                ),
                encoding="utf-8",
            )

            feedback = bench.build_planner_feedback(out_dir)
            markdown = bench.render_planner_feedback_markdown(feedback)

        self.assertEqual(feedback["schema"], "rrtl-pyrtl-planner-feedback-v1")
        self.assertEqual(feedback["summary"]["profiles"], 3)
        self.assertEqual(feedback["summary"]["plan_hits"], 1)
        self.assertEqual(feedback["summary"]["plan_misses"], 2)
        self.assertAlmostEqual(feedback["summary"]["plan_hit_rate"], 1 / 3)
        self.assertEqual(feedback["summary"]["miss_reasons"]["planned-cpu-slower"], 1)
        self.assertEqual(feedback["summary"]["miss_reasons"]["planned-gpu-slower"], 1)
        self.assertEqual(
            feedback["summary"]["recommended_backends"]["rrtl_gpu_measured_trace"],
            1,
        )
        self.assertEqual(feedback["summary"]["gpu_option_hits"], 1)
        self.assertEqual(feedback["summary"]["gpu_option_misses"], 2)
        self.assertEqual(feedback["summary"]["static_profitability_hits"], 1)
        self.assertEqual(feedback["summary"]["static_profitability_misses"], 2)
        self.assertAlmostEqual(
            feedback["summary"]["static_profitability_hit_rate"],
            1 / 3,
        )
        self.assertEqual(
            feedback["summary"]["static_profitability_backends"]["gpu-fused"],
            1,
        )
        self.assertEqual(
            feedback["summary"]["static_profitability_miss_reasons"][
                "static-profitability-slower"
            ],
            2,
        )
        self.assertEqual(
            feedback["summary"]["gpu_option_miss_reasons"]["planned-gpu-option-slower"],
            1,
        )
        self.assertEqual(
            feedback["summary"]["static_plan_vs_recommended_speedup"]["buckets"]["gt_2_00x"],
            1,
        )
        self.assertEqual(feedback["summary"]["hot_profile_replay"]["profiles"], 2)
        self.assertEqual(feedback["summary"]["hot_profile_replay"]["valid_profiles"], 2)
        self.assertEqual(
            feedback["summary"]["hot_profile_replay"]["selected_backends"]["rrtl_gpu_measured_trace"],
            1,
        )
        self.assertEqual(
            feedback["summary"]["hot_profile_replay"]["replay_ns_per_lane_step"]["min"],
            15.0,
        )
        self.assertEqual(
            feedback["summary"]["hot_profile_replay"]["setup_to_hot_ratio"]["max"],
            6.0,
        )
        self.assertAlmostEqual(
            feedback["summary"]["hot_profile_replay"]["hot_replay_speedup"]["max"],
            1.5,
        )
        hot_gpu = next(target for target in feedback["targets"] if target["target"] == "gpu_miss")
        cpu_miss = next(target for target in feedback["targets"] if target["target"] == "cpu_miss")
        self.assertEqual(
            cpu_miss["profitability_feature_buckets"],
            ["simd_coverage>=70", "native_simd>=50"],
        )
        self.assertEqual(hot_gpu["hot_selected_backend"], "rrtl_gpu_measured_trace")
        self.assertEqual(hot_gpu["hot_first_replay_ns"], 90)
        self.assertEqual(hot_gpu["hot_setup_ns_total"], 360)
        self.assertEqual(hot_gpu["hot_replay_ns_per_lane_step"], 15.0)
        self.assertEqual(hot_gpu["hot_setup_to_hot_ratio"], 6.0)
        self.assertEqual(hot_gpu["hot_replay_speedup"], 1.5)
        self.assertEqual(feedback["summary"]["worst_misses"][0]["target"], "gpu_miss")
        self.assertGreaterEqual(len(feedback["warnings"]), 1)
        self.assertIn("# RRTL Planner Feedback", markdown)
        self.assertIn("| planned-cpu-slower | 1 |", markdown)
        self.assertIn("Static profitability hit rate: 33.33%", markdown)
        self.assertIn("## Hot Profile Replay", markdown)
        self.assertIn("Median ns per lane-step: 18.75", markdown)
        self.assertIn("Median setup/hot ratio: 5.00x", markdown)
        self.assertIn("Median hot replay speedup:", markdown)
        self.assertIn("| gpu_miss | rrtl_gpu_measured_trace | planned-gpu-slower | 2.40x |", markdown)

        calibration = bench.build_planner_calibration(feedback)
        calibration_md = bench.render_planner_calibration_markdown(calibration)
        self.assertEqual(calibration["schema"], "rrtl-pyrtl-planner-calibration-v1")
        self.assertEqual(
            calibration["summary"]["threaded_layout_preferences"][0]["signature"],
            "simd-cpu:2",
        )
        self.assertEqual(
            calibration["summary"]["gpu_option_preferences"][0]["signature"],
            "workgroup=128,memory=word-major,reuse=false",
        )
        self.assertEqual(
            calibration["summary"]["hot_backend_preferences"][0]["signature"],
            "rrtl_gpu_measured_trace",
        )
        self.assertEqual(
            calibration["summary"]["profitability_backend_preferences"][0]["signature"],
            "gpu-fused",
        )
        self.assertEqual(
            calibration["summary"]["profitability_penalties"][0]["signature"],
            "static-profitability-slower",
        )
        self.assertIn("# RRTL Planner Calibration", calibration_md)
        self.assertIn("## Hot Backend Preferences", calibration_md)
        self.assertIn("## Profitability Backend Preferences", calibration_md)
        self.assertIn("## Profitability Penalties", calibration_md)
        self.assertIn("| simd-cpu:2 | 2.70 | 1 |", calibration_md)

    def test_main_planner_feedback_only_writes_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            (out_dir / "profiled.runtime_profile.json").write_text(
                json.dumps(
                    {
                        "schema": "rrtl-pyrtl-runtime-profile-v1",
                        "recommended_runtime_backend": "rrtl_threaded_trace",
                        "recommended_runtime_source": "static-plan",
                        "static_plan_vs_recommended_speedup": 1.0,
                        "planner_feedback": {"plan_hit": True, "miss_reason": "none"},
                    }
                ),
                encoding="utf-8",
            )

            code = bench.main(["--out-dir", str(out_dir), "--planner-feedback-only"])
            feedback = json.loads((out_dir / "planner_feedback.json").read_text(encoding="utf-8"))
            markdown = (out_dir / "planner_feedback.md").read_text(encoding="utf-8")

        self.assertEqual(code, 0)
        self.assertEqual(feedback["summary"]["profiles"], 1)
        self.assertIn("Plan hits: 1", markdown)

    def test_main_planner_calibration_only_writes_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            feedback = {
                "schema": "rrtl-pyrtl-planner-feedback-v1",
                "out_dir": str(out_dir),
                "summary": {"profiles": 1},
                "targets": [
                    {
                        "recommended_runtime_backend": "rrtl_gpu_measured_trace",
                        "static_plan_vs_recommended_speedup": 2.0,
                        "static_profitability_backend": "simd-cpu",
                        "static_profitability_hit": False,
                        "static_profitability_miss_reason": "static-profitability-slower",
                        "profitability_feature_buckets": [
                            "simd_coverage>=70",
                            "lane_work>=4096",
                        ],
                        "selected_gpu_option_signature": "workgroup=128,memory=word-major,reuse=false",
                    }
                ],
                "warnings": [],
            }
            (out_dir / "planner_feedback.json").write_text(
                json.dumps(feedback),
                encoding="utf-8",
            )

            code = bench.main(["--out-dir", str(out_dir), "--planner-calibration-only"])
            calibration = json.loads(
                (out_dir / "planner_calibration.json").read_text(encoding="utf-8")
            )
            markdown = (out_dir / "planner_calibration.md").read_text(encoding="utf-8")

        self.assertEqual(code, 0)
        self.assertEqual(calibration["summary"]["profiles"], 1)
        self.assertEqual(
            calibration["summary"]["profitability_feature_preferences"][0]["signature"],
            "gpu-fused|lane_work>=4096",
        )
        self.assertEqual(
            calibration["summary"]["profitability_feature_penalties"][0]["signature"],
            "simd-cpu|lane_work>=4096",
        )
        self.assertIn("workgroup=128,memory=word-major,reuse=false", markdown)
        self.assertIn("## Profitability Feature Preferences", markdown)

    def test_main_planner_comparison_only_writes_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            before_dir = tmp / "before"
            after_dir = tmp / "after"
            out_dir = tmp / "compare"
            before_dir.mkdir()
            after_dir.mkdir()
            before_feedback = {
                "schema": "rrtl-pyrtl-planner-feedback-v1",
                "summary": {
                    "profiles": 2,
                    "plan_hit_rate": 0.5,
                    "static_profitability_hit_rate": 0.0,
                    "recommended_backends": {"rrtl_threaded_trace": 2},
                    "miss_reasons": {"planned-cpu-slower": 1},
                    "worst_misses": [],
                },
                "targets": [
                    {
                        "target": "a",
                        "recommended_runtime_backend": "rrtl_threaded_trace",
                        "recommended_runtime_source": "static-plan",
                        "static_profitability_hit": False,
                    },
                    {
                        "target": "b",
                        "recommended_runtime_backend": "rrtl_threaded_trace",
                        "recommended_runtime_source": "static-plan",
                        "static_profitability_hit": False,
                    },
                ],
            }
            after_feedback = {
                "schema": "rrtl-pyrtl-planner-feedback-v1",
                "summary": {
                    "profiles": 2,
                    "plan_hit_rate": 1.0,
                    "static_profitability_hit_rate": 0.5,
                    "recommended_backends": {"rrtl_gpu_measured_trace": 1, "rrtl_threaded_trace": 1},
                    "miss_reasons": {},
                    "worst_misses": [
                        {
                            "target": "b",
                            "recommended_runtime_backend": "rrtl_threaded_trace",
                            "miss_reason": "planned-cpu-slower",
                            "static_plan_vs_recommended_speedup": 1.2,
                        }
                    ],
                },
                "targets": [
                    {
                        "target": "a",
                        "recommended_runtime_backend": "rrtl_gpu_measured_trace",
                        "recommended_runtime_source": "measured-gpu",
                        "static_profitability_hit": True,
                        "profitability_feature_buckets": ["gpu_suitability>=70"],
                    },
                    {
                        "target": "b",
                        "recommended_runtime_backend": "rrtl_threaded_trace",
                        "recommended_runtime_source": "static-plan",
                        "static_profitability_hit": False,
                    },
                ],
            }
            (before_dir / "planner_feedback.json").write_text(
                json.dumps(before_feedback),
                encoding="utf-8",
            )
            (after_dir / "planner_feedback.json").write_text(
                json.dumps(after_feedback),
                encoding="utf-8",
            )

            code = bench.main(
                [
                    "--out-dir",
                    str(out_dir),
                    "--planner-comparison-only",
                    "--before-dir",
                    str(before_dir),
                    "--after-dir",
                    str(after_dir),
                ]
            )
            comparison = json.loads(
                (out_dir / "planner_comparison.json").read_text(encoding="utf-8")
            )
            markdown = (out_dir / "planner_comparison.md").read_text(encoding="utf-8")

        self.assertEqual(code, 0)
        self.assertEqual(comparison["summary"]["shared_targets"], 2)
        self.assertEqual(comparison["summary"]["backend_shift_count"], 1)
        self.assertEqual(comparison["summary"]["static_profitability_improvements"], 1)
        self.assertAlmostEqual(
            comparison["summary"]["static_profitability_hit_rate_delta"],
            0.5,
        )
        self.assertIn("Static profitability hit rate: 0.00% -> 50.00%", markdown)
        self.assertIn(
            "| a | rrtl_threaded_trace | rrtl_gpu_measured_trace | gpu_suitability>=70 |",
            markdown,
        )

    def test_run_benchmark_surfaces_bench_trace_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = mock.Mock(returncode=1, stdout="", stderr="trace mismatches")
            with mock.patch("rrtl_pyrtl.bench.subprocess.run", return_value=completed):
                with self.assertRaisesRegex(RuntimeError, "trace mismatches"):
                    bench.run_benchmark(
                        out_dir=Path(tmpdir),
                        rows=1,
                        cols=1,
                        steps=2,
                        data_width=4,
                        acc_width=16,
                        repeat=1,
                        warmup=0,
                        pyrtl2rrtl=["pyrtl2rrtl"],
                    )

    def test_main_profile_replay_only_writes_profile_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out"
            profile_path = Path(tmpdir) / "runtime_profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "schema": "rrtl-pyrtl-runtime-profile-v1",
                        "recommended_runtime_backend": "rrtl_backend:simd-cpu",
                        "recommended_runtime_source": "measured-cpu",
                        "selected_backend": {"backend": "rrtl_backend:simd-cpu"},
                    }
                ),
                encoding="utf-8",
            )
            completed = mock.Mock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "schema": "rrtl-pyrtl-profile-replay-hot-v1",
                        "selected_backend": "rrtl_backend:simd-cpu",
                        "replay_ns_best": 10,
                        "replay_ns_median": 11,
                        "mismatch_count": 0,
                    }
                ),
                stderr="",
            )
            with mock.patch("rrtl_pyrtl.bench.subprocess.run", return_value=completed):
                code = bench.main(
                    [
                        "--out-dir",
                        str(out_dir),
                        "--rows",
                        "1",
                        "--cols",
                        "1",
                        "--steps",
                        "2",
                        "--data-width",
                        "4",
                        "--acc-width",
                        "16",
                        "--repeat",
                        "1",
                        "--warmup",
                        "0",
                        "--packed-lanes",
                        "2",
                        "--pyrtl2rrtl",
                        "pyrtl2rrtl",
                        "--runtime-profile",
                        str(profile_path),
                        "--profile-replay-only",
                    ]
                )

            replay = json.loads((out_dir / "profile_replay.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(replay["schema"], "rrtl-pyrtl-profile-replay-v1")
        self.assertEqual(replay["selected_backend"], "rrtl_backend:simd-cpu")
        self.assertEqual(replay["replay"]["schema"], "rrtl-pyrtl-profile-replay-hot-v1")
        self.assertEqual(replay["replay"]["selected_backend"], "rrtl_backend:simd-cpu")

    def test_run_benchmark_surfaces_bench_packed_trace_failure(self):
        fake_rrtl = {
            "schema": "rrtl-pyrtl-bench-trace-v1",
            "steps": 2,
            "repeat": 1,
            "warmup": 0,
            "import_ns": 10,
            "setup_ns": 20,
            "replay_ns_samples": [100],
            "replay_ns_best": 100,
            "replay_ns_median": 100,
            "mismatch_count": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            scalar_completed = mock.Mock(returncode=0, stdout=json.dumps(fake_rrtl), stderr="")
            packed_completed = mock.Mock(returncode=1, stdout="", stderr="packed mismatches")
            with mock.patch(
                "rrtl_pyrtl.bench.subprocess.run",
                side_effect=[scalar_completed, packed_completed],
            ):
                with self.assertRaisesRegex(RuntimeError, "packed mismatches"):
                    bench.run_benchmark(
                        out_dir=Path(tmpdir),
                        rows=1,
                        cols=1,
                        steps=2,
                        data_width=4,
                        acc_width=16,
                        repeat=1,
                        warmup=0,
                        pyrtl2rrtl=["pyrtl2rrtl"],
                    )

    def test_run_benchmark_surfaces_bench_single_trace_failure(self):
        fake_rrtl = {
            "schema": "rrtl-pyrtl-bench-trace-v1",
            "steps": 2,
            "repeat": 1,
            "warmup": 0,
            "import_ns": 10,
            "setup_ns": 20,
            "replay_ns_samples": [100],
            "replay_ns_best": 100,
            "replay_ns_median": 100,
            "mismatch_count": 0,
        }
        fake_rrtl_packed = {
            "schema": "rrtl-pyrtl-bench-packed-trace-v1",
            "steps": 2,
            "repeat": 1,
            "warmup": 0,
            "lanes": 1,
            "import_ns": 10,
            "setup_ns": 30,
            "replay_ns_samples": [80],
            "replay_ns_best": 80,
            "replay_ns_median": 80,
            "mismatch_count": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            scalar_completed = mock.Mock(returncode=0, stdout=json.dumps(fake_rrtl), stderr="")
            packed_completed = mock.Mock(
                returncode=0, stdout=json.dumps(fake_rrtl_packed), stderr=""
            )
            single_completed = mock.Mock(returncode=1, stdout="", stderr="single mismatches")
            with mock.patch(
                "rrtl_pyrtl.bench.subprocess.run",
                side_effect=[scalar_completed, packed_completed, single_completed],
            ):
                with self.assertRaisesRegex(RuntimeError, "single mismatches"):
                    bench.run_benchmark(
                        out_dir=Path(tmpdir),
                        rows=1,
                        cols=1,
                        steps=2,
                        data_width=4,
                        acc_width=16,
                        repeat=1,
                        warmup=0,
                        pyrtl2rrtl=["pyrtl2rrtl"],
                    )

    def test_main_returns_one_on_invalid_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code = bench.main(["--out-dir", tmpdir, "--rows", "0"])

        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
