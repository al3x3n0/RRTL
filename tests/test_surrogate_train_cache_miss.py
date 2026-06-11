import contextlib
import io
import importlib.util
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from rrtl_surrogate.gnn_transformer import EVENT_TRAINER_SCHEMA
import rrtl_surrogate.train_cache_miss as train_cache_miss


class TestSurrogateTrainCacheMiss(unittest.TestCase):
    def test_generate_corpus_is_deterministic(self):
        lhs = train_cache_miss.generate_corpus(samples=4, window=3, seed=11)
        rhs = train_cache_miss.generate_corpus(samples=4, window=3, seed=11)

        self.assertEqual(lhs, rhs)
        self.assertEqual(lhs["schema"], train_cache_miss.CORPUS_SCHEMA)
        self.assertEqual(lhs["events"][0]["schema"], train_cache_miss.EVENT_SCHEMA)
        self.assertEqual(lhs["events"][0]["target"], "cache_miss")
        self.assertNotIn("lane", lhs["events"][0])

    def test_generate_corpus_assigns_stable_lanes(self):
        corpus = train_cache_miss.generate_corpus(
            samples=5,
            window=3,
            seed=11,
            lanes=2,
        )

        self.assertEqual([event["lane"] for event in corpus["events"]], [0, 1, 0, 1, 0])
        self.assertEqual(
            [
                event["label"]["cache_miss"]
                for event in train_cache_miss.generate_corpus(
                    samples=5,
                    window=3,
                    seed=11,
                )["events"]
            ],
            [event["label"]["cache_miss"] for event in corpus["events"]],
        )

    def test_tensor_manifest_describes_event_predictor_contract(self):
        corpus = train_cache_miss.generate_corpus(samples=5, window=4, seed=2)
        manifest = train_cache_miss.build_tensor_manifest(corpus)

        self.assertEqual(
            manifest["schema"], "rrtl-surrogate-event-predictor-tensors-v1"
        )
        self.assertEqual(
            manifest["task"]["surrogate_class"],
            "event_predictor",
        )
        self.assertEqual(manifest["task"]["prediction_target"], "cache_miss")
        self.assertEqual(
            manifest["inputs"]["signal_window"]["shape"],
            [5, 4, len(train_cache_miss.SIGNAL_FEATURES)],
        )
        self.assertEqual(
            manifest["inputs"]["program_context"]["features"],
            train_cache_miss.PROGRAM_FEATURES,
        )
        self.assertNotIn("lane_id", manifest["inputs"])

    def test_tensor_manifest_keeps_lanes_as_metadata(self):
        corpus = train_cache_miss.generate_corpus(
            samples=5,
            window=4,
            seed=2,
            lanes=2,
        )
        manifest = train_cache_miss.build_tensor_manifest(corpus)

        self.assertEqual(manifest["batching"]["lane_metadata"], True)
        self.assertEqual(manifest["batching"]["lanes"], [0, 1])
        self.assertNotIn("lane_id", manifest["inputs"])

    def test_tensor_manifest_rejects_mixed_windows(self):
        corpus = train_cache_miss.generate_corpus(samples=2, window=4, seed=3)
        corpus["events"][1]["window_cycles"] = 3

        with self.assertRaisesRegex(ValueError, "window does not match"):
            train_cache_miss.build_tensor_manifest(corpus)

    def test_tensor_bundle_exports_values_in_manifest_order(self):
        corpus = train_cache_miss.generate_corpus(samples=3, window=4, seed=5, lanes=2)
        bundle = train_cache_miss.build_tensor_bundle(corpus)

        first_event = corpus["events"][0]
        self.assertEqual(
            bundle["schema"],
            train_cache_miss.EVENT_TENSOR_BUNDLE_SCHEMA,
        )
        self.assertEqual(
            bundle["manifest"]["inputs"]["signal_window"]["features"],
            train_cache_miss.SIGNAL_FEATURES,
        )
        self.assertEqual(
            bundle["manifest"]["inputs"]["signal_window"]["shape"],
            [3, 4, len(train_cache_miss.SIGNAL_FEATURES)],
        )
        self.assertEqual(
            bundle["manifest"]["inputs"]["program_context"]["features"],
            train_cache_miss.PROGRAM_FEATURES,
        )
        self.assertEqual(len(bundle["inputs"]["signal_window"]), 3)
        self.assertEqual(len(bundle["inputs"]["signal_window"][0]), 4)
        self.assertEqual(
            bundle["inputs"]["signal_window"][0][0],
            [
                first_event["signals"][0][feature]
                for feature in train_cache_miss.SIGNAL_FEATURES
            ],
        )
        self.assertEqual(
            bundle["inputs"]["program_context"][0],
            [
                first_event["program"][feature]
                for feature in train_cache_miss.PROGRAM_FEATURES
            ],
        )
        self.assertEqual(
            bundle["labels"]["predicted_event"],
            [[event["label"]["cache_miss"]] for event in corpus["events"]],
        )
        self.assertEqual(bundle["metadata"]["sample_ids"], [0, 1, 2])
        self.assertEqual(bundle["metadata"]["lanes"], [0, 1, 0])
        self.assertEqual(bundle["metadata"]["target"], "cache_miss")

    def test_stall_event_profile_exports_target_specific_tensors(self):
        corpus = train_cache_miss.generate_corpus(
            samples=3,
            window=4,
            seed=5,
            target="stall_event",
        )
        bundle = train_cache_miss.build_tensor_bundle(corpus, target="stall_event")

        self.assertEqual(corpus["events"][0]["target"], "stall_event")
        self.assertIn("stall_event", corpus["events"][0]["label"])
        self.assertEqual(
            bundle["manifest"]["task"]["prediction_target"],
            "stall_event",
        )
        self.assertEqual(
            bundle["manifest"]["inputs"]["signal_window"]["features"],
            train_cache_miss.STALL_SIGNAL_FEATURES,
        )
        self.assertEqual(
            bundle["manifest"]["inputs"]["program_context"]["features"],
            train_cache_miss.STALL_PROGRAM_FEATURES,
        )
        self.assertEqual(
            bundle["labels"]["predicted_event"],
            [[event["label"]["stall_event"]] for event in corpus["events"]],
        )
        self.assertEqual(bundle["metadata"]["profile"], "stall_event")

    def test_custom_event_profile_exports_target_specific_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "custom_profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "schema": train_cache_miss.EVENT_PROFILE_SCHEMA,
                        "target": "custom_event",
                        "surrogate_id": "custom_event_predictor",
                        "signal_features": ["load"],
                        "program_features": ["pc"],
                        "label": {
                            "name": "custom_event",
                            "kind": "binary",
                            "positive_value": 1,
                        },
                        "input_window_cycles_default": 4,
                        "horizon_cycles_default": 1,
                        "mock_rule": {
                            "kind": "linear_threshold",
                            "threshold": 2,
                            "terms": [
                                {
                                    "source": "signal",
                                    "feature": "load",
                                    "reduction": "sum",
                                    "weight": 1,
                                }
                            ],
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            corpus = train_cache_miss.generate_corpus(
                samples=2,
                window=4,
                seed=5,
                target=None,
                profile_path=profile_path,
            )
            bundle = train_cache_miss.build_tensor_bundle(
                corpus,
                target=None,
                profile_path=profile_path,
            )

        self.assertEqual(corpus["events"][0]["target"], "custom_event")
        self.assertEqual(
            bundle["manifest"]["inputs"]["signal_window"]["features"],
            ["load"],
        )
        self.assertEqual(
            bundle["manifest"]["inputs"]["program_context"]["features"],
            ["pc"],
        )
        self.assertEqual(bundle["metadata"]["target"], "custom_event")

    def test_custom_event_profile_rejects_target_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "custom_profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "schema": train_cache_miss.EVENT_PROFILE_SCHEMA,
                        "target": "custom_event",
                        "surrogate_id": "custom_event_predictor",
                        "signal_features": ["load"],
                        "program_features": ["pc"],
                        "label": {
                            "name": "custom_event",
                            "kind": "binary",
                            "positive_value": 1,
                        },
                        "input_window_cycles_default": 4,
                        "horizon_cycles_default": 1,
                        "mock_rule": {
                            "kind": "linear_threshold",
                            "threshold": 1,
                            "terms": [
                                {
                                    "source": "signal",
                                    "feature": "load",
                                    "reduction": "sum",
                                    "weight": 1,
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "does not match requested target"):
                train_cache_miss.load_event_profile(
                    target="cache_miss",
                    profile_path=profile_path,
                )

    def test_tensor_bundle_rejects_missing_signal_feature(self):
        corpus = train_cache_miss.generate_corpus(samples=2, window=4, seed=6)
        del corpus["events"][0]["signals"][0]["load"]

        with self.assertRaisesRegex(ValueError, "missing feature 'load'"):
            train_cache_miss.build_tensor_bundle(corpus)

    def test_tensor_bundle_rejects_missing_program_feature(self):
        corpus = train_cache_miss.generate_corpus(samples=2, window=4, seed=6)
        del corpus["events"][0]["program"]["stride"]

        with self.assertRaisesRegex(ValueError, "missing feature 'stride'"):
            train_cache_miss.build_tensor_bundle(corpus)

    def test_tensor_bundle_rejects_mixed_targets(self):
        corpus = train_cache_miss.generate_corpus(samples=2, window=4, seed=6)
        corpus["events"][1]["target"] = "branch_miss"

        with self.assertRaisesRegex(ValueError, "target 'branch_miss' does not match"):
            train_cache_miss.build_tensor_bundle(corpus)

    def test_cli_export_tensors_writes_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = Path(tmpdir) / "events.json"
            out = Path(tmpdir) / "tensor_bundle.json"
            corpus = train_cache_miss.generate_corpus(samples=2, window=4, seed=9)
            corpus_path.write_text(
                json.dumps(corpus, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                code = train_cache_miss.main(
                    [
                        "export-tensors",
                        "--corpus",
                        str(corpus_path),
                        "--out",
                        str(out),
                    ]
                )
            bundle = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(bundle["schema"], train_cache_miss.EVENT_TENSOR_BUNDLE_SCHEMA)
        self.assertEqual(len(bundle["inputs"]["signal_window"]), 2)

    def test_train_and_export_writes_manifest_and_rule_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = Path(tmpdir) / "corpus.json"
            out_dir = Path(tmpdir) / "train"
            corpus = train_cache_miss.generate_corpus(samples=6, window=5, seed=4)
            corpus_path.write_text(
                json.dumps(corpus, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            summary = train_cache_miss.train_and_export(
                corpus_path=corpus_path,
                out_dir=out_dir,
                seed=4,
            )
            manifest = json.loads(
                (out_dir / "manifest.json").read_text(encoding="utf-8")
            )
            tensor_bundle = json.loads(
                (out_dir / "tensor_bundle.json").read_text(encoding="utf-8")
            )
            trainer_manifest = json.loads(
                (out_dir / "trainer_manifest.json").read_text(encoding="utf-8")
            )

            self.assertTrue(Path(summary["outputs"]["artifact"]).exists())
            self.assertTrue(Path(summary["outputs"]["tensors"]).exists())
            self.assertTrue(Path(summary["outputs"]["tensor_bundle"]).exists())
            self.assertTrue(Path(summary["outputs"]["trainer_manifest"]).exists())
            self.assertEqual(manifest["artifact"]["path"], "cache_miss_rule.json")
            self.assertTrue((out_dir / manifest["artifact"]["path"]).exists())

        self.assertEqual(summary["samples"], 6)
        self.assertEqual(
            tensor_bundle["schema"],
            train_cache_miss.EVENT_TENSOR_BUNDLE_SCHEMA,
        )
        self.assertEqual(trainer_manifest["schema"], EVENT_TRAINER_SCHEMA)
        self.assertEqual(trainer_manifest["task"]["sample_count"], 6)
        self.assertEqual(manifest["surrogate_class"], "event_predictor")
        self.assertEqual(manifest["model_family"], "rule-baseline")
        self.assertEqual(manifest["task"]["prediction_target"], "cache_miss")
        self.assertEqual(manifest["artifact"]["format"], "mock-event-predictor")
        self.assertEqual(manifest["policy"]["mode"], "telemetry_only")

    def test_train_and_export_writes_stall_event_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = Path(tmpdir) / "corpus.json"
            out_dir = Path(tmpdir) / "train"
            corpus = train_cache_miss.generate_corpus(
                samples=4,
                window=3,
                seed=8,
                target="stall_event",
            )
            corpus_path.write_text(
                json.dumps(corpus, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            summary = train_cache_miss.train_and_export(
                corpus_path=corpus_path,
                out_dir=out_dir,
                seed=8,
                target="stall_event",
            )
            manifest = json.loads(
                (out_dir / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["target"], "stall_event")
        self.assertEqual(manifest["surrogate_id"], "stall_event_predictor")
        self.assertEqual(manifest["artifact"]["path"], "stall_event_rule.json")
        self.assertEqual(manifest["task"]["prediction_target"], "stall_event")
        self.assertEqual(
            manifest["task"]["signal_features"],
            train_cache_miss.STALL_SIGNAL_FEATURES,
        )
        self.assertEqual(manifest["task"]["label"]["name"], "stall_event")

    def test_learned_manifest_uses_onnx_artifact_contract(self):
        corpus = train_cache_miss.generate_corpus(samples=3, window=4, seed=4)
        tensor_manifest = train_cache_miss.build_tensor_manifest(corpus)
        manifest = train_cache_miss.build_learned_manifest(
            corpus,
            artifact_path=Path("model.onnx"),
            artifact_hash="model-hash",
            tensor_manifest=tensor_manifest,
        )

        self.assertEqual(manifest["surrogate_class"], "event_predictor")
        self.assertEqual(manifest["model_family"], "gnn-transformer")
        self.assertEqual(manifest["artifact"]["format"], "onnx")
        self.assertEqual(manifest["artifact"]["path"], "model.onnx")
        self.assertEqual(
            manifest["artifact"]["input_tensors"],
            ["signal_window", "program_context"],
        )
        self.assertEqual(
            manifest["artifact"]["output_tensors"],
            ["event_probability", "predicted_event"],
        )
        self.assertEqual(manifest["artifact"]["opset"], 17)
        self.assertEqual(manifest["policy"]["mode"], "telemetry_only")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for export")
    def test_train_learned_and_export_writes_manifest_and_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = Path(tmpdir) / "corpus.json"
            out_dir = Path(tmpdir) / "train_learned"
            corpus = train_cache_miss.generate_corpus(samples=4, window=3, seed=4)
            corpus_path.write_text(
                json.dumps(corpus, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            summary = train_cache_miss.train_learned_and_export(
                corpus_path=corpus_path,
                out_dir=out_dir,
                epochs=0,
                seed=4,
            )
            manifest = json.loads(
                (out_dir / "manifest.json").read_text(encoding="utf-8")
            )
            trainer_manifest = json.loads(
                (out_dir / "trainer_manifest.json").read_text(encoding="utf-8")
            )

            self.assertTrue(Path(summary["outputs"]["model"]).exists())
            self.assertTrue(Path(summary["outputs"]["manifest"]).exists())
            self.assertTrue(Path(summary["outputs"]["tensors"]).exists())
            self.assertTrue(Path(summary["outputs"]["heldout"]).exists())
            self.assertTrue(Path(summary["outputs"]["tensor_bundle"]).exists())
            self.assertTrue(Path(summary["outputs"]["trainer_manifest"]).exists())
            self.assertEqual(
                set(summary["outputs"]),
                {
                    "model",
                    "manifest",
                    "tensors",
                    "tensor_bundle",
                    "trainer_manifest",
                    "heldout",
                },
            )

        self.assertEqual(summary["schema"], "rrtl-surrogate-event-learned-training-summary-v1")
        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["heldout"], 4)
        self.assertEqual(summary["model"]["family"], "gnn-transformer")
        self.assertIn("accuracy", summary["metrics"])
        self.assertIn(summary["export_backend"], {"torch-onnx", "minimal-onnx"})
        self.assertEqual(summary["metrics"]["export_backend"], summary["export_backend"])
        self.assertEqual(manifest["artifact"]["format"], "onnx")
        self.assertEqual(manifest["model_family"], "gnn-transformer")
        self.assertEqual(trainer_manifest["schema"], EVENT_TRAINER_SCHEMA)
        self.assertEqual(trainer_manifest["task"]["sample_count"], 4)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for export")
    def test_learned_export_falls_back_without_python_onnx(self):
        corpus = train_cache_miss.generate_corpus(samples=3, window=3, seed=4)
        bundle = train_cache_miss.build_tensor_bundle(corpus)
        real_import = __import__

        def import_without_onnx(name, *args, **kwargs):
            if name == "onnx":
                raise ImportError("no onnx")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.onnx"
            with mock.patch("builtins.__import__", side_effect=import_without_onnx):
                metrics = train_cache_miss.export_learned_onnx_model(
                    bundle,
                    model_path,
                    epochs=0,
                    seed=4,
                    hidden_dim=16,
                    heads=2,
                )

            self.assertTrue(model_path.exists())
            self.assertGreater(model_path.stat().st_size, 0)
        self.assertEqual(metrics["export_backend"], "minimal-onnx")
        self.assertIn("accuracy", metrics)

    def test_learned_export_reports_missing_torch_cleanly(self):
        corpus = train_cache_miss.generate_corpus(samples=2, window=3, seed=4)
        bundle = train_cache_miss.build_tensor_bundle(corpus)
        real_import = __import__

        def import_without_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("builtins.__import__", side_effect=import_without_torch):
                with self.assertRaisesRegex(RuntimeError, "PyTorch is required"):
                    train_cache_miss.export_learned_onnx_model(
                        bundle,
                        Path(tmpdir) / "model.onnx",
                        epochs=0,
                        seed=4,
                        hidden_dim=16,
                        heads=2,
                    )

    def test_model_fast_golden_uses_labels_and_fast_predictions(self):
        corpus = train_cache_miss.generate_corpus(samples=3, window=4, seed=9)
        fast_run = self._event_fast_run_from_corpus(corpus)

        golden = train_cache_miss.build_model_fast_golden(
            corpus,
            fast_run,
            op_id="cache0",
        )

        labels = [[event["label"]["cache_miss"]] for event in corpus["events"]]
        predictions = [[item["predicted"]] for item in fast_run["results"]]
        self.assertEqual(
            golden["schema"],
            train_cache_miss.MODEL_FAST_GOLDEN_SCHEMA,
        )
        self.assertEqual(golden["op_id"], "cache0")
        self.assertEqual(golden["op_kind"], "event")
        self.assertEqual(golden["expected"]["items"], 3)
        self.assertEqual(golden["expected"]["surrogate_replacements"], 3)
        self.assertEqual(golden["expected"]["exact_fallbacks"], 0)
        self.assertEqual(golden["expected"]["fail_closed"], 0)
        self.assertEqual(golden["expected"]["shadow_sampled"], 1)
        self.assertEqual(golden["expected_tensors"]["predicted_event"], labels)
        self.assertEqual(golden["actual_tensors"]["predicted_event"], predictions)
        self.assertEqual(golden["max_abs_error"], 0)

    def test_model_fast_golden_rejects_sample_id_mismatch(self):
        corpus = train_cache_miss.generate_corpus(samples=2, window=4, seed=9)
        fast_run = self._event_fast_run_from_corpus(corpus)
        fast_run["results"][1]["sample_id"] = 99

        with self.assertRaisesRegex(ValueError, "sample_id mismatch at index 1"):
            train_cache_miss.build_model_fast_golden(
                corpus,
                fast_run,
                op_id="cache0",
            )

    def test_model_fast_golden_rejects_missing_prediction(self):
        corpus = train_cache_miss.generate_corpus(samples=2, window=4, seed=9)
        fast_run = self._event_fast_run_from_corpus(corpus)
        del fast_run["results"][0]["predicted"]

        with self.assertRaisesRegex(ValueError, "missing predicted value"):
            train_cache_miss.build_model_fast_golden(
                corpus,
                fast_run,
                op_id="cache0",
            )

    def test_model_fast_golden_rejects_non_binary_prediction(self):
        corpus = train_cache_miss.generate_corpus(samples=2, window=4, seed=9)
        fast_run = self._event_fast_run_from_corpus(corpus)
        fast_run["results"][0]["predicted"] = 2

        with self.assertRaisesRegex(ValueError, "predicted must be binary"):
            train_cache_miss.build_model_fast_golden(
                corpus,
                fast_run,
                op_id="cache0",
            )

    def test_cli_model_fast_golden_writes_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = Path(tmpdir) / "events.json"
            fast_run_path = Path(tmpdir) / "fast_run.json"
            out = Path(tmpdir) / "cache0_golden.json"
            corpus = train_cache_miss.generate_corpus(samples=2, window=4, seed=9)
            fast_run = self._event_fast_run_from_corpus(corpus)
            corpus_path.write_text(
                json.dumps(corpus, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            fast_run_path.write_text(
                json.dumps(fast_run, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                code = train_cache_miss.main(
                    [
                        "model-fast-golden",
                        "--corpus",
                        str(corpus_path),
                        "--fast-run",
                        str(fast_run_path),
                        "--op-id",
                        "cache0",
                        "--out",
                        str(out),
                    ]
                )
            golden = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(golden["op_id"], "cache0")
        self.assertEqual(
            golden["expected_tensors"]["predicted_event"][0],
            [corpus["events"][0]["label"]["cache_miss"]],
        )

    def test_cli_generate_writes_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "cache_miss.json"
            code = train_cache_miss.main(
                [
                    "generate",
                    "--out",
                    str(out),
                    "--samples",
                    "3",
                    "--window",
                    "4",
                    "--seed",
                    "9",
                    "--lanes",
                    "2",
                ]
            )
            data = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(len(data["events"]), 3)
        self.assertEqual([event["lane"] for event in data["events"]], [0, 1, 0])

    def test_validate_with_rrtl_reads_output_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.json"
            corpus = Path(tmpdir) / "corpus.json"
            out = Path(tmpdir) / "report.json"
            fake = Path(tmpdir) / "fake_validator.py"
            manifest.write_text("{}", encoding="utf-8")
            corpus.write_text("{}", encoding="utf-8")
            fake.write_text(
                "\n".join(
                    [
                        "import json",
                        "import sys",
                        "out = sys.argv[sys.argv.index('--out') + 1]",
                        "open(out, 'w', encoding='utf-8').write(json.dumps({'schema': 'rrtl-surrogate-event-validation-v1', 'ok': True, 'metrics': {'accuracy': 1.0}}))",
                    ]
                ),
                encoding="utf-8",
            )

            report = train_cache_miss.validate_with_rrtl(
                manifest_path=manifest,
                corpus_path=corpus,
                pyrtl2rrtl=["python", str(fake)],
                out_path=out,
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["metrics"]["accuracy"], 1.0)

    def test_shadow_with_rrtl_reads_output_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.json"
            corpus = Path(tmpdir) / "corpus.json"
            out = Path(tmpdir) / "shadow.json"
            fake = Path(tmpdir) / "fake_shadow.py"
            manifest.write_text("{}", encoding="utf-8")
            corpus.write_text("{}", encoding="utf-8")
            fake.write_text(
                "\n".join(
                    [
                        "import json",
                        "import sys",
                        "out = sys.argv[sys.argv.index('--out') + 1]",
                        "open(out, 'w', encoding='utf-8').write(json.dumps({'schema': 'rrtl-surrogate-event-shadow-v1', 'ok': True, 'results': [{'sample_id': 0, 'ok': True}]}))",
                    ]
                ),
                encoding="utf-8",
            )

            report = train_cache_miss.shadow_with_rrtl(
                manifest_path=manifest,
                corpus_path=corpus,
                pyrtl2rrtl=["python", str(fake)],
                out_path=out,
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["results"][0]["sample_id"], 0)

    def test_inspect_with_rrtl_reads_output_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = Path(tmpdir) / "corpus.json"
            out = Path(tmpdir) / "inspect.json"
            fake = Path(tmpdir) / "fake_inspector.py"
            corpus.write_text("{}", encoding="utf-8")
            fake.write_text(
                "\n".join(
                    [
                        "import json",
                        "import sys",
                        "out = sys.argv[sys.argv.index('--out') + 1]",
                        "open(out, 'w', encoding='utf-8').write(json.dumps({'schema': 'rrtl-surrogate-event-inspection-v1', 'ok': True, 'corpus': {'samples': 3}}))",
                    ]
                ),
                encoding="utf-8",
            )

            report = train_cache_miss.inspect_with_rrtl(
                corpus_path=corpus,
                pyrtl2rrtl=["python", str(fake)],
                out_path=out,
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["corpus"]["samples"], 3)

    def test_emit_with_rrtl_reads_output_corpus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace = Path(tmpdir) / "trace.json"
            config = Path(tmpdir) / "config.json"
            out = Path(tmpdir) / "events.json"
            fake = Path(tmpdir) / "fake_emitter.py"
            trace.write_text("{}", encoding="utf-8")
            config.write_text("{}", encoding="utf-8")
            fake.write_text(
                "\n".join(
                    [
                        "import json",
                        "import sys",
                        "out = sys.argv[sys.argv.index('--out') + 1]",
                        "open(out, 'w', encoding='utf-8').write(json.dumps({'schema': 'rrtl-surrogate-instrumentation-corpus-v1', 'events': [{'sample_id': 0}]}))",
                    ]
                ),
                encoding="utf-8",
            )

            corpus = train_cache_miss.emit_with_rrtl(
                trace_path=trace,
                config_path=config,
                pyrtl2rrtl=["python", str(fake)],
                out_path=out,
            )

        self.assertEqual(corpus["schema"], train_cache_miss.CORPUS_SCHEMA)
        self.assertEqual(corpus["events"][0]["sample_id"], 0)

    def test_emit_instrumented_with_rrtl_reads_output_corpus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace = Path(tmpdir) / "instrumentation.json"
            config = Path(tmpdir) / "config.json"
            out = Path(tmpdir) / "events.json"
            fake = Path(tmpdir) / "fake_emitter.py"
            trace.write_text("{}", encoding="utf-8")
            config.write_text("{}", encoding="utf-8")
            fake.write_text(
                "\n".join(
                    [
                        "import json",
                        "import sys",
                        "assert 'emit-instrumented-events' in sys.argv",
                        "out = sys.argv[sys.argv.index('--out') + 1]",
                        "open(out, 'w', encoding='utf-8').write(json.dumps({'schema': 'rrtl-surrogate-instrumentation-corpus-v1', 'events': [{'sample_id': 3, 'target': 'cache_miss'}]}))",
                    ]
                ),
                encoding="utf-8",
            )

            corpus = train_cache_miss.emit_instrumented_with_rrtl(
                trace_path=trace,
                config_path=config,
                pyrtl2rrtl=["python", str(fake)],
                out_path=out,
            )

        self.assertEqual(corpus["schema"], train_cache_miss.CORPUS_SCHEMA)
        self.assertEqual(corpus["events"][0]["sample_id"], 3)
        self.assertEqual(corpus["events"][0]["target"], "cache_miss")

    def test_inspect_instrumentation_with_rrtl_reads_output_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace = Path(tmpdir) / "instrumentation.json"
            config = Path(tmpdir) / "config.json"
            out = Path(tmpdir) / "inspection.json"
            fake = Path(tmpdir) / "fake_inspector.py"
            trace.write_text("{}", encoding="utf-8")
            config.write_text("{}", encoding="utf-8")
            fake.write_text(
                "\n".join(
                    [
                        "import json",
                        "import sys",
                        "assert 'inspect-instrumentation' in sys.argv",
                        "assert '--config' in sys.argv",
                        "out = sys.argv[sys.argv.index('--out') + 1]",
                        "open(out, 'w', encoding='utf-8').write(json.dumps({'schema': 'rrtl-instrumentation-trace-inspection-v1', 'ok': True, 'steps': 4, 'compatibility': {'emittable_samples': 2}}))",
                    ]
                ),
                encoding="utf-8",
            )

            report = train_cache_miss.inspect_instrumentation_with_rrtl(
                trace_path=trace,
                config_path=config,
                pyrtl2rrtl=["python", str(fake)],
                out_path=out,
            )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "rrtl-instrumentation-trace-inspection-v1")
        self.assertEqual(report["steps"], 4)
        self.assertEqual(report["compatibility"]["emittable_samples"], 2)

    def test_catalog_use_cases_cli_includes_builtin_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "catalog.json"

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(["catalog-use-cases", "--out", str(out)])

            catalog = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(catalog, json.loads(stdout.getvalue()))
        self.assertEqual(catalog["schema"], train_cache_miss.USE_CASE_CATALOG_SCHEMA)
        self.assertIn("cache_miss", catalog["targets"])
        self.assertIn("stall_event", catalog["targets"])
        cache_miss = {
            entry["target"]: entry for entry in catalog["use_cases"]
        }["cache_miss"]
        self.assertEqual(cache_miss["surrogate_class"], "event_predictor")
        self.assertEqual(cache_miss["supported_model_modes"], ["rule", "learned"])
        self.assertIn("pending_misses", cache_miss["signal_features"])

    def test_catalog_use_cases_cli_includes_custom_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = Path(tmpdir) / "branch_taken.json"
            out = Path(tmpdir) / "catalog.json"
            custom.write_text(
                json.dumps(
                    {
                        "schema": train_cache_miss.EVENT_PROFILE_SCHEMA,
                        "target": "branch_taken",
                        "surrogate_id": "branch_taken_predictor",
                        "signal_features": ["cycle_delta", "branch_valid"],
                        "program_features": ["pc", "opcode_id"],
                        "label": {
                            "name": "branch_taken",
                            "kind": "binary",
                            "positive_value": 1,
                        },
                        "input_window_cycles_default": 4,
                        "horizon_cycles_default": 1,
                        "mock_rule": {
                            "kind": "linear_threshold",
                            "threshold": 1,
                            "terms": [
                                {
                                    "source": "signal",
                                    "feature": "branch_valid",
                                    "reduction": "sum",
                                    "weight": 1,
                                }
                            ],
                        },
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                code = train_cache_miss.main(
                    ["catalog-use-cases", "--out", str(out), "--profile", str(custom)]
                )

            catalog = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertIn("branch_taken", catalog["targets"])
        custom_entry = {
            entry["target"]: entry for entry in catalog["use_cases"]
        }["branch_taken"]
        self.assertEqual(custom_entry["surrogate_id"], "branch_taken_predictor")
        self.assertEqual(custom_entry["input_window_cycles_default"], 4)

    def test_catalog_use_cases_rejects_invalid_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.json"
            bad.write_text(
                json.dumps({"schema": train_cache_miss.EVENT_PROFILE_SCHEMA}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "event profile target"):
                train_cache_miss.build_use_case_catalog(profile_paths=[bad])

    def test_instrumentation_use_case_match_accepts_matching_config(self):
        report = train_cache_miss.build_instrumentation_use_case_match(
            config=self._cache_miss_emitter_config(),
            use_case=train_cache_miss.build_use_case_contract(
                train_cache_miss.load_event_profile(target="cache_miss")
            ),
            instrumentation_inspection=self._instrumentation_inspection(),
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(
            report["schema"],
            train_cache_miss.INSTRUMENTATION_USE_CASE_MATCH_SCHEMA,
        )
        self.assertTrue(report["target_match"])
        self.assertTrue(report["label_match"])
        self.assertEqual(report["feature_match"]["missing_signal_features"], [])
        self.assertEqual(report["instrumentation_compatibility"]["emittable_samples"], 2)

    def test_instrumentation_use_case_match_rejects_target_mismatch(self):
        config = self._cache_miss_emitter_config()
        config["target"] = "stall_event"

        report = train_cache_miss.build_instrumentation_use_case_match(
            config=config,
            use_case=train_cache_miss.build_use_case_contract(
                train_cache_miss.load_event_profile(target="cache_miss")
            ),
            instrumentation_inspection=self._instrumentation_inspection(),
        )

        self.assertFalse(report["ok"])
        self.assertIn("target mismatch", "\n".join(report["errors"]))

    def test_instrumentation_use_case_match_rejects_missing_signal_feature(self):
        config = self._cache_miss_emitter_config()
        config["signal_features"] = [
            item for item in config["signal_features"] if item["name"] != "tag_delta"
        ]

        report = train_cache_miss.build_instrumentation_use_case_match(
            config=config,
            use_case=train_cache_miss.build_use_case_contract(
                train_cache_miss.load_event_profile(target="cache_miss")
            ),
            instrumentation_inspection=self._instrumentation_inspection(),
        )

        self.assertFalse(report["ok"])
        self.assertEqual(report["feature_match"]["missing_signal_features"], ["tag_delta"])

    def test_instrumentation_use_case_match_rejects_missing_program_feature(self):
        config = self._cache_miss_emitter_config()
        config["program_features"] = [
            item for item in config["program_features"] if item["name"] != "stride"
        ]

        report = train_cache_miss.build_instrumentation_use_case_match(
            config=config,
            use_case=train_cache_miss.build_use_case_contract(
                train_cache_miss.load_event_profile(target="cache_miss")
            ),
            instrumentation_inspection=self._instrumentation_inspection(),
        )

        self.assertFalse(report["ok"])
        self.assertEqual(report["feature_match"]["missing_program_features"], ["stride"])

    def test_instrumentation_use_case_match_rejects_label_mismatch(self):
        config = self._cache_miss_emitter_config()
        config["label"]["name"] = "wrong_label"

        report = train_cache_miss.build_instrumentation_use_case_match(
            config=config,
            use_case=train_cache_miss.build_use_case_contract(
                train_cache_miss.load_event_profile(target="cache_miss")
            ),
            instrumentation_inspection=self._instrumentation_inspection(),
        )

        self.assertFalse(report["ok"])
        self.assertFalse(report["label_match"])
        self.assertIn("label mismatch", "\n".join(report["errors"]))

    def test_match_instrumentation_use_case_cli_writes_out_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace = Path(tmpdir) / "instrumentation.json"
            config = Path(tmpdir) / "config.json"
            out = Path(tmpdir) / "match.json"
            fake = Path(tmpdir) / "fake_inspector.py"
            trace.write_text("{}", encoding="utf-8")
            config.write_text(
                json.dumps(self._cache_miss_emitter_config(), indent=2),
                encoding="utf-8",
            )
            fake.write_text(
                "\n".join(
                    [
                        "import json",
                        "import sys",
                        "assert 'inspect-instrumentation' in sys.argv",
                        "print(json.dumps({'schema': 'rrtl-instrumentation-trace-inspection-v1', 'ok': True, 'compatibility': {'target': 'cache_miss', 'window_cycles': 2, 'horizon_cycles': 1, 'emittable_samples': 2, 'missing_fields': []}, 'warnings': [], 'errors': []}))",
                    ]
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(
                    [
                        "match-instrumentation-use-case",
                        "--trace",
                        str(trace),
                        "--config",
                        str(config),
                        "--out",
                        str(out),
                        "--pyrtl2rrtl",
                        "python",
                        str(fake),
                    ]
                )

            report = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(code, 0, report)
        self.assertEqual(stdout.getvalue(), "")
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["use_case"]["target"], "cache_miss")

    def test_inspect_flow_cli_returns_zero_for_ok_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = Path(tmpdir) / "flow_bundle.json"
            bundle.write_text(
                json.dumps(
                    {
                        "schema": train_cache_miss.FLOW_BUNDLE_SCHEMA,
                        "ok": True,
                        "readiness": {"runtime_ready": True},
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(["inspect-flow", "--bundle", str(bundle)])

        self.assertEqual(code, 0)
        self.assertTrue(json.loads(stdout.getvalue())["ok"])

    def test_inspect_flow_cli_returns_nonzero_for_failed_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = Path(tmpdir) / "flow_bundle.json"
            bundle.write_text(
                json.dumps(
                    {
                        "schema": train_cache_miss.FLOW_BUNDLE_SCHEMA,
                        "ok": False,
                        "errors": ["runtime not ready"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(["inspect-flow", "--bundle", str(bundle)])

        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout.getvalue())["errors"], ["runtime not ready"])

    def test_quality_gate_cli_returns_zero_for_good_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = self._write_quality_gate_bundle(Path(tmpdir))

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(["quality-gate", "--bundle", str(bundle)])

        report = json.loads(stdout.getvalue())
        self.assertEqual(code, 0, report)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], train_cache_miss.QUALITY_GATE_SCHEMA)
        self.assertTrue(report["checks"]["validation_accuracy"]["ok"])
        self.assertTrue(report["checks"]["runtime_ready"]["ok"])

    def test_quality_gate_fails_on_validation_accuracy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._write_quality_gate_bundle(root)
            self._write_json(root / "validate.json", self._validation_report(accuracy=0.5))

            report = train_cache_miss.inspect_quality_gate(bundle_path=bundle)

        self.assertFalse(report["ok"])
        self.assertIn("validation accuracy below threshold", "\n".join(report["errors"]))

    def test_quality_gate_fails_on_shadow_accuracy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._write_quality_gate_bundle(root)
            self._write_json(root / "shadow.json", self._shadow_report(accuracy=0.5))

            report = train_cache_miss.inspect_quality_gate(bundle_path=bundle)

        self.assertFalse(report["ok"])
        self.assertIn("shadow accuracy below threshold", "\n".join(report["errors"]))

    def test_quality_gate_fails_on_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._write_quality_gate_bundle(root)
            self._write_json(root / "fast_run.json", self._fast_run_report(fail_closed=1))

            report = train_cache_miss.inspect_quality_gate(bundle_path=bundle)

        self.assertFalse(report["ok"])
        self.assertIn("fail_closed above threshold", "\n".join(report["errors"]))

    def test_quality_gate_fails_on_model_fast_not_ok(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._write_quality_gate_bundle(root)
            self._write_json(root / "model_fast_report.json", {"ok": False})

            report = train_cache_miss.inspect_quality_gate(bundle_path=bundle)

        self.assertFalse(report["ok"])
        self.assertIn("model FAST report is not ok", report["errors"])

    def test_quality_gate_fails_on_runtime_not_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._write_quality_gate_bundle(root)
            self._write_json(root / "runtime_execution.json", {"ready": False})

            report = train_cache_miss.inspect_quality_gate(bundle_path=bundle)

        self.assertFalse(report["ok"])
        self.assertIn("runtime execution is not ready", report["errors"])

    def test_quality_gate_accepts_custom_thresholds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._write_quality_gate_bundle(root)
            thresholds = root / "thresholds.json"
            out = root / "quality_gate.json"
            self._write_json(root / "validate.json", self._validation_report(accuracy=0.5))
            self._write_json(root / "shadow.json", self._shadow_report(accuracy=0.5))
            self._write_json(
                thresholds,
                {
                    "min_validation_accuracy": 0.5,
                    "min_shadow_accuracy": 0.5,
                    "max_fail_closed": 0,
                    "max_shadow_failed": 0,
                    "require_model_fast_ok": True,
                    "require_runtime_ready": True,
                },
            )

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(
                    [
                        "quality-gate",
                        "--bundle",
                        str(bundle),
                        "--thresholds",
                        str(thresholds),
                        "--out",
                        str(out),
                    ]
                )

            report = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(code, 0, report)
        self.assertEqual(stdout.getvalue(), "")
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["thresholds"]["min_validation_accuracy"], 0.5)

    def test_package_runtime_handoff_cli_returns_zero_for_good_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = self._write_runtime_handoff_bundle(Path(tmpdir))
            out = Path(tmpdir) / "handoff.json"

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(
                    ["package-runtime-handoff", "--bundle", str(bundle), "--out", str(out)]
                )

            handoff = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(code, 0, handoff)
        self.assertEqual(stdout.getvalue(), "")
        self.assertTrue(handoff["ok"], handoff)
        self.assertEqual(handoff["schema"], train_cache_miss.RUNTIME_HANDOFF_SCHEMA)
        self.assertEqual(handoff["target"], "cache_miss")
        self.assertEqual(handoff["manifest"]["surrogate_id"], "cache_miss_event_predictor")
        self.assertTrue(handoff["runtime"]["ready"])

    def test_package_runtime_handoff_fails_without_quality_gate_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._write_runtime_handoff_bundle(root)
            data = json.loads(bundle.read_text(encoding="utf-8"))
            data["readiness"]["quality_gate"] = False
            self._write_json(bundle, data)

            report = train_cache_miss.package_runtime_handoff(bundle_path=bundle)

        self.assertFalse(report["ok"])
        self.assertIn("flow bundle quality_gate readiness is not true", report["errors"])

    def test_package_runtime_handoff_fails_when_runtime_not_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._write_runtime_handoff_bundle(root)
            self._write_json(root / "runtime_execution.json", {"ready": False})

            report = train_cache_miss.package_runtime_handoff(bundle_path=bundle)

        self.assertFalse(report["ok"])
        self.assertIn("runtime execution is not ready", report["errors"])

    def test_package_runtime_handoff_fails_when_required_artifact_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._write_runtime_handoff_bundle(root)
            data = json.loads(bundle.read_text(encoding="utf-8"))
            data["artifacts"].pop("runtime_plan")
            self._write_json(bundle, data)

            report = train_cache_miss.package_runtime_handoff(bundle_path=bundle)

        self.assertFalse(report["ok"])
        self.assertIn("runtime_plan artifact not present", report["errors"])

    def test_inspect_runtime_telemetry_cli_returns_zero_for_matching_telemetry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle, telemetry = self._write_runtime_telemetry_gate_inputs(Path(tmpdir))

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(
                    [
                        "inspect-runtime-telemetry",
                        "--bundle",
                        str(bundle),
                        "--telemetry",
                        str(telemetry),
                    ]
                )

        report = json.loads(stdout.getvalue())
        self.assertEqual(code, 0, report)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], train_cache_miss.RUNTIME_TELEMETRY_GATE_SCHEMA)
        self.assertTrue(report["telemetry_has_surrogate_execution"])
        self.assertTrue(report["runtime_ready"])
        self.assertEqual(report["plan_schema_match"], True)
        self.assertEqual(report["counter_matches"]["attached_items"], True)

    def test_inspect_runtime_telemetry_cli_fails_when_execution_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle, telemetry = self._write_runtime_telemetry_gate_inputs(Path(tmpdir))
            telemetry.write_text(
                json.dumps({"format_version": 1, "module_name": "cache0"}),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(
                    [
                        "inspect-runtime-telemetry",
                        "--bundle",
                        str(bundle),
                        "--telemetry",
                        str(telemetry),
                    ]
                )

        report = json.loads(stdout.getvalue())
        self.assertEqual(code, 1)
        self.assertIn("telemetry missing surrogate_execution", report["errors"])

    def test_inspect_runtime_telemetry_cli_fails_when_not_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle, telemetry = self._write_runtime_telemetry_gate_inputs(Path(tmpdir))
            data = json.loads(telemetry.read_text(encoding="utf-8"))
            data["surrogate_execution"]["ready"] = False
            telemetry.write_text(json.dumps(data), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(
                    [
                        "inspect-runtime-telemetry",
                        "--bundle",
                        str(bundle),
                        "--telemetry",
                        str(telemetry),
                    ]
                )

        report = json.loads(stdout.getvalue())
        self.assertEqual(code, 1)
        self.assertIn("surrogate_execution is not ready", report["errors"])

    def test_inspect_runtime_telemetry_cli_fails_on_counter_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle, telemetry = self._write_runtime_telemetry_gate_inputs(Path(tmpdir))
            data = json.loads(telemetry.read_text(encoding="utf-8"))
            data["surrogate_execution"]["attached_items"] = 1
            telemetry.write_text(json.dumps(data), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(
                    [
                        "inspect-runtime-telemetry",
                        "--bundle",
                        str(bundle),
                        "--telemetry",
                        str(telemetry),
                    ]
                )

        report = json.loads(stdout.getvalue())
        self.assertEqual(code, 1)
        self.assertEqual(report["counter_matches"]["attached_items"], False)
        self.assertIn("attached_items mismatch", "\n".join(report["errors"]))

    def test_inspect_runtime_telemetry_cli_writes_out_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle, telemetry = self._write_runtime_telemetry_gate_inputs(Path(tmpdir))
            out = Path(tmpdir) / "runtime_gate.json"

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(
                    [
                        "inspect-runtime-telemetry",
                        "--bundle",
                        str(bundle),
                        "--telemetry",
                        str(telemetry),
                        "--out",
                        str(out),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(stdout.getvalue(), "")
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(report["ok"], report)

    def test_inspect_runtime_handoff_telemetry_cli_returns_zero_for_matching_telemetry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handoff, telemetry = self._write_runtime_handoff_telemetry_gate_inputs(
                Path(tmpdir)
            )

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(
                    [
                        "inspect-runtime-handoff-telemetry",
                        "--handoff",
                        str(handoff),
                        "--telemetry",
                        str(telemetry),
                    ]
                )

        report = json.loads(stdout.getvalue())
        self.assertEqual(code, 0, report)
        self.assertTrue(report["ok"], report)
        self.assertEqual(
            report["schema"],
            train_cache_miss.RUNTIME_HANDOFF_TELEMETRY_GATE_SCHEMA,
        )
        self.assertTrue(report["handoff_ok"])
        self.assertTrue(report["handoff_runtime_ready"])
        self.assertEqual(report["plan_schema_match"], True)
        self.assertEqual(report["counter_matches"]["attached_items"], True)

    def test_inspect_runtime_handoff_telemetry_cli_fails_when_handoff_not_ok(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handoff, telemetry = self._write_runtime_handoff_telemetry_gate_inputs(
                Path(tmpdir)
            )
            data = json.loads(handoff.read_text(encoding="utf-8"))
            data["ok"] = False
            self._write_json(handoff, data)

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(
                    [
                        "inspect-runtime-handoff-telemetry",
                        "--handoff",
                        str(handoff),
                        "--telemetry",
                        str(telemetry),
                    ]
                )

        report = json.loads(stdout.getvalue())
        self.assertEqual(code, 1)
        self.assertIn("runtime handoff is not ok", report["errors"])

    def test_inspect_runtime_handoff_telemetry_cli_fails_when_execution_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handoff, telemetry = self._write_runtime_handoff_telemetry_gate_inputs(
                Path(tmpdir)
            )
            self._write_json(telemetry, {"format_version": 1, "module_name": "cache0"})

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(
                    [
                        "inspect-runtime-handoff-telemetry",
                        "--handoff",
                        str(handoff),
                        "--telemetry",
                        str(telemetry),
                    ]
                )

        report = json.loads(stdout.getvalue())
        self.assertEqual(code, 1)
        self.assertIn("telemetry missing surrogate_execution", report["errors"])

    def test_inspect_runtime_handoff_telemetry_cli_fails_when_not_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handoff, telemetry = self._write_runtime_handoff_telemetry_gate_inputs(
                Path(tmpdir)
            )
            data = json.loads(telemetry.read_text(encoding="utf-8"))
            data["surrogate_execution"]["ready"] = False
            self._write_json(telemetry, data)

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(
                    [
                        "inspect-runtime-handoff-telemetry",
                        "--handoff",
                        str(handoff),
                        "--telemetry",
                        str(telemetry),
                    ]
                )

        report = json.loads(stdout.getvalue())
        self.assertEqual(code, 1)
        self.assertIn("surrogate_execution is not ready", report["errors"])

    def test_inspect_runtime_handoff_telemetry_cli_fails_on_counter_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handoff, telemetry = self._write_runtime_handoff_telemetry_gate_inputs(
                Path(tmpdir)
            )
            data = json.loads(telemetry.read_text(encoding="utf-8"))
            data["surrogate_execution"]["attached_items"] = 1
            self._write_json(telemetry, data)

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(
                    [
                        "inspect-runtime-handoff-telemetry",
                        "--handoff",
                        str(handoff),
                        "--telemetry",
                        str(telemetry),
                    ]
                )

        report = json.loads(stdout.getvalue())
        self.assertEqual(code, 1)
        self.assertEqual(report["counter_matches"]["attached_items"], False)
        self.assertIn("attached_items mismatch", "\n".join(report["errors"]))

    def test_inspect_runtime_handoff_telemetry_cli_writes_out_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handoff, telemetry = self._write_runtime_handoff_telemetry_gate_inputs(
                Path(tmpdir)
            )
            out = Path(tmpdir) / "runtime_handoff_gate.json"

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = train_cache_miss.main(
                    [
                        "inspect-runtime-handoff-telemetry",
                        "--handoff",
                        str(handoff),
                        "--telemetry",
                        str(telemetry),
                        "--out",
                        str(out),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(stdout.getvalue(), "")
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(report["ok"], report)

    def _write_runtime_telemetry_gate_inputs(self, tmpdir: Path) -> tuple[Path, Path]:
        runtime_execution = tmpdir / "runtime_execution.json"
        runtime_execution.write_text(
            json.dumps(
                {
                    "ready": True,
                    "executed": False,
                    "plan_schema": "rrtl-surrogate-event-runtime-plan-v1",
                    "total_lanes": 2,
                    "attached_items": 4,
                    "surrogate_eligible_items": 2,
                    "exact_fallback_items": 2,
                    "invalid_items": 0,
                    "shadow_compared_items": 2,
                    "shadow_passed_items": 2,
                    "shadow_failed_items": 0,
                    "shadow_unavailable_items": 0,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        bundle = tmpdir / "flow_bundle.json"
        bundle.write_text(
            json.dumps(
                {
                    "schema": train_cache_miss.FLOW_BUNDLE_SCHEMA,
                    "ok": True,
                    "samples": 4,
                    "artifacts": {"runtime_execution": str(runtime_execution)},
                    "counters": {
                        "lanes": 2,
                        "surrogate_replacements": 2,
                        "exact_fallbacks": 2,
                        "shadow_sampled": 2,
                        "shadow_passed": 2,
                        "shadow_failed": 0,
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        telemetry = tmpdir / "runtime_telemetry.json"
        telemetry.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "module_name": "cache0",
                    "surrogate_execution": {
                        "ready": True,
                        "executed": False,
                        "plan_schema": "rrtl-surrogate-event-runtime-plan-v1",
                        "total_lanes": 2,
                        "attached_items": 4,
                        "surrogate_eligible_items": 2,
                        "exact_fallback_items": 2,
                        "invalid_items": 0,
                        "shadow_compared_items": 2,
                        "shadow_passed_items": 2,
                        "shadow_failed_items": 0,
                        "shadow_unavailable_items": 0,
                        "diagnostics": [],
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return bundle, telemetry

    def _write_runtime_handoff_telemetry_gate_inputs(
        self,
        tmpdir: Path,
    ) -> tuple[Path, Path]:
        runtime_execution = tmpdir / "runtime_execution.json"
        self._write_json(
            runtime_execution,
            {
                "ready": True,
                "executed": False,
                "plan_schema": "rrtl-surrogate-event-runtime-plan-v1",
                "total_lanes": 2,
                "attached_items": 4,
                "surrogate_eligible_items": 2,
                "exact_fallback_items": 2,
                "invalid_items": 0,
                "shadow_compared_items": 2,
                "shadow_passed_items": 2,
                "shadow_failed_items": 0,
                "shadow_unavailable_items": 0,
            },
        )
        handoff = tmpdir / "runtime_handoff.json"
        self._write_json(
            handoff,
            {
                "schema": train_cache_miss.RUNTIME_HANDOFF_SCHEMA,
                "ok": True,
                "target": "cache_miss",
                "model": "rule",
                "runtime": {
                    "execution": str(runtime_execution),
                    "ready": True,
                    "attached_items": 4,
                },
                "counters": {
                    "samples": 4,
                    "lanes": 2,
                    "surrogate_replacements": 2,
                    "exact_fallbacks": 2,
                    "fail_closed": 0,
                    "shadow_sampled": 2,
                    "shadow_passed": 2,
                    "shadow_failed": 0,
                },
                "artifacts": {"runtime_execution": str(runtime_execution)},
            },
        )
        telemetry = tmpdir / "runtime_telemetry.json"
        self._write_json(
            telemetry,
            {
                "format_version": 1,
                "module_name": "cache0",
                "surrogate_execution": {
                    "ready": True,
                    "executed": False,
                    "plan_schema": "rrtl-surrogate-event-runtime-plan-v1",
                    "total_lanes": 2,
                    "attached_items": 4,
                    "surrogate_eligible_items": 2,
                    "exact_fallback_items": 2,
                    "invalid_items": 0,
                    "shadow_compared_items": 2,
                    "shadow_passed_items": 2,
                    "shadow_failed_items": 0,
                    "shadow_unavailable_items": 0,
                    "diagnostics": [],
                },
            },
        )
        return handoff, telemetry

    def _write_quality_gate_bundle(self, root: Path) -> Path:
        artifacts = {
            "training_summary": "training_summary.json",
            "validate": "validate.json",
            "shadow": "shadow.json",
            "fast_run": "fast_run.json",
            "model_fast_report": "model_fast_report.json",
            "runtime_execution": "runtime_execution.json",
        }
        self._write_json(root / "training_summary.json", self._training_summary())
        self._write_json(root / "validate.json", self._validation_report())
        self._write_json(root / "shadow.json", self._shadow_report())
        self._write_json(root / "fast_run.json", self._fast_run_report())
        self._write_json(root / "model_fast_report.json", {"ok": True})
        self._write_json(root / "runtime_execution.json", {"ready": True})
        bundle = root / "flow_bundle.json"
        self._write_json(
            bundle,
            {
                "schema": train_cache_miss.FLOW_BUNDLE_SCHEMA,
                "ok": True,
                "target": "cache_miss",
                "artifacts": artifacts,
            },
        )
        return bundle

    def _write_runtime_handoff_bundle(self, root: Path) -> Path:
        artifacts = {
            "manifest": "manifest.json",
            "runtime_plan": "runtime_plan.json",
            "runtime_attachment": "runtime_attachment.json",
            "runtime_execution": "runtime_execution.json",
            "quality_gate": "quality_gate.json",
            "model_fast_report": "model_fast_report.json",
        }
        self._write_json(
            root / "manifest.json",
            {
                "schema": train_cache_miss.MANIFEST_SCHEMA,
                "surrogate_id": "cache_miss_event_predictor",
                "surrogate_class": "event_predictor",
                "model_family": "rule-baseline",
                "artifact": {"sha256": "artifact-hash"},
                "source": {"source_hash": "source-hash"},
            },
        )
        self._write_json(root / "runtime_plan.json", {"schema": "rrtl-surrogate-event-runtime-plan-v1"})
        self._write_json(root / "runtime_attachment.json", {"schema": "rrtl-runtime-surrogate-attachment-v1"})
        self._write_json(root / "runtime_execution.json", {"ready": True, "attached_items": 2})
        self._write_json(root / "quality_gate.json", {"schema": train_cache_miss.QUALITY_GATE_SCHEMA, "ok": True})
        self._write_json(root / "model_fast_report.json", {"ok": True})
        bundle = root / "flow_bundle.json"
        self._write_json(
            bundle,
            {
                "schema": train_cache_miss.FLOW_BUNDLE_SCHEMA,
                "ok": True,
                "target": "cache_miss",
                "model": "rule",
                "samples": 2,
                "use_case": {"target": "cache_miss"},
                "readiness": {"quality_gate": True, "runtime_ready": True},
                "counters": {
                    "lanes": 2,
                    "surrogate_replacements": 0,
                    "exact_fallbacks": 2,
                    "fail_closed": 0,
                    "shadow_sampled": 1,
                    "shadow_passed": 1,
                    "shadow_failed": 0,
                },
                "artifacts": artifacts,
            },
        )
        return bundle

    def _training_summary(self, accuracy: float = 1.0):
        return {"schema": "rrtl-surrogate-event-training-summary-v1", "metrics": {"accuracy": accuracy}}

    def _validation_report(self, accuracy: float = 1.0):
        return {"schema": "rrtl-surrogate-event-validation-v1", "ok": True, "metrics": {"accuracy": accuracy}}

    def _shadow_report(self, accuracy: float = 1.0):
        return {"schema": "rrtl-surrogate-event-shadow-v1", "ok": True, "metrics": {"accuracy": accuracy}}

    def _fast_run_report(self, fail_closed: int = 0, shadow_failed: int = 0):
        return {
            "schema": train_cache_miss.EVENT_FAST_RUN_SCHEMA,
            "ok": True,
            "fail_closed": fail_closed,
            "shadow_failed": shadow_failed,
        }

    def _write_json(self, path: Path, value):
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _cache_miss_emitter_config(self):
        return {
            "schema": "rrtl-surrogate-event-emitter-config-v1",
            "target": "cache_miss",
            "window_cycles": 2,
            "horizon_cycles": 1,
            "signal_features": [{"name": name} for name in train_cache_miss.SIGNAL_FEATURES],
            "program_features": [{"name": name} for name in train_cache_miss.PROGRAM_FEATURES],
            "label": {"name": "cache_miss"},
        }

    def _instrumentation_inspection(self):
        return {
            "schema": "rrtl-instrumentation-trace-inspection-v1",
            "ok": True,
            "compatibility": {
                "target": "cache_miss",
                "window_cycles": 2,
                "horizon_cycles": 1,
                "emittable_samples": 2,
                "missing_fields": [],
            },
            "warnings": [],
            "errors": [],
        }

    def _event_fast_run_from_corpus(self, corpus):
        results = []
        for index, event in enumerate(corpus["events"]):
            predicted = event["label"]["cache_miss"]
            results.append(
                {
                    "index": index,
                    "sample_id": event["sample_id"],
                    "lane": event.get("lane", 0),
                    "target": event["target"],
                    "decision": "surrogate_used",
                    "source_result": "surrogate",
                    "predicted": predicted,
                    "expected": event["label"]["cache_miss"],
                    "provenance": {
                        "tag": "instrumentation_prediction",
                        "exact": False,
                        "surrogate_id": "cache_miss_event_predictor",
                        "surrogate_class": "event_predictor",
                    },
                    "shadow_sampled": index == 0,
                }
            )
        return {
            "schema": train_cache_miss.EVENT_FAST_RUN_SCHEMA,
            "ok": True,
            "count": len(results),
            "total_lanes": 1,
            "surrogate_replacements": len(results),
            "exact_fallbacks": 0,
            "fail_closed": 0,
            "shadow_compared": 0,
            "shadow_passed": 0,
            "shadow_failed": 0,
            "workers": [],
            "lanes": [],
            "results": results,
            "errors": [],
        }


if __name__ == "__main__":
    unittest.main()
