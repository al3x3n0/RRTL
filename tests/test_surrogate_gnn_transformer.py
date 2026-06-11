import json
import tempfile
import unittest
from pathlib import Path

import rrtl_surrogate.train_cache_miss as train_cache_miss
from rrtl_surrogate.gnn_transformer import (
    EVENT_TRAINER_SCHEMA,
    build_event_trainer_manifest,
    build_tensor_manifest,
    main,
)


class TestSurrogateGnnTransformer(unittest.TestCase):
    def test_build_tensor_manifest(self):
        manifest = build_tensor_manifest(
            {
                "schema": "rrtl-surrogate-dataset-v1",
                "source_hash": "abc",
                "top_name": "Top",
                "graph": {
                    "node_features": [
                        {"name": "a", "kind": "input", "bitwidth": 8},
                        {"name": "out", "kind": "output", "bitwidth": 8},
                    ],
                    "edge_index": [[0, 1]],
                    "edge_features": [{"net_index": 0, "op": "w", "role": "arg-to-dest"}],
                    "metadata": {},
                },
                "trace": {
                    "schema": "rrtl-pyrtl-trace-v1",
                    "steps": [
                        {"inputs": {"a": 1}, "outputs": {"out": 1}},
                        {"inputs": {"a": 2}, "outputs": {"out": 2}},
                    ],
                },
            }
        )

        self.assertEqual(
            manifest["schema"], "rrtl-surrogate-gnn-transformer-tensors-v1"
        )
        self.assertEqual(manifest["graph_inputs"]["node_features"]["shape"], [2, 4])
        self.assertEqual(manifest["graph_inputs"]["edge_index"]["shape"], [2, 1])
        self.assertEqual(manifest["sequence_inputs"]["input_trace"]["signals"], ["a"])
        self.assertEqual(manifest["outputs"]["c_tensor"]["shape"], ["rows", "cols"])

    def test_build_event_trainer_manifest_from_tensor_bundle(self):
        corpus = train_cache_miss.generate_corpus(samples=4, window=3, seed=7, lanes=2)
        bundle = train_cache_miss.build_tensor_bundle(corpus)

        manifest = build_event_trainer_manifest(bundle)

        self.assertEqual(manifest["schema"], EVENT_TRAINER_SCHEMA)
        self.assertEqual(manifest["source_hash"], bundle["source_hash"])
        self.assertEqual(manifest["top_name"], "InstrumentedCache")
        self.assertEqual(manifest["task"]["prediction_target"], "cache_miss")
        self.assertEqual(manifest["task"]["sample_count"], 4)
        self.assertEqual(
            manifest["inputs"]["signal_window"]["shape"],
            [4, 3, len(train_cache_miss.SIGNAL_FEATURES)],
        )
        self.assertEqual(
            manifest["inputs"]["signal_window"]["features"],
            train_cache_miss.SIGNAL_FEATURES,
        )
        self.assertEqual(
            manifest["inputs"]["program_context"]["features"],
            train_cache_miss.PROGRAM_FEATURES,
        )
        self.assertEqual(manifest["labels"]["predicted_event"]["shape"], [4, 1])
        self.assertEqual(manifest["labels"]["predicted_event"]["kind"], "binary")
        self.assertEqual(manifest["metadata"]["sample_ids"], [0, 1, 2, 3])
        self.assertEqual(manifest["metadata"]["lanes"], [0, 1, 0, 1])
        self.assertEqual(manifest["metadata"]["batching"], ["sample_ids", "lanes"])

    def test_event_trainer_manifest_rejects_missing_tensor(self):
        corpus = train_cache_miss.generate_corpus(samples=2, window=3, seed=7)
        bundle = train_cache_miss.build_tensor_bundle(corpus)
        del bundle["inputs"]["program_context"]

        with self.assertRaisesRegex(ValueError, "requires program_context"):
            build_event_trainer_manifest(bundle)

    def test_event_trainer_manifest_rejects_mismatched_sample_count(self):
        corpus = train_cache_miss.generate_corpus(samples=2, window=3, seed=7)
        bundle = train_cache_miss.build_tensor_bundle(corpus)
        bundle["labels"]["predicted_event"].append([0])

        with self.assertRaisesRegex(ValueError, "sample count 3 does not match"):
            build_event_trainer_manifest(bundle)

    def test_event_trainer_manifest_rejects_mismatched_feature_width(self):
        corpus = train_cache_miss.generate_corpus(samples=2, window=3, seed=7)
        bundle = train_cache_miss.build_tensor_bundle(corpus)
        bundle["inputs"]["signal_window"][0][0].append(99)

        with self.assertRaisesRegex(ValueError, "width 9 does not match manifest width 8"):
            build_event_trainer_manifest(bundle)

    def test_cli_event_bundle_writes_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "tensor_bundle.json"
            out = Path(tmpdir) / "trainer_manifest.json"
            corpus = train_cache_miss.generate_corpus(samples=2, window=3, seed=7)
            bundle = train_cache_miss.build_tensor_bundle(corpus)
            bundle_path.write_text(
                json.dumps(bundle, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            code = main(["event-bundle", str(bundle_path), "--out", str(out)])
            manifest = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["schema"], EVENT_TRAINER_SCHEMA)
        self.assertEqual(manifest["task"]["sample_count"], 2)

    def test_cli_legacy_dataset_mode_writes_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.json"
            out = Path(tmpdir) / "tensor_manifest.json"
            dataset_path.write_text(
                json.dumps(_graph_dataset(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            code = main([str(dataset_path), "--out", str(out)])
            manifest = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            manifest["schema"],
            "rrtl-surrogate-gnn-transformer-tensors-v1",
        )

def _graph_dataset():
    return {
        "schema": "rrtl-surrogate-dataset-v1",
        "source_hash": "abc",
        "top_name": "Top",
        "graph": {
            "node_features": [
                {"name": "a", "kind": "input", "bitwidth": 8},
                {"name": "out", "kind": "output", "bitwidth": 8},
            ],
            "edge_index": [[0, 1]],
            "edge_features": [{"net_index": 0, "op": "w", "role": "arg-to-dest"}],
            "metadata": {},
        },
        "trace": {
            "schema": "rrtl-pyrtl-trace-v1",
            "steps": [
                {"inputs": {"a": 1}, "outputs": {"out": 1}},
                {"inputs": {"a": 2}, "outputs": {"out": 2}},
            ],
        },
    }


if __name__ == "__main__":
    unittest.main()
