import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import rrtl_surrogate.train_gemm as train_gemm


class TestSurrogateTrainGemm(unittest.TestCase):
    def test_generate_corpus_is_deterministic(self):
        lhs = train_gemm.generate_corpus(rows=2, cols=2, k=2, samples=3, seed=7)
        rhs = train_gemm.generate_corpus(rows=2, cols=2, k=2, samples=3, seed=7)

        self.assertEqual(lhs, rhs)
        self.assertEqual(len(lhs), 3)
        self.assertEqual(lhs[0]["schema"], "rrtl-surrogate-gemm-transaction-v1")
        self.assertNotIn("lane", lhs[0])

    def test_generate_corpus_assigns_lanes(self):
        corpus = train_gemm.generate_corpus(
            rows=2,
            cols=2,
            k=2,
            samples=5,
            seed=7,
            lanes=2,
        )

        self.assertEqual([item["lane"] for item in corpus], [0, 1, 0, 1, 0])

    def test_transaction_expected_outputs(self):
        shape = train_gemm.GemmShape(rows=2, cols=2, k=2)
        tx = train_gemm.build_transaction(
            shape,
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        )

        self.assertEqual(tx["expected_c"], [[19, 22], [43, 50]])
        self.assertEqual(tx["expected_latency_cycles"], 5)

    def test_manifest_hash_and_tensor_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.onnx"
            model.write_bytes(b"fake-onnx")
            digest = hashlib.sha256(b"fake-onnx").hexdigest()
            manifest = train_gemm.build_manifest(
                train_gemm.GemmShape(rows=2, cols=2, k=2),
                model_path=model,
                model_hash=digest,
                source_hash="source",
                tolerance=0,
            )

        self.assertEqual(manifest["artifact"]["sha256"], digest)
        self.assertEqual(
            manifest["artifact"]["input_tensors"],
            ["gemm_descriptor", "a_tensor", "w_tensor"],
        )
        self.assertEqual(
            manifest["artifact"]["output_tensors"],
            ["c_tensor", "telemetry"],
        )

    def test_infer_shape_rejects_mixed_shapes(self):
        corpus = train_gemm.generate_corpus(rows=2, cols=2, k=2, samples=2, seed=1)
        corpus[1]["k"] = 3

        with self.assertRaisesRegex(ValueError, "shape does not match"):
            train_gemm.infer_shape(corpus)

    def test_cli_generate_writes_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "corpus.json"
            code = train_gemm.main(
                [
                    "generate",
                    "--out",
                    str(out),
                    "--rows",
                    "2",
                    "--cols",
                    "2",
                    "--k",
                    "2",
                    "--samples",
                    "2",
                    "--seed",
                    "9",
                ]
            )
            data = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(len(data), 2)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("onnx"),
        "torch and onnx are required for ONNX export",
    )
    def test_train_and_export_writes_manifest_and_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = Path(tmpdir) / "corpus.json"
            out_dir = Path(tmpdir) / "train"
            corpus = train_gemm.generate_corpus(
                rows=2, cols=2, k=2, samples=2, seed=3
            )
            corpus_path.write_text(
                json.dumps(corpus, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            summary = train_gemm.train_and_export(
                corpus_path=corpus_path,
                out_dir=out_dir,
                epochs=0,
                seed=3,
            )
            manifest = json.loads(
                (out_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(Path(summary["outputs"]["model"]).exists())

        self.assertEqual(summary["samples"], 2)
        self.assertEqual(manifest["artifact"]["format"], "onnx")
        self.assertEqual(manifest["validation"]["max_abs_error"], 0)

    def test_cli_generate_writes_lane_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "corpus.json"
            code = train_gemm.main(
                [
                    "generate",
                    "--out",
                    str(out),
                    "--rows",
                    "2",
                    "--cols",
                    "2",
                    "--k",
                    "2",
                    "--samples",
                    "3",
                    "--seed",
                    "9",
                    "--lanes",
                    "2",
                ]
            )
            data = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual([item["lane"] for item in data], [0, 1, 0])

    def test_validate_with_rrtl_uses_batch_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.json"
            transactions = Path(tmpdir) / "transactions.json"
            out = Path(tmpdir) / "report.json"
            fake = Path(tmpdir) / "fake_pyrtl2rrtl.py"
            manifest.write_text("{}", encoding="utf-8")
            corpus = train_gemm.generate_corpus(
                rows=2,
                cols=2,
                k=2,
                samples=2,
                seed=5,
                lanes=2,
            )
            transactions.write_text(
                json.dumps(corpus, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            fake.write_text(
                "\n".join(
                    [
                        "import json",
                        "import sys",
                        "batch = json.load(open(sys.argv[-1], encoding='utf-8'))",
                        "lanes = sorted({item.get('lane', i) for i, item in enumerate(batch['transactions'])})",
                        "print(json.dumps({'schema': 'rrtl-surrogate-gemm-batch-result-v1', 'ok': True, 'count': len(batch['transactions']), 'total_lanes': len(lanes), 'metrics': {'max_abs_error': 0, 'max_mean_abs_error': 0.0, 'max_latency_error_cycles': 0}, 'lanes': [{'lane': lane, 'count': 1, 'ok': 1, 'failed': 0, 'metrics': {'max_abs_error': 0, 'max_mean_abs_error': 0.0, 'max_latency_error_cycles': 0}} for lane in lanes], 'results': [{'index': i, 'lane': item.get('lane', i), 'ok': True} for i, item in enumerate(batch['transactions'])]}))",
                    ]
                ),
                encoding="utf-8",
            )

            report = train_gemm.validate_with_rrtl(
                manifest_path=manifest,
                transactions_path=transactions,
                pyrtl2rrtl=["python", str(fake)],
                out_path=out,
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["schema"], "rrtl-surrogate-gemm-batch-result-v1")
        self.assertEqual(report["total_lanes"], 2)
        self.assertEqual(report["results"][1]["lane"], 1)

    def test_policy_with_rrtl_reads_decision_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.json"
            transactions = Path(tmpdir) / "transactions.json"
            out = Path(tmpdir) / "policy.json"
            fake = Path(tmpdir) / "fake_policy.py"
            manifest.write_text("{}", encoding="utf-8")
            corpus = train_gemm.generate_corpus(
                rows=2,
                cols=2,
                k=2,
                samples=2,
                seed=5,
                lanes=2,
            )
            transactions.write_text(
                json.dumps(corpus, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            fake.write_text(
                "\n".join(
                    [
                        "import json",
                        "import sys",
                        "batch = json.load(open(sys.argv[-1], encoding='utf-8'))",
                        "print(json.dumps({'schema': 'rrtl-surrogate-gemm-policy-report-v1', 'ok': True, 'count': len(batch['transactions']), 'used_surrogate': 1, 'exact_fallbacks': 1, 'fail_closed': 0, 'results': [{'index': 0, 'lane': batch['transactions'][0].get('lane', 0), 'decision': 'surrogate_used', 'ok': True}, {'index': 1, 'lane': batch['transactions'][1].get('lane', 1), 'decision': 'exact_fallback', 'ok': True}]}))",
                    ]
                ),
                encoding="utf-8",
            )

            report = train_gemm.policy_with_rrtl(
                manifest_path=manifest,
                transactions_path=transactions,
                pyrtl2rrtl=["python", str(fake)],
                out_path=out,
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["schema"], "rrtl-surrogate-gemm-policy-report-v1")
        self.assertEqual(report["used_surrogate"], 1)
        self.assertEqual(report["exact_fallbacks"], 1)
        self.assertEqual(report["results"][1]["decision"], "exact_fallback")
        self.assertEqual(report["results"][1]["lane"], 1)


if __name__ == "__main__":
    unittest.main()
