import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rrtl_pyrtl import corpus, gate


class GateTests(unittest.TestCase):
    def test_gate_discovers_runs_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "src"
            out_dir = Path(tmpdir) / "out"
            design = root / "gate_ok_pkg" / "design.py"
            design.parent.mkdir(parents=True)
            design.write_text(
                """
import pyrtl

def build_counter():
    en = pyrtl.Input(1, "en")
    out = pyrtl.Output(1, "out")
    out <<= en
    return pyrtl.working_block()
""",
                encoding="utf-8",
            )

            with mock.patch.object(corpus, "_run_cmd", return_value=""):
                result = gate.run_gate(
                    [str(root)],
                    package_roots=[str(root)],
                    out_dir=out_dir,
                    pyrtl2rrtl=["python3", "-V"],
                )

            manifest = json.loads((out_dir / "discovered.json").read_text(encoding="utf-8"))
            gate_json = json.loads((out_dir / "gate.json").read_text(encoding="utf-8"))
            gate_md = (out_dir / "gate.md").read_text(encoding="utf-8")

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["ok"])
        self.assertEqual(result["totals"]["passed"], 1)
        self.assertEqual(manifest["targets"][0]["target"], "gate_ok_pkg.design:build_counter")
        self.assertEqual(gate_json["schema"], "rrtl-pyrtl-gate-v1")
        self.assertIn("summary_json", gate_json["outputs"])
        self.assertIn("# PyRTL Private Corpus Gate", gate_md)

    def test_gate_stops_before_corpus_on_validation_failure(self):
        manifest = {
            "targets": [
                {
                    "name": "missing",
                    "target": "no_such_module:build",
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "src"
            out_dir = Path(tmpdir) / "out"
            root.mkdir()
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = gate.run_gate(
                [str(root)],
                package_roots=[str(root)],
                out_dir=out_dir,
                manifest_path=manifest_path,
                pyrtl2rrtl=["python3", "-V"],
            )

        self.assertEqual(result["status"], "validation_failed")
        self.assertFalse(result["ok"])
        self.assertEqual(result["validation"]["total_findings"], 1)

    def test_gate_reports_import_failure_from_build_as_target_failure(self):
        manifest = {
            "targets": [
                {
                    "name": "bad_build",
                    "target": "gate_bad_build_pkg.design:build",
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "src"
            out_dir = Path(tmpdir) / "out"
            design = root / "gate_bad_build_pkg" / "design.py"
            design.parent.mkdir(parents=True)
            design.write_text(
                """
def build():
    raise RuntimeError("builder exploded")
""",
                encoding="utf-8",
            )
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = gate.run_gate(
                [str(root)],
                package_roots=[str(root)],
                out_dir=out_dir,
                manifest_path=manifest_path,
                pyrtl2rrtl=["python3", "-V"],
            )

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["ok"])
        self.assertEqual(result["totals"]["failed"], 1)
        self.assertEqual(result["failure_buckets"]["build"], 1)
        self.assertEqual(result["actionable_failures"][0]["name"], "bad_build")

    def test_gate_uses_provided_manifest_without_discovery(self):
        manifest = {"targets": [{"name": "ok", "target": "rrtl_pyrtl.examples:counter"}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "src"
            out_dir = Path(tmpdir) / "out"
            root.mkdir()
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with mock.patch.object(corpus, "_run_cmd", return_value=""):
                result = gate.run_gate(
                    [str(root)],
                    package_roots=[str(root)],
                    out_dir=out_dir,
                    manifest_path=manifest_path,
                    pyrtl2rrtl=["python3", "-V"],
                )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["outputs"]["manifest"], str(manifest_path))
        self.assertNotIn("discovery_report", result["outputs"])
        self.assertFalse((out_dir / "discovered.json").exists())

    def test_gate_main_returns_zero_for_target_failures(self):
        manifest = {
            "targets": [
                {
                    "name": "bad_build",
                    "target": "gate_main_bad_pkg.design:build",
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "src"
            out_dir = Path(tmpdir) / "out"
            design = root / "gate_main_bad_pkg" / "design.py"
            design.parent.mkdir(parents=True)
            design.write_text(
                """
def build():
    raise RuntimeError("builder exploded")
""",
                encoding="utf-8",
            )
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            code = gate.main(
                [
                    str(root),
                    "--package-root",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--out-dir",
                    str(out_dir),
                    "--pyrtl2rrtl",
                    "python3 -V",
                ]
            )

        self.assertEqual(code, 0)

    def test_gate_main_returns_one_for_validation_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "src"
            out_dir = Path(tmpdir) / "out"
            root.mkdir()
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(json.dumps({"targets": ["not an object"]}), encoding="utf-8")

            code = gate.main(
                [
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--out-dir",
                    str(out_dir),
                    "--pyrtl2rrtl",
                    "python3 -V",
                ]
            )

            self.assertEqual(code, 1)
            self.assertTrue((out_dir / "gate.json").exists())

    def test_gate_rejects_missing_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing"
            code = gate.main([str(missing), "--pyrtl2rrtl", "python3 -V"])

        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
