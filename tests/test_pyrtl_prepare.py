import json
import sys
import tempfile
import unittest
from pathlib import Path

from rrtl_pyrtl import prepare_corpus


class PrepareCorpusTests(unittest.TestCase):
    def test_prepare_success_writes_discovery_and_validation_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "src"
            out_dir = Path(tmpdir) / "out"
            design = root / "prep_success_pkg" / "design.py"
            design.parent.mkdir(parents=True)
            design.write_text(
                """
import pyrtl

def build_counter():
    pyrtl.Input(1, "en")
    return pyrtl.working_block()
""",
                encoding="utf-8",
            )

            sys.path.insert(0, str(root))
            try:
                result = prepare_corpus.prepare_corpus(
                    [str(root)],
                    package_roots=[str(root)],
                    out_dir=out_dir,
                )
            finally:
                sys.path.remove(str(root))

            manifest = json.loads((out_dir / "discovered.json").read_text(encoding="utf-8"))
            validation = json.loads((out_dir / "validation.json").read_text(encoding="utf-8"))
            discovery_report = (out_dir / "discovered.md").read_text(encoding="utf-8")
            validation_report = (out_dir / "validation.md").read_text(encoding="utf-8")
            next_commands = (out_dir / "next_commands.md").read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertIn("next_commands", result["outputs"])
        self.assertEqual(manifest["targets"][0]["target"], "prep_success_pkg.design:build_counter")
        self.assertEqual(validation, {"findings": [], "ok": True, "total_findings": 0})
        self.assertIn("prep_success_pkg.design:build_counter", discovery_report)
        self.assertIn("Manifest validation passed.", validation_report)
        self.assertIn("rrtl_pyrtl.corpus", next_commands)
        self.assertIn(str(out_dir / "discovered.json"), next_commands)

    def test_prepare_validation_failure_exits_one_and_writes_findings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "src"
            out_dir = Path(tmpdir) / "out"
            design = root / "prep_fail_pkg" / "design.py"
            design.parent.mkdir(parents=True)
            design.write_text(
                """
import pyrtl

def build_counter():
    pyrtl.Input(1, "en")
    return pyrtl.working_block()
""",
                encoding="utf-8",
            )

            code = prepare_corpus.main(
                [
                    str(root),
                    "--package-root",
                    str(root),
                    "--out-dir",
                    str(out_dir),
                ]
            )
            validation = json.loads((out_dir / "validation.json").read_text(encoding="utf-8"))
            validation_report = (out_dir / "validation.md").read_text(encoding="utf-8")

        self.assertEqual(code, 1)
        self.assertFalse(validation["ok"])
        self.assertEqual(validation["total_findings"], 1)
        self.assertIn("target import failed", validation["findings"][0]["error"])
        self.assertIn("target import failed", validation_report)

    def test_prepare_respects_custom_output_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "src"
            out_dir = Path(tmpdir) / "out"
            design = root / "prep_custom_pkg" / "design.py"
            design.parent.mkdir(parents=True)
            design.write_text(
                """
import pyrtl

def make_top():
    pyrtl.Output(1, "out")
""",
                encoding="utf-8",
            )

            sys.path.insert(0, str(root))
            try:
                code = prepare_corpus.main(
                    [
                        str(root),
                        "--package-root",
                        str(root),
                        "--out-dir",
                        str(out_dir),
                        "--manifest-name",
                        "manifest.json",
                        "--report-name",
                        "report.md",
                        "--validation-json",
                        "valid.json",
                        "--validation-md",
                        "valid.md",
                        "--next-commands-name",
                        "next.md",
                    ]
                )
                self.assertEqual(code, 0)
                self.assertTrue((out_dir / "manifest.json").exists())
                self.assertTrue((out_dir / "report.md").exists())
                self.assertTrue((out_dir / "valid.json").exists())
                self.assertTrue((out_dir / "valid.md").exists())
                self.assertTrue((out_dir / "next.md").exists())
                self.assertIn("manifest.json", (out_dir / "next.md").read_text(encoding="utf-8"))
            finally:
                sys.path.remove(str(root))

    def test_prepare_forwards_multiple_roots_and_package_roots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root_a = base / "src_a"
            root_b = base / "src_b"
            out_dir = base / "out"
            design_a = root_a / "pkg_a" / "a.py"
            design_b = root_b / "pkg_b" / "b.py"
            design_a.parent.mkdir(parents=True)
            design_b.parent.mkdir(parents=True)
            design_a.write_text(
                """
import pyrtl

def build_a():
    pyrtl.Input(1, "a")
""",
                encoding="utf-8",
            )
            design_b.write_text(
                """
import pyrtl

def build_b():
    pyrtl.Input(1, "b")
""",
                encoding="utf-8",
            )

            sys.path[:0] = [str(root_a), str(root_b)]
            try:
                result = prepare_corpus.prepare_corpus(
                    [str(root_a), str(root_b)],
                    package_roots=[str(root_a), str(root_b)],
                    out_dir=out_dir,
                )
            finally:
                sys.path.remove(str(root_a))
                sys.path.remove(str(root_b))

            targets = {
                target["target"]
                for target in json.loads((out_dir / "discovered.json").read_text(encoding="utf-8"))[
                    "targets"
                ]
            }

        self.assertTrue(result["ok"])
        self.assertEqual(targets, {"pkg_a.a:build_a", "pkg_b.b:build_b"})


if __name__ == "__main__":
    unittest.main()
