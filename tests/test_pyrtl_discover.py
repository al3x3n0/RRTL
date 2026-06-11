import json
import tempfile
import unittest
from pathlib import Path

from rrtl_pyrtl import discover


class DiscoverTests(unittest.TestCase):
    def test_derives_module_path_from_package_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "src"
            design = root / "pkg" / "designs.py"
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

            candidates = discover.discover_targets([root], package_roots=[root])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["target"], "pkg.designs:build_counter")
        self.assertEqual(candidates[0]["top_name"], "BuildCounter")

    def test_detects_import_pyrtl_and_from_import_usage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.py").write_text(
                """
import pyrtl as pr

def make_a():
    pr.Output(1, "out")
""",
                encoding="utf-8",
            )
            (root / "b.py").write_text(
                """
from pyrtl import Input, working_block

def make_b():
    Input(1, "in")
    return working_block()
""",
                encoding="utf-8",
            )

            targets = {item["target"] for item in discover.discover_targets([root], [root])}

        self.assertEqual(targets, {"a:make_a", "b:make_b"})

    def test_ignores_functions_without_pyrtl_usage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "plain.py").write_text(
                """
import pyrtl

def build_number():
    return 1
""",
                encoding="utf-8",
            )

            candidates = discover.discover_targets([root], [root])

        self.assertEqual(candidates, [])

    def test_skips_ignored_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skipped = root / "target" / "generated.py"
            skipped.parent.mkdir()
            skipped.write_text(
                """
import pyrtl

def build_generated():
    pyrtl.Input(1, "x")
""",
                encoding="utf-8",
            )

            candidates = discover.discover_targets([root], [root])

        self.assertEqual(candidates, [])

    def test_write_manifest_outputs_valid_corpus_json(self):
        candidate = {
            "name": "pkg_design_build",
            "target": "pkg.design:build",
            "top_name": "Build",
            "clock_name": "clk",
            "reset_working_block": True,
            "confidence": 90,
            "source_path": "/tmp/design.py",
            "line": 4,
            "reasons": ["uses Input"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "manifest.json"
            discover.write_manifest([candidate], out)
            data = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(
            data,
            {
                "targets": [
                    {
                        "name": "pkg_design_build",
                        "target": "pkg.design:build",
                        "top_name": "Build",
                        "clock_name": "clk",
                        "reset_working_block": True,
                    }
                ]
            },
        )

    def test_render_discovery_report_includes_candidate_details(self):
        text = discover.render_discovery_report(
            [
                {
                    "name": "pkg_design_build",
                    "target": "pkg.design:build",
                    "top_name": "Build",
                    "clock_name": "clk",
                    "reset_working_block": True,
                    "confidence": 80,
                    "source_path": "pkg/design.py",
                    "line": 10,
                    "reasons": ["uses Input", "builder-like name"],
                }
            ]
        )

        self.assertIn("# PyRTL Corpus Discovery", text)
        self.assertIn("- High confidence: 1", text)
        self.assertIn("- Medium confidence: 0", text)
        self.assertIn("- Low confidence: 0", text)
        self.assertIn("## Next Steps", text)
        self.assertIn("| pkg_design_build | pkg.design:build | 80 | pkg/design.py:10 |", text)
        self.assertIn("uses Input", text)

    def test_cli_writes_manifest_and_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "src"
            design = root / "pkg" / "top.py"
            design.parent.mkdir(parents=True)
            design.write_text(
                """
import pyrtl

def elaborate_top():
    pyrtl.Register(4, "r")
    return pyrtl.working_block()
""",
                encoding="utf-8",
            )
            out = Path(tmpdir) / "discovered.json"
            report = Path(tmpdir) / "discovered.md"

            code = discover.main(
                [
                    str(root),
                    "--package-root",
                    str(root),
                    "--out",
                    str(out),
                    "--report",
                    str(report),
                ]
            )

            data = json.loads(out.read_text(encoding="utf-8"))
            report_text = report.read_text(encoding="utf-8")

        self.assertEqual(code, 0)
        self.assertEqual(data["targets"][0]["target"], "pkg.top:elaborate_top")
        self.assertIn("pkg.top:elaborate_top", report_text)


if __name__ == "__main__":
    unittest.main()
