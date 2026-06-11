"""Prepare a PyRTL corpus manifest from private source trees.

This command performs static discovery and manifest validation. It does not
execute candidate design builders.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from .corpus import render_validation_report, validate_manifest
from .discover import discover_targets, render_discovery_report, write_manifest


def prepare_corpus(
    roots: list[str],
    *,
    package_roots: list[str] | None,
    out_dir: Path,
    manifest_name: str = "discovered.json",
    report_name: str = "discovered.md",
    validation_json: str = "validation.json",
    validation_md: str = "validation.md",
    next_commands_name: str = "next_commands.md",
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / manifest_name
    report_path = out_dir / report_name
    validation_json_path = out_dir / validation_json
    validation_md_path = out_dir / validation_md
    next_commands_path = out_dir / next_commands_name

    candidates = discover_targets(roots, package_roots=package_roots)
    write_manifest(candidates, manifest_path)
    report_path.write_text(render_discovery_report(candidates), encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    findings = validate_manifest(manifest)
    validation = {
        "ok": not findings,
        "total_findings": len(findings),
        "findings": findings,
    }
    validation_json_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation_md_path.write_text(render_validation_report(findings), encoding="utf-8")
    next_commands_path.write_text(
        render_next_commands(
            roots,
            package_roots=package_roots,
            out_dir=out_dir,
            manifest_path=manifest_path,
            report_path=report_path,
            validation_json_path=validation_json_path,
            validation_md_path=validation_md_path,
            next_commands_name=next_commands_name,
        ),
        encoding="utf-8",
    )

    return {
        "ok": not findings,
        "candidates": candidates,
        "findings": findings,
        "outputs": {
            "manifest": str(manifest_path),
            "report": str(report_path),
            "validation_json": str(validation_json_path),
            "validation_md": str(validation_md_path),
            "next_commands": str(next_commands_path),
        },
    }


def render_next_commands(
    roots: list[str],
    *,
    package_roots: list[str] | None,
    out_dir: Path,
    manifest_path: Path,
    report_path: Path,
    validation_json_path: Path,
    validation_md_path: Path,
    next_commands_name: str,
) -> str:
    prepare_parts = ["PYTHONPATH=python:${PYTHONPATH:-}", "python3", "-m", "rrtl_pyrtl.prepare_corpus"]
    prepare_parts.extend(roots)
    for package_root in package_roots or []:
        prepare_parts.extend(["--package-root", package_root])
    prepare_parts.extend(
        [
            "--out-dir",
            str(out_dir),
            "--next-commands-name",
            next_commands_name,
        ]
    )

    validate_parts = [
        "PYTHONPATH=python:${PYTHONPATH:-}",
        "python3",
        "-m",
        "rrtl_pyrtl.corpus",
        "--validate-only",
        str(manifest_path),
    ]
    triage_parts = [
        "PYTHONPATH=python:${PYTHONPATH:-}",
        "python3",
        "-m",
        "rrtl_pyrtl.corpus",
        str(manifest_path),
        "--emit-sv",
        "--emit-json",
        "--summary-json",
        str(out_dir / "summary.json"),
        "--summary-md",
        str(out_dir / "summary.md"),
    ]

    return "\n".join(
        [
            "# PyRTL Corpus Next Commands",
            "",
            "Generated artifacts:",
            "",
            f"- Manifest: `{manifest_path}`",
            f"- Discovery report: `{report_path}`",
            f"- Validation JSON: `{validation_json_path}`",
            f"- Validation report: `{validation_md_path}`",
            "",
            "Re-run prepare:",
            "",
            "```sh",
            _shell_join(prepare_parts),
            "```",
            "",
            "Validate generated manifest:",
            "",
            "```sh",
            _shell_join(validate_parts),
            "```",
            "",
            "Run full corpus triage:",
            "",
            "```sh",
            _shell_join(triage_parts),
            "```",
            "",
        ]
    )


def _shell_join(parts: list[str]) -> str:
    return " ".join(part if part.startswith("PYTHONPATH=") else shlex.quote(part) for part in parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover and validate a PyRTL corpus manifest")
    parser.add_argument("roots", nargs="+", help="Python files or directories to scan")
    parser.add_argument(
        "--package-root",
        action="append",
        help="root used to derive module paths; defaults to each scan root",
    )
    parser.add_argument("--out-dir", required=True, help="directory for generated corpus artifacts")
    parser.add_argument("--manifest-name", default="discovered.json")
    parser.add_argument("--report-name", default="discovered.md")
    parser.add_argument("--validation-json", default="validation.json")
    parser.add_argument("--validation-md", default="validation.md")
    parser.add_argument("--next-commands-name", default="next_commands.md")
    args = parser.parse_args(argv)

    result = prepare_corpus(
        args.roots,
        package_roots=args.package_root,
        out_dir=Path(args.out_dir),
        manifest_name=args.manifest_name,
        report_name=args.report_name,
        validation_json=args.validation_json,
        validation_md=args.validation_md,
        next_commands_name=args.next_commands_name,
    )

    print(f"discovered {len(result['candidates'])} candidate(s)")
    for label, path in result["outputs"].items():
        print(f"{label}: {path}")
    if result["ok"]:
        print("manifest validation passed")
        return 0

    print(f"{len(result['findings'])} manifest validation error(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
