"""Report-only compatibility gate for private PyRTL corpora."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from . import corpus
from .discover import discover_targets, render_discovery_report, write_manifest


def run_gate(
    roots: list[str],
    *,
    package_roots: list[str] | None,
    out_dir: Path,
    manifest_path: Path | None = None,
    pyrtl2rrtl: list[str] | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Run a report-only PyRTL compatibility gate and return its summary."""

    if not roots:
        raise ValueError("at least one root is required")
    for root in roots:
        if not Path(root).exists():
            raise FileNotFoundError(f"root does not exist: {root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    package_roots = package_roots or roots
    pyrtl2rrtl = pyrtl2rrtl or ["cargo", "run", "-q", "-p", "rrtl-pyrtl", "--bin", "pyrtl2rrtl", "--"]
    _check_command_available(pyrtl2rrtl)

    outputs: dict[str, str] = {}
    candidates: list[dict[str, Any]] | None = None
    chosen_manifest = manifest_path

    if chosen_manifest is None:
        chosen_manifest = out_dir / "discovered.json"
        discovery_report_path = out_dir / "discovered.md"
        candidates = discover_targets(roots, package_roots=package_roots)
        write_manifest(candidates, chosen_manifest)
        discovery_report_path.write_text(render_discovery_report(candidates), encoding="utf-8")
        outputs["manifest"] = str(chosen_manifest)
        outputs["discovery_report"] = str(discovery_report_path)
    else:
        outputs["manifest"] = str(chosen_manifest)

    manifest = _read_json(chosen_manifest)

    validation_json_path = out_dir / "validation.json"
    validation_md_path = out_dir / "validation.md"
    with _temporary_sys_path(package_roots):
        findings = corpus.validate_manifest(manifest)
    validation = {
        "ok": not findings,
        "total_findings": len(findings),
        "findings": findings,
    }
    validation_json_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation_md_path.write_text(corpus.render_validation_report(findings), encoding="utf-8")
    outputs["validation_json"] = str(validation_json_path)
    outputs["validation_md"] = str(validation_md_path)

    if findings:
        gate = _build_gate_report(
            status="validation_failed",
            outputs=outputs,
            candidates=candidates,
            validation=validation,
            corpus_summary=None,
        )
        _write_gate_outputs(gate, out_dir)
        return gate

    summary_json_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"
    with _temporary_sys_path(package_roots):
        results = corpus.run_manifest(
            manifest,
            out_dir=out_dir,
            pyrtl2rrtl=pyrtl2rrtl,
            emit_sv=True,
            emit_json=True,
            fail_fast=fail_fast,
        )
    corpus_summary = corpus.build_summary(results)
    summary_json_path.write_text(
        json.dumps(corpus_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_md_path.write_text(corpus.render_summary_markdown(corpus_summary), encoding="utf-8")
    outputs["summary_json"] = str(summary_json_path)
    outputs["summary_md"] = str(summary_md_path)

    gate = _build_gate_report(
        status="completed",
        outputs=outputs,
        candidates=candidates,
        validation=validation,
        corpus_summary=corpus_summary,
    )
    _write_gate_outputs(gate, out_dir)
    return gate


def render_gate_markdown(gate: dict[str, Any]) -> str:
    """Render a human-readable gate report."""

    totals = gate.get("totals", {})
    phase_counts = gate.get("phase_counts", {})
    failure_buckets = gate.get("failure_buckets", {})
    failures = gate.get("actionable_failures", [])
    validation = gate.get("validation", {})
    outputs = gate.get("outputs", {})

    lines = [
        "# PyRTL Private Corpus Gate",
        "",
        f"- Status: {gate.get('status', 'unknown')}",
        f"- Total targets: {totals.get('total', 0)}",
        f"- Passed: {totals.get('passed', 0)}",
        f"- Failed: {totals.get('failed', 0)}",
        f"- Pass rate: {totals.get('pass_rate_percent', 0):.1f}%",
        f"- Validation findings: {validation.get('total_findings', 0)}",
        "",
        "## Artifacts",
        "",
    ]
    for label, path in sorted(outputs.items()):
        lines.append(f"- {label}: `{path}`")

    lines.extend(["", "## Phases", ""])
    if phase_counts:
        lines.extend(["| Phase | Count |", "| --- | ---: |"])
        for phase, count in sorted(phase_counts.items()):
            if count:
                lines.append(f"| {_md(phase)} | {count} |")
    else:
        lines.append("No corpus targets were run.")

    lines.extend(["", "## Failure Buckets", ""])
    if failure_buckets:
        lines.extend(["| Bucket | Count |", "| --- | ---: |"])
        for bucket, count in sorted(failure_buckets.items()):
            lines.append(f"| {_md(bucket)} | {count} |")
    else:
        lines.append("No target failures.")

    lines.extend(["", "## First Actionable Failures", ""])
    if failures:
        lines.extend(["| Name | Phase | Bucket | Error |", "| --- | --- | --- | --- |"])
        for item in failures:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md(item.get("name")),
                        _md(item.get("phase")),
                        _md(item.get("bucket")),
                        _md(item.get("error")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No target failures to triage.")

    return "\n".join(lines) + "\n"


def _build_gate_report(
    *,
    status: str,
    outputs: dict[str, str],
    candidates: list[dict[str, Any]] | None,
    validation: dict[str, Any],
    corpus_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    meta = (corpus_summary or {}).get("summary", {})
    total = int(meta.get("total", 0))
    passed = int(meta.get("passed", 0))
    failed = int(meta.get("failed", 0))
    pass_rate = (passed / total * 100.0) if total else 0.0
    failures = [
        {
            "name": result.get("name"),
            "target": result.get("target"),
            "phase": result.get("phase"),
            "bucket": result.get("bucket") or corpus.classify_failure(result),
            "error": result.get("error"),
            "outputs": result.get("outputs", {}),
        }
        for result in (corpus_summary or {}).get("results", [])
        if not result.get("ok")
    ][:10]

    return {
        "schema": "rrtl-pyrtl-gate-v1",
        "status": status,
        "ok": status == "completed",
        "report_only": True,
        "discovered_candidates": len(candidates) if candidates is not None else None,
        "validation": validation,
        "totals": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate_percent": round(pass_rate, 1),
        },
        "phase_counts": meta.get("phase_counts", {}),
        "failure_buckets": meta.get("failure_buckets", {}),
        "actionable_failures": failures,
        "outputs": outputs,
    }


def _write_gate_outputs(gate: dict[str, Any], out_dir: Path) -> None:
    gate_json = out_dir / "gate.json"
    gate_md = out_dir / "gate.md"
    gate["outputs"]["gate_json"] = str(gate_json)
    gate["outputs"]["gate_md"] = str(gate_md)
    gate_json.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gate_md.write_text(render_gate_markdown(gate), encoding="utf-8")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"failed to parse JSON `{path}`: {err}") from err


def _check_command_available(cmd: list[str]) -> None:
    if not cmd:
        raise ValueError("pyrtl2rrtl command must not be empty")
    executable = cmd[0]
    if "/" in executable or "\\" in executable:
        if not Path(executable).exists():
            raise FileNotFoundError(f"pyrtl2rrtl executable does not exist: {executable}")
        return
    if shutil.which(executable) is None:
        raise FileNotFoundError(f"pyrtl2rrtl executable not found on PATH: {executable}")


@contextmanager
def _temporary_sys_path(paths: Iterable[str]):
    original = list(sys.path)
    for path in reversed([str(Path(item)) for item in paths]):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = original


def _md(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a report-only private PyRTL corpus gate")
    parser.add_argument("root", nargs="+", help="Python files or directories to scan")
    parser.add_argument(
        "--package-root",
        action="append",
        help="root used to derive and import module paths; defaults to each scan root",
    )
    parser.add_argument("--manifest", help="reviewed corpus manifest; skips discovery when provided")
    parser.add_argument("--out-dir", default="target/rrtl-pyrtl-corpus/gate")
    parser.add_argument(
        "--pyrtl2rrtl",
        default="cargo run -q -p rrtl-pyrtl --bin pyrtl2rrtl --",
        help="command used to invoke the Rust bridge CLI",
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)

    try:
        gate = run_gate(
            args.root,
            package_roots=args.package_root,
            out_dir=Path(args.out_dir),
            manifest_path=Path(args.manifest) if args.manifest else None,
            pyrtl2rrtl=shlex.split(args.pyrtl2rrtl),
            fail_fast=args.fail_fast,
        )
    except Exception as err:  # noqa: BLE001 - CLI should report tooling errors cleanly.
        print(f"rrtl_pyrtl.gate: {err}", file=sys.stderr)
        return 1

    outputs = gate.get("outputs", {})
    for label in ("gate_json", "gate_md", "summary_json", "summary_md", "validation_json", "validation_md"):
        if label in outputs:
            print(f"{label}: {outputs[label]}")
    totals = gate.get("totals", {})
    print(
        f"{totals.get('passed', 0)}/{totals.get('total', 0)} targets passed "
        f"({totals.get('pass_rate_percent', 0):.1f}%)"
    )
    return 0 if gate.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
