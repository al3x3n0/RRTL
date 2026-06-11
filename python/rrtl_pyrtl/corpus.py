"""Batch runner for hardening the PyRTL-to-RRTL bridge.

Manifest shape:

{
  "targets": [
    {
      "name": "counter",
      "target": "my_designs.counter:build",
      "top_name": "Counter",
      "clock_name": "clk",
      "inputs": [{"en": 1}, {"en": 0}]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import importlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyrtl

from . import bench
from .export import _load_target, export_block_json, simulate_lane_trace, simulate_trace


PHASES = (
    "manifest",
    "build",
    "export",
    "check",
    "sv",
    "json",
    "trace",
    "compare",
    "profile",
    "profile_replay",
    "done",
)


def run_manifest(
    manifest: dict[str, Any],
    *,
    out_dir: Path,
    pyrtl2rrtl: list[str],
    emit_sv: bool = False,
    emit_json: bool = False,
    fail_fast: bool = False,
    profile_replay_hot_repeat: int | None = None,
    planner_calibration: Path | None = None,
) -> list[dict[str, Any]]:
    if profile_replay_hot_repeat is not None and profile_replay_hot_repeat < 0:
        raise ValueError("profile_replay_hot_repeat must be non-negative")
    targets, error = _manifest_targets(manifest)
    if error is not None:
        raise SystemExit(error)
    assert targets is not None

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, spec in enumerate(targets):
        if not isinstance(spec, dict):
            result = _failure(
                f"target-{index}",
                None,
                "manifest",
                "manifest entry is not an object",
            )
            results.append(result)
            _print_result(result)
            if fail_fast:
                break
            continue
        result = _run_one(
            spec,
            out_dir=out_dir,
            pyrtl2rrtl=pyrtl2rrtl,
            emit_sv=emit_sv,
            emit_json=emit_json,
            profile_replay_hot_repeat=profile_replay_hot_repeat,
            planner_calibration=planner_calibration,
        )
        results.append(result)
        _print_result(result)
        if not result["ok"]:
            if fail_fast:
                break
    return results


def validate_manifest(manifest: Any) -> list[dict[str, Any]]:
    targets, error = _manifest_targets(manifest)
    if error is not None:
        return [_validation_failure(None, None, None, error)]
    assert targets is not None

    findings = []
    seen_names: dict[str, int] = {}
    for index, spec in enumerate(targets):
        if not isinstance(spec, dict):
            findings.append(
                _validation_failure(
                    index,
                    f"target-{index}",
                    None,
                    "manifest entry is not an object",
                )
            )
            continue

        name = _effective_name(spec, index)
        target = spec.get("target")

        if "name" in spec and not isinstance(spec["name"], str):
            findings.append(
                _validation_failure(index, name, target, "name must be a string when provided")
            )
        if name in seen_names:
            findings.append(
                _validation_failure(
                    index,
                    name,
                    target,
                    f"duplicate target name `{name}` also used at index {seen_names[name]}",
                )
            )
        else:
            seen_names[name] = index

        if not isinstance(target, str):
            findings.append(
                _validation_failure(index, name, None, "target must be a module:function string")
            )
        elif ":" not in target:
            findings.append(
                _validation_failure(index, name, target, "target must be in module:function form")
            )
        else:
            try:
                value = _resolve_target(target)
                if not callable(value):
                    findings.append(
                        _validation_failure(index, name, target, "target resolves but is not callable")
                    )
            except Exception as err:  # noqa: BLE001 - validation reports import/attribute failures.
                findings.append(
                    _validation_failure(index, name, target, f"target import failed: {err}")
                )

        for field in ("top_name", "clock_name"):
            if field in spec and not isinstance(spec[field], str):
                findings.append(
                    _validation_failure(index, name, target, f"{field} must be a string")
                )

        if "reset_working_block" in spec and not isinstance(spec["reset_working_block"], bool):
            findings.append(
                _validation_failure(index, name, target, "reset_working_block must be a boolean")
            )

        if "inputs" in spec:
            findings.extend(_validate_inputs(index, name, target, spec["inputs"]))
        if "benchmark_profile" in spec and not isinstance(spec["benchmark_profile"], bool):
            findings.append(
                _validation_failure(index, name, target, "benchmark_profile must be a boolean")
            )
        for field in ("packed_lanes", "repeat", "warmup", "profile_replay_hot_repeat"):
            if field in spec:
                value = spec[field]
                if not isinstance(value, int):
                    findings.append(
                        _validation_failure(index, name, target, f"{field} must be an integer")
                    )
                elif field not in {"warmup", "profile_replay_hot_repeat"} and value <= 0:
                    findings.append(
                        _validation_failure(index, name, target, f"{field} must be greater than zero")
                    )
                elif field in {"warmup", "profile_replay_hot_repeat"} and value < 0:
                    findings.append(
                        _validation_failure(index, name, target, f"{field} must be non-negative")
                    )

    return findings


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    phase_counts = {phase: 0 for phase in PHASES}
    bucket_counts: dict[str, int] = {}
    recommended_backends: dict[str, int] = {}
    planner_calibrations: dict[str, int] = {}
    profiled = 0
    failed = 0
    for result in results:
        phase = str(result.get("phase", "unknown"))
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        profile = (result.get("outputs") or {}).get("runtime_profile")
        runtime_profile = (
            result.get("runtime_profile") if isinstance(result.get("runtime_profile"), dict) else None
        )
        if profile or runtime_profile:
            profiled += 1
        if runtime_profile:
            backend = runtime_profile.get("recommended_runtime_backend")
            if backend:
                recommended_backends[str(backend)] = recommended_backends.get(str(backend), 0) + 1
        calibration_path = (result.get("outputs") or {}).get("planner_calibration")
        if calibration_path:
            planner_calibrations[str(calibration_path)] = (
                planner_calibrations.get(str(calibration_path), 0) + 1
            )
        if not result.get("ok"):
            failed += 1
            bucket = str(result.get("bucket") or classify_failure(result))
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    return {
        "summary": {
            "total": len(results),
            "passed": len(results) - failed,
            "failed": failed,
            "phase_counts": phase_counts,
            "failure_buckets": dict(sorted(bucket_counts.items())),
            "profiled": profiled,
            "recommended_backends": dict(sorted(recommended_backends.items())),
            "planner_calibrations": dict(sorted(planner_calibrations.items())),
        },
        "results": results,
    }


def _print_result(result: dict[str, Any]) -> None:
    status = "ok" if result["ok"] else "fail"
    print(f"[{status}] {result['name']} ({result['phase']})")
    if not result["ok"]:
        print(f"  {result['error']}", file=sys.stderr)


def render_summary_markdown(summary: dict[str, Any]) -> str:
    meta = summary.get("summary", {})
    results = summary.get("results", [])
    phase_counts = meta.get("phase_counts", {})
    failure_buckets = meta.get("failure_buckets", {})

    lines = [
        "# PyRTL-to-RRTL Corpus Triage",
        "",
        f"- Total: {meta.get('total', 0)}",
        f"- Passed: {meta.get('passed', 0)}",
        f"- Failed: {meta.get('failed', 0)}",
        f"- Profiled: {meta.get('profiled', 0)}",
        "",
        "## Phases",
        "",
        "| Phase | Count |",
        "| --- | ---: |",
    ]
    for phase in PHASES:
        count = phase_counts.get(phase, 0)
        if count:
            lines.append(f"| {phase} | {count} |")

    lines.extend(["", "## Failure Buckets", ""])
    if failure_buckets:
        lines.extend(["| Bucket | Count |", "| --- | ---: |"])
        for bucket, count in sorted(failure_buckets.items()):
            lines.append(f"| {_md(bucket)} | {count} |")
    else:
        lines.append("No failures.")

    recommended_backends = meta.get("recommended_backends") or {}
    lines.extend(["", "## Runtime Profile Recommendations", ""])
    if recommended_backends:
        lines.extend(["| Backend | Count |", "| --- | ---: |"])
        for backend, count in sorted(recommended_backends.items()):
            lines.append(f"| {_md(backend)} | {count} |")
    else:
        lines.append("No profiled recommendations.")

    planner_calibrations = meta.get("planner_calibrations") or {}
    if planner_calibrations:
        lines.extend(["", "## Calibration Inputs", ""])
        lines.extend(["| Calibration | Targets |", "| --- | ---: |"])
        for path, count in sorted(planner_calibrations.items()):
            lines.append(f"| {_md(path)} | {count} |")

    planner_feedback = summary.get("planner_feedback") or {}
    if planner_feedback:
        feedback_summary = planner_feedback.get("summary") or {}
        lines.extend([
            "",
            "## Planner Feedback",
            "",
            f"- Profiles: {feedback_summary.get('profiles', 0)}",
            f"- Plan hits: {feedback_summary.get('plan_hits', 0)}",
            f"- Plan misses: {feedback_summary.get('plan_misses', 0)}",
            f"- Plan hit rate: {feedback_summary.get('plan_hit_rate', 0.0):.2%}",
            f"- Static profitability hit rate: {feedback_summary.get('static_profitability_hit_rate', 0.0):.2%}",
        ])
        miss_reasons = feedback_summary.get("miss_reasons") or {}
        if miss_reasons:
            lines.extend(["", "| Miss Reason | Count |", "| --- | ---: |"])
            for reason, count in sorted(miss_reasons.items()):
                lines.append(f"| {_md(reason)} | {count} |")

    planner_calibration = summary.get("planner_calibration") or {}
    if planner_calibration:
        calibration_summary = planner_calibration.get("summary") or {}
        backend_preferences = calibration_summary.get("backend_preferences") or []
        gpu_preferences = calibration_summary.get("gpu_option_preferences") or []
        profitability_preferences = calibration_summary.get(
            "profitability_backend_preferences"
        ) or []
        profitability_penalties = calibration_summary.get("profitability_penalties") or []
        profitability_feature_preferences = calibration_summary.get(
            "profitability_feature_preferences"
        ) or []
        profitability_feature_penalties = calibration_summary.get(
            "profitability_feature_penalties"
        ) or []
        lines.extend([
            "",
            "## Planner Calibration",
            "",
            f"- Backend preferences: {len(backend_preferences)}",
            f"- GPU option preferences: {len(gpu_preferences)}",
            f"- Profitability backend preferences: {len(profitability_preferences)}",
            f"- Profitability penalties: {len(profitability_penalties)}",
            f"- Profitability feature preferences: {len(profitability_feature_preferences)}",
            f"- Profitability feature penalties: {len(profitability_feature_penalties)}",
        ])
        if backend_preferences:
            top = backend_preferences[0]
            lines.append(
                f"- Top backend preference: {_md(top.get('signature'))} ({float(top.get('score') or 0.0):.2f})"
            )
        if profitability_preferences:
            top = profitability_preferences[0]
            lines.append(
                f"- Top profitability backend: {_md(top.get('signature'))} ({float(top.get('score') or 0.0):.2f})"
            )
        if profitability_penalties:
            top = profitability_penalties[0]
            lines.append(
                f"- Top profitability penalty: {_md(top.get('signature'))} ({float(top.get('score') or 0.0):.2f})"
            )
        if profitability_feature_preferences:
            top = profitability_feature_preferences[0]
            lines.append(
                f"- Top profitability feature: {_md(top.get('signature'))} ({float(top.get('score') or 0.0):.2f})"
            )
        if profitability_feature_penalties:
            top = profitability_feature_penalties[0]
            lines.append(
                f"- Top profitability feature penalty: {_md(top.get('signature'))} ({float(top.get('score') or 0.0):.2f})"
            )

    failures = [result for result in results if not result.get("ok")]
    lines.extend(["", "## Failures", ""])
    if failures:
        lines.extend(["| Name | Phase | Bucket | Error |", "| --- | --- | --- | --- |"])
        for result in failures:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md(result.get("name")),
                        _md(result.get("phase")),
                        _md(result.get("bucket") or classify_failure(result)),
                        _md(result.get("error")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("All targets passed.")

    return "\n".join(lines) + "\n"


def render_validation_report(findings: list[dict[str, Any]]) -> str:
    lines = [
        "# PyRTL Corpus Manifest Validation",
        "",
        f"- Status: {'failed' if findings else 'passed'}",
        f"- Findings: {len(findings)}",
        "",
    ]
    if not findings:
        lines.append("Manifest validation passed.")
        return "\n".join(lines) + "\n"

    grouped: dict[str, int] = {}
    for finding in findings:
        error = str(finding.get("error") or "")
        grouped[error] = grouped.get(error, 0) + 1

    lines.extend(["## Error Groups", "", "| Error | Count |", "| --- | ---: |"])
    for error, count in sorted(grouped.items()):
        lines.append(f"| {_md(error)} | {count} |")

    lines.extend(["", "## Findings", ""])
    lines.extend(["| Name | Target | Error |", "| --- | --- | --- |"])
    for finding in findings:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(finding.get("name") or "manifest"),
                    _md(finding.get("target")),
                    _md(finding.get("error")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def classify_failure(result: dict[str, Any]) -> str | None:
    if result.get("ok"):
        return None
    phase = str(result.get("phase") or "unknown")
    error = str(result.get("error") or "")
    lower = error.lower()

    if "unsupported pyrtl op" in lower:
        return "unsupported_op"
    if "trace mismatch" in lower or phase == "compare" and "mismatch" in lower:
        return "trace_mismatch"
    if any(
        token in lower
        for token in (
            "missing signal",
            "unknown signal",
            "unknown wire",
            "no signal",
            "not found",
        )
    ):
        return "missing_signal"
    if any(token in lower for token in ("width", "bitwidth", " type ", "type mismatch")):
        return "width_or_type"
    if any(token in lower for token in ("memory", "mem ", "rom")):
        return "memory"
    if "register" in lower:
        return "register"
    if phase in {
        "manifest",
        "build",
        "export",
        "check",
        "sv",
        "json",
        "trace",
        "compare",
        "profile",
        "profile_replay",
    }:
        return phase
    return "other"


def _manifest_targets(manifest: Any) -> tuple[list[Any] | None, str | None]:
    if isinstance(manifest, list):
        return manifest, None
    if isinstance(manifest, dict) and isinstance(manifest.get("targets"), list):
        return manifest["targets"], None
    return None, "manifest must be a list or an object with a 'targets' list"


def _effective_name(spec: dict[str, Any], index: int) -> str:
    name = spec.get("name")
    if isinstance(name, str):
        return name
    target = spec.get("target")
    if isinstance(target, str):
        return target.replace(":", "_").replace(".", "_")
    return f"target-{index}"


def _resolve_target(target: str) -> Any:
    module_name, func_name = target.split(":", 1)
    if not module_name or not func_name:
        raise ValueError("target must include both module and function")
    module = importlib.import_module(module_name)
    value = module
    for part in func_name.split("."):
        if not part:
            raise ValueError("target function path contains an empty segment")
        value = getattr(value, part)
    return value


def _validate_inputs(
    index: int,
    name: str,
    target: Any,
    inputs: Any,
) -> list[dict[str, Any]]:
    if not isinstance(inputs, list):
        return [_validation_failure(index, name, target, "inputs must be a list of dictionaries")]

    findings = []
    for step_index, step in enumerate(inputs):
        if not isinstance(step, dict):
            findings.append(
                _validation_failure(
                    index,
                    name,
                    target,
                    f"inputs[{step_index}] must be a dictionary",
                )
            )
            continue
        for key, value in step.items():
            if not isinstance(key, str):
                findings.append(
                    _validation_failure(
                        index,
                        name,
                        target,
                        f"inputs[{step_index}] key {key!r} must be a string",
                    )
                )
            if not isinstance(value, int):
                findings.append(
                    _validation_failure(
                        index,
                        name,
                        target,
                        f"inputs[{step_index}][{key!r}] must be an integer",
                    )
                )
    return findings


def _run_one(
    spec: dict[str, Any],
    *,
    out_dir: Path,
    pyrtl2rrtl: list[str],
    emit_sv: bool,
    emit_json: bool,
    profile_replay_hot_repeat: int | None,
    planner_calibration: Path | None,
) -> dict[str, Any]:
    target = spec.get("target")
    name = spec.get("name") or (
        target.replace(":", "_").replace(".", "_") if isinstance(target, str) else "target"
    )
    if not isinstance(target, str):
        return _failure(name, None, "manifest", "target must be a module:function string")

    phase = "build"
    outputs: dict[str, str] = {}
    try:
        if spec.get("reset_working_block", True):
            pyrtl.reset_working_block()
        result = _load_target(target)()
        block = result if isinstance(result, pyrtl.Block) else pyrtl.working_block()
        top_name = str(spec.get("top_name", name))
        clock_name = str(spec.get("clock_name", "clk"))

        phase = "export"
        export_path = out_dir / f"{name}.pyrtl.json"
        with export_path.open("w", encoding="utf-8") as dest:
            export_block_json(dest, block, top_name=top_name, clock_name=clock_name)
        outputs["export"] = str(export_path)

        phase = "check"
        _run_cmd([*pyrtl2rrtl, "check", str(export_path)], name, "check")

        if emit_sv:
            phase = "sv"
            sv = _run_cmd([*pyrtl2rrtl, "sv", str(export_path)], name, "sv")
            sv_path = out_dir / f"{name}.sv"
            sv_path.write_text(sv, encoding="utf-8")
            outputs["sv"] = str(sv_path)

        if emit_json:
            phase = "json"
            compiled = _run_cmd([*pyrtl2rrtl, "json", str(export_path)], name, "json")
            json_path = out_dir / f"{name}.rrtl.json"
            json_path.write_text(compiled, encoding="utf-8")
            outputs["json"] = str(json_path)

        inputs = spec.get("inputs")
        if inputs is not None:
            if not isinstance(inputs, list):
                return _failure(
                    name,
                    target,
                    "manifest",
                    "inputs must be a list of input dictionaries",
                    outputs,
                )
            phase = "trace"
            trace = simulate_trace(inputs, block)
            trace_path = out_dir / f"{name}.trace.json"
            trace_path.write_text(
                json.dumps(trace, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            outputs["trace"] = str(trace_path)
            phase = "compare"
            _run_cmd([*pyrtl2rrtl, "compare", str(export_path), str(trace_path)], name, "compare")

        runtime_profile = None
        if spec.get("benchmark_profile", False):
            if inputs is None:
                return _failure(
                    name,
                    target,
                    "manifest",
                    "benchmark_profile requires inputs",
                    outputs,
                )
            phase = "profile"
            runtime_profile = _run_profile_for_target(
                name,
                block,
                inputs,
                export_path,
                trace_path,
                out_dir,
                pyrtl2rrtl,
                outputs,
                packed_lanes=int(spec.get("packed_lanes", 2)),
                repeat=int(spec.get("repeat", 1)),
                warmup=int(spec.get("warmup", 0)),
                profile_replay_hot_repeat=int(
                    spec.get(
                        "profile_replay_hot_repeat",
                        profile_replay_hot_repeat or 0,
                    )
                ),
                planner_calibration=planner_calibration,
            )

        result = {
            "name": name,
            "target": target,
            "phase": "done",
            "ok": True,
            "error": None,
            "bucket": None,
            "outputs": outputs,
        }
        if runtime_profile is not None:
            result["runtime_profile"] = runtime_profile
        return result
    except Exception as err:  # noqa: BLE001 - corpus runs should classify every target failure.
        failure_phase = phase
        if phase == "profile" and "profile_replay failed" in str(err):
            failure_phase = "profile_replay"
        return _failure(name, target, failure_phase, str(err), outputs)


def _run_profile_for_target(
    name: str,
    block: pyrtl.Block,
    inputs: list[dict[str, int]],
    export_path: Path,
    trace_path: Path,
    out_dir: Path,
    pyrtl2rrtl: list[str],
    outputs: dict[str, str],
    *,
    packed_lanes: int,
    repeat: int,
    warmup: int,
    profile_replay_hot_repeat: int,
    planner_calibration: Path | None,
) -> dict[str, Any]:
    lane_vectors = [[dict(step) for step in inputs] for _ in range(packed_lanes)]
    lane_trace = simulate_lane_trace(lane_vectors, block)
    lane_trace_path = out_dir / f"{name}.lane_trace.json"
    lane_trace_path.write_text(
        json.dumps(lane_trace, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["lane_trace"] = str(lane_trace_path)
    planner_calibration_args = _planner_calibration_args(planner_calibration)
    if planner_calibration is not None:
        outputs["planner_calibration"] = str(planner_calibration)

    backend_plan_path = out_dir / f"{name}.backend_plan.json"
    backend_plan = json.loads(
        _run_cmd(
            [
                *pyrtl2rrtl,
                "plan-backends",
                str(export_path),
                str(lane_trace_path),
                *planner_calibration_args,
            ],
            name,
            "profile",
        )
    )
    backend_plan_path.write_text(
        json.dumps(backend_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["backend_plan"] = str(backend_plan_path)

    threaded = json.loads(
        _run_cmd(
            [
                *pyrtl2rrtl,
                "bench-threaded-trace",
                str(export_path),
                str(lane_trace_path),
                "--repeat",
                str(repeat),
                "--warmup",
                str(warmup),
                "--plan-first",
                *planner_calibration_args,
            ],
            name,
            "profile",
        )
    )
    threaded_autotune = json.loads(
        _run_cmd(
            [
                *pyrtl2rrtl,
                "bench-threaded-trace",
                str(export_path),
                str(lane_trace_path),
                "--repeat",
                str(repeat),
                "--warmup",
                str(warmup),
                "--autotune",
            ],
            name,
            "profile",
        )
    )
    gpu_combined = json.loads(
        _run_cmd(
            [
                *pyrtl2rrtl,
                "bench-gpu-combined",
                str(export_path),
                str(lane_trace_path),
                "--repeat",
                str(repeat),
                "--warmup",
                str(warmup),
                *planner_calibration_args,
            ],
            name,
            "profile",
        )
    )
    gpu_trace = gpu_combined.get("static_trace") or {}
    gpu_sweep = gpu_combined.get("option_sweep") or {}
    gpu_measured = bench._annotate_measured_gpu_trace(
        gpu_combined.get("measured_trace") or {},
        gpu_sweep,
    )
    unavailable = {
        "available": False,
        "replay_ns_best": 0,
        "replay_ns_median": 0,
        "mismatch_count": 0,
        "error": "not-run",
    }
    evaluation = bench.evaluate_backend_plan(
        rrtl=unavailable,
        rrtl_packed=unavailable,
        rrtl_single=unavailable,
        rrtl_backends={"backends": []},
        rrtl_threaded=threaded,
        rrtl_threaded_autotune=threaded_autotune,
        rrtl_gpu=gpu_trace,
        backend_plan=backend_plan,
        rrtl_gpu_measured=gpu_measured,
        rrtl_gpu_option_sweep=gpu_sweep,
    )
    runtime_profile = bench.build_runtime_profile(
        {
            "config": {
                "repeat": repeat,
                "warmup": warmup,
                "packed_lanes": packed_lanes,
                "steps": len(inputs),
            },
            "backend_plan_evaluation": evaluation,
            "rrtl_threaded_autotune_trace": threaded_autotune,
            "rrtl_gpu_measured_trace": gpu_measured,
        }
    )
    runtime_profile_path = out_dir / f"{name}.runtime_profile.json"
    runtime_profile_path.write_text(
        json.dumps(runtime_profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["runtime_profile"] = str(runtime_profile_path)

    profile_replay_path = out_dir / f"{name}.profile_replay.json"
    profile_replay = _run_cmd(
        [
            *pyrtl2rrtl,
            "bench-profile-replay",
            str(export_path),
            str(trace_path),
            str(lane_trace_path),
            str(runtime_profile_path),
            "--repeat",
            str(repeat),
            "--warmup",
            str(warmup),
            "--lanes",
            str(packed_lanes),
        ],
        name,
        "profile_replay",
    )
    profile_replay_path.write_text(profile_replay, encoding="utf-8")
    outputs["profile_replay"] = str(profile_replay_path)
    if profile_replay_hot_repeat:
        hot_profile_sweep_path = out_dir / f"{name}.hot_profile_sweep.json"
        hot_profile_replay_path = out_dir / f"{name}.hot_profile_replay.json"
        hot_profile_sweep = bench.run_hot_profile_sweep(
            runtime_profile=runtime_profile,
            backend_plan=backend_plan,
            rrtl_threaded=threaded,
            rrtl_threaded_autotune=threaded_autotune,
            rrtl_gpu_measured=gpu_measured,
            pyrtl2rrtl=pyrtl2rrtl,
            export_path=export_path,
            trace_path=trace_path,
            lane_trace_path=lane_trace_path,
            out_dir=out_dir,
            profile_prefix=f"{name}.hot_sweep",
            repeat=profile_replay_hot_repeat,
            warmup=warmup,
            lanes=packed_lanes,
            command_runner=lambda cmd, phase: _run_cmd(cmd, name, phase),
        )
        hot_profile_sweep_path.write_text(
            json.dumps(hot_profile_sweep, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        outputs["hot_profile_sweep"] = str(hot_profile_sweep_path)
        if hot_profile_sweep.get("selected_runtime_profile"):
            runtime_profile = hot_profile_sweep["selected_runtime_profile"]
            runtime_profile_path.write_text(
                json.dumps(runtime_profile, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        hot_profile_replay = hot_profile_sweep.get("selected_replay") or {}
        hot_profile_replay_path.write_text(
            json.dumps(hot_profile_replay, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        outputs["hot_profile_replay"] = str(hot_profile_replay_path)
    return runtime_profile


def _planner_calibration_args(path: Path | None) -> list[str]:
    return ["--planner-calibration", str(path)] if path is not None else []


def _run_cmd(cmd: list[str], target_name: str, phase: str) -> str:
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{target_name} {phase} failed: {detail}")
    return result.stdout


def _failure(
    name: str,
    target: str | None,
    phase: str,
    error: str,
    outputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    result = {
        "name": name,
        "target": target,
        "phase": phase,
        "ok": False,
        "error": error,
        "outputs": outputs or {},
    }
    result["bucket"] = classify_failure(result)
    return result


def _validation_failure(
    index: int | None,
    name: str | None,
    target: Any,
    error: str,
) -> dict[str, Any]:
    return {
        "index": index,
        "name": name,
        "target": target if isinstance(target, str) else None,
        "phase": "manifest",
        "ok": False,
        "error": error,
        "bucket": "manifest",
    }


def _print_validation_findings(findings: list[dict[str, Any]]) -> None:
    for finding in findings:
        label = finding.get("name") or "manifest"
        print(f"[invalid] {label}: {finding['error']}", file=sys.stderr)
    print(f"{len(findings)} manifest validation error(s)", file=sys.stderr)


def _md(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a PyRTL-to-RRTL corpus manifest")
    parser.add_argument("manifest", help="JSON manifest file")
    parser.add_argument("--out-dir", default="target/rrtl-pyrtl-corpus")
    parser.add_argument(
        "--pyrtl2rrtl",
        default="cargo run -q -p rrtl-pyrtl --bin pyrtl2rrtl --",
        help="command used to invoke the Rust bridge CLI",
    )
    parser.add_argument("--emit-sv", action="store_true")
    parser.add_argument("--emit-json", action="store_true")
    parser.add_argument("--summary-json")
    parser.add_argument("--summary-md")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate manifest shape and target importability without building designs",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first target failure instead of triaging the whole corpus",
    )
    parser.add_argument(
        "--profile-replay-hot-repeat",
        type=int,
        default=0,
        help="run an additional hot selected-profile replay for each profiled target",
    )
    parser.add_argument(
        "--planner-calibration",
        help="planner_calibration.json to feed into static backend planning commands",
    )
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.validate_only:
        findings = validate_manifest(manifest)
        if findings:
            _print_validation_findings(findings)
            return 1
        print("manifest validation passed")
        return 0

    out_dir = Path(args.out_dir)
    results = run_manifest(
        manifest,
        out_dir=out_dir,
        pyrtl2rrtl=shlex.split(args.pyrtl2rrtl),
        emit_sv=args.emit_sv,
        emit_json=args.emit_json,
        fail_fast=args.fail_fast,
        profile_replay_hot_repeat=args.profile_replay_hot_repeat,
        planner_calibration=Path(args.planner_calibration)
        if args.planner_calibration
        else None,
    )
    summary = build_summary(results)
    if summary.get("summary", {}).get("profiled", 0):
        planner_feedback = bench.build_planner_feedback(out_dir)
        feedback_json_path = out_dir / "planner_feedback.json"
        feedback_md_path = out_dir / "planner_feedback.md"
        feedback_json_path.write_text(
            json.dumps(planner_feedback, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        feedback_md_path.write_text(
            bench.render_planner_feedback_markdown(planner_feedback),
            encoding="utf-8",
        )
        planner_calibration = bench.build_planner_calibration(planner_feedback)
        calibration_json_path = out_dir / "planner_calibration.json"
        calibration_md_path = out_dir / "planner_calibration.md"
        calibration_json_path.write_text(
            json.dumps(planner_calibration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        calibration_md_path.write_text(
            bench.render_planner_calibration_markdown(planner_calibration),
            encoding="utf-8",
        )
        summary["planner_feedback"] = planner_feedback
        summary["summary"]["planner_feedback"] = planner_feedback["summary"]
        summary["planner_calibration"] = planner_calibration
        summary["summary"]["planner_calibration"] = planner_calibration["summary"]
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.summary_md:
        summary_path = Path(args.summary_md)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(render_summary_markdown(summary), encoding="utf-8")
    failures = [result for result in results if not result["ok"]]
    print(f"{len(results) - len(failures)}/{len(results)} targets passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
