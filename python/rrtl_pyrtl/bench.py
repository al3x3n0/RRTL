"""Synthetic PyRTL-to-RRTL performance benchmark harness."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from statistics import median
from typing import Any

import pyrtl
from pyrtl import wire

from .export import export_block_json, simulate_lane_trace, simulate_trace


def build_systolic_mac(
    *,
    rows: int = 2,
    cols: int = 2,
    data_width: int = 8,
    acc_width: int = 32,
) -> pyrtl.Block:
    """Build a small signed systolic-style MAC array as a PyRTL block."""

    _validate_positive("rows", rows)
    _validate_positive("cols", cols)
    _validate_positive("data_width", data_width)
    _validate_positive("acc_width", acc_width)
    if acc_width < data_width * 2:
        raise ValueError("acc_width must be at least 2 * data_width")

    pyrtl.reset_working_block()
    acts = [pyrtl.Input(data_width, f"act_{row}") for row in range(rows)]
    weights = [pyrtl.Input(data_width, f"weight_{col}") for col in range(cols)]

    act_regs = [
        [pyrtl.Register(data_width, f"act_r{row}_c{col}") for col in range(cols)]
        for row in range(rows)
    ]
    weight_regs = [
        [pyrtl.Register(data_width, f"weight_r{row}_c{col}") for col in range(cols)]
        for row in range(rows)
    ]
    psum_regs = [
        [pyrtl.Register(acc_width, f"psum_r{row}_c{col}") for col in range(cols)]
        for row in range(rows)
    ]

    for row in range(rows):
        for col in range(cols):
            act_src = acts[row] if col == 0 else act_regs[row][col - 1]
            weight_src = weights[col] if row == 0 else weight_regs[row - 1][col]
            act_regs[row][col].next <<= act_src
            weight_regs[row][col].next <<= weight_src

            product = pyrtl.signed_mult(act_regs[row][col], weight_regs[row][col])
            product = _fit_signed(product, acc_width)
            psum_next = (psum_regs[row][col] + product).truncate(acc_width)
            psum_regs[row][col].next <<= psum_next

            out = pyrtl.Output(acc_width, f"out_r{row}_c{col}")
            out <<= psum_regs[row][col]

    return pyrtl.working_block()


def input_vectors(
    *,
    rows: int,
    cols: int,
    data_width: int,
    steps: int,
) -> list[dict[str, int]]:
    """Return deterministic signed two's-complement input vectors."""

    _validate_positive("steps", steps)
    mask = (1 << data_width) - 1
    vectors = []
    for step in range(steps):
        item: dict[str, int] = {}
        for row in range(rows):
            item[f"act_{row}"] = ((step * 3 + row * 5 + 1) & mask)
        for col in range(cols):
            item[f"weight_{col}"] = ((step * 7 + col * 11 + 3) & mask)
        vectors.append(item)
    return vectors


def lane_input_vectors(
    *,
    rows: int,
    cols: int,
    data_width: int,
    steps: int,
    lanes: int,
) -> list[list[dict[str, int]]]:
    """Return deterministic independent input vectors for each replay lane."""

    _validate_positive("lanes", lanes)
    mask = (1 << data_width) - 1
    lane_vectors = []
    for lane in range(lanes):
        vectors = []
        for step in range(steps):
            item: dict[str, int] = {}
            for row in range(rows):
                item[f"act_{row}"] = ((step * 3 + row * 5 + lane * 13 + 1) & mask)
            for col in range(cols):
                item[f"weight_{col}"] = ((step * 7 + col * 11 + lane * 17 + 3) & mask)
            vectors.append(item)
        lane_vectors.append(vectors)
    return lane_vectors


def run_benchmark(
    *,
    out_dir: Path,
    rows: int = 2,
    cols: int = 2,
    steps: int = 32,
    data_width: int = 8,
    acc_width: int = 32,
    repeat: int = 3,
    warmup: int = 1,
    packed_lanes: int = 1,
    profile_replay_hot_repeat: int = 0,
    hot_profile_select: bool = True,
    pyrtl2rrtl: list[str] | None = None,
) -> dict[str, Any]:
    """Run the synthetic benchmark and write JSON/Markdown artifacts."""

    _validate_positive("repeat", repeat)
    _validate_positive("packed_lanes", packed_lanes)
    if profile_replay_hot_repeat < 0:
        raise ValueError("profile_replay_hot_repeat must be non-negative")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    out_dir.mkdir(parents=True, exist_ok=True)
    pyrtl2rrtl = pyrtl2rrtl or [
        "cargo",
        "run",
        "-q",
        "-p",
        "rrtl-pyrtl",
        "--bin",
        "pyrtl2rrtl",
        "--",
    ]

    block = build_systolic_mac(
        rows=rows,
        cols=cols,
        data_width=data_width,
        acc_width=acc_width,
    )
    vectors = input_vectors(rows=rows, cols=cols, data_width=data_width, steps=steps)

    export_path = out_dir / "systolic_mac.pyrtl.json"
    trace_path = out_dir / "systolic_mac.trace.json"
    lane_trace_path = out_dir / "systolic_mac.lane_trace.json"
    backend_plan_path = out_dir / "backend_plan.json"
    bench_json_path = out_dir / "bench.json"
    bench_md_path = out_dir / "bench.md"
    runtime_profile_path = out_dir / "runtime_profile.json"
    hot_profile_replay_path = out_dir / "hot_profile_replay.json"
    hot_profile_sweep_path = out_dir / "hot_profile_sweep.json"

    with export_path.open("w", encoding="utf-8") as dest:
        export_block_json(dest, block, top_name="SyntheticSystolicMac")
    trace = simulate_trace(vectors, block)
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lane_vectors = lane_input_vectors(
        rows=rows,
        cols=cols,
        data_width=data_width,
        steps=steps,
        lanes=packed_lanes,
    )
    lane_trace = simulate_lane_trace(lane_vectors, block)
    lane_trace_path.write_text(
        json.dumps(lane_trace, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fast = time_fast_simulation(block, vectors, repeat=repeat, warmup=warmup)
    rrtl = _run_bench_trace(
        pyrtl2rrtl,
        export_path,
        trace_path,
        repeat=repeat,
        warmup=warmup,
    )
    rrtl_packed = _run_bench_packed_trace(
        pyrtl2rrtl,
        export_path,
        trace_path,
        repeat=repeat,
        warmup=warmup,
        lanes=packed_lanes,
    )
    rrtl_single = _run_bench_single_trace(
        pyrtl2rrtl,
        export_path,
        trace_path,
        repeat=repeat,
        warmup=warmup,
    )
    rrtl_backends = _run_bench_backends(
        pyrtl2rrtl,
        export_path,
        trace_path,
        repeat=repeat,
        warmup=warmup,
        lanes=packed_lanes,
        backends="scalar,packed-cpu,simd-cpu,jit-cpu",
    )
    rrtl_backend_plan = _run_plan_backends(
        pyrtl2rrtl,
        export_path,
        lane_trace_path,
    )
    backend_plan_path.write_text(
        json.dumps(rrtl_backend_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rrtl_threaded = _run_bench_threaded_trace(
        pyrtl2rrtl,
        export_path,
        lane_trace_path,
        repeat=repeat,
        warmup=warmup,
    )
    rrtl_threaded_autotune = _run_bench_threaded_autotune_trace(
        pyrtl2rrtl,
        export_path,
        lane_trace_path,
        repeat=repeat,
        warmup=warmup,
    )
    rrtl_gpu_combined = _run_bench_gpu_combined(
        pyrtl2rrtl,
        export_path,
        lane_trace_path,
        repeat=repeat,
        warmup=warmup,
    )
    rrtl_gpu = rrtl_gpu_combined["static_trace"]
    rrtl_gpu_option_sweep = rrtl_gpu_combined["option_sweep"]
    rrtl_gpu_measured = _annotate_measured_gpu_trace(
        rrtl_gpu_combined["measured_trace"],
        rrtl_gpu_option_sweep,
    )

    speedup_best = _speedup(fast["run_ns_best"], rrtl["replay_ns_best"])
    speedup_median = _speedup(fast["run_ns_median"], rrtl["replay_ns_median"])
    packed_speedup_best = _speedup(fast["run_ns_best"], rrtl_packed["replay_ns_best"])
    packed_speedup_median = _speedup(fast["run_ns_median"], rrtl_packed["replay_ns_median"])
    single_speedup_best = _speedup(fast["run_ns_best"], rrtl_single["replay_ns_best"])
    single_speedup_median = _speedup(fast["run_ns_median"], rrtl_single["replay_ns_median"])
    threaded_speedup_best = _speedup(fast["run_ns_best"], rrtl_threaded["replay_ns_best"])
    threaded_speedup_median = _speedup(fast["run_ns_median"], rrtl_threaded["replay_ns_median"])
    threaded_autotune_speedup_best = _speedup(
        fast["run_ns_best"], rrtl_threaded_autotune["replay_ns_best"]
    )
    threaded_autotune_speedup_median = _speedup(
        fast["run_ns_median"], rrtl_threaded_autotune["replay_ns_median"]
    )
    gpu_speedup_best = _speedup(fast["run_ns_best"], rrtl_gpu["replay_ns_best"])
    gpu_speedup_median = _speedup(fast["run_ns_median"], rrtl_gpu["replay_ns_median"])
    gpu_measured_speedup_best = _speedup(
        fast["run_ns_best"], rrtl_gpu_measured["replay_ns_best"]
    )
    gpu_measured_speedup_median = _speedup(
        fast["run_ns_median"], rrtl_gpu_measured["replay_ns_median"]
    )
    backend_plan_evaluation = evaluate_backend_plan(
        rrtl=rrtl,
        rrtl_packed=rrtl_packed,
        rrtl_single=rrtl_single,
        rrtl_backends=rrtl_backends,
        rrtl_threaded=rrtl_threaded,
        rrtl_threaded_autotune=rrtl_threaded_autotune,
        rrtl_gpu=rrtl_gpu,
        backend_plan=rrtl_backend_plan,
        rrtl_gpu_measured=rrtl_gpu_measured,
        rrtl_gpu_option_sweep=rrtl_gpu_option_sweep,
    )
    report = {
        "schema": "rrtl-pyrtl-bench-v1",
        "config": {
            "rows": rows,
            "cols": cols,
            "steps": steps,
            "data_width": data_width,
            "acc_width": acc_width,
            "repeat": repeat,
            "warmup": warmup,
            "packed_lanes": packed_lanes,
            "profile_replay_hot_repeat": profile_replay_hot_repeat,
            "hot_profile_select": hot_profile_select,
        },
        "outputs": {
            "export": str(export_path),
            "trace": str(trace_path),
            "lane_trace": str(lane_trace_path),
            "backend_plan": str(backend_plan_path),
            "bench_json": str(bench_json_path),
            "bench_md": str(bench_md_path),
            "runtime_profile": str(runtime_profile_path),
        },
        "pyrtl_fast": fast,
        "rrtl_trace": rrtl,
        "rrtl_packed_trace": rrtl_packed,
        "rrtl_single_trace": rrtl_single,
        "rrtl_backends": rrtl_backends,
        "rrtl_backend_plan": rrtl_backend_plan,
        "backend_plan_evaluation": backend_plan_evaluation,
        "rrtl_threaded_trace": rrtl_threaded,
        "rrtl_threaded_autotune_trace": rrtl_threaded_autotune,
        "rrtl_gpu_trace": rrtl_gpu,
        "rrtl_gpu_combined": rrtl_gpu_combined,
        "rrtl_gpu_option_sweep": rrtl_gpu_option_sweep,
        "rrtl_gpu_measured_trace": rrtl_gpu_measured,
        "speedup_best": speedup_best,
        "speedup_median": speedup_median,
        "packed_speedup_best": packed_speedup_best,
        "packed_speedup_median": packed_speedup_median,
        "single_speedup_best": single_speedup_best,
        "single_speedup_median": single_speedup_median,
        "threaded_speedup_best": threaded_speedup_best,
        "threaded_speedup_median": threaded_speedup_median,
        "threaded_autotune_speedup_best": threaded_autotune_speedup_best,
        "threaded_autotune_speedup_median": threaded_autotune_speedup_median,
        "gpu_speedup_best": gpu_speedup_best,
        "gpu_speedup_median": gpu_speedup_median,
        "gpu_measured_speedup_best": gpu_measured_speedup_best,
        "gpu_measured_speedup_median": gpu_measured_speedup_median,
    }
    runtime_profile = build_runtime_profile(report)
    report["runtime_profile"] = runtime_profile

    runtime_profile_path.write_text(
        json.dumps(runtime_profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if profile_replay_hot_repeat:
        hot_sweep = run_hot_profile_sweep(
            runtime_profile=runtime_profile,
            backend_plan=rrtl_backend_plan,
            rrtl_threaded=rrtl_threaded,
            rrtl_threaded_autotune=rrtl_threaded_autotune,
            rrtl_gpu_measured=rrtl_gpu_measured,
            pyrtl2rrtl=pyrtl2rrtl,
            export_path=export_path,
            trace_path=trace_path,
            lane_trace_path=lane_trace_path,
            out_dir=out_dir,
            profile_prefix="hot_sweep",
            repeat=profile_replay_hot_repeat,
            warmup=warmup,
            lanes=packed_lanes,
        )
        hot_profile_sweep_path.write_text(
            json.dumps(hot_sweep, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report["profile_replay_hot_sweep"] = hot_sweep
        report["outputs"]["hot_profile_sweep"] = str(hot_profile_sweep_path)
        hot_replay = hot_sweep.get("selected_replay") or {}
        if hot_profile_select and hot_sweep.get("selected_runtime_profile"):
            runtime_profile = hot_sweep["selected_runtime_profile"]
            report["runtime_profile"] = runtime_profile
            runtime_profile_path.write_text(
                json.dumps(runtime_profile, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        hot_profile_replay_path.write_text(
            json.dumps(hot_replay, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report["profile_replay_hot"] = hot_replay
        report["outputs"]["hot_profile_replay"] = str(hot_profile_replay_path)
    bench_json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bench_md_path.write_text(render_benchmark_markdown(report), encoding="utf-8")
    return report


def run_profile_selected_replay(
    *,
    out_dir: Path,
    runtime_profile_path: Path,
    rows: int = 2,
    cols: int = 2,
    steps: int = 32,
    data_width: int = 8,
    acc_width: int = 32,
    repeat: int = 3,
    warmup: int = 1,
    packed_lanes: int = 1,
    pyrtl2rrtl: list[str] | None = None,
) -> dict[str, Any]:
    """Run only the backend selected by a saved runtime profile."""

    _validate_positive("repeat", repeat)
    _validate_positive("packed_lanes", packed_lanes)
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    out_dir.mkdir(parents=True, exist_ok=True)
    pyrtl2rrtl = pyrtl2rrtl or [
        "cargo",
        "run",
        "-q",
        "-p",
        "rrtl-pyrtl",
        "--bin",
        "pyrtl2rrtl",
        "--",
    ]

    profile = load_runtime_profile(runtime_profile_path)
    block = build_systolic_mac(
        rows=rows,
        cols=cols,
        data_width=data_width,
        acc_width=acc_width,
    )
    vectors = input_vectors(rows=rows, cols=cols, data_width=data_width, steps=steps)
    export_path = out_dir / "systolic_mac.pyrtl.json"
    trace_path = out_dir / "systolic_mac.trace.json"
    lane_trace_path = out_dir / "systolic_mac.lane_trace.json"
    profile_replay_path = out_dir / "profile_replay.json"

    with export_path.open("w", encoding="utf-8") as dest:
        export_block_json(dest, block, top_name="SyntheticSystolicMac")
    trace = simulate_trace(vectors, block)
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lane_vectors = lane_input_vectors(
        rows=rows,
        cols=cols,
        data_width=data_width,
        steps=steps,
        lanes=packed_lanes,
    )
    lane_trace = simulate_lane_trace(lane_vectors, block)
    lane_trace_path.write_text(
        json.dumps(lane_trace, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    replay = _run_profile_selected_replay(
        profile,
        pyrtl2rrtl,
        export_path,
        trace_path,
        lane_trace_path,
        runtime_profile_path=runtime_profile_path,
        repeat=repeat,
        warmup=warmup,
        lanes=packed_lanes,
    )
    report = {
        "schema": "rrtl-pyrtl-profile-replay-v1",
        "config": {
            "rows": rows,
            "cols": cols,
            "steps": steps,
            "data_width": data_width,
            "acc_width": acc_width,
            "repeat": repeat,
            "warmup": warmup,
            "packed_lanes": packed_lanes,
        },
        "outputs": {
            "export": str(export_path),
            "trace": str(trace_path),
            "lane_trace": str(lane_trace_path),
            "runtime_profile": str(runtime_profile_path),
            "profile_replay": str(profile_replay_path),
        },
        "runtime_profile": profile,
        "selected_backend": profile["recommended_runtime_backend"],
        "selected_source": profile.get("recommended_runtime_source"),
        "replay": replay,
    }
    profile_replay_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def load_runtime_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    _validate_runtime_profile(profile)
    return profile


def _validate_runtime_profile(profile: dict[str, Any]) -> None:
    schema = profile.get("schema")
    if schema != "rrtl-pyrtl-runtime-profile-v1":
        raise ValueError(f"unsupported runtime profile schema `{schema}`")
    backend = profile.get("recommended_runtime_backend")
    if not backend:
        raise ValueError("runtime profile does not select a backend")
    if profile.get("recommended_runtime_source") == "no-valid-measurements":
        raise ValueError("runtime profile has no valid measurements")
    if not profile.get("selected_backend"):
        raise ValueError("runtime profile missing selected backend details")


def _run_profile_selected_replay(
    profile: dict[str, Any],
    pyrtl2rrtl: list[str],
    export_path: Path,
    trace_path: Path,
    lane_trace_path: Path,
    *,
    runtime_profile_path: Path,
    repeat: int,
    warmup: int,
    lanes: int,
) -> dict[str, Any]:
    _validate_runtime_profile(profile)
    return _run_json_cmd(
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
            str(lanes),
        ],
        "profile replay",
    )


def _run_json_cmd(cmd: list[str], label: str) -> dict[str, Any]:
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"pyrtl2rrtl {label} failed: {detail}")
    return json.loads(result.stdout)


def run_hot_profile_sweep(
    runtime_profile: dict[str, Any],
    backend_plan: dict[str, Any] | None,
    rrtl_threaded: dict[str, Any] | None,
    rrtl_threaded_autotune: dict[str, Any] | None,
    rrtl_gpu_measured: dict[str, Any] | None,
    pyrtl2rrtl: list[str],
    export_path: Path,
    trace_path: Path,
    lane_trace_path: Path,
    *,
    out_dir: Path,
    profile_prefix: str,
    repeat: int,
    warmup: int,
    lanes: int,
    command_runner: Any | None = None,
) -> dict[str, Any]:
    """Run replayable runtime-profile candidates through native hot replay."""

    _validate_positive("repeat", repeat)
    _validate_positive("lanes", lanes)
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = _hot_profile_sweep_candidates(
        runtime_profile,
        backend_plan,
        rrtl_threaded,
        rrtl_threaded_autotune,
        rrtl_gpu_measured,
    )
    reports = []
    for index, candidate in enumerate(candidates):
        profile_path = out_dir / f"{profile_prefix}.{index}.{candidate['name']}.runtime_profile.json"
        profile_path.write_text(
            json.dumps(candidate["runtime_profile"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        command = [
            *pyrtl2rrtl,
            "bench-profile-replay",
            str(export_path),
            str(trace_path),
            str(lane_trace_path),
            str(profile_path),
            "--repeat",
            str(repeat),
            "--warmup",
            str(warmup),
            "--lanes",
            str(lanes),
        ]
        report = {
            "candidate_index": index,
            "candidate_name": candidate["name"],
            "candidate_source": candidate["source"],
            "runtime_backend": candidate["runtime_profile"].get("recommended_runtime_backend"),
            "runtime_profile": str(profile_path),
            "available": False,
            "valid": False,
            "replay_ns_best": 0,
            "replay_ns_median": 0,
            "first_replay_ns": 0,
            "hot_replay_ns_best": 0,
            "hot_replay_ns_median": 0,
            "replay_ns_per_lane_step": 0.0,
            "setup_ns_total": 0,
            "setup_to_hot_ratio": 0.0,
            "hot_replay_speedup": 0.0,
            "mismatch_count": 0,
            "error": "",
        }
        try:
            replay = (
                json.loads(command_runner(command, "profile_replay"))
                if command_runner is not None
                else _run_json_cmd(command, f"hot profile sweep {candidate['name']}")
            )
            report["replay"] = replay
            report["available"] = True
            report["replay_ns_best"] = int(replay.get("replay_ns_best") or 0)
            report["replay_ns_median"] = int(replay.get("replay_ns_median") or 0)
            report["first_replay_ns"] = int(replay.get("first_replay_ns") or 0)
            report["hot_replay_ns_best"] = int(
                replay.get("hot_replay_ns_best") or replay.get("replay_ns_best") or 0
            )
            report["hot_replay_ns_median"] = int(
                replay.get("hot_replay_ns_median") or replay.get("replay_ns_median") or 0
            )
            report["replay_ns_per_lane_step"] = float(
                replay.get("replay_ns_per_lane_step") or 0.0
            )
            report["setup_ns_total"] = int(replay.get("setup_ns_total") or 0)
            report["setup_to_hot_ratio"] = float(
                replay.get("setup_to_hot_ratio")
                or replay.get("setup_to_replay_ratio")
                or 0.0
            )
            report["hot_replay_speedup"] = float(replay.get("hot_replay_speedup") or 0.0)
            report["mismatch_count"] = int(replay.get("mismatch_count") or 0)
            report["valid"] = _valid_hot_profile_replay(replay)
        except Exception as err:  # noqa: BLE001 - sweeps should keep failed candidates.
            report["error"] = str(err)
        reports.append(report)

    ranked = sorted(
        (report for report in reports if report.get("valid")),
        key=lambda report: (
            float(report.get("replay_ns_per_lane_step") or 0.0),
            int(report.get("replay_ns_best") or 0),
            str(report.get("candidate_name") or ""),
        ),
    )
    selected = ranked[0] if ranked else None
    selected_profile = None
    selected_replay = None
    if selected is not None:
        source_candidate = candidates[int(selected["candidate_index"])]
        selected_profile = _hot_selected_runtime_profile(
            source_candidate["runtime_profile"],
            selected,
            len(candidates),
        )
        selected_replay = selected.get("replay") or {}
    return {
        "schema": "rrtl-pyrtl-hot-profile-sweep-v1",
        "repeat": repeat,
        "warmup": warmup,
        "lanes": lanes,
        "candidate_count": len(candidates),
        "valid_candidate_count": len(ranked),
        "selected_candidate_index": selected.get("candidate_index") if selected else None,
        "selected_candidate_name": selected.get("candidate_name") if selected else None,
        "selected_backend": (selected_profile or {}).get("recommended_runtime_backend"),
        "selected_source": (selected_profile or {}).get("recommended_runtime_source"),
        "selected_replay": selected_replay,
        "selected_runtime_profile": selected_profile,
        "candidates": reports,
    }


def _hot_profile_sweep_candidates(
    runtime_profile: dict[str, Any],
    backend_plan: dict[str, Any] | None,
    rrtl_threaded: dict[str, Any] | None,
    rrtl_threaded_autotune: dict[str, Any] | None,
    rrtl_gpu_measured: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    candidates = []
    seen: set[str] = set()
    for backend in ("scalar", "packed-cpu", "simd-cpu", "jit-cpu"):
        profile = _candidate_runtime_profile(
            runtime_profile,
            f"rrtl_backend:{backend}",
            "hot-profile-sweep-cpu",
            {"backend": f"rrtl_backend:{backend}"},
        )
        _append_hot_candidate(candidates, seen, f"backend-{backend}", "cpu-backend", profile)

    planned_layout = (backend_plan or {}).get("selected_threaded_layout") or {}
    if planned_layout.get("workers"):
        profile = _candidate_runtime_profile(
            runtime_profile,
            "rrtl_threaded_autotune_trace",
            "hot-profile-sweep-planned-threaded",
            {
                "backend": "rrtl_threaded_autotune_trace",
                "selected_threaded_layout": planned_layout,
            },
        )
        _append_hot_candidate(candidates, seen, "planned-threaded", "planned-threaded", profile)

    autotune_layout = (rrtl_threaded_autotune or {}).get("selected_threaded_layout") or {}
    if _valid_benchmark_report(rrtl_threaded_autotune) and autotune_layout.get("workers"):
        profile = _candidate_runtime_profile(
            runtime_profile,
            "rrtl_threaded_autotune_trace",
            "hot-profile-sweep-autotuned-threaded",
            {
                "backend": "rrtl_threaded_autotune_trace",
                "selected_threaded_layout": autotune_layout,
            },
        )
        _append_hot_candidate(candidates, seen, "autotuned-threaded", "autotuned-threaded", profile)

    gpu_options = (rrtl_gpu_measured or {}).get("selected_gpu_options") or {}
    if _valid_benchmark_report(rrtl_gpu_measured) and gpu_options:
        profile = _candidate_runtime_profile(
            runtime_profile,
            "rrtl_gpu_measured_trace",
            "hot-profile-sweep-measured-gpu",
            {
                "backend": "rrtl_gpu_measured_trace",
                "selected_gpu_option_index": (rrtl_gpu_measured or {}).get(
                    "selected_gpu_option_index"
                ),
                "selected_gpu_options": gpu_options,
            },
        )
        _append_hot_candidate(candidates, seen, "measured-gpu", "measured-gpu", profile)
    return candidates


def _append_hot_candidate(
    candidates: list[dict[str, Any]],
    seen: set[str],
    name: str,
    source: str,
    profile: dict[str, Any],
) -> None:
    signature = json.dumps(profile.get("selected_backend") or {}, sort_keys=True)
    key = f"{profile.get('recommended_runtime_backend')}:{signature}"
    if key in seen:
        return
    seen.add(key)
    candidates.append({"name": name, "source": source, "runtime_profile": profile})


def _candidate_runtime_profile(
    base: dict[str, Any],
    backend: str,
    source: str,
    selected_backend: dict[str, Any],
) -> dict[str, Any]:
    profile = json.loads(json.dumps(base))
    profile["schema"] = "rrtl-pyrtl-runtime-profile-v1"
    profile["recommended_runtime_backend"] = backend
    profile["recommended_runtime_source"] = source
    profile["selected_backend"] = selected_backend
    profile.setdefault("measured_backends", [])
    return profile


def _valid_hot_profile_replay(report: dict[str, Any]) -> bool:
    return (
        report.get("schema") == "rrtl-pyrtl-profile-replay-hot-v1"
        and int(report.get("mismatch_count") or 0) == 0
        and int(report.get("replay_ns_best") or 0) > 0
        and float(report.get("replay_ns_per_lane_step") or 0.0) > 0.0
    )


def _hot_selected_runtime_profile(
    profile: dict[str, Any],
    selected: dict[str, Any],
    candidate_count: int,
) -> dict[str, Any]:
    result = json.loads(json.dumps(profile))
    result["recommended_runtime_source"] = "hot-profile-sweep"
    result["recommended_backend_reason"] = "hot-profile-sweep-fastest"
    result["hot_profile_sweep"] = {
        "selected_candidate_index": selected.get("candidate_index"),
        "selected_candidate_name": selected.get("candidate_name"),
        "selected_backend": result.get("recommended_runtime_backend"),
        "best_ns": int(selected.get("replay_ns_best") or 0),
        "median_ns": int(selected.get("replay_ns_median") or 0),
        "first_replay_ns": int(selected.get("first_replay_ns") or 0),
        "hot_replay_ns_best": int(
            selected.get("hot_replay_ns_best") or selected.get("replay_ns_best") or 0
        ),
        "hot_replay_ns_median": int(
            selected.get("hot_replay_ns_median") or selected.get("replay_ns_median") or 0
        ),
        "replay_ns_per_lane_step": float(selected.get("replay_ns_per_lane_step") or 0.0),
        "setup_ns_total": int(selected.get("setup_ns_total") or 0),
        "setup_to_hot_ratio": float(selected.get("setup_to_hot_ratio") or 0.0),
        "hot_replay_speedup": float(selected.get("hot_replay_speedup") or 0.0),
        "candidate_count": candidate_count,
    }
    return result


def build_runtime_profile(report: dict[str, Any]) -> dict[str, Any]:
    """Return a compact reusable backend profile from a benchmark report."""

    evaluation = report.get("backend_plan_evaluation") or {}
    recommended_backend = evaluation.get("recommended_runtime_backend")
    measured = [
        _runtime_profile_backend_evidence(backend)
        for backend in evaluation.get("measured_backends", [])
        if backend.get("valid")
    ]
    profile = {
        "schema": "rrtl-pyrtl-runtime-profile-v1",
        "config": dict(report.get("config") or {}),
        "recommended_runtime_backend": recommended_backend,
        "recommended_runtime_source": evaluation.get("recommended_runtime_source"),
        "static_plan_vs_recommended_speedup": evaluation.get(
            "static_plan_vs_recommended_speedup",
            0.0,
        ),
        "recommended_backend_reason": evaluation.get("recommended_backend_reason"),
        "planner_feedback": _runtime_profile_planner_feedback(evaluation),
        "measured_backends": measured,
    }
    backend_plan = report.get("rrtl_backend_plan") or {}
    if backend_plan.get("profitability"):
        profile["static_backend_profitability"] = backend_plan.get("profitability")
    if backend_plan.get("backend_candidates"):
        profile["static_backend_candidates"] = backend_plan.get("backend_candidates")

    threaded = _runtime_profile_threaded_selection(
        report.get("rrtl_threaded_autotune_trace"),
    )
    if threaded:
        profile["threaded_autotune"] = threaded

    gpu = _runtime_profile_gpu_selection(report.get("rrtl_gpu_measured_trace"))
    if gpu:
        profile["measured_gpu"] = gpu

    selected = _runtime_profile_selected_backend(report, recommended_backend)
    if selected:
        profile["selected_backend"] = selected
    return profile


def _runtime_profile_planner_feedback(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan_hit": bool(evaluation.get("plan_hit", False)),
        "miss_reason": evaluation.get("miss_reason") or "unknown",
        "planned_cpu_rank": evaluation.get("planned_cpu_rank"),
        "planned_gpu_rank": evaluation.get("planned_gpu_rank"),
        "planned_gpu_selected": bool(evaluation.get("planned_gpu_selected", False)),
        "gpu_option_plan_hit": bool(evaluation.get("gpu_option_plan_hit", False)),
        "gpu_option_miss_reason": evaluation.get("gpu_option_miss_reason") or "unknown",
        "planned_gpu_option_rank": evaluation.get("planned_gpu_option_rank"),
        "measured_gpu_option_best_index": evaluation.get("measured_gpu_option_best_index"),
        "static_profitability_backend": evaluation.get("static_profitability_backend"),
        "static_profitability_rank": evaluation.get("static_profitability_rank"),
        "static_profitability_hit": bool(evaluation.get("static_profitability_hit", False)),
        "static_profitability_miss_reason": evaluation.get(
            "static_profitability_miss_reason"
        )
        or "unknown",
    }


def _runtime_profile_backend_evidence(backend: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": backend.get("name"),
        "label": backend.get("label"),
        "best_ns": int(backend.get("best_ns") or 0),
        "median_ns": int(backend.get("median_ns") or 0),
    }


def _runtime_profile_threaded_selection(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not _valid_benchmark_report(report):
        return None
    autotune = report.get("autotune") or {}
    pruned = report.get("autotune_pruned_candidates") or autotune.get("pruned_candidates", [])
    return {
        "backend": "rrtl_threaded_autotune_trace",
        "selected_threaded_layout": report.get("selected_threaded_layout") or {},
        "selected_reason": report.get("selected_reason", ""),
        "autotune_selected_candidate": autotune.get("selected_candidate"),
        "autotune_candidate_count": len(autotune.get("candidates", [])),
        "autotune_pruned_candidate_count": len(pruned),
        "best_ns": int(report.get("replay_ns_best") or 0),
        "median_ns": int(report.get("replay_ns_median") or 0),
    }


def _runtime_profile_gpu_selection(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not _valid_benchmark_report(report):
        return None
    return {
        "backend": "rrtl_gpu_measured_trace",
        "selected_gpu_option_index": report.get("selected_gpu_option_index"),
        "selected_gpu_options": report.get("selected_gpu_options") or {},
        "gpu_replay_mode": report.get("gpu_replay_mode", ""),
        "best_ns": int(report.get("replay_ns_best") or 0),
        "median_ns": int(report.get("replay_ns_median") or 0),
    }


def _runtime_profile_selected_backend(
    report: dict[str, Any],
    backend_name: str | None,
) -> dict[str, Any] | None:
    if not backend_name:
        return None
    if backend_name == "rrtl_threaded_autotune_trace":
        return _runtime_profile_threaded_selection(report.get("rrtl_threaded_autotune_trace"))
    if backend_name == "rrtl_gpu_measured_trace":
        return _runtime_profile_gpu_selection(report.get("rrtl_gpu_measured_trace"))
    evidence = next(
        (
            backend
            for backend in (report.get("backend_plan_evaluation") or {}).get(
                "measured_backends",
                [],
            )
            if backend.get("name") == backend_name and backend.get("valid")
        ),
        None,
    )
    if not evidence:
        return None
    return {
        "backend": backend_name,
        "best_ns": int(evidence.get("best_ns") or 0),
        "median_ns": int(evidence.get("median_ns") or 0),
    }


def _valid_benchmark_report(report: dict[str, Any] | None) -> bool:
    if not report:
        return False
    return (
        bool(report.get("available", True))
        and int(report.get("mismatch_count") or 0) == 0
        and int(report.get("replay_ns_best") or 0) > 0
    )


def build_planner_feedback(out_dir: Path) -> dict[str, Any]:
    """Aggregate saved runtime profiles into planner calibration feedback."""

    profile_paths = [
        path
        for path in sorted(out_dir.glob("*.runtime_profile.json"))
        if not _is_hot_sweep_candidate_profile(path)
    ]
    default_profile = out_dir / "runtime_profile.json"
    if default_profile.exists():
        profile_paths.append(default_profile)

    targets = []
    warnings = []
    for profile_path in profile_paths:
        target_name = _feedback_target_name(out_dir, profile_path)
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception as err:  # noqa: BLE001 - feedback should report bad artifacts.
            warnings.append(f"{profile_path.name}: failed to read profile: {err}")
            continue
        if profile.get("schema") != "rrtl-pyrtl-runtime-profile-v1":
            warnings.append(f"{profile_path.name}: unsupported runtime profile schema")
            continue

        plan_path = _feedback_backend_plan_path(out_dir, target_name, profile_path)
        if plan_path is None:
            warnings.append(f"{profile_path.name}: matching backend plan not found")
        hot_path = _feedback_hot_profile_replay_path(out_dir, target_name, profile_path)
        hot_replay = None
        if hot_path is not None:
            try:
                hot_replay = json.loads(hot_path.read_text(encoding="utf-8"))
            except Exception as err:  # noqa: BLE001 - feedback should report bad artifacts.
                warnings.append(f"{hot_path.name}: failed to read hot replay: {err}")
        backend_plan = None
        if plan_path is not None:
            try:
                backend_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            except Exception as err:  # noqa: BLE001 - feedback should report bad artifacts.
                warnings.append(f"{plan_path.name}: failed to read backend plan: {err}")
        item = _planner_feedback_target(
            target_name,
            profile_path,
            plan_path,
            profile,
            hot_path,
            hot_replay,
            backend_plan,
        )
        targets.append(item)

    summary = _planner_feedback_summary(targets, warnings)
    return {
        "schema": "rrtl-pyrtl-planner-feedback-v1",
        "out_dir": str(out_dir),
        "summary": summary,
        "targets": targets,
        "warnings": warnings,
    }


def render_planner_feedback_markdown(feedback: dict[str, Any]) -> str:
    summary = feedback.get("summary") or {}
    lines = [
        "# RRTL Planner Feedback",
        "",
        f"- Profiles: {summary.get('profiles', 0)}",
        f"- Plan hits: {summary.get('plan_hits', 0)}",
        f"- Plan misses: {summary.get('plan_misses', 0)}",
        f"- Plan hit rate: {summary.get('plan_hit_rate', 0.0):.2%}",
        f"- GPU option hits: {summary.get('gpu_option_hits', 0)}",
        f"- GPU option misses: {summary.get('gpu_option_misses', 0)}",
        f"- Static profitability hits: {summary.get('static_profitability_hits', 0)}",
        f"- Static profitability misses: {summary.get('static_profitability_misses', 0)}",
        f"- Static profitability hit rate: {summary.get('static_profitability_hit_rate', 0.0):.2%}",
        "",
        "## Recommended Backends",
        "",
    ]
    recommended = summary.get("recommended_backends") or {}
    if recommended:
        lines.extend(["| Backend | Count |", "| --- | ---: |"])
        for backend, count in sorted(recommended.items()):
            lines.append(f"| {_md(backend)} | {count} |")
    else:
        lines.append("No backend recommendations.")

    lines.extend(["", "## Miss Reasons", ""])
    miss_reasons = summary.get("miss_reasons") or {}
    if miss_reasons:
        lines.extend(["| Reason | Count |", "| --- | ---: |"])
        for reason, count in sorted(miss_reasons.items()):
            lines.append(f"| {_md(reason)} | {count} |")
    else:
        lines.append("No planner misses.")

    lines.extend(["", "## Speedup Distribution", ""])
    speedup = summary.get("static_plan_vs_recommended_speedup") or {}
    lines.extend([
        f"- Count: {speedup.get('count', 0)}",
        f"- Median: {speedup.get('median', 0.0):.2f}x",
        f"- Max: {speedup.get('max', 0.0):.2f}x",
    ])

    hot = summary.get("hot_profile_replay") or {}
    lines.extend(["", "## Hot Profile Replay", ""])
    if hot.get("profiles", 0):
        lane_step = hot.get("replay_ns_per_lane_step") or {}
        setup_ratio = hot.get("setup_to_hot_ratio") or hot.get("setup_to_replay_ratio") or {}
        hot_speedup = hot.get("hot_replay_speedup") or {}
        lines.extend([
            f"- Profiles: {hot.get('profiles', 0)}",
            f"- Valid profiles: {hot.get('valid_profiles', 0)}",
            f"- Median ns per lane-step: {lane_step.get('median', 0.0):.2f}",
            f"- Best ns per lane-step: {lane_step.get('min', 0.0):.2f}",
            f"- Median setup/hot ratio: {setup_ratio.get('median', 0.0):.2f}x",
            f"- Median hot replay speedup: {hot_speedup.get('median', 0.0):.2f}x",
            "",
            "| Backend | Count |",
            "| --- | ---: |",
        ])
        for backend, count in sorted((hot.get("selected_backends") or {}).items()):
            lines.append(f"| {_md(backend)} | {count} |")
    else:
        lines.append("No hot replay artifacts.")

    lines.extend(["", "## Worst Misses", ""])
    worst = summary.get("worst_misses") or []
    if worst:
        lines.extend([
            "| Target | Recommended Backend | Miss Reason | Speedup |",
            "| --- | --- | --- | ---: |",
        ])
        for item in worst:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md(item.get("target")),
                        _md(item.get("recommended_runtime_backend")),
                        _md(item.get("miss_reason")),
                        f"{float(item.get('static_plan_vs_recommended_speedup') or 0.0):.2f}x",
                    ]
                )
                + " |"
            )
    else:
        lines.append("No planner misses.")

    warnings = feedback.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {_md(warning)}")
    return "\n".join(lines) + "\n"


def build_planner_calibration(feedback: dict[str, Any]) -> dict[str, Any]:
    targets = feedback.get("targets") or []
    threaded_scores: dict[str, float] = {}
    gpu_scores: dict[str, float] = {}
    backend_scores: dict[str, float] = {}
    hot_backend_scores: dict[str, float] = {}
    profitability_scores: dict[str, float] = {}
    profitability_penalty_scores: dict[str, float] = {}
    profitability_feature_scores: dict[str, float] = {}
    profitability_feature_penalty_scores: dict[str, float] = {}
    threaded_counts: dict[str, int] = {}
    gpu_counts: dict[str, int] = {}
    backend_counts: dict[str, int] = {}
    hot_backend_counts: dict[str, int] = {}
    profitability_counts: dict[str, int] = {}
    profitability_penalty_counts: dict[str, int] = {}
    profitability_feature_counts: dict[str, int] = {}
    profitability_feature_penalty_counts: dict[str, int] = {}
    for target in targets:
        speedup = float(target.get("static_plan_vs_recommended_speedup") or 1.0)
        weight = _planner_calibration_weight(target, speedup)
        hot_backend = str(target.get("hot_selected_backend") or "")
        backend = hot_backend or str(target.get("recommended_runtime_backend") or "")
        if backend:
            backend_scores[backend] = backend_scores.get(backend, 0.0) + weight
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
        if hot_backend:
            hot_backend_scores[hot_backend] = hot_backend_scores.get(hot_backend, 0.0) + weight
            hot_backend_counts[hot_backend] = hot_backend_counts.get(hot_backend, 0) + 1
        profitability_backend = _profitability_backend_from_target(target)
        if profitability_backend:
            profitability_scores[profitability_backend] = (
                profitability_scores.get(profitability_backend, 0.0) + weight
            )
            profitability_counts[profitability_backend] = (
                profitability_counts.get(profitability_backend, 0) + 1
            )
            for feature in target.get("profitability_feature_buckets") or []:
                signature = _profitability_feature_signature(
                    profitability_backend,
                    str(feature),
                )
                profitability_feature_scores[signature] = (
                    profitability_feature_scores.get(signature, 0.0) + weight
                )
                profitability_feature_counts[signature] = (
                    profitability_feature_counts.get(signature, 0) + 1
                )
        penalty = str(target.get("static_profitability_miss_reason") or "")
        selected_static = str(target.get("static_profitability_backend") or "")
        if (
            penalty
            and penalty not in {"none", "unknown"}
            and selected_static
            and not target.get("static_profitability_hit")
        ):
            penalty_weight = max(1.0, weight)
            penalty_keys = {penalty}
            penalty_keys.update(_profitability_backend_penalty_reasons(selected_static))
            for key in penalty_keys:
                profitability_penalty_scores[key] = (
                    profitability_penalty_scores.get(key, 0.0) + penalty_weight
                )
                profitability_penalty_counts[key] = (
                    profitability_penalty_counts.get(key, 0) + 1
                )
            for feature in target.get("profitability_feature_buckets") or []:
                signature = _profitability_feature_signature(selected_static, str(feature))
                profitability_feature_penalty_scores[signature] = (
                    profitability_feature_penalty_scores.get(signature, 0.0) + penalty_weight
                )
                profitability_feature_penalty_counts[signature] = (
                    profitability_feature_penalty_counts.get(signature, 0) + 1
                )
        threaded = target.get("selected_threaded_signature")
        if threaded:
            threaded_scores[str(threaded)] = threaded_scores.get(str(threaded), 0.0) + weight
            threaded_counts[str(threaded)] = threaded_counts.get(str(threaded), 0) + 1
        gpu = target.get("selected_gpu_option_signature")
        if gpu:
            gpu_scores[str(gpu)] = gpu_scores.get(str(gpu), 0.0) + weight
            gpu_counts[str(gpu)] = gpu_counts.get(str(gpu), 0) + 1
    return {
        "schema": "rrtl-pyrtl-planner-calibration-v1",
        "source_schema": feedback.get("schema"),
        "source_out_dir": feedback.get("out_dir"),
        "summary": {
            "profiles": (feedback.get("summary") or {}).get("profiles", 0),
            "backend_preferences": _rank_scores(backend_scores, backend_counts),
            "hot_backend_preferences": _rank_scores(hot_backend_scores, hot_backend_counts),
            "threaded_layout_preferences": _rank_scores(threaded_scores, threaded_counts),
            "gpu_option_preferences": _rank_scores(gpu_scores, gpu_counts),
            "profitability_backend_preferences": _rank_scores(
                profitability_scores,
                profitability_counts,
            ),
            "profitability_penalties": _rank_scores(
                profitability_penalty_scores,
                profitability_penalty_counts,
            ),
            "profitability_feature_preferences": _rank_scores(
                profitability_feature_scores,
                profitability_feature_counts,
            ),
            "profitability_feature_penalties": _rank_scores(
                profitability_feature_penalty_scores,
                profitability_feature_penalty_counts,
            ),
        },
    }


def _profitability_backend_from_target(target: dict[str, Any]) -> str:
    backend = str(target.get("hot_selected_backend") or target.get("recommended_runtime_backend") or "")
    if backend == "rrtl_threaded_trace" or backend == "rrtl_threaded_autotune_trace":
        return "threaded-mixed"
    if backend == "rrtl_gpu_trace" or backend == "rrtl_gpu_measured_trace":
        return "gpu-fused"
    if backend.startswith("rrtl_backend:"):
        direct = backend.split(":", 1)[1]
        if direct in {"scalar", "packed-cpu", "simd-cpu"}:
            return direct
    if backend == "rrtl_packed_trace":
        return "packed-cpu"
    if backend in {"rrtl_trace", "rrtl_single_trace"}:
        return "scalar"
    return ""


def _profitability_backend_penalty_reasons(backend: str) -> list[str]:
    if backend == "gpu-fused":
        return ["gpu-launch-not-amortized"]
    return []


def _planner_calibration_weight(target: dict[str, Any], speedup: float) -> float:
    base = speedup if speedup > 0.0 else 1.0
    if not target.get("hot_valid"):
        return base
    hot_speedup = float(target.get("hot_replay_speedup") or 0.0)
    setup_ratio = float(target.get("hot_setup_to_hot_ratio") or 0.0)
    hot_multiplier = max(1.5, min(4.0, hot_speedup if hot_speedup > 0.0 else 1.5))
    if setup_ratio >= 10.0:
        hot_multiplier = max(hot_multiplier, 2.0)
    return base * hot_multiplier


def _profitability_feature_signature(backend: str, feature: str) -> str:
    return f"{backend}|{feature}"


def render_planner_calibration_markdown(calibration: dict[str, Any]) -> str:
    summary = calibration.get("summary") or {}
    lines = [
        "# RRTL Planner Calibration",
        "",
        f"- Profiles: {summary.get('profiles', 0)}",
        "",
    ]
    for title, key in (
        ("Backend Preferences", "backend_preferences"),
        ("Hot Backend Preferences", "hot_backend_preferences"),
        ("Threaded Layout Preferences", "threaded_layout_preferences"),
        ("GPU Option Preferences", "gpu_option_preferences"),
        ("Profitability Backend Preferences", "profitability_backend_preferences"),
        ("Profitability Penalties", "profitability_penalties"),
        ("Profitability Feature Preferences", "profitability_feature_preferences"),
        ("Profitability Feature Penalties", "profitability_feature_penalties"),
    ):
        lines.extend([f"## {title}", ""])
        rows = summary.get(key) or []
        if rows:
            lines.extend(["| Signature | Score | Count |", "| --- | ---: | ---: |"])
            for row in rows:
                lines.append(
                    f"| {_md(row.get('signature'))} | {float(row.get('score') or 0.0):.2f} | {row.get('count', 0)} |"
                )
        else:
            lines.append("No preferences.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_planner_comparison(before_dir: Path, after_dir: Path) -> dict[str, Any]:
    before = _load_or_build_planner_feedback(before_dir)
    after = _load_or_build_planner_feedback(after_dir)
    before_summary = before.get("summary") or {}
    after_summary = after.get("summary") or {}
    before_targets = {
        str(target.get("target")): target
        for target in before.get("targets") or []
        if target.get("target")
    }
    after_targets = {
        str(target.get("target")): target
        for target in after.get("targets") or []
        if target.get("target")
    }
    shared_targets = sorted(set(before_targets) & set(after_targets))
    backend_shifts = []
    improved_targets = []
    regressed_targets = []
    for target in shared_targets:
        before_target = before_targets[target]
        after_target = after_targets[target]
        before_backend = before_target.get("recommended_runtime_backend")
        after_backend = after_target.get("recommended_runtime_backend")
        if before_backend != after_backend:
            backend_shifts.append(
                {
                    "target": target,
                    "before_backend": before_backend,
                    "after_backend": after_backend,
                    "before_source": before_target.get("recommended_runtime_source"),
                    "after_source": after_target.get("recommended_runtime_source"),
                    "after_profitability_features": (
                        after_target.get("profitability_feature_buckets") or []
                    )[:8],
                }
            )
        before_hit = bool(before_target.get("static_profitability_hit"))
        after_hit = bool(after_target.get("static_profitability_hit"))
        if not before_hit and after_hit:
            improved_targets.append(target)
        elif before_hit and not after_hit:
            regressed_targets.append(target)
    return {
        "schema": "rrtl-pyrtl-planner-comparison-v1",
        "before_out_dir": str(before_dir),
        "after_out_dir": str(after_dir),
        "summary": {
            "before_profiles": before_summary.get("profiles", 0),
            "after_profiles": after_summary.get("profiles", 0),
            "shared_targets": len(shared_targets),
            "plan_hit_rate_before": float(before_summary.get("plan_hit_rate") or 0.0),
            "plan_hit_rate_after": float(after_summary.get("plan_hit_rate") or 0.0),
            "plan_hit_rate_delta": float(after_summary.get("plan_hit_rate") or 0.0)
            - float(before_summary.get("plan_hit_rate") or 0.0),
            "static_profitability_hit_rate_before": float(
                before_summary.get("static_profitability_hit_rate") or 0.0
            ),
            "static_profitability_hit_rate_after": float(
                after_summary.get("static_profitability_hit_rate") or 0.0
            ),
            "static_profitability_hit_rate_delta": float(
                after_summary.get("static_profitability_hit_rate") or 0.0
            )
            - float(before_summary.get("static_profitability_hit_rate") or 0.0),
            "backend_shift_count": len(backend_shifts),
            "static_profitability_improvements": len(improved_targets),
            "static_profitability_regressions": len(regressed_targets),
            "before_recommended_backends": before_summary.get("recommended_backends") or {},
            "after_recommended_backends": after_summary.get("recommended_backends") or {},
            "before_miss_reasons": before_summary.get("miss_reasons") or {},
            "after_miss_reasons": after_summary.get("miss_reasons") or {},
            "remaining_worst_misses": after_summary.get("worst_misses") or [],
        },
        "backend_shifts": backend_shifts,
        "static_profitability_improved_targets": improved_targets,
        "static_profitability_regressed_targets": regressed_targets,
    }


def render_planner_comparison_markdown(comparison: dict[str, Any]) -> str:
    summary = comparison.get("summary") or {}
    lines = [
        "# RRTL Planner Comparison",
        "",
        f"- Before: {_md(comparison.get('before_out_dir'))}",
        f"- After: {_md(comparison.get('after_out_dir'))}",
        f"- Shared targets: {summary.get('shared_targets', 0)}",
        f"- Plan hit rate: {summary.get('plan_hit_rate_before', 0.0):.2%} -> {summary.get('plan_hit_rate_after', 0.0):.2%} ({summary.get('plan_hit_rate_delta', 0.0):+.2%})",
        f"- Static profitability hit rate: {summary.get('static_profitability_hit_rate_before', 0.0):.2%} -> {summary.get('static_profitability_hit_rate_after', 0.0):.2%} ({summary.get('static_profitability_hit_rate_delta', 0.0):+.2%})",
        f"- Backend shifts: {summary.get('backend_shift_count', 0)}",
        f"- Static profitability improvements: {summary.get('static_profitability_improvements', 0)}",
        f"- Static profitability regressions: {summary.get('static_profitability_regressions', 0)}",
        "",
        "## Backend Shifts",
        "",
    ]
    shifts = comparison.get("backend_shifts") or []
    if shifts:
        lines.extend(["| Target | Before | After | Features |", "| --- | --- | --- | --- |"])
        for shift in shifts:
            lines.append(
                f"| {_md(shift.get('target'))} | {_md(shift.get('before_backend'))} | {_md(shift.get('after_backend'))} | {_md(','.join(shift.get('after_profitability_features') or []))} |"
            )
    else:
        lines.append("No backend recommendation shifts.")

    lines.extend(["", "## Remaining Worst Misses", ""])
    worst = summary.get("remaining_worst_misses") or []
    if worst:
        lines.extend([
            "| Target | Recommended Backend | Miss Reason | Speedup |",
            "| --- | --- | --- | ---: |",
        ])
        for item in worst:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md(item.get("target")),
                        _md(item.get("recommended_runtime_backend")),
                        _md(item.get("miss_reason")),
                        f"{float(item.get('static_plan_vs_recommended_speedup') or 0.0):.2f}x",
                    ]
                )
                + " |"
            )
    else:
        lines.append("No remaining planner misses.")
    return "\n".join(lines) + "\n"


def _load_or_build_planner_feedback(out_dir: Path) -> dict[str, Any]:
    feedback_path = out_dir / "planner_feedback.json"
    if feedback_path.exists():
        return json.loads(feedback_path.read_text(encoding="utf-8"))
    return build_planner_feedback(out_dir)


def _feedback_target_name(out_dir: Path, profile_path: Path) -> str:
    if profile_path.name == "runtime_profile.json":
        return out_dir.name
    suffix = ".runtime_profile"
    stem = profile_path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def _is_hot_sweep_candidate_profile(path: Path) -> bool:
    name = path.name
    return ".hot_sweep." in name or name.startswith("hot_sweep.")


def _feedback_backend_plan_path(
    out_dir: Path,
    target_name: str,
    profile_path: Path,
) -> Path | None:
    candidates = [
        out_dir / f"{target_name}.backend_plan.json",
        profile_path.with_name("backend_plan.json"),
    ]
    return next((path for path in candidates if path.exists()), None)


def _feedback_hot_profile_replay_path(
    out_dir: Path,
    target_name: str,
    profile_path: Path,
) -> Path | None:
    candidates = [
        out_dir / f"{target_name}.hot_profile_replay.json",
        profile_path.with_name("hot_profile_replay.json"),
        profile_path.with_name(f"{target_name}.hot_profile_replay.json"),
    ]
    return next((path for path in candidates if path.exists()), None)


def _planner_feedback_target(
    target_name: str,
    profile_path: Path,
    plan_path: Path | None,
    profile: dict[str, Any],
    hot_path: Path | None = None,
    hot_replay: dict[str, Any] | None = None,
    backend_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    planner = profile.get("planner_feedback") or {}
    recommended_backend = profile.get("recommended_runtime_backend")
    source = profile.get("recommended_runtime_source") or "unknown"
    speedup = float(profile.get("static_plan_vs_recommended_speedup") or 0.0)
    static_plan_hit = bool(planner.get("plan_hit", source == "static-plan"))
    plan_hit = static_plan_hit and source == "static-plan"
    miss_reason = str(
        "none" if plan_hit else _effective_feedback_miss_reason(profile, planner)
    )
    gpu_option_hit = bool(planner.get("gpu_option_plan_hit", False))
    gpu_option_reason = str(planner.get("gpu_option_miss_reason") or "unknown")
    item = {
        "target": target_name,
        "profile": str(profile_path),
        "backend_plan": str(plan_path) if plan_path else None,
        "recommended_runtime_backend": recommended_backend,
        "recommended_runtime_source": source,
        "recommended_backend_reason": profile.get("recommended_backend_reason"),
        "static_plan_vs_recommended_speedup": speedup,
        "plan_hit": plan_hit,
        "miss_reason": miss_reason,
        "planned_cpu_rank": planner.get("planned_cpu_rank"),
        "planned_gpu_rank": planner.get("planned_gpu_rank"),
        "planned_gpu_selected": bool(planner.get("planned_gpu_selected", False)),
        "gpu_option_plan_hit": gpu_option_hit,
        "gpu_option_miss_reason": "none" if gpu_option_hit else gpu_option_reason,
        "static_profitability_backend": planner.get("static_profitability_backend"),
        "static_profitability_rank": planner.get("static_profitability_rank"),
        "static_profitability_hit": bool(planner.get("static_profitability_hit", False)),
        "static_profitability_miss_reason": planner.get(
            "static_profitability_miss_reason"
        )
        or "unknown",
        "selected_threaded_signature": _threaded_signature(profile),
        "selected_gpu_option_signature": _gpu_option_signature(profile),
        "profitability_features": _profitability_features_from_plan(backend_plan),
        "profitability_feature_buckets": _profitability_feature_buckets_from_plan(
            backend_plan
        ),
    }
    hot = _hot_replay_feedback(hot_replay)
    if hot_path is not None:
        item["hot_profile_replay"] = str(hot_path)
    if hot is not None:
        item.update(hot)
    return item


def _profitability_features_from_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    if not plan:
        return {}
    features = plan.get("profitability_features") or {}
    if features:
        return dict(features)
    profitability = plan.get("profitability") or {}
    op_profile = profitability.get("op_profile") or {}
    if not op_profile:
        return {}
    instr_count = int(op_profile.get("instr_count") or 0)
    memory_ops = int(op_profile.get("memory_ops") or 0)
    memory_ratio = int(memory_ops * 100 / instr_count) if instr_count else 0
    derived = {
        "lanes": int(plan.get("lanes") or 0),
        "steps": int(plan.get("steps") or 0),
        "lane_steps": int(plan.get("lanes") or 0) * int(plan.get("steps") or 0),
        "estimated_lane_work_units": int(
            op_profile.get("estimated_lane_work_units") or 0
        ),
        "instr_count": instr_count,
        "simd_coverage_score_x100": int(
            profitability.get("simd_coverage_score_x100") or 0
        ),
        "native_simd_score_x100": 0,
        "fallback_ratio_x100": 0,
        "memory_op_ratio_x100": memory_ratio,
        "pure_compute_packets": int(op_profile.get("pure_compute_packets") or 0),
        "memory_hostile_packets": int(op_profile.get("memory_hostile_packets") or 0),
        "wide_fallback_ops": int(op_profile.get("wide_fallback_ops") or 0),
        "gpu_suitability_score_x100": int(
            profitability.get("gpu_suitability_score_x100") or 0
        ),
        "threading_score_x100": int(profitability.get("threading_score_x100") or 0),
    }
    derived["feature_buckets"] = _profitability_feature_buckets(derived)
    return derived


def _profitability_feature_buckets_from_plan(plan: dict[str, Any] | None) -> list[str]:
    features = _profitability_features_from_plan(plan)
    return [str(feature) for feature in features.get("feature_buckets") or []]


def _profitability_feature_buckets(features: dict[str, Any]) -> list[str]:
    buckets: list[str] = []
    specs = (
        ("lanes", "lanes", (4, 16, 64, 256)),
        ("steps", "steps", (16, 64, 256, 1024)),
        ("lane_steps", "lane_steps", (256, 4096, 65_536, 1_048_576)),
        ("lane_work", "estimated_lane_work_units", (4096, 65_536, 1_048_576)),
        ("instr", "instr_count", (32, 128, 512, 2048)),
        ("simd_coverage", "simd_coverage_score_x100", (50, 70, 90)),
        ("native_simd", "native_simd_score_x100", (50, 70, 90)),
        ("fallback", "fallback_ratio_x100", (10, 25, 50)),
        ("memory_ops", "memory_op_ratio_x100", (10, 25, 50)),
        ("gpu_pure_packets", "pure_compute_packets", (4, 16, 64)),
        ("gpu_memory_hostile", "memory_hostile_packets", (1, 4, 16)),
        ("wide_fallback", "wide_fallback_ops", (1, 16, 64)),
        ("gpu_suitability", "gpu_suitability_score_x100", (50, 70, 90)),
        ("threading", "threading_score_x100", (50, 70, 90)),
    )
    for bucket_name, field, thresholds in specs:
        value = int(features.get(field) or 0)
        for threshold in thresholds:
            if value >= threshold:
                buckets.append(f"{bucket_name}>={threshold}")
    return buckets


def _hot_replay_feedback(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    valid = (
        report.get("schema") == "rrtl-pyrtl-profile-replay-hot-v1"
        and int(report.get("mismatch_count") or 0) == 0
        and int(report.get("replay_ns_best") or 0) > 0
    )
    return {
        "hot_valid": valid,
        "hot_selected_backend": report.get("selected_backend") if valid else None,
        "hot_selected_source": report.get("selected_source") if valid else None,
        "hot_repeat": int(report.get("repeat") or 0),
        "hot_first_replay_ns": int(report.get("first_replay_ns") or 0),
        "hot_replay_ns_best": int(report.get("replay_ns_best") or 0),
        "hot_replay_ns_median": int(report.get("replay_ns_median") or 0),
        "hot_replay_ns_best_excluding_setup": int(
            report.get("hot_replay_ns_best") or report.get("replay_ns_best") or 0
        ),
        "hot_replay_ns_median_excluding_setup": int(
            report.get("hot_replay_ns_median") or report.get("replay_ns_median") or 0
        ),
        "hot_replay_ns_per_step": float(report.get("replay_ns_per_step") or 0.0),
        "hot_replay_ns_per_lane_step": float(report.get("replay_ns_per_lane_step") or 0.0),
        "hot_setup_ns_total": int(report.get("setup_ns_total") or 0),
        "hot_setup_to_replay_ratio": float(report.get("setup_to_replay_ratio") or 0.0),
        "hot_setup_to_hot_ratio": float(
            report.get("setup_to_hot_ratio") or report.get("setup_to_replay_ratio") or 0.0
        ),
        "hot_replay_speedup": float(report.get("hot_replay_speedup") or 0.0),
        "hot_mismatch_count": int(report.get("mismatch_count") or 0),
    }


def _threaded_signature(profile: dict[str, Any]) -> str | None:
    selected = profile.get("selected_backend") or {}
    layout = selected.get("selected_threaded_layout") or {}
    workers = layout.get("workers") or []
    if not workers:
        threaded = profile.get("threaded_autotune") or {}
        layout = threaded.get("selected_threaded_layout") or {}
        workers = layout.get("workers") or []
    if not workers:
        return None
    return ",".join(f"{worker.get('backend')}:{int(worker.get('lanes') or 0)}" for worker in workers)


def _gpu_option_signature(profile: dict[str, Any]) -> str | None:
    selected = profile.get("selected_backend") or {}
    options = selected.get("selected_gpu_options") or {}
    if not options:
        gpu = profile.get("measured_gpu") or {}
        options = gpu.get("selected_gpu_options") or {}
    if not options:
        return None
    return (
        f"workgroup={int(options.get('workgroup_size') or 0)},"
        f"memory={options.get('memory_layout') or 'lane-major'},"
        f"reuse={str(bool(options.get('reuse_temporaries', False))).lower()}"
    )


def _rank_scores(scores: dict[str, float], counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"signature": signature, "score": score, "count": counts.get(signature, 0)}
        for signature, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def _fallback_miss_reason(profile: dict[str, Any]) -> str:
    if profile.get("recommended_runtime_source") == "static-plan":
        return "none"
    return str(profile.get("recommended_backend_reason") or "profile-missing-plan-feedback")


def _effective_feedback_miss_reason(
    profile: dict[str, Any],
    planner: dict[str, Any],
) -> str:
    planner_reason = str(planner.get("miss_reason") or "")
    if planner_reason and planner_reason != "none":
        return planner_reason
    return _fallback_miss_reason(profile)


def _planner_feedback_summary(
    targets: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    plan_hits = sum(1 for target in targets if target.get("plan_hit"))
    plan_misses = len(targets) - plan_hits
    gpu_option_hits = sum(1 for target in targets if target.get("gpu_option_plan_hit"))
    gpu_option_misses = sum(
        1
        for target in targets
        if target.get("gpu_option_miss_reason") not in (None, "", "none")
    )
    static_profitability_hits = sum(
        1 for target in targets if target.get("static_profitability_hit")
    )
    static_profitability_misses = len(targets) - static_profitability_hits
    speedups = [
        float(target.get("static_plan_vs_recommended_speedup") or 0.0)
        for target in targets
        if float(target.get("static_plan_vs_recommended_speedup") or 0.0) > 0.0
    ]
    hot_targets = [target for target in targets if target.get("hot_profile_replay")]
    valid_hot_targets = [target for target in hot_targets if target.get("hot_valid")]
    hot_lane_step = [
        float(target.get("hot_replay_ns_per_lane_step") or 0.0)
        for target in valid_hot_targets
        if float(target.get("hot_replay_ns_per_lane_step") or 0.0) > 0.0
    ]
    hot_setup_ratio = [
        float(target.get("hot_setup_to_hot_ratio") or target.get("hot_setup_to_replay_ratio") or 0.0)
        for target in valid_hot_targets
        if float(target.get("hot_setup_to_hot_ratio") or target.get("hot_setup_to_replay_ratio") or 0.0) > 0.0
    ]
    hot_replay_speedup = [
        float(target.get("hot_replay_speedup") or 0.0)
        for target in valid_hot_targets
        if float(target.get("hot_replay_speedup") or 0.0) > 0.0
    ]
    worst = sorted(
        (target for target in targets if not target.get("plan_hit")),
        key=lambda target: float(target.get("static_plan_vs_recommended_speedup") or 0.0),
        reverse=True,
    )[:10]
    return {
        "profiles": len(targets),
        "warnings": len(warnings),
        "plan_hits": plan_hits,
        "plan_misses": plan_misses,
        "plan_hit_rate": (plan_hits / len(targets)) if targets else 0.0,
        "miss_reasons": _count_by(targets, "miss_reason", skip={"none"}),
        "recommended_backends": _count_by(targets, "recommended_runtime_backend"),
        "recommended_sources": _count_by(targets, "recommended_runtime_source"),
        "gpu_option_hits": gpu_option_hits,
        "gpu_option_misses": gpu_option_misses,
        "gpu_option_miss_reasons": _count_by(
            targets,
            "gpu_option_miss_reason",
            skip={"none"},
        ),
        "static_profitability_hits": static_profitability_hits,
        "static_profitability_misses": static_profitability_misses,
        "static_profitability_hit_rate": (
            static_profitability_hits / len(targets)
        )
        if targets
        else 0.0,
        "static_profitability_backends": _count_by(targets, "static_profitability_backend"),
        "static_profitability_miss_reasons": _count_by(
            targets,
            "static_profitability_miss_reason",
            skip={"none"},
        ),
        "static_plan_vs_recommended_speedup": _speedup_distribution(speedups),
        "hot_profile_replay": {
            "profiles": len(hot_targets),
            "valid_profiles": len(valid_hot_targets),
            "selected_backends": _count_by(valid_hot_targets, "hot_selected_backend"),
            "selected_sources": _count_by(valid_hot_targets, "hot_selected_source"),
            "replay_ns_per_lane_step": _numeric_distribution(hot_lane_step),
            "setup_to_replay_ratio": _numeric_distribution(hot_setup_ratio),
            "setup_to_hot_ratio": _numeric_distribution(hot_setup_ratio),
            "hot_replay_speedup": _numeric_distribution(hot_replay_speedup),
        },
        "worst_misses": worst,
    }


def _count_by(
    items: list[dict[str, Any]],
    key: str,
    *,
    skip: set[str] | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    skip = skip or set()
    for item in items:
        value = item.get(key)
        if value is None:
            continue
        text = str(value)
        if not text or text in skip:
            continue
        counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items()))


def _speedup_distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "average": 0.0,
            "buckets": {},
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "median": median(ordered),
        "average": sum(ordered) / len(ordered),
        "buckets": {
            "lte_1_00x": sum(1 for value in ordered if value <= 1.0),
            "gt_1_00_lte_1_25x": sum(1 for value in ordered if 1.0 < value <= 1.25),
            "gt_1_25_lte_2_00x": sum(1 for value in ordered if 1.25 < value <= 2.0),
            "gt_2_00x": sum(1 for value in ordered if value > 2.0),
        },
    }


def _numeric_distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "average": 0.0,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "median": median(ordered),
        "average": sum(ordered) / len(ordered),
    }


def time_fast_simulation(
    block: pyrtl.Block,
    vectors: list[dict[str, int]],
    *,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    """Time PyRTL FastSimulation replay for the generated input vectors."""

    setup_start = time.perf_counter_ns()
    pyrtl.FastSimulation(tracer=None, block=block)
    setup_ns = time.perf_counter_ns() - setup_start

    for _ in range(warmup):
        sim = pyrtl.FastSimulation(tracer=None, block=block)
        _replay_fast(sim, block, vectors)

    samples = []
    checksums = []
    for _ in range(repeat):
        sim = pyrtl.FastSimulation(tracer=None, block=block)
        start = time.perf_counter_ns()
        checksum = _replay_fast(sim, block, vectors)
        samples.append(time.perf_counter_ns() - start)
        checksums.append(checksum)

    sorted_samples = sorted(samples)
    return {
        "schema": "rrtl-pyrtl-fast-bench-v1",
        "steps": len(vectors),
        "repeat": repeat,
        "warmup": warmup,
        "setup_ns": setup_ns,
        "run_ns_samples": samples,
        "run_ns_best": sorted_samples[0],
        "run_ns_median": int(median(sorted_samples)),
        "checksum": checksums[-1] if checksums else 0,
    }


def render_benchmark_markdown(report: dict[str, Any]) -> str:
    config = report["config"]
    fast = report["pyrtl_fast"]
    rrtl = report["rrtl_trace"]
    rrtl_packed = report["rrtl_packed_trace"]
    rrtl_single = report["rrtl_single_trace"]
    rrtl_backends = report.get("rrtl_backends", {}).get("backends", [])
    backend_plan = report.get("rrtl_backend_plan")
    plan_evaluation = report.get("backend_plan_evaluation")
    rrtl_threaded = report.get("rrtl_threaded_trace")
    rrtl_threaded_autotune = report.get("rrtl_threaded_autotune_trace")
    rrtl_gpu = report.get("rrtl_gpu_trace")
    rrtl_gpu_option_sweep = report.get("rrtl_gpu_option_sweep")
    rrtl_gpu_measured = report.get("rrtl_gpu_measured_trace")
    profile_replay_hot = report.get("profile_replay_hot")
    profile_replay_hot_sweep = report.get("profile_replay_hot_sweep")
    lines = [
        "# PyRTL-to-RRTL Synthetic Benchmark",
        "",
        f"- Array: {config['rows']}x{config['cols']}",
        f"- Steps: {config['steps']}",
        f"- Data width: {config['data_width']}",
        f"- Accumulator width: {config['acc_width']}",
        f"- Repeat: {config['repeat']}",
        f"- Warmup: {config['warmup']}",
        f"- Packed lanes: {config['packed_lanes']}",
        "",
        "## Timing",
        "",
        "| Engine | Best ns | Median ns |",
        "| --- | ---: | ---: |",
        f"| PyRTL FastSimulation | {fast['run_ns_best']} | {fast['run_ns_median']} |",
        f"| RRTL trace replay | {rrtl['replay_ns_best']} | {rrtl['replay_ns_median']} |",
        f"| RRTL packed trace replay | {rrtl_packed['replay_ns_best']} | {rrtl_packed['replay_ns_median']} |",
        f"| RRTL single-lane machine trace replay | {rrtl_single['replay_ns_best']} | {rrtl_single['replay_ns_median']} |",
    ]
    if rrtl_threaded:
        lines.append(
            f"| RRTL threaded independent-lane replay | {rrtl_threaded['replay_ns_best']} | {rrtl_threaded['replay_ns_median']} |"
        )
    if rrtl_threaded_autotune:
        lines.append(
            f"| RRTL threaded autotune replay | {rrtl_threaded_autotune['replay_ns_best']} | {rrtl_threaded_autotune['replay_ns_median']} |"
        )
    for backend in rrtl_backends:
        label = f"RRTL backend {backend['backend']}"
        if not backend.get("available", True):
            label += " (unavailable)"
        lines.append(
            f"| {label} | {backend['replay_ns_best']} | {backend['replay_ns_median']} |"
        )
    backend_detail_rows = []
    for backend in rrtl_backends:
        timing = backend.get("replay_timing") or {}
        simd = backend.get("simd_stats") or {}
        if timing or simd:
            backend_detail_rows.append((backend, timing, simd))
    if backend_detail_rows:
        lines.extend([
            "",
            "## Backend Details",
            "",
            "| Backend | Input ns | Eval ns | Compare ns | Tick ns | SIMD fast | SIMD fallback | Lane materializations | 1-limb fast | 2-limb fast | Native 2-limb | 2-limb mul | Mem read fast | Mux fast | Mem write effects | Sext fb | Mem fb | Wide fb | Signed lt fb | Wide concat fb |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for backend, timing, simd in backend_detail_rows:
            reasons = simd.get("fallback_reasons") or {}
            fast_paths = simd.get("fast_paths") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        backend["backend"],
                        str(timing.get("input_ns", 0)),
                        str(timing.get("eval_ns", 0)),
                        str(timing.get("compare_ns", 0)),
                        str(timing.get("tick_ns", 0)),
                        str(simd.get("fast_instrs", 0)),
                        str(simd.get("fallback_instrs", 0)),
                        str(simd.get("lane_materializations", 0)),
                        str(fast_paths.get("one_limb_ops", 0)),
                        str(fast_paths.get("two_limb_ops", 0)),
                        str(fast_paths.get("native_two_limb_ops", 0)),
                        str(fast_paths.get("two_limb_mul_ops", 0)),
                        str(fast_paths.get("two_limb_memory_reads", 0)),
                        str(fast_paths.get("two_limb_mux_ops", 0)),
                        str(fast_paths.get("memory_write_effects", 0)),
                        str(reasons.get("sext", 0)),
                        str(reasons.get("mem_read", 0)),
                        str(reasons.get("wide_op", 0)),
                        str(reasons.get("signed_lt", 0)),
                        str(reasons.get("wide_concat", 0)),
                    ]
                )
                + " |"
            )
    suitability = report.get("rrtl_backends", {}).get("simd_suitability")
    if not suitability and rrtl_threaded:
        suitability = rrtl_threaded.get("simd_suitability")
    if suitability:
        total = suitability.get("total") or {}
        reasons = total.get("fallback_reasons") or {}
        fast_profile = total.get("fast_path_profile") or {}
        lines.extend([
            "",
            "## SIMD Suitability",
            "",
            f"- Recommendation: {suitability.get('recommendation', 'unknown')}",
            f"- Score x100: {suitability.get('score_x100', 0)}",
            f"- Fallback ratio x100: {suitability.get('fallback_ratio_x100', 0)}",
            f"- Estimated fast cost: {suitability.get('estimated_fast_cost', 0)}",
            f"- Estimated fallback cost: {suitability.get('estimated_fallback_cost', 0)}",
            f"- Estimated materialization cost: {suitability.get('estimated_materialization_cost', 0)}",
            f"- Predicted fast instrs: {total.get('fast_instrs', 0)}",
            f"- Predicted fallback instrs: {total.get('fallback_instrs', 0)}",
            f"- Predicted lane materializations per lane: {total.get('lane_materializations_per_lane', 0)}",
            f"- Predicted fast paths: one_limb={fast_profile.get('one_limb_ops', 0)}, two_limb={fast_profile.get('two_limb_ops', 0)}, two_limb_mul={fast_profile.get('two_limb_mul_ops', 0)}, mem_read={fast_profile.get('two_limb_memory_reads', 0)}, mux={fast_profile.get('two_limb_mux_ops', 0)}, mem_write={fast_profile.get('memory_write_effects', 0)}",
            f"- Predicted fallback reasons: sext={reasons.get('sext', 0)}, mem_read={reasons.get('mem_read', 0)}, wide_op={reasons.get('wide_op', 0)}, signed_lt={reasons.get('signed_lt', 0)}, wide_concat={reasons.get('wide_concat', 0)}",
        ])
    if backend_plan:
        selected = backend_plan.get("selected_threaded_layout") or {}
        workers = ",".join(
            f"{worker.get('backend', 'unknown')}:{worker.get('lanes', 0)}"
            for worker in selected.get("workers", [])
        )
        workload = backend_plan.get("replay_workload") or {}
        selected_gpu = backend_plan.get("selected_gpu_options") or {}
        profitability = backend_plan.get("profitability") or {}
        op_profile = profitability.get("op_profile") or {}
        candidates = backend_plan.get("backend_candidates") or []
        top_candidate = candidates[0] if candidates else {}
        lines.extend([
            "",
            "## Backend Plan",
            "",
            f"- Selected workers: {workers}",
            f"- Selected reason: {backend_plan.get('selected_reason', 'unknown')}",
            f"- Pruned candidates: {len(backend_plan.get('pruned_threaded_layouts', []))}",
            f"- SIMD recommendation: {(backend_plan.get('simd_suitability') or {}).get('recommendation', 'unknown')}",
            f"- Backend affinity recommendation: {(backend_plan.get('backend_affinity') or {}).get('recommendation', 'unknown')}",
            f"- GPU recommendation: {(backend_plan.get('gpu_region_analysis') or {}).get('recommendation', 'unknown')}",
            f"- Selected GPU reason: {backend_plan.get('selected_gpu_reason') or 'none'}",
            f"- Selected GPU workgroup size: {selected_gpu.get('workgroup_size', 0)}",
            f"- Selected GPU memory layout: {selected_gpu.get('memory_layout', 'none')}",
            f"- Selected GPU reusable temporaries: {selected_gpu.get('reuse_temporaries', False)}",
            f"- Recommended GPU options: {len(backend_plan.get('recommended_gpu_options', []))}",
            f"- Pruned GPU options: {len(backend_plan.get('pruned_gpu_options', []))}",
            f"- Estimated lane work units: {workload.get('estimated_lane_work_units', 0)}",
            f"- Static profitability backend: {profitability.get('recommended_backend', 'unknown')}",
            f"- Static profitability reason: {profitability.get('recommended_reason', 'unknown')}",
            f"- Selected runtime backend: {backend_plan.get('selected_runtime_backend') or 'unknown'}",
            f"- Selected runtime reason: {backend_plan.get('selected_runtime_reason') or 'unknown'}",
            f"- Pruned runtime candidates: {len(backend_plan.get('pruned_runtime_candidates', []))}",
            f"- SIMD coverage x100: {profitability.get('simd_coverage_score_x100', 0)}",
            f"- GPU suitability x100: {profitability.get('gpu_suitability_score_x100', 0)}",
            f"- Threading score x100: {profitability.get('threading_score_x100', 0)}",
            f"- Static top candidate: {top_candidate.get('backend', 'unknown')} score={top_candidate.get('score', 0)}",
            f"- Static op profile: one_limb={op_profile.get('one_limb_ops', 0)}, two_limb={op_profile.get('two_limb_ops', 0)}, native_two_limb={op_profile.get('native_two_limb_ops', 0)}, memory={op_profile.get('memory_ops', 0)}, wide_fallback={op_profile.get('wide_fallback_ops', 0)}",
        ])
    if plan_evaluation:
        fastest = plan_evaluation.get("fastest_rrtl_available") or {}
        fastest_overall = plan_evaluation.get("fastest_overall_backend") or {}
        lines.extend([
            "",
            "## Plan Evaluation",
            "",
            f"- Fastest measured RRTL backend: {fastest.get('name', 'none')}",
            f"- Fastest measured best ns: {fastest.get('best_ns', 0)}",
            f"- Fastest overall backend: {fastest_overall.get('name', 'none')}",
            f"- Recommended runtime backend: {plan_evaluation.get('recommended_runtime_backend') or 'none'}",
            f"- Recommended runtime source: {plan_evaluation.get('recommended_runtime_source', 'unknown')}",
            f"- Static plan vs recommended speedup: {plan_evaluation.get('static_plan_vs_recommended_speedup', 0.0):.2f}x",
            f"- Recommended reason: {plan_evaluation.get('recommended_backend_reason', 'unknown')}",
            f"- Planned CPU rank: {plan_evaluation.get('planned_cpu_rank') or 'none'}",
            f"- Planned GPU rank: {plan_evaluation.get('planned_gpu_rank') or 'none'}",
            f"- Plan hit: {plan_evaluation.get('plan_hit', False)}",
            f"- Miss reason: {plan_evaluation.get('miss_reason', 'unknown')}",
            f"- Static profitability backend: {plan_evaluation.get('static_profitability_backend') or 'none'}",
            f"- Static profitability rank: {plan_evaluation.get('static_profitability_rank') or 'none'}",
            f"- Static profitability hit: {plan_evaluation.get('static_profitability_hit', False)}",
            f"- Static profitability miss reason: {plan_evaluation.get('static_profitability_miss_reason', 'unknown')}",
            f"- Planned GPU option rank: {plan_evaluation.get('planned_gpu_option_rank') or 'none'}",
            f"- Measured best GPU option: {plan_evaluation.get('measured_gpu_option_best_index') if plan_evaluation.get('measured_gpu_option_best_index') is not None else 'none'}",
            f"- GPU option plan hit: {plan_evaluation.get('gpu_option_plan_hit', False)}",
            f"- GPU option miss reason: {plan_evaluation.get('gpu_option_miss_reason', 'unknown')}",
        ])
    runtime_profile = report.get("runtime_profile") or {}
    if runtime_profile:
        selected = runtime_profile.get("selected_backend") or {}
        threaded = runtime_profile.get("threaded_autotune") or {}
        gpu = runtime_profile.get("measured_gpu") or {}
        lines.extend([
            "",
            "## Runtime Profile",
            "",
            f"- Recommended backend: {runtime_profile.get('recommended_runtime_backend') or 'none'}",
            f"- Source: {runtime_profile.get('recommended_runtime_source') or 'unknown'}",
            f"- Static plan speedup: {runtime_profile.get('static_plan_vs_recommended_speedup', 0.0):.2f}x",
            f"- Reason: {runtime_profile.get('recommended_backend_reason') or 'unknown'}",
            f"- Selected best ns: {selected.get('best_ns', 0)}",
        ])
        static_profitability = runtime_profile.get("static_backend_profitability") or {}
        if static_profitability:
            lines.append(
                f"- Static profitability backend: {static_profitability.get('recommended_backend', 'unknown')}"
            )
        if threaded:
            selected_layout = threaded.get("selected_threaded_layout") or {}
            workers = ",".join(
                f"{worker.get('backend', 'unknown')}:{worker.get('lanes', 0)}"
                for worker in selected_layout.get("workers", [])
            )
            lines.append(f"- Profile threaded workers: {workers}")
        if gpu:
            options = gpu.get("selected_gpu_options") or {}
            lines.append(
                f"- Profile GPU option: workgroup={options.get('workgroup_size', 0)}, memory={options.get('memory_layout', 'none')}, reuse={options.get('reuse_temporaries', False)}"
            )
    if profile_replay_hot:
        lines.extend([
            "",
            "## Hot Profile Replay",
            "",
            f"- Selected backend: {profile_replay_hot.get('selected_backend', 'none')}",
            f"- Selected source: {profile_replay_hot.get('selected_source', 'unknown')}",
            f"- Repeat: {profile_replay_hot.get('repeat', 0)}",
            f"- First replay ns: {profile_replay_hot.get('first_replay_ns', 0)}",
            f"- Best ns: {profile_replay_hot.get('replay_ns_best', 0)}",
            f"- Median ns: {profile_replay_hot.get('replay_ns_median', 0)}",
            f"- Setup ns total: {profile_replay_hot.get('setup_ns_total', 0)}",
            f"- Setup to hot ratio: {profile_replay_hot.get('setup_to_hot_ratio', profile_replay_hot.get('setup_to_replay_ratio', 0.0)):.2f}x",
            f"- Hot replay speedup: {profile_replay_hot.get('hot_replay_speedup', 0.0):.2f}x",
            f"- Ns per lane-step: {profile_replay_hot.get('replay_ns_per_lane_step', 0.0):.2f}",
            f"- Mismatches: {profile_replay_hot.get('mismatch_count', 0)}",
        ])
    if profile_replay_hot_sweep:
        lines.extend([
            "",
            "## Hot Profile Sweep",
            "",
            f"- Candidates: {profile_replay_hot_sweep.get('candidate_count', 0)}",
            f"- Valid candidates: {profile_replay_hot_sweep.get('valid_candidate_count', 0)}",
            f"- Selected candidate: {profile_replay_hot_sweep.get('selected_candidate_name') or 'none'}",
            f"- Selected backend: {profile_replay_hot_sweep.get('selected_backend') or 'none'}",
        ])
    if rrtl_gpu:
        gpu_timing = rrtl_gpu.get("gpu_timing") or {}
        lines.extend([
            "",
            "## GPU Replay",
            "",
            f"- Available: {rrtl_gpu.get('available', False)}",
            f"- Error: {rrtl_gpu.get('error') or ''}",
            f"- Replay mode: {rrtl_gpu.get('gpu_replay_mode', 'unknown')}",
            f"- Prepared runner setup ns: {rrtl_gpu.get('prepared_runner_setup_ns', 0)}",
            f"- Prepared snapshot setup ns: {rrtl_gpu.get('prepared_snapshot_setup_ns', 0)}",
            f"- Prepared trace bytes: {rrtl_gpu.get('prepared_trace_bytes', 0)}",
            f"- Prepared trace uncompressed bytes: {rrtl_gpu.get('prepared_trace_uncompressed_bytes', 0)}",
            f"- Prepared trace compression ratio x100: {rrtl_gpu.get('prepared_trace_compression_ratio_x100', 100)}",
            f"- Prepared trace uniform input ops: {rrtl_gpu.get('prepared_trace_uniform_input_ops', 0)}",
            f"- Prepared trace uniform check ops: {rrtl_gpu.get('prepared_trace_uniform_check_ops', 0)}",
            f"- Prepared trace layout: {rrtl_gpu.get('prepared_trace_layout', '')}",
            f"- Prepared trace template input ops: {rrtl_gpu.get('prepared_trace_template_input_ops', 0)}",
            f"- Prepared trace template check ops: {rrtl_gpu.get('prepared_trace_template_check_ops', 0)}",
            f"- Prepared trace metadata saved words: {rrtl_gpu.get('prepared_trace_metadata_saved_words', 0)}",
            f"- Prepared trace fixed template: {rrtl_gpu.get('prepared_trace_fixed_template', False)}",
            f"- Prepared trace value metadata saved words: {rrtl_gpu.get('prepared_trace_value_metadata_saved_words', 0)}",
            f"- Prepared trace value stride words: {rrtl_gpu.get('prepared_trace_value_stride_words', 0)}",
            f"- Hot restore best ns: {rrtl_gpu.get('hot_restore_ns_best', 0)}",
            f"- Hot restore median ns: {rrtl_gpu.get('hot_restore_ns_median', 0)}",
            f"- Hot GPU replay best ns: {rrtl_gpu.get('hot_gpu_replay_ns_best', rrtl_gpu.get('replay_ns_best', 0))}",
            f"- Hot GPU replay median ns: {rrtl_gpu.get('hot_gpu_replay_ns_median', rrtl_gpu.get('replay_ns_median', 0))}",
            f"- GPU single-submit profitable: {rrtl_gpu.get('gpu_single_submit_profitable', False)}",
            f"- GPU planner calibration reason: {rrtl_gpu.get('gpu_planner_calibration_reason') or ''}",
            f"- Count readback ns: {gpu_timing.get('count_readback_ns', 0)}",
            f"- Full readback ns: {gpu_timing.get('full_readback_ns', 0)}",
            f"- Full readback words: {gpu_timing.get('full_readback_words', 0)}",
            f"- Single submit used: {gpu_timing.get('single_submit_used', False)}",
            f"- Single submit ns: {gpu_timing.get('single_submit_ns', 0)}",
            f"- Single submit count readback ns: {gpu_timing.get('single_submit_count_readback_ns', 0)}",
            f"- Best ns: {rrtl_gpu.get('replay_ns_best', 0)}",
            f"- Median ns: {rrtl_gpu.get('replay_ns_median', 0)}",
            f"- Mismatches: {rrtl_gpu.get('mismatch_count', 0)}",
        ])
    if rrtl_gpu_option_sweep:
        selected_index = rrtl_gpu_option_sweep.get("selected_candidate_index")
        candidates = rrtl_gpu_option_sweep.get("candidates") or []
        selected = (
            candidates[selected_index]
            if isinstance(selected_index, int) and selected_index < len(candidates)
            else {}
        )
        selected_plan = selected.get("planned") or {}
        selected_options = selected_plan.get("options") or {}
        lines.extend([
            "",
            "## GPU Option Sweep",
            "",
            f"- Candidates: {len(candidates)}",
            f"- Selected candidate: {selected_index if selected_index is not None else 'none'}",
            f"- Selected reason: {selected_plan.get('reason', 'none')}",
            f"- Selected workgroup size: {selected_options.get('workgroup_size', 0)}",
            f"- Selected memory layout: {selected_options.get('memory_layout', 'none')}",
            f"- Selected reusable temporaries: {selected_options.get('reuse_temporaries', False)}",
            f"- Selected best ns: {selected.get('replay_ns_best', 0)}",
        ])
    if rrtl_gpu_measured:
        gpu_timing = rrtl_gpu_measured.get("gpu_timing") or {}
        static_options = ((backend_plan or {}).get("selected_gpu_options") or {})
        measured_options = rrtl_gpu_measured.get("selected_gpu_options") or {}
        differs = bool(
            measured_options
            and static_options
            and not _gpu_options_match(static_options, measured_options)
        )
        lines.extend([
            "",
            "## GPU Measured Replay",
            "",
            f"- Available: {rrtl_gpu_measured.get('available', False)}",
            f"- Error: {rrtl_gpu_measured.get('error') or ''}",
            f"- Prepared runner setup ns: {rrtl_gpu_measured.get('prepared_runner_setup_ns', 0)}",
            f"- Prepared snapshot setup ns: {rrtl_gpu_measured.get('prepared_snapshot_setup_ns', 0)}",
            f"- Prepared trace bytes: {rrtl_gpu_measured.get('prepared_trace_bytes', 0)}",
            f"- Prepared trace uncompressed bytes: {rrtl_gpu_measured.get('prepared_trace_uncompressed_bytes', 0)}",
            f"- Prepared trace compression ratio x100: {rrtl_gpu_measured.get('prepared_trace_compression_ratio_x100', 100)}",
            f"- Prepared trace uniform input ops: {rrtl_gpu_measured.get('prepared_trace_uniform_input_ops', 0)}",
            f"- Prepared trace uniform check ops: {rrtl_gpu_measured.get('prepared_trace_uniform_check_ops', 0)}",
            f"- Prepared trace layout: {rrtl_gpu_measured.get('prepared_trace_layout', '')}",
            f"- Prepared trace template input ops: {rrtl_gpu_measured.get('prepared_trace_template_input_ops', 0)}",
            f"- Prepared trace template check ops: {rrtl_gpu_measured.get('prepared_trace_template_check_ops', 0)}",
            f"- Prepared trace metadata saved words: {rrtl_gpu_measured.get('prepared_trace_metadata_saved_words', 0)}",
            f"- Prepared trace fixed template: {rrtl_gpu_measured.get('prepared_trace_fixed_template', False)}",
            f"- Prepared trace value metadata saved words: {rrtl_gpu_measured.get('prepared_trace_value_metadata_saved_words', 0)}",
            f"- Prepared trace value stride words: {rrtl_gpu_measured.get('prepared_trace_value_stride_words', 0)}",
            f"- Hot restore best ns: {rrtl_gpu_measured.get('hot_restore_ns_best', 0)}",
            f"- Hot restore median ns: {rrtl_gpu_measured.get('hot_restore_ns_median', 0)}",
            f"- Hot GPU replay best ns: {rrtl_gpu_measured.get('hot_gpu_replay_ns_best', rrtl_gpu_measured.get('replay_ns_best', 0))}",
            f"- Hot GPU replay median ns: {rrtl_gpu_measured.get('hot_gpu_replay_ns_median', rrtl_gpu_measured.get('replay_ns_median', 0))}",
            f"- GPU single-submit profitable: {rrtl_gpu_measured.get('gpu_single_submit_profitable', False)}",
            f"- GPU planner calibration reason: {rrtl_gpu_measured.get('gpu_planner_calibration_reason') or ''}",
            f"- Count readback ns: {gpu_timing.get('count_readback_ns', 0)}",
            f"- Full readback ns: {gpu_timing.get('full_readback_ns', 0)}",
            f"- Full readback words: {gpu_timing.get('full_readback_words', 0)}",
            f"- Single submit used: {gpu_timing.get('single_submit_used', False)}",
            f"- Single submit ns: {gpu_timing.get('single_submit_ns', 0)}",
            f"- Single submit count readback ns: {gpu_timing.get('single_submit_count_readback_ns', 0)}",
            f"- Best ns: {rrtl_gpu_measured.get('replay_ns_best', 0)}",
            f"- Median ns: {rrtl_gpu_measured.get('replay_ns_median', 0)}",
            f"- Differs from static GPU option: {differs}",
        ])
    if rrtl_threaded_autotune:
        selected = rrtl_threaded_autotune.get("selected_threaded_layout") or {}
        workers = ",".join(
            f"{worker.get('backend', 'unknown')}:{worker.get('lanes', 0)}"
            for worker in selected.get("workers", [])
        )
        autotune = rrtl_threaded_autotune.get("autotune") or {}
        pruned = rrtl_threaded_autotune.get("autotune_pruned_candidates") or autotune.get(
            "pruned_candidates", []
        )
        lines.extend([
            "",
            "## Threaded Autotune Replay",
            "",
            f"- Selected workers: {workers}",
            f"- Selected reason: {rrtl_threaded_autotune.get('selected_reason', 'unknown')}",
            f"- Best ns: {rrtl_threaded_autotune.get('replay_ns_best', 0)}",
            f"- Median ns: {rrtl_threaded_autotune.get('replay_ns_median', 0)}",
            f"- Autotune selected candidate: {autotune.get('selected_candidate') if autotune.get('selected_candidate') is not None else 'none'}",
            f"- Autotune candidates: {len(autotune.get('candidates', []))}",
            f"- Autotune pruned candidates: {len(pruned)}",
            f"- Mismatches: {rrtl_threaded_autotune.get('mismatch_count', 0)}",
        ])
    if rrtl_threaded and rrtl_threaded.get("autotune"):
        autotune = rrtl_threaded["autotune"]
        pruned = rrtl_threaded.get("autotune_pruned_candidates") or autotune.get(
            "pruned_candidates", []
        )
        lines.extend([
            "",
            "## Autotune Pruning",
            "",
            f"- Kept candidates: {len(autotune.get('candidates', []))}",
            f"- Pruned candidates: {len(pruned)}",
        ])
        for candidate in pruned:
            workers = ",".join(
                f"{worker['backend']}:{worker['lanes']}"
                for worker in candidate.get("layout", {}).get("workers", [])
            )
            lines.append(
                f"- Pruned candidate {candidate.get('candidate_index', 0)} ({workers}): {candidate.get('reason', 'unknown')}"
            )
    if rrtl_threaded and rrtl_threaded.get("replay_workload"):
        workload = rrtl_threaded["replay_workload"]
        lines.extend([
            "",
            "## Replay Workload",
            "",
            f"- Steps: {workload.get('steps', 0)}",
            f"- Lanes: {workload.get('lanes', 0)}",
            f"- Input ops: {workload.get('input_ops', 0)}",
            f"- Check ops: {workload.get('check_ops', 0)}",
            f"- One-limb input batches: {workload.get('one_limb_input_batches', 0)}",
            f"- One-limb check batches: {workload.get('one_limb_check_batches', 0)}",
            f"- Generic input ops: {workload.get('generic_input_ops', 0)}",
            f"- Generic check ops: {workload.get('generic_check_ops', 0)}",
            f"- Estimated lane work units: {workload.get('estimated_lane_work_units', 0)}",
        ])
    lines.extend([
        "",
        f"- Scalar best speedup: {report['speedup_best']:.2f}x",
        f"- Scalar median speedup: {report['speedup_median']:.2f}x",
        f"- Packed best speedup: {report['packed_speedup_best']:.2f}x",
        f"- Packed median speedup: {report['packed_speedup_median']:.2f}x",
        f"- Single-lane machine best speedup: {report['single_speedup_best']:.2f}x",
        f"- Single-lane machine median speedup: {report['single_speedup_median']:.2f}x",
        f"- Threaded independent-lane best speedup: {report.get('threaded_speedup_best', 0.0):.2f}x",
        f"- Threaded independent-lane median speedup: {report.get('threaded_speedup_median', 0.0):.2f}x",
        f"- Threaded autotune best speedup: {report.get('threaded_autotune_speedup_best', 0.0):.2f}x",
        f"- Threaded autotune median speedup: {report.get('threaded_autotune_speedup_median', 0.0):.2f}x",
        f"- GPU best speedup: {report.get('gpu_speedup_best', 0.0):.2f}x",
        f"- GPU median speedup: {report.get('gpu_speedup_median', 0.0):.2f}x",
        f"- GPU measured best speedup: {report.get('gpu_measured_speedup_best', 0.0):.2f}x",
        f"- GPU measured median speedup: {report.get('gpu_measured_speedup_median', 0.0):.2f}x",
        f"- RRTL scalar mismatches: {rrtl['mismatch_count']}",
        f"- RRTL packed mismatches: {rrtl_packed['mismatch_count']}",
        f"- RRTL single-lane machine mismatches: {rrtl_single['mismatch_count']}",
    ])
    if rrtl_threaded:
        lines.append(
            f"- RRTL threaded independent-lane mismatches: {rrtl_threaded['mismatch_count']}"
        )
    if rrtl_threaded_autotune:
        lines.append(
            f"- RRTL threaded autotune mismatches: {rrtl_threaded_autotune['mismatch_count']}"
        )
    if rrtl_gpu:
        lines.append(f"- RRTL GPU mismatches: {rrtl_gpu['mismatch_count']}")
    for backend in rrtl_backends:
        lines.append(
            f"- RRTL backend {backend['backend']} mismatches: {backend['mismatch_count']}"
        )
        if not backend.get("available", True):
            lines.append(
                f"- RRTL backend {backend['backend']} unavailable: {backend.get('error', '')}"
            )
    lines.append("")
    return "\n".join(lines)


def evaluate_backend_plan(
    *,
    rrtl: dict[str, Any],
    rrtl_packed: dict[str, Any],
    rrtl_single: dict[str, Any],
    rrtl_backends: dict[str, Any],
    rrtl_threaded: dict[str, Any] | None,
    rrtl_gpu: dict[str, Any] | None,
    backend_plan: dict[str, Any] | None,
    rrtl_threaded_autotune: dict[str, Any] | None = None,
    rrtl_gpu_measured: dict[str, Any] | None = None,
    rrtl_gpu_option_sweep: dict[str, Any] | None = None,
) -> dict[str, Any]:
    measured = [
        _measured_rrtl_backend("rrtl_trace", "RRTL trace replay", rrtl),
        _measured_rrtl_backend(
            "rrtl_packed_trace",
            "RRTL packed trace replay",
            rrtl_packed,
        ),
        _measured_rrtl_backend(
            "rrtl_single_trace",
            "RRTL single-lane machine trace replay",
            rrtl_single,
        ),
    ]
    for backend in (rrtl_backends.get("backends") or []):
        measured.append(
            _measured_rrtl_backend(
                f"rrtl_backend:{backend.get('backend', 'unknown')}",
                f"RRTL backend {backend.get('backend', 'unknown')}",
                backend,
            )
        )
    if rrtl_threaded:
        measured.append(
            _measured_rrtl_backend(
                "rrtl_threaded_trace",
                "RRTL threaded independent-lane replay",
                rrtl_threaded,
            )
        )
    if rrtl_gpu:
        measured.append(
            _measured_rrtl_backend(
                "rrtl_gpu_trace",
                "RRTL GPU replay",
                rrtl_gpu,
            )
        )

    plan_ranked = sorted(
        (backend for backend in measured if backend["valid"]),
        key=lambda backend: (backend["best_ns"], backend["median_ns"], backend["name"]),
    )
    if rrtl_threaded_autotune:
        measured.append(
            _measured_rrtl_backend(
                "rrtl_threaded_autotune_trace",
                "RRTL threaded autotune replay",
                rrtl_threaded_autotune,
            )
        )
    if rrtl_gpu_measured:
        measured.append(
            _measured_rrtl_backend(
                "rrtl_gpu_measured_trace",
                "RRTL GPU measured replay",
                rrtl_gpu_measured,
            )
        )
    ranked = sorted(
        (backend for backend in measured if backend["valid"]),
        key=lambda backend: (backend["best_ns"], backend["median_ns"], backend["name"]),
    )
    ranks = {backend["name"]: rank for rank, backend in enumerate(ranked, start=1)}
    fastest = plan_ranked[0] if plan_ranked else None
    fastest_overall = ranked[0] if ranked else None
    planned_gpu_selected = bool((backend_plan or {}).get("selected_gpu_options"))
    planned_cpu_rank = ranks.get("rrtl_threaded_trace")
    planned_gpu_rank = ranks.get("rrtl_gpu_trace")
    plan_hit = bool(
        fastest
        and fastest["name"] in {"rrtl_threaded_trace", "rrtl_gpu_trace"}
        and (
            fastest["name"] == "rrtl_threaded_trace"
            or (fastest["name"] == "rrtl_gpu_trace" and planned_gpu_selected)
        )
    )
    miss_reason = "none"
    if not fastest:
        miss_reason = "no-valid-measurements"
    elif not plan_hit:
        gpu_valid = bool(
            rrtl_gpu and _measured_rrtl_backend("rrtl_gpu_trace", "", rrtl_gpu)["valid"]
        )
        if planned_gpu_selected and rrtl_gpu and not gpu_valid:
            miss_reason = "gpu-unavailable"
        elif planned_gpu_selected and planned_gpu_rank and planned_gpu_rank > 1:
            miss_reason = "planned-gpu-slower"
        else:
            miss_reason = "planned-cpu-slower"
    gpu_option_feedback = _evaluate_gpu_option_plan(
        (backend_plan or {}).get("selected_gpu_options"),
        rrtl_gpu_option_sweep,
    )
    profitability_feedback = _evaluate_static_profitability_plan(
        backend_plan or {},
        fastest_overall,
        ranks,
    )
    recommendation = _recommend_runtime_backend(ranked, fastest_overall)

    return {
        "schema": "rrtl-pyrtl-backend-plan-evaluation-v1",
        "measured_backends": measured,
        "fastest_rrtl_available": fastest,
        "fastest_overall_backend": fastest_overall,
        **recommendation,
        "planned_cpu_backend": _planned_workers_summary(
            ((backend_plan or {}).get("selected_threaded_layout") or {}).get(
                "workers", []
            )
        ),
        "planned_gpu_selected": planned_gpu_selected,
        "planned_cpu_rank": planned_cpu_rank,
        "planned_gpu_rank": planned_gpu_rank,
        "plan_hit": plan_hit,
        "miss_reason": miss_reason,
        **profitability_feedback,
        **gpu_option_feedback,
    }


def _evaluate_static_profitability_plan(
    backend_plan: dict[str, Any],
    fastest_overall: dict[str, Any] | None,
    ranks: dict[str, int],
) -> dict[str, Any]:
    selected = str(
        backend_plan.get("selected_runtime_backend")
        or (backend_plan.get("profitability") or {}).get("recommended_backend")
        or ""
    )
    measured_name = _static_profitability_measured_name(selected)
    rank = ranks.get(measured_name) if measured_name else None
    hit = bool(fastest_overall and measured_name and fastest_overall.get("name") == measured_name)
    if not selected:
        reason = "static-profitability-missing"
    elif measured_name is None:
        reason = "static-profitability-unknown-backend"
    elif rank is None:
        reason = "static-profitability-unmeasured"
    elif hit:
        reason = "none"
    else:
        reason = "static-profitability-slower"
    return {
        "static_profitability_backend": selected or None,
        "static_profitability_measured_backend": measured_name,
        "static_profitability_rank": rank,
        "static_profitability_hit": hit,
        "static_profitability_miss_reason": reason,
    }


def _static_profitability_measured_name(backend: str) -> str | None:
    return {
        "scalar": "rrtl_backend:scalar",
        "packed-cpu": "rrtl_backend:packed-cpu",
        "simd-cpu": "rrtl_backend:simd-cpu",
        "threaded-mixed": "rrtl_threaded_trace",
        "gpu-fused": "rrtl_gpu_trace",
    }.get(backend)


def _recommend_runtime_backend(
    ranked: list[dict[str, Any]],
    fastest_overall: dict[str, Any] | None,
) -> dict[str, Any]:
    static_plan_names = {"rrtl_threaded_trace", "rrtl_gpu_trace"}
    best_static_plan = next(
        (backend for backend in ranked if backend["name"] in static_plan_names),
        None,
    )
    if not fastest_overall:
        return {
            "recommended_runtime_backend": None,
            "recommended_runtime_source": "no-valid-measurements",
            "static_plan_vs_recommended_speedup": 0.0,
            "recommended_backend_reason": "no-valid-measurements",
        }

    name = fastest_overall["name"]
    if name in static_plan_names:
        source = "static-plan"
        reason = "static-plan-fastest"
    elif name == "rrtl_gpu_measured_trace":
        source = "measured-gpu"
        reason = (
            "measured-gpu-beats-static-plan"
            if best_static_plan
            else "measured-gpu-only-valid"
        )
    else:
        source = "measured-cpu"
        reason = (
            "measured-cpu-beats-static-plan"
            if best_static_plan
            else "measured-cpu-only-valid"
        )

    speedup = 0.0
    if best_static_plan:
        speedup = _speedup(best_static_plan["best_ns"], fastest_overall["best_ns"])
    return {
        "recommended_runtime_backend": name,
        "recommended_runtime_source": source,
        "static_plan_vs_recommended_speedup": speedup,
        "recommended_backend_reason": reason,
    }


def _measured_rrtl_backend(
    name: str,
    label: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    best_ns = int(report.get("replay_ns_best") or 0)
    median_ns = int(report.get("replay_ns_median") or 0)
    mismatch_count = int(report.get("mismatch_count") or 0)
    available = bool(report.get("available", True))
    valid = available and mismatch_count == 0 and best_ns > 0
    invalid_reason = "none"
    if not available:
        invalid_reason = "unavailable"
    elif mismatch_count != 0:
        invalid_reason = "mismatch"
    elif best_ns <= 0:
        invalid_reason = "invalid-timing"
    return {
        "name": name,
        "label": label,
        "available": available,
        "valid": valid,
        "best_ns": best_ns,
        "median_ns": median_ns,
        "mismatch_count": mismatch_count,
        "invalid_reason": invalid_reason,
        "error": report.get("error") or "",
    }


def _evaluate_gpu_option_plan(
    planned_gpu_options: dict[str, Any] | None,
    sweep: dict[str, Any] | None,
) -> dict[str, Any]:
    feedback = {
        "planned_gpu_option_rank": None,
        "planned_gpu_option_best_ns": None,
        "measured_gpu_option_best_index": None,
        "measured_gpu_option_best_ns": None,
        "gpu_option_plan_hit": False,
        "gpu_option_miss_reason": "no-sweep",
    }
    if not planned_gpu_options:
        feedback["gpu_option_miss_reason"] = "gpu-not-planned"
        return feedback
    candidates = (sweep or {}).get("candidates") or []
    if not candidates:
        feedback["gpu_option_miss_reason"] = "no-valid-gpu-options"
        return feedback

    ranked = sorted(
        (
            candidate
            for candidate in candidates
            if _valid_gpu_option_candidate(candidate)
        ),
        key=lambda candidate: (
            int(candidate.get("replay_ns_best") or 0),
            int(candidate.get("replay_ns_median") or 0),
            int(candidate.get("candidate_index") or 0),
        ),
    )
    if not ranked:
        feedback["gpu_option_miss_reason"] = "no-valid-gpu-options"
        return feedback

    ranks = {
        int(candidate.get("candidate_index") or 0): rank
        for rank, candidate in enumerate(ranked, start=1)
    }
    best = ranked[0]
    best_index = int(best.get("candidate_index") or 0)
    feedback["measured_gpu_option_best_index"] = best_index
    feedback["measured_gpu_option_best_ns"] = int(best.get("replay_ns_best") or 0)

    planned_candidate = next(
        (
            candidate
            for candidate in candidates
            if _gpu_options_match(
                planned_gpu_options,
                ((candidate.get("planned") or {}).get("options") or {}),
            )
        ),
        None,
    )
    if not planned_candidate:
        feedback["gpu_option_miss_reason"] = "planned-option-not-in-sweep"
        return feedback

    planned_index = int(planned_candidate.get("candidate_index") or 0)
    feedback["planned_gpu_option_best_ns"] = int(
        planned_candidate.get("replay_ns_best") or 0
    )
    if not _valid_gpu_option_candidate(planned_candidate):
        feedback["gpu_option_miss_reason"] = "planned-option-invalid"
        return feedback

    feedback["planned_gpu_option_rank"] = ranks.get(planned_index)
    feedback["gpu_option_plan_hit"] = planned_index == best_index
    feedback["gpu_option_miss_reason"] = (
        "none" if planned_index == best_index else "planned-gpu-option-slower"
    )
    return feedback


def _valid_gpu_option_candidate(candidate: dict[str, Any]) -> bool:
    return (
        bool(candidate.get("available", True))
        and int(candidate.get("mismatch_count") or 0) == 0
        and int(candidate.get("replay_ns_best") or 0) > 0
    )


def _gpu_options_match(lhs: dict[str, Any], rhs: dict[str, Any]) -> bool:
    return (
        int(lhs.get("workgroup_size") or 0) == int(rhs.get("workgroup_size") or 0)
        and lhs.get("memory_layout") == rhs.get("memory_layout")
        and bool(lhs.get("reuse_temporaries", False))
        == bool(rhs.get("reuse_temporaries", False))
    )


def _planned_workers_summary(workers: list[dict[str, Any]]) -> str:
    if not workers:
        return ""
    return ",".join(
        f"{worker.get('backend', 'unknown')}:{worker.get('lanes', 0)}" for worker in workers
    )


def _replay_fast(
    sim: pyrtl.FastSimulation,
    block: pyrtl.Block,
    vectors: list[dict[str, int]],
) -> int:
    outputs = sorted(w.name for w in block.wirevector_subset(wire.Output))
    checksum = 0
    mask = (1 << 128) - 1
    for vector in vectors:
        sim.step(dict(vector))
        for name in outputs:
            checksum = ((checksum * 1_315_423_911) ^ int(sim.inspect(name))) & mask
    return checksum


def _run_bench_trace(
    pyrtl2rrtl: list[str],
    export_path: Path,
    trace_path: Path,
    *,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    cmd = [
        *pyrtl2rrtl,
        "bench-trace",
        str(export_path),
        str(trace_path),
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"pyrtl2rrtl bench-trace failed: {detail}")
    return json.loads(result.stdout)


def _run_bench_packed_trace(
    pyrtl2rrtl: list[str],
    export_path: Path,
    trace_path: Path,
    *,
    repeat: int,
    warmup: int,
    lanes: int,
) -> dict[str, Any]:
    cmd = [
        *pyrtl2rrtl,
        "bench-packed-trace",
        str(export_path),
        str(trace_path),
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
        "--lanes",
        str(lanes),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"pyrtl2rrtl bench-packed-trace failed: {detail}")
    return json.loads(result.stdout)


def _run_bench_single_trace(
    pyrtl2rrtl: list[str],
    export_path: Path,
    trace_path: Path,
    *,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    cmd = [
        *pyrtl2rrtl,
        "bench-single-trace",
        str(export_path),
        str(trace_path),
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"pyrtl2rrtl bench-single-trace failed: {detail}")
    return json.loads(result.stdout)


def _run_bench_backends(
    pyrtl2rrtl: list[str],
    export_path: Path,
    trace_path: Path,
    *,
    repeat: int,
    warmup: int,
    lanes: int,
    backends: str,
) -> dict[str, Any]:
    cmd = [
        *pyrtl2rrtl,
        "bench-backends",
        str(export_path),
        str(trace_path),
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
        "--lanes",
        str(lanes),
        "--backend",
        backends,
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"pyrtl2rrtl bench-backends failed: {detail}")
    return json.loads(result.stdout)


def _run_plan_backends(
    pyrtl2rrtl: list[str],
    export_path: Path,
    lane_trace_path: Path,
) -> dict[str, Any]:
    cmd = [
        *pyrtl2rrtl,
        "plan-backends",
        str(export_path),
        str(lane_trace_path),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"pyrtl2rrtl plan-backends failed: {detail}")
    return json.loads(result.stdout)


def _run_bench_threaded_trace(
    pyrtl2rrtl: list[str],
    export_path: Path,
    lane_trace_path: Path,
    *,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    cmd = [
        *pyrtl2rrtl,
        "bench-threaded-trace",
        str(export_path),
        str(lane_trace_path),
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
        "--plan-first",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"pyrtl2rrtl bench-threaded-trace failed: {detail}")
    return json.loads(result.stdout)


def _run_bench_threaded_autotune_trace(
    pyrtl2rrtl: list[str],
    export_path: Path,
    lane_trace_path: Path,
    *,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    cmd = [
        *pyrtl2rrtl,
        "bench-threaded-trace",
        str(export_path),
        str(lane_trace_path),
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
        "--autotune",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"pyrtl2rrtl bench-threaded-trace --autotune failed: {detail}")
    return json.loads(result.stdout)


def _run_bench_threaded_trace_with_workers(
    pyrtl2rrtl: list[str],
    export_path: Path,
    lane_trace_path: Path,
    workers: list[dict[str, Any]],
    *,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    cmd = [
        *pyrtl2rrtl,
        "bench-threaded-trace",
        str(export_path),
        str(lane_trace_path),
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
        "--no-autotune",
    ]
    for worker in workers:
        backend = worker.get("backend")
        lanes = int(worker.get("lanes") or 0)
        if not backend or lanes <= 0:
            raise ValueError("runtime profile worker requires backend and positive lanes")
        cmd.extend(["--worker", f"{backend}:{lanes}"])
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"pyrtl2rrtl bench-threaded-trace workers failed: {detail}")
    return json.loads(result.stdout)


def _run_bench_gpu_trace(
    pyrtl2rrtl: list[str],
    export_path: Path,
    lane_trace_path: Path,
    *,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    cmd = [
        *pyrtl2rrtl,
        "bench-gpu-trace",
        str(export_path),
        str(lane_trace_path),
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
        "--plan-first",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"pyrtl2rrtl bench-gpu-trace failed: {detail}")
    return json.loads(result.stdout)


def _run_bench_gpu_combined(
    pyrtl2rrtl: list[str],
    export_path: Path,
    lane_trace_path: Path,
    *,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    cmd = [
        *pyrtl2rrtl,
        "bench-gpu-combined",
        str(export_path),
        str(lane_trace_path),
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"pyrtl2rrtl bench-gpu-combined failed: {detail}")
    return json.loads(result.stdout)


def _annotate_measured_gpu_trace(
    report: dict[str, Any],
    sweep: dict[str, Any],
) -> dict[str, Any]:
    selected_index = sweep.get("selected_candidate_index")
    candidates = sweep.get("candidates") or []
    if isinstance(selected_index, int) and selected_index < len(candidates):
        options = (candidates[selected_index].get("planned") or {}).get("options") or {}
        report = dict(report)
        report["selected_gpu_option_index"] = selected_index
        report["selected_gpu_options"] = options
    return report


def _run_bench_gpu_measured_trace(
    pyrtl2rrtl: list[str],
    export_path: Path,
    lane_trace_path: Path,
    sweep: dict[str, Any],
    *,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    selected_index = sweep.get("selected_candidate_index")
    candidates = sweep.get("candidates") or []
    if not isinstance(selected_index, int) or selected_index >= len(candidates):
        return _empty_measured_gpu_trace(sweep, "gpu-measured-option-not-selected")
    selected = candidates[selected_index]
    if not _valid_gpu_option_candidate(selected):
        return _empty_measured_gpu_trace(sweep, "gpu-measured-option-invalid")
    options = (selected.get("planned") or {}).get("options") or {}
    report = _run_bench_gpu_trace_with_options(
        pyrtl2rrtl,
        export_path,
        lane_trace_path,
        options,
        repeat=repeat,
        warmup=warmup,
    )
    report["selected_gpu_option_index"] = selected_index
    report["selected_gpu_options"] = options
    return report


def _empty_measured_gpu_trace(sweep: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "schema": "rrtl-pyrtl-bench-gpu-trace-v1",
        "steps": sweep.get("steps", 0),
        "lanes": sweep.get("lanes", 0),
        "repeat": sweep.get("repeat", 0),
        "warmup": sweep.get("warmup", 0),
        "import_ns": 0,
        "setup_ns": 0,
        "available": False,
        "error": error,
        "backend_affinity": {},
        "gpu_region_analysis": {},
        "shader_stats": {},
        "gpu_replay_mode": "fused-kernel",
        "gpu_timing": {},
        "replay_ns_samples": [],
        "replay_ns_best": 0,
        "replay_ns_median": 0,
        "mismatch_count": 0,
        "mismatches": [],
    }


def _run_bench_gpu_trace_with_options(
    pyrtl2rrtl: list[str],
    export_path: Path,
    lane_trace_path: Path,
    options: dict[str, Any],
    *,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    cmd = [
        *pyrtl2rrtl,
        "bench-gpu-trace",
        str(export_path),
        str(lane_trace_path),
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
        "--workgroup-size",
        str(options.get("workgroup_size", 128)),
        "--memory-layout",
        str(options.get("memory_layout", "lane-major")),
    ]
    if options.get("reuse_temporaries", False):
        cmd.append("--reuse-temporaries")
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"pyrtl2rrtl bench-gpu-trace failed: {detail}")
    return json.loads(result.stdout)


def _run_bench_gpu_options(
    pyrtl2rrtl: list[str],
    export_path: Path,
    lane_trace_path: Path,
    *,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    cmd = [
        *pyrtl2rrtl,
        "bench-gpu-options",
        str(export_path),
        str(lane_trace_path),
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"pyrtl2rrtl bench-gpu-options failed: {detail}")
    return json.loads(result.stdout)


def _fit_signed(value, width: int):
    if len(value) < width:
        return value.sign_extended(width)
    if len(value) > width:
        return value.truncate(width)
    return value


def _speedup(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _md(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a synthetic PyRTL-to-RRTL benchmark")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--data-width", type=int, default=8)
    parser.add_argument("--acc-width", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--packed-lanes", type=int, default=1)
    parser.add_argument(
        "--profile-replay-hot-repeat",
        type=int,
        default=0,
        help="also run the selected runtime profile in the native hot replay loop",
    )
    parser.add_argument(
        "--no-hot-profile-select",
        action="store_true",
        help="record the hot profile sweep without replacing runtime_profile.json",
    )
    parser.add_argument(
        "--pyrtl2rrtl",
        default="cargo run -q -p rrtl-pyrtl --bin pyrtl2rrtl --",
        help="command used to invoke the Rust bridge CLI",
    )
    parser.add_argument(
        "--runtime-profile",
        help="runtime_profile.json used for profile-selected replay",
    )
    parser.add_argument(
        "--profile-replay-only",
        action="store_true",
        help="run only the backend selected by --runtime-profile",
    )
    parser.add_argument(
        "--planner-feedback-only",
        action="store_true",
        help="scan --out-dir for runtime profiles and write planner feedback reports",
    )
    parser.add_argument(
        "--planner-calibration-only",
        action="store_true",
        help="scan --out-dir for planner feedback and write planner calibration reports",
    )
    parser.add_argument(
        "--planner-comparison-only",
        action="store_true",
        help="compare planner feedback from --before-dir and --after-dir",
    )
    parser.add_argument("--before-dir", help="baseline corpus/benchmark output directory")
    parser.add_argument("--after-dir", help="calibrated corpus/benchmark output directory")
    args = parser.parse_args(argv)

    try:
        if args.planner_comparison_only:
            if not args.before_dir or not args.after_dir:
                raise ValueError("--planner-comparison-only requires --before-dir and --after-dir")
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            comparison = build_planner_comparison(
                Path(args.before_dir),
                Path(args.after_dir),
            )
            json_path = out_dir / "planner_comparison.json"
            md_path = out_dir / "planner_comparison.md"
            json_path.write_text(
                json.dumps(comparison, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            md_path.write_text(
                render_planner_comparison_markdown(comparison),
                encoding="utf-8",
            )
            print(f"planner_comparison_json: {json_path}")
            print(f"planner_comparison_md: {md_path}")
            print(f"planner_comparison_shared_targets: {comparison['summary']['shared_targets']}")
            return 0
        if args.planner_feedback_only:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            feedback = build_planner_feedback(out_dir)
            json_path = out_dir / "planner_feedback.json"
            md_path = out_dir / "planner_feedback.md"
            json_path.write_text(
                json.dumps(feedback, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            md_path.write_text(render_planner_feedback_markdown(feedback), encoding="utf-8")
            print(f"planner_feedback_json: {json_path}")
            print(f"planner_feedback_md: {md_path}")
            print(f"planner_feedback_profiles: {feedback['summary']['profiles']}")
            return 0
        if args.planner_calibration_only:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            feedback_path = out_dir / "planner_feedback.json"
            if feedback_path.exists():
                feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
            else:
                feedback = build_planner_feedback(out_dir)
            calibration = build_planner_calibration(feedback)
            json_path = out_dir / "planner_calibration.json"
            md_path = out_dir / "planner_calibration.md"
            json_path.write_text(
                json.dumps(calibration, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            md_path.write_text(
                render_planner_calibration_markdown(calibration),
                encoding="utf-8",
            )
            print(f"planner_calibration_json: {json_path}")
            print(f"planner_calibration_md: {md_path}")
            print(f"planner_calibration_profiles: {calibration['summary']['profiles']}")
            return 0
        if args.profile_replay_only:
            if not args.runtime_profile:
                raise ValueError("--profile-replay-only requires --runtime-profile")
            replay_report = run_profile_selected_replay(
                out_dir=Path(args.out_dir),
                runtime_profile_path=Path(args.runtime_profile),
                rows=args.rows,
                cols=args.cols,
                steps=args.steps,
                data_width=args.data_width,
                acc_width=args.acc_width,
                repeat=args.repeat,
                warmup=args.warmup,
                packed_lanes=args.packed_lanes,
                pyrtl2rrtl=shlex.split(args.pyrtl2rrtl),
            )
            print(f"profile_replay: {replay_report['outputs']['profile_replay']}")
            print(f"selected_backend: {replay_report['selected_backend']}")
            replay = replay_report.get("replay") or {}
            if "replay_ns_best" in replay:
                print(f"profile_replay_best_ns: {replay.get('replay_ns_best', 0)}")
                print(f"profile_replay_median_ns: {replay.get('replay_ns_median', 0)}")
            return 0
        report = run_benchmark(
            out_dir=Path(args.out_dir),
            rows=args.rows,
            cols=args.cols,
            steps=args.steps,
            data_width=args.data_width,
            acc_width=args.acc_width,
            repeat=args.repeat,
            warmup=args.warmup,
            packed_lanes=args.packed_lanes,
            profile_replay_hot_repeat=args.profile_replay_hot_repeat,
            hot_profile_select=not args.no_hot_profile_select,
            pyrtl2rrtl=shlex.split(args.pyrtl2rrtl),
        )
    except Exception as err:  # noqa: BLE001 - CLI should surface benchmark failures cleanly.
        print(f"rrtl_pyrtl.bench: {err}")
        return 1

    print(f"bench_json: {report['outputs']['bench_json']}")
    print(f"bench_md: {report['outputs']['bench_md']}")
    print(f"best_speedup: {report['speedup_best']:.2f}x")
    print(f"median_speedup: {report['speedup_median']:.2f}x")
    print(f"packed_best_speedup: {report['packed_speedup_best']:.2f}x")
    print(f"packed_median_speedup: {report['packed_speedup_median']:.2f}x")
    print(f"single_best_speedup: {report['single_speedup_best']:.2f}x")
    print(f"single_median_speedup: {report['single_speedup_median']:.2f}x")
    print(f"threaded_best_speedup: {report['threaded_speedup_best']:.2f}x")
    print(f"threaded_median_speedup: {report['threaded_speedup_median']:.2f}x")
    print(f"threaded_autotune_best_speedup: {report['threaded_autotune_speedup_best']:.2f}x")
    print(f"threaded_autotune_median_speedup: {report['threaded_autotune_speedup_median']:.2f}x")
    print(f"gpu_best_speedup: {report['gpu_speedup_best']:.2f}x")
    print(f"gpu_median_speedup: {report['gpu_speedup_median']:.2f}x")
    print(f"gpu_measured_best_speedup: {report['gpu_measured_speedup_best']:.2f}x")
    print(f"gpu_measured_median_speedup: {report['gpu_measured_speedup_median']:.2f}x")
    if "hot_profile_replay" in report["outputs"]:
        print(f"hot_profile_replay: {report['outputs']['hot_profile_replay']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
