"""Synthetic GEMM surrogate corpus generation, ONNX export, and validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRANSACTION_SCHEMA = "rrtl-surrogate-gemm-transaction-v1"
MANIFEST_SCHEMA = "rrtl-surrogate-manifest-v1"


@dataclass(frozen=True)
class GemmShape:
    rows: int
    cols: int
    k: int
    data_width: int = 8
    acc_width: int = 32


def generate_corpus(
    *,
    rows: int,
    cols: int,
    k: int,
    samples: int,
    seed: int,
    data_width: int = 8,
    acc_width: int = 32,
    lanes: int = 1,
) -> list[dict[str, Any]]:
    """Generate deterministic signed integer GEMM transactions."""

    _validate_positive("rows", rows)
    _validate_positive("cols", cols)
    _validate_positive("k", k)
    _validate_positive("samples", samples)
    _validate_positive("lanes", lanes)
    rng = random.Random(seed)
    shape = GemmShape(rows=rows, cols=cols, k=k, data_width=data_width, acc_width=acc_width)
    low = -(1 << (data_width - 1))
    high = (1 << (data_width - 1)) - 1
    transactions = []
    for index in range(samples):
        transaction = build_transaction(
            shape,
            [[rng.randint(low, high) for _ in range(k)] for _ in range(rows)],
            [[rng.randint(low, high) for _ in range(cols)] for _ in range(k)],
        )
        if lanes > 1:
            transaction["lane"] = index % lanes
        transactions.append(transaction)
    return transactions


def build_transaction(
    shape: GemmShape,
    a: list[list[int]],
    w: list[list[int]],
) -> dict[str, Any]:
    c = gemm(a, w, shape.rows, shape.cols, shape.k)
    return {
        "schema": TRANSACTION_SCHEMA,
        "rows": shape.rows,
        "cols": shape.cols,
        "k": shape.k,
        "a": a,
        "w": w,
        "expected_c": c,
        "expected_latency_cycles": latency_cycles(shape.rows, shape.cols, shape.k),
    }


def gemm(
    a: list[list[int]],
    w: list[list[int]],
    rows: int,
    cols: int,
    k: int,
) -> list[list[int]]:
    return [
        [sum(int(a[row][kk]) * int(w[kk][col]) for kk in range(k)) for col in range(cols)]
        for row in range(rows)
    ]


def latency_cycles(rows: int, cols: int, k: int) -> int:
    return k + rows + cols - 1


def train_and_export(
    *,
    corpus_path: Path,
    out_dir: Path,
    epochs: int,
    seed: int,
    tolerance: int = 0,
) -> dict[str, Any]:
    """Export a deterministic PyTorch GEMM surrogate and return a summary."""

    _validate_nonnegative("epochs", epochs)
    corpus = read_corpus(corpus_path)
    shape = infer_shape(corpus)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.onnx"
    manifest_path = out_dir / "manifest.json"
    heldout_path = out_dir / "heldout.json"
    summary_path = out_dir / "summary.json"

    export_onnx_model(shape, model_path, seed=seed)
    model_hash = sha256_file(model_path)
    heldout = corpus[: min(len(corpus), 8)]
    heldout_path.write_text(json.dumps(heldout, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = build_manifest(
        shape,
        model_path=manifest_relative_path(model_path=model_path, manifest_path=manifest_path),
        model_hash=model_hash,
        source_hash=sha256_json(corpus),
        tolerance=tolerance,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": "rrtl-surrogate-gemm-training-summary-v1",
        "samples": len(corpus),
        "heldout": len(heldout),
        "epochs": epochs,
        "seed": seed,
        "shape": {
            "rows": shape.rows,
            "cols": shape.cols,
            "k": shape.k,
            "data_width": shape.data_width,
            "acc_width": shape.acc_width,
        },
        "outputs": {
            "model": str(model_path),
            "manifest": str(manifest_path),
            "heldout": str(heldout_path),
        },
        "artifact_sha256": model_hash,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def export_onnx_model(shape: GemmShape, model_path: Path, *, seed: int) -> None:
    try:
        import torch
    except Exception as err:  # noqa: BLE001 - provide a clean CLI error.
        raise RuntimeError("PyTorch is required for GEMM surrogate export") from err

    try:
        import onnx  # noqa: F401
    except Exception as err:  # noqa: BLE001 - torch's legacy exporter requires onnx.
        raise RuntimeError(
            "Python package `onnx` is required for export; install it in a venv "
            "and rerun `python -m rrtl_surrogate.train_gemm train ...`"
        ) from err

    torch.manual_seed(seed)

    class GemmSurrogate(torch.nn.Module):
        def forward(self, gemm_descriptor, a_tensor, w_tensor):  # type: ignore[no-untyped-def]
            c_tensor = torch.matmul(a_tensor, w_tensor)
            latency = gemm_descriptor[2] + gemm_descriptor[0] + gemm_descriptor[1] - 1.0
            active = gemm_descriptor[2]
            utilization = active / latency
            telemetry = torch.stack([latency, active, utilization])
            return c_tensor, telemetry

    descriptor = torch.tensor(
        [
            float(shape.rows),
            float(shape.cols),
            float(shape.k),
            float(shape.data_width),
            float(shape.acc_width),
            0.0,
        ],
        dtype=torch.float32,
    )
    a_tensor = torch.zeros((shape.rows, shape.k), dtype=torch.float32)
    w_tensor = torch.zeros((shape.k, shape.cols), dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.filterwarnings(
            "ignore",
            message="You are using the legacy TorchScript-based ONNX export",
            category=DeprecationWarning,
        )
        torch.onnx.export(
            GemmSurrogate(),
            (descriptor, a_tensor, w_tensor),
            model_path,
            input_names=["gemm_descriptor", "a_tensor", "w_tensor"],
            output_names=["c_tensor", "telemetry"],
            opset_version=17,
            dynamo=False,
        )


def build_manifest(
    shape: GemmShape,
    *,
    model_path: Path,
    model_hash: str,
    source_hash: str,
    tolerance: int,
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "surrogate_id": f"synthetic_gemm_{shape.rows}x{shape.cols}x{shape.k}",
        "surrogate_class": "transaction_kernel",
        "model_family": "gnn-transformer",
        "source": {
            "top_name": "SyntheticGemmTransaction",
            "export_schema": "rrtl-surrogate-gemm-corpus-v1",
            "source_hash": source_hash,
        },
        "artifact": {
            "format": "onnx",
            "path": str(model_path),
            "sha256": model_hash,
            "input_tensors": ["gemm_descriptor", "a_tensor", "w_tensor"],
            "output_tensors": ["c_tensor", "telemetry"],
            "opset": 17,
        },
        "domain": {
            "rows": shape.rows,
            "cols": shape.cols,
            "k_min": shape.k,
            "k_max": shape.k,
            "data_width": shape.data_width,
            "acc_width": shape.acc_width,
        },
        "validation": {
            "max_abs_error": tolerance,
            "max_mean_abs_error": float(tolerance),
            "max_latency_error_cycles": 0,
        },
        "policy": {
            "mode": "approximate_with_tolerance",
            "fallback": "fail_closed",
            "provenance_tag": "approximate",
        },
    }


def validate_with_rrtl(
    *,
    manifest_path: Path,
    transactions_path: Path,
    pyrtl2rrtl: list[str],
    out_path: Path | None = None,
) -> dict[str, Any]:
    transactions = read_corpus(transactions_path)
    batch = {
        "schema": "rrtl-surrogate-gemm-batch-v1",
        "source_hash": sha256_json(transactions),
        "transactions": transactions,
    }
    temp_path = transactions_path.parent / ".rrtl_surrogate_gemm_batch.json"
    temp_path.write_text(
        json.dumps(batch, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        completed = subprocess.run(
            [
                *pyrtl2rrtl,
                "surrogate",
                "run-gemm-batch",
                str(manifest_path),
                str(temp_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if completed.stdout:
        summary = json.loads(completed.stdout)
    else:
        summary = {
            "schema": "rrtl-surrogate-gemm-batch-result-v1",
            "ok": False,
            "count": len(transactions),
            "total_lanes": 0,
            "metrics": {
                "max_abs_error": 0,
                "max_mean_abs_error": 0.0,
                "max_latency_error_cycles": 0,
            },
            "lanes": [],
            "results": [],
        }
    if completed.returncode != 0:
        summary["ok"] = False
        if completed.stderr:
            summary.setdefault("errors", []).append(completed.stderr.strip())
    if out_path is not None:
        out_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def policy_with_rrtl(
    *,
    manifest_path: Path,
    transactions_path: Path,
    pyrtl2rrtl: list[str],
    out_path: Path | None = None,
) -> dict[str, Any]:
    transactions = read_corpus(transactions_path)
    batch = {
        "schema": "rrtl-surrogate-gemm-batch-v1",
        "source_hash": sha256_json(transactions),
        "transactions": transactions,
    }
    temp_path = transactions_path.parent / ".rrtl_surrogate_gemm_policy_batch.json"
    temp_path.write_text(
        json.dumps(batch, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        completed = subprocess.run(
            [
                *pyrtl2rrtl,
                "surrogate",
                "policy-gemm-batch",
                str(manifest_path),
                str(temp_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if completed.stdout:
        summary = json.loads(completed.stdout)
    else:
        summary = {
            "schema": "rrtl-surrogate-gemm-policy-report-v1",
            "ok": False,
            "count": len(transactions),
            "used_surrogate": 0,
            "exact_fallbacks": 0,
            "fail_closed": 0,
            "results": [],
        }
    if completed.returncode != 0:
        summary["ok"] = False
        if completed.stderr:
            summary.setdefault("errors", []).append(completed.stderr.strip())
    if out_path is not None:
        out_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def read_corpus(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("transactions"), list):
        data = data["transactions"]
    if not isinstance(data, list):
        raise ValueError("corpus must be a list or an object with a transactions list")
    return data


def infer_shape(corpus: list[dict[str, Any]]) -> GemmShape:
    if not corpus:
        raise ValueError("corpus must not be empty")
    first = corpus[0]
    shape = GemmShape(
        rows=int(first["rows"]),
        cols=int(first["cols"]),
        k=int(first["k"]),
    )
    for index, item in enumerate(corpus):
        if (item.get("rows"), item.get("cols"), item.get("k")) != (
            shape.rows,
            shape.cols,
            shape.k,
        ):
            raise ValueError(f"transaction {index} shape does not match first transaction")
    return shape


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(data: Any) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def manifest_relative_path(*, model_path: Path, manifest_path: Path) -> Path:
    model_resolved = model_path.resolve()
    manifest_dir = manifest_path.parent.resolve()
    try:
        return model_resolved.relative_to(manifest_dir)
    except ValueError:
        return model_resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate synthetic GEMM transactions")
    gen.add_argument("--out", required=True)
    gen.add_argument("--rows", type=int, default=2)
    gen.add_argument("--cols", type=int, default=2)
    gen.add_argument("--k", type=int, default=2)
    gen.add_argument("--samples", type=int, default=32)
    gen.add_argument("--seed", type=int, default=1)
    gen.add_argument("--data-width", type=int, default=8)
    gen.add_argument("--acc-width", type=int, default=32)
    gen.add_argument("--lanes", type=int, default=1)

    train = sub.add_parser("train", help="export ONNX surrogate and manifest")
    train.add_argument("--corpus", required=True)
    train.add_argument("--out-dir", required=True)
    train.add_argument("--epochs", type=int, default=0)
    train.add_argument("--seed", type=int, default=1)
    train.add_argument("--tolerance", type=int, default=0)

    val = sub.add_parser("validate", help="validate heldout transactions through pyrtl2rrtl")
    val.add_argument("--manifest", required=True)
    val.add_argument("--transactions", required=True)
    val.add_argument("--out")
    val.add_argument(
        "--pyrtl2rrtl",
        nargs="+",
        default=[
            "cargo",
            "run",
            "-q",
            "-p",
            "rrtl-pyrtl",
            "--features",
            "onnx-ort",
            "--bin",
            "pyrtl2rrtl",
            "--",
        ],
    )

    policy = sub.add_parser("policy", help="evaluate GEMM replacement policy through pyrtl2rrtl")
    policy.add_argument("--manifest", required=True)
    policy.add_argument("--transactions", required=True)
    policy.add_argument("--out")
    policy.add_argument(
        "--pyrtl2rrtl",
        nargs="+",
        default=[
            "cargo",
            "run",
            "-q",
            "-p",
            "rrtl-pyrtl",
            "--features",
            "onnx-ort",
            "--bin",
            "pyrtl2rrtl",
            "--",
        ],
    )

    args = parser.parse_args(argv)
    if args.command == "generate":
        corpus = generate_corpus(
            rows=args.rows,
            cols=args.cols,
            k=args.k,
            samples=args.samples,
            seed=args.seed,
            data_width=args.data_width,
            acc_width=args.acc_width,
            lanes=args.lanes,
        )
        Path(args.out).write_text(
            json.dumps(corpus, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    if args.command == "train":
        summary = train_and_export(
            corpus_path=Path(args.corpus),
            out_dir=Path(args.out_dir),
            epochs=args.epochs,
            seed=args.seed,
            tolerance=args.tolerance,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        summary = validate_with_rrtl(
            manifest_path=Path(args.manifest),
            transactions_path=Path(args.transactions),
            pyrtl2rrtl=args.pyrtl2rrtl,
            out_path=Path(args.out) if args.out else None,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["ok"] else 1
    if args.command == "policy":
        summary = policy_with_rrtl(
            manifest_path=Path(args.manifest),
            transactions_path=Path(args.transactions),
            pyrtl2rrtl=args.pyrtl2rrtl,
            out_path=Path(args.out) if args.out else None,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["ok"] else 1
    raise AssertionError(args.command)


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_nonnegative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


if __name__ == "__main__":
    raise SystemExit(main())
