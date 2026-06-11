"""GNN/Transformer dataset preparation for RRTL surrogate models.

This module intentionally stops at deterministic tensor-manifest generation.
Actual model training should consume this contract, train in PyTorch or another
ML stack, and export an ONNX artifact referenced by the RRTL surrogate manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NODE_KIND_IDS = {
    "input": 1,
    "output": 2,
    "wire": 3,
    "register": 4,
    "const": 5,
}
EVENT_TENSOR_BUNDLE_SCHEMA = "rrtl-surrogate-event-tensor-bundle-v1"
EVENT_TRAINER_SCHEMA = "rrtl-surrogate-gnn-transformer-event-trainer-v1"


def build_tensor_manifest(dataset: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic tensor metadata for a GNN+Transformer trainer."""

    if dataset.get("schema") != "rrtl-surrogate-dataset-v1":
        raise ValueError(f"unsupported dataset schema {dataset.get('schema')!r}")

    graph = dataset["graph"]
    trace = dataset["trace"]
    nodes = graph["node_features"]
    edges = graph["edge_index"]
    edge_features = graph["edge_features"]
    steps = trace["steps"]

    input_names = sorted({name for step in steps for name in step["inputs"]})
    output_names = sorted({name for step in steps for name in step["outputs"]})

    return {
        "schema": "rrtl-surrogate-gnn-transformer-tensors-v1",
        "source_hash": dataset["source_hash"],
        "top_name": dataset["top_name"],
        "graph_inputs": {
            "node_features": {
                "shape": [len(nodes), 4],
                "columns": ["kind_id", "bitwidth", "has_value", "has_reset_value"],
            },
            "edge_index": {"shape": [2, len(edges)]},
            "edge_features": {
                "shape": [len(edge_features), 2],
                "columns": ["net_index", "op_id"],
            },
        },
        "sequence_inputs": {
            "input_trace": {
                "shape": [len(steps), len(input_names)],
                "signals": input_names,
            },
            "output_trace": {
                "shape": [len(steps), len(output_names)],
                "signals": output_names,
            },
        },
        "transaction_inputs": {
            "gemm_descriptor": {
                "shape": [6],
                "columns": ["rows", "cols", "k", "data_width", "acc_width", "mode_id"],
            },
            "a_tensor": {"shape": ["rows", "k"]},
            "w_tensor": {"shape": ["k", "cols"]},
        },
        "outputs": {
            "c_tensor": {"shape": ["rows", "cols"]},
            "telemetry": {
                "shape": [3],
                "columns": ["latency_cycles", "active_cycles", "utilization"],
            },
            "uncertainty": {"shape": [1]},
        },
        "node_kind_ids": NODE_KIND_IDS,
    }


def build_event_trainer_manifest(bundle: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic GNN/Transformer trainer metadata for an event bundle."""

    if bundle.get("schema") != EVENT_TENSOR_BUNDLE_SCHEMA:
        raise ValueError(f"unsupported event tensor bundle schema {bundle.get('schema')!r}")
    tensor_manifest = bundle.get("manifest")
    if not isinstance(tensor_manifest, dict):
        raise ValueError("event tensor bundle requires manifest")
    inputs = bundle.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("event tensor bundle requires inputs")
    labels = bundle.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("event tensor bundle requires labels")
    metadata = bundle.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("event tensor bundle metadata must be an object")

    signal_window = _required_tensor(inputs, "signal_window")
    program_context = _required_tensor(inputs, "program_context")
    predicted_event = _required_tensor(labels, "predicted_event")
    sample_count = len(signal_window)
    _validate_sample_count("program_context", program_context, sample_count)
    _validate_sample_count("predicted_event", predicted_event, sample_count)

    manifest_inputs = tensor_manifest.get("inputs")
    if not isinstance(manifest_inputs, dict):
        raise ValueError("event tensor manifest requires inputs")
    signal_spec = _required_spec(manifest_inputs, "signal_window")
    program_spec = _required_spec(manifest_inputs, "program_context")
    label_spec = _required_spec(tensor_manifest.get("outputs", {}), "predicted_event")
    _validate_feature_list(signal_spec, "features", "signal_window")
    _validate_feature_list(program_spec, "features", "program_context")
    _validate_shape(signal_spec, "signal_window", sample_count)
    _validate_shape(program_spec, "program_context", sample_count)
    _validate_shape(label_spec, "predicted_event", sample_count)
    _validate_inner_width(
        "signal_window",
        signal_window,
        len(signal_spec["features"]),
        depth=2,
    )
    _validate_inner_width(
        "program_context",
        program_context,
        len(program_spec["features"]),
        depth=1,
    )
    _validate_inner_width("predicted_event", predicted_event, 1, depth=1)

    return {
        "schema": EVENT_TRAINER_SCHEMA,
        "source_hash": bundle["source_hash"],
        "top_name": bundle["top_name"],
        "task": {
            "surrogate_class": "event_predictor",
            "prediction_target": metadata.get(
                "target",
                tensor_manifest.get("task", {}).get("prediction_target"),
            ),
            "sample_count": sample_count,
        },
        "inputs": {
            "signal_window": {
                "shape": signal_spec["shape"],
                "features": signal_spec["features"],
            },
            "program_context": {
                "shape": program_spec["shape"],
                "features": program_spec["features"],
            },
        },
        "labels": {
            "predicted_event": {
                "shape": label_spec["shape"],
                "kind": tensor_manifest.get("label", {}).get("kind", "binary"),
            }
        },
        "metadata": {
            "sample_ids": metadata.get("sample_ids", []),
            "lanes": metadata.get("lanes", []),
            "batching": ["sample_ids", "lanes"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else None
    if argv and argv[0] == "event-bundle":
        parser = argparse.ArgumentParser(
            description="Prepare GNN/Transformer trainer metadata from an event tensor bundle"
        )
        parser.add_argument("command")
        parser.add_argument("bundle", help="rrtl-surrogate-event-tensor-bundle-v1 JSON")
        parser.add_argument("--out", required=True, help="output trainer metadata JSON")
        args = parser.parse_args(argv)
        bundle = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
        trainer_manifest = build_event_trainer_manifest(bundle)
        Path(args.out).write_text(
            json.dumps(trainer_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0

    parser = argparse.ArgumentParser(
        description="Prepare tensor metadata for an RRTL GNN/Transformer surrogate trainer"
    )
    parser.add_argument("dataset", help="rrtl-surrogate-dataset-v1 JSON")
    parser.add_argument("--out", required=True, help="output tensor metadata JSON")
    args = parser.parse_args(argv)

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    tensor_manifest = build_tensor_manifest(dataset)
    Path(args.out).write_text(
        json.dumps(tensor_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _required_tensor(container: dict[str, Any], name: str) -> list[Any]:
    tensor = container.get(name)
    if not isinstance(tensor, list):
        raise ValueError(f"event tensor bundle requires {name}")
    return tensor


def _required_spec(container: Any, name: str) -> dict[str, Any]:
    if not isinstance(container, dict):
        raise ValueError(f"event tensor manifest requires {name}")
    spec = container.get(name)
    if not isinstance(spec, dict):
        raise ValueError(f"event tensor manifest requires {name}")
    return spec


def _validate_sample_count(name: str, tensor: list[Any], sample_count: int) -> None:
    if len(tensor) != sample_count:
        raise ValueError(
            f"{name} sample count {len(tensor)} does not match signal_window sample count {sample_count}"
        )


def _validate_feature_list(spec: dict[str, Any], field: str, name: str) -> None:
    features = spec.get(field)
    if not isinstance(features, list) or not all(isinstance(item, str) for item in features):
        raise ValueError(f"{name} requires a {field} list")


def _validate_shape(spec: dict[str, Any], name: str, sample_count: int) -> None:
    shape = spec.get("shape")
    if not isinstance(shape, list) or not shape:
        raise ValueError(f"{name} requires a shape")
    if shape[0] != sample_count:
        raise ValueError(
            f"{name} manifest sample count {shape[0]} does not match tensor sample count {sample_count}"
        )


def _validate_inner_width(
    name: str,
    tensor: list[Any],
    expected_width: int,
    *,
    depth: int,
) -> None:
    for sample_index, sample in enumerate(tensor):
        rows = sample if depth == 2 else [sample]
        if not isinstance(rows, list):
            raise ValueError(f"{name} sample {sample_index} must be a list")
        for row_index, row in enumerate(rows):
            if not isinstance(row, list):
                raise ValueError(f"{name} sample {sample_index} row {row_index} must be a list")
            if len(row) != expected_width:
                raise ValueError(
                    f"{name} sample {sample_index} row {row_index} width {len(row)} "
                    f"does not match manifest width {expected_width}"
                )


if __name__ == "__main__":
    raise SystemExit(main())
