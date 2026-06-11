"""Instrumentation-driven cache-miss event predictor scaffold.

This module models the near-term RRTL instrumentation contract: a surrogate can
predict a named event from a sampled signal window plus program context without
replacing architectural state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import warnings
from pathlib import Path
from typing import Any

from .gnn_transformer import build_event_trainer_manifest
from .minimal_onnx import write_minimal_event_predictor_onnx


EVENT_SCHEMA = "rrtl-surrogate-instrumentation-event-v1"
CORPUS_SCHEMA = "rrtl-surrogate-instrumentation-corpus-v1"
MANIFEST_SCHEMA = "rrtl-surrogate-manifest-v1"
EVENT_FAST_RUN_SCHEMA = "rrtl-surrogate-event-fast-run-v1"
MODEL_FAST_GOLDEN_SCHEMA = "rrtl-surrogate-model-fast-golden-v1"
EVENT_TENSOR_BUNDLE_SCHEMA = "rrtl-surrogate-event-tensor-bundle-v1"
EVENT_PROFILE_SCHEMA = "rrtl-surrogate-event-profile-v1"
FLOW_BUNDLE_SCHEMA = "rrtl-surrogate-flow-bundle-v1"
RUNTIME_TELEMETRY_GATE_SCHEMA = "rrtl-surrogate-runtime-telemetry-gate-v1"
RUNTIME_HANDOFF_TELEMETRY_GATE_SCHEMA = (
    "rrtl-surrogate-runtime-handoff-telemetry-gate-v1"
)
USE_CASE_CATALOG_SCHEMA = "rrtl-surrogate-use-case-catalog-v1"
INSTRUMENTATION_USE_CASE_MATCH_SCHEMA = "rrtl-surrogate-instrumentation-use-case-match-v1"
QUALITY_GATE_SCHEMA = "rrtl-surrogate-quality-gate-v1"
RUNTIME_HANDOFF_SCHEMA = "rrtl-surrogate-runtime-handoff-v1"

DEFAULT_QUALITY_THRESHOLDS = {
    "min_validation_accuracy": 1.0,
    "min_shadow_accuracy": 1.0,
    "max_fail_closed": 0,
    "max_shadow_failed": 0,
    "require_model_fast_ok": True,
    "require_runtime_ready": True,
}

SIGNAL_FEATURES = [
    "cycle_delta",
    "load",
    "store",
    "addr_low",
    "cache_set",
    "tag_delta",
    "pending_misses",
    "store_buffer_occupancy",
]
PROGRAM_FEATURES = ["pc", "opcode_id", "stride", "working_set_log2"]

STALL_SIGNAL_FEATURES = [
    "cycle_delta",
    "load",
    "store",
    "pending_misses",
    "store_buffer_occupancy",
]
STALL_PROGRAM_FEATURES = ["pc", "opcode_id"]

BUILTIN_EVENT_TARGETS = ("cache_miss", "stall_event")
EVENT_PROFILE_DIR = Path(__file__).with_name("event_profiles")


def generate_corpus(
    *,
    samples: int,
    window: int,
    seed: int,
    sets: int = 16,
    lanes: int = 1,
    target: str | None = "cache_miss",
    profile_path: Path | None = None,
) -> dict[str, Any]:
    """Generate deterministic instrumentation windows with profile-selected labels."""

    profile = load_event_profile(target=target, profile_path=profile_path)
    _validate_positive("samples", samples)
    window = window or int(profile["input_window_cycles_default"])
    _validate_positive("window", window)
    _validate_positive("sets", sets)
    _validate_positive("lanes", lanes)
    rng = random.Random(seed)
    events = []
    for index in range(samples):
        pc = 0x1000 + rng.randrange(0, 64) * 4
        opcode_id = rng.choice([1, 2, 3, 4])
        stride = rng.choice([0, 4, 16, 64, 128])
        working_set_log2 = rng.choice([8, 10, 12, 14, 16])
        base_addr = rng.randrange(0, 1 << 16)
        pending = rng.randrange(0, 5)
        store_buffer = rng.randrange(0, 8)
        previous_tag = base_addr >> 10
        signals = []
        for cycle in range(window):
            is_load = 1 if rng.random() < 0.58 else 0
            is_store = 1 if not is_load and rng.random() < 0.45 else 0
            addr = (base_addr + cycle * stride + rng.randrange(0, 4) * 4) & 0xFFFF
            cache_set = (addr >> 4) % sets
            tag = addr >> 10
            signals.append(
                {
                    "cycle_delta": cycle,
                    "load": is_load,
                    "store": is_store,
                    "addr_low": addr & 0xFF,
                    "cache_set": cache_set,
                    "tag_delta": abs(tag - previous_tag),
                    "pending_misses": pending,
                    "store_buffer_occupancy": store_buffer,
                }
            )
            pending = max(0, pending + is_load - rng.randrange(0, 2))
            store_buffer = max(0, min(7, store_buffer + is_store - rng.randrange(0, 2)))
            previous_tag = tag

        program = {
            "pc": pc,
            "opcode_id": opcode_id,
            "stride": stride,
            "working_set_log2": working_set_log2,
        }
        for feature in profile["program_features"]:
            program.setdefault(feature, rng.randrange(0, 16))
        for step in signals:
            for feature in profile["signal_features"]:
                step.setdefault(feature, rng.randrange(0, 8))
        label = predict_event_rule(profile, signals, program)
        event = {
            "schema": EVENT_SCHEMA,
            "sample_id": index,
            "target": profile["target"],
            "window_cycles": window,
            "horizon_cycles": int(profile["horizon_cycles_default"]),
            "program": program,
            "signals": signals,
            "label": {profile["label"]["name"]: label},
        }
        if lanes > 1:
            event["lane"] = index % lanes
        events.append(event)
    source = {"seed": seed, "samples": samples, "target": profile["target"], "window": window}
    if lanes > 1:
        source["lanes"] = lanes
    return {
        "schema": CORPUS_SCHEMA,
        "source_hash": sha256_json(source),
        "top_name": "InstrumentedCache",
        "events": events,
    }


def predict_cache_miss_rule(
    signals: list[dict[str, int]],
    working_set_log2: int,
    stride: int,
) -> int:
    """A deterministic baseline labeler standing in for exact instrumentation."""

    loads = sum(step["load"] for step in signals)
    tag_motion = sum(step["tag_delta"] for step in signals)
    pending_peak = max(step["pending_misses"] for step in signals)
    set_span = len({step["cache_set"] for step in signals})
    pressure = loads * 2 + tag_motion * 3 + pending_peak + set_span
    if working_set_log2 >= 14:
        pressure += 4
    if stride >= 64:
        pressure += 3
    return 1 if pressure >= 14 else 0


def predict_stall_event_rule(signals: list[dict[str, int]]) -> int:
    """A deterministic baseline labeler for generic stall-like events."""

    stores = sum(step["store"] for step in signals)
    loads = sum(step["load"] for step in signals)
    pending_peak = max(step["pending_misses"] for step in signals)
    store_buffer_peak = max(step["store_buffer_occupancy"] for step in signals)
    mixed_memory_ops = 2 if stores > 0 and loads > 0 else 0
    pressure = pending_peak * 2 + store_buffer_peak + stores + mixed_memory_ops
    return 1 if pressure >= 12 else 0


def predict_event_rule(
    profile_or_target: dict[str, Any] | str,
    signals: list[dict[str, int]],
    program: dict[str, int],
) -> int:
    profile = (
        load_event_profile(target=profile_or_target)
        if isinstance(profile_or_target, str)
        else profile_or_target
    )
    return evaluate_linear_threshold_rule(profile["mock_rule"], signals, program)


def evaluate_linear_threshold_rule(
    rule: dict[str, Any],
    signals: list[dict[str, int]],
    program: dict[str, int],
) -> int:
    if rule.get("kind") != "linear_threshold":
        raise ValueError(f"unsupported mock rule kind {rule.get('kind')!r}")
    score = 0
    for term in rule.get("terms", []):
        value = _rule_value(term, signals, program)
        score += int(value) * int(term.get("weight", 1))
    for bonus in rule.get("bonuses", []):
        value = _rule_value(bonus, signals, program)
        if _compare_rule_value(value, str(bonus.get("op", ">=")), int(bonus.get("value", 0))):
            score += int(bonus.get("bonus", 0))
    return 1 if score >= int(rule["threshold"]) else 0


def _rule_value(
    spec: dict[str, Any],
    signals: list[dict[str, int]],
    program: dict[str, int],
) -> int:
    source = spec.get("source")
    feature = str(spec.get("feature", ""))
    if source == "program":
        return int(program.get(feature, 0))
    if source != "signal":
        raise ValueError(f"unsupported mock rule source {source!r}")
    values = [int(step.get(feature, 0)) for step in signals]
    reduction = spec.get("reduction", "sum")
    if reduction == "sum":
        return sum(values)
    if reduction == "max":
        return max(values) if values else 0
    if reduction == "min":
        return min(values) if values else 0
    if reduction == "last":
        return values[-1] if values else 0
    if reduction == "unique_count":
        return len(set(values))
    raise ValueError(f"unsupported mock rule reduction {reduction!r}")


def _compare_rule_value(actual: int, op: str, expected: int) -> bool:
    if op == ">=":
        return actual >= expected
    if op == ">":
        return actual > expected
    if op == "<=":
        return actual <= expected
    if op == "<":
        return actual < expected
    if op in ("==", "="):
        return actual == expected
    if op == "!=":
        return actual != expected
    raise ValueError(f"unsupported mock rule comparison {op!r}")


def build_tensor_manifest(
    corpus: dict[str, Any],
    *,
    target: str | None = "cache_miss",
    profile_path: Path | None = None,
) -> dict[str, Any]:
    profile = load_event_profile(target=target, profile_path=profile_path)
    events, window = _validated_events(corpus, target=profile["target"])
    tensor_manifest = {
        "schema": "rrtl-surrogate-event-predictor-tensors-v1",
        "source_hash": corpus["source_hash"],
        "top_name": corpus["top_name"],
        "task": {
            "surrogate_class": "event_predictor",
            "prediction_target": profile["target"],
            "input_window_cycles": window,
            "horizon_cycles": 1,
        },
        "inputs": {
            "signal_window": {
                "shape": [len(events), window, len(profile["signal_features"])],
                "features": profile["signal_features"],
            },
            "program_context": {
                "shape": [len(events), len(profile["program_features"])],
                "features": profile["program_features"],
            },
        },
        "outputs": {
            "event_probability": {"shape": [len(events), 1]},
            "predicted_event": {"shape": [len(events), 1]},
        },
        "label": profile["label"],
    }
    lanes = sorted(
        {
            int(event["lane"])
            for event in events
            if "lane" in event and event["lane"] is not None
        }
    )
    if lanes:
        tensor_manifest["batching"] = {"lane_metadata": True, "lanes": lanes}
    return tensor_manifest


def build_tensor_bundle(
    corpus: dict[str, Any],
    *,
    target: str | None = "cache_miss",
    profile_path: Path | None = None,
) -> dict[str, Any]:
    profile = load_event_profile(target=target, profile_path=profile_path)
    events, window = _validated_events(corpus, target=profile["target"])
    signal_window = []
    program_context = []
    labels = []
    sample_ids = []
    lanes = []
    for index, event in enumerate(events):
        signals = event.get("signals")
        if not isinstance(signals, list) or len(signals) != window:
            raise ValueError(f"event {index} signals must have {window} entries")
        signal_window.append(
            [
                [
                    _required_int(step, feature, f"event {index} signal {step_index}")
                    for feature in profile["signal_features"]
                ]
                for step_index, step in enumerate(signals)
            ]
        )
        program = event.get("program")
        if not isinstance(program, dict):
            raise ValueError(f"event {index} program must be an object")
        program_context.append(
            [
                _required_int(program, feature, f"event {index} program")
                for feature in profile["program_features"]
            ]
        )
        labels.append(
            [
                _validate_binary_value(
                    f"event {index} label {target}",
                    event["label"][profile["label"]["name"]],
                )
            ]
        )
        sample_ids.append(int(event["sample_id"]))
        lanes.append(int(event.get("lane", 0)))

    return {
        "schema": EVENT_TENSOR_BUNDLE_SCHEMA,
        "source_hash": corpus["source_hash"],
        "top_name": corpus["top_name"],
        "manifest": build_tensor_manifest(corpus, target=profile["target"], profile_path=profile_path),
        "inputs": {
            "signal_window": signal_window,
            "program_context": program_context,
        },
        "labels": {"predicted_event": labels},
        "metadata": {
            "sample_ids": sample_ids,
            "lanes": lanes,
            "profile": profile["target"],
            "target": profile["target"],
        },
    }


def write_tensor_bundle(
    *,
    corpus_path: Path,
    out_path: Path,
    target: str | None = "cache_miss",
    profile_path: Path | None = None,
) -> dict[str, Any]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    bundle = build_tensor_bundle(corpus, target=target, profile_path=profile_path)
    out_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle


def train_and_export(
    *,
    corpus_path: Path,
    out_dir: Path,
    seed: int,
    target: str | None = "cache_miss",
    profile_path: Path | None = None,
) -> dict[str, Any]:
    """Write a rule artifact, manifest, tensor contract, and holdout metrics."""

    profile = load_event_profile(target=target, profile_path=profile_path)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    tensor_manifest = build_tensor_manifest(corpus, target=profile["target"], profile_path=profile_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / f"{profile['target']}_rule.json"
    manifest_path = out_dir / "manifest.json"
    tensors_path = out_dir / "tensors.json"
    tensor_bundle_path = out_dir / "tensor_bundle.json"
    trainer_manifest_path = out_dir / "trainer_manifest.json"
    summary_path = out_dir / "summary.json"

    artifact = {
        "schema": "rrtl-surrogate-rule-artifact-v1",
        "model_family": "rule-baseline",
        "prediction_target": profile["target"],
        "signal_features": profile["signal_features"],
        "program_features": profile["program_features"],
        "label": profile["label"],
        "rule": profile["mock_rule"].get("description", ""),
        "profile": profile,
        "mock_rule": profile["mock_rule"],
        "seed": seed,
    }
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_hash = sha256_file(artifact_path)
    manifest = build_manifest(
        corpus,
        artifact_path=manifest_relative_path(
            artifact_path=artifact_path,
            manifest_path=manifest_path,
        ),
        artifact_hash=artifact_hash,
        tensor_manifest=tensor_manifest,
        profile_path=profile_path,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tensors_path.write_text(
        json.dumps(tensor_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tensor_bundle = build_tensor_bundle(corpus, target=profile["target"], profile_path=profile_path)
    tensor_bundle_path.write_text(
        json.dumps(tensor_bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trainer_manifest = build_event_trainer_manifest(tensor_bundle)
    trainer_manifest_path.write_text(
        json.dumps(trainer_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = evaluate_rule(corpus, target=profile["target"], profile_path=profile_path)
    summary = {
        "schema": "rrtl-surrogate-event-training-summary-v1",
        "samples": len(corpus["events"]),
        "seed": seed,
        "target": profile["target"],
        "metrics": metrics,
        "outputs": {
            "artifact": str(artifact_path),
            "manifest": str(manifest_path),
            "tensors": str(tensors_path),
            "tensor_bundle": str(tensor_bundle_path),
            "trainer_manifest": str(trainer_manifest_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def train_learned_and_export(
    *,
    corpus_path: Path,
    out_dir: Path,
    epochs: int,
    seed: int,
    hidden_dim: int = 16,
    heads: int = 2,
    target: str | None = "cache_miss",
    profile_path: Path | None = None,
) -> dict[str, Any]:
    """Train and export an ONNX event predictor from instrumentation tensors."""

    _validate_nonnegative("epochs", epochs)
    _validate_positive("hidden_dim", hidden_dim)
    _validate_positive("heads", heads)
    if hidden_dim % heads != 0:
        raise ValueError("hidden_dim must be divisible by heads")

    profile = load_event_profile(target=target, profile_path=profile_path)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    tensor_bundle = build_tensor_bundle(corpus, target=profile["target"], profile_path=profile_path)
    trainer_manifest = build_event_trainer_manifest(tensor_bundle)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.onnx"
    manifest_path = out_dir / "manifest.json"
    tensors_path = out_dir / "tensors.json"
    tensor_bundle_path = out_dir / "tensor_bundle.json"
    trainer_manifest_path = out_dir / "trainer_manifest.json"
    heldout_path = out_dir / "heldout.json"
    summary_path = out_dir / "summary.json"

    metrics = export_learned_onnx_model(
        tensor_bundle,
        model_path,
        epochs=epochs,
        seed=seed,
        hidden_dim=hidden_dim,
        heads=heads,
    )
    model_hash = sha256_file(model_path)
    tensor_manifest = tensor_bundle["manifest"]
    manifest = build_learned_manifest(
        corpus,
        artifact_path=manifest_relative_path(
            artifact_path=model_path,
            manifest_path=manifest_path,
        ),
        artifact_hash=model_hash,
        tensor_manifest=tensor_manifest,
        profile_path=profile_path,
    )
    heldout = corpus["events"][: min(len(corpus["events"]), 8)]

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tensors_path.write_text(
        json.dumps(tensor_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tensor_bundle_path.write_text(
        json.dumps(tensor_bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trainer_manifest_path.write_text(
        json.dumps(trainer_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    heldout_path.write_text(
        json.dumps(heldout, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": "rrtl-surrogate-event-learned-training-summary-v1",
        "samples": len(corpus["events"]),
        "heldout": len(heldout),
        "epochs": epochs,
        "seed": seed,
        "target": profile["target"],
        "model": {
            "family": "gnn-transformer",
            "hidden_dim": hidden_dim,
            "heads": heads,
        },
        "metrics": metrics,
        "export_backend": metrics["export_backend"],
        "outputs": {
            "model": str(model_path),
            "manifest": str(manifest_path),
            "tensors": str(tensors_path),
            "tensor_bundle": str(tensor_bundle_path),
            "trainer_manifest": str(trainer_manifest_path),
            "heldout": str(heldout_path),
        },
        "artifact_sha256": model_hash,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def export_learned_onnx_model(
    tensor_bundle: dict[str, Any],
    model_path: Path,
    *,
    epochs: int,
    seed: int,
    hidden_dim: int,
    heads: int,
) -> dict[str, Any]:
    try:
        import torch
    except Exception as err:  # noqa: BLE001 - provide a clean CLI error.
        raise RuntimeError("PyTorch is required for event predictor ONNX export") from err

    try:
        import onnx  # noqa: F401
        has_onnx = True
    except Exception:
        has_onnx = False

    torch.manual_seed(seed)
    signal_window = torch.tensor(
        tensor_bundle["inputs"]["signal_window"],
        dtype=torch.float32,
    )
    program_context = torch.tensor(
        tensor_bundle["inputs"]["program_context"],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        tensor_bundle["labels"]["predicted_event"],
        dtype=torch.float32,
    )
    signal_std, signal_mean = torch.std_mean(signal_window, dim=(0, 1), unbiased=False)
    program_std, program_mean = torch.std_mean(program_context, dim=0, unbiased=False)
    signal_std = torch.where(signal_std > 0, signal_std, torch.ones_like(signal_std))
    program_std = torch.where(program_std > 0, program_std, torch.ones_like(program_std))

    model = _build_event_predictor_model(
        signal_features=signal_window.shape[-1],
        program_features=program_context.shape[-1],
        hidden_dim=hidden_dim,
        heads=heads,
        signal_mean=signal_mean,
        signal_std=signal_std,
        program_mean=program_mean,
        program_std=program_std,
    )
    if epochs > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model.logits(signal_window, program_context)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model.logits(signal_window, program_context)
        probability = torch.sigmoid(logits)
        predicted = (probability >= 0.5).to(torch.float32)
        accuracy = float((predicted == labels).to(torch.float32).mean().item())
        loss = float(torch.nn.functional.binary_cross_entropy(probability, labels).item())

    if has_onnx:
        sample_signal = signal_window[:1]
        sample_program = program_context[:1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.filterwarnings(
                "ignore",
                message="You are using the legacy TorchScript-based ONNX export",
                category=DeprecationWarning,
            )
            torch.onnx.export(
                model,
                (sample_signal, sample_program),
                model_path,
                input_names=["signal_window", "program_context"],
                output_names=["event_probability", "predicted_event"],
                dynamic_axes={
                    "signal_window": {0: "batch"},
                    "program_context": {0: "batch"},
                    "event_probability": {0: "batch"},
                    "predicted_event": {0: "batch"},
                },
                opset_version=17,
                dynamo=False,
            )
        export_backend = "torch-onnx"
    else:
        positive_count = int(labels.sum().item())
        predicted_value = 1 if positive_count * 2 >= labels.numel() else 0
        probability_value = float(predicted_value)
        fallback_predictions = torch.full_like(labels, float(predicted_value))
        accuracy = float((fallback_predictions == labels).to(torch.float32).mean().item())
        loss = 0.0 if accuracy == 1.0 else 1.0
        write_minimal_event_predictor_onnx(
            model_path,
            signal_window_shape=[
                1,
                int(signal_window.shape[1]),
                int(signal_window.shape[2]),
            ],
            program_context_shape=[1, int(program_context.shape[1])],
            probability=probability_value,
            predicted=predicted_value,
        )
        export_backend = "minimal-onnx"
    return {"accuracy": accuracy, "loss": loss, "export_backend": export_backend}


def _build_event_predictor_model(
    *,
    signal_features: int,
    program_features: int,
    hidden_dim: int,
    heads: int,
    signal_mean: Any,
    signal_std: Any,
    program_mean: Any,
    program_std: Any,
) -> Any:
    import torch

    class EventPredictor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("signal_mean", signal_mean)
            self.register_buffer("signal_std", signal_std)
            self.register_buffer("program_mean", program_mean)
            self.register_buffer("program_std", program_std)
            self.signal_projection = torch.nn.Linear(signal_features, hidden_dim)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.0,
                batch_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.program_projection = torch.nn.Sequential(
                torch.nn.Linear(program_features, hidden_dim),
                torch.nn.ReLU(),
            )
            self.head = torch.nn.Linear(hidden_dim * 2, 1)

        def logits(self, signal_window, program_context):  # type: ignore[no-untyped-def]
            signal = (signal_window - self.signal_mean) / self.signal_std
            program = (program_context - self.program_mean) / self.program_std
            encoded = self.encoder(self.signal_projection(signal))
            pooled = encoded.mean(dim=1)
            program_embedding = self.program_projection(program)
            return self.head(torch.cat([pooled, program_embedding], dim=1))

        def forward(self, signal_window, program_context):  # type: ignore[no-untyped-def]
            probability = torch.sigmoid(self.logits(signal_window, program_context))
            predicted = (probability >= 0.5).to(torch.float32)
            return probability, predicted

    return EventPredictor()


def validate_with_rrtl(
    *,
    manifest_path: Path,
    corpus_path: Path,
    pyrtl2rrtl: list[str],
    out_path: Path | None = None,
) -> dict[str, Any]:
    command = [
        *pyrtl2rrtl,
        "surrogate",
        "validate-events",
        str(manifest_path),
        str(corpus_path),
    ]
    if out_path is not None:
        command.extend(["--out", str(out_path)])
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if out_path is not None and out_path.exists():
        summary = json.loads(out_path.read_text(encoding="utf-8"))
    elif completed.stdout:
        summary = json.loads(completed.stdout)
    else:
        summary = {
            "schema": "rrtl-surrogate-event-validation-v1",
            "ok": False,
            "errors": [completed.stderr.strip() or "validate-events produced no report"],
        }
    if completed.returncode != 0:
        errors = summary.setdefault("errors", [])
        if completed.stderr and completed.stderr.strip() not in errors:
            errors.append(completed.stderr.strip())
        summary["ok"] = False
    return summary


def shadow_with_rrtl(
    *,
    manifest_path: Path,
    corpus_path: Path,
    pyrtl2rrtl: list[str],
    out_path: Path | None = None,
) -> dict[str, Any]:
    command = [
        *pyrtl2rrtl,
        "surrogate",
        "shadow-events",
        str(manifest_path),
        str(corpus_path),
    ]
    if out_path is not None:
        command.extend(["--out", str(out_path)])
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if out_path is not None and out_path.exists():
        summary = json.loads(out_path.read_text(encoding="utf-8"))
    elif completed.stdout:
        summary = json.loads(completed.stdout)
    else:
        summary = {
            "schema": "rrtl-surrogate-event-shadow-v1",
            "ok": False,
            "errors": [completed.stderr.strip() or "shadow-events produced no report"],
        }
    if completed.returncode != 0:
        errors = summary.setdefault("errors", [])
        if completed.stderr and completed.stderr.strip() not in errors:
            errors.append(completed.stderr.strip())
        summary["ok"] = False
    return summary


def inspect_instrumentation_with_rrtl(
    *,
    trace_path: Path,
    pyrtl2rrtl: list[str],
    config_path: Path | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    command = [
        *pyrtl2rrtl,
        "surrogate",
        "inspect-instrumentation",
        str(trace_path),
    ]
    if config_path is not None:
        command.extend(["--config", str(config_path)])
    if out_path is not None:
        command.extend(["--out", str(out_path)])
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if out_path is not None and out_path.exists():
        summary = json.loads(out_path.read_text(encoding="utf-8"))
    elif completed.stdout:
        summary = json.loads(completed.stdout)
    else:
        summary = {
            "schema": "rrtl-instrumentation-trace-inspection-v1",
            "ok": False,
            "errors": [completed.stderr.strip() or "inspect-instrumentation produced no report"],
        }
    if completed.returncode != 0:
        errors = summary.setdefault("errors", [])
        if completed.stderr and completed.stderr.strip() not in errors:
            errors.append(completed.stderr.strip())
        summary["ok"] = False
    return summary


def match_instrumentation_use_case(
    *,
    trace_path: Path,
    config_path: Path,
    pyrtl2rrtl: list[str] | None = None,
    target: str | None = "cache_miss",
    profile_path: Path | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    cli = pyrtl2rrtl or [
        "cargo",
        "run",
        "-q",
        "-p",
        "rrtl-pyrtl",
        "--bin",
        "pyrtl2rrtl",
        "--",
    ]
    inspection = inspect_instrumentation_with_rrtl(
        trace_path=trace_path,
        config_path=config_path,
        pyrtl2rrtl=cli,
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profile = load_event_profile(target=target, profile_path=profile_path)
    report = build_instrumentation_use_case_match(
        config=config,
        use_case=build_use_case_contract(profile),
        instrumentation_inspection=inspection,
    )
    if out_path is not None:
        out_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def build_instrumentation_use_case_match(
    *,
    config: dict[str, Any],
    use_case: dict[str, Any],
    instrumentation_inspection: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    config_target = config.get("target")
    if config_target != use_case["target"]:
        errors.append(
            f"target mismatch: config {config_target!r}, use case {use_case['target']!r}"
        )

    config_signal_features = _mapping_names(config.get("signal_features"))
    config_program_features = _mapping_names(config.get("program_features"))
    label = config.get("label") if isinstance(config.get("label"), dict) else {}
    config_label = label.get("name")
    use_case_label = use_case["label"]["name"]
    if config_label != use_case_label:
        errors.append(
            f"label mismatch: config {config_label!r}, use case {use_case_label!r}"
        )

    signal_expected = list(use_case["signal_features"])
    program_expected = list(use_case["program_features"])
    missing_signal, extra_signal = _feature_set_delta(signal_expected, config_signal_features)
    missing_program, extra_program = _feature_set_delta(program_expected, config_program_features)
    for feature in missing_signal:
        errors.append(f"missing signal feature {feature!r}")
    for feature in extra_signal:
        errors.append(f"extra signal feature {feature!r}")
    for feature in missing_program:
        errors.append(f"missing program feature {feature!r}")
    for feature in extra_program:
        errors.append(f"extra program feature {feature!r}")

    compatibility = instrumentation_inspection.get("compatibility")
    if not instrumentation_inspection.get("ok"):
        errors.extend(_collect_report_messages("instrumentation", instrumentation_inspection, "errors"))
    if not isinstance(compatibility, dict):
        errors.append("instrumentation inspection missing compatibility report")
        compatibility_summary = {}
    else:
        compatibility_summary = {
            "target": compatibility.get("target"),
            "window_cycles": compatibility.get("window_cycles"),
            "horizon_cycles": compatibility.get("horizon_cycles"),
            "emittable_samples": compatibility.get("emittable_samples"),
            "missing_fields": compatibility.get("missing_fields", []),
        }
        if compatibility.get("target") != use_case["target"]:
            errors.append(
                "instrumentation compatibility target mismatch: "
                f"{compatibility.get('target')!r} != {use_case['target']!r}"
            )
        if int(compatibility.get("emittable_samples", 0) or 0) <= 0:
            errors.append("instrumentation compatibility has no emittable samples")
        missing_fields = compatibility.get("missing_fields")
        if isinstance(missing_fields, list) and missing_fields:
            errors.append("instrumentation compatibility has missing fields")
    warnings.extend(_collect_report_messages("instrumentation", instrumentation_inspection, "warnings"))

    ok = not errors
    return {
        "schema": INSTRUMENTATION_USE_CASE_MATCH_SCHEMA,
        "ok": ok,
        "use_case": use_case,
        "config": {
            "target": config_target,
            "signal_features": config_signal_features,
            "program_features": config_program_features,
            "label": config_label,
            "window_cycles": config.get("window_cycles"),
            "horizon_cycles": config.get("horizon_cycles"),
        },
        "feature_match": {
            "missing_signal_features": missing_signal,
            "extra_signal_features": extra_signal,
            "missing_program_features": missing_program,
            "extra_program_features": extra_program,
        },
        "label_match": config_label == use_case_label,
        "target_match": config_target == use_case["target"],
        "instrumentation_compatible": bool(instrumentation_inspection.get("ok")),
        "instrumentation_compatibility": compatibility_summary,
        "errors": errors,
        "warnings": warnings,
    }


def _mapping_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    names = []
    for item in value:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])
    return names


def _feature_set_delta(expected: list[str], actual: list[str]) -> tuple[list[str], list[str]]:
    expected_set = set(expected)
    actual_set = set(actual)
    return (
        sorted(expected_set - actual_set),
        sorted(actual_set - expected_set),
    )


def build_flow_bundle(
    *,
    model: str,
    target: str,
    use_case: dict[str, Any],
    instrumentation_match: dict[str, Any],
    quality_gate: dict[str, Any] | None = None,
    artifacts: dict[str, str],
    instrumentation_inspection: dict[str, Any],
    corpus: dict[str, Any],
    inspection: dict[str, Any],
    validation: dict[str, Any],
    shadow: dict[str, Any],
    fast_run: dict[str, Any],
    runtime_execution: dict[str, Any],
    model_fast_report: dict[str, Any],
) -> dict[str, Any]:
    compatibility = instrumentation_inspection.get("compatibility") or {}
    events = corpus.get("events") if isinstance(corpus.get("events"), list) else []
    fast_results = (
        fast_run.get("results")
        if isinstance(fast_run.get("results"), list)
        else []
    )
    model_totals = (
        model_fast_report.get("totals")
        if isinstance(model_fast_report.get("totals"), dict)
        else {}
    )
    readiness = {
        "instrumentation_use_case_match": bool(instrumentation_match.get("ok")),
        "instrumentation_compatible": bool(instrumentation_inspection.get("ok")),
        "corpus_ok": corpus.get("schema") == CORPUS_SCHEMA and bool(events),
        "validation_ok": bool(validation.get("ok")),
        "shadow_ok": bool(shadow.get("ok")),
        "runtime_ready": bool(runtime_execution.get("ready")),
        "model_fast_ok": bool(model_fast_report.get("ok")),
    }
    if quality_gate is not None:
        readiness["quality_gate"] = bool(quality_gate.get("ok"))
    errors = _collect_report_messages(
        "instrumentation_match",
        instrumentation_match,
        "errors",
    )
    errors.extend(_collect_report_messages(
        "instrumentation",
        instrumentation_inspection,
        "errors",
    ))
    errors.extend(_collect_report_messages("inspection", inspection, "errors"))
    errors.extend(_collect_report_messages("validation", validation, "errors"))
    errors.extend(_collect_report_messages("shadow", shadow, "errors"))
    errors.extend(_collect_report_messages("fast_run", fast_run, "errors"))
    errors.extend(_collect_report_messages("runtime_execution", runtime_execution, "errors"))
    errors.extend(_collect_report_messages("model_fast", model_fast_report, "errors"))
    if quality_gate is not None:
        errors.extend(_collect_report_messages("quality_gate", quality_gate, "errors"))
    warnings = _collect_report_messages(
        "instrumentation_match",
        instrumentation_match,
        "warnings",
    )
    warnings.extend(_collect_report_messages(
        "instrumentation",
        instrumentation_inspection,
        "warnings",
    ))
    warnings.extend(_collect_report_messages("inspection", inspection, "warnings"))

    counters = {
        "lanes": int(
            instrumentation_inspection.get(
                "lane_count",
                fast_run.get("total_lanes", 0),
            )
            or 0
        ),
        "emittable_samples": int(compatibility.get("emittable_samples", 0) or 0),
        "surrogate_replacements": int(
            fast_run.get(
                "surrogate_replacements",
                model_totals.get("surrogate_replacements", 0),
            )
            or 0
        ),
        "exact_fallbacks": int(
            fast_run.get("exact_fallbacks", model_totals.get("exact_fallbacks", 0))
            or 0
        ),
        "fail_closed": int(
            fast_run.get("fail_closed", model_totals.get("fail_closed", 0)) or 0
        ),
        "shadow_sampled": int(
            model_totals.get(
                "shadow_sampled",
                sum(1 for item in fast_results if item.get("shadow_sampled")),
            )
            or 0
        ),
        "shadow_passed": int(fast_run.get("shadow_passed", 0) or 0),
        "shadow_failed": int(fast_run.get("shadow_failed", 0) or 0),
    }
    ok = all(readiness.values()) and not errors
    return {
        "schema": FLOW_BUNDLE_SCHEMA,
        "ok": ok,
        "target": target,
        "model": model,
        "use_case": use_case,
        "instrumentation_match": {
            "ok": bool(instrumentation_match.get("ok")),
            "schema": instrumentation_match.get("schema"),
        },
        "quality_gate": {
            "ok": bool(quality_gate.get("ok")),
            "schema": quality_gate.get("schema"),
        } if quality_gate is not None else None,
        "samples": len(events),
        "source_hash": instrumentation_inspection.get("source_hash")
        or corpus.get("source_hash"),
        "top_name": instrumentation_inspection.get("top_name") or corpus.get("top_name"),
        "program_id": instrumentation_inspection.get("program_id"),
        "artifacts": artifacts,
        "readiness": readiness,
        "counters": counters,
        "errors": errors,
        "warnings": warnings,
    }


def inspect_flow_bundle(bundle_path: Path) -> dict[str, Any]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schema") != FLOW_BUNDLE_SCHEMA:
        raise ValueError(f"unsupported flow bundle schema {bundle.get('schema')!r}")
    return bundle


def inspect_quality_gate(
    *,
    bundle_path: Path,
    thresholds_path: Path | None = None,
) -> dict[str, Any]:
    bundle = inspect_flow_bundle(bundle_path)
    thresholds = dict(DEFAULT_QUALITY_THRESHOLDS)
    if thresholds_path is not None:
        overrides = json.loads(thresholds_path.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ValueError("quality thresholds must be a JSON object")
        thresholds.update(overrides)
    return build_quality_gate_report(
        bundle=bundle,
        bundle_path=bundle_path,
        thresholds=thresholds,
    )


def build_quality_gate_report(
    *,
    bundle: dict[str, Any],
    bundle_path: Path,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thresholds = dict(DEFAULT_QUALITY_THRESHOLDS if thresholds is None else thresholds)
    errors: list[str] = []
    warnings: list[str] = []
    artifacts = bundle.get("artifacts") if isinstance(bundle.get("artifacts"), dict) else {}
    reports: dict[str, Any] = {}
    for name in (
        "training_summary",
        "validate",
        "shadow",
        "fast_run",
        "model_fast_report",
        "runtime_execution",
    ):
        path = artifacts.get(name)
        if not path:
            if name == "training_summary":
                warnings.append("training_summary artifact not present")
            else:
                errors.append(f"{name} artifact not present")
            continue
        try:
            reports[name] = _read_bundle_artifact_json(bundle_path, str(path))
        except FileNotFoundError:
            errors.append(f"{name} artifact not found: {path}")
        except json.JSONDecodeError as err:
            errors.append(f"{name} artifact is not valid JSON: {err}")

    checks: dict[str, Any] = {}
    training = reports.get("training_summary")
    if isinstance(training, dict):
        training_accuracy = _metrics_accuracy(training)
        checks["training_accuracy"] = {"observed": training_accuracy}

    validation_accuracy = _metrics_accuracy(reports.get("validate"))
    checks["validation_accuracy"] = _threshold_min_check(
        validation_accuracy,
        float(thresholds["min_validation_accuracy"]),
    )
    if not checks["validation_accuracy"]["ok"]:
        errors.append(
            "validation accuracy below threshold: "
            f"{validation_accuracy} < {thresholds['min_validation_accuracy']}"
        )

    shadow_accuracy = _metrics_accuracy(reports.get("shadow"))
    checks["shadow_accuracy"] = _threshold_min_check(
        shadow_accuracy,
        float(thresholds["min_shadow_accuracy"]),
    )
    if not checks["shadow_accuracy"]["ok"]:
        errors.append(
            "shadow accuracy below threshold: "
            f"{shadow_accuracy} < {thresholds['min_shadow_accuracy']}"
        )

    fast_run = reports.get("fast_run") if isinstance(reports.get("fast_run"), dict) else {}
    fail_closed = int(fast_run.get("fail_closed", 0) or 0)
    shadow_failed = int(fast_run.get("shadow_failed", 0) or 0)
    checks["fail_closed"] = _threshold_max_check(
        fail_closed,
        int(thresholds["max_fail_closed"]),
    )
    checks["shadow_failed"] = _threshold_max_check(
        shadow_failed,
        int(thresholds["max_shadow_failed"]),
    )
    if not checks["fail_closed"]["ok"]:
        errors.append(
            f"fail_closed above threshold: {fail_closed} > {thresholds['max_fail_closed']}"
        )
    if not checks["shadow_failed"]["ok"]:
        errors.append(
            f"shadow_failed above threshold: {shadow_failed} > {thresholds['max_shadow_failed']}"
        )

    model_fast = reports.get("model_fast_report")
    model_fast_ok = bool(model_fast.get("ok")) if isinstance(model_fast, dict) else False
    checks["model_fast_ok"] = {
        "ok": model_fast_ok or not bool(thresholds["require_model_fast_ok"]),
        "observed": model_fast_ok,
        "required": bool(thresholds["require_model_fast_ok"]),
    }
    if not checks["model_fast_ok"]["ok"]:
        errors.append("model FAST report is not ok")

    runtime_execution = reports.get("runtime_execution")
    runtime_ready = (
        bool(runtime_execution.get("ready")) if isinstance(runtime_execution, dict) else False
    )
    checks["runtime_ready"] = {
        "ok": runtime_ready or not bool(thresholds["require_runtime_ready"]),
        "observed": runtime_ready,
        "required": bool(thresholds["require_runtime_ready"]),
    }
    if not checks["runtime_ready"]["ok"]:
        errors.append("runtime execution is not ready")

    return {
        "schema": QUALITY_GATE_SCHEMA,
        "ok": not errors,
        "bundle": str(bundle_path),
        "target": bundle.get("target"),
        "thresholds": thresholds,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


def _read_bundle_artifact_json(bundle_path: Path, raw_path: str) -> dict[str, Any]:
    path = _resolve_bundle_artifact_path(bundle_path, raw_path)
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_bundle_artifact_path(bundle_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else [path, bundle_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(raw_path)


def _metrics_accuracy(report: Any) -> float | None:
    if not isinstance(report, dict):
        return None
    metrics = report.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("accuracy") is None:
        return None
    return float(metrics["accuracy"])


def _threshold_min_check(value: float | None, threshold: float) -> dict[str, Any]:
    return {
        "ok": value is not None and value >= threshold,
        "observed": value,
        "threshold": threshold,
    }


def _threshold_max_check(value: int, threshold: int) -> dict[str, Any]:
    return {
        "ok": value <= threshold,
        "observed": value,
        "threshold": threshold,
    }


def package_runtime_handoff(*, bundle_path: Path) -> dict[str, Any]:
    bundle = inspect_flow_bundle(bundle_path)
    return build_runtime_handoff(bundle=bundle, bundle_path=bundle_path)


def build_runtime_handoff(*, bundle: dict[str, Any], bundle_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    artifacts = bundle.get("artifacts") if isinstance(bundle.get("artifacts"), dict) else {}
    required = (
        "manifest",
        "runtime_plan",
        "runtime_attachment",
        "runtime_execution",
        "quality_gate",
        "model_fast_report",
    )
    resolved: dict[str, str] = {}
    loaded: dict[str, dict[str, Any]] = {}
    for name in required:
        raw_path = artifacts.get(name)
        if not raw_path:
            errors.append(f"{name} artifact not present")
            continue
        try:
            resolved_path = _resolve_bundle_artifact_path(bundle_path, str(raw_path))
            resolved[name] = str(resolved_path)
            if name in ("manifest", "runtime_execution", "quality_gate", "model_fast_report"):
                loaded[name] = json.loads(resolved_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            errors.append(f"{name} artifact not found: {raw_path}")
        except json.JSONDecodeError as err:
            errors.append(f"{name} artifact is not valid JSON: {err}")

    if not bundle.get("ok"):
        errors.append("flow bundle is not ok")
    readiness = bundle.get("readiness") if isinstance(bundle.get("readiness"), dict) else {}
    if not readiness.get("quality_gate"):
        errors.append("flow bundle quality_gate readiness is not true")

    quality_gate = loaded.get("quality_gate", {})
    if not quality_gate.get("ok"):
        errors.append("quality gate is not ok")
    runtime_execution = loaded.get("runtime_execution", {})
    if not runtime_execution.get("ready"):
        errors.append("runtime execution is not ready")
    model_fast = loaded.get("model_fast_report", {})
    if not model_fast.get("ok"):
        errors.append("model FAST report is not ok")

    manifest = loaded.get("manifest", {})
    manifest_artifact = manifest.get("artifact") if isinstance(manifest.get("artifact"), dict) else {}
    manifest_source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    counters = bundle.get("counters") if isinstance(bundle.get("counters"), dict) else {}
    return {
        "schema": RUNTIME_HANDOFF_SCHEMA,
        "ok": not errors,
        "bundle": str(bundle_path),
        "target": bundle.get("target"),
        "model": bundle.get("model"),
        "use_case": bundle.get("use_case"),
        "manifest": {
            "path": resolved.get("manifest"),
            "surrogate_id": manifest.get("surrogate_id"),
            "surrogate_class": manifest.get("surrogate_class"),
            "model_family": manifest.get("model_family"),
            "artifact_hash": manifest_artifact.get("sha256"),
            "source_hash": manifest_source.get("source_hash"),
        },
        "runtime": {
            "plan": resolved.get("runtime_plan"),
            "attachment": resolved.get("runtime_attachment"),
            "execution": resolved.get("runtime_execution"),
            "ready": bool(runtime_execution.get("ready")),
            "attached_items": runtime_execution.get("attached_items"),
        },
        "acceptance": {
            "quality_gate": resolved.get("quality_gate"),
            "quality_ok": bool(quality_gate.get("ok")),
            "model_fast_report": resolved.get("model_fast_report"),
            "model_fast_ok": bool(model_fast.get("ok")),
            "readiness": readiness,
        },
        "counters": {
            "samples": bundle.get("samples"),
            "lanes": counters.get("lanes"),
            "surrogate_replacements": counters.get("surrogate_replacements"),
            "exact_fallbacks": counters.get("exact_fallbacks"),
            "fail_closed": counters.get("fail_closed"),
            "shadow_sampled": counters.get("shadow_sampled"),
            "shadow_passed": counters.get("shadow_passed"),
            "shadow_failed": counters.get("shadow_failed"),
        },
        "artifacts": resolved,
        "errors": errors,
        "warnings": warnings,
    }


def inspect_runtime_telemetry_gate(
    *,
    bundle_path: Path,
    telemetry_path: Path,
) -> dict[str, Any]:
    bundle = inspect_flow_bundle(bundle_path)
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    surrogate_execution = telemetry.get("surrogate_execution")
    errors: list[str] = []
    warnings: list[str] = []

    if not bundle.get("ok"):
        errors.append("flow bundle is not ok")
    if not isinstance(surrogate_execution, dict):
        errors.append("telemetry missing surrogate_execution")
        surrogate_execution = {}

    runtime_execution = _load_flow_bundle_runtime_execution(bundle, bundle_path, warnings)
    expected = _runtime_telemetry_expected_counters(bundle, runtime_execution)
    observed = _runtime_telemetry_observed_counters(surrogate_execution)
    counter_matches: dict[str, bool] = {}
    for key, expected_value in expected.items():
        observed_value = observed.get(key)
        matched = observed_value == expected_value
        counter_matches[key] = matched
        if not matched:
            errors.append(
                f"{key} mismatch: expected {expected_value}, observed {observed_value}"
            )

    runtime_ready = bool(surrogate_execution.get("ready"))
    if not runtime_ready:
        errors.append("surrogate_execution is not ready")

    expected_plan_schema = (
        runtime_execution.get("plan_schema") if isinstance(runtime_execution, dict) else None
    )
    observed_plan_schema = surrogate_execution.get("plan_schema")
    plan_schema_match: bool | None = None
    if expected_plan_schema:
        plan_schema_match = observed_plan_schema == expected_plan_schema
        if not plan_schema_match:
            errors.append(
                "plan_schema mismatch: "
                f"expected {expected_plan_schema!r}, observed {observed_plan_schema!r}"
            )
    elif observed_plan_schema is None:
        warnings.append("telemetry surrogate_execution has no plan_schema")

    diagnostics = surrogate_execution.get("diagnostics")
    if isinstance(diagnostics, list):
        warnings.extend(f"runtime diagnostic: {item}" for item in diagnostics if str(item))

    return {
        "schema": RUNTIME_TELEMETRY_GATE_SCHEMA,
        "ok": not errors,
        "bundle": str(bundle_path),
        "telemetry": str(telemetry_path),
        "bundle_ok": bool(bundle.get("ok")),
        "telemetry_has_surrogate_execution": bool(surrogate_execution),
        "runtime_ready": runtime_ready,
        "plan_schema_match": plan_schema_match,
        "expected": expected,
        "observed": observed,
        "counter_matches": counter_matches,
        "errors": errors,
        "warnings": warnings,
    }


def inspect_runtime_handoff_telemetry_gate(
    *,
    handoff_path: Path,
    telemetry_path: Path,
) -> dict[str, Any]:
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    surrogate_execution = telemetry.get("surrogate_execution")
    errors: list[str] = []
    warnings: list[str] = []

    if handoff.get("schema") != RUNTIME_HANDOFF_SCHEMA:
        errors.append("handoff schema is not rrtl-surrogate-runtime-handoff-v1")
    if not handoff.get("ok"):
        errors.append("runtime handoff is not ok")

    runtime = handoff.get("runtime") if isinstance(handoff.get("runtime"), dict) else {}
    handoff_runtime_ready = bool(runtime.get("ready"))
    if not handoff_runtime_ready:
        errors.append("handoff runtime is not ready")

    if not isinstance(surrogate_execution, dict):
        errors.append("telemetry missing surrogate_execution")
        surrogate_execution = {}

    runtime_execution = _load_handoff_runtime_execution(handoff, handoff_path, warnings)
    expected = _runtime_telemetry_expected_counters_from_handoff(
        handoff,
        runtime_execution,
    )
    observed = _runtime_telemetry_observed_counters(surrogate_execution)
    counter_matches: dict[str, bool] = {}
    for key, expected_value in expected.items():
        observed_value = observed.get(key)
        matched = observed_value == expected_value
        counter_matches[key] = matched
        if not matched:
            errors.append(
                f"{key} mismatch: expected {expected_value}, observed {observed_value}"
            )

    runtime_ready = bool(surrogate_execution.get("ready"))
    if not runtime_ready:
        errors.append("surrogate_execution is not ready")

    expected_plan_schema = (
        runtime_execution.get("plan_schema") if isinstance(runtime_execution, dict) else None
    )
    observed_plan_schema = surrogate_execution.get("plan_schema")
    plan_schema_match: bool | None = None
    if expected_plan_schema:
        plan_schema_match = observed_plan_schema == expected_plan_schema
        if not plan_schema_match:
            errors.append(
                "plan_schema mismatch: "
                f"expected {expected_plan_schema!r}, observed {observed_plan_schema!r}"
            )
    elif observed_plan_schema is None:
        warnings.append("telemetry surrogate_execution has no plan_schema")

    diagnostics = surrogate_execution.get("diagnostics")
    if isinstance(diagnostics, list):
        warnings.extend(f"runtime diagnostic: {item}" for item in diagnostics if str(item))

    return {
        "schema": RUNTIME_HANDOFF_TELEMETRY_GATE_SCHEMA,
        "ok": not errors,
        "handoff": str(handoff_path),
        "telemetry": str(telemetry_path),
        "handoff_ok": bool(handoff.get("ok")),
        "handoff_runtime_ready": handoff_runtime_ready,
        "telemetry_has_surrogate_execution": bool(surrogate_execution),
        "runtime_ready": runtime_ready,
        "plan_schema_match": plan_schema_match,
        "expected": expected,
        "observed": observed,
        "counter_matches": counter_matches,
        "errors": errors,
        "warnings": warnings,
    }


def _load_handoff_runtime_execution(
    handoff: dict[str, Any],
    handoff_path: Path,
    warnings: list[str],
) -> dict[str, Any]:
    runtime = handoff.get("runtime") if isinstance(handoff.get("runtime"), dict) else {}
    artifacts = handoff.get("artifacts") if isinstance(handoff.get("artifacts"), dict) else {}
    raw_path = runtime.get("execution") or artifacts.get("runtime_execution")
    if not raw_path:
        return {}
    try:
        path = _resolve_handoff_artifact_path(handoff_path, str(raw_path))
        report = json.loads(path.read_text(encoding="utf-8"))
        return report if isinstance(report, dict) else {}
    except FileNotFoundError:
        warnings.append(f"runtime_execution artifact not found: {raw_path}")
    except json.JSONDecodeError as err:
        warnings.append(f"runtime_execution artifact is not valid JSON: {err}")
    return {}


def _resolve_handoff_artifact_path(handoff_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else [path, handoff_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(raw_path)


def _load_flow_bundle_runtime_execution(
    bundle: dict[str, Any],
    bundle_path: Path,
    warnings: list[str],
) -> dict[str, Any]:
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, dict):
        return {}
    raw_path = artifacts.get("runtime_execution")
    if not raw_path:
        return {}
    try:
        report = _read_bundle_artifact_json(bundle_path, str(raw_path))
        return report if isinstance(report, dict) else {}
    except FileNotFoundError:
        pass
    warnings.append(f"runtime_execution artifact not found: {raw_path}")
    return {}


def _runtime_telemetry_expected_counters(
    bundle: dict[str, Any],
    runtime_execution: dict[str, Any],
) -> dict[str, int]:
    counters = bundle.get("counters") if isinstance(bundle.get("counters"), dict) else {}
    expected: dict[str, int] = {}
    for key in (
        "total_lanes",
        "attached_items",
        "surrogate_eligible_items",
        "exact_fallback_items",
        "invalid_items",
        "shadow_compared_items",
        "shadow_passed_items",
        "shadow_failed_items",
        "shadow_unavailable_items",
    ):
        if key in runtime_execution:
            expected[key] = int(runtime_execution.get(key) or 0)

    if "total_lanes" not in expected and "lanes" in counters:
        expected["total_lanes"] = int(counters.get("lanes") or 0)
    if "attached_items" not in expected and "samples" in bundle:
        expected["attached_items"] = int(bundle.get("samples") or 0)
    if "surrogate_eligible_items" not in expected and "surrogate_replacements" in counters:
        expected["surrogate_eligible_items"] = int(
            counters.get("surrogate_replacements") or 0
        )
    if "exact_fallback_items" not in expected and "exact_fallbacks" in counters:
        expected["exact_fallback_items"] = int(counters.get("exact_fallbacks") or 0)
    if "shadow_passed_items" not in expected and "shadow_passed" in counters:
        expected["shadow_passed_items"] = int(counters.get("shadow_passed") or 0)
    if "shadow_failed_items" not in expected and "shadow_failed" in counters:
        expected["shadow_failed_items"] = int(counters.get("shadow_failed") or 0)
    if "shadow_compared_items" not in expected:
        if "shadow_sampled" in counters:
            expected["shadow_compared_items"] = int(counters.get("shadow_sampled") or 0)
        elif "shadow_passed_items" in expected or "shadow_failed_items" in expected:
            expected["shadow_compared_items"] = expected.get(
                "shadow_passed_items",
                0,
            ) + expected.get("shadow_failed_items", 0)
    return expected


def _runtime_telemetry_expected_counters_from_handoff(
    handoff: dict[str, Any],
    runtime_execution: dict[str, Any],
) -> dict[str, int]:
    handoff_counters = (
        handoff.get("counters") if isinstance(handoff.get("counters"), dict) else {}
    )
    counter_map = {
        "lanes": handoff_counters.get("lanes"),
        "surrogate_replacements": handoff_counters.get("surrogate_replacements"),
        "exact_fallbacks": handoff_counters.get("exact_fallbacks"),
        "shadow_sampled": handoff_counters.get("shadow_sampled"),
        "shadow_passed": handoff_counters.get("shadow_passed"),
        "shadow_failed": handoff_counters.get("shadow_failed"),
    }
    bundle_like = {
        "samples": handoff_counters.get("samples"),
        "counters": {key: value for key, value in counter_map.items() if value is not None},
    }
    runtime = handoff.get("runtime") if isinstance(handoff.get("runtime"), dict) else {}
    if runtime.get("attached_items") is not None:
        runtime_execution = dict(runtime_execution)
        runtime_execution.setdefault("attached_items", runtime.get("attached_items"))
    return _runtime_telemetry_expected_counters(bundle_like, runtime_execution)


def _runtime_telemetry_observed_counters(
    surrogate_execution: dict[str, Any],
) -> dict[str, int | None]:
    keys = (
        "total_lanes",
        "attached_items",
        "surrogate_eligible_items",
        "exact_fallback_items",
        "invalid_items",
        "shadow_compared_items",
        "shadow_passed_items",
        "shadow_failed_items",
        "shadow_unavailable_items",
    )
    observed: dict[str, int | None] = {}
    for key in keys:
        value = surrogate_execution.get(key)
        observed[key] = int(value) if value is not None else None
    return observed


def _collect_report_messages(prefix: str, report: dict[str, Any], key: str) -> list[str]:
    messages = report.get(key, [])
    if not isinstance(messages, list):
        return []
    return [f"{prefix}: {message}" for message in messages if str(message)]


def run_instrumented_flow(
    *,
    trace_path: Path,
    config_path: Path,
    out_dir: Path,
    model: str = "rule",
    pyrtl2rrtl: list[str] | None = None,
    seed: int = 1,
    epochs: int = 4,
    target: str | None = "cache_miss",
    profile_path: Path | None = None,
) -> dict[str, Any]:
    """Run the instrumented trace through corpus, runtime, and model-FAST artifacts."""

    if model not in ("rule", "learned"):
        raise ValueError("model must be 'rule' or 'learned'")
    profile = load_event_profile(target=target, profile_path=profile_path)
    selected_target = profile["target"]
    use_case = build_use_case_contract(profile)
    cli = pyrtl2rrtl or [
        "cargo",
        "run",
        "-q",
        "-p",
        "rrtl-pyrtl",
        "--bin",
        "pyrtl2rrtl",
        "--",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)

    instrumentation_inspection_path = out_dir / "instrumentation_inspection.json"
    instrumentation_match_path = out_dir / "instrumentation_use_case_match.json"
    corpus_path = out_dir / "events.json"
    inspection_path = out_dir / "inspection.json"
    tensor_bundle_path = out_dir / "tensor_bundle.json"
    trainer_manifest_path = out_dir / "trainer_manifest.json"
    model_dir = out_dir / f"{model}_model"
    validate_path = out_dir / "validate.json"
    shadow_path = out_dir / "shadow.json"
    policy_path = out_dir / "policy.json"
    runtime_plan_path = out_dir / "runtime_plan.json"
    fast_run_path = out_dir / "fast_run.json"
    golden_path = out_dir / "cache0_golden.json"
    model_fast_plan_path = out_dir / "model_fast_plan.json"
    model_fast_report_path = out_dir / "model_fast_report.json"
    runtime_attachment_path = out_dir / "runtime_attachment.json"
    runtime_execution_path = out_dir / "runtime_execution.json"
    quality_gate_path = out_dir / "quality_gate.json"
    runtime_handoff_path = out_dir / "runtime_handoff.json"
    flow_bundle_path = out_dir / "flow_bundle.json"
    summary_path = out_dir / "summary.json"

    instrumentation_inspection = inspect_instrumentation_with_rrtl(
        trace_path=trace_path,
        config_path=config_path,
        pyrtl2rrtl=cli,
        out_path=instrumentation_inspection_path,
    )
    if not instrumentation_inspection.get("ok"):
        raise RuntimeError(
            f"inspect-instrumentation failed: {instrumentation_inspection.get('errors')}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    instrumentation_match = build_instrumentation_use_case_match(
        config=config,
        use_case=use_case,
        instrumentation_inspection=instrumentation_inspection,
    )
    instrumentation_match_path.write_text(
        json.dumps(instrumentation_match, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not instrumentation_match.get("ok"):
        raise RuntimeError(
            f"instrumentation use-case match failed: {instrumentation_match.get('errors')}"
        )

    corpus = emit_instrumented_with_rrtl(
        trace_path=trace_path,
        config_path=config_path,
        pyrtl2rrtl=cli,
        out_path=corpus_path,
    )
    if corpus.get("schema") != CORPUS_SCHEMA or corpus.get("errors"):
        raise RuntimeError(f"emit-instrumented failed: {corpus.get('errors')}")

    inspection = inspect_with_rrtl(
        corpus_path=corpus_path,
        pyrtl2rrtl=cli,
        out_path=inspection_path,
    )
    if not inspection.get("ok"):
        raise RuntimeError(f"inspect-events failed: {inspection.get('errors')}")

    tensor_bundle = write_tensor_bundle(
        corpus_path=corpus_path,
        out_path=tensor_bundle_path,
        target=selected_target,
        profile_path=profile_path,
    )
    trainer_manifest = build_event_trainer_manifest(tensor_bundle)
    trainer_manifest_path.write_text(
        json.dumps(trainer_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if model == "learned":
        train_summary = train_learned_and_export(
            corpus_path=corpus_path,
            out_dir=model_dir,
            epochs=epochs,
            seed=seed,
            target=selected_target,
            profile_path=profile_path,
        )
    else:
        train_summary = train_and_export(
            corpus_path=corpus_path,
            out_dir=model_dir,
            seed=seed,
            target=selected_target,
            profile_path=profile_path,
        )
    manifest_path = model_dir / "manifest.json"

    validation = validate_with_rrtl(
        manifest_path=manifest_path,
        corpus_path=corpus_path,
        pyrtl2rrtl=cli,
        out_path=validate_path,
    )
    if not validation.get("ok"):
        raise RuntimeError(f"validate-events failed: {validation.get('errors')}")

    shadow = shadow_with_rrtl(
        manifest_path=manifest_path,
        corpus_path=corpus_path,
        pyrtl2rrtl=cli,
        out_path=shadow_path,
    )
    if not shadow.get("ok"):
        raise RuntimeError(f"shadow-events failed: {shadow.get('errors')}")

    _run_rrtl_json_command(
        [
            *cli,
            "surrogate",
            "policy-events",
            str(manifest_path),
            str(corpus_path),
            "--out",
            str(policy_path),
        ],
        out_path=policy_path,
    )
    _run_rrtl_json_command(
        [
            *cli,
            "surrogate",
            "plan-runtime-events",
            str(policy_path),
            "--worker",
            "cpu-a:1",
            "--worker",
            "cpu-b:1",
            "--out",
            str(runtime_plan_path),
        ],
        out_path=runtime_plan_path,
    )
    fast_run = _run_rrtl_json_command(
        [
            *cli,
            "surrogate",
            "run-fast-events",
            str(manifest_path),
            str(corpus_path),
            str(runtime_plan_path),
            "--shadow-sample-stride",
            "2",
            "--out",
            str(fast_run_path),
        ],
        out_path=fast_run_path,
    )

    write_model_fast_golden(
        corpus_path=corpus_path,
        fast_run_path=fast_run_path,
        op_id="cache0",
        out_path=golden_path,
        target=selected_target,
        profile_path=profile_path,
    )
    _run_rrtl_json_command(
        [
            *cli,
            "surrogate",
            "plan-model-fast",
            "--op",
            f"cache0:{fast_run_path}:{selected_target} predictor",
            "--golden",
            f"cache0:{golden_path}",
            "--out",
            str(model_fast_plan_path),
        ],
        out_path=model_fast_plan_path,
    )
    model_fast_report = _run_rrtl_json_command(
        [
            *cli,
            "surrogate",
            "run-model-fast",
            str(model_fast_plan_path),
            "--out",
            str(model_fast_report_path),
        ],
        out_path=model_fast_report_path,
    )
    _run_rrtl_json_command(
        [
            *cli,
            "surrogate",
            "attach-runtime-events",
            str(runtime_plan_path),
            "--worker",
            "cpu-a:1",
            "--worker",
            "cpu-b:1",
            "--out",
            str(runtime_attachment_path),
        ],
        out_path=runtime_attachment_path,
    )
    runtime_execution = _run_rrtl_json_command(
        [
            *cli,
            "surrogate",
            "inspect-runtime-attachment",
            str(runtime_attachment_path),
            "--out",
            str(runtime_execution_path),
        ],
        out_path=runtime_execution_path,
    )

    artifacts = {
        "instrumentation_inspection": str(instrumentation_inspection_path),
        "instrumentation_use_case_match": str(instrumentation_match_path),
        "corpus": str(corpus_path),
        "inspection": str(inspection_path),
        "tensor_bundle": str(tensor_bundle_path),
        "trainer_manifest": str(trainer_manifest_path),
        "model_dir": str(model_dir),
        "training_summary": str(model_dir / "summary.json"),
        "manifest": str(manifest_path),
        "validate": str(validate_path),
        "shadow": str(shadow_path),
        "policy": str(policy_path),
        "runtime_plan": str(runtime_plan_path),
        "fast_run": str(fast_run_path),
        "golden": str(golden_path),
        "model_fast_plan": str(model_fast_plan_path),
        "model_fast_report": str(model_fast_report_path),
        "runtime_attachment": str(runtime_attachment_path),
        "runtime_execution": str(runtime_execution_path),
    }
    flow_bundle = build_flow_bundle(
        model=model,
        target=selected_target,
        use_case=use_case,
        instrumentation_match=instrumentation_match,
        artifacts=artifacts,
        instrumentation_inspection=instrumentation_inspection,
        corpus=corpus,
        inspection=inspection,
        validation=validation,
        shadow=shadow,
        fast_run=fast_run,
        runtime_execution=runtime_execution,
        model_fast_report=model_fast_report,
    )
    flow_bundle_path.write_text(
        json.dumps(flow_bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    quality_gate = inspect_quality_gate(bundle_path=flow_bundle_path)
    quality_gate_path.write_text(
        json.dumps(quality_gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts["quality_gate"] = str(quality_gate_path)
    artifacts["flow_bundle"] = str(flow_bundle_path)
    flow_bundle = build_flow_bundle(
        model=model,
        target=selected_target,
        use_case=use_case,
        instrumentation_match=instrumentation_match,
        quality_gate=quality_gate,
        artifacts=artifacts,
        instrumentation_inspection=instrumentation_inspection,
        corpus=corpus,
        inspection=inspection,
        validation=validation,
        shadow=shadow,
        fast_run=fast_run,
        runtime_execution=runtime_execution,
        model_fast_report=model_fast_report,
    )
    flow_bundle_path.write_text(
        json.dumps(flow_bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime_handoff = package_runtime_handoff(bundle_path=flow_bundle_path)
    runtime_handoff_path.write_text(
        json.dumps(runtime_handoff, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts["runtime_handoff"] = str(runtime_handoff_path)
    flow_bundle["artifacts"] = artifacts
    flow_bundle_path.write_text(
        json.dumps(flow_bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema": "rrtl-surrogate-instrumented-flow-summary-v1",
        "model": model,
        "target": selected_target,
        "ok": bool(flow_bundle.get("ok")),
        "samples": len(corpus.get("events", [])),
        "outputs": artifacts,
        "training": train_summary,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["outputs"]["summary"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _run_rrtl_json_command(command: list[str], *, out_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if out_path.exists():
        summary = json.loads(out_path.read_text(encoding="utf-8"))
    elif completed.stdout:
        summary = json.loads(completed.stdout)
    else:
        summary = {"errors": [completed.stderr.strip() or "command produced no JSON output"]}
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return summary


def inspect_with_rrtl(
    *,
    corpus_path: Path,
    pyrtl2rrtl: list[str],
    out_path: Path | None = None,
) -> dict[str, Any]:
    command = [
        *pyrtl2rrtl,
        "surrogate",
        "inspect-events",
        str(corpus_path),
    ]
    if out_path is not None:
        command.extend(["--out", str(out_path)])
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if out_path is not None and out_path.exists():
        summary = json.loads(out_path.read_text(encoding="utf-8"))
    elif completed.stdout:
        summary = json.loads(completed.stdout)
    else:
        summary = {
            "schema": "rrtl-surrogate-event-inspection-v1",
            "ok": False,
            "errors": [completed.stderr.strip() or "inspect-events produced no report"],
        }
    if completed.returncode != 0:
        errors = summary.setdefault("errors", [])
        if completed.stderr and completed.stderr.strip() not in errors:
            errors.append(completed.stderr.strip())
        summary["ok"] = False
    return summary


def emit_with_rrtl(
    *,
    trace_path: Path,
    config_path: Path,
    pyrtl2rrtl: list[str],
    out_path: Path | None = None,
) -> dict[str, Any]:
    command = [
        *pyrtl2rrtl,
        "surrogate",
        "emit-events",
        str(trace_path),
        "--config",
        str(config_path),
    ]
    if out_path is not None:
        command.extend(["--out", str(out_path)])
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if out_path is not None and out_path.exists():
        summary = json.loads(out_path.read_text(encoding="utf-8"))
    elif completed.stdout:
        summary = json.loads(completed.stdout)
    else:
        summary = {
            "schema": CORPUS_SCHEMA,
            "events": [],
            "errors": [completed.stderr.strip() or "emit-events produced no corpus"],
        }
    if completed.returncode != 0:
        errors = summary.setdefault("errors", [])
        if completed.stderr and completed.stderr.strip() not in errors:
            errors.append(completed.stderr.strip())
    return summary


def emit_instrumented_with_rrtl(
    *,
    trace_path: Path,
    config_path: Path,
    pyrtl2rrtl: list[str],
    out_path: Path | None = None,
) -> dict[str, Any]:
    command = [
        *pyrtl2rrtl,
        "surrogate",
        "emit-instrumented-events",
        str(trace_path),
        "--config",
        str(config_path),
    ]
    if out_path is not None:
        command.extend(["--out", str(out_path)])
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if out_path is not None and out_path.exists():
        summary = json.loads(out_path.read_text(encoding="utf-8"))
    elif completed.stdout:
        summary = json.loads(completed.stdout)
    else:
        summary = {
            "schema": CORPUS_SCHEMA,
            "events": [],
            "errors": [completed.stderr.strip() or "emit-instrumented-events produced no corpus"],
        }
    if completed.returncode != 0:
        errors = summary.setdefault("errors", [])
        if completed.stderr and completed.stderr.strip() not in errors:
            errors.append(completed.stderr.strip())
    return summary


def build_model_fast_golden(
    corpus: dict[str, Any],
    fast_run: dict[str, Any],
    *,
    op_id: str,
    op_kind: str = "event",
    target: str | None = "cache_miss",
    profile_path: Path | None = None,
) -> dict[str, Any]:
    """Build a model FAST golden from cache-miss labels and event FAST predictions."""

    profile = load_event_profile(target=target, profile_path=profile_path)
    target = profile["target"]
    if corpus.get("schema") != CORPUS_SCHEMA:
        raise ValueError(f"unsupported corpus schema {corpus.get('schema')!r}")
    if fast_run.get("schema") != EVENT_FAST_RUN_SCHEMA:
        raise ValueError(f"unsupported event FAST run schema {fast_run.get('schema')!r}")
    events = corpus.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("corpus.events must be a non-empty list")
    results = fast_run.get("results")
    if not isinstance(results, list):
        raise ValueError("fast_run.results must be a list")
    if int(fast_run.get("count", len(results))) != len(results):
        raise ValueError("fast_run.count does not match results length")
    if len(events) != len(results):
        raise ValueError(
            f"event count {len(events)} does not match FAST result count {len(results)}"
        )

    expected_tensor = []
    actual_tensor = []
    for index, (event, result) in enumerate(zip(events, results)):
        if event.get("schema") != EVENT_SCHEMA:
            raise ValueError(f"event {index} has unsupported schema {event.get('schema')!r}")
        if event.get("target") != target:
            raise ValueError(
                f"event {index} target {event.get('target')!r} does not match {target!r}"
            )
        if result.get("target") != target:
            raise ValueError(
                f"FAST result {index} target {result.get('target')!r} does not match {target!r}"
            )
        if int(event.get("sample_id", -1)) != int(result.get("sample_id", -2)):
            raise ValueError(
                f"sample_id mismatch at index {index}: event {event.get('sample_id')!r} "
                f"FAST result {result.get('sample_id')!r}"
            )
        label = event.get("label", {}).get(target)
        if label is None:
            raise ValueError(f"event {index} missing label {target!r}")
        predicted = result.get("predicted")
        if predicted is None:
            raise ValueError(f"FAST result {index} missing predicted value")
        label_int = _validate_binary_value(f"event {index} label", label)
        predicted_int = _validate_binary_value(f"FAST result {index} predicted", predicted)
        if result.get("expected") is not None:
            expected_int = _validate_binary_value(
                f"FAST result {index} expected",
                result["expected"],
            )
            if expected_int != label_int:
                raise ValueError(
                    f"FAST result {index} expected {expected_int} does not match label {label_int}"
                )
        expected_tensor.append([label_int])
        actual_tensor.append([predicted_int])

    return {
        "schema": MODEL_FAST_GOLDEN_SCHEMA,
        "op_id": op_id,
        "op_kind": op_kind,
        "expected": {
            "items": int(fast_run.get("count", len(results))),
            "surrogate_replacements": int(fast_run.get("surrogate_replacements", 0)),
            "exact_fallbacks": int(fast_run.get("exact_fallbacks", 0)),
            "fail_closed": int(fast_run.get("fail_closed", 0)),
            "shadow_sampled": sum(1 for item in results if item.get("shadow_sampled")),
        },
        "expected_tensors": {"predicted_event": expected_tensor},
        "actual_tensors": {"predicted_event": actual_tensor},
        "max_abs_error": 0,
    }


def write_model_fast_golden(
    *,
    corpus_path: Path,
    fast_run_path: Path,
    op_id: str,
    out_path: Path,
    op_kind: str = "event",
    target: str | None = "cache_miss",
    profile_path: Path | None = None,
) -> dict[str, Any]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    fast_run = json.loads(fast_run_path.read_text(encoding="utf-8"))
    golden = build_model_fast_golden(
        corpus,
        fast_run,
        op_id=op_id,
        op_kind=op_kind,
        target=target,
        profile_path=profile_path,
    )
    out_path.write_text(
        json.dumps(golden, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return golden


def build_manifest(
    corpus: dict[str, Any],
    *,
    artifact_path: Path,
    artifact_hash: str,
    tensor_manifest: dict[str, Any],
    profile_path: Path | None = None,
) -> dict[str, Any]:
    task = tensor_manifest["task"]
    profile = load_event_profile(target=task["prediction_target"], profile_path=profile_path)
    return {
        "schema": MANIFEST_SCHEMA,
        "surrogate_id": profile["surrogate_id"],
        "surrogate_class": "event_predictor",
        "model_family": "rule-baseline",
        "task": {
            "prediction_target": task["prediction_target"],
            "input_window_cycles": task["input_window_cycles"],
            "horizon_cycles": task["horizon_cycles"],
            "signal_features": tensor_manifest["inputs"]["signal_window"]["features"],
            "program_features": tensor_manifest["inputs"]["program_context"]["features"],
            "label": tensor_manifest["label"],
        },
        "source": {
            "top_name": corpus["top_name"],
            "export_schema": CORPUS_SCHEMA,
            "source_hash": corpus["source_hash"],
        },
        "artifact": {
            "format": "mock-event-predictor",
            "path": str(artifact_path),
            "sha256": artifact_hash,
            "input_tensors": ["signal_window", "program_context"],
            "output_tensors": ["event_probability", "predicted_event"],
        },
        "domain": {
            "rows": 1,
            "cols": 1,
            "k_min": 1,
            "k_max": task["input_window_cycles"],
            "data_width": 64,
            "acc_width": 64,
        },
        "validation": {
            "max_abs_error": 0,
            "max_mean_abs_error": 0.0,
            "max_latency_error_cycles": 0,
        },
        "policy": {
            "mode": "telemetry_only",
            "fallback": "fail_closed",
            "provenance_tag": "instrumentation_prediction",
        },
    }


def build_learned_manifest(
    corpus: dict[str, Any],
    *,
    artifact_path: Path,
    artifact_hash: str,
    tensor_manifest: dict[str, Any],
    profile_path: Path | None = None,
) -> dict[str, Any]:
    task = tensor_manifest["task"]
    profile = load_event_profile(target=task["prediction_target"], profile_path=profile_path)
    return {
        "schema": MANIFEST_SCHEMA,
        "surrogate_id": profile["surrogate_id"],
        "surrogate_class": "event_predictor",
        "model_family": "gnn-transformer",
        "task": {
            "prediction_target": task["prediction_target"],
            "input_window_cycles": task["input_window_cycles"],
            "horizon_cycles": task["horizon_cycles"],
            "signal_features": tensor_manifest["inputs"]["signal_window"]["features"],
            "program_features": tensor_manifest["inputs"]["program_context"]["features"],
            "label": tensor_manifest["label"],
        },
        "source": {
            "top_name": corpus["top_name"],
            "export_schema": CORPUS_SCHEMA,
            "source_hash": corpus["source_hash"],
        },
        "artifact": {
            "format": "onnx",
            "path": str(artifact_path),
            "sha256": artifact_hash,
            "input_tensors": ["signal_window", "program_context"],
            "output_tensors": ["event_probability", "predicted_event"],
            "opset": 17,
        },
        "domain": {
            "rows": 1,
            "cols": 1,
            "k_min": 1,
            "k_max": task["input_window_cycles"],
            "data_width": 64,
            "acc_width": 64,
        },
        "validation": {
            "max_abs_error": 0,
            "max_mean_abs_error": 0.0,
            "max_latency_error_cycles": 0,
        },
        "policy": {
            "mode": "telemetry_only",
            "fallback": "fail_closed",
            "provenance_tag": "instrumentation_prediction",
        },
    }


def evaluate_rule(
    corpus: dict[str, Any],
    *,
    target: str | None = "cache_miss",
    profile_path: Path | None = None,
) -> dict[str, Any]:
    profile = load_event_profile(target=target, profile_path=profile_path)
    events, _ = _validated_events(corpus, target=profile["target"])
    total = 0
    correct = 0
    false_positive = 0
    false_negative = 0
    for event in events:
        program = event["program"]
        predicted = predict_event_rule(profile, event["signals"], program)
        expected = int(event["label"][profile["label"]["name"]])
        total += 1
        correct += int(predicted == expected)
        false_positive += int(predicted == 1 and expected == 0)
        false_negative += int(predicted == 0 and expected == 1)
    return {
        "accuracy": correct / total if total else 0.0,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(data: Any) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def manifest_relative_path(*, artifact_path: Path, manifest_path: Path) -> Path:
    """Return a manifest artifact path resolved relative to the manifest file."""

    artifact_resolved = artifact_path.resolve()
    manifest_dir = manifest_path.parent.resolve()
    try:
        return artifact_resolved.relative_to(manifest_dir)
    except ValueError:
        return artifact_resolved


def build_use_case_contract(profile: dict[str, Any]) -> dict[str, Any]:
    _validate_event_profile(profile)
    label = profile["label"]
    return {
        "target": profile["target"],
        "surrogate_id": profile["surrogate_id"],
        "surrogate_class": "event_predictor",
        "signal_features": list(profile["signal_features"]),
        "program_features": list(profile["program_features"]),
        "label": {
            "name": label["name"],
            "kind": label["kind"],
            "positive_value": int(label["positive_value"]),
        },
        "input_window_cycles_default": int(profile["input_window_cycles_default"]),
        "horizon_cycles_default": int(profile["horizon_cycles_default"]),
        "supported_model_modes": ["rule", "learned"],
    }


def build_use_case_catalog(
    *,
    profile_dir: Path | None = None,
    profile_paths: list[Path] | None = None,
) -> dict[str, Any]:
    entries_by_target: dict[str, dict[str, Any]] = {}
    sources: list[str] = []
    for path in _catalog_profile_paths(profile_dir=profile_dir, profile_paths=profile_paths):
        profile = json.loads(path.read_text(encoding="utf-8"))
        contract = build_use_case_contract(profile)
        contract["source"] = str(path)
        entries_by_target[contract["target"]] = contract
        sources.append(str(path))
    entries = [entries_by_target[target] for target in sorted(entries_by_target)]
    return {
        "schema": USE_CASE_CATALOG_SCHEMA,
        "count": len(entries),
        "targets": [entry["target"] for entry in entries],
        "supported_model_modes": ["rule", "learned"],
        "sources": sources,
        "use_cases": entries,
    }


def _catalog_profile_paths(
    *,
    profile_dir: Path | None,
    profile_paths: list[Path] | None,
) -> list[Path]:
    selected_dir = profile_dir or EVENT_PROFILE_DIR
    paths = sorted(selected_dir.glob("*.json")) if selected_dir.exists() else []
    paths.extend(profile_paths or [])
    return paths


def load_event_profile(
    *,
    target: str | None = None,
    profile_path: Path | None = None,
) -> dict[str, Any]:
    if profile_path is not None:
        profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        _validate_event_profile(profile)
        if target is not None and target != profile["target"]:
            raise ValueError(
                f"profile target {profile['target']!r} does not match requested target {target!r}"
            )
        return profile
    selected = target or "cache_miss"
    if selected not in BUILTIN_EVENT_TARGETS:
        raise ValueError(
            f"unknown event target {selected!r}; supported targets: {', '.join(BUILTIN_EVENT_TARGETS)}"
        )
    profile = json.loads((EVENT_PROFILE_DIR / f"{selected}.json").read_text(encoding="utf-8"))
    _validate_event_profile(profile)
    return profile


def event_profile(target: str) -> dict[str, Any]:
    return load_event_profile(target=target)


def _validate_event_profile(profile: dict[str, Any]) -> None:
    if profile.get("schema") != EVENT_PROFILE_SCHEMA:
        raise ValueError(f"unsupported event profile schema {profile.get('schema')!r}")
    for field in ("target", "surrogate_id"):
        if not isinstance(profile.get(field), str) or not profile[field].strip():
            raise ValueError(f"event profile {field} must be a non-empty string")
    _validate_feature_names(profile.get("signal_features"), "signal_features")
    _validate_feature_names(profile.get("program_features"), "program_features")
    label = profile.get("label")
    if not isinstance(label, dict):
        raise ValueError("event profile label must be an object")
    if not isinstance(label.get("name"), str) or not label["name"].strip():
        raise ValueError("event profile label.name must be a non-empty string")
    if label.get("kind") != "binary":
        raise ValueError("event profile label.kind must be 'binary'")
    if int(label.get("positive_value", -1)) != 1:
        raise ValueError("event profile label.positive_value must be 1")
    for field in ("input_window_cycles_default", "horizon_cycles_default"):
        if not isinstance(profile.get(field), int) or profile[field] <= 0:
            raise ValueError(f"event profile {field} must be a positive integer")
    _validate_mock_rule(profile.get("mock_rule"))


def _validate_feature_names(value: Any, field: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"event profile {field} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"event profile {field} entries must be non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"event profile {field} must not contain duplicates")


def _validate_mock_rule(rule: Any) -> None:
    if not isinstance(rule, dict):
        raise ValueError("event profile mock_rule must be an object")
    if rule.get("kind") != "linear_threshold":
        raise ValueError("event profile mock_rule.kind must be 'linear_threshold'")
    if not isinstance(rule.get("threshold"), int):
        raise ValueError("event profile mock_rule.threshold must be an integer")
    terms = rule.get("terms")
    if not isinstance(terms, list) or not terms:
        raise ValueError("event profile mock_rule.terms must be a non-empty list")
    for term in terms:
        _validate_rule_value_spec(term, "term")
        if not isinstance(term.get("weight"), int):
            raise ValueError("event profile mock_rule term weight must be an integer")
    bonuses = rule.get("bonuses", [])
    if not isinstance(bonuses, list):
        raise ValueError("event profile mock_rule.bonuses must be a list")
    for bonus in bonuses:
        _validate_rule_value_spec(bonus, "bonus")
        if bonus.get("op") not in (">=", ">", "<=", "<", "==", "=", "!="):
            raise ValueError("event profile mock_rule bonus op is unsupported")
        if not isinstance(bonus.get("value"), int) or not isinstance(bonus.get("bonus"), int):
            raise ValueError("event profile mock_rule bonus value and bonus must be integers")


def _validate_rule_value_spec(spec: Any, context: str) -> None:
    if not isinstance(spec, dict):
        raise ValueError(f"event profile mock_rule {context} must be an object")
    if spec.get("source") not in ("signal", "program"):
        raise ValueError(f"event profile mock_rule {context} source is unsupported")
    if not isinstance(spec.get("feature"), str) or not spec["feature"].strip():
        raise ValueError(f"event profile mock_rule {context} feature must be non-empty")
    if spec.get("source") == "signal":
        reduction = spec.get("reduction", "sum")
        if reduction not in ("sum", "max", "min", "last", "unique_count"):
            raise ValueError(f"event profile mock_rule {context} reduction is unsupported")


def _validate_nonnegative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate synthetic instrumentation event windows")
    gen.add_argument("--out", required=True)
    gen.add_argument("--samples", type=int, default=32)
    gen.add_argument("--window", type=int, default=8)
    gen.add_argument("--seed", type=int, default=1)
    gen.add_argument("--sets", type=int, default=16)
    gen.add_argument("--lanes", type=int, default=1)
    gen.add_argument("--target")
    gen.add_argument("--profile")

    tensors = sub.add_parser("tensors", help="write tensor metadata for an event corpus")
    tensors.add_argument("--corpus", required=True)
    tensors.add_argument("--out", required=True)
    tensors.add_argument("--target")
    tensors.add_argument("--profile")

    export_tensors = sub.add_parser(
        "export-tensors",
        help="write tensor values and metadata for an event corpus",
    )
    export_tensors.add_argument("--corpus", required=True)
    export_tensors.add_argument("--out", required=True)
    export_tensors.add_argument("--target")
    export_tensors.add_argument("--profile")

    catalog = sub.add_parser(
        "catalog-use-cases",
        help="write a catalog of surrogate event use-case contracts",
    )
    catalog.add_argument("--out", required=True)
    catalog.add_argument("--profile-dir")
    catalog.add_argument("--profile", action="append", default=[])

    emit = sub.add_parser("emit", help="emit event corpus from trace through pyrtl2rrtl")
    emit.add_argument("--trace", required=True)
    emit.add_argument("--config", required=True)
    emit.add_argument("--out", required=True)
    emit.add_argument(
        "--pyrtl2rrtl",
        nargs="+",
        default=[
            "cargo",
            "run",
            "-q",
            "-p",
            "rrtl-pyrtl",
            "--bin",
            "pyrtl2rrtl",
            "--",
        ],
    )

    emit_instrumented = sub.add_parser(
        "emit-instrumented",
        help="emit event corpus from RRTL instrumentation trace through pyrtl2rrtl",
    )
    emit_instrumented.add_argument("--trace", required=True)
    emit_instrumented.add_argument("--config", required=True)
    emit_instrumented.add_argument("--out", required=True)
    emit_instrumented.add_argument(
        "--pyrtl2rrtl",
        nargs="+",
        default=[
            "cargo",
            "run",
            "-q",
            "-p",
            "rrtl-pyrtl",
            "--bin",
            "pyrtl2rrtl",
            "--",
        ],
    )

    inspect_instrumentation = sub.add_parser(
        "inspect-instrumentation",
        help="inspect RRTL instrumentation trace compatibility through pyrtl2rrtl",
    )
    inspect_instrumentation.add_argument("--trace", required=True)
    inspect_instrumentation.add_argument("--config")
    inspect_instrumentation.add_argument("--out")
    inspect_instrumentation.add_argument(
        "--pyrtl2rrtl",
        nargs="+",
        default=[
            "cargo",
            "run",
            "-q",
            "-p",
            "rrtl-pyrtl",
            "--bin",
            "pyrtl2rrtl",
            "--",
        ],
    )

    match_instrumentation = sub.add_parser(
        "match-instrumentation-use-case",
        help="validate an instrumentation trace/config against a surrogate use-case profile",
    )
    match_instrumentation.add_argument("--trace", required=True)
    match_instrumentation.add_argument("--config", required=True)
    match_instrumentation.add_argument("--target")
    match_instrumentation.add_argument("--profile")
    match_instrumentation.add_argument("--out")
    match_instrumentation.add_argument(
        "--pyrtl2rrtl",
        nargs="+",
        default=[
            "cargo",
            "run",
            "-q",
            "-p",
            "rrtl-pyrtl",
            "--bin",
            "pyrtl2rrtl",
            "--",
        ],
    )

    instrumented_flow = sub.add_parser(
        "instrumented-flow",
        help="run RRTL instrumentation through event training, FAST, and runtime attachment artifacts",
    )
    instrumented_flow.add_argument("--trace", required=True)
    instrumented_flow.add_argument("--config", required=True)
    instrumented_flow.add_argument("--out-dir", required=True)
    instrumented_flow.add_argument("--model", choices=["rule", "learned"], default="rule")
    instrumented_flow.add_argument("--seed", type=int, default=1)
    instrumented_flow.add_argument("--epochs", type=int, default=4)
    instrumented_flow.add_argument("--target")
    instrumented_flow.add_argument("--profile")
    instrumented_flow.add_argument(
        "--pyrtl2rrtl",
        nargs="+",
        default=[
            "cargo",
            "run",
            "-q",
            "-p",
            "rrtl-pyrtl",
            "--bin",
            "pyrtl2rrtl",
            "--",
        ],
    )

    inspect_flow = sub.add_parser(
        "inspect-flow",
        help="inspect a surrogate flow bundle and fail when it is not ready",
    )
    inspect_flow.add_argument("--bundle", required=True)
    inspect_flow.add_argument("--out")

    quality_gate = sub.add_parser(
        "quality-gate",
        help="validate surrogate flow quality thresholds against bundle artifacts",
    )
    quality_gate.add_argument("--bundle", required=True)
    quality_gate.add_argument("--thresholds")
    quality_gate.add_argument("--out")

    package_handoff = sub.add_parser(
        "package-runtime-handoff",
        help="package an accepted surrogate flow bundle for runtime orchestration",
    )
    package_handoff.add_argument("--bundle", required=True)
    package_handoff.add_argument("--out")

    inspect_runtime_telemetry = sub.add_parser(
        "inspect-runtime-telemetry",
        help="validate runtime telemetry against a surrogate flow bundle",
    )
    inspect_runtime_telemetry.add_argument("--bundle", required=True)
    inspect_runtime_telemetry.add_argument("--telemetry", required=True)
    inspect_runtime_telemetry.add_argument("--out")

    inspect_runtime_handoff_telemetry = sub.add_parser(
        "inspect-runtime-handoff-telemetry",
        help="validate runtime telemetry against an accepted runtime handoff package",
    )
    inspect_runtime_handoff_telemetry.add_argument("--handoff", required=True)
    inspect_runtime_handoff_telemetry.add_argument("--telemetry", required=True)
    inspect_runtime_handoff_telemetry.add_argument("--out")

    inspect = sub.add_parser("inspect", help="inspect event corpus through pyrtl2rrtl")
    inspect.add_argument("--corpus", required=True)
    inspect.add_argument("--out")
    inspect.add_argument(
        "--pyrtl2rrtl",
        nargs="+",
        default=[
            "cargo",
            "run",
            "-q",
            "-p",
            "rrtl-pyrtl",
            "--bin",
            "pyrtl2rrtl",
            "--",
        ],
    )

    train = sub.add_parser("train", help="write rule artifact and event-predictor manifest")
    train.add_argument("--corpus", required=True)
    train.add_argument("--out-dir", required=True)
    train.add_argument("--seed", type=int, default=1)
    train.add_argument("--target")
    train.add_argument("--profile")

    train_learned = sub.add_parser(
        "train-learned",
        help="train ONNX GNN/Transformer event-predictor artifact and manifest",
    )
    train_learned.add_argument("--corpus", required=True)
    train_learned.add_argument("--out-dir", required=True)
    train_learned.add_argument("--epochs", type=int, default=4)
    train_learned.add_argument("--seed", type=int, default=1)
    train_learned.add_argument("--hidden-dim", type=int, default=16)
    train_learned.add_argument("--heads", type=int, default=2)
    train_learned.add_argument("--target")
    train_learned.add_argument("--profile")

    model_fast = sub.add_parser(
        "model-fast-golden",
        help="write a model FAST golden from an event corpus and FAST run",
    )
    model_fast.add_argument("--corpus", required=True)
    model_fast.add_argument("--fast-run", required=True)
    model_fast.add_argument("--op-id", required=True)
    model_fast.add_argument("--op-kind", default="event")
    model_fast.add_argument("--target")
    model_fast.add_argument("--profile")
    model_fast.add_argument("--out", required=True)

    val = sub.add_parser("validate", help="validate event corpus through pyrtl2rrtl")
    val.add_argument("--manifest", required=True)
    val.add_argument("--corpus", required=True)
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
            "--bin",
            "pyrtl2rrtl",
            "--",
        ],
    )

    shadow = sub.add_parser("shadow", help="shadow-compare event corpus through pyrtl2rrtl")
    shadow.add_argument("--manifest", required=True)
    shadow.add_argument("--corpus", required=True)
    shadow.add_argument("--out")
    shadow.add_argument(
        "--pyrtl2rrtl",
        nargs="+",
        default=[
            "cargo",
            "run",
            "-q",
            "-p",
            "rrtl-pyrtl",
            "--bin",
            "pyrtl2rrtl",
            "--",
        ],
    )

    args = parser.parse_args(argv)
    if args.command == "generate":
        corpus = generate_corpus(
            samples=args.samples,
            window=args.window,
            seed=args.seed,
            sets=args.sets,
            lanes=args.lanes,
            target=_cli_target(args),
            profile_path=Path(args.profile) if args.profile else None,
        )
        Path(args.out).write_text(
            json.dumps(corpus, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    if args.command == "tensors":
        corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
        manifest = build_tensor_manifest(
            corpus,
            target=_cli_target(args),
            profile_path=Path(args.profile) if args.profile else None,
        )
        Path(args.out).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    if args.command == "export-tensors":
        bundle = write_tensor_bundle(
            corpus_path=Path(args.corpus),
            out_path=Path(args.out),
            target=_cli_target(args),
            profile_path=Path(args.profile) if args.profile else None,
        )
        print(json.dumps(bundle, indent=2, sort_keys=True))
        return 0
    if args.command == "catalog-use-cases":
        catalog = build_use_case_catalog(
            profile_dir=Path(args.profile_dir) if args.profile_dir else None,
            profile_paths=[Path(path) for path in args.profile],
        )
        text = json.dumps(catalog, indent=2, sort_keys=True) + "\n"
        Path(args.out).write_text(text, encoding="utf-8")
        print(text, end="")
        return 0
    if args.command == "emit":
        corpus = emit_with_rrtl(
            trace_path=Path(args.trace),
            config_path=Path(args.config),
            pyrtl2rrtl=args.pyrtl2rrtl,
            out_path=Path(args.out),
        )
        print(json.dumps(corpus, indent=2, sort_keys=True))
        return 0 if corpus.get("schema") == CORPUS_SCHEMA and not corpus.get("errors") else 1
    if args.command == "emit-instrumented":
        corpus = emit_instrumented_with_rrtl(
            trace_path=Path(args.trace),
            config_path=Path(args.config),
            pyrtl2rrtl=args.pyrtl2rrtl,
            out_path=Path(args.out),
        )
        print(json.dumps(corpus, indent=2, sort_keys=True))
        return 0 if corpus.get("schema") == CORPUS_SCHEMA and not corpus.get("errors") else 1
    if args.command == "inspect-instrumentation":
        summary = inspect_instrumentation_with_rrtl(
            trace_path=Path(args.trace),
            config_path=Path(args.config) if args.config else None,
            pyrtl2rrtl=args.pyrtl2rrtl,
            out_path=Path(args.out) if args.out else None,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary.get("ok") else 1
    if args.command == "match-instrumentation-use-case":
        report = match_instrumentation_use_case(
            trace_path=Path(args.trace),
            config_path=Path(args.config),
            pyrtl2rrtl=args.pyrtl2rrtl,
            target=_cli_target(args),
            profile_path=Path(args.profile) if args.profile else None,
            out_path=Path(args.out) if args.out else None,
        )
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if not args.out:
            print(text, end="")
        return 0 if report.get("ok") else 1
    if args.command == "instrumented-flow":
        summary = run_instrumented_flow(
            trace_path=Path(args.trace),
            config_path=Path(args.config),
            out_dir=Path(args.out_dir),
            model=args.model,
            pyrtl2rrtl=args.pyrtl2rrtl,
            seed=args.seed,
            epochs=args.epochs,
            target=_cli_target(args),
            profile_path=Path(args.profile) if args.profile else None,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary.get("ok") else 1
    if args.command == "inspect-flow":
        bundle = inspect_flow_bundle(Path(args.bundle))
        text = json.dumps(bundle, indent=2, sort_keys=True) + "\n"
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0 if bundle.get("ok") else 1
    if args.command == "quality-gate":
        report = inspect_quality_gate(
            bundle_path=Path(args.bundle),
            thresholds_path=Path(args.thresholds) if args.thresholds else None,
        )
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0 if report.get("ok") else 1
    if args.command == "package-runtime-handoff":
        report = package_runtime_handoff(bundle_path=Path(args.bundle))
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0 if report.get("ok") else 1
    if args.command == "inspect-runtime-telemetry":
        report = inspect_runtime_telemetry_gate(
            bundle_path=Path(args.bundle),
            telemetry_path=Path(args.telemetry),
        )
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0 if report.get("ok") else 1
    if args.command == "inspect-runtime-handoff-telemetry":
        report = inspect_runtime_handoff_telemetry_gate(
            handoff_path=Path(args.handoff),
            telemetry_path=Path(args.telemetry),
        )
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0 if report.get("ok") else 1
    if args.command == "inspect":
        summary = inspect_with_rrtl(
            corpus_path=Path(args.corpus),
            pyrtl2rrtl=args.pyrtl2rrtl,
            out_path=Path(args.out) if args.out else None,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary.get("ok") else 1
    if args.command == "train":
        summary = train_and_export(
            corpus_path=Path(args.corpus),
            out_dir=Path(args.out_dir),
            seed=args.seed,
            target=_cli_target(args),
            profile_path=Path(args.profile) if args.profile else None,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "train-learned":
        summary = train_learned_and_export(
            corpus_path=Path(args.corpus),
            out_dir=Path(args.out_dir),
            epochs=args.epochs,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            target=_cli_target(args),
            profile_path=Path(args.profile) if args.profile else None,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "model-fast-golden":
        golden = write_model_fast_golden(
            corpus_path=Path(args.corpus),
            fast_run_path=Path(args.fast_run),
            op_id=args.op_id,
            op_kind=args.op_kind,
            target=_cli_target(args),
            profile_path=Path(args.profile) if args.profile else None,
            out_path=Path(args.out),
        )
        print(json.dumps(golden, indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        summary = validate_with_rrtl(
            manifest_path=Path(args.manifest),
            corpus_path=Path(args.corpus),
            pyrtl2rrtl=args.pyrtl2rrtl,
            out_path=Path(args.out) if args.out else None,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary.get("ok") else 1
    if args.command == "shadow":
        summary = shadow_with_rrtl(
            manifest_path=Path(args.manifest),
            corpus_path=Path(args.corpus),
            pyrtl2rrtl=args.pyrtl2rrtl,
            out_path=Path(args.out) if args.out else None,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary.get("ok") else 1
    raise AssertionError(args.command)


def _cli_target(args: argparse.Namespace) -> str | None:
    if getattr(args, "target", None) is not None:
        return args.target
    if getattr(args, "profile", None):
        return None
    return "cache_miss"


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validated_events(
    corpus: dict[str, Any],
    *,
    target: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if corpus.get("schema") != CORPUS_SCHEMA:
        raise ValueError(f"unsupported corpus schema {corpus.get('schema')!r}")
    events = corpus.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("corpus.events must be a non-empty list")
    window = int(events[0]["window_cycles"])
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"event {index} must be an object")
        if event.get("schema") != EVENT_SCHEMA:
            raise ValueError(f"event {index} has unsupported schema {event.get('schema')!r}")
        if int(event["window_cycles"]) != window:
            raise ValueError(f"event {index} window does not match first event")
        if target is not None:
            if event.get("target") != target:
                raise ValueError(
                    f"event {index} target {event.get('target')!r} does not match {target!r}"
                )
            label = event.get("label")
            if not isinstance(label, dict) or target not in label:
                raise ValueError(f"event {index} missing label {target!r}")
    return events, window


def _required_int(data: dict[str, Any], name: str, context: str) -> int:
    if name not in data:
        raise ValueError(f"{context} missing feature {name!r}")
    value = data[name]
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, int):
        raise ValueError(f"{context} feature {name!r} must be an integer")
    return value


def _validate_binary_value(name: str, value: Any) -> int:
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, int) or value not in (0, 1):
        raise ValueError(f"{name} must be binary 0/1")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
