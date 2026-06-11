"""Small ONNX ModelProto writer for deterministic event-predictor fixtures."""

from __future__ import annotations

import struct
from pathlib import Path


WIRE_VARINT = 0
WIRE_LEN = 2
TENSOR_FLOAT = 1


def build_minimal_event_predictor_onnx(
    *,
    signal_window_shape: list[int],
    program_context_shape: list[int],
    probability: float,
    predicted: int,
) -> bytes:
    """Return a tiny ONNX model with event predictor tensor names."""

    graph = b""
    graph += _field_msg(1, _node("Identity", ["probability_const"], ["event_probability"]))
    graph += _field_msg(1, _node("Identity", ["prediction_const"], ["predicted_event"]))
    graph += _field_str(2, "rrtl_tiny_event_predictor")
    graph += _field_msg(5, _tensor("probability_const", [1, 1], [float(probability)]))
    graph += _field_msg(5, _tensor("prediction_const", [1, 1], [float(predicted)]))
    graph += _field_msg(11, _value_info("signal_window", signal_window_shape))
    graph += _field_msg(11, _value_info("program_context", program_context_shape))
    graph += _field_msg(12, _value_info("event_probability", [1, 1]))
    graph += _field_msg(12, _value_info("predicted_event", [1, 1]))

    model = _field_int(1, 9)
    model += _field_str(2, "rrtl-surrogate")
    model += _field_msg(7, graph)
    model += _field_msg(8, _opset(17))
    return model


def write_minimal_event_predictor_onnx(
    path: Path,
    *,
    signal_window_shape: list[int],
    program_context_shape: list[int],
    probability: float,
    predicted: int,
) -> None:
    path.write_bytes(
        build_minimal_event_predictor_onnx(
            signal_window_shape=signal_window_shape,
            program_context_shape=program_context_shape,
            probability=probability,
            predicted=predicted,
        )
    )


def _varint(value: int) -> bytes:
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _field_int(field: int, value: int) -> bytes:
    return _key(field, WIRE_VARINT) + _varint(value)


def _field_str(field: int, value: str) -> bytes:
    data = value.encode("utf-8")
    return _key(field, WIRE_LEN) + _varint(len(data)) + data


def _field_bytes(field: int, value: bytes) -> bytes:
    return _key(field, WIRE_LEN) + _varint(len(value)) + value


def _field_msg(field: int, value: bytes) -> bytes:
    return _field_bytes(field, value)


def _dimension(value: int) -> bytes:
    return _field_int(1, value)


def _shape(dims: list[int]) -> bytes:
    return b"".join(_field_msg(1, _dimension(dim)) for dim in dims)


def _tensor_type(dims: list[int]) -> bytes:
    return _field_int(1, TENSOR_FLOAT) + _field_msg(2, _shape(dims))


def _type_proto(dims: list[int]) -> bytes:
    return _field_msg(1, _tensor_type(dims))


def _value_info(name: str, dims: list[int]) -> bytes:
    return _field_str(1, name) + _field_msg(2, _type_proto(dims))


def _tensor(name: str, dims: list[int], values: list[float]) -> bytes:
    raw = struct.pack("<" + "f" * len(values), *values)
    body = b"".join(_field_int(1, dim) for dim in dims)
    body += _field_int(2, TENSOR_FLOAT)
    body += _field_str(8, name)
    body += _field_bytes(9, raw)
    return body


def _node(op_type: str, inputs: list[str], outputs: list[str]) -> bytes:
    body = b"".join(_field_str(1, name) for name in inputs)
    body += b"".join(_field_str(2, name) for name in outputs)
    body += _field_str(4, op_type)
    return body


def _opset(version: int) -> bytes:
    return _field_int(2, version)
