"""Generate a tiny deterministic GEMM ONNX fixture for rrtl-surrogate tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def build_model() -> onnx.ModelProto:
    descriptor = helper.make_tensor_value_info(
        "gemm_descriptor", TensorProto.FLOAT, [6]
    )
    a_tensor = helper.make_tensor_value_info("a_tensor", TensorProto.FLOAT, [2, 2])
    w_tensor = helper.make_tensor_value_info("w_tensor", TensorProto.FLOAT, [2, 2])
    c_tensor = helper.make_tensor_value_info("c_tensor", TensorProto.FLOAT, [2, 2])
    telemetry = helper.make_tensor_value_info("telemetry", TensorProto.FLOAT, [3])

    latency = helper.make_tensor(
        "latency_const", TensorProto.FLOAT, [1], [5.0]
    )
    active = helper.make_tensor("active_const", TensorProto.FLOAT, [1], [2.0])
    utilization = helper.make_tensor("utilization_const", TensorProto.FLOAT, [1], [0.4])

    nodes = [
        helper.make_node("MatMul", ["a_tensor", "w_tensor"], ["c_tensor"]),
        helper.make_node(
            "Concat",
            ["latency_const", "active_const", "utilization_const"],
            ["telemetry"],
            axis=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "rrtl_tiny_gemm",
        [descriptor, a_tensor, w_tensor],
        [c_tensor, telemetry],
        initializer=[latency, active, utilization],
    )
    model = helper.make_model(
        graph,
        producer_name="rrtl-surrogate",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="crates/rrtl-surrogate/tests/fixtures/tiny_gemm.onnx",
        help="output ONNX fixture path",
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
