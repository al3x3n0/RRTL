"""Generate a tiny deterministic event-predictor ONNX fixture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from rrtl_surrogate.minimal_onnx import write_minimal_event_predictor_onnx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="crates/rrtl-surrogate/tests/fixtures/tiny_event_predictor.onnx",
        help="output ONNX fixture path",
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_minimal_event_predictor_onnx(
        out,
        signal_window_shape=[1, 2, 1],
        program_context_shape=[1, 1],
        probability=1.0,
        predicted=1,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
