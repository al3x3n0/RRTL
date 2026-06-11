"""JSON exporter for elaborated PyRTL blocks.

The exporter intentionally consumes PyRTL's normalized ``Block``/``LogicNet``
representation. It does not try to translate Python source.
"""

from __future__ import annotations

import importlib
import json
from typing import Any, IO

import pyrtl
from pyrtl import wire


def export_block(
    block: pyrtl.Block | None = None,
    *,
    top_name: str = "Top",
    clock_name: str = "clk",
) -> dict[str, Any]:
    """Return an RRTL/PyRTL bridge JSON object for ``block``."""

    block = pyrtl.working_block() if block is None else block
    block.sanity_check()

    wires = []
    for w in sorted(block.wirevector_set, key=lambda item: item.name):
        item: dict[str, Any] = {
            "name": w.name,
            "kind": _wire_kind(w),
            "bitwidth": len(w),
        }
        if isinstance(w, wire.Const):
            item["value"] = int(w.val)
        if isinstance(w, wire.Register):
            item["reset_value"] = int(getattr(w, "reset_value", 0) or 0)
        wires.append(item)

    memories = []
    for mem in sorted(block.memblock_by_name.values(), key=lambda item: item.name):
        item = {
            "name": mem.name,
            "id": int(mem.id),
            "kind": "rom" if isinstance(mem, pyrtl.RomBlock) else "mem",
            "bitwidth": int(mem.bitwidth),
            "addrwidth": int(mem.addrwidth),
            "asynchronous": bool(getattr(mem, "asynchronous", False)),
            "initial": _memory_initial(mem),
        }
        memories.append(item)

    nets = []
    for index, net in enumerate(block.logic):
        nets.append(
            {
                "index": index,
                "op": net.op,
                "op_param": _op_param(net),
                "args": [arg.name for arg in net.args],
                "dests": [dest.name for dest in net.dests],
            }
        )

    return {
        "schema": "rrtl-pyrtl-block-v1",
        "top_name": top_name,
        "clock_name": clock_name,
        "wires": wires,
        "memories": memories,
        "nets": nets,
    }


def export_block_json(
    dest: IO[str] | None = None,
    block: pyrtl.Block | None = None,
    *,
    top_name: str = "Top",
    clock_name: str = "clk",
) -> str | None:
    """Serialize ``block`` as JSON.

    If ``dest`` is provided, JSON is written to it and ``None`` is returned.
    Otherwise the JSON string is returned.
    """

    data = export_block(block, top_name=top_name, clock_name=clock_name)
    if dest is None:
        return json.dumps(data, indent=2, sort_keys=True)
    json.dump(data, dest, indent=2, sort_keys=True)
    dest.write("\n")
    return None


def simulate_trace(
    input_vectors: list[dict[str, int]],
    block: pyrtl.Block | None = None,
    *,
    register_value_map=None,
    memory_value_map=None,
    default_value: int = 0,
) -> dict[str, Any]:
    """Run PyRTL and return a co-simulation trace for RRTL replay."""

    block = pyrtl.working_block() if block is None else block
    block.sanity_check()
    sim = pyrtl.Simulation(
        tracer=None,
        register_value_map=register_value_map,
        memory_value_map=memory_value_map,
        default_value=default_value,
        block=block,
    )
    outputs = sorted(w.name for w in block.wirevector_subset(wire.Output))
    steps = []
    for inputs in input_vectors:
        sim.step(dict(inputs))
        steps.append(
            {
                "inputs": {name: int(value) for name, value in inputs.items()},
                "outputs": {name: int(sim.inspect(name)) for name in outputs},
            }
        )
    return {"schema": "rrtl-pyrtl-trace-v1", "steps": steps}


def simulate_lane_trace(
    input_vectors_by_lane: list[list[dict[str, int]]],
    block: pyrtl.Block | None = None,
    *,
    register_value_map=None,
    memory_value_map=None,
    default_value: int = 0,
) -> dict[str, Any]:
    """Run independent PyRTL simulations and return a lane-vector replay trace."""

    if not input_vectors_by_lane:
        raise ValueError("input_vectors_by_lane must include at least one lane")
    step_count = len(input_vectors_by_lane[0])
    for lane, vectors in enumerate(input_vectors_by_lane):
        if len(vectors) != step_count:
            raise ValueError(
                f"lane {lane} has {len(vectors)} steps, expected {step_count}"
            )

    block = pyrtl.working_block() if block is None else block
    block.sanity_check()
    outputs = sorted(w.name for w in block.wirevector_subset(wire.Output))
    lane_steps = []
    for vectors in input_vectors_by_lane:
        sim = pyrtl.Simulation(
            tracer=None,
            register_value_map=register_value_map,
            memory_value_map=memory_value_map,
            default_value=default_value,
            block=block,
        )
        steps = []
        for inputs in vectors:
            sim.step(dict(inputs))
            steps.append(
                {
                    "inputs": {name: int(value) for name, value in inputs.items()},
                    "outputs": {name: int(sim.inspect(name)) for name in outputs},
                }
            )
        lane_steps.append(steps)

    steps = []
    for step_index in range(step_count):
        input_names = sorted(
            {
                name
                for lane in lane_steps
                for name in lane[step_index]["inputs"]
            }
        )
        output_names = sorted(
            {
                name
                for lane in lane_steps
                for name in lane[step_index]["outputs"]
            }
        )
        steps.append(
            {
                "inputs": {
                    name: [
                        int(lane[step_index]["inputs"].get(name, 0))
                        for lane in lane_steps
                    ]
                    for name in input_names
                },
                "outputs": {
                    name: [
                        int(lane[step_index]["outputs"][name])
                        for lane in lane_steps
                    ]
                    for name in output_names
                },
            }
        )
    return {
        "schema": "rrtl-pyrtl-lane-trace-v1",
        "lanes": len(input_vectors_by_lane),
        "steps": steps,
    }


def _wire_kind(w: wire.WireVector) -> str:
    if isinstance(w, wire.Input):
        return "input"
    if isinstance(w, wire.Output):
        return "output"
    if isinstance(w, wire.Register):
        return "register"
    if isinstance(w, wire.Const):
        return "const"
    return "wire"


def _op_param(net: pyrtl.LogicNet) -> Any:
    if net.op in {"m", "@"}:
        memid, mem = net.op_param
        return {"memory_id": int(memid), "memory": mem.name}
    if isinstance(net.op_param, tuple):
        return list(net.op_param)
    return net.op_param


def _memory_initial(mem: pyrtl.MemBlock) -> list[dict[str, int]]:
    if not isinstance(mem, pyrtl.RomBlock):
        return []
    data = getattr(mem, "data", None)
    if callable(data):
        return []

    depth = 1 << int(mem.addrwidth)
    values = list(data)
    if getattr(mem, "pad_with_zeros", False):
        values = values + [0] * max(0, depth - len(values))

    out = []
    for addr, value in enumerate(values[:depth]):
        out.append({"addr": int(addr), "value": int(value)})
    return out


def _load_target(target: str) -> Any:
    if ":" not in target:
        raise SystemExit("target must be in module:function form")
    module_name, func_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    value = module
    for part in func_name.split("."):
        value = getattr(value, part)
    return value


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Export a PyRTL Block for RRTL import")
    parser.add_argument(
        "target",
        nargs="?",
        help="optional module:function that builds and optionally returns a PyRTL Block",
    )
    parser.add_argument("--top-name", default="Top")
    parser.add_argument("--clock-name", default="clk")
    parser.add_argument("--out", "-o")
    args = parser.parse_args(argv)

    block = None
    if args.target:
        result = _load_target(args.target)()
        if isinstance(result, pyrtl.Block):
            block = result

    if args.out:
        with open(args.out, "w", encoding="utf-8") as dest:
            export_block_json(
                dest,
                block,
                top_name=args.top_name,
                clock_name=args.clock_name,
            )
    else:
        text = export_block_json(block=block, top_name=args.top_name, clock_name=args.clock_name)
        assert text is not None
        sys.stdout.write(text)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
