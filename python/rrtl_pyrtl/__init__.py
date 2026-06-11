"""Export elaborated PyRTL blocks for RRTL import."""

from .export import export_block, export_block_json, simulate_lane_trace, simulate_trace

__all__ = [
    "discover_targets",
    "export_block",
    "export_block_json",
    "build_systolic_mac",
    "render_discovery_report",
    "render_benchmark_markdown",
    "render_gate_markdown",
    "render_validation_report",
    "run_benchmark",
    "run_gate",
    "simulate_lane_trace",
    "simulate_trace",
    "write_manifest",
]


def __getattr__(name):
    if name in {"discover_targets", "render_discovery_report", "write_manifest"}:
        from . import discover

        return getattr(discover, name)
    if name == "render_validation_report":
        from . import corpus

        return getattr(corpus, name)
    if name in {"render_gate_markdown", "run_gate"}:
        from . import gate

        return getattr(gate, name)
    if name in {"build_systolic_mac", "render_benchmark_markdown", "run_benchmark"}:
        from . import bench

        return getattr(bench, name)
    raise AttributeError(name)
