"""Static discovery for PyRTL corpus manifest candidates.

The scanner is intentionally conservative: it parses Python source with
``ast`` and never imports or executes private design code.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any, Iterable


IGNORED_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    "target",
    "node_modules",
}

PYRTL_CALLS = {
    "concat",
    "Input",
    "Output",
    "Register",
    "MemBlock",
    "mux",
    "RomBlock",
    "select",
    "WireVector",
    "working_block",
    "reset_working_block",
    "Block",
}
DESIGN_CONSTRUCTION_CALLS = PYRTL_CALLS - {"working_block", "reset_working_block", "Block"}

BUILDER_NAME_HINTS = ("build", "make", "create", "elaborate", "top")


def discover_targets(
    roots: Iterable[str | Path],
    package_roots: Iterable[str | Path] | None = None,
) -> list[dict[str, Any]]:
    """Return static PyRTL builder candidates under ``roots``."""

    scan_roots = [Path(root).resolve() for root in roots]
    if not scan_roots:
        raise ValueError("at least one scan root is required")
    module_roots = [Path(root).resolve() for root in (package_roots or scan_roots)]

    candidates: list[dict[str, Any]] = []
    seen_targets = set()
    for root in scan_roots:
        for path in _iter_python_files(root):
            module = _module_name(path, module_roots)
            if module is None:
                continue
            for candidate in _discover_file(path, module):
                if candidate["target"] in seen_targets:
                    continue
                seen_targets.add(candidate["target"])
                candidates.append(candidate)

    return sorted(candidates, key=lambda item: (item["target"], item["source_path"], item["line"]))


def render_discovery_report(candidates: list[dict[str, Any]]) -> str:
    """Render a Markdown report for discovered candidates."""

    confidence = _confidence_counts(candidates)
    lines = [
        "# PyRTL Corpus Discovery",
        "",
        f"- Candidates: {len(candidates)}",
        f"- High confidence: {confidence['high']}",
        f"- Medium confidence: {confidence['medium']}",
        f"- Low confidence: {confidence['low']}",
        "",
        "## Next Steps",
        "",
        "Review the candidates below, edit the generated manifest if needed, then run manifest validation before full corpus triage.",
        "",
    ]
    if not candidates:
        lines.append("No candidates found.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| Name | Target | Confidence | Source | Reasons |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for candidate in candidates:
        source = f"{candidate['source_path']}:{candidate['line']}"
        reasons = ", ".join(candidate.get("reasons", []))
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(candidate.get("name")),
                    _md(candidate.get("target")),
                    str(candidate.get("confidence", 0)),
                    _md(source),
                    _md(reasons),
                ]
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def _confidence_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "low": 0}
    for candidate in candidates:
        confidence = int(candidate.get("confidence", 0))
        if confidence >= 80:
            counts["high"] += 1
        elif confidence >= 50:
            counts["medium"] += 1
        else:
            counts["low"] += 1
    return counts


def write_manifest(candidates: list[dict[str, Any]], path: str | Path) -> None:
    """Write candidates as a draft corpus manifest."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    targets = [
        {
            "name": candidate["name"],
            "target": candidate["target"],
            "top_name": candidate["top_name"],
            "clock_name": candidate["clock_name"],
            "reset_working_block": candidate["reset_working_block"],
        }
        for candidate in candidates
    ]
    out.write_text(json.dumps({"targets": targets}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iter_python_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix == ".py":
            yield root
        return

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_DIRS)
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                yield Path(dirpath) / filename


def _module_name(path: Path, package_roots: list[Path]) -> str | None:
    resolved = path.resolve()
    for root in package_roots:
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            continue
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            return None
        return ".".join(parts)
    return None


def _discover_file(path: Path, module: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return []

    import_info = _pyrtl_imports(tree)
    if not import_info["has_pyrtl"]:
        return []

    candidates = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        candidate = _function_candidate(node, module, path, import_info)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _pyrtl_imports(tree: ast.Module) -> dict[str, Any]:
    aliases = set()
    imported_names = set()
    star_import = False

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pyrtl" or alias.name.startswith("pyrtl."):
                    aliases.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module == "pyrtl":
            for alias in node.names:
                if alias.name == "*":
                    star_import = True
                else:
                    imported_names.add(alias.asname or alias.name)

    return {
        "has_pyrtl": bool(aliases or imported_names or star_import),
        "aliases": aliases,
        "imported_names": imported_names,
        "star_import": star_import,
    }


def _function_candidate(
    node: ast.FunctionDef,
    module: str,
    path: Path,
    import_info: dict[str, Any],
) -> dict[str, Any] | None:
    calls = set()
    returns_block = False
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            call_name = _call_name(child.func, import_info)
            if call_name:
                calls.add(call_name)
        elif isinstance(child, ast.Return):
            returns_block = returns_block or _returns_block(child.value, import_info)

    reasons = []
    if calls:
        reasons.append("uses " + ", ".join(sorted(calls)))
    if returns_block:
        reasons.append("returns PyRTL block")
    if _builder_name_hint(node.name):
        reasons.append("builder-like name")

    has_design_call = bool(calls & DESIGN_CONSTRUCTION_CALLS)
    if not has_design_call and not returns_block and not (_builder_name_hint(node.name) and calls):
        return None

    confidence = 40
    confidence += min(30, len(calls & DESIGN_CONSTRUCTION_CALLS) * 10)
    if returns_block:
        confidence += 20
    if _builder_name_hint(node.name):
        confidence += 10
    confidence = min(confidence, 100)

    target = f"{module}:{node.name}"
    name = target.replace(":", "_").replace(".", "_")
    return {
        "name": name,
        "target": target,
        "top_name": _top_name(node.name),
        "clock_name": "clk",
        "reset_working_block": True,
        "confidence": confidence,
        "source_path": str(path),
        "line": node.lineno,
        "reasons": reasons,
    }


def _call_name(func: ast.expr, import_info: dict[str, Any]) -> str | None:
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id in import_info["aliases"] and func.attr in PYRTL_CALLS:
            return func.attr
    if isinstance(func, ast.Name):
        if func.id in import_info["imported_names"] or import_info["star_import"]:
            if func.id in PYRTL_CALLS:
                return func.id
    return None


def _returns_block(value: ast.expr | None, import_info: dict[str, Any]) -> bool:
    if value is None:
        return False
    if isinstance(value, ast.Call):
        return _call_name(value.func, import_info) in {"working_block", "Block"}
    if isinstance(value, ast.Name):
        return value.id.lower().endswith("block")
    return False


def _builder_name_hint(name: str) -> bool:
    lower = name.lower()
    return any(hint in lower for hint in BUILDER_NAME_HINTS)


def _top_name(function_name: str) -> str:
    parts = [part for part in function_name.replace("-", "_").split("_") if part]
    if not parts:
        return "Top"
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _md(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover likely PyRTL corpus manifest targets")
    parser.add_argument("roots", nargs="+", help="Python files or directories to scan")
    parser.add_argument(
        "--package-root",
        action="append",
        help="root used to derive module paths; defaults to each scan root",
    )
    parser.add_argument("--out", required=True, help="draft manifest JSON output path")
    parser.add_argument("--report", required=True, help="Markdown discovery report output path")
    args = parser.parse_args(argv)

    candidates = discover_targets(args.roots, package_roots=args.package_root)
    write_manifest(candidates, args.out)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_discovery_report(candidates), encoding="utf-8")
    print(f"discovered {len(candidates)} candidate(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
