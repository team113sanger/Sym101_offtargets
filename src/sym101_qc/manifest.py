#!/usr/bin/env python3
"""Load conf/manifest.yaml and resolve its ``${a.b.c}`` cross-references.

The manifest holds runnable config only: paths, tools, thresholds and I/O.

Usage::

    from sym101_qc.manifest import load_manifest, get, resolve_path

    m = load_manifest()
    get(m, "steps.step_02.parameters.max_mismatches")   # -> 2
    resolve_path(m["paths"]["results"])

Run as a script for a readable dump of one step::

    python src/sym101_qc/manifest.py step_02
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "conf" / "manifest.yaml"
SCRATCH_ENV = "SGSQ_SCRATCH"

_REF = re.compile(r"\$\{([a-zA-Z0-9_.]+)\}")
_MAX_RESOLVE_PASSES = 8


class ManifestError(RuntimeError):
    """The manifest is missing a key, or a reference does not resolve."""


def get(node: Any, dotted: str) -> Any:
    """Fetch ``a.b.c`` from nested dicts, raising with the full path on a miss."""
    cur = node
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ManifestError(f"manifest has no key {dotted!r} (failed at {part!r})")
        cur = cur[part]
    return cur


def _resolve_str(text: str, root: dict) -> str:
    def sub(match: re.Match[str]) -> str:
        value = get(root, match.group(1))
        if isinstance(value, dict | list):
            raise ManifestError(f"${{{match.group(1)}}} points at a {type(value).__name__}, not a scalar")
        return str(value)

    return _REF.sub(sub, text)


def _walk(node: Any, root: dict) -> tuple[Any, bool]:
    """Resolve references one pass deep. Returns (node, changed)."""
    if isinstance(node, str):
        new = _resolve_str(node, root)
        return new, new != node
    if isinstance(node, dict):
        changed = False
        for key, value in node.items():
            node[key], hit = _walk(value, root)
            changed = changed or hit
        return node, changed
    if isinstance(node, list):
        changed = False
        for i, value in enumerate(node):
            node[i], hit = _walk(value, root)
            changed = changed or hit
        return node, changed
    return node, False


def _apply_env_overrides(root: dict) -> None:
    """Apply portable env overrides before ${...} resolution."""
    paths = root.setdefault("paths", {})
    scratch = os.environ.get(SCRATCH_ENV)
    if scratch:
        paths["scratch"] = scratch
        # Keep results aligned with scratch unless the user set an absolute results path.
        if not Path(str(paths.get("results", "results"))).is_absolute():
            paths["results"] = scratch


def load_manifest(path: Path | None = None) -> dict:
    """Parse the manifest, apply env overrides, and resolve every ``${...}`` reference."""
    path = path or MANIFEST_PATH
    root = yaml.safe_load(path.read_text())
    if not isinstance(root, dict):
        raise ManifestError(f"{path} did not parse to a mapping")
    _apply_env_overrides(root)
    for _ in range(_MAX_RESOLVE_PASSES):
        _, changed = _walk(root, root)
        if not changed:
            return root
    raise ManifestError("manifest references did not settle - is there a reference cycle?")


def resolve_path(value: str | Path) -> Path:
    """Turn a manifest path into an absolute one, relative to the repository root.

    Absolute paths (e.g. an ``SGSQ_SCRATCH`` override) are returned unchanged.
    """
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def iter_paths(node: Any) -> list[str]:
    """Every absolute or repo-relative path-looking string under ``node``."""
    out: list[str] = []
    if isinstance(node, str):
        if (
            node.startswith("/")
            or node.startswith("inputs/")
            or node.startswith("results/")
            or node.startswith("scripts/")
            or node.startswith("references/")
        ):
            out.append(node)
    elif isinstance(node, dict):
        for value in node.values():
            out.extend(iter_paths(value))
    elif isinstance(node, list):
        for value in node:
            out.extend(iter_paths(value))
    return out


def paths_exist(node: Any) -> dict[str, bool]:
    """Check which paths under ``node`` are still on disk."""
    out: dict[str, bool] = {}
    for raw in iter_paths(node):
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if "{" in raw:
            candidate = candidate.parent
        out[raw] = candidate.exists()
    return out


def _dump(step: str) -> None:
    manifest = load_manifest()
    node = manifest["steps"].get(step) or manifest.get(step)
    if node is None:
        available = ", ".join(manifest["steps"])
        raise SystemExit(f"unknown step {step!r}; available: {available}")
    print(yaml.safe_dump(node, sort_keys=False, width=100))
    missing = [p for p, ok in paths_exist(node).items() if not ok]
    if missing:
        print("# MISSING ON DISK:")
        for p in missing:
            print(f"#   {p}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    _dump(sys.argv[1])
