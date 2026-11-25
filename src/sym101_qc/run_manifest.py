#!/usr/bin/env python3
"""Write a run JSON manifest beside methodological outputs.
Incl job id, git commit, environment, input files with checksums, parameters, and headline results.

Usage::

    from sym101_qc.run_manifest import RunManifest

    manifest = RunManifest(script="scripts/.../x.py", output_dir=out)
    manifest.add_inputs({"guide_table": path, "annotation": gtf})
    manifest.add_parameters({"min_frac": 0.5, "chronos_essential": -0.5})
    ...
    manifest.add_results({"n_genes": 23})
    manifest.write()

"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

# Files larger than this are recorded by size and mtime rather than hashed.
MAX_CHECKSUM_BYTES = 256 * 1024 * 1024

TRACKED_PACKAGES = ("pandas", "numpy", "scipy", "statsmodels", "matplotlib", "openpyxl", "yaml")


def job_id() -> str:
    """LSF job id when running under LSF, otherwise a unique local id."""
    lsf = os.environ.get("LSB_JOBID")
    if lsf:
        index = os.environ.get("LSB_JOBINDEX")
        return f"{lsf}_{index}" if index and index != "0" else lsf
    return f"local-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"


def git_state() -> dict[str, Any]:
    """Commit, branch and cleanliness. A dirty tree means the committed code is NOT what ran."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-c", f"safe.directory={REPO_ROOT}", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return {"error": f"git state unavailable: {exc}"}


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            module = __import__(name)
        except ImportError:
            versions[name] = "NOT INSTALLED"
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": package_versions(),
        "lsf": {
            key: os.environ[key]
            for key in ("LSB_JOBID", "LSB_QUEUE", "LSB_HOSTS", "LSB_JOBNAME", "LSB_MAX_NUM_PROCESSORS")
            if key in os.environ
        },
        "pixi_environment": os.environ.get("PIXI_ENVIRONMENT_NAME"),
    }


def file_record(path: str | Path) -> dict[str, Any]:
    """Describe one input: resolved path, size, mtime, and sha256 when it is small enough."""
    path = Path(path)
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.is_file():
        return record
    stat = path.stat()
    record["size_bytes"] = stat.st_size
    record["modified_utc"] = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(timespec="seconds")
    if stat.st_size <= MAX_CHECKSUM_BYTES:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        record["sha256"] = digest.hexdigest()
    else:
        record["sha256"] = None
        record["checksum_skipped"] = f"larger than {MAX_CHECKSUM_BYTES} bytes"
    return record


class RunManifest:
    """Accumulates provenance during a run and writes it as JSON at the end."""

    def __init__(
        self,
        script: str,
        output_dir: str | Path,
        status: str = "analysis",
        description: str | None = None,
    ) -> None:
        self.script = script
        self.output_dir = Path(output_dir)
        self.job_id = job_id()
        self.started_utc = datetime.now(UTC).isoformat(timespec="seconds")
        self.payload: dict[str, Any] = {
            "script": script,
            "status": status,
            "description": description,
            "job_id": self.job_id,
            "started_utc": self.started_utc,
            "command_line": [sys.executable, *sys.argv],
            "git": git_state(),
            "environment": environment(),
            "inputs": {},
            "parameters": {},
            "outputs": {},
            "results": {},
            "manifest_keys_used": [],
            "notes": [],
        }

    def add_inputs(self, inputs: dict[str, Any]) -> None:
        for key, value in inputs.items():
            self.payload["inputs"][key] = file_record(value)

    def add_parameters(self, parameters: dict[str, Any]) -> None:
        self.payload["parameters"].update(parameters)

    def add_outputs(self, outputs: dict[str, Any]) -> None:
        for key, value in outputs.items():
            path = Path(value)
            self.payload["outputs"][key] = {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }

    def add_results(self, results: dict[str, Any]) -> None:
        self.payload["results"].update(results)

    def add_manifest_keys(self, keys: list[str]) -> None:
        """Record which conf/manifest.yaml keys this run resolved its settings from."""
        self.payload["manifest_keys_used"].extend(keys)

    def note(self, text: str) -> None:
        self.payload["notes"].append(text)

    def write(self) -> Path:
        self.payload["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / f"run_manifest.{self.job_id}.json"
        destination.write_text(json.dumps(self.payload, indent=2, default=str) + "\n")
        return destination
