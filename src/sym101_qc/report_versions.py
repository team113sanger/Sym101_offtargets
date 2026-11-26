#!/usr/bin/env python3
"""Print the environment's tool versions next to the versions the published run used.

`pixi run versions`

"""

from __future__ import annotations

import shutil
import subprocess
import sys

from sym101_qc.manifest import load_manifest


def _first_version_token(text: str) -> str:
    # bedtools prints "bedtools v2.31.1", bowtie prints "bowtie-align-s version 1.3.1":
    for token in text.replace(",", " ").split():
        token = token.removeprefix("v")
        if token[:1].isdigit() and "." in token:
            return token
    return "?"


def _probe(binary: str, args: list[str]) -> str:
    path = shutil.which(binary)
    if path is None:
        return "NOT ON PATH"
    try:
        proc = subprocess.run(  # noqa: S603 - fixed binary names, no shell
            [path, *args], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"probe failed: {exc}"
    return _first_version_token(proc.stdout or proc.stderr)


def main() -> int:
    manifest = load_manifest()
    tools = manifest["tools"]

    rows = [
        ("bowtie", tools["bowtie"]["version"], _probe("bowtie", ["--version"])),
        ("bedtools", tools["bedtools"]["version"], _probe("bedtools", ["--version"])),
        ("python", tools["python"]["version"], ".".join(map(str, sys.version_info[:3]))),
    ]

    width = max(len(name) for name, _, _ in rows)
    print(f"{'tool'.ljust(width)}  {'manifest':<12}  installed")
    mismatched = []
    for name, declared, installed in rows:
        print(f"{name.ljust(width)}  {str(declared):<12}  {installed}")
        if name != "python" and not installed.startswith(str(declared)):
            mismatched.append(name)

    if mismatched:
        print()
        print("MISMATCH: " + ", ".join(mismatched) + " differ from conf/manifest.yaml.")
        print("Results produced with this environment will not match the published freeze.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
