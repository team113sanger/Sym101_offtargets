#!/usr/bin/env python3
"""Regex bulk rename with a mandatory dry-run default.

    pixi run python src/sym101_qc/bulk_rename.py results/supplementary \\
        --pattern 'run_manifest\\.local-.+\\.json' \\
        --replacement '' --dry-run

    # After checking:
    pixi run python src/sym101_qc/bulk_rename.py … --execute
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def bulk_rename(
    directory: Path,
    pattern: str,
    replacement: str,
    *,
    dry_run: bool = True,
    recursive: bool = False,
) -> list[tuple[str, str]]:
    """Rename files whose names match ``pattern``. Returns (old, new) pairs."""
    target = directory.expanduser().resolve()
    if not target.is_dir():
        raise SystemExit(f"not a directory: {target}")

    compiled = re.compile(pattern)
    files = sorted(target.rglob("*") if recursive else target.iterdir())
    renamed: list[tuple[str, str]] = []

    for path in files:
        if not path.is_file():
            continue
        new_name = compiled.sub(replacement, path.name)
        if new_name == path.name:
            continue
        if not new_name:
            print(f"  skip (empty name): {path.name}", file=sys.stderr)
            continue
        new_path = path.with_name(new_name)
        if new_path.exists():
            print(f"  skip (target exists): {path.name} -> {new_name}", file=sys.stderr)
            continue
        if dry_run:
            print(f"  would rename: {path.relative_to(target)} -> {new_name}")
        else:
            path.rename(new_path)
            print(f"  renamed: {path.relative_to(target)} -> {new_name}")
        renamed.append((path.name, new_name))

    verb = "Would rename" if dry_run else "Renamed"
    print(f"\n{verb} {len(renamed)} file(s) under {target}")
    return renamed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Folder whose files to rename")
    parser.add_argument("--pattern", required=True, help="Regex matched against the file name")
    parser.add_argument("--replacement", required=True, help="re.sub replacement (may use \\1…)")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also rename under subdirectories",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print the plan only (default)",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Actually rename (disables dry-run)",
    )
    args = parser.parse_args()
    bulk_rename(
        args.directory,
        args.pattern,
        args.replacement,
        dry_run=not args.execute,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
