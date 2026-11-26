#!/usr/bin/env python3
"""Quantify same-strand overlaps between screen lncRNA transcripts and protein-coding gene bodies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sym101_qc.overlap_analysis import (
    DEFAULT_THRESHOLDS,
    OverlapThreshold,
    run_overlap_analysis,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_thresholds(
    thresholds: list[float] | None,
    include_any: bool,
    same_strand: bool,
) -> list[OverlapThreshold]:
    out: list[OverlapThreshold] = []
    if include_any:
        out.append(OverlapThreshold(name="any_overlap", min_frac_query=0.0, same_strand=same_strand))
    for frac in thresholds or []:
        label = f"{int(frac * 100)}pct_of_lncrna"
        out.append(OverlapThreshold(name=label, min_frac_query=frac, same_strand=same_strand))
    return out or DEFAULT_THRESHOLDS


def main() -> None:
    ann = REPO_ROOT / "inputs" / "annotations"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lncrna-bed", type=Path, default=ann / "screen_lncrna_transcripts.bed")
    parser.add_argument("--pc-genes-bed", type=Path, default=ann / "protein_coding_genes.bed")
    parser.add_argument("--screen-summary", type=Path, default=ann / "screen_lncrna_summary.tsv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "01_same_strand_overlap" / "lncrna_pc_overlap",
    )
    parser.add_argument("--threshold", type=float, action="append")
    parser.add_argument("--include-any-overlap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--same-strand-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bedtools", default="bedtools")
    args = parser.parse_args()

    thresholds = build_thresholds(args.threshold, args.include_any_overlap, args.same_strand_only)
    outputs = run_overlap_analysis(
        args.lncrna_bed,
        args.pc_genes_bed,
        args.output_dir,
        screen_summary_path=args.screen_summary,
        thresholds=thresholds,
        bedtools=args.bedtools,
    )
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
