#!/usr/bin/env python3
"""Map lncRNA guides to genomic coordinates and assess overlap with protein-coding annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sym101_qc.guide_specificity import run_guide_specificity_analysis

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ann = REPO_ROOT / "inputs" / "annotations"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supp-table", type=Path, default=REPO_ROOT / "inputs" / "Supp_table_1_.csv")
    parser.add_argument(
        "--gtf", type=Path, default=REPO_ROOT / "inputs" / "gencode_v46" / "gencode.v46.annotation.gtf.gz"
    )
    parser.add_argument("--pc-exons-bed", type=Path, default=ann / "protein_coding_exons.bed")
    parser.add_argument("--pc-genes-bed", type=Path, default=ann / "protein_coding_genes.bed")
    parser.add_argument("--pc-cds-bed", type=Path, default=ann / "protein_coding_cds.bed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "01_same_strand_overlap" / "guide_specificity",
    )
    parser.add_argument("--bedtools", default="bedtools")
    args = parser.parse_args()

    pc_cds = args.pc_cds_bed if args.pc_cds_bed.exists() else None
    outputs = run_guide_specificity_analysis(
        args.supp_table,
        args.gtf,
        args.pc_exons_bed,
        args.pc_genes_bed,
        args.output_dir,
        pc_cds_bed=pc_cds,
        bedtools=args.bedtools,
    )
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
