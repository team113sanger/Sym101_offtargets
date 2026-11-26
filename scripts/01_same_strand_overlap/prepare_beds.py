#!/usr/bin/env python3
"""Build BED files for lncRNA / protein-coding overlap analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from sym101_qc.gtf_utils import (
    exons_to_bed,
    filter_gtf_transcripts,
    genes_to_bed,
    load_screen_lncrna_transcripts,
    read_gtf,
    transcripts_to_bed,
    write_bed,
    write_bed_with_extra,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT / "lncrna_resources"


def prepare_beds(
    gtf_path: str | Path,
    supp_table_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    screen = load_screen_lncrna_transcripts(supp_table_path)
    screen_path = output_dir / "screen_lncrna_summary.tsv"
    screen.to_csv(screen_path, sep="\t", index=False)

    gtf = read_gtf(gtf_path, gene_types={"protein_coding"})
    lnc_tx = filter_gtf_transcripts(gtf, screen["transcript_id_base"])
    missing = set(screen["transcript_id_base"]) - set(lnc_tx["transcript_id_base"])
    if missing:
        missing_path = output_dir / "missing_transcripts.txt"
        missing_path.write_text("\n".join(sorted(missing)) + "\n")
        print(f"WARNING: {len(missing)} screen transcript(s) not found in GTF; see {missing_path}")

    lnc_bed = transcripts_to_bed(lnc_tx)
    return {
        "screen_summary": screen_path,
        "lncrna_transcripts": write_bed(lnc_bed, output_dir / "screen_lncrna_transcripts.bed"),
        "lncrna_transcripts_full": write_bed_with_extra(
            lnc_bed, output_dir / "screen_lncrna_transcripts.full.bed"
        ),
        "pc_genes": write_bed(
            genes_to_bed(gtf, gene_type="protein_coding"),
            output_dir / "protein_coding_genes.bed",
        ),
        "pc_exons": write_bed(
            exons_to_bed(gtf, gene_type="protein_coding"),
            output_dir / "protein_coding_exons.bed",
        ),
        "pc_cds": write_bed(
            exons_to_bed(gtf, gene_type="protein_coding", feature="CDS"),
            output_dir / "protein_coding_cds.bed",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gtf",
        type=Path,
        default=REPO_ROOT / "inputs" / "gencode_v46" / "gencode.v46.annotation.gtf.gz",
    )
    parser.add_argument(
        "--supp-table",
        type=Path,
        default=REPO_ROOT / "inputs" / "Supp_table_1_.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "inputs" / "annotations",
    )
    args = parser.parse_args()

    paths = prepare_beds(args.gtf, args.supp_table, args.output_dir)
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
