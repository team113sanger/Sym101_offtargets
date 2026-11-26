#!/usr/bin/env python3
"""Write the published Sym101 guide list for HISAT2 offtarget re-annotation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sym101_qc.gtf_utils import EXCLUDED_SCREEN_GENES, SCREEN_LNCRNA_BIOTYPES, strip_ensembl_version

REPO_ROOT = Path(__file__).resolve().parents[2]


def query_class(row: pd.Series) -> str:
    if row.get("transcript_biotype") == "NT_Control" or row.get("Gene") == "NT_Control":
        return "NT"
    if (
        row.get("transcript_biotype") in SCREEN_LNCRNA_BIOTYPES
        and row.get("Gene") not in EXCLUDED_SCREEN_GENES
    ):
        return "lncRNA"
    return "protein_coding"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supp-table", type=Path, default=REPO_ROOT / "inputs" / "Supp_table_1_.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "inputs" / "offtarget_rerun",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.supp_table)
    df["GuideSeq"] = df["GuideSeq"].astype(str).str.upper()
    df["query_class"] = df.apply(query_class, axis=1)
    df["intended_gene_id"] = df["ensembl_gene_id"].map(
        lambda x: strip_ensembl_version(str(x)) if pd.notna(x) else ""
    )
    df["intended_transcript_id"] = df["Transcript_ID"].map(
        lambda x: strip_ensembl_version(str(x)) if pd.notna(x) else ""
    )

    cols = [
        "unique_guide_name",
        "GuideSeq",
        "Gene",
        "intended_gene_id",
        "intended_transcript_id",
        "transcript_biotype",
        "query_class",
        "GuideName",
        "MatchPos",
        "Guide_Score",
    ]
    table = df[cols].copy()
    table_path = args.output_dir / "sym101_guides.tsv"
    fasta_path = args.output_dir / "sym101_guides.fa"
    table.to_csv(table_path, sep="\t", index=False)

    with fasta_path.open("w") as handle:
        for row in table.itertuples(index=False):
            handle.write(f">{row.unique_guide_name}\n{row.GuideSeq}\n")

    print(f"Wrote {len(table)} spacers")
    print(table["query_class"].value_counts().to_string())
    print(table_path)
    print(fasta_path)


if __name__ == "__main__":
    main()
