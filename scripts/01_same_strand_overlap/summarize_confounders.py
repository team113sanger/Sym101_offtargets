#!/usr/bin/env python3
"""Combine lncRNA–PC overlap and guide specificity for reviewer summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "01_same_strand_overlap"

GENE_MAP = {
    "ENST00000437930": "ACTA2-AS1",
    "ENST00000542933": "LINC01089",
    "ENST00000600801": "SOX2-OT",
    "ENST00000304425": "MIR31HG",
}


def _gene_from_screen(transcript_id: str, screen: pd.DataFrame | None) -> str | None:
    if screen is None or screen.empty:
        return GENE_MAP.get(transcript_id)
    hit = screen.loc[screen["transcript_id_base"] == transcript_id, "Gene"]
    if len(hit):
        return hit.iloc[0]
    return GENE_MAP.get(transcript_id)


def load_overlap_pairs(overlap_dir: Path) -> pd.DataFrame:
    path = overlap_dir / "lncrna_pc_overlap_fractions.tsv"
    if not path.exists():
        path = overlap_dir / "lncrna_pc_overlaps_any_overlap.tsv"
        df = pd.read_csv(path, sep="\t")
        return df.rename(columns={"lnc_name": "lnc_name", "pc_name": "pc_name"})
    return pd.read_csv(path, sep="\t")


def summarize_guides(guide_coords: Path, genes: list[str]) -> pd.DataFrame:
    df = pd.read_csv(guide_coords, sep="\t", low_memory=False)
    sub = df.loc[df["Gene"].isin(genes)].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for gene, gdf in sub.groupby("Gene"):
        rows.append(
            {
                "Gene": gene,
                "n_guides": gdf["unique_guide_name"].nunique(),
                "n_lncrna_specific": int((gdf["guide_specificity"] == "lncrna_specific").sum()),
                "n_overlaps_pc_intron_only": int(
                    (gdf["guide_specificity"] == "overlaps_pc_intron_only").sum()
                ),
                "n_overlaps_pc_exon": int(gdf["guide_specificity"].str.startswith("overlaps_pc_exon").sum()),
                "n_overlaps_pc_cds": int((gdf["guide_specificity"] == "overlaps_pc_cds").sum()),
                "n_unmapped": int((gdf["guide_specificity"] == "unmapped").sum()),
                "n_junction_unmapped": int((gdf["guide_specificity"] == "junction_spanning_unmapped").sum()),
                "overlapping_pc_genes": ";".join(
                    sorted({x for x in gdf["overlapping_pc_genes"].dropna().astype(str) if x and x != "nan"})
                )
                or None,
            }
        )
    return pd.DataFrame(rows)


def build_summary(overlap_dir: Path, guide_dir: Path) -> pd.DataFrame:
    pairs = load_overlap_pairs(overlap_dir)
    if "lnc_name" in pairs.columns:
        pairs = pairs.rename(columns={"lnc_name": "transcript_id", "pc_name": "pc_gene"})
    else:
        pairs["transcript_id"] = pairs.iloc[:, 3]
        pairs["pc_gene"] = pairs.iloc[:, 10]

    screen_path = REPO_ROOT / "inputs" / "annotations" / "screen_lncrna_summary.tsv"
    screen = pd.read_csv(screen_path, sep="\t") if screen_path.exists() else None
    pairs["Gene"] = pairs["transcript_id"].map(lambda tx: _gene_from_screen(tx, screen))
    genes = [g for g in pairs["Gene"].dropna().tolist() if g]

    guides = summarize_guides(guide_dir / "lncrna_guide_genomic_coords.tsv", genes)
    out = pairs.merge(guides, on="Gene", how="left")

    # Liang-style flags
    out["passes_50pct_lncrna"] = out.get("frac_of_lncrna", 0) >= 0.5
    out["passes_80pct_lncrna"] = out.get("frac_of_lncrna", 0) >= 0.8
    out["any_guide_hits_pc_exon_or_cds"] = (
        out["n_overlaps_pc_exon"].fillna(0) + out["n_overlaps_pc_cds"].fillna(0)
    ) > 0
    out["reviewer_risk"] = out.apply(_risk_label, axis=1)
    return out


def _risk_label(row: pd.Series) -> str:
    if row.get("passes_50pct_lncrna") and row.get("any_guide_hits_pc_exon_or_cds"):
        return "high — substantial lncRNA/PC overlap and guides hit PC exons/CDS"
    if row.get("passes_50pct_lncrna") and row.get("n_overlaps_pc_intron_only", 0) > 0:
        return "moderate — guides in PC introns only (same-strand gene-body overlap)"
    if not row.get("passes_50pct_lncrna", False) and row.get("n_lncrna_specific", 0) == row.get(
        "n_guides", 0
    ):
        return "low — minimal lncRNA/PC overlap; all guides lncRNA-specific"
    if not row.get("passes_50pct_lncrna", False):
        return "low-moderate — limited lncRNA/PC overlap; check guide placement"
    return "review manually"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-dir", type=Path, default=RESULTS / "lncrna_pc_overlap")
    parser.add_argument("--guide-dir", type=Path, default=RESULTS / "guide_specificity")
    parser.add_argument("--output", type=Path, default=RESULTS / "lncrna_confounder_summary.tsv")
    args = parser.parse_args()

    summary = build_summary(args.overlap_dir, args.guide_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, sep="\t", index=False)
    print(
        summary[
            ["Gene", "pc_gene", "frac_of_lncrna", "frac_of_pc_gene", "n_guides", "reviewer_risk"]
        ].to_string(index=False)
    )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
