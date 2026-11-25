"""Map screen guides to genomic coordinates and assess overlap with protein-coding annotation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sym101_qc.gtf_utils import (
    is_screen_lncrna_row,
    parse_guide_transcript_coords,
    read_gtf,
    strip_ensembl_version,
    transcript_to_genomic,
)
from sym101_qc.overlap_analysis import OverlapThreshold, run_bedtools_intersect


def load_lncrna_guides(supp_table_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(supp_table_path)
    guides = df.loc[is_screen_lncrna_row(df)].copy()
    guides["transcript_id_base"] = guides["Transcript_ID"].map(strip_ensembl_version)
    return guides


def annotate_guide_genomic_coords(
    guides: pd.DataFrame,
    gtf: pd.DataFrame,
) -> pd.DataFrame:
    """Add genomic coordinates for each guide from transcript-space GuideName intervals."""
    records = []
    for _, row in guides.iterrows():
        try:
            tx_start, tx_end = parse_guide_transcript_coords(row["GuideName"])
        except ValueError:
            tx_end = int(row["MatchPos"])
            tx_start = tx_end - 22

        mapping = transcript_to_genomic(row["Transcript_ID"], tx_start, tx_end, gtf)
        rec = row.to_dict()
        rec.update(
            {
                "tx_start": tx_start,
                "tx_end": tx_end,
                "spans_junction": mapping["spans_junction"] if mapping else None,
                "genomic_chrom": mapping.get("chrom") if mapping else None,
                "genomic_start": mapping.get("genomic_start") if mapping else None,
                "genomic_end": mapping.get("genomic_end") if mapping else None,
                "guide_strand": mapping.get("strand") if mapping else None,
                "target_exon_id": mapping.get("exon_id") if mapping else None,
            }
        )
        records.append(rec)
    return pd.DataFrame(records)


def guides_to_bed(guides: pd.DataFrame) -> pd.DataFrame:
    mapped = guides.dropna(subset=["genomic_chrom", "genomic_start", "genomic_end"]).copy()
    mapped["genomic_start"] = mapped["genomic_start"].astype(int)
    mapped["genomic_end"] = mapped["genomic_end"].astype(int)
    return mapped.assign(
        chrom=mapped["genomic_chrom"],
        chromStart=mapped["genomic_start"] - 1,
        chromEnd=mapped["genomic_end"],
        name=mapped["unique_guide_name"],
        score=0,
        strand=mapped["guide_strand"],
    )[["chrom", "chromStart", "chromEnd", "name", "score", "strand"]]


def classify_guide_pc_overlap(
    guide_hits: pd.DataFrame,
    pc_exons_bed: str | Path,
    pc_genes_bed: str | Path,
    pc_cds_bed: str | Path | None = None,
    bedtools: str = "bedtools",
) -> pd.DataFrame:
    """
    Classify each guide by overlap with protein-coding exons, CDS, and gene bodies.

    """
    if guide_hits.empty:
        return guide_hits

    tmp_dir = Path(pc_exons_bed).parent / "_guide_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    guide_bed_path = tmp_dir / "guides.bed"
    guides_to_bed(guide_hits).to_csv(guide_bed_path, sep="\t", header=False, index=False)

    any_hit = run_bedtools_intersect(
        guide_bed_path,
        pc_genes_bed,
        OverlapThreshold(name="any", min_frac_query=0.0, same_strand=True),
        bedtools=bedtools,
    ).rename(
        columns={
            "lnc_name": "unique_guide_name",
            "pc_name": "overlapping_pc_gene",
            "pc_strand": "overlapping_pc_strand",
        }
    )
    exon_hit = run_bedtools_intersect(
        guide_bed_path,
        pc_exons_bed,
        OverlapThreshold(name="any", min_frac_query=0.0, same_strand=True),
        bedtools=bedtools,
    ).rename(columns={"lnc_name": "unique_guide_name", "pc_name": "overlapping_pc_exon_tx"})

    classified = guide_hits.copy()
    if not any_hit.empty:
        gene_map = (
            any_hit.groupby("unique_guide_name")
            .agg(
                overlapping_pc_genes=("overlapping_pc_gene", lambda s: ";".join(sorted(set(s)))),
                n_overlapping_pc_genes=("overlapping_pc_gene", "nunique"),
            )
            .reset_index()
        )
        classified = classified.merge(gene_map, on="unique_guide_name", how="left")
    else:
        classified["overlapping_pc_genes"] = None
        classified["n_overlapping_pc_genes"] = 0

    if not exon_hit.empty:
        exon_map = (
            exon_hit.groupby("unique_guide_name")
            .agg(
                overlaps_pc_exon=("overlapping_pc_exon_tx", "count"),
            )
            .reset_index()
        )
        classified = classified.merge(exon_map, on="unique_guide_name", how="left")
        classified["overlaps_pc_exon"] = classified["overlaps_pc_exon"].fillna(0).astype(int) > 0
    else:
        classified["overlaps_pc_exon"] = False

    classified["overlaps_pc_gene_body"] = classified["n_overlapping_pc_genes"].fillna(0).astype(int) > 0
    classified["overlaps_pc_intron_only"] = (
        classified["overlaps_pc_gene_body"] & ~classified["overlaps_pc_exon"]
    )

    if pc_cds_bed and Path(pc_cds_bed).exists():
        cds_hit = run_bedtools_intersect(
            guide_bed_path,
            pc_cds_bed,
            OverlapThreshold(name="any", min_frac_query=0.0, same_strand=True),
            bedtools=bedtools,
        ).rename(columns={"lnc_name": "unique_guide_name"})
        cds_guides = set(cds_hit["unique_guide_name"]) if not cds_hit.empty else set()
        classified["overlaps_pc_cds"] = classified["unique_guide_name"].isin(cds_guides)
    else:
        classified["overlaps_pc_cds"] = False

    classified["guide_specificity"] = classified.apply(_specificity_label, axis=1)
    return classified


def _specificity_label(row: pd.Series) -> str:
    if row.get("spans_junction"):
        return "junction_spanning_unmapped"
    if pd.isna(row.get("genomic_start")):
        return "unmapped"
    if row.get("overlaps_pc_cds"):
        return "overlaps_pc_cds"
    if row.get("overlaps_pc_exon"):
        return "overlaps_pc_exon_not_cds"
    if row.get("overlaps_pc_intron_only"):
        return "overlaps_pc_intron_only"
    return "lncrna_specific"


def run_guide_specificity_analysis(
    supp_table_path: str | Path,
    gtf_path: str | Path,
    pc_exons_bed: str | Path,
    pc_genes_bed: str | Path,
    output_dir: str | Path,
    pc_cds_bed: str | Path | None = None,
    bedtools: str = "bedtools",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    guides = load_lncrna_guides(supp_table_path)
    gtf = read_gtf(gtf_path, gene_types={"protein_coding", "lncRNA"})
    mapped = annotate_guide_genomic_coords(guides, gtf)
    classified = classify_guide_pc_overlap(mapped, pc_exons_bed, pc_genes_bed, pc_cds_bed, bedtools=bedtools)

    guide_path = output_dir / "lncrna_guide_genomic_coords.tsv"
    classified.to_csv(guide_path, sep="\t", index=False)

    summary = (
        classified.groupby(["Gene", "transcript_id_base", "guide_specificity"], as_index=False)
        .size()
        .rename(columns={"size": "n_guides"})
    )
    summary_path = output_dir / "lncrna_guide_specificity_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)

    by_gene = classified.groupby("Gene", as_index=False).agg(
        n_guides=("unique_guide_name", "nunique"),
        n_lncrna_specific=("guide_specificity", lambda s: (s == "lncrna_specific").sum()),
        n_overlaps_pc=("guide_specificity", lambda s: s.str.startswith("overlaps_pc").sum()),
        n_unmapped=("guide_specificity", lambda s: (s == "unmapped").sum()),
    )
    by_gene_path = output_dir / "lncrna_guide_specificity_by_gene.tsv"
    by_gene.to_csv(by_gene_path, sep="\t", index=False)

    return {
        "guides": guide_path,
        "summary": summary_path,
        "by_gene": by_gene_path,
    }
