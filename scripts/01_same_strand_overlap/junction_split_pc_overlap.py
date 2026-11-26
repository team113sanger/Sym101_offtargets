#!/usr/bin/env python3
"""
Split-map junction-spanning guides to genomic segments and test PC overlap.

Answers: do guide binding sites (including junction guides) overlap protein-coding
gene bodies, exons, or CDS on the same strand?

For junction guides the 22 bp target is contiguous on mature spliced lncRNA but spans
two lncRNA exons in the genome; each exon-flanking segment is mapped separately and
tested with bedtools intersect (-s).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sym101_qc.gtf_utils import read_gtf, strip_ensembl_version
from sym101_qc.overlap_analysis import OverlapThreshold, run_bedtools_intersect

REPO_ROOT = Path(__file__).resolve().parents[2]
ANN = REPO_ROOT / "inputs" / "annotations"
RESULTS = REPO_ROOT / "results" / "01_same_strand_overlap"


def build_exon_model(gtf: pd.DataFrame, transcript_id: str) -> pd.DataFrame:
    tx_base = strip_ensembl_version(transcript_id)
    exons = gtf.loc[(gtf["feature"] == "exon") & (gtf["transcript_id_base"] == tx_base)].copy()
    if exons.empty:
        return exons

    exon_num = pd.to_numeric(exons["exon_number"], errors="coerce")
    if exon_num.notna().all():
        exons = exons.sort_values("exon_number")
    else:
        exons = exons.sort_values("start")
        if exons.iloc[0]["strand"] == "-":
            exons = exons.sort_values("start", ascending=False)

    exons["length"] = exons["end"] - exons["start"] + 1
    exons["rel_start"] = [1] + list(exons["length"].cumsum()[:-1])
    exons["rel_end"] = exons["length"].cumsum()
    return exons


def map_interval_to_segments(
    transcript_id: str,
    tx_start: int,
    tx_end: int,
    exons: pd.DataFrame,
) -> list[dict]:
    """Map [tx_start, tx_end] to one or more genomic segments (spliced transcript coords)."""
    if exons.empty:
        return []

    hit = exons[(exons["rel_end"] >= tx_start) & (exons["rel_start"] <= tx_end)].sort_values("rel_start")

    if hit.empty:
        return []

    segments: list[dict] = []
    for _, exon in hit.iterrows():
        seg_tx_start = max(tx_start, int(exon["rel_start"]))
        seg_tx_end = min(tx_end, int(exon["rel_end"]))
        if seg_tx_end < seg_tx_start:
            continue
        offset_start = seg_tx_start - int(exon["rel_start"])
        offset_end = seg_tx_end - int(exon["rel_start"])
        g_start = int(exon["start"] + offset_start)
        g_end = int(exon["start"] + offset_end)
        segments.append(
            {
                "chrom": exon["chrom"],
                "genomic_start": g_start,
                "genomic_end": g_end,
                "strand": exon["strand"],
                "exon_id": exon["exon_id"],
                "tx_seg_start": seg_tx_start,
                "tx_seg_end": seg_tx_end,
            }
        )
    return segments


def segments_to_bed(segments_df: pd.DataFrame) -> pd.DataFrame:
    return segments_df.assign(
        chrom=segments_df["chrom"],
        chromStart=segments_df["genomic_start"] - 1,
        chromEnd=segments_df["genomic_end"],
        name=segments_df["segment_id"],
        score=0,
        strand=segments_df["strand"],
    )[["chrom", "chromStart", "chromEnd", "name", "score", "strand"]]


def classify_pc_hits(
    hits: pd.DataFrame,
    segment_meta: pd.DataFrame,
    hit_type: str,
) -> pd.DataFrame:
    if hits.empty:
        return pd.DataFrame()
    h = hits.rename(columns={"lnc_name": "segment_id", "pc_name": "pc_hit"})
    return h.merge(
        segment_meta[["segment_id", "unique_guide_name", "Gene", "Transcript_ID"]],
        on="segment_id",
        how="left",
    ).assign(pc_hit_type=hit_type)


def run_junction_analysis(
    guide_coords_path: Path,
    gtf_path: Path,
    pc_genes_bed: Path,
    pc_exons_bed: Path,
    pc_cds_bed: Path,
    output_dir: Path,
    bedtools: str = "bedtools",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    guides = pd.read_csv(guide_coords_path, sep="\t", low_memory=False)
    gtf = read_gtf(gtf_path, gene_types={"lncRNA", "protein_coding"})

    # Re-map ALL guides to segments (single-exon guides → 1 segment; junction → 2+)
    records: list[dict] = []
    for _, row in guides.iterrows():
        tx_start = int(row["tx_start"])
        tx_end = int(row["tx_end"])
        exons = build_exon_model(gtf, row["Transcript_ID"])
        segs = map_interval_to_segments(row["Transcript_ID"], tx_start, tx_end, exons)
        for i, seg in enumerate(segs):
            records.append(
                {
                    "unique_guide_name": row["unique_guide_name"],
                    "Gene": row["Gene"],
                    "Transcript_ID": row["Transcript_ID"],
                    "GuideName": row["GuideName"],
                    "tx_start": tx_start,
                    "tx_end": tx_end,
                    "original_class": row["guide_specificity"],
                    "segment_idx": i,
                    "n_segments": len(segs),
                    "segment_id": f"{row['unique_guide_name']}_seg{i}",
                    **seg,
                }
            )

    seg_df = pd.DataFrame(records)
    seg_path = output_dir / "guide_segments_all.tsv"
    seg_df.to_csv(seg_path, sep="\t", index=False)

    bed_path = output_dir / "guide_segments.bed"
    segments_to_bed(seg_df).to_csv(bed_path, sep="\t", header=False, index=False)

    meta = seg_df[["segment_id", "unique_guide_name", "Gene", "Transcript_ID"]]

    thresh = OverlapThreshold(name="any", min_frac_query=0.0, same_strand=True)
    gene_hits = run_bedtools_intersect(bed_path, pc_genes_bed, thresh, bedtools)
    exon_hits = run_bedtools_intersect(bed_path, pc_exons_bed, thresh, bedtools)
    cds_hits = run_bedtools_intersect(bed_path, pc_cds_bed, thresh, bedtools)

    gene_cls = classify_pc_hits(gene_hits, meta, "gene_body")
    exon_cls = classify_pc_hits(exon_hits, meta, "exon")
    cds_cls = classify_pc_hits(cds_hits, meta, "cds")

    all_hits = pd.concat([gene_cls, exon_cls, cds_cls], ignore_index=True)
    hits_path = output_dir / "guide_segment_pc_hits.tsv"
    all_hits.to_csv(hits_path, sep="\t", index=False)

    # Per-guide summary after split mapping
    seg_df["pc_exon_hit"] = seg_df["segment_id"].map(
        exon_cls.groupby("segment_id")["pc_hit"].first() if not exon_cls.empty else {}
    )
    seg_df["pc_cds_hit"] = seg_df["segment_id"].map(
        cds_cls.groupby("segment_id")["pc_hit"].first() if not cds_cls.empty else {}
    )
    seg_df["pc_gene_body_hit"] = seg_df["segment_id"].map(
        gene_cls.groupby("segment_id")["pc_hit"].apply(lambda s: ";".join(sorted(set(s))))
        if not gene_cls.empty
        else {}
    )

    def guide_class(gdf: pd.DataFrame) -> str:
        if gdf["pc_cds_hit"].notna().any():
            return "segment_overlaps_pc_cds"
        if gdf["pc_exon_hit"].notna().any():
            return "segment_overlaps_pc_exon_not_cds"
        if gdf["pc_gene_body_hit"].notna().any():
            return "segment_overlaps_pc_gene_body_only"
        return "segment_lncrna_specific"

    guide_rows = []
    for key, gdf in seg_df.groupby(
        ["unique_guide_name", "Gene", "Transcript_ID", "GuideName", "original_class", "n_segments"]
    ):
        guide_rows.append(
            {
                **dict(
                    zip(
                        [
                            "unique_guide_name",
                            "Gene",
                            "Transcript_ID",
                            "GuideName",
                            "original_class",
                            "n_segments",
                        ],
                        key,
                        strict=False,
                    )
                ),
                "split_class": guide_class(gdf),
            }
        )
    guide_summary = pd.DataFrame(guide_rows)
    guide_summary_path = output_dir / "guide_split_pc_summary.tsv"
    guide_summary.to_csv(guide_summary_path, sep="\t", index=False)

    # Screen lncRNA exons vs PC (structure-level, not guides)
    screen_tx = guides["Transcript_ID"].map(strip_ensembl_version).unique()
    lnc_exons = gtf.loc[
        (gtf["feature"] == "exon")
        & (gtf["gene_type"] == "lncRNA")
        & (gtf["transcript_id_base"].isin(screen_tx))
    ].copy()
    lnc_exon_bed = output_dir / "screen_lncrna_exons.bed"
    lnc_exons.assign(
        chromStart=lnc_exons["start"] - 1,
        chromEnd=lnc_exons["end"],
        name=lnc_exons["transcript_id"],
        score=0,
    )[["chrom", "chromStart", "chromEnd", "name", "score", "strand"]].to_csv(
        lnc_exon_bed, sep="\t", header=False, index=False
    )

    lnc_exon_gene = run_bedtools_intersect(lnc_exon_bed, pc_genes_bed, thresh, bedtools)
    lnc_exon_exon = run_bedtools_intersect(lnc_exon_bed, pc_exons_bed, thresh, bedtools)
    lnc_exon_cds = run_bedtools_intersect(lnc_exon_bed, pc_cds_bed, thresh, bedtools)

    struct_rows = [
        {"layer": "lncrna_exon", "pc_target": "gene_body", "n_intervals": len(lnc_exon_gene)},
        {"layer": "lncrna_exon", "pc_target": "exon", "n_intervals": len(lnc_exon_exon)},
        {"layer": "lncrna_exon", "pc_target": "cds", "n_intervals": len(lnc_exon_cds)},
    ]
    struct_path = output_dir / "lncrna_exon_pc_overlap_counts.tsv"
    pd.DataFrame(struct_rows).to_csv(struct_path, sep="\t", index=False)

    # Count summary
    guide_summary["split_class"].value_counts()
    junction_only = guide_summary[guide_summary["original_class"] == "junction_spanning_unmapped"]
    junc_counts = junction_only["split_class"].value_counts()

    overview = pd.DataFrame(
        [
            {"metric": "n_guides_total", "value": len(guide_summary)},
            {"metric": "n_junction_guides", "value": len(junction_only)},
            {
                "metric": "n_split_pc_cds",
                "value": int((guide_summary["split_class"] == "segment_overlaps_pc_cds").sum()),
            },
            {
                "metric": "n_split_pc_exon",
                "value": int((guide_summary["split_class"] == "segment_overlaps_pc_exon_not_cds").sum()),
            },
            {
                "metric": "n_split_pc_gene_body_only",
                "value": int((guide_summary["split_class"] == "segment_overlaps_pc_gene_body_only").sum()),
            },
            {
                "metric": "n_split_lncrna_specific",
                "value": int((guide_summary["split_class"] == "segment_lncrna_specific").sum()),
            },
            {
                "metric": "n_junction_split_pc_gene_body_only",
                "value": int((junction_only["split_class"] == "segment_overlaps_pc_gene_body_only").sum()),
            },
            {
                "metric": "n_junction_split_lncrna_specific",
                "value": int((junction_only["split_class"] == "segment_lncrna_specific").sum()),
            },
            {"metric": "lncrna_exon_intervals_hit_pc_exon", "value": len(lnc_exon_exon)},
            {"metric": "lncrna_exon_intervals_hit_pc_cds", "value": len(lnc_exon_cds)},
        ]
    )
    overview_path = output_dir / "junction_split_overview.tsv"
    overview.to_csv(overview_path, sep="\t", index=False)

    print(overview.to_string(index=False))
    if len(junction_only):
        print("\nJunction guides after split:")
        print(junc_counts.to_string())

    return {
        "segments": seg_path,
        "hits": hits_path,
        "guide_summary": guide_summary_path,
        "overview": overview_path,
        "structure": struct_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--guide-coords",
        type=Path,
        default=RESULTS / "guide_specificity" / "lncrna_guide_genomic_coords.tsv",
    )
    parser.add_argument(
        "--gtf",
        type=Path,
        default=ANN / "genes.gtf",
        help="Filtered GTF (default: annotations/genes.gtf; use gencode.v46 for full annotation)",
    )
    parser.add_argument("--pc-genes-bed", type=Path, default=ANN / "protein_coding_genes.bed")
    parser.add_argument("--pc-exons-bed", type=Path, default=ANN / "protein_coding_exons.bed")
    parser.add_argument("--pc-cds-bed", type=Path, default=ANN / "protein_coding_cds.bed")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "junction_split_pc_overlap")
    parser.add_argument("--bedtools", default="bedtools")
    args = parser.parse_args()

    run_junction_analysis(
        args.guide_coords,
        args.gtf,
        args.pc_genes_bed,
        args.pc_exons_bed,
        args.pc_cds_bed,
        args.output_dir,
        args.bedtools,
    )


if __name__ == "__main__":
    main()
