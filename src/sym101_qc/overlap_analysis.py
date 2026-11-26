"""bedtools-based overlap analysis between screen lncRNAs and protein-coding genes."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BEDTOOLS_QUERY_COLS = [
    "lnc_chrom",
    "lnc_start",
    "lnc_end",
    "lnc_name",
    "lnc_score",
    "lnc_strand",
]
BEDTOOLS_SUBJECT_COLS = [
    "pc_chrom",
    "pc_start",
    "pc_end",
    "pc_name",
    "pc_score",
    "pc_strand",
]


@dataclass
class OverlapThreshold:
    """Parameters for one bedtools intersect run."""

    name: str
    min_frac_query: float = 0.0  # bedtools -f (fraction of A overlapping B)
    min_frac_subject: float | None = None  # bedtools -F (fraction of B overlapping A)
    same_strand: bool = True

    def bedtools_f_args(self) -> list[str]:
        args: list[str] = []
        if self.min_frac_query > 0:
            args.extend(["-f", str(self.min_frac_query)])
        if self.min_frac_subject is not None and self.min_frac_subject > 0:
            args.extend(["-F", str(self.min_frac_subject)])
        return args


# Default thresholds requested for reviewer analysis
DEFAULT_THRESHOLDS = [
    OverlapThreshold(name="any_overlap", min_frac_query=0.0),
    OverlapThreshold(name="10pct_of_lncrna", min_frac_query=0.1),
    OverlapThreshold(name="25pct_of_lncrna", min_frac_query=0.25),
    OverlapThreshold(name="50pct_of_lncrna", min_frac_query=0.5),
    OverlapThreshold(name="80pct_of_lncrna", min_frac_query=0.8),
]


def run_bedtools_intersect(
    query_bed: str | Path,
    subject_bed: str | Path,
    threshold: OverlapThreshold,
    bedtools: str = "bedtools",
) -> pd.DataFrame:
    """
    Run bedtools intersect and return parsed overlap table.

    Uses -wa -wb to retain both lncRNA (query) and protein-coding (subject) intervals.
    """
    query_bed = Path(query_bed)
    subject_bed = Path(subject_bed)
    cmd = [
        bedtools,
        "intersect",
        "-a",
        str(query_bed),
        "-b",
        str(subject_bed),
        "-wa",
        "-wb",
        *threshold.bedtools_f_args(),
    ]
    if threshold.same_strand:
        cmd.append("-s")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"bedtools intersect failed (exit {proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        ) from None
    if not proc.stdout.strip():
        return pd.DataFrame(columns=BEDTOOLS_QUERY_COLS + BEDTOOLS_SUBJECT_COLS)

    rows = [line.split("\t") for line in proc.stdout.strip().splitlines()]
    cols = BEDTOOLS_QUERY_COLS + BEDTOOLS_SUBJECT_COLS
    return pd.DataFrame(rows, columns=cols[: len(rows[0])])


def compute_overlap_fractions(overlaps: pd.DataFrame) -> pd.DataFrame:
    """
    For each overlap pair, compute overlap length and fraction of lncRNA / PC gene covered.

    BED coordinates are 0-based half-open (end exclusive).
    """
    if overlaps.empty:
        return pd.DataFrame(
            columns=[
                "lnc_name",
                "pc_name",
                "lnc_strand",
                "pc_strand",
                "overlap_bp",
                "lnc_length_bp",
                "pc_length_bp",
                "frac_of_lncrna",
                "frac_of_pc_gene",
            ]
        )

    df = overlaps.copy()
    lnc_start = df["lnc_start"].astype(int)
    lnc_end = df["lnc_end"].astype(int)
    pc_start = df["pc_start"].astype(int)
    pc_end = df["pc_end"].astype(int)

    ov_start = pd.concat([lnc_start, pc_start], axis=1).max(axis=1)
    ov_end = pd.concat([lnc_end, pc_end], axis=1).min(axis=1)
    overlap_bp = (ov_end - ov_start).clip(lower=0)

    lnc_len = (lnc_end - lnc_start).clip(lower=1)
    pc_len = (pc_end - pc_start).clip(lower=1)

    return pd.DataFrame(
        {
            "lnc_name": df["lnc_name"],
            "pc_name": df["pc_name"],
            "lnc_strand": df["lnc_strand"],
            "pc_strand": df["pc_strand"],
            "overlap_bp": overlap_bp,
            "lnc_length_bp": lnc_len,
            "pc_length_bp": pc_len,
            "frac_of_lncrna": (overlap_bp / lnc_len).round(4),
            "frac_of_pc_gene": (overlap_bp / pc_len).round(4),
        }
    )


def summarize_overlaps(
    overlaps: pd.DataFrame,
    screen_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarise overlap pairs to one row per lncRNA transcript."""
    if overlaps.empty:
        return pd.DataFrame(
            columns=[
                "lnc_name",
                "n_overlapping_pc_genes",
                "overlapping_pc_genes",
                "same_strand_overlaps",
            ]
        )

    df = overlaps.copy()
    df["same_strand"] = df["lnc_strand"] == df["pc_strand"]
    summary = (
        df.groupby("lnc_name", as_index=False)
        .agg(
            n_overlapping_pc_genes=("pc_name", "nunique"),
            overlapping_pc_genes=("pc_name", lambda s: ";".join(sorted(set(s)))),
            same_strand_overlaps=("same_strand", "sum"),
        )
        .rename(columns={"lnc_name": "transcript_id"})
    )

    if screen_summary is not None:
        summary = screen_summary.merge(
            summary,
            left_on="transcript_id_base",
            right_on="transcript_id",
            how="left",
        )
        summary["n_overlapping_pc_genes"] = summary["n_overlapping_pc_genes"].fillna(0).astype(int)
        summary["same_strand_overlaps"] = summary["same_strand_overlaps"].fillna(0).astype(int)
        summary["has_same_strand_pc_overlap"] = summary["same_strand_overlaps"] > 0
    return summary


def run_overlap_analysis(
    lncrna_bed: str | Path,
    pc_genes_bed: str | Path,
    output_dir: str | Path,
    screen_summary_path: str | Path | None = None,
    thresholds: list[OverlapThreshold] | None = None,
    bedtools: str = "bedtools",
) -> dict[str, Path]:
    """
    Run overlap analysis across multiple thresholds.

    Returns paths to detailed overlap TSVs and per-threshold summaries.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = thresholds or DEFAULT_THRESHOLDS

    screen_summary = None
    if screen_summary_path and Path(screen_summary_path).exists():
        screen_summary = pd.read_csv(screen_summary_path, sep="\t")

    outputs: dict[str, Path] = {}
    overview_rows = []

    for threshold in thresholds:
        overlaps = run_bedtools_intersect(lncrna_bed, pc_genes_bed, threshold, bedtools=bedtools)
        detail_path = output_dir / f"lncrna_pc_overlaps_{threshold.name}.tsv"
        overlaps.to_csv(detail_path, sep="\t", index=False)
        outputs[f"detail_{threshold.name}"] = detail_path

        if threshold.name == "any_overlap" and not overlaps.empty:
            frac_path = output_dir / "lncrna_pc_overlap_fractions.tsv"
            compute_overlap_fractions(overlaps).to_csv(frac_path, sep="\t", index=False)
            outputs["overlap_fractions"] = frac_path

        summary = summarize_overlaps(overlaps, screen_summary)
        summary_path = output_dir / f"lncrna_pc_overlap_summary_{threshold.name}.tsv"
        summary.to_csv(summary_path, sep="\t", index=False)
        outputs[f"summary_{threshold.name}"] = summary_path

        n_lnc = screen_summary.shape[0] if screen_summary is not None else overlaps["lnc_name"].nunique()
        n_hit = (
            summary["has_same_strand_pc_overlap"].sum()
            if "has_same_strand_pc_overlap" in summary.columns
            else summary["n_overlapping_pc_genes"].gt(0).sum()
        )
        overview_rows.append(
            {
                "threshold": threshold.name,
                "min_frac_query": threshold.min_frac_query,
                "same_strand_only": threshold.same_strand,
                "n_screen_lncrnas": n_lnc,
                "n_with_pc_overlap": int(n_hit),
                "frac_with_overlap": round(n_hit / n_lnc, 4) if n_lnc else 0,
                "n_overlap_pairs": len(overlaps),
            }
        )

    overview = pd.DataFrame(overview_rows)
    overview_path = output_dir / "lncrna_pc_overlap_overview.tsv"
    overview.to_csv(overview_path, sep="\t", index=False)
    outputs["overview"] = overview_path
    return outputs
