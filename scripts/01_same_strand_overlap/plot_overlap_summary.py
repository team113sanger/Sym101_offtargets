#!/usr/bin/env python3
"""Generate summary plots from overlap and guide-specificity results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "01_same_strand_overlap"


def plot_overlap_overview(overview_path: Path, output_dir: Path) -> Path:
    df = pd.read_csv(overview_path, sep="\t")
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C78A8" if f == 0 else "#F58518" if f == 0.5 else "#E45756" for f in df["min_frac_query"]]
    bars = ax.bar(df["threshold"], df["frac_with_overlap"], color=colors)
    ax.set_ylabel("Fraction of screen lncRNAs with overlap")
    ax.set_xlabel("Overlap threshold")
    ax.set_ylim(0, 1)
    ax.set_title("Same-strand lncRNA vs protein-coding gene overlap")
    for bar, n, total in zip(bars, df["n_with_pc_overlap"], df["n_screen_lncrnas"], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{int(n)}/{int(total)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out = output_dir / "overlap_fraction_by_threshold.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_guide_specificity(guide_summary_path: Path, output_dir: Path) -> Path:
    df = pd.read_csv(guide_summary_path, sep="\t")
    counts = df.groupby("guide_specificity")["n_guides"].sum().sort_values(ascending=False)
    risk_order = [
        "overlaps_pc_cds",
        "overlaps_pc_exon_not_cds",
        "overlaps_pc_intron_only",
        "lncrna_specific",
        "junction_spanning_unmapped",
        "unmapped",
    ]
    counts = counts.reindex([c for c in risk_order if c in counts.index])
    color_map = {
        "overlaps_pc_cds": "#E45756",
        "overlaps_pc_exon_not_cds": "#F58518",
        "overlaps_pc_intron_only": "#FFC857",
        "lncrna_specific": "#54A24B",
        "junction_spanning_unmapped": "#BAB0AC",
        "unmapped": "#9D9D9D",
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        counts.index,
        counts.values,
        color=[color_map.get(k, "#4C78A8") for k in counts.index],
    )
    ax.set_ylabel("Number of guides")
    ax.set_xlabel("Guide specificity class")
    ax.set_title("lncRNA guide overlap with protein-coding annotation")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = output_dir / "guide_specificity_counts.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_gene_level_specificity(by_gene_path: Path, output_dir: Path, top_n: int = 20) -> Path | None:
    df = pd.read_csv(by_gene_path, sep="\t")
    df["n_confounded"] = df["n_guides"] - df["n_lncrna_specific"] - df["n_unmapped"]
    flagged = df.loc[df["n_confounded"] > 0].sort_values("n_confounded", ascending=False).head(top_n)
    if flagged.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(flagged))))
    y = range(len(flagged))
    ax.barh(list(y), flagged["n_lncrna_specific"], label="lncRNA-specific", color="#54A24B")
    ax.barh(
        list(y),
        flagged["n_confounded"],
        left=flagged["n_lncrna_specific"],
        label="overlaps PC",
        color="#E45756",
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(flagged["Gene"])
    ax.set_xlabel("Guide count")
    ax.set_title(f"Top {len(flagged)} lncRNAs with PC-overlapping guides")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = output_dir / "guide_specificity_top_genes.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_confounder_pairs(summary_path: Path, output_dir: Path) -> Path | None:
    """Bar chart of guide classes for the four overlapping lncRNA–PC pairs."""
    if not summary_path.exists():
        return None
    df = pd.read_csv(summary_path, sep="\t")
    if df.empty:
        return None

    metrics = ["n_lncrna_specific", "n_overlaps_pc_intron_only", "n_overlaps_pc_exon", "n_overlaps_pc_cds"]
    present = [m for m in metrics if m in df.columns]
    if not present:
        return None

    labels = [f"{g}\n(vs {p})" for g, p in zip(df["Gene"], df["pc_gene"], strict=False)]
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = [0] * len(df)
    colors = {
        "n_lncrna_specific": "#54A24B",
        "n_overlaps_pc_intron_only": "#FFC857",
        "n_overlaps_pc_exon": "#F58518",
        "n_overlaps_pc_cds": "#E45756",
    }
    for m in present:
        vals = df[m].fillna(0).astype(int)
        ax.bar(labels, vals, bottom=bottom, label=m.replace("n_", ""), color=colors.get(m, "#4C78A8"))
        bottom = [b + v for b, v in zip(bottom, vals, strict=False)]
    ax.set_ylabel("Guide count")
    ax.set_title("Guide specificity for lncRNAs with same-strand PC overlap")
    ax.legend(loc="upper right", fontsize=8)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out = output_dir / "confounder_pair_guides.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-dir", type=Path, default=RESULTS / "lncrna_pc_overlap")
    parser.add_argument("--guide-dir", type=Path, default=RESULTS / "guide_specificity")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "plots")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overview = args.overlap_dir / "lncrna_pc_overlap_overview.tsv"
    if overview.exists():
        print(f"wrote {plot_overlap_overview(overview, args.output_dir)}")
    else:
        print(f"skip overlap plot — not found: {overview}")

    guide_summary = args.guide_dir / "lncrna_guide_specificity_summary.tsv"
    if guide_summary.exists():
        print(f"wrote {plot_guide_specificity(guide_summary, args.output_dir)}")
    else:
        print(f"skip guide plot — not found: {guide_summary}")

    by_gene = args.guide_dir / "lncrna_guide_specificity_by_gene.tsv"
    if by_gene.exists():
        p = plot_gene_level_specificity(by_gene, args.output_dir)
        print(f"wrote {p}" if p else "skip gene-level plot — no confounded guides")

    conf_summary = args.overlap_dir.parent / "lncrna_confounder_summary.tsv"
    if not conf_summary.exists():
        conf_summary = RESULTS / "lncrna_confounder_summary.tsv"
    p = plot_confounder_pairs(conf_summary, args.output_dir)
    print(f"wrote {p}" if p else "skip confounder pair plot — run summarize_confounders.py first")


if __name__ == "__main__":
    main()
