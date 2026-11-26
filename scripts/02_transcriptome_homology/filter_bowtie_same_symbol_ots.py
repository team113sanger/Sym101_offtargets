#!/usr/bin/env python3
"""Post-filter Sym101 Bowtie v50 off-target tables.

Bowtie QC already drops alignments to the *intended Ensembl gene ID* (same ENSG,
including other isoforms). GENCODE nevertheless assigns the same gene *symbol*
to more than one ENSG (e.g. ACTG1 ENSG00000184009 → ACTG1 ENSG00000291420).
Those same-symbol / different-ENSG hits remain in the raw sites table and inflate
protein-coding fail rates.

This script:
  1. Archives the full non-self sites table (same ENSG already excluded).
  2. Writes a filtered sites table (different gene *symbol* only) in the Excel
     column layout used for sharing.
  3. Rebuilds guide- and gene-level library summaries under that filter.

  Usage:
  python3 scripts/filter_bowtie_same_symbol_ots.py \\
    --results-dir results/transcriptome_bowtie_qc_v50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SITES_COLS = [
    "unique_guide_name",
    "intended_gene",
    "query_class",
    "spacer_sequence",
    "perfect_match_rna_site",
    "offtarget_gene_symbol",
    "offtarget_gene_id",
    "offtarget_transcript_id",
    "transcript_header",
    "alignment_start_on_transcript",
    "n_mismatches",
    "mismatch_positions_on_spacer",
    "n_seed_mismatches_spacer_15_21",
    "has_seed_mismatch",
    "reverse_strand",
]

CLASS_ORDER = ["lncRNA", "protein_coding", "NT"]
CLOSEST_ORDER = {"exact": 0, "1_mismatch": 1, "2_mismatch": 2, "none": 3}


def bool_excel(series: pd.Series) -> pd.Series:
    """Match Excel TRUE/FALSE export style."""
    return series.map(lambda x: "TRUE" if bool(x) else "FALSE")


def mm_class(n: int) -> str:
    return {0: "exact", 1: "1_mismatch", 2: "2_mismatch"}[int(n)]


def library_summary_by_guide(guides: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls in CLASS_ORDER:
        sub = guides[guides["query_class"] == cls]
        fail = sub[sub["n_offtarget_genes_le2mm"] > 0]
        n = len(sub)
        n_fail = len(fail)
        rows.append(
            {
                "Class": cls,
                "n": n,
                "Off-target matches": f"{n_fail} ({100 * n_fail / n:.1f}%)" if n else "0",
                "Exact": int((fail["closest_offtarget_match"] == "exact").sum()),
                "1-mm": int((fail["closest_offtarget_match"] == "1_mismatch").sum()),
                "2-mm": int((fail["closest_offtarget_match"] == "2_mismatch").sum()),
                "Pass": int((sub["n_offtarget_genes_le2mm"] == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def library_summary_by_gene(guides: pd.DataFrame) -> pd.DataFrame:
    gene_rows = []
    for (cls, gene), sub in guides.groupby(["query_class", "Gene"], sort=False):
        ranks = sub["closest_offtarget_match"].map(CLOSEST_ORDER)
        closest = sub.loc[ranks.idxmin(), "closest_offtarget_match"]
        n_fail_g = int((sub["n_offtarget_genes_le2mm"] > 0).sum())
        gene_rows.append(
            {
                "query_class": cls,
                "Gene": gene,
                "n_guides": len(sub),
                "n_guides_with_OT": n_fail_g,
                "gene_has_OT": int(n_fail_g > 0),
                "closest_offtarget_match": closest,
            }
        )
    gene_df = pd.DataFrame(gene_rows)

    rows = []
    for cls in CLASS_ORDER:
        sub = gene_df[gene_df["query_class"] == cls]
        fail = sub[sub["gene_has_OT"] == 1]
        n = len(sub)
        n_fail = len(fail)
        rows.append(
            {
                "Class": cls,
                "n_genes": n,
                "Off-target matches": f"{n_fail} ({100 * n_fail / n:.1f}%)" if n else "0",
                "Exact": int((fail["closest_offtarget_match"] == "exact").sum()),
                "1-mm": int((fail["closest_offtarget_match"] == "1_mismatch").sum()),
                "2-mm": int((fail["closest_offtarget_match"] == "2_mismatch").sum()),
                "Pass": n - n_fail,
            }
        )
    return pd.DataFrame(rows), gene_df


def rebuild_guides_from_sites(all_guides: pd.DataFrame, sites_filt: pd.DataFrame) -> pd.DataFrame:
    """Recompute exclusive other-gene counts from filtered sites."""
    if sites_filt.empty:
        best = pd.DataFrame(
            columns=["unique_guide_name", "offtarget_gene_id", "offtarget_gene_symbol", "best_mm"]
        )
    else:
        best = (
            sites_filt.groupby(
                ["unique_guide_name", "offtarget_gene_id", "offtarget_gene_symbol"],
                as_index=False,
            )
            .agg(best_mm=("n_mismatches", "min"))
            .sort_values(["unique_guide_name", "best_mm", "offtarget_gene_id"])
            .drop_duplicates(["unique_guide_name", "offtarget_gene_id"])
        )

    site_counts = (
        sites_filt.groupby("unique_guide_name")
        .agg(
            n_offtarget_sites_le2mm=("offtarget_transcript_id", "size"),
            n_offtarget_transcripts_le2mm=("offtarget_transcript_id", "nunique"),
        )
        .reset_index()
        if not sites_filt.empty
        else pd.DataFrame(
            columns=[
                "unique_guide_name",
                "n_offtarget_sites_le2mm",
                "n_offtarget_transcripts_le2mm",
            ]
        )
    )

    rows = []
    for guide, g in best.groupby("unique_guide_name"):
        n_exact = int((g["best_mm"] == 0).sum())
        n_1 = int((g["best_mm"] == 1).sum())
        n_2 = int((g["best_mm"] == 2).sum())
        if n_exact:
            closest = "exact"
        elif n_1:
            closest = "1_mismatch"
        elif n_2:
            closest = "2_mismatch"
        else:
            closest = "none"
        rows.append(
            {
                "unique_guide_name": guide,
                "n_offtarget_genes_exact": n_exact,
                "n_offtarget_genes_1_mismatch": n_1,
                "n_offtarget_genes_2_mismatch": n_2,
                "n_offtarget_genes_le2mm": n_exact + n_1 + n_2,
                "closest_offtarget_match": closest,
                "offtarget_genes_exact": ";".join(
                    sorted(g.loc[g["best_mm"] == 0, "offtarget_gene_symbol"].astype(str))
                ),
                "offtarget_genes_1_mismatch": ";".join(
                    sorted(g.loc[g["best_mm"] == 1, "offtarget_gene_symbol"].astype(str))
                ),
                "offtarget_genes_2_mismatch": ";".join(
                    sorted(g.loc[g["best_mm"] == 2, "offtarget_gene_symbol"].astype(str))
                ),
            }
        )
    rebuilt = pd.DataFrame(rows)
    if not rebuilt.empty:
        rebuilt = rebuilt.merge(site_counts, on="unique_guide_name", how="left")
    else:
        rebuilt = site_counts.copy()
        rebuilt["n_offtarget_genes_exact"] = 0
        rebuilt["n_offtarget_genes_1_mismatch"] = 0
        rebuilt["n_offtarget_genes_2_mismatch"] = 0
        rebuilt["n_offtarget_genes_le2mm"] = 0
        rebuilt["closest_offtarget_match"] = "none"
        rebuilt["offtarget_genes_exact"] = ""
        rebuilt["offtarget_genes_1_mismatch"] = ""
        rebuilt["offtarget_genes_2_mismatch"] = ""

    keep = [
        "unique_guide_name",
        "query_class",
        "Gene",
        "intended_gene_id",
        "intended_transcript_id",
        "GuideSeq",
        "n_alignments_to_intended_gene_le2mm",
    ]
    # GuideSeq may be named spacer_sequence in guides.tsv
    base = all_guides.copy()
    if "GuideSeq" not in base.columns and "spacer_sequence" in base.columns:
        base = base.rename(columns={"spacer_sequence": "GuideSeq"})
    for col in keep:
        if col not in base.columns and col != "GuideSeq":
            raise SystemExit(f"Missing column in guide table: {col}")

    out = base[keep].merge(rebuilt, on="unique_guide_name", how="left")
    for col in [
        "n_offtarget_genes_exact",
        "n_offtarget_genes_1_mismatch",
        "n_offtarget_genes_2_mismatch",
        "n_offtarget_genes_le2mm",
        "n_offtarget_sites_le2mm",
        "n_offtarget_transcripts_le2mm",
    ]:
        out[col] = out[col].fillna(0).astype(int)
    out["closest_offtarget_match"] = out["closest_offtarget_match"].fillna("none")
    for col in [
        "offtarget_genes_exact",
        "offtarget_genes_1_mismatch",
        "offtarget_genes_2_mismatch",
    ]:
        out[col] = out[col].fillna("")
    return out


def gene_to_gene(sites_filt: pd.DataFrame, guides: pd.DataFrame) -> pd.DataFrame:
    if sites_filt.empty:
        return pd.DataFrame()
    best = sites_filt.groupby(
        [
            "unique_guide_name",
            "intended_gene",
            "query_class",
            "offtarget_gene_symbol",
            "offtarget_gene_id",
        ],
        as_index=False,
    ).agg(best_n_mismatches=("n_mismatches", "min"))
    best["best_match"] = best["best_n_mismatches"].map(mm_class)
    best = best.merge(guides[["unique_guide_name", "intended_gene_id"]], on="unique_guide_name", how="left")
    xy = best.groupby(
        [
            "query_class",
            "intended_gene",
            "intended_gene_id",
            "offtarget_gene_symbol",
            "offtarget_gene_id",
            "best_match",
        ],
        as_index=False,
    ).agg(n_guides=("unique_guide_name", "nunique"))
    wide = xy.pivot_table(
        index=[
            "query_class",
            "intended_gene",
            "intended_gene_id",
            "offtarget_gene_symbol",
            "offtarget_gene_id",
        ],
        columns="best_match",
        values="n_guides",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()
    for c in ["exact", "1_mismatch", "2_mismatch"]:
        if c not in wide.columns:
            wide[c] = 0
    wide["n_guides_with_OT_to_this_gene"] = wide["exact"] + wide["1_mismatch"] + wide["2_mismatch"]
    n_guides = (
        guides.groupby("Gene")
        .size()
        .rename("n_guides_targeting_intended")
        .reset_index()
        .rename(columns={"Gene": "intended_gene"})
    )
    wide = wide.merge(n_guides, on="intended_gene", how="left")
    wide["frac_guides_with_OT_to_this_gene"] = (
        wide["n_guides_with_OT_to_this_gene"] / wide["n_guides_targeting_intended"]
    ).round(4)

    def closest(r):
        if r["exact"] > 0:
            return "exact"
        if r["1_mismatch"] > 0:
            return "1_mismatch"
        return "2_mismatch"

    wide["closest_match_any_guide"] = wide.apply(closest, axis=1)
    return wide.sort_values(
        ["query_class", "intended_gene", "n_guides_with_OT_to_this_gene"],
        ascending=[True, True, False],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/transcriptome_bowtie_qc_v50"),
        help="Directory with sym101_guides_bowtie_gencode_v50.*.tsv*",
    )
    p.add_argument(
        "--sites-in",
        type=Path,
        default=None,
        help="Input sites TSV.gz (default: results-dir/...sites.tsv.gz or "
        "...sites.including_same_symbol.tsv.gz if already archived)",
    )
    args = p.parse_args()
    out = args.results_dir

    archived = out / "sym101_guides_bowtie_gencode_v50.sites.including_same_symbol.tsv.gz"
    default_sites = out / "sym101_guides_bowtie_gencode_v50.sites.tsv.gz"
    if args.sites_in is not None:
        sites_path = args.sites_in
    elif archived.exists():
        sites_path = archived
    else:
        sites_path = default_sites

    guides_path = out / "sym101_guides_bowtie_gencode_v50.tsv"
    if not guides_path.exists():
        guides_path = out / "sym101_guides_bowtie_gencode_v50.guides.tsv"

    sites = pd.read_csv(sites_path, sep="\t")
    missing = [c for c in SITES_COLS if c not in sites.columns]
    if missing:
        raise SystemExit(f"Sites table missing columns: {missing}")

    guides_raw = pd.read_csv(guides_path, sep="\t")

    same_symbol = sites["offtarget_gene_symbol"] == sites["intended_gene"]
    n_same = int(same_symbol.sum())
    sites_filt = sites.loc[~same_symbol, SITES_COLS].copy()

    # Archive full table once, then overwrite shareable sites with filtered
    if not archived.exists():
        sites[SITES_COLS].to_csv(archived, sep="\t", index=False, compression="gzip")
        print(f"Archived full non-self sites → {archived}")
    elif sites_path.resolve() != archived.resolve():
        # Re-archive from current input if caller pointed at a different full file
        sites[SITES_COLS].to_csv(archived, sep="\t", index=False, compression="gzip")
        print(f"Refreshed archive → {archived}")

    sites_out = sites_filt.copy()
    sites_out["has_seed_mismatch"] = bool_excel(sites_out["has_seed_mismatch"])
    sites_out["reverse_strand"] = bool_excel(sites_out["reverse_strand"])
    # Empty mismatch position cells as blank (not nan)
    sites_out["mismatch_positions_on_spacer"] = sites_out["mismatch_positions_on_spacer"].fillna("")
    sites_out.to_csv(default_sites, sep="\t", index=False, compression="gzip")

    guides_filt = rebuild_guides_from_sites(guides_raw, sites_filt)
    guide_sum = library_summary_by_guide(guides_filt)
    gene_sum, gene_df = library_summary_by_gene(guides_filt)
    xy = gene_to_gene(sites_filt, guides_filt)

    guides_filt.to_csv(out / "sym101_guides_bowtie_gencode_v50.excl_same_symbol.tsv", sep="\t", index=False)
    guide_sum.to_csv(out / "library_summary_by_guide_excl_same_symbol.tsv", sep="\t", index=False)
    gene_sum.to_csv(out / "library_summary_by_gene_excl_same_symbol.tsv", sep="\t", index=False)
    gene_df.to_csv(out / "gene_level_pass_fail_excl_same_symbol.tsv", sep="\t", index=False)
    xy.to_csv(out / "gene_to_gene_offtargets_excl_same_symbol.tsv", sep="\t", index=False)

    guide_sum.to_csv(out / "library_summary_by_guide.tsv", sep="\t", index=False)
    gene_sum.to_csv(out / "library_summary_by_gene.tsv", sep="\t", index=False)

    print(f"Input sites: {sites_path} ({len(sites)} rows)")
    print(f"Dropped same-symbol / different-ENSG rows: {n_same}")
    print(f"Filtered sites written: {default_sites} ({len(sites_filt)} rows)")
    print("Guide-level summary (excl. same-symbol):")
    print(guide_sum.to_string(index=False))
    print("Gene-level summary (excl. same-symbol):")
    print(gene_sum.to_string(index=False))


if __name__ == "__main__":
    main()
