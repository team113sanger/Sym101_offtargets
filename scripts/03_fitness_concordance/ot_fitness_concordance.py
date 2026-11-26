#!/usr/bin/env python3
"""Quantify transcriptome off-target status vs Sym101 fitness.

Uses:
  - Bowtie v50 OT freeze (excl. same-symbol)
  - Fig1c guide-level abundances → LFC = mean(TP6) − mean(TP1)
  - Supplementary Table 3 MAGeCK MLE gene β / Wald FDR (TP6)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

LINES = ["A549", "H1299", "H1975"]
# Fig1f-style thresholds (plot_fig1f_mle_volcano.py)
BETA_HIT = -0.5
WALD_FDR_HIT = 0.01  # -log10(FDR) > 2


def load_ot_guides(path: Path) -> pd.DataFrame:
    g = pd.read_csv(path, sep="\t")
    g["is_ot"] = g["n_offtarget_genes_le2mm"] > 0
    g["ot_class"] = g["closest_offtarget_match"].fillna("none")
    return g


def load_fig1c_lfc(path: Path) -> pd.DataFrame:
    """Guide-level LFC from Fig1c multi-header abundances."""
    hdr = pd.read_csv(path, header=None, nrows=5)
    df = pd.read_csv(path, header=None, skiprows=5)
    df = df.rename(columns={0: "unique_guide_name", 1: "fig_biotype", 2: "essential_flag"})
    cell = hdr.iloc[0, 3:].tolist()
    tp = hdr.iloc[2, 3:].tolist()

    out = df[["unique_guide_name", "fig_biotype", "essential_flag"]].copy()
    for line in LINES:
        tp1_idx = [i for i, c in enumerate(cell) if c == line and tp[i] == "TP1"]
        tp6_idx = [i for i, c in enumerate(cell) if c == line and tp[i] == "TP6"]
        if not tp1_idx or not tp6_idx:
            raise SystemExit(f"Missing TP1/TP6 columns for {line}")
        tp1 = df.iloc[:, [i + 3 for i in tp1_idx]].astype(float).mean(axis=1)
        tp6 = df.iloc[:, [i + 3 for i in tp6_idx]].astype(float).mean(axis=1)
        out[f"LFC_{line}"] = tp6 - tp1
        out[f"TP1_{line}"] = tp1
        out[f"TP6_{line}"] = tp6
    return out


def load_mle_tp6(path: Path) -> pd.DataFrame:
    mle = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    mle = mle[~mle.index.isin(["Gene", "variable"])]
    mle.index.name = "Gene"
    rows = []
    for line in LINES:
        beta = mle[("TP6", line, "beta")].astype(float)
        wald = mle[("TP6", line, "wald-fdr")].astype(float)
        tmp = pd.DataFrame({"Gene": beta.index, "line": line, "beta": beta.values, "wald_fdr": wald.values})
        tmp["is_hit"] = (tmp["beta"] < BETA_HIT) & (tmp["wald_fdr"] < WALD_FDR_HIT)
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def gene_ot_burden(guides: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene, sub in guides.groupby("Gene"):
        n = len(sub)
        n_ot = int(sub["is_ot"].sum())
        rows.append(
            {
                "Gene": gene,
                "query_class": sub["query_class"].iloc[0],
                "n_guides": n,
                "n_ot": n_ot,
                "n_clean": n - n_ot,
                "frac_ot": n_ot / n if n else np.nan,
                "n_exact": int((sub["ot_class"] == "exact").sum()),
                "n_1mm": int((sub["ot_class"] == "1_mismatch").sum()),
                "n_2mm": int((sub["ot_class"] == "2_mismatch").sum()),
                "any_ot": n_ot > 0,
                "all_clean": n_ot == 0,
                "mixed": n_ot > 0 and n_ot < n,
            }
        )
    return pd.DataFrame(rows)


def within_gene_contrast(joined: pd.DataFrame) -> pd.DataFrame:
    """Per gene × line: mean LFC OT vs clean; Wilcoxon if both sides n>=2."""
    rows = []
    for (gene, line), sub in joined.groupby(["Gene", "line"]):
        ot = sub.loc[sub["is_ot"], "LFC"].dropna()
        cl = sub.loc[~sub["is_ot"], "LFC"].dropna()
        if len(ot) == 0 or len(cl) == 0:
            continue
        diff = float(ot.mean() - cl.mean())  # negative => OT more depleted
        p = np.nan
        if len(ot) >= 2 and len(cl) >= 2:
            try:
                p = float(stats.mannwhitneyu(ot, cl, alternative="two-sided").pvalue)
            except ValueError:
                p = np.nan
        rows.append(
            {
                "Gene": gene,
                "query_class": sub["query_class"].iloc[0],
                "line": line,
                "n_ot": len(ot),
                "n_clean": len(cl),
                "mean_LFC_ot": float(ot.mean()),
                "mean_LFC_clean": float(cl.mean()),
                "delta_OT_minus_clean": diff,
                "p_mannwhitney": p,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_bh"] = _bh(out["p_mannwhitney"])
    return out


def _bh(pvals: pd.Series) -> pd.Series:
    p = pvals.astype(float).values
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return pd.Series(out, index=pvals.index)
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out_ok = np.empty(n)
    out_ok[order] = adj
    out[ok] = out_ok
    return pd.Series(out, index=pvals.index)


def seed_flags(sites_path: Path, guides: pd.DataFrame) -> pd.DataFrame:
    """Per guide: any OT site with seed mismatch (from filtered sites)."""
    sites = pd.read_csv(sites_path, sep="\t")
    seed = sites["has_seed_mismatch"].astype(str).str.upper().isin(["TRUE", "1", "T"])
    per = (
        sites.assign(seed=seed)
        .groupby("unique_guide_name", as_index=False)
        .agg(n_ot_sites=("n_mismatches", "size"), n_ot_sites_seed_mm=("seed", "sum"))
    )
    per["has_seed_mm_ot"] = per["n_ot_sites_seed_mm"] > 0
    return guides.merge(per, on="unique_guide_name", how="left").fillna(
        {"n_ot_sites": 0, "n_ot_sites_seed_mm": 0, "has_seed_mm_ot": False}
    )


def rad_style_lncrna(guides: pd.DataFrame, xy: pd.DataFrame, mle: pd.DataFrame) -> pd.DataFrame:
    """For lncRNA genes with OT: are OT partners essential PC hits in MLE?"""
    lnc_ot = guides[(guides["query_class"] == "lncRNA") & guides["is_ot"]]
    genes = sorted(lnc_ot["Gene"].unique())
    mle_hit = mle[mle["is_hit"]][["Gene", "line"]].drop_duplicates()
    rows = []
    for gene in genes:
        partners = xy[xy["intended_gene"] == gene]
        if partners.empty:
            continue
        for line in LINES:
            hits = set(mle_hit.loc[mle_hit["line"] == line, "Gene"])
            ot_genes = partners["offtarget_gene_symbol"].astype(str).tolist()
            partners["offtarget_gene_id"].astype(str).tolist()
            # match by symbol primarily
            n_essential_ot = sum(1 for g in ot_genes if g in hits)
            rows.append(
                {
                    "intended_lncRNA": gene,
                    "line": line,
                    "n_ot_partner_genes": len(partners),
                    "n_ot_partners_essential_hit": n_essential_ot,
                    "ot_partners": ";".join(partners["offtarget_gene_symbol"].astype(str)),
                    "essential_ot_partners": ";".join([g for g in ot_genes if g in hits]),
                }
            )
    return pd.DataFrame(rows)


def sensitivity_drop_ot(joined: pd.DataFrame, burden: pd.DataFrame, mle: pd.DataFrame) -> pd.DataFrame:
    """Approximate gene score as median guide LFC; compare all vs clean-only."""
    rows = []
    for line in LINES:
        sub = joined[joined["line"] == line]
        med_all = sub.groupby("Gene")["LFC"].median()
        med_clean = sub.loc[~sub["is_ot"]].groupby("Gene")["LFC"].median()
        tmp = burden[["Gene", "query_class", "frac_ot", "n_ot", "n_clean", "mixed"]].merge(
            med_all.rename("median_LFC_all").reset_index(), on="Gene", how="left"
        )
        tmp = tmp.merge(med_clean.rename("median_LFC_clean_only").reset_index(), on="Gene", how="left")
        tmp["line"] = line
        tmp["delta_median_LFC"] = tmp["median_LFC_clean_only"] - tmp["median_LFC_all"]
        # join MLE
        m = mle[mle["line"] == line][["Gene", "beta", "wald_fdr", "is_hit"]]
        tmp = tmp.merge(m, on="Gene", how="left")
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lncrna-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    p.add_argument(
        "--updating-plot",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "inputs" / "figure_source_data",
    )
    args = p.parse_args()

    ot_dir = args.lncrna_root / "results" / "03_fitness_concordance"
    out = ot_dir / "ot_fitness_concordance"
    plot_dir = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    guides = load_ot_guides(ot_dir / "sym101_guides_bowtie_gencode_v50.excl_same_symbol.tsv")
    guides = seed_flags(ot_dir / "sym101_guides_bowtie_gencode_v50.sites.tsv.gz", guides)
    lfc = load_fig1c_lfc(args.updating_plot / "Fig1c_source_data.csv")
    mle = load_mle_tp6(args.updating_plot / "Supplementary_Table_3_SYM101_MAGeCK_MLE.csv")
    xy = pd.read_csv(ot_dir / "gene_to_gene_offtargets_excl_same_symbol.tsv", sep="\t")
    burden = gene_ot_burden(guides)

    # Long guide×line table
    long = guides.merge(lfc, on="unique_guide_name", how="inner")
    print(f"Guides with Fig1c LFC: {long['unique_guide_name'].nunique()} / {len(guides)}")
    parts = []
    for line in LINES:
        t = long.copy()
        t["line"] = line
        t["LFC"] = t[f"LFC_{line}"]
        parts.append(t)
    joined = pd.concat(parts, ignore_index=True)

    # --- Part 1: within-gene ---
    contrast = within_gene_contrast(joined)
    contrast.to_csv(out / "within_gene_OT_vs_clean_LFC.tsv", sep="\t", index=False)
    lnc_contrast = contrast[contrast["query_class"] == "lncRNA"].copy()
    pc_contrast = contrast[contrast["query_class"] == "protein_coding"].copy()
    lnc_contrast.to_csv(out / "within_gene_OT_vs_clean_LFC_lncRNA.tsv", sep="\t", index=False)
    pc_contrast.to_csv(out / "within_gene_OT_vs_clean_LFC_PC.tsv", sep="\t", index=False)

    # Feature controls
    feature_genes = ["SNHG15", "SNHG16", "SNHG6", "SNHG1", "PPIA"]
    feat_rows = []
    for gene in feature_genes:
        g = guides[guides["Gene"] == gene]
        feat_rows.append(
            {
                "Gene": gene,
                "n_guides": len(g),
                "n_ot": int(g["is_ot"].sum()),
                "frac_ot": float(g["is_ot"].mean()) if len(g) else np.nan,
                "ot_classes": dict(g["ot_class"].value_counts()),
                "note": (
                    "negative_control_no_OT"
                    if gene.startswith("SNHG") and g["is_ot"].sum() == 0
                    else ("positive_control_paralog_OT" if gene == "PPIA" else "")
                ),
            }
        )
    pd.DataFrame(feat_rows).to_csv(out / "feature_gene_OT_status.tsv", sep="\t", index=False)

    # --- Part 2: gene-level frac_OT vs beta ---
    gene_fit = burden.merge(mle, on="Gene", how="inner")
    gene_fit.to_csv(out / "gene_fracOT_vs_MLE_beta.tsv", sep="\t", index=False)

    # Enrichment: high frac_OT among hits
    enrich_rows = []
    for line in LINES:
        sub = gene_fit[gene_fit["line"] == line].dropna(subset=["frac_ot", "is_hit"])
        for cls in ["lncRNA", "protein_coding", "all"]:
            s = sub if cls == "all" else sub[sub["query_class"] == cls]
            if s.empty:
                continue
            high = s["frac_ot"] >= 0.5
            # contingency: high OT vs hit
            table = pd.crosstab(high, s["is_hit"])
            # ensure 2x2
            for a in [False, True]:
                for b in [False, True]:
                    if a not in table.index:
                        table.loc[a] = 0
                    if b not in table.columns:
                        table[b] = 0
            table = table.reindex(index=[False, True], columns=[False, True]).fillna(0)
            oddsr, p = stats.fisher_exact(table.values)
            enrich_rows.append(
                {
                    "line": line,
                    "class": cls,
                    "n_genes": len(s),
                    "n_hits": int(s["is_hit"].sum()),
                    "n_high_fracOT_ge0.5": int(high.sum()),
                    "n_hits_and_high_fracOT": int(((s["is_hit"]) & high).sum()),
                    "odds_ratio_hit_vs_high_fracOT": oddsr,
                    "fisher_p": p,
                    "median_fracOT_hits": float(s.loc[s["is_hit"], "frac_ot"].median())
                    if s["is_hit"].any()
                    else np.nan,
                    "median_fracOT_nonhits": float(s.loc[~s["is_hit"], "frac_ot"].median())
                    if (~s["is_hit"]).any()
                    else np.nan,
                }
            )
    enrich = pd.DataFrame(enrich_rows)
    enrich.to_csv(out / "hit_enrichment_vs_fracOT.tsv", sep="\t", index=False)

    # SNHG15/16 under frac_OT==0 restriction
    snhg = gene_fit[gene_fit["Gene"].isin(["SNHG15", "SNHG16", "SNHG6"])]
    snhg.to_csv(out / "SNHG_status_under_OT_filter.tsv", sep="\t", index=False)

    # --- Part 3 extras ---
    # Stratify guide LFC by OT class
    strat_rows = []
    for line in LINES:
        sub = joined[joined["line"] == line]
        for cls in ["none", "exact", "1_mismatch", "2_mismatch"]:
            s = sub[sub["ot_class"] == cls]["LFC"].dropna()
            if s.empty:
                continue
            strat_rows.append(
                {
                    "line": line,
                    "ot_class": cls,
                    "n_guides": len(s),
                    "mean_LFC": float(s.mean()),
                    "median_LFC": float(s.median()),
                    "mean_LFC_lncRNA": float(
                        sub[(sub.ot_class == cls) & (sub.query_class == "lncRNA")]["LFC"].mean()
                    )
                    if ((sub.ot_class == cls) & (sub.query_class == "lncRNA")).any()
                    else np.nan,
                }
            )
    pd.DataFrame(strat_rows).to_csv(out / "LFC_by_OT_class.tsv", sep="\t", index=False)

    # Seed mismatch among OT guides
    seed_sum = []
    for line in LINES:
        sub = joined[(joined["line"] == line) & joined["is_ot"]]
        for flag, label in [(True, "OT_with_seed_mm"), (False, "OT_no_seed_mm")]:
            s = sub[sub["has_seed_mm_ot"] == flag]["LFC"].dropna()
            seed_sum.append(
                {
                    "line": line,
                    "group": label,
                    "n": len(s),
                    "mean_LFC": float(s.mean()) if len(s) else np.nan,
                    "median_LFC": float(s.median()) if len(s) else np.nan,
                }
            )
    pd.DataFrame(seed_sum).to_csv(out / "LFC_by_seed_mismatch.tsv", sep="\t", index=False)

    sens = sensitivity_drop_ot(joined, burden, mle)
    sens.to_csv(out / "sensitivity_drop_OT_guides_medianLFC.tsv", sep="\t", index=False)

    rad = rad_style_lncrna(guides, xy, mle)
    rad.to_csv(out / "lncRNA_OT_partners_essential_Rad_style.tsv", sep="\t", index=False)

    # --- Plots ---
    # 1) within-gene delta for lncRNA mixed
    if not lnc_contrast.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, line in zip(axes, LINES, strict=False):
            s = lnc_contrast[lnc_contrast["line"] == line].sort_values("delta_OT_minus_clean")
            colors = ["#c44e52" if d < 0 else "#4c72b0" for d in s["delta_OT_minus_clean"]]
            ax.barh(s["Gene"], s["delta_OT_minus_clean"], color=colors)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"lncRNA mixed — {line}")
            ax.set_xlabel("mean LFC(OT) − LFC(clean)")
        fig.suptitle("Within-gene: OT vs clean guide LFC (TP6−TP1)", y=1.02)
        fig.tight_layout()
        fig.savefig(plot_dir / "within_gene_lncRNA_delta_LFC.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 2) frac_OT vs beta
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, line in zip(axes, LINES, strict=False):
        s = gene_fit[gene_fit["line"] == line]
        for cls, color, lab in [
            ("protein_coding", "#4c72b0", "PC"),
            ("lncRNA", "#55a868", "lncRNA"),
        ]:
            ss = s[s["query_class"] == cls]
            ax.scatter(
                ss["frac_ot"],
                ss["beta"],
                s=12,
                alpha=0.45,
                c=color,
                label=lab,
                linewidths=0,
            )
        # highlight SNHG / PPIA
        for gene, marker in [("SNHG15", "*"), ("SNHG16", "D"), ("SNHG6", "s"), ("PPIA", "X")]:
            ss = s[s["Gene"] == gene]
            if not ss.empty:
                ax.scatter(
                    ss["frac_ot"],
                    ss["beta"],
                    s=80,
                    marker=marker,
                    c="black",
                    label=gene,
                    zorder=5,
                )
        ax.axhline(BETA_HIT, color="k", ls="--", lw=0.8)
        ax.set_xlabel("frac_OT guides")
        ax.set_title(line)
        if line == "A549":
            ax.set_ylabel("MLE β (TP6)")
            ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Gene-level OT burden vs MAGeCK MLE β", y=1.02)
    fig.tight_layout()
    fig.savefig(plot_dir / "fracOT_vs_MLE_beta.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3) LFC by OT class
    strat = pd.read_csv(out / "LFC_by_OT_class.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(7, 4))
    classes = ["none", "exact", "1_mismatch", "2_mismatch"]
    x = np.arange(len(classes))
    width = 0.25
    for i, line in enumerate(LINES):
        vals = [strat.loc[(strat.line == line) & (strat.ot_class == c), "median_LFC"].values for c in classes]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(x + i * width, vals, width, label=line)
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Median guide LFC (TP6−TP1)")
    ax.set_title("Guide LFC by closest OT class")
    ax.legend()
    ax.axhline(0, color="k", lw=0.6)
    fig.tight_layout()
    fig.savefig(plot_dir / "LFC_by_OT_class.png", dpi=150)
    plt.close()

    # Summary markdown
    n_mixed_lnc = int(burden[(burden.query_class == "lncRNA") & burden.mixed].shape[0])
    n_mixed_pc = int(burden[(burden.query_class == "protein_coding") & burden.mixed].shape[0])
    lines_md = [
        "# OT ↔ fitness concordance (Sym101)",
        "",
        "## Inputs",
        "- OT freeze: `sym101_guides_bowtie_gencode_v50.excl_same_symbol.tsv`",
        "- Guide LFC: Fig1c abundances, **LFC = mean(TP6) − mean(TP1)** per line (not a MAGeCK sgrna export; none found under `updating_plot/` besides Fig1c).",
        "- Gene β: `Supplementary_Table_3_SYM101_MAGeCK_MLE.csv` TP6; hit = β < −0.5 and Wald FDR < 0.01 (Fig1f thresholds).",
        "",
        "## Library OT burden (excl. same-symbol)",
        f"- Mixed genes (OT + clean guides): lncRNA **{n_mixed_lnc}**, PC **{n_mixed_pc}**",
        "- SNHG15 / SNHG16 / SNHG6: **0 OT guides** (negative controls for this analysis)",
        "- PPIA: high OT fraction (paralog / PPIAL4-family; positive control)",
        "",
        "## Key outputs",
        "- `within_gene_OT_vs_clean_LFC_lncRNA.tsv` — Part 1 primary",
        "- `gene_fracOT_vs_MLE_beta.tsv`, `hit_enrichment_vs_fracOT.tsv` — Part 2",
        "- `LFC_by_OT_class.tsv`, `sensitivity_drop_OT_guides_medianLFC.tsv`,",
        "  `lncRNA_OT_partners_essential_Rad_style.tsv` — Part 3",
        "- `plots/` — figures",
        "",
        "## Framing",
        "Homology ≠ cleavage. These tests ask whether OT-flagged guides/genes show",
        "different fitness behaviour than clean counterparts.",
        "",
    ]
    # Add quick numeric headlines
    if not lnc_contrast.empty:
        for line in LINES:
            s = lnc_contrast[lnc_contrast["line"] == line]
            more_dep = (s["delta_OT_minus_clean"] < 0).mean()
            lines_md.append(
                f"- lncRNA mixed {line}: {len(s)} genes; "
                f"{100 * more_dep:.0f}% have OT guides more depleted than clean "
                f"(median Δ={s['delta_OT_minus_clean'].median():.3f})"
            )
    for line in LINES:
        e = enrich[(enrich.line == line) & (enrich["class"] == "lncRNA")]
        if not e.empty:
            r = e.iloc[0]
            lines_md.append(
                f"- lncRNA hit enrichment vs frac_OT≥0.5 ({line}): "
                f"OR={r['odds_ratio_hit_vs_high_fracOT']:.2f}, Fisher p={r['fisher_p']:.3g}"
            )
    for gene in ["SNHG15", "SNHG16", "SNHG6"]:
        ss = snhg[snhg.Gene == gene]
        if ss.empty:
            continue
        hits = ", ".join(f"{r.line}:β={r.beta:.2f},hit={bool(r.is_hit)}" for _, r in ss.iterrows())
        lines_md.append(f"- {gene} (frac_OT=0): {hits}")

    (out / "SUMMARY_ot_fitness.md").write_text("\n".join(lines_md) + "\n")
    print("Wrote", out)
    print((out / "SUMMARY_ot_fitness.md").read_text())


if __name__ == "__main__":
    main()
