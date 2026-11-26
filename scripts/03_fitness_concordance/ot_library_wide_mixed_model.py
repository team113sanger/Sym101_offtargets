#!/usr/bin/env python3
"""Library-wide OT fitness sensitivity for lncRNA guides.

Pools guide-level LFCs across genes and asks: after accounting for gene identity
and cell line, is is_ot associated with a more negative LFC?

Primary model (fixed effects / OLS):
    LFC ~ is_ot + C(Gene) + C(line)

Optional mixed model if statsmodels used:
    LFC ~ is_ot + C(line) + (1 | Gene)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LINES = ["A549", "H1299", "H1975"]


def load_fig1c_lfc(path: Path) -> pd.DataFrame:
    hdr = pd.read_csv(path, header=None, nrows=5)
    df = pd.read_csv(path, header=None, skiprows=5)
    df = df.rename(columns={0: "unique_guide_name", 1: "fig_biotype", 2: "essential_flag"})
    cell = hdr.iloc[0, 3:].tolist()
    tp = hdr.iloc[2, 3:].tolist()
    out = df[["unique_guide_name"]].copy()
    for line in LINES:
        tp1_idx = [i for i, c in enumerate(cell) if c == line and tp[i] == "TP1"]
        tp6_idx = [i for i, c in enumerate(cell) if c == line and tp[i] == "TP6"]
        tp1 = df.iloc[:, [i + 3 for i in tp1_idx]].astype(float).mean(axis=1)
        tp6 = df.iloc[:, [i + 3 for i in tp6_idx]].astype(float).mean(axis=1)
        out[f"LFC_{line}"] = tp6 - tp1
    return out


def ols_with_dummies(y: np.ndarray, X: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Ordinary least squares with analytic SE; first column is intercept."""
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = n - p
    sigma2 = float((resid @ resid) / df) if df > 0 else np.nan
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    t = beta / se
    pvals = 2 * stats.t.sf(np.abs(t), df)
    ci_lo = beta - stats.t.ppf(0.975, df) * se
    ci_hi = beta + stats.t.ppf(0.975, df) * se
    return pd.DataFrame(
        {
            "term": names,
            "coef": beta,
            "se": se,
            "t": t,
            "p": pvals,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n": n,
            "df_resid": df,
            "r2": 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean())),
        }
    )


def fit_ols_gene_line_fe(df: pd.DataFrame) -> pd.DataFrame:
    """LFC ~ is_ot + C(Gene) + C(line) via dummy encoding (drop first)."""
    y = df["LFC"].to_numpy(dtype=float)
    is_ot = df["is_ot"].astype(float).to_numpy()
    gene_d = pd.get_dummies(df["Gene"], prefix="Gene", drop_first=True, dtype=float)
    line_d = pd.get_dummies(df["line"], prefix="line", drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(df)), is_ot, gene_d.to_numpy(), line_d.to_numpy()])
    names = ["Intercept", "is_ot"] + gene_d.columns.tolist() + line_d.columns.tolist()
    return ols_with_dummies(y, X, names)


def fit_within_gene_demean(df: pd.DataFrame) -> pd.DataFrame:
    """Within-gene demeaned regression: equivalent FE for is_ot (+ line)."""
    tmp = df.copy()
    line_d = pd.get_dummies(tmp["line"], prefix="line", drop_first=True, dtype=float)
    tmp = pd.concat([tmp.reset_index(drop=True), line_d.reset_index(drop=True)], axis=1)
    tmp["LFC_dm"] = tmp["LFC"] - tmp.groupby("Gene")["LFC"].transform("mean")
    tmp["is_ot_dm"] = tmp["is_ot"].astype(float) - tmp.groupby("Gene")["is_ot"].transform("mean")
    line_cols = list(line_d.columns)
    for c in line_cols:
        tmp[f"{c}_dm"] = tmp[c] - tmp.groupby("Gene")[c].transform("mean")
    y = tmp["LFC_dm"].to_numpy(dtype=float)
    X = np.column_stack([tmp["is_ot_dm"].to_numpy()] + [tmp[f"{c}_dm"].to_numpy() for c in line_cols])
    names = ["is_ot"] + [f"{c}_dm" for c in line_cols]
    return ols_with_dummies(y, X, names)


def try_mixedlm(df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None
    model = smf.mixedlm("LFC ~ is_ot + C(line)", df, groups=df["Gene"])
    res = model.fit(reml=False, method="lbfgs")
    rows = []
    for term, coef in res.fe_params.items():
        se = res.bse_fe[term]
        z = coef / se if se else np.nan
        p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
        rows.append(
            {
                "term": term,
                "coef": float(coef),
                "se": float(se),
                "z": float(z),
                "p": float(p),
                "ci_lo": float(coef - 1.96 * se),
                "ci_hi": float(coef + 1.96 * se),
                "n": int(res.nobs),
                "model": "mixedlm_Gene_RE",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--lncrna-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    ap.add_argument(
        "--fig1c",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "inputs"
        / "figure_source_data"
        / "Fig1c_source_data.csv",
    )
    args = ap.parse_args()

    ot_dir = args.lncrna_root / "results" / "03_fitness_concordance"
    out = ot_dir / "ot_fitness_concordance"
    out.mkdir(parents=True, exist_ok=True)

    guides = pd.read_csv(ot_dir / "sym101_guides_bowtie_gencode_v50.excl_same_symbol.tsv", sep="\t")
    guides["is_ot"] = guides["n_offtarget_genes_le2mm"] > 0
    lfc = load_fig1c_lfc(args.fig1c)
    long = guides.merge(lfc, on="unique_guide_name", how="inner")
    parts = []
    for line in LINES:
        t = long.copy()
        t["line"] = line
        t["LFC"] = t[f"LFC_{line}"]
        parts.append(t)
    joined = pd.concat(parts, ignore_index=True).dropna(subset=["LFC"])
    lnc = joined.query("query_class == 'lncRNA'").copy()
    lnc["is_ot"] = lnc["is_ot"].astype(int)

    n_guides = lnc["unique_guide_name"].nunique()
    n_ot = int((lnc.groupby("unique_guide_name")["is_ot"].first() == 1).sum())
    n_genes = lnc["Gene"].nunique()
    n_mixed = int(lnc.groupby("Gene")["is_ot"].agg(lambda s: s.min() == 0 and s.max() == 1).sum())

    ols = fit_ols_gene_line_fe(lnc)
    ols["model"] = "OLS_LFC_is_ot_CGene_Cline"
    within = fit_within_gene_demean(lnc)
    within["model"] = "within_gene_demean"

    ols_is = ols.loc[ols["term"] == "is_ot"].iloc[0]
    within_is = within.loc[within["term"] == "is_ot"].iloc[0]

    mixed = try_mixedlm(lnc)
    if mixed is not None:
        mixed.to_csv(out / "library_wide_OT_mixedlm.tsv", sep="\t", index=False)

    summary = {
        "n_lncRNA_guide_x_line_rows": int(len(lnc)),
        "n_unique_lncRNA_guides": int(n_guides),
        "n_OT_guides": int(n_ot),
        "n_lncRNA_genes": int(n_genes),
        "n_mixed_genes_OT_and_clean": int(n_mixed),
        "OLS_is_ot_coef": float(ols_is["coef"]),
        "OLS_is_ot_ci_lo": float(ols_is["ci_lo"]),
        "OLS_is_ot_ci_hi": float(ols_is["ci_hi"]),
        "OLS_is_ot_p": float(ols_is["p"]),
        "OLS_r2": float(ols_is["r2"]),
        "within_is_ot_coef": float(within_is["coef"]),
        "within_is_ot_ci_lo": float(within_is["ci_lo"]),
        "within_is_ot_ci_hi": float(within_is["ci_hi"]),
        "within_is_ot_p": float(within_is["p"]),
        "mixedlm_available": mixed is not None,
        "interpretation": (
            "is_ot coef ≈ average LFC shift for OT vs no-OT guides after absorbing "
            "gene identity (+ line). Near zero / NS ⇒ no detectable library-wide "
            "OT→fitness link among lncRNAs; still homology ≠ cleavage."
        ),
    }
    if mixed is not None:
        m_is = mixed.loc[mixed["term"] == "is_ot"].iloc[0]
        summary.update(
            {
                "mixedlm_is_ot_coef": float(m_is["coef"]),
                "mixedlm_is_ot_ci_lo": float(m_is["ci_lo"]),
                "mixedlm_is_ot_ci_hi": float(m_is["ci_hi"]),
                "mixedlm_is_ot_p": float(m_is["p"]),
            }
        )

    ols.to_csv(out / "library_wide_OT_OLS_full.tsv", sep="\t", index=False)
    within.to_csv(out / "library_wide_OT_within_gene.tsv", sep="\t", index=False)
    pd.DataFrame([summary]).to_csv(out / "library_wide_OT_effect_summary.tsv", sep="\t", index=False)
    with (out / "library_wide_OT_effect_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print("=== Library-wide lncRNA OT → LFC (sensitivity) ===")
    print(
        f"rows={summary['n_lncRNA_guide_x_line_rows']}  "
        f"guides={n_guides} (OT={n_ot})  genes={n_genes}  mixed_genes={n_mixed}"
    )
    print(
        f"OLS  is_ot = {ols_is['coef']:+.4f}  "
        f"95% CI [{ols_is['ci_lo']:+.4f}, {ols_is['ci_hi']:+.4f}]  "
        f"p={ols_is['p']:.3g}  R²={ols_is['r2']:.3f}"
    )
    print(
        f"Within-gene demean is_ot = {within_is['coef']:+.4f}  "
        f"95% CI [{within_is['ci_lo']:+.4f}, {within_is['ci_hi']:+.4f}]  "
        f"p={within_is['p']:.3g}"
    )
    if mixed is not None:
        print(
            f"MixedLM is_ot = {summary['mixedlm_is_ot_coef']:+.4f}  "
            f"95% CI [{summary['mixedlm_is_ot_ci_lo']:+.4f}, "
            f"{summary['mixedlm_is_ot_ci_hi']:+.4f}]  p={summary['mixedlm_is_ot_p']:.3g}"
        )
    else:
        print("MixedLM skipped (statsmodels not installed); OLS / within-FE written.")
    print(f"wrote {out}/library_wide_OT_effect_summary.tsv")


if __name__ == "__main__":
    main()
