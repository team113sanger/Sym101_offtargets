#!/usr/bin/env python3
"""Look up DepMap Chronos scores for the lncRNA off-target partners, for any release.

One script for both releases.

    --release 24Q4   regenerate the published table from raw, and (with --verify) check
                     it reproduces the frozen copies exactly
    --release 26Q3   re-score the same partners against a later release

Usage::

    python scripts/03_fitness_concordance/depmap_partner_chronos.py --release 24Q4 --verify
    python scripts/03_fitness_concordance/depmap_partner_chronos.py --release 24Q4
    python scripts/03_fitness_concordance/depmap_partner_chronos.py --release 26Q3

Inputs and output locations default to conf/manifest.yaml.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

from sym101_qc.manifest import get, load_manifest, resolve_path  # noqa: E402
from sym101_qc.run_manifest import RunManifest  # noqa: E402

RELEASES = {"24Q4": "references.depmap_24Q4.path", "26Q3": "references.depmap_26Q3.path"}
CHRONOS_ESSENTIAL = -0.5
CHRONOS_SENSITIVE = -1.0


def require_input(path: Path, key: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing input {key}: {path}\n  (resolved from conf/manifest.yaml)")
    return path


def partner_list(gene_to_gene: Path) -> pd.DataFrame:
    """The lncRNA off-target partners, from the deposited gene-to-gene table."""
    table = pd.read_csv(gene_to_gene, sep="\t", low_memory=False)
    return (
        table.loc[
            table["query_class"] == "lncRNA",
            ["intended_gene", "offtarget_gene_symbol", "offtarget_gene_id"],
        ]
        .drop_duplicates()
        .sort_values(["intended_gene", "offtarget_gene_symbol"])
    )


def chronos_lookup(partners: pd.DataFrame, gene_effect: Path, models: Path, aliases: dict) -> pd.DataFrame:
    """One row per partner x screen line, with the Chronos score where it exists."""
    header = pd.read_csv(gene_effect, nrows=0)
    symbols = set(partners["offtarget_gene_symbol"])
    # DepMap names gene columns "SYMBOL (ENTREZ)"; the matrix is never loaded whole.
    columns = {c.split(" (")[0]: c for c in header.columns if c.split(" (")[0] in symbols}

    index_column = header.columns[0]
    wanted = {spellings["depmap_model"] for spellings in aliases.values()}
    effect = pd.read_csv(gene_effect, usecols=[index_column, *columns.values()]).set_index(index_column)
    effect = effect.loc[effect.index.intersection(wanted)]

    model_table = pd.read_csv(models, usecols=["ModelID", "StrippedCellLineName"])
    display = dict(zip(model_table["ModelID"], model_table["StrippedCellLineName"], strict=False))
    screen_name = {s["depmap_model"]: s["screen"] for s in aliases.values()}

    rows = []
    for partner in partners.itertuples(index=False):
        column = columns.get(partner.offtarget_gene_symbol)
        for model_id in sorted(effect.index, key=lambda m: screen_name[m]):
            rows.append(
                {
                    "intended_gene": partner.intended_gene,
                    "offtarget_gene": partner.offtarget_gene_symbol,
                    "offtarget_gene_id": partner.offtarget_gene_id,
                    "cell_line": screen_name[model_id],
                    "depmap_id": model_id,
                    "depmap_display_name": display.get(model_id, ""),
                    "Chronos": float(effect.loc[model_id, column]) if column else None,
                    "Chronos_available": column is not None,
                }
            )
    annotated = pd.DataFrame(rows)
    annotated["Chronos_lt_neg0.5"] = annotated["Chronos"] < CHRONOS_ESSENTIAL
    annotated["Chronos_lt_neg1"] = annotated["Chronos"] < CHRONOS_SENSITIVE
    # The frozen table leaves the model identifiers blank where there is no score, even
    # though the model is known. Matched so the rebuild reproduces the archive.
    annotated.loc[~annotated["Chronos_available"], ["depmap_id", "depmap_display_name"]] = pd.NA
    return annotated


def scored_view(annotated: pd.DataFrame) -> pd.DataFrame:
    scored = annotated.loc[annotated["Chronos_available"]].copy()
    return scored[
        ["offtarget_gene", "cell_line", "Chronos", "depmap_id", "depmap_display_name", "intended_gene"]
    ].sort_values(["cell_line", "offtarget_gene"])


def compare(rebuilt: pd.DataFrame, frozen: pd.DataFrame, key: list[str], name: str) -> dict:
    shared = [c for c in rebuilt.columns if c in frozen.columns]
    left = rebuilt.sort_values(key).reset_index(drop=True)
    right = frozen.sort_values(key).reset_index(drop=True)
    mismatched = {}
    for column in shared:
        a, b = left[column], right[column]
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            equal = ((pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce")).abs() < 1e-9) | (
                a.isna() & b.isna()
            )
        else:
            equal = (a.astype(str) == b.astype(str)) | (a.isna() & b.isna())
        if not bool(equal.all()):
            mismatched[column] = int((~equal).sum())
    return {
        "table": name,
        "rebuilt_rows": len(rebuilt),
        "frozen_rows": len(frozen),
        "columns_compared": len(shared),
        "columns_mismatching": mismatched,
        "identical": not mismatched and len(rebuilt) == len(frozen),
    }


def main() -> None:
    manifest = load_manifest()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--release", choices=sorted(RELEASES), default="24Q4")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verify", action="store_true", help="24Q4 only: compare against the frozen tables")
    args = parser.parse_args()

    release_root = resolve_path(get(manifest, RELEASES[args.release]))
    gene_effect = require_input(release_root / "CRISPRGeneEffect.csv", f"{args.release} gene effect")
    models = require_input(release_root / "Model.csv", f"{args.release} models")
    gene_to_gene = require_input(
        resolve_path(get(manifest, "steps.step_02.outputs.gene_to_gene")), "gene_to_gene"
    )

    concordance = resolve_path(get(manifest, "steps.step_03.outputs.dir"))
    output_dir = args.output_dir or (
        concordance / ("depmap_partners" if args.release == "24Q4" else "depmap_release_check")
    )

    partners = partner_list(gene_to_gene)
    annotated = chronos_lookup(partners, gene_effect, models, get(manifest, "project.cell_line_aliases"))
    scored = scored_view(annotated)
    print(
        f"{args.release}: {len(partners)} partner entries, "
        f"{partners['offtarget_gene_symbol'].nunique()} unique partners, "
        f"{scored['offtarget_gene'].nunique()} with a Chronos score"
    )

    summary = {
        "release": args.release,
        "n_partner_entries": int(len(partners)),
        "n_unique_partners": int(partners["offtarget_gene_symbol"].nunique()),
        "n_with_chronos": int(scored["offtarget_gene"].nunique()),
        "n_reaching_chronos_essential": int((scored["Chronos"] < CHRONOS_ESSENTIAL).sum()),
        "n_reaching_chronos_sensitive": int((scored["Chronos"] < CHRONOS_SENSITIVE).sum()),
        "most_negative": {
            "gene": scored.loc[scored["Chronos"].idxmin(), "offtarget_gene"],
            "line": scored.loc[scored["Chronos"].idxmin(), "cell_line"],
            "chronos": round(float(scored["Chronos"].min()), 4),
        },
    }

    # Against the release of record, so a later release can be read as a difference.
    frozen_scored = concordance / "lncRNA_OT_partners_DepMap_Chronos_24Q4.tsv"
    if args.release != "24Q4" and frozen_scored.is_file():
        reference = pd.read_csv(frozen_scored, sep="\t").rename(columns={"Chronos": "chronos_24Q4"})
        merged = reference.merge(
            scored[["offtarget_gene", "cell_line", "Chronos"]].rename(
                columns={"Chronos": f"chronos_{args.release}"}
            ),
            on=["offtarget_gene", "cell_line"],
            how="left",
        )
        merged[f"delta_{args.release}_minus_24Q4"] = (
            merged[f"chronos_{args.release}"] - merged["chronos_24Q4"]
        )
        merged["essential_24Q4"] = merged["chronos_24Q4"] < CHRONOS_ESSENTIAL
        merged[f"essential_{args.release}"] = merged[f"chronos_{args.release}"] < CHRONOS_ESSENTIAL
        summary["vs_24Q4"] = {
            "n_changing_essentiality_call": int(
                (merged["essential_24Q4"] != merged[f"essential_{args.release}"]).sum()
            ),
            "max_abs_delta": round(float(merged[f"delta_{args.release}_minus_24Q4"].abs().max()), 4),
            "median_delta": round(float(merged[f"delta_{args.release}_minus_24Q4"].median()), 4),
        }
        comparison = merged

    reports = []
    if args.verify and args.release == "24Q4":
        if frozen_scored.is_file():
            reports.append(
                compare(
                    scored,
                    pd.read_csv(frozen_scored, sep="\t"),
                    ["offtarget_gene", "cell_line", "intended_gene"],
                    "scored_partners",
                )
            )
        frozen_annotated = concordance / "OT_partners_DepMap_Chronos.tsv"
        if frozen_annotated.is_file():
            frozen = pd.read_csv(frozen_annotated, sep="\t")
            shared = [c for c in annotated.columns if c in frozen.columns]
            reports.append(
                compare(
                    annotated[shared],
                    frozen[shared],
                    ["intended_gene", "offtarget_gene", "cell_line"],
                    "annotated_partners",
                )
            )
        summary["verification"] = reports
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.release == "24Q4":
            scored.to_csv(output_dir / "lncRNA_OT_partners_DepMap_Chronos_24Q4.tsv", sep="\t", index=False)
            annotated.to_csv(output_dir / "OT_partners_DepMap_Chronos.tsv", sep="\t", index=False)
        else:
            comparison.to_csv(
                output_dir / f"OT_partners_DepMap_Chronos_{args.release}_vs_24Q4.tsv",
                sep="\t",
                index=False,
            )

        run = RunManifest(
            script="scripts/03_fitness_concordance/depmap_partner_chronos.py",
            output_dir=output_dir,
            status="reconstruction" if args.release == "24Q4" else "robustness_only_not_published",
            description=f"lncRNA off-target partner Chronos lookup, DepMap {args.release}",
        )
        run.add_manifest_keys([RELEASES[args.release], "steps.step_02.outputs.gene_to_gene"])
        run.add_inputs({"gene_to_gene": gene_to_gene, "gene_effect": gene_effect, "models": models})
        run.add_parameters(
            {
                "release": args.release,
                "release_of_record": "24Q4",
                "chronos_essential": CHRONOS_ESSENTIAL,
                "chronos_strongly_sensitive": CHRONOS_SENSITIVE,
                "partner_source": "gene_to_gene_offtargets_excl_same_symbol.tsv, query_class == lncRNA",
            }
        )
        run.add_results(summary)
        print(f"wrote {run.write()}")

    print(json.dumps(summary, indent=2))
    if reports and not all(r["identical"] for r in reports):
        raise SystemExit("rebuild does NOT reproduce the frozen tables; see above")


if __name__ == "__main__":
    main()
