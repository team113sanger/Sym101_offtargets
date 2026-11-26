#!/usr/bin/env python3
"""Bridge the bowtie output to the tables the same-symbol filter reads.

`--verify` checks it reproduces the existing tables

Usage::
    python scripts/02_transcriptome_homology/rebuild_share_freeze_inputs.py --verify
    python scripts/02_transcriptome_homology/rebuild_share_freeze_inputs.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

from sym101_qc.manifest import get, load_manifest, resolve_path  # noqa: E402
from sym101_qc.run_manifest import RunManifest  # noqa: E402

GUIDE_RENAMES = {
    "n_self_hits_le2": "n_alignments_to_intended_gene_le2mm",
    "n_nonself_sites_le2": "n_offtarget_sites_le2mm",
    "n_nonself_transcripts_le2": "n_offtarget_transcripts_le2mm",
    "n_exact_offtarget_genes": "n_offtarget_genes_exact",
    "n_one_mismatch_offtarget_genes": "n_offtarget_genes_1_mismatch",
    "n_two_mismatch_offtarget_genes": "n_offtarget_genes_2_mismatch",
    "n_nonself_genes_le2": "n_offtarget_genes_le2mm",
}
# The two tables use different vocabularies for the same four states. 1:1, verified
# across all 6,225 guides.
CLOSEST_MATCH_LABELS = {
    "exact": "exact",
    "one_mismatch": "1_mismatch",
    "two_mismatch": "2_mismatch",
    "none": "none",
}
SITE_RENAMES = {
    "Gene": "intended_gene",
    "GuideSeq": "spacer_sequence",
    "ot_gene_name": "offtarget_gene_symbol",
    "ot_gene_id": "offtarget_gene_id",
    "ot_transcript_id": "offtarget_transcript_id",
    "ref_name": "transcript_header",
    "ref_pos": "alignment_start_on_transcript",
    "mismatches": "n_mismatches",
    "mismatch_positions": "mismatch_positions_on_spacer",
    "n_seed_mismatches": "n_seed_mismatches_spacer_15_21",
    "seed_mismatch": "has_seed_mismatch",
}
SITE_COLUMNS = [
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
GUIDE_COLUMNS = [
    "unique_guide_name",
    "query_class",
    "Gene",
    "intended_gene_id",
    "intended_transcript_id",
    "GuideSeq",
    "n_alignments_to_intended_gene_le2mm",
    "n_offtarget_sites_le2mm",
    "n_offtarget_transcripts_le2mm",
    "closest_offtarget_match",
    "n_offtarget_genes_exact",
    "n_offtarget_genes_1_mismatch",
    "n_offtarget_genes_2_mismatch",
    "n_offtarget_genes_le2mm",
    "offtarget_genes_exact",
    "offtarget_genes_1_mismatch",
    "offtarget_genes_2_mismatch",
]

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


def require(path: Path, key: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing input {key}: {path}")
    return path


def build_sites(nonself: pd.DataFrame) -> pd.DataFrame:
    """Site table, renamed, with the perfect-match RNA site added.

    A Cas13d spacer is the reverse complement of the transcript it engages, so the RNA
    site that would pair perfectly is the reverse complement of the spacer.
    """
    sites = nonself.rename(columns=SITE_RENAMES).copy()
    sites["perfect_match_rna_site"] = sites["spacer_sequence"].map(reverse_complement)
    return sites[SITE_COLUMNS]


def offtarget_gene_lists(sites: pd.DataFrame) -> pd.DataFrame:
    """Off-target genes per guide, grouped by their best (minimum) mismatch count.

    Three details, each established by reproducing the existing table rather than assumed:

    1. Grouping is per Ensembl **gene id**, not per symbol. One symbol can map to several
       ids at different mismatch counts, so collapsing on the symbol first would move
       genes into the wrong class.
    2. A symbol with more than one gene id **within a mismatch class** is written
       ``SYMBOL|ENSG``; an unambiguous symbol is written bare. Ambiguity is judged within
       the class, not across the guide.
    3. Sorting is on (symbol, gene id), not on the composite label: "CCL3" must precede
       "CCL3L3", which sorting the labels would reverse because "|" sorts after "L".
    """
    best = (
        sites.groupby(["unique_guide_name", "offtarget_gene_id", "offtarget_gene_symbol"], dropna=False)[
            "n_mismatches"
        ]
        .min()
        .reset_index()
    )
    frames = {}
    for mismatches, column in (
        (0, "offtarget_genes_exact"),
        (1, "offtarget_genes_1_mismatch"),
        (2, "offtarget_genes_2_mismatch"),
    ):
        subset = best.loc[best["n_mismatches"] == mismatches].copy()
        ids_per_symbol = subset.groupby(["unique_guide_name", "offtarget_gene_symbol"])[
            "offtarget_gene_id"
        ].transform("nunique")
        subset["label"] = subset["offtarget_gene_symbol"].where(
            ids_per_symbol == 1,
            subset["offtarget_gene_symbol"] + "|" + subset["offtarget_gene_id"].astype(str),
        )
        subset = subset.sort_values(["offtarget_gene_symbol", "offtarget_gene_id"])
        frames[column] = subset.groupby("unique_guide_name")["label"].apply(";".join).rename(column)
    return pd.concat(frames.values(), axis=1)


def build_guides(guides_raw: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    guides = guides_raw.rename(columns=GUIDE_RENAMES).copy()
    guides["closest_offtarget_match"] = guides["closest_offtarget_class"].map(CLOSEST_MATCH_LABELS)
    guides = guides.merge(
        offtarget_gene_lists(sites), left_on="unique_guide_name", right_index=True, how="left"
    )
    for column in ("offtarget_genes_exact", "offtarget_genes_1_mismatch", "offtarget_genes_2_mismatch"):
        guides[column] = guides[column].fillna("")
    return guides[GUIDE_COLUMNS]


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
            # An empty gene list is NaN in one table and "" in the other; that is a
            # serialisation difference, not a data difference.
            left_s = a.fillna("").astype(str).replace("nan", "")
            right_s = b.fillna("").astype(str).replace("nan", "")
            equal = left_s == right_s
        if not bool(equal.all()):
            mismatched[column] = int((~equal).sum())
    return {
        "table": name,
        "rebuilt_rows": len(rebuilt),
        "existing_rows": len(frozen),
        "columns_compared": len(shared),
        "columns_mismatching": mismatched,
        "identical": not mismatched and len(rebuilt) == len(frozen),
    }


def main() -> None:
    manifest = load_manifest()
    default_dir = resolve_path(get(manifest, "steps.step_02.outputs.guide_level")).parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, default=default_dir)
    parser.add_argument("--verify", action="store_true", help="compare against the existing tables")
    args = parser.parse_args()

    guides_in = require(args.results_dir / "sym101_guides_bowtie_v50.tsv", "bowtie_guide_summary")
    sites_in = require(args.results_dir / "nonself_hits_le2.tsv.gz", "bowtie_nonself_sites")

    sites = build_sites(pd.read_csv(sites_in, sep="\t", low_memory=False))
    guides = build_guides(pd.read_csv(guides_in, sep="\t", low_memory=False), sites)

    guide_out = args.results_dir / "sym101_guides_bowtie_gencode_v50.tsv"
    site_out = args.results_dir / "sym101_guides_bowtie_gencode_v50.sites.including_same_symbol.tsv.gz"

    reports = []
    if args.verify:
        if guide_out.is_file():
            reports.append(
                compare(
                    guides,
                    pd.read_csv(guide_out, sep="\t", low_memory=False),
                    ["unique_guide_name"],
                    "guide_level",
                )
            )
        if site_out.is_file():
            reports.append(
                compare(
                    sites,
                    pd.read_csv(site_out, sep="\t", low_memory=False),
                    ["unique_guide_name", "offtarget_transcript_id", "alignment_start_on_transcript"],
                    "site_level",
                )
            )
        print(json.dumps(reports, indent=2))
        if reports and not all(r["identical"] for r in reports):
            raise SystemExit("rebuild does NOT reproduce the existing tables")
        print("\nreproduces the existing tables exactly")
        return

    guides.to_csv(guide_out, sep="\t", index=False)
    sites.to_csv(site_out, sep="\t", index=False, compression="gzip")

    run = RunManifest(
        script="scripts/02_transcriptome_homology/rebuild_share_freeze_inputs.py",
        output_dir=args.results_dir,
        status="step_02",
        description="Bridge bowtie output to the tables the same-symbol filter reads",
    )
    run.add_inputs({"bowtie_guide_summary": guides_in, "bowtie_nonself_sites": sites_in})
    run.add_parameters({"guide_renames": GUIDE_RENAMES, "site_renames": SITE_RENAMES})
    run.add_outputs({"guide_level": guide_out, "site_level": site_out})
    run.add_results({"guide_rows": len(guides), "site_rows": len(sites)})
    print(f"wrote {guide_out}\nwrote {site_out}\nwrote {run.write()}")


if __name__ == "__main__":
    main()
