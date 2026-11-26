#!/usr/bin/env python3
"""Join Liang Hamming QC with Sym101 splice-junction guide labels.

Uses post-hoc genome-alignment splice flags from guide_splice_junctions.csv

Writes per-class counts for junctional vs non-junctional guides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_SJ = Path(__file__).resolve().parents[2] / "inputs" / "guide_splice_junctions.csv"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True, help="guide_mismatch_summary.tsv")
    p.add_argument("--splice-junctions", type=Path, default=DEFAULT_SJ)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    summary = pd.read_csv(args.summary, sep="\t")
    sj = pd.read_csv(args.splice_junctions)
    # Prefer a spliced flag if any alignment for that guide is spliced.
    prim = (
        sj.sort_values(["qname", "is_spliced"], ascending=[True, False])
        .drop_duplicates("qname")[["qname", "is_spliced", "splice_junction"]]
        .rename(columns={"qname": "unique_guide_name"})
    )
    m = summary.merge(prim, on="unique_guide_name", how="left")
    m["guide_site_class"] = m["is_spliced"].map({True: "splice_junction", False: "non_splice"})
    m.loc[m["is_spliced"].isna(), "guide_site_class"] = "unknown"

    rows = []
    for cls in ["lncRNA", "protein_coding", "NT"]:
        for site in ["splice_junction", "non_splice", "unknown"]:
            sub = m[(m["query_class"] == cls) & (m["guide_site_class"] == site)]
            n = len(sub)
            if cls == "NT":
                n_fail = int((sub["n_nonself_hits_le2"] > 0).sum()) if n else 0
            else:
                n_fail = int(sub["fails_liang_qc"].fillna(False).sum()) if n else 0
            rows.append(
                {
                    "query_class": cls,
                    "guide_site_class": site,
                    "n_guides": n,
                    "liang_PC_lncRNA_fail": n_fail,
                    "frac_fail": round(n_fail / n, 4) if n else None,
                }
            )
    counts = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_joined = args.output_dir / "guide_mismatch_summary_with_splice_flags.tsv"
    out_counts = args.output_dir / "SUMMARY_splice_vs_nonsplice.tsv"
    m.to_csv(out_joined, sep="\t", index=False)
    counts.to_csv(out_counts, sep="\t", index=False)

    meta = {
        "note": (
            "guide_site_class comes from genome alignment splice CIGAR "
            "(Exon-Exon / Exon-Intron), not from the transcript FASTA itself. "
            "Transcript FASTA is always spliced mature RNA (exons concatenated)."
        ),
        "splice_source": str(args.splice_junctions),
        "summary_source": str(args.summary),
    }
    (args.output_dir / "SUMMARY_splice_vs_nonsplice.json").write_text(
        json.dumps({"meta": meta, "counts": rows}, indent=2)
    )
    print(counts.to_string(index=False))
    print(f"Wrote {out_joined}")
    print(f"Wrote {out_counts}")


if __name__ == "__main__":
    main()
