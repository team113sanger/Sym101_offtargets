#!/usr/bin/env python3
"""Summarise Bowtie transcriptome off-targets for Sym101 Cas13d guides.

Reproduces the Guo / cas13_webtool homology search:

  bowtie --nofw -a -n 2 -f INDEX guides.fa -S alignments.sam

then counts unique *other genes* at 0 / ≤1 / ≤2 mismatches (mapped0e:mapped1e:mapped2e),
while also keeping site- and transcript-level tables that the original webtool collapsed.

Sequence-homology search, not a cleavage-activity model.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd

SEED_START = 15  # 1-based spacer coordinates from Wessels et al. 2020
SEED_END = 21

MD_TOKEN = re.compile(r"(\d+)|(\^[A-Z]+)|([A-Z])")
NM_RE = re.compile(r"NM:i:(\d+)")
MD_RE = re.compile(r"MD:Z:([0-9A-Z\^]+)")


def strip_version(value: str) -> str:
    return str(value).split(".")[0] if value and str(value) != "nan" else ""


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open("rt")


def parse_gencode_header(header: str) -> dict:
    name = header[1:].split()[0]
    parts = name.split("|")
    transcript_id = strip_version(parts[0]) if parts else ""
    gene_id = strip_version(parts[1]) if len(parts) > 1 else ""
    gene_name = parts[5] if len(parts) > 5 else ""
    biotype = parts[7].rstrip("|") if len(parts) > 7 else ""
    return {
        "ref_name": name,
        "ot_transcript_id": transcript_id,
        "ot_gene_id": gene_id,
        "ot_gene_name": gene_name,
        "ot_biotype": biotype,
    }


def load_reference_map(fasta: Path) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    with open_text(fasta) as handle:
        for line in handle:
            if line.startswith(">"):
                rec = parse_gencode_header(line.rstrip("\n"))
                mapping[rec["ref_name"]] = rec
                mapping[rec["ot_transcript_id"]] = rec
    return mapping


def parse_md_query_positions(md: str, query_len: int, reverse: bool) -> list[int]:
    """Return 1-based mismatch positions on the original query/spacer."""
    if not md:
        return []
    ref_consumed = 0
    positions: list[int] = []
    for span, deletion, substitution in MD_TOKEN.findall(md):
        if span:
            ref_consumed += int(span)
        elif deletion:
            continue
        elif substitution:
            ref_consumed += 1
            aligned_pos = ref_consumed
            qpos = query_len - aligned_pos + 1 if reverse else aligned_pos
            if 1 <= qpos <= query_len:
                positions.append(qpos)
    return positions


def parse_nm(tags: str) -> int | None:
    match = NM_RE.search(tags)
    return int(match.group(1)) if match else None


def parse_md(tags: str) -> str:
    match = MD_RE.search(tags)
    return match.group(1) if match else ""


def classify_closest(min_mm: int | float | None) -> str:
    if min_mm is None or pd.isna(min_mm):
        return "none"
    min_mm = int(min_mm)
    if min_mm == 0:
        return "exact"
    if min_mm == 1:
        return "one_mismatch"
    if min_mm == 2:
        return "two_mismatch"
    return "none"


def load_guides(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["unique_guide_name"] = df["unique_guide_name"].astype(str)
    df["GuideSeq"] = df["GuideSeq"].astype(str).str.upper()
    df["intended_gene_id"] = df["intended_gene_id"].map(strip_version)
    df["intended_transcript_id"] = df["intended_transcript_id"].map(strip_version)
    return df


def iter_sam_hits(sam: Path, max_mm: int = 2):
    with open_text(sam) as handle:
        for line in handle:
            if not line or line.startswith("@"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 11:
                continue
            flag = int(fields[1])
            if flag & 4:
                continue
            tags = "\t".join(fields[11:])
            nm = parse_nm(tags)
            if nm is None or nm > max_mm:
                continue
            cigar = fields[5]
            seq = fields[9]
            reverse = bool(flag & 16)
            md = parse_md(tags)
            qlen = len(seq) if seq not in {".", "*"} else 23
            mm_pos = parse_md_query_positions(md, qlen, reverse)
            yield {
                "unique_guide_name": fields[0],
                "flag": flag,
                "reverse_strand": reverse,
                "ref_name": fields[2],
                "ref_pos": int(fields[3]),
                "cigar": cigar,
                "mismatches": nm,
                "mismatch_positions": ",".join(str(x) for x in mm_pos),
                "n_seed_mismatches": sum(SEED_START <= p <= SEED_END for p in mm_pos),
                "seed_mismatch": any(SEED_START <= p <= SEED_END for p in mm_pos),
            }


def summarise(guides: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = dict(hits.groupby("unique_guide_name", sort=False)) if not hits.empty else {}
    for rec in guides.itertuples(index=False):
        gid = rec.unique_guide_name
        gh = grouped.get(gid)
        if gh is None or gh.empty:
            rows.append(
                {
                    "unique_guide_name": gid,
                    "query_class": rec.query_class,
                    "Gene": rec.Gene,
                    "intended_gene_id": rec.intended_gene_id,
                    "intended_transcript_id": rec.intended_transcript_id,
                    "GuideSeq": rec.GuideSeq,
                    "n_self_hits_le2": 0,
                    "n_nonself_sites_le2": 0,
                    "n_nonself_transcripts_le2": 0,
                    "n_nonself_genes_mm0": 0,
                    "n_nonself_genes_le1": 0,
                    "n_nonself_genes_le2": 0,
                    "mapped0e:mapped1e:mapped2e": "0:0:0",
                    "fails_liang_qc": False,
                    "closest_offtarget_class": "none",
                    "min_nonself_mm": pd.NA,
                    "n_exact_offtarget_genes": 0,
                    "n_one_mismatch_offtarget_genes": 0,
                    "n_two_mismatch_offtarget_genes": 0,
                    "n_nonself_sites_with_seed_mm": 0,
                    "nonself_ot_genes": "",
                    "nonself_ot_gene_ids": "",
                    "nonself_ot_transcripts": "",
                    "nonself_ot_biotypes": "",
                }
            )
            continue

        self_mask = gh["is_self_gene"]
        nonself = gh.loc[~self_mask]
        gene_min = (
            nonself.groupby(["ot_gene_id", "ot_gene_name"], dropna=False)["mismatches"].min()
            if not nonself.empty
            else pd.Series(dtype="int64")
        )
        n0 = int((gene_min == 0).sum())
        n1 = int((gene_min <= 1).sum())
        n2 = int((gene_min <= 2).sum())
        n_exact = n0
        n_one = int((gene_min == 1).sum())
        n_two = int((gene_min == 2).sum())
        min_mm = int(gene_min.min()) if len(gene_min) else None
        gene_names = sorted({str(name or gid2) for gid2, name in gene_min.index if gid2 or name})
        gene_ids = sorted({str(gid2) for gid2, _ in gene_min.index if gid2})
        tx = sorted(set(nonself["ot_transcript_id"].dropna().astype(str)))
        biot = sorted(set(nonself["ot_biotype"].dropna().astype(str)))
        rows.append(
            {
                "unique_guide_name": gid,
                "query_class": rec.query_class,
                "Gene": rec.Gene,
                "intended_gene_id": rec.intended_gene_id,
                "intended_transcript_id": rec.intended_transcript_id,
                "GuideSeq": rec.GuideSeq,
                "n_self_hits_le2": int(self_mask.sum()),
                "n_nonself_sites_le2": int(len(nonself)),
                "n_nonself_transcripts_le2": int(nonself["ot_transcript_id"].nunique())
                if not nonself.empty
                else 0,
                "n_nonself_genes_mm0": n0,
                "n_nonself_genes_le1": n1,
                "n_nonself_genes_le2": n2,
                "mapped0e:mapped1e:mapped2e": f"{n0}:{n1}:{n2}",
                "fails_liang_qc": n2 > 0,
                "closest_offtarget_class": classify_closest(min_mm),
                "min_nonself_mm": min_mm if min_mm is not None else pd.NA,
                "n_exact_offtarget_genes": n_exact,
                "n_one_mismatch_offtarget_genes": n_one,
                "n_two_mismatch_offtarget_genes": n_two,
                "n_nonself_sites_with_seed_mm": int(nonself["seed_mismatch"].fillna(False).sum())
                if not nonself.empty
                else 0,
                "nonself_ot_genes": ";".join(gene_names),
                "nonself_ot_gene_ids": ";".join(gene_ids),
                "nonself_ot_transcripts": ";".join(tx),
                "nonself_ot_biotypes": ";".join(biot),
            }
        )
    return pd.DataFrame(rows)


def count_fail_rows(summary: pd.DataFrame) -> list[dict]:
    rows = []
    for cls in ["lncRNA", "protein_coding", "NT"]:
        sub = summary[summary["query_class"] == cls]
        n = len(sub)
        fail = sub[sub["fails_liang_qc"]] if n else sub
        rows.append(
            {
                "query_class": cls,
                "n_guides": n,
                "liang_fail": int(fail["fails_liang_qc"].sum()) if n else 0,
                "frac_fail": round(float(fail["fails_liang_qc"].mean()), 4) if n else None,
                "fail_exact": int((fail["closest_offtarget_class"] == "exact").sum()) if n else 0,
                "fail_one_mismatch": int((fail["closest_offtarget_class"] == "one_mismatch").sum())
                if n
                else 0,
                "fail_two_mismatch": int((fail["closest_offtarget_class"] == "two_mismatch").sum())
                if n
                else 0,
                "pass": n - int(fail["fails_liang_qc"].sum()) if n else 0,
            }
        )
    return rows


def write_markdown(path: Path, counts: list[dict], meta: dict) -> None:
    lines = [
        "# Sym101 GENCODE v50 Bowtie off-target QC",
        "",
        "Primary fail rule: any non-self alignment with ≤2 mismatches to a protein-coding or lncRNA transcript.",
        "Counts are unique *genes*, matching Guo et al. 2021 `mapped0e:mapped1e:mapped2e`.",
        "",
        f"- Bowtie: `{meta.get('bowtie_command', '')}`",
        f"- Reference: `{meta.get('reference_fasta', '')}`",
        f"- Spacers: `{meta.get('spacer_fasta', '')}`",
        "",
        "| Class | n | Liang fail | Exact | 1-mismatch | 2-mismatch | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in counts:
        lines.append(
            "| {query_class} | {n_guides} | {liang_fail} | {fail_exact} | {fail_one_mismatch} | {fail_two_mismatch} | {pass} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Closest class is the *best* non-self hit for that guide (0 before 1 before 2).",
            "Exact and one-mismatch hits deserve highest scrutiny (Wessels et al. 2020; TIGER, Wessels & Stirn et al. 2023).",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sam", type=Path, required=True)
    p.add_argument("--guides", type=Path, required=True, help="sym101_guides.tsv")
    p.add_argument("--reference-fasta", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bowtie-command", default="")
    p.add_argument("--join-hamming", type=Path, default=None, help="optional guide_mismatch_summary.tsv")
    p.add_argument(
        "--join-sanjana", type=Path, default=None, help="optional full_joined TSV with Sanjana mapped fields"
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    guides = load_guides(args.guides)
    ref_map = load_reference_map(args.reference_fasta)
    guide_idx = guides.set_index("unique_guide_name")

    hit_rows = []
    for hit in iter_sam_hits(args.sam):
        info = ref_map.get(hit["ref_name"]) or parse_gencode_header(">" + hit["ref_name"])
        intended = ""
        if hit["unique_guide_name"] in guide_idx.index:
            intended = str(guide_idx.loc[hit["unique_guide_name"], "intended_gene_id"] or "")
            if isinstance(intended, pd.Series):
                intended = str(intended.iloc[0] or "")
        ot_gene = info.get("ot_gene_id") or ""
        hit.update(info)
        hit["is_self_gene"] = bool(intended) and ot_gene == intended
        hit_rows.append(hit)

    hits = pd.DataFrame(hit_rows)
    if hits.empty:
        hits = pd.DataFrame(
            columns=[
                "unique_guide_name",
                "ref_name",
                "ot_transcript_id",
                "ot_gene_id",
                "ot_gene_name",
                "ot_biotype",
                "mismatches",
                "is_self_gene",
                "seed_mismatch",
                "mismatch_positions",
                "n_seed_mismatches",
                "ref_pos",
                "cigar",
                "reverse_strand",
            ]
        )
    else:
        hits = hits.merge(
            guides[["unique_guide_name", "query_class", "Gene", "GuideSeq"]],
            on="unique_guide_name",
            how="left",
        )

    summary = summarise(guides, hits)
    counts = count_fail_rows(summary)
    meta = {
        "note": (
            "Retrospective Bowtie homology QC against GENCODE v50 PC + lncRNA transcripts. "
            "Primary fail = any non-self gene with ≤2 mismatches. "
            "mapped0e:mapped1e:mapped2e are cumulative unique genes, not sites."
        ),
        "seed_region": f"{SEED_START}-{SEED_END}",
        "bowtie_command": args.bowtie_command,
        "reference_fasta": str(args.reference_fasta),
        "spacer_fasta": str(args.guides),
        "n_guides": int(len(summary)),
        "counts": counts,
    }

    hits_path = args.output_dir / "all_hits_le2.tsv.gz"
    nonself = hits.loc[~hits["is_self_gene"]] if not hits.empty else hits
    summary_path = args.output_dir / "sym101_guides_bowtie_v50.tsv"
    counts_path = args.output_dir / "SUMMARY_bowtie_v50_counts.tsv"
    md_path = args.output_dir / "SUMMARY_bowtie_v50.md"
    json_path = args.output_dir / "qc_summary.json"

    if args.join_hamming and args.join_hamming.exists():
        ham = pd.read_csv(args.join_hamming, sep="\t")
        keep = [
            c
            for c in [
                "unique_guide_name",
                "fails_liang_qc",
                "n_nonself_hits_le2",
                "min_nonself_mm",
                "nonself_ot_genes",
                "n_nonself_ot_genes",
            ]
            if c in ham.columns
        ]
        ham = ham[keep].rename(columns={c: f"hamming_{c}" for c in keep if c != "unique_guide_name"})
        summary = summary.merge(ham, on="unique_guide_name", how="left")
    if args.join_sanjana and args.join_sanjana.exists():
        san = pd.read_csv(args.join_sanjana, sep="\t")
        keep = [
            c
            for c in [
                "unique_guide_name",
                "in_sanjana_resource",
                "sanjana_resource",
                "sanjana_mapped0e_1e_2e",
                "sanjana_any_gt0",
                "sanjana_weighted",
                "sanjana_weighted_gt1",
            ]
            if c in san.columns
        ]
        summary = summary.merge(san[keep], on="unique_guide_name", how="left")

    hits.to_csv(hits_path, sep="\t", index=False)
    nonself.to_csv(args.output_dir / "nonself_hits_le2.tsv.gz", sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    pd.DataFrame(counts).to_csv(counts_path, sep="\t", index=False)
    write_markdown(md_path, counts, meta)
    json_path.write_text(json.dumps(meta, indent=2))

    fail = summary[summary["fails_liang_qc"]]
    fail.to_csv(args.output_dir / "liang_fail_guides.tsv", sep="\t", index=False)
    print(pd.DataFrame(counts).to_string(index=False))
    print(f"Wrote {summary_path}")
    print(f"Wrote {hits_path}")


if __name__ == "__main__":
    main()
