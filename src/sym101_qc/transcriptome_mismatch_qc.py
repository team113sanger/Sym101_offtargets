"""Retrospective Cas13 spacer vs GENCODE-transcriptome Hamming QC (Liang-style).
It counts non-self transcript hits at ≤2 mismatches using transcriptome FASTA
(same freeze as the overlap GTF: GENCODE v46).

Primary fail rule (hard-coded): ≥1 non-self ≤2 mm hit to a protein_coding or
lncRNA transcript (see LIANG_OFFTARGET_BIOTYPES). Hits to misc_RNA / other
biotypes are secondary only (fails_any_biotype_le2).

"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from sym101_qc.gtf_utils import (
    EXCLUDED_SCREEN_GENES,
    SCREEN_LNCRNA_BIOTYPES,
    parse_guide_transcript_coords,
    strip_ensembl_version,
)

DNA_COMP = str.maketrans({"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"})
TILES_23 = ((0, 8), (8, 8), (16, 7))

LIANG_OFFTARGET_BIOTYPES = frozenset(
    {
        "protein_coding",
        "protein_coding_CDS_not_defined",
        "lncRNA",
        "retained_intron",
    }
)


def rc_dna(seq: str) -> str:
    return seq.upper().translate(DNA_COMP)[::-1]


def hamming(a: bytes, b: bytes) -> int:
    return sum(x != y for x, y in zip(a, b, strict=True))


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open("rt")


def parse_fasta_header(header: str) -> dict:
    """Parse GENCODE transcript headers; fall back to the first token as transcript ID."""
    h = header[1:].strip().split()[0]
    parts = h.split("|")
    tx = parts[0]
    gene = parts[1] if len(parts) > 1 and parts[1].startswith("ENSG") else ""
    gene_name = parts[5] if len(parts) > 5 else ""
    biotype = parts[7] if len(parts) > 7 else ""
    return {
        "header": h,
        "transcript_id": tx,
        "transcript_id_base": strip_ensembl_version(tx),
        "gene_id": gene,
        "gene_id_base": strip_ensembl_version(gene) if gene else "",
        "gene_name": gene_name,
        "biotype": biotype.rstrip("|"),
    }


def inspect_fasta(path: Path, n_headers: int = 8) -> dict:
    """Decide whether a file is a transcript FASTA, genome FASTA, or not FASTA."""
    headers: list[str] = []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith(">"):
                headers.append(line[1:].strip())
                if len(headers) >= n_headers:
                    break
    if not headers:
        first = path.name.lower()
        kind = "names_or_empty"
        if first.endswith(".txt") or "names" in first:
            kind = "names_txt"
        return {"path": str(path), "kind": kind, "n_headers_sampled": 0, "examples": []}

    joined = " ".join(headers).lower()
    enst = sum(h.startswith("ENST") or "|ENST" in h or h.split("|")[0].startswith("ENST") for h in headers)
    chrom = sum(
        h.split()[0].lower().startswith("chr")
        or h.split()[0] in {str(i) for i in range(1, 23)} | {"X", "Y", "MT", "M"}
        for h in headers
    )
    if enst >= max(1, len(headers) // 2):
        kind = "transcript_fasta"
    elif chrom >= max(1, len(headers) // 2) or "homo sapiens" in joined:
        kind = "genome_fasta"
    else:
        kind = "unknown_fasta"
    return {
        "path": str(path),
        "kind": kind,
        "n_headers_sampled": len(headers),
        "examples": headers[:4],
        "n_enst_like": int(enst),
    }


def load_transcriptome(path: Path) -> tuple[list[bytes], pd.DataFrame]:
    seqs: list[bytes] = []
    meta_rows: list[dict] = []
    tx_idx = 0
    header = None
    chunks: list[str] = []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(chunks).upper().replace("U", "T")
                    seqs.append(seq.encode("ascii"))
                    rec = parse_fasta_header(header)
                    rec["tx_idx"] = tx_idx
                    rec["length"] = len(seq)
                    meta_rows.append(rec)
                    tx_idx += 1
                header = line
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            seq = "".join(chunks).upper().replace("U", "T")
            seqs.append(seq.encode("ascii"))
            rec = parse_fasta_header(header)
            rec["tx_idx"] = tx_idx
            rec["length"] = len(seq)
            meta_rows.append(rec)
    meta = pd.DataFrame(meta_rows)
    if meta.empty:
        raise ValueError(f"No FASTA records in {path}")
    return seqs, meta


def query_class(row: pd.Series) -> str:
    if row.get("transcript_biotype") == "NT_Control" or row.get("Gene") == "NT_Control":
        return "NT"
    if (
        row.get("transcript_biotype") in SCREEN_LNCRNA_BIOTYPES
        and row.get("Gene") not in EXCLUDED_SCREEN_GENES
    ):
        return "lncRNA"
    return "protein_coding"


def extract_site(seq: bytes, start_1based: int, end_1based: int) -> str | None:
    """1-based inclusive transcript coordinates -> DNA string, or None if OOB."""
    if start_1based < 1 or end_1based > len(seq) or end_1based < start_1based:
        return None
    return seq[start_1based - 1 : end_1based].decode("ascii")


def orientation_check(
    guides: pd.DataFrame,
    seqs: list[bytes],
    meta: pd.DataFrame,
    n_sample: int = 80,
) -> tuple[pd.DataFrame, str]:
    """Compare GuideSeq vs intended transcript slice; choose RNA-sense vs spacer RC."""
    by_tx = {row.transcript_id_base: int(row.tx_idx) for row in meta.itertuples(index=False)}
    rows = []
    candidates = guides.loc[guides["Transcript_ID"].notna() & (guides["transcript_biotype"] != "NT_Control")]
    for _, g in candidates.head(n_sample).iterrows():
        tx = strip_ensembl_version(str(g["Transcript_ID"]))
        idx = by_tx.get(tx)
        rec = {
            "unique_guide_name": g["unique_guide_name"],
            "Transcript_ID": g["Transcript_ID"],
            "GuideSeq": g["GuideSeq"],
            "intended_tx_in_fasta": idx is not None,
            "match_guide": False,
            "match_rc": False,
        }
        if idx is None:
            rows.append(rec)
            continue
        try:
            tx_start, tx_end = parse_guide_transcript_coords(str(g["GuideName"]))
        except ValueError:
            tx_end = int(g["MatchPos"])
            tx_start = tx_end - 22
        site = extract_site(seqs[idx], tx_start, tx_end)
        rec["fasta_site"] = site
        if site is not None:
            rec["match_guide"] = site == g["GuideSeq"]
            rec["match_rc"] = site == rc_dna(g["GuideSeq"])
        rows.append(rec)
    check = pd.DataFrame(rows)
    n_guide = int(check["match_guide"].sum())
    n_rc = int(check["match_rc"].sum())
    if n_guide >= n_rc and n_guide > 0:
        orientation = "guide_seq_is_rna_sense"
    elif n_rc > n_guide:
        orientation = "guide_seq_is_spacer_rc"
    else:
        orientation = "unresolved_try_both"
    return check, orientation


def query_sequence(guide_seq: str, orientation: str) -> list[str]:
    s = guide_seq.upper()
    if orientation == "guide_seq_is_rna_sense":
        return [s]
    if orientation == "guide_seq_is_spacer_rc":
        return [rc_dna(s)]
    return [s, rc_dna(s)]


def build_query_kmer_index(query_seqs: list[bytes]) -> dict[bytes, list[tuple[int, int]]]:
    """K-mers to look up while scanning the transcriptome."""
    wanted: set[bytes] = set()
    for seq in query_seqs:
        for offset, k in TILES_23:
            wanted.add(seq[offset : offset + k])
    return {k: [] for k in wanted}


def scan_transcriptome(
    seqs: list[bytes],
    buckets: dict[bytes, list[tuple[int, int]]],
) -> None:
    keys = buckets
    for tx_idx, seq in enumerate(seqs):
        n = len(seq)
        for _offset, k in ((0, 8), (0, 7)):  # generate all 8-mers and 7-mers
            last = n - k
            for pos in range(0, last + 1):
                mer = seq[pos : pos + k]
                hits = keys.get(mer)
                if hits is not None:
                    hits.append((tx_idx, pos))


def align_guides(
    guides: pd.DataFrame,
    seqs: list[bytes],
    meta: pd.DataFrame,
    orientation: str,
) -> pd.DataFrame:
    query_map: dict[str, bytes] = {}
    for _, g in guides.iterrows():
        for q in query_sequence(str(g["GuideSeq"]), orientation):
            query_map[f"{g['unique_guide_name']}|{q}"] = q.encode("ascii")

    buckets = build_query_kmer_index(list(query_map.values()))
    scan_transcriptome(seqs, buckets)

    meta_by_idx = meta.set_index("tx_idx")
    records: list[dict] = []
    seen: set[tuple[str, int, int]] = set()

    for _, g in guides.iterrows():
        gid = g["unique_guide_name"]
        intended_gene = (
            strip_ensembl_version(str(g["ensembl_gene_id"])) if pd.notna(g["ensembl_gene_id"]) else ""
        )
        intended_tx = strip_ensembl_version(str(g["Transcript_ID"])) if pd.notna(g["Transcript_ID"]) else ""
        for q in query_sequence(str(g["GuideSeq"]), orientation):
            q_b = q.encode("ascii")
            if len(q_b) != 23:
                continue
            candidates: list[tuple[int, int]] = []
            for offset, k in TILES_23:
                mer = q_b[offset : offset + k]
                for tx_idx, pos in buckets.get(mer, ()):
                    start = pos - offset
                    if start < 0:
                        continue
                    candidates.append((tx_idx, start))
            for tx_idx, start in candidates:
                key = (gid, tx_idx, start)
                if key in seen:
                    continue
                seq = seqs[tx_idx]
                if start + 23 > len(seq):
                    continue
                window = seq[start : start + 23]
                mm = hamming(q_b, window)
                if mm > 2:
                    continue
                seen.add(key)
                info = meta_by_idx.loc[tx_idx]
                ot_gene = str(info["gene_id_base"] or "")
                is_self = bool(intended_gene) and ot_gene == intended_gene
                records.append(
                    {
                        "unique_guide_name": gid,
                        "query_class": query_class(g),
                        "Gene": g["Gene"],
                        "intended_gene_id": intended_gene,
                        "intended_transcript_id": intended_tx,
                        "query_seq": q,
                        "ot_transcript_id": info["transcript_id_base"],
                        "ot_gene_id": ot_gene,
                        "ot_gene_name": info["gene_name"],
                        "ot_biotype": info["biotype"],
                        "ot_start0": start,
                        "mismatches": mm,
                        "is_self_gene": is_self,
                    }
                )
    return pd.DataFrame.from_records(records)


def summarise(
    guides: pd.DataFrame,
    hits: pd.DataFrame,
    orientation: str,
    fasta_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    guides = guides.copy()
    guides["query_class"] = guides.apply(query_class, axis=1)
    empty = hits.empty
    if empty:
        hits = pd.DataFrame(
            columns=[
                "unique_guide_name",
                "is_self_gene",
                "mismatches",
                "ot_gene_id",
                "ot_gene_name",
                "ot_biotype",
                "ot_transcript_id",
            ]
        )

    nonself_all = hits.loc[~hits["is_self_gene"]] if not empty else hits
    # Primary Liang fail: non-self hit to PC or lncRNA only
    nonself_liang = (
        nonself_all.loc[nonself_all["ot_biotype"].isin(LIANG_OFFTARGET_BIOTYPES)]
        if not empty and "ot_biotype" in nonself_all.columns
        else nonself_all
    )
    fail_ids_liang = set(nonself_liang["unique_guide_name"]) if not empty else set()
    fail_ids_any = set(nonself_all["unique_guide_name"]) if not empty else set()

    def ot_table(gids: pd.Series, pool: pd.DataFrame, prefix: str) -> pd.Series:
        sub = pool.loc[pool["unique_guide_name"].isin(gids)]
        keys = [
            f"{prefix}ot_genes",
            f"{prefix}n_ot_genes",
            f"{prefix}min_mm",
            f"{prefix}ot_biotypes",
        ]
        if sub.empty:
            return pd.Series(["", 0, pd.NA, ""], index=keys)
        genes = ";".join(sorted({x for x in sub["ot_gene_name"].fillna(sub["ot_gene_id"]) if x}))
        biot = ";".join(sorted({x for x in sub["ot_biotype"].fillna("") if x}))
        return pd.Series(
            [genes, int(sub["ot_gene_id"].nunique()), int(sub["mismatches"].min()), biot],
            index=keys,
        )

    rows = []
    for _, g in guides.iterrows():
        gid = g["unique_guide_name"]
        gh = hits.loc[hits["unique_guide_name"] == gid] if not empty else hits
        n_self0 = int(((gh["is_self_gene"]) & (gh["mismatches"] == 0)).sum()) if not gh.empty else 0
        n_nonself_any = int((~gh["is_self_gene"]).sum()) if not gh.empty else 0
        n_nonself_liang = (
            int(((~gh["is_self_gene"]) & gh["ot_biotype"].isin(LIANG_OFFTARGET_BIOTYPES)).sum())
            if not gh.empty
            else 0
        )
        min_mm_self = (
            int(gh.loc[gh["is_self_gene"], "mismatches"].min())
            if n_self0 or (not gh.empty and gh["is_self_gene"].any())
            else pd.NA
        )
        ot_liang = ot_table(pd.Series([gid]), nonself_liang, prefix="")
        ot_any = ot_table(pd.Series([gid]), nonself_all, prefix="any_biotype_")
        rows.append(
            {
                "unique_guide_name": gid,
                "query_class": g["query_class"],
                "Gene": g["Gene"],
                "Transcript_ID": g["Transcript_ID"],
                "ensembl_gene_id": g["ensembl_gene_id"],
                "transcript_biotype": g["transcript_biotype"],
                "GuideSeq": g["GuideSeq"],
                "n_self_perfect": n_self0,
                "has_intended_0mm": n_self0 > 0,
                # Primary: Liang PC/lncRNA rule
                "n_nonself_hits_le2": n_nonself_liang,
                "fails_liang_qc": bool(gid in fail_ids_liang) if g["query_class"] != "NT" else pd.NA,
                "min_self_mm": min_mm_self,
                "min_nonself_mm": ot_liang["min_mm"],
                "nonself_ot_genes": ot_liang["ot_genes"],
                "n_nonself_ot_genes": ot_liang["n_ot_genes"],
                "nonself_ot_biotypes": ot_liang["ot_biotypes"],
                # Secondary: any GENCODE biotype (stricter; not Liang wording)
                "n_nonself_hits_le2_any_biotype": n_nonself_any,
                "fails_any_biotype_le2": bool(gid in fail_ids_any) if g["query_class"] != "NT" else pd.NA,
                "any_biotype_ot_genes": ot_any["any_biotype_ot_genes"],
                "any_biotype_ot_biotypes": ot_any["any_biotype_ot_biotypes"],
            }
        )
    summary = pd.DataFrame(rows)

    def count_fail(cls: str) -> dict:
        sub = summary.loc[summary["query_class"] == cls]
        n = len(sub)
        if cls == "NT":
            return {
                "n": n,
                "n_with_PC_or_lncRNA_le2_hit": int((sub["n_nonself_hits_le2"] > 0).sum()),
                "n_with_any_biotype_le2_hit": int((sub["n_nonself_hits_le2_any_biotype"] > 0).sum()),
            }
        n_fail = int(sub["fails_liang_qc"].fillna(False).sum())
        n_fail_any = int(sub["fails_any_biotype_le2"].fillna(False).sum())
        n_intended = int(sub["has_intended_0mm"].sum())
        return {
            "n": n,
            "n_fail_PC_or_lncRNA_le2": n_fail,
            "frac_fail_PC_or_lncRNA": round(n_fail / n, 4) if n else None,
            "n_fail_any_biotype_le2": n_fail_any,
            "frac_fail_any_biotype": round(n_fail_any / n, 4) if n else None,
            "n_intended_0mm": n_intended,
            "frac_intended_0mm": round(n_intended / n, 4) if n else None,
        }

    counts = {
        "note": (
            "Retrospective QC. Primary fail rule = Liang: any non-self ≤2-mismatch hit "
            "to a protein_coding or lncRNA transcript (retained_intron included). "
            "Do not describe as design-time filtering."
        ),
        "liang_offtarget_biotypes": sorted(LIANG_OFFTARGET_BIOTYPES),
        "transcriptome_fasta": str(fasta_path),
        "orientation": orientation,
        "n_guides_total": int(len(summary)),
        "lncRNA": count_fail("lncRNA"),
        "protein_coding": count_fail("protein_coding"),
        "NT": count_fail("NT"),
    }

    lnc_fail = summary.loc[
        (summary["query_class"] == "lncRNA") & (summary["fails_liang_qc"] == True)  # noqa: E712
    ][
        [
            "unique_guide_name",
            "Gene",
            "Transcript_ID",
            "n_nonself_hits_le2",
            "min_nonself_mm",
            "nonself_ot_genes",
            "n_nonself_ot_genes",
            "nonself_ot_biotypes",
        ]
    ].sort_values(["n_nonself_ot_genes", "Gene"], ascending=[False, True])

    return summary, lnc_fail, counts


def run_qc(
    supp_table: Path,
    transcriptome_fasta: Path,
    output_dir: Path,
) -> dict:
    inspect = inspect_fasta(transcriptome_fasta)
    if inspect["kind"] != "transcript_fasta":
        raise ValueError(
            "Need a GENCODE transcript FASTA (headers starting with ENST), not a genome "
            f"FASTA or names.txt. Saw kind={inspect['kind']} in {transcriptome_fasta}. "
            f"Examples: {inspect.get('examples')}"
        )

    guides = pd.read_csv(supp_table)
    guides["GuideSeq"] = guides["GuideSeq"].astype(str).str.upper()
    seqs, meta = load_transcriptome(transcriptome_fasta)
    check, orientation = orientation_check(guides, seqs, meta)
    hits = align_guides(guides, seqs, meta, orientation)
    summary, lnc_fail, counts = summarise(guides, hits, orientation, transcriptome_fasta)
    counts["fasta_inspect"] = inspect
    counts["n_transcripts_in_fasta"] = int(len(meta))

    output_dir.mkdir(parents=True, exist_ok=True)
    check.to_csv(output_dir / "orientation_check.tsv", sep="\t", index=False)
    summary.to_csv(output_dir / "guide_mismatch_summary.tsv", sep="\t", index=False)
    lnc_fail.to_csv(output_dir / "lncrna_nonself_offtarget_genes.tsv", sep="\t", index=False)
    if not hits.empty:
        hits.to_csv(output_dir / "all_hits_le2.tsv.gz", sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "all_hits_le2.tsv.gz", sep="\t", index=False)
    (output_dir / "qc_summary.json").write_text(json.dumps(counts, indent=2) + "\n")
    return counts
