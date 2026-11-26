#!/usr/bin/env python3
"""Re-score the Sym101 off-target alignments in RNA space, counting G·U wobbles as matches.

Bowtie aligns DNA so there is no concept of RNA base-pairing, so a G·U pair is charged
as a mismatch even though it is a stable pair Cas13d tolerates. The published `-n 2` search
therefore under-calls off-targets, and it under-calls precisely the ones most likely to be
cut: an NM=2 alignment whose two mismatches are both G·U is a perfect RNA-space match
that any exact only criterion misses entirely.

At an aligned column the crRNA base pairing with reference base t is complement(SEQ[i]). So:

    crRNA G · target U  ->  complement(SEQ[i]) == G  ->  SEQ[i] == C  and  ref == T
    crRNA U · target G  ->  complement(SEQ[i]) == T  ->  SEQ[i] == A  and  ref == G

A mismatch is a tolerated wobble iff (SEQ base, ref base) is (C, T) or (A, G). Everything
else stays a real mismatch. `n_eff = NM - n_wobble`.

Wessels et al. 2020 put the Cas13d seed at spacer nucleotides 15–21. Because SEQ
is reverse-complemented, spacer position = 24 − (1-based alignment column) for a 23-nt
spacer. Wobbles inside the seed are counted separately: wobbles are tolerated, but the
seed is where tolerance is lowest, so a seed wobble is the least safe kind to discount.

Usage::

    python scripts/02_transcriptome_homology/wobble_rescore_sym101.py --self-test
    python scripts/02_transcriptome_homology/wobble_rescore_sym101.py [--output-dir results/02_transcriptome_homology/wobble_rna_space]
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

from sym101_qc.manifest import get, load_manifest, resolve_path  # noqa: E402
from sym101_qc.run_manifest import RunManifest  # noqa: E402

# (SAM SEQ base, reference base) pairs that are a G·U wobble in RNA space.
WOBBLE = {("C", "T"), ("A", "G")}
SPACER_LEN = 23
SEED = (15, 21)  # 1-based spacer positions, Wessels et al. 2020

_MD_TOKEN = re.compile(r"(\d+)|(\^[A-Za-z]+)|([A-Za-z])")


def parse_md(md: str) -> list[tuple[int, str]]:
    """Reference bases at mismatched read positions, as (0-based read offset, ref base).

    MD:Z counts matched bases, names each mismatched reference base, and prefixes
    deletions with '^'. bowtie 1 emits no indels, so deletions are not expected; they are
    skipped rather than silently mis-offsetting the read position if one ever appears.
    """
    offset = 0
    out: list[tuple[int, str]] = []
    for run, deletion, mismatch in _MD_TOKEN.findall(md):
        if run:
            offset += int(run)
        elif deletion:
            continue
        elif mismatch:
            out.append((offset, mismatch.upper()))
            offset += 1
    return out


def spacer_position(read_offset: int) -> int:
    """1-based spacer position for a 0-based column of the reverse-complemented SEQ."""
    return SPACER_LEN - read_offset


def classify_alignment(seq: str, md: str) -> dict:
    """Split an alignment's mismatches into true mismatches and tolerated wobbles."""
    wobble = 0
    real = 0
    seed_wobble = 0
    for offset, ref_base in parse_md(md):
        if offset >= len(seq):
            continue
        pair = (seq[offset].upper(), ref_base)
        position = spacer_position(offset)
        if pair in WOBBLE:
            wobble += 1
            if SEED[0] <= position <= SEED[1]:
                seed_wobble += 1
        else:
            real += 1
    return {"n_wobble": wobble, "n_real_mismatch": real, "n_seed_wobble": seed_wobble}


def self_test() -> None:
    """Check the wobble rule on built examples before trusting it on 151,000 alignments."""
    seq = "ACGTACGTACGTACGTACGTACG"

    # Negative control: seq[2] is G, and (G, T) is not a wobble pair.
    assert seq[2] == "G"
    assert classify_alignment(seq, "2T20")["n_wobble"] == 0, "(G, T) must stay a real mismatch"
    checks = [
        # (seq, md, expected wobble, expected real)
        ("C" + seq[1:], "0T22", 1, 0),  # SEQ C vs ref T  -> wobble
        ("A" + seq[1:], "0G22", 1, 0),  # SEQ A vs ref G  -> wobble
        ("G" + seq[1:], "0A22", 0, 1),  # SEQ G vs ref A  -> real mismatch
        ("T" + seq[1:], "0C22", 0, 1),  # SEQ T vs ref C  -> real mismatch
        ("C" + seq[1:2] + "A" + seq[3:], "0T1G20", 2, 0),  # two wobbles -> perfect in RNA space
    ]
    for sequence, md, want_wobble, want_real in checks:
        got = classify_alignment(sequence, md)
        assert got["n_wobble"] == want_wobble, (sequence, md, got)
        assert got["n_real_mismatch"] == want_real, (sequence, md, got)

    # Seed arithmetic: read offset 0 is spacer position 23; offset 22 is position 1.
    assert spacer_position(0) == 23
    assert spacer_position(22) == 1
    # A wobble at read offset 5 is spacer position 18, the centre of the seed.
    assert SEED[0] <= spacer_position(5) <= SEED[1]
    assert classify_alignment(seq[:5] + "C" + seq[6:], "5T17")["n_seed_wobble"] == 1
    print("self-test passed: wobble rule and seed arithmetic")


def gene_of(reference_name: str) -> tuple[str, str]:
    """(gene id without version, gene symbol) from a GENCODE transcript FASTA header."""
    fields = reference_name.split("|")
    gene_id = fields[1].split(".")[0] if len(fields) > 1 else ""
    symbol = fields[5] if len(fields) > 5 else ""
    return gene_id, symbol


def rescore_sam(sam_path: Path) -> pd.DataFrame:
    """One row per (guide, off-target gene): best DNA-space and best RNA-space mismatch."""
    best: dict[tuple[str, str, str], dict] = {}
    opener = gzip.open if sam_path.suffix == ".gz" else open
    n_alignments = 0
    with opener(sam_path, "rt") as handle:
        for line in handle:
            if line.startswith("@"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 11 or fields[2] == "*":
                continue
            n_alignments += 1
            guide, flag, reference, seq = fields[0], int(fields[1]), fields[2], fields[9]
            if not flag & 16:
                continue  # --nofw - antisense only; a sense hit is not a Cas13d target
            md = next((f[5:] for f in fields[11:] if f.startswith("MD:Z:")), None)
            if md is None:
                continue
            scores = classify_alignment(seq, md)
            gene_id, symbol = gene_of(reference)
            key = (guide, gene_id, symbol)
            dna_mm = scores["n_wobble"] + scores["n_real_mismatch"]
            rna_mm = scores["n_real_mismatch"]
            current = best.get(key)
            if current is None or (rna_mm, dna_mm) < (current["rna_mismatches"], current["dna_mismatches"]):
                best[key] = {
                    "unique_guide_name": guide,
                    "offtarget_gene_id": gene_id,
                    "offtarget_gene_symbol": symbol,
                    "dna_mismatches": dna_mm,
                    "rna_mismatches": rna_mm,
                    "n_wobble": scores["n_wobble"],
                    "n_seed_wobble": scores["n_seed_wobble"],
                }
    print(f"{n_alignments} alignment records read from {sam_path.name}")
    return pd.DataFrame(best.values())


def main() -> None:
    manifest = load_manifest()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sam", type=Path, default=resolve_path(get(manifest, "steps.step_02.outputs.sam")))
    parser.add_argument(
        "--guide-table", type=Path, default=resolve_path(get(manifest, "steps.step_02.outputs.guide_level"))
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "02_transcriptome_homology" / "wobble_rna_space",
    )
    parser.add_argument("--self-test", action="store_true", help="check the rule and exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    self_test()  # never score 151,000 alignments on an unchecked rule
    for key, path in (("sam", args.sam), ("guide_table", args.guide_table)):
        if not path.is_file():
            raise SystemExit(f"missing input {key}: {path}")

    guides = pd.read_csv(args.guide_table, sep="\t", low_memory=False)
    intended_gene_id = dict(
        zip(
            guides["unique_guide_name"],
            guides["intended_gene_id"].astype(str).str.split(".").str[0],
            strict=False,
        )
    )
    intended_symbol = dict(zip(guides["unique_guide_name"], guides["Gene"], strict=False))
    query_class = dict(zip(guides["unique_guide_name"], guides["query_class"], strict=False))

    hits = rescore_sam(args.sam)

    # Self-exclusion exactly as published: the intended ENSG, and the same symbol under a
    # different ENSG (annotation duplicates).
    hits["intended_gene_id"] = hits["unique_guide_name"].map(intended_gene_id)
    hits["intended_symbol"] = hits["unique_guide_name"].map(intended_symbol)
    hits["query_class"] = hits["unique_guide_name"].map(query_class)
    is_self = (hits["offtarget_gene_id"] == hits["intended_gene_id"]) | (
        hits["offtarget_gene_symbol"] == hits["intended_symbol"]
    )
    offtargets = hits.loc[~is_self].copy()

    def gene_counts(frame: pd.DataFrame, column: str, limit: int) -> pd.Series:
        return frame.loc[frame[column] <= limit].groupby("unique_guide_name")["offtarget_gene_id"].nunique()

    per_guide = pd.DataFrame({"unique_guide_name": guides["unique_guide_name"]})
    per_guide["query_class"] = per_guide["unique_guide_name"].map(query_class)
    for label, column in (("dna", "dna_mismatches"), ("rna", "rna_mismatches")):
        for limit, suffix in ((0, "exact"), (2, "le2mm")):
            per_guide[f"n_{label}_{suffix}"] = (
                per_guide["unique_guide_name"]
                .map(gene_counts(offtargets, column, limit))
                .fillna(0)
                .astype(int)
            )
    per_guide["gained_offtarget_in_rna_space"] = (per_guide["n_rna_le2mm"] > per_guide["n_dna_le2mm"]) | (
        per_guide["n_rna_exact"] > per_guide["n_dna_exact"]
    )

    summary = {
        "n_offtarget_gene_pairs": int(len(offtargets)),
        "exact_offtarget_genes": {
            "dna_space": int((offtargets["dna_mismatches"] == 0).sum()),
            "rna_space": int((offtargets["rna_mismatches"] == 0).sum()),
        },
        "guides_with_any_offtarget_le2mm": {
            "dna_space": int((per_guide["n_dna_le2mm"] > 0).sum()),
            "rna_space": int((per_guide["n_rna_le2mm"] > 0).sum()),
        },
        "by_class": {},
        "n_alignments_with_a_wobble": int((offtargets["n_wobble"] > 0).sum()),
        "n_wobbles_inside_the_seed": int(offtargets["n_seed_wobble"].sum()),
        "n_guides_clean_in_dna_dirty_in_rna": int(
            ((per_guide["n_dna_le2mm"] == 0) & (per_guide["n_rna_le2mm"] > 0)).sum()
        ),
    }
    for klass, subset in per_guide.groupby("query_class"):
        summary["by_class"][klass] = {
            "n_guides": int(len(subset)),
            "with_offtarget_dna_space": int((subset["n_dna_le2mm"] > 0).sum()),
            "with_offtarget_rna_space": int((subset["n_rna_le2mm"] > 0).sum()),
            "exact_offtarget_dna_space": int((subset["n_dna_exact"] > 0).sum()),
            "exact_offtarget_rna_space": int((subset["n_rna_exact"] > 0).sum()),
            "clean_in_dna_dirty_in_rna": int(
                ((subset["n_dna_le2mm"] == 0) & (subset["n_rna_le2mm"] > 0)).sum()
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_guide_path = args.output_dir / "guide_offtarget_wobble_sym101.tsv"
    pairs_path = args.output_dir / "offtarget_gene_pairs_wobble_sym101.tsv.gz"
    per_guide.to_csv(per_guide_path, sep="\t", index=False)
    offtargets.to_csv(pairs_path, sep="\t", index=False, compression="gzip")

    run = RunManifest(
        script="scripts/02_transcriptome_homology/wobble_rescore_sym101.py",
        output_dir=args.output_dir,
        status="robustness_only_not_published",
        description="RNA-space re-score of the step 02 SAM, counting G.U wobbles as matches",
    )
    run.add_manifest_keys(["steps.step_02.outputs.sam", "steps.step_02.outputs.guide_level"])
    run.add_inputs({"sam": args.sam, "guide_table": args.guide_table})
    run.add_parameters(
        {
            "wobble_pairs_seq_ref": sorted(WOBBLE),
            "spacer_len": SPACER_LEN,
            "seed_spacer_positions": SEED,
            "self_exclusion": ["intended ENSG", "same symbol, different ENSG"],
            "rule_source": "scripts/sym201_ported/offtarget_wobble_rescore.py",
        }
    )
    run.add_outputs({"per_guide": per_guide_path, "gene_pairs": pairs_path})
    run.add_results(summary)
    run.note("Not published. The manuscript's counts are the DNA-space -n 2 counts.")
    manifest_path = run.write()

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {per_guide_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
