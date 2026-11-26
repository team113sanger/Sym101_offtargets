#!/usr/bin/env python3
"""CLI for retrospective ≤2-mismatch transcriptome QC of Symphony Cas13d guides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sym101_qc.transcriptome_mismatch_qc import inspect_fasta, run_qc

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FASTA_CANDIDATES = (
    REPO_ROOT / "inputs" / "gencode_v46" / "gencode.v46.transcripts.fa.gz",
    REPO_ROOT / "inputs" / "gencode_v46" / "gencode.v46.transcripts.fa",
    REPO_ROOT / "inputs" / "gencode_v46" / "gencode.v46.transcripts.fa.gz",
    REPO_ROOT / "inputs" / "gencode_v46" / "gencode.v46.transcripts.fa",
)


def find_transcript_fasta(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return explicit

    search_dirs = [
        REPO_ROOT / "inputs",
        REPO_ROOT / "inputs" / "gencode_v46",
    ]
    candidates = list(DEFAULT_FASTA_CANDIDATES)
    for folder in search_dirs:
        if folder.is_dir():
            candidates.extend(sorted(folder.glob("*.fa")))
            candidates.extend(sorted(folder.glob("*.fasta")))
            candidates.extend(sorted(folder.glob("*.fa.gz")))
            candidates.extend(sorted(folder.glob("*.fasta.gz")))

    seen: set[Path] = set()
    transcript_hits: list[tuple[Path, dict]] = []
    other: list[tuple[Path, dict]] = []
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        info = inspect_fasta(path)
        if info["kind"] == "transcript_fasta":
            transcript_hits.append((path, info))
        else:
            other.append((path, info))

    if len(transcript_hits) == 1:
        return transcript_hits[0][0]
    if len(transcript_hits) > 1:

        def score(item: tuple[Path, dict]) -> tuple[int, str]:
            name = item[0].name.lower()
            pref = 0
            if "transcripts" in name and "gencode" in name:
                pref += 3
            if "pc_transcripts" in name or "lncrna_transcripts" in name:
                pref -= 1
            return (pref, name)

        transcript_hits.sort(key=score, reverse=True)
        return transcript_hits[0][0]

    msg = [
        "No GENCODE transcript FASTA found (headers should start with ENST).",
        "Copy the guide-design transcriptome FASTA into:",
        f"  {REPO_ROOT / 'inputs' / 'gencode_v46'}/",
        "Use the sequence FASTA (gencode.v46.transcripts.fa / .fa.gz), not names.txt and not a genome FASTA.",
        "Then rerun with --transcriptome-fasta PATH",
    ]
    if other:
        msg.append("Files seen but not usable as a transcript FASTA:")
        for path, info in other:
            msg.append(f"  {path}  kind={info['kind']}  examples={info.get('examples', [])[:2]}")
    raise FileNotFoundError("\n".join(msg))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supp-table", type=Path, default=REPO_ROOT / "inputs" / "Supp_table_1_.csv")
    parser.add_argument(
        "--transcriptome-fasta",
        type=Path,
        default=None,
        help="GENCODE v46 transcript FASTA from the guide-design folder (not genome, not names.txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "02_transcriptome_homology" / "transcriptome_mismatch_qc",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Classify FASTA files in inputs/ and exit (no alignment).",
    )
    args = parser.parse_args()

    if args.inspect_only:
        folders = [REPO_ROOT / "inputs", REPO_ROOT / "inputs" / "gencode_v46"]
        reports = []
        for folder in folders:
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in {".fa", ".fasta", ".gz", ".txt", ".fna"}:
                    reports.append(inspect_fasta(path))
        print(json.dumps(reports, indent=2))
        return

    fasta = find_transcript_fasta(args.transcriptome_fasta)
    print(f"Using transcriptome FASTA: {fasta}", flush=True)
    counts = run_qc(args.supp_table, fasta, args.output_dir)
    print(json.dumps(counts, indent=2))
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
