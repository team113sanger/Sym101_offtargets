#!/usr/bin/env python3
"""Filter a GENCODE transcript FASTA to set of transcript biotypes.
Usage::

    python scripts/02_transcriptome_homology/filter_fasta_by_biotype.py \
        --transcriptome-fasta inputs/gencode_v50/gencode.v50.transcripts.fa.gz \
        --output inputs/gencode_v50/gencode.v50.PC_lncRNA_retained_intron.fa.gz
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

from sym101_qc.transcriptome_mismatch_qc import (
    LIANG_OFFTARGET_BIOTYPES,
    open_text,
    parse_fasta_header,
)


def filter_by_biotype(source: Path, destination: Path, biotypes: frozenset[str]) -> tuple[int, int]:
    """Copy records whose biotype is in ``biotypes``; return (kept, seen).

    Written to a temporary file first, so an interrupted run cannot leave a truncated
    """
    tmp = destination.with_name(destination.name + ".tmp")
    n_seen = n_kept = 0
    keeping = False
    with open_text(source) as handle, gzip.open(tmp, "wt") as out:
        for line in handle:
            if line.startswith(">"):
                n_seen += 1
                keeping = parse_fasta_header(line)["biotype"] in biotypes
                n_kept += keeping
            if keeping:
                out.write(line)
    tmp.replace(destination)
    return n_kept, n_seen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--transcriptome-fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="gzipped FASTA to write")
    parser.add_argument(
        "--biotype",
        action="append",
        default=None,
        help=f"repeatable; defaults to {sorted(LIANG_OFFTARGET_BIOTYPES)}",
    )
    args = parser.parse_args()

    if not args.transcriptome_fasta.is_file():
        raise SystemExit(f"missing transcriptome FASTA: {args.transcriptome_fasta}")
    biotypes = frozenset(args.biotype) if args.biotype else LIANG_OFFTARGET_BIOTYPES

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_kept, n_seen = filter_by_biotype(args.transcriptome_fasta, args.output, biotypes)
    print(f"kept {n_kept}/{n_seen} transcripts in {sorted(biotypes)} -> {args.output}")


if __name__ == "__main__":
    main()
