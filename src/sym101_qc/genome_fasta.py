#!/usr/bin/env python3
"""Random access to a plain (uncompressed) genome FASTA, without samtools or pysam.

Usage:
    from sym101_qc.genome_fasta import GenomeFasta

    with GenomeFasta(fasta_path) as genome:
        genome.fetch("chr21", 43811563, 43811585)   # 1-based, inclusive
        genome.revcomp("ACGT")

Run as a script to build the index and print a region::
    python src/sym101_qc/genome_fasta.py <fasta> [chrom start end]

"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(sequence: str) -> str:
    """Reverse complement, preserving case and leaving N as N."""
    return sequence.translate(_COMPLEMENT)[::-1]


class GenomeFastaError(RuntimeError):
    """The FASTA is unusable: missing, empty, or not line-wrapped uniformly."""


class GenomeFasta:
    """An indexed, read-only FASTA. One file handle, held open for the object's life."""

    def __init__(self, fasta: str | Path, index_dir: str | Path | None = None) -> None:
        self.fasta = Path(fasta)
        if not self.fasta.is_file():
            raise GenomeFastaError(f"genome FASTA not found: {self.fasta}")
        index_dir = Path(index_dir) if index_dir else self.fasta.parent
        self.index_path = index_dir / f"{self.fasta.name}.fai"
        self.index = self._load_or_build_index()
        self._handle = self.fasta.open("rb")

    # -- index ------------------------------------------------------------------

    def _load_or_build_index(self) -> dict[str, dict[str, int]]:
        if self.index_path.is_file():
            return self._read_fai(self.index_path)
        index = self._build_index()
        with contextlib.suppress(OSError):
            self._write_fai(index, self.index_path)
        return index

    @staticmethod
    def _read_fai(path: Path) -> dict[str, dict[str, int]]:
        index: dict[str, dict[str, int]] = {}
        for line in path.read_text().splitlines():
            name, length, offset, linebases, linewidth = line.split("\t")
            index[name] = {
                "length": int(length),
                "offset": int(offset),
                "linebases": int(linebases),
                "linewidth": int(linewidth),
            }
        return index

    @staticmethod
    def _write_fai(index: dict[str, dict[str, int]], path: Path) -> None:
        lines = [
            f"{name}\t{m['length']}\t{m['offset']}\t{m['linebases']}\t{m['linewidth']}"
            for name, m in index.items()
        ]
        path.write_text("\n".join(lines) + "\n")

    def _build_index(self) -> dict[str, dict[str, int]]:
        """One sequential pass, recording byte offsets only - no sequence is held."""
        headers: list[tuple[int, str]] = []
        line_bases: int | None = None
        with self.fasta.open("rb") as handle:
            offset = 0
            for raw in handle:
                if raw.startswith(b">"):
                    headers.append((offset, raw.decode().rstrip("\n")))
                elif line_bases is None:
                    line_bases = len(raw.rstrip(b"\n"))
                offset += len(raw)
            total = offset

        if not headers or line_bases is None:
            raise GenomeFastaError(f"no FASTA records in {self.fasta}")

        index: dict[str, dict[str, int]] = {}
        for i, (header_offset, header) in enumerate(headers):
            name = header[1:].split()[0]
            seq_offset = header_offset + len(header) + 1
            seq_end = headers[i + 1][0] if i + 1 < len(headers) else total
            n_bytes = seq_end - seq_offset
            # Every line but possibly the last carries one newline; ceil-divide to count
            # them, then subtract, which gives the base count exactly for a uniformly
            # wrapped FASTA.
            n_lines = -(-n_bytes // (line_bases + 1))
            index[name] = {
                "length": n_bytes - n_lines,
                "offset": seq_offset,
                "linebases": line_bases,
                "linewidth": line_bases + 1,
            }
        return index

    # -- access -----------------------------------------------------------------

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Sequence for ``chrom:start-end``, 1-based and inclusive, upper-cased."""
        if chrom not in self.index:
            raise GenomeFastaError(f"{chrom!r} is not in {self.fasta.name}")
        meta = self.index[chrom]
        if start < 1 or end > meta["length"] or end < start:
            raise GenomeFastaError(f"{chrom}:{start}-{end} is outside {chrom} (length {meta['length']})")
        line_bases, line_width = meta["linebases"], meta["linewidth"]
        zero_based = start - 1
        byte_offset = meta["offset"] + (zero_based // line_bases) * line_width + (zero_based % line_bases)
        n_bases = end - start + 1
        self._handle.seek(byte_offset)
        # Read enough to cover the newlines that fall inside the span.
        raw = self._handle.read(n_bases + (n_bases // line_bases) + 2)
        return raw.replace(b"\n", b"")[:n_bases].decode().upper()

    @staticmethod
    def revcomp(sequence: str) -> str:
        return reverse_complement(sequence)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> GenomeFasta:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def main() -> None:
    if len(sys.argv) not in (2, 5):
        print(__doc__)
        raise SystemExit(2)
    with GenomeFasta(sys.argv[1]) as genome:
        print(f"{len(genome.index)} sequences, index at {genome.index_path}")
        if len(sys.argv) == 5:
            chrom, start, end = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
            print(genome.fetch(chrom, start, end))


if __name__ == "__main__":
    main()
