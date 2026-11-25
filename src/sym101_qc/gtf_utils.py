"""Shared GTF parsing and coordinate utilities for lncRNA overlap analysis."""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

GTF_COLUMNS = [
    "chrom",
    "source",
    "feature",
    "start",
    "end",
    "score",
    "strand",
    "frame",
    "attributes",
]

# Features used by prepare_beds / guide_specificity (skip UTR, start_codon, etc.)
DEFAULT_GTF_FEATURES = frozenset({"gene", "transcript", "exon", "CDS"})
# gene / exon / CDS rows filtered to these biotypes when gene_types is set
GENE_TYPE_FILTER_FEATURES = frozenset({"gene", "exon", "CDS"})

_ATTR_KEYS = (
    "gene_id",
    "transcript_id",
    "gene_name",
    "gene_type",
    "transcript_type",
    "exon_number",
    "exon_id",
)
_ATTR_RE = re.compile(r'(\w+)\s+"([^"]+)"')


def strip_ensembl_version(feature_id: str) -> str:
    """Remove Ensembl version suffix (e.g. ENST00000400385.2 -> ENST00000400385)."""
    return feature_id.split(".")[0]


def parse_gtf_attributes(attr_string: str) -> dict[str, str]:
    """Parse a GTF attribute field into a dictionary."""
    return dict(_ATTR_RE.findall(attr_string))


def read_gtf(
    path: str | Path,
    features: set[str] | None = None,
    gene_types: set[str] | None = None,
) -> pd.DataFrame:
    """
      Load a GTF (or gzipped GTF) into a DataFrame with parsed attribute columns.

      Streams line-by-line (suitable for comprehensive Gencode). By default keeps
      gene / transcript / exon / CDS rows only. When gene_types is set (e.g.
    {"protein_coding"}), gene/exon/CDS rows are filtered to those biotypes;
      transcript rows are kept unfiltered so screen lncRNA IDs can be matched.
    """
    path = Path(path)
    features = set(features or DEFAULT_GTF_FEATURES)
    gene_types = set(gene_types) if gene_types else None

    open_fn = gzip.open if path.suffix == ".gz" else open
    rows: list[dict] = []

    with open_fn(path, "rt") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            feature = parts[2]
            if feature not in features:
                continue

            attrs = parse_gtf_attributes(parts[8])
            if (
                gene_types
                and feature in GENE_TYPE_FILTER_FEATURES
                and attrs.get("gene_type") not in gene_types
            ):
                continue

            rows.append(
                {
                    "chrom": parts[0],
                    "source": parts[1],
                    "feature": feature,
                    "start": int(parts[3]),
                    "end": int(parts[4]),
                    "score": parts[5],
                    "strand": parts[6],
                    "frame": parts[7],
                    "attributes": parts[8],
                    **{key: attrs.get(key) for key in _ATTR_KEYS},
                }
            )

    df = pd.DataFrame(rows)
    for col in ("gene_id", "transcript_id", "exon_id"):
        if col in df.columns:
            df[f"{col}_base"] = df[col].map(lambda x: strip_ensembl_version(x) if isinstance(x, str) else x)
    return df


SCREEN_LNCRNA_BIOTYPES = frozenset({"lncRNA", "retained_intron"})
EXCLUDED_SCREEN_GENES = frozenset({"TUG1"})


def is_screen_lncrna_row(
    df: pd.DataFrame,
    biotype_col: str = "transcript_biotype",
    gene_col: str = "Gene",
) -> pd.Series:
    """Boolean mask for Sym101 lncRNA-query rows in Supp_table_1."""
    return df[biotype_col].isin(SCREEN_LNCRNA_BIOTYPES) & ~df[gene_col].isin(EXCLUDED_SCREEN_GENES)


def load_screen_lncrna_transcripts(
    supp_table_path: str | Path,
    biotype_col: str = "transcript_biotype",
    transcript_col: str = "Transcript_ID",
) -> pd.DataFrame:
    """Return unique lncRNA transcripts from the Symphony screen supplementary table."""
    df = pd.read_csv(supp_table_path)
    lnc = df.loc[is_screen_lncrna_row(df, biotype_col)].copy()
    lnc["transcript_id_base"] = lnc[transcript_col].map(strip_ensembl_version)
    return (
        lnc.groupby("transcript_id_base", as_index=False)
        .agg(
            Transcript_ID=(transcript_col, "first"),
            ensembl_gene_id=("ensembl_gene_id", "first"),
            Gene=("Gene", "first"),
            transcript_biotype=(biotype_col, "first"),
            n_guides=("unique_guide_name", "nunique"),
        )
        .sort_values("Gene")
    )


def filter_gtf_transcripts(
    gtf: pd.DataFrame,
    transcript_ids: Iterable[str],
    feature: str = "transcript",
) -> pd.DataFrame:
    """Subset GTF rows for screen transcript IDs (version-agnostic matching)."""
    wanted = {strip_ensembl_version(t) for t in transcript_ids}
    return gtf.loc[(gtf["feature"] == feature) & (gtf["transcript_id_base"].isin(wanted))].copy()


def transcripts_to_bed(
    gtf_transcripts: pd.DataFrame,
    name_col: str = "transcript_id_base",
) -> pd.DataFrame:
    """Convert transcript GTF rows to BED6 (+ optional extra columns)."""
    bed = gtf_transcripts.assign(
        chrom=gtf_transcripts["chrom"],
        chromStart=gtf_transcripts["start"] - 1,  # GTF is 1-based; BED is 0-based
        chromEnd=gtf_transcripts["end"],
        name=gtf_transcripts[name_col],
        score=0,
        strand=gtf_transcripts["strand"],
    )[
        [
            "chrom",
            "chromStart",
            "chromEnd",
            "name",
            "score",
            "strand",
            "gene_name",
            "gene_id",
            "gene_type",
            "transcript_id_base",
        ]
    ]
    return bed.sort_values(["chrom", "chromStart"])


def genes_to_bed(
    gtf: pd.DataFrame,
    gene_type: str,
    feature: str = "gene",
) -> pd.DataFrame:
    """Extract gene-body BED intervals for a given Gencode gene_type."""
    genes = gtf.loc[(gtf["feature"] == feature) & (gtf["gene_type"] == gene_type)].copy()
    bed = genes.assign(
        chromStart=genes["start"] - 1,
        chromEnd=genes["end"],
        name=genes["gene_name"].fillna(genes["gene_id"]),
        score=0,
    )[
        [
            "chrom",
            "chromStart",
            "chromEnd",
            "name",
            "score",
            "strand",
            "gene_id",
            "gene_type",
        ]
    ]
    return bed.sort_values(["chrom", "chromStart"])


def exons_to_bed(
    gtf: pd.DataFrame,
    gene_type: str,
    feature: str = "exon",
) -> pd.DataFrame:
    """Extract exon BED intervals for transcripts of a given gene_type."""
    exons = gtf.loc[(gtf["feature"] == feature) & (gtf["gene_type"] == gene_type)].copy()
    bed = exons.assign(
        chromStart=exons["start"] - 1,
        chromEnd=exons["end"],
        name=exons["transcript_id"].fillna(exons["gene_id"]),
        score=0,
    )[
        [
            "chrom",
            "chromStart",
            "chromEnd",
            "name",
            "score",
            "strand",
            "gene_name",
            "gene_id",
            "transcript_id",
            "exon_number",
            "exon_id",
        ]
    ]
    return bed.sort_values(["chrom", "chromStart"])


def write_bed(bed: pd.DataFrame, path: str | Path, bed_cols: int = 6) -> Path:
    """Write BED file (first bed_cols columns only, for bedtools compatibility)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bed.iloc[:, :bed_cols].to_csv(path, sep="\t", header=False, index=False)
    return path


def write_bed_with_extra(bed: pd.DataFrame, path: str | Path) -> Path:
    """Write full BED-like table (extra columns kept for downstream parsing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bed.to_csv(path, sep="\t", header=False, index=False)
    return path


def parse_guide_transcript_coords(guide_name: str) -> tuple[int, int]:
    """
    Parse transcript coordinates from GuideName.

    Examples:
        crRNA0983:983-1005 -> (983, 1005)
        crRNA1279 -> raises if no interval present; use MatchPos as fallback upstream
    """
    if ":" in guide_name:
        interval = guide_name.split(":")[-1]
        if "-" in interval:
            start, end = interval.split("-", maxsplit=1)
            return int(start), int(end)
    raise ValueError(f"Could not parse transcript interval from GuideName: {guide_name}")


def transcript_to_genomic(
    transcript_id: str,
    tx_start: int,
    tx_end: int,
    gtf: pd.DataFrame,
) -> dict | None:
    """
    Map transcript coordinates to genomic coordinates via exon structure.

    Returns None for guides spanning splice junctions (multiple exons).
    """
    tx_base = strip_ensembl_version(transcript_id)
    exons = gtf.loc[(gtf["feature"] == "exon") & (gtf["transcript_id_base"] == tx_base)].copy()
    if exons.empty:
        return None

    # Sort exons in transcript order; fall back to genomic position if exon_number missing
    exon_num = pd.to_numeric(exons["exon_number"], errors="coerce")
    if exon_num.notna().all():
        exons = exons.assign(_exon_order=exon_num).sort_values("_exon_order")
    else:
        exons = exons.sort_values("start")
        if exons.iloc[0]["strand"] == "-":
            exons = exons.sort_values("start", ascending=False)
    exons["length"] = exons["end"] - exons["start"] + 1
    # Exon k begins at (sum of lengths 1..k-1) + 1 in 1-based transcript space.
    cumulative = exons["length"].cumsum()
    exons["relative_start"] = [1] + list(cumulative[:-1] + 1)
    exons["relative_end"] = cumulative

    hit = ((tx_start >= exons["relative_start"]) & (tx_start <= exons["relative_end"])) | (
        (tx_end >= exons["relative_start"]) & (tx_end <= exons["relative_end"])
    )
    if hit.sum() == 0:
        return None
    if hit.sum() > 1:
        return {
            "spans_junction": True,
            "genomic_start": None,
            "genomic_end": None,
            "exon_id": ";".join(exons.loc[hit, "exon_id"].astype(str)),
        }

    exon = exons.loc[hit].iloc[0]
    e_start = max(0, tx_start - exon["relative_start"])
    e_end = min(exon["length"] - 1, tx_end - exon["relative_start"])
    # Transcript coords run 5'→3' along the transcript: add to exon.start on +,
    # subtract from exon.end on −.
    if exon["strand"] == "-":
        genomic_start = int(exon["end"] - e_end)
        genomic_end = int(exon["end"] - e_start)
    else:
        genomic_start = int(exon["start"] + e_start)
        genomic_end = int(exon["start"] + e_end)
    return {
        "spans_junction": False,
        "genomic_start": genomic_start,
        "genomic_end": genomic_end,
        "chrom": exon["chrom"],
        "strand": exon["strand"],
        "exon_id": exon["exon_id"],
        "exon_number": exon["exon_number"],
    }
