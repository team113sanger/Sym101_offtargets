"""Shared library for the Symphony Cas13d guide-specificity QC.
Importable as a package:
    from sym101_qc.manifest import load_manifest, resolve_path
    from sym101_qc.gtf_utils import read_gtf
Installed editable into the pixi environment; see pyproject.toml.
"""

__all__ = [
    "genome_fasta",
    "gtf_utils",
    "guide_specificity",
    "manifest",
    "overlap_analysis",
    "run_manifest",
    "transcriptome_mismatch_qc",
]
