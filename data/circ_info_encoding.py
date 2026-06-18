"""
CircInfo encoding for PanCirc-Fungi.

Functions to:
- Parse CircInfo CSV files
- Filter to exon/intron types only (discard Unknown/antisense/intergenic)
- Map categorical variables to integer IDs
- Build positive-sample gene annotation
"""

import logging
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Only these circ_types are considered (must have gene_id for gene-level prediction)
VALID_CIRC_TYPES = {"exon", "intron"}

# CircType => integer mapping
CIRC_TYPE_MAP = {
    "exon": 0,
    "intron": 1,
}

# Strand => integer map
STRAND_MAP = {
    "+": 0,
    "-": 1,
    ".": 2,
}


def load_circ_info(path: str) -> pd.DataFrame:
    """Load a CircInfo CSV file.

    Expected columns: circ_id, strand, circ_type, gene_id, gene_name, gene_type
    """
    if not os.path.isfile(path):
        logger.warning(f"CircInfo file not found: {path}")
        return pd.DataFrame(columns=[
            "circ_id", "strand", "circ_type", "gene_id", "gene_name", "gene_type"
        ])
    df = pd.read_csv(path)
    # Standardize column names
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def filter_circ_info(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only exon/intron circRNAs (have gene_id).

    Removes antisense, intergenic, Unknown types.
    Returns a copy with reset index.
    """
    if df.empty:
        return df
    valid = df[df["circ_type"].str.lower().isin(VALID_CIRC_TYPES)].copy()
    # Also drop rows without gene_id
    valid = valid[valid["gene_id"].notna() & (valid["gene_id"] != "")].copy()
    valid.reset_index(drop=True, inplace=True)
    return valid


def get_positive_genes(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Extract the set of genes that have circRNA annotations.

    Returns dict mapping gene_id -> circRNA rows for that gene.
    """
    positive_genes: Dict[str, pd.DataFrame] = {}
    for gene_id, group in df.groupby("gene_id"):
        if gene_id and str(gene_id).strip():
            positive_genes[str(gene_id)] = group
    return positive_genes


def extract_circ_features(df_row: pd.Series) -> np.ndarray:
    """Extract feature vector from a single CircInfo row.

    Returns array of shape (13,):
        [circ_type_0, circ_type_1,     # one-hot of binary circ_type
         strand_0, strand_1, strand_2,  # one-hot of strand (3)
         gene_type_code]                 # gene type integer
        + 8 reserved positions
    """
    # Circ type (one-hot, only exon=0 or intron=1)
    circ_type = str(df_row.get("circ_type", "")).lower()
    ct_vec = np.zeros(2, dtype=np.float32)
    if circ_type in CIRC_TYPE_MAP:
        ct_vec[CIRC_TYPE_MAP[circ_type]] = 1.0

    # Strand (one-hot over 3 values)
    strand = str(df_row.get("strand", "."))
    st_vec = np.zeros(3, dtype=np.float32)
    if strand in STRAND_MAP:
        st_vec[STRAND_MAP[strand]] = 1.0

    # Gene type (categorical)
    gene_type = str(df_row.get("gene_type", "")).lower()
    GENE_TYPE_MAP = {
        "protein_coding": 0,
        "snonna": 1,
        "snorna": 1,
        "trna": 2,
        "rrna": 3,
        "ncrna": 4,
        "mirna": 5,
        "pseudogene": 6,
        "lncrna": 7,
        "other": 8,
    }
    gt_code = GENE_TYPE_MAP.get(gene_type, 8)

    feats = np.concatenate([ct_vec, st_vec, [float(gt_code)]])
    # Pad to 13
    padded = np.zeros(13, dtype=np.float32)
    padded[: len(feats)] = feats
    return padded


def encode_circ_info_full(df: pd.DataFrame) -> np.ndarray:
    """Encode an entire CircInfo DataFrame to a feature matrix.

    Returns (n_rows, 13) array.
    """
    if df.empty:
        return np.zeros((0, 13), dtype=np.float32)
    rows = []
    for _, row in df.iterrows():
        rows.append(extract_circ_features(row))
    return np.stack(rows)


def parse_circ_id(circ_id: str) -> Optional[Dict[str, any]]:
    """Parse a circular RNA ID into components.

    Typical format: ``chr:start|end``
    Examples:
        Ca22chr1A_C_albicans_SC5314:1083580|1083674
        1:1000|2000
    """
    if not circ_id or not isinstance(circ_id, str):
        return None
    try:
        chrom_rest = circ_id.split(":")
        if len(chrom_rest) != 2:
            return None
        chrom = chrom_rest[0]
        coords = chrom_rest[1].split("|")
        if len(coords) != 2:
            return None
        return {
            "chrom": chrom,
            "start": int(coords[0]),
            "end": int(coords[1]),
        }
    except (ValueError, IndexError):
        return None
