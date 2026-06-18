"""
Expression data encoding for PanCirc-Fungi.

Handles:
- Reading CircExp (BSJ counts) and GeneExp (gene counts) CSV files
- log1p transformation and z-score normalization
- Alignment of circRNA to host gene expression pairs
- Handling variable replicate counts across strains
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_expression_csv(path: str) -> Optional[pd.DataFrame]:
    """Load an expression CSV file.

    Expected format: first column is ID (circ_id or gene_id),
    remaining columns are replicate expression values.

    Returns None if file is missing or empty.
    """
    if not os.path.isfile(path):
        logger.warning(f"Expression file not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        # Check if file has data beyond header
        if df.shape[0] == 0 or df.shape[1] < 2:
            logger.warning(f"Expression file empty: {path}")
            return None
        return df
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return None


def log1p_transform(df: pd.DataFrame, skip_col: int = 0) -> pd.DataFrame:
    """Apply log1p transformation to numeric columns.

    ``skip_col``: index of the ID column to skip (default 0 = first column).
    Returns a copy.
    """
    result = df.copy()
    for col in result.columns[skip_col:]:
        result[col] = np.log1p(result[col].astype(np.float32))
    return result


def zscore_normalize(df: pd.DataFrame, skip_col: int = 0) -> pd.DataFrame:
    """Z-score normalize numeric columns (skip the ID column).

    Normalization is per-column (per-replicate).
    Returns a copy.
    """
    result = df.copy()
    for col in result.columns[skip_col:]:
        vals = result[col].astype(np.float32)
        mean = vals.mean()
        std = vals.std()
        if std > 0:
            result[col] = (vals - mean) / std
        else:
            result[col] = vals - mean
    return result


def get_replicate_count(path: str) -> int:
    """Return number of replicate columns in an expression CSV."""
    df = load_expression_csv(path)
    if df is None:
        return 0
    return df.shape[1] - 1  # subtract ID column


def encode_expression_values(
    df: pd.DataFrame,
    log1p: bool = True,
    zscore: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert an expression DataFrame to normalized feature arrays.

    Returns
    -------
    values : np.ndarray (n_rows, n_replicates)
        Normalized expression values.
    id_col : np.ndarray (n_rows,)
        The ID column (gene_id or circ_id) as strings.
    replicate_names : List[str]
        Column names of replicates.
    """
    if df is None or df.empty:
        return np.array([]), np.array([]), []

    id_col = df.iloc[:, 0].astype(str).values
    replicate_names = list(df.columns[1:])

    # Extract numeric values
    values = df.iloc[:, 1:].astype(np.float32).values

    if log1p:
        values = np.log1p(values)
    if zscore:
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        values = (values - mean) / std

    return values, id_col, replicate_names


def align_circ_to_gene_expression(
    circ_info_df: pd.DataFrame,
    circ_exp_df: Optional[pd.DataFrame],
    gene_exp_df: Optional[pd.DataFrame],
) -> Dict[str, Dict]:
    """Align circRNA expression to their host gene expression.

    Returns dict::
        {
            gene_id: {
                "circ_ids": [circ_id_1, ...],
                "circ_exp": np.ndarray (n_circs, n_rep),
                "gene_exp": np.ndarray (n_rep,),
            }
        }
    """
    result: Dict[str, Dict] = {}

    if circ_info_df.empty:
        return result

    # Load circ expression
    circ_vals, circ_ids, _ = encode_expression_values(
        circ_exp_df) if circ_exp_df is not None else (None, [], [])

    # Load gene expression
    gene_vals, gene_ids, _ = encode_expression_values(
        gene_exp_df) if gene_exp_df is not None else (None, [], [])

    # Build gene_id -> expression lookup
    gene_exp_map: Dict[str, np.ndarray] = {}
    if gene_vals is not None and len(gene_vals) > 0:
        for i, gid in enumerate(gene_ids):
            gene_exp_map[gid] = gene_vals[i]

    # Build circ_id -> expression lookup
    circ_exp_map: Dict[str, np.ndarray] = {}
    if circ_vals is not None and len(circ_vals) > 0:
        for i, cid in enumerate(circ_ids):
            circ_exp_map[cid] = circ_vals[i]

    # For each positive gene (with circRNA info), align
    for gene_id, group in circ_info_df.groupby("gene_id"):
        gene_id = str(gene_id).strip()
        if not gene_id:
            continue

        circ_ids_list = group["circ_id"].tolist()
        circ_exp_list = [
            circ_exp_map.get(cid, np.zeros(circ_exp_map.get(list(circ_exp_map.keys())[0], np.zeros(1)).shape))
            for cid in circ_ids_list
        ] if circ_exp_map else []

        gene_exp = gene_exp_map.get(gene_id, None)

        result[gene_id] = {
            "circ_ids": circ_ids_list,
            "circ_exp": np.stack(circ_exp_list) if circ_exp_list else np.array([]),
            "gene_exp": gene_exp,
        }

    return result


def pad_to_max_replicates(
    exp_values: np.ndarray,
    max_replicates: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Pad or truncate expression values to a fixed replicate count.

    If exp_values is 1D (single gene), it's treated as (n_rep,).
    If 2D (n_samples, n_rep), each row is padded/truncated.
    """
    if exp_values.ndim == 0 or exp_values.size == 0:
        return np.full(max_replicates, pad_value, dtype=np.float32)

    if exp_values.ndim == 1:
        current_n = len(exp_values)
        if current_n >= max_replicates:
            return exp_values[:max_replicates]
        else:
            return np.pad(exp_values, (0, max_replicates - current_n),
                          constant_values=pad_value)

    # 2D
    if exp_values.shape[1] >= max_replicates:
        return exp_values[:, :max_replicates]
    else:
        pad_width = ((0, 0), (0, max_replicates - exp_values.shape[1]))
        return np.pad(exp_values, pad_width, constant_values=pad_value)
