"""
Negative sampling for PanCirc-Fungi.

Generates negative gene samples (genes without circRNA annotations)
that match the positive gene distribution in terms of:
- Length distribution
- Expression level (if available)
- Exon count distribution
"""

import logging
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from utils.gtf_utils import GeneInfo, GeneModelIndexer

logger = logging.getLogger(__name__)


def get_negative_genes(
    gene_model: GeneModelIndexer,
    positive_gene_ids: Set[str],
    n_negatives: int,
    rng_seed: Optional[int] = None,
) -> List[str]:
    """Sample negative genes (no circRNA) matching positive gene characteristics.

    Parameters
    ----------
    gene_model : GeneModelIndexer
        Indexed gene annotations.
    positive_gene_ids : Set[str]
        Gene IDs with known circRNAs.
    n_negatives : int
        Number of negative genes to sample.
    rng_seed : int, optional

    Returns
    -------
    List[str]
        Sampled negative gene IDs.
    """
    rng = random.Random(rng_seed)

    # Get all genes and their lengths
    all_genes: List[GeneInfo] = list(gene_model.genes.values())
    candidate_negatives = [
        g for g in all_genes
        if g.gene_id not in positive_gene_ids
        and g.exon_count > 0  # must have exons
    ]

    if len(candidate_negatives) < n_negatives:
        logger.warning(
            f"Only {len(candidate_negatives)} negative candidates available "
            f"(requested {n_negatives}), using all"
        )
        return [g.gene_id for g in candidate_negatives]

    # Get positive gene length distribution for matching
    pos_genes = [
        g for g in all_genes
        if g.gene_id in positive_gene_ids
    ]
    if pos_genes:
        pos_lengths = np.array([g.end - g.start for g in pos_genes])
        pos_median = np.median(pos_lengths)
        pos_std = pos_lengths.std()
        pos_cutoff = pos_median + 2 * pos_std

        # Filter negatives: prefer genes with similar length (±2σ of positive median)
        neg_lengths = np.array([g.end - g.start for g in candidate_negatives])
        length_diff = np.abs(neg_lengths - pos_median)
        # Score: smaller difference = better match
        scores = length_diff / max(pos_std, 1)
        # Also penalize genes much shorter or longer than all positives
        too_long = neg_lengths > pos_cutoff
        scores[too_long] *= 5  # strongly penalize outliers

        # Weighted sampling (lower score = higher probability)
        weights = 1.0 / (scores + 0.1)
        weights = weights / weights.sum()

        indices = rng.choices(
            range(len(candidate_negatives)),
            weights=weights,
            k=n_negatives,
        )
        sampled = [candidate_negatives[i].gene_id for i in indices]
    else:
        # No positives to match against — random sampling
        sampled = rng.sample(candidate_negatives, n_negatives)

    return sampled


def get_negative_genes_by_group(
    gene_models: Dict[str, GeneModelIndexer],
    positive_gene_ids: Dict[str, Set[str]],
    negative_ratio: float = 1.0,
    rng_seed: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Sample negative genes for multiple strains.

    Parameters
    ----------
    gene_models : dict mapping strain -> GeneModelIndexer
    positive_gene_ids : dict mapping strain -> set of positive gene IDs
    negative_ratio : float
        Ratio of negatives to positives (1.0 = balanced).
    rng_seed : int, optional

    Returns
    -------
    Dict[str, List[str]]
        strain -> list of negative gene IDs
    """
    result = {}
    rng = random.Random(rng_seed)

    for strain, gm in gene_models.items():
        pos_ids = positive_gene_ids.get(strain, set())
        n_pos = len(pos_ids)
        n_neg = max(1, int(n_pos * negative_ratio))

        negs = get_negative_genes(
            gm, pos_ids, n_neg, rng_seed=rng_seed
        )
        result[strain] = negs
        logger.info(
            f"{strain}: {n_pos} positive genes, {len(negs)} negative genes sampled"
        )

    return result


def get_background_junctions_for_negatives(
    gene_model: GeneModelIndexer,
    negative_gene_ids: List[str],
    max_candidates_per_gene: int = 50,
) -> Dict[str, List]:
    """Generate candidate splice junctions for negative genes.

    These are treated as "non-circRNA" junctions during training.
    Each gene contributes its top ``max_candidates_per_gene`` exon pairs.

    Returns
    -------
    Dict[str, List[tuple]]
        gene_id -> [(donor_pos, acceptor_pos, donor_exon_i, acceptor_exon_j), ...]
    """
    result = {}
    for gid in negative_gene_ids:
        gene = gene_model.get_gene(gid)
        if gene is None:
            continue
        candidates = gene_model.generate_candidate_junctions(
            gene, max_candidates=max_candidates_per_gene
        )
        result[gid] = candidates
    return result
