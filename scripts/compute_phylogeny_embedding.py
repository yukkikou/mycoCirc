#!/usr/bin/env python3
"""
Compute phylogenetic PCA embedding from ultrametric tree.

Usage:
    python scripts/compute_phylogeny_embedding.py

Outputs:
    checkpoints/phylo_pca_features.npy - (n_species, n_components) array
    checkpoints/phylo_strain_order.txt  - strain IDs in order
"""

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from data.tsv_parser import parse_strain_registry, build_strain_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_distance_matrix(entries, tree_path: str) -> np.ndarray:
    """Compute pairwise phylogenetic distance matrix from Newick tree.

    Returns (n_species, n_species) distance matrix.
    """
    if not os.path.isfile(tree_path):
        logger.warning(f"Tree file not found: {tree_path}")
        logger.warning("Using identity + small random distances as fallback")
        n = len(entries)
        D = np.eye(n) * 0.01
        # Add hierarchical grouping
        for i, ei in enumerate(entries):
            for j, ej in enumerate(entries):
                if ei.group == ej.group:
                    D[i, j] = 0.3 if ei.species != ej.species else 0.05
                else:
                    D[i, j] = 0.8
        np.fill_diagonal(D, 0)
        return D

    try:
        from Bio import Phylo
    except ImportError:
        logger.error("biopython required for tree parsing. Using fallback.")
        return compute_distance_matrix.__wrapped__(entries, "")

    tree = Phylo.read(tree_path, "newick")
    # Map species to tree tips
    # Tree tip names format may vary: need to match to entry.species
    tips = {tip.name: tip for tip in tree.get_terminals()}

    n = len(entries)
    D = np.zeros((n, n))

    for i, ei in enumerate(entries):
        for j, ej in enumerate(entries):
            if i == j:
                continue
            # Find matching tips
            name_i = ei.species.replace("_", " ")
            name_j = ej.species.replace("_", " ")
            tip_i = tips.get(name_i) or tips.get(ei.species)
            tip_j = tips.get(name_j) or tips.get(ej.species)
            if tip_i and tip_j:
                D[i, j] = tree.distance(tip_i, tip_j)
            else:
                D[i, j] = 0.5  # fallback distance

    return D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default="all_lib_model_full.tsv")
    parser.add_argument("--tree",
                        default="../1_data/AllFungi/phylogeny/1_nwk/"
                                "Final_ultrametric_tree.nwk")
    parser.add_argument("--output", default="checkpoints/phylogeny")
    parser.add_argument("--n-components", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    entries = parse_strain_registry(args.tsv)
    active = [e for e in entries if not e.is_excluded]
    logger.info(f"Computing phylogeny for {len(active)} strains")

    D = compute_distance_matrix(active, args.tree)

    # PCA via eigendecomposition of distance matrix
    # Center the distance matrix
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D**2) @ J

    eigvals, eigvecs = np.linalg.eigh(B)

    # Sort by eigenvalue magnitude (descending)
    idx = np.argsort(-np.abs(eigvals))
    n_comp = min(args.n_components, n)
    components = eigvecs[:, idx[:n_comp]] * np.sqrt(np.abs(eigvals[idx[:n_comp]]))

    # Save
    np.save(os.path.join(args.output, "phylo_pca_features.npy"), components)

    strain_order = [e.strain for e in active]
    with open(os.path.join(args.output, "phylo_strain_order.txt"), "w") as f:
        f.write("\n".join(strain_order))

    logger.info(f"Phylogeny PCA features saved: {components.shape}")
    logger.info(f"Strain order: {strain_order}")

    # Print variance explained
    total_var = np.sum(np.abs(eigvals))
    var_explained = np.abs(eigvals[idx[:n_comp]]) / total_var
    for i, ve in enumerate(var_explained):
        logger.info(f"  PC{i+1}: {ve:.3f} variance explained")


if __name__ == "__main__":
    main()
