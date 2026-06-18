#!/usr/bin/env python3
"""
Pre-compute features for all strains.

Usage:
    python scripts/extract_features.py [--tsv all_lib_model_full.tsv] \\
        [--output checkpoints/features] [--workers 4]

Generates per-strain .npz files containing pre-computed features::
    checkpoints/features/
        P1/
            gtf_features.npz
            genome_profiles.npz
            positive_genes.npy
            negative_genes.npy
        ...
"""

import argparse
import logging
import os
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Optional

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from data.circ_info_encoding import (
    load_circ_info,
    filter_circ_info,
    get_positive_genes,
)
from data.genome_encoding import (
    GenomeIndexer,
    compute_genome_context_features,
)
from data.tsv_parser import (
    parse_strain_registry,
    get_group_members,
    StrainEntry,
)
from data.negative_sampling import get_negative_genes
from utils.gtf_utils import GeneModelIndexer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_strain(
    entry: StrainEntry,
    output_dir: str,
    flank_size: int = 150,
    genome_window_size: int = 10000,
) -> Dict:
    """Pre-compute features for a single strain.

    Returns summary dict.
    """
    strain = entry.strain
    strain_dir = os.path.join(output_dir, strain)
    os.makedirs(strain_dir, exist_ok=True)

    result = {"strain": strain, "status": "ok", "n_pos_genes": 0, "n_neg_genes": 0}

    try:
        # 1. Load CircInfo
        circ_df = load_circ_info(entry.circinfo_path)
        circ_df = filter_circ_info(circ_df)
        pos_genes = get_positive_genes(circ_df)
        n_pos = len(pos_genes)
        result["n_pos_genes"] = n_pos
        logger.info(f"{strain}: {n_pos} positive genes")

        # Save positive gene IDs
        np.save(os.path.join(strain_dir, "positive_gene_ids.npy"),
                np.array(list(pos_genes.keys()), dtype=object),
                allow_pickle=True)

        # 2. Index genome
        genome_idx = GenomeIndexer(entry.genome_path)

        # 3. Index GTF
        gene_model = GeneModelIndexer(entry.gtf_path)
        logger.info(f"{strain}: {gene_model.n_genes()} genes in GTF")

        # 4. Compute GTF features for all positive genes
        gtf_feats = {}
        for gid in pos_genes:
            gene = gene_model.get_gene(gid)
            if gene is not None:
                feats = gene_model.extract_features(gene)
                gtf_feats[gid] = feats

        # Save GTF features
        gtf_save = {gid: feats for gid, feats in gtf_feats.items()}
        np.savez(os.path.join(strain_dir, "gtf_features.npz"), **gtf_save)

        # 5. Compute genome context profiles for positive genes
        genome_profiles = {}
        for gid, gene in [(gid, gene_model.get_gene(gid)) for gid in pos_genes]:
            if gene is None:
                continue
            mid = (gene.start + gene.end) // 2
            window = genome_idx.extract_window(
                gene.chrom, mid, genome_window_size // 2
            )
            if window:
                profile = compute_genome_context_features(window)
                # Trim to 200 bins
                if profile.shape[0] > 200:
                    profile = profile[:200]
                elif profile.shape[0] < 200:
                    # Pad
                    padded = np.zeros((200, 8), dtype=np.float32)
                    padded[:profile.shape[0]] = profile
                    profile = padded
                genome_profiles[gid] = profile

        if genome_profiles:
            np.savez(os.path.join(strain_dir, "genome_profiles.npz"),
                     **{gid: p for gid, p in genome_profiles.items()})

        # 6. Sample negative genes
        neg_genes = get_negative_genes(
            gene_model, set(pos_genes.keys()), n_pos
        )
        result["n_neg_genes"] = len(neg_genes)
        logger.info(f"{strain}: {len(neg_genes)} negative genes sampled")

        np.save(os.path.join(strain_dir, "negative_gene_ids.npy"),
                np.array(neg_genes, dtype=object), allow_pickle=True)

        # 7. Save strain metadata
        meta = {
            "strain": strain,
            "species": entry.species,
            "group": entry.group,
            "n_positive_genes": n_pos,
            "n_negative_genes": len(neg_genes),
            "n_total_genes": gene_model.n_genes(),
        }
        np.save(os.path.join(strain_dir, "metadata.npy"), meta, allow_pickle=True)

        genome_idx.close()

    except Exception as e:
        logger.error(f"{strain} failed: {e}")
        result["status"] = f"error: {e}"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute features for PanCirc-Fungi"
    )
    parser.add_argument("--tsv", default="all_lib_model_full.tsv")
    parser.add_argument("--output", default="checkpoints/features")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--strains", nargs="*",
                        help="Only process specific strains (default: all)")
    parser.add_argument("--flank-size", type=int, default=150)
    parser.add_argument("--genome-window", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Parse TSV
    entries = parse_strain_registry(args.tsv)
    active = [e for e in entries if not e.is_excluded]

    if args.strains:
        active = [e for e in active if e.strain in args.strains]

    logger.info(f"Processing {len(active)} strains with {args.workers} workers")

    results = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for entry in active:
                r = pool.apply_async(
                    process_strain, (entry, args.output, args.flank_size, args.genome_window)
                )
                results.append(r)
            results = [r.get() for r in results]
    else:
        for entry in active:
            r = process_strain(entry, args.output, args.flank_size, args.genome_window)
            results.append(r)

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    total_pos = sum(r.get("n_pos_genes", 0) for r in results)
    total_neg = sum(r.get("n_neg_genes", 0) for r in results)
    errors = [r for r in results if r["status"] != "ok"]

    print(f"\n{'='*50}")
    print(f"Feature extraction complete:")
    print(f"  Strains processed: {ok}/{len(active)}")
    print(f"  Total positive genes: {total_pos}")
    print(f"  Total negative genes: {total_neg}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors:
            print(f"    {e['strain']}: {e['status']}")
    print(f"  Output: {os.path.abspath(args.output)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
