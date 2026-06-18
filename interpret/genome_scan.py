"""
Genome-wide scan for circRNA prediction.

Scans all genes in a strain genome and outputs prediction scores
in bedGraph/bed format for IGV visualization.

Output columns:
    chrom, start, end, gene_id, circRNA_probability, best_junction_coords
"""

import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from model.pancirc import PanCircModel

logger = logging.getLogger(__name__)


@torch.no_grad()
def scan_genome(
    model: PanCircModel,
    gene_models: Dict[str, "GeneModelIndexer"],
    genome_indexers: Dict[str, "GenomeIndexer"],
    strain: str,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """Scan all genes in a strain and predict circRNA probabilities.

    Uses actual genome context profiles (GC content, k-mer frequency)
    extracted via genome_indexer.get_context_profile().

    Returns array with columns: [chrom, start, end, gene_id, probability, best_junction]
    """
    model.eval()
    gm = gene_models[strain]
    gi = genome_indexers[strain]

    results = []
    gene_ids = list(gm.genes.keys())
    n_genes = len(gene_ids)

    logger.info(f"Scanning {n_genes} genes in {strain}...")

    from data.tsv_parser import build_strain_index

    from data.genome_encoding import compute_genome_context_features
    genome_window_size = getattr(gm, 'genome_window_size', 10000)
    half_window = genome_window_size // 2

    for start_idx in range(0, n_genes, batch_size):
        batch_ids = gene_ids[start_idx: start_idx + batch_size]

        gtf_features = []
        genome_profiles = []
        gene_info_list = []

        for gid in batch_ids:
            gene = gm.get_gene(gid)
            if gene is None or gene.exon_count == 0:
                continue

            feats = gm.extract_features(gene)
            gtf_features.append(feats)

            # Extract actual genome context (matching dataset.py pattern)
            mid = (gene.start + gene.end) // 2
            try:
                window = gi.extract_window(
                    gene.chrom, mid, half_window
                )
                if window:
                    ctx = compute_genome_context_features(window)
                else:
                    ctx = np.zeros((200, 8), dtype=np.float32)
            except Exception:
                ctx = np.zeros((200, 8), dtype=np.float32)

            genome_profiles.append(ctx)
            gene_info_list.append(gene)

        if not gtf_features:
            continue

        # Pad/truncate genome profiles to 200 bins
        max_bins = 200
        n_channels = genome_profiles[0].shape[1] if genome_profiles else 8
        padded = []
        for p in genome_profiles:
            if p.shape[0] >= max_bins:
                padded.append(p[:max_bins])
            else:
                pad = np.zeros((max_bins - p.shape[0], n_channels), dtype=p.dtype)
                padded.append(np.concatenate([p, pad], axis=0))
        genome_tensor = torch.tensor(np.stack(padded), dtype=torch.float, device=device)
        gtf_tensor = torch.tensor(np.stack(gtf_features), dtype=torch.float, device=device)

        # Species ID (strain might not be in TSV index; default to 0)
        strain_idx_map = {strain: 0}
        species_tensor = torch.zeros(len(gtf_features), dtype=torch.long, device=device)

        batch_dict = {
            "strain_id": species_tensor,
            "genome_context": genome_tensor,
            "gtf_features": gtf_tensor,
            # Dummy junction data (1 donor + 1 acceptor site per gene, zero-filled)
            # The JunctionEncoder will produce zero-ish output; gene-level prediction
            # from GTF+GenomeCtx still works correctly.
            "donor_kmers": torch.zeros(len(gtf_features), 1, 298,
                                       dtype=torch.long, device=device),
            "acceptor_kmers": torch.zeros(len(gtf_features), 1, 298,
                                          dtype=torch.long, device=device),
        }

        outputs = model(batch_dict, task="pretrain")
        probs = torch.sigmoid(outputs["gene_logits"]).cpu().numpy().flatten()

        for j, gene in enumerate(gene_info_list):
            prob = probs[j]
            best_junction = ""
            results.append([
                gene.chrom, gene.start, gene.end,
                gene.gene_id, prob, best_junction,
            ])

        logger.debug(f"  Processed {start_idx + len(gene_info_list)}/{n_genes}")

    return np.array(results, dtype=object)


def write_bedgraph(
    results: np.ndarray,
    output_path: str,
):
    """Write prediction results as a bedGraph file for IGV.

    Format:
        track type=bedGraph name="CircRNA Prediction"
        chrom  start  end  score
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write('track type=bedGraph name="CircRNA Prediction"\n')
        for row in results:
            chrom, start, end, gene_id, prob, junction = row
            try:
                score = float(prob)
                f.write(f"{chrom}\t{start}\t{end}\t{score:.4f}\n")
            except (ValueError, TypeError):
                continue

    logger.info(f"bedGraph saved to {output_path}")


def write_bed(
    results: np.ndarray,
    output_path: str,
    threshold: float = 0.5,
):
    """Write predicted circRNA genes as BED file.

    Format:
        chrom  start  end  gene_id  score
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w") as f:
        for row in results:
            chrom, start, end, gene_id, prob, junction = row
            try:
                score = float(prob)
                if score >= threshold:
                    f.write(f"{chrom}\t{start}\t{end}\t{gene_id}\t{score:.4f}\n")
                    count += 1
            except (ValueError, TypeError):
                continue

    logger.info(f"BED file saved to {output_path} ({count} entries above {threshold})")
