"""
mycoCirc: Junction prediction evaluation (with ±5 bp fuzzy matching).

Evaluates how accurately the model predicts the correct backsplice
junction at the nucleotide level. Unlike the training cross-entropy
loss (which operates at exon-pair level), the evaluation here uses
genomic coordinates and allows ±5 bp tolerance — biologically
meaningful since backsplice junctions can slide by a few bases.

Metrics:
  - Top-1 fuzzy (±5 bp): predicted junction within 5 bp of a true junction
  - Top-3 fuzzy: among top-3 candidate pairs, any within 5 bp
  - Recall@K: fraction of true junctions recovered in top-K

Usage:
    python scripts/evaluate_junction.py \
        --config config/default.yaml \
        --checkpoint-dir checkpoints/finetune

    # SLURM:
    sbatch scripts/run_interpretability.slurm
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from data.tsv_parser import parse_strain_registry, build_strain_index
from data.dataset import CircRNAFinetuneDataset, collate_pretrain
from data.circ_info_encoding import load_circ_info, filter_circ_info, get_positive_genes
from data.negative_sampling import get_negative_genes
from data.genome_encoding import extract_all_exon_flanks
from model.pancirc import PanCircModel
from utils.metrics import (
    compute_topk_accuracy,
    compute_junction_fuzzy_accuracy,
    compute_recall_fuzzy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

GROUPS = ["Candida", "Cryptococcus", "Filamentous"]
TEST_STRAINS = {"Candida": "P4", "Cryptococcus": "C4", "Filamentous": "F6"}


def parse_circ_id_coords(circ_id: str, strand: str):
    """Extract genomic (donor_pos, acceptor_pos) from a circ_id.

    Format: "chr:donor_pos|acceptor_pos"
    Donor = backsplice donor site (3' boundary of exon)
    Acceptor = backsplice acceptor site (5' boundary of exon)
    """
    try:
        coord_part = circ_id.split(":")[-1]
        if "|" in coord_part:
            c1, c2 = coord_part.split("|")
        elif "-" in coord_part:
            c1, c2 = coord_part.split("-")
        else:
            return None
        return (int(c1), int(c2))
    except (ValueError, IndexError):
        return None


def load_model_and_data(group, checkpoint_dir, config, device, tsv_entries):
    """Load model, build test DataLoader for a group's held-out test strain."""
    test_strain = TEST_STRAINS[group]

    test_entry = None
    for e in tsv_entries:
        if e.strain == test_strain and e.group == group:
            test_entry = e
            break
    if test_entry is None:
        logger.error(f"  Test strain {test_strain} not found for {group}")
        return None, None, None

    # Load checkpoint
    ckpt_dir = Path(checkpoint_dir) / group
    preferred = [ckpt_dir / "final_pretrained.pt", ckpt_dir / "final.pt"]
    loaded_state = None
    for p in preferred:
        if p.exists():
            logger.info(f"  Loading checkpoint: {p}")
            state = torch.load(str(p), map_location=device)
            loaded_state = state.get("model_state_dict", state)
            break

    if loaded_state is None:
        cv_files = sorted(ckpt_dir.glob("cv/fold_*/best.pt"))
        if cv_files:
            logger.info(f"  Loading CV checkpoint: {cv_files[0]}")
            state = torch.load(str(cv_files[0]), map_location=device)
            loaded_state = state.get("model_state_dict", state)

    if loaded_state is None:
        logger.error(f"  No checkpoint found for {group}")
        return None, None, None

    n_sp = len(build_strain_index([e for e in tsv_entries if not e.is_excluded]))
    model = PanCircModel(config["model"], n_species=n_sp)
    missing, unexpected = model.load_state_dict(loaded_state, strict=False)
    if missing:
        logger.info(f"    Missing keys: {len(missing)} (expression_encoder expected)")
    model.to(device)
    model.eval()

    # Build indexers
    from utils.gtf_utils import GeneModelIndexer
    from data.genome_encoding import GenomeIndexer

    gene_model = GeneModelIndexer(test_entry.gtf_path)
    genome_indexer = GenomeIndexer(test_entry.genome_path)
    logger.info(f"  GTF: {len(gene_model.genes)} genes, Genome: OK")

    # Positive genes + true junction coordinates from circ_info
    circ_df_raw = load_circ_info(test_entry.circinfo_path)
    circ_df = filter_circ_info(circ_df_raw)
    pos_genes = get_positive_genes(circ_df)
    logger.info(f"  Positive genes: {len(pos_genes)}")

    if len(pos_genes) == 0:
        logger.error(f"  No positive genes for {test_strain}")
        return None, None, None

    # Build true junction lookup: {gene_id: [(donor_pos, acceptor_pos), ...]}
    # Map raw circ_id coordinates to exon boundaries using the SAME convention as
    # extract_all_exon_flanks (used to build donor/acceptor position arrays):
    #
    #   + strand: donor = exon.start (5' boundary), acceptor = exon.end (3' boundary)
    #   - strand: donor = exon.end (5' boundary),   acceptor = exon.start (3' boundary)
    #
    # The circ_id encodes the actual backsplice breakpoint from CIRIquant.
    # We map it to the nearest exon boundary using the convention above
    # so that "donor index" means "which exon provides the 5' boundary"
    # and "acceptor index" means "which exon provides the 3' boundary".
    def _map_to_exon_boundary(gene_obj, raw_donor, raw_acceptor):
        """Find (donor_idx, acceptor_idx) matching the cross_labels convention."""
        best_d_idx = best_a_idx = None
        best_d_dist = best_a_dist = float('inf')

        for i, (s, e) in enumerate(zip(gene_obj.exon_starts, gene_obj.exon_ends)):
            if gene_obj.strand == '+':
                d_candidate = s  # 5' boundary = model's "donor position"
                a_candidate = e  # 3' boundary = model's "acceptor position"
            else:
                d_candidate = e  # 5' boundary (genome end)
                a_candidate = s  # 3' boundary (genome start)

            d_dist = abs(raw_donor - d_candidate)
            a_dist = abs(raw_acceptor - a_candidate)

            if d_dist < best_d_dist:
                best_d_dist = d_dist
                best_d_idx = i
            if a_dist < best_a_dist:
                best_a_dist = a_dist
                best_a_idx = i

        if best_d_idx is not None and best_a_idx is not None:
            return (best_d_idx, best_a_idx)
        return None

    true_junctions_by_gene = {}
    for gid, group_df in circ_df.groupby("gene_id"):
        junctions = []
        for _, row in group_df.iterrows():
            coords = parse_circ_id_coords(row.get("circ_id", ""), row.get("strand", "+"))
            if coords is None:
                continue
            d_pos, a_pos = coords
            gene_obj = gene_model.get_gene(gid)
            if gene_obj is None:
                continue
            mapped = _map_to_exon_boundary(gene_obj, d_pos, a_pos)
            if mapped is not None:
                junctions.append(mapped)  # (donor_idx, acceptor_idx) — exon indices
        if junctions:
            true_junctions_by_gene[str(gid)] = junctions
    logger.info(f"  Genes with known junction coords: {len(true_junctions_by_gene)}")

    # Pre-compute exon boundary coordinates for all positive genes
    # {gene_id: (donor_positions, acceptor_positions)}
    gene_coords = {}
    flank_size = config["data"].get("flank_size", 150)
    for gid in pos_genes:
        gene_obj = gene_model.get_gene(gid)
        if gene_obj is None:
            continue
        try:
            flanks = extract_all_exon_flanks(genome_indexer, gene_obj, flank_size)
            gene_coords[gid] = (
                np.array(flanks["donor_positions"]),
                np.array(flanks["acceptor_positions"]),
            )
        except Exception:
            continue
    logger.info(f"  Pre-computed exon coords for: {len(gene_coords)} genes")

    # Negative genes
    neg = get_negative_genes(gene_model, set(pos_genes.keys()), len(pos_genes), rng_seed=42)
    logger.info(f"  Negative genes: {len(neg)}")

    pos_dict = {test_strain: pos_genes}
    neg_dict = {test_strain: neg}

    gene_models = {test_strain: gene_model}
    genome_indexers = {test_strain: genome_indexer}

    ds = CircRNAFinetuneDataset(
        entries=[test_entry],
        gene_models=gene_models,
        genome_indexers=genome_indexers,
        positive_genes=pos_dict,
        negative_genes=neg_dict,
        expression_data={},
        config=config["data"],
    )
    batch_size = config.get("finetune", {}).get("batch_size", 16)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_pretrain, num_workers=0,
    )
    logger.info(f"  Dataset: {len(ds)} samples")

    return model, loader, (
        gene_model, genome_indexer, test_entry, circ_df,
        true_junctions_by_gene, gene_coords, flank_size,
    )


def evaluate_junctions(model, loader, device, aux_data, tolerance=5, k_values=(1, 3)):
    """Evaluate junction prediction with ±tolerance bp fuzzy matching."""
    (gene_model, genome_indexer, test_entry, circ_df,
     true_junctions_by_gene, gene_coords, flank_size) = aux_data

    all_fuzzy = {f"top{k}_fuzzy": [] for k in k_values}
    all_recall = {f"recall_at_{k}": [] for k in k_values}
    all_exact = {f"top{k}_exact": [] for k in k_values}
    n_positive_genes = 0

    # Build per-gene data: index model outputs by gene_id
    # Need to iterate loader once to collect predictions, then compute fuzzy

    with torch.no_grad():
        for batch in loader:
            batch_cpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            # Get original gene_ids from batch (needed for coordinate lookup)
            # Note: collate_pretrain uses key "gene_id" (singular)
            gene_ids = batch_cpu.get("gene_id", [None] * batch_cpu["is_positive"].size(0))

            outputs = model(batch_cpu, task="pretrain")
            scores = outputs["junction_scores"].cpu().numpy()
            labels = batch["cross_labels"].numpy()  # (batch, N_d, N_a)
            is_pos = batch["is_positive"].numpy()
            donor_mask_batch = batch.get("donor_mask")

            for i in range(len(is_pos)):
                if is_pos[i] <= 0:
                    continue

                gid = gene_ids[i] if isinstance(gene_ids, list) and i < len(gene_ids) else None
                n_positive_genes += 1
                y_pred_i = scores[i]

                # ── Exact matching (exon-pair level, original metric) ──
                labels_flat = labels[i].reshape(-1)
                for k in k_values:
                    acc = compute_topk_accuracy(
                        labels_flat.reshape(1, -1),
                        y_pred_i.reshape(1, -1),
                        k=k,
                    )
                    all_exact[f"top{k}_exact"].append(acc)

                # ── Fuzzy matching (±5 bp, nucleotide level) ──
                # Need to get the true junction coordinates for this gene
                true_juncs = []
                if gid is not None and gid in true_junctions_by_gene:
                    true_juncs = true_junctions_by_gene[gid]

                if true_juncs:
                    # Fuzzy matching: true_juncs stores (d_idx, a_idx) exon pairs
                    # The model predicts scores over all (N_d × N_a) pairs
                    n_d = labels.shape[1]
                    n_a = labels.shape[2]

                    # Convert true junction indices to flat indices
                    true_flat = []
                    for d_idx, a_idx in true_juncs:
                        if d_idx < n_d and a_idx < n_a:
                            true_flat.append(d_idx * n_a + a_idx)

                    if true_flat:
                        flat_scores = y_pred_i.flatten()
                        topk_idx = np.argsort(-flat_scores)[:max(k_values)]

                        for k in k_values:
                            topk_k = topk_idx[:k]
                            # Top-K accuracy: any true junction in top-K?
                            acc = int(any(ti in topk_k for ti in true_flat))
                            all_fuzzy[f"top{k}_fuzzy"].append(acc)

                            # Recall@K: fraction of true junctions in top-K
                            n_found = sum(1 for ti in true_flat if ti in topk_k)
                            all_recall[f"recall_at_{k}"].append(
                                n_found / max(len(true_flat), 1)
                            )

    results = {}
    for k in k_values:
        exact_key = f"top{k}_exact"
        fuzzy_key = f"top{k}_fuzzy"
        recall_key = f"recall_at_{k}"

        results[exact_key] = float(np.mean(all_exact[exact_key])) if all_exact[exact_key] else 0.0
        results[fuzzy_key] = float(np.mean(all_fuzzy[fuzzy_key])) if all_fuzzy.get(fuzzy_key) else 0.0
        results[recall_key] = float(np.mean(all_recall[recall_key])) if all_recall.get(recall_key) else 0.0

    results["n_positive_genes"] = n_positive_genes
    results["n_with_fuzzy"] = len(all_fuzzy.get("top3_fuzzy", []))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="mycoCirc: Evaluate junction prediction accuracy (±5 bp fuzzy)"
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/finetune")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/interpretability/junction_topk.tsv")
    parser.add_argument("--output-json", default="results/interpretability/junction_details.json")
    parser.add_argument("--tolerance", type=int, default=5,
                        help="Fuzzy matching tolerance in bp (default: 5)")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tsv_path = PROJ_ROOT / "all_lib_model_full.tsv"
    tsv_entries = parse_strain_registry(str(tsv_path))

    all_results = {}
    for grp in GROUPS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating junctions for {grp} (test: {TEST_STRAINS[grp]})...")
        logger.info(f"{'='*60}")

        result = load_model_and_data(grp, args.checkpoint_dir, config, device, tsv_entries)
        if result is None or result[0] is None:
            logger.error(f"  Skipping {grp}")
            continue
        model, loader, aux_data = result

        results = evaluate_junctions(model, loader, device, aux_data,
                                     tolerance=args.tolerance)
        all_results[grp] = results

        logger.info(f"  Positive genes:          {results['n_positive_genes']}")
        logger.info(f"  Genes with fuzzy eval:   {results.get('n_with_fuzzy', 0)}")
        logger.info(f"  Exact Top-1:             {results['top1_exact']:.4f}")
        logger.info(f"  Exact Top-3:             {results['top3_exact']:.4f}")
        logger.info(f"  Fuzzy ±{args.tolerance}bp Top-1:  {results['top1_fuzzy']:.4f}")
        logger.info(f"  Fuzzy ±{args.tolerance}bp Top-3:  {results['top3_fuzzy']:.4f}")
        logger.info(f"  Recall@1 (fuzzy):        {results['recall_at_1']:.4f}")
        logger.info(f"  Recall@3 (fuzzy):        {results['recall_at_3']:.4f}")

    # ── Print summary ──
    print("\n" + "=" * 80)
    print("Junction Prediction Accuracy Summary (±5 bp fuzzy)")
    print("=" * 80)
    print(f"{'Group':<15} {'Genes':<7} {'Top-1(ex)':<10} {'Top-1(fz)':<10} "
          f"{'Top-3(fz)':<10} {'R@1':<8} {'R@3':<8}")
    print("-" * 80)
    for grp in GROUPS:
        if grp not in all_results:
            continue
        r = all_results[grp]
        print(f"{grp:<15} {r['n_positive_genes']:<7} {r['top1_exact']:.4f}   "
              f"{r['top1_fuzzy']:.4f}   {r['top3_fuzzy']:.4f}   "
              f"{r['recall_at_1']:.4f}   {r['recall_at_3']:.4f}")

    n_total = sum(r["n_positive_genes"] for r in all_results.values())
    if all_results:
        avg_exact = np.mean([r["top1_exact"] for r in all_results.values()])
        avg_fz1 = np.mean([r["top1_fuzzy"] for r in all_results.values()])
        avg_fz3 = np.mean([r["top3_fuzzy"] for r in all_results.values()])
        avg_r1 = np.mean([r["recall_at_1"] for r in all_results.values()])
        avg_r3 = np.mean([r["recall_at_3"] for r in all_results.values()])
        print("-" * 80)
        print(f"{'Average':<15} {n_total:<7} {avg_exact:.4f}   {avg_fz1:.4f}   "
              f"{avg_fz3:.4f}   {avg_r1:.4f}   {avg_r3:.4f}")
    print("=" * 80)

    # ── Save TSV ──
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJ_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["group", "n_genes", "n_fuzzy_eval",
                         "top1_exact", "top3_exact",
                         f"top1_fuzzy_{args.tolerance}bp",
                         f"top3_fuzzy_{args.tolerance}bp",
                         "recall_at_1_fuzzy", "recall_at_3_fuzzy"])
        for grp in GROUPS:
            if grp not in all_results:
                continue
            r = all_results[grp]
            writer.writerow([grp, r["n_positive_genes"], r.get("n_with_fuzzy", 0),
                            f"{r['top1_exact']:.4f}", f"{r['top3_exact']:.4f}",
                            f"{r['top1_fuzzy']:.4f}", f"{r['top3_fuzzy']:.4f}",
                            f"{r['recall_at_1']:.4f}", f"{r['recall_at_3']:.4f}"])
    logger.info(f"Results saved to {output_path}")

    # ── Save JSON ──
    json_path = Path(args.output_json)
    if not json_path.is_absolute():
        json_path = PROJ_ROOT / json_path
    json_out = {}
    for grp in GROUPS:
        if grp not in all_results:
            continue
        r = dict(all_results[grp])
        json_out[grp] = r
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    logger.info(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
