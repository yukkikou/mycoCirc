"""
mycoCirc: Master interpretability pipeline.

Runs all interpretability analyses across all three taxonomic groups:

  1. GTF Feature Importance (Integrated Gradients)
  2. Junction cross-attention visualization
  3. K-mer motif enrichment at backsplice junctions
  4. Modality ablation bar chart (from existing results)
  5. Genome browser track (bedGraph) for a representative strain

Outputs figures and TSV tables to results/interpretability/ and figures/.

Usage:
    # All groups sequentially (GPU recommended):
    python scripts/run_interpretability.py \
        --config config/default.yaml \
        --checkpoint-dir checkpoints/finetune

    # Single group:
    python scripts/run_interpretability.py --group Candida

    # SLURM:
    sbatch scripts/run_interpretability.slurm
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from data.tsv_parser import parse_strain_registry, build_strain_index
from data.dataset import CircRNAFinetuneDataset, collate_pretrain
from model.pancirc import PanCircModel
from utils.gtf_utils import GeneModelIndexer
from data.genome_encoding import GenomeIndexer
from data.circ_info_encoding import load_circ_info, filter_circ_info, get_positive_genes
from data.negative_sampling import get_negative_genes

# Interpretability modules
from interpret.feature_importance import (
    compute_integrated_gradients,
    aggregate_importances,
    print_feature_ranking,
    GTF_FEATURE_NAMES,
)
from interpret.attention_viz import (
    plot_modality_ablation,
    plot_cross_attention_examples,
)
from interpret.motif_discovery import (
    extract_high_attention_kmers,
    find_enriched_motifs,
    plot_sequence_logo,
)
from interpret.genome_scan import scan_genome, write_bedgraph, write_bed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

GROUPS = ["Candida", "Cryptococcus", "Filamentous"]
TEST_STRAINS = {"Candida": "P4", "Cryptococcus": "C4", "Filamentous": "F6"}


def find_test_entry(tsv_entries, group):
    """Find the held-out test strain entry for a group."""
    ts = TEST_STRAINS[group]
    for e in tsv_entries:
        if e.strain == ts and e.group == group:
            return e
    return None


def load_group_model_and_data(group, checkpoint_dir, config, device, tsv_entries,
                               expression_data=None):
    """Load model and build test DataLoader for one group."""
    test_entry = find_test_entry(tsv_entries, group)
    if test_entry is None:
        logger.error(f"  Test strain not found for {group}")
        return None, None, None, None

    # Load checkpoint
    ckpt_dir = Path(checkpoint_dir) / group
    preferred = [
        ckpt_dir / "final_pretrained.pt",
        ckpt_dir / "final.pt",
    ]
    loaded_state = None
    for p in preferred:
        if p.exists():
            logger.info(f"  Loading: {p}")
            state = torch.load(str(p), map_location=device)
            loaded_state = state.get("model_state_dict", state)
            break

    if loaded_state is None:
        cv_files = sorted(ckpt_dir.glob("cv/fold_*/best.pt"))
        if cv_files:
            logger.info(f"  Loading CV: {cv_files[0]}")
            state = torch.load(str(cv_files[0]), map_location=device)
            loaded_state = state.get("model_state_dict", state)

    if loaded_state is None:
        logger.error(f"  No checkpoint for {group}")
        return None, None, None, None

    n_sp = len(build_strain_index([e for e in tsv_entries if not e.is_excluded]))
    model = PanCircModel(config["model"], n_species=n_sp)
    missing, unexpected = model.load_state_dict(loaded_state, strict=False)
    if missing:
        logger.info(f"    Missing keys: {len(missing)} (expected)")
    model.to(device)
    model.eval()

    # Build indexers
    gene_model = GeneModelIndexer(test_entry.gtf_path)
    genome_indexer = GenomeIndexer(test_entry.genome_path)
    logger.info(f"  GTF: {len(gene_model.genes)} genes, Genome: OK")

    # Positive/negative genes
    circ_df = filter_circ_info(load_circ_info(test_entry.circinfo_path))
    pos_genes = get_positive_genes(circ_df)
    if len(pos_genes) == 0:
        logger.error(f"  No positive genes for {test_entry.strain}")
        return None, None, None, None

    neg_list = get_negative_genes(gene_model, set(pos_genes.keys()), len(pos_genes), rng_seed=42)
    logger.info(f"  Positives: {len(pos_genes)}, Negatives: {len(neg_list)}")

    pos_dict = {test_entry.strain: pos_genes}
    neg_dict = {test_entry.strain: neg_list}
    gene_models = {test_entry.strain: gene_model}
    genome_indexers = {test_entry.strain: genome_indexer}

    ds = CircRNAFinetuneDataset(
        entries=[test_entry],
        gene_models=gene_models,
        genome_indexers=genome_indexers,
        positive_genes=pos_dict,
        negative_genes=neg_dict,
        expression_data=expression_data or {},
        config=config["data"],
    )
    batch_size = config.get("finetune", {}).get("batch_size", 16)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_pretrain, num_workers=0,
    )

    return model, loader, gene_model, genome_indexer


def run_feature_importance(model, loader, device, output_dir):
    """Compute Integrated Gradients for GTF features and save results."""
    logger.info("  Computing GTF feature importance (Integrated Gradients)...")
    all_ig = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        gtf_input = batch["gtf_features"]
        ig = compute_integrated_gradients(model, gtf_input, n_steps=25)
        all_ig.append(ig)
        if len(all_ig) * len(gtf_input) >= 256:  # limit samples
            break

    all_ig = np.concatenate(all_ig, axis=0)
    agg = aggregate_importances(all_ig, GTF_FEATURE_NAMES)

    # Save TSV
    tsv_path = output_dir / "feature_importance.tsv"
    with open(tsv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["feature", "importance"])
        for name, score in agg.items():
            w.writerow([name, f"{score:.6f}"])
    logger.info(f"  Feature importance saved to {tsv_path}")

    # Print ranking
    print_feature_ranking(agg, n_top=17)

    return agg


def run_motif_discovery(model, loader, device, output_dir, k=3):
    """Extract high-attention k-mers and find enriched motifs."""
    logger.info("  Discovering sequence motifs at backsplice junctions...")
    pos_kmers_all = []
    bg_kmers_all = []

    n_processed = 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        outputs = model(batch, task="pretrain")
        probs = torch.sigmoid(outputs["gene_logits"].squeeze(-1))
        labels = batch["is_positive"]
        donor_attn = outputs.get("donor_attn")
        acceptor_attn = outputs.get("acceptor_attn")

        if donor_attn is None:
            continue

        donor_attn_np = donor_attn.detach().cpu().numpy()
        acceptor_attn_np = acceptor_attn.detach().cpu().numpy()
        donor_kmers = batch.get("donor_kmers")
        acceptor_kmers = batch.get("acceptor_kmers")

        if donor_kmers is None:
            continue

        donor_kmers_np = donor_kmers.detach().cpu().numpy()
        acceptor_kmers_np = acceptor_kmers.detach().cpu().numpy()

        batch_size = len(probs)
        for i in range(batch_size):
            if labels[i].item() <= 0:
                continue  # skip negatives

            # Get donor attention and k-mers
            d_attn_sites = donor_attn_np[i]  # (N_d,)
            d_kmers_sites = donor_kmers_np[i]  # (N_d, L)

            for site_idx in range(len(d_attn_sites)):
                if d_attn_sites[site_idx] < 0.05:
                    continue
                kmers = d_kmers_sites[site_idx]
                # Create flat attention per position (uniform across site)
                attn_per_pos = np.full(len(kmers), d_attn_sites[site_idx] / len(kmers))
                high_kmers = extract_high_attention_kmers(attn_per_pos, kmers, k=k)
                pos_kmers_all.extend(high_kmers)
                bg_kmers_all.extend([str(t) for t in kmers[:min(10, len(kmers))]])

            # Same for acceptor
            a_attn_sites = acceptor_attn_np[i]
            a_kmers_sites = acceptor_kmers_np[i]
            for site_idx in range(len(a_attn_sites)):
                if a_attn_sites[site_idx] < 0.05:
                    continue
                kmers = a_kmers_sites[site_idx]
                attn_per_pos = np.full(len(kmers), a_attn_sites[site_idx] / len(kmers))
                high_kmers = extract_high_attention_kmers(attn_per_pos, kmers, k=k)
                pos_kmers_all.extend(high_kmers)

        n_processed += batch_size
        if n_processed >= 200:
            break

    if not pos_kmers_all:
        logger.warning("  No positive kmers found for motif discovery")
        return

    enriched = find_enriched_motifs(pos_kmers_all, bg_kmers_all, n_top=10)

    # Save TSV
    tsv_path = output_dir / "motif_enrichment.tsv"
    with open(tsv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["motif", "log2_enrichment"])
        for motif, score in enriched.items():
            w.writerow([motif, f"{score:.4f}"])
    logger.info(f"  Motif enrichment saved to {tsv_path}")

    # Plot
    logo_path = output_dir / "motif_logo.png"
    try:
        plot_sequence_logo(enriched, str(logo_path))
    except Exception as ex:
        logger.warning(f"  Motif logo plot failed: {ex}")

    # Print
    print(f"\n  Top enriched k-mers near backsplice junctions:")
    for i, (motif, score) in enumerate(list(enriched.items())[:10], 1):
        print(f"    {i}. {motif:>4s}  log2(FC)={score:+.2f}")

    return enriched


def main():
    parser = argparse.ArgumentParser(description="PanCirc-Fungi interpretability pipeline")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/finetune")
    parser.add_argument("--output-dir", default="results/interpretability")
    parser.add_argument("--fig-dir", default="figures")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--group", default=None, choices=GROUPS)
    parser.add_argument("--skip-feature-importance", action="store_true")
    parser.add_argument("--skip-cross-attention", action="store_true")
    parser.add_argument("--skip-motif", action="store_true")
    parser.add_argument("--skip-genome-scan", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJ_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = Path(args.fig_dir)
    if not fig_dir.is_absolute():
        fig_dir = PROJ_ROOT / fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = PROJ_ROOT / "all_lib_model_full.tsv"
    tsv_entries = parse_strain_registry(str(tsv_path))

    groups = [args.group] if args.group else GROUPS

    all_feature_importance = {}
    for grp in groups:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {grp}...")
        logger.info(f"{'='*60}")

        model, loader, gm, gi = load_group_model_and_data(
            grp, args.checkpoint_dir, config, device, tsv_entries
        )
        if model is None:
            logger.warning(f"  Skipping {grp}")
            continue

        grp_out = output_dir / grp
        grp_out.mkdir(parents=True, exist_ok=True)

        # 1. Feature importance
        if not args.skip_feature_importance:
            try:
                feat_imp = run_feature_importance(model, loader, device, grp_out)
                all_feature_importance[grp] = feat_imp
            except Exception as ex:
                logger.error(f"  Feature importance failed: {ex}")

        # 2. Cross-attention visualization
        if not args.skip_cross_attention:
            try:
                # Get a batch of positive examples for cross-attention plot
                pos_loader = None
                for batch in loader:
                    # Filter to positive-only subset
                    pos_idx = batch["is_positive"] > 0
                    if pos_idx.sum() > 0:
                        pos_loader = [batch]
                        break

                if pos_loader:
                    ca_path = grp_out / "cross_attention.png"
                    plot_cross_attention_examples(
                        model, pos_loader, device,
                        save_path=str(ca_path), n_examples=5,
                    )
            except Exception as ex:
                logger.error(f"  Cross-attention failed: {ex}")

        # 3. Motif discovery
        if not args.skip_motif:
            try:
                run_motif_discovery(model, loader, device, grp_out, k=3)
            except Exception as ex:
                logger.error(f"  Motif discovery failed: {ex}")

    # 4. Modality ablation plot
    ablation_tsv = PROJ_ROOT / "results" / "ablations.tsv"
    if ablation_tsv.exists():
        logger.info("\nGenerating modality ablation plot...")
        try:
            plot_modality_ablation(
                str(ablation_tsv),
                save_path=str(fig_dir / "fig_modality_ablation.pdf"),
            )
        except Exception as ex:
            logger.error(f"  Ablation plot failed: {ex}")
    else:
        logger.warning(f"  Ablation TSV not found at {ablation_tsv}")

    # 5. Genome scan (Filamentous → F6 as representative)
    if not args.skip_genome_scan and ("Filamentous" in groups):
        logger.info("\nRunning genome-wide scan for Filamentous (F6)...")
        try:
            model_f6, _, gm_f6, gi_f6 = load_group_model_and_data(
                "Filamentous", args.checkpoint_dir, config, device, tsv_entries
            )
            if model_f6 is not None:
                test_entry_f6 = find_test_entry(tsv_entries, "Filamentous")
                gene_models = {test_entry_f6.strain: gm_f6}
                genome_indexers = {test_entry_f6.strain: gi_f6}

                scan_results = scan_genome(
                    model_f6, gene_models, genome_indexers,
                    strain=test_entry_f6.strain, device=device,
                    batch_size=64,
                )
                if len(scan_results) > 0:
                    bg_path = output_dir / "predictions_F6.bedgraph"
                    bed_path = output_dir / "predictions_F6.bed"
                    write_bedgraph(scan_results, str(bg_path))
                    write_bed(scan_results, str(bed_path), threshold=0.5)
        except Exception as ex:
            logger.error(f"  Genome scan failed: {ex}")

    # 6. Save aggregated feature importance
    if all_feature_importance:
        agg_path = output_dir / "feature_importance_all.tsv"
        with open(agg_path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            groups_list = list(all_feature_importance.keys())
            header = ["feature"] + groups_list
            w.writerow(header)

            # Collect all feature names
            all_feats = set()
            for v in all_feature_importance.values():
                all_feats.update(v.keys())
            for feat in sorted(all_feats):
                row = [feat]
                for grp in groups_list:
                    row.append(f"{all_feature_importance[grp].get(feat, 0):.6f}")
                w.writerow(row)
        logger.info(f"Aggregated feature importance saved to {agg_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Interpretability pipeline complete!")
    logger.info(f"  Outputs: {output_dir}")
    logger.info(f"  Figures: {fig_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
