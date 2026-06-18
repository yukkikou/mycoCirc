"""
PanCirc-Fungi: Test Talaromyces marneffei PM1
==============================================
Tests a completely unseen species (PM1) on finetuned models from all 3 groups.

Tests:
  - Each group's finetuned model → PM1 Mode A (Genome+GTF) & Mode B (+GeneExp)
  - Reports AUROC, AUPRC, F1, accuracy, MCC, and gene count

Usage:
    python scripts/test_pm1.py \
        --config config/default.yaml \
        --checkpoint-dir checkpoints/finetune

    # SLURM:
    sbatch scripts/test_pm1.slurm
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

from data.tsv_parser import parse_strain_registry, StrainEntry, build_strain_index
from data.dataset import CircRNAFinetuneDataset, collate_pretrain
from data.circ_info_encoding import load_circ_info, filter_circ_info, get_positive_genes
from data.negative_sampling import get_negative_genes
from data.expression_encoding import load_expression_csv, align_circ_to_gene_expression, pad_to_max_replicates
from utils.gtf_utils import GeneModelIndexer
from data.genome_encoding import GenomeIndexer
from model.pancirc import PanCircModel
from utils.metrics import classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

GROUPS = ["Candida", "Cryptococcus", "Filamentous"]

# Hardcoded PM1 strain entry (from test_sample.tsv)
PM1_ENTRY = StrainEntry(
    group="Talaromyces",
    species="Talaromyces_marneffei",
    strain="PM1",
    genome_path="/media/share/resource/reference/talaromyces_marneffei/genome/pm1.soft.fasta",
    gtf_path="/media/share/resource/reference/talaromyces_marneffei/annotation/pm1_scallop_final.gtf",
    circinfo_path="/media/share/data5/1610305236/circTS/3_CIRIquant/circTS_circRNA_info.csv",
    circexp_path="/media/share/data5/1610305236/circTS/3_CIRIquant/circTS_circRNA_bsj.csv",
    geneexp_path="/media/share/data5/1610305236/circTS/3_CIRIquant/circTS_gene_count_matrix.csv",
    is_excluded=False,
)


def load_pm1_data(max_replicates=3):
    """Build indexers and expression data for PM1."""
    logger.info("Loading PM1 data...")

    # Build indexers
    gene_model = None
    genome_indexer = None
    try:
        gene_model = GeneModelIndexer(PM1_ENTRY.gtf_path)
        logger.info(f"  GTF loaded: {len(gene_model.genes)} genes")
    except Exception as ex:
        logger.error(f"  Failed to load GTF: {ex}")
        return None, None, None, None, None

    try:
        genome_indexer = GenomeIndexer(PM1_ENTRY.genome_path)
        logger.info(f"  Genome loaded: {genome_indexer}")
    except Exception as ex:
        logger.error(f"  Failed to load genome: {ex}")
        return None, None, None, None, None

    # Load circ info for positive/negative genes
    circ_df = load_circ_info(PM1_ENTRY.circinfo_path)
    circ_df = filter_circ_info(circ_df)
    logger.info(f"  CircInfo: {len(circ_df)} filtered records")
    pos_genes = get_positive_genes(circ_df)
    logger.info(f"  Positive genes: {len(pos_genes)}")

    if len(pos_genes) == 0:
        logger.error("  No positive genes found for PM1!")
        return gene_model, genome_indexer, None, None, None

    # Negative sampling
    neg = get_negative_genes(gene_model, set(pos_genes.keys()), len(pos_genes), rng_seed=42)
    logger.info(f"  Negative genes: {len(neg)}")

    pos_genes_dict = {PM1_ENTRY.strain: pos_genes}

    # Expression data
    expression_data = {}
    try:
        circ_exp_raw = load_expression_csv(PM1_ENTRY.circexp_path)
        gene_exp_raw = load_expression_csv(PM1_ENTRY.geneexp_path)

        gene_exp_map = {}
        if gene_exp_raw is not None:
            ge_df = gene_exp_raw.copy()
            if "gene_id" in ge_df.columns:
                ge_df = ge_df.set_index("gene_id")
            for gid in ge_df.index:
                vals = ge_df.loc[gid].values
                if vals.ndim == 0:
                    vals = np.array([vals])
                vals = np.log1p(vals.astype(np.float64))
                vals = (vals - vals.mean()) / max(vals.std(), 1e-8)
                gene_exp_map[str(gid)] = pad_to_max_replicates(vals, max_replicates)

        aligned = {}
        if circ_exp_raw is not None:
            aligned = align_circ_to_gene_expression(circ_df, circ_exp_raw, gene_exp_raw)
            logger.info(f"  Aligned genes: {len(aligned)}")

        expression_data[PM1_ENTRY.strain] = {"gene_exp": gene_exp_map, "aligned": aligned}
        logger.info(f"  GeneExp entries: {len(gene_exp_map)}")
    except Exception as ex:
        logger.warning(f"  Expression loading issue: {ex}")

    return gene_model, genome_indexer, pos_genes_dict, neg, expression_data


def load_group_checkpoint(group_name, checkpoint_dir, config, device, active_entries):
    """Load a group's best finetuned checkpoint. Try final_pretrained.pt first."""
    ckpt_dir = Path(checkpoint_dir) / group_name

    # Prefer pretrained backup
    preferred = [
        ckpt_dir / "final_pretrained.pt",
        ckpt_dir / "final.pt",
    ]

    loaded_state = None
    loaded_from = None
    for p in preferred:
        if p.exists():
            logger.info(f"  Loading: {p}")
            state = torch.load(str(p), map_location=device)
            if "model_state_dict" in state:
                loaded_state = state["model_state_dict"]
            else:
                loaded_state = state
            loaded_from = str(p)
            break

    if loaded_state is None:
        # Try CV checkpoints
        cv_files = sorted(ckpt_dir.glob("cv/fold_*/best.pt"))
        if cv_files:
            logger.info(f"  Loading CV checkpoint: {cv_files[0]}")
            state = torch.load(str(cv_files[0]), map_location=device)
            loaded_state = state["model_state_dict"]
            loaded_from = str(cv_files[0])

    if loaded_state is None:
        logger.error(f"  No checkpoint found for {group_name}")
        return None

    n_sp = len(build_strain_index(active_entries))
    model = PanCircModel(config["model"], n_species=n_sp)
    missing, unexpected = model.load_state_dict(loaded_state, strict=False)
    if missing:
        logger.info(f"  Missing keys: {len(missing)} (expected for expression_encoder)")
    if unexpected:
        logger.info(f"  Unexpected keys: {len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def evaluate(model, loader, device, task="pretrain", label=""):
    """Run inference and return metrics."""
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(batch, task=task)
            probs = torch.sigmoid(outputs["gene_logits"].squeeze(-1))
            all_labels.append(batch["is_positive"].cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    valid = y_true >= 0
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    metrics = classification_metrics(y_true, y_prob)
    logger.info(f"  {label:35s} → AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
                f"F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}  MCC={metrics['mcc']:.4f}  "
                f"N={len(y_true)}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Test PM1 on all finetuned models")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/finetune")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/pm1_test_results.json")
    parser.add_argument("--output-tsv", default="results/pm1_test_results.tsv")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_config = config["data"]
    batch_size = config.get("finetune", {}).get("batch_size", 16)

    # Load full TSV for species index building
    tsv_path = PROJ_ROOT / "all_lib_model_full.tsv"
    all_entries = parse_strain_registry(str(tsv_path))
    active = [e for e in all_entries if not e.is_excluded]

    # Load PM1 data
    gene_model, genome_indexer, pos_genes, neg_list, expression_data = load_pm1_data()
    if gene_model is None or pos_genes is None:
        logger.error("Failed to load PM1 data. Aborting.")
        sys.exit(1)

    neg_genes = {PM1_ENTRY.strain: neg_list}

    # Build PM1 dataset
    pm1_entries = [PM1_ENTRY]
    gene_models = {PM1_ENTRY.strain: gene_model}
    genome_indexers = {PM1_ENTRY.strain: genome_indexer}

    ds = CircRNAFinetuneDataset(
        entries=pm1_entries,
        gene_models=gene_models,
        genome_indexers=genome_indexers,
        positive_genes=pos_genes,
        negative_genes=neg_genes,
        expression_data=expression_data,
        config=data_config,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_pretrain, num_workers=0,
    )
    logger.info(f"PM1 dataset: {len(ds)} samples ({len(neg_list)} pos, ? neg)")

    # Test each group's finetuned model
    all_results = {}
    for grp in GROUPS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {grp} finetuned model on PM1...")
        logger.info(f"{'='*60}")

        model = load_group_checkpoint(grp, args.checkpoint_dir, config, device, active)
        if model is None:
            logger.error(f"  Skipping {grp} — no checkpoint")
            continue

        # Mode A: Genome+GTF (task="pretrain", no expression encoder)
        metrics_a = evaluate(model, loader, device, task="pretrain",
                             label=f"[{grp}] Mode A (Genome+GTF)")

        # Mode B: Genome+GTF+GeneExp (task="finetune")
        metrics_b = evaluate(model, loader, device, task="finetune",
                             label=f"[{grp}] Mode B (+GeneExp)")

        all_results[grp] = {
            "Mode_A": metrics_a,
            "Mode_B": metrics_b,
        }

    # ── Summary table ──────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("PM1 Test Summary — Talaromyces marneffei (Filamentous, unseen)")
    print("=" * 100)

    header = f"{'Model':<20} {'Mode':<12} {'AUROC':<9} {'AUPRC':<9} {'F1':<9} {'Acc':<9} {'MCC':<9} {'N':<7}"
    print(header)
    print("-" * 100)

    for grp in GROUPS:
        if grp not in all_results:
            continue
        for mode in ["Mode_A", "Mode_B"]:
            m = all_results[grp][mode]
            label = f"{grp} finetuned"
            mode_label = "GTF+Genome" if mode == "Mode_A" else "+GeneExp"
            print(f"{label:<20} {mode_label:<12} {m['auroc']:<9.4f} {m['auprc']:<9.4f} "
                  f"{m['f1']:<9.4f} {m['accuracy']:<9.4f} {m['mcc']:<9.4f} {m.get('count',len(loader.dataset)):<7}")
        print("-" * 100)

    print("=" * 100)

    # ── Save JSON ──────────────────────────────────────────────────
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJ_ROOT / output_path
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # ── Save TSV ───────────────────────────────────────────────────
    tsv_path = Path(args.output_tsv)
    if not tsv_path.is_absolute():
        tsv_path = PROJ_ROOT / tsv_path
    with open(tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["group", "mode", "auroc", "auprc", "f1", "accuracy", "mcc"])
        for grp in GROUPS:
            if grp not in all_results:
                continue
            for mode_key, mode_label in [("Mode_A", "Genome+GTF"), ("Mode_B", "+GeneExp")]:
                m = all_results[grp][mode_key]
                writer.writerow([grp, mode_label,
                                f"{m['auroc']:.4f}", f"{m['auprc']:.4f}",
                                f"{m['f1']:.4f}", f"{m['accuracy']:.4f}",
                                f"{m['mcc']:.4f}"])
    logger.info(f"Results saved to {tsv_path}")


if __name__ == "__main__":
    main()
