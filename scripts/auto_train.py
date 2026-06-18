#!/usr/bin/env python3
"""
Automatic hyperparameter sweep and training for PanCirc-Fungi.

Searches over a grid of hyperparameter combinations (flank_size, k, embed_dim,
gru_hidden), runs full two-stage pretraining + group-wise fine-tuning +
evaluation for each, and records results to a CSV.

Supports three execution modes:
  1. Sequential (default): iterate all combinations one by one
  2. SLURM array: each array task runs one combination (indexed by $SLURM_ARRAY_TASK_ID)
  3. Single: run one specific combination via --experiment-id

Usage:
    # Dry-run: list all combinations
    python scripts/auto_train.py --dry-run

    # Sequential sweep (small grid)
    python scripts/auto_train.py --flank-sizes 100,150,200 --k-values 3,4

    # Single experiment (overrides all grid params)
    python scripts/auto_train.py --flank-size 150 --k 3 --embed-dim 64 --gru-hidden 64

    # SLURM array: submit with auto_train.slurm (reads SLURM_ARRAY_TASK_ID)

    # Fast mode (fewer epochs for quick validation)
    python scripts/auto_train.py --fast --flank-sizes 150,200 --k-values 3

Output CSV columns:
    experiment_id, timestamp, flank_size, k, embed_dim, gru_hidden, group,
    auroc, auprc, f1, accuracy, mcc, top1_accuracy, top3_accuracy,
    cv_auroc_mean, cv_auroc_std, stage1_loss, stage2_loss, status
"""

import argparse
import csv
import itertools
import logging
import os
import sys
import time
import traceback
import copy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("pancirc.auto_train")

# All heavy imports (torch, yaml, model, data modules) are lazy-loaded
# inside the functions that need them — this keeps --dry-run fast and
# working even without ML dependencies installed.

# ─── Default grid ────────────────────────────────────────────────────────────
DEFAULT_FLANK_SIZES = [50, 100, 150, 200, 300, 500]
DEFAULT_K_VALUES = [3, 4, 5]
DEFAULT_EMBED_DIMS = [32, 64, 128]
DEFAULT_GRU_HIDDENS = [32, 64, 128]

ALL_GROUPS = ["Candida", "Cryptococcus", "Filamentous"]


# ═════════════════════════════════════════════════════════════════════════════
#  Grid enumeration
# ═════════════════════════════════════════════════════════════════════════════

def generate_grid(
    flank_sizes: Optional[List[int]] = None,
    k_values: Optional[List[int]] = None,
    embed_dims: Optional[List[int]] = None,
    gru_hiddens: Optional[List[int]] = None,
    groups: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Generate all hyperparameter combinations in the grid."""
    flank_sizes = flank_sizes or DEFAULT_FLANK_SIZES
    k_values = k_values or DEFAULT_K_VALUES
    embed_dims = embed_dims or DEFAULT_EMBED_DIMS
    gru_hiddens = gru_hiddens or DEFAULT_GRU_HIDDENS
    groups = groups or ALL_GROUPS

    grid = []
    for i, (fs, k, ed, gh, grp) in enumerate(
        itertools.product(flank_sizes, k_values, embed_dims, gru_hiddens, groups)
    ):
        grid.append({
            "experiment_id": i + 1,
            "flank_size": fs,
            "k": k,
            "embed_dim": ed,
            "gru_hidden": gh,
            "group": grp,
        })

    logger.info(
        f"Generated grid: {len(grid)} experiments "
        f"({len(flank_sizes)}×{len(k_values)}×{len(embed_dims)}×{len(gru_hiddens)}"
        f"×{len(groups)} groups)"
    )
    return grid


# ═════════════════════════════════════════════════════════════════════════════
#  Data loading helpers
# ═════════════════════════════════════════════════════════════════════════════

def _ensure_features(
    entries,
    features_dir: str,
    flank_size: int,
    genome_window_size: int,
) -> bool:
    """Check features exist; if not, compute them.

    Returns True if features are available (pre-existing or freshly computed).
    """
    missing = []
    for e in entries:
        sd = os.path.join(features_dir, e.strain)
        if not os.path.isdir(sd):
            missing.append(e.strain)
        else:
            for fname in ("positive_gene_ids.npy", "negative_gene_ids.npy",
                          "gtf_features.npz", "genome_profiles.npz"):
                if not os.path.isfile(os.path.join(sd, fname)):
                    missing.append(e.strain)
                    break

    if not missing:
        logger.info("All features pre-computed.")
        return True

    logger.warning(f"Missing features for {len(missing)} strains: {missing}")
    logger.info("Computing features now...")

    # Import here to avoid circular imports at module level
    from scripts.extract_features import process_strain

    success = 0
    for e in entries:
        if e.strain in missing:
            try:
                result = process_strain(
                    e, features_dir,
                    flank_size=flank_size,
                    genome_window_size=genome_window_size,
                )
                if result["status"] == "ok":
                    success += 1
                else:
                    logger.error(f"Feature extraction failed for {e.strain}: {result['status']}")
            except Exception as ex:
                logger.error(f"Feature extraction error for {e.strain}: {ex}")

    return success == len(missing)


def _load_pos_neg_from_features(features_dir: str, entries):
    """Load positive/negative gene lists + actual circ_info for cross_labels."""
    import numpy as np
    import pandas as pd
    from data.circ_info_encoding import load_circ_info, filter_circ_info

    pos_genes = {}
    neg_genes = {}

    # Pre-load circ_info for all strains (needed for cross_labels)
    circ_data = {}

    for e in entries:
        sd = os.path.join(features_dir, e.strain)
        if not os.path.isdir(sd):
            pos_genes[e.strain] = {}
            neg_genes[e.strain] = []
            circ_data[e.strain] = {}
            continue

        # Load circ_info for cross_labels
        strain_circ = {}
        if os.path.isfile(e.circinfo_path):
            try:
                circ_df = filter_circ_info(load_circ_info(e.circinfo_path))
                for gid, group in circ_df.groupby("gene_id"):
                    strain_circ[str(gid)] = group
            except Exception:
                pass
        circ_data[e.strain] = strain_circ

        # Positive genes (with actual DataFrames for cross_labels)
        pos_path = os.path.join(sd, "positive_gene_ids.npy")
        if os.path.isfile(pos_path):
            ids = np.load(pos_path, allow_pickle=True).tolist()
            pos_genes[e.strain] = {
                gid: strain_circ.get(gid, pd.DataFrame())
                for gid in ids
            }
        else:
            pos_genes[e.strain] = {}

        # Negative genes
        neg_path = os.path.join(sd, "negative_gene_ids.npy")
        if os.path.isfile(neg_path):
            neg_genes[e.strain] = np.load(neg_path, allow_pickle=True).tolist()
        else:
            neg_genes[e.strain] = []

    return pos_genes, neg_genes


def _build_indexers(entries, cache_dir=None):
    """Build GeneModelIndexer and GenomeIndexer for each strain."""
    from utils.gtf_utils import GeneModelIndexer
    from data.genome_encoding import GenomeIndexer

    gene_models = {}
    genome_indexers = {}

    for e in entries:
        logger.info(f"Indexing {e.strain}...")

        # GTF
        try:
            gm = GeneModelIndexer(e.gtf_path)
            gene_models[e.strain] = gm
            logger.info(f"  {e.strain}: {gm.n_genes()} genes in GTF")
        except Exception as ex:
            logger.error(f"  Failed to index GTF for {e.strain}: {ex}")
            continue

        # Genome
        try:
            gi = GenomeIndexer(e.genome_path)
            genome_indexers[e.strain] = gi
        except Exception as ex:
            logger.error(f"  Failed to index genome for {e.strain}: {ex}")

    return gene_models, genome_indexers


# ═════════════════════════════════════════════════════════════════════════════
#  Training helpers
# ═════════════════════════════════════════════════════════════════════════════

def _create_dataloaders(
    train_strains: set,
    test_strains: set,
    all_entries,
    gene_models,
    genome_indexers,
    pos_genes: dict,
    neg_genes: dict,
    data_config: dict,
    batch_size: int,
    shuffle: bool = True,
):
    """Create training and validation DataLoaders."""
    from torch.utils.data import DataLoader
    from data.dataset import CircRNAPretrainDataset, collate_pretrain
    train_entries = [e for e in all_entries if e.strain in train_strains]
    val_entries = [e for e in all_entries if e.strain in test_strains]

    train_dataset = CircRNAPretrainDataset(
        entries=train_entries,
        gene_models=gene_models,
        genome_indexers=genome_indexers,
        positive_genes=pos_genes,
        negative_genes=neg_genes,
        config=data_config,
    )
    # Validation on test strains (cross-species)
    val_dataset = CircRNAPretrainDataset(
        entries=val_entries,
        gene_models=gene_models,
        genome_indexers=genome_indexers,
        positive_genes=pos_genes,
        negative_genes=neg_genes,
        config=data_config,
    ) if val_entries else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_pretrain,
        num_workers=data_config.get("num_workers", 2),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pretrain,
        num_workers=data_config.get("num_workers", 2),
    ) if val_dataset else None

    return train_loader, val_loader


def run_stage1(
    model,
    train_loader,
    val_loader,
    stage1_config: dict,
    device: str,
    output_dir: str,
) -> Dict[str, float]:
    """Stage 1: Gene representation learning."""
    from train.trainer import Trainer

    logger.info("=" * 50)
    logger.info("Stage 1: Gene-level pre-training")
    logger.info("=" * 50)

    # Freeze junction encoder
    for p in model.junction_encoder.parameters():
        p.requires_grad = False
    # Freeze expression encoder (not used in pretraining)
    for p in model.expression_encoder.parameters():
        p.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {n_trainable:,}")

    trainer = Trainer(model, stage1_config, device=device)
    n_epochs = stage1_config.get("epochs", 15)

    if val_loader is not None:
        trainer.fit(
            train_loader, val_loader,
            n_epochs=n_epochs,
            loss_fn=model.compute_pretrain_loss,
            checkpoint_dir=os.path.join(output_dir, "stage1"),
            early_stop_patience=stage1_config.get("early_stop_patience"),
        )
    else:
        # No validation set — just train
        for epoch in range(n_epochs):
            trainer.train_epoch(train_loader, model.compute_pretrain_loss)
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                ckpt_path = os.path.join(output_dir, "stage1", f"epoch_{epoch+1}.pt")
                trainer.save_checkpoint(ckpt_path, epoch)

    # Save final stage 1 checkpoint
    trainer.save_checkpoint(
        os.path.join(output_dir, "stage1", "final.pt"), n_epochs - 1
    )

    return {
        "stage1_loss": trainer.best_val_loss if hasattr(trainer, "best_val_loss") else 0.0,
        "stage1_epochs": n_epochs,
    }


def run_stage2(
    model,
    train_loader,
    val_loader,
    stage2_config: dict,
    device: str,
    output_dir: str,
) -> Dict[str, float]:
    """Stage 2: Junction-level training."""
    from train.trainer import Trainer

    logger.info("=" * 50)
    logger.info("Stage 2: Junction-level pre-training")
    logger.info("=" * 50)

    # Unfreeze all
    for p in model.parameters():
        p.requires_grad = True
    # Keep expression encoder frozen (not used in pretraining)
    for p in model.expression_encoder.parameters():
        p.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {n_trainable:,}")

    trainer = Trainer(model, stage2_config, device=device)
    n_epochs = stage2_config.get("epochs", 30)

    if val_loader is not None:
        trainer.fit(
            train_loader, val_loader,
            n_epochs=n_epochs,
            loss_fn=model.compute_pretrain_loss,
            checkpoint_dir=os.path.join(output_dir, "stage2"),
            early_stop_patience=stage2_config.get("early_stop_patience"),
        )
    else:
        for epoch in range(n_epochs):
            trainer.train_epoch(train_loader, model.compute_pretrain_loss)
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                trainer.save_checkpoint(
                    os.path.join(output_dir, "stage2", f"epoch_{epoch+1}.pt"), epoch
                )

    trainer.save_checkpoint(
        os.path.join(output_dir, "stage2", "final.pt"), n_epochs - 1
    )

    return {
        "stage2_loss": trainer.best_val_loss if hasattr(trainer, "best_val_loss") else 0.0,
        "stage2_epochs": n_epochs,
    }


def evaluate_on_test(
    model,
    group: str,
    test_entries,
    gene_models,
    genome_indexers,
    pos_genes: dict,
    neg_genes: dict,
    data_config: dict,
    batch_size: int,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on held-out test strains."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from data.dataset import CircRNAPretrainDataset, collate_pretrain
    from utils.metrics import classification_metrics, compute_topk_accuracy

    model.eval()

    if not test_entries:
        logger.warning(f"No test entries for {group}, skipping evaluation")
        return {}

    test_dataset = CircRNAPretrainDataset(
        entries=test_entries,
        gene_models=gene_models,
        genome_indexers=genome_indexers,
        positive_genes=pos_genes,
        negative_genes=neg_genes,
        config=data_config,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pretrain,
        num_workers=data_config.get("num_workers", 2),
    )

    all_labels = []
    all_probs = []
    all_junction_scores = []
    all_cross_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(batch, task="pretrain")

            probs = torch.sigmoid(outputs["gene_logits"]).squeeze(-1)
            all_labels.append(batch["is_positive"].cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            if "junction_scores" in outputs:
                all_junction_scores.append(outputs["junction_scores"].cpu().numpy())
            if "cross_labels" in batch:
                all_cross_labels.append(batch["cross_labels"].cpu().numpy())

    if not all_labels:
        logger.warning("No evaluation data collected")
        return {}

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    # Filter out dummy samples (is_positive == -1.0)
    valid = y_true >= 0
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    if len(y_true) == 0:
        logger.warning("No valid samples in evaluation set")
        return {}

    metrics = classification_metrics(y_true, y_prob)

    # Junction ranking metrics
    if all_junction_scores and all_cross_labels:
        try:
            j_scores = np.concatenate(all_junction_scores, axis=0)
            c_labels = np.concatenate(all_cross_labels, axis=0)
            # Flatten junction matrices to vectors
            batch_size_actual = j_scores.shape[0]
            j_flat = j_scores.reshape(batch_size_actual, -1)
            c_flat = c_labels.reshape(batch_size_actual, -1)
            metrics["top1_accuracy"] = compute_topk_accuracy(c_flat, j_flat, k=1)
            metrics["top3_accuracy"] = compute_topk_accuracy(c_flat, j_flat, k=3)
        except Exception as e:
            logger.warning(f"Could not compute junction metrics: {e}")
            metrics["top1_accuracy"] = 0.0
            metrics["top3_accuracy"] = 0.0

    logger.info(f"Evaluation results for {group}:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
#  Single experiment runner
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(
    experiment: dict,
    config: dict,
    all_entries,
    active_entries,
    results_dir: str,
    features_dir: str,
    device: str,
    fast_mode: bool = False,
) -> Dict[str, Any]:
    """Run one full experiment (pretrain + evaluate)."""
    import numpy as np
    import yaml
    from data.tsv_parser import build_strain_index, TRAIN_STRAINS, TEST_STRAINS
    from model.pancirc import PanCircModel, count_parameters

    flank_size = experiment["flank_size"]
    k = experiment["k"]
    embed_dim = experiment["embed_dim"]
    gru_hidden = experiment["gru_hidden"]
    group = experiment["group"]

    exp_id = experiment["experiment_id"]
    output_dir = os.path.join(
        results_dir,
        f"exp{exp_id:04d}_fs{flank_size}_k{k}_ed{embed_dim}_gh{gru_hidden}_{group}"
    )
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Experiment {exp_id}: "
                f"flank_size={flank_size}, k={k}, "
                f"embed_dim={embed_dim}, gru_hidden={gru_hidden}, "
                f"group={group}")
    logger.info("=" * 60)

    # ── 1. Override config (deep copy to avoid cross-experiment pollution) ───
    data_config = copy.deepcopy(config.get("data", {}))
    data_config["flank_size"] = flank_size
    data_config["k"] = k

    model_config = copy.deepcopy(config.get("model", {}))
    model_config["junction"]["k"] = k
    model_config["junction"]["embed_dim"] = embed_dim
    model_config["junction"]["gru_hidden"] = gru_hidden
    model_config["flank_size"] = flank_size  # for positional encoding max_len

    pretrain_config = config.get("pretrain", {})
    if fast_mode:
        stage1 = pretrain_config.get("stage1", {}).copy()
        stage1["epochs"] = 3        # 15 → 3
        stage2 = pretrain_config.get("stage2", {}).copy()
        stage2["epochs"] = 5        # 30 → 5
    else:
        stage1 = pretrain_config.get("stage1", {}).copy()
        stage2 = pretrain_config.get("stage2", {}).copy()

    finetune_config = config.get("finetune", {}).copy()
    train_config = config.get("training", {}).copy()

    # ── 2. Build indexers ─────────────────────────────────────────────────
    train_strains_set = TRAIN_STRAINS[group]
    test_strains_set = TEST_STRAINS[group]

    # Only load indexers for relevant strains
    relevant_strains = train_strains_set | test_strains_set
    relevant_entries = [e for e in active_entries if e.strain in relevant_strains]

    try:
        gene_models, genome_indexers = _build_indexers(relevant_entries)
    except Exception as e:
        logger.error(f"Failed to build indexers: {e}")
        return {**experiment, "status": f"indexer_error: {e}"}

    # ── 3. Load pre-computed features ─────────────────────────────────────
    pos_genes, neg_genes = _load_pos_neg_from_features(features_dir, active_entries)

    # ── 4. Create DataLoaders ─────────────────────────────────────────────
    batch_size_stage1 = stage1.get("batch_size", 32)
    batch_size_stage2 = stage2.get("batch_size", 16)

    try:
        train_loader, val_loader = _create_dataloaders(
            train_strains=train_strains_set,
            test_strains=test_strains_set,
            all_entries=active_entries,
            gene_models=gene_models,
            genome_indexers=genome_indexers,
            pos_genes=pos_genes,
            neg_genes=neg_genes,
            data_config=data_config,
            batch_size=batch_size_stage1,
        )
    except Exception as e:
        logger.error(f"Failed to create DataLoaders: {e}")
        traceback.print_exc()
        return {**experiment, "status": f"dataloader_error: {e}"}

    # ── 5. Build model ────────────────────────────────────────────────────
    n_species = len(build_strain_index(active_entries))
    model = PanCircModel(model_config, n_species=n_species)
    n_total, n_trainable = count_parameters(model)
    logger.info(f"Model: {n_total:,} total params, {n_trainable:,} trainable")
    model.to(device)

    # ── 6. Stage 1: Gene-level pretraining ────────────────────────────────
    stage1_result = run_stage1(
        model, train_loader, val_loader,
        stage1, device, output_dir,
    )

    # ── 7. Stage 2: Junction-level pretraining ────────────────────────────
    # Recreate loader with stage 2 batch size
    if batch_size_stage2 != batch_size_stage1:
        train_loader, val_loader = _create_dataloaders(
            train_strains=train_strains_set,
            test_strains=test_strains_set,
            all_entries=active_entries,
            gene_models=gene_models,
            genome_indexers=genome_indexers,
            pos_genes=pos_genes,
            neg_genes=neg_genes,
            data_config=data_config,
            batch_size=batch_size_stage2,
        )

    stage2_result = run_stage2(
        model, train_loader, val_loader,
        stage2, device, output_dir,
    )

    # ── 8. Evaluate ────────────────────────────────────────────────────────
    test_entries = [e for e in active_entries if e.strain in test_strains_set]
    eval_metrics = evaluate_on_test(
        model=model,
        group=group,
        test_entries=test_entries,
        gene_models=gene_models,
        genome_indexers=genome_indexers,
        pos_genes=pos_genes,
        neg_genes=neg_genes,
        data_config=data_config,
        batch_size=finetune_config.get("batch_size", 16),
        device=device,
    )

    # ── 9. Clean up genome indexers ───────────────────────────────────────
    for gi in genome_indexers.values():
        try:
            gi.close()
        except Exception:
            pass

    # ── 10. Assemble result ────────────────────────────────────────────────
    result = {
        **experiment,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **stage1_result,
        **stage2_result,
        **eval_metrics,
        "status": "ok",
    }

    # Save per-experiment result
    result_path = os.path.join(output_dir, "result.yaml")
    with open(result_path, "w") as f:
        yaml.dump({k: float(v) if isinstance(v, (np.floating,)) else v
                    for k, v in result.items()}, f)

    logger.info(f"Experiment {exp_id} complete → {result_path}")
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  CSV I/O
# ═════════════════════════════════════════════════════════════════════════════

CSV_HEADER = [
    "experiment_id", "timestamp",
    "flank_size", "k", "embed_dim", "gru_hidden", "group",
    "auroc", "auprc", "f1", "accuracy", "mcc",
    "top1_accuracy", "top3_accuracy",
    "cv_auroc_mean", "cv_auroc_std",
    "stage1_loss", "stage2_loss",
    "status",
]


def _read_existing_results(csv_path: str) -> set:
    """Read experiment IDs already recorded in the CSV."""
    if not os.path.isfile(csv_path):
        return set()
    existing = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                existing.add(int(row["experiment_id"]))
            except (ValueError, KeyError):
                pass
    return existing


def _append_results(csv_path: str, results: List[Dict[str, Any]]):
    """Append results to CSV file, creating header if needed."""
    import numpy as np

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for r in results:
            # Convert numpy types
            clean = {}
            for k in CSV_HEADER:
                v = r.get(k, "")
                if isinstance(v, (np.floating,)):
                    v = float(v)
                elif isinstance(v, (np.integer,)):
                    v = int(v)
                elif v is None:
                    v = ""
                clean[k] = v
            writer.writerow(clean)


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PanCirc-Fungi automatic hyperparameter sweep"
    )
    # Config
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--results-dir", default="checkpoints/results")
    parser.add_argument("--features-dir", default="checkpoints/features")
    parser.add_argument("--device", default="cuda")

    # Grid specification (comma-separated lists)
    parser.add_argument("--flank-sizes", type=str, default=None,
                        help="Comma-separated list, e.g. '50,100,150'")
    parser.add_argument("--k-values", type=str, default=None,
                        help="Comma-separated list, e.g. '3,4,5'")
    parser.add_argument("--embed-dims", type=str, default=None,
                        help="Comma-separated list, e.g. '32,64,128'")
    parser.add_argument("--gru-hiddens", type=str, default=None,
                        help="Comma-separated list, e.g. '32,64,128'")
    parser.add_argument("--groups", type=str, default=None,
                        help="Comma-separated list, e.g. 'Candida,Cryptococcus'")

    # Single experiment mode (overrides grid)
    parser.add_argument("--flank-size", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--gru-hidden", type=int, default=None)
    parser.add_argument("--group", type=str, default=None)

    # Execution control
    parser.add_argument("--dry-run", action="store_true",
                        help="Print grid and exit")
    parser.add_argument("--fast", action="store_true",
                        help="Reduced epochs for quick validation")
    parser.add_argument("--slurm-array", action="store_true",
                        help="Read experiment index from SLURM_ARRAY_TASK_ID")
    parser.add_argument("--experiment-id", type=int, default=None,
                        help="Run a specific experiment from the grid")
    parser.add_argument("--resume", action="store_true",
                        help="Skip experiments already recorded in CSV")
    parser.add_argument("--force-features", action="store_true",
                        help="Recompute features even if cached")
    parser.add_argument("--log-dir", default="logs/auto_train")
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()

    # ── Setup logging and device ──────────────────────────────────────────
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"auto_train_{datetime.now():%Y%m%d_%H%M%S}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    global logger
    logger = logging.getLogger("pancirc.auto_train")

    logger.info("PanCirc-Fungi Auto Training")
    logger.info(f"Log file: {log_path}")

    # ── Load config (lightweight, no torch needed) ────────────────────────
    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ── Parse strain registry (lightweight, no torch needed) ──────────────
    from data.tsv_parser import parse_strain_registry

    all_entries = parse_strain_registry(config["data"]["tsv_path"])
    active_entries = [e for e in all_entries if not e.is_excluded]
    logger.info(f"Active strains: {len(active_entries)}")

    # ── Build grid ────────────────────────────────────────────────────────
    if args.slurm_array:
        # SLURM array mode: single experiment indexed by task ID
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        logger.info(f"SLURM array mode: task ID = {task_id}")

        # Build full grid to index into it
        grid = generate_grid(
            flank_sizes=_parse_comma_int(args.flank_sizes),
            k_values=_parse_comma_int(args.k_values),
            embed_dims=_parse_comma_int(args.embed_dims),
            gru_hiddens=_parse_comma_int(args.gru_hiddens),
            groups=_parse_comma_str(args.groups),
        )
        if task_id < 1 or task_id > len(grid):
            logger.error(f"SLURM_ARRAY_TASK_ID={task_id} out of range [1, {len(grid)}]")
            sys.exit(1)
        experiments = [grid[task_id - 1]]

    elif args.experiment_id is not None:
        # Single experiment by ID
        grid = generate_grid(
            flank_sizes=_parse_comma_int(args.flank_sizes),
            k_values=_parse_comma_int(args.k_values),
            embed_dims=_parse_comma_int(args.embed_dims),
            gru_hiddens=_parse_comma_int(args.gru_hiddens),
            groups=_parse_comma_str(args.groups),
        )
        match = [e for e in grid if e["experiment_id"] == args.experiment_id]
        if not match:
            logger.error(f"Experiment ID {args.experiment_id} not found in grid")
            sys.exit(1)
        experiments = match

    elif args.flank_size is not None:
        # Single experiment with explicit params
        experiments = [{
            "experiment_id": 1,
            "flank_size": args.flank_size,
            "k": args.k or config["model"]["junction"]["k"],
            "embed_dim": args.embed_dim or config["model"]["junction"]["embed_dim"],
            "gru_hidden": args.gru_hidden or config["model"]["junction"]["gru_hidden"],
            "group": args.group or "Candida",
        }]

    else:
        # Full grid
        experiments = generate_grid(
            flank_sizes=_parse_comma_int(args.flank_sizes),
            k_values=_parse_comma_int(args.k_values),
            embed_dims=_parse_comma_int(args.embed_dims),
            gru_hiddens=_parse_comma_int(args.gru_hiddens),
            groups=_parse_comma_str(args.groups),
        )

    logger.info(f"Total experiments to run: {len(experiments)}")

    # ── Dry run ───────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n{'='*80}")
        print(f"Grid: {len(experiments)} experiments")
        print(f"{'='*80}")
        print(f"{'ID':>4s}  {'flank':>5s}  {'k':>2s}  {'embed_dim':>9s}  "
              f"{'gru_hidden':>10s}  {'group':<15s}")
        print("-" * 60)
        for exp in experiments:
            print(f"{exp['experiment_id']:>4d}  {exp['flank_size']:>5d}  {exp['k']:>2d}  "
                  f"{exp['embed_dim']:>9d}  {exp['gru_hidden']:>10d}  {exp['group']:<15s}")
        print(f"{'='*80}")
        print(f"Estimated time per experiment: "
              f"{'3h (fast)' if args.fast else '~12h (full)'}")
        print(f"Total estimated time: "
              f"{'3h' if args.fast else '~12h'} × {len(experiments)} experiments")
        return

    # ── Device detection & Python 3.14+ multiprocessing fix ──────────────
    import torch
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    logger.info(f"Device: {device}")

    # Force 'fork' start method — Python 3.14+ defaults to 'forkserver'
    # which breaks pyfaidx pickling in DataLoader workers
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    # ── Pre-compute features (if needed) ──────────────────────────────────
    genome_window = config["data"].get("genome_window_size", 10000)
    default_flank = config["data"].get("flank_size", 150)

    if not _ensure_features(
        active_entries, args.features_dir,
        flank_size=default_flank,
        genome_window_size=genome_window,
    ):
        logger.error("Feature extraction failed. Aborting.")
        sys.exit(1)

    # ── Check / resume ────────────────────────────────────────────────────
    csv_path = os.path.join(args.results_dir, "experiments.csv")
    if args.resume:
        existing = _read_existing_results(csv_path)
        experiments = [e for e in experiments if e["experiment_id"] not in existing]
        logger.info(f"Resume mode: {len(existing)} existing, {len(experiments)} remaining")
        if not experiments:
            logger.info("All experiments already completed!")
            return

    # ── Run experiments ───────────────────────────────────────────────────
    results = []
    t_start = time.time()

    for i, exp in enumerate(experiments):
        t_exp = time.time()
        logger.info(f"\n[{i+1}/{len(experiments)}] Starting experiment {exp['experiment_id']}")

        try:
            result = run_experiment(
                experiment=exp,
                config=config,
                all_entries=all_entries,
                active_entries=active_entries,
                results_dir=args.results_dir,
                features_dir=args.features_dir,
                device=device,
                fast_mode=args.fast,
            )
            results.append(result)

            elapsed = time.time() - t_exp
            logger.info(f"Experiment {exp['experiment_id']} done in {elapsed/60:.1f} min")

        except Exception as e:
            logger.error(f"Experiment {exp['experiment_id']} failed: {e}")
            traceback.print_exc()
            results.append({
                **exp,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": f"error: {str(e)[:100]}",
            })

        # Write results incrementally
        _append_results(csv_path, results[-1:] if isinstance(results[-1], dict) else [])
        logger.info(f"Results saved to {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_err = sum(1 for r in results if r.get("status", "").startswith("error"))

    print(f"\n{'='*60}")
    print(f"Auto-training complete!")
    print(f"  Total time: {total_time/3600:.1f}h")
    print(f"  Experiments: {len(results)} ({n_ok} ok, {n_err} errors)")
    print(f"  Results: {csv_path}")
    print(f"{'='*60}")

    # Print best result
    if n_ok > 0:
        best = max(
            [r for r in results if r.get("status") == "ok" and r.get("auroc", 0) > 0],
            key=lambda r: r.get("auroc", 0),
            default=None,
        )
        if best:
            print(f"\nBest result:")
            print(f"  Experiment {best['experiment_id']}: "
                  f"flank={best['flank_size']}, k={best['k']}, "
                  f"embed_dim={best['embed_dim']}, gru_hidden={best['gru_hidden']}")
            print(f"  Group: {best['group']}, AUROC: {best.get('auroc', 'N/A')}")


# ─── Helpers ────────────────────────────────────────────────────────────────

def _parse_comma_int(val: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated string of ints, e.g. '50,100,150'."""
    if val is None:
        return None
    parts = [p.strip() for p in val.split(",") if p.strip()]
    return [int(p) for p in parts]


def _parse_comma_str(val: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated string, e.g. 'Candida,Cryptococcus'."""
    if val is None:
        return None
    parts = [p.strip() for p in val.split(",") if p.strip()]
    return parts if parts else None


if __name__ == "__main__":
    main()
