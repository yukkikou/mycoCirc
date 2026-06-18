#!/usr/bin/env python3
"""
Fine-tuning for PanCirc-Fungi.

Each group is fine-tuned independently using its own CircExp/GeneExp
as additional input features, with its held-out species as test set.

Groups:
  Candida:     train P1,P2,P3,P5,P6,S8 → test P4 (C. auris)
  Cryptococcus: train C1,C2,C3,C5,C6,C7 → test C4 (C. neoformans var neoformans)
  Filamentous:  train F3,F4,N1,A1,A3   → test F6 (F. venenatum)

5-fold CV by strain:
  Each fold: train on N-1 strains, validate on 1 held-out strain.
  Reports cv_auroc_mean ± cv_auroc_std.
  Best fold model → evaluate on held-out test strain.

Usage:
    python train/finetune.py --group Candida \\
        --pretrained checkpoints/pretrain/best.pt \\
        --config config/default.yaml

    # From scratch (no pretrained weights):
    python train/finetune.py --group Candida --from-scratch \\
        --config config/default.yaml
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.multiprocessing as mp

from data.dataset import CircRNAFinetuneDataset, collate_pretrain
from data.tsv_parser import (
    parse_strain_registry,
    build_strain_index,
    TRAIN_STRAINS,
    TEST_STRAINS,
)
from data.expression_encoding import (
    load_expression_csv,
    encode_expression_values,
    align_circ_to_gene_expression,
    pad_to_max_replicates,
)
from data.circ_info_encoding import load_circ_info, filter_circ_info
from model.pancirc import PanCircModel, count_parameters
from train.trainer import Trainer
from utils.logging import setup_logging
from utils.metrics import classification_metrics

logger = logging.getLogger("pancirc.finetune")


# ─── Helpers ────────────────────────────────────────────────────────────────


def _build_indexers(entries):
    """Build GeneModelIndexer and GenomeIndexer for each strain."""
    from utils.gtf_utils import GeneModelIndexer
    from data.genome_encoding import GenomeIndexer

    gene_models = {}
    genome_indexers = {}
    for e in entries:
        logger.info(f"  Indexing {e.strain}...")
        try:
            gm = GeneModelIndexer(e.gtf_path)
            gene_models[e.strain] = gm
            logger.info(f"    {e.strain}: {gm.n_genes()} genes")
        except Exception as ex:
            logger.error(f"    GTF failed for {e.strain}: {ex}")
            continue
        try:
            gi = GenomeIndexer(e.genome_path)
            genome_indexers[e.strain] = gi
        except Exception as ex:
            logger.error(f"    Genome failed for {e.strain}: {ex}")
    return gene_models, genome_indexers


def _load_features(features_dir, entries):
    """Load positive/negative gene lists with real circ_info DataFrames."""
    import pandas as pd

    pos_genes = {}
    neg_genes = {}
    for e in entries:
        sd = os.path.join(features_dir, e.strain)
        if not os.path.isdir(sd):
            pos_genes[e.strain] = {}
            neg_genes[e.strain] = []
            continue
        strain_circ = {}
        if os.path.isfile(e.circinfo_path):
            try:
                circ_df = filter_circ_info(load_circ_info(e.circinfo_path))
                for gid, group in circ_df.groupby("gene_id"):
                    strain_circ[str(gid)] = group
            except Exception:
                pass
        pos_path = os.path.join(sd, "positive_gene_ids.npy")
        if os.path.isfile(pos_path):
            ids = np.load(pos_path, allow_pickle=True).tolist()
            pos_genes[e.strain] = {gid: strain_circ.get(gid, pd.DataFrame())
                                   for gid in ids}
        else:
            pos_genes[e.strain] = {}
        neg_path = os.path.join(sd, "negative_gene_ids.npy")
        if os.path.isfile(neg_path):
            neg_genes[e.strain] = np.load(neg_path, allow_pickle=True).tolist()
        else:
            neg_genes[e.strain] = []
    return pos_genes, neg_genes


def _load_expression_data(entries, max_replicates=3):
    """Load and normalize CircExp + GeneExp for each strain.

    CircExp is available during training but NOT during inference
    (you don't know circRNA expression before predicting it).

    Returns dict:
        {strain: {"gene_exp": {gene_id: np.ndarray}, "aligned": {gene_id: {...}}}}
    """
    expression_data = {}
    for e in entries:
        circ_exp_raw = load_expression_csv(e.circexp_path)
        gene_exp_raw = load_expression_csv(e.geneexp_path)

        gene_exp_map = {}
        if gene_exp_raw is not None:
            ge_vals, ge_ids, _ = encode_expression_values(
                gene_exp_raw, log1p=True, zscore=True
            )
            for i, gid in enumerate(ge_ids):
                gene_exp_map[gid] = pad_to_max_replicates(
                    ge_vals[i], max_replicates
                )

        aligned = {}
        if circ_exp_raw is not None and os.path.isfile(e.circinfo_path):
            circ_info = filter_circ_info(load_circ_info(e.circinfo_path))
            if circ_info is not None and not circ_info.empty:
                aligned = align_circ_to_gene_expression(
                    circ_info, circ_exp_raw, gene_exp_raw
                )

        expression_data[e.strain] = {"gene_exp": gene_exp_map, "aligned": aligned}
        logger.info(
            f"  {e.strain}: {len(gene_exp_map)} genes, "
            f"{len(aligned)} positive w/ aligned circ expression"
        )
    return expression_data


def _make_model(config, n_species, pretrained_path, device, strategy="full"):
    """Create a fresh model, optionally load pretrained weights, apply freeze strategy."""
    model = PanCircModel(config["model"], n_species=n_species)
    if pretrained_path and os.path.isfile(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info(f"  Loaded pretrained weights — missing={len(missing)} unexpected={len(unexpected)}")
    else:
        logger.info("  No pretrained weights loaded (training from scratch)")

    # Enable expression encoder
    for p in model.expression_encoder.parameters():
        p.requires_grad = True

    # Apply unfreeze strategy
    if strategy in ("linear_probe", "gradual"):
        for name, p in model.named_parameters():
            if "head" not in name and "expression" not in name:
                p.requires_grad = False

    model.to(device)
    _, n_train = count_parameters(model)
    logger.info(f"  Strategy: {strategy} ({n_train:,} trainable params)")
    return model


def _make_dataloader(entries, gene_models, genome_indexers, pos_genes,
                     neg_genes, expression_data, config,
                     max_replicates, batch_size, num_workers, shuffle=True):
    """Build a single DataLoader from a list of strain entries."""
    ds = CircRNAFinetuneDataset(
        entries=entries,
        gene_models=gene_models,
        genome_indexers=genome_indexers,
        positive_genes=pos_genes,
        negative_genes=neg_genes,
        expression_data=expression_data,
        config=config,
        max_replicates=max_replicates,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_pretrain, num_workers=num_workers,
        drop_last=shuffle,
    )


def evaluate(model, loader, device, task="finetune"):
    """Evaluate model on a DataLoader, return classification metrics dict."""
    model.eval()
    all_labels = []
    all_probs = []
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
    if valid.any():
        y_true = y_true[valid]
        y_prob = y_prob[valid]
    if len(y_true) == 0:
        return {}
    return classification_metrics(y_true, y_prob)


def evaluate_with_zeroed_circexp(model, loader, device):
    """Evaluate with circ_exp zeroed out (simulating no CircExp input)."""
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch["circ_exp"] = torch.zeros_like(batch["circ_exp"])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(batch, task="finetune")
            probs = torch.sigmoid(outputs["gene_logits"].squeeze(-1))
            all_labels.append(batch["is_positive"].cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    valid = y_true >= 0
    if valid.any():
        y_true = y_true[valid]
        y_prob = y_prob[valid]
    if len(y_true) == 0:
        return {}
    return classification_metrics(y_true, y_prob)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True,
                        choices=["Candida", "Cryptococcus", "Filamentous"])
    parser.add_argument("--pretrained", default=None,
                        help="Path to pre-trained checkpoint (omit for --from-scratch)")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Skip loading pretrained weights (random init)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output-dir", default="checkpoints/finetune")
    parser.add_argument("--log-dir", default="logs/finetune")
    parser.add_argument("--features-dir", default="checkpoints/features")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--unfreeze-strategy", default="full",
                        choices=["linear_probe", "gradual", "full"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if not args.from_scratch and not args.pretrained:
        parser.error("Either --pretrained or --from-scratch must be specified")
    if args.from_scratch and args.pretrained:
        parser.error("--pretrained and --from-scratch are mutually exclusive")

    # ── Config & logging ───────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = os.path.join(args.output_dir, args.group)
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(args.log_dir, args.group)
    os.makedirs(log_dir, exist_ok=True)
    logger_r = setup_logging(log_dir, name=f"finetune_{args.group}")

    logger_r.info("=" * 50)
    logger_r.info(f"Fine-tuning: {args.group}")
    logger_r.info("=" * 50)

    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    device = args.device if torch.cuda.is_available() else "cpu"
    logger_r.info(f"Device: {device}")

    # ── Strains ────────────────────────────────────────────────────────
    train_strains = sorted(TRAIN_STRAINS[args.group])
    test_strains = sorted(TEST_STRAINS[args.group])

    all_entries = parse_strain_registry(config["data"]["tsv_path"])
    active = [e for e in all_entries if not e.is_excluded]

    train_entries = [e for e in active if e.strain in train_strains]
    test_entries = [e for e in active if e.strain in test_strains]
    all_group_entries = train_entries + test_entries

    n_sp = len(build_strain_index(active))
    data_config = config["data"]
    batch_size = config["finetune"]["batch_size"]
    max_rep = config["model"]["expression"]["max_replicates"]
    ft_config = dict(config["finetune"])
    if args.epochs:
        ft_config["epochs"] = args.epochs
    ft_config["save_interval"] = 9999  # save at end only in CV mode

    logger_r.info(
        f"Train strains ({len(train_entries)}): {[e.strain for e in train_entries]}"
    )
    logger_r.info(
        f"Test strains  ({len(test_entries)}): {[e.strain for e in test_entries]}"
    )

    # ── Shared resources ───────────────────────────────────────────────
    logger_r.info("Building indexers...")
    gene_models, genome_indexers = _build_indexers(all_group_entries)
    if not gene_models:
        logger_r.error("No gene models built. Aborting.")
        sys.exit(1)

    logger_r.info("Loading features...")
    pos_genes, neg_genes = _load_features(args.features_dir, all_group_entries)

    logger_r.info("Loading expression data...")
    expression_data = _load_expression_data(all_group_entries, max_rep)

    # Build per-strain DataLoaders (so folds can recombine them).
    # Skip strains with 0 usable genes (e.g. C3: all circRNAs are
    # intergenic/antisense, filtered out by filter_circ_info).
    strain_loaders = {}
    for e in train_entries:
        ds_test = CircRNAFinetuneDataset(
            entries=[e], gene_models=gene_models,
            genome_indexers=genome_indexers,
            positive_genes=pos_genes, negative_genes=neg_genes,
            expression_data=expression_data, config=data_config,
            max_replicates=max_rep,
        )
        if len(ds_test) == 0:
            logger_r.warning(f"  {e.strain}: 0 usable genes — excluding from CV")
            continue
        strain_loaders[e.strain] = DataLoader(
            ds_test, batch_size=batch_size, shuffle=True,
            collate_fn=collate_pretrain, num_workers=args.num_workers,
            drop_last=True,
        )
    logger_r.info(f"  Usable strains for CV: {list(strain_loaders.keys())}")

    # Test loader (used only at final evaluation)
    test_loader = _make_dataloader(
        entries=test_entries, gene_models=gene_models,
        genome_indexers=genome_indexers,
        pos_genes=pos_genes, neg_genes=neg_genes,
        expression_data=expression_data, config=data_config,
        max_replicates=max_rep, batch_size=batch_size,
        num_workers=args.num_workers, shuffle=False,
    ) if test_entries else None

    # ── Pre-training baseline ──────────────────────────────────────────
    if test_loader:
        logger_r.info("\n--- Baseline (pre-trained, before fine-tuning) ---")
        base_model = _make_model(config, n_sp, args.pretrained, device, args.unfreeze_strategy)
        base_metrics = evaluate(base_model, test_loader, device, task="finetune")
        for k, v in base_metrics.items():
            logger_r.info(f"  {k}: {v:.4f}")
        del base_model

    # ═══════════════════════════════════════════════════════════════════
    #  5-fold Cross-Validation (leave-one-strain-out)
    # ═══════════════════════════════════════════════════════════════════

    n_folds = len(strain_loaders)
    cv_strains = list(strain_loaders.keys())
    logger_r.info(f"\n{'='*50}")
    logger_r.info(f"  {n_folds}-fold CV: leave-one-strain-out ({cv_strains})")
    logger_r.info(f"{'='*50}")

    fold_aurocs = []
    cv_dir = os.path.join(output_dir, "cv")
    os.makedirs(cv_dir, exist_ok=True)
    best_fold_auroc = -1.0
    best_fold_path = ""

    for fold_idx, val_strain in enumerate(cv_strains):
        train_strains_fold = [s for s in cv_strains if s != val_strain]
        logger_r.info(f"\n--- Fold {fold_idx+1}/{n_folds}: val={val_strain}, train={train_strains_fold} ---")

        # Fresh model from pretrained
        model = _make_model(config, n_sp, args.pretrained, device, args.unfreeze_strategy)

        # Build train loader (concat all training strains for this fold)
        train_loader_fold = ConcatDataset(
            [strain_loaders[s].dataset for s in train_strains_fold]
        )
        train_loader_fold = DataLoader(
            train_loader_fold, batch_size=batch_size, shuffle=True,
            collate_fn=collate_pretrain, num_workers=args.num_workers,
            drop_last=True,
        )
        val_loader_fold = strain_loaders[val_strain]

        # Train
        trainer = Trainer(model, ft_config, device=device, task="finetune")
        trainer.fit(
            train_loader_fold, val_loader_fold,
            n_epochs=ft_config["epochs"],
            loss_fn=model.compute_finetune_loss,
            checkpoint_dir=os.path.join(cv_dir, f"fold_{fold_idx+1}"),
            early_stop_patience=ft_config.get("early_stop_patience", 10),
        )

        # Evaluate on validation strain
        val_metrics = evaluate(model, val_loader_fold, device, task="finetune")
        fold_auroc = val_metrics.get("auroc", 0.0)
        fold_aurocs.append(fold_auroc)

        logger_r.info(f"  Fold {fold_idx+1} val AUROC: {fold_auroc:.4f}")

        # Track best fold
        if fold_auroc > best_fold_auroc:
            best_fold_auroc = fold_auroc
            best_fold_path = os.path.join(cv_dir, f"fold_{fold_idx+1}", "best.pt")
            logger_r.info(f"  ← Best fold so far")

        del model

    # ── CV summary ────────────────────────────────────────────────────
    cv_mean = float(np.mean(fold_aurocs))
    cv_std = float(np.std(fold_aurocs))
    logger_r.info(f"\n{'='*50}")
    logger_r.info(f"  CV AUROC: {cv_mean:.4f} ± {cv_std:.4f}")
    logger_r.info(f"  Per-fold: {[f'{v:.4f}' for v in fold_aurocs]}")
    logger_r.info(f"  Best fold AUROC: {best_fold_auroc:.4f}")
    logger_r.info(f"{'='*50}")

    # ═══════════════════════════════════════════════════════════════════
    #  Final evaluation on held-out test strain (best fold model)
    # ═══════════════════════════════════════════════════════════════════

    metrics_baseline = {}
    metrics_genexp = {}
    best_model_state = {}

    if test_loader and os.path.isfile(best_fold_path):
        logger_r.info("\n--- Final evaluation on held-out test strain ---")

        # Load best fold model
        model = _make_model(config, n_sp, args.pretrained, device, args.unfreeze_strategy)
        best_ckpt = torch.load(best_fold_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        best_model_state = best_ckpt["model_state_dict"]
        logger_r.info(f"Loaded best fold checkpoint: {best_fold_path}")

        # Mode A — Genome+GTF only
        metrics_baseline = evaluate(model, test_loader, device, task="pretrain")
        logger_r.info("Mode A — Genome+GTF only:")
        for k, v in metrics_baseline.items():
            logger_r.info(f"  {k}: {v:.4f}")

        # Mode B — Genome+GTF+GeneExp (CircExp=0)
        metrics_genexp = evaluate_with_zeroed_circexp(model, test_loader, device)
        logger_r.info("Mode B — Genome+GTF+GeneExp:")
        for k, v in metrics_genexp.items():
            logger_r.info(f"  {k}: {v:.4f}")

    # ── Save final model ───────────────────────────────────────────────
    final_path = os.path.join(output_dir, "final.pt")
    torch.save({
        "model_state_dict": best_model_state,
        "config": config,
        "group": args.group,
        "train_strains": list(train_strains),
        "test_strains": list(test_strains),
        "cv_auroc_mean": cv_mean,
        "cv_auroc_std": cv_std,
        "fold_aurocs": fold_aurocs,
        "best_fold_path": best_fold_path,
        "metrics_baseline": metrics_baseline,
        "metrics_genexp": metrics_genexp,
    }, final_path)
    logger_r.info(f"\nFinal model saved to {final_path}")
    logger_r.info("Fine-tuning complete!")

    # Cleanup
    for gi in genome_indexers.values():
        try:
            gi.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
