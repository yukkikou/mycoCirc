#!/usr/bin/env python3
"""
Two-stage pre-training for PanCirc-Fungi.

Loads all training strains, builds indexers & DataLoaders,
runs Stage 1 (gene-level, 15 epochs) then Stage 2
(junction-level, 30 epochs), saves checkpoints.

Usage:
    python train/pretrain.py --config config/default.yaml
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from torch.utils.data import DataLoader

from data.dataset import CircRNAPretrainDataset, collate_pretrain
from data.tsv_parser import (
    parse_strain_registry,
    build_strain_index,
    TRAIN_STRAINS,
    TEST_STRAINS,
)
from model.pancirc import PanCircModel, count_parameters
from train.trainer import Trainer
from utils.logging import setup_logging

logger = logging.getLogger("pancirc.pretrain")


# ─── Helpers (lifted from auto_train.py) ──────────────────────────────────────

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
    import numpy as np
    import pandas as pd
    from data.circ_info_encoding import load_circ_info, filter_circ_info

    pos_genes = {}
    neg_genes = {}
    for e in entries:
        sd = os.path.join(features_dir, e.strain)
        if not os.path.isdir(sd):
            pos_genes[e.strain] = {}
            neg_genes[e.strain] = []
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
        # Positive genes
        pos_path = os.path.join(sd, "positive_gene_ids.npy")
        if os.path.isfile(pos_path):
            ids = np.load(pos_path, allow_pickle=True).tolist()
            pos_genes[e.strain] = {gid: strain_circ.get(gid, pd.DataFrame())
                                   for gid in ids}
        else:
            pos_genes[e.strain] = {}
        # Negative genes
        neg_path = os.path.join(sd, "negative_gene_ids.npy")
        if os.path.isfile(neg_path):
            neg_genes[e.strain] = np.load(neg_path, allow_pickle=True).tolist()
        else:
            neg_genes[e.strain] = []
    return pos_genes, neg_genes


def _run_stage(model, loader, val_loader, stage_config, device, output_dir, stage_name):
    """Run one stage of training."""
    from train.trainer import Trainer

    logger.info(f"\n{'='*50}")
    logger.info(f"  {stage_name}")
    logger.info(f"{'='*50}")

    trainer = Trainer(model, stage_config, device=device)
    n_epochs = stage_config.get("epochs", 15)
    ckpt_dir = os.path.join(output_dir, stage_name.lower().replace(" ", "_"))

    if val_loader is not None:
        trainer.fit(loader, val_loader, n_epochs=n_epochs,
                    loss_fn=model.compute_pretrain_loss,
                    checkpoint_dir=ckpt_dir,
                    early_stop_patience=stage_config.get("early_stop_patience"))
    else:
        for epoch in range(n_epochs):
            trainer.train_epoch(loader, model.compute_pretrain_loss)
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                trainer.save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt"), epoch)
    trainer.save_checkpoint(os.path.join(ckpt_dir, "final.pt"), n_epochs - 1)

    best = getattr(trainer, "best_val_loss", 0.0)
    logger.info(f"  ✓ {stage_name} done (best val loss: {best:.4f})")
    return trainer


def main():
    parser = argparse.ArgumentParser(description="PanCirc-Fungi pre-training")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output-dir", default="checkpoints/pretrain")
    parser.add_argument("--log-dir", default="logs/pretrain")
    parser.add_argument("--features-dir", default="checkpoints/features")
    parser.add_argument("--flank-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger_r = setup_logging(args.log_dir, name="pretrain")

    logger_r.info("=" * 50)
    logger_r.info("PanCirc-Fungi: Two-Stage Pre-training")
    logger_r.info("=" * 50)

    if args.flank_size is not None:
        config["data"]["flank_size"] = args.flank_size
    flank_size = config["data"]["flank_size"]

    # ── Python 3.14 multiprocessing fix ───────────────────────────────────
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    device = args.device if torch.cuda.is_available() else "cpu"
    logger_r.info(f"Device: {device}, flank_size: {flank_size}")

    # ── Strain registry ───────────────────────────────────────────────────
    all_entries = parse_strain_registry(config["data"]["tsv_path"])
    active = [e for e in all_entries if not e.is_excluded]

    # Combine all training strains (cross all groups)
    train_strains = set()
    for s in TRAIN_STRAINS.values():
        train_strains |= s
    test_strains = set()
    for s in TEST_STRAINS.values():
        test_strains |= s

    train_entries = [e for e in active if e.strain in train_strains]
    val_entries = [e for e in active if e.strain in test_strains]

    logger_r.info(f"Train strains ({len(train_entries)}): {[e.strain for e in train_entries]}")
    logger_r.info(f"Val strains  ({len(val_entries)}): {[e.strain for e in val_entries]}")

    # ── Build indexers (all relevant strains) ─────────────────────────────
    relevant_strains = train_strains | test_strains
    relevant_entries = [e for e in active if e.strain in relevant_strains]
    logger_r.info("Building indexers...")
    gene_models, genome_indexers = _build_indexers(relevant_entries)
    if not gene_models:
        logger_r.error("No gene models could be built. Aborting.")
        sys.exit(1)

    # ── Load features ─────────────────────────────────────────────────────
    pos_genes, neg_genes = _load_features(args.features_dir, active)

    # ── Create DataLoaders ────────────────────────────────────────────────
    data_config = config["data"]
    batch_size_s1 = config["pretrain"]["stage1"].get("batch_size", 32)
    batch_size_s2 = config["pretrain"]["stage2"].get("batch_size", 16)

    train_dataset = CircRNAPretrainDataset(
        entries=train_entries, gene_models=gene_models,
        genome_indexers=genome_indexers,
        positive_genes=pos_genes, negative_genes=neg_genes,
        config=data_config,
    )
    val_dataset = CircRNAPretrainDataset(
        entries=val_entries, gene_models=gene_models,
        genome_indexers=genome_indexers,
        positive_genes=pos_genes, negative_genes=neg_genes,
        config=data_config,
    ) if val_entries else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_s1, shuffle=True,
        collate_fn=collate_pretrain, num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size_s1, shuffle=False,
        collate_fn=collate_pretrain, num_workers=args.num_workers,
    ) if val_dataset else None

    # ── Build model ───────────────────────────────────────────────────────
    n_species = len(build_strain_index(active))
    model = PanCircModel(config["model"], n_species=n_species)
    n_total, n_trainable = count_parameters(model)
    logger_r.info(f"Model: {n_total:,} total params, {n_trainable:,} trainable")
    model.to(device)

    # ── Stage 1 ───────────────────────────────────────────────────────────
    for p in model.junction_encoder.parameters():
        p.requires_grad = False
    for p in model.expression_encoder.parameters():
        p.requires_grad = False

    s1_config = dict(config["pretrain"]["stage1"])
    s1_config["epochs"] = args.stage1_epochs if hasattr(args, "stage1_epochs") \
                          and args.stage1_epochs else s1_config.get("epochs", 15)

    _run_stage(model, train_loader, val_loader, s1_config, device,
               args.output_dir, "Stage 1: Gene-level pre-training")

    # ── Stage 2 ───────────────────────────────────────────────────────────
    for p in model.junction_encoder.parameters():
        p.requires_grad = True
    for p in model.expression_encoder.parameters():
        p.requires_grad = False

    s2_config = dict(config["pretrain"]["stage2"])

    # Recreate DataLoader with stage 2 batch size if different
    if batch_size_s2 != batch_size_s1:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size_s2, shuffle=True,
            collate_fn=collate_pretrain, num_workers=args.num_workers,
            drop_last=True,
        )

    _run_stage(model, train_loader, val_loader, s2_config, device,
               args.output_dir, "Stage 2: Junction-level pre-training")

    # ── Save final model (best Stage 2 checkpoint, not final epoch) ─────────
    s2_best_path = os.path.join(args.output_dir, "stage_2:_junction-level_pre-training", "best.pt")
    if os.path.isfile(s2_best_path):
        best_ckpt = torch.load(s2_best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        logger_r.info(f"Loaded best Stage 2 checkpoint from {s2_best_path} "
                      f"(val loss: {best_ckpt.get('best_val_loss', 'N/A'):.4f})")

    final_path = os.path.join(args.output_dir, "best.pt")
    torch.save({"model_state_dict": model.state_dict(),
                "config": config}, final_path)
    logger_r.info(f"Final model saved to {final_path}")
    logger_r.info("Pre-training complete!")

    # Cleanup
    for gi in genome_indexers.values():
        try:
            gi.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
