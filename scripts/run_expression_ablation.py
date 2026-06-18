#!/usr/bin/env python3
"""
Expression ablation for PanCirc-Fungi.

Evaluates the contribution of GeneExp by ablating it.

Usage:
    python scripts/run_expression_ablation.py \\
        --pretrained-dir checkpoints/finetune \\
        [--groups Candida Cryptococcus Filamentous]

Output:
    Tab-separated table printed to stdout:
    group   mode    auroc   auprc   f1      accuracy    mcc

Modes tested:
    original        — GeneExp intact (Genome+GTF+GeneExp)
    shuffle_geneexp — GeneExp shuffled across genes (label broken)
    zero_genexp     — GeneExp set to zero
    no_expression   — pretrain mode (Genome+GTF only, no expression path)
"""

import argparse
import copy
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.multiprocessing as mp

from data.dataset import CircRNAFinetuneDataset, collate_pretrain
from data.expression_encoding import (
    encode_expression_values,
    load_expression_csv,
    pad_to_max_replicates,
)
from data.genome_encoding import GenomeIndexer
from data.tsv_parser import TEST_STRAINS, TRAIN_STRAINS, build_strain_index, parse_strain_registry
from model.pancirc import PanCircModel
from utils.gtf_utils import GeneModelIndexer
from utils.metrics import classification_metrics

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("ablation")


# ─── Helpers ───────────────────────────────────────────────────────────────


def load_gene_expression(entries, max_replicates=3):
    """Load and normalise GeneExp for each strain.

    Returns dict: {strain: {"gene_exp": {gene_id: np.ndarray}}}
    """
    expression_data = {}
    for e in entries:
        gene_exp_map = {}
        ge_raw = load_expression_csv(e.geneexp_path)
        if ge_raw is not None:
            ge_vals, ge_ids, _ = encode_expression_values(
                ge_raw, log1p=True, zscore=True
            )
            for i, gid in enumerate(ge_ids):
                gene_exp_map[gid] = pad_to_max_replicates(ge_vals[i], max_replicates)
        expression_data[e.strain] = {"gene_exp": gene_exp_map}
    return expression_data


def shuffle_geneexp(expr_data, rng=None):
    """Shuffle GeneExp across genes within each strain."""
    if rng is None:
        rng = np.random.RandomState(42)
    out = copy.deepcopy(expr_data)
    for strain in out:
        ge = out[strain].get("gene_exp", {})
        if not ge:
            continue
        keys = list(ge.keys())
        vals = list(ge.values())
        rng.shuffle(vals)
        out[strain]["gene_exp"] = dict(zip(keys, vals))
    return out


def zero_geneexp(expr_data):
    """Zero out all GeneExp values."""
    out = copy.deepcopy(expr_data)
    for strain in out:
        for gid in out[strain].get("gene_exp", {}):
            out[strain]["gene_exp"][gid] = np.zeros(3, dtype=np.float32)
    return out


def build_indexers(entries):
    """Build GeneModelIndexer and GenomeIndexer for each strain."""
    gene_models, genome_indexers = {}, {}
    for e in entries:
        try:
            gene_models[e.strain] = GeneModelIndexer(e.gtf_path)
        except Exception:
            pass
        try:
            genome_indexers[e.strain] = GenomeIndexer(e.genome_path)
        except Exception:
            pass
    return gene_models, genome_indexers


def load_features(features_dir, entries):
    """Load positive/negative gene lists."""
    from data.circ_info_encoding import filter_circ_info, load_circ_info

    pos_genes, neg_genes = {}, {}
    for e in entries:
        sd = os.path.join(features_dir, e.strain)
        if not os.path.isdir(sd):
            pos_genes[e.strain] = {}
            neg_genes[e.strain] = []
            continue
        circ_df = filter_circ_info(load_circ_info(e.circinfo_path))
        circ_map = {str(gid): g for gid, g in circ_df.groupby("gene_id")}
        pos_path = os.path.join(sd, "positive_gene_ids.npy")
        if os.path.isfile(pos_path):
            ids = np.load(pos_path, allow_pickle=True).tolist()
            pos_genes[e.strain] = {gid: circ_map.get(gid) for gid in ids}
        else:
            pos_genes[e.strain] = {}
        neg_path = os.path.join(sd, "negative_gene_ids.npy")
        if os.path.isfile(neg_path):
            neg_genes[e.strain] = np.load(neg_path, allow_pickle=True).tolist()
        else:
            neg_genes[e.strain] = []
    return pos_genes, neg_genes


def evaluate(model, loader, device, task="finetune"):
    """Return classification_metrics dict."""
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for batch in loader:
            b = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(b, task=task)
            probs = torch.sigmoid(out["gene_logits"].squeeze(-1))
            all_y.append(b["is_positive"].cpu())
            all_p.append(probs.cpu())
    y_true = torch.cat(all_y)
    y_prob = torch.cat(all_p)
    valid = y_true >= 0
    return classification_metrics(y_true[valid].numpy(), y_prob[valid].numpy())


# ─── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="GeneExp ablation for PanCirc-Fungi")
    parser.add_argument(
        "--pretrained-dir",
        default="checkpoints/finetune",
        help="Directory containing {group}/best.pt",
    )
    parser.add_argument(
        "--groups", nargs="+",
        default=["Candida", "Cryptococcus", "Filamentous"],
    )
    parser.add_argument("--features-dir", default="checkpoints/features")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    max_rep = config["model"]["expression"]["max_replicates"]
    entries = [
        e for e in parse_strain_registry(config["data"]["tsv_path"])
        if not e.is_excluded
    ]

    # ── Table header ──────────────────────────────────────────────────
    header = "\t".join(["group", "mode", "auroc", "auprc", "f1", "accuracy", "mcc"])
    print(header)
    print("#" + "=" * (len(header) + 1))

    for group in args.groups:
        train_strains = TRAIN_STRAINS[group]
        test_strains = TEST_STRAINS[group]
        all_strains = train_strains | test_strains
        group_entries = [e for e in entries if e.strain in all_strains]
        test_entries = [e for e in group_entries if e.strain in test_strains]

        if not test_entries:
            logger.warning(f"No test entries for {group}, skipping")
            continue

        gene_models, genome_indexers = build_indexers(group_entries)
        pos_genes, neg_genes = load_features(args.features_dir, group_entries)
        expr_data = load_gene_expression(group_entries, max_rep)

        # Load model
        ckpt_path = os.path.join(args.pretrained_dir, group, "best.pt")
        if not os.path.isfile(ckpt_path):
            ckpt_path = os.path.join(args.pretrained_dir, group, "final.pt")
        if not os.path.isfile(ckpt_path):
            logger.error(f"No checkpoint found for {group}")
            continue

        n_species = len(build_strain_index(entries))
        model = PanCircModel(config["model"], n_species=n_species)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.to(device)
        model.eval()

        rng = np.random.RandomState(42)

        def make_loader(expr):
            ds = CircRNAFinetuneDataset(
                test_entries, gene_models, genome_indexers,
                pos_genes, neg_genes, expression_data=expr,
                config=config["data"], max_replicates=max_rep,
            )
            return DataLoader(ds, batch_size=64, collate_fn=collate_pretrain, shuffle=False)

        # 1. Original GenuineGenome+GTF+GeneExp
        loader_orig = make_loader(expr_data)
        m = evaluate(model, loader_orig, device, task="finetune")
        print(f"{group}\toriginal\t{m['auroc']:.4f}\t{m['auprc']:.4f}\t{m['f1']:.4f}\t{m['accuracy']:.4f}\t{m['mcc']:.4f}")

        # 2. Shuffle GeneExp
        expr_shuf = shuffle_geneexp(expr_data, rng)
        m = evaluate(model, make_loader(expr_shuf), device, task="finetune")
        print(f"{group}\tshuffle_geneexp\t{m['auroc']:.4f}\t{m['auprc']:.4f}\t{m['f1']:.4f}\t{m['accuracy']:.4f}\t{m['mcc']:.4f}")

        # 3. Zero GeneExp
        expr_zero = zero_geneexp(expr_data)
        m = evaluate(model, make_loader(expr_zero), device, task="finetune")
        print(f"{group}\tzero_geneexp\t{m['auroc']:.4f}\t{m['auprc']:.4f}\t{m['f1']:.4f}\t{m['accuracy']:.4f}\t{m['mcc']:.4f}")

        # 4. No expression (pretrain mode, Genome+GTF only)
        m = evaluate(model, loader_orig, device, task="pretrain")
        print(f"{group}\tno_expression\t{m['auroc']:.4f}\t{m['auprc']:.4f}\t{m['f1']:.4f}\t{m['accuracy']:.4f}\t{m['mcc']:.4f}")


if __name__ == "__main__":
    main()
