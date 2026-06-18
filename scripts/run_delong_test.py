"""
PanCirc-Fungi: Statistical Significance Tests
==============================================
Computes DeLong test for ROC AUC comparisons:
- PanCirc Mode A (Genome+GTF) vs PanCirc Mode B (+GeneExp)
- PanCirc Mode A vs PanCirc ablation conditions
- (Cross-model comparisons use bootstrap CIs)

Usage:
    python scripts/run_delong_test.py --tsv all_lib_model_full.tsv \\
        --config config/default.yaml --run-all --output results/stats.json
"""

import argparse
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

from data.tsv_parser import (
    parse_strain_registry,
    build_strain_index,
    TEST_STRAINS,
    TRAIN_STRAINS,
)
from data.dataset import CircRNAFinetuneDataset, collate_pretrain
from data.circ_info_encoding import load_circ_info, filter_circ_info, get_positive_genes
from data.negative_sampling import get_negative_genes
from data.expression_encoding import load_expression_csv, align_circ_to_gene_expression, pad_to_max_replicates
from utils.gtf_utils import GeneModelIndexer
from data.genome_encoding import GenomeIndexer
from model.pancirc import PanCircModel
from utils.metrics import delong_roc_test, classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

GROUPS = ["Candida", "Cryptococcus", "Filamentous"]


def load_expression_data(entries, max_replicates=3):
    """Build expression data dict."""
    expression_data = {}
    for e in entries:
        circ_exp_raw = load_expression_csv(e.circexp_path)
        gene_exp_raw = load_expression_csv(e.geneexp_path)
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
        if circ_exp_raw is not None and os.path.isfile(e.circinfo_path):
            circ_info = load_circ_info(e.circinfo_path)
            aligned = align_circ_to_gene_expression(circ_info, circ_exp_raw, gene_exp_raw)
        expression_data[e.strain] = {"gene_exp": gene_exp_map, "aligned": aligned}
    return expression_data


def get_predictions(model, loader, device, task="pretrain", batch_mod_fn=None):
    """Run inference and return (y_true, y_prob)."""
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            if batch_mod_fn:
                batch = batch_mod_fn(batch)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(batch, task=task)
            probs = torch.sigmoid(outputs["gene_logits"].squeeze(-1))
            all_labels.append(batch["is_positive"].cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    valid = y_true >= 0
    return y_true[valid], y_prob[valid]


def process_group(group_name, tsv_path, config_path, checkpoint_dir,
                  device="cuda", max_replicates=3):
    """Run significance tests for one group."""
    base = PROJ_ROOT
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = base / checkpoint_dir

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = base / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    batch_size = config.get("finetune", {}).get("batch_size", 16)

    tsv_path = Path(tsv_path)
    if not tsv_path.is_absolute():
        tsv_path = base / tsv_path
    all_entries = parse_strain_registry(str(tsv_path))
    active = [e for e in all_entries if not e.is_excluded]

    train_strains = TRAIN_STRAINS.get(group_name, set())
    test_strains = TEST_STRAINS.get(group_name, set())
    train_entries = [e for e in active if e.strain in train_strains]
    test_entries = [e for e in active if e.strain in test_strains]

    logger.info(f"Group: {group_name} => test: {[e.strain for e in test_entries]}")

    # Build indexers
    gene_models = {}
    genome_indexers = {}
    for e in train_entries + test_entries:
        try:
            gene_models[e.strain] = GeneModelIndexer(e.gtf_path)
        except Exception as ex:
            logger.warning(f"  Failed GTF for {e.strain}: {ex}")
        try:
            genome_indexers[e.strain] = GenomeIndexer(e.genome_path)
        except Exception as ex:
            logger.warning(f"  Failed genome for {e.strain}: {ex}")

    # Positive/negative genes
    pos_genes = {}
    neg_genes = {}
    for e in active:
        if e.strain not in gene_models:
            continue
        circ_df = load_circ_info(e.circinfo_path)
        circ_df = filter_circ_info(circ_df)
        pos = get_positive_genes(circ_df)
        pos_genes[e.strain] = pos
        if len(pos) > 0 and e.strain in train_strains:
            neg = get_negative_genes(gene_models[e.strain], set(pos.keys()), len(pos), rng_seed=42)
            neg_genes[e.strain] = neg
        elif len(pos) > 0 and e.strain in test_strains:
            neg = get_negative_genes(gene_models[e.strain], set(pos.keys()), len(pos), rng_seed=42)
            neg_genes[e.strain] = neg
        else:
            neg_genes[e.strain] = []

    # Expression data
    expression_data = load_expression_data(train_entries + test_entries, max_replicates)

    # Test loader
    ds_test = CircRNAFinetuneDataset(
        entries=test_entries, gene_models=gene_models, genome_indexers=genome_indexers,
        positive_genes=pos_genes, negative_genes=neg_genes,
        expression_data=expression_data, config=data_config, max_replicates=max_replicates,
    )
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        collate_fn=collate_pretrain, num_workers=0,
    )

    # Load model
    n_sp = len(build_strain_index(active))
    model = PanCircModel(config["model"], n_species=n_sp)

    final_ckpt_path = checkpoint_dir / group_name / "final.pt"
    if final_ckpt_path.exists():
        ckpt = torch.load(str(final_ckpt_path), map_location=device)
        state = ckpt["model_state_dict"]
    else:
        cv_files = sorted((checkpoint_dir / group_name / "cv").glob("fold_*/best.pt"))
        if cv_files:
            state = torch.load(str(cv_files[0]), map_location=device)["model_state_dict"]
        else:
            logger.error(f"  No checkpoint for {group_name}")
            return None
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # ── Collect predictions for each condition ────────────────────────
    results = {}
    test_strain = list(test_strains)[0]

    # Mode A (pretrain task = no expression encoder)
    y_true, y_prob_mode_a = get_predictions(model, test_loader, device, task="pretrain")
    metrics_a = classification_metrics(y_true, y_prob_mode_a)
    results["Mode_A"] = {"y_true": y_true.tolist(), "y_prob": y_prob_mode_a.tolist(),
                         "auroc": metrics_a["auroc"]}

    # Mode B (finetune with circ_exp=0, gene_exp real)
    def zero_circ(batch):
        batch["circ_exp"] = torch.zeros_like(batch["circ_exp"])
        return batch
    y_true_b, y_prob_mode_b = get_predictions(model, test_loader, device,
                                              task="finetune", batch_mod_fn=zero_circ)
    metrics_b = classification_metrics(y_true_b, y_prob_mode_b)
    results["Mode_B"] = {"y_true": y_true_b.tolist(), "y_prob": y_prob_mode_b.tolist(),
                         "auroc": metrics_b["auroc"]}

    # No GTF
    def zero_gtf(batch):
        batch["gtf_features"] = torch.zeros_like(batch["gtf_features"])
        return batch
    _, y_prob_no_gtf = get_predictions(model, test_loader, device,
                                        task="pretrain", batch_mod_fn=zero_gtf)
    results["no_GTF"] = {"y_true": y_true.tolist(), "y_prob": y_prob_no_gtf.tolist()}

    # No genome
    def zero_genome(batch):
        batch["genome_context"] = torch.zeros_like(batch["genome_context"])
        return batch
    _, y_prob_no_genome = get_predictions(model, test_loader, device,
                                          task="pretrain", batch_mod_fn=zero_genome)
    results["no_Genome"] = {"y_true": y_true.tolist(), "y_prob": y_prob_no_genome.tolist()}

    # ── Compute DeLong tests ─────────────────────────────────────────
    comparisons = [
        ("Mode_A", "Mode_B", "GeneExp contribution"),
        ("Mode_A", "no_GTF", "GTFEncoder contribution"),
        ("Mode_A", "no_Genome", "GenomicContext contribution"),
    ]

    stats_results = {}
    for name_a, name_b, label in comparisons:
        if name_a not in results or name_b not in results:
            continue
        ra = results[name_a]
        rb = results[name_b]
        dl = delong_roc_test(np.array(ra["y_true"]),
                              np.array(ra["y_prob"]),
                              np.array(rb["y_prob"]))
        dl["label"] = label
        dl["model_a"] = name_a
        dl["model_b"] = name_b
        stats_results[f"{name_a}_vs_{name_b}"] = dl
        p = dl["p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        logger.info(f"  {label}: AUC {dl['auc_a']:.4f} vs {dl['auc_b']:.4f}, "
                    f"diff={dl['auc_diff']:.4f}, p={p:.4e} {sig}")

    return {
        "group": group_name,
        "test_strain": test_strain,
        "metrics": {
            "Mode_A": metrics_a,
            "Mode_B": metrics_b,
        },
        "delong_tests": stats_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default="all_lib_model_full.tsv")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/finetune")
    parser.add_argument("--group", default=None, choices=GROUPS + [None])
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/significance_tests.json")
    args = parser.parse_args()

    groups_to_run = [args.group] if args.group else (GROUPS if args.run_all else [])
    if not groups_to_run:
        parser.print_help()
        sys.exit(1)

    all_results = {}
    for grp in groups_to_run:
        r = process_group(grp, args.tsv, args.config, args.checkpoint_dir, device=args.device)
        if r:
            all_results[grp] = r

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Statistical Significance Summary (DeLong Test)")
    print("=" * 80)
    for grp, data in all_results.items():
        print(f"\n  {grp} (test: {data['test_strain']}):")
        print(f"    Mode A: {data['metrics']['Mode_A']['auroc']:.4f}")
        print(f"    Mode B: {data['metrics']['Mode_B']['auroc']:.4f}")
        print(f"    --- DeLong tests ---")
        for name, dl in data.get("delong_tests", {}).items():
            p = dl["p_value"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ci = dl["auc_diff_ci95"]
            print(f"    {dl['label']}: diff={dl['auc_diff']:.4f} "
                  f"95%CI=({ci[0]:.4f},{ci[1]:.4f}) p={p:.4e} {sig}")

    # ── Save ──────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
