"""
PanCirc-Fungi: Component Ablation Evaluation
=============================================
For each group's best finetuned model, zero one modality at a time and
measure the AUROC drop on the held-out test strain.

Conditions:
  full              — Mode A (Genome+GTF, no expression)
  no_gtf            — zero gtf_features
  no_genome          — zero genome_context
  no_species         — set strain_id=0
  no_junction        — zero all junction inputs (kmers, onehot, freq)
  no_expression      — as Mode A (task="pretrain", skip ExpressionEncoder)
  all_zero           — zero everything (random baseline sanity check)

Usage:
    python scripts/run_ablations.py --tsv all_lib_model_full.tsv \
        --group Candida --checkpoint checkpoints/finetune/Candida/final.pt \
        --config config/default.yaml

    # Or run all 3 groups:
    python scripts/run_ablations.py --run-all --config config/default.yaml

    # Via SLURM (all 3 groups, fast):
    sbatch scripts/run_ablations.slurm
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
from utils.metrics import classification_metrics

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

GROUPS = ["Candida", "Cryptococcus", "Filamentous"]

# ─── Ablation conditions ──────────────────────────────────────────────────────
ABLATION_CONDITIONS = [
    "full",          # Mode A: task="pretrain", no expression
    "mode_b",        # Mode B: task="finetune", circ_exp=0, gene_exp real
    "no_gtf",        # zero gtf_features, task="pretrain"
    "no_genome",     # zero genome_context, task="pretrain"
    "no_species",    # set strain_id=0, task="pretrain"
    "no_junction",   # zero all junction inputs, task="pretrain"
    "no_expression", # zero gene_exp, task="finetune" (circ_exp already 0)
    "all_zero",      # all inputs zeroed, task="pretrain"
]


def load_expression_data(entries, max_replicates=3):
    """Build expression data dict for a list of StrainEntry objects."""
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
            from data.expression_encoding import align_circ_to_gene_expression
            circ_info = load_circ_info(e.circinfo_path)
            aligned = align_circ_to_gene_expression(
                circ_info, circ_exp_raw, gene_exp_raw
            )
        expression_data[e.strain] = {
            "gene_exp": gene_exp_map,
            "aligned": aligned,
        }
    return expression_data


def zero_junction_in_batch(batch):
    """Zero all junction-related inputs in a batch dict (in-place)."""
    batch_keys_flat = [
        "donor_kmers", "acceptor_kmers",
        "donor_onehot", "acceptor_onehot",
        "donor_kmer_freq", "acceptor_kmer_freq",
    ]
    for key in batch_keys_flat:
        if key in batch:
            batch[key] = torch.zeros_like(batch[key])
    return batch


def evaluate_ablation(model, loader, device, condition):
    """Evaluate model under one ablation condition.

    Args:
        model: PanCircModel
        loader: DataLoader (test strain)
        device: torch device
        condition: str — ablation condition name

    Returns:
        dict of metrics
    """
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in loader:
            # ── Apply ablation ─────────────────────────────────────────
            if condition == "mode_b":
                batch["circ_exp"] = torch.zeros_like(batch["circ_exp"])
                # gene_exp kept real

            elif condition == "no_gtf":
                batch["gtf_features"] = torch.zeros_like(batch["gtf_features"])

            elif condition == "no_genome":
                batch["genome_context"] = torch.zeros_like(batch["genome_context"])

            elif condition == "no_species":
                batch["strain_id"] = torch.zeros_like(batch["strain_id"])

            elif condition == "no_junction":
                batch = zero_junction_in_batch(batch)

            elif condition == "no_expression":
                batch["circ_exp"] = torch.zeros_like(batch["circ_exp"])
                batch["gene_exp"] = torch.zeros_like(batch["gene_exp"])

            elif condition == "all_zero":
                batch["gtf_features"] = torch.zeros_like(batch["gtf_features"])
                batch["genome_context"] = torch.zeros_like(batch["genome_context"])
                batch["strain_id"] = torch.zeros_like(batch["strain_id"])
                batch = zero_junction_in_batch(batch)
                batch["circ_exp"] = torch.zeros_like(batch["circ_exp"])
                batch["gene_exp"] = torch.zeros_like(batch["gene_exp"])

            # condition == "full": no modification

            # ── Determine task ─────────────────────────────────────────
            # Mode A (no expression at all): task="pretrain"
            # Mode B (with expression encoder): task="finetune"
            # Ablations (no_gtf, no_genome, etc.) use task="pretrain" so
            # expression encoder is NOT included — isolates modality effect.
            # Only no_expression uses task="finetune" to test gene_exp zeroing.
            if condition == "no_expression":
                task = "finetune"
            else:
                task = "pretrain"

            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(batch, task=task)
            probs = torch.sigmoid(outputs["gene_logits"].squeeze(-1))
            all_labels.append(batch["is_positive"].cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    # Filter out dummy samples (is_positive == -1)
    valid = y_true >= 0
    if valid.any():
        y_true = y_true[valid]
        y_prob = y_prob[valid]

    if len(y_true) == 0:
        return {"auroc": float("nan"), "auprc": float("nan"),
                "f1": float("nan"), "accuracy": float("nan"), "mcc": float("nan")}

    return classification_metrics(y_true, y_prob)


def process_group(
    group_name,
    tsv_path,
    config_path,
    checkpoint_dir,
    device="cuda",
    max_replicates=3,
):
    """Run all ablations for one group."""
    base = PROJ_ROOT
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = base / checkpoint_dir

    # ── Load config ────────────────────────────────────────────────────────
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = base / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    batch_size = config.get("finetune", {}).get("batch_size", 16)

    # ── Parse TSV ──────────────────────────────────────────────────────────
    tsv_path = Path(tsv_path)
    if not tsv_path.is_absolute():
        tsv_path = base / tsv_path
    all_entries = parse_strain_registry(str(tsv_path))
    active = [e for e in all_entries if not e.is_excluded]

    train_strains = TRAIN_STRAINS.get(group_name, set())
    test_strains = TEST_STRAINS.get(group_name, set())
    train_entries = [e for e in active if e.strain in train_strains]
    test_entries = [e for e in active if e.strain in test_strains]

    if not test_entries:
        logger.error(f"No test entries for {group_name}")
        return []

    logger.info(f"Group: {group_name}")
    logger.info(f"  Train strains ({len(train_entries)}): {[e.strain for e in train_entries]}")
    logger.info(f"  Test strains  ({len(test_entries)}): {[e.strain for e in test_entries]}")

    # ── Build indexers ────────────────────────────────────────────────────
    gene_models = {}
    genome_indexers = {}
    for e in train_entries + test_entries:
        try:
            gene_models[e.strain] = GeneModelIndexer(e.gtf_path)
        except Exception as ex:
            logger.warning(f"  Failed to load GTF for {e.strain}: {ex}")
        try:
            genome_indexers[e.strain] = GenomeIndexer(e.genome_path)
        except Exception as ex:
            logger.warning(f"  Failed to load genome for {e.strain}: {ex}")

    # ── Build positive/negative gene lists ────────────────────────────────
    pos_genes = {}
    neg_genes = {}
    for e in active:
        if e.strain not in gene_models:
            continue
        circ_df = load_circ_info(e.circinfo_path)
        circ_df = filter_circ_info(circ_df)
        pos = get_positive_genes(circ_df)
        pos_genes[e.strain] = pos
        if len(pos) > 0 and e.strain in train_strains.union(test_strains):
            neg = get_negative_genes(
                gene_models[e.strain], set(pos.keys()), len(pos), rng_seed=42
            )
            neg_genes[e.strain] = neg

    # ── Load expression data ──────────────────────────────────────────────
    logger.info("  Loading expression data...")
    expression_data = load_expression_data(train_entries + test_entries, max_replicates)

    # ── Build test loader ─────────────────────────────────────────────────
    logger.info("  Building test DataLoader...")
    ds_test = CircRNAFinetuneDataset(
        entries=test_entries,
        gene_models=gene_models,
        genome_indexers=genome_indexers,
        positive_genes=pos_genes,
        negative_genes=neg_genes,
        expression_data=expression_data,
        config=data_config,
        max_replicates=max_replicates,
    )
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        collate_fn=collate_pretrain, num_workers=0,
    )
    logger.info(f"  Test samples: {len(ds_test)} ({len(test_entries)} strain{'s' if len(test_entries)>1 else ''})")

    # ── Load model ────────────────────────────────────────────────────────
    n_sp = len(build_strain_index(active))
    model = PanCircModel(config["model"], n_species=n_sp)

    # Load the best fold checkpoint (from final.pt or best.pt)
    final_ckpt_path = checkpoint_dir / group_name / "final.pt"
    if not final_ckpt_path.exists():
        # Try cv subdirectories
        cv_files = sorted((checkpoint_dir / group_name / "cv").glob("fold_*/best.pt"))
        if cv_files:
            ckpt_path = str(cv_files[0])
            logger.info(f"  Loading CV checkpoint: {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)["model_state_dict"]
        else:
            logger.error(f"  No checkpoint found for {group_name}")
            return []
    else:
        logger.info(f"  Loading final checkpoint: {final_ckpt_path}")
        ckpt = torch.load(str(final_ckpt_path), map_location=device)
        # final.pt stores model_state_dict directly under "model_state_dict" key
        state = ckpt["model_state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.info(f"  Missing keys (expected for expression_encoder): {len(missing)}")
    if unexpected:
        logger.info(f"  Unexpected keys: {len(unexpected)}")
    model.to(device)
    model.eval()

    # ── Run ablations ─────────────────────────────────────────────────────
    results = []
    for cond in ABLATION_CONDITIONS:
        metrics = evaluate_ablation(model, test_loader, device, cond)
        results.append({
            "group": group_name,
            "condition": cond,
            "auroc": metrics.get("auroc", float("nan")),
            "auprc": metrics.get("auprc", float("nan")),
            "f1": metrics.get("f1", float("nan")),
            "accuracy": metrics.get("accuracy", float("nan")),
            "mcc": metrics.get("mcc", float("nan")),
        })
        logger.info(f"  {cond:16s} → AUROC={metrics.get('auroc',float('nan')):.4f}  "
                    f"AUPRC={metrics.get('auprc',float('nan')):.4f}  "
                    f"F1={metrics.get('f1',float('nan')):.4f}")

    # ── Compute drops ─────────────────────────────────────────────────────
    full_auroc = results[0]["auroc"] if results else float("nan")
    for r in results:
        r["drop_from_full"] = full_auroc - r["auroc"] if not np.isnan(full_auroc) and not np.isnan(r["auroc"]) else float("nan")

    return results


def print_results_table(all_results):
    """Print formatted summary table."""
    print("\n" + "=" * 90)
    print("Component Ablation Results")
    print("=" * 90)

    header = f"{'Group':<14} {'Condition':<18} {'AUROC':<9} {'AUPRC':<9} {'F1':<9} {'Drop':<9}"
    print(header)
    print("-" * 90)

    for grp in GROUPS:
        results = all_results.get(grp, [])
        if not results:
            continue
        for r in results:
            drop = r.get("drop_from_full", float("nan"))
            drop_str = f"{drop:.4f}" if not np.isnan(drop) else "N/A"
            print(f"{grp:<14} {r['condition']:<18} {r['auroc']:<9.4f} {r['auprc']:<9.4f} "
                  f"{r['f1']:<9.4f} {drop_str:<9}")
        print("-" * 90)

    print("=" * 90)


def save_tsv(all_results, output_path):
    """Save all ablation results to a TSV file."""
    import csv
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["group", "condition", "auroc", "auprc", "f1", "accuracy", "mcc", "drop_from_full"])
        for grp in GROUPS:
            for r in all_results.get(grp, []):
                writer.writerow([
                    r["group"], r["condition"],
                    f"{r['auroc']:.4f}" if not np.isnan(r['auroc']) else "N/A",
                    f"{r['auprc']:.4f}" if not np.isnan(r['auprc']) else "N/A",
                    f"{r['f1']:.4f}" if not np.isnan(r['f1']) else "N/A",
                    f"{r['accuracy']:.4f}" if not np.isnan(r['accuracy']) else "N/A",
                    f"{r['mcc']:.4f}" if not np.isnan(r['mcc']) else "N/A",
                    f"{r['drop_from_full']:.4f}" if not np.isnan(r['drop_from_full']) else "N/A",
                ])
    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PanCirc-Fungi component ablation")
    parser.add_argument("--tsv", default="all_lib_model_full.tsv")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/finetune")
    parser.add_argument("--group", default=None, choices=GROUPS + [None])
    parser.add_argument("--run-all", action="store_true", help="Run all 3 groups")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/ablations.tsv")

    args = parser.parse_args()

    groups_to_run = [args.group] if args.group else (GROUPS if args.run_all else [])
    if not groups_to_run:
        parser.print_help()
        print("\nSpecify --group or --run-all")
        sys.exit(1)

    all_results = {}
    for grp in groups_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {grp}...")
        logger.info(f"{'='*60}")
        results = process_group(
            grp, args.tsv, args.config,
            args.checkpoint_dir, device=args.device,
        )
        if results:
            all_results[grp] = results

    # Print and save results
    if all_results:
        print_results_table(all_results)
        save_tsv(all_results, args.output)


if __name__ == "__main__":
    main()
