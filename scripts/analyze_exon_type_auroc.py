"""
mycoCirc: Single-exon vs multi-exon AUROC breakdown.

Evaluates whether the model performs differently on single-exon genes
vs multi-exon genes. This is a critical biological question because
~94% of fungal circRNA-positive genes are single-exon.

Output: results/exon_type_auroc.tsv + results/exon_type_auroc_summary.tsv

Usage:
    sbatch -A ylab -p gpu --gres=gpu:1 -D /media/share/workdir/1610305236/Panfungi/4_model \
        --cpus-per-task=4 --mem=16G \
        -o /home/1610305236/log/exon-%j.out -e /home/1610305236/log/exon-%j.err \
        --wrap "source /media/share/home/1610305236/.local/share/mamba/etc/profile.d/mamba.sh \
        && micromamba activate pancirc-fungi \
        && python scripts/analyze_exon_type_auroc.py \
        --config config/default.yaml --checkpoint-dir checkpoints/finetune"
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from data.tsv_parser import parse_strain_registry, build_strain_index
from data.dataset import CircRNAFinetuneDataset, collate_pretrain
from data.circ_info_encoding import load_circ_info, filter_circ_info, get_positive_genes
from data.negative_sampling import get_negative_genes
from model.pancirc import PanCircModel
from utils.metrics import classification_metrics
from utils.gtf_utils import GeneModelIndexer
from data.genome_encoding import GenomeIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

GROUPS = ["Candida", "Cryptococcus", "Filamentous"]
TEST_STRAINS = {"Candida": "P4", "Cryptococcus": "C4", "Filamentous": "F6"}


def load_data(group: str, checkpoint_dir: str, config: dict, device: torch.device,
              tsv_entries: list):
    """Load model + build test loader, return model, loader, gene_obj_map."""
    test_strain = TEST_STRAINS[group]
    test_entry = next((e for e in tsv_entries
                       if e.strain == test_strain and e.group == group), None)
    if test_entry is None:
        logger.error(f"  Test strain not found for {group}")
        return None, None, None

    # Load checkpoint
    ckpt_dir = Path(checkpoint_dir) / group
    preferred = [ckpt_dir / "final_pretrained.pt", ckpt_dir / "final.pt"]
    state = None
    for p in preferred:
        if p.exists():
            state = torch.load(str(p), map_location=device)
            state = state.get("model_state_dict", state)
            break
    if state is None:
        cv_files = sorted(ckpt_dir.glob("cv/fold_*/best.pt"))
        if cv_files:
            state = torch.load(str(cv_files[0]), map_location=device)
            state = state.get("model_state_dict", state)
    if state is None:
        logger.error(f"  No checkpoint for {group}")
        return None, None, None

    n_sp = len(build_strain_index([e for e in tsv_entries if not e.is_excluded]))
    model = PanCircModel(config["model"], n_species=n_sp)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Build indexers
    gene_model = GeneModelIndexer(test_entry.gtf_path)
    genome_indexer = GenomeIndexer(test_entry.genome_path)

    # Positive/negative genes
    circ_df = filter_circ_info(load_circ_info(test_entry.circinfo_path))
    pos_genes = get_positive_genes(circ_df)
    if not pos_genes:
        return None, None, None
    neg = get_negative_genes(gene_model, set(pos_genes.keys()), len(pos_genes), rng_seed=42)

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
    bs = config.get("finetune", {}).get("batch_size", 16)
    loader = DataLoader(ds, batch_size=bs, shuffle=False,
                        collate_fn=collate_pretrain, num_workers=0)

    # Pre-classify each sample by exon count
    gene_exon_map = {}
    for idx in range(len(ds)):
        entry, gid, is_pos = ds.samples[idx]
        gene = gene_model.get_gene(gid)
        if gene is not None:
            n_exons = gene.exon_count
        else:
            n_exons = -1
        gene_exon_map[idx] = {
            "gene_id": gid,
            "n_exons": n_exons,
            "is_single_exon": n_exons == 1,
        }

    return model, loader, gene_exon_map


def evaluate_by_exon_type(model, loader, device, gene_exon_map):
    """Run inference and split metrics by single-exon vs multi-exon."""
    all_probs = {}
    all_labels = {}
    all_exon_type = {}

    with torch.no_grad():
        sample_offset = 0
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(batch, task="pretrain")
            probs = torch.sigmoid(outputs["gene_logits"].squeeze(-1))
            labels = batch["is_positive"]

            bs = len(labels)
            for i in range(bs):
                idx = sample_offset + i
                if idx not in gene_exon_map:
                    continue
                info = gene_exon_map[idx]
                all_probs[idx] = probs[i].item()
                all_labels[idx] = labels[i].item()
                all_exon_type[idx] = info["is_single_exon"]

            sample_offset += bs

    # Separate by exon type (exclude dummy = label < 0)
    se_labels, se_probs = [], []
    me_labels, me_probs = [], []
    for idx in all_labels:
        if all_labels[idx] < 0:
            continue
        if all_exon_type.get(idx, False):
            se_labels.append(all_labels[idx])
            se_probs.append(all_probs[idx])
        else:
            me_labels.append(all_labels[idx])
            me_probs.append(all_probs[idx])

    results = {
        "single_exon": {"n": len(se_labels)},
        "multi_exon": {"n": len(me_labels)},
        "total": {"n": len(se_labels) + len(me_labels)},
    }

    if se_labels:
        results["single_exon"]["pos"] = int(sum(se_labels))
        results["single_exon"].update(classification_metrics(
            np.array(se_labels), np.array(se_probs)))
    if me_labels:
        results["multi_exon"]["pos"] = int(sum(me_labels))
        results["multi_exon"].update(classification_metrics(
            np.array(me_labels), np.array(me_probs)))
    # Compute total metrics over all genes
    all_labels = np.array(se_labels + me_labels)
    all_probs = np.array(se_probs + me_probs)
    if len(all_labels) > 0:
        results["total"]["pos"] = int(all_labels.sum())
        results["total"].update(classification_metrics(all_labels, all_probs))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="mycoCirc: single-exon vs multi-exon AUROC breakdown")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/finetune")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/exon_type_auroc.tsv")
    parser.add_argument("--output-summary", default="results/exon_type_auroc_summary.tsv")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tsv_entries = parse_strain_registry(config["data"]["tsv_path"])

    all_results = {}
    for grp in GROUPS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {grp} (test: {TEST_STRAINS[grp]})...")
        logger.info(f"{'='*60}")

        out = load_data(grp, args.checkpoint_dir, config, device, tsv_entries)
        if out is None or out[0] is None:
            continue
        model, loader, gene_exon_map = out
        results = evaluate_by_exon_type(model, loader, device, gene_exon_map)
        all_results[grp] = results

        for subtype in ["single_exon", "multi_exon", "total"]:
            r = results.get(subtype, {})
            if r:
                logger.info(f"  {subtype:15s}: N={r.get('n',0):>5d}  "
                            f"Pos={r.get('pos',0):>4d}  "
                            f"AUROC={r.get('auroc',0):.4f}  "
                            f"AUPRC={r.get('auprc',0):.4f}  "
                            f"F1={r.get('f1',0):.4f}")

    # Print table
    print("\n" + "=" * 90)
    print("Single-Exon vs Multi-Exon AUROC Breakdown")
    print("=" * 90)
    header = f"{'Group':<15} {'Type':<15} {'N':>6} {'Pos':>5} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Acc':>8} {'MCC':>8}"
    print(header)
    print("-" * 90)
    for grp in GROUPS:
        r = all_results.get(grp, {})
        for subtype in ["single_exon", "multi_exon", "total"]:
            s = r.get(subtype, {})
            if s:
                print(f"{grp:<15} {subtype:<15} {s['n']:>6} {s.get('pos',0):>5} "
                      f"{s.get('auroc',0):>8.4f} {s.get('auprc',0):>8.4f} "
                      f"{s.get('f1',0):>8.4f} {s.get('accuracy',0):>8.4f} "
                      f"{s.get('mcc',0):>8.4f}")
    print("=" * 90)

    # Save results TSV (per-group, per-type)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJ_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["group", "exon_type", "n_genes", "n_positive",
                     "auroc", "auprc", "f1", "accuracy", "mcc"])
        for grp in GROUPS:
            r = all_results.get(grp, {})
            for subtype in ["single_exon", "multi_exon", "total"]:
                s = r.get(subtype, {})
                if s:
                    w.writerow([grp, subtype, s["n"], s.get("pos", 0),
                                f"{s.get('auroc', 0):.4f}",
                                f"{s.get('auprc', 0):.4f}",
                                f"{s.get('f1', 0):.4f}",
                                f"{s.get('accuracy', 0):.4f}",
                                f"{s.get('mcc', 0):.4f}"])
    logger.info(f"Results saved to {output_path}")

    # Save summary TSV (single-exon stats for the "94% claim")
    summary_path = Path(args.output_summary)
    if not summary_path.is_absolute():
        summary_path = PROJ_ROOT / summary_path
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["group", "pct_single_exon_of_all", "pct_single_exon_of_positive",
                     "auroc_single", "auroc_multi", "auroc_gap",
                     "n_total", "n_positive", "n_single_exon_total",
                     "n_single_exon_positive", "n_multi_exon_positive"])
        for grp in GROUPS:
            r = all_results.get(grp, {})
            se = r.get("single_exon", {})
            me = r.get("multi_exon", {})
            total = r.get("total", {})
            n_pos = total.get("n_positive", 0) if "n_positive" in total else 0
            n_se_pos = se.get("pos", 0)
            n_me_pos = me.get("pos", 0)
            n_se_total = se.get("n", 0)
            n_total = total.get("n", 0)
            pct_se_of_all = 100 * n_se_total / max(n_total, 1)
            pct_se_of_pos = 100 * n_se_pos / max(n_se_pos + n_me_pos, 1)
            gap = se.get("auroc", 0) - me.get("auroc", 0)
            w.writerow([grp, f"{pct_se_of_all:.1f}", f"{pct_se_of_pos:.1f}",
                        f"{se.get('auroc', 0):.4f}", f"{me.get('auroc', 0):.4f}",
                        f"{gap:+.4f}",
                        n_total, n_se_pos + n_me_pos, n_se_total,
                        n_se_pos, n_me_pos])
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
