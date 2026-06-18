"""
PanCirc-Fungi × JEDI Comparison
=================================
Extract fungal data in JEDI raw JSONL format, convert to k-mer encoded
format via generate_input.py, train/test JEDI, and collect metrics.

Usage:
    python scripts/run_jedi_comparison.py --tsv all_lib_model_full.tsv \\
        --group Candida --jedi-dir other_models/JEDI-master \\
        --out-dir results/jedi [--k 3 --L 4 --epochs 20]

Or via SLURM:
    sbatch --array=1-3 scripts/run_jedi.slurm
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

# Add project root
PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from data.tsv_parser import (
    parse_strain_registry,
    TEST_STRAINS,
    TRAIN_STRAINS,
)
from data.circ_info_encoding import load_circ_info, filter_circ_info, get_positive_genes
from data.negative_sampling import get_negative_genes
from utils.gtf_utils import GeneModelIndexer
from data.genome_encoding import GenomeIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
SEQUENCE_PADDING = 2000  # extra bases on each side of gene for the seq field

GROUPS = ["Candida", "Cryptococcus", "Filamentous"]


def get_all_exon_pairs(gene):
    """Get ALL possible backsplice exon-exon pairs for a gene.

    Returns list of (acceptor_pos, donor_pos) in genomic coordinates.
    - Multi-exon: all i<j pairs.
    - Single-exon: one pair (exon_end, exon_start) representing circularization.
    """
    pairs = []
    n = gene.exon_count
    if n == 1:
        # Single-exon backsplice: 3' end -> 5' start
        if gene.strand == "+":
            pairs.append((gene.exon_ends[0], gene.exon_starts[0]))
        else:
            pairs.append((gene.exon_starts[0], gene.exon_ends[0]))
        return pairs
    for i in range(n - 1):
        for j in range(i + 1, n):
            if gene.strand == "+":
                acceptor = gene.exon_ends[i]
                donor = gene.exon_starts[j]
                if donor <= acceptor:
                    continue
            else:
                acceptor = gene.exon_starts[i]
                donor = gene.exon_ends[j]
                if donor >= acceptor:
                    continue
            pairs.append((acceptor, donor))
    return pairs


def prepare_jedi_raw_data(
    entries,
    gene_models,
    genome_indexers,
    positive_genes,
    negative_genes,
    output_dir,
    gene_padding=SEQUENCE_PADDING,
):
    """Prepare JEDI raw JSONL files (pos_raw.json, neg_raw.json) for a group.

    Each line contains a gene with ALL exon boundary pairs:
        {"seq": "...", "strand": "+/-", "junctions": {"head": [...], "tail": [...]}}

    The label 1/0 is assigned per gene based on whether it has circRNA
    annotations (JEDI learns at the gene level, not per junction).

    Both single-exon and multi-exon genes are included.
    """
    os.makedirs(output_dir, exist_ok=True)

    pos_path = os.path.join(output_dir, "pos_raw.json")
    neg_path = os.path.join(output_dir, "neg_raw.json")

    n_pos = 0
    n_neg = 0

    def _pairs_to_junctions(pairs, seq_start, seq):
        hlist, tlist = [], []
        for acc, don in pairs:
            h = acc - seq_start
            t = don - seq_start
            if 0 <= h < len(seq) and 0 <= t < len(seq):
                hlist.append(h)
                tlist.append(t)
        return hlist, tlist

    # ─── Process positive genes ────────────────────────────────────────────
    with open(pos_path, "w") as fpos:
        for entry in entries:
            strain = entry.strain
            gm = gene_models.get(strain)
            gi = genome_indexers.get(strain)
            if gm is None or gi is None:
                continue

            strain_pos = positive_genes.get(strain, {})
            for gene_id in strain_pos:
                gene = gm.get_gene(gene_id)
                if gene is None or gene.exon_count < 1:
                    continue

                seq_start = max(1, gene.start - gene_padding)
                seq_end = gene.end + gene_padding
                seq = gi.get_seq(gene.chrom, seq_start, seq_end, "+")
                if not seq or len(seq) < 10:
                    continue

                pairs = get_all_exon_pairs(gene)
                head_list, tail_list = _pairs_to_junctions(pairs, seq_start, seq)
                if not head_list:
                    continue

                fpos.write(json.dumps({
                    "seq": seq, "strand": gene.strand,
                    "junctions": {"head": head_list, "tail": tail_list},
                }) + "\n")
                n_pos += 1

    # ─── Process negative genes ────────────────────────────────────────────
    with open(neg_path, "w") as fneg:
        for entry in entries:
            strain = entry.strain
            gm = gene_models.get(strain)
            gi = genome_indexers.get(strain)
            if gm is None or gi is None:
                continue

            strain_neg = negative_genes.get(strain, [])
            for gene_id in strain_neg:
                gene = gm.get_gene(gene_id)
                if gene is None or gene.exon_count < 1:
                    continue

                seq_start = max(1, gene.start - gene_padding)
                seq_end = gene.end + gene_padding
                seq = gi.get_seq(gene.chrom, seq_start, seq_end, "+")
                if not seq or len(seq) < 10:
                    continue

                pairs = get_all_exon_pairs(gene)
                head_list, tail_list = _pairs_to_junctions(pairs, seq_start, seq)
                if not head_list:
                    continue

                fneg.write(json.dumps({
                    "seq": seq, "strand": gene.strand,
                    "junctions": {"head": head_list, "tail": tail_list},
                }) + "\n")
                n_neg += 1

    logger.info(f"  Raw data: {n_pos} positive, {n_neg} negative records")
    return pos_path, neg_path


def convert_to_jedi_format(
    jedi_dir,
    pos_raw_path,
    neg_raw_path,
    output_path,
    k=3,
    L=4,
    python_cmd=None,
):
    python_cmd = python_cmd or sys.executable
    gen_script = os.path.join(jedi_dir, "src", "generate_input.py")

    if not os.path.isfile(gen_script):
        logger.error(f"generate_input.py not found at {gen_script}")
        return False

    cmd = [
        python_cmd, gen_script,
        pos_raw_path, neg_raw_path,
        str(k), str(L), output_path,
    ]

    logger.info(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"  generate_input.py failed:\n{result.stderr}")
        return False

    logger.info(f"  Generated: {output_path}")
    return True


def run_jedi_training(
    jedi_dir,
    data_dir,
    output_dir,
    group_name,
    k=3,
    L=4,
    epochs=20,
    batch_size=64,
    python_cmd=None,
):
    python_cmd = python_cmd or sys.executable
    run_script = os.path.join(jedi_dir, "src", "run.py")
    config_yml = os.path.join(jedi_dir, "src", "config.yml")

    if not os.path.isfile(run_script):
        logger.error(f"run.py not found at {run_script}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    group_config = os.path.join(output_dir, "config.yml")
    with open(config_yml) as f:
        cfg_template = f.read()
    cfg_content = cfg_template.replace(
        "PATH_TO_PROCESSED_DATA_DIR", data_dir
    ).replace(
        "PATH_TO_PREDICTION_DIR", output_dir
    )
    with open(group_config, "w") as f:
        f.write(cfg_content)

    cmd = [
        python_cmd, run_script,
        f"--config={group_config}",
        "--cv=0",
        f"--K={k}",
        f"--L={L}",
        f"--num_epochs={epochs}",
        f"--batch_size={batch_size}",
    ]

    logger.info(f"  Running JEDI training: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    with open(os.path.join(output_dir, "jedi_stdout.log"), "w") as f:
        f.write(result.stdout)
    with open(os.path.join(output_dir, "jedi_stderr.log"), "w") as f:
        f.write(result.stderr)

    if result.returncode != 0:
        logger.error(f"  JEDI training failed:\n{result.stderr}")
        return None

    metrics = parse_jedi_metrics(result.stderr)
    logger.info(f"  JEDI metrics: {metrics}")

    return metrics


def parse_jedi_metrics(output):
    metrics = {}

    test_pat = r"Testing.*\nLs: ([0-9.e-]+)\tA: ([0-9.]+)\t P: ([0-9.]+)\tF: ([0-9.]+),\tM: ([0-9.e-]+)\tSe: ([0-9.]+)\tSp: ([0-9.]+)"
    match = re.search(test_pat, output)
    if match:
        metrics["test_loss"] = float(match.group(1))
        metrics["test_accuracy"] = float(match.group(2))
        metrics["test_precision"] = float(match.group(3))
        metrics["test_f1"] = float(match.group(4))
        metrics["test_mcc"] = float(match.group(5))
        metrics["test_sensitivity"] = float(match.group(6))
        metrics["test_specificity"] = float(match.group(7))

    train_pat = r"Epoch (\d+).*\nLs: ([0-9.e-]+)\tA: ([0-9.]+)\t P: ([0-9.]+)\tF: ([0-9.]+),\tM: ([0-9.e-]+)\tSe: ([0-9.]+)\tSp: ([0-9.]+)"
    matches = list(re.finditer(train_pat, output))
    if matches:
        last = matches[-1]
        metrics["train_epoch"] = int(last.group(1))
        metrics["train_loss"] = float(last.group(2))
        metrics["train_accuracy"] = float(last.group(3))
        metrics["train_precision"] = float(last.group(4))
        metrics["train_f1"] = float(last.group(5))
        metrics["train_mcc"] = float(last.group(6))
        metrics["train_sensitivity"] = float(last.group(7))
        metrics["train_specificity"] = float(last.group(8))

    return metrics


def compute_auroc_auprc_from_predictions(pred_path):
    try:
        with open(pred_path) as f:
            preds = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning(f"  Prediction file not found or invalid: {pred_path}")
        return {}

    from sklearn.metrics import roc_auc_score, average_precision_score

    scores = np.array([p[0] for p in preds])
    labels = np.array([p[1] for p in preds])

    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "n_samples": len(preds)}

    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "n_samples": len(preds),
    }


def process_group(
    entries,
    group_name,
    jedi_dir,
    output_base_dir,
    k=3,
    L=4,
    epochs=20,
    batch_size=64,
    python_cmd=None,
    gene_padding=SEQUENCE_PADDING,
):
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing group: {group_name}")
    logger.info(f"{'='*60}")

    group_members = [e for e in entries if e.group == group_name and not e.is_excluded]
    train_strains = TRAIN_STRAINS.get(group_name, set())
    test_strains = TEST_STRAINS.get(group_name, set())

    train_entries = [e for e in group_members if e.strain in train_strains]
    test_entries = [e for e in group_members if e.strain in test_strains]

    if not train_entries:
        logger.error(f"  No training entries for {group_name}")
        return None

    logger.info(f"  Train strains ({len(train_entries)}): {[e.strain for e in train_entries]}")
    logger.info(f"  Test strains  ({len(test_entries)}): {[e.strain for e in test_entries]}")

    group_out_dir = os.path.join(output_base_dir, group_name)
    os.makedirs(group_out_dir, exist_ok=True)

    # ── 1. Build gene models & genome indexers ────────────────────────────
    def load_strain_data(strain_entries):
        gm_dict, gi_dict, pos_dict, neg_dict = {}, {}, {}, {}
        for entry in strain_entries:
            strain = entry.strain
            logger.info(f"  Loading {strain}...")
            try:
                gm = GeneModelIndexer(entry.gtf_path)
            except Exception as e:
                logger.warning(f"    Failed to load GTF for {strain}: {e}")
                continue
            try:
                gi = GenomeIndexer(entry.genome_path)
            except Exception as e:
                logger.warning(f"    Failed to load genome for {strain}: {e}")
                continue
            gm_dict[strain] = gm
            gi_dict[strain] = gi

            circ_df = load_circ_info(entry.circinfo_path)
            circ_df = filter_circ_info(circ_df)
            pos = get_positive_genes(circ_df)
            pos_dict[strain] = pos
            n_pos = len(pos)
            if n_pos == 0:
                logger.info(f"    {strain}: 0 usable circRNA genes")
                neg_dict[strain] = []
                continue
            neg = get_negative_genes(gm, set(pos.keys()), n_pos, rng_seed=42)
            neg_dict[strain] = neg
            logger.info(f"    {strain}: {n_pos} pos, {len(neg)} neg")
        return gm_dict, gi_dict, pos_dict, neg_dict

    train_gm, train_gi, train_pos, train_neg = load_strain_data(train_entries)
    test_gm, test_gi, test_pos, test_neg = load_strain_data(test_entries) if test_entries else ({}, {}, {}, {})

    if not train_gm:
        logger.error(f"  No valid training data for {group_name}")
        return None

    # ── 2. Prepare TRAINING set ──────────────────────────────────────────
    train_valid = [e for e in train_entries if e.strain in train_gm]
    train_raw_dir = os.path.join(group_out_dir, "raw_train")
    os.makedirs(train_raw_dir, exist_ok=True)
    logger.info(f"  Preparing TRAINING raw data...")
    prepare_jedi_raw_data(train_valid, train_gm, train_gi,
                          train_pos, train_neg, train_raw_dir, gene_padding)

    jedi_data_dir = os.path.join(group_out_dir, "jedi_data")
    os.makedirs(jedi_data_dir, exist_ok=True)
    train_path = os.path.join(jedi_data_dir, f"data.0.K{k}.L{L}.train")

    logger.info(f"  Converting TRAINING to JEDI format (K={k}, L={L})...")
    train_pos_raw = os.path.join(train_raw_dir, "pos_raw.json")
    train_neg_raw = os.path.join(train_raw_dir, "neg_raw.json")
    if not convert_to_jedi_format(jedi_dir, train_pos_raw, train_neg_raw, train_path, k, L, python_cmd):
        logger.error(f"  Failed to convert training data")
        return None

    # ── 3. Prepare TEST set ──────────────────────────────────────────────
    test_path = os.path.join(jedi_data_dir, f"data.0.K{k}.L{L}.test")
    if test_gm:
        test_valid = [e for e in test_entries if e.strain in test_gm]
        test_raw_dir = os.path.join(group_out_dir, "raw_test")
        os.makedirs(test_raw_dir, exist_ok=True)
        logger.info(f"  Preparing TEST raw data...")
        prepare_jedi_raw_data(test_valid, test_gm, test_gi,
                              test_pos, test_neg, test_raw_dir, gene_padding)
        test_pos_raw = os.path.join(test_raw_dir, "pos_raw.json")
        test_neg_raw = os.path.join(test_raw_dir, "neg_raw.json")
        if not convert_to_jedi_format(jedi_dir, test_pos_raw, test_neg_raw, test_path, k, L, python_cmd):
            import shutil
            shutil.copy(train_path, test_path)
    else:
        import shutil
        shutil.copy(train_path, test_path)

    # ── 4. Run JEDI training ──────────────────────────────────────────────
    logger.info(f"  Running JEDI training ({epochs} epochs)...")
    jedi_out_dir = os.path.join(group_out_dir, "jedi_output")
    jedi_metrics = run_jedi_training(jedi_dir, jedi_data_dir, jedi_out_dir,
                                     group_name, k, L, epochs, batch_size, python_cmd)

    if jedi_metrics is None:
        logger.error(f"  JEDI training failed for {group_name}")
    else:
        logger.info(f"  JEDI training complete.")
        pred_file = os.path.join(jedi_out_dir, f"pred.0.K{k}.L{L}")
        if os.path.isfile(pred_file):
            am = compute_auroc_auprc_from_predictions(pred_file)
            jedi_metrics.update(am)
            logger.info(f"  Test AUROC: {am.get('auroc', 'N/A'):.4f}, "
                       f"AUPRC: {am.get('auprc', 'N/A'):.4f}")

    # ── 5. Save summary ──────────────────────────────────────────────────
    train_pos_count = sum(len(train_pos.get(e.strain, {})) for e in train_valid)
    train_neg_count = sum(len(train_neg.get(e.strain, [])) for e in train_valid)
    test_pos_count = sum(len(test_pos.get(e.strain, {})) for e in test_entries) if test_entries else 0
    test_neg_count = sum(len(test_neg.get(e.strain, [])) for e in test_entries) if test_entries else 0

    summary_path = os.path.join(group_out_dir, "summary.json")
    results = {
        "group": group_name,
        "train_strains": [e.strain for e in train_valid],
        "test_strains": [e.strain for e in test_entries] if test_entries else [],
        "n_train_pos": train_pos_count,
        "n_train_neg": train_neg_count,
        "n_test_pos": test_pos_count,
        "n_test_neg": test_neg_count,
        "jedi_hyperparameters": {"K": k, "L": L, "epochs": epochs, "batch_size": batch_size},
        "jedi_metrics": jedi_metrics or {},
    }
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Summary saved to {summary_path}")

    for gi in {**train_gi, **test_gi}.values():
        gi.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="PanCirc-Fungi x JEDI comparison")
    parser.add_argument("--tsv", default="all_lib_model_full.tsv")
    parser.add_argument("--group", choices=GROUPS + ["all"], default="all")
    parser.add_argument("--jedi-dir", default="other_models/JEDI-master")
    parser.add_argument("--out-dir", default="results/jedi_comparison")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--python-cmd", default=None)
    parser.add_argument("--gene-padding", type=int, default=SEQUENCE_PADDING)

    args = parser.parse_args()
    python_cmd = args.python_cmd or sys.executable

    tsv_path = os.path.join(PROJ_ROOT, args.tsv) if not os.path.isabs(args.tsv) else args.tsv
    entries = parse_strain_registry(tsv_path)

    jedi_dir = os.path.join(PROJ_ROOT, args.jedi_dir) if not os.path.isabs(args.jedi_dir) else args.jedi_dir
    out_dir = os.path.join(PROJ_ROOT, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    groups_to_process = GROUPS if args.group == "all" else [args.group]

    all_results = {}
    for group in groups_to_process:
        r = process_group(entries, group, jedi_dir, out_dir, args.k, args.L,
                          args.epochs, args.batch_size, python_cmd, args.gene_padding)
        if r:
            all_results[group] = r

    # Summary
    print("\n" + "=" * 60)
    print("JEDI Comparison Summary")
    print("=" * 60)
    for group, result in all_results.items():
        m = result.get("jedi_metrics", {})
        print(f"\n  {group}:")
        print(f"    Train: {', '.join(result['train_strains'])}")
        print(f"    Test:  {', '.join(result['test_strains']) if result['test_strains'] else 'N/A'}")
        print(f"    Genes: {result['n_train_pos']} pos + {result['n_train_neg']} neg")
        print(f"    AUROC: {m.get('auroc', 'N/A')}")
        print(f"    AUPRC: {m.get('auprc', 'N/A')}")
        print(f"    F1:    {m.get('test_f1', 'N/A')}")

    summary_path = os.path.join(out_dir, "all_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {summary_path}")


if __name__ == "__main__":
    main()
