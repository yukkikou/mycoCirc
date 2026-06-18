"""
PanCirc-Fungi × CircPCBL Comparison
====================================
Extract fungal gene sequences in FASTA format, run CircPCBL's pretrained
Plant model, and evaluate predictions vs ground truth labels.

Usage:
    # Zero-shot (pretrained Plant model on fungal test strains)
    python scripts/run_circpcbl_comparison.py --tsv all_lib_model_full.tsv \\
        --group Candida --mode zero-shot --out-dir results/circpcbl

    # Train from scratch on fungal data
    python scripts/run_circpcbl_comparison.py --tsv all_lib_model_full.tsv \\
        --group Candida --mode train --out-dir results/circpcbl

    # Or via SLURM array:
    sbatch --array=1-3 scripts/run_circpcbl.slurm
"""

import argparse
import json
import logging
import os
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
    get_group_members,
    TEST_STRAINS,
    TRAIN_STRAINS,
    EXCLUDED_STRAINS,
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
SEQUENCE_LENGTH = 1500  # CircPCBL fixed sequence length
HALF_LENGTH = SEQUENCE_LENGTH // 2  # 750bp on each side

GROUPS = ["Candida", "Cryptococcus", "Filamentous"]
MODES = ["zero-shot", "train"]


def extract_gene_sequence(genome_indexer, gene, half_window=HALF_LENGTH):
    """Extract gene sequence centered on gene body, truncated/padded to 1500bp.

    Returns the sequence as string (always SEQUENCE_LENGTH bp).
    """
    gene_mid = (gene.start + gene.end) // 2
    half_w = half_window

    seq_start = max(1, gene_mid - half_w)
    seq_end = gene_mid + half_w

    seq = genome_indexer.get_seq(gene.chrom, seq_start, seq_end, gene.strand)

    if not seq:
        return "N" * SEQUENCE_LENGTH

    if len(seq) < SEQUENCE_LENGTH:
        seq = seq + "N" * (SEQUENCE_LENGTH - len(seq))
    elif len(seq) > SEQUENCE_LENGTH:
        seq = seq[:SEQUENCE_LENGTH]

    return seq


def write_fasta(gene_ids, sequences, labels, output_path):
    """Write sequences in FASTA format with labels in header.

    Sequences are uppercased; only A/C/G/T/U/N characters expected.
    """
    with open(output_path, "w") as f:
        for gid, seq, label in zip(gene_ids, sequences, labels):
            seq = seq.upper()
            f.write(f">{gid}|label={label}\n")
            # Write sequence in 80-char lines
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")


def run_circpcbl_pretrained(
    circpcbl_dir,
    fasta_path,
    output_path,
    model_type="Plant",
    batch_size=16,
    python_cmd=sys.executable,
):
    """Run CircPCBL's pretrained model on a FASTA file.

    model_type: "Plant" or "Animal"
    """
    if model_type == "Plant":
        script = os.path.join(circpcbl_dir, "Model", "Plant_GPU.py")
        param = os.path.join(circpcbl_dir, "Param", "Plant.pkl")
    else:
        script = os.path.join(circpcbl_dir, "Model", "Animal_GPU.py")
        param = os.path.join(circpcbl_dir, "Param", "Animal.pkl")

    if not os.path.isfile(script):
        logger.error(f"  CircPCBL script not found: {script}")
        return None

    if not os.path.isfile(param):
        logger.error(f"  CircPCBL params not found: {param}")
        return None

    cmd = [
        python_cmd, script,
        f"--input={fasta_path}",
        f"--output={output_path}",
        f"--batch_size={batch_size}",
    ]

    logger.info(f"  Running: {' '.join(cmd)}")
    # CircPCBL uses cuda directly; if cuda not available, try CPU version
    env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                            cwd=os.path.join(circpcbl_dir, "Model"))

    if result.returncode != 0:
        # Try CPU version
        logger.warning(f"  GPU version failed, trying CPU: {result.stderr[:200]}")
        cpu_script = os.path.join(circpcbl_dir, "Model", f"{model_type}_CPU.py")
        if os.path.isfile(cpu_script):
            cmd[0] = python_cmd
            cmd[1] = cpu_script
            result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                    cwd=os.path.join(circpcbl_dir, "Model"))

    if result.returncode != 0:
        logger.error(f"  CircPCBL failed:\n{result.stderr}")
        return None

    logger.info(f"  CircPCBL output: {output_path}")
    return output_path


def load_circpcbl_results(result_csv_path, gene_ids, labels):
    """Load CircPCBL predictions and compute metrics vs ground truth."""
    import pandas as pd

    try:
        df = pd.read_csv(result_csv_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        logger.error(f"  Cannot read CircPCBL results: {result_csv_path}")
        return None

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Verify required columns exist
    required_cols = ["Index", "Predict_result"]
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"  Missing expected column '{col}' in CircPCBL output; "
                         f"found columns: {list(df.columns)}")
            return None

    # CircPCBL output has: Index, Predict_result (circRNA/lncRNA)
    if len(df) != len(gene_ids):
        logger.warning(f"  Mismatch: {len(df)} predictions vs {len(gene_ids)} genes")
        df = df.iloc[:len(gene_ids)]

    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        accuracy_score, f1_score, matthews_corrcoef,
        precision_score, recall_score, confusion_matrix,
    )

    # CircPCBL predicts class directly (not probability)
    preds = np.array([1 if r.strip() == "circRNA" else 0 for r in df["Predict_result"]])
    true = np.array(labels[:len(preds)])

    unique_true = np.unique(true)
    unique_pred = np.unique(preds)

    metrics = {
        "n_samples": int(len(true)),
        "n_positive": int(true.sum()),
        "n_predicted_circrna": int(preds.sum()),
    }

    if len(unique_true) < 2 or len(unique_pred) < 2:
        logger.warning("  Only one class in predictions or labels")
        metrics["accuracy"] = float(accuracy_score(true, preds))
        return metrics

    metrics["accuracy"] = float(accuracy_score(true, preds))
    metrics["precision"] = float(precision_score(true, preds, zero_division=0))
    metrics["recall"] = float(recall_score(true, preds, zero_division=0))
    metrics["f1"] = float(f1_score(true, preds, zero_division=0))
    metrics["mcc"] = float(matthews_corrcoef(true, preds))

    # For AUROC/AUPRC, we use predicted class as probability (since CircPCBL
    # only outputs class, not probability). This gives a lower bound estimate.
    metrics["auroc_approx"] = float(roc_auc_score(true, preds))
    metrics["auprc_approx"] = float(average_precision_score(true, preds))

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(true, preds).ravel()
    metrics["specificity"] = float(tn / max(tn + fp, 1))
    metrics["sensitivity"] = float(tp / max(tp + fn, 1))

    return metrics


def prepare_and_run_zero_shot(
    circpcbl_dir,
    entries,
    positive_genes,
    negative_genes,
    gene_models,
    genome_indexers,
    output_dir,
    group_name,
    batch_size=16,
    python_cmd=sys.executable,
    half_window=HALF_LENGTH,
):
    """Zero-shot evaluation: run pretrained CircPCBL Plant model on test strains."""
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    all_gene_ids = []
    all_sequences = []
    all_labels = []
    all_strains = []

    for entry in entries:
        strain = entry.strain
        gm = gene_models.get(strain)
        gi = genome_indexers.get(strain)
        if gm is None or gi is None:
            continue

        # Collect positive and negative genes
        strain_pos = positive_genes.get(strain, {})
        strain_neg = negative_genes.get(strain, [])

        strain_seq = []
        strain_gid = []
        strain_lbl = []

        for gene_id in strain_pos:
            gene = gm.get_gene(gene_id)
            if gene is None:
                continue
            seq = extract_gene_sequence(gi, gene, half_window)
            strain_seq.append(seq)
            strain_gid.append(gene_id)
            strain_lbl.append(1)

        for gene_id in strain_neg:
            gene = gm.get_gene(gene_id)
            if gene is None:
                continue
            seq = extract_gene_sequence(gi, gene, half_window)
            strain_seq.append(seq)
            strain_gid.append(gene_id)
            strain_lbl.append(0)

        if not strain_seq:
            logger.info(f"    {strain}: no genes to classify")
            continue

        fasta_path = os.path.join(output_dir, f"{strain}.fasta")
        result_csv = os.path.join(output_dir, f"{strain}_result.csv")

        write_fasta(strain_gid, strain_seq, strain_lbl, fasta_path)
        logger.info(f"  {strain}: {len(strain_seq)} genes -> {fasta_path}")

        # Run CircPCBL
        output = run_circpcbl_pretrained(
            circpcbl_dir, fasta_path, result_csv,
            model_type="Plant", batch_size=batch_size,
            python_cmd=python_cmd,
        )

        if output is None:
            logger.warning(f"  CircPCBL failed for {strain}")
            continue

        # Evaluate
        strain_metrics = load_circpcbl_results(result_csv, strain_gid, strain_lbl)
        if strain_metrics:
            strain_metrics["strain"] = strain
            strain_metrics["n_pos"] = sum(strain_lbl)
            strain_metrics["n_neg"] = len(strain_lbl) - sum(strain_lbl)
            all_results[strain] = strain_metrics

            logger.info(f"    {strain}: Acc={strain_metrics.get('accuracy', 'N/A'):.4f}, "
                       f"F1={strain_metrics.get('f1', 'N/A'):.4f}, "
                       f"AUROC≈{strain_metrics.get('auroc_approx', 'N/A'):.4f}")

        all_gene_ids.extend(strain_gid)
        all_sequences.extend(strain_seq)
        all_labels.extend(strain_lbl)
        all_strains.extend([strain] * len(strain_gid))

    # Overall metrics for the group
    if all_labels:
        overall_fasta = os.path.join(output_dir, f"{group_name}_all.fasta")
        overall_csv = os.path.join(output_dir, f"{group_name}_all_result.csv")
        write_fasta(all_gene_ids, all_sequences, all_labels, overall_fasta)

        output = run_circpcbl_pretrained(
            circpcbl_dir, overall_fasta, overall_csv,
            model_type="Plant", batch_size=batch_size,
            python_cmd=python_cmd,
        )

        if output:
            overall_metrics = load_circpcbl_results(overall_csv, all_gene_ids, all_labels)
            if overall_metrics:
                all_results["overall"] = overall_metrics

    return all_results


def prepare_and_run_training(
    circpcbl_dir,
    entries,
    positive_genes,
    negative_genes,
    gene_models,
    genome_indexers,
    output_dir,
    group_name,
    batch_size=16,
    python_cmd=sys.executable,
    half_window=HALF_LENGTH,
):
    """Train CircPCBL from scratch on fungal training strains.

    This requires modifying CircPCBL to support training on new data.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Prepare training FASTA ─────────────────────────────────────────────
    train_seqs = []
    train_labels = []
    train_ids = []

    for entry in entries:
        strain = entry.strain
        gm = gene_models.get(strain)
        gi = genome_indexers.get(strain)
        if gm is None or gi is None:
            continue

        strain_pos = positive_genes.get(strain, {})
        strain_neg = negative_genes.get(strain, [])

        for gene_id in strain_pos:
            gene = gm.get_gene(gene_id)
            if gene is None:
                continue
            seq = extract_gene_sequence(gi, gene, half_window)
            train_seqs.append(seq)
            train_ids.append(gene_id)
            train_labels.append(1)

        for gene_id in strain_neg:
            gene = gm.get_gene(gene_id)
            if gene is None:
                continue
            seq = extract_gene_sequence(gi, gene, half_window)
            train_seqs.append(seq)
            train_ids.append(gene_id)
            train_labels.append(0)

    train_fasta = os.path.join(output_dir, f"{group_name}_train.fasta")
    write_fasta(train_ids, train_seqs, train_labels, train_fasta)
    logger.info(f"  Training set: {len(train_seqs)} genes -> {train_fasta}")

    # ── Create training ground truth CSV ────────────────────────────────────
    import pandas as pd
    train_gt = pd.DataFrame({
        "gene_id": train_ids,
        "label": train_labels,
        "sequence": train_seqs,
    })
    train_gt.to_csv(os.path.join(output_dir, f"{group_name}_train_labels.csv"), index=False)

    # ── Note on training from scratch ──────────────────────────────────────
    logger.info(f"  {'='*50}")
    logger.info(f"  CircPCBL 'train' mode: generating training data only.")
    logger.info(f"  To train from scratch, modify CircPCBL to use:")
    logger.info(f"    {train_fasta}")
    logger.info(f"  and the corresponding labels file. This requires")
    logger.info(f"  adapting CircPCBL's training loop (Animal_GPU.py)")
    logger.info(f"  to accept custom training data.")
    logger.info(f"  {'='*50}")

    return {"status": "training_data_prepared", "train_fasta": train_fasta}


def process_group(
    entries,
    group_name,
    circpcbl_dir,
    mode,
    output_base_dir,
    batch_size=16,
    python_cmd=sys.executable,
    half_window=HALF_LENGTH,
):
    """Process one group for CircPCBL comparison."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing group: {group_name} (mode: {mode})")
    logger.info(f"{'='*60}")

    group_members = [e for e in entries if e.group == group_name and not e.is_excluded]
    train_strains = TRAIN_STRAINS.get(group_name, set())
    test_strains = TEST_STRAINS.get(group_name, set())

    train_entries = [e for e in group_members if e.strain in train_strains]
    test_entries = [e for e in group_members if e.strain in test_strains]

    if not train_entries and mode == "train":
        logger.error(f"  No training entries for {group_name}")
        return None

    logger.info(f"  Train strains: {[e.strain for e in train_entries]}")
    logger.info(f"  Test strains:  {[e.strain for e in test_entries]}")

    # ── Build gene models and genome indexers ──────────────────────────────
    gene_models = {}
    genome_indexers = {}
    positive_genes = {}
    negative_genes = {}

    # For zero-shot, only need test strains
    # For train, need both train and test
    target_entries = test_entries if mode == "zero-shot" else group_members

    for entry in target_entries:
        strain = entry.strain
        logger.info(f"  Loading {strain}...")

        try:
            gm = GeneModelIndexer(entry.gtf_path)
            gene_models[strain] = gm
        except Exception as e:
            logger.warning(f"    Failed to load GTF for {strain}: {e}")
            continue

        try:
            gi = GenomeIndexer(entry.genome_path)
            genome_indexers[strain] = gi
        except Exception as e:
            logger.warning(f"    Failed to load genome for {strain}: {e}")
            continue

        # Positive genes
        circ_df = load_circ_info(entry.circinfo_path)
        circ_df = filter_circ_info(circ_df)
        pos = get_positive_genes(circ_df)
        positive_genes[strain] = pos
        n_pos = len(pos)

        if n_pos == 0:
            logger.info(f"    {strain}: 0 usable circRNA genes")
            if mode == "train":
                continue

        # Negative genes (balanced)
        if n_pos > 0 and gm is not None:
            neg = get_negative_genes(
                gm, set(pos.keys()), n_pos, rng_seed=42
            )
            negative_genes[strain] = neg
            logger.info(f"    {strain}: {n_pos} pos, {len(neg)} neg")

    group_out_dir = os.path.join(output_base_dir, group_name)
    os.makedirs(group_out_dir, exist_ok=True)

    if mode == "zero-shot":
        results = prepare_and_run_zero_shot(
            circpcbl_dir, test_entries,
            positive_genes, negative_genes,
            gene_models, genome_indexers,
            group_out_dir, group_name,
            batch_size=batch_size,
            python_cmd=python_cmd,
            half_window=half_window,
        )
    else:
        results = prepare_and_run_training(
            circpcbl_dir, train_entries,
            positive_genes, negative_genes,
            gene_models, genome_indexers,
            group_out_dir, group_name,
            batch_size=batch_size,
            python_cmd=python_cmd,
            half_window=half_window,
        )

    # Save summary
    summary_path = os.path.join(group_out_dir, "summary.json")
    summary = {
        "group": group_name,
        "mode": mode,
        "train_strains": [e.strain for e in train_entries],
        "test_strains": [e.strain for e in test_entries],
        "results": results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Summary saved to {summary_path}")

    for gi in genome_indexers.values():
        gi.close()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="PanCirc-Fungi × CircPCBL comparison"
    )
    parser.add_argument("--tsv", default="all_lib_model_full.tsv",
                        help="Strain registry TSV")
    parser.add_argument("--group", choices=GROUPS + ["all"], default="all",
                        help="Group to process")
    parser.add_argument("--mode", choices=MODES, default="zero-shot",
                        help="Comparison mode: zero-shot (pretrained Plant model) or train")
    parser.add_argument("--circpcbl-dir", default="other_models/CircPCBL-main",
                        help="CircPCBL-main directory")
    parser.add_argument("--out-dir", default="results/circpcbl_comparison",
                        help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--python-cmd", default=sys.executable,
                        help="Python command (default: sys.executable)")
    parser.add_argument("--half-window", type=int, default=HALF_LENGTH,
                        help=f"Half window for sequence extraction (default: {HALF_LENGTH})")

    args = parser.parse_args()

    tsv_path = os.path.join(PROJ_ROOT, args.tsv) if not os.path.isabs(args.tsv) else args.tsv
    entries = parse_strain_registry(tsv_path)

    circpcbl_dir = os.path.join(PROJ_ROOT, args.circpcbl_dir) if not os.path.isabs(args.circpcbl_dir) else args.circpcbl_dir
    out_dir = os.path.join(PROJ_ROOT, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    groups_to_process = GROUPS if args.group == "all" else [args.group]

    all_results = {}
    for group in groups_to_process:
        result = process_group(
            entries, group, circpcbl_dir,
            mode=args.mode,
            output_base_dir=out_dir,
            batch_size=args.batch_size,
            python_cmd=args.python_cmd,
            half_window=args.half_window,
        )
        if result:
            all_results[group] = result

    # Summary table
    print("\n" + "=" * 60)
    print(f"CircPCBL Comparison Summary (mode: {args.mode})")
    print("=" * 60)

    for group, result in all_results.items():
        results_dict = result.get("results", {})
        print(f"\n  {group}:")

        if args.mode == "zero-shot":
            for strain, metrics in results_dict.items():
                if strain == "overall":
                    continue
                if isinstance(metrics, dict):
                    print(f"    {strain}: Acc={metrics.get('accuracy', 'N/A'):.4f}, "
                          f"F1={metrics.get('f1', 'N/A'):.4f}, "
                          f"AUROC≈{metrics.get('auroc_approx', 'N/A'):.4f}")
            overall = results_dict.get("overall", {})
            if overall:
                print(f"    Overall: Acc={overall.get('accuracy', 'N/A'):.4f}, "
                      f"F1={overall.get('f1', 'N/A'):.4f}, "
                      f"AUROC≈{overall.get('auroc_approx', 'N/A'):.4f}")
        else:
            status = results_dict.get("status", "N/A")
            print(f"    Status: {status}")

    summary_path = os.path.join(out_dir, "all_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
