#!/usr/bin/env python3
"""A4 (revision): genome-wide (naturally imbalanced) evaluation — thresholded
metrics, precision/FDR, sensitivity/specificity, calibration (ECE/Brier,
reliability curves).

Responds to: R1-M4 + R3-2 (balanced AUROC is not enough; practical imbalanced
performance and calibration at the mycoCircAtlas threshold p>=0.8 required).

Ground truth:
  * 3 external species with multi-tool detection: gene-level max tool support
    from circ/External/{condition}/{condition}_merged_weighted_1based.tsv,
    mapped circ_id->gene_id via the CIRIquant merge info (logic replicated
    from scripts/roc_external.py).
  * T. marneffei (PM1): CIRIquant circTS gene ids (exon/intron circRNAs).

Predictions: genome-wide per-gene p_circ from predictions_external/
(one TSV per checkpoint variant; unsuffixed file = Filamentous).

Outputs (revision/results/):
  a4_genomewide_roc_prc.csv   — AUROC/AUPRC + bootstrap CI per stringency
  a4_threshold_metrics.csv    — TP/FP/FN/TN, precision, FDR, sens, spec, F1,
                                enrichment at p in {0.5,0.6,0.7,0.8,0.9}
  a4_calibration.csv          — ECE, Brier, per-bin reliability data
  a4_summary.md               — human-readable summary for the response letter

Run:
  micromamba run -n pancirc-fungi python revision/scripts/a4_threshold_calibration.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

PROJ = Path(__file__).resolve().parents[2]
OUT = PROJ / "revision" / "results"
OUT.mkdir(parents=True, exist_ok=True)

PRED_DIR = PROJ / "predictions_external"
WORKDIR = Path("/media/share/workdir/1610305236/Panfungi/circ/External")
CONFIG_PATH = "/media/share/data5/1610305236/Panfungi/1_data/configs/exte_lib_config.tsv"
MERGE_DIR = "/media/share/data5/1610305236/Panfungi/1_data/External/merge"
PM1_CIRCINFO = "/media/share/data5/1610305236/circTS/3_CIRIquant/circTS_circRNA_info.csv"

CHECKPOINTS = ["Candida", "Cryptococcus", "Filamentous"]
STRINGENCY = [1, 2, 3, 4]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
N_BOOT, BOOT_SEED = 1000, 2026


# ── GT loading (replicated from scripts/roc_external.py) ────────────────────

def load_circ_to_gene(species):
    info_path = Path(MERGE_DIR) / species / f"{species}_circRNA_info.csv"
    mapping = {}
    if not info_path.is_file():
        return mapping
    with open(info_path) as f:
        for row in csv.DictReader(f):
            circ_id = row["circ_id"].strip("\"'")
            gene_id = row["gene_id"].strip("\"'")
            if gene_id and gene_id != "NA":
                mapping[circ_id] = gene_id
    return mapping


def load_gene_support(species):
    """Per-gene max multi-tool support across all conditions of a species."""
    conditions = []
    with open(CONFIG_PATH) as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if cols and cols[0] == species:
                conditions.append(cols[1])
    circ_to_gene = load_circ_to_gene(species)
    gene_info = defaultdict(lambda: {"max_support": 0, "max_bsj": 0.0})
    for cond in conditions:
        tsv = WORKDIR / cond / f"{cond}_merged_weighted_1based.tsv"
        if not tsv.is_file():
            continue
        with open(tsv) as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 7:
                    continue
                circ_id = parts[0]
                support = int(parts[6])
                gene_id = circ_to_gene.get(circ_id, "")
                if not gene_id:
                    continue
                gi = gene_info[gene_id]
                gi["max_support"] = max(gi["max_support"], support)
                bsj = max(float(x) if x else 0 for x in parts[2:6])
                gi["max_bsj"] = max(gi["max_bsj"], bsj)
    return gene_info


def load_pm1_positives():
    """PM1 GT: CIRIquant gene ids (exon/intron only, as in training filters)."""
    df = pd.read_csv(PM1_CIRCINFO)
    df = df[df["circ_type"].isin(["exon", "intron"])]
    df = df[df["gene_id"].notna() & (df["gene_id"].astype(str) != "NA")]
    return set(df["gene_id"].astype(str))


def load_predictions(species, ckpt):
    name = f"{species}_{ckpt}.tsv"
    path = PRED_DIR / name
    if not path.exists():
        path = PRED_DIR / f"{species}.tsv"  # unsuffixed = Filamentous
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df["gene_id"], df["p_circ"]))


# ── metrics helpers ──────────────────────────────────────────────────────────

def bootstrap_ci(y, p, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    aurocs, auprcs = [], []
    for _ in range(n_boot):
        sel = rng.choice(idx, size=len(idx), replace=True)
        yt, yp = y[sel], p[sel]
        if len(np.unique(yt)) < 2:
            continue
        aurocs.append(roc_auc_score(yt, yp))
        auprcs.append(average_precision_score(yt, yp))
    lo_a, hi_a = np.percentile(aurocs, [2.5, 97.5])
    lo_p, hi_p = np.percentile(auprcs, [2.5, 97.5])
    return (float(lo_a), float(hi_a)), (float(lo_p), float(hi_p))


def calibration(y, p, n_bins=10):
    """ECE + Brier + reliability bins (equal-width)."""
    y = np.asarray(y, int); p = np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece, wsum = [], 0.0, len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            rows.append(dict(bin=f"[{lo:.1f},{hi:.1f})", n=0))
            continue
        conf = p[m].mean()
        frac = y[m].mean()
        ece += (m.sum() / wsum) * abs(conf - frac)
        rows.append(dict(bin=f"[{lo:.1f},{hi:.1f}]", n=int(m.sum()),
                         mean_pred=float(conf), obs_rate=float(frac)))
    brier = float(np.mean((p - y) ** 2))
    return float(ece), brier, rows


def main():
    roc_rows, thr_rows, cal_rows = [], [], []
    summary_lines = ["# A4 genome-wide (imbalanced) evaluation summary", ""]

    jobs = []
    for sp in ["Aspergillus_cristatus", "Trichophyton_mentagrophytes",
               "Trichophyton_rubrum"]:
        jobs.append((sp, "multitool", load_gene_support(sp)))
    pm1_pos = load_pm1_positives()
    jobs.append(("Talaromyces_marneffei", "ciriquant", pm1_pos))

    for sp, gt_kind, gt in jobs:
        print(f"\n=== {sp} ({gt_kind}) ===")
        for ckpt in CHECKPOINTS:
            preds = load_predictions(sp, ckpt)
            if gt_kind == "multitool":
                all_genes = sorted(set(preds) | set(gt))
                support = {g: gt.get(g, {"max_support": 0})["max_support"]
                           for g in all_genes}
            else:
                all_genes = sorted(set(preds) | set(gt))
                support = {g: (1 if g in gt else 0) for g in all_genes}
            p_vec = np.array([preds.get(g, 0.0) for g in all_genes])

            for k in STRINGENCY:
                if gt_kind == "multitool":
                    y_vec = np.array([1 if support[g] >= k else 0
                                      for g in all_genes])
                elif k == 1:
                    y_vec = np.array([support[g] for g in all_genes])
                else:
                    continue  # single-tool GT has no stringency levels
                n_pos = int(y_vec.sum())
                if n_pos == 0 or n_pos == len(y_vec):
                    continue
                auc = roc_auc_score(y_vec, p_vec)
                auprc = average_precision_score(y_vec, p_vec)
                (alo, ahi), (plo, phi) = bootstrap_ci(y_vec, p_vec)
                roc_rows.append(dict(
                    species=sp, gt=gt_kind, stringency=f">={k}", ckpt=ckpt,
                    n_genes=len(all_genes), n_pos=n_pos,
                    prevalence=round(n_pos / len(all_genes), 5),
                    auroc=round(float(auc), 4),
                    auroc_ci=f"[{alo:.3f},{ahi:.3f}]",
                    auprc=round(float(auprc), 4),
                    auprc_ci=f"[{plo:.3f},{phi:.3f}]"))

                for t in THRESHOLDS:
                    pred_pos = p_vec >= t
                    tp = int((pred_pos & (y_vec == 1)).sum())
                    fp = int((pred_pos & (y_vec == 0)).sum())
                    fn = int((~pred_pos & (y_vec == 1)).sum())
                    tn = int((~pred_pos & (y_vec == 0)).sum())
                    prec = tp / (tp + fp) if tp + fp else np.nan
                    sens = tp / (tp + fn) if tp + fn else np.nan
                    spec = tn / (tn + fp) if tn + fp else np.nan
                    fdr = 1 - prec if prec == prec else np.nan
                    prev = n_pos / len(all_genes)
                    thr_rows.append(dict(
                        species=sp, gt=gt_kind, stringency=f">={k}",
                        ckpt=ckpt, thr=t,
                        tp=tp, fp=fp, fn=fn, tn=tn,
                        precision=round(float(prec), 4) if prec == prec else "",
                        fdr=round(float(fdr), 4) if fdr == fdr else "",
                        sensitivity=round(float(sens), 4) if sens == sens else "",
                        specificity=round(float(spec), 4) if spec == spec else "",
                        f1=round(2 * tp / (2 * tp + fp + fn), 4),
                        enrichment_vs_prev=(
                            round(float(prec / prev), 2)
                            if prec == prec and prev > 0 else "")))

                if ckpt == CHECKPOINTS[-1] and (gt_kind != "multitool" or k == 2):
                    ece, brier, bins = calibration(y_vec, p_vec)
                    for b in bins:
                        cal_rows.append(dict(species=sp, gt=gt_kind,
                                             stringency=f">={k}",
                                             ckpt=ckpt, **b))
                    cal_rows.append(dict(species=sp, gt=gt_kind,
                                         stringency=f">={k}", ckpt=ckpt,
                                         bin="ECE", n=N_BOOT,
                                         mean_pred=ece, obs_rate=brier))

    pd.DataFrame(roc_rows).to_csv(OUT / "a4_genomewide_roc_prc.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(OUT / "a4_threshold_metrics.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(OUT / "a4_calibration.csv", index=False)

    # human-readable highlight: p>=0.8, >=2 tools, all checkpoints
    df = pd.DataFrame(thr_rows)
    if len(df):
        hl = df[(df.thr == 0.8) & (df.stringency == ">=2")]
        summary_lines += ["## Precision at the atlas threshold (p>=0.8, "
                          "GT = detected by >=2 tools)", "",
                          "| species | ckpt | precision | enrichment | sens |",
                          "|---|---|---|---|---|"]
        for _, r in hl.iterrows():
            summary_lines.append(
                f"| {r.species} | {r.ckpt} | {r.precision} | "
                f"{r.enrichment_vs_prev}x | {r.sensitivity} |")
    (OUT / "a4_summary.md").write_text("\n".join(summary_lines) + "\n")
    print(f"\nDone → {OUT}")


if __name__ == "__main__":
    main()
