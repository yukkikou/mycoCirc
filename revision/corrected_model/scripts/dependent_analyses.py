#!/usr/bin/env python3
"""Re-run the analyses that depended on mycoCirc scores, using the CORRECTED
(CircExp-zeroed) model's genome-wide Mode A predictions.

  A2  matched-negative evaluation (3 draws, default caliper)
  A4  genome-wide threshold metrics + calibration against multi-tool GT

Inputs : revision/corrected_model/eval/scores/{species}_{ckpt}_A.tsv
Outputs: revision/corrected_model/eval/results/
           corrected_a2_matched_negative.csv
           corrected_a4_threshold_metrics.csv
           corrected_a4_calibration.csv
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ))
BASE = PROJ / "revision" / "corrected_model"
SCORES = BASE / "eval" / "scores"
RES = BASE / "eval" / "results"

from data.circ_info_encoding import (load_circ_info, filter_circ_info,  # noqa: E402
                                     get_positive_genes)
from data.negative_sampling import get_negative_genes  # noqa: E402
from utils.gtf_utils import GeneModelIndexer  # noqa: E402

MERGE = "/media/share/data5/1610305236/Panfungi/1_data/External/merge"
CFG = "/media/share/data5/1610305236/Panfungi/1_data/configs/exte_lib_config.tsv"
WORKDIR = Path("/media/share/workdir/1610305236/Panfungi/circ/External")
PM1_CI = "/media/share/data5/1610305236/circTS/3_CIRIquant/circTS_circRNA_info.csv"
PM1_GTF = ("/media/share/resource/reference/talaromyces_marneffei/"
           "annotation/pm1_scallop_final.gtf")
GROUPS = ["Candida", "Cryptococcus", "Filamentous"]
SPECIES = ["Aspergillus_cristatus", "Trichophyton_rubrum",
           "Trichophyton_mentagrophytes", "Talaromyces_marneffei"]
THRESHOLDS = [0.5, 0.8]
DRAWS = [42, 43, 44]


# ---------- helpers ----------
def spec_paths(sp):
    if sp == "Talaromyces_marneffei":
        return dict(gtf=PM1_GTF, circinfo=PM1_CI)
    with open(CFG) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[0] == sp:
                return dict(gtf=c[5], circinfo=f"{MERGE}/{sp}/{sp}_circRNA_info.csv")
    raise KeyError(sp)


def gene_table(gm, ids):
    out = []
    for g in ids:
        x = gm.get_gene(g)
        if x is None or x.exon_count < 1:
            continue
        L = max(int(sum(x.exon_ends) - sum(x.exon_starts) + x.exon_count), 1)
        out.append((g, np.log(L), float(x.exon_count)))
    return out


def match(pos, cand, seed, len_cal=0.25):
    rng = np.random.default_rng(seed)
    pl = np.array([p[1] for p in pos]); pc = np.array([p[2] for p in pos])
    cl = np.array([c[1] for c in cand]); cc = np.array([c[2] for c in cand])
    avail = np.ones(len(cand), bool); used = []
    for i in rng.permutation(len(pos)):
        d_len = np.abs(cl - pl[i])
        d_cnt = np.abs(cc - pc[i])
        d_cnt = np.where(d_cnt == 0, 0.0, np.where(pc[i] <= 3, 1e6, d_cnt))
        score = d_len + np.minimum(d_cnt, 1e6)
        ok = (d_len <= len_cal) & (d_cnt < 1e6)
        gated = np.where(ok & avail, score, np.inf)
        j = int(np.argmin(gated))
        if not np.isfinite(gated[j]):
            ung = np.where(avail, score, np.inf)
            if not np.isfinite(ung).any():
                continue
            j = int(np.argmin(ung))
        avail[j] = False
        used.append(cand[j][0])
    return used


def load_corrected_scores(sp, ck):
    p = SCORES / f"{sp}_{ck}_A.tsv"
    df = pd.read_csv(p, sep="\t")
    return dict(zip(df.gene_id, df.p_circ))


def load_gene_support(sp):
    conds = []
    with open(CFG) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c and c[0] == sp:
                conds.append(c[1])
    c2g = {}
    with open(f"{MERGE}/{sp}/{sp}_circRNA_info.csv") as f:
        for row in csv.DictReader(f):
            gid = row["gene_id"].strip("\"'")
            if gid and gid != "NA":
                c2g[row["circ_id"].strip("\"'")] = gid
    sup = defaultdict(int)
    for cond in conds:
        tsv = WORKDIR / cond / f"{cond}_merged_weighted_1based.tsv"
        if not tsv.is_file():
            continue
        with open(tsv) as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 7:
                    continue
                gid = c2g.get(parts[0], "")
                if gid:
                    sup[gid] = max(sup[gid], int(parts[6]))
    return sup


def main():
    a2, a4, cal = [], [], []
    for sp in SPECIES:
        paths = spec_paths(sp)
        gm = GeneModelIndexer(paths["gtf"])
        raw = load_circ_info(paths["circinfo"])
        filt = filter_circ_info(raw)
        pos_ids = list(get_positive_genes(filt))
        pos_set = set(pos_ids)
        pos = gene_table(gm, pos_ids)
        cand = gene_table(gm, [g for g in gm.genes if g not in pos_set])
        print(f"{sp}: pos={len(pos)} cand={len(cand)}", flush=True)

        for ck in GROUPS:
            sc = load_corrected_scores(sp, ck)
            # ---- A2: matched negatives, 3 draws ----
            aucs = []
            for seed in DRAWS:
                neg = match(pos, cand, seed)
                genes = [p[0] for p in pos] + neg
                y = np.array([1] * len(pos) + [0] * len(neg))
                pv = np.array([sc.get(g, 0.0) for g in genes])
                aucs.append(roc_auc_score(y, pv))
            a2.append(dict(species=sp, checkpoint=ck, recipe="nocircexp",
                           auroc_matched=round(float(np.mean(aucs)), 4),
                           sd_3draws=round(float(np.std(aucs)), 4)))

            # ---- A4: genome-wide vs detection GT ----
            if sp == "Talaromyces_marneffei":
                gt = {g: 1 for g in pos_ids}
                allg = sorted(set(sc) | set(gt))
                y = np.array([gt.get(g, 0) for g in allg])
                pv = np.array([sc.get(g, 0.0) for g in allg])
                levels = [1]
            else:
                sup = load_gene_support(sp)
                allg = sorted(set(sc) | set(sup))
                pv = np.array([sc.get(g, 0.0) for g in allg])
                levels = [1, 2, 3]
            for k in levels:
                if sp == "Talaromyces_marneffei":
                    yk = y
                else:
                    yk = np.array([1 if sup.get(g, 0) >= k else 0 for g in allg])
                if yk.sum() < 10 or yk.sum() == len(yk):
                    continue
                auc = roc_auc_score(yk, pv)
                ap = average_precision_score(yk, pv)
                row = dict(species=sp, checkpoint=ck, gt=("ciriquant" if
                           sp == "Talaromyces_marneffei" else "multitool"),
                           stringency=k, n=len(yk), n_pos=int(yk.sum()),
                           prevalence=round(float(yk.mean()), 5),
                           auroc=round(float(auc), 4),
                           auprc=round(float(ap), 4))
                for t in THRESHOLDS:
                    pos_pred = pv >= t
                    tp = int((pos_pred & (yk == 1)).sum())
                    fp = int((pos_pred & (yk == 0)).sum())
                    fn = int((~pos_pred & (yk == 1)).sum())
                    tn = int((~pos_pred & (yk == 0)).sum())
                    row[f"prec@{t}"] = round(tp / (tp + fp), 4) if tp + fp else np.nan
                    row[f"sens@{t}"] = round(tp / (tp + fn), 4) if tp + fn else np.nan
                    row[f"spec@{t}"] = round(tn / (tn + fp), 4) if tn + fp else np.nan
                a4.append(row)
                # calibration at the most reliable GT level
                if k == levels[-1]:
                    bins = np.linspace(0, 1, 11)
                    ece = 0.0
                    for i in range(10):
                        m = (pv >= bins[i]) & (pv < bins[i + 1] if i < 9 else pv <= 1)
                        if m.sum():
                            ece += (m.sum() / len(pv)) * abs(pv[m].mean() - yk[m].mean())
                    cal.append(dict(species=sp, checkpoint=ck,
                                    gt=row["gt"], stringency=k,
                                    ece=round(float(ece), 4),
                                    brier=round(float(np.mean((pv - yk) ** 2)), 4)))

    pd.DataFrame(a2).to_csv(RES / "corrected_a2_matched_negative.csv", index=False)
    pd.DataFrame(a4).to_csv(RES / "corrected_a4_threshold_metrics.csv", index=False)
    pd.DataFrame(cal).to_csv(RES / "corrected_a4_calibration.csv", index=False)
    print("\n=== A2 matched-negative (corrected) ===")
    print(pd.DataFrame(a2).pivot(index="species", columns="checkpoint",
                                 values="auroc_matched").round(3).to_string())
    print("\n=== A4 calibration (corrected) ===")
    print(pd.DataFrame(cal).to_string(index=False))


if __name__ == "__main__":
    main()
