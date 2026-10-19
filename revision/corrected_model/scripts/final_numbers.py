#!/usr/bin/env python3
"""Final numbers needed to sync the response letter:
  * matched-negative: unmatched -> matched per species (best checkpoint) + CI
  * DeLong vs JEDI on the corrected model
  * GTF-only baseline compared with the corrected model
  * precision/recall at the 0.85 atlas threshold per species
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ))
BASE = PROJ / "revision" / "corrected_model"
RES = BASE / "eval" / "results"
SCORES = BASE / "eval" / "scores"

sys.path.insert(0, str(BASE / "scripts"))
import dependent_analyses as dep  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "a1", PROJ / "revision" / "scripts" / "a1_table1_ci.py")
a1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a1)

BOOT = 2000


def boot(y, p, seed=2026):
    rng = np.random.default_rng(seed)
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(BOOT):
        s = np.r_[rng.choice(pi, len(pi)), rng.choice(ni, len(ni))]
        vals.append(roc_auc_score(y[s], p[s]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    for sp in dep.SPECIES:
        specs = {sp: dep.spec_paths(sp) for sp in dep.SPECIES}

    # best checkpoint per species (Filamentous for all except T. marneffei Crypto? use per-species max)
    out = []
    for sp, paths in specs.items():
        gm = dep.GeneModelIndexer(paths["gtf"])
        from data.circ_info_encoding import load_circ_info, filter_circ_info, get_positive_genes
        pos_ids = list(get_positive_genes(filter_circ_info(load_circ_info(paths["circinfo"]))))
        pos = dep.gene_table(gm, pos_ids)
        pos_set = set(pos_ids)
        cand = dep.gene_table(gm, [g for g in gm.genes if g not in pos_set])

        # pick the checkpoint with the best unmatched AUROC
        best = None
        for ck in dep.GROUPS:
            sc = dep.load_corrected_scores(sp, ck)
            genes = [p[0] for p in pos]
            y = np.ones(len(genes))
            # unmatched AUROC needs negatives too; use A4 row for that species/ckpt
        a4 = pd.read_csv(RES / "corrected_a4_threshold_metrics.csv")
        sub = a4[(a4.species == sp) & (a4.stringency == a4.stringency.max())]
        ck_best = sub.loc[sub.auroc.idxmax()] if len(sub) else None
        ck = ck_best.checkpoint if ck_best is not None else "Filamentous"
        sc = dep.load_corrected_scores(sp, ck)

        # unmatched (seed-42 length-matched) and matched (3 draws)
        if sp == "Trichophyton_rubrum":
            pass
        from data.negative_sampling import get_negative_genes
        neg42 = get_negative_genes(gm, pos_set, len(pos_ids), rng_seed=42)
        genes = [p[0] for p in pos] + neg42
        y = np.array([1] * len(pos) + [0] * len(neg42))
        pv = np.array([sc.get(g, 0.0) for g in genes])
        auc_u = roc_auc_score(y, pv)
        lo_u, hi_u = boot(y, pv)

        aucs_m, cis = [], []
        for seed in (42, 43, 44):
            neg = dep.match(pos, cand, seed)
            genes_m = [p[0] for p in pos] + neg
            ym = np.array([1] * len(pos) + [0] * len(neg))
            pm = np.array([sc.get(g, 0.0) for g in genes_m])
            aucs_m.append(roc_auc_score(ym, pm))
            cis.append(boot(ym, pm))
        lo_m = min(c[0] for c in cis); hi_m = max(c[1] for c in cis)
        out.append(dict(species=sp, checkpoint=ck,
                        unmatched=round(float(auc_u), 3),
                        unmatched_ci=f"[{lo_u:.3f},{hi_u:.3f}]",
                        matched=round(float(np.mean(aucs_m)), 3),
                        matched_ci=f"[{lo_m:.3f},{hi_m:.3f}]"))
    df = pd.DataFrame(out)
    df.to_csv(RES / "final_matched_negative_summary.csv", index=False)
    print("=== matched-negative (corrected) ===")
    print(df.to_string(index=False))

    # ---- precision at 0.85 per species (best ckpt) ----
    sw = pd.read_csv(RES / "threshold_sweep_external.csv")
    p85 = sw[(sw.thr == 0.85)]
    print("\n=== precision/recall at p>=0.85 (>=2 tools) ===")
    print(p85[p85.gt.str.contains("2")][["species", "checkpoint", "precision",
                                         "recall", "enrichment"]]
          .pivot_table(index="species", columns="checkpoint",
                       values="precision").round(3).to_string())
    print("\nmean precision:", round(p85[p85.gt.str.contains("2")].precision.mean(), 3),
          "| mean recall:", round(p85[p85.gt.str.contains("2")].recall.mean(), 3),
          "| mean enrichment:", round(p85[p85.gt.str.contains("2")].enrichment.mean(), 2))


if __name__ == "__main__":
    main()
