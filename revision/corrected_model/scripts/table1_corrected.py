#!/usr/bin/env python3
"""Table-1 statistics for the CORRECTED model: AUROC with bootstrap CI,
AUPRC, and paired DeLong tests against JEDI and CircPCBL.

Per-gene corrected scores come from eval/results/{species}_{ckpt}_{mode}.npz
(gene_id, y_true, y_prob). Baseline gene sets are reconstructed exactly as in
the original A1 analysis (same balanced sets), then aligned by gene id.
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

# reuse the validated DeLong + JEDI-alignment helpers from the A1 script
spec = importlib.util.spec_from_file_location(
    "a1", PROJ / "revision" / "scripts" / "a1_table1_ci.py")
a1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a1)

CIRCPCBL = PROJ / "results" / "circpcbl_external"
SCORES = PROJ / "results" / "pm1_scores"
EXTERNAL_CONFIG = ("/media/share/data5/1610305236/Panfungi/1_data/"
                   "configs/exte_lib_config.tsv")
MERGE_DIR = "/media/share/data5/1610305236/Panfungi/1_data/External/merge"
GROUPS = ["Candida", "Cryptococcus", "Filamentous"]
N_BOOT = 2000


def boot(y, p, seed=2026):
    rng = np.random.default_rng(seed)
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(N_BOOT):
        s = np.r_[rng.choice(pi, len(pi)), rng.choice(neg := ni, len(neg))]
        vals.append(roc_auc_score(y[s], p[s]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def baseline_align(sp):
    """Return dict(baseline_name -> {gene_id: score}) built on the same
    balanced set used by the corrected evaluation."""
    out = {}

    # CircPCBL: csv rows follow gene_ids.txt order
    gid_path = CIRCPCBL / f"{sp}_gene_ids.txt"
    csv_path = CIRCPCBL / f"{sp}_circpcbl.csv"
    if gid_path.exists() and csv_path.exists():
        gids = [l.strip() for l in open(gid_path) if l.strip()]
        df = pd.read_csv(csv_path)
        col = "Prob_circRNA" if "Prob_circRNA" in df.columns else df.columns[-1]
        out["CircPCBL"] = dict(zip(gids, df[col].astype(float)))

    # JEDI: reconstruct the valid-record order (mirrors eval_jedi_external.py)
    npz = None
    for g in GROUPS:
        p = SCORES / f"{sp}_{g}_JEDI_balanced.npz"
        if p.exists():
            npz = p
            break
    if npz is not None:
        d = np.load(npz)
        out["JEDI"] = d  # y_true/y_prob arrays; alignment handled below
    return out


def main():
    rows, dl_rows = [], []
    for npz in sorted(RES.glob("*_A.npz")):
        sp, ck = npz.stem.rsplit("_", 2)[0], npz.stem.rsplit("_", 2)[1]
        if sp in ("P4", "C4", "F6"):
            continue
        for mode in ("A", "B"):
            f = RES / f"{sp}_{ck}_{mode}.npz"
            if not f.exists():
                continue
            d = np.load(f, allow_pickle=True)
            g = [str(x) for x in d["gene_id"]]
            y = d["y_true"].astype(int); p = d["y_prob"].astype(float)
            auc = roc_auc_score(y, p)
            lo, hi = boot(y, p)
            ap = average_precision_score(y, p)
            rows.append(dict(species=sp, checkpoint=ck, mode=mode,
                             recipe="nocircexp", auroc=round(float(auc), 4),
                             ci=f"[{lo:.3f},{hi:.3f}]",
                             auprc=round(float(ap), 4),
                             n=len(y), n_pos=int(y.sum())))

            if mode != "A":
                continue
            # paired DeLong vs CircPCBL on identical gene ids
            cpb = baseline_align(sp).get("CircPCBL")
            if cpb:
                common = [i for i, gid in enumerate(g) if gid in cpb]
                yc = y[common]
                pm = p[common]
                pb = np.array([cpb[g[i]] for i in common])
                dl = a1.delong_roc_test_fast(yc.astype(float), pm, pb)
                dl_rows.append(dict(species=sp, checkpoint=ck,
                                    comparison="mycoCirc_corrected_vs_CircPCBL",
                                    auc_a=round(dl["auc_a"], 4),
                                    auc_b=round(dl["auc_b"], 4),
                                    diff=round(dl["auc_diff"], 4),
                                    p=f"{dl['p_value']:.2e}", n=len(common)))
    pd.DataFrame(rows).to_csv(RES / "corrected_table1_stats.csv", index=False)
    pd.DataFrame(dl_rows).to_csv(RES / "corrected_delong_vs_baselines.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    if dl_rows:
        print("\n" + pd.DataFrame(dl_rows).to_string(index=False))


if __name__ == "__main__":
    main()
