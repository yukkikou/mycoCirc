#!/usr/bin/env python3
"""Summarize A1b (held-out per-gene via predict.py path): AUROC + bootstrap CI.

Also computes the A. cristatus / T. rubrum strain-id sweep AUROCs (A7) from
whatever sweep TSVs exist, and writes revision/results/a7_strainid_sensitivity.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJ = Path("/media/share/workdir/1610305236/Panfungi/4_model")
OUT = PROJ / "revision" / "results"
RES = OUT / "rescore"
sys.path.insert(0, str(PROJ))

rng = np.random.default_rng(2026)
rows = []
for f in sorted(RES.glob("heldout_*.npz")):
    d = np.load(f, allow_pickle=True)
    y, p = d["y_true"], d["y_prob"]
    auc = roc_auc_score(y, p)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    vals = []
    for _ in range(2000):
        sel = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        vals.append(roc_auc_score(y[sel], p[sel]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    strain, grp = f.stem.replace("heldout_", "").rsplit("_", 1)
    rows.append(dict(strain=strain, checkpoint=grp, auroc=round(float(auc), 4),
                     ci=f"[{lo:.3f},{hi:.3f}]", n=len(y), n_pos=int(y.sum())))
    print(f"{strain:3s} {grp:12s} AUROC={auc:.4f} CI=[{lo:.3f},{hi:.3f}] "
          f"n={len(y)} pos={int(y.sum())}")
pd.DataFrame(rows).to_csv(OUT / "a1b_heldout_with_ci.csv", index=False)

# ── A7 sweep ────────────────────────────────────────────────────────────────
from data.circ_info_encoding import (load_circ_info, filter_circ_info,
                                     get_positive_genes)
from data.negative_sampling import get_negative_genes
from utils.gtf_utils import GeneModelIndexer

MERGE = "/media/share/data5/1610305236/Panfungi/1_data/External/merge"
CFG = ("/media/share/data5/1610305236/Panfungi/1_data/configs/"
       "exte_lib_config.tsv")
specs = {
    "Aspergillus_cristatus":
        "/media/share/workdir/1610305236/Panfungi/circ/External/gtf/"
        "Aspergillus_cristatus_gca_001717485.ASM171748v1.60.gtf",
    "Trichophyton_rubrum":
        "/media/share/workdir/1610305236/Panfungi/circ/External/gtf/"
        "Trichophyton_rubrum_gca_001651445.CMCC_F_T1i.60.gtf",
}
a7 = []
for sp, gtf in specs.items():
    info = f"{MERGE}/{sp}/{sp}_circRNA_info.csv"
    gm = GeneModelIndexer(gtf)
    pos = list(get_positive_genes(filter_circ_info(load_circ_info(info))))
    neg = get_negative_genes(gm, set(pos), len(pos), rng_seed=42)
    labels = {g: 1 for g in pos}
    labels.update({g: 0 for g in neg})
    for sid in range(21):
        f = RES / "strainid" / f"{sp}_sid{sid}.tsv"
        if not f.exists():
            continue
        df = pd.read_csv(f, sep="\t")
        common = [g for g in df.gene_id if g in labels]
        y = np.array([labels[g] for g in common])
        p = df.set_index("gene_id").loc[common, "p_circ"].values
        a7.append(dict(species=sp, strain_id=sid,
                       auroc=round(float(roc_auc_score(y, p)), 5),
                       n=len(y)))
df7 = pd.DataFrame(a7)
df7.to_csv(OUT / "a7_strainid_sensitivity.csv", index=False)
for sp, g in df7.groupby("species"):
    print(f"\nA7 {sp}: n_slots={len(g)} min={g.auroc.min():.4f} "
          f"max={g.auroc.max():.4f} range={g.auroc.max()-g.auroc.min():.5f} "
          f"sd={g.auroc.std():.5f}")
print("\nDone.")
