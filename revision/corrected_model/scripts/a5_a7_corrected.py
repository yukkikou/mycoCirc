#!/usr/bin/env python3
"""A5 (atlas tier) and A7 (species-embedding sweep) for the CORRECTED model.

A5  per-species counts from the regenerated atlas + empirical precision of the
    p>=0.8 tier from corrected A4 outputs
A7  AUROC across the 21 strain slots for two external species
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[3]
BASE = PROJ / "revision" / "corrected_model"
RES = BASE / "eval" / "results"
ATLAS_PRED = BASE / "atlas" / "predictions"
ATLAS_ALL = BASE / "atlas" / "predictions_all"
STRAINID = BASE / "eval" / "strainid"

MERGE = "/media/share/data5/1610305236/Panfungi/1_data/External/merge"


def nrows(p):
    with open(p, "rb") as f:
        return max(sum(1 for _ in f) - 1, 0)


# ---------------- A5 ----------------
def a5():
    per = {p.stem: nrows(p) for p in sorted(ATLAS_PRED.glob("*.tsv"))}
    tot = sum(per.values())
    rows = [dict(item="p>=0.8 total", value=tot),
            dict(item="species", value=len(per))]
    if per:
        vals = np.array(list(per.values()))
        rows += [dict(item="per-species median", value=int(np.median(vals))),
                 dict(item="per-species max", value=int(vals.max()))]
    pd.DataFrame(rows).to_csv(RES / "corrected_a5_atlas_counts.csv", index=False)
    print(f"A5: {len(per)} species, {tot:,} predictions at p>=0.8 "
          f"(median {int(np.median(list(per.values())))}/species)" if per else "A5: no atlas yet")

    # empirical precision of the new tier, from the corrected A4 table
    f = RES / "corrected_a4_threshold_metrics.csv"
    if f.exists():
        d = pd.read_csv(f)
        d[d.stringency == 2][["species", "checkpoint", "prec@0.8",
                              "sens@0.8", "spec@0.8", "auprc"]].to_csv(
            RES / "corrected_a5_tier_precision.csv", index=False)


# ---------------- A7 ----------------
def a7():
    import sys
    sys.path.insert(0, str(PROJ))
    from data.circ_info_encoding import (load_circ_info, filter_circ_info,
                                         get_positive_genes)
    from data.negative_sampling import get_negative_genes
    from utils.gtf_utils import GeneModelIndexer

    SPEC = {"Aspergillus_cristatus":
            "/media/share/workdir/1610305236/Panfungi/circ/External/gtf/"
            "Aspergillus_cristatus_gca_001717485.ASM171748v1.60.gtf",
            "Trichophyton_rubrum":
            "/media/share/workdir/1610305236/Panfungi/circ/External/gtf/"
            "Trichophyton_rubrum_gca_001651445.CMCC_F_T1i.60.gtf"}
    rows = []
    for sp, gtf in SPEC.items():
        gm = GeneModelIndexer(gtf)
        circ = filter_circ_info(load_circ_info(f"{MERGE}/{sp}/{sp}_circRNA_info.csv"))
        pos = list(get_positive_genes(circ))
        neg = get_negative_genes(gm, set(pos), len(pos), rng_seed=42)
        lab = {g: 1 for g in pos}
        lab.update({g: 0 for g in neg})
        for f in sorted(STRAINID.glob(f"{sp}_*_sid*.tsv")):
            sid = int(f.stem.split("sid")[-1])
            df = pd.read_csv(f, sep="\t")
            common = [g for g in df.gene_id if g in lab]
            y = np.array([lab[g] for g in common])
            p = df.set_index("gene_id").loc[common, "p_circ"].values
            rows.append(dict(species=sp, strain_id=sid,
                             auroc=round(float(roc_auc_score(y, p)), 5),
                             n=len(y)))
    if rows:
        d = pd.DataFrame(rows)
        d.to_csv(RES / "corrected_a7_strainid_sensitivity.csv", index=False)
        for sp, g in d.groupby("species"):
            print(f"A7 {sp}: n_slots={len(g)} min={g.auroc.min():.4f} "
                  f"max={g.auroc.max():.4f} range={g.auroc.max()-g.auroc.min():.5f}")
    else:
        print("A7: no sweep outputs yet")


if __name__ == "__main__":
    a5()
    a7()
