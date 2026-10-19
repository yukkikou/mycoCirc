#!/usr/bin/env python3
"""Choose the atlas probability threshold empirically for the corrected model.

Two curves, on the same x-axis (threshold):
  (a) precision / recall / FDR of the p>=t tier on the external species with
      multi-tool detection ground truth (>=1/>=2/>=3 tools) and on
      T. marneffei (CIRIquant);
  (b) size of the tier across all 768 atlas species (from predictions_all).

Outputs
  results/threshold_sweep_external.csv
  results/threshold_sweep_atlas.csv
  results/threshold_sweep.md
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
ATLAS_ALL = BASE / "atlas" / "predictions_all"
RES = BASE / "eval" / "results"

CFG = "/media/share/data5/1610305236/Panfungi/1_data/configs/exte_lib_config.tsv"
MERGE = "/media/share/data5/1610305236/Panfungi/1_data/External/merge"
WD = Path("/media/share/workdir/1610305236/Panfungi/circ/External")
PM1 = ("/media/share/data5/1610305236/circTS/3_CIRIquant/"
       "circTS_circRNA_info.csv")
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
GROUPS = ["Candida", "Cryptococcus", "Filamentous"]
MULTI = ["Aspergillus_cristatus", "Trichophyton_mentagrophytes",
         "Trichophyton_rubrum"]


def circ_to_gene(sp):
    m = {}
    with open(f"{MERGE}/{sp}/{sp}_circRNA_info.csv") as f:
        for r in csv.DictReader(f):
            g = r["gene_id"].strip("\"'")
            if g and g != "NA":
                m[r["circ_id"].strip("\"'")] = g
    return m


def support(sp):
    c2g = circ_to_gene(sp)
    conds = []
    with open(CFG) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c and c[0] == sp:
                conds.append(c[1])
    sup = defaultdict(int)
    for cond in conds:
        p = WD / cond / f"{cond}_merged_weighted_1based.tsv"
        if not p.is_file():
            continue
        with open(p) as f:
            next(f)
            for line in f:
                c = line.rstrip("\n").split("\t")
                if len(c) < 7:
                    continue
                g = c2g.get(c[0], "")
                if g:
                    sup[g] = max(sup[g], int(c[6]))
    return sup


def main():
    ext_rows = []
    # ---------- (a) external precision curves ----------
    for sp in MULTI:
        sup = support(sp)
        for ck in GROUPS:
            f = SCORES / f"{sp}_{ck}_A.tsv"
            if not f.exists():
                continue
            df = pd.read_csv(f, sep="\t")
            genes = df.gene_id.tolist()
            pv = df.p_circ.values.astype(float)
            for k in (1, 2, 3):
                y = np.array([1 if sup.get(g, 0) >= k else 0 for g in genes])
                if y.sum() < 20:
                    continue
                for t in THRESHOLDS:
                    sel = pv >= t
                    tp = int((sel & (y == 1)).sum())
                    fp = int((sel & (y == 0)).sum())
                    ext_rows.append(dict(
                        species=sp, checkpoint=ck, gt=f">={k}tools", thr=t,
                        n_pred=int(sel.sum()),
                        frac_pred=round(float(sel.mean()), 4),
                        precision=round(tp / (tp + fp), 4) if tp + fp else np.nan,
                        recall=round(tp / int(y.sum()), 4),
                        prevalence=round(float(y.mean()), 4),
                        enrichment=round((tp / (tp + fp)) / float(y.mean()), 2)
                        if tp + fp and y.mean() > 0 else np.nan))
    # PM1 (CIRIquant)
    dfp = pd.read_csv(f"{MERGE}/Talaromyces_marneffei/Talaromyces_marneffei_circRNA_info.csv") \
        if Path(f"{MERGE}/Talaromyces_marneffei/Talaromyces_marneffei_circRNA_info.csv").exists() else pd.read_csv(PM1)
    pm1_pos = set(dfp.loc[dfp.circ_type.isin(["exon", "intron"]), "gene_id"].astype(str))
    for ck in GROUPS:
        f = SCORES / f"Talaromyces_marneffei_{ck}_A.tsv"
        if not f.exists():
            continue
        df = pd.read_csv(f, sep="\t")
        y = np.array([1 if g in pm1_pos else 0 for g in df.gene_id])
        pv = df.p_circ.values.astype(float)
        for t in THRESHOLDS:
            sel = pv >= t
            tp = int((sel & (y == 1)).sum())
            fp = int((sel & (y == 0)).sum())
            ext_rows.append(dict(
                species="Talaromyces_marneffei", checkpoint=ck,
                gt="CIRIquant", thr=t, n_pred=int(sel.sum()),
                frac_pred=round(float(sel.mean()), 4),
                precision=round(tp / (tp + fp), 4) if tp + fp else np.nan,
                recall=round(tp / int(y.sum()), 4),
                prevalence=round(float(y.mean()), 4),
                enrichment=round((tp / (tp + fp)) / float(y.mean()), 2)
                if tp + fp and y.mean() > 0 else np.nan))
    pd.DataFrame(ext_rows).to_csv(RES / "threshold_sweep_external.csv", index=False)

    # ---------- (b) atlas tier size per threshold ----------
    counts = {t: 0 for t in THRESHOLDS}
    n_genes = 0
    per_sp = []
    for f in sorted(ATLAS_ALL.glob("*.tsv")):
        df = pd.read_csv(f, sep="\t", usecols=["p_circ"])
        p = df.p_circ.values
        n_genes += len(p)
        for t in THRESHOLDS:
            counts[t] += int((p >= t).sum())
        per_sp.append(dict(species=f.stem,
                           n=len(p),
                           **{f"n_ge_{t}": int((p >= t).sum()) for t in THRESHOLDS}))
    pd.DataFrame(per_sp).to_csv(RES / "threshold_sweep_atlas_per_species.csv",
                                index=False)
    at = pd.DataFrame([dict(thr=t, atlas_predictions=counts[t],
                            frac_of_all_genes=round(counts[t] / n_genes, 4))
                       for t in THRESHOLDS])
    at.to_csv(RES / "threshold_sweep_atlas.csv", index=False)

    # ---------- summary ----------
    ext = pd.DataFrame(ext_rows)
    lines = ["# Atlas threshold selection (corrected model)", "",
             f"Total genes in atlas: {n_genes:,}", "",
             "## Tier size across 768 species", "",
             at.to_markdown(index=False) if hasattr(at, "to_markdown") else
             at.to_string(index=False), "",
             "## Precision / recall on external GT (mean over models)", ""]
    piv = (ext[ext.gt == ">=2tools"]
           .groupby("thr")[["precision", "recall", "enrichment"]]
           .mean().round(3))
    piv["atlas_predictions"] = at.set_index("thr").atlas_predictions
    txt = piv.to_string()
    lines += ["```", txt, "```", ""]

    def md(df):
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in df.iterrows():
            out.append("| " + " | ".join(str(v) for v in r.values) + " |")
        return "\n".join(out)

    md_lines = ["# Atlas threshold selection (corrected model)", "",
                f"Total genes in atlas: {n_genes:,}", "",
                "## Tier size across 768 species", "",
                md(at.rename(columns={"thr": "threshold",
                                      "atlas_predictions": "predictions",
                                      "frac_of_all_genes": "fraction_of_genes"})),
                "", "## External GT (≥2 tools), mean over the 3 group models", "",
                md(piv.reset_index().rename(columns={"thr": "threshold"}))]
    (RES / "threshold_sweep.md").write_text("\n".join(md_lines) + "\n")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
