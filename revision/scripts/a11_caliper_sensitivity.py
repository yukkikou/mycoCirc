#!/usr/bin/env python3
"""A11: sensitivity of the matched-negative evaluation to the caliper.

Answers a likely follow-up: "why these values (exact <=3 / +/-1 exon count,
+/-0.25 log exon length)?"  We re-run the matched-negative evaluation under
four pre-specified caliper regimes and three draws each, structure-only
matching (as in A2: expression and GC channels are NaN for gene-level
matching, so they never bind).

Regimes
  strict      log-length +/-0.10 ; exon count exact <=3, else +/-1
  default     log-length +/-0.25 ; exon count exact <=3, else +/-1   (A2)
  loose       log-length +/-0.50 ; exon count exact <=3, else +/-1
  all-exact   log-length +/-0.01 ; exon count exact for all

Matching is greedy nearest-neighbour without replacement, with the
"nearest available candidate" fallback when a caliper is exhausted, exactly
as in determinants/lib/matched_sampling.py.

Outputs (revision/results/):
  a11_caliper_sensitivity.csv          per (species, ckpt, regime, draw)
  a11_caliper_summary.csv              mean AUROC per (species, ckpt, regime)
  a11_caliper_summary.md               human-readable table for the response
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))
OUT = PROJ / "revision" / "results"

from data.circ_info_encoding import (load_circ_info, filter_circ_info,  # noqa: E402
                                     get_positive_genes)
from utils.gtf_utils import GeneModelIndexer  # noqa: E402

EXTERNAL_CONFIG = ("/media/share/data5/1610305236/Panfungi/1_data/"
                   "configs/exte_lib_config.tsv")
MERGE_DIR = "/media/share/data5/1610305236/Panfungi/1_data/External/merge"
CHECKPOINTS = ["Candida", "Cryptococcus", "Filamentous"]
REGIMES = {                     # name: (log-length caliper, count mode)
    "strict": (0.10, "exact_small"),
    "default": (0.25, "exact_small"),
    "loose": (0.50, "exact_small"),
    "all-exact": (0.01, "exact_all"),
}
DRAWS = [42, 43, 44]


def gene_table(gm, gene_ids):
    rows = []
    for gid in gene_ids:
        g = gm.get_gene(gid)
        if g is None or g.exon_count < 1:
            continue
        exon_len = max(int(sum(g.exon_ends) - sum(g.exon_starts)
                           + g.exon_count), 1)
        rows.append((gid, np.log(exon_len), float(g.exon_count)))
    return rows


def match(pos, cand, len_cal, count_mode, seed, n_per_pos=1):
    """Greedy structure-only matching. Returns (negative gene ids, n_relaxed)."""
    rng = np.random.default_rng(seed)
    pos_log = np.array([p[1] for p in pos])
    pos_cnt = np.array([p[2] for p in pos])
    cand_log = np.array([c[1] for c in cand])
    cand_cnt = np.array([c[2] for c in cand])
    avail = np.ones(len(cand), bool)
    used, relaxed = [], 0
    for i in rng.permutation(len(pos)):
        pl, pc = pos_log[i], pos_cnt[i]
        d_len = np.abs(cand_log - pl)
        d_cnt = np.abs(cand_cnt - pc)
        if count_mode == "exact_small":
            d_cnt = np.where(d_cnt == 0, 0.0, np.where(pc <= 3, 1e6, d_cnt))
        elif count_mode == "exact_all":
            d_cnt = np.where(d_cnt == 0, 0.0, 1e6)
        ok = (d_len <= len_cal) & (d_cnt < 1e6)
        score = d_len + np.minimum(d_cnt, 1e6)
        gated = np.where(ok, score, np.inf)
        gated[~avail] = np.inf
        if not np.isfinite(gated).any():          # caliper exhausted → relax
            relaxed += 1
            ung = np.where(avail, score, np.inf)
            if not np.isfinite(ung).any():
                continue
            j = int(np.argmin(ung))
        else:
            j = int(np.argmin(gated))
        avail[j] = False
        used.append(cand[j][0])
    return used, relaxed


def main():
    specs = {
        "Talaromyces_marneffei":
            "/media/share/resource/reference/talaromyces_marneffei/annotation/pm1_scallop_final.gtf",
    }
    seen = {}
    with open(EXTERNAL_CONFIG) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) > 5 and c[0] not in seen:
                seen[c[0]] = c[5]
    specs.update(seen)

    rows = []
    for sp, gtf in specs.items():
        gm = GeneModelIndexer(gtf)
        circ = filter_circ_info(load_circ_info(
            f"{MERGE_DIR}/{sp}/{sp}_circRNA_info.csv"
            if sp != "Talaromyces_marneffei"
            else "/media/share/data5/1610305236/circTS/3_CIRIquant/circTS_circRNA_info.csv"))
        pos_ids = list(get_positive_genes(circ))
        pos_set = set(pos_ids)
        cand_ids = [g for g in gm.genes if g not in pos_set]
        pos = gene_table(gm, pos_ids)
        cand = gene_table(gm, cand_ids)
        print(f"{sp}: pos={len(pos)} cand={len(cand)}")

        # genome-wide scores per checkpoint
        sc = {}
        for ck in CHECKPOINTS:
            p = PROJ / "predictions_external" / f"{sp}_{ck}.tsv"
            if not p.exists():
                p = PROJ / "predictions_external" / f"{sp}.tsv"
            df = pd.read_csv(p, sep="\t")
            sc[ck] = dict(zip(df["gene_id"], df["p_circ"]))

        for rname, (lcal, cmode) in REGIMES.items():
            for seed in DRAWS:
                negs, nrel = match(pos, cand, lcal, cmode, seed)
                y = np.array([1] * len(pos) + [0] * len(negs))
                for ck in CHECKPOINTS:
                    s = sc[ck]
                    pv = np.array([s.get(g, 0.0) for g in
                                   [x[0] for x in pos] + negs])
                    rows.append(dict(species=sp, ckpt=ck, regime=rname,
                                     draw=seed, auroc=round(float(roc_auc_score(y, pv)), 4),
                                     n_pos=len(pos), n_neg=len(negs),
                                     n_relaxed=nrel))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "a11_caliper_sensitivity.csv", index=False)

    summ = (df.groupby(["species", "ckpt", "regime"])["auroc"]
              .agg(["mean", "min", "max"]).round(4).reset_index()
              .rename(columns={"mean": "auroc_mean", "min": "auroc_min",
                               "max": "auroc_max"}))
    summ["spread_3draws"] = (summ.auroc_max - summ.auroc_min).round(4)
    summ.to_csv(OUT / "a11_caliper_summary.csv", index=False)

    piv = summ.pivot_table(index=["species", "ckpt"], columns="regime",
                           values="auroc_mean")
    def md_table(df):
        cols = [str(c) for c in df.columns]
        out = ["| " + " | ".join([df.index.names[0] or ""] + cols) + " |",
               "|" + "---|" * (len(cols) + 1)]
        for idx, row in df.iterrows():
            key = " / ".join(idx) if isinstance(idx, tuple) else str(idx)
            out.append("| " + key + " | "
                       + " | ".join("" if pd.isna(v) else f"{v:.3f}" for v in row)
                       + " |")
        return "\n".join(out)

    spread = float((piv.max(axis=1) - piv.min(axis=1)).max())
    lines = ["# A11 — caliper sensitivity of the matched-negative evaluation", "",
             "Mean AUROC over three independent draws; structure-only matching.",
             "", md_table(piv.round(3)), "",
             f"Largest spread across the four caliper regimes: {spread:.3f} AUROC.", ""]
    (OUT / "a11_caliper_summary.md").write_text("\n".join(lines))
    print(piv.round(3).to_string())
    print(f"\nmax spread across regimes: "
          f"{float((piv.max(axis=1) - piv.min(axis=1)).max()):.3f}")
    print(f"Done → {OUT}")


if __name__ == "__main__":
    main()
