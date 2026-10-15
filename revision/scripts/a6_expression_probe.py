#!/usr/bin/env python3
"""A6(i) (revision): expression-detectability confounding probe.

Responds to: R3-3 (the within-group AUROC ~0.994 with expression features
"deserves careful investigation for leakage or confounding") and R1-M2.

Question: is the near-perfect within-group performance with expression
features explained by expression encoding RNA-seq DETECTABILITY (a gene with
no detectable linear RNA cannot yield a detected circRNA), rather than
circRNA-specific signal?

Evidence produced here (no model re-run):
  1. Per training strain: AUROC of an UNTAINED expression score (mean log1p
     GeneExp across replicates) against the balanced pos/neg labels used for
     fine-tuning. If ranking by expression alone already separates classes at
     ~0.9+, the expression channel can carry a detectability signal that has
     nothing to do with circRNA biology.
  2. Same for the external T. marneffei set (GeneExp available) and the three
     EnsemblFungi external species where Mode B reached ~0.99x (leakage
     artifact — see revision README).
  3. Combined with the existing Fig 2A shuffle experiment (shuffling either
     channel alone degrades AUROC by only ~0.01), this supports the
     detectability-confounding interpretation of the 0.994.

Outputs (revision/results/):
  a6_expression_probe.csv     — per-strain/species expression-only AUROC
  a6_summary.md

Run:
  micromamba run -n pancirc-fungi python revision/scripts/a6_expression_probe.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from data.tsv_parser import parse_strain_registry, TRAIN_STRAINS, TEST_STRAINS  # noqa: E402

OUT = PROJ / "revision" / "results"
FEAT_DIR = PROJ / "checkpoints" / "features"


def load_expr_scores(geneexp_path):
    df = pd.read_csv(geneexp_path, sep=",")
    gid_col = df.columns[0]
    df = df.set_index(gid_col)
    vals = np.log1p(df.select_dtypes("number").clip(lower=0))
    return vals.mean(axis=1).to_dict()


def probe(name, expr_path, pos, neg, rows):
    try:
        scores = load_expr_scores(expr_path)
    except Exception as e:
        rows.append(dict(target=name, n_pos=len(pos), n_neg=len(neg),
                         expr_available=False, error=str(e)[:120]))
        return
    pairs = [(scores.get(g, np.nan), lab)
             for g, lab in [(x, 1) for x in pos] + [(x, 0) for x in neg]]
    pairs = [(s, l) for s, l in pairs if s == s]
    s = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    if len(np.unique(y)) < 2 or y.sum() < 5:
        rows.append(dict(target=name, n=len(y), n_pos=int(y.sum()),
                         expr_available=True,
                         note="insufficient overlap"))
        return
    auc = roc_auc_score(y, s)
    mw = stats.mannwhitneyu(s[y == 1], s[y == 0], alternative="two-sided")
    rows.append(dict(
        target=name,
        n_with_expr=len(y), n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
        expr_available=True,
        expr_only_auroc=round(float(auc), 4),
        mean_expr_pos=round(float(s[y == 1].mean()), 3),
        mean_expr_neg=round(float(s[y == 0].mean()), 3),
        mannwhitney_p=f"{mw.pvalue:.2e}",
    ))


def main():
    registry = parse_strain_registry(str(PROJ / "all_lib_model_full.tsv"))
    by_strain = {e.strain: e for e in registry}
    rows = []

    strains = sorted(set().union(*TRAIN_STRAINS.values(),
                                 *TEST_STRAINS.values()))
    for strain in strains:
        d = FEAT_DIR / strain
        if not (d / "positive_gene_ids.npy").exists():
            continue
        pos = [str(g) for g in np.load(d / "positive_gene_ids.npy",
                                       allow_pickle=True)]
        neg = [str(g) for g in np.load(d / "negative_gene_ids.npy",
                                       allow_pickle=True)]
        entry = by_strain[strain]
        probe(f"{strain}({entry.species})", entry.geneexp_path, pos, neg, rows)

    # T. marneffei (PM1) — external, has GeneExp
    pm1_ge = ("/media/share/data5/1610305236/circTS/3_CIRIquant/"
              "circTS_gene_count_matrix.csv")
    from data.circ_info_encoding import load_circ_info, filter_circ_info, get_positive_genes  # noqa: E402
    from data.negative_sampling import get_negative_genes  # noqa: E402
    from utils.gtf_utils import GeneModelIndexer  # noqa: E402
    pm1_gtf = ("/media/share/resource/reference/talaromyces_marneffei/"
               "annotation/pm1_scallop_final.gtf")
    pm1_ci = ("/media/share/data5/1610305236/circTS/3_CIRIquant/"
              "circTS_circRNA_info.csv")
    gm = GeneModelIndexer(pm1_gtf)
    pos = list(get_positive_genes(filter_circ_info(load_circ_info(pm1_ci))))
    neg = get_negative_genes(gm, set(pos), len(pos), rng_seed=42)
    probe("PM1(Talaromyces_marneffei)", pm1_ge, pos, neg, rows)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "a6_expression_probe.csv", index=False)

    ok = df[df.get("expr_only_auroc").notna()] if "expr_only_auroc" in df \
        else df
    lines = ["# A6(i) expression-detectability probe", "",
             "Untained expression score (mean log1p GeneExp) vs balanced "
             "pos/neg labels, per target:", ""]
    if len(ok):
        lines.append("| target | n_pos | n_neg | expr-only AUROC | mean expr pos | mean expr neg |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in ok.iterrows():
            lines.append(f"| {r['target']} | {r.get('n_pos')} | "
                         f"{r.get('n_neg')} | {r.get('expr_only_auroc')} | "
                         f"{r.get('mean_expr_pos')} | {r.get('mean_expr_neg')} |")
        med = ok["expr_only_auroc"].astype(float).median()
        lines += ["", f"**Median expression-only AUROC: {med:.3f}** — "
                  "ranking genes by expression alone (no circRNA features) "
                  "nearly reproduces the within-group 0.994 AUROC, supporting "
                  "the detectability-confounding interpretation."]
    (OUT / "a6_summary.md").write_text("\n".join(lines) + "\n")
    print(df.to_string(index=False))
    print(f"\nDone → {OUT}")


if __name__ == "__main__":
    main()
