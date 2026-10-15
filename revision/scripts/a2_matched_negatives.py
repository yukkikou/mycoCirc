#!/usr/bin/env python3
"""A2 (revision): matched-negative robustness of the external evaluation.

Responds to: R2-2 (are results stable when negatives are matched by exon
count / gene structure?), R1-M1 (false-negative contamination robustness),
R3-3 (structural confounding).

Matching (greedy, no replacement — reuses determinants/lib/matched_sampling.py
MatchedExonSampler at GENE level):
  * log total-exon-length caliper +/-0.25 (relaxed +/-0.5)
  * exon count exact when <= 3, else +/-1
  * expression/GC channels disabled (NaN<->NaN) — structure-only matching,
    exactly what R2-2 requests

Evaluated methods on each matched set:
  * mycoCirc Mode A (per checkpoint, genome-wide TSV scores)
  * GTF-only logistic regression (trained on the group's TRAIN strains) —
    probes whether gene-structure features ALONE still separate matched sets

Three independent draws (seeds 42/43/44, n_per_pos=1) per species to show
stability. Unmatched (original seed-42 length-matched) AUROC shown alongside.

Outputs (revision/results/):
  a2_matched_negative_results.csv
  a2_matched_sets/  — matched gene id lists per draw

Run:
  micromamba run -n pancirc-fungi python revision/scripts/a2_matched_negatives.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

_spec = importlib.util.spec_from_file_location(
    "matched_sampling",
    str(PROJ / "determinants" / "lib" / "matched_sampling.py"))
matched_sampling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(matched_sampling)
MatchedExonSampler = matched_sampling.MatchedExonSampler

# reuse a3 helpers (GTF feature matrix + per-strain training loader)
_spec3 = importlib.util.spec_from_file_location(
    "a3_helpers", str(PROJ / "revision" / "scripts" / "a3_gtf_only_baseline.py"))
a3 = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(a3)

from data.tsv_parser import parse_strain_registry, TRAIN_STRAINS  # noqa: E402
from utils.gtf_utils import GeneModelIndexer  # noqa: E402

OUT = PROJ / "revision" / "results"
SETS = OUT / "a2_matched_sets"
SETS.mkdir(parents=True, exist_ok=True)

EXTERNAL_CONFIG = ("/media/share/data5/1610305236/Panfungi/1_data/"
                   "configs/exte_lib_config.tsv")
MERGE_DIR = "/media/share/data5/1610305236/Panfungi/1_data/External/merge"
CHECKPOINTS = ["Candida", "Cryptococcus", "Filamentous"]
DRAWS = [42, 43, 44]
N_BOOT, BOOT_SEED = 1000, 2026


def bootstrap_ci(y, p, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n_boot):
        sel = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[sel])) < 2:
            continue
        vals.append(roc_auc_score(y[sel], p[sel]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def gene_rows(gm, gene_ids):
    """Build sampler-format rows at gene level from a GeneModelIndexer."""
    rows = []
    for gid in gene_ids:
        g = gm.get_gene(gid)
        if g is None or g.exon_count < 1:
            continue
        exon_len = int(sum(g.exon_ends) - sum(g.exon_starts) + g.exon_count)
        rows.append(dict(
            gene_id=gid, chrom=g.chrom, exon_start=g.start, exon_end=g.end,
            strand=g.strand,
            exon_len=max(exon_len, 1),
            host_exon_count=float(g.exon_count),
            host_expr=np.nan, flank_gc=np.nan,
            circ_len=max(exon_len, 1),  # for positives
            circ_id=gid,
        ))
    return rows


def main():
    registry = parse_strain_registry(str(PROJ / "all_lib_model_full.tsv"))
    by_strain = {e.strain: e for e in registry}

    # GTF-only logreg per group (trained on TRAIN strains)
    lr_models = {}
    from data.tsv_parser import TRAIN_STRAINS as TS
    for grp in ["Candida", "Cryptococcus", "Filamentous"]:
        X_l, y_l = [], []
        for strain in sorted(TS[grp]):
            _, X, y = a3.load_strain_xy(strain, by_strain[strain])
            X_l.append(X)
            y_l.append(y)
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=42))
        m.fit(np.vstack(X_l), np.concatenate(y_l))
        lr_models[grp] = m
        print(f"logreg {grp} trained")

    ext_defs = {
        "Talaromyces_marneffei": dict(
            genome="/media/share/resource/reference/talaromyces_marneffei/genome/pm1.soft.fasta",
            gtf="/media/share/resource/reference/talaromyces_marneffei/annotation/pm1_scallop_final.gtf",
            circinfo="/media/share/data5/1610305236/circTS/3_CIRIquant/circTS_circRNA_info.csv",
        ),
    }
    seen = {}
    with open(EXTERNAL_CONFIG) as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) > 5 and cols[0] not in seen:
                seen[cols[0]] = (cols[4], cols[5])
    for name, (g, t) in seen.items():
        merge = Path(MERGE_DIR) / name / f"{name}_circRNA_info.csv"
        ext_defs[name] = dict(genome=g, gtf=t, circinfo=str(merge))

    rows = []
    for sp, spec in ext_defs.items():
        print(f"\n=== {sp} ===")
        gm = GeneModelIndexer(spec["gtf"])
        from data.circ_info_encoding import (load_circ_info, filter_circ_info,
                                             get_positive_genes)
        pos = list(get_positive_genes(
            filter_circ_info(load_circ_info(spec["circinfo"]))))
        pos_set = set(pos)
        cands = [g for g in gm.genes if g not in pos_set]

        pos_df = pd.DataFrame(gene_rows(gm, pos))
        cand_df = pd.DataFrame(gene_rows(gm, cands))
        print(f"pos={len(pos_df)} candidates={len(cand_df)}")

        # mycoCirc scores
        score_maps = {}
        for ckpt in CHECKPOINTS:
            tsv = PROJ / "predictions_external" / f"{sp}_{ckpt}.tsv"
            if not tsv.exists():
                tsv = PROJ / "predictions_external" / f"{sp}.tsv"
            df = pd.read_csv(tsv, sep="\t")
            score_maps[ckpt] = dict(zip(df["gene_id"], df["p_circ"]))

        # original unmatched AUROC (seed-42 length-matched balanced set)
        for ckpt in CHECKPOINTS:
            sm = score_maps[ckpt]
            y = pos_df["gene_id"].isin(sm).astype(int).values
            p = pos_df["gene_id"].map(sm).fillna(0).values
            # recompute on pos + seed42 negatives
            from data.negative_sampling import get_negative_genes
            neg42 = get_negative_genes(gm, pos_set, len(pos), rng_seed=42)
            genes = list(pos_df["gene_id"]) + neg42
            y = np.array([1] * len(pos_df) + [0] * len(neg42))
            p = np.array([sm.get(g, 0.0) for g in genes])
            auc = roc_auc_score(y, p)
            rows.append(dict(species=sp, ckpt=ckpt, method="mycoCirc",
                             neg_set="unmatched_len", draw="seed42",
                             auroc=round(float(auc), 4), n=len(y)))

        # matched draws
        for draw_seed in DRAWS:
            sampler = MatchedExonSampler(cand_df, rng_seed=draw_seed)
            neg_df = sampler.sample(pos_df, n_per_pos=1)
            neg_genes = sorted(set(neg_df["gene_id"]))
            pd.Series(neg_genes).to_csv(
                SETS / f"{sp}_matched_seed{draw_seed}.txt",
                index=False, header=False)
            diag = sampler.diagnostics()
            print(f"  draw {draw_seed}: matched {len(neg_genes)} negatives "
                  f"({diag})")

            genes = list(pos_df["gene_id"]) + neg_genes
            y = np.array([1] * len(pos_df) + [0] * len(neg_genes))
            for ckpt in CHECKPOINTS:
                sm = score_maps[ckpt]
                p = np.array([sm.get(g, 0.0) for g in genes])
                auc = roc_auc_score(y, p)
                lo, hi = bootstrap_ci(y, p)
                rows.append(dict(species=sp, ckpt=ckpt, method="mycoCirc",
                                 neg_set="matched_exoncount_len",
                                 draw=f"seed{draw_seed}",
                                 auroc=round(float(auc), 4),
                                 ci=f"[{lo:.3f},{hi:.3f}]", n=len(y)))

            # GTF-only logreg on the matched set
            X = np.vstack([a3.feat_matrix(gm.extract_features(gm.get_gene(g)))
                           for g in genes if gm.get_gene(g) is not None])
            keep = [i for i, g in enumerate(genes)
                    if gm.get_gene(g) is not None]
            y_k = y[keep]
            for grp, m in lr_models.items():
                p = m.predict_proba(X)[:, 1]
                auc = roc_auc_score(y_k, p)
                lo, hi = bootstrap_ci(y_k, p)
                rows.append(dict(species=sp, ckpt=grp, method="gtf_logreg",
                                 neg_set="matched_exoncount_len",
                                 draw=f"seed{draw_seed}",
                                 auroc=round(float(auc), 4),
                                 ci=f"[{lo:.3f},{hi:.3f}]", n=len(y_k)))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "a2_matched_negative_results.csv", index=False)

    # summary pivot: mean matched AUROC vs unmatched per species/ckpt
    piv = (df[df.method == "mycoCirc"]
           .groupby(["species", "ckpt", "neg_set"])["auroc"]
           .mean().unstack())
    piv.to_csv(OUT / "a2_matched_vs_unmatched_summary.csv")
    print("\n" + piv.round(4).to_string())
    print(f"\nDone → {OUT}")


if __name__ == "__main__":
    main()
