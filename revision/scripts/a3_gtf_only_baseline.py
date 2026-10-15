#!/usr/bin/env python3
"""A3 (revision): GTF-only classical baselines (logistic regression +
histogram gradient boosting) on the 8 numeric gene-structure features
(+ biotype one-hot).

Responds to: R2-3 ("a simple GTF-based baseline would help determine how much
of the reported performance can be explained by gene-structure features
alone") and R3-3 (structural-feature confounding).

Protocol (matched to mycoCirc evaluation):
  * Training strains: the group's TRAIN strains (registry split, cached
    gtf_features.npz + positive/negative_gene_ids.npy — identical gene sets to
    mycoCirc training).
  * Held-out: P4/C4/F6 (own-group baseline).
  * External: 4 external species on the same seed-42 balanced sets used by
    eval_external.py; features extracted live with GeneModelIndexer.
  * 5-fold CV on training strains for an internal estimate; bootstrap 95% CI
    on every test AUROC; paired DeLong vs mycoCirc (genome-wide TSV scores).

Outputs (revision/results/):
  a3_gtf_only_results.csv      — all cells (model × eval set × AUROC/CI)
  a3_delong_vs_mycocirc.csv    — paired DeLong per external species
  a3_predictions/              — per-gene test predictions (npz)

Run:
  micromamba run -n pancirc-fungi python revision/scripts/a3_gtf_only_baseline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from data.tsv_parser import parse_strain_registry, TRAIN_STRAINS, TEST_STRAINS  # noqa: E402
from data.circ_info_encoding import (  # noqa: E402
    load_circ_info, filter_circ_info, get_positive_genes,
)
from data.negative_sampling import get_negative_genes  # noqa: E402
from utils.gtf_utils import GeneModelIndexer  # noqa: E402
from utils.metrics import delong_roc_test  # noqa: E402

OUT = PROJ / "revision" / "results"
PRED_DIR = OUT / "a3_predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

FEAT_DIR = PROJ / "checkpoints" / "features"
EXTERNAL_CONFIG = ("/media/share/data5/1610305236/Panfungi/1_data/"
                   "configs/exte_lib_config.tsv")
MERGE_DIR = "/media/share/data5/1610305236/Panfungi/1_data/External/merge"

GROUPS = ["Candida", "Cryptococcus", "Filamentous"]
N_BOOT, BOOT_SEED = 2000, 2026
BIOTYPES = 12  # n_biotypes used by GTFEncoder


def load_strain_xy(strain, registry_entry, max_retry=4):
    """Balanced training labels (cached ids) + GTF features extracted live.

    The cached gtf_features.npz only contains POSITIVE genes (see
    scripts/extract_features.py), so features for negatives are computed here
    with the same GeneModelIndexer.extract_features used by the model.
    Shared-FS GTF reads can silently return truncated parses, so the parse is
    validated (enough genes + >=50% positive-id intersection) and retried.
    """
    import time
    d = FEAT_DIR / strain
    pos = [str(g) for g in np.load(d / "positive_gene_ids.npy",
                                    allow_pickle=True)]
    neg = [str(g) for g in np.load(d / "negative_gene_ids.npy",
                                    allow_pickle=True)]
    last_err = None
    if len(pos) == 0 or len(neg) == 0:
        # degenerate strain (e.g. C3: no usable circRNAs after filtering) —
        # legitimately empty, not a parse failure
        return np.array([]), np.zeros((0, 20), dtype=float), np.array([])
    for attempt in range(max_retry):
        try:
            gm = GeneModelIndexer(registry_entry.gtf_path)
            n_genes = len(gm.genes)
            hit = sum(1 for g in pos if gm.get_gene(g) is not None)
            if n_genes < 1000 or hit < 0.5 * len(pos):
                raise RuntimeError(
                    f"{strain}: suspicious GTF parse (genes={n_genes}, "
                    f"pos hit {hit}/{len(pos)}) — attempt {attempt + 1}")
            genes, X, y = [], [], []
            for gid, lab in [(g, 1) for g in pos] + [(g, 0) for g in neg]:
                gene = gm.get_gene(gid)
                if gene is None:
                    continue
                genes.append(gid)
                X.append(feat_matrix(gm.extract_features(gene)))
                y.append(lab)
            if len(X) == 0:
                raise RuntimeError(f"{strain}: 0 usable genes")
            return np.array(genes), np.array(X, dtype=float), np.array(y)
        except RuntimeError as e:
            last_err = e
            print(f"  [retry] {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{strain}: GTF parse failed after {max_retry} "
                       f"attempts: {last_err}")


def feat_matrix(vec17):
    """8 numeric dims + biotype one-hot (reserved dims 9-16 are always 0)."""
    numeric = vec17[:8]
    bt = int(vec17[8]) if vec17[8] < BIOTYPES else BIOTYPES - 1
    onehot = np.zeros(BIOTYPES, dtype=float)
    onehot[bt] = 1.0
    return np.concatenate([numeric, onehot])


def external_dataset(name, spec):
    gm = GeneModelIndexer(spec["gtf"])
    circ_df = filter_circ_info(load_circ_info(spec["circinfo"]))
    pos = get_positive_genes(circ_df)
    neg = get_negative_genes(gm, set(pos.keys()), len(pos), rng_seed=42)
    genes, X, y = [], [], []
    for gid, lab in [(g, 1) for g in pos] + [(g, 0) for g in neg]:
        gene = gm.get_gene(gid)
        if gene is None or gid not in gm.genes:
            continue
        X.append(feat_matrix(gm.extract_features(gene)))
        genes.append(gid)
        y.append(lab)
    return np.array(genes), np.array(X), np.array(y)


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


def make_models(seed=42):
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
        "hgb": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=seed),
    }


def main():
    registry = parse_strain_registry(str(PROJ / "all_lib_model_full.tsv"))
    by_strain = {e.strain: e for e in registry}

    # ── assemble per-group training matrices ─────────────────────────────
    group_train = {}
    for grp in GROUPS:
        X_l, y_l = [], []
        for strain in sorted(TRAIN_STRAINS[grp]):
            g, X, y = load_strain_xy(strain, by_strain[strain])
            if int(y.sum()) < 10 or int((1 - y).sum()) < 10:
                print(f"  {strain}: SKIPPED (degenerate: "
                      f"pos={int(y.sum())} neg={int((1 - y).sum())} — "
                      f"also excluded from mycoCirc fine-tuning)")
                continue
            print(f"  {strain}: n={len(y)} pos={int(y.sum())}")
            X_l.append(X)
            y_l.append(y)
        X_all = np.vstack(X_l)
        y_all = np.concatenate(y_l)
        group_train[grp] = (X_all, y_all)
        print(f"{grp}: train n={len(y_all)} pos={int(y_all.sum())}")

    # ── external species definitions ─────────────────────────────────────
    ext_defs = {
        "Talaromyces_marneffei": dict(
            strain="PM1",
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
        ext_defs[name] = dict(strain=name, genome=g, gtf=t,
                              circinfo=str(merge))

    rows, delong_rows = [], []

    for grp in GROUPS:
        X_tr, y_tr = group_train[grp]
        models = make_models()
        for mname, model in models.items():
            # 5-fold CV on training strains (gene-level folds)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_aucs = []
            for tr, va in skf.split(X_tr, y_tr):
                m = make_models()[mname]
                m.fit(X_tr[tr], y_tr[tr])
                cv_aucs.append(roc_auc_score(y_tr[va], m.predict_proba(X_tr[va])[:, 1]))
            model.fit(X_tr, y_tr)

            # held-out strain (own group)
            test_strain = next(iter(TEST_STRAINS[grp]))
            g_h, X_h, y_h = load_strain_xy(test_strain, by_strain[test_strain])
            p_h = model.predict_proba(X_h)[:, 1]
            auc_h = roc_auc_score(y_h, p_h)
            lo, hi = bootstrap_ci(y_h, p_h)
            np.savez(PRED_DIR / f"heldout_{test_strain}_{grp}_{mname}.npz",
                     gene_id=g_h, y_true=y_h, y_prob=p_h)
            rows.append(dict(
                eval=f"heldout_{test_strain}", group=grp, model=mname,
                auroc=round(float(auc_h), 4),
                auroc_ci=f"[{lo:.3f},{hi:.3f}]",
                cv_auroc=round(float(np.mean(cv_aucs)), 4),
                cv_sd=round(float(np.std(cv_aucs)), 4),
                n=len(y_h), n_pos=int(y_h.sum())))

            # external species
            for sp, spec in ext_defs.items():
                g_e, X_e, y_e = external_dataset(sp, spec)
                p_e = model.predict_proba(X_e)[:, 1]
                auc_e = roc_auc_score(y_e, p_e)
                lo, hi = bootstrap_ci(y_e, p_e)
                np.savez(PRED_DIR / f"external_{sp}_{grp}_{mname}.npz",
                         gene_id=g_e, y_true=y_e, y_prob=p_e)
                rows.append(dict(
                    eval=f"external_{sp}", group=grp, model=mname,
                    auroc=round(float(auc_e), 4),
                    auroc_ci=f"[{lo:.3f},{hi:.3f}]",
                    cv_auroc=round(float(np.mean(cv_aucs)), 4),
                    cv_sd=round(float(np.std(cv_aucs)), 4),
                    n=len(y_e), n_pos=int(y_e.sum())))

                # paired DeLong vs mycoCirc (genome-wide TSV scores)
                tsv = PROJ / "predictions_external" / f"{sp}_{grp}.tsv"
                if not tsv.exists():
                    tsv = PROJ / "predictions_external" / f"{sp}.tsv"
                df = pd.read_csv(tsv, sep="\t")
                myco = dict(zip(df["gene_id"], df["p_circ"]))
                common = [i for i, gid in enumerate(g_e) if gid in myco]
                y_c = y_e[common]
                p_mine = p_e[common]
                p_myco = np.array([myco[g_e[i]] for i in common])
                dl = delong_roc_test(y_c.astype(float), p_myco, p_mine)
                delong_rows.append(dict(
                    species=sp, group=grp, baseline=mname,
                    comparison="mycoCirc_A_vs_GTFbaseline",
                    auc_a=round(dl["auc_a"], 4), auc_b=round(dl["auc_b"], 4),
                    diff=round(dl["auc_diff"], 4),
                    diff_ci95="[%.3f,%.3f]" % dl["auc_diff_ci95"],
                    p=f"{dl['p_value']:.2e}", n=len(common)))
            print(f"{grp}/{mname}: held-out "
                  f"{[r for r in rows if r['group']==grp and r['model']==mname and r['eval'].startswith('heldout')][0]['auroc']}")

    pd.DataFrame(rows).to_csv(OUT / "a3_gtf_only_results.csv", index=False)
    pd.DataFrame(delong_rows).to_csv(
        OUT / "a3_delong_vs_mycocirc.csv", index=False)
    print(f"\nDone → {OUT}")


if __name__ == "__main__":
    main()
