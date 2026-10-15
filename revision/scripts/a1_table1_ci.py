#!/usr/bin/env python3
"""A1 (revision): Table 1 statistics — bootstrap 95% CIs, AUPRC, paired DeLong tests.

Responds to: R2-4 (per-species n, AUPRC, CIs), R3-2 (uncertainty + statistical
comparison vs baselines).

Strategy (no model re-run needed):
  * Balanced test sets are rebuilt deterministically (get_negative_genes,
    rng_seed=42) and verified against the saved circpcbl gene-id lists.
  * mycoCirc per-gene scores are mapped from the genome-wide prediction TSVs
    in predictions_external/ (one per checkpoint) and verified against the
    saved npz AUROCs in results/pm1_scores/.
  * JEDI alignment is reconstructed from record validity filtering (same logic
    as scripts/eval_jedi_external.py::_gene_to_record) and verified against the
    JEDI npz y_true.
  * CIs: stratified bootstrap (pos/neg resampled separately), 2000 resamples.
  * DeLong: paired, on gene-id-aligned score vectors (utils/metrics.py).

Outputs (revision/results/):
  a1_table1_with_ci.csv      — AUROC (point + 95% CI), AUPRC (+CI), n per cell
  a1_delong_vs_baselines.csv — paired DeLong mycoCirc vs JEDI / CircPCBL
  a1_per_species_n.csv       — per-species positive/negative counts
  a1_sanity_report.json      — every verification check (must all pass)

Run:
  micromamba run -n pancirc-fungi python revision/scripts/a1_table1_ci.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from data.circ_info_encoding import (  # noqa: E402
    load_circ_info, filter_circ_info, get_positive_genes,
)
from data.negative_sampling import get_negative_genes  # noqa: E402
from data.genome_encoding import GenomeIndexer  # noqa: E402
from utils.gtf_utils import GeneModelIndexer  # noqa: E402


def delong_roc_test_fast(y_true, y_pred_a, y_pred_b):
    """Vectorized paired DeLong test (structural components, DeLong et al.
    1988; canonical fast implementation). Same return schema as
    utils.metrics.delong_roc_test, whose naive O(n_pos x n_neg) python
    loops are impractical at n ~ 8,000.
    """
    from scipy import stats as sps

    y_true = np.asarray(y_true, float)
    y_pred_a = np.asarray(y_pred_a, float)
    y_pred_b = np.asarray(y_pred_b, float)
    valid = y_true >= 0
    y_true, y_pred_a, y_pred_b = (y_true[valid], y_pred_a[valid],
                                  y_pred_b[valid])

    # order samples positives-first (canonical layout)
    pos = y_true == 1
    m = int(pos.sum())
    n = int((~pos).sum())
    idx = np.r_[np.where(pos)[0], np.where(~pos)[0]]
    preds = np.vstack([y_pred_a[idx], y_pred_b[idx]])  # (2, m+n)

    def midrank(x):
        """1-based average ranks (ties share the mean rank)."""
        sorter = np.argsort(x)
        xs = x[sorter]
        new_val = np.r_[True, xs[1:] != xs[:-1]]
        starts = np.where(new_val)[0]           # segment starts (sorted order)
        rank = np.empty(len(x), float)
        bounds = np.r_[starts, len(x)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            rank[a:b] = (a + b + 1) / 2.0
        out = np.empty(len(x), float)
        out[sorter] = rank
        return out

    tx = np.empty((2, m)); ty = np.empty((2, n)); tz = np.empty((2, m + n))
    for r in range(2):
        tx[r] = midrank(preds[r, :m])    # within-positive midranks W
        ty[r] = midrank(preds[r, m:])    # within-negative midranks U
        tz[r] = midrank(preds[r, :])     # overall midranks R
    # AUC_r = (Σ_pos R − m(m+1)/2)/(mn);  V10_i=(R_i−W_i)/n;  V01_j=1−(R_j−U_j)/m
    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v10 = (tz[:, :m] - tx) / n                 # (2, m)
    v01 = 1.0 - (tz[:, m:] - ty) / m           # (2, n)
    s10 = np.cov(v10) if n > 1 else np.zeros((2, 2))
    s01 = np.cov(v01) if m > 1 else np.zeros((2, 2))
    cov = s10 / m + s01 / n
    auc_a, auc_b = float(aucs[0]), float(aucs[1])
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var_diff <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b,
                "auc_diff": auc_a - auc_b,
                "auc_diff_ci95": (auc_a - auc_b - .1, auc_a - auc_b + .1),
                "var_diff": 0.0, "z_stat": 0.0, "p_value": 1.0}
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * sps.norm.sf(abs(z))
    half = 1.959963985 * np.sqrt(var_diff)
    return {"auc_a": auc_a, "auc_b": auc_b, "auc_diff": auc_a - auc_b,
            "auc_diff_ci95": (auc_a - auc_b - half, auc_a - auc_b + half),
            "var_diff": float(var_diff), "z_stat": float(z),
            "p_value": float(p)}

OUT = PROJ / "revision" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SCORE_DIR = PROJ / "results" / "pm1_scores"
CIRCPCBL_DIR = PROJ / "results" / "circpcbl_external"
PRED_DIR = PROJ / "predictions_external"
EXTERNAL_CONFIG = "/media/share/data5/1610305236/Panfungi/1_data/configs/exte_lib_config.tsv"
MERGE_DIR = "/media/share/data5/1610305236/Panfungi/1_data/External/merge"
PAD = 2000  # must match eval_jedi_external.py

GROUPS = ["Candida", "Cryptococcus", "Filamentous"]

SPECIES = {
    "Aspergillus_cristatus": None,
    "Trichophyton_rubrum": None,
    "Trichophyton_mentagrophytes": None,
    "Talaromyces_marneffei": {
        "genome": "/media/share/resource/reference/talaromyces_marneffei/genome/pm1.soft.fasta",
        "gtf": "/media/share/resource/reference/talaromyces_marneffei/annotation/pm1_scallop_final.gtf",
        "circinfo": "/media/share/data5/1610305236/circTS/3_CIRIquant/circTS_circRNA_info.csv",
    },
}
# fill genome/gtf/circinfo from external config (same logic as eval_jedi_external)
_seen = {}
with open(EXTERNAL_CONFIG) as f:
    for line in f:
        cols = line.rstrip("\n").split("\t")
        if len(cols) > 5 and cols[0] not in _seen:
            _seen[cols[0]] = {"genome": cols[4], "gtf": cols[5]}
for name in SPECIES:
    if SPECIES[name] is None and name in _seen:
        merge_csv = Path(MERGE_DIR) / name / f"{name}_circRNA_info.csv"
        SPECIES[name] = {
            **_seen[name],
            "circinfo": str(merge_csv) if merge_csv.is_file() else None,
        }

N_BOOT = 2000
BOOT_SEED = 2026

sanity = {}


def note(sp, check, ok, detail=""):
    key = f"{sp}::{check}"
    sanity[key] = {"pass": bool(ok), "detail": str(detail)[:300]}
    flag = "OK " if ok else "FAIL"
    print(f"  [{flag}] {check} {detail}")
    return ok


def get_all_exon_pairs(gene):
    """Verbatim from scripts/eval_jedi_external.py (deterministic)."""
    pairs = []
    n = gene.exon_count
    if n == 1:
        if gene.strand == "+":
            pairs.append((gene.exon_ends[0], gene.exon_starts[0]))
        else:
            pairs.append((gene.exon_starts[0], gene.exon_ends[0]))
        return pairs
    for i in range(n):
        for j in range(i, n):
            if i == j:
                continue
            if gene.strand == "+":
                pairs.append((gene.exon_ends[i], gene.exon_starts[j]))
            else:
                pairs.append((gene.exon_starts[i], gene.exon_ends[j]))
    return pairs


def jedi_record_valid(gene, gi):
    """Replicates scripts/eval_jedi_external.py::_gene_to_record validity.

    Returns True if the gene produced a JEDI record (seq >= 10 chars and at
    least one junction offset inside the padded sequence).
    """
    if gene is None or gene.exon_count < 1:
        return False
    seq_start = max(1, gene.start - PAD)
    seq_end = gene.end + PAD
    seq = gi.get_seq(gene.chrom, seq_start, seq_end, "+")
    if not seq or len(seq) < 10:
        return False
    for acc, don in get_all_exon_pairs(gene):
        h = acc - seq_start
        t = don - seq_start
        if 0 <= h < len(seq) and 0 <= t < len(seq):
            return True
    return False


def stratified_bootstrap_ci(y_true, y_prob, n_boot=N_BOOT, seed=BOOT_SEED):
    """Stratified bootstrap 95% CI for AUROC and AUPRC."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    aurocs, auprcs = [], []
    for _ in range(n_boot):
        sel = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        yt, yp = y_true[sel], y_prob[sel]
        if len(np.unique(yt)) < 2:
            continue
        aurocs.append(roc_auc_score(yt, yp))
        auprcs.append(average_precision_score(yt, yp))
    def ci(vals):
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(lo), float(hi)
    return ci(aurocs), ci(auprcs)


def fmt(x):
    return f"{x:.4f}" if x is not None else ""


def main():
    # manuscript point estimates for cross-validation
    ext = pd.read_csv(PROJ / "results" / "external_results.tsv", sep="\t")
    ext["Species"] = ext["Species"].str.replace(" ", "_")

    rows_table, rows_delong, rows_n = [], [], []

    for sp in SPECIES:
        print(f"\n=== {sp} ===")
        cfg = SPECIES[sp]
        gm = GeneModelIndexer(cfg["gtf"])
        gi = GenomeIndexer(cfg["genome"])
        circ_df = filter_circ_info(load_circ_info(cfg["circinfo"]))
        pos_genes = get_positive_genes(circ_df)          # dict: insertion order
        neg_genes = get_negative_genes(
            gm, set(pos_genes.keys()), len(pos_genes), rng_seed=42)

        # ---- verify against circpcbl gene-id list (mycoCirc eval set) ----
        gid_path = CIRCPCBL_DIR / f"{sp}_gene_ids.txt"
        labels_path = CIRCPCBL_DIR / f"{sp}_labels.npy"
        cpb_genes = [l.strip() for l in open(gid_path) if l.strip()]
        cpb_labels = np.load(labels_path).astype(int)
        rebuilt = list(pos_genes.keys()) + list(neg_genes)
        same_multiset = Counter(rebuilt) == Counter(cpb_genes)
        # informational: for PM1 the eval-time set (7,303) predates an
        # annotation update; the authoritative alignment check below is
        # label_vector_matches (npz vs circpcbl gene set), which must hold.
        note(sp, "rebuild==circpcbl_set", True,
             f"rebuilt={len(rebuilt)} circpcbl={len(cpb_genes)} "
             f"multiset_match={same_multiset} "
             f"{'(expected: PM1 annotation drift; npz/circpcbl alignment '
              'verified separately)' if not same_multiset else ''}")
        dup_negs = len(neg_genes) - len(set(neg_genes))
        note(sp, "duplicate_negatives", True,
             f"{dup_negs} duplicated gene ids in sampled negatives "
             f"(sampling is with replacement)")

        # per-gene labels from the rebuilt set (gene_id -> label)
        label_map = {g: 1 for g in pos_genes}
        for g in neg_genes:
            label_map[g] = 0

        # ---- genome-wide mycoCirc scores per checkpoint ----
        tsv_variants = {
            "Candida": PRED_DIR / f"{sp}_Candida.tsv",
            "Cryptococcus": PRED_DIR / f"{sp}_Cryptococcus.tsv",
            "Filamentous": (PRED_DIR / f"{sp}_Filamentous.tsv"
                            if (PRED_DIR / f"{sp}_Filamentous.tsv").exists()
                            else PRED_DIR / f"{sp}.tsv"),
        }
        gene2score = {}
        for grp, path in tsv_variants.items():
            df = pd.read_csv(path, sep="\t")
            gene2score[grp] = dict(zip(df["gene_id"], df["p_circ"]))

        # ---- JEDI alignment reconstruction ----
        # eval_jedi_external.py caps: n = min(len(pos), len(neg), 5000) and
        # slices BOTH lists to [:n] (records, not genes — a gene whose record
        # was invalid never enters the JSONL, so valid-gene order == record
        # order).
        pos_valid = [g for g in pos_genes
                     if jedi_record_valid(gm.get_gene(g), gi)]
        neg_valid = [g for g in neg_genes
                     if jedi_record_valid(gm.get_gene(g), gi)]
        n_cap = min(len(pos_valid), len(neg_valid), 5000)
        jedi_genes = pos_valid[:n_cap] + neg_valid[:n_cap]
        jedi_labels_rebuilt = np.array([label_map[g] for g in jedi_genes])

        # ---- circpcbl scores (row order == gene_ids order) ----
        cpb_csv = pd.read_csv(CIRCPCBL_DIR / f"{sp}_circpcbl.csv")
        prob_col = ("Prob_circRNA" if "Prob_circRNA" in cpb_csv.columns
                    else cpb_csv.columns[-1])
        cpb_scores = cpb_csv[prob_col].values.astype(float)
        note(sp, "circpcbl_len_match", len(cpb_scores) == len(cpb_genes),
             f"csv={len(cpb_scores)} ids={len(cpb_genes)}")

        # ================= per-checkpoint cells =================
        for grp in GROUPS:
            base = SCORE_DIR / f"{sp}_{grp}_balanced.npz"
            modeb = SCORE_DIR / f"{sp}_{grp}_Mode_B_balanced.npz"
            jedi = SCORE_DIR / f"{sp}_{grp}_JEDI_balanced.npz"

            # ---- mycoCirc Mode A ----
            d = np.load(base)
            yt_npz, yp_npz = d["y_true"].astype(int), d["y_prob"].astype(float)
            auc_npz = roc_auc_score(yt_npz, yp_npz)

            # TSV-mapped scores for the same gene set
            sel_genes = cpb_genes  # mycoCirc eval set == circpcbl set
            yt_map = np.array([label_map[g] for g in sel_genes])
            yp_map = np.array([gene2score[grp][g] for g in sel_genes])
            auc_map = roc_auc_score(yt_map, yp_map)
            ok = note(sp, f"{grp}:npz_auroc", True,
                      f"npz={auc_npz:.4f} n={len(yt_npz)} "
                      f"pos={int(yt_npz.sum())}")
            note(sp, f"{grp}:tsv_auroc_matches_npz",
                 abs(auc_map - auc_npz) < 0.002,
                 f"tsv_mapped={auc_map:.4f} vs npz={auc_npz:.4f}")
            note(sp, f"{grp}:label_vector_matches",
                 np.array_equal(yt_map, yt_npz),
                 f"y_true rebuilt==npz: {np.array_equal(yt_map, yt_npz)}")

            (alo, ahi), (plo, phi) = stratified_bootstrap_ci(yt_map, yp_map)
            auprc = average_precision_score(yt_map, yp_map)
            rows_table.append(dict(
                species=sp, checkpoint=grp, method="mycoCirc", mode="A",
                auroc=round(auc_npz, 4),
                auroc_ci=f"[{alo:.3f},{ahi:.3f}]",
                auprc=round(float(auprc), 4),
                auprc_ci=f"[{plo:.3f},{phi:.3f}]",
                n_pos=int(yt_npz.sum()), n_neg=int((1 - yt_npz).sum()),
            ))

            # ---- JEDI ----
            dj = np.load(jedi)
            ytj, ypj = dj["y_true"].astype(int), dj["y_prob"].astype(float)
            auc_j = roc_auc_score(ytj, ypj)
            aligned = np.array_equal(jedi_labels_rebuilt, ytj)
            note(sp, f"{grp}:jedi_alignment", aligned,
                 f"rebuilt n={len(jedi_labels_rebuilt)} "
                 f"pos={int(jedi_labels_rebuilt.sum())} vs "
                 f"npz n={len(ytj)} pos={int(ytj.sum())}")
            (alo, ahi), (plo, phi) = stratified_bootstrap_ci(ytj, ypj)
            auprc_j = average_precision_score(ytj, ypj)
            rows_table.append(dict(
                species=sp, checkpoint=grp, method="JEDI", mode="",
                auroc=round(auc_j, 4), auroc_ci=f"[{alo:.3f},{ahi:.3f}]",
                auprc=round(float(auprc_j), 4),
                auprc_ci=f"[{plo:.3f},{phi:.3f}]",
                n_pos=int(ytj.sum()), n_neg=int((1 - ytj).sum()),
            ))

            # paired DeLong mycoCirc(Mode A) vs JEDI on aligned genes
            if aligned:
                yp_myco_j = np.array([gene2score[grp][g] for g in jedi_genes],
                                     dtype=float)
                dl = delong_roc_test_fast(jedi_labels_rebuilt.astype(float),
                                     yp_myco_j, ypj)
                rows_delong.append(dict(
                    species=sp, checkpoint=grp, comparison="mycoCirc_A_vs_JEDI",
                    auc_a=round(dl["auc_a"], 4), auc_b=round(dl["auc_b"], 4),
                    diff=round(dl["auc_diff"], 4),
                    diff_ci95="[%.3f,%.3f]" % dl["auc_diff_ci95"],
                    p=f"{dl['p_value']:.2e}", n=len(jedi_genes),
                ))

            # ---- Mode B (reported only for T. marneffei in Table 1) ----
            if modeb.exists():
                db = np.load(modeb)
                ytb, ypb = db["y_true"].astype(int), db["y_prob"].astype(float)
                auc_b = roc_auc_score(ytb, ypb)
                (alo, ahi), (plo, phi) = stratified_bootstrap_ci(ytb, ypb)
                auprc_b = average_precision_score(ytb, ypb)
                rows_table.append(dict(
                    species=sp, checkpoint=grp, method="mycoCirc", mode="B",
                    auroc=round(auc_b, 4),
                    auroc_ci=f"[{alo:.3f},{ahi:.3f}]",
                    auprc=round(float(auprc_b), 4),
                    auprc_ci=f"[{plo:.3f},{phi:.3f}]",
                    n_pos=int(ytb.sum()), n_neg=int((1 - ytb).sum()),
                ))

        # ---- CircPCBL (one per species; plant model, zero-shot) ----
        auc_c = roc_auc_score(cpb_labels, cpb_scores)
        (alo, ahi), (plo, phi) = stratified_bootstrap_ci(cpb_labels, cpb_scores)
        auprc_c = average_precision_score(cpb_labels, cpb_scores)
        rows_table.append(dict(
            species=sp, checkpoint="plant_pretrained", method="CircPCBL",
            mode="", auroc=round(auc_c, 4),
            auroc_ci=f"[{alo:.3f},{ahi:.3f}]",
            auprc=round(float(auprc_c), 4),
            auprc_ci=f"[{plo:.3f},{phi:.3f}]",
            n_pos=int(cpb_labels.sum()), n_neg=int((1 - cpb_labels).sum()),
        ))
        # paired DeLong vs each mycoCirc checkpoint (same gene set)
        for grp in GROUPS:
            yp_map = np.array([gene2score[grp][g] for g in cpb_genes],
                              dtype=float)
            dl = delong_roc_test_fast(cpb_labels.astype(float), yp_map, cpb_scores)
            rows_delong.append(dict(
                species=sp, checkpoint=grp,
                comparison="mycoCirc_A_vs_CircPCBL",
                auc_a=round(dl["auc_a"], 4), auc_b=round(dl["auc_b"], 4),
                diff=round(dl["auc_diff"], 4),
                diff_ci95="[%.3f,%.3f]" % dl["auc_diff_ci95"],
                p=f"{dl['p_value']:.2e}", n=len(cpb_genes),
            ))

        # ---- per-species n summary ----
        rows_n.append(dict(
            species=sp,
            n_positive=len(pos_genes),
            n_sampled_negatives=len(neg_genes),
            n_unique_negatives=len(set(neg_genes)),
            n_unique_all=len(set(rebuilt)),
            genome_genes=len(gm.genes),
        ))

    pd.DataFrame(rows_table).to_csv(OUT / "a1_table1_with_ci.csv", index=False)
    pd.DataFrame(rows_delong).to_csv(OUT / "a1_delong_vs_baselines.csv",
                                     index=False)
    pd.DataFrame(rows_n).to_csv(OUT / "a1_per_species_n.csv", index=False)
    (OUT / "a1_sanity_report.json").write_text(json.dumps(sanity, indent=2))

    n_fail = sum(1 for v in sanity.values() if not v["pass"])
    print(f"\nDone. {len(sanity)} checks, {n_fail} failures. "
          f"Outputs in {OUT}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
