#!/usr/bin/env python3
"""Convert the corrected-model results into the exact table schemas that
figure_rivise.R reads, writing them to revision/r_inputs/.

The R script then only needs its input paths repointed here; every column
name, ordering and unit matches the originals so no other edit is required.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
REV = Path(__file__).resolve().parent
CORR = REV / "corrected_model" / "eval" / "results"
OUT = REV / "r_inputs"
OUT.mkdir(exist_ok=True)


def auprc_of(species, ckpt, mode):
    f = CORR / f"{species}_{ckpt}_{mode}.npz"
    if not f.exists():
        return ""
    d = np.load(f, allow_pickle=True)
    from sklearn.metrics import average_precision_score
    return f"{average_precision_score(d['y_true'], d['y_prob']):.4f}"


def extra_metrics(species, ckpt, mode):
    """Compute f1/accuracy/mcc at the 0.5 threshold from the per-gene scores."""
    f = CORR / f"{species}_{ckpt}_{mode}.npz"
    if not f.exists():
        return {}
    d = np.load(f, allow_pickle=True)
    y = d["y_true"].astype(int); p = d["y_prob"].astype(float)
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    acc = (tp + tn) / len(y)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return dict(f1=f"{f1:.4f}", accuracy=f"{acc:.4f}", mcc=f"{mcc:.4f}")

GROUPS = ["Candida", "Cryptococcus", "Filamentous"]
# genome-wide share of single-exon genes in the group's held-out strain
# (all parsed genes with >=1 exon): computed from the GTF, see README
SINGLE_ALL = {"Candida": 92.4, "Cryptococcus": 2.5, "Filamentous": 28.6}
STRAIN = {"Candida": "P4", "Cryptococcus": "C4", "Filamentous": "F6"}
BASE = pd.read_csv(PROJ / "comparison_model_benchmark.tsv", sep="\t")
allm_full = pd.read_csv(CORR / "corrected_all_metrics.csv")

# ---------------------------------------------------------------- 1) benchmark
# keep JEDI / CircPCBL rows verbatim; refresh the two mycoCirc rows
allm = pd.read_csv(CORR / "corrected_heldout_own.csv")
rows = []
for g in GROUPS:
    st = STRAIN[g]
    held = allm[(allm.species == st) & (allm.checkpoint == g)]
    for mode, label in (("A", "PanCirc (Genome+GTF)"), ("B", "PanCirc (+GeneExp)")):
        r = held[held["mode"] == mode]
        if not len(r):
            continue
        r = r.iloc[0]
        extra = extra_metrics(st, g, mode)
        old = BASE[(BASE.group == g) & (BASE.method == label)]
        note = old.note.iloc[0] if len(old) else "Default (k=3, embed=64) — held-out test"
        rows.append(dict(group=g, test_strain=st, method=label,
                         auroc=f"{r.auroc:.4f}", auprc=auprc_of(st, g, mode),
                         f1=extra.get("f1", ""), accuracy=extra.get("accuracy", ""),
                         mcc=extra.get("mcc", ""), note=note))
for g in GROUPS:
    for m in ("JEDI", "CircPCBL"):
        sub = BASE[(BASE.group == g) & (BASE.method == m)]
        if len(sub):
            rows.append(sub.iloc[0].to_dict())
pd.DataFrame(rows).to_csv(OUT / "comparison_model_benchmark.tsv", sep="\t", index=False)

# ---------------------------------------------------------------- 2) exon type
d = pd.read_csv(CORR / "exon_type_auroc_corrected.tsv", sep="\t")
d = d[d["group"].isin(GROUPS)]
out = []
for g in GROUPS:
    s = d[(d.group == g) & (d.exon_type == "single_exon")].iloc[0]
    m = d[(d.group == g) & (d.exon_type == "multi_exon")].iloc[0]
    t = d[(d.group == g) & (d.exon_type == "total")].iloc[0]
    out.append(dict(
        group=g,
        pct_single_exon_of_all=SINGLE_ALL[g],
        pct_single_exon_of_positive=round(100 * s.n_positive / t.n_positive, 1),
        auroc_single=f"{s.auroc:.4f}", auroc_multi=f"{m.auroc:.4f}",
        auroc_gap=f"{s.auroc - m.auroc:+.4f}",
        n_total=int(t.n_genes), n_positive=int(t.n_positive),
        n_single_exon_total=int(s.n_genes), n_single_exon_positive=int(s.n_positive),
        n_multi_exon_positive=int(m.n_positive)))
pd.DataFrame(out).to_csv(OUT / "exon_type_auroc_summary.tsv", sep="\t", index=False)

# ---------------------------------------------------------------- 3) PM1
# figure reads model_config == "default"; supply Mode A and Mode B rows
t1 = pd.read_csv(CORR / "corrected_table1_stats.csv")
pm1 = {"Candida": ("Talaromyces_marneffei", "Candida"),
       "Cryptococcus": ("Talaromyces_marneffei", "Cryptococcus"),
       "Filamentous": ("Talaromyces_marneffei", "Filamentous")}
rows = []
for g, (sp, ck) in pm1.items():
    for mode, mlabel in (("A", "Genome+GTF"), ("B", "+GeneExp")):
        r = t1[(t1.species == sp) & (t1.checkpoint == ck) & (t1["mode"] == mode)]
        if not len(r):
            continue
        r = r.iloc[0]
        extra = extra_metrics(sp, ck, mode)
        rows.append(dict(group=g, test_strain="PM1", model_config="default",
                         mode=mlabel, auroc=f"{r.auroc:.4f}", auprc=f"{r.auprc:.4f}",
                         f1=extra.get("f1", ""), accuracy=extra.get("accuracy", ""),
                         mcc=extra.get("mcc", ""), n_samples=int(r.n),
                         note=f"{g} finetuned -> PM1 (Talaromyces marneffei)"))
pd.DataFrame(rows).to_csv(OUT / "pm1_cross_species_test.tsv", sep="\t", index=False)

# ---------------------------------------------------------------- 4) expression ablation
ea = pd.read_csv(CORR / "expression_ablation_corrected.tsv", sep="\t", comment="#")
ea = ea[ea["group"].isin(GROUPS)]
map_cond = {"original": "full (CircExp+GeneExp)",
            "shuffle_geneexp": "shuffle GeneExp",
            "zero_geneexp": "zero both",
            "no_expression": "no expression"}
rows = []
for _, r in ea.iterrows():
    c = map_cond.get(r["mode"])
    if c is None:
        continue
    rows.append(dict(group=r["group"], cv_condition=c,
                     auroc=f"{r.auroc:.4f}", auprc=f"{r.auprc:.4f}",
                     f1=f"{r.f1:.4f}", accuracy=f"{r.accuracy:.4f}",
                     mcc=f"{r.mcc:.4f}",
                     note=("no perturbation" if c.startswith("full")
                           else "GeneExp perturbed at inference")))
pd.DataFrame(rows).to_csv(OUT / "comparison_ablation.tsv", sep="\t", index=False)

# ---------------------------------------------------------------- 5) feature importance
IG = REV / "corrected_model" / "eval" / "interpretability"
feat_frames = {}
for g in GROUPS:
    f = IG / g / "feature_importance.tsv"
    feat_frames[g] = pd.read_csv(f, sep="\t").set_index("feature")["importance"]
fi = pd.DataFrame(feat_frames)
fi.insert(0, "feature", fi.index)
fi = fi.reset_index(drop=True)
# keep the same feature names/order the R script expects (it filters reserved_)
fi.to_csv(OUT / "feature_importance_all.tsv", sep="\t", index=False)

# ---------------------------------------------------------------- 6) junction
j = pd.read_csv(CORR / "junction_topk_corrected.tsv", sep="\t")
j = j[j["group"].isin(GROUPS)]
j.to_csv(OUT / "junction_topk.tsv", sep="\t", index=False)

# ---------------------------------------------------------------- 7) from-scratch
fs = pd.read_csv(CORR / "fromscratch_corrected.csv")
rows = []
for g in GROUPS:
    r = fs[fs.group == g].iloc[0]
    rows.append(dict(group=g, mode="Mode_A",
                     from_scratch_auroc=f"{r.fromscratch_A:.4f}",
                     from_scratch_auprc="", from_scratch_f1="",
                     from_scratch_accuracy="", from_scratch_mcc="",
                     pretrained_auroc=f"{r.pretrained_A:.4f}",
                     pretrained_auprc="", pretrained_f1="",
                     pretrained_accuracy="", pretrained_mcc="",
                     auroc_diff=f"{r.fromscratch_A - r.pretrained_A:+.4f}"))
pd.DataFrame(rows).to_csv(OUT / "comparison_fromscratch_full.tsv", sep="\t", index=False)

print("wrote 7 tables to", OUT)
for f in sorted(OUT.iterdir()):
    if f.is_file():
        print(f"  {f.name}: {sum(1 for _ in open(f))-1} rows")
