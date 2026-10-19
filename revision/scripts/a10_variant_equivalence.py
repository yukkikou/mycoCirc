#!/usr/bin/env python3
"""A10: is the alternatively-trained (CircExp-zeroed) model statistically
distinguishable from the manuscript model in Mode A?

Motivation: the revision reports that the alternative recipe improves Mode B
substantially. A reviewer could reasonably ask why the manuscript model is
retained. This script provides the evidence-based answer: paired DeLong tests
between the two checkpoints on the SAME held-out genes, in the deployment
mode (Mode A, no RNA-seq), plus the external species.

Source data: per-gene probability TSVs produced by scripts/predict.py for
both checkpoints (revision/results/rescore/predict_*.tsv and *_nocircexp.tsv).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
RES = PROJ / "revision" / "results"
OUT = RES / "a10_variant_equivalence.csv"
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "revision" / "scripts"))

from a1_table1_ci import delong_roc_test_fast  # noqa: E402

PAIRS = [
    ("P4", "Candida"), ("C4", "Cryptococcus"), ("F6", "Filamentous"),
]


def labels_for(strain):
    d = PROJ / "checkpoints" / "features" / strain
    pos = [str(g) for g in np.load(d / "positive_gene_ids.npy",
                                   allow_pickle=True)]
    neg = [str(g) for g in np.load(d / "negative_gene_ids.npy",
                                   allow_pickle=True)]
    lab = {g: 1 for g in pos}
    lab.update({g: 0 for g in neg})
    return lab


rows = []
for strain, grp in PAIRS:
    lab = labels_for(strain)
    f_orig = RES / "rescore" / f"predict_{strain}_{grp}.tsv"
    f_alt = RES / "rescore" / f"nocircexp_{strain}_{grp}.tsv"
    if not (f_orig.exists() and f_alt.exists()):
        print(f"skip {strain}/{grp}: missing input")
        continue
    a = pd.read_csv(f_orig, sep="\t").set_index("gene_id")["p_circ"]
    b = pd.read_csv(f_alt, sep="\t").set_index("gene_id")["p_circ"]
    genes = [g for g in a.index if g in lab and g in b.index]
    y = np.array([lab[g] for g in genes], float)
    pa = a.loc[genes].values.astype(float)
    pb = b.loc[genes].values.astype(float)
    dl = delong_roc_test_fast(y, pa, pb)
    rows.append(dict(
        strain=strain, checkpoint=grp, mode="A", n=len(genes),
        auroc_original=round(dl["auc_a"], 4),
        auroc_alt_recipe=round(dl["auc_b"], 4),
        delta=round(dl["auc_diff"], 4),
        delta_ci95="[%.3f,%.3f]" % dl["auc_diff_ci95"],
        p=f"{dl['p_value']:.3g}",
        significant=bool(dl["p_value"] < 0.05)))
    print(f"{strain} ({grp}): original {dl['auc_a']:.4f} vs alt "
          f"{dl['auc_b']:.4f}  delta {dl['auc_diff']:+.4f}  "
          f"p={dl['p_value']:.3g}  "
          f"{'DIFFERENT' if dl['p_value'] < 0.05 else 'not distinguishable'}")

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\nsaved → {OUT}")
if len(df):
    print("significant(α=0.05):", int(df.significant.sum()), "/", len(df))
