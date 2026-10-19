#!/usr/bin/env python3
"""Build the p>=0.85 atlas tier from the corrected model's full predictions.

Reads  atlas/predictions_all/*.tsv   (all genes with p_circ)
Writes atlas_p085/predictions/*.tsv  (p_circ >= 0.85)
       atlas_p085/summary.csv        (per-species counts)
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "atlas" / "predictions_all"
DST = BASE / "atlas_p085" / "predictions"
DST.mkdir(parents=True, exist_ok=True)
THR = 0.85

rows = []
for f in sorted(SRC.glob("*.tsv")):
    df = pd.read_csv(f, sep="\t")
    keep = df[df.p_circ >= THR]
    keep.to_csv(DST / f.name, sep="\t", index=False)
    rows.append(dict(species=f.stem, n_genes=len(df), n_p_ge_085=len(keep),
                     frac=round(len(keep) / len(df), 5) if len(df) else 0))
s = pd.DataFrame(rows)
s.to_csv(BASE / "atlas_p085" / "summary.csv", index=False)
print(f"species: {len(s)}")
print(f"genes: {s.n_genes.sum():,}")
print(f"p>=0.85 predictions: {s.n_p_ge_085.sum():,}")
print(f"median per species: {int(s.n_p_ge_085.median())}")
print(s.n_p_ge_085.describe().round(1).to_string())
