#!/usr/bin/env python3
"""Aggregate corrected-model evaluation JSONs into manuscript-ready tables.

Outputs (revision/corrected_model/eval/results/):
  corrected_all_metrics.csv   one row per (species, checkpoint, mode)
  corrected_heldout_own.csv   held-out strains with their own-group checkpoint
  corrected_table1_modeA.csv  external species x checkpoint, Mode A (Table 1 layout)
  corrected_table1_modeB.csv  external species x checkpoint, Mode B (Table 1 layout)
  corrected_atlas_counts.csv  per-group and total p>=0.8 counts from the new atlas
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
BASE = PROJ / "revision" / "corrected_model"
RES = BASE / "eval" / "results"
ATLAS = BASE / "atlas"

rows = [json.loads(p.read_text()) for p in sorted(RES.glob("*.json"))]
if not rows:
    raise SystemExit("no eval JSONs yet")

df = pd.DataFrame(rows)
df["ci"] = df["ci"].apply(lambda v: f"[{v[0]:.3f},{v[1]:.3f}]" if isinstance(v, list) else v)
df = df.sort_values(["species", "checkpoint", "mode"]).reset_index(drop=True)
df.to_csv(RES / "corrected_all_metrics.csv", index=False)

heldout = ["P4", "C4", "F6"]
own = {"P4": "Candida", "C4": "Cryptococcus", "F6": "Filamentous"}
h = df[df.species.isin(heldout)].copy()
h = h[h.apply(lambda r: own.get(r.species) == r.checkpoint, axis=1)]
h.to_csv(RES / "corrected_heldout_own.csv", index=False)

ext = ~df.species.isin(heldout)
for mode in ("A", "B"):
    t = df[ext & (df["mode"] == mode)].pivot_table(
        index="species", columns="checkpoint", values="auroc")
    t.to_csv(RES / f"corrected_table1_mode{mode}.csv")

# ---- atlas counts ----
pred_dir = ATLAS / "predictions"
all_dir = ATLAS / "predictions_all"
if pred_dir.exists() and any(pred_dir.glob("*.tsv")):
    def nrows(p):
        with open(p, "rb") as f:
            return max(sum(1 for _ in f) - 1, 0)
    per = {p.stem: nrows(p) for p in sorted(pred_dir.glob("*.tsv"))}
    tot = sum(per.values())
    pd.DataFrame([dict(item="p>=0.8 total", value=tot),
                  dict(item="species predicted", value=len(per))]).to_csv(
        RES / "corrected_atlas_counts.csv", index=False)
    print(f"atlas: {len(per)} species, {tot:,} predictions at p>=0.8")

print(df.to_string(index=False))
