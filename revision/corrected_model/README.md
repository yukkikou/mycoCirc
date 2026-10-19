# Corrected-model results (single-model revision)

**Decision (2026-10-17):** the manuscript must describe exactly one model.
We adopt the **CircExp-zeroed fine-tuning recipe** (circRNA-expression channel
zeroed during fine-tuning, i.e. train–inference symmetric) and recompute
**everything**, including Mode A and the atlas.

Nothing here overwrites the original results: all outputs live under
`revision/corrected_model/`; the original `revision/results/`,
`mycoCircAtlas/predictions/` and `model_weights/` are untouched.

Checkpoints: `revision/checkpoints/nocircexp/{Candida,Cryptococcus,Filamentous}/final.pt`

---

## 1. Held-out strains (own-group checkpoint, cached evaluation sets)

| strain | Mode A (manuscript → corrected) | Mode B (manuscript → corrected) |
|---|---|---|
| P4 (C. auris) | 0.699 → **0.738** [0.683, 0.792] | 0.453 → **0.811** [0.761, 0.860] |
| C4 (C. neoformans var. neoformans) | 0.690 → **0.704** [0.673, 0.737] | 0.570 → **0.794** [0.766, 0.820] |
| F6 (F. venenatum) | 0.698 → **0.695** [0.660, 0.728] | 0.723 → **0.885** [0.863, 0.906] |

Source: `eval/results/corrected_heldout_own.csv`

## 2. External species — Mode A (12 cells)

Mean change vs manuscript **−0.016**; **10 of 12 cells decrease** (largest
−0.055, A. cristatus × Cryptococcus); one improves (+0.037, T. marneffei ×
Cryptococcus).

| species | Candida | Cryptococcus | Filamentous |
|---|---|---|---|
| A. cristatus | 0.695 | 0.631 | 0.702 |
| T. mentagrophytes | 0.685 | 0.631 | 0.686 |
| T. rubrum | 0.598 | 0.590 | 0.624 |
| T. marneffei | 0.666 | 0.691 | 0.676 |

Source: `eval/results/corrected_table1_modeA.csv`, `corrected_all_metrics.csv`

## 3. External species — Mode B (12 cells, corrected)

AUROC 0.690–0.802; e.g. T. marneffei 0.771 / 0.802 / 0.786 (manuscript reported
0.703 / 0.565 / 0.702 for its three models).
Source: `eval/results/corrected_table1_modeB.csv`

## 4. Statistics (Table 1)

AUROC with bootstrap 95% CI and AUPRC for every cell; paired DeLong vs
CircPCBL on identical gene sets: **all 12 comparisons significant**
(Δ = +0.040 to +0.182, p ≤ 7.5e-5).
Source: `eval/results/corrected_table1_stats.csv`,
`corrected_delong_vs_baselines.csv`

## 5. Matched-negative robustness (A2)

External Mode A AUROC under structure-matched negatives: **0.543–0.624**
(original: 0.53–0.63) — conclusion unchanged.
Source: `eval/results/corrected_a2_matched_negative.csv`

## 6. Threshold / calibration (A4)

Precision at p ≥ 0.8 (≥2-tool GT): 0.07–0.54; sensitivity 0.08–0.22;
specificity 0.90–0.96. Calibration ECE 0.10–0.46 — **worse calibrated than
the original model** (0.11–0.18), consistent with its larger high-probability
mass.
Source: `eval/results/corrected_a4_threshold_metrics.csv`,
`corrected_a4_calibration.csv`

## 7. Atlas (A5) — regenerated

| | original | corrected |
|---|---|---|
| species | 768 | 768 |
| predictions at p ≥ 0.8 | 319,860 | **650,284** |
| per-species median | ~418 | 723 |
| Candida / Cryptococcus / Filamentous | 36,835 / 2,361 / 280,664 | 52,561 / 7,085 / 590,638 |

Database: `atlas/database/myco_circ_atlas.db` (73 MB).
Source: `eval/results/corrected_a5_atlas_counts.csv`,
`corrected_a5_tier_precision.csv`

> **Note:** the p ≥ 0.8 tier roughly doubles because the corrected model's
> probability distribution is no longer collapsed/over-extreme. Precision per
> prediction is similar to the original model, so the tier is not worse per
> call — it is simply larger. A manuscript threshold choice should be revisited
> (raising to 0.9 does not improve precision; it saturates).

## 8. Species-embedding sweep (A7)

21 strain slots per species: A. cristatus 0.569–0.695 (range 0.126),
T. rubrum 0.504–0.628 (range 0.124) — same qualitative picture as before
(idiosyncratic, default near-best).
Source: `eval/results/corrected_a7_strainid_sensitivity.csv`

---

## Reproduce

```bash
# evaluation (7 species x 3 checkpoints x 2 modes)
sbatch revision/corrected_model/slurm/eval_corrected.slurm
python revision/corrected_model/scripts/aggregate_metrics.py
python revision/corrected_model/scripts/table1_corrected.py
python revision/corrected_model/scripts/dependent_analyses.py   # A2 + A4
# atlas (256 parallel tasks)
sbatch revision/corrected_model/slurm/atlas_corrected.slurm
python revision/corrected_model/scripts/a5_a7_corrected.py      # A5 + A7
python mycoCircAtlas/build_database.py --predictions-dir \
  revision/corrected_model/atlas/predictions \
  --db-path revision/corrected_model/atlas/database/myco_circ_atlas.db
```

## Open issue

`S5_open_issue.md` — S5 (S. pombe) is registered but in no split; author
decided to record it and not act yet.
