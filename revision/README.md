# revision/ — analysis code

Code for the analyses added during revision of the manuscript. **Only the
code is published here**; result tables, the response letter, manuscript
edit lists and figures remain local (they travel with the manuscript
submission and the archived release instead).

## Scripts

| script | purpose |
|---|---|
| `a1_table1_ci.py` | bootstrap CIs, AUPRC, paired DeLong tests, per-species n (Table 1 statistics) |
| `a1b_heldout_via_predict.sh` | held-out per-gene scoring through the production `scripts/predict.py` path |
| `a1b_summarize.py` | summarise held-out CIs and the strain-slot sweep |
| `a2_matched_negatives.py` | matched-negative robustness (evaluation side) |
| `a3_gtf_only_baseline.py` | GTF-only logistic-regression / gradient-boosting baselines |
| `a4_threshold_calibration.py` | genome-wide imbalanced evaluation: thresholds, FDR, calibration |
| `a5_atlas_precision.py` | atlas high-precision tier: count reconciliation, precision evidence |
| `a6_expression_probe.py` | expression–detectability probe (untrained expression ranking) |
| `a7_strainid_via_predict.sh` | species-embedding (strain-slot) sensitivity sweep |
| `a8_circpcbl_retrain.py` | CircPCBL retrained from scratch on fungal data |
| `a9_phylo_structure_probe.py` | lookup-embedding vs phylogenetic-distance structure |
| `a_variants_heldout.sh` | held-out scoring for the alternatively-trained variants |
| `finetune_nocircexp.py` | fine-tuning with the circRNA-expression channel zeroed |
| `make_matched_features.py` | build matched-negative feature sets for re-training |
| `rescore.py` | per-gene re-scoring helper (held-out / external / strain-slot) |
| `plot_supp_figs.py` | supplementary figures (calibration, embedding sensitivity) |
| `build_release.sh` | assemble the archived release bundle |
| `wait_rescore.sh`, `a1b_sum.slurm`-equivalents | job helpers |

Job scripts live in `revision/slurm/` (not tracked: the repository ignores
`*.slurm`).

## Environment

Python ≥ 3.9, PyTorch ≥ 2.0, numpy, pandas, scikit-learn, scipy,
biopython, matplotlib. Analyses were run in a single micromamba
environment. Per-gene inference uses `scripts/predict.py` from the
repository root.

## Note on the two model sets

The revision trains alternative checkpoints (circRNA-expression channel
zeroed during fine-tuning) in addition to the manuscript-reproducing
checkpoints. Both sets are distributed with the archived release; the
manuscript-reproducing checkpoints remain the model of record for every
number reported in the paper.
