#!/bin/bash
# Assemble the v2 release bundle: the SINGLE symmetric-recipe model, its
# recomputed results, the regenerated p>=0.85 atlas, and the new figures.
#
# Unlike v1 there is only ONE model set — the manuscript results and the
# atlas both come from these checkpoints, so no paper/alternate split is
# needed. The superseded (asymmetric-recipe) release remains on Zenodo as
# the previous version.
set -euo pipefail
cd "$(dirname "$0")"
REV=$(pwd)
OUT="$REV/release_v2"

rm -rf "$OUT"
mkdir -p "$OUT"/{models,code,results,figures,figures_v2,atlas}

# ── models: the single set used for every reported number ──────────────────
cp "$REV"/checkpoints/nocircexp/Candida/final.pt        "$OUT/models/mycoCirc_candida.pt"
cp "$REV"/checkpoints/nocircexp/Cryptococcus/final.pt   "$OUT/models/mycoCirc_cryptococcus.pt"
cp "$REV"/checkpoints/nocircexp/Filamentous/final.pt    "$OUT/models/mycoCirc_filamentous.pt"
# from-scratch controls used for the pre-training ablation (Fig 1C)
mkdir -p "$OUT/models/fromscratch"
for g in Candida Cryptococcus Filamentous; do
  cp "$REV/corrected_model/checkpoints/fromscratch/$g/final.pt" \
     "$OUT/models/fromscratch/$g.pt" 2>/dev/null || true
done

# ── code ───────────────────────────────────────────────────────────────────
cp "$REV"/corrected_model/scripts/*.py "$OUT/code/" 2>/dev/null || true
cp "$REV"/corrected_model/slurm/*.slurm "$OUT/code/" 2>/dev/null || true
cp "$REV"/scripts/*.py "$REV"/scripts/*.sh "$OUT/code/" 2>/dev/null || true

# ── results (tables + per-gene scores) ─────────────────────────────────────
cp "$REV"/corrected_model/eval/results/*.csv "$OUT/results/" 2>/dev/null || true
cp "$REV"/corrected_model/eval/results/*.tsv "$OUT/results/" 2>/dev/null || true
cp "$REV"/corrected_model/eval/results/*.json "$OUT/results/" 2>/dev/null || true
cp "$REV"/corrected_model/eval/results/*.npz "$OUT/results/" 2>/dev/null || true
mkdir -p "$OUT/results/interpretability"
cp -r "$REV"/corrected_model/eval/interpretability/* "$OUT/results/interpretability/" 2>/dev/null || true
cp "$REV"/final_tables/*.csv "$OUT/results/" 2>/dev/null || true

# ── figures ────────────────────────────────────────────────────────────────
cp "$REV"/figures_v2/*.svg "$REV"/figures_v2/*.pdf "$OUT/figures_v2/" 2>/dev/null || true
cp "$REV"/figures/supp_fig_*.svg "$REV"/figures/supp_fig_*.pdf "$OUT/figures/" 2>/dev/null || true
cp "$REV"/figures/supp_fig_captions.md "$OUT/figures/" 2>/dev/null || true

# ── atlas: p>=0.85 predictions for all 768 species ────────────────────────
# per-species TSVs only (the full all-gene tables stay out of the archive;
# they are reproducible from the checkpoints with the shipped code)
tar -C "$REV/corrected_model/atlas_p085" -cf - predictions | tar -C "$OUT/atlas" -xf -
cp "$REV"/corrected_model/atlas_p085/summary.csv "$OUT/atlas/" 2>/dev/null || true

cp "$REV"/release_v2_README.md "$OUT/README.md"

# ── manifest ───────────────────────────────────────────────────────────────
{
  echo -e "path\tmd5\tsize_bytes\trole"
  cd "$OUT"
  while IFS= read -r f; do
    role="analysis support"
    case "$f" in
      README.md)          role="package overview — READ FIRST" ;;
      models/fromscratch/*) role="from-scratch control (Fig 1C)" ;;
      models/*)           role="FINAL model — reproduces all manuscript numbers and the atlas" ;;
      atlas/*)            role="regenerated p>=0.85 atlas prediction" ;;
      code/*)             role="analysis / training / inference code" ;;
      results/*)          role="result table or per-gene score" ;;
      figures*/*)         role="manuscript figure" ;;
    esac
    printf "%s\t%s\t%s\t%s\n" "$f" "$(md5sum "$f" | cut -d' ' -f1)" \
           "$(stat -c%s "$f")" "$role"
  done < <(find . -type f ! -name MANIFEST.tsv | sed 's|^\./||' | sort)
} > "$OUT/MANIFEST.tsv"

echo "assembled $OUT"
du -sh "$OUT"
find "$OUT" -type f | wc -l
