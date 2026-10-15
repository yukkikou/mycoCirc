#!/bin/bash
# Assemble the Zenodo release staging directory for the mycoCirc revision.
#
# Two model sets are shipped, EXPLICITLY SEPARATED:
#   models/paper_version/   — the checkpoints that reproduce every number in
#                             the manuscript (Table 1, Figs 1–2, atlas).
#   models/updated_recipe/  — alternatively-trained checkpoints (circRNA-
#                             expression channel zeroed during fine-tuning,
#                             train–inference symmetric). Better Mode B
#                             performance; NOT the source of manuscript
#                             numbers. Documented in README.md.
#
# Everything else (code, results, figures) is revision-side analysis support.
set -euo pipefail
cd "$(dirname "$0")/.."           # → revision/
REV=$(pwd)
ROOT=$(dirname "$REV")
OUT="$REV/release"

rm -rf "$OUT"
mkdir -p "$OUT"/{models/paper_version,models/updated_recipe,code,results,figures,manuscript_edits,response}

# ── models ─────────────────────────────────────────────────────────────────
cp "$ROOT"/model_weights/mycoCirc_candida.pt        "$OUT/models/paper_version/"
cp "$ROOT"/model_weights/mycoCirc_cryptococcus.pt   "$OUT/models/paper_version/"
cp "$ROOT"/model_weights/mycoCirc_filamentous.pt    "$OUT/models/paper_version/"
cp "$REV"/checkpoints/nocircexp/Candida/final.pt      "$OUT/models/updated_recipe/mycoCirc_candida_nocircexp.pt"
cp "$REV"/checkpoints/nocircexp/Cryptococcus/final.pt "$OUT/models/updated_recipe/mycoCirc_cryptococcus_nocircexp.pt"
cp "$REV"/checkpoints/nocircexp/Filamentous/final.pt  "$OUT/models/updated_recipe/mycoCirc_filamentous_nocircexp.pt"

# ── release README (template lives outside $OUT so rebuilds don't wipe it) ─
cp "$REV"/release_README.md "$OUT/README.md"

# ── code / results / figures / documents ───────────────────────────────────
cp "$REV"/scripts/*.py "$REV"/scripts/*.sh "$OUT/code/" 2>/dev/null || true
cp "$REV"/slurm/*.slurm "$OUT/code/" 2>/dev/null || true
cp "$REV"/results/*.csv "$REV"/results/*.json "$REV"/results/*.md "$OUT/results/" 2>/dev/null || true
# per-gene score matrices (source data behind the tables)
mkdir -p "$OUT/results/rescore" "$OUT/results/a3_predictions"
cp "$REV"/results/*.npz "$OUT/results/" 2>/dev/null || true
cp "$REV"/results/rescore/*.npz "$REV"/results/rescore/*.tsv "$OUT/results/rescore/" 2>/dev/null || true
cp "$REV"/results/a3_predictions/*.npz "$OUT/results/a3_predictions/" 2>/dev/null || true
cp -r "$REV"/results/a2_matched_sets "$OUT/results/" 2>/dev/null || true
cp "$REV"/figures/supp_fig_*.svg "$REV"/figures/supp_fig_*.pdf \
   "$REV"/figures/supp_fig_captions.md "$OUT/figures/" 2>/dev/null || true
cp "$REV"/manuscript_edits/*.md "$OUT/manuscript_edits/" 2>/dev/null || true
cp "$REV"/response/*.md "$OUT/response/" 2>/dev/null || true

# ── manifest with checksums ────────────────────────────────────────────────
{
  echo -e "path\tmd5\tsize_bytes\trole"
  cd "$OUT"
  while IFS= read -r f; do
    role="analysis support"
    case "$f" in
      README.md)               role="package overview — READ FIRST" ;;
      models/paper_version/*)  role="MANUSCRIPT-REPRODUCING checkpoint" ;;
      models/updated_recipe/*) role="ALT-RECIPE checkpoint (different numbers)" ;;
      code/*)                  role="analysis code" ;;
      results/*)               role="result table / source data" ;;
      figures/*)               role="supplementary figure" ;;
    esac
    printf "%s\t%s\t%s\t%s\n" "$f" "$(md5sum "$f" | cut -d' ' -f1)" \
           "$(stat -c%s "$f")" "$role"
  done < <(find . -type f ! -name MANIFEST.tsv | sed 's|^\./||' | sort)
} > "$OUT/MANIFEST.tsv"

echo "release assembled at $OUT"
du -sh "$OUT"
wc -l "$OUT/MANIFEST.tsv"
