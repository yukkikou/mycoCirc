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
mkdir -p "$OUT"/{models/paper_version,models/updated_recipe,code,results,figures}

# ── models ─────────────────────────────────────────────────────────────────
cp "$ROOT"/model_weights/mycoCirc_candida.pt        "$OUT/models/paper_version/"
cp "$ROOT"/model_weights/mycoCirc_cryptococcus.pt   "$OUT/models/paper_version/"
cp "$ROOT"/model_weights/mycoCirc_filamentous.pt    "$OUT/models/paper_version/"
cp "$REV"/checkpoints/nocircexp/Candida/final.pt      "$OUT/models/updated_recipe/mycoCirc_candida_nocircexp.pt"
cp "$REV"/checkpoints/nocircexp/Cryptococcus/final.pt "$OUT/models/updated_recipe/mycoCirc_cryptococcus_nocircexp.pt"
cp "$REV"/checkpoints/nocircexp/Filamentous/final.pt  "$OUT/models/updated_recipe/mycoCirc_filamentous_nocircexp.pt"

# ── release README (template lives outside $OUT so rebuilds don't wipe it) ─
cp "$REV"/release_README.md "$OUT/README.md"

# ── VERSION ────────────────────────────────────────────────────────────────
cat > "$OUT/VERSION" <<'VER'
mycoCirc revision release v1.0
assembled: __DATE__
contents: two checkpoint sets (paper_version = manuscript-reproducing;
          updated_recipe = circRNA-expression-zeroed alternative),
          analysis code, result tables, supplementary figures,
          response letter and manuscript edit lists.
see README.md; file list with md5 in MANIFEST.tsv
VER
sed -i "s/__DATE__/$(date -u +%Y-%m-%d)/" "$OUT/VERSION"

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
# Local count-reconciliation bookkeeping stays private (working-copy drift
# relative to the released database version is not part of the public record)
rm -f "$OUT/results/a5_atlas_count_reconciliation.csv"
if [ -f "$OUT/results/a5_high_confidence_evidence.md" ]; then
  python3 - "$OUT/results/a5_high_confidence_evidence.md" <<'PYSTRIP'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
a = s.find("## Atlas count reconciliation")
b = s.find("## Empirical precision")
if a != -1 and b > a:
    p.write_text(s[:a] + s[b:])
PYSTRIP
fi

cp "$REV"/figures/supp_fig_*.svg "$REV"/figures/supp_fig_*.pdf \
   "$REV"/figures/supp_fig_captions.md "$OUT/figures/" 2>/dev/null || true
# PUBLIC release contains code, data, figures and models only.
# Internal working documents are deliberately NOT shipped:
#   manuscript_edits/ (strategy notes, reviewer-response drafting, verbatim
#                      manuscript text) and response/ (response letter,
#                      traceability map) are peer-review correspondence and
#                      stay private until/unless the journal publishes them.

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
