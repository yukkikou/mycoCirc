#!/bin/bash
# Package the regenerated mycoCircAtlas (p >= 0.85, corrected model) as a
# deployable web-application bundle. Nothing outside
# revision/atlas_p085_deploy/ is written.
set -euo pipefail

PROJ="/media/share/workdir/1610305236/Panfungi/4_model"
REV="${PROJ}/revision"
SRC="${REV}/corrected_model/atlas_p085"
DEPLOY="${REV}/atlas_p085_deploy"
PKG="${REV}/mycoCircAtlas_v2_p085.tar.gz"

echo "=============================================================="
echo "  Packaging mycoCircAtlas v2 (p >= 0.85)"
echo "=============================================================="

rm -rf "${DEPLOY}"
mkdir -p "${DEPLOY}"/{data,predictions,templates,static,logs}

# ── application code (unchanged from the published release) ────────────────
echo "  Copying application files..."
cp "${PROJ}/mycoCircAtlas/app.py"                       "${DEPLOY}/"
cp "${PROJ}/mycoCircAtlas/templates/"*.html             "${DEPLOY}/templates/"
cp "${PROJ}/mycoCircAtlas/static/"*.css                 "${DEPLOY}/static/"
cp "${PROJ}/mycoCircAtlas/build_database.py"            "${DEPLOY}/"
cp "${PROJ}/mycoCircAtlas/requirements-web.txt"         "${DEPLOY}/"

# ── regenerated database ───────────────────────────────────────────────────
echo "  Copying database (p>=0.85)..."
cp "${SRC}/database/myco_circ_atlas.db" "${DEPLOY}/data/"

# ── per-species prediction tables (p>=0.85) ────────────────────────────────
echo "  Copying prediction tables..."
cp "${SRC}/predictions/"*.tsv "${DEPLOY}/predictions/"
cp "${SRC}/summary.csv"       "${DEPLOY}/predictions/species_summary.csv"

# ── launch script ──────────────────────────────────────────────────────────
cat > "${DEPLOY}/launch.sh" << 'SCRIPT'
#!/bin/bash
# mycoCircAtlas v2 (p >= 0.85) — launch script
#   bash launch.sh [--port 8080] [--host 0.0.0.0]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON=""
for cmd in python3 python; do
    command -v "$cmd" &>/dev/null && { PYTHON="$cmd"; break; }
done
[ -z "$PYTHON" ] && { echo "ERROR: Python 3 not found (need >= 3.9)"; exit 1; }
PORT=5000; HOST=127.0.0.1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    *) shift ;;
  esac
done
echo "Starting mycoCircAtlas on ${HOST}:${PORT}"
"$PYTHON" app.py --port "${PORT}" --host "${HOST}"
SCRIPT
chmod +x "${DEPLOY}/launch.sh"

# ── deployment README ─────────────────────────────────────────────────────
cat > "${DEPLOY}/README.md" << 'DOC'
# mycoCircAtlas v2 (probability >= 0.85)

Web application for browsing precomputed fungal circRNA predictions.
**This version supersedes the earlier p >= 0.80 release.**

| | previous release | **this release** |
|---|---|---|
| prediction threshold | p >= 0.80 | **p >= 0.85** |
| predictions | 319,860 | **406,902** |
| species | 768 | 768 |
| model | pre-revision checkpoints | **single revised model** (train–inference symmetric) |

Per-species breakdown (`predictions/species_summary.csv`):
Candida-lineage model 41,324 · Cryptosporidium-lineage model 4,033 ·
Filamentous-lineage model 361,545 predictions.

The threshold was chosen from a sweep over the external validation species
(0.80 → 650,284 predictions at precision 0.248; **0.85 → 406,902 at 0.254**;
0.90 → 198,151 at 0.280 with recall falling to 0.056).

## Run

```bash
pip install -r requirements-web.txt
bash launch.sh --port 5000
```

## Contents

```
app.py                  Flask application (species browser, search, primer design)
build_database.py       rebuilds data/myco_circ_atlas.db from predictions/
data/                   SQLite database used by the app
predictions/            768 per-species TSV tables + species_summary.csv
templates/ static/      web assets
launch.sh               convenience launcher
```

Genome and GTF files are **not** bundled (they are large); the primer-design
module requires `--genomes-dir` pointing at a local Ensembl Fungi mirror.
DOC

# ── package ───────────────────────────────────────────────────────────────
echo "  Creating tarball..."
tar -C "${DEPLOY}" -czf "${PKG}" .

echo
echo "  bundle: ${PKG}"
ls -lh "${PKG}" | awk '{print "  size:  "$5}'
echo "  files: $(find "${DEPLOY}" -type f | wc -l)"
echo "  db:    $(du -h "${DEPLOY}/data/myco_circ_atlas.db" | cut -f1)"
echo "  preds: $(ls "${DEPLOY}/predictions/"*.tsv | wc -l) species tables"
