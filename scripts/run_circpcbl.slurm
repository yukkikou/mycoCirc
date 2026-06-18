#!/bin/bash
#SBATCH -A ylab
#SBATCH -J pancirc-circpcbl
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -D /media/share/workdir/1610305236/Panfungi/4_model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /home/1610305236/log/%x-%A_%a.out
#SBATCH -e /home/1610305236/log/%x-%A_%a.err
#SBATCH --array 1-3

# ── Usage ─────────────────────────────────────────────────────────────
# Zero-shot:  sbatch --array=1-3 scripts/run_circpcbl.slurm
# Train:      sbatch --array=1-3 scripts/run_circpcbl.slurm train
# ──────────────────────────────────────────────────────────────────────

GROUP_NAMES=("Candida" "Cryptococcus" "Filamentous")
MODE="${1:-zero-shot}"

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    IDX=$((SLURM_ARRAY_TASK_ID - 1))
    GROUP="${GROUP_NAMES[$IDX]}"
else
    echo "Usage: sbatch --array=1-3 scripts/run_circpcbl.slurm [zero-shot|train]"
    exit 1
fi

echo "=========================================="
echo "  CircPCBL Comparison: $GROUP (mode: $MODE)"
echo "=========================================="

# ── Activate environment ──
source /media/share/home/1610305236/.local/share/mamba/etc/profile.d/mamba.sh
micromamba activate pancirc-fungi

# ── Run CircPCBL comparison ──
python scripts/run_circpcbl_comparison.py \
    --tsv all_lib_model_full.tsv \
    --group "$GROUP" \
    --mode "$MODE" \
    --circpcbl-dir other_models/CircPCBL-main \
    --out-dir results/circpcbl_comparison \
    --batch-size 16

echo "=========================================="
echo "  CircPCBL $GROUP ($MODE) complete"
echo "=========================================="
