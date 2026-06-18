#!/bin/bash
#SBATCH -A ylab
#SBATCH -J pancirc-jedi
#SBATCH -p mix
#SBATCH -D /media/share/workdir/1610305236/Panfungi/4_model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /home/1610305236/log/%x-%A_%a.out
#SBATCH -e /home/1610305236/log/%x-%A_%a.err
#SBATCH --array 1-3

# ── Usage ─────────────────────────────────────────────────────────────
# sbatch --array=1-3 scripts/run_jedi.slurm
#   Array task ID → group: 1=Candida, 2=Cryptococcus, 3=Filamentous
# ──────────────────────────────────────────────────────────────────────

GROUP_NAMES=("Candida" "Cryptococcus" "Filamentous")

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    IDX=$((SLURM_ARRAY_TASK_ID - 1))
    GROUP="${GROUP_NAMES[$IDX]}"
else
    echo "Usage: sbatch --array=1-3 scripts/run_jedi.slurm"
    exit 1
fi

echo "=========================================="
echo "  JEDI Comparison: $GROUP"
echo "=========================================="

# ── Activate JEDI environment ──
source /media/share/home/1610305236/.local/share/mamba/etc/profile.d/mamba.sh
micromamba activate pancirc-jedi

# ── Step 1: Prepare data + Run JEDI ──
python scripts/run_jedi_comparison.py \
    --tsv all_lib_model_full.tsv \
    --group "$GROUP" \
    --jedi-dir other_models/JEDI-master \
    --out-dir results/jedi_comparison \
    --k 3 --L 4 --epochs 20 --batch-size 64 \
    --python-cmd python3

echo "=========================================="
echo "  JEDI $GROUP complete"
echo "=========================================="
