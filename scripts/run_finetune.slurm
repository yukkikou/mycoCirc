#!/bin/bash
#SBATCH -A ylab
#SBATCH -J pancirc-finetune
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -D /media/share/workdir/1610305236/Panfungi/4_model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /home/1610305236/log/%x-%A_%a.out
#SBATCH -e /home/1610305236/log/%x-%A_%a.err
#SBATCH -w L40vm2
#SBATCH --array 1-3

# ── Usage ─────────────────────────────────────────────────────────────
# Single job:   sbatch scripts/run_finetune.slurm Candida
# Array job:    sbatch --array=1-3 scripts/run_finetune.slurm
# From scratch: sbatch --array=1-3 scripts/run_finetune.slurm "" "--from-scratch"
#   Array task ID → group: 1=Candida, 2=Cryptococcus, 3=Filamentous
# ──────────────────────────────────────────────────────────────────────

GROUP_NAMES=("Candida" "Cryptococcus" "Filamentous")

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # Array mode: pick group by task ID
    IDX=$((SLURM_ARRAY_TASK_ID - 1))
    GROUP="${GROUP_NAMES[$IDX]}"
    PRETRAINED="checkpoints/pretrain/best.pt"
    EXTRA_ARGS="$@"
elif [ -n "$1" ]; then
    # Direct argument mode: $1=group $2=pretrained (optional)
    GROUP="$1"
    PRETRAINED="${2:-checkpoints/pretrain/best.pt}"
    shift 2 2>/dev/null
    EXTRA_ARGS="$@"
else
    echo "Usage: sbatch [--array=1-3] scripts/run_finetune.slurm [Candida|Cryptococcus|Filamentous] [pretrained.pt] [--extra-args...]"
    exit 1
fi

echo "=========================================="
echo "  Fine-tuning: $GROUP"
echo "  Pretrained:  $PRETRAINED"
echo "  Extra args:  $EXTRA_ARGS"
echo "=========================================="

# ── Activate environment ──
source /media/share/home/1610305236/.local/share/mamba/etc/profile.d/mamba.sh
micromamba activate pancirc-fungi

# Build command
CMD="python train/finetune.py \
    --group \"$GROUP\" \
    --config config/default.yaml \
    --output-dir checkpoints/finetune \
    --log-dir logs/finetune \
    --device cuda"

# If --from-scratch is in EXTRA_ARGS, don't pass --pretrained
if [[ "$EXTRA_ARGS" == *"--from-scratch"* ]]; then
    CMD="$CMD --from-scratch"
    # Remove --from-scratch from EXTRA_ARGS to avoid duplicate
    EXTRA_ARGS="${EXTRA_ARGS/--from-scratch/}"
else
    CMD="$CMD --pretrained $PRETRAINED"
fi

CMD="$CMD $EXTRA_ARGS"

echo "Running: $CMD"
eval $CMD
