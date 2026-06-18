#!/bin/bash
#SBATCH -A ylab
#SBATCH -J pancirc-finetune-hd
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -D /media/share/workdir/1610305236/Panfungi/4_model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /home/1610305236/log/%x-%A_%a.out
#SBATCH -e /home/1610305236/log/%x-%A_%a.err
#SBATCH --array 1-3

# Array task ID → group: 1=Candida, 2=Cryptococcus, 3=Filamentous
GROUP_NAMES=("Candida" "Cryptococcus" "Filamentous")
IDX=$((SLURM_ARRAY_TASK_ID - 1))
GROUP="${GROUP_NAMES[$IDX]}"
PRETRAINED="checkpoints/pretrain_highdim/best.pt"

echo "=========================================="
echo "  HighDim Finetune: $GROUP"
echo "  Pretrained: $PRETRAINED"
echo "=========================================="

source /media/share/home/1610305236/.local/share/mamba/etc/profile.d/mamba.sh
micromamba activate pancirc-fungi

python train/finetune.py \
    --group "$GROUP" \
    --config config/highdim.yaml \
    --pretrained $PRETRAINED \
    --output-dir checkpoints/finetune_highdim \
    --log-dir logs/finetune_highdim \
    --device cuda

echo "Finetune $GROUP complete"
