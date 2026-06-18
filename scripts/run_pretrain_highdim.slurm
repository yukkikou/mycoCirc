#!/bin/bash
#SBATCH -A ylab
#SBATCH -J pancirc-highdim
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -D /media/share/workdir/1610305236/Panfungi/4_model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /home/1610305236/log/%x-%j.out
#SBATCH -e /home/1610305236/log/%x-%j.err

echo "=========================================="
echo "  HighDim Pretrain (k=5, embed_dim=128)"
echo "=========================================="

source /media/share/home/1610305236/.local/share/mamba/etc/profile.d/mamba.sh
micromamba activate pancirc-fungi

python train/pretrain.py \
    --config config/highdim.yaml \
    --output-dir checkpoints/pretrain_highdim \
    --log-dir logs/pretrain_highdim \
    --device cuda

echo "=========================================="
echo "  HighDim pretrain complete"
echo "=========================================="
