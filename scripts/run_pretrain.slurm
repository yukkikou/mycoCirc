#!/bin/bash
#SBATCH -A ylab
#SBATCH -J pancirc-pretrain
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -D /media/share/workdir/1610305236/Panfungi/4_model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /home/1610305236/log/%x-%A_%a.out
#SBATCH -e /home/1610305236/log/%x-%A_%a.err
#SBATCH -w L40vm2

# ── Activate environment ──
source /media/share/home/1610305236/.local/share/mamba/etc/profile.d/mamba.sh
micromamba activate pancirc-fungi

python train/pretrain.py \
    --config config/default.yaml \
    --output-dir checkpoints/pretrain \
    --log-dir logs/pretrain \
    --device cuda
