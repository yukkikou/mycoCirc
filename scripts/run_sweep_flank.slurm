#!/bin/bash
#SBATCH -A ylab
#SBATCH -J pancirc-flk
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -D /media/share/workdir/1610305236/Panfungi/4_model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /home/1610305236/log/%x-%A_%a.out
#SBATCH -e /home/1610305236/log/%x-%A_%a.err
#SBATCH --array=1-15

source /media/share/home/1610305236/.local/share/mamba/etc/profile.d/mamba.sh
micromamba activate pancirc-fungi

python scripts/auto_train.py \
    --config config/default.yaml \
    --results-dir checkpoints/results \
    --features-dir checkpoints/features \
    --slurm-array \
    --device cuda \
    --num-workers 4 \
    --flank-sizes 50,100,150,200,300 \
    --k-values 3 \
    --embed-dims 64 \
    --gru-hiddens 64 \
    --fast
