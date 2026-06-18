#!/bin/bash
#SBATCH -A ylab
#SBATCH -J pancirc-extract
#SBATCH -p mix               # CPU 任务，mix 分区 12 个节点排队更快
#SBATCH -D /media/share/workdir/1610305236/Panfungi/4_model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G             # GTF 索引需要较多内存
#SBATCH -o /home/1610305236/log/%x-%A_%a.out
#SBATCH -e /home/1610305236/log/%x-%A_%a.err

# ── Activate environment ──
source /media/share/home/1610305236/.local/share/mamba/etc/profile.d/mamba.sh
micromamba activate pancirc-fungi

python scripts/extract_features.py \
    --tsv all_lib_model_full.tsv \
    --output checkpoints/features \
    --workers 8 \
    --flank-size 150
