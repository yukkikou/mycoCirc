#!/bin/bash
#SBATCH -A ylab
#SBATCH -J pancirc-ablation
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -D /media/share/workdir/1610305236/Panfungi/4_model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /home/1610305236/log/%x-%j.out
#SBATCH -e /home/1610305236/log/%x-%j.err
#SBATCH -w L40vm2

echo "=========================================="
echo "  Component Ablation (all 3 groups)"
echo "=========================================="

# ── Activate environment ──
source /media/share/home/1610305236/.local/share/mamba/etc/profile.d/mamba.sh
micromamba activate pancirc-fungi

python scripts/run_ablations.py \
    --tsv all_lib_model_full.tsv \
    --config config/default.yaml \
    --checkpoint-dir checkpoints/finetune \
    --run-all \
    --output results/ablations.tsv

echo "=========================================="
echo "  Ablation complete"
echo "=========================================="
