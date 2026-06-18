#!/bin/bash
#SBATCH -A ylab
#SBATCH -J pancirc-pm1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -D /media/share/workdir/1610305236/Panfungi/4_model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /home/1610305236/log/%x-%j.out
#SBATCH -e /home/1610305236/log/%x-%j.err

echo "=========================================="
echo "  PM1 Test — Talaromyces marneffei"
echo "=========================================="

source /media/share/home/1610305236/.local/share/mamba/etc/profile.d/mamba.sh
micromamba activate pancirc-fungi

python scripts/test_pm1.py \
    --config config/default.yaml \
    --checkpoint-dir checkpoints/finetune \
    --output results/pm1_test_results.json \
    --output-tsv results/pm1_test_results.tsv

echo "=========================================="
echo "  PM1 test complete"
echo "=========================================="
