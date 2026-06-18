#!/bin/bash
#SBATCH -A ylab
#SBATCH -J mycoCirc-interpret
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -D /media/share/workdir/1610305236/Panfungi/4_model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /home/1610305236/log/%x-%j.out
#SBATCH -e /home/1610305236/log/%x-%j.err

echo "=========================================="
echo "  mycoCirc Interpretability Pipeline"
echo "=========================================="
echo "  Started: $(date)"
echo "=========================================="

source /media/share/home/1610305236/.local/share/mamba/etc/profile.d/mamba.sh
micromamba activate pancirc-fungi

# Step 1: Junction prediction evaluation (±5 bp fuzzy)
echo ""
echo ">>> Step 1: Junction prediction accuracy (±5 bp fuzzy)"
python scripts/evaluate_junction.py \
    --config config/default.yaml \
    --checkpoint-dir checkpoints/finetune \
    --output results/interpretability/junction_topk.tsv \
    --output-json results/interpretability/junction_details.json \
    --tolerance 5

# Step 2: Interpretability analysis
echo ""
echo ">>> Step 2: Interpretability analysis"
python scripts/run_interpretability.py \
    --config config/default.yaml \
    --checkpoint-dir checkpoints/finetune \
    --output-dir results/interpretability \
    --fig-dir figures

echo ""
echo "=========================================="
echo "  Pipeline complete: $(date)"
echo "=========================================="
