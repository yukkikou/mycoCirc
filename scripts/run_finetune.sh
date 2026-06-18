#!/bin/bash
# PanCirc-Fungi: Fine-tuning launch script
# Usage: bash scripts/run_finetune.sh Candida

set -e

GROUP=${1:-Candida}
PRETRAINED=${2:-checkpoints/pretrain/best.pt}
CONFIG="config/default.yaml"
OUTPUT_DIR="checkpoints/finetune"
LOG_DIR="logs/finetune"

python train/finetune.py \
    --group "$GROUP" \
    --pretrained "$PRETRAINED" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --log-dir "$LOG_DIR" \
    --device cuda
