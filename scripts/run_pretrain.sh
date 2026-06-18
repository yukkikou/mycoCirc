#!/bin/bash
# PanCirc-Fungi: Pre-training launch script (single GPU)
# Usage: bash scripts/run_pretrain.sh

set -e

CONFIG="config/default.yaml"
OUTPUT_DIR="checkpoints/pretrain"
LOG_DIR="logs/pretrain"

python train/pretrain.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --log-dir "$LOG_DIR" \
    --device cuda
