#!/bin/bash
# =============================================================================
# mycoCirc — Test Dataset Prediction Script
# =============================================================================
# Runs mycoCirc on the 4 test species using the Filamentous fungi checkpoint.
#
# Usage:
#     bash test_data/predict_test.sh [--checkpoint model_weights/mycoCirc_filamentous.pt]
#
# Input:   test_data/genomes/{species}.fa
#          test_data/annotations/{species}.gtf
# Output:  test_data/expected_output/{species}_predicted.tsv
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEST_DIR="$PROJECT_DIR/test_data"
CHECKPOINT="${1:-$PROJECT_DIR/model_weights/mycoCirc_filamentous.pt}"
CONFIG="$PROJECT_DIR/config/default.yaml"

# Verify checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Download the model weights first:"
    echo "  wget https://github.com/yukkikou/mycoCirc/releases/download/v1.0/model_weights.tar.gz"
    echo "  tar xzf model_weights.tar.gz"
    exit 1
fi

# Predict each species
echo "========================================"
echo "  mycoCirc — Test Dataset Prediction"
echo "========================================"
echo ""

for genome in "$TEST_DIR/genomes/"*.fa; do
    species=$(basename "$genome" .fa)
    gtf="$TEST_DIR/annotations/${species}.gtf"
    output="$TEST_DIR/expected_output/${species}_predicted.tsv"

    echo "────────────────────────────────────────"
    echo "Species: $species"
    echo "Genome:  $genome"
    echo "GTF:     $gtf"
    echo "Output:  $output"
    echo "────────────────────────────────────────"

    # Run prediction
    python "$PROJECT_DIR/scripts/predict.py" \
        --genome "$genome" \
        --gtf "$gtf" \
        --checkpoint "$CHECKPOINT" \
        --config "$CONFIG" \
        --output "$output" \
        --batch-size 32 \
        --num-workers 4

    # Summary
    echo ""
    echo "  Results summary:"
    total=$(tail -n +2 "$output" | wc -l)
    n_positive=$(awk -F'\t' 'NR>1 && $6 >= 0.5 {count++} END {print count}' "$output")
    n_high=$(awk -F'\t' 'NR>1 && $6 >= 0.8 {count++} END {print count}' "$output")
    echo "    Genes predicted:     $total"
    echo "    circRNA-positive (p≥0.5): $n_positive"
    echo "    High confidence (p≥0.8):  $n_high"
    echo ""
done

echo "========================================"
echo "  All test species completed!"
echo "========================================"

# Show comparison command
echo ""
echo "To compare your predictions with pre-computed results:"
echo "  diff <(cut -f1,6 test_data/expected_output/Aspergillus_cristatus_predicted.tsv) \\"
echo "       <(cut -f1,6 test_data/expected_output/Aspergillus_cristatus_predicted.tsv)"
