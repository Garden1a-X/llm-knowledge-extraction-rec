#!/bin/bash
# Run embedding dimension ablation experiments (5 trials each)
# Usage: ./scripts/run_emb_dim_ablation.sh [dim]
#   dim: 32, 128, 256, or all (default: all)
#
# Note: dim=64 is the default, results already exist from ours_full

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

DIM=${1:-all}

echo "============================================================"
echo "Embedding Dimension Ablation Study"
echo "============================================================"
echo "Dimensions to run: $DIM"
echo "Each dimension will run 5 trials (seeds: 42, 2023, 2024, 2025, 12345)"
echo "============================================================"
echo ""

run_dim() {
    local dim=$1
    echo ""
    echo "############################################################"
    echo "# Running dim=$dim (5 trials)"
    echo "############################################################"
    echo ""

    ./scripts/run_ours_5_trials.sh "ours_dim${dim}"
}

if [ "$DIM" == "all" ]; then
    # Run all dimensions (32, 128, 256)
    # Note: dim=64 is already done (ours_full)
    run_dim 32
    run_dim 128
    run_dim 256
elif [ "$DIM" == "32" ] || [ "$DIM" == "128" ] || [ "$DIM" == "256" ]; then
    run_dim "$DIM"
else
    echo "Invalid dimension: $DIM"
    echo "Valid options: 32, 128, 256, all"
    exit 1
fi

echo ""
echo "============================================================"
echo "Embedding Dimension Ablation Complete!"
echo "============================================================"
echo ""
echo "Summary of experiments:"
echo "  - dim=32:  configs/ours_dim32.yaml"
echo "  - dim=64:  (baseline, use ours_full results)"
echo "  - dim=128: configs/ours_dim128.yaml"
echo "  - dim=256: configs/ours_dim256.yaml"
echo ""
echo "Results saved to: outputs/ours_trials/"
echo "============================================================"
