#!/bin/bash
# Run embedding dimension ablation experiments (1 trial each, seed=42)
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
echo "Seed: 42 (single trial)"
echo "============================================================"
echo ""

run_dim() {
    local dim=$1
    local config="configs/ours_dim${dim}.yaml"

    echo ""
    echo "############################################################"
    echo "# Running dim=$dim"
    echo "############################################################"
    echo ""

    if [ ! -f "$config" ]; then
        echo "Error: Config file $config not found!"
        return 1
    fi

    python scripts/train_model_fast.py --config "$config"
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
echo "Expected results table:"
echo "  | dim |  NDCG@10  | vs dim=64 |"
echo "  |-----|-----------|-----------|"
echo "  |  32 |    ???    |    ???    |"
echo "  |  64 |   0.2707  |     -     | (baseline)"
echo "  | 128 |    ???    |    ???    |"
echo "  | 256 |    ???    |    ???    |"
echo "============================================================"
