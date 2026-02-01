#!/bin/bash
# Run 5-trial experiments for Amazon Beauty dataset
# Usage: bash scripts/run_beauty_5trials.sh [ours|bpr|all]

set -e

SEEDS=(42 123 456 789 2024)
METHOD=${1:-all}

echo "========================================"
echo "Amazon Beauty 5-Trial Experiments"
echo "========================================"
echo "Seeds: ${SEEDS[*]}"
echo "Method: $METHOD"
echo ""

# Run Ours method
run_ours() {
    echo "----------------------------------------"
    echo "Running Ours (Full) - 5 Trials"
    echo "----------------------------------------"

    for seed in "${SEEDS[@]}"; do
        echo ""
        echo ">>> Trial with seed=$seed"
        python scripts/train_model.py \
            --config configs/ours_full_beauty.yaml \
            --seed $seed \
            --output_dir outputs/beauty/ours_full/seed_$seed
    done

    echo ""
    echo "✓ Ours completed. Results in outputs/beauty/ours_full/"
}

# Run BPR baseline
run_bpr() {
    echo "----------------------------------------"
    echo "Running BPR Baseline - 5 Trials"
    echo "----------------------------------------"

    for seed in "${SEEDS[@]}"; do
        echo ""
        echo ">>> Trial with seed=$seed"
        python -m recbole.quick_start.quick_start \
            --config_files configs/recbole_beauty_bpr.yaml \
            --seed $seed
    done

    echo ""
    echo "✓ BPR completed."
}

# Main
case $METHOD in
    ours)
        run_ours
        ;;
    bpr)
        run_bpr
        ;;
    all)
        run_ours
        run_bpr
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Usage: bash scripts/run_beauty_5trials.sh [ours|bpr|all]"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"
