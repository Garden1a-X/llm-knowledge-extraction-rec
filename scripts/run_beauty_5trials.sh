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

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Run Ours method
run_ours() {
    echo "----------------------------------------"
    echo "Running Ours (Full) - 5 Trials"
    echo "----------------------------------------"

    CONFIG_FILE="configs/ours_full_beauty.yaml"
    OUTPUT_DIR="outputs/beauty/ours_full_${TIMESTAMP}"
    mkdir -p "${OUTPUT_DIR}"

    # Backup original config
    cp "${CONFIG_FILE}" "${OUTPUT_DIR}/original_config.yaml"

    for i in {0..4}; do
        SEED=${SEEDS[$i]}
        TRIAL_NUM=$((i + 1))

        echo ""
        echo ">>> Trial ${TRIAL_NUM}/5 - Seed: ${SEED}"

        # Create temporary config with modified seed
        TEMP_CONFIG="${OUTPUT_DIR}/config_seed_${SEED}.yaml"
        cp "${CONFIG_FILE}" "${TEMP_CONFIG}"

        # Modify random_seed in the temp config
        sed -i "s/random_seed: .*/random_seed: ${SEED}/" "${TEMP_CONFIG}"

        # Run the training
        python scripts/train_model_fast.py \
            --config "${TEMP_CONFIG}" \
            2>&1 | tee "${OUTPUT_DIR}/trial_${TRIAL_NUM}_seed_${SEED}.log"

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✓ Trial ${TRIAL_NUM} completed successfully"
        else
            echo "✗ Trial ${TRIAL_NUM} failed"
        fi
    done

    echo ""
    echo "✓ Ours completed. Results in ${OUTPUT_DIR}"
}

# Run BPR baseline
run_bpr() {
    echo "----------------------------------------"
    echo "Running BPR Baseline - 5 Trials"
    echo "----------------------------------------"

    CONFIG_FILE="configs/recbole_beauty_bpr.yaml"
    OUTPUT_DIR="outputs/beauty/bpr_${TIMESTAMP}"
    mkdir -p "${OUTPUT_DIR}"

    # Backup original config
    cp "${CONFIG_FILE}" "${OUTPUT_DIR}/original_config.yaml"

    for i in {0..4}; do
        SEED=${SEEDS[$i]}
        TRIAL_NUM=$((i + 1))

        echo ""
        echo ">>> Trial ${TRIAL_NUM}/5 - Seed: ${SEED}"

        # Create temporary config with modified seed
        TEMP_CONFIG="${OUTPUT_DIR}/config_seed_${SEED}.yaml"
        cp "${CONFIG_FILE}" "${TEMP_CONFIG}"

        # Modify seed in the temp config
        sed -i "s/^seed: .*/seed: ${SEED}/" "${TEMP_CONFIG}"

        # Run RecBole
        python -m recbole.quick_start.quick_start \
            --config_files "${TEMP_CONFIG}" \
            2>&1 | tee "${OUTPUT_DIR}/trial_${TRIAL_NUM}_seed_${SEED}.log"

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✓ Trial ${TRIAL_NUM} completed successfully"
        else
            echo "✗ Trial ${TRIAL_NUM} failed"
        fi
    done

    echo ""
    echo "✓ BPR completed. Results in ${OUTPUT_DIR}"
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
