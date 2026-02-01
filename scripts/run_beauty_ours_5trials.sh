#!/bin/bash
# Run 5-trial experiments for Ours method on Amazon Beauty dataset
# Usage: bash scripts/run_beauty_ours_5trials.sh

set -e

SEEDS=(42 123 456 789 2024)
CONFIG_FILE="configs/ours_full_beauty.yaml"

echo "========================================"
echo "Amazon Beauty - Ours 5-Trial Experiments"
echo "========================================"
echo "Seeds: ${SEEDS[*]}"
echo "Config: ${CONFIG_FILE}"
echo ""

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file ${CONFIG_FILE} not found!"
    exit 1
fi

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/beauty/ours_full_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# Backup original config
cp "${CONFIG_FILE}" "${OUTPUT_DIR}/original_config.yaml"

echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Run each trial
for i in {0..4}; do
    SEED=${SEEDS[$i]}
    TRIAL_NUM=$((i + 1))

    echo ""
    echo "========================================"
    echo "Trial ${TRIAL_NUM}/5 - Seed: ${SEED}"
    echo "========================================"

    # Create temporary config with modified seed
    TEMP_CONFIG="${OUTPUT_DIR}/config_seed_${SEED}.yaml"
    cp "${CONFIG_FILE}" "${TEMP_CONFIG}"

    # Modify random_seed in the temp config
    sed -i "s/random_seed: .*/random_seed: ${SEED}/" "${TEMP_CONFIG}"

    echo "Config: ${TEMP_CONFIG}"
    echo ""

    # Run the training
    python scripts/train_model_fast.py \
        --config "${TEMP_CONFIG}" \
        2>&1 | tee "${OUTPUT_DIR}/trial_${TRIAL_NUM}_seed_${SEED}.log"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ""
        echo "✓ Trial ${TRIAL_NUM} completed successfully"
    else
        echo ""
        echo "✗ Trial ${TRIAL_NUM} failed"
    fi
done

echo ""
echo "========================================"
echo "All 5 trials completed!"
echo "========================================"
echo "Logs and configs saved to: ${OUTPUT_DIR}"
echo ""
