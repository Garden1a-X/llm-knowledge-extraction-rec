#!/bin/bash
# Run 5 trials with different seeds for Ours model
# Usage: ./run_ours_5_trials.sh CONFIG_NAME
# Example: ./run_ours_5_trials.sh ours_full

CONFIG_NAME=${1:-ours_full}
CONFIG_FILE="configs/${CONFIG_NAME}.yaml"
DEVICE="cuda"
EPOCHS=300

# Common seeds used in research
SEEDS=(42 2023 2024 2025 12345)

echo "=========================================="
echo "Running 5 trials for ${CONFIG_NAME}"
echo "=========================================="
echo "Config: ${CONFIG_FILE}"
echo "Seeds: ${SEEDS[@]}"
echo "Epochs: ${EPOCHS}"
echo "=========================================="
echo ""

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file ${CONFIG_FILE} not found!"
    exit 1
fi

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/ours_trials/${CONFIG_NAME}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# Backup original config
cp "${CONFIG_FILE}" "${OUTPUT_DIR}/original_config.yaml"

# Run each trial
for i in {0..4}; do
    SEED=${SEEDS[$i]}
    TRIAL_NUM=$((i + 1))

    echo ""
    echo "=========================================="
    echo "Trial ${TRIAL_NUM}/5 - Seed: ${SEED}"
    echo "=========================================="
    echo ""

    # Create temporary config with modified seed
    TEMP_CONFIG="${OUTPUT_DIR}/config_seed_${SEED}.yaml"
    cp "${CONFIG_FILE}" "${TEMP_CONFIG}"

    # Modify random_seed in the temp config
    sed -i "s/random_seed: .*/random_seed: ${SEED}/" "${TEMP_CONFIG}"

    echo "Modified config saved to: ${TEMP_CONFIG}"
    echo "Running training..."
    echo ""

    # Run the training
    python scripts/train_model_fast.py \
        --config "${TEMP_CONFIG}" \
        2>&1 | tee "${OUTPUT_DIR}/trial_${TRIAL_NUM}_seed_${SEED}.log"

    # Check exit status from PIPESTATUS (tee doesn't pass through exit code)
    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Trial ${TRIAL_NUM} completed successfully"
    else
        echo "✗ Trial ${TRIAL_NUM} failed (exit code: $EXIT_CODE)"
        echo "Check log: ${OUTPUT_DIR}/trial_${TRIAL_NUM}_seed_${SEED}.log"
    fi

    echo ""
done

echo ""
echo "=========================================="
echo "All trials completed!"
echo "=========================================="
echo "Logs and configs saved to: ${OUTPUT_DIR}"
echo ""
echo "To extract results, check the individual log files."
echo ""
