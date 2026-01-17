#!/bin/bash
# Run 5 trials with different seeds for Beauty dataset
# Usage: ./run_beauty_5_trials.sh

CONFIG_NAME="beauty_full"
CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

# Common seeds used in research (same as ML-1M for consistency)
SEEDS=(42 2023 2024 2025 12345)

echo "=========================================="
echo "Running 5 trials for Amazon Beauty dataset"
echo "=========================================="
echo "Config: ${CONFIG_FILE}"
echo "Seeds: ${SEEDS[@]}"
echo "=========================================="
echo ""

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file ${CONFIG_FILE} not found!"
    exit 1
fi

# Check if required data files exist
echo "Checking required data files..."
REQUIRED_FILES=(
    "data/recbole/amazon-beauty/amazon-beauty.inter"
    "data/recbole/amazon-beauty/amazon-beauty.item.kg"
    "data/recbole/amazon-beauty/amazon-beauty.user.kg"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${file}" ]; then
        echo "Error: Required file ${file} not found!"
        echo ""
        echo "Please run the Beauty dataset preparation pipeline first:"
        echo "  1. python scripts/prepare_beauty_dataset.py"
        echo "  2. python scripts/beauty_phase4_extraction.py --api_key YOUR_KEY"
        echo "  3. python scripts/convert_beauty_phase4_to_kg.py"
        echo "  4. python scripts/extract_beauty_user_interests.py --api_key YOUR_KEY"
        echo "  5. python scripts/convert_beauty_user_interests_to_kg.py"
        exit 1
    fi
done

echo "✓ All required files found"
echo ""

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/beauty_trials/${CONFIG_NAME}_${TIMESTAMP}"
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
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "To summarize results, run:"
echo "  python scripts/summarize_beauty_trials.py ${OUTPUT_DIR}"
echo ""
