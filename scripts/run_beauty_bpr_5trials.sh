#!/bin/bash
# Run BPR baseline on Beauty dataset with 5 different seeds
# Usage: ./run_beauty_bpr_5trials.sh

CONFIG_FILE="configs/recbole_beauty_bpr.yaml"
MODEL="BPR"
DATASET="amazon-beauty"

# Common seeds (same as our model for consistency)
SEEDS=(42 2023 2024 2025 12345)

echo "=========================================="
echo "Running BPR Baseline on Amazon Beauty"
echo "=========================================="
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
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
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${file}" ]; then
        echo "Error: Required file ${file} not found!"
        echo ""
        echo "Please run the Beauty dataset preparation pipeline first:"
        echo "  python scripts/prepare_beauty_dataset.py"
        exit 1
    fi
done

echo "✓ All required files found"
echo ""

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/beauty_baselines/bpr_${TIMESTAMP}"
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
    echo "=========================================="
    echo "Trial ${TRIAL_NUM}/5 - Seed: ${SEED}"
    echo "=========================================="
    echo ""

    # Create temporary config with modified seed
    TEMP_CONFIG="${OUTPUT_DIR}/config_seed_${SEED}.yaml"
    cp "${CONFIG_FILE}" "${TEMP_CONFIG}"

    # Modify seed in the temp config
    sed -i "s/seed: .*/seed: ${SEED}/" "${TEMP_CONFIG}"

    echo "Modified config saved to: ${TEMP_CONFIG}"
    echo "Running RecBole BPR..."
    echo ""

    # Run RecBole
    python -c "
from recbole.quick_start import run_recbole

run_recbole(
    model='${MODEL}',
    dataset='${DATASET}',
    config_file_list=['${TEMP_CONFIG}']
)
" 2>&1 | tee "${OUTPUT_DIR}/trial_${TRIAL_NUM}_seed_${SEED}.log"

    # Check exit status
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
echo "  python scripts/summarize_beauty_bpr_trials.py ${OUTPUT_DIR}"
echo ""
