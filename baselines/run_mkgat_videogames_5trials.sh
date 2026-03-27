#!/bin/bash
# Run MKGAT on Video Games dataset with 5 different seeds
# Usage: bash baselines/run_mkgat_videogames_5trials.sh

DATASET="amazon-videogames"
DATA_DIR="data/recbole/${DATASET}"
VISUAL_FEATURES="${DATA_DIR}/visual_features.npy"
OUTPUT_DIR="outputs/mkgat_videogames_5trials"
DEVICE="cuda"
EPOCHS=300

# Seeds for 5 trials (same as other baselines)
SEEDS=(42 2023 2024 2025 12345)

echo "=========================================="
echo "Running MKGAT on Video Games - 5 Trials"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Seeds: ${SEEDS[@]}"
echo "Epochs: ${EPOCHS}"
echo "=========================================="
echo ""

# Check if visual features exist
if [ ! -f "${VISUAL_FEATURES}" ]; then
    echo "ERROR: Visual features not found at ${VISUAL_FEATURES}"
    echo ""
    echo "Please run feature extraction first:"
    echo "  python baselines/extract_videogames_visual_features.py \\"
    echo "      --image_dir ${DATA_DIR}/images \\"
    echo "      --item_file ${DATA_DIR}/${DATASET}.item \\"
    echo "      --id_mapping ${DATA_DIR}/mappings/item_mapping.json \\"
    echo "      --output ${VISUAL_FEATURES}"
    echo ""
    exit 1
fi

echo "✓ Visual features found: ${VISUAL_FEATURES}"
echo ""

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRIAL_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
mkdir -p "${TRIAL_DIR}"

echo "Output directory: ${TRIAL_DIR}"
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

    # Run training
    python baselines/train_mkgat.py \
        --data_dir ${DATA_DIR} \
        --visual_features ${VISUAL_FEATURES} \
        --output_dir ${TRIAL_DIR} \
        --device ${DEVICE} \
        --epochs ${EPOCHS} \
        --seed ${SEED} \
        2>&1 | tee "${TRIAL_DIR}/trial_${TRIAL_NUM}_seed_${SEED}.log"

    if [ $? -eq 0 ]; then
        echo "✓ Trial ${TRIAL_NUM} completed successfully"
    else
        echo "✗ Trial ${TRIAL_NUM} failed"
    fi

    echo ""
done

echo ""
echo "=========================================="
echo "All trials completed!"
echo "=========================================="
echo "Logs and results saved to: ${TRIAL_DIR}"
echo ""

# Extract and aggregate results
python baselines/aggregate_mkgat_results.py ${TRIAL_DIR}

echo ""
