#!/bin/bash
# Run 5 trials with different seeds for a baseline model
# Usage: ./run_5_trials.sh MODEL_NAME
# Example: ./run_5_trials.sh LightGCN

MODEL=${1:-LightGCN}
DATASET="ml-1m"
DATA_PATH="data/recbole"
DEVICE="cuda"
EPOCHS=300

# Common seeds used in research
SEEDS=(42 2023 2024 2025 12345)

echo "=========================================="
echo "Running 5 trials for ${MODEL}"
echo "=========================================="
echo "Model: ${MODEL}"
echo "Seeds: ${SEEDS[@]}"
echo "Epochs: ${EPOCHS}"
echo "=========================================="
echo ""

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/baselines/multiple_trials/${MODEL}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# Run each trial
for i in {0..4}; do
    SEED=${SEEDS[$i]}
    TRIAL_NUM=$((i + 1))

    echo ""
    echo "=========================================="
    echo "Trial ${TRIAL_NUM}/5 - Seed: ${SEED}"
    echo "=========================================="
    echo ""

    # Run the baseline
    python baselines/run_baseline.py \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --data_path ${DATA_PATH} \
        --device ${DEVICE} \
        --epochs ${EPOCHS} \
        --seed ${SEED} \
        2>&1 | tee "${OUTPUT_DIR}/trial_${TRIAL_NUM}_seed_${SEED}.log"

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
echo "Logs saved to: ${OUTPUT_DIR}"
echo ""
echo "To extract results, run:"
echo "  python baselines/extract_trial_results.py ${OUTPUT_DIR}"
echo ""
