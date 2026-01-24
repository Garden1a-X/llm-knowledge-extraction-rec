#!/bin/bash
# Run VBPR with 5 different random seeds for statistical significance

set -e

# Configuration
DATA_DIR="data/recbole/ml-1m"
VISUAL_FEATURES="data/recbole/ml-1m/visual_features.npy"
OUTPUT_BASE="outputs/vbpr_5trials"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"

# Seeds for 5 trials
SEEDS=(42 2023 2024 2025 12345)

# Hyperparameters (default)
EMBEDDING_DIM=64
REG_WEIGHT=1e-5
EPOCHS=300
BATCH_SIZE=1024
LR=0.001
EARLY_STOP=10
DEVICE="cuda"

echo "=========================================="
echo "VBPR - 5 Trials Evaluation"
echo "=========================================="
echo "Data: ${DATA_DIR}"
echo "Visual Features: ${VISUAL_FEATURES}"
echo "Output: ${OUTPUT_DIR}"
echo "Seeds: ${SEEDS[@]}"
echo "=========================================="

# Check visual features exist
if [ ! -f "${VISUAL_FEATURES}" ]; then
    echo "ERROR: Visual features not found at ${VISUAL_FEATURES}"
    echo "Please run: python baselines/extract_visual_features.py"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Save configuration
cat > "${OUTPUT_DIR}/config.json" <<EOF
{
  "data_dir": "${DATA_DIR}",
  "visual_features": "${VISUAL_FEATURES}",
  "embedding_dim": ${EMBEDDING_DIM},
  "reg_weight": ${REG_WEIGHT},
  "epochs": ${EPOCHS},
  "batch_size": ${BATCH_SIZE},
  "lr": ${LR},
  "early_stop": ${EARLY_STOP},
  "device": "${DEVICE}",
  "seeds": [${SEEDS[@]}]
}
EOF

# Run 5 trials
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    TRIAL_NUM=$((i + 1))
    TRIAL_OUTPUT="${OUTPUT_DIR}/seed_${SEED}_$(date +%Y%m%d_%H%M%S)"

    echo ""
    echo "=========================================="
    echo "Trial ${TRIAL_NUM}/5 (Seed: ${SEED})"
    echo "=========================================="

    python baselines/train_vbpr.py \
        --data_dir "${DATA_DIR}" \
        --visual_features "${VISUAL_FEATURES}" \
        --output_dir "${TRIAL_OUTPUT}" \
        --embedding_dim ${EMBEDDING_DIM} \
        --reg_weight ${REG_WEIGHT} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --early_stop ${EARLY_STOP} \
        --device ${DEVICE} \
        --seed ${SEED} \
        2>&1 | tee "${OUTPUT_DIR}/trial_${TRIAL_NUM}_seed_${SEED}.log"

    echo "✓ Trial ${TRIAL_NUM} completed"
done

echo ""
echo "=========================================="
echo "All 5 trials completed!"
echo "=========================================="

# Aggregate results
echo ""
echo "Aggregating results..."

python -c "
import json
import numpy as np
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')
seeds = [${SEEDS[@]}]

# Collect results from all trials
all_results = []
for seed in seeds:
    # Find the results.json for this seed
    trial_dirs = list(output_dir.glob(f'seed_{seed}_*'))
    if trial_dirs:
        results_file = trial_dirs[0] / 'results.json'
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
                all_results.append(results)

if not all_results:
    print('ERROR: No results found!')
    exit(1)

# Extract metrics
metrics = ['ndcg@10', 'recall@10', 'precision@10']
aggregated = {}

for metric in metrics:
    values = [r['test_metrics'][metric] for r in all_results]
    aggregated[metric] = {
        'mean': np.mean(values),
        'std': np.std(values),
        'values': values
    }

# Save aggregated results
output = {
    'n_trials': len(all_results),
    'seeds': seeds,
    'test_metrics': aggregated
}

with open(output_dir / 'aggregated_results.json', 'w') as f:
    json.dump(output, f, indent=2)

# Print summary
print('')
print('='*50)
print('Aggregated Results (5 trials):')
print('='*50)
for metric in metrics:
    mean = aggregated[metric]['mean']
    std = aggregated[metric]['std']
    print(f'{metric:15s}: {mean:.4f} ± {std:.4f}')
print('='*50)
print(f'Full results saved to: {output_dir / \"aggregated_results.json\"}')
"

echo ""
echo "Done!"
