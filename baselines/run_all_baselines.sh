#!/bin/bash
# Run all baseline models on MovieLens 1M

# Configuration
DATASET="ml-1m"
DATA_PATH="data/recbole"
OUTPUT_DIR="outputs/baselines"
DEVICE="cuda"
EPOCHS=300

echo "=================================="
echo "Running All Baseline Models"
echo "=================================="
echo "Dataset: $DATASET"
echo "Device: $DEVICE"
echo "Epochs: $EPOCHS"
echo "=================================="
echo ""

# Activate conda environment if needed
# conda activate xuao_llm_kg_rec

# 1. LightGCN - Pure collaborative filtering (no KG)
echo ">>> Running LightGCN..."
python baselines/run_baseline.py \
    --model LightGCN \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE \
    --epochs $EPOCHS

echo ""
echo "=================================="
echo ""

# 2. BPR - Traditional matrix factorization baseline
echo ">>> Running BPR..."
python baselines/run_baseline.py \
    --model BPR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE \
    --epochs $EPOCHS

echo ""
echo "=================================="
echo ""

# 3. NGCF - Another GNN baseline
echo ">>> Running NGCF..."
python baselines/run_baseline.py \
    --model NGCF \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE \
    --epochs $EPOCHS

echo ""
echo "=================================="
echo ""

# Note: KGAT and RippleNet require knowledge graph data
# We'll need to prepare KG data separately for these models

echo ""
echo "=================================="
echo "All baselines completed!"
echo "=================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
