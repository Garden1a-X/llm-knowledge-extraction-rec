#!/bin/bash
# Run VBPR and MMGCN experiments on Amazon Video Games dataset
# Usage:
#   ./scripts/run_videogames_multimodal.sh vbpr    # Run VBPR only
#   ./scripts/run_videogames_multimodal.sh mmgcn   # Run MMGCN only
#   ./scripts/run_videogames_multimodal.sh all     # Run both (default)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

METHOD=${1:-all}

echo "============================================================"
echo "Video Games Multimodal Baseline Experiments"
echo "============================================================"
echo "Method: $METHOD"
echo "Data: /data/xuao/llm-knowledge-extraction-rec/data/recbole/amazon-videogames"
echo "============================================================"
echo ""

# Activate conda environment if needed
# source activate your_env

python scripts/run_videogames_multimodal.py --method "$METHOD"

echo ""
echo "============================================================"
echo "Experiments completed!"
echo "Results saved to: outputs/videogames/"
echo "============================================================"
