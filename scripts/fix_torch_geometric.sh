#!/bin/bash
# Fix PyTorch Geometric installation
# This script installs the required dependencies for PyTorch Geometric

echo "=========================================="
echo "PyTorch Geometric Dependency Fix"
echo "=========================================="
echo ""

# Check current environment
echo "Current conda environment: $CONDA_DEFAULT_ENV"
if [ "$CONDA_DEFAULT_ENV" != "xuao_llm_kg_rec" ]; then
    echo "WARNING: Not in xuao_llm_kg_rec environment!"
    echo "Please activate: conda activate xuao_llm_kg_rec"
    exit 1
fi

echo ""
echo "Step 1: Checking PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch not installed!"
    exit 1
fi

echo ""
echo "Step 2: Getting PyTorch and CUDA versions..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'cpu')")

echo "Detected PyTorch: $TORCH_VERSION"
echo "Detected CUDA: $CUDA_VERSION"

echo ""
echo "Step 3: Installing PyTorch Geometric dependencies..."
echo "This will install: torch-scatter, torch-sparse, torch-cluster, torch-spline-conv"
echo ""

# Install PyG dependencies
# These must match PyTorch version exactly
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html

echo ""
echo "Step 4: Verifying installation..."
python -c "
import torch
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv
print('✓ PyTorch:', torch.__version__)
print('✓ PyTorch Geometric:', torch_geometric.__version__)
print('✓ All imports successful!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Installation successful!"
    echo "=========================================="
    echo ""
    echo "You can now run: bash scripts/run_ours_5_trials.sh ours_full"
else
    echo ""
    echo "=========================================="
    echo "✗ Installation failed!"
    echo "=========================================="
    echo ""
    echo "Please try manual installation:"
    echo "1. Check your PyTorch version: python -c 'import torch; print(torch.__version__)'"
    echo "2. Visit: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
    echo "3. Follow the installation guide for your specific PyTorch version"
    exit 1
fi
