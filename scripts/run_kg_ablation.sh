#!/bin/bash
# 图谱有效性消融实验启动脚本

echo "========================================"
echo "KG Ablation Study Launcher"
echo "========================================"
echo ""
echo "This script will run 5 experiments:"
echo "  1. KGAT + Visual KG"
echo "  2. KGCN + Metadata KG"
echo "  3. KGCN + Visual KG"
echo "  4. KGIN + Metadata KG"
echo "  5. KGIN + Visual KG"
echo ""
echo "Each experiment will run for 300 epochs (1 trial only)"
echo "Estimated time: ~2-3 hours for all 5 experiments"
echo ""

# 激活conda环境（如果需要）
# conda activate your_env_name

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit

# 检查Python脚本是否存在
if [ ! -f "scripts/run_kg_ablation.py" ]; then
    echo "Error: scripts/run_kg_ablation.py not found!"
    exit 1
fi

# 先进行dry run检查
echo "Checking experiment configuration..."
python scripts/run_kg_ablation.py --dry-run

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Configuration check failed!"
    exit 1
fi

echo ""
read -p "Do you want to proceed with running all experiments? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting experiments..."
    echo ""

    # 运行实验
    python scripts/run_kg_ablation.py

    echo ""
    echo "========================================"
    echo "All experiments completed!"
    echo "Check results in: outputs/kg_ablation/"
    echo "========================================"
else
    echo "Aborted."
    exit 0
fi
