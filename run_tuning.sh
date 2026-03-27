#!/bin/bash
# 一键启动并行调参（使用GPU 0,5,6,7）

echo "========================================"
echo "启动4个并行超参数调优实验"
echo "使用GPU: 0, 5, 6, 7"
echo "========================================"
echo ""

# 检查GPU是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 错误: 未找到nvidia-smi，请确认CUDA环境"
    exit 1
fi

echo "📊 当前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo ""

# 检查目标GPU是否可用
for gpu in 0 5 6 7; do
    if ! nvidia-smi -i $gpu &> /dev/null; then
        echo "⚠️  警告: GPU $gpu 不可用"
    fi
done
echo ""

# 启动并行调优
echo "🚀 启动并行调优..."
python scripts/run_parallel_tuning.py --multi-gpu 0,5,6,7

echo ""
echo "✅ 实验启动完成！"
echo ""
echo "📈 监控命令:"
echo "  实时GPU状态:  watch -n 1 nvidia-smi"
echo "  查看日志:      tail -f log/tune_parallel_*.log"
echo "  TensorBoard:   tensorboard --logdir log_tensorboard"
echo ""
echo "🔍 完成后查看结果:"
echo "  python scripts/compare_tuning_results.py"
