#!/bin/bash
# 重新运行所有我们的方法（修复数据泄露后）
# 2026-01-18: 修复了图构建和评估中的数据泄露问题，需要重跑所有实验

set -e  # 遇到错误立即退出

echo "=========================================="
echo "重新运行所有我们的方法（修复数据泄露后）"
echo "=========================================="
echo ""
echo "修复内容："
echo "1. 图构建只用训练集（不再包含test/val数据）"
echo "2. 测试评估排除val items"
echo "3. 训练脚本调用顺序修正"
echo ""
echo "预期：分数会显著降低到合理水平"
echo ""

# 配置列表
configs=(
    "configs/ours_full.yaml"
    "configs/ours_cf_only.yaml"
    "configs/ours_kg_only.yaml"
    "configs/ours_wo_contrast.yaml"
    "configs/ours_wo_mask.yaml"
)

# 配置名称（用于日志）
config_names=(
    "Ours-Full (完整模型)"
    "Ours-CF-Only (只用CF视图)"
    "Ours-KG-Only (只用KG视图)"
    "Ours-wo-Contrast (无对比学习)"
    "Ours-wo-Mask (无Mask机制)"
)

# 记录开始时间
start_time=$(date +%s)

# 逐个运行
for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    name="${config_names[$i]}"

    echo ""
    echo "=========================================="
    echo "[$((i+1))/${#configs[@]}] 运行: $name"
    echo "配置文件: $config"
    echo "=========================================="
    echo ""

    # 检查配置文件是否存在
    if [ ! -f "$config" ]; then
        echo "❌ 错误: 配置文件不存在: $config"
        exit 1
    fi

    # 运行训练
    python scripts/train_model.py --config "$config"

    echo ""
    echo "✓ 完成: $name"
    echo ""
done

# 计算总耗时
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo ""
echo "=========================================="
echo "🎉 所有实验运行完成！"
echo "=========================================="
echo "总耗时: ${hours}h ${minutes}m ${seconds}s"
echo ""
echo "下一步："
echo "1. 查看结果日志"
echo "2. 对比修复前后的分数差异"
echo "3. 分析真实性能表现"
echo "4. 更新EXPERIMENT_TRACKING.md"
echo ""
