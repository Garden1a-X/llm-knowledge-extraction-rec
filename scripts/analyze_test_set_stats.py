#!/usr/bin/env python3
"""
分析 ML-1M 测试集统计信息

计算每个用户在测试集中的平均交互数，用于校正 Recall 指标。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import split_data


def analyze_test_set(inter_path: str):
    """分析测试集统计信息"""

    print("="*60)
    print("ML-1M 测试集统计分析")
    print("="*60)

    # 使用和训练一样的配置
    train_df, val_df, test_df = split_data(
        inter_path=inter_path,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        time_based=True,
        random_seed=42,
        per_user_split=True
    )

    print(f"\n总交互数: {len(train_df) + len(val_df) + len(test_df)}")
    print(f"  训练集: {len(train_df)} ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
    print(f"  验证集: {len(val_df)} ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
    print(f"  测试集: {len(test_df)} ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")

    # 统计测试集中每个用户的交互数
    test_user_counts = test_df.groupby('user_id:token').size()

    print(f"\n测试集用户数: {len(test_user_counts)}")
    print(f"测试集总交互数: {len(test_df)}")

    avg_test_items = test_user_counts.mean()
    median_test_items = test_user_counts.median()
    std_test_items = test_user_counts.std()
    min_test_items = test_user_counts.min()
    max_test_items = test_user_counts.max()

    print(f"\n每个用户的测试集交互数统计:")
    print(f"  平均值: {avg_test_items:.2f}")
    print(f"  中位数: {median_test_items:.2f}")
    print(f"  标准差: {std_test_items:.2f}")
    print(f"  最小值: {min_test_items}")
    print(f"  最大值: {max_test_items}")

    # 分布统计
    print(f"\n交互数分布:")
    for i in range(1, 11):
        count = (test_user_counts == i).sum()
        pct = count / len(test_user_counts) * 100
        print(f"  {i} 个交互: {count} 用户 ({pct:.1f}%)")

    count_10plus = (test_user_counts > 10).sum()
    pct_10plus = count_10plus / len(test_user_counts) * 100
    print(f"  >10 个交互: {count_10plus} 用户 ({pct_10plus:.1f}%)")

    # Recall 校正系数
    print(f"\n" + "="*60)
    print("Recall 校正系数")
    print("="*60)
    print(f"\n校正系数 (平均测试集交互数): {avg_test_items:.4f}")
    print(f"\n用法：RecBole_Recall ≈ Your_Recall / {avg_test_items:.4f}")
    print(f"     Your_Recall ≈ RecBole_Recall × {avg_test_items:.4f}")

    # 示例
    print(f"\n示例校正:")
    example_recalls = [
        ("Ours-Full", 0.5006),
        ("VBPR", 0.4397),
        ("MKGAT", 0.4347),
    ]

    print(f"\n{'方法':<15} {'原始 Recall@10':<18} {'校正后 Recall@10':<18}")
    print("-" * 60)
    for method, recall in example_recalls:
        corrected_recall = recall / avg_test_items
        print(f"{method:<15} {recall:<18.4f} {corrected_recall:<18.4f}")

    # 对比 RecBole 方法
    print(f"\n与 RecBole 基线对比:")
    recbole_baselines = [
        ("BPR", 0.1518),
        ("LightGCN", 0.1536),
        ("KGAT", 0.1518),
    ]

    print(f"\n{'方法':<15} {'RecBole Recall@10':<20} {'反推 Hit@10':<15}")
    print("-" * 60)
    for method, recall in recbole_baselines:
        implied_hit = recall * avg_test_items
        print(f"{method:<15} {recall:<20.4f} {implied_hit:<15.4f}")

    return avg_test_items


if __name__ == '__main__':
    inter_path = 'data/recbole/ml-1m/ml-1m.inter'
    avg_test_items = analyze_test_set(inter_path)

    print(f"\n" + "="*60)
    print(f"✅ 结论: 使用校正系数 {avg_test_items:.4f} 来对齐 Recall 指标")
    print("="*60)
