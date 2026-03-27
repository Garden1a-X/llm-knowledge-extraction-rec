#!/usr/bin/env python3
"""
分析 ML-1M 测试集统计信息（不依赖 pandas）

计算每个用户在测试集中的平均交互数，用于校正 Recall 指标。
"""

from collections import defaultdict
import statistics


def split_data_simple(inter_path: str, train_ratio=0.7, val_ratio=0.1):
    """简单版本的数据分割（per-user time-based）"""

    # 读取数据
    with open(inter_path, 'r') as f:
        lines = f.readlines()

    # 跳过 header
    header = lines[0].strip()
    lines = lines[1:]

    # 按用户组织数据
    user_interactions = defaultdict(list)
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 4:
            continue
        user_id = parts[0]
        item_id = parts[1]
        rating = parts[2]
        timestamp = float(parts[3])

        user_interactions[user_id].append((timestamp, item_id))

    # 对每个用户按时间排序并分割
    train_interactions = []
    val_interactions = []
    test_interactions = []

    for user_id, interactions in user_interactions.items():
        # 按时间排序
        interactions.sort(key=lambda x: x[0])

        n = len(interactions)

        if n < 3:
            # 少于3条：全部放入训练集
            train_interactions.extend([(user_id, item) for _, item in interactions])
            continue

        # 按比例划分
        train_end = max(1, int(n * train_ratio))
        val_end = min(n - 1, train_end + max(1, int(n * val_ratio)))

        train_interactions.extend([(user_id, item) for _, item in interactions[:train_end]])
        val_interactions.extend([(user_id, item) for _, item in interactions[train_end:val_end]])
        test_interactions.extend([(user_id, item) for _, item in interactions[val_end:]])

    return train_interactions, val_interactions, test_interactions


def analyze_test_set(inter_path: str):
    """分析测试集统计信息"""

    print("="*60)
    print("ML-1M 测试集统计分析")
    print("="*60)

    # 分割数据
    print("\n正在分割数据...")
    train, val, test = split_data_simple(inter_path)

    total = len(train) + len(val) + len(test)
    print(f"\n总交互数: {total}")
    print(f"  训练集: {len(train)} ({len(train)/total*100:.1f}%)")
    print(f"  验证集: {len(val)} ({len(val)/total*100:.1f}%)")
    print(f"  测试集: {len(test)} ({len(test)/total*100:.1f}%)")

    # 统计测试集中每个用户的交互数
    test_user_counts = defaultdict(int)
    for user_id, item_id in test:
        test_user_counts[user_id] += 1

    counts = list(test_user_counts.values())

    print(f"\n测试集用户数: {len(test_user_counts)}")
    print(f"测试集总交互数: {len(test)}")

    avg_test_items = statistics.mean(counts)
    median_test_items = statistics.median(counts)
    std_test_items = statistics.stdev(counts) if len(counts) > 1 else 0
    min_test_items = min(counts)
    max_test_items = max(counts)

    print(f"\n每个用户的测试集交互数统计:")
    print(f"  平均值: {avg_test_items:.4f}")
    print(f"  中位数: {median_test_items:.2f}")
    print(f"  标准差: {std_test_items:.2f}")
    print(f"  最小值: {min_test_items}")
    print(f"  最大值: {max_test_items}")

    # 分布统计
    print(f"\n交互数分布:")
    count_dist = defaultdict(int)
    for count in counts:
        if count <= 10:
            count_dist[count] += 1
        else:
            count_dist['>10'] += 1

    for i in range(1, 11):
        count = count_dist[i]
        pct = count / len(counts) * 100
        print(f"  {i} 个交互: {count} 用户 ({pct:.1f}%)")

    count_10plus = count_dist['>10']
    pct_10plus = count_10plus / len(counts) * 100
    print(f"  >10 个交互: {count_10plus} 用户 ({pct_10plus:.1f}%)")

    # Recall 校正系数
    print(f"\n" + "="*60)
    print("Recall 校正系数")
    print("="*60)
    print(f"\n校正系数 (平均测试集交互数): {avg_test_items:.4f}")
    print(f"\n用法：RecBole_Recall ≈ Your_Recall / {avg_test_items:.4f}")
    print(f"     Your_Recall ≈ RecBole_Recall × {avg_test_items:.4f}")

    # 示例
    print(f"\n示例校正 (将我们的 Recall 除以校正系数以对齐 RecBole):")
    example_recalls = [
        ("Ours-Full", 0.5006),
        ("VBPR", 0.4397),
        ("MKGAT", 0.4347),
    ]

    print(f"\n{'方法':<15} {'原始 Recall@10':<18} {'校正后 Recall@10':<18} {'提升 vs BPR':<15}")
    print("-" * 70)
    bpr_recall_corrected = 0.1518  # RecBole BPR baseline
    for method, recall in example_recalls:
        corrected_recall = recall / avg_test_items
        improvement = (corrected_recall / bpr_recall_corrected - 1) * 100
        print(f"{method:<15} {recall:<18.4f} {corrected_recall:<18.4f} +{improvement:.1f}%")

    # 对比 RecBole 方法
    print(f"\n与 RecBole 基线对比 (验证一致性):")
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
