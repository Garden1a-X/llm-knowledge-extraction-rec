#!/usr/bin/env python3
"""
分析用户交互分布和LLM调用成本估算
"""

import pandas as pd
import numpy as np
from datetime import datetime

# 配置参数
SHORT_TERM_DAYS = 21  # 短期兴趣桶大小
LONG_TERM_BUCKETS = 4  # 每4个短期桶总结一次长期兴趣

# 加载数据
print("加载评分数据...")
df = pd.read_csv('../data/recbole/ml-1m/ml-1m.inter', sep='\t')
df.columns = [col.split(':')[0] for col in df.columns]

print(f"✓ 数据规模: {len(df):,} 条评分")
print(f"✓ 用户数: {df['user_id'].nunique():,}")
print(f"✓ 物品数: {df['item_id'].nunique():,}")
print()

# 按用户和时间排序
df_sorted = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

total_interactions = len(df)
num_users = df['user_id'].nunique()
avg_interactions_per_user = total_interactions / num_users

# 计算每个用户的桶数和LLM调用次数
user_llm_calls = []

print("按活跃时间分桶分析...")
print("="*70)

for user_id in df['user_id'].unique():
    user_data = df_sorted[df_sorted['user_id'] == user_id].copy()

    timestamps = user_data['timestamp'].values

    if len(timestamps) == 0:
        continue

    # 计算时间跨度
    time_span_seconds = timestamps[-1] - timestamps[0]
    time_span_days = time_span_seconds / (24 * 3600)

    # 统计活跃天数（有交互的不同日期数）
    unique_dates = set()
    for ts in timestamps:
        date = datetime.fromtimestamp(ts).date()
        unique_dates.add(date)

    active_days = len(unique_dates)

    # 计算短期桶数（每21天活跃时间）
    num_short_term_buckets = max(1, int(np.ceil(active_days / SHORT_TERM_DAYS)))

    # 计算LLM调用次数（每4个短期桶调用1次）
    # 即使不足4个桶，也至少调用1次
    num_llm_calls = max(1, int(np.ceil(num_short_term_buckets / LONG_TERM_BUCKETS)))

    user_llm_calls.append({
        'user_id': user_id,
        'num_interactions': len(user_data),
        'active_days': active_days,
        'time_span_days': time_span_days,
        'num_short_buckets': num_short_term_buckets,
        'num_llm_calls': num_llm_calls
    })

# 转换为DataFrame
llm_calls_df = pd.DataFrame(user_llm_calls)

print(f"✓ 分析完成")
print(f"  用户数: {len(llm_calls_df):,}")
print()

# 统计信息
print("活跃天数统计:")
print(llm_calls_df['active_days'].describe())
print()

print("短期桶数统计:")
print(llm_calls_df['num_short_buckets'].describe())
print()

print("LLM调用次数统计（每用户）:")
print(llm_calls_df['num_llm_calls'].describe())
print()

# 总LLM调用次数
total_llm_calls = llm_calls_df['num_llm_calls'].sum()
avg_llm_calls = llm_calls_df['num_llm_calls'].mean()
median_llm_calls = llm_calls_df['num_llm_calls'].median()

print("="*70)
print("LLM调用成本估算（最终方案）")
print("="*70)
print(f"总LLM调用次数: {total_llm_calls:,}")
print(f"平均每用户调用: {avg_llm_calls:.1f} 次")
print(f"中位数每用户调用: {median_llm_calls:.0f} 次")
print()

# 成本估算
costs = {
    'gpt-4o-mini ($0.001/call)': total_llm_calls * 0.001,
    'gpt-4o-mini ($0.002/call)': total_llm_calls * 0.002,
    'gpt-4o ($0.01/call)': total_llm_calls * 0.01,
}

for model, cost in costs.items():
    print(f"  {model:30s}: ${cost:7.2f}")

print("="*70)
print()

# 最终方案总结
print("="*70)
print("最终方案总结")
print("="*70)
print()
print("📊 数据规模:")
print(f"  - 总用户数: {len(llm_calls_df):,}")
print(f"  - 总交互数: {total_interactions:,}")
print(f"  - 平均交互/用户: {avg_interactions_per_user:.1f}")
print(f"  - 平均活跃天数/用户: {llm_calls_df['active_days'].mean():.0f}")
print(f"  - 中位数活跃天数/用户: {llm_calls_df['active_days'].median():.0f}")
print()
print("🎯 提取策略:")
print(f"  - 短期兴趣: 每 {SHORT_TERM_DAYS} 天活跃时间，统计提取（高分电影，≥4星）")
print(f"  - 短期兴趣数量: 前10个（出现次数>1）")
print(f"  - 长期兴趣: 每 {LONG_TERM_BUCKETS} 个短期桶 ({SHORT_TERM_DAYS * LONG_TERM_BUCKETS} 天)，LLM总结")
print(f"  - 长期兴趣数量: 5个（带年龄记录）")
print(f"  - 按用户实际活跃时间分桶，非全局日历时间")
print()
print("💰 成本估算:")
print(f"  - 总LLM调用: {total_llm_calls:,} 次")
print(f"  - 平均调用/用户: {avg_llm_calls:.1f} 次")
print(f"  - 中位数调用/用户: {median_llm_calls:.0f} 次")
print(f"  - 预估成本 (gpt-4o-mini): ${total_llm_calls * 0.001:.2f} - ${total_llm_calls * 0.002:.2f}")
print()
print("✅ 方案优势:")
print("  1. 成本可控 (~$60-120)")
print("  2. LLM用于高层次推理（长期兴趣演化）")
print("  3. 统计方法提取短期兴趣（快速、免费）")
print("  4. 理论依据：21天习惯养成，季度总结")
print("  5. 支持消融实验：长期vs短期、统计vs LLM")
print("  6. 符合论文叙事：LLM as information extractor")
print()
print("📝 最终输出 (用户KG):")
print("  - 5个长期兴趣 (relation: long_term_interest)")
print("  - 10个短期兴趣 (relation: short_term_interest)")
print("  - 长期兴趣带年龄标记（连续保留的天数）")
print()
print("📊 分布特征:")
print(f"  - 短期桶数: 平均{llm_calls_df['num_short_buckets'].mean():.1f}, 中位数{llm_calls_df['num_short_buckets'].median():.0f}")
print(f"  - LLM调用: 平均{avg_llm_calls:.1f}, 中位数{median_llm_calls:.0f}")
print("="*70)
