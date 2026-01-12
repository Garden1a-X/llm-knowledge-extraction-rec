#!/usr/bin/env python3
"""
检查ML-1M数据集的时间戳分布问题
"""

from datetime import datetime
from collections import Counter
import random

# 读取数据
print("加载数据...")
timestamps = []
user_data = {}

with open('../data/recbole/ml-1m/ml-1m.inter', 'r') as f:
    header = f.readline()
    for line in f:
        parts = line.strip().split('\t')
        user_id = int(parts[0])
        item_id = int(parts[1])
        rating = float(parts[2])
        ts = int(parts[3])

        if user_id not in user_data:
            user_data[user_id] = []
        user_data[user_id].append({
            'item_id': item_id,
            'rating': rating,
            'timestamp': ts,
            'date': datetime.fromtimestamp(ts).strftime('%Y-%m-%d'),
            'datetime': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        })

print(f"✓ 加载完成: {len(user_data)} 用户")
print()

# 分析几个代表性用户
print("="*70)
print("抽样检查用户交互模式")
print("="*70)
print()

# 随机选几个用户
random.seed(42)
sample_users = random.sample(list(user_data.keys()), 5)

for user_id in sample_users:
    interactions = user_data[user_id]
    interactions.sort(key=lambda x: x['timestamp'])

    # 统计不同日期数
    unique_dates = set(x['date'] for x in interactions)

    print(f"用户 {user_id}:")
    print(f"  总交互数: {len(interactions)}")
    print(f"  不同日期数: {len(unique_dates)}")
    print(f"  时间跨度: {interactions[0]['date']} → {interactions[-1]['date']}")

    if len(unique_dates) <= 5:
        print(f"  具体日期分布:")
        date_counts = Counter(x['date'] for x in interactions)
        for date, count in sorted(date_counts.items()):
            print(f"    {date}: {count} 条评分")

    # 显示前3条和后3条
    print(f"  前3条交互:")
    for i in range(min(3, len(interactions))):
        print(f"    {interactions[i]['datetime']}: 电影{interactions[i]['item_id']}, 评分{interactions[i]['rating']}")

    if len(interactions) > 6:
        print(f"  ...")
        print(f"  后3条交互:")
        for i in range(max(0, len(interactions)-3), len(interactions)):
            print(f"    {interactions[i]['datetime']}: 电影{interactions[i]['item_id']}, 评分{interactions[i]['rating']}")

    print()

# 全局统计
print("="*70)
print("全局时间分布统计")
print("="*70)

users_by_active_days = {}
for user_id, interactions in user_data.items():
    unique_dates = set(x['date'] for x in interactions)
    num_dates = len(unique_dates)

    if num_dates not in users_by_active_days:
        users_by_active_days[num_dates] = 0
    users_by_active_days[num_dates] += 1

print("\n活跃天数分布（前20个）:")
for days in sorted(users_by_active_days.keys())[:20]:
    count = users_by_active_days[days]
    pct = count / len(user_data) * 100
    bar = '█' * int(pct / 2)
    print(f"  {days:3d} 天: {count:4d} 用户 ({pct:5.1f}%) {bar}")

if len(users_by_active_days) > 20:
    remaining = sum(users_by_active_days[d] for d in sorted(users_by_active_days.keys())[20:])
    print(f"  ... (还有 {len(users_by_active_days)-20} 个不同值, 共{remaining}用户)")

# 统计单日评分数量
single_day_users = sum(1 for uid, ints in user_data.items() if len(set(x['date'] for x in ints)) == 1)
print(f"\n只在1天内完成所有评分的用户: {single_day_users} / {len(user_data)} ({single_day_users/len(user_data)*100:.1f}%)")

# 分析单日评分的用户特征
print("\n单日评分用户的交互数分布:")
single_day_interactions = [len(user_data[uid]) for uid in user_data if len(set(x['date'] for x in user_data[uid])) == 1]
print(f"  平均评分数: {sum(single_day_interactions)/len(single_day_interactions):.1f}")
print(f"  中位数: {sorted(single_day_interactions)[len(single_day_interactions)//2]}")
print(f"  最小: {min(single_day_interactions)}")
print(f"  最大: {max(single_day_interactions)}")

print("\n="*70)
print("结论")
print("="*70)
print("""
这是ML-1M数据集的已知特性：
- 时间戳记录的是"评分提交时间"，不是"观影时间"
- 很多用户在注册MovieLens时，批量导入了历史观影记录
- 导致大量评分集中在同一天（注册日）

这对我们的影响：
- 按"活跃天数"分桶不适用于这个数据集
- 需要改用其他方法：
  1. 按交互序号分桶（每N次交互）
  2. 按时间戳均匀划分（不管是否同一天）
  3. 完全放弃时序，只提取整体兴趣
""")
