#!/usr/bin/env python3
"""
分析Stage 1 Filtering结果
提取关键统计信息，验证数据完整性
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

def analyze_stage1_results(json_path):
    """分析Stage 1 filtering结果"""

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 80)
    print("Stage 1 Filtering 结果分析")
    print("=" * 80)

    # 1. 元数据
    metadata = data.get('metadata', {})
    print(f"\n【元数据】")
    print(f"处理的relations数量: {metadata.get('total_relations', 0)}")
    print(f"Relations列表: {', '.join(metadata.get('relations_processed', []))}")

    # 2. 统计每个relation的结果
    results = data.get('results', {})

    print(f"\n{'='*80}")
    print(f"{'Relation':<25} {'原始':<8} {'保留':<8} {'移除':<8} {'保留率':<10}")
    print(f"{'='*80}")

    total_original = 0
    total_keep = 0
    total_remove = 0

    relation_stats = []

    for relation_name in sorted(results.keys()):
        rel_data = results[relation_name]

        original = rel_data.get('original_entity_count', 0)
        keep_list = rel_data.get('keep', [])
        remove_dict = rel_data.get('remove', {})

        keep_count = len(keep_list)
        remove_count = len(remove_dict)

        # 验证完整性
        if original != keep_count + remove_count:
            print(f"⚠️  {relation_name}: 数量不匹配! {original} != {keep_count} + {remove_count}")

        keep_rate = keep_count / original * 100 if original > 0 else 0

        print(f"{relation_name:<25} {original:<8} {keep_count:<8} {remove_count:<8} {keep_rate:>6.1f}%")

        total_original += original
        total_keep += keep_count
        total_remove += remove_count

        relation_stats.append({
            'relation': relation_name,
            'original': original,
            'keep': keep_count,
            'remove': remove_count,
            'keep_rate': keep_rate
        })

    print(f"{'='*80}")
    print(f"{'总计':<25} {total_original:<8} {total_keep:<8} {total_remove:<8} {total_keep/total_original*100:>6.1f}%")
    print(f"{'='*80}")

    # 3. 分析被移除entities的建议去向
    print(f"\n{'='*80}")
    print("被移除Entities的建议去向统计")
    print(f"{'='*80}")

    suggested_destinations = defaultdict(lambda: defaultdict(int))

    for relation_name, rel_data in results.items():
        remove_dict = rel_data.get('remove', {})

        for entity_name, entity_data in remove_dict.items():
            suggested = entity_data.get('suggested_relation')
            if suggested is None:
                suggested = 'null/unknown'
            suggested_destinations[relation_name][suggested] += 1

    for relation_name in sorted(suggested_destinations.keys()):
        destinations = suggested_destinations[relation_name]
        total_removed = sum(destinations.values())

        print(f"\n{relation_name} ({total_removed} entities被移除):")
        for dest, count in sorted(destinations.items(), key=lambda x: -x[1]):
            percentage = count / total_removed * 100
            print(f"  → {dest:<25} {count:>3} ({percentage:>5.1f}%)")

    # 4. 找出接收最多entities的relations
    print(f"\n{'='*80}")
    print("接收Entity最多的Relations (Top 10)")
    print(f"{'='*80}")

    receiving_count = Counter()
    for relation_name, destinations in suggested_destinations.items():
        for dest, count in destinations.items():
            receiving_count[dest] += count

    for dest, count in receiving_count.most_common(10):
        print(f"{dest:<25} 接收 {count:>3} entities")

    # 5. 识别潜在冲突（entity可能被多个relation建议）
    print(f"\n{'='*80}")
    print("数据完整性检查")
    print(f"{'='*80}")

    # 检查keep和remove是否有重叠
    conflicts = []
    for relation_name, rel_data in results.items():
        keep_set = set(rel_data.get('keep', []))
        remove_dict = rel_data.get('remove', {})
        remove_set = set(remove_dict.keys())

        overlap = keep_set & remove_set
        if overlap:
            conflicts.append((relation_name, overlap))
            print(f"⚠️  {relation_name}: keep和remove有重叠 - {overlap}")

    if not conflicts:
        print("✅ 所有relations的keep和remove列表无重叠")

    # 验证总数
    if total_original == total_keep + total_remove:
        print(f"✅ 总数验证通过: {total_original} = {total_keep} + {total_remove}")
    else:
        print(f"⚠️  总数不匹配: {total_original} != {total_keep} + {total_remove}")

    # 6. 保存摘要
    summary = {
        'total_relations': len(results),
        'total_entities_original': total_original,
        'total_entities_keep': total_keep,
        'total_entities_remove': total_remove,
        'overall_keep_rate': total_keep / total_original * 100,
        'relation_stats': relation_stats,
        'receiving_count': dict(receiving_count),
        'has_conflicts': len(conflicts) > 0
    }

    summary_path = Path(json_path).parent / 'stage1_filtering_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 摘要已保存到: {summary_path}")

    return summary

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = 'results/entity_redistribution_stage1_filtering.json'

    analyze_stage1_results(json_path)
