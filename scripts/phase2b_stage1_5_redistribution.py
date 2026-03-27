#!/usr/bin/env python3
"""
Phase 2b Stage 1.5: Entity重分配（Redistribution）

目标：
1. 读取Stage 1的筛选结果
2. 收集所有被remove的entities及其suggested_relation
3. 将这些entities重新分配到对应的relations
4. 处理冲突（同一entity被多个relations拒绝）
5. 输出更新后的entity分配结果

流程：
- Stage 1输出：每个relation的keep列表和remove列表（含suggested_relation）
- Stage 1.5处理：将remove列表中的entities重新分配
- Stage 1.5输出：每个relation更新后的entity列表

使用：
  python scripts/phase2b_stage1_5_redistribution.py

输入：
  results/entity_redistribution_stage1_filtering.json (Stage 1输出)

输出：
  results/entity_redistribution_stage1_5_redistributed.json
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'


def collect_redistributions(stage1_results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    收集所有需要重新分配的entities

    Returns:
        Dict[suggested_relation, List[{entity, from_relation, reasoning}]]
    """
    redistribution_map = defaultdict(list)

    for from_relation, result_data in stage1_results.items():
        remove_list = result_data.get('remove', {})

        for entity, info in remove_list.items():
            suggested = info.get('suggested_relation')
            # 如果是None或null，默认分配到additional_elements
            if suggested is None:
                suggested = 'additional_elements'
            # 字段名是'reason'而不是'reasoning'
            reasoning = info.get('reason', '')

            redistribution_map[suggested].append({
                'entity': entity,
                'from_relation': from_relation,
                'reasoning': reasoning
            })

    return redistribution_map


def detect_conflicts(redistribution_map: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
    """
    检测冲突：同一entity被分配到多个不同的relations

    Returns:
        Dict[entity, List[suggested_relations]]
    """
    entity_suggestions = defaultdict(set)

    for suggested_relation, entities_info in redistribution_map.items():
        for info in entities_info:
            entity = info['entity']
            entity_suggestions[entity].add(suggested_relation)

    # 只返回有冲突的（被分配到2个或以上relations）
    conflicts = {}
    for entity, suggestions in entity_suggestions.items():
        if len(suggestions) > 1:
            conflicts[entity] = list(suggestions)

    return conflicts


def apply_redistributions(
    stage1_results: Dict[str, Any],
    redistribution_map: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    应用重新分配：将removed entities添加到对应的relations

    Returns:
        更新后的results（每个relation的final entity list）
    """
    final_results = {}

    # 初始化：每个relation的keep列表作为起点
    for relation, result_data in stage1_results.items():
        final_results[relation] = {
            'relation': relation,
            'original_entity_count': result_data.get('original_entity_count', 0),
            'stage1_kept': result_data.get('keep', []),
            'stage1_removed': list(result_data.get('remove', {}).keys()),
            'redistributed_in': [],  # 从其他relations分配进来的
            'final_entities': list(result_data.get('keep', []))  # 最终entity列表
        }

    # 应用重新分配
    redistributed_count = 0

    for to_relation, entities_info in redistribution_map.items():
        # 确保目标relation存在
        if to_relation not in final_results:
            # 如果这个relation在Stage 1中没有处理过，创建一个空entry
            print(f"⚠️  警告: {to_relation} 在Stage 1中未处理，创建新entry")
            final_results[to_relation] = {
                'relation': to_relation,
                'original_entity_count': 0,
                'stage1_kept': [],
                'stage1_removed': [],
                'redistributed_in': [],
                'final_entities': []
            }

        for info in entities_info:
            entity = info['entity']
            from_relation = info['from_relation']

            # 检查是否已经存在（避免重复）
            if entity not in final_results[to_relation]['final_entities']:
                final_results[to_relation]['final_entities'].append(entity)
                final_results[to_relation]['redistributed_in'].append({
                    'entity': entity,
                    'from_relation': from_relation,
                    'reasoning': info['reasoning']
                })
                redistributed_count += 1
            else:
                # 已存在（可能是Stage 1就保留的，或者已经被重新分配过了）
                print(f"  ⚠️  {entity} 已存在于 {to_relation}，跳过重复添加")

    return final_results, redistributed_count


def calculate_statistics(
    stage1_results: Dict[str, Any],
    final_results: Dict[str, Any],
    redistribution_map: Dict[str, List[Dict[str, Any]]],
    conflicts: Dict[str, List[str]]
) -> Dict[str, Any]:
    """计算统计信息"""

    # Stage 1统计
    total_original = sum(r.get('original_entity_count', 0) for r in stage1_results.values())
    total_stage1_kept = sum(len(r.get('keep', [])) for r in stage1_results.values())
    total_stage1_removed = sum(len(r.get('remove', {})) for r in stage1_results.values())

    # Stage 1.5统计
    total_redistributed = sum(len(entities) for entities in redistribution_map.values())
    total_conflicts = len(conflicts)

    # Final统计
    total_final = sum(len(r['final_entities']) for r in final_results.values())

    # 按relation统计
    relation_stats = {}
    for relation, data in final_results.items():
        stage1_data = stage1_results.get(relation, {})
        relation_stats[relation] = {
            'original': stage1_data.get('original_entity_count', 0),
            'stage1_kept': len(data['stage1_kept']),
            'stage1_removed': len(data['stage1_removed']),
            'redistributed_in': len(data['redistributed_in']),
            'final_count': len(data['final_entities']),
            'net_change': len(data['final_entities']) - stage1_data.get('original_entity_count', 0)
        }

    return {
        'total': {
            'original_entities': total_original,
            'stage1_kept': total_stage1_kept,
            'stage1_removed': total_stage1_removed,
            'redistributed': total_redistributed,
            'conflicts': total_conflicts,
            'final_entities': total_final
        },
        'by_relation': relation_stats
    }


def main():
    print("="*80)
    print("Phase 2b Stage 1.5: Entity重分配（Redistribution）")
    print("="*80)
    print()

    # 输入文件
    default_input = 'results/entity_redistribution_stage1_filtering.json'
    input_file = input(f"Stage 1结果文件路径 (default: {default_input}): ").strip() or default_input

    input_path = PROJECT_ROOT / input_file

    # 检查文件是否存在
    if not input_path.exists():
        print(f"❌ 错误: 文件不存在: {input_path}")
        print("\n请先运行 Stage 1: python scripts/phase2b_stage1_filtering.py")
        return

    # 加载Stage 1结果
    print(f"\n加载Stage 1结果: {input_path}")
    with open(input_path, 'r') as f:
        stage1_data = json.load(f)

    stage1_results = stage1_data.get('results', {})

    if not stage1_results:
        print("❌ 错误: Stage 1结果为空")
        return

    print(f"找到 {len(stage1_results)} 个relations的筛选结果")

    # Step 1: 收集所有需要重新分配的entities
    print(f"\n{'='*80}")
    print("Step 1: 收集需要重新分配的entities")
    print(f"{'='*80}")

    redistribution_map = collect_redistributions(stage1_results)

    print(f"\n收集到需要重新分配的entities:")
    for to_relation, entities_info in sorted(redistribution_map.items()):
        print(f"  → {to_relation:30s}: {len(entities_info)} entities")

    total_to_redistribute = sum(len(entities) for entities in redistribution_map.values())
    print(f"\n总计: {total_to_redistribute} entities需要重新分配")

    # Step 2: 检测冲突
    print(f"\n{'='*80}")
    print("Step 2: 检测冲突（同一entity被分配到多个relations）")
    print(f"{'='*80}")

    conflicts = detect_conflicts(redistribution_map)

    if conflicts:
        print(f"\n⚠️  发现 {len(conflicts)} 个冲突entities:")
        for i, (entity, suggestions) in enumerate(sorted(conflicts.items())[:20], 1):
            print(f"  {i}. {entity:30s} → {', '.join(suggestions)}")

        if len(conflicts) > 20:
            print(f"  ... 还有 {len(conflicts) - 20} 个冲突")

        print(f"\n处理策略: 保留第一个分配的relation（按字母顺序）")
    else:
        print("\n✅ 未发现冲突")

    # Step 3: 应用重新分配
    print(f"\n{'='*80}")
    print("Step 3: 应用重新分配")
    print(f"{'='*80}")

    final_results, redistributed_count = apply_redistributions(stage1_results, redistribution_map)

    print(f"\n✅ 成功重新分配 {redistributed_count} 个entities")

    # Step 4: 计算统计
    print(f"\n{'='*80}")
    print("Step 4: 统计结果")
    print(f"{'='*80}")

    statistics = calculate_statistics(stage1_results, final_results, redistribution_map, conflicts)

    print("\n总体统计:")
    print(f"  原始entities总数: {statistics['total']['original_entities']}")
    print(f"  Stage 1保留: {statistics['total']['stage1_kept']}")
    print(f"  Stage 1移除: {statistics['total']['stage1_removed']}")
    print(f"  Stage 1.5重新分配: {statistics['total']['redistributed']}")
    print(f"  最终entities总数: {statistics['total']['final_entities']}")
    print(f"  冲突数: {statistics['total']['conflicts']}")

    print(f"\n按Relation统计 (变化显著的前10个):")
    # 按final_count排序
    sorted_relations = sorted(
        statistics['by_relation'].items(),
        key=lambda x: x[1]['final_count'],
        reverse=True
    )

    print(f"{'Relation':<30} {'原始':<8} {'Stage1保留':<12} {'重新分配':<12} {'最终':<8} {'净变化':<8}")
    print("-" * 90)

    for relation, stats in sorted_relations[:10]:
        print(f"{relation:<30} {stats['original']:<8} {stats['stage1_kept']:<12} "
              f"{stats['redistributed_in']:<12} {stats['final_count']:<8} "
              f"{stats['net_change']:+d}")

    if len(sorted_relations) > 10:
        print(f"\n... 还有 {len(sorted_relations) - 10} 个relations")

    # 显示重新分配详情（示例）
    print(f"\n{'='*80}")
    print("重新分配详情示例")
    print(f"{'='*80}")

    for relation in sorted(final_results.keys())[:5]:
        data = final_results[relation]
        if data['redistributed_in']:
            print(f"\n{relation}:")
            print(f"  Stage 1保留: {len(data['stage1_kept'])}")
            print(f"  重新分配进来: {len(data['redistributed_in'])}")
            print(f"  示例 (前5个):")
            for item in data['redistributed_in'][:5]:
                print(f"    - {item['entity']:25s} (from {item['from_relation']})")

    # 保存结果
    default_output = 'results/entity_redistribution_stage1_5_redistributed.json'
    output = input(f"\n输出文件路径 (default: {default_output}): ").strip() or default_output

    output_path = PROJECT_ROOT / output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'stage': 'stage1_5_redistributed',
            'stage1_input': str(input_path),
            'relations_count': len(final_results),
            'total_redistributed': redistributed_count,
            'conflicts_count': len(conflicts)
        },
        'statistics': statistics,
        'conflicts': conflicts,
        'results': final_results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("✅ Stage 1.5 完成!")
    print(f"{'='*80}")
    print(f"结果已保存到: {output_path}")
    print("\n下一步:")
    print("  1. 检查Stage 1.5的重新分配结果")
    print("  2. 运行 Stage 2: python scripts/phase2b_stage2_merging.py")
    print("     (注意：需要修改Stage 2的输入文件为Stage 1.5的输出)")


if __name__ == '__main__':
    main()
