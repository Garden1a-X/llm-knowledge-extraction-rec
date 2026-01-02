"""
展示轻量级relations的所有entities及其频数，用于手动聚类
"""

import json
from collections import defaultdict

# 轻量处理的6个relation
LIGHT_RELATIONS = [
    'color_palette',
    'interaction',
    'graphic_element',
    'genre',
    'symbolism',
    'others_relation'
]

def main():
    # 1. 加载Phase 1数据
    with open('../results/phase1_5percent_exploration.json', 'r') as f:
        phase1_data = json.load(f)

    # 2. 加载relation mapping
    with open('../results/relation_mapping_final_v1.json', 'r') as f:
        relation_mapping_data = json.load(f)

    relation_mapping = relation_mapping_data['relation_mapping']

    # 3. 提取所有knowledge points，按标准relation分组
    relation_entities = defaultdict(lambda: defaultdict(int))  # {standard_relation: {entity: count}}

    for movie_result in phase1_data['results']:
        if movie_result['status'] != 'success':
            continue

        for kp in movie_result['knowledge_points']:
            original_relation = kp['relation']
            entity = kp['entity']

            # 映射到标准relation
            standard_relation = relation_mapping.get(original_relation, 'others_relation')

            # 只统计轻量级relations
            if standard_relation in LIGHT_RELATIONS:
                relation_entities[standard_relation][entity] += 1

    # 4. 按relation展示
    print("=" * 80)
    print("轻量级Relations的Entity分布（用于手动聚类）")
    print("=" * 80)
    print()

    for relation in LIGHT_RELATIONS:
        entities = relation_entities[relation]

        # 按频数降序排序
        sorted_entities = sorted(entities.items(), key=lambda x: (-x[1], x[0]))

        print(f"\n{'='*80}")
        print(f"📋 {relation.upper()}")
        print(f"{'='*80}")
        print(f"总实体数: {len(entities)}")
        print(f"总出现次数: {sum(entities.values())}")
        print(f"\n{'Entity':<50} {'频数':>10}")
        print("-" * 80)

        for entity, count in sorted_entities:
            print(f"{entity:<50} {count:>10}")

        print()

if __name__ == '__main__':
    main()
