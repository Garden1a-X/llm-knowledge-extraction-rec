#!/usr/bin/env python3
"""
从refined entity mapping中提取标准entity词汇表
用于Phase 3全量提取
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_vocabulary(mapping_file: str, output_file: str):
    """
    从entity mapping中提取标准vocabulary

    Args:
        mapping_file: refined entity mapping JSON文件
        output_file: 输出vocabulary JSON文件
    """
    print(f"📥 读取refined mapping: {mapping_file}")
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entity_mappings = data.get('entity_mappings', {})

    # 提取每个relation的标准entities
    vocabulary = {}
    statistics = {
        'total_relations': 0,
        'total_standard_entities': 0,
        'total_noise_entities': 0,
        'relations_in_target_range': 0,  # 10-15 entities
        'per_relation': {}
    }

    print("\n" + "="*80)
    print("提取标准Entity词汇表")
    print("="*80 + "\n")

    for relation, mapping in sorted(entity_mappings.items()):
        # 提取所有canonical entity names（去重）
        canonical_entities = set(mapping.values())

        # 分离标准entities和噪声entities
        standard_entities = sorted([
            e for e in canonical_entities
            if not e.startswith('other_')
        ])

        noise_entities = sorted([
            e for e in canonical_entities
            if e.startswith('other_')
        ])

        # 保存到vocabulary
        vocabulary[relation] = {
            'standard_entities': standard_entities,
            'noise_entities': noise_entities,
            'total': len(canonical_entities),
            'in_target_range': 10 <= len(standard_entities) <= 15
        }

        # 更新统计
        statistics['total_relations'] += 1
        statistics['total_standard_entities'] += len(standard_entities)
        statistics['total_noise_entities'] += len(noise_entities)
        if 10 <= len(standard_entities) <= 15:
            statistics['relations_in_target_range'] += 1

        statistics['per_relation'][relation] = {
            'standard_count': len(standard_entities),
            'noise_count': len(noise_entities),
            'total_count': len(canonical_entities),
            'in_target_range': 10 <= len(standard_entities) <= 15
        }

        # 打印
        status = "✅" if 10 <= len(standard_entities) <= 15 else "⚠️ "
        print(f"{status} {relation:25s}: {len(standard_entities):2d} standard + {len(noise_entities):2d} noise = {len(canonical_entities):2d} total")

    # 打印统计摘要
    print("\n" + "="*80)
    print("统计摘要")
    print("="*80)
    print(f"总Relations数: {statistics['total_relations']}")
    print(f"总标准Entities数: {statistics['total_standard_entities']}")
    print(f"总噪声Entities数: {statistics['total_noise_entities']}")
    print(f"总Entities数: {statistics['total_standard_entities'] + statistics['total_noise_entities']}")
    print(f"\n目标范围(10-15 entities)达标率: {statistics['relations_in_target_range']}/{statistics['total_relations']} " +
          f"({statistics['relations_in_target_range']/statistics['total_relations']*100:.1f}%)")

    # 保存结果
    output_data = {
        'vocabulary': vocabulary,
        'statistics': statistics,
        'metadata': {
            'source': mapping_file,
            'description': 'Standard entity vocabulary for Phase 3 full extraction',
            'target_range': '10-15 entities per relation',
            'noise_pattern': 'other_{relation} for outlier entities'
        }
    }

    print(f"\n💾 保存vocabulary到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n✅ 词汇表提取完成!")

    return vocabulary, statistics


if __name__ == '__main__':
    mapping_file = 'results/temp_refined_data.json'
    output_file = 'results/standard_entity_vocabulary.json'

    if not Path(mapping_file).exists():
        print(f"❌ 错误: 找不到mapping文件: {mapping_file}")
        sys.exit(1)

    extract_vocabulary(mapping_file, output_file)
