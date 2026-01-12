#!/usr/bin/env python3
"""
为RecBole KGAT创建item-entity链接文件

KGAT需要.link文件来定义item_id到entity_id的映射。
对于我们的知识图谱，从ml-1m.item.kg中提取item作为head的三元组，
将每个item映射到对应的entity。

Usage:
    python baselines/create_link_file.py \
        --item_kg data/recbole/ml-1m/ml-1m.item.kg \
        --output data/recbole/ml-1m/ml-1m.link
"""

import argparse
from pathlib import Path
from collections import defaultdict


def create_link_file(item_kg_path: str, output_path: str):
    """
    从item.kg创建.link文件

    Args:
        item_kg_path: item.kg文件路径
        output_path: 输出的.link文件路径
    """
    print(f"Creating link file from: {item_kg_path}")

    # 读取item.kg，提取所有item和对应的entity
    item_to_entities = defaultdict(set)

    with open(item_kg_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 跳过header
    start_idx = 1 if lines[0].startswith('head_id') else 0

    for line in lines[start_idx:]:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            item_id = parts[0]  # head_id (item)
            relation = parts[1]  # relation_id
            entity_id = parts[2]  # tail_id (entity)

            # 将item映射到entity
            item_to_entities[item_id].add(entity_id)

    print(f"Found {len(item_to_entities)} items with entities")

    # 写入.link文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入header
        f.write('item_id:token\tentity_id:token\n')

        # 写入item-entity映射
        for item_id in sorted(item_to_entities.keys()):
            entities = item_to_entities[item_id]
            # 如果一个item对应多个entity，每个都写一行
            for entity_id in sorted(entities):
                f.write(f'{item_id}\t{entity_id}\n')

    # 统计
    total_links = sum(len(entities) for entities in item_to_entities.values())
    print(f"\nCreated link file:")
    print(f"  Items: {len(item_to_entities)}")
    print(f"  Total links: {total_links}")
    print(f"  Avg entities per item: {total_links / len(item_to_entities):.2f}")
    print(f"  Saved to: {output_path}")
    print("✓ Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create item-entity link file for RecBole KGAT')
    parser.add_argument('--item_kg', type=str, required=True,
                        help='Path to item.kg file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for .link file')

    args = parser.parse_args()

    create_link_file(args.item_kg, args.output)
