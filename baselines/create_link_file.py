#!/usr/bin/env python3
"""
为RecBole KGAT创建item-entity链接文件

KGAT的标准设置中，items本身也是entities的一部分。
.link文件将item_id映射到对应的entity_id。

对于我们的场景，最简单的方式是让每个item映射到自己的ID：
item 1 -> entity 1, item 2 -> entity 2, etc.

Usage:
    python baselines/create_link_file.py \
        --inter data/recbole/ml-1m/ml-1m.inter \
        --output data/recbole/ml-1m/ml-1m.link
"""

import argparse
from pathlib import Path


def create_link_file(inter_path: str, output_path: str):
    """
    从.inter文件创建.link文件（item_id -> entity_id映射）

    Args:
        inter_path: .inter文件路径
        output_path: 输出的.link文件路径
    """
    print(f"Creating link file from: {inter_path}")

    # 读取.inter文件，提取所有唯一的item_id
    unique_items = set()

    with open(inter_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 跳过header
    start_idx = 1 if lines[0].startswith('user_id') else 0

    for line in lines[start_idx:]:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            item_id = parts[1]  # 第二列是item_id
            unique_items.add(item_id)

    print(f"Found {len(unique_items)} unique items")

    # 写入.link文件（每个item映射到自己的ID作为entity）
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入header (RecBole期望的列名)
        f.write('item_id\tentity_id\n')

        # 每个item映射到自己的ID
        for item_id in sorted(unique_items, key=lambda x: int(x) if x.isdigit() else x):
            f.write(f'{item_id}\t{item_id}\n')

    print(f"\nCreated link file:")
    print(f"  Items: {len(unique_items)}")
    print(f"  Mapping: item_id -> entity_id (1:1, item作为自身entity)")
    print(f"  Saved to: {output_path}")
    print("✓ Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create item-entity link file for RecBole KGAT')
    parser.add_argument('--inter', type=str, required=True,
                        help='Path to .inter file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for .link file')

    args = parser.parse_args()

    create_link_file(args.inter, args.output)
