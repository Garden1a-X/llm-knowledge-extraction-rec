#!/usr/bin/env python3
"""
合并item.kg和user.kg为统一的.kg文件供RecBole使用

Usage:
    python baselines/merge_kg_files.py \
        --item_kg data/recbole/ml-1m/ml-1m.item.kg \
        --user_kg data/recbole/ml-1m/ml-1m.user.kg \
        --output data/recbole/ml-1m/ml-1m.kg
"""

import argparse
from pathlib import Path


def merge_kg_files(item_kg_path: str, user_kg_path: str, output_path: str):
    """
    合并item KG和user KG为统一的KG文件

    Args:
        item_kg_path: item.kg文件路径
        user_kg_path: user.kg文件路径
        output_path: 输出的.kg文件路径
    """
    print(f"Merging KG files:")
    print(f"  Item KG: {item_kg_path}")
    print(f"  User KG: {user_kg_path}")
    print(f"  Output:  {output_path}")

    # 读取item.kg
    with open(item_kg_path, 'r', encoding='utf-8') as f:
        item_lines = f.readlines()

    print(f"\nItem KG: {len(item_lines)} triplets")

    # 读取user.kg
    with open(user_kg_path, 'r', encoding='utf-8') as f:
        user_lines = f.readlines()

    print(f"User KG: {len(user_lines)} triplets")

    # 合并（去掉重复的header如果有的话）
    header = item_lines[0] if item_lines[0].startswith('head_id') else None

    all_triplets = []

    # 添加header
    if header:
        all_triplets.append(header)

    # 添加item triplets（跳过header）
    start_idx = 1 if header else 0
    all_triplets.extend(item_lines[start_idx:])

    # 添加user triplets（跳过header）
    user_start_idx = 1 if (user_lines and user_lines[0].startswith('head_id')) else 0
    all_triplets.extend(user_lines[user_start_idx:])

    # 写入输出文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(all_triplets)

    total_triplets = len(all_triplets) - (1 if header else 0)
    print(f"\nMerged KG: {total_triplets} triplets")
    print(f"Saved to: {output_path}")
    print("✓ Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge item.kg and user.kg for RecBole')
    parser.add_argument('--item_kg', type=str, required=True,
                        help='Path to item.kg file')
    parser.add_argument('--user_kg', type=str, required=True,
                        help='Path to user.kg file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for merged .kg file')

    args = parser.parse_args()

    merge_kg_files(args.item_kg, args.user_kg, args.output)
