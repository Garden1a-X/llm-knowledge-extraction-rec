#!/usr/bin/env python3
"""
Calculate Avg. Items/Entity for Metadata KG and Visual KG on ML-1M.

Usage:
    python scripts/calc_kg_items_per_entity.py

Paths (modify if needed):
    Metadata KG: /data/xuao/KG4RecEval/dataset/ml-1m/ml-1m.kg
    Visual KG:   /data/xuao/llm-knowledge-extraction-rec/data/recbole/ml-1m/ml-1m.item.kg
"""

import pandas as pd
from collections import Counter

METADATA_KG_PATH = '/data/xuao/KG4RecEval/dataset/ml-1m/ml-1m.kg'
VISUAL_KG_PATH = '/data/xuao/llm-knowledge-extraction-rec/data/recbole/ml-1m/ml-1m.item.kg'


def calc_items_per_entity(path, name):
    df = pd.read_csv(path, sep='\t', names=['head_id', 'relation_id', 'tail_id'], skiprows=1)
    df = df.dropna()

    # head_id = item, tail_id = entity
    # 统计每个 entity 连接到多少个不同的 item
    entity_items = df.groupby('tail_id')['head_id'].nunique()

    n_entities = len(entity_items)
    n_triples = len(df)
    n_items = df['head_id'].nunique()
    avg = entity_items.mean()

    print(f"\n{name}")
    print(f"  # Triples:          {n_triples:,}")
    print(f"  # Items:            {n_items:,}")
    print(f"  # Entities:         {n_entities}")
    print(f"  # Relations:        {df['relation_id'].nunique()}")
    print(f"  Avg. Items/Entity:  {avg:.2f}")
    print(f"  Min Items/Entity:   {entity_items.min()}")
    print(f"  Max Items/Entity:   {entity_items.max()}")
    print(f"  Median:             {entity_items.median():.1f}")

    return avg, n_entities


def main():
    print("=" * 60)
    print("Avg. Items/Entity Comparison (ML-1M)")
    print("=" * 60)

    avg_meta, n_meta = calc_items_per_entity(METADATA_KG_PATH, "Metadata KG")
    avg_visual, n_visual = calc_items_per_entity(VISUAL_KG_PATH, "Visual KG")

    increase = (avg_visual - avg_meta) / avg_meta * 100

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Metadata KG - Avg. Items/Entity: {avg_meta:.2f}  ({n_meta} entities)")
    print(f"  Visual KG   - Avg. Items/Entity: {avg_visual:.2f}  ({n_visual} entities)")
    print(f"  Increase: {'+' if increase > 0 else ''}{increase:.1f}%")


if __name__ == '__main__':
    main()
