#!/usr/bin/env python3
"""
Calculate dataset statistics for paper.

Usage:
    python scripts/calc_dataset_stats.py

Expects data at: data/recbole/{dataset}/{dataset}.inter
"""

import pandas as pd
from pathlib import Path

# Dataset configurations
DATASETS = {
    'ml-1m': {
        'inter_path': 'data/recbole/ml-1m/ml-1m.inter',
        'user_col': 'user_id:token',
        'item_col': 'item_id:token',
        'note': '5-core filtered',
    },
    'amazon-beauty': {
        'inter_path': 'data/recbole/amazon-beauty/amazon-beauty.inter',
        'user_col': 'user_id:token',
        'item_col': 'item_id:token',
        'note': 'No 5-core filtering (all interactions)',
    },
    'amazon-videogames': {
        'inter_path': 'data/recbole/amazon-videogames/amazon-videogames.inter',
        'user_col': 'user_id:token',
        'item_col': 'item_id:token',
        'note': '5-core filtered',
    },
}


def calc_stats(name, config):
    path = Path(config['inter_path'])
    if not path.exists():
        print(f"\n[SKIP] {name}: {path} not found")
        return None

    df = pd.read_csv(path, sep='\t')
    n_users = df[config['user_col']].nunique()
    n_items = df[config['item_col']].nunique()
    n_inter = len(df)
    density = n_inter / (n_users * n_items) * 100
    avg_per_user = n_inter / n_users
    avg_per_item = n_inter / n_items

    return {
        'Dataset': name,
        '# Users': n_users,
        '# Items': n_items,
        '# Interactions': n_inter,
        'Density (%)': f"{density:.4f}",
        'Avg. Inter/User': f"{avg_per_user:.2f}",
        'Avg. Inter/Item': f"{avg_per_item:.2f}",
        'Note': config['note'],
    }


def main():
    print("=" * 80)
    print("Dataset Statistics")
    print("=" * 80)

    results = []
    for name, config in DATASETS.items():
        stats = calc_stats(name, config)
        if stats:
            results.append(stats)
            print(f"\n{name} ({config['note']})")
            print(f"  # Users:          {stats['# Users']:,}")
            print(f"  # Items:          {stats['# Items']:,}")
            print(f"  # Interactions:   {stats['# Interactions']:,}")
            print(f"  Density:          {stats['Density (%)']}%")
            print(f"  Avg. Inter/User:  {stats['Avg. Inter/User']}")
            print(f"  Avg. Inter/Item:  {stats['Avg. Inter/Item']}")

    # Print LaTeX-style table
    if results:
        print("\n" + "=" * 80)
        print("Markdown Table")
        print("=" * 80)
        print("| Dataset | # Users | # Items | # Interactions | Density (%) | Avg. Inter/User | Avg. Inter/Item |")
        print("|---------|---------|---------|----------------|-------------|-----------------|-----------------|")
        for r in results:
            print(f"| {r['Dataset']} | {r['# Users']:,} | {r['# Items']:,} | {r['# Interactions']:,} | {r['Density (%)']} | {r['Avg. Inter/User']} | {r['Avg. Inter/Item']} |")


if __name__ == '__main__':
    main()
