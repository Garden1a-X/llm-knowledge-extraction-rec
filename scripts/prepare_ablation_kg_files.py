#!/usr/bin/env python3
"""
Prepare KG files for ablation experiments.

This script creates the following ablation KG variants:
1. *.user.kg.empty - Empty user KG (for item_kg_only ablation)
2. *.item.kg.empty - Empty item KG (for user_kg_only ablation)
3. *.user.kg.short_term_only - Only short-term interests
4. *.user.kg.long_term_only - Only long-term interests

Usage:
    # Prepare all datasets
    python scripts/prepare_ablation_kg_files.py --dataset all

    # Prepare specific dataset
    python scripts/prepare_ablation_kg_files.py --dataset ml-1m
    python scripts/prepare_ablation_kg_files.py --dataset beauty
    python scripts/prepare_ablation_kg_files.py --dataset videogames
"""

import argparse
from pathlib import Path
import sys

# Dataset configurations
DATASETS = {
    'ml-1m': {
        'dir': 'data/recbole/ml-1m',
        'prefix': 'ml-1m',
    },
    'beauty': {
        'dir': 'data/recbole/amazon-beauty',
        'prefix': 'amazon-beauty',
    },
    'videogames': {
        'dir': 'data/recbole/amazon-videogames',
        'prefix': 'amazon-videogames',
    },
}

# Relation type mappings
SHORT_TERM_RELATIONS = ['short_term_interest', 'recent_interest', 'current_interest']
LONG_TERM_RELATIONS = ['long_term_interest', 'preference', 'historical_interest']


def create_empty_kg(output_path, kg_type='user'):
    """Create an empty KG file with only header."""
    if kg_type == 'user':
        header = "head_id:token\trelation_id:token\ttail_id:token\n"
    else:  # item
        header = "head_id:token\trelation_id:token\ttail_id:token\n"

    with open(output_path, 'w') as f:
        f.write(header)

    print(f"  ✅ Created: {output_path} (empty)")


def filter_user_kg_by_relation(input_path, output_path, relation_types, exclude=False):
    """
    Filter user KG by relation types.

    Args:
        input_path: Path to the original user KG
        output_path: Path for the filtered output
        relation_types: List of relation types to include (or exclude if exclude=True)
        exclude: If True, exclude these relation types instead of including
    """
    if not Path(input_path).exists():
        print(f"  ⚠️ Warning: {input_path} not found, skipping...")
        return False

    with open(input_path, 'r') as f:
        lines = f.readlines()

    if len(lines) <= 1:
        print(f"  ⚠️ Warning: {input_path} is empty or only has header")
        return False

    header = lines[0]
    data_lines = lines[1:]

    filtered_lines = [header]

    for line in data_lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            relation = parts[1].lower()

            # Check if relation matches any of the target types
            matches = any(rt in relation for rt in relation_types)

            if exclude:
                # Keep lines that DON'T match
                if not matches:
                    filtered_lines.append(line)
            else:
                # Keep lines that DO match
                if matches:
                    filtered_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(filtered_lines)

    print(f"  ✅ Created: {output_path} ({len(filtered_lines)-1} edges)")
    return True


def prepare_dataset(dataset_name, config):
    """Prepare ablation KG files for a dataset."""
    print(f"\n{'='*60}")
    print(f"Preparing ablation files for: {dataset_name}")
    print(f"{'='*60}")

    data_dir = Path(config['dir'])
    prefix = config['prefix']

    if not data_dir.exists():
        print(f"  ⚠️ Warning: Directory {data_dir} does not exist!")
        print(f"  Creating directory...")
        data_dir.mkdir(parents=True, exist_ok=True)

    # Check for required base files
    user_kg_path = data_dir / f"{prefix}.user.kg"
    item_kg_path = data_dir / f"{prefix}.item.kg"

    print(f"\n  Checking base files:")
    print(f"    User KG: {user_kg_path} {'✅ exists' if user_kg_path.exists() else '❌ missing'}")
    print(f"    Item KG: {item_kg_path} {'✅ exists' if item_kg_path.exists() else '❌ missing'}")

    # 1. Create empty user KG
    print(f"\n  Creating empty KG files...")
    empty_user_kg = data_dir / f"{prefix}.user.kg.empty"
    create_empty_kg(empty_user_kg, 'user')

    # 2. Create empty item KG
    empty_item_kg = data_dir / f"{prefix}.item.kg.empty"
    create_empty_kg(empty_item_kg, 'item')

    # 3. Create short-term only user KG
    print(f"\n  Creating filtered user KG files...")
    if user_kg_path.exists():
        short_term_path = data_dir / f"{prefix}.user.kg.short_term_only"
        filter_user_kg_by_relation(
            user_kg_path,
            short_term_path,
            ['short_term'],  # Include relations containing 'short_term'
            exclude=False
        )

        # 4. Create long-term only user KG
        long_term_path = data_dir / f"{prefix}.user.kg.long_term_only"
        filter_user_kg_by_relation(
            user_kg_path,
            long_term_path,
            ['long_term'],  # Include relations containing 'long_term'
            exclude=False
        )
    else:
        print(f"  ⚠️ Skipping filtered KGs: {user_kg_path} not found")
        print(f"  Please generate user KG first using the user interest extraction pipeline.")

    # Summary
    print(f"\n  Summary for {dataset_name}:")
    for fname in [f"{prefix}.user.kg.empty", f"{prefix}.item.kg.empty",
                  f"{prefix}.user.kg.short_term_only", f"{prefix}.user.kg.long_term_only"]:
        fpath = data_dir / fname
        if fpath.exists():
            with open(fpath, 'r') as f:
                line_count = sum(1 for _ in f) - 1  # Exclude header
            print(f"    {fname}: {line_count} edges")
        else:
            print(f"    {fname}: ❌ not created")


def show_user_kg_relations(dataset_name, config):
    """Show unique relations in user KG to help with filtering."""
    data_dir = Path(config['dir'])
    prefix = config['prefix']
    user_kg_path = data_dir / f"{prefix}.user.kg"

    if not user_kg_path.exists():
        print(f"  User KG not found: {user_kg_path}")
        return

    print(f"\n  Unique relations in {user_kg_path}:")

    relations = {}
    with open(user_kg_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                rel = parts[1]
                relations[rel] = relations.get(rel, 0) + 1

    for rel, count in sorted(relations.items(), key=lambda x: -x[1]):
        print(f"    {rel}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Prepare KG files for ablation experiments')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'ml-1m', 'beauty', 'videogames'],
                        help='Dataset to prepare')
    parser.add_argument('--show-relations', action='store_true',
                        help='Show unique relations in user KG (for debugging)')

    args = parser.parse_args()

    # Determine which datasets to process
    if args.dataset == 'all':
        datasets = list(DATASETS.keys())
    else:
        datasets = [args.dataset]

    print("=" * 60)
    print("Preparing Ablation KG Files")
    print("=" * 60)
    print(f"Datasets: {datasets}")

    for ds in datasets:
        config = DATASETS[ds]

        if args.show_relations:
            show_user_kg_relations(ds, config)
        else:
            prepare_dataset(ds, config)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run ablation experiments:")
    print("     python scripts/run_beauty_ablation_1trial.py --method all")
    print("     python scripts/run_videogames_ablation_1trial.py --method all")
    print("\n  2. If user KG is missing, generate it first:")
    print("     python scripts/extract_user_interests.py --dataset beauty")
    print("     python scripts/extract_user_interests.py --dataset videogames")


if __name__ == '__main__':
    main()
