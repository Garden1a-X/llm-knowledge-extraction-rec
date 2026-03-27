#!/usr/bin/env python3
"""
Create filtered KG files for ablation experiments.

This script creates:
1. ml-1m.user.kg.short_term_only - User KG with only short-term interests
2. ml-1m.user.kg.long_term_only - User KG with only long-term interests
3. ml-1m.user.kg.empty - Empty user KG (for item-only ablation)
4. ml-1m.item.kg.empty - Empty item KG (for user-only ablation)

Usage:
    python scripts/create_ablation_kg_files.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_filtered_user_kg(input_path, output_path, keep_relations):
    """
    Create filtered user KG by keeping only specified relations.

    Args:
        input_path: Path to original user KG file
        output_path: Path to output filtered KG file
        keep_relations: Set of relation names to keep
    """
    kept_count = 0
    total_count = 0

    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        # Write header
        header = f_in.readline()
        f_out.write(header)

        for line in f_in:
            total_count += 1
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue

            head, relation, tail = parts

            if relation in keep_relations:
                f_out.write(line)
                kept_count += 1

    print(f"Created {output_path}")
    print(f"  Kept {kept_count}/{total_count} triplets ({100*kept_count/total_count:.1f}%)")
    print(f"  Relations kept: {keep_relations}")


def create_empty_kg(output_path, kg_type='user'):
    """
    Create empty KG file (header only).

    Args:
        output_path: Path to output empty KG file
        kg_type: 'user' or 'item'
    """
    with open(output_path, 'w') as f:
        f.write("head_id:token\trelation_id:token\ttail_id:token\n")

    print(f"Created empty {kg_type} KG: {output_path}")


def main():
    data_dir = Path("data/recbole/ml-1m")

    # Original KG files
    user_kg_path = data_dir / "ml-1m.user.kg"
    item_kg_path = data_dir / "ml-1m.item.kg"

    print("=" * 60)
    print("Creating Ablation KG Files")
    print("=" * 60)
    print()

    # 1. User KG with only short-term interests (remove long-term)
    print("1. Creating User KG (short-term only)...")
    create_filtered_user_kg(
        user_kg_path,
        data_dir / "ml-1m.user.kg.short_term_only",
        keep_relations={'short_term_interest'}
    )
    print()

    # 2. User KG with only long-term interests (remove short-term)
    print("2. Creating User KG (long-term only)...")
    create_filtered_user_kg(
        user_kg_path,
        data_dir / "ml-1m.user.kg.long_term_only",
        keep_relations={'long_term_interest'}
    )
    print()

    # 3. Empty User KG (for item-only ablation)
    print("3. Creating empty User KG...")
    create_empty_kg(data_dir / "ml-1m.user.kg.empty", 'user')
    print()

    # 4. Empty Item KG (for user-only ablation)
    print("4. Creating empty Item KG...")
    create_empty_kg(data_dir / "ml-1m.item.kg.empty", 'item')
    print()

    print("=" * 60)
    print("All ablation KG files created successfully!")
    print("=" * 60)

    # Verify files
    print("\nVerification:")
    for f in data_dir.glob("*.kg*"):
        lines = sum(1 for _ in open(f)) - 1  # Exclude header
        print(f"  {f.name}: {lines} triplets")


if __name__ == '__main__':
    main()
