#!/usr/bin/env python3
"""
Convert user interests to RecBole .kg format.

Usage:
    python scripts/convert_user_interests_to_kg.py \
        --input results/user_interests_hybrid.json \
        --output_user data/recbole/ml-1m/ml-1m.user.kg \
        --output_merged data/recbole/ml-1m/ml-1m.kg \
        --item_kg data/recbole/ml-1m/ml-1m.item.kg
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse


def convert_user_interests_to_kg(
    user_interests_file: Path,
    output_user_kg: Path,
    verbose: bool = True
):
    """
    Convert user interests JSON to RecBole .kg format.

    Output format (user KG):
        user_id  long_term_interest  entity
        user_id  short_term_interest  entity
    """
    with open(user_interests_file, 'r') as f:
        data = json.load(f)

    results = data['results']

    # Collect triplets
    triplets = []
    stats = {
        'total_users': len(results),
        'users_with_long_term': 0,
        'users_with_short_term': 0,
        'total_long_term': 0,
        'total_short_term': 0,
        'total_triplets': 0
    }

    for result in results:
        user_id = result['user_id']

        # Long-term interests
        long_term = result.get('long_term_interests', [])
        if long_term:
            stats['users_with_long_term'] += 1
            for interest in long_term:
                # Use original relation name from interest
                relation = interest['relation']
                entity = interest['entity']
                # Normalize entity (replace spaces with underscores)
                entity = entity.replace(' ', '_')
                triplets.append((user_id, 'long_term_interest', f"{relation}:{entity}"))
                stats['total_long_term'] += 1

        # Short-term interests
        short_term = result.get('short_term_interests', [])
        if short_term:
            stats['users_with_short_term'] += 1
            for interest in short_term:
                relation = interest['relation']
                entity = interest['entity']
                entity = entity.replace(' ', '_')
                triplets.append((user_id, 'short_term_interest', f"{relation}:{entity}"))
                stats['total_short_term'] += 1

    stats['total_triplets'] = len(triplets)

    if verbose:
        print(f"\nUser Interest Statistics:")
        print(f"  Total users: {stats['total_users']}")
        print(f"  Users with long-term: {stats['users_with_long_term']} ({stats['users_with_long_term']/stats['total_users']*100:.1f}%)")
        print(f"  Users with short-term: {stats['users_with_short_term']} ({stats['users_with_short_term']/stats['total_users']*100:.1f}%)")
        print(f"  Total long-term: {stats['total_long_term']}")
        print(f"  Total short-term: {stats['total_short_term']}")
        print(f"  Total triplets: {stats['total_triplets']}")
        print()

    # Write to file
    output_user_kg.parent.mkdir(parents=True, exist_ok=True)

    with open(output_user_kg, 'w', encoding='utf-8') as f:
        # Header
        f.write("head_id:token\trelation_id:token\ttail_id:token\n")

        # Triplets (sorted by user_id)
        for head_id, relation_id, tail_id in sorted(triplets):
            f.write(f"{head_id}\t{relation_id}\t{tail_id}\n")

    if verbose:
        print(f"✓ Wrote user KG: {output_user_kg}")
        print(f"  {stats['total_triplets']} triplets")
        print()

    return stats


def merge_item_and_user_kg(
    item_kg_file: Path,
    user_kg_file: Path,
    output_merged: Path,
    verbose: bool = True
):
    """
    Merge item KG and user KG into a single file.
    """
    if verbose:
        print("Merging item KG and user KG...")

    # Read both files (skip headers)
    item_lines = []
    with open(item_kg_file, 'r') as f:
        header = f.readline()  # Skip header
        item_lines = f.readlines()

    user_lines = []
    with open(user_kg_file, 'r') as f:
        header = f.readline()  # Skip header
        user_lines = f.readlines()

    # Write merged file
    output_merged.parent.mkdir(parents=True, exist_ok=True)

    with open(output_merged, 'w', encoding='utf-8') as f:
        # Header
        f.write("head_id:token\trelation_id:token\ttail_id:token\n")

        # Item KG first
        for line in item_lines:
            f.write(line)

        # Then user KG
        for line in user_lines:
            f.write(line)

    if verbose:
        print(f"✓ Wrote merged KG: {output_merged}")
        print(f"  Item triplets: {len(item_lines):,}")
        print(f"  User triplets: {len(user_lines):,}")
        print(f"  Total triplets: {len(item_lines) + len(user_lines):,}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Convert user interests to RecBole KG format'
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Input user interests JSON')
    parser.add_argument('--output_user', type=str, required=True,
                        help='Output user KG file (.user.kg)')
    parser.add_argument('--output_merged', type=str, required=True,
                        help='Output merged KG file (.kg)')
    parser.add_argument('--item_kg', type=str, required=True,
                        help='Input item KG file')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("="*70)
        print("USER INTEREST TO RECBOLE KG CONVERSION")
        print("="*70)
        print()

    # Convert user interests
    stats = convert_user_interests_to_kg(
        user_interests_file=Path(args.input),
        output_user_kg=Path(args.output_user),
        verbose=verbose
    )

    # Merge with item KG
    merge_item_and_user_kg(
        item_kg_file=Path(args.item_kg),
        user_kg_file=Path(args.output_user),
        output_merged=Path(args.output_merged),
        verbose=verbose
    )

    if verbose:
        print("="*70)
        print("SUMMARY")
        print("="*70)
        print(f"✓ User KG: {args.output_user}")
        print(f"  - {stats['total_users']} users")
        print(f"  - {stats['total_triplets']} triplets")
        print(f"✓ Merged KG: {args.output_merged}")
        print(f"  - Item + User combined")
        print("="*70)


if __name__ == '__main__':
    main()
