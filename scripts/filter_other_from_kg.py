#!/usr/bin/env python3
"""
Filter out 'other' entities from KG files.

Removes all triples where tail_id contains '_other' or 'other_'.
"""

import argparse
from pathlib import Path


def filter_kg_file(input_path: Path, output_path: Path):
    """
    Filter KG file to remove 'other' entities.

    Args:
        input_path: Input .kg file
        output_path: Output .kg file
    """
    print(f"Filtering {input_path} -> {output_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Keep header
    header = lines[0]
    triples = lines[1:]

    # Filter triples
    kept_triples = []
    removed_count = 0

    for line in triples:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            head, relation, tail = parts[0], parts[1], parts[2]

            # Remove if tail contains 'other'
            if '_other' in tail.lower() or 'other_' in tail.lower():
                removed_count += 1
                continue

            kept_triples.append(line)

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(kept_triples)

    print(f"  Original: {len(triples)} triples")
    print(f"  Kept:     {len(kept_triples)} triples")
    print(f"  Removed:  {removed_count} triples")
    print(f"  Reduction: {removed_count/len(triples)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Filter other entities from KG files')
    parser.add_argument('--item_kg', type=str, required=True, help='Input item.kg file')
    parser.add_argument('--user_kg', type=str, default=None, help='Input user.kg file (optional)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Filtering 'other' entities from KG files")
    print("="*70)
    print()

    # Filter item.kg
    item_kg_input = Path(args.item_kg)
    item_kg_output = output_dir / item_kg_input.name
    filter_kg_file(item_kg_input, item_kg_output)
    print()

    # Filter user.kg if provided
    if args.user_kg:
        user_kg_input = Path(args.user_kg)
        user_kg_output = output_dir / user_kg_input.name
        filter_kg_file(user_kg_input, user_kg_output)
        print()

    print("="*70)
    print("Filtering complete!")
    print(f"Output directory: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
