#!/usr/bin/env python3
"""
Generate standard Knowledge Graph for Amazon Beauty dataset.
Based on item metadata (categories + price), for KGAT baseline training.

Input: amazon-beauty.item file
Output:
  - amazon-beauty.kg (KG triples)
  - amazon-beauty.link (item-entity mapping)
  - amazon-beauty.inter (copy of interactions)

Usage:
    python scripts/generate_beauty_standard_kg.py \
        --item_file data/recbole/amazon-beauty/amazon-beauty.item \
        --inter_file data/recbole/amazon-beauty/amazon-beauty.inter \
        --output_dir /data/xuao/KG4RecEval/dataset/amazon-beauty
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Set


def parse_item_file(item_file: Path) -> List[Tuple[str, str, List[str], float]]:
    """
    Parse amazon-beauty.item file.

    Format: item_id:token	title:token_seq	categories:token_seq	price:float

    Returns:
        List of (item_id, title, categories_list, price)
    """
    items = []

    with open(item_file, 'r', encoding='utf-8') as f:
        # Skip header
        header = f.readline()

        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 4:
                continue

            item_id = parts[0]
            title = parts[1]
            categories_str = parts[2]
            price_str = parts[3]

            # Parse categories (pipe-separated)
            categories = []
            if categories_str and categories_str != 'Unknown':
                categories = categories_str.split('|')

            # Parse price
            try:
                price = float(price_str)
            except ValueError:
                price = 0.0

            items.append((item_id, title, categories, price))

    return items


def get_price_range(price: float) -> str:
    """
    Map price to price range category.
    """
    if price == 0.0:
        return "free"
    elif price < 10.0:
        return "under_10"
    elif price < 20.0:
        return "10_to_20"
    elif price < 30.0:
        return "20_to_30"
    elif price < 50.0:
        return "30_to_50"
    else:
        return "over_50"


def clean_entity_name(text: str) -> str:
    """
    Clean entity name to be CSV/TSV safe.
    """
    if not text:
        return "unknown"

    cleaned = (
        text.lower()
        .replace('"', '')
        .replace("'", '')
        .replace('\t', '_')
        .replace('\n', '_')
        .replace('\r', '')
        .replace('\\', '_')
        .replace(' ', '_')
        .replace('&', 'and')
        .replace(',', '')
        .replace(';', '')
        .replace(':', '')
        .replace('(', '')
        .replace(')', '')
        .replace('[', '')
        .replace(']', '')
        .replace('{', '')
        .replace('}', '')
        .replace('/', '_')
        .replace('|', '_')
        .replace('<', '')
        .replace('>', '')
        .replace('?', '')
        .replace('!', '')
        .replace('*', '')
        .replace('#', '')
        .replace('@', 'at')
        .replace('$', '')
        .replace('%', 'percent')
        .replace('^', '')
        .replace('~', '')
        .replace('`', '')
        .replace('+', 'plus')
        .replace('=', 'equals')
    )

    # Remove consecutive underscores
    while '__' in cleaned:
        cleaned = cleaned.replace('__', '_')

    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')

    if not cleaned:
        cleaned = 'unknown'

    return cleaned


def generate_kg_triples(items: List[Tuple[str, str, List[str], float]]) -> List[Tuple[str, str, str]]:
    """
    Generate KG triples from item metadata.

    Relations:
    - has_category: item -> category
    - has_price_range: item -> price_range
    """
    triples = []

    for item_id, title, categories, price in items:
        # Add category relations
        for category in categories:
            category_cleaned = clean_entity_name(category)
            if category_cleaned and category_cleaned != 'unknown':
                triples.append((item_id, "has_category", category_cleaned))

        # Add price range relation
        price_range = get_price_range(price)
        triples.append((item_id, "has_price_range", price_range))

    return triples


def generate_link_file(items: List[Tuple[str, str, List[str], float]]) -> List[Tuple[str, str]]:
    """
    Generate item-entity link mapping (1-to-1).
    """
    return [(item_id, item_id) for item_id, _, _, _ in items]


def save_kg(triples: List[Tuple[str, str, str]], output_file: Path):
    """Save KG triples to file in RecBole format."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("head_id:token\trelation_id:token\ttail_id:token\n")
        for head, relation, tail in triples:
            f.write(f"{head}\t{relation}\t{tail}\n")

    print(f"  ✓ Saved {len(triples)} triples to {output_file}")


def save_link(links: List[Tuple[str, str]], output_file: Path):
    """Save item-entity link file in RecBole format."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("item_id:token\tentity_id:token\n")
        for item_id, entity_id in links:
            f.write(f"{item_id}\t{entity_id}\n")

    print(f"  ✓ Saved {len(links)} links to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate standard KG for Amazon Beauty (for KGAT baseline)'
    )
    parser.add_argument('--item_file', type=str,
                       default='data/recbole/amazon-beauty/amazon-beauty.item',
                       help='Path to amazon-beauty.item file')
    parser.add_argument('--inter_file', type=str,
                       default='data/recbole/amazon-beauty/amazon-beauty.inter',
                       help='Path to amazon-beauty.inter file')
    parser.add_argument('--output_dir', type=str,
                       default='/data/xuao/KG4RecEval/dataset/amazon-beauty',
                       help='Output directory for KGAT baseline')

    args = parser.parse_args()

    item_file = Path(args.item_file)
    inter_file = Path(args.inter_file)
    output_dir = Path(args.output_dir)

    print("="*70)
    print("Generating Standard KG for Amazon Beauty (KGAT Baseline)")
    print("="*70)
    print(f"\nInput:")
    print(f"  Item file:  {item_file}")
    print(f"  Inter file: {inter_file}")
    print(f"Output: {output_dir}")
    print()

    # Check if files exist
    if not item_file.exists():
        print(f"Error: Item file not found: {item_file}")
        return

    if not inter_file.exists():
        print(f"Error: Inter file not found: {inter_file}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse item file
    print("Parsing item file...")
    items = parse_item_file(item_file)
    print(f"  ✓ Parsed {len(items)} items")

    # Generate KG triples
    print("\nGenerating KG triples...")
    triples = generate_kg_triples(items)
    print(f"  ✓ Generated {len(triples)} triples")

    # Statistics
    num_category = sum(1 for _, r, _ in triples if r == "has_category")
    num_price = sum(1 for _, r, _ in triples if r == "has_price_range")

    print(f"\nTriple Statistics:")
    print(f"  Category triples: {num_category}")
    print(f"  Price range triples: {num_price}")
    print(f"  Total: {len(triples)}")

    # Unique entities
    unique_categories = set(tail for _, r, tail in triples if r == "has_category")
    unique_prices = set(tail for _, r, tail in triples if r == "has_price_range")

    print(f"\nUnique Entities:")
    print(f"  Categories: {len(unique_categories)}")
    print(f"  Price ranges: {len(unique_prices)}")
    print(f"  Total: {len(unique_categories) + len(unique_prices)}")

    if unique_categories:
        print(f"  Example categories: {sorted(list(unique_categories))[:5]}")
    print(f"  Price ranges: {sorted(unique_prices)}")

    # Generate link file
    print("\nGenerating item-entity links...")
    links = generate_link_file(items)
    print(f"  ✓ Generated {len(links)} links")

    # Save files
    print(f"\nSaving files to {output_dir}...")
    save_kg(triples, output_dir / "amazon-beauty.kg")
    save_link(links, output_dir / "amazon-beauty.link")

    # Copy inter file
    print(f"\nCopying inter file...")
    shutil.copy(inter_file, output_dir / "amazon-beauty.inter")
    print(f"  ✓ Copied {inter_file} to {output_dir / 'amazon-beauty.inter'}")

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"  - amazon-beauty.kg ({len(triples)} triples)")
    print(f"  - amazon-beauty.link ({len(links)} links)")
    print(f"  - amazon-beauty.inter (copied)")
    print()
    print("Next: Run KGAT with:")
    print(f"  python scripts/run_beauty_kgat_5trials.py \\")
    print(f"      --data_path /data/xuao/KG4RecEval/dataset")
    print("="*70)


if __name__ == '__main__':
    main()
