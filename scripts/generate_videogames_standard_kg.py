#!/usr/bin/env python3
"""
Generate standard Knowledge Graph for Amazon Video Games dataset.
Similar to ML-1M KG format, for KGAT baseline training.

Input: amazon-videogames.item file
Output:
  - amazon-videogames.kg (KG triples)
  - amazon-videogames.link (item-entity mapping)
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Set


def parse_item_file(item_file: Path) -> List[Tuple[str, str, List[str], float]]:
    """
    Parse amazon-videogames.item file.

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
                print(f"Warning: Skipping malformed line: {line[:50]}...")
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

    Ranges:
    - free: $0
    - under_10: $0.01-$9.99
    - 10_to_20: $10-$19.99
    - 20_to_40: $20-$39.99
    - 40_to_60: $40-$59.99
    - over_60: $60+
    """
    if price == 0.0:
        return "free"
    elif price < 10.0:
        return "under_10"
    elif price < 20.0:
        return "10_to_20"
    elif price < 40.0:
        return "20_to_40"
    elif price < 60.0:
        return "40_to_60"
    else:
        return "over_60"


def generate_kg_triples(items: List[Tuple[str, str, List[str], float]]) -> List[Tuple[str, str, str]]:
    """
    Generate KG triples from item metadata.

    Relations:
    - has_category: item -> category
    - has_price_range: item -> price_range

    Returns:
        List of (head_id, relation_id, tail_id) triples
    """
    triples = []

    for item_id, title, categories, price in items:
        # Add category relations
        for category in categories:
            # Normalize category to lowercase and replace spaces/special chars
            category_normalized = (
                category.lower()
                .replace("'", "")
                .replace(" ", "_")
                .replace("&", "and")
                .replace(",", "")
                .replace("(", "")
                .replace(")", "")
            )
            triples.append((item_id, "has_category", category_normalized))

        # Add price range relation
        price_range = get_price_range(price)
        triples.append((item_id, "has_price_range", price_range))

    return triples


def generate_link_file(items: List[Tuple[str, str, List[str], float]]) -> List[Tuple[str, str]]:
    """
    Generate item-entity link mapping.

    For standard KG, each item is its own entity (1-to-1 mapping).

    Returns:
        List of (item_id, entity_id) pairs
    """
    links = []

    for item_id, _, _, _ in items:
        # Simple 1-to-1 mapping
        links.append((item_id, item_id))

    return links


def save_kg(triples: List[Tuple[str, str, str]], output_file: Path):
    """
    Save KG triples to file in RecBole format.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("head_id:token\trelation_id:token\ttail_id:token\n")

        # Write triples
        for head, relation, tail in triples:
            f.write(f"{head}\t{relation}\t{tail}\n")

    print(f"  ✓ Saved {len(triples)} triples to {output_file}")


def save_link(links: List[Tuple[str, str]], output_file: Path):
    """
    Save item-entity link file in RecBole format.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("item_id:token\tentity_id:token\n")

        # Write links
        for item_id, entity_id in links:
            f.write(f"{item_id}\t{entity_id}\n")

    print(f"  ✓ Saved {len(links)} links to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate standard KG for Amazon Video Games (for KGAT baseline)'
    )
    parser.add_argument('--item_file', type=str,
                       default='data/recbole/amazon-videogames/amazon-videogames.item',
                       help='Path to amazon-videogames.item file')
    parser.add_argument('--output_dir', type=str,
                       default='data/recbole/amazon-videogames',
                       help='Output directory')

    args = parser.parse_args()

    item_file = Path(args.item_file)
    output_dir = Path(args.output_dir)

    kg_file = output_dir / "amazon-videogames.kg"
    link_file = output_dir / "amazon-videogames.link"

    print("="*70)
    print("Generating Standard KG for Amazon Video Games")
    print("="*70)
    print(f"\nInput:  {item_file}")
    print(f"Output: {kg_file}")
    print(f"        {link_file}")
    print()

    # Check if item file exists
    if not item_file.exists():
        print(f"Error: Item file not found: {item_file}")
        print("\nPlease run prepare_videogames_dataset.py first to generate")
        print("RecBole format data files.")
        return

    # Parse item file
    print("Parsing item file...")
    items = parse_item_file(item_file)
    print(f"  ✓ Parsed {len(items)} items")

    # Generate KG triples
    print("\nGenerating KG triples...")
    triples = generate_kg_triples(items)
    print(f"  ✓ Generated {len(triples)} triples")

    # Statistics
    num_category_triples = sum(1 for _, r, _ in triples if r == "has_category")
    num_price_triples = sum(1 for _, r, _ in triples if r == "has_price_range")

    print(f"\nTriple Statistics:")
    print(f"  Category triples: {num_category_triples}")
    print(f"  Price range triples: {num_price_triples}")
    print(f"  Total triples: {len(triples)}")

    # Get unique entities
    unique_categories = set(tail for _, r, tail in triples if r == "has_category")
    unique_price_ranges = set(tail for _, r, tail in triples if r == "has_price_range")

    print(f"\nUnique Entities:")
    print(f"  Categories: {len(unique_categories)}")
    print(f"  Price ranges: {len(unique_price_ranges)}")
    print(f"  Total entities: {len(unique_categories) + len(unique_price_ranges)}")

    if unique_categories:
        print(f"  Example categories: {sorted(list(unique_categories))[:5]}")
    print(f"  Price ranges: {sorted(unique_price_ranges)}")

    # Generate link file
    print("\nGenerating item-entity links...")
    links = generate_link_file(items)
    print(f"  ✓ Generated {len(links)} links (1-to-1 mapping)")

    # Save files
    print(f"\nSaving files...")
    save_kg(triples, kg_file)
    save_link(links, link_file)

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"KG file: {kg_file}")
    print(f"  - {len(triples)} triples")
    print(f"  - {len(unique_categories) + len(unique_price_ranges)} unique entities")
    print(f"  - 2 relation types: has_category, has_price_range")
    print()
    print(f"Link file: {link_file}")
    print(f"  - {len(links)} item-entity mappings (1-to-1)")
    print()
    print("Next steps:")
    print("  1. Copy these files to KGAT training directory")
    print("  2. Update KGAT config to use these KG files")
    print("  3. Run KGAT baseline training")
    print("="*70)
    print()


if __name__ == '__main__':
    main()
