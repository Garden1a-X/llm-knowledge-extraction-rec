#!/usr/bin/env python3
"""
Build traditional metadata-based KG for Video Games dataset.
Used for KGAT baseline comparison.

Constructs KG from:
- Categories (item -> has_category -> category_entity)
- Price ranges (item -> has_price_range -> price_entity)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import defaultdict


def load_item_file(item_path):
    """Load item file and parse metadata."""
    items = []

    with open(item_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()  # Skip header

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue

            item_id = int(parts[0])
            title = parts[1]
            categories = parts[2].split('|') if parts[2] else []
            price = float(parts[3]) if parts[3] else 0.0

            items.append({
                'item_id': item_id,
                'title': title,
                'categories': categories,
                'price': price
            })

    return items


def discretize_price(price):
    """Discretize price into bins."""
    if price <= 0:
        return 'price_unknown'
    elif price < 10:
        return 'price_0_10'
    elif price < 20:
        return 'price_10_20'
    elif price < 30:
        return 'price_20_30'
    elif price < 50:
        return 'price_30_50'
    else:
        return 'price_50_plus'


def build_metadata_kg(items):
    """Build KG from item metadata."""

    print("Building metadata KG...")

    # Collect all entities
    all_categories = set()
    all_price_bins = set()

    for item in items:
        for cat in item['categories']:
            if cat and cat.strip():
                all_categories.add(cat.strip())

        price_bin = discretize_price(item['price'])
        all_price_bins.add(price_bin)

    # Assign entity IDs
    # Format: entity_id starts from 1 (following RecBole convention)
    entity_to_id = {}
    current_id = 1

    # Category entities
    for cat in sorted(all_categories):
        entity_to_id[f'category_{cat}'] = current_id
        current_id += 1

    # Price entities
    for price_bin in sorted(all_price_bins):
        entity_to_id[price_bin] = current_id
        current_id += 1

    print(f"  Total entities: {len(entity_to_id)}")
    print(f"    Categories: {len(all_categories)}")
    print(f"    Price bins: {len(all_price_bins)}")

    # Build KG triplets
    kg_triplets = []

    # Relation IDs
    HAS_CATEGORY = 0
    HAS_PRICE = 1

    for item in items:
        item_id = item['item_id']

        # Item -> has_category -> Category
        for cat in item['categories']:
            if cat and cat.strip():
                entity_key = f'category_{cat.strip()}'
                if entity_key in entity_to_id:
                    entity_id = entity_to_id[entity_key]
                    kg_triplets.append((item_id, HAS_CATEGORY, entity_id))

        # Item -> has_price -> PriceRange
        price_bin = discretize_price(item['price'])
        if price_bin in entity_to_id:
            entity_id = entity_to_id[price_bin]
            kg_triplets.append((item_id, HAS_PRICE, entity_id))

    print(f"  Total triplets: {len(kg_triplets)}")

    return kg_triplets, entity_to_id


def build_link_file(items, entity_to_id):
    """Build .link file (item -> entity mapping)."""

    links = []

    for item in items:
        item_id = item['item_id']

        # Link to category entities
        for cat in item['categories']:
            if cat and cat.strip():
                entity_key = f'category_{cat.strip()}'
                if entity_key in entity_to_id:
                    entity_id = entity_to_id[entity_key]
                    links.append((item_id, entity_id))

        # Link to price entity
        price_bin = discretize_price(item['price'])
        if price_bin in entity_to_id:
            entity_id = entity_to_id[price_bin]
            links.append((item_id, entity_id))

    # Deduplicate
    links = list(set(links))

    return links


def save_kg_file(kg_triplets, output_path):
    """Save KG to RecBole format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write("head_id:token\trelation_id:token\ttail_id:token\n")

        # Write triplets
        for head, rel, tail in kg_triplets:
            f.write(f"{head}\t{rel}\t{tail}\n")

    print(f"  ✓ Saved KG: {output_path}")


def save_link_file(links, output_path):
    """Save .link file to RecBole format."""

    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write("item_id:token\tentity_id:token\n")

        # Write links
        for item_id, entity_id in sorted(links):
            f.write(f"{item_id}\t{entity_id}\n")

    print(f"  ✓ Saved links: {output_path}")


def save_entity_mapping(entity_to_id, output_path):
    """Save entity ID mapping for reference."""

    import json

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'entity_to_id': entity_to_id,
            'id_to_entity': {v: k for k, v in entity_to_id.items()}
        }, f, indent=2)

    print(f"  ✓ Saved entity mapping: {output_path}")


def main():
    print("="*70)
    print("Building Metadata KG for Video Games (KGAT baseline)")
    print("="*70)
    print()

    # Paths
    data_dir = Path("data/recbole/amazon-videogames")
    item_path = data_dir / "amazon-videogames.item"

    # Output paths
    # For KGAT: save to KG4RecEval format
    kg4rec_dir = Path("/data/xuao/KG4RecEval/dataset/amazon-videogames")
    kg4rec_dir.mkdir(parents=True, exist_ok=True)

    kg_output = kg4rec_dir / "amazon-videogames.kg"
    link_output = kg4rec_dir / "amazon-videogames.link"
    mapping_output = kg4rec_dir / "entity_mapping.json"

    # Also save to RecBole dir for convenience
    kg_output_recbole = data_dir / "amazon-videogames.kg"
    link_output_recbole = data_dir / "amazon-videogames.link"

    # Check input file
    if not item_path.exists():
        print(f"Error: Item file not found: {item_path}")
        print("Please run prepare_videogames_dataset.py first")
        return 1

    # Load items
    print(f"Loading items from: {item_path}")
    items = load_item_file(item_path)
    print(f"  Loaded {len(items)} items")
    print()

    # Build KG
    kg_triplets, entity_to_id = build_metadata_kg(items)
    print()

    # Build link file
    print("Building item-entity links...")
    links = build_link_file(items, entity_to_id)
    print(f"  Total links: {len(links)}")
    print()

    # Save files
    print("Saving files...")

    # KG4RecEval format (for running KGAT)
    save_kg_file(kg_triplets, kg_output)
    save_link_file(links, link_output)
    save_entity_mapping(entity_to_id, mapping_output)

    # Also save to RecBole dir
    save_kg_file(kg_triplets, kg_output_recbole)
    save_link_file(links, link_output_recbole)

    print()
    print("="*70)
    print("Summary")
    print("="*70)
    print(f"Items: {len(items)}")
    print(f"Entities: {len(entity_to_id)}")
    print(f"KG triplets: {len(kg_triplets)}")
    print(f"Item-entity links: {len(links)}")
    print()
    print(f"KG4RecEval output: {kg4rec_dir}/")
    print(f"RecBole output: {data_dir}/")
    print()
    print("Next: Copy .inter and .item files to KG4RecEval directory:")
    print(f"  cp {data_dir}/amazon-videogames.inter {kg4rec_dir}/")
    print(f"  cp {data_dir}/amazon-videogames.item {kg4rec_dir}/")
    print("="*70)

    return 0


if __name__ == '__main__':
    exit(main())
