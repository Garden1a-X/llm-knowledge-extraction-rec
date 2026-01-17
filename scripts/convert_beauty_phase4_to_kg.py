#!/usr/bin/env python3
"""
Convert Beauty Phase 4 extraction results to RecBole KG format.

Generates:
- amazon-beauty.item.kg: Item-side knowledge graph (product visual features)
- amazon-beauty.link: Item-entity mapping

Usage:
    python scripts/convert_beauty_phase4_to_kg.py \
        --phase4 results/beauty_phase4_extraction.json \
        --item_mapping data/recbole/amazon-beauty/mappings/item_mapping.json \
        --output_kg data/recbole/amazon-beauty/amazon-beauty.item.kg \
        --output_link data/recbole/amazon-beauty/amazon-beauty.link
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from typing import Dict, List, Set
from collections import Counter


def load_item_mapping(mapping_file: Path) -> Dict[str, int]:
    """Load item mapping: ASIN -> RecBole item_id."""
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    return data['original_to_recbole']


def convert_phase4_to_kg(
    phase4_results: List[Dict],
    asin_to_recbole: Dict[str, int],
    verbose: bool = True
) -> tuple[List[tuple], Dict]:
    """
    Convert Phase 4 extraction results to RecBole KG triplets.

    Args:
        phase4_results: List of extraction results (ASIN-based)
        asin_to_recbole: Mapping from ASIN to RecBole item_id
        verbose: Print statistics

    Returns:
        (triplets, statistics)
        triplets: List of (item_id, relation, entity) tuples
    """
    triplets = []

    # Statistics
    total_products = 0
    products_with_kg = 0
    skipped_asins = []
    relation_counts = Counter()
    entity_counts = Counter()
    unique_entities = set()
    total_triplets = 0

    for result in phase4_results:
        if result.get('status') != 'success':
            continue

        total_products += 1
        asin = result['asin']

        # Get RecBole item_id
        if asin not in asin_to_recbole:
            skipped_asins.append(asin)
            continue

        item_id = asin_to_recbole[asin]
        knowledge_points = result.get('knowledge_points', [])

        if knowledge_points:
            products_with_kg += 1

        for kp in knowledge_points:
            relation = kp['relation']
            entity = kp['entity']

            # Normalize entity format: replace spaces with underscores
            entity = entity.replace(' ', '_')

            # Create triplet: (item_id, relation, entity)
            triplets.append((item_id, relation, entity))

            total_triplets += 1
            relation_counts[relation] += 1
            entity_counts[entity] += 1
            unique_entities.add(entity)

    # Statistics
    stats = {
        'total_products': total_products,
        'products_with_knowledge': products_with_kg,
        'skipped_asins': len(skipped_asins),
        'coverage_percentage': products_with_kg / total_products * 100 if total_products > 0 else 0,
        'total_triplets': total_triplets,
        'avg_triplets_per_product': total_triplets / products_with_kg if products_with_kg > 0 else 0,
        'unique_relations': len(relation_counts),
        'unique_entities': len(unique_entities),
        'relation_distribution': dict(relation_counts),
        'top_entities': dict(entity_counts.most_common(20))
    }

    if verbose:
        print(f"\nConversion Statistics:")
        print(f"  Total products: {total_products}")
        print(f"  Products with KG: {products_with_kg} ({stats['coverage_percentage']:.2f}%)")
        print(f"  Skipped ASINs: {len(skipped_asins)}")
        print(f"  Total triplets: {total_triplets}")
        print(f"  Avg triplets/product: {stats['avg_triplets_per_product']:.2f}")
        print(f"  Unique relations: {len(relation_counts)}")
        print(f"  Unique entities: {len(unique_entities)}")
        print()

        print("  Relation distribution:")
        for relation, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
            print(f"    {relation:25s}: {count:5d} ({count/total_triplets*100:5.2f}%)")
        print()

    return triplets, stats


def write_item_kg(
    triplets: List[tuple],
    output_file: Path,
    verbose: bool = True
):
    """
    Write item-side knowledge graph to RecBole .item.kg format.

    Format:
        - Tab-separated values
        - Header: head_id:token\trelation_id:token\ttail_id:token
        - Rows: item_id\trelation\tentity
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Writing item KG: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("head_id:token\trelation_id:token\ttail_id:token\n")

        # Write triplets (sorted by item_id for readability)
        for item_id, relation, entity in sorted(triplets):
            f.write(f"{item_id}\t{relation}\t{entity}\n")

    if verbose:
        print(f"  ✓ Wrote {len(triplets)} triplets")


def write_link_file(
    item_ids: Set[int],
    output_file: Path,
    verbose: bool = True
):
    """
    Write item-entity link file.

    For Beauty dataset, we use 1:1 mapping (entity_id = item_id)
    since each product is its own entity in the KG.

    Format:
        - Tab-separated values
        - Header: item_id:token\tentity_id:token
        - Rows: item_id\titem_id
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nWriting link file: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("item_id:token\tentity_id:token\n")

        # Write links (1:1 mapping)
        for item_id in sorted(item_ids):
            f.write(f"{item_id}\t{item_id}\n")

    if verbose:
        print(f"  ✓ Wrote {len(item_ids)} item-entity links")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Beauty Phase 4 results to RecBole KG format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--phase4', type=str,
                        default='results/beauty_phase4_extraction.json',
                        help='Phase 4 extraction results JSON')
    parser.add_argument('--item_mapping', type=str,
                        default='data/recbole/amazon-beauty/mappings/item_mapping.json',
                        help='Item mapping JSON (ASIN to RecBole ID)')
    parser.add_argument('--output_kg', type=str,
                        default='data/recbole/amazon-beauty/amazon-beauty.item.kg',
                        help='Output item KG file')
    parser.add_argument('--output_link', type=str,
                        default='data/recbole/amazon-beauty/amazon-beauty.link',
                        help='Output link file')
    parser.add_argument('--stats', type=str, default=None,
                        help='Optional statistics output file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("="*70)
        print("BEAUTY PHASE 4 TO RECBOLE KG CONVERSION")
        print("="*70)
        print()

    # Load Phase 4 results
    phase4_file = Path(args.phase4)
    if not phase4_file.exists():
        print(f"Error: Phase 4 file not found: {phase4_file}")
        return 1

    if verbose:
        print(f"Loading Phase 4 results: {phase4_file}")

    with open(phase4_file, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])

    if verbose:
        print(f"  ✓ Loaded {len(results)} results")

    # Load item mapping
    mapping_file = Path(args.item_mapping)
    if not mapping_file.exists():
        print(f"Error: Item mapping not found: {mapping_file}")
        print("\nPlease run prepare_beauty_dataset.py first to create mappings")
        return 1

    if verbose:
        print(f"\nLoading item mapping: {mapping_file}")

    asin_to_recbole = load_item_mapping(mapping_file)

    if verbose:
        print(f"  ✓ Loaded {len(asin_to_recbole)} item mappings")
        print()

    # Convert to KG triplets
    if verbose:
        print("Converting to RecBole KG triplets...")

    triplets, stats = convert_phase4_to_kg(results, asin_to_recbole, verbose=verbose)

    # Get unique item IDs
    unique_items = set(t[0] for t in triplets)

    # Write item KG file
    output_kg = Path(args.output_kg)
    write_item_kg(triplets, output_kg, verbose=verbose)

    # Write link file
    output_link = Path(args.output_link)
    write_link_file(unique_items, output_link, verbose=verbose)

    # Save statistics if requested
    if args.stats:
        stats_file = Path(args.stats)
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\nSaving statistics: {stats_file}")

        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        if verbose:
            print(f"  ✓ Saved conversion statistics")

    # Summary
    if verbose:
        print()
        print("="*70)
        print("SUMMARY")
        print("="*70)
        print(f"✓ Converted {stats['products_with_knowledge']} products")
        print(f"✓ Generated {stats['total_triplets']} knowledge triplets")
        print(f"✓ {stats['unique_relations']} relations × {stats['unique_entities']} entities")
        print(f"✓ Item KG file: {output_kg}")
        print(f"✓ Link file: {output_link}")
        print("="*70)

    return 0


if __name__ == '__main__':
    exit(main())
