#!/usr/bin/env python3
"""
Prepare KGAT-compatible data with numeric entity IDs.

KGAT requires:
1. .inter file: user-item interactions
2. .kg file: knowledge graph with NUMERIC entity IDs (not entity names!)
3. .link file: item-entity mapping

The key challenge: Our KG has entity NAMES (e.g., "adventure", "warm_colors"),
but RecBole KGAT expects numeric IDs. We need to:
- Create an entity vocabulary (entity_name -> entity_id mapping)
- Convert entity names to IDs in .kg file
- Generate .link file (item_id -> entity_id for items as entities)

Usage:
    # Step 1: Check what data we have
    python baselines/prepare_kgat_data.py --check

    # Step 2: Prepare KGAT data (creates entity vocabulary + converts KG)
    python baselines/prepare_kgat_data.py \
        --extraction_results results/phase4_full_extraction_filtered.json \
        --inter_file data/recbole/ml-1m/ml-1m.inter \
        --output_dir data/recbole/ml-1m
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from collections import Counter
from typing import Dict, List, Set, Tuple


def create_entity_vocabulary(
    extraction_results: List[Dict]
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create entity vocabulary from extraction results.

    Entity IDs start after item IDs to avoid conflicts.
    If we have N items (1 to N), entities start from N+1.

    Returns:
        (entity_to_id, id_to_entity) mappings
    """
    print("\nCreating entity vocabulary...")

    # Collect all unique entities
    all_entities = set()
    for result in extraction_results:
        if result.get('status') != 'success':
            continue

        knowledge_points = result.get('knowledge_points', [])
        for kp in knowledge_points:
            entity = kp['entity'].replace(' ', '_')
            all_entities.add(entity)

    # Get max item ID to determine entity ID start
    max_item_id = 0
    for result in extraction_results:
        if result.get('status') == 'success':
            item_id = result.get('recbole_id', 0)
            max_item_id = max(max_item_id, item_id)

    # Create mappings (entities start after items)
    entity_start_id = max_item_id + 1
    sorted_entities = sorted(all_entities)

    entity_to_id = {}
    id_to_entity = {}

    for idx, entity in enumerate(sorted_entities, start=entity_start_id):
        entity_to_id[entity] = idx
        id_to_entity[idx] = entity

    print(f"  ✓ Created vocabulary:")
    print(f"    - Max item ID: {max_item_id}")
    print(f"    - Entity IDs: [{entity_start_id}, {entity_start_id + len(sorted_entities) - 1}]")
    print(f"    - Total entities: {len(sorted_entities)}")

    return entity_to_id, id_to_entity


def convert_kg_with_numeric_ids(
    extraction_results: List[Dict],
    entity_to_id: Dict[str, int],
    output_kg_path: Path
):
    """
    Convert KG to RecBole format with numeric entity IDs.
    """
    print(f"\nConverting KG to numeric IDs...")

    triplets = []
    total_kps = 0

    for result in extraction_results:
        if result.get('status') != 'success':
            continue

        item_id = result['recbole_id']
        knowledge_points = result.get('knowledge_points', [])

        for kp in knowledge_points:
            relation = kp['relation']
            entity_name = kp['entity'].replace(' ', '_')
            entity_id = entity_to_id[entity_name]

            # Triplet: (item_id, relation, entity_id)
            triplets.append((item_id, relation, entity_id))
            total_kps += 1

    # Write to file
    output_kg_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_kg_path, 'w', encoding='utf-8') as f:
        # Header (NO :token suffix for IDs that are numeric!)
        f.write('head_id:token\trelation_id:token\ttail_id:token\n')

        # Triplets (sorted by item_id)
        for head_id, relation_id, tail_id in sorted(triplets):
            f.write(f'{head_id}\t{relation_id}\t{tail_id}\n')

    print(f"  ✓ Wrote {len(triplets)} triplets to {output_kg_path}")
    print(f"  ✓ Format: head_id:token (item) -> relation_id:token -> tail_id:token (entity)")

    return triplets


def create_link_file(
    extraction_results: List[Dict],
    entity_to_id: Dict[str, int],
    output_link_path: Path
):
    """
    Create .link file mapping items to entities.

    For KGAT, items are also part of the entity space.
    We map each item to itself: item_id -> item_id
    """
    print(f"\nCreating .link file...")

    # Collect all item IDs
    item_ids = set()
    for result in extraction_results:
        if result.get('status') == 'success':
            item_ids.add(result['recbole_id'])

    # Write link file (item -> item mapping)
    output_link_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_link_path, 'w', encoding='utf-8') as f:
        # Header (NO :token suffix!)
        f.write('item_id\tentity_id\n')

        # Each item maps to itself
        for item_id in sorted(item_ids):
            f.write(f'{item_id}\t{item_id}\n')

    print(f"  ✓ Wrote {len(item_ids)} item-entity mappings to {output_link_path}")
    print(f"  ✓ Mapping: item_id -> item_id (items as self-entities)")


def check_existing_data(output_dir: Path):
    """
    Check what RecBole data files already exist.
    """
    print("\nChecking existing RecBole data...")
    print(f"Directory: {output_dir}")
    print()

    required_files = {
        'ml-1m.inter': 'User-item interactions (required)',
        'ml-1m.kg': 'Knowledge graph triplets (required for KGAT)',
        'ml-1m.link': 'Item-entity mapping (required for KGAT)',
        'entity_vocab.json': 'Entity vocabulary (our mapping)'
    }

    for filename, description in required_files.items():
        filepath = output_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  ✓ {filename:25s} - {size:>10,} bytes - {description}")
        else:
            print(f"  ✗ {filename:25s} - MISSING - {description}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Prepare KGAT-compatible data with numeric entity IDs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--check', action='store_true',
                        help='Check what data files already exist')
    parser.add_argument('--extraction_results', type=str,
                        default='results/phase4_full_extraction_filtered.json',
                        help='Path to extraction results JSON')
    parser.add_argument('--inter_file', type=str,
                        default='data/recbole/ml-1m/ml-1m.inter',
                        help='Path to existing .inter file (must exist!)')
    parser.add_argument('--output_dir', type=str,
                        default='data/recbole/ml-1m',
                        help='Output directory for KGAT data')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Check mode
    if args.check:
        check_existing_data(output_dir)
        return 0

    print("="*70)
    print("PREPARE KGAT-COMPATIBLE DATA")
    print("="*70)

    # 1. Check if .inter file exists
    inter_file = Path(args.inter_file)
    if not inter_file.exists():
        print(f"\n❌ ERROR: .inter file not found: {inter_file}")
        print("\nYou need to run prepare_data_for_recbole.py first!")
        print("Example:")
        print("  python baselines/prepare_data_for_recbole.py \\")
        print("      --ml_data_dir /path/to/ml-1m \\")
        print("      --output_dir data/recbole/ml-1m")
        return 1

    print(f"\n✓ Found .inter file: {inter_file}")

    # 2. Load extraction results
    extraction_file = Path(args.extraction_results)
    if not extraction_file.exists():
        print(f"\n❌ ERROR: Extraction results not found: {extraction_file}")
        return 1

    print(f"✓ Found extraction results: {extraction_file}")

    with open(extraction_file, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    print(f"  Loaded {len(results)} extraction results")

    # 3. Create entity vocabulary
    entity_to_id, id_to_entity = create_entity_vocabulary(results)

    # Save entity vocabulary
    vocab_path = output_dir / 'entity_vocab.json'
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    vocab_data = {
        'entity_to_id': entity_to_id,
        'id_to_entity': {str(k): v for k, v in id_to_entity.items()},
        'num_entities': len(entity_to_id)
    }

    with open(vocab_path, 'w') as f:
        json.dump(vocab_data, f, indent=2)

    print(f"\n✓ Saved entity vocabulary to {vocab_path}")

    # 4. Convert KG with numeric IDs
    kg_path = output_dir / 'ml-1m.kg'
    triplets = convert_kg_with_numeric_ids(results, entity_to_id, kg_path)

    # 5. Create .link file
    link_path = output_dir / 'ml-1m.link'
    create_link_file(results, entity_to_id, link_path)

    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Entity vocabulary: {len(entity_to_id)} entities")
    print(f"✓ KG triplets: {len(triplets)}")
    print(f"✓ Output directory: {output_dir}")
    print()
    print("Files created:")
    print(f"  - {kg_path.name}: KG with numeric entity IDs")
    print(f"  - {link_path.name}: Item-entity mapping")
    print(f"  - {vocab_path.name}: Entity vocabulary")
    print()
    print("Next step: Run KGAT baseline")
    print("  python baselines/run_baseline.py --model KGAT --dataset ml-1m")
    print("="*70)

    return 0


if __name__ == '__main__':
    exit(main())
