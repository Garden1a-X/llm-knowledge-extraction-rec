#!/usr/bin/env python3
"""
Convert Amazon Beauty dataset to RecBole format
Creates .inter, item/user mappings, and prepares for knowledge extraction
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def load_jsonl(filepath):
    """Load JSON lines file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def process_beauty_dataset():
    """Process Beauty dataset to RecBole format"""

    print("="*70)
    print("Converting Amazon Beauty to RecBole Format")
    print("="*70)
    print()

    # Paths
    base_path = Path("data/raw/amazon-beauty")
    reviews_path = base_path / "All_Beauty_5.json"
    meta_path = base_path / "meta_All_Beauty.json"

    output_dir = Path("data/recbole/amazon-beauty")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading reviews...")
    reviews = load_jsonl(reviews_path)
    print(f"  Total reviews: {len(reviews):,}")

    print("\nLoading metadata...")
    metadata = load_jsonl(meta_path)
    meta_dict = {item['asin']: item for item in metadata if 'asin' in item}
    print(f"  Total metadata entries: {len(metadata):,}")

    # Get unique users and items from reviews
    review_users = sorted(set(r['reviewerID'] for r in reviews))
    review_items = sorted(set(r['asin'] for r in reviews))

    print(f"\nDataset statistics:")
    print(f"  Unique users: {len(review_users):,}")
    print(f"  Unique items: {len(review_items):,}")

    # Filter items: only keep those with metadata
    items_with_meta = [asin for asin in review_items if asin in meta_dict]
    print(f"  Items with metadata: {len(items_with_meta):,} ({len(items_with_meta)/len(review_items)*100:.1f}%)")

    # Filter items: only keep those with image
    items_with_image = []
    for asin in items_with_meta:
        meta = meta_dict[asin]
        has_image = ('imageURL' in meta and meta['imageURL']) or \
                   ('imageURLHighRes' in meta and meta['imageURLHighRes'])

        if has_image:
            items_with_image.append(asin)

    print(f"  Items with image: {len(items_with_image):,} ({len(items_with_image)/len(review_items)*100:.1f}%)")

    # Use items with image
    valid_items = set(items_with_image)

    # Filter reviews: only keep those with valid items
    filtered_reviews = [r for r in reviews if r['asin'] in valid_items]
    print(f"\nFiltered reviews: {len(filtered_reviews):,} (kept {len(filtered_reviews)/len(reviews)*100:.1f}%)")

    # Get final user/item sets
    final_users = sorted(set(r['reviewerID'] for r in filtered_reviews))
    final_items = sorted(set(r['asin'] for r in filtered_reviews))

    print(f"Final dataset:")
    print(f"  Users: {len(final_users):,}")
    print(f"  Items: {len(final_items):,}")
    print(f"  Interactions: {len(filtered_reviews):,}")

    # Create mappings
    user_map = {user_id: idx for idx, user_id in enumerate(final_users)}
    item_map = {asin: idx for idx, asin in enumerate(final_items)}

    # Create .inter file
    print(f"\nCreating {output_dir / 'amazon-beauty.inter'}...")
    inter_path = output_dir / "amazon-beauty.inter"

    with open(inter_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")

        # Write interactions
        for review in filtered_reviews:
            user_idx = user_map[review['reviewerID']]
            item_idx = item_map[review['asin']]
            rating = review['overall']
            timestamp = review['unixReviewTime']

            f.write(f"{user_idx}\t{item_idx}\t{rating}\t{timestamp}\n")

    print(f"  ✓ Wrote {len(filtered_reviews):,} interactions")

    # Save mappings for later use
    mapping_dir = output_dir / "mappings"
    mapping_dir.mkdir(exist_ok=True)

    # User mapping
    with open(mapping_dir / "user_mapping.json", 'w') as f:
        json.dump({
            'original_to_recbole': user_map,
            'recbole_to_original': {v: k for k, v in user_map.items()}
        }, f, indent=2)

    # Item mapping
    with open(mapping_dir / "item_mapping.json", 'w') as f:
        json.dump({
            'original_to_recbole': item_map,
            'recbole_to_original': {v: k for k, v in item_map.items()}
        }, f, indent=2)

    print(f"  ✓ Saved mappings to {mapping_dir}/")

    # Save filtered metadata for knowledge extraction
    filtered_meta = {asin: meta_dict[asin] for asin in final_items}
    with open(mapping_dir / "filtered_metadata.json", 'w') as f:
        json.dump(filtered_meta, f, indent=2)

    print(f"  ✓ Saved filtered metadata ({len(filtered_meta):,} items)")

    # Print statistics
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - amazon-beauty.inter ({len(filtered_reviews):,} interactions)")
    print(f"  - mappings/user_mapping.json ({len(final_users):,} users)")
    print(f"  - mappings/item_mapping.json ({len(final_items):,} items with images)")
    print(f"  - mappings/filtered_metadata.json ({len(filtered_meta):,} items)")
    print()
    print("Next steps:")
    print("  1. Extract item knowledge from images: python scripts/extract_beauty_knowledge.py")
    print("  2. Extract user interests: python scripts/extract_beauty_user_interests.py")
    print("  3. Create RecBole KG files: python scripts/convert_beauty_to_kg.py")
    print(f"{'='*70}")
    print()


if __name__ == '__main__':
    process_beauty_dataset()
