#!/usr/bin/env python3
"""
Convert Amazon Video Games dataset to RecBole format
Creates .inter, .item, .user files and mappings for knowledge extraction
"""

import json
import gzip
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def load_jsonl(filepath):
    """Load JSON lines file (supports .gz)"""
    data = []

    if str(filepath).endswith('.gz'):
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

    return data


def process_videogames_dataset():
    """Process Video Games dataset to RecBole format"""

    print("="*70)
    print("Converting Amazon Video Games to RecBole Format")
    print("="*70)
    print()

    # Paths
    base_path = Path("data/raw/amazon-videogames")
    reviews_path = base_path / "Video_Games_5.json.gz"

    # Try alternative names
    if not reviews_path.exists():
        reviews_path = base_path / "Video_Games.json.gz"
    if not reviews_path.exists():
        reviews_path = base_path / "Video_Games_5.json"

    # Meta file
    meta_path = base_path / "meta_Video_Games.json.gz"
    if not meta_path.exists():
        meta_path = base_path / "meta_Video_Games.json"

    # Check files exist
    if not reviews_path.exists():
        print(f"Error: Reviews file not found!")
        print(f"Expected: {reviews_path}")
        print("\nPlease download Amazon Video Games dataset first:")
        print("  Reviews: Video_Games_5.json.gz")
        print("  Metadata: meta_Video_Games.json.gz")
        return

    if not meta_path.exists():
        print(f"Warning: Metadata file not found at {meta_path}")
        print("Continuing without metadata (will use ASIN as title)...")
        meta_dict = {}
    else:
        print("Loading metadata...")
        metadata = load_jsonl(meta_path)
        meta_dict = {item['asin']: item for item in metadata if 'asin' in item}
        print(f"  Total metadata entries: {len(meta_dict):,}")

    output_dir = Path("data/recbole/amazon-videogames")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reviews
    print("\nLoading reviews...")
    reviews = load_jsonl(reviews_path)
    print(f"  Total reviews: {len(reviews):,}")

    if len(reviews) == 0:
        print("Error: No reviews loaded! Please check the data file.")
        return

    # Get unique users and items from reviews
    review_users = sorted(set(r['reviewerID'] for r in reviews if 'reviewerID' in r))
    review_items = sorted(set(r['asin'] for r in reviews if 'asin' in r))

    print(f"\nDataset statistics:")
    print(f"  Unique users: {len(review_users):,}")
    print(f"  Unique items: {len(review_items):,}")

    # Filter items based on availability
    if meta_dict:
        # Only keep items with metadata
        items_with_meta = [asin for asin in review_items if asin in meta_dict]
        print(f"  Items with metadata: {len(items_with_meta):,} ({len(items_with_meta)/len(review_items)*100:.1f}%)")

        # Filter items: only keep those with image
        items_with_image = []
        for asin in items_with_meta:
            meta = meta_dict[asin]
            has_image = ('imageURL' in meta and meta['imageURL']) or \
                       ('imageURLHighRes' in meta and meta['imageURLHighRes']) or \
                       ('images' in meta and meta['images'])

            if has_image:
                items_with_image.append(asin)

        print(f"  Items with image: {len(items_with_image):,} ({len(items_with_image)/len(review_items)*100:.1f}%)")
        valid_items = set(items_with_image)
    else:
        # No metadata, use all items
        valid_items = set(review_items)
        print(f"  Warning: No metadata filtering applied")

    # Filter reviews: only keep those with valid items
    filtered_reviews = [r for r in reviews if r.get('asin') in valid_items and 'reviewerID' in r]
    print(f"\nFiltered reviews: {len(filtered_reviews):,} (kept {len(filtered_reviews)/len(reviews)*100:.1f}%)")

    # Get final user/item sets
    final_users = sorted(set(r['reviewerID'] for r in filtered_reviews))
    final_items = sorted(set(r['asin'] for r in filtered_reviews))

    print(f"Final dataset:")
    print(f"  Users: {len(final_users):,}")
    print(f"  Items: {len(final_items):,}")
    print(f"  Interactions: {len(filtered_reviews):,}")

    # Calculate density
    density = len(filtered_reviews) / (len(final_users) * len(final_items)) * 100
    print(f"  Density: {density:.4f}%")

    # Create mappings (RecBole format requires IDs starting from 1, not 0)
    user_map = {user_id: idx + 1 for idx, user_id in enumerate(final_users)}
    item_map = {asin: idx + 1 for idx, asin in enumerate(final_items)}

    # Create .inter file
    print(f"\nCreating {output_dir / 'amazon-videogames.inter'}...")
    inter_path = output_dir / "amazon-videogames.inter"

    with open(inter_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")

        # Write interactions
        for review in filtered_reviews:
            user_idx = user_map[review['reviewerID']]
            item_idx = item_map[review['asin']]
            rating = review.get('overall', 5.0)
            timestamp = review.get('unixReviewTime', 0)

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
    if meta_dict:
        filtered_meta = {asin: meta_dict[asin] for asin in final_items if asin in meta_dict}
        with open(mapping_dir / "filtered_metadata.json", 'w') as f:
            json.dump(filtered_meta, f, indent=2)
        print(f"  ✓ Saved filtered metadata ({len(filtered_meta):,} items)")

    # Create .item file (item features)
    print(f"\nCreating {output_dir / 'amazon-videogames.item'}...")
    item_path = output_dir / "amazon-videogames.item"

    with open(item_path, 'w', encoding='utf-8') as f:
        # Header: item_id, title, categories, price
        f.write("item_id:token\ttitle:token_seq\tcategories:token_seq\tprice:float\n")

        for asin in final_items:
            item_idx = item_map[asin]

            if asin in meta_dict:
                meta = meta_dict[asin]

                # Extract title (clean it)
                title = meta.get('title', '').replace('\t', ' ').replace('\n', ' ').strip()
                if not title:
                    title = f"Game_{asin}"

                # Extract categories (flatten nested list and join with |)
                categories = meta.get('category', [])
                if categories:
                    # Flatten if nested and clean
                    cat_list = []
                    for cat in categories:
                        if isinstance(cat, str):
                            cat_list.append(cat.replace('\t', ' ').replace('\n', ' ').strip())
                    category_str = '|'.join(cat_list) if cat_list else 'Unknown'
                else:
                    category_str = 'Unknown'

                # Extract price
                price = meta.get('price', 0.0)
                try:
                    price = float(price) if price else 0.0
                except (ValueError, TypeError):
                    price = 0.0
            else:
                # No metadata
                title = f"Game_{asin}"
                category_str = 'Unknown'
                price = 0.0

            f.write(f"{item_idx}\t{title}\t{category_str}\t{price}\n")

    print(f"  ✓ Wrote {len(final_items):,} items")

    # Create .user file (user features - placeholder)
    print(f"\nCreating {output_dir / 'amazon-videogames.user'}...")
    user_path = output_dir / "amazon-videogames.user"

    with open(user_path, 'w', encoding='utf-8') as f:
        # Header: just user_id (Video Games dataset has no user demographics)
        f.write("user_id:token\n")

        for user_id in final_users:
            user_idx = user_map[user_id]
            f.write(f"{user_idx}\n")

    print(f"  ✓ Wrote {len(final_users):,} users")

    # Print statistics
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"\nRecBole format files created:")
    print(f"  - amazon-videogames.inter ({len(filtered_reviews):,} interactions)")
    print(f"  - amazon-videogames.item ({len(final_items):,} items with features)")
    print(f"  - amazon-videogames.user ({len(final_users):,} users)")
    print(f"\nMapping files created:")
    print(f"  - mappings/user_mapping.json (users: IDs 1-{len(final_users)})")
    print(f"  - mappings/item_mapping.json (items: IDs 1-{len(final_items)})")
    if meta_dict:
        print(f"  - mappings/filtered_metadata.json ({len(filtered_meta):,} items)")
    print()
    print("Next steps:")
    print("  1. Download game images for visual extraction")
    print("  2. Run Phase 4 extraction (vocabulary-constrained)")
    print("  3. Generate item KG from extraction results")
    print("  4. Extract user interests (21-day bucket strategy)")
    print("  5. Generate user KG from user interests")
    print(f"{'='*70}")
    print()


if __name__ == '__main__':
    process_videogames_dataset()
