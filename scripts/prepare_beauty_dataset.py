#!/usr/bin/env python3
"""
Convert Amazon Beauty dataset to RecBole format
NO 5-core filtering - use all interactions

Data source: https://amazon-reviews-2023.github.io/
Expected files:
  - All_Beauty.jsonl.gz (reviews)
  - meta_All_Beauty.jsonl.gz (metadata)

Or older format:
  - All_Beauty_5.json.gz or All_Beauty.json.gz
  - meta_All_Beauty.json.gz
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


def find_data_files(base_path):
    """Find review and metadata files with various naming conventions"""
    base_path = Path(base_path)

    # Possible review file names
    review_candidates = [
        "All_Beauty.jsonl.gz",
        "All_Beauty.jsonl",
        "All_Beauty_5.json.gz",
        "All_Beauty.json.gz",
        "All_Beauty_5.json",
        "All_Beauty.json",
        "reviews_All_Beauty.json.gz",
        "reviews_All_Beauty.json",
    ]

    # Possible metadata file names
    meta_candidates = [
        "meta_All_Beauty.jsonl.gz",
        "meta_All_Beauty.jsonl",
        "meta_All_Beauty.json.gz",
        "meta_All_Beauty.json",
    ]

    review_path = None
    meta_path = None

    for name in review_candidates:
        path = base_path / name
        if path.exists():
            review_path = path
            break

    for name in meta_candidates:
        path = base_path / name
        if path.exists():
            meta_path = path
            break

    return review_path, meta_path


def process_beauty_dataset(base_path="/data/xuao/llm-knowledge-extraction-rec/data/raw/amazon-beauty",
                           output_base="/data/xuao/llm-knowledge-extraction-rec/data/recbole"):
    """
    Process Beauty dataset to RecBole format
    NO 5-core filtering - use all interactions
    """

    print("="*70)
    print("Converting Amazon Beauty to RecBole Format")
    print("NOTE: No 5-core filtering - using ALL interactions")
    print("="*70)
    print()

    base_path = Path(base_path)

    # Find data files
    reviews_path, meta_path = find_data_files(base_path)

    if reviews_path is None:
        print("Error: Review file not found!")
        print(f"Searched in: {base_path}")
        print("\nPlease download Amazon Beauty dataset:")
        print("  https://amazon-reviews-2023.github.io/")
        print("\nExpected files:")
        print("  - All_Beauty.jsonl.gz (reviews)")
        print("  - meta_All_Beauty.jsonl.gz (metadata)")
        return

    print(f"Found review file: {reviews_path}")

    if meta_path is None:
        print(f"Warning: Metadata file not found!")
        print("Continuing without metadata (limited functionality)...")
        meta_dict = {}
    else:
        print(f"Found metadata file: {meta_path}")
        print("\nLoading metadata...")
        metadata = load_jsonl(meta_path)
        # Handle both 'asin' and 'parent_asin' keys
        meta_dict = {}
        for item in metadata:
            asin = item.get('asin') or item.get('parent_asin')
            if asin:
                meta_dict[asin] = item
        print(f"  Total metadata entries: {len(meta_dict):,}")

    output_dir = Path(output_base) / "amazon-beauty"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reviews
    print("\nLoading reviews...")
    reviews = load_jsonl(reviews_path)
    print(f"  Total reviews: {len(reviews):,}")

    if len(reviews) == 0:
        print("Error: No reviews loaded! Please check the data file.")
        return

    # Get user/item fields (handle different field names)
    def get_user_id(r):
        return r.get('reviewerID') or r.get('user_id')

    def get_item_id(r):
        return r.get('asin') or r.get('parent_asin')

    # Get unique users and items
    review_users = sorted(set(get_user_id(r) for r in reviews if get_user_id(r)))
    review_items = sorted(set(get_item_id(r) for r in reviews if get_item_id(r)))

    print(f"\nRaw dataset statistics:")
    print(f"  Unique users: {len(review_users):,}")
    print(f"  Unique items: {len(review_items):,}")
    print(f"  Total interactions: {len(reviews):,}")

    # Filter items based on metadata/image availability
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
                       ('images' in meta and meta['images']) or \
                       ('main_image' in meta and meta['main_image'])

            if has_image:
                items_with_image.append(asin)

        print(f"  Items with image: {len(items_with_image):,} ({len(items_with_image)/len(review_items)*100:.1f}%)")
        valid_items = set(items_with_image)
    else:
        # No metadata, use all items
        valid_items = set(review_items)
        print(f"  Warning: No metadata filtering applied")

    # Filter reviews: only keep those with valid items
    filtered_reviews = [r for r in reviews if get_item_id(r) in valid_items and get_user_id(r)]
    print(f"\nFiltered reviews: {len(filtered_reviews):,} (kept {len(filtered_reviews)/len(reviews)*100:.1f}%)")

    # Get final user/item sets
    final_users = sorted(set(get_user_id(r) for r in filtered_reviews))
    final_items = sorted(set(get_item_id(r) for r in filtered_reviews))

    print(f"\nFinal dataset (NO 5-core filtering):")
    print(f"  Users: {len(final_users):,}")
    print(f"  Items: {len(final_items):,}")
    print(f"  Interactions: {len(filtered_reviews):,}")

    # Calculate density
    density = len(filtered_reviews) / (len(final_users) * len(final_items)) * 100
    print(f"  Density: {density:.6f}%")

    # Avg interactions per user/item
    avg_per_user = len(filtered_reviews) / len(final_users)
    avg_per_item = len(filtered_reviews) / len(final_items)
    print(f"  Avg interactions per user: {avg_per_user:.2f}")
    print(f"  Avg interactions per item: {avg_per_item:.2f}")

    # Create mappings (RecBole format requires IDs starting from 1)
    user_map = {user_id: idx + 1 for idx, user_id in enumerate(final_users)}
    item_map = {asin: idx + 1 for idx, asin in enumerate(final_items)}

    # Create .inter file
    print(f"\nCreating {output_dir / 'amazon-beauty.inter'}...")
    inter_path = output_dir / "amazon-beauty.inter"

    with open(inter_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")

        # Write interactions
        for review in filtered_reviews:
            user_idx = user_map[get_user_id(review)]
            item_idx = item_map[get_item_id(review)]
            rating = review.get('overall') or review.get('rating', 5.0)
            timestamp = review.get('unixReviewTime') or review.get('timestamp', 0)

            f.write(f"{user_idx}\t{item_idx}\t{rating}\t{timestamp}\n")

    print(f"  ✓ Wrote {len(filtered_reviews):,} interactions")

    # Save mappings
    mapping_dir = output_dir

    # Combined ID mappings
    id_mappings = {
        'user': {
            'original_to_recbole': user_map,
            'recbole_to_original': {str(v): k for k, v in user_map.items()}
        },
        'item': {
            'original_to_recbole': item_map,
            'recbole_to_original': {str(v): k for k, v in item_map.items()}
        }
    }

    with open(mapping_dir / "id_mappings.json", 'w') as f:
        json.dump(id_mappings, f, indent=2)

    print(f"  ✓ Saved id_mappings.json")

    # Save filtered metadata for knowledge extraction
    if meta_dict:
        filtered_meta = {asin: meta_dict[asin] for asin in final_items if asin in meta_dict}
        with open(mapping_dir / "filtered_metadata.json", 'w') as f:
            json.dump(filtered_meta, f, indent=2)
        print(f"  ✓ Saved filtered_metadata.json ({len(filtered_meta):,} items)")

    # Create .item file
    print(f"\nCreating {output_dir / 'amazon-beauty.item'}...")
    item_path = output_dir / "amazon-beauty.item"

    with open(item_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("item_id:token\ttitle:token_seq\tcategories:token_seq\tprice:float\n")

        for asin in final_items:
            item_idx = item_map[asin]

            if asin in meta_dict:
                meta = meta_dict[asin]

                # Extract title
                title = meta.get('title', '').replace('\t', ' ').replace('\n', ' ').strip()
                if not title:
                    title = f"Beauty_{asin}"

                # Extract categories
                categories = meta.get('category', []) or meta.get('categories', [])
                if categories:
                    cat_list = []
                    for cat in categories:
                        if isinstance(cat, str):
                            cat_list.append(cat.replace('\t', ' ').replace('\n', ' ').strip())
                        elif isinstance(cat, list):
                            cat_list.extend([c.replace('\t', ' ').replace('\n', ' ').strip() for c in cat if isinstance(c, str)])
                    category_str = '|'.join(cat_list) if cat_list else 'Unknown'
                else:
                    category_str = 'Unknown'

                # Extract price
                price = meta.get('price', 0.0)
                try:
                    if isinstance(price, str):
                        price = float(price.replace('$', '').replace(',', ''))
                    else:
                        price = float(price) if price else 0.0
                except (ValueError, TypeError):
                    price = 0.0
            else:
                title = f"Beauty_{asin}"
                category_str = 'Unknown'
                price = 0.0

            f.write(f"{item_idx}\t{title}\t{category_str}\t{price}\n")

    print(f"  ✓ Wrote {len(final_items):,} items")

    # Create .user file
    print(f"\nCreating {output_dir / 'amazon-beauty.user'}...")
    user_path = output_dir / "amazon-beauty.user"

    with open(user_path, 'w', encoding='utf-8') as f:
        f.write("user_id:token\n")
        for user_id in final_users:
            user_idx = user_map[user_id]
            f.write(f"{user_idx}\n")

    print(f"  ✓ Wrote {len(final_users):,} users")

    # Print summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"\nRecBole format files created:")
    print(f"  - amazon-beauty.inter ({len(filtered_reviews):,} interactions)")
    print(f"  - amazon-beauty.item ({len(final_items):,} items)")
    print(f"  - amazon-beauty.user ({len(final_users):,} users)")
    print(f"  - id_mappings.json")
    if meta_dict:
        print(f"  - filtered_metadata.json ({len(filtered_meta):,} items)")
    print()
    print("Dataset characteristics:")
    print(f"  - NO 5-core filtering (sparse dataset)")
    print(f"  - Good for cold-start evaluation")
    print(f"  - Density: {density:.6f}%")
    print()
    print("Next steps:")
    print("  1. Download product images for visual extraction")
    print("  2. Extract visual knowledge using LLM")
    print("  3. Build knowledge graph")
    print("  4. Train models")
    print(f"{'='*70}")
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare Amazon Beauty dataset')
    parser.add_argument('--input', type=str,
                        default='/data/xuao/llm-knowledge-extraction-rec/data/raw/amazon-beauty',
                        help='Input directory containing raw data')
    parser.add_argument('--output', type=str,
                        default='/data/xuao/llm-knowledge-extraction-rec/data/recbole',
                        help='Output base directory')

    args = parser.parse_args()

    process_beauty_dataset(args.input, args.output)
