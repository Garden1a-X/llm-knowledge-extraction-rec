#!/usr/bin/env python3
"""
Explore Amazon datasets (Video Games and Beauty)
Display basic statistics and sample records
"""

import json
import gzip
from pathlib import Path
from collections import Counter, defaultdict


def load_jsonl(filepath):
    """Load JSON lines file (supports both .json and .json.gz)"""
    data = []
    filepath = Path(filepath)

    # Try .json.gz first, then .json
    if filepath.suffix == '.gz':
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        # Try without .gz
        json_path = filepath.with_suffix('') if filepath.suffix == '.gz' else filepath
        if not json_path.exists():
            # Maybe it's .json.gz but passed without .gz
            json_path = Path(str(filepath) + '.gz')
            if json_path.exists():
                with gzip.open(json_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
            else:
                raise FileNotFoundError(f"File not found: {filepath}")
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
    return data


def explore_reviews(filepath, name):
    """Explore reviews dataset"""
    print(f"\n{'='*60}")
    print(f"Reviews: {name}")
    print(f"{'='*60}\n")

    reviews = load_jsonl(filepath)

    # Basic statistics
    print(f"Total reviews: {len(reviews):,}")

    users = set(r['reviewerID'] for r in reviews)
    items = set(r['asin'] for r in reviews)
    print(f"Unique users: {len(users):,}")
    print(f"Unique items: {len(items):,}")

    # Rating distribution
    ratings = [r['overall'] for r in reviews]
    rating_dist = Counter(ratings)
    print(f"\nRating distribution:")
    for rating in sorted(rating_dist.keys()):
        count = rating_dist[rating]
        pct = count / len(reviews) * 100
        print(f"  {rating:.1f}: {count:,} ({pct:.1f}%)")

    # Sparsity
    sparsity = (1 - len(reviews) / (len(users) * len(items))) * 100
    print(f"\nSparsity: {sparsity:.2f}%")

    # Sample review
    print(f"\nSample review:")
    sample = reviews[0]
    print(f"  User: {sample.get('reviewerID', 'N/A')}")
    print(f"  Item: {sample.get('asin', 'N/A')}")
    print(f"  Rating: {sample.get('overall', 'N/A')}")
    print(f"  Time: {sample.get('reviewTime', 'N/A')}")
    if 'reviewText' in sample:
        text = sample['reviewText'][:100] + "..." if len(sample['reviewText']) > 100 else sample['reviewText']
        print(f"  Review: {text}")


def explore_metadata(filepath, name):
    """Explore metadata dataset"""
    print(f"\n{'='*60}")
    print(f"Metadata: {name}")
    print(f"{'='*60}\n")

    metadata = load_jsonl(filepath)

    # Basic statistics
    print(f"Total items: {len(metadata):,}")

    # Field coverage
    fields = {
        'title': lambda x: x.get('title'),
        'description': lambda x: x.get('description') and (x['description'] if isinstance(x['description'], str) else any(x['description'])),
        'price': lambda x: x.get('price'),
        'brand': lambda x: x.get('brand'),
        'category': lambda x: x.get('category') and len(x['category']) > 0 if isinstance(x.get('category'), list) else x.get('category'),
        'imageURL': lambda x: x.get('imageURL') or x.get('image'),
        'imageURLHighRes': lambda x: x.get('imageURLHighRes'),
        'feature': lambda x: x.get('feature') and (len(x['feature']) > 0 if isinstance(x['feature'], list) else x['feature'])
    }
    print(f"\nField coverage:")
    for field, check_fn in fields.items():
        count = sum(1 for item in metadata if check_fn(item))
        pct = count / len(metadata) * 100
        print(f"  {field:15s}: {count:6,} ({pct:5.1f}%)")

    # Category distribution (top 10)
    categories = []
    for item in metadata:
        if 'category' in item:
            cats = item['category']
            if isinstance(cats, list) and len(cats) > 0:
                # Take the last (most specific) category
                if isinstance(cats[-1], list):
                    categories.extend(cats[-1])
                else:
                    categories.append(cats[-1])

    if categories:
        cat_dist = Counter(categories)
        print(f"\nTop 10 categories:")
        for cat, count in cat_dist.most_common(10):
            pct = count / len(metadata) * 100
            print(f"  {cat[:40]:40s}: {count:5,} ({pct:5.1f}%)")

    # Brand distribution (top 10)
    brands = [item['brand'] for item in metadata if 'brand' in item and item['brand']]
    if brands:
        brand_dist = Counter(brands)
        print(f"\nTop 10 brands:")
        for brand, count in brand_dist.most_common(10):
            pct = count / len(metadata) * 100
            brand_str = brand[:40] if isinstance(brand, str) else str(brand)[:40]
            print(f"  {brand_str:40s}: {count:5,} ({pct:5.1f}%)")

    # Sample item
    print(f"\nSample item:")
    sample = metadata[0]
    print(f"  ASIN: {sample.get('asin', 'N/A')}")
    print(f"  Title: {sample.get('title', 'N/A')[:60]}")
    if 'description' in sample:
        desc = sample['description']
        if isinstance(desc, list):
            desc = ' '.join(desc)
        desc = desc[:100] + "..." if len(desc) > 100 else desc
        print(f"  Description: {desc}")
    if 'brand' in sample:
        print(f"  Brand: {sample['brand']}")
    if 'category' in sample:
        print(f"  Category: {sample['category']}")
    # Check for both image field names
    image_field = 'imageURL' if 'imageURL' in sample else ('image' if 'image' in sample else None)
    if image_field and sample[image_field]:
        print(f"  Images ({image_field}): {len(sample[image_field])} URLs")
        print(f"    Example: {sample[image_field][0][:80]}...")
    if 'imageURLHighRes' in sample and sample['imageURLHighRes']:
        print(f"  High-res images: {len(sample['imageURLHighRes'])} URLs")


def find_file(directory, basename):
    """Find file with .json or .json.gz extension"""
    json_path = directory / f"{basename}.json"
    gz_path = directory / f"{basename}.json.gz"

    if json_path.exists():
        return json_path
    elif gz_path.exists():
        return gz_path
    else:
        return None


def main():
    """Main function"""
    base_path = Path("data/raw")

    # Video Games
    vg_dir = base_path / "amazon-videogames"
    vg_reviews = find_file(vg_dir, "Video_Games_5")
    vg_meta = find_file(vg_dir, "meta_Video_Games")

    if vg_reviews and vg_meta:
        explore_reviews(vg_reviews, "Video Games")
        explore_metadata(vg_meta, "Video Games")
    else:
        print(f"⚠ Video Games dataset not found at {vg_dir}")
        if not vg_reviews:
            print(f"   Missing: Video_Games_5.json or .json.gz")
        if not vg_meta:
            print(f"   Missing: meta_Video_Games.json or .json.gz")

    # Beauty
    beauty_dir = base_path / "amazon-beauty"
    beauty_reviews = find_file(beauty_dir, "All_Beauty_5")
    beauty_meta = find_file(beauty_dir, "meta_All_Beauty")

    if beauty_reviews and beauty_meta:
        explore_reviews(beauty_reviews, "Beauty")
        explore_metadata(beauty_meta, "Beauty")
    else:
        print(f"⚠ Beauty dataset not found at {beauty_dir}")
        if not beauty_reviews:
            print(f"   Missing: All_Beauty_5.json or .json.gz")
        if not beauty_meta:
            print(f"   Missing: meta_All_Beauty.json or .json.gz")

    # Toys and Games
    toys_dir = base_path / "amazon-toys"
    toys_reviews = find_file(toys_dir, "Toys_and_Games_5")
    toys_meta = find_file(toys_dir, "meta_Toys_and_Games")

    if toys_reviews and toys_meta:
        explore_reviews(toys_reviews, "Toys and Games")
        explore_metadata(toys_meta, "Toys and Games")
    else:
        print(f"⚠ Toys and Games dataset not found at {toys_dir}")
        if not toys_reviews:
            print(f"   Missing: Toys_and_Games_5.json or .json.gz")
        if not toys_meta:
            print(f"   Missing: meta_Toys_and_Games.json or .json.gz")

    print(f"\n{'='*60}")
    print("Exploration complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
