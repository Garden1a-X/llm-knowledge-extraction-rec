#!/usr/bin/env python3
"""
Explore Amazon datasets (Video Games and Beauty)
Display basic statistics and sample records
"""

import json
import gzip
from pathlib import Path
from collections import Counter, defaultdict


def load_jsonl_gz(filepath):
    """Load gzipped JSON lines file"""
    data = []
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def explore_reviews(filepath, name):
    """Explore reviews dataset"""
    print(f"\n{'='*60}")
    print(f"Reviews: {name}")
    print(f"{'='*60}\n")

    reviews = load_jsonl_gz(filepath)

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

    metadata = load_jsonl_gz(filepath)

    # Basic statistics
    print(f"Total items: {len(metadata):,}")

    # Field coverage
    fields = ['title', 'description', 'price', 'brand', 'category', 'image', 'feature']
    print(f"\nField coverage:")
    for field in fields:
        count = sum(1 for item in metadata if field in item and item[field])
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
    if 'image' in sample:
        print(f"  Images: {len(sample['image'])} URLs")
        if sample['image']:
            print(f"    Example: {sample['image'][0][:60]}...")


def main():
    """Main function"""
    base_path = Path("data/raw")

    # Video Games
    vg_reviews = base_path / "amazon-videogames" / "Video_Games_5.json.gz"
    vg_meta = base_path / "amazon-videogames" / "meta_Video_Games.json.gz"

    if vg_reviews.exists() and vg_meta.exists():
        explore_reviews(vg_reviews, "Video Games")
        explore_metadata(vg_meta, "Video Games")
    else:
        print(f"⚠ Video Games dataset not found at {base_path / 'amazon-videogames'}")

    # Beauty
    beauty_reviews = base_path / "amazon-beauty" / "All_Beauty_5.json.gz"
    beauty_meta = base_path / "amazon-beauty" / "meta_All_Beauty.json.gz"

    if beauty_reviews.exists() and beauty_meta.exists():
        explore_reviews(beauty_reviews, "Beauty")
        explore_metadata(beauty_meta, "Beauty")
    else:
        print(f"⚠ Beauty dataset not found at {base_path / 'amazon-beauty'}")

    print(f"\n{'='*60}")
    print("Exploration complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
