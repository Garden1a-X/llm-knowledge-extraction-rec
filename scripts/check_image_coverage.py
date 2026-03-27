#!/usr/bin/env python3
"""
Check how many items in reviews have image URLs in metadata
For Video Games and Beauty datasets
"""

import json
import gzip
from pathlib import Path
from collections import defaultdict


def load_jsonl(filepath):
    """Load JSON lines file (supports both .json and .json.gz)"""
    data = []
    filepath = Path(filepath)

    if filepath.suffix == '.gz':
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    return data


def check_dataset_images(reviews_path, meta_path, dataset_name):
    """Check image coverage for a dataset"""
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}\n")

    # Load reviews
    print("Loading reviews...")
    reviews = load_jsonl(reviews_path)
    review_items = set(r['asin'] for r in reviews)
    print(f"  Reviews: {len(reviews):,}")
    print(f"  Unique items in reviews: {len(review_items):,}")

    # Load metadata
    print("\nLoading metadata...")
    metadata = load_jsonl(meta_path)
    print(f"  Total metadata entries: {len(metadata):,}")

    # Create metadata index
    meta_dict = {item['asin']: item for item in metadata if 'asin' in item}

    # Check coverage
    items_with_meta = 0
    items_with_image = 0
    items_with_highres = 0
    items_with_title_desc_image = 0

    for asin in review_items:
        if asin in meta_dict:
            items_with_meta += 1
            meta = meta_dict[asin]

            # Check for images
            has_image = False
            has_highres = False

            if 'imageURL' in meta and meta['imageURL']:
                has_image = True
                items_with_image += 1
            if 'imageURLHighRes' in meta and meta['imageURLHighRes']:
                has_highres = True
                items_with_highres += 1

            # Check for complete info (title + description + image)
            has_title = 'title' in meta and meta['title']
            has_desc = False
            if 'description' in meta:
                desc = meta['description']
                if isinstance(desc, str) and desc.strip():
                    has_desc = True
                elif isinstance(desc, list) and any(d.strip() for d in desc):
                    has_desc = True

            if has_title and has_desc and (has_image or has_highres):
                items_with_title_desc_image += 1

    # Print results
    print(f"\n{'='*70}")
    print("Coverage Analysis:")
    print(f"{'='*70}")
    print(f"Items in reviews:                 {len(review_items):6,} (100.0%)")
    print(f"Items with metadata:              {items_with_meta:6,} ({items_with_meta/len(review_items)*100:5.1f}%)")
    print(f"Items with imageURL:              {items_with_image:6,} ({items_with_image/len(review_items)*100:5.1f}%)")
    print(f"Items with imageURLHighRes:       {items_with_highres:6,} ({items_with_highres/len(review_items)*100:5.1f}%)")
    print(f"Items with title+desc+image:      {items_with_title_desc_image:6,} ({items_with_title_desc_image/len(review_items)*100:5.1f}%)")
    print(f"{'='*70}\n")

    # Sample items with complete info
    if items_with_title_desc_image > 0:
        print("Sample items with complete info (title + description + image):\n")
        count = 0
        for asin in review_items:
            if asin in meta_dict:
                meta = meta_dict[asin]
                has_title = 'title' in meta and meta['title']
                has_desc = False
                if 'description' in meta:
                    desc = meta['description']
                    if isinstance(desc, str) and desc.strip():
                        has_desc = True
                    elif isinstance(desc, list) and any(d.strip() for d in desc):
                        has_desc = True
                has_image = ('imageURL' in meta and meta['imageURL']) or \
                           ('imageURLHighRes' in meta and meta['imageURLHighRes'])

                if has_title and has_desc and has_image:
                    print(f"  ASIN: {asin}")
                    print(f"    Title: {meta['title'][:60]}")
                    if 'brand' in meta and meta['brand']:
                        print(f"    Brand: {meta['brand']}")
                    if 'imageURL' in meta and meta['imageURL']:
                        print(f"    Image: {meta['imageURL'][0][:60]}...")
                    print()
                    count += 1
                    if count >= 3:
                        break

    return {
        'total_items': len(review_items),
        'items_with_meta': items_with_meta,
        'items_with_image': items_with_image,
        'items_with_complete': items_with_title_desc_image
    }


def main():
    """Main function"""
    base_path = Path("data/raw")

    results = {}

    # Video Games
    vg_reviews = base_path / "amazon-videogames" / "Video_Games_5.json"
    vg_meta = base_path / "amazon-videogames" / "meta_Video_Games.json"

    if vg_reviews.exists() and vg_meta.exists():
        results['Video Games'] = check_dataset_images(vg_reviews, vg_meta, "Video Games")
    else:
        print(f"⚠ Video Games dataset not found")

    # Beauty
    beauty_reviews = base_path / "amazon-beauty" / "All_Beauty_5.json"
    beauty_meta = base_path / "amazon-beauty" / "meta_All_Beauty.json"

    if beauty_reviews.exists() and beauty_meta.exists():
        results['Beauty'] = check_dataset_images(beauty_reviews, beauty_meta, "Beauty (All_Beauty)")
    else:
        print(f"⚠ Beauty dataset not found")

    # Summary
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}\n")
        for dataset, stats in results.items():
            print(f"{dataset}:")
            print(f"  Total items: {stats['total_items']:,}")
            print(f"  With complete info (title+desc+image): {stats['items_with_complete']:,} "
                  f"({stats['items_with_complete']/stats['total_items']*100:.1f}%)")
            print()


if __name__ == '__main__':
    main()
