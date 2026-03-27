#!/usr/bin/env python3
"""
Download product images for Amazon Beauty dataset
"""

import json
import requests
from pathlib import Path
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# Global counters with thread lock
lock = threading.Lock()
stats = {
    'success': 0,
    'failed': 0,
    'skipped': 0,
    'no_url': 0
}


def download_image(url, save_path, timeout=10, retries=3):
    """Download image from URL with retries"""
    for attempt in range(retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, timeout=timeout, stream=True, headers=headers)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            else:
                return False

    return False


def get_image_url(meta):
    """Extract image URL from metadata (handles various formats)"""
    # Try different image URL fields

    # New Amazon format (2023)
    if 'main_image' in meta and meta['main_image']:
        return meta['main_image'].get('large') or meta['main_image'].get('medium') or meta['main_image'].get('small')

    # Older formats
    if 'imageURLHighRes' in meta and meta['imageURLHighRes']:
        if isinstance(meta['imageURLHighRes'], list):
            return meta['imageURLHighRes'][0] if meta['imageURLHighRes'] else None
        return meta['imageURLHighRes']

    if 'imageURL' in meta and meta['imageURL']:
        if isinstance(meta['imageURL'], list):
            return meta['imageURL'][0] if meta['imageURL'] else None
        return meta['imageURL']

    if 'images' in meta and meta['images']:
        if isinstance(meta['images'], list) and len(meta['images']) > 0:
            first_img = meta['images'][0]
            if isinstance(first_img, dict):
                return first_img.get('large') or first_img.get('medium') or first_img.get('small') or first_img.get('hi_res')
            elif isinstance(first_img, str):
                return first_img

    return None


def download_single_item(asin, meta, output_dir):
    """Download image for a single item"""
    output_path = output_dir / f"{asin}.jpg"

    # Skip if already exists
    if output_path.exists():
        with lock:
            stats['skipped'] += 1
        return 'skipped', None

    # Get image URL
    image_url = get_image_url(meta)

    if not image_url:
        with lock:
            stats['no_url'] += 1
        return 'no_url', asin

    # Download
    if download_image(image_url, output_path):
        with lock:
            stats['success'] += 1
        return 'success', None
    else:
        with lock:
            stats['failed'] += 1
        return 'failed', (asin, image_url)


def main():
    """Download all Beauty product images"""

    print("="*70)
    print("Downloading Amazon Beauty Product Images")
    print("="*70)
    print()

    # Paths
    data_dir = Path("/data/xuao/llm-knowledge-extraction-rec/data/recbole/amazon-beauty")
    metadata_path = data_dir / "filtered_metadata.json"
    output_dir = data_dir / "images"

    # Check metadata exists
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        print("Please run prepare_beauty_dataset.py first!")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"Total items: {len(metadata):,}")
    print(f"Output directory: {output_dir}")
    print()

    # Check existing images
    existing = list(output_dir.glob("*.jpg"))
    print(f"Existing images: {len(existing):,}")

    # Download with progress bar
    print("\nDownloading images...")
    failed_items = []
    no_url_items = []

    items = list(metadata.items())

    # Use thread pool for parallel downloads
    max_workers = 10

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_single_item, asin, meta, output_dir): asin
            for asin, meta in items
        }

        with tqdm(total=len(items), desc="Downloading") as pbar:
            for future in as_completed(futures):
                result, info = future.result()

                if result == 'failed' and info:
                    failed_items.append(info)
                elif result == 'no_url' and info:
                    no_url_items.append(info)

                pbar.update(1)
                pbar.set_postfix({
                    'success': stats['success'],
                    'skipped': stats['skipped'],
                    'failed': stats['failed'],
                    'no_url': stats['no_url']
                })

    # Print summary
    print(f"\n{'='*70}")
    print("Download Summary")
    print(f"{'='*70}")
    print(f"  Success: {stats['success']:,}")
    print(f"  Skipped (already exist): {stats['skipped']:,}")
    print(f"  Failed: {stats['failed']:,}")
    print(f"  No URL: {stats['no_url']:,}")
    print(f"\nTotal images in {output_dir}: {len(list(output_dir.glob('*.jpg'))):,}")

    # Save failed items for retry
    if failed_items:
        failed_path = data_dir / "failed_downloads.json"
        with open(failed_path, 'w') as f:
            json.dump(failed_items, f, indent=2)
        print(f"\nFailed items saved to: {failed_path}")

    if no_url_items:
        no_url_path = data_dir / "no_url_items.json"
        with open(no_url_path, 'w') as f:
            json.dump(no_url_items, f, indent=2)
        print(f"Items without URL saved to: {no_url_path}")

    print(f"{'='*70}")


if __name__ == '__main__':
    main()
