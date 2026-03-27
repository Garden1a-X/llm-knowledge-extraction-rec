#!/usr/bin/env python3
"""
Download product images for Video Games dataset from Amazon URLs
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
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)  # Wait before retry
                continue
            else:
                return False

    return False


def download_single_item(asin, meta, output_dir):
    """Download image for a single item"""
    # Output filename
    output_path = output_dir / f"{asin}.jpg"

    # Skip if already exists
    if output_path.exists():
        with lock:
            stats['skipped'] += 1
        return 'skipped', None

    # Get image URL (prefer high-res)
    image_url = None

    # Try different image URL fields
    if 'imageURLHighRes' in meta and meta['imageURLHighRes']:
        if isinstance(meta['imageURLHighRes'], list):
            image_url = meta['imageURLHighRes'][0]
        else:
            image_url = meta['imageURLHighRes']
    elif 'imageURL' in meta and meta['imageURL']:
        if isinstance(meta['imageURL'], list):
            image_url = meta['imageURL'][0]
        else:
            image_url = meta['imageURL']
    elif 'images' in meta and meta['images']:
        if isinstance(meta['images'], list) and len(meta['images']) > 0:
            if isinstance(meta['images'][0], dict):
                image_url = meta['images'][0].get('large') or meta['images'][0].get('medium') or meta['images'][0].get('small')
            else:
                image_url = meta['images'][0]

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
    """Download all Video Games product images"""

    print("="*70)
    print("Downloading Video Games Product Images")
    print("="*70)
    print()

    # Load filtered metadata
    metadata_path = Path("data/recbole/amazon-videogames/mappings/filtered_metadata.json")

    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found!")
        print("Please run prepare_videogames_dataset.py first.")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"Loaded metadata for {len(metadata):,} items")

    # Create output directory
    output_dir = Path("data/recbole/amazon-videogames/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving images to: {output_dir}")
    print()

    # Use ThreadPoolExecutor for parallel downloads
    max_workers = 20
    print(f"Using {max_workers} parallel workers for downloading...")
    print()

    failed_items = []
    no_url_items = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        futures = {
            executor.submit(download_single_item, asin, meta, output_dir): asin
            for asin, meta in metadata.items()
        }

        # Process results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            try:
                status, data = future.result()
                if status == 'failed' and data:
                    failed_items.append(data)
                elif status == 'no_url' and data:
                    no_url_items.append(data)
            except Exception as e:
                asin = futures[future]
                print(f"\n  ✗ Exception for {asin}: {e}")
                with lock:
                    stats['failed'] += 1

    # Summary
    print()
    print("="*70)
    print("Download Summary")
    print("="*70)
    print(f"Total items: {len(metadata):,}")
    print(f"Successfully downloaded: {stats['success']:,}")
    print(f"Already existed: {stats['skipped']:,}")
    print(f"No image URL: {stats['no_url']:,}")
    print(f"Failed to download: {stats['failed']:,}")
    print(f"Images saved to: {output_dir}")

    # Show some failed items if any
    if failed_items:
        print(f"\nFirst 5 failed downloads:")
        for asin, url in failed_items[:5]:
            print(f"  - {asin}: {url}")

    if no_url_items:
        print(f"\nFirst 5 items without image URL:")
        for asin in no_url_items[:5]:
            print(f"  - {asin}")

    print("="*70)
    print()


if __name__ == '__main__':
    main()
