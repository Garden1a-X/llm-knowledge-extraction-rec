#!/usr/bin/env python3
"""
Download product images for Beauty dataset from Amazon URLs
"""

import json
import requests
from pathlib import Path
from tqdm import tqdm
import time


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
                print(f"  ✗ Failed to download {url}: {e}")
                return False

    return False


def main():
    """Download all Beauty product images"""

    print("="*70)
    print("Downloading Beauty Product Images")
    print("="*70)
    print()

    # Load filtered metadata
    metadata_path = Path("data/recbole/amazon-beauty/mappings/filtered_metadata.json")

    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found!")
        print("Please run prepare_beauty_dataset.py first.")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"Loaded metadata for {len(metadata)} items")

    # Create output directory
    output_dir = Path("data/raw/amazon-beauty/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving images to: {output_dir}")
    print()

    # Download images
    success_count = 0
    failed_count = 0
    skipped_count = 0

    for asin, meta in tqdm(metadata.items(), desc="Downloading images"):
        # Output filename
        output_path = output_dir / f"{asin}.jpg"

        # Skip if already exists
        if output_path.exists():
            skipped_count += 1
            continue

        # Get image URL (prefer high-res)
        image_url = None
        if 'imageURLHighRes' in meta and meta['imageURLHighRes']:
            image_url = meta['imageURLHighRes'][0]
        elif 'imageURL' in meta and meta['imageURL']:
            image_url = meta['imageURL'][0]

        if not image_url:
            print(f"  ⚠ No image URL for {asin}")
            failed_count += 1
            continue

        # Download
        if download_image(image_url, output_path):
            success_count += 1
        else:
            failed_count += 1

    # Summary
    print()
    print("="*70)
    print("Download Summary")
    print("="*70)
    print(f"Total items: {len(metadata)}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Already existed: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Images saved to: {output_dir}")
    print("="*70)
    print()


if __name__ == '__main__':
    main()
