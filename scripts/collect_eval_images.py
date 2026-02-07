#!/usr/bin/env python3
"""
Collect evaluation images for the 300 sampled items.
Copies images from original locations to a dedicated folder for upload.
"""

import json
import shutil
import os
from pathlib import Path

# Configuration - adjust these paths to your environment
BASE_DATA_PATH = "/data/xuao/llm-knowledge-extraction-rec/data"
IMAGE_PATHS = {
    "ML-1M": f"{BASE_DATA_PATH}/raw/ml-1m/posters",
    "Video Games": f"{BASE_DATA_PATH}/recbole/amazon-videogames/images",
    "Beauty": f"{BASE_DATA_PATH}/recbole/amazon-beauty/images"
}
ID_MAPPING_PATHS = {
    "ML-1M": f"{BASE_DATA_PATH}/recbole/ml-1m/id_mappings.json",
    "Video Games": f"{BASE_DATA_PATH}/recbole/amazon-videogames/id_mappings.json",
    "Beauty": f"{BASE_DATA_PATH}/recbole/amazon-beauty/id_mappings.json"
}

# Output directory
OUTPUT_DIR = "results/quality_eval/images"

def main():
    # Load sampled items
    with open("results/quality_eval/sampled_items.json") as f:
        sampled_items = json.load(f)

    print(f"Total sampled items: {len(sampled_items)}")

    # Load ID mappings for ML-1M (need to convert new_id -> original_id)
    id_mappings = {}
    for dataset, path in ID_MAPPING_PATHS.items():
        if os.path.exists(path):
            with open(path) as f:
                id_mappings[dataset] = json.load(f)
            print(f"Loaded ID mapping for {dataset}")
        else:
            print(f"Warning: ID mapping not found for {dataset}: {path}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Track statistics
    stats = {"copied": 0, "missing": 0, "errors": []}

    for item in sampled_items:
        dataset = item["dataset"]
        item_id = item["item_id"]

        # Determine the original ID / filename
        if dataset == "ML-1M":
            # Convert new_id to original movie ID
            mapping = id_mappings.get(dataset, {}).get("item_id_map", {}).get("new_to_original", {})
            original_id = mapping.get(str(item_id), item_id)
            src_filename = f"{original_id}.jpg"
        else:
            # Amazon datasets - use ASIN directly
            asin = item.get("asin", "")
            if not asin:
                # Fallback: try to get from ID mapping
                mapping = id_mappings.get(dataset, {}).get("item_id_map", {}).get("new_to_original", {})
                asin = mapping.get(str(item_id), str(item_id))
            src_filename = f"{asin}.jpg"

        # Source and destination paths
        src_path = os.path.join(IMAGE_PATHS[dataset], src_filename)

        # Use a consistent naming: dataset_itemid.jpg
        dst_filename = f"{dataset.replace(' ', '_')}_{item_id}.jpg"
        dst_path = os.path.join(OUTPUT_DIR, dst_filename)

        # Copy the file
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            stats["copied"] += 1
        else:
            stats["missing"] += 1
            stats["errors"].append(f"{dataset}: {src_path}")
            print(f"Missing: {src_path}")

    print(f"\n=== Summary ===")
    print(f"Copied: {stats['copied']}")
    print(f"Missing: {stats['missing']}")

    if stats["missing"] > 0:
        print(f"\nMissing files (first 10):")
        for err in stats["errors"][:10]:
            print(f"  {err}")

    print(f"\nImages saved to: {OUTPUT_DIR}/")
    print(f"Total size: ", end="")
    os.system(f"du -sh {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
