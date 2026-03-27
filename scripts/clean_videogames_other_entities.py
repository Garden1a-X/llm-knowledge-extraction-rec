#!/usr/bin/env python3
"""
Clean 'other' entities from Video Games extraction results.

Removes all knowledge points with entities containing '_other' or 'other_'.
These are noise from the extraction process.
"""

import json
from pathlib import Path
from collections import Counter

def clean_extraction_results(input_path: str, output_path: str):
    """
    Remove 'other' entities from extraction results.

    Args:
        input_path: Path to phase2 extraction JSON
        output_path: Path to save cleaned JSON
    """
    print("="*70)
    print("Cleaning 'other' entities from Video Games extraction")
    print("="*70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    # Load extraction results
    with open(input_path, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    print(f"Total items: {len(results)}")

    # Statistics
    total_kps_before = 0
    total_kps_after = 0
    removed_entities = Counter()

    # Clean each item
    cleaned_results = []
    for item in results:
        kps = item.get('knowledge_points', [])
        total_kps_before += len(kps)

        # Filter out 'other' entities
        cleaned_kps = []
        for kp in kps:
            entity = kp.get('entity', '')

            # Check if entity contains 'other'
            if '_other' in entity.lower() or 'other_' in entity.lower():
                removed_entities[entity] += 1
                continue

            cleaned_kps.append(kp)

        total_kps_after += len(cleaned_kps)

        # Update item
        item['knowledge_points'] = cleaned_kps
        cleaned_results.append(item)

    # Update data
    data['results'] = cleaned_results

    # Save cleaned data
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    # Print statistics
    print(f"\nStatistics:")
    print(f"  Knowledge points before: {total_kps_before}")
    print(f"  Knowledge points after:  {total_kps_after}")
    print(f"  Removed:                 {total_kps_before - total_kps_after}")
    print(f"  Reduction:               {((total_kps_before - total_kps_after) / total_kps_before * 100):.1f}%")

    print(f"\nRemoved entities (top 20):")
    for entity, count in removed_entities.most_common(20):
        print(f"  {entity}: {count}")

    print(f"\n✓ Saved cleaned results to {output_path}")
    print("="*70)


def main():
    input_file = "results/videogames/phase2_constrained_5pct.json"
    output_file = "results/videogames/phase2_constrained_5pct_cleaned.json"

    clean_extraction_results(input_file, output_file)

    print("\nNext steps:")
    print("  1. Rebuild entity vocabulary from cleaned data")
    print("  2. Generate KG files")
    print("  3. Run 5 trials")


if __name__ == '__main__':
    main()
