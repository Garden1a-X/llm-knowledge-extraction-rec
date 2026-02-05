#!/usr/bin/env python3
"""
Calculate Stage 1 (5%) and Stage 2 (20%) sample sizes for Knowledge Extraction.

For paper Table 9: Knowledge Extraction stages.
- Stage 1: 5% of items (pilot stage)
- Stage 2: 20% of items (refinement stage)
- Stage 3: remaining 75% (production stage)

Usage:
    python scripts/calc_stage_samples.py
"""

from pathlib import Path


# Known item counts from dataset statistics (calculated earlier)
KNOWN_ITEM_COUNTS = {
    'ML-1M': 3416,
    'Beauty': 16340,
    'Video Games': 14969,
}


def count_items(inter_file):
    """Count unique items in .inter file."""
    items = set()
    with open(inter_file, 'r') as f:
        header = f.readline().strip().split('\t')
        item_col = header.index('item_id:token')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > item_col:
                items.add(parts[item_col])
    return len(items)


def main():
    # Try multiple data paths
    data_paths = [
        Path('/data/xuao/llm-knowledge-extraction-rec/data/recbole'),
        Path('anonymous_repo/data'),
        Path('data/recbole'),
    ]

    datasets_config = {
        'ML-1M': ['ml-1m/ml-1m.inter', 'ml-1m/ml-1m.inter'],
        'Beauty': ['amazon-beauty/amazon-beauty.inter', 'amazon-beauty/amazon-beauty.inter'],
        'Video Games': ['amazon-videogames/amazon-videogames.inter', 'amazon-videogames/amazon-videogames.inter'],
    }

    print("=" * 70)
    print("Knowledge Extraction Stage Sample Sizes (for Paper Table 9)")
    print("=" * 70)
    print(f"{'Dataset':<15} {'Total Items':<12} {'Stage1 (5%)':<14} {'Stage2 (20%)':<14} {'Stage3 (75%)':<14}")
    print("-" * 70)

    for name in ['ML-1M', 'Beauty', 'Video Games']:
        total_items = None

        # Try to find the data file
        for data_root in data_paths:
            for rel_path in datasets_config[name]:
                inter_file = data_root / rel_path
                if inter_file.exists():
                    total_items = count_items(inter_file)
                    break
            if total_items:
                break

        # Fall back to known values
        if total_items is None:
            total_items = KNOWN_ITEM_COUNTS.get(name)

        if total_items is None:
            print(f"{name:<15} [DATA NOT FOUND]")
            continue

        stage1 = int(total_items * 0.05)
        stage2 = int(total_items * 0.20)
        stage3 = total_items - stage1 - stage2  # remaining 75%

        print(f"{name:<15} {total_items:<12} {stage1:<14} {stage2:<14} {stage3:<14}")

    print("-" * 70)
    print("\nFor Paper Table 9:")
    print("  - Stage 1 (Pilot): 5% of items, 2 iterations")
    print("  - Stage 2 (Refinement): 20% of items, 2 iterations")
    print("  - Stage 3 (Production): 75% of items, 2 iterations")


if __name__ == '__main__':
    main()
