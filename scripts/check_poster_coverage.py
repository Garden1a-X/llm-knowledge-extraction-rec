#!/usr/bin/env python3
"""
Check poster coverage for filtered MovieLens items.

This script verifies that all items in the 5-core filtered dataset
have corresponding poster images.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from pathlib import Path


def check_poster_coverage(
    id_mapping_path: str,
    poster_dir: str,
    verbose: bool = True
):
    """
    Check which items have posters available.

    Args:
        id_mapping_path: Path to id_mappings.json
        poster_dir: Directory containing poster images (named by original movie_id)
        verbose: Print detailed information

    Returns:
        Dict with coverage statistics
    """
    # Load ID mappings
    with open(id_mapping_path, 'r') as f:
        mappings = json.load(f)

    item_map = mappings['item_id_map']['new_to_original']
    total_items = len(item_map)

    print(f"Checking poster coverage for {total_items} items...")
    print(f"ID mapping file: {id_mapping_path}")
    print(f"Poster directory: {poster_dir}")
    print()

    # Check each item
    poster_dir = Path(poster_dir)
    available = []
    missing = []

    for recbole_id_str, original_movie_id in item_map.items():
        recbole_id = int(recbole_id_str)
        poster_path = poster_dir / f"{original_movie_id}.jpg"

        if poster_path.exists():
            available.append({
                'recbole_id': recbole_id,
                'original_id': original_movie_id,
                'poster_path': str(poster_path)
            })
        else:
            missing.append({
                'recbole_id': recbole_id,
                'original_id': original_movie_id,
                'expected_path': str(poster_path)
            })

    # Print results
    print("="*60)
    print("POSTER COVERAGE REPORT")
    print("="*60)
    print(f"Total items (5-core filtered): {total_items}")
    print(f"Posters available: {len(available)} ({100*len(available)/total_items:.2f}%)")
    print(f"Posters missing: {len(missing)} ({100*len(missing)/total_items:.2f}%)")
    print()

    if missing:
        print(f"⚠️  WARNING: {len(missing)} items are missing posters!")
        print()
        if verbose and len(missing) <= 20:
            print("Missing posters (original movie IDs):")
            for item in missing[:20]:
                print(f"  - Movie ID {item['original_id']} (RecBole ID {item['recbole_id']})")
        elif verbose:
            print(f"Missing posters (showing first 20 of {len(missing)}):")
            for item in missing[:20]:
                print(f"  - Movie ID {item['original_id']} (RecBole ID {item['recbole_id']})")
            print(f"  ... and {len(missing) - 20} more")
    else:
        print("✅ All items have posters available!")

    print()
    print("="*60)

    # Return statistics
    return {
        'total_items': total_items,
        'available_count': len(available),
        'missing_count': len(missing),
        'available_items': available,
        'missing_items': missing,
        'coverage_rate': len(available) / total_items if total_items > 0 else 0
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Check poster coverage for filtered items')
    parser.add_argument('--id_mapping', type=str,
                        default='data/recbole/ml-1m/id_mappings.json',
                        help='Path to id_mappings.json')
    parser.add_argument('--poster_dir', type=str,
                        default='data/raw/ml-1m/posters',
                        help='Directory containing posters')
    parser.add_argument('--save_report', type=str, default=None,
                        help='Save detailed report to JSON file')
    parser.add_argument('--quiet', action='store_true',
                        help='Only print summary statistics')

    args = parser.parse_args()

    # Check coverage
    results = check_poster_coverage(
        id_mapping_path=args.id_mapping,
        poster_dir=args.poster_dir,
        verbose=not args.quiet
    )

    # Save report if requested
    if args.save_report:
        with open(args.save_report, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed report saved to: {args.save_report}")

    # Return exit code based on coverage
    if results['missing_count'] > 0:
        print(f"\n⚠️  {results['missing_count']} items need poster images")
        return 1
    else:
        print("\n✅ All items have posters - ready for knowledge extraction!")
        return 0


if __name__ == '__main__':
    exit(main())
