#!/usr/bin/env python3
"""
Filter extraction results to vocabulary-only entities.

This script filters Phase 4 extraction results to keep only entities
that exist in the standard vocabulary. Also handles format variants.

Usage:
    python scripts/filter_to_vocabulary.py \
        --input results/phase4_full_extraction.json \
        --vocabulary results/standard_entity_vocabulary_v2.json \
        --output results/phase4_full_extraction_filtered.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from typing import Dict, List, Set
from collections import Counter


def load_vocabulary(vocab_file: Path) -> Dict[str, List[str]]:
    """Load standard vocabulary from JSON file."""
    with open(vocab_file, 'r') as f:
        data = json.load(f)

    vocabulary = {}
    for relation, info in data['vocabulary'].items():
        vocabulary[relation] = info['standard_entities']

    return vocabulary


def find_vocabulary_match(entity: str, relation: str, vocabulary: Dict[str, List[str]]) -> str | None:
    """
    Find matching entity in vocabulary (with format variants).

    Returns matched vocabulary entity or None if no match.
    """
    if relation not in vocabulary:
        return None

    vocab_entities = vocabulary[relation]

    # Try exact match first
    if entity in vocab_entities:
        return entity

    # Try format variants
    variants = [
        entity.replace('_', ' '),  # underscore to space
        entity.replace(' ', '_'),  # space to underscore
        entity + 's',              # add plural
        entity.rstrip('s'),        # remove plural
        entity.replace('_', ' ') + 's',
        entity.replace('_', ' ').rstrip('s'),
        entity.lower(),            # lowercase
        entity.title(),            # title case
    ]

    for variant in variants:
        if variant in vocab_entities:
            return variant

    return None


def filter_results(
    results: List[Dict],
    vocabulary: Dict[str, List[str]],
    verbose: bool = True
) -> tuple[List[Dict], Dict]:
    """
    Filter results to keep only vocabulary entities.

    Args:
        results: List of extraction results
        vocabulary: Standard vocabulary dict
        verbose: Print statistics

    Returns:
        (filtered_results, statistics)
    """
    filtered_results = []

    # Statistics
    total_kps = 0
    kept_kps = 0
    removed_kps = 0
    format_corrected = 0

    removed_entities = []
    format_corrections = []

    for result in results:
        if result.get('status') != 'success':
            filtered_results.append(result)
            continue

        filtered_kps = []

        for kp in result.get('knowledge_points', []):
            total_kps += 1
            relation = kp['relation']
            entity = kp['entity']

            # Remove NEW_ prefix if present (shouldn't be there in Phase 4, but just in case)
            if entity.startswith('NEW_'):
                entity = entity.replace('NEW_', '')

            # Try to match to vocabulary
            matched_entity = find_vocabulary_match(entity, relation, vocabulary)

            if matched_entity:
                kept_kps += 1
                filtered_kps.append({
                    'relation': relation,
                    'entity': matched_entity
                })

                # Track if format was corrected
                if matched_entity != entity:
                    format_corrected += 1
                    format_corrections.append({
                        'recbole_id': result['recbole_id'],
                        'relation': relation,
                        'original': entity,
                        'corrected': matched_entity
                    })
            else:
                # Not in vocabulary - remove
                removed_kps += 1
                removed_entities.append({
                    'recbole_id': result['recbole_id'],
                    'relation': relation,
                    'entity': entity
                })

        # Update result with filtered KPs
        filtered_result = result.copy()
        filtered_result['knowledge_points'] = filtered_kps
        filtered_result['num_knowledge_points'] = len(filtered_kps)
        filtered_results.append(filtered_result)

    # Calculate statistics
    retention_rate = kept_kps / total_kps if total_kps > 0 else 0

    stats = {
        'total_knowledge_points': total_kps,
        'kept_knowledge_points': kept_kps,
        'removed_knowledge_points': removed_kps,
        'format_corrected': format_corrected,
        'retention_rate': retention_rate,
        'retention_percentage': retention_rate * 100,
        'removed_entities': removed_entities,
        'format_corrections': format_corrections
    }

    if verbose:
        print(f"\nFiltering Statistics:")
        print(f"  Total KPs: {total_kps}")
        print(f"  Kept (in vocabulary): {kept_kps} ({retention_rate*100:.2f}%)")
        print(f"  Removed (not in vocab): {removed_kps} ({removed_kps/total_kps*100:.2f}%)")
        print(f"  Format corrections: {format_corrected}")
        print()

        # Show top removed entities
        if removed_entities:
            removed_counter = Counter([
                f"{r['relation']}:{r['entity']}"
                for r in removed_entities
            ])
            print("  Top removed entities:")
            for entity_str, count in removed_counter.most_common(10):
                print(f"    {count:3}x  {entity_str}")
            print()

    return filtered_results, stats


def main():
    parser = argparse.ArgumentParser(
        description='Filter extraction results to vocabulary-only entities',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Input extraction results JSON')
    parser.add_argument('--vocabulary', type=str, required=True,
                        help='Vocabulary JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output filtered results JSON')
    parser.add_argument('--report', type=str, default=None,
                        help='Optional filtering report file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("="*70)
        print("VOCABULARY FILTERING")
        print("="*70)
        print()

    # Load input
    if verbose:
        print(f"Loading input: {args.input}")

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return 1

    with open(input_file, 'r') as f:
        data = json.load(f)

    # Load vocabulary
    if verbose:
        print(f"Loading vocabulary: {args.vocabulary}")

    vocab_file = Path(args.vocabulary)
    if not vocab_file.exists():
        print(f"Error: Vocabulary file not found: {vocab_file}")
        return 1

    vocabulary = load_vocabulary(vocab_file)
    vocab_size = sum(len(ents) for ents in vocabulary.values())

    if verbose:
        print(f"  ✓ {len(vocabulary)} relations, {vocab_size} entities")
        print()

    # Original stats
    if verbose:
        total_movies = data.get('total_movies', len(data.get('results', [])))
        successful = data.get('successful', sum(1 for r in data.get('results', []) if r['status'] == 'success'))
        print(f"Input dataset:")
        print(f"  Movies: {total_movies}")
        print(f"  Successful: {successful}")
        print()

    # Filter results
    if verbose:
        print("Filtering to vocabulary entities...")

    filtered_results, filter_stats = filter_results(
        data.get('results', []),
        vocabulary,
        verbose=verbose
    )

    # Save filtered results
    output_data = data.copy()
    output_data['results'] = filtered_results
    output_data['filtered'] = True
    output_data['filter_stats'] = filter_stats
    output_data['vocabulary_file'] = str(vocab_file)

    # Recalculate summary stats
    output_data['total_movies'] = len(filtered_results)
    output_data['successful'] = sum(1 for r in filtered_results if r['status'] == 'success')
    output_data['total_knowledge_points'] = filter_stats['kept_knowledge_points']

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nSaving filtered results to: {output_file}")

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print(f"  ✓ Saved {len(filtered_results)} results")

    # Save report if requested
    if args.report:
        report_file = Path(args.report)
        report_file.parent.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\nSaving filtering report to: {report_file}")

        with open(report_file, 'w') as f:
            json.dump(filter_stats, f, indent=2)

        if verbose:
            print(f"  ✓ Saved filtering details")

    # Summary
    if verbose:
        print()
        print("="*70)
        print("SUMMARY")
        print("="*70)
        print(f"✓ Filtered {filter_stats['kept_knowledge_points']} vocabulary entities")
        print(f"✓ Retention rate: {filter_stats['retention_percentage']:.2f}%")
        print(f"✓ Removed {filter_stats['removed_knowledge_points']} non-vocabulary entities")
        print(f"✓ Output saved to: {output_file}")
        print("="*70)

    return 0


if __name__ == '__main__':
    exit(main())
