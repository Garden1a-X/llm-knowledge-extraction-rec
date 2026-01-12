#!/usr/bin/env python3
"""
Convert movie knowledge graph to RecBole format.

This script converts extraction results (JSON) to RecBole .kg format:
- Tab-separated values
- Header: head_id:token, relation_id:token, tail_id:token
- Rows: movie_id, relation, entity

Usage:
    python scripts/convert_to_recbole_kg.py \
        --input results/phase4_full_extraction_filtered.json \
        --output data/recbole/ml-1m/ml-1m.kg \
        --stats results/recbole_conversion_stats.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from typing import Dict, List
from collections import Counter


def convert_to_recbole_kg(
    results: List[Dict],
    verbose: bool = True
) -> tuple[List[tuple], Dict]:
    """
    Convert extraction results to RecBole KG triplets.

    Args:
        results: List of extraction results from JSON
        verbose: Print statistics

    Returns:
        (triplets, statistics)
        triplets: List of (head_id, relation_id, tail_id) tuples
    """
    triplets = []

    # Statistics
    total_movies = 0
    total_triplets = 0
    movies_with_kg = 0
    relation_counts = Counter()
    entity_counts = Counter()
    unique_entities = set()

    for result in results:
        if result.get('status') != 'success':
            continue

        total_movies += 1
        movie_id = result['recbole_id']
        knowledge_points = result.get('knowledge_points', [])

        if knowledge_points:
            movies_with_kg += 1

        for kp in knowledge_points:
            relation = kp['relation']
            entity = kp['entity']

            # Normalize entity format: replace spaces with underscores
            entity = entity.replace(' ', '_')

            # Create triplet: (movie_id, relation, entity)
            triplets.append((movie_id, relation, entity))

            total_triplets += 1
            relation_counts[relation] += 1
            entity_counts[entity] += 1
            unique_entities.add(entity)

    # Statistics
    stats = {
        'total_movies': total_movies,
        'movies_with_knowledge': movies_with_kg,
        'coverage_percentage': movies_with_kg / total_movies * 100 if total_movies > 0 else 0,
        'total_triplets': total_triplets,
        'avg_triplets_per_movie': total_triplets / total_movies if total_movies > 0 else 0,
        'unique_relations': len(relation_counts),
        'unique_entities': len(unique_entities),
        'relation_distribution': dict(relation_counts),
        'top_entities': dict(entity_counts.most_common(20))
    }

    if verbose:
        print(f"\nConversion Statistics:")
        print(f"  Total movies: {total_movies}")
        print(f"  Movies with KG: {movies_with_kg} ({stats['coverage_percentage']:.2f}%)")
        print(f"  Total triplets: {total_triplets}")
        print(f"  Avg triplets/movie: {stats['avg_triplets_per_movie']:.2f}")
        print(f"  Unique relations: {len(relation_counts)}")
        print(f"  Unique entities: {len(unique_entities)}")
        print()

        print("  Relation distribution:")
        for relation, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
            print(f"    {relation:25s}: {count:5d} ({count/total_triplets*100:5.2f}%)")
        print()

    return triplets, stats


def write_recbole_kg(
    triplets: List[tuple],
    output_file: Path,
    verbose: bool = True
):
    """
    Write triplets to RecBole .kg file format.

    Format:
        - Tab-separated values
        - Header: head_id:token\trelation_id:token\ttail_id:token
        - Rows: movie_id\trelation\tentity
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Writing to RecBole format: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("head_id:token\trelation_id:token\ttail_id:token\n")

        # Write triplets (sorted by movie_id for readability)
        for head_id, relation_id, tail_id in sorted(triplets):
            f.write(f"{head_id}\t{relation_id}\t{tail_id}\n")

    if verbose:
        print(f"  ✓ Wrote {len(triplets)} triplets")


def main():
    parser = argparse.ArgumentParser(
        description='Convert knowledge graph to RecBole format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Input extraction results JSON')
    parser.add_argument('--output', type=str, required=True,
                        help='Output RecBole .kg file')
    parser.add_argument('--stats', type=str, default=None,
                        help='Optional statistics output file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("="*70)
        print("RECBOLE KNOWLEDGE GRAPH CONVERSION")
        print("="*70)
        print()

    # Load input
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return 1

    if verbose:
        print(f"Loading input: {input_file}")

    with open(input_file, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])

    if verbose:
        print(f"  ✓ Loaded {len(results)} results")
        print()

    # Convert to triplets
    if verbose:
        print("Converting to RecBole triplets...")

    triplets, stats = convert_to_recbole_kg(results, verbose=verbose)

    # Write RecBole format
    output_file = Path(args.output)
    write_recbole_kg(triplets, output_file, verbose=verbose)

    # Save statistics if requested
    if args.stats:
        stats_file = Path(args.stats)
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\nSaving statistics: {stats_file}")

        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        if verbose:
            print(f"  ✓ Saved conversion statistics")

    # Summary
    if verbose:
        print()
        print("="*70)
        print("SUMMARY")
        print("="*70)
        print(f"✓ Converted {stats['total_movies']} movies")
        print(f"✓ Generated {stats['total_triplets']} knowledge triplets")
        print(f"✓ {stats['unique_relations']} relations × {stats['unique_entities']} entities")
        print(f"✓ RecBole .kg file: {output_file}")
        print("="*70)

    return 0


if __name__ == '__main__':
    exit(main())
