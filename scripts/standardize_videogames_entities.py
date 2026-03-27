#!/usr/bin/env python3
"""
Analyze Phase 2 constrained extraction results and standardize entities.

1. Check relation distribution
2. Count entities per relation
3. Standardize entity values within each relation
4. Generate final vocabulary
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import re


def normalize_entity(entity: str) -> str:
    """
    Normalize entity string for standardization.
    - Lowercase
    - Replace spaces/underscores consistently
    - Remove extra whitespace
    """
    entity = entity.lower().strip()
    # Normalize separators (keep underscores)
    entity = re.sub(r'\s+', '_', entity)
    # Remove extra underscores
    entity = re.sub(r'_+', '_', entity)
    entity = entity.strip('_')
    return entity


def standardize_entities_for_relation(entities: list[str], relation: str) -> dict:
    """
    Standardize entities for a specific relation.
    Returns mapping from normalized to canonical form and statistics.
    """
    # Count raw entities
    entity_counter = Counter(entities)

    # Normalize and group
    normalized_groups = defaultdict(list)
    for entity, count in entity_counter.items():
        normalized = normalize_entity(entity)
        normalized_groups[normalized].append((entity, count))

    # Choose canonical form (most frequent original)
    canonical_mapping = {}
    entity_stats = []

    for normalized, variants in normalized_groups.items():
        # Sort by count descending, take most frequent as canonical
        variants_sorted = sorted(variants, key=lambda x: x[1], reverse=True)
        canonical = variants_sorted[0][0]
        total_count = sum(count for _, count in variants)

        canonical_mapping[normalized] = canonical
        entity_stats.append({
            'canonical': canonical,
            'normalized': normalized,
            'count': total_count,
            'variants': [{'raw': raw, 'count': count} for raw, count in variants_sorted]
        })

    # Sort by count descending
    entity_stats.sort(key=lambda x: x['count'], reverse=True)

    return {
        'canonical_mapping': canonical_mapping,
        'entity_stats': entity_stats,
        'total_entities': len(entities),
        'unique_normalized': len(normalized_groups)
    }


def analyze_phase2_results(results_file: Path):
    """Analyze Phase 2 constrained extraction results."""

    print("="*70)
    print("Phase 2 Constrained Extraction Analysis")
    print("="*70)
    print()

    # Load results
    print(f"Loading results from: {results_file}")
    with open(results_file, 'r') as f:
        data = json.load(f)

    results = data['results']
    print(f"  Total results: {len(results)}")

    # Filter successful extractions
    successful = [r for r in results if r['status'] == 'success']
    print(f"  Successful: {len(successful)}")
    print()

    # Collect entities by relation
    relation_entities = defaultdict(list)
    total_knowledge_points = 0

    for result in successful:
        for kp in result.get('knowledge_points', []):
            relation = kp['relation']
            entity = kp['entity']
            relation_entities[relation].append(entity)
            total_knowledge_points += 1

    print(f"Total knowledge points: {total_knowledge_points:,}")
    print(f"Relations used: {len(relation_entities)}")
    print()

    # Relation distribution
    print("="*70)
    print("Relation Distribution")
    print("="*70)
    print()

    relation_counts = {rel: len(entities) for rel, entities in relation_entities.items()}
    sorted_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Relation':<35} {'Count':>10} {'%':>8}")
    print("-"*70)
    for relation, count in sorted_relations:
        percentage = (count / total_knowledge_points) * 100
        print(f"{relation:<35} {count:>10,} {percentage:>7.2f}%")
    print()

    # Standardize entities for each relation
    print("="*70)
    print("Entity Standardization")
    print("="*70)
    print()

    standardized_vocabulary = {}

    for relation, entities in sorted(relation_entities.items()):
        print(f"\n{relation}:")
        print("-" * 70)

        result = standardize_entities_for_relation(entities, relation)
        standardized_vocabulary[relation] = result

        print(f"  Total entities: {result['total_entities']:,}")
        print(f"  Unique normalized: {result['unique_normalized']:,}")
        print(f"  Top 10 entities:")

        for i, stat in enumerate(result['entity_stats'][:10], 1):
            print(f"    {i:2d}. {stat['canonical']:<40} (count: {stat['count']:>4})")
            if len(stat['variants']) > 1:
                print(f"        Variants: {len(stat['variants'])}")

    print()

    return standardized_vocabulary, relation_counts


def save_standardized_vocabulary(vocabulary: dict, relation_counts: dict, output_file: Path):
    """Save standardized vocabulary to JSON."""

    # Prepare output format
    output = {
        'dataset': 'amazon-videogames',
        'phase': 'phase2_constrained',
        'vocabulary_version': '2.0',
        'description': 'Standardized entity vocabulary from Phase 2 constrained extraction',
        'statistics': {
            'total_relations': len(vocabulary),
            'relation_counts': relation_counts
        },
        'relations': {}
    }

    # Add standardized entities for each relation
    for relation, data in vocabulary.items():
        output['relations'][relation] = {
            'total_entities': data['total_entities'],
            'unique_normalized': data['unique_normalized'],
            'entities': [
                {
                    'canonical': stat['canonical'],
                    'normalized': stat['normalized'],
                    'count': stat['count'],
                    'num_variants': len(stat['variants'])
                }
                for stat in data['entity_stats']
            ]
        }

    # Save
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Standardized vocabulary saved to: {output_file}")


def main():
    results_file = Path("results/videogames/phase2_constrained_5pct.json")
    output_file = Path("results/videogames/entity_vocabulary_standardized.json")

    # Analyze
    vocabulary, relation_counts = analyze_phase2_results(results_file)

    # Save
    print()
    print("="*70)
    print("Saving Standardized Vocabulary")
    print("="*70)
    print()
    save_standardized_vocabulary(vocabulary, relation_counts, output_file)

    print()
    print("="*70)
    print("Analysis Complete")
    print("="*70)


if __name__ == '__main__':
    main()
