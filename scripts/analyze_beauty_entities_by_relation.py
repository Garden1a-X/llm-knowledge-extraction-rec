#!/usr/bin/env python3
"""
Analyze Beauty Phase 1 entities grouped by standard relations.

Maps Phase 1 extraction results to standard relation vocabulary,
then outputs entity distribution for each standard relation.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import argparse


def load_relation_vocabulary(vocab_file: Path) -> dict:
    """Load standard relation vocabulary."""
    with open(vocab_file, 'r') as f:
        return json.load(f)


def load_phase1_results(results_file: Path) -> dict:
    """Load Phase 1 extraction results."""
    with open(results_file, 'r') as f:
        return json.load(f)


def analyze_entities(results: list, relation_mapping: dict) -> dict:
    """
    Group entities by standard relations.

    Returns:
        dict: {standard_relation: {entity: count}}
    """
    entities_by_relation = defaultdict(Counter)
    unmapped_relations = Counter()

    for item in results:
        if item.get('status') != 'success':
            continue

        for kp in item.get('knowledge_points', []):
            orig_relation = kp.get('relation', '')
            entity = kp.get('entity', '')

            if not orig_relation or not entity:
                continue

            # Map to standard relation
            std_relation = relation_mapping.get(orig_relation)

            if std_relation:
                entities_by_relation[std_relation][entity] += 1
            else:
                # Unmapped relation -> additional_property
                entities_by_relation['additional_property'][entity] += 1
                unmapped_relations[orig_relation] += 1

    return dict(entities_by_relation), dict(unmapped_relations)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Beauty entities by standard relations'
    )
    parser.add_argument('--results', type=str,
                        default='results/beauty/phase1_results.json',
                        help='Phase 1 results file')
    parser.add_argument('--vocab', type=str,
                        default='data/beauty_relation_vocabulary.json',
                        help='Relation vocabulary file')
    parser.add_argument('--output', type=str,
                        default='results/beauty/entities_by_relation.json',
                        help='Output file')
    parser.add_argument('--top_n', type=int, default=50,
                        help='Show top N entities per relation')

    args = parser.parse_args()

    print("="*70)
    print("Beauty Entity Analysis by Standard Relations")
    print("="*70)

    # Load data
    vocab = load_relation_vocabulary(Path(args.vocab))
    results_data = load_phase1_results(Path(args.results))
    results = results_data.get('results', [])

    relation_mapping = vocab['relation_mapping']
    standard_relations = [r['relation'] for r in vocab['relations']]

    print(f"\nLoaded {len(results)} items from Phase 1")
    print(f"Standard relations: {len(standard_relations)}")
    print()

    # Analyze
    entities_by_relation, unmapped = analyze_entities(results, relation_mapping)

    # Print results
    print("="*70)
    print("Entity Distribution by Standard Relation")
    print("="*70)

    output_data = {
        'source': args.results,
        'vocabulary': args.vocab,
        'relations': {}
    }

    for std_rel in standard_relations:
        entities = entities_by_relation.get(std_rel, Counter())
        total_count = sum(entities.values())
        unique_count = len(entities)

        print(f"\n{'='*70}")
        print(f"{std_rel.upper()}")
        print(f"  Total: {total_count}, Unique: {unique_count}")
        print("-"*70)

        # Top entities
        top_entities = entities.most_common(args.top_n)
        for entity, count in top_entities:
            print(f"  {entity}: {count}")

        if len(entities) > args.top_n:
            print(f"  ... and {len(entities) - args.top_n} more")

        # Save to output
        output_data['relations'][std_rel] = {
            'total_count': total_count,
            'unique_count': unique_count,
            'entities': dict(entities.most_common())  # All entities with counts
        }

    # Unmapped relations
    if unmapped:
        print(f"\n{'='*70}")
        print("UNMAPPED RELATIONS (went to additional_property)")
        print("-"*70)
        for rel, count in sorted(unmapped.items(), key=lambda x: -x[1]):
            print(f"  {rel}: {count}")

    # Save output
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Saved to: {output_file}")
    print("="*70)


if __name__ == '__main__':
    main()
