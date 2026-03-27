#!/usr/bin/env python3
"""
Analyze NEW_ entities from Phase 2 results.

Check which NEW entities might actually match existing vocabulary.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


def load_results(result_file: Path) -> dict:
    with open(result_file, 'r') as f:
        return json.load(f)


def load_vocabulary(vocab_file: Path) -> dict:
    with open(vocab_file, 'r') as f:
        return json.load(f)


def analyze_new_entities(results: list, vocabulary: dict) -> dict:
    """Analyze all NEW_ entities."""

    # Collect NEW entities by relation
    new_by_relation = defaultdict(list)

    for result in results:
        if result.get('status') != 'success':
            continue

        for kp in result.get('knowledge_points', []):
            entity = kp['entity']
            relation = kp['relation']

            if entity.startswith('NEW_'):
                # Remove NEW_ prefix for analysis
                clean_entity = entity[4:].lower().strip()
                new_by_relation[relation].append(clean_entity)

    # Count and analyze
    analysis = {}

    for relation, entities in sorted(new_by_relation.items()):
        counter = Counter(entities)

        # Get standard entities for this relation
        if relation in vocabulary['relations']:
            standard = set(e.lower() for e in vocabulary['relations'][relation]['standard_entities'])
        else:
            standard = set()

        # Check for potential matches
        potential_matches = []
        for entity, count in counter.most_common():
            # Check exact match (case-insensitive)
            if entity in standard:
                potential_matches.append({
                    'new_entity': entity,
                    'count': count,
                    'match_type': 'EXACT_MATCH_EXISTS',
                    'suggestion': f'Should use: {entity}'
                })
            else:
                # Check partial matches
                for std in standard:
                    if entity in std or std in entity:
                        potential_matches.append({
                            'new_entity': entity,
                            'count': count,
                            'match_type': 'PARTIAL_MATCH',
                            'suggestion': f'Similar to: {std}'
                        })
                        break

        analysis[relation] = {
            'total_new': len(entities),
            'unique_new': len(counter),
            'top_20': [{'entity': e, 'count': c} for e, c in counter.most_common(20)],
            'potential_matches': potential_matches[:20],
            'standard_entities': list(vocabulary['relations'].get(relation, {}).get('standard_entities', []))
        }

    return analysis


def main():
    result_file = Path("results/beauty/phase2_results.json")
    vocab_file = Path("data/beauty_entity_vocabulary.json")

    print("Loading data...")
    results_data = load_results(result_file)
    vocabulary = load_vocabulary(vocab_file)

    results = results_data['results']
    print(f"  {len(results)} results loaded")

    # Analyze
    print("\nAnalyzing NEW_ entities...")
    analysis = analyze_new_entities(results, vocabulary)

    # Print report
    print("\n" + "="*70)
    print("NEW ENTITY ANALYSIS REPORT")
    print("="*70)

    total_new = 0
    total_potential_issues = 0

    for relation, data in sorted(analysis.items()):
        print(f"\n【{relation}】")
        print(f"  Total NEW: {data['total_new']}, Unique: {data['unique_new']}")
        print(f"  Standard entities: {data['standard_entities']}")
        print(f"\n  Top 20 NEW entities:")
        for item in data['top_20']:
            print(f"    - {item['entity']}: {item['count']}")

        if data['potential_matches']:
            print(f"\n  ⚠️ Potential issues ({len(data['potential_matches'])}):")
            for match in data['potential_matches'][:10]:
                print(f"    - NEW_{match['new_entity']} ({match['count']}x) -> {match['match_type']}: {match['suggestion']}")

        total_new += data['total_new']
        total_potential_issues += len(data['potential_matches'])

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total NEW entities: {total_new}")
    print(f"Potential issues (matches exist): {total_potential_issues}")

    # Also analyze invalid entities
    print("\n" + "="*70)
    print("INVALID ENTITIES (not NEW_, not in vocab)")
    print("="*70)

    invalid_by_relation = defaultdict(list)
    for result in results:
        if result.get('status') != 'success':
            continue
        for kp in result.get('knowledge_points', []):
            entity = kp['entity']
            relation = kp['relation']
            if not entity.startswith('NEW_'):
                if relation in vocabulary['relations']:
                    standard = set(vocabulary['relations'][relation]['standard_entities'])
                    if entity not in standard:
                        invalid_by_relation[relation].append(entity)

    for relation, entities in sorted(invalid_by_relation.items()):
        counter = Counter(entities)
        print(f"\n【{relation}】 ({len(entities)} invalid)")
        print(f"  Standard: {vocabulary['relations'].get(relation, {}).get('standard_entities', [])}")
        print(f"  Top invalid:")
        for entity, count in counter.most_common(15):
            print(f"    - '{entity}': {count}")

    # Save detailed analysis
    output_file = Path("results/beauty/new_entity_analysis.json")
    with open(output_file, 'w') as f:
        json.dump({
            'analysis': analysis,
            'invalid_by_relation': {r: dict(Counter(e)) for r, e in invalid_by_relation.items()}
        }, f, indent=2)
    print(f"\nDetailed analysis saved to: {output_file}")


if __name__ == '__main__':
    main()
