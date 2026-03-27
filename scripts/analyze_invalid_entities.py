#!/usr/bin/env python3
"""
Analyze invalid entities from Phase 2 results.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


def main():
    result_file = Path("results/beauty/phase2_results_NEW.json")
    vocab_file = Path("data/beauty_entity_vocabulary.json")

    print("Loading data...")
    with open(result_file, 'r') as f:
        results_data = json.load(f)
    with open(vocab_file, 'r') as f:
        vocabulary = json.load(f)

    results = results_data['results']
    print(f"  {len(results)} results loaded")

    # Collect invalid entities
    invalid_by_relation = defaultdict(list)
    valid_relations = set(vocabulary['relations'].keys())

    for result in results:
        if result.get('status') != 'success':
            continue

        for kp in result.get('knowledge_points', []):
            entity = kp['entity']
            relation = kp['relation']

            # Skip NEW_ entities
            if entity.startswith('NEW_'):
                continue

            # Check if relation is valid
            if relation not in valid_relations:
                invalid_by_relation[f"INVALID_RELATION:{relation}"].append(entity)
                continue

            # Check if entity is in vocabulary
            standard = set(vocabulary['relations'][relation]['standard_entities'])
            if entity not in standard:
                invalid_by_relation[relation].append(entity)

    # Print report
    print("\n" + "="*70)
    print("INVALID ENTITIES ANALYSIS")
    print("="*70)

    total_invalid = 0
    for relation, entities in sorted(invalid_by_relation.items()):
        counter = Counter(entities)
        total_invalid += len(entities)

        print(f"\n【{relation}】 ({len(entities)} invalid, {len(counter)} unique)")

        if relation in vocabulary['relations']:
            standard = vocabulary['relations'][relation]['standard_entities']
            print(f"  Standard entities: {standard}")

        print(f"  Top invalid entities:")
        for entity, count in counter.most_common(20):
            # Check if it's close to any standard entity
            hint = ""
            if relation in vocabulary['relations']:
                for std in vocabulary['relations'][relation]['standard_entities']:
                    if entity.lower() == std.lower():
                        hint = f" → CASE MISMATCH, should be '{std}'"
                        break
                    elif entity.replace('_', '') == std.replace('_', ''):
                        hint = f" → SIMILAR to '{std}'"
                        break
                    elif entity in std or std in entity:
                        hint = f" → PARTIAL match '{std}'"
                        break
            print(f"    '{entity}': {count}{hint}")

    print("\n" + "="*70)
    print(f"Total invalid: {total_invalid}")
    print("="*70)

    # Categorize invalid entities
    print("\n" + "="*70)
    print("CATEGORIZATION OF INVALID ENTITIES")
    print("="*70)

    case_mismatch = 0
    wrong_relation = 0
    missing_from_vocab = 0

    for relation, entities in invalid_by_relation.items():
        if relation.startswith("INVALID_RELATION:"):
            wrong_relation += len(entities)
            continue

        counter = Counter(entities)
        for entity, count in counter.items():
            # Check case mismatch
            standard = vocabulary['relations'][relation]['standard_entities']
            matched = False
            for std in standard:
                if entity.lower() == std.lower() and entity != std:
                    case_mismatch += count
                    matched = True
                    break
            if not matched:
                # Check if entity belongs to another relation
                found_elsewhere = False
                for other_rel, other_info in vocabulary['relations'].items():
                    if other_rel != relation and entity in other_info['standard_entities']:
                        wrong_relation += count
                        found_elsewhere = True
                        break
                if not found_elsewhere:
                    missing_from_vocab += count

    print(f"\nCase mismatch (e.g., 'Glossy' vs 'glossy'): {case_mismatch}")
    print(f"Wrong relation (entity exists but in different relation): {wrong_relation}")
    print(f"Missing from vocabulary: {missing_from_vocab}")


if __name__ == '__main__':
    main()
