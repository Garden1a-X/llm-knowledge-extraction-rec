#!/usr/bin/env python3
"""
Post-processing script to fix false NEW_ entities.

This script corrects cases where LLM marked entities with NEW_ prefix
even though they exist in the vocabulary. This happens when the LLM
fails to properly review the vocabulary during self-review.

Usage:
    python scripts/fix_false_new_entities.py \
        --input results/phase3_5percent_test_v2.json \
        --vocabulary results/standard_entity_vocabulary_v2.json \
        --output results/phase3_5percent_test_v2_corrected.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from typing import Dict, List, Set


def load_vocabulary(vocab_file: Path) -> Dict[str, List[str]]:
    """Load standard vocabulary from JSON file."""
    with open(vocab_file, 'r') as f:
        data = json.load(f)

    vocabulary = {}
    for relation, info in data['vocabulary'].items():
        vocabulary[relation] = info['standard_entities']

    return vocabulary


def fix_false_new_entities(
    results: List[Dict],
    vocabulary: Dict[str, List[str]],
    verbose: bool = True
) -> tuple[List[Dict], Dict]:
    """
    Fix false NEW_ entities by removing NEW_ prefix if entity exists in vocabulary.

    Also handles format variants (space vs underscore, plural vs singular, etc.)

    Args:
        results: List of extraction results
        vocabulary: Standard vocabulary dict
        verbose: Print correction details

    Returns:
        (corrected_results, correction_stats)
    """
    corrected_results = []

    # Statistics
    total_new = 0
    corrected_new = 0
    corrections_by_relation = {}
    corrections_detail = []

    for result in results:
        if result.get('status') != 'success':
            corrected_results.append(result)
            continue

        corrected_kps = []
        for kp in result.get('knowledge_points', []):
            relation = kp['relation']
            entity = kp['entity']

            if entity.startswith('NEW_'):
                total_new += 1
                clean_entity = entity.replace('NEW_', '')

                # Check if clean entity exists in vocabulary (with format variants)
                matched_entity = None
                if relation in vocabulary:
                    vocab_entities = vocabulary[relation]

                    # Try exact match first
                    if clean_entity in vocab_entities:
                        matched_entity = clean_entity
                    else:
                        # Try format variants
                        variants = [
                            clean_entity.replace('_', ' '),  # underscore to space
                            clean_entity.replace(' ', '_'),  # space to underscore
                            clean_entity + 's',              # add plural
                            clean_entity.rstrip('s'),        # remove plural
                            clean_entity.replace('_', ' ') + 's',
                            clean_entity.replace('_', ' ').rstrip('s'),
                        ]

                        for variant in variants:
                            if variant in vocab_entities:
                                matched_entity = variant
                                break

                if matched_entity:
                    # This is a false NEW_ - correct it
                    corrected_new += 1
                    corrected_kps.append({
                        'relation': relation,
                        'entity': matched_entity  # Use matched vocab entity
                    })

                    # Track correction
                    if relation not in corrections_by_relation:
                        corrections_by_relation[relation] = []
                    corrections_by_relation[relation].append({
                        'recbole_id': result['recbole_id'],
                        'original': entity,
                        'corrected': matched_entity
                    })
                    corrections_detail.append({
                        'recbole_id': result['recbole_id'],
                        'relation': relation,
                        'original': entity,
                        'corrected': matched_entity
                    })
                else:
                    # This is a true NEW_ - keep it
                    corrected_kps.append(kp)
            else:
                # Not a NEW_ entity - check if it needs format correction
                if relation in vocabulary:
                    vocab_entities = vocabulary[relation]

                    if entity in vocab_entities:
                        # Correctly formatted - keep as is
                        corrected_kps.append(kp)
                    else:
                        # Try format variants to fix invalid entities
                        variants = [
                            entity.replace(' ', '_'),  # space to underscore
                            entity.replace('_', ' '),  # underscore to space
                            entity + 's',
                            entity.rstrip('s'),
                            entity.replace(' ', '_') + 's',
                            entity.replace(' ', '_').rstrip('s'),
                        ]

                        matched_entity = None
                        for variant in variants:
                            if variant in vocab_entities:
                                matched_entity = variant
                                break

                        if matched_entity:
                            # Format mismatch - correct it
                            corrected_kps.append({
                                'relation': relation,
                                'entity': matched_entity
                            })
                            corrections_detail.append({
                                'recbole_id': result['recbole_id'],
                                'relation': relation,
                                'original': entity,
                                'corrected': matched_entity,
                                'type': 'format_fix'
                            })
                        else:
                            # Still invalid - keep as is (will be caught by coverage stats)
                            corrected_kps.append(kp)
                else:
                    # Invalid relation - keep as is
                    corrected_kps.append(kp)

        # Update result with corrected knowledge points
        corrected_result = result.copy()
        corrected_result['knowledge_points'] = corrected_kps
        corrected_result['num_knowledge_points'] = len(corrected_kps)
        corrected_results.append(corrected_result)

    correction_stats = {
        'total_new_entities': total_new,
        'corrected_false_new': corrected_new,
        'true_new_remaining': total_new - corrected_new,
        'correction_rate': corrected_new / total_new if total_new > 0 else 0,
        'corrections_by_relation': corrections_by_relation,
        'corrections_detail': corrections_detail
    }

    if verbose:
        print(f"\nCorrection Statistics:")
        print(f"  Total NEW_ entities found: {total_new}")
        print(f"  False NEW_ corrected: {corrected_new} ({100*corrected_new/total_new:.1f}%)")
        print(f"  True NEW_ remaining: {total_new - corrected_new} ({100*(total_new-corrected_new)/total_new:.1f}%)")
        print()

        if corrections_by_relation:
            print("  Corrections by relation:")
            for rel, corrs in sorted(corrections_by_relation.items(), key=lambda x: len(x[1]), reverse=True):
                print(f"    {rel}: {len(corrs)} corrections")

    return corrected_results, correction_stats


def calculate_coverage_stats(results: List[Dict], vocabulary: Dict[str, List[str]]) -> Dict:
    """Calculate vocabulary coverage statistics."""
    total_kps = 0
    new_kps = 0
    invalid_kps = 0
    new_entities = []
    invalid_entities = []

    for result in results:
        if result.get('status') != 'success':
            continue

        for kp in result.get('knowledge_points', []):
            total_kps += 1
            relation = kp['relation']
            entity = kp['entity']

            if entity.startswith('NEW_'):
                new_kps += 1
                new_entities.append({
                    'relation': relation,
                    'entity': entity,
                    'recbole_id': result['recbole_id']
                })
            else:
                # Validate: check if entity is actually in the vocabulary for this relation
                if relation in vocabulary:
                    if entity not in vocabulary[relation]:
                        invalid_kps += 1
                        invalid_entities.append({
                            'relation': relation,
                            'entity': entity,
                            'recbole_id': result['recbole_id'],
                            'reason': f'Entity "{entity}" not in vocabulary for relation "{relation}"'
                        })
                else:
                    # Relation itself is invalid
                    invalid_kps += 1
                    invalid_entities.append({
                        'relation': relation,
                        'entity': entity,
                        'recbole_id': result['recbole_id'],
                        'reason': f'Invalid relation "{relation}"'
                    })

    valid_kps = total_kps - new_kps - invalid_kps
    coverage_rate = valid_kps / total_kps if total_kps > 0 else 0.0

    return {
        'total_knowledge_points': total_kps,
        'valid_entities': valid_kps,
        'new_entities_count': new_kps,
        'invalid_entities_count': invalid_kps,
        'coverage_rate': coverage_rate,
        'coverage_percentage': coverage_rate * 100,
        'new_entities': new_entities,
        'invalid_entities': invalid_entities,
        'target_coverage': 90.0,
        'meets_target': coverage_rate >= 0.90
    }


def main():
    parser = argparse.ArgumentParser(
        description='Fix false NEW_ entities in extraction results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Input extraction results JSON')
    parser.add_argument('--vocabulary', type=str, required=True,
                        help='Vocabulary JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output corrected results JSON')
    parser.add_argument('--report', type=str, default=None,
                        help='Optional correction report file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("="*70)
        print("FALSE NEW_ ENTITY CORRECTION")
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
        print(f"  ✓ Loaded {len(vocabulary)} relations, {vocab_size} standard entities")
        print()

    # Original stats
    if verbose and 'vocabulary_stats' in data:
        orig_stats = data['vocabulary_stats']
        print("Original Statistics:")
        print(f"  Total KPs: {orig_stats['total_knowledge_points']}")
        print(f"  Valid: {orig_stats['valid_entities']} ({orig_stats['coverage_percentage']:.2f}%)")
        print(f"  NEW_: {orig_stats['new_entities_count']}")
        print(f"  Invalid: {orig_stats['invalid_entities_count']}")
        print()

    # Fix false NEW_ entities
    if verbose:
        print("Correcting false NEW_ entities...")

    corrected_results, correction_stats = fix_false_new_entities(
        data.get('results', []),
        vocabulary,
        verbose=verbose
    )

    # Recalculate coverage stats
    if verbose:
        print("\nRecalculating coverage statistics...")

    new_coverage_stats = calculate_coverage_stats(corrected_results, vocabulary)

    if verbose:
        print()
        print("Corrected Statistics:")
        print(f"  Total KPs: {new_coverage_stats['total_knowledge_points']}")
        print(f"  Valid: {new_coverage_stats['valid_entities']} ({new_coverage_stats['coverage_percentage']:.2f}%)")
        print(f"  NEW_: {new_coverage_stats['new_entities_count']}")
        print(f"  Invalid: {new_coverage_stats['invalid_entities_count']}")
        print()

        if 'vocabulary_stats' in data:
            orig_cov = data['vocabulary_stats']['coverage_percentage']
            new_cov = new_coverage_stats['coverage_percentage']
            improvement = new_cov - orig_cov
            print(f"Coverage Improvement: {orig_cov:.2f}% → {new_cov:.2f}% (+{improvement:.2f}%)")
            print()

    # Save corrected results
    output_data = data.copy()
    output_data['results'] = corrected_results
    output_data['vocabulary_stats'] = new_coverage_stats
    output_data['correction_applied'] = True
    output_data['correction_stats'] = correction_stats

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Saving corrected results to: {output_file}")

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print(f"  ✓ Saved {len(corrected_results)} results")

    # Save correction report if requested
    if args.report:
        report_file = Path(args.report)
        report_file.parent.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\nSaving correction report to: {report_file}")

        with open(report_file, 'w') as f:
            json.dump(correction_stats, f, indent=2)

        if verbose:
            print(f"  ✓ Saved correction details")

    # Summary
    if verbose:
        print()
        print("="*70)
        print("SUMMARY")
        print("="*70)
        print(f"✓ Corrected {correction_stats['corrected_false_new']} false NEW_ entities")
        print(f"✓ Coverage improved from {data['vocabulary_stats']['coverage_percentage']:.2f}% to {new_coverage_stats['coverage_percentage']:.2f}%")
        print(f"✓ Output saved to: {output_file}")
        print("="*70)

    return 0


if __name__ == '__main__':
    exit(main())
