#!/usr/bin/env python3
"""
Analyze NEW_ entities from Phase 3 validation results.

This script:
1. Extracts all NEW_ entities from phase3_20percent_validation.json
2. Groups by relation
3. Computes frequency statistics
4. Identifies high/medium/low frequency NEW_ entities
5. Generates analysis report for vocabulary v2 expansion

Usage:
    python scripts/analyze_new_entities.py \
        --input results/phase3_20percent_validation.json \
        --output results/new_entities_analysis.json \
        --report results/new_entities_report.txt
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple


def load_phase3_results(filepath: Path) -> Dict:
    """Load Phase 3 validation results."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_new_entities(phase3_data: Dict) -> List[Dict]:
    """
    Extract all NEW_ entities from validation results.

    Returns:
        List of dicts with 'relation', 'entity', 'recbole_id'
    """
    return phase3_data.get('vocabulary_stats', {}).get('new_entities', [])


def analyze_by_relation(new_entities: List[Dict]) -> Dict[str, Counter]:
    """
    Group NEW_ entities by relation and count frequencies.

    Args:
        new_entities: List of NEW_ entity records

    Returns:
        Dict mapping relation -> Counter of entity frequencies
    """
    relation_entities = defaultdict(list)

    # Group by relation (ignore recbole_id)
    for record in new_entities:
        relation = record['relation']
        entity = record['entity']
        relation_entities[relation].append(entity)

    # Count frequencies for each relation
    relation_counters = {}
    for relation, entities in relation_entities.items():
        relation_counters[relation] = Counter(entities)

    return relation_counters


def categorize_by_frequency(relation_counters: Dict[str, Counter]) -> Dict:
    """
    Categorize NEW_ entities by frequency thresholds.

    Thresholds:
    - High: ≥10 occurrences (likely real needs, direct add to v2)
    - Medium: 3-9 occurrences (review & possibly merge)
    - Low: 1-2 occurrences (possibly noise, careful review)

    Returns:
        Dict with categorized entities per relation
    """
    categorized = {}

    for relation, counter in relation_counters.items():
        high_freq = []
        medium_freq = []
        low_freq = []

        for entity, count in counter.most_common():
            if count >= 10:
                high_freq.append((entity, count))
            elif count >= 3:
                medium_freq.append((entity, count))
            else:
                low_freq.append((entity, count))

        categorized[relation] = {
            'high_frequency': high_freq,
            'medium_frequency': medium_freq,
            'low_frequency': low_freq,
            'total_unique': len(counter),
            'total_occurrences': sum(counter.values())
        }

    return categorized


def generate_statistics(
    new_entities: List[Dict],
    relation_counters: Dict[str, Counter],
    categorized: Dict
) -> Dict:
    """Generate overall statistics."""

    total_new_occurrences = len(new_entities)
    total_unique_new = sum(len(counter) for counter in relation_counters.values())

    # Count by frequency category
    high_count = sum(len(cat['high_frequency']) for cat in categorized.values())
    medium_count = sum(len(cat['medium_frequency']) for cat in categorized.values())
    low_count = sum(len(cat['low_frequency']) for cat in categorized.values())

    # Relation with most NEW_ entities
    relation_totals = {
        rel: sum(counter.values())
        for rel, counter in relation_counters.items()
    }
    top_relation = max(relation_totals.items(), key=lambda x: x[1])

    return {
        'total_new_occurrences': total_new_occurrences,
        'total_unique_new_entities': total_unique_new,
        'average_frequency': total_new_occurrences / total_unique_new if total_unique_new > 0 else 0,
        'high_frequency_count': high_count,
        'medium_frequency_count': medium_count,
        'low_frequency_count': low_count,
        'relations_with_new_entities': len(relation_counters),
        'top_relation': {
            'name': top_relation[0],
            'new_entity_occurrences': top_relation[1]
        }
    }


def generate_report(
    stats: Dict,
    categorized: Dict,
    relation_counters: Dict[str, Counter]
) -> str:
    """Generate human-readable analysis report."""

    lines = []
    lines.append("=" * 80)
    lines.append("NEW ENTITIES ANALYSIS REPORT")
    lines.append("Phase 3 Validation - Vocabulary v1 Coverage Gap")
    lines.append("=" * 80)
    lines.append("")

    # Overall statistics
    lines.append("## OVERALL STATISTICS")
    lines.append("")
    lines.append(f"Total NEW_ entity occurrences: {stats['total_new_occurrences']}")
    lines.append(f"Total unique NEW_ entities: {stats['total_unique_new_entities']}")
    lines.append(f"Average frequency: {stats['average_frequency']:.2f}")
    lines.append("")
    lines.append(f"High frequency (≥10): {stats['high_frequency_count']} entities")
    lines.append(f"Medium frequency (3-9): {stats['medium_frequency_count']} entities")
    lines.append(f"Low frequency (1-2): {stats['low_frequency_count']} entities")
    lines.append("")
    lines.append(f"Relations with NEW_ entities: {stats['relations_with_new_entities']}")
    lines.append(f"Top relation: {stats['top_relation']['name']} ({stats['top_relation']['new_entity_occurrences']} occurrences)")
    lines.append("")

    # Frequency distribution recommendation
    lines.append("## EXPANSION RECOMMENDATION")
    lines.append("")
    lines.append("✅ High frequency (≥10): Direct candidates for v2 (likely real needs)")
    lines.append("⚠️  Medium frequency (3-9): Review & consider merging with existing")
    lines.append("🔍 Low frequency (1-2): Careful review (may be noise or edge cases)")
    lines.append("")

    # Per-relation analysis
    lines.append("=" * 80)
    lines.append("## PER-RELATION ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    # Sort relations by total occurrences (descending)
    sorted_relations = sorted(
        categorized.items(),
        key=lambda x: x[1]['total_occurrences'],
        reverse=True
    )

    for relation, data in sorted_relations:
        lines.append(f"### Relation: {relation}")
        lines.append(f"Total unique NEW_ entities: {data['total_unique']}")
        lines.append(f"Total occurrences: {data['total_occurrences']}")
        lines.append("")

        # High frequency
        if data['high_frequency']:
            lines.append(f"✅ High Frequency (≥10) - {len(data['high_frequency'])} entities:")
            for entity, count in data['high_frequency']:
                lines.append(f"   - {entity}: {count} times")
            lines.append("")

        # Medium frequency
        if data['medium_frequency']:
            lines.append(f"⚠️  Medium Frequency (3-9) - {len(data['medium_frequency'])} entities:")
            for entity, count in data['medium_frequency']:
                lines.append(f"   - {entity}: {count} times")
            lines.append("")

        # Low frequency (show top 10 only if too many)
        if data['low_frequency']:
            lines.append(f"🔍 Low Frequency (1-2) - {len(data['low_frequency'])} entities:")
            shown = data['low_frequency'][:10]
            for entity, count in shown:
                lines.append(f"   - {entity}: {count} times")
            if len(data['low_frequency']) > 10:
                lines.append(f"   ... and {len(data['low_frequency']) - 10} more")
            lines.append("")

        lines.append("-" * 80)
        lines.append("")

    # Next steps
    lines.append("=" * 80)
    lines.append("## NEXT STEPS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. Review high-frequency NEW_ entities:")
    lines.append("   - Verify they are genuine visual features")
    lines.append("   - Check if they should merge with existing entities")
    lines.append("   - Add confirmed entities to vocabulary v2")
    lines.append("")
    lines.append("2. Cluster medium-frequency entities:")
    lines.append("   - Use embedding similarity (similar to Phase 2b)")
    lines.append("   - Merge semantically similar NEW_ entities")
    lines.append("   - Decide which to add to v2")
    lines.append("")
    lines.append("3. Review low-frequency entities:")
    lines.append("   - Check if they are noise or special cases")
    lines.append("   - Consider dropping 1-occurrence entities")
    lines.append("   - Keep only those with clear visual meaning")
    lines.append("")
    lines.append("4. Expected outcome:")
    lines.append("   - Vocabulary v2 = v1 (165) + selected NEW_ entities")
    lines.append("   - Target: Achieve ≥90% coverage in validation")
    lines.append("")

    return "\n".join(lines)


def save_results(
    categorized: Dict,
    stats: Dict,
    report: str,
    output_json: Path,
    output_report: Path
):
    """Save analysis results to files."""

    # Save JSON
    output_data = {
        'statistics': stats,
        'categorized_by_relation': categorized,
        'timestamp': str(Path(output_json).stat().st_mtime) if output_json.exists() else None
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ JSON analysis saved to: {output_json}")

    # Save report
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ Text report saved to: {output_report}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze NEW_ entities from Phase 3 validation'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default='results/phase3_20percent_validation.json',
        help='Input Phase 3 validation results file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default='results/new_entities_analysis.json',
        help='Output JSON analysis file'
    )
    parser.add_argument(
        '--report',
        type=Path,
        default='results/new_entities_report.txt',
        help='Output text report file'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("NEW ENTITIES ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    print(f"📂 Loading Phase 3 results from: {args.input}")
    phase3_data = load_phase3_results(args.input)

    # Extract NEW_ entities
    new_entities = extract_new_entities(phase3_data)
    print(f"✅ Extracted {len(new_entities)} NEW_ entity occurrences")
    print()

    # Analyze by relation
    print("🔍 Analyzing by relation...")
    relation_counters = analyze_by_relation(new_entities)
    print(f"✅ Found NEW_ entities in {len(relation_counters)} relations")
    print()

    # Categorize by frequency
    print("📊 Categorizing by frequency...")
    categorized = categorize_by_frequency(relation_counters)
    print()

    # Generate statistics
    print("📈 Computing statistics...")
    stats = generate_statistics(new_entities, relation_counters, categorized)
    print()

    # Generate report
    print("📝 Generating report...")
    report = generate_report(stats, categorized, relation_counters)
    print()

    # Save results
    print("💾 Saving results...")
    save_results(categorized, stats, report, args.output, args.report)
    print()

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Total unique NEW_ entities: {stats['total_unique_new_entities']}")
    print(f"  - High frequency (≥10): {stats['high_frequency_count']}")
    print(f"  - Medium frequency (3-9): {stats['medium_frequency_count']}")
    print(f"  - Low frequency (1-2): {stats['low_frequency_count']}")
    print()
    print(f"Top relation: {stats['top_relation']['name']} ({stats['top_relation']['new_entity_occurrences']} occurrences)")
    print()
    print("📄 Check the report for detailed analysis!")
    print()


if __name__ == '__main__':
    main()
