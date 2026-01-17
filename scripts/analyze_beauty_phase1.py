#!/usr/bin/env python3
"""
Analyze Beauty Phase 1 extraction results and prepare vocabulary standardization.
Similar to ML-1M Phase 2, but simpler due to small dataset size (58 items).
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set


def load_phase1_results(results_file: Path) -> Dict:
    """Load Phase 1 extraction results."""
    with open(results_file, 'r') as f:
        return json.load(f)


def analyze_relations(results: Dict) -> Dict[str, int]:
    """Count all relations used."""
    relation_counts = Counter()

    for result in results['results']:
        if result['status'] == 'success':
            for kp in result['knowledge_points']:
                relation_counts[kp['relation']] += 1

    return dict(relation_counts)


def analyze_entities(results: Dict) -> Dict[str, List[str]]:
    """Group entities by relation."""
    relation_entities = defaultdict(set)

    for result in results['results']:
        if result['status'] == 'success':
            for kp in result['knowledge_points']:
                relation_entities[kp['relation']].add(kp['entity'])

    # Convert sets to sorted lists
    return {rel: sorted(list(ents)) for rel, ents in relation_entities.items()}


def suggest_relation_standardization(relation_counts: Dict[str, int]) -> Dict[str, List[str]]:
    """
    Suggest relation clustering/standardization.

    Group similar relations together.
    """
    # Manual clustering based on semantic similarity
    clusters = {
        # Color-related
        'color': ['dominant_color', 'color_scheme', 'accent_color', 'secondary_color',
                  'primary_color', 'label_color', 'text_color'],

        # Product type/category
        'product_category': ['product_type', 'product_category', 'product_general_function'],

        # Packaging
        'packaging_style': ['packaging_style', 'packaging_material', 'packaging_shape'],

        # Design/Aesthetics
        'design_style': ['design_style', 'design_aesthetic', 'branding_style', 'brand_identity',
                        'design_elements', 'graphic_elements', 'visual_theme', 'design_motif'],

        # Material/Texture
        'material': ['material', 'material_type', 'texture', 'finish', 'finish_type'],

        # Size/Shape
        'physical_properties': ['size', 'shape', 'size_category', 'product_size', 'height'],

        # Target audience
        'target_audience': ['target_audience', 'intended_user', 'user_audience'],

        # Text/Label
        'label_design': ['label_style', 'label_design', 'text_style', 'font_style', 'text_type'],

        # Features/Benefits
        'product_feature': ['product_feature', 'functional_feature', 'functionality',
                           'feature_highlight', 'benefit'],

        # Brand elements
        'branding_elements': ['branding_element', 'branding_elements', 'logo', 'brand_visibility'],
    }

    # Find which cluster each relation belongs to
    relation_to_cluster = {}
    for cluster_name, relations in clusters.items():
        for rel in relations:
            if rel in relation_counts:
                relation_to_cluster[rel] = cluster_name

    # Group relations by cluster
    clustered = defaultdict(list)
    for rel in relation_counts:
        cluster = relation_to_cluster.get(rel, 'other')
        clustered[cluster].append(rel)

    return dict(clustered)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Beauty Phase 1 results and suggest vocabulary standardization'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Path to Phase 1 results JSON')
    parser.add_argument('--output', type=str, default='results/beauty_phase1_analysis.json',
                       help='Path to save analysis output')

    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Beauty Phase 1 Vocabulary Analysis")
    print("="*70)
    print(f"\nInput: {input_file}")
    print(f"Output: {output_file}")
    print()

    # Load results
    print("Loading Phase 1 results...")
    results = load_phase1_results(input_file)
    print(f"  ✓ Loaded {results['successful']} successful extractions")

    # Analyze relations
    print("\nAnalyzing relations...")
    relation_counts = analyze_relations(results)
    print(f"  ✓ Found {len(relation_counts)} unique relations")
    print(f"\nTop 10 most frequent relations:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {rel}: {count}")

    # Analyze entities
    print("\nAnalyzing entities...")
    relation_entities = analyze_entities(results)
    total_entities = sum(len(ents) for ents in relation_entities.values())
    print(f"  ✓ Found {total_entities} unique entities across all relations")

    # Suggest clustering
    print("\nSuggesting relation standardization...")
    relation_clusters = suggest_relation_standardization(relation_counts)
    print(f"  ✓ Grouped into {len(relation_clusters)} clusters")
    print("\nRelation clusters:")
    for cluster, relations in sorted(relation_clusters.items()):
        print(f"  {cluster}:")
        for rel in sorted(relations):
            print(f"    - {rel} ({relation_counts[rel]} uses)")

    # Save analysis
    analysis = {
        'dataset': 'beauty',
        'phase': 'phase1_analysis',
        'total_products': results['total_products'],
        'successful': results['successful'],
        'statistics': {
            'total_unique_relations': len(relation_counts),
            'total_unique_entities': total_entities,
            'total_knowledge_points': sum(relation_counts.values())
        },
        'relation_counts': relation_counts,
        'relation_clusters': relation_clusters,
        'relation_entities': relation_entities,
        'suggestions': {
            'next_steps': [
                '1. Review relation clusters and merge similar relations',
                '2. Standardize entity names (e.g., "mens_grooming" -> "men", "adults" -> "adult")',
                '3. Create final vocabulary with standardized relations and entities',
                '4. Re-run Phase 4 extraction with standardized vocabulary'
            ]
        }
    }

    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n✓ Analysis saved to {output_file}")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review the analysis JSON file")
    print("  2. Decide on final relation names (or use suggested clusters)")
    print("  3. Standardize entity names")
    print("  4. Create vocabulary file for Phase 4")


if __name__ == '__main__':
    main()
