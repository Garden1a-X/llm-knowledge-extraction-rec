#!/usr/bin/env python3
"""
Build complete Beauty vocabulary from Phase 4 extraction results.

Extracts all entities, applies standardization rules, and generates
vocabulary file in ML-1M format with standard_entities lists.

Usage:
    python scripts/build_beauty_vocabulary.py \
        --phase4_results <phase4_json> \
        --output results/beauty_vocabulary_v2.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import re
from typing import Dict, List, Set
from collections import defaultdict, Counter


# 13 standardized relations
STANDARD_RELATIONS = [
    "product_type",
    "color",
    "packaging",
    "material",
    "texture",
    "design_style",
    "size",
    "shape",
    "label_style",
    "target_audience",
    "product_feature",
    "scent",
    "brand_element"
]


def standardize_entity(entity: str, relation: str) -> str:
    """
    Standardize entity name.

    Rules:
    1. Convert to lowercase
    2. Replace spaces with underscores
    3. Remove special characters except underscores and numbers
    4. Trim excessive underscores
    5. Apply relation-specific mappings
    """
    # Convert to lowercase
    entity = entity.lower()

    # Replace spaces/hyphens with underscores
    entity = entity.replace(' ', '_').replace('-', '_')

    # Remove special characters (keep letters, numbers, underscores)
    entity = re.sub(r'[^a-z0-9_]', '', entity)

    # Remove duplicate underscores
    entity = re.sub(r'_+', '_', entity)

    # Trim leading/trailing underscores
    entity = entity.strip('_')

    # Apply relation-specific mappings
    entity = apply_entity_mappings(entity, relation)

    return entity


def apply_entity_mappings(entity: str, relation: str) -> str:
    """Apply relation-specific entity mappings."""

    # Target audience mappings
    if relation == 'target_audience':
        mappings = {
            'adults': 'adult',
            'men': 'men',
            'women': 'women',
            'children': 'children',
            'child': 'children',
            'general': 'adult',
            'general_consumers': 'adult',
            'general_use': 'adult',
            'beauty_enthusiasts': 'adult',
            'makeup_users': 'adult',
            'makeup_enthusiasts': 'adult',
            'skincare_users': 'adult',
            'nail_care_enthusiasts': 'adult',
            'dental_hygiene_users': 'adult',
            'dental_health_conscious_users': 'adult',
            'environmentally_conscious_users': 'adult',
            'eco_conscious_consumers': 'adult',
            'health_conscious_consumers': 'adult',
            'incense_users': 'adult',
            'young_adults': 'adult',
            'men_and_women': 'adult',
            'mens_grooming': 'men',
            'male': 'men',
            'all_skin_types': 'adult',
            'sensitive_skin': 'adult',
            'all_hair_types': 'adult'
        }
        return mappings.get(entity, entity)

    # Color mappings
    if relation == 'color':
        mappings = {
            'creamy_natural': 'cream',
            'light_tones': 'light',
            'warm_tones': 'warm',
            'cool_tones': 'cool',
            'dark_tones': 'dark'
        }
        return mappings.get(entity, entity)

    # Size normalization
    if relation == 'size':
        # Normalize ounce formats
        entity = entity.replace('_ounce', '_oz')
        entity = entity.replace('ounce', 'oz')
        entity = entity.replace('fluid_', '')
        entity = entity.replace('fluidoz', '_oz')

        # Normalize inches
        if entity.endswith('_in') and not entity.endswith('_inch'):
            entity = entity.replace('_in', '_inch')

        # Standardize common sizes
        size_map = {
            'standard_compact_size': 'standard',
            'small': 'small',
            'medium': 'medium',
            'large': 'large',
            'compact': 'compact',
            'standard': 'standard'
        }

        return size_map.get(entity, entity)

    # Packaging mappings
    if relation == 'packaging':
        mappings = {
            'plastic_bottle': 'bottle',
            'pump_bottle': 'bottle',
            'clear_glass_bottle': 'bottle',
            'boxed': 'box',
            'wrapped': 'wrapper',
            'paper_bands': 'wrapper',
            'plastic_case': 'case',
            'electronic_device': 'standalone',
            'none': 'standalone'
        }
        return mappings.get(entity, entity)

    # Material mappings
    if relation == 'material':
        mappings = {
            'organic_cotton': 'cotton',
            'plant_based_ingredients': 'natural',
            'vegetable': 'natural'
        }
        return mappings.get(entity, entity)

    # Texture mappings
    if relation == 'texture':
        mappings = {
            'soft_rough': 'textured',
            'swirled': 'textured'
        }
        return mappings.get(entity, entity)

    # Design style mappings
    if relation == 'design_style':
        mappings = {
            'minimalist': 'modern',
            'minimalistic': 'modern',
            'simple': 'modern',
            'sleek': 'modern',
            'modern_minimalist': 'modern',
            'modern_design': 'modern',
            'rustic': 'natural',
            'animal_print': 'decorative',
            'floral_print': 'decorative',
            'traditional': 'classic',
            'vintage': 'classic'
        }
        return mappings.get(entity, entity)

    # Label style mappings
    if relation == 'label_style':
        mappings = {
            'bold_typography': 'bold_text',
            'printed_text': 'printed',
            'modern_typography': 'modern',
            'clear_text': 'printed',
            'minimalist': 'minimalistic',
            'minimal': 'minimalistic',
            'informative': 'detailed',
            'multi_colored': 'colorful',
            'colorful_illustrative': 'colorful',
            'earthy_tone_design': 'natural',
            'graphic_elements': 'graphic',
            'ornate': 'decorative',
            'embossed_text': 'embossed',
            'modern_design': 'modern',
            'bold_text_on_dark_background': 'bold_text',
            'logo_on_handle': 'branded',
            'logo_on_packaging': 'branded',
            'logo_on_bottle': 'branded',
            'prominent_brand_name': 'branded',
            'printed_logo': 'branded',
            'logo_oral_b': 'branded',
            'brand_logo': 'branded',
            'logo': 'branded'
        }
        return mappings.get(entity, entity)

    # Shape mappings
    if relation == 'shape':
        mappings = {
            'elongated': 'cylindrical',
            'round': 'circular',
            'upright': 'vertical',
            'tapered': 'curved',
            'angular': 'geometric',
            'hexagonal': 'geometric',
            'flat': 'flat'
        }
        return mappings.get(entity, entity)

    # Brand element mappings
    if relation == 'brand_element':
        mappings = {
            'logo_visible': 'logo',
            'brand_logo': 'logo',
            'logo_on_front': 'logo',
            'colgate_logo': 'logo',
            'pantene_logo': 'logo',
            'oral_b': 'logo'
        }
        return mappings.get(entity, entity)

    return entity


def extract_entities_from_phase4(phase4_data: Dict) -> Dict[str, List[str]]:
    """
    Extract all entities from Phase 4 results, grouped by relation.

    Returns:
        Dict mapping relation -> list of raw entities
    """
    entities_by_relation = defaultdict(list)

    for result in phase4_data.get('results', []):
        if result.get('status') != 'success':
            continue

        for kp in result.get('knowledge_points', []):
            relation = kp['relation']
            entity = kp['entity']

            # Only keep valid relations
            if relation in STANDARD_RELATIONS:
                entities_by_relation[relation].append(entity)

    return dict(entities_by_relation)


def build_vocabulary(entities_by_relation: Dict[str, List[str]]) -> Dict:
    """
    Build vocabulary with standardized entities.

    Returns:
        Vocabulary dict in ML-1M format
    """
    vocabulary = {}

    for relation in STANDARD_RELATIONS:
        raw_entities = entities_by_relation.get(relation, [])

        # Standardize all entities
        standardized = []
        for entity in raw_entities:
            std_entity = standardize_entity(entity, relation)
            if std_entity:  # Skip empty
                standardized.append(std_entity)

        # Get unique sorted list
        unique_entities = sorted(set(standardized))

        # Get entity frequency
        entity_counts = Counter(standardized)

        vocabulary[relation] = {
            'standard_entities': unique_entities,
            'num_entities': len(unique_entities),
            'total_occurrences': len(standardized),
            'top_entities': [
                {'entity': e, 'count': entity_counts[e]}
                for e in sorted(entity_counts.keys(), key=lambda x: entity_counts[x], reverse=True)[:10]
            ]
        }

    return vocabulary


def main():
    parser = argparse.ArgumentParser(
        description='Build complete Beauty vocabulary from Phase 4 results'
    )

    parser.add_argument('--phase4_results', type=str, required=True,
                       help='Phase 4 extraction results JSON')
    parser.add_argument('--output', type=str,
                       default='results/beauty_vocabulary_v2.json',
                       help='Output vocabulary file')

    args = parser.parse_args()

    print("="*70)
    print("BUILD BEAUTY VOCABULARY FROM PHASE 4")
    print("="*70)
    print()

    # Load Phase 4 results
    phase4_file = Path(args.phase4_results)
    if not phase4_file.exists():
        print(f"Error: Phase 4 file not found: {phase4_file}")
        return 1

    print(f"Loading Phase 4 results: {phase4_file}")
    with open(phase4_file, 'r') as f:
        phase4_data = json.load(f)

    total_products = phase4_data.get('total_products', 0)
    successful = phase4_data.get('successful', 0)
    print(f"  Products: {total_products}, Successful: {successful}")
    print()

    # Extract entities
    print("Extracting entities from Phase 4 results...")
    entities_by_relation = extract_entities_from_phase4(phase4_data)

    total_raw = sum(len(entities) for entities in entities_by_relation.values())
    print(f"  Extracted {total_raw} raw entity occurrences")
    print()

    # Build vocabulary
    print("Standardizing entities and building vocabulary...")
    vocabulary = build_vocabulary(entities_by_relation)

    total_unique = sum(v['num_entities'] for v in vocabulary.values())
    print(f"  ✓ {total_unique} unique standardized entities")
    print()

    # Show statistics per relation
    print("Entities per relation:")
    for relation in STANDARD_RELATIONS:
        info = vocabulary.get(relation, {})
        num = info.get('num_entities', 0)
        occurrences = info.get('total_occurrences', 0)
        print(f"  {relation:20s}: {num:3d} unique ({occurrences:3d} occurrences)")
    print()

    # Build output
    output_data = {
        'dataset': 'amazon-beauty',
        'phase': 'vocabulary_v2',
        'source': 'phase4_extraction_58_products',
        'vocabulary': vocabulary,
        'statistics': {
            'num_relations': len(STANDARD_RELATIONS),
            'num_unique_entities': total_unique,
            'num_products': successful,
            'total_knowledge_points': total_raw
        }
    }

    # Save
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving vocabulary to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  ✓ Saved")
    print()

    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ {len(STANDARD_RELATIONS)} relations")
    print(f"✓ {total_unique} unique standardized entities")
    print(f"✓ Vocabulary saved to: {output_file}")
    print("="*70)

    return 0


if __name__ == '__main__':
    exit(main())
