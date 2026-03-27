#!/usr/bin/env python3
"""
Create final standardized entity vocabulary for Video Games Phase 3 extraction.

Different standardization strategies for different relations:
- Strict: Keep canonical forms (e.g., has_visual_style, shows_perspective)
- Moderate: Group similar entities (e.g., colors, environments)
- Flexible: Keep diverse entities but normalize format (e.g., characters, text)
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import re


def normalize_entity(entity: str) -> str:
    """Basic normalization: lowercase, underscore-separated."""
    entity = entity.lower().strip()
    entity = re.sub(r'\s+', '_', entity)
    entity = re.sub(r'_+', '_', entity)
    entity = entity.strip('_')
    return entity


def merge_similar_entities(entities_with_counts: list[tuple[str, int]],
                           similarity_patterns: list[tuple[str, list[str]]]) -> dict:
    """
    Merge entities based on similarity patterns.

    Args:
        entities_with_counts: List of (entity, count) tuples
        similarity_patterns: List of (canonical, [pattern1, pattern2, ...]) tuples

    Returns:
        Mapping from original entity to canonical form
    """
    mapping = {}

    for entity, count in entities_with_counts:
        normalized = normalize_entity(entity)
        matched = False

        for canonical, patterns in similarity_patterns:
            for pattern in patterns:
                if re.search(pattern, normalized):
                    mapping[entity] = canonical
                    matched = True
                    break
            if matched:
                break

        if not matched:
            mapping[entity] = entity  # Keep original if no match

    return mapping


def create_visual_style_vocabulary(entities_with_counts: list) -> dict:
    """Standardize visual style entities - STRICT."""

    # Define canonical styles
    canonical_styles = {
        'realistic_3d': ['realistic.*3d', '3d.*realistic', 'realistic_graphics'],
        '3d': ['^3d$', '^3d_graphics$'],
        'cartoon': ['cartoon'],
        'anime': ['anime'],
        'pixel_art': ['pixel.*art', '8.*bit', '16.*bit'],
        'realistic': ['^realistic$', 'photorealistic'],
        'stylized': ['stylized'],
        'abstract': ['abstract'],
        'cel_shaded': ['cel.*shad'],
        'low_poly': ['low.*poly'],
        'hand_drawn': ['hand.*drawn', 'illustrated']
    }

    return merge_similar_entities(entities_with_counts, list(canonical_styles.items()))


def create_perspective_vocabulary(entities_with_counts: list) -> dict:
    """Standardize perspective entities - STRICT."""

    canonical_perspectives = {
        'first_person': ['first.*person', 'fps_view'],
        'third_person': ['third.*person'],
        'top_down': ['top.*down'],
        'side_scrolling': ['side.*scroll'],
        'isometric': ['isometric'],
        'overhead': ['overhead'],
        '2.5d': ['2.5d', '2_5d']
    }

    return merge_similar_entities(entities_with_counts, list(canonical_perspectives.items()))


def create_graphics_quality_vocabulary(entities_with_counts: list) -> dict:
    """Standardize graphics quality entities - MODERATE."""

    canonical_quality = {
        'high_fidelity': ['high.*fidelity', 'high.*quality', 'high.*definition'],
        'stylized': ['stylized'],
        'low_fidelity': ['low.*fidelity', 'low.*quality'],
        'retro': ['retro']
    }

    return merge_similar_entities(entities_with_counts, list(canonical_quality.items()))


def create_color_palette_vocabulary(entities_with_counts: list) -> dict:
    """Standardize color palette entities - MODERATE."""

    # For colors, normalize order (alphabetical) and format
    mapping = {}

    for entity, count in entities_with_counts:
        normalized = normalize_entity(entity)

        # Extract color words
        color_words = re.findall(r'\b(?:red|blue|green|yellow|orange|purple|pink|'
                                  r'black|white|gray|grey|brown|cyan|magenta|'
                                  r'dark|light|bright|vibrant|muted|neon|pastel)\b',
                                  normalized)

        if len(color_words) >= 2:
            # Sort colors alphabetically (except adjectives)
            adjectives = {'dark', 'light', 'bright', 'vibrant', 'muted', 'neon', 'pastel'}
            colors = [w for w in color_words if w not in adjectives]
            modifiers = [w for w in color_words if w in adjectives]

            if colors:
                colors_sorted = sorted(set(colors))
                if modifiers:
                    canonical = f"{'_'.join(modifiers)}_{'_and_'.join(colors_sorted)}"
                else:
                    canonical = '_and_'.join(colors_sorted)
                mapping[entity] = canonical
            else:
                mapping[entity] = normalized
        else:
            # Single color or descriptive phrase
            mapping[entity] = normalized

    return mapping


def create_platform_vocabulary(entities_with_counts: list) -> dict:
    """Standardize platform entities - STRICT."""

    canonical_platforms = {
        'playstation': ['^playstation$', '^ps$'],
        'playstation_2': ['playstation.*2', 'ps2'],
        'playstation_3': ['playstation.*3', 'ps3'],
        'playstation_4': ['playstation.*4', 'ps4'],
        'playstation_5': ['playstation.*5', 'ps5'],
        'xbox': ['^xbox$'],
        'xbox_360': ['xbox.*360'],
        'xbox_one': ['xbox.*one'],
        'nintendo_switch': ['nintendo.*switch', '^switch$'],
        'nintendo_wii': ['nintendo.*wii', '^wii$'],
        'nintendo_ds': ['nintendo.*ds', '^ds$'],
        'game_boy': ['game.*boy'],
        'pc': ['^pc$'],
        'mobile': ['mobile', 'ios', 'android']
    }

    return merge_similar_entities(entities_with_counts, list(canonical_platforms.items()))


def standardize_all_relations(phase2_results: dict) -> dict:
    """Apply appropriate standardization to each relation."""

    # Collect entities by relation
    relation_entities = defaultdict(list)

    for result in phase2_results['results']:
        if result['status'] != 'success':
            continue

        for kp in result.get('knowledge_points', []):
            relation = kp['relation']
            entity = kp['entity']
            relation_entities[relation].append(entity)

    # Fix typo: merge show_platform into shows_platform
    if 'show_platform' in relation_entities:
        relation_entities['shows_platform'].extend(relation_entities.pop('show_platform'))

    # Count entities per relation
    relation_entity_counts = {}
    for relation, entities in relation_entities.items():
        counter = Counter(entities)
        relation_entity_counts[relation] = counter.most_common()

    # Apply standardization strategies
    standardized = {}

    # STRICT standardization
    if 'has_visual_style' in relation_entity_counts:
        standardized['has_visual_style'] = create_visual_style_vocabulary(
            relation_entity_counts['has_visual_style']
        )

    if 'shows_perspective' in relation_entity_counts:
        standardized['shows_perspective'] = create_perspective_vocabulary(
            relation_entity_counts['shows_perspective']
        )

    if 'shows_platform' in relation_entity_counts:
        standardized['shows_platform'] = create_platform_vocabulary(
            relation_entity_counts['shows_platform']
        )

    if 'has_graphics_quality' in relation_entity_counts:
        standardized['has_graphics_quality'] = create_graphics_quality_vocabulary(
            relation_entity_counts['has_graphics_quality']
        )

    # MODERATE standardization
    if 'has_color_palette' in relation_entity_counts:
        standardized['has_color_palette'] = create_color_palette_vocabulary(
            relation_entity_counts['has_color_palette']
        )

    # FLEXIBLE standardization (just normalize format)
    flexible_relations = [
        'features_character', 'features_creature', 'features_vehicle', 'features_weapon',
        'set_in_environment', 'has_atmosphere', 'has_genre_indicator',
        'has_ui_elements', 'has_text_element', 'has_additional_property'
    ]

    for relation in flexible_relations:
        if relation in relation_entity_counts:
            # Just normalize format, keep diversity
            mapping = {}
            for entity, count in relation_entity_counts[relation]:
                mapping[entity] = normalize_entity(entity)
            standardized[relation] = mapping

    return standardized, relation_entity_counts


def generate_final_vocabulary(standardized_mappings: dict,
                              entity_counts: dict) -> dict:
    """Generate final vocabulary with statistics."""

    vocabulary = {
        'dataset': 'amazon-videogames',
        'phase': 'phase2_constrained_standardized',
        'vocabulary_version': '3.0',
        'description': 'Final standardized entity vocabulary for Phase 3 extraction',
        'relations': {}
    }

    for relation in sorted(standardized_mappings.keys()):
        mapping = standardized_mappings[relation]
        counts = entity_counts.get(relation, [])

        # Apply mapping to counts
        canonical_counts = Counter()
        for entity, count in counts:
            canonical = mapping.get(entity, normalize_entity(entity))
            canonical_counts[canonical] += count

        # Build entity list
        entities = []
        for canonical, count in canonical_counts.most_common():
            entities.append({
                'entity': canonical,
                'count': count
            })

        vocabulary['relations'][relation] = {
            'total_entities': sum(count for _, count in counts),
            'unique_original': len(counts),
            'unique_standardized': len(canonical_counts),
            'entities': entities
        }

    return vocabulary


def print_vocabulary_summary(vocabulary: dict):
    """Print summary of vocabulary."""

    print("\n" + "="*70)
    print("Final Vocabulary Summary")
    print("="*70)
    print()

    for relation, data in sorted(vocabulary['relations'].items()):
        reduction = (1 - data['unique_standardized'] / data['unique_original']) * 100
        print(f"{relation}:")
        print(f"  Original unique: {data['unique_original']}")
        print(f"  Standardized unique: {data['unique_standardized']}")
        print(f"  Reduction: {reduction:.1f}%")
        print(f"  Total occurrences: {data['total_entities']}")
        print(f"  Top 5 entities:")
        for i, entity_data in enumerate(data['entities'][:5], 1):
            print(f"    {i}. {entity_data['entity']:<40} ({entity_data['count']:>4} occurrences)")
        print()


def main():
    # Load Phase 2 results
    phase2_file = Path("results/videogames/phase2_constrained_5pct.json")
    output_file = Path("results/videogames/entity_vocabulary_final.json")

    print("="*70)
    print("Creating Final Entity Vocabulary")
    print("="*70)
    print()

    print(f"Loading Phase 2 results from: {phase2_file}")
    with open(phase2_file, 'r') as f:
        phase2_data = json.load(f)
    print(f"  ✓ Loaded {len(phase2_data['results'])} results")
    print()

    # Standardize
    print("Applying standardization strategies...")
    standardized_mappings, entity_counts = standardize_all_relations(phase2_data)
    print(f"  ✓ Standardized {len(standardized_mappings)} relations")
    print()

    # Generate vocabulary
    print("Generating final vocabulary...")
    vocabulary = generate_final_vocabulary(standardized_mappings, entity_counts)
    print(f"  ✓ Generated vocabulary")
    print()

    # Print summary
    print_vocabulary_summary(vocabulary)

    # Save
    with open(output_file, 'w') as f:
        json.dump(vocabulary, f, indent=2)

    print("="*70)
    print(f"Final vocabulary saved to: {output_file}")
    print("="*70)


if __name__ == '__main__':
    main()
