#!/usr/bin/env python3
"""
Analyze Video Games Phase 1 relations and create vocabulary.

Similar to ML-1M's relation analysis, but adapted for video game domain.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import re


def normalize_relation(relation: str) -> str:
    """Normalize relation name for grouping similar relations."""
    # Convert to lowercase
    rel = relation.lower().strip()

    # Remove common prefixes/suffixes
    rel = re.sub(r'^(has_|features_|shows_|includes_|displays_|contains_)', '', rel)
    rel = re.sub(r'(_element|_elements|_detail|_details)$', '', rel)

    return rel


def analyze_relations(phase1_file: Path):
    """Analyze relations from Phase 1 results."""

    with open(phase1_file, 'r') as f:
        data = json.load(f)

    # Extract all relations
    all_relations = []
    relation_examples = defaultdict(list)

    for result in data['results']:
        if result['status'] == 'success':
            for kp in result.get('knowledge_points', []):
                relation = kp.get('relation', '')
                entity = kp.get('entity', '')
                all_relations.append(relation)

                # Store examples
                if len(relation_examples[relation]) < 5:
                    relation_examples[relation].append(entity)

    # Count relations
    relation_counts = Counter(all_relations)

    # Group by normalized form
    normalized_groups = defaultdict(list)
    for relation in relation_counts.keys():
        normalized = normalize_relation(relation)
        normalized_groups[normalized].append(relation)

    # Calculate group counts
    group_counts = {}
    for normalized, variants in normalized_groups.items():
        total_count = sum(relation_counts[v] for v in variants)
        group_counts[normalized] = {
            'count': total_count,
            'variants': variants,
            'examples': relation_examples[variants[0]]  # Use first variant's examples
        }

    # Sort by count
    sorted_groups = sorted(group_counts.items(), key=lambda x: x[1]['count'], reverse=True)

    # Print analysis
    print("="*80)
    print("VIDEO GAMES PHASE 1 RELATION ANALYSIS")
    print("="*80)
    print(f"\nTotal knowledge points: {len(all_relations)}")
    print(f"Unique relations (raw): {len(relation_counts)}")
    print(f"Unique relations (normalized): {len(normalized_groups)}")
    print()

    print("="*80)
    print("TOP RELATION GROUPS (Normalized)")
    print("="*80)
    print()

    for i, (normalized, info) in enumerate(sorted_groups[:30], 1):
        count = info['count']
        percentage = count / len(all_relations) * 100
        variants = info['variants']
        examples = info['examples'][:3]

        print(f"{i}. {normalized}")
        print(f"   Count: {count} ({percentage:.2f}%)")
        print(f"   Variants: {', '.join(variants[:5])}")
        if len(variants) > 5:
            print(f"             ... and {len(variants)-5} more")
        print(f"   Examples: {', '.join(examples)}")
        print()

    return sorted_groups, relation_counts, all_relations


def propose_core_relations(sorted_groups):
    """Propose 14 core relations based on analysis."""

    print("="*80)
    print("PROPOSED CORE RELATIONS (14 + 1 Additional)")
    print("="*80)
    print()

    # Manual curation based on top groups and game-specific needs
    core_relations = [
        {
            'relation': 'has_visual_style',
            'description': 'Visual art style (realistic, cartoon, pixel art, anime, etc.)',
            'normalized_forms': ['visual_style', 'art_style', 'graphics_style', 'style']
        },
        {
            'relation': 'features_character',
            'description': 'Character designs, types, clothing, equipment',
            'normalized_forms': ['character', 'characters']
        },
        {
            'relation': 'set_in_environment',
            'description': 'Environment/setting (urban, fantasy, sci-fi, nature, etc.)',
            'normalized_forms': ['environment', 'setting', 'location', 'scene']
        },
        {
            'relation': 'has_color_palette',
            'description': 'Color scheme and palette',
            'normalized_forms': ['color_palette', 'color', 'colors']
        },
        {
            'relation': 'shows_perspective',
            'description': 'Camera perspective (first-person, third-person, top-down, side-scrolling)',
            'normalized_forms': ['perspective', 'viewpoint', 'view', 'camera_angle']
        },
        {
            'relation': 'has_ui_elements',
            'description': 'User interface elements visible in game',
            'normalized_forms': ['ui', 'interface', 'hud']
        },
        {
            'relation': 'has_atmosphere',
            'description': 'Mood and atmosphere (dark, bright, gritty, whimsical, etc.)',
            'normalized_forms': ['atmosphere', 'mood', 'tone']
        },
        {
            'relation': 'has_graphics_quality',
            'description': 'Graphics fidelity and quality indicators',
            'normalized_forms': ['graphics_quality', 'graphics', 'quality']
        },
        {
            'relation': 'features_vehicle',
            'description': 'Vehicles, mounts, or transportation',
            'normalized_forms': ['vehicle', 'vehicles', 'transportation']
        },
        {
            'relation': 'features_weapon',
            'description': 'Weapons, combat equipment visible',
            'normalized_forms': ['weapon', 'weapons', 'armament', 'combat']
        },
        {
            'relation': 'has_genre_indicator',
            'description': 'Visual indicators of game genre (RPG, FPS, racing, etc.)',
            'normalized_forms': ['genre', 'game_type', 'gameplay_type']
        },
        {
            'relation': 'shows_platform',
            'description': 'Platform or era indicators (console, PC, mobile, retro)',
            'normalized_forms': ['platform', 'platform_era', 'era']
        },
        {
            'relation': 'has_text_element',
            'description': 'Visible text, logos, branding',
            'normalized_forms': ['text', 'logo', 'branding', 'title']
        },
        {
            'relation': 'features_creature',
            'description': 'Non-human creatures, monsters, animals',
            'normalized_forms': ['creature', 'creatures', 'monster', 'animal']
        },
        {
            'relation': 'has_additional_property',
            'description': 'Other visual properties not covered above',
            'normalized_forms': ['additional', 'other', 'property', 'misc']
        }
    ]

    for i, rel in enumerate(core_relations, 1):
        print(f"{i}. {rel['relation']}")
        print(f"   Description: {rel['description']}")
        print(f"   Matches: {', '.join(rel['normalized_forms'])}")
        print()

    return core_relations


def save_relation_vocabulary(core_relations, output_file: Path):
    """Save relation vocabulary to JSON."""

    vocabulary = {
        'domain': 'video_games',
        'version': '1.0',
        'num_relations': len(core_relations),
        'relations': []
    }

    for rel in core_relations:
        vocabulary['relations'].append({
            'relation': rel['relation'],
            'description': rel['description'],
            'normalized_forms': rel['normalized_forms']
        })

    with open(output_file, 'w') as f:
        json.dump(vocabulary, f, indent=2)

    print(f"✓ Saved relation vocabulary to: {output_file}")


def main():
    phase1_file = Path('results/videogames/phase1_5percent_exploration.json')
    output_file = Path('results/videogames/relation_vocabulary_v1.json')

    if not phase1_file.exists():
        print(f"Error: Phase 1 results not found: {phase1_file}")
        return

    # Analyze relations
    sorted_groups, relation_counts, all_relations = analyze_relations(phase1_file)

    # Propose core relations
    core_relations = propose_core_relations(sorted_groups)

    # Save vocabulary
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_relation_vocabulary(core_relations, output_file)

    print()
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Review the proposed 14+1 core relations above")
    print("2. Adjust if needed based on domain knowledge")
    print("3. Use this vocabulary for constrained extraction (Phase 2)")
    print("="*80)


if __name__ == '__main__':
    main()
