#!/usr/bin/env python3
"""
Generate Case Study data for paper Appendix B.8.

This script extracts:
1. Example 1 (ML-1M): A famous movie's extraction results
2. Example 2 (Video Games): A game's extraction results
3. Table 11: Cross-item knowledge sharing statistics
"""

import json
from pathlib import Path
from collections import defaultdict


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_item_mapping(item_file):
    """Load item ID to title mapping."""
    mapping = {}
    with open(item_file, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                item_id = int(parts[0])
                title = parts[1]
                mapping[item_id] = title
    return mapping


def load_kg(kg_file):
    """Load knowledge graph and build entity->items mapping."""
    entity_items = defaultdict(list)  # entity -> [(item_id, relation), ...]
    item_entities = defaultdict(list)  # item_id -> [(relation, entity), ...]

    with open(kg_file, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                item_id = int(parts[0])
                relation = parts[1]
                entity = parts[2]
                entity_items[entity].append((item_id, relation))
                item_entities[item_id].append((relation, entity))

    return entity_items, item_entities


def find_movie_by_keyword(item_mapping, keyword):
    """Find movies containing keyword."""
    results = []
    for item_id, title in item_mapping.items():
        if keyword.lower() in title.lower():
            results.append((item_id, title))
    return results


def get_extraction_result(extraction_data, movie_id, id_field='original_movie_id'):
    """Get extraction result for a specific movie."""
    for result in extraction_data.get('results', []):
        if result.get(id_field) == movie_id:
            return result
    return None


def main():
    print("=" * 80)
    print("CASE STUDY DATA FOR PAPER APPENDIX B.8")
    print("=" * 80)

    # === ML-1M ===
    print("\n" + "=" * 80)
    print("EXAMPLE 1: ML-1M Dataset")
    print("=" * 80)

    ml1m_item_file = Path('data/recbole/ml-1m/ml-1m.item')
    ml1m_kg_file = Path('data/recbole/ml-1m/ml-1m.item.kg')
    ml1m_extraction_file = Path('results/phase4_full_extraction_filtered.json')

    if ml1m_item_file.exists():
        item_mapping = load_item_mapping(ml1m_item_file)

        # Find Star Wars
        star_wars = find_movie_by_keyword(item_mapping, 'Star Wars')
        print("\nStar Wars movies found:")
        for item_id, title in star_wars:
            print(f"  ID {item_id}: {title}")

        # Find The Matrix
        matrix = find_movie_by_keyword(item_mapping, 'Matrix')
        print("\nMatrix movies found:")
        for item_id, title in matrix:
            print(f"  ID {item_id}: {title}")

    if ml1m_kg_file.exists():
        entity_items, item_entities = load_kg(ml1m_kg_file)

        # Star Wars: Episode IV - A New Hope (1977) has item_id 242
        target_movie_id = 242
        if target_movie_id in item_mapping:
            print(f"\n--- Selected Movie: {item_mapping[target_movie_id]} ---")
            print(f"Item ID: {target_movie_id}")

            if target_movie_id in item_entities:
                print("\nStage 2/3 Standardized Knowledge Points:")
                for i, (rel, ent) in enumerate(item_entities[target_movie_id], 1):
                    print(f"  {i}. ({rel}, {ent})")

                # Knowledge Sharing
                print("\nKnowledge Sharing:")
                for rel, ent in item_entities[target_movie_id][:3]:
                    shared_items = entity_items.get(ent, [])
                    if len(shared_items) > 1:
                        other_items = [(iid, item_mapping.get(iid, f'ID:{iid}'))
                                      for iid, r in shared_items if iid != target_movie_id][:5]
                        print(f"  - '{ent}' ({rel}) is shared by {len(shared_items)} items:")
                        for iid, title in other_items:
                            print(f"      [{iid}] {title[:50]}...")

    if ml1m_extraction_file.exists():
        extraction_data = load_json(ml1m_extraction_file)
        result = get_extraction_result(extraction_data, 260)
        if result:
            print(f"\nRaw Output from MLLM:")
            print(f"  {result.get('raw_output', 'N/A')[:500]}")

    # === Video Games ===
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Video Games Dataset")
    print("=" * 80)

    vg_phase1_file = Path('results/videogames/phase1_5percent_exploration.json')
    vg_phase2_file = Path('results/videogames/phase2_constrained_5pct.json')

    if vg_phase1_file.exists():
        vg_phase1 = load_json(vg_phase1_file)

        # Find a good example (Sonic the Hedgehog)
        print("\nLooking for Sonic the Hedgehog 2...")
        for result in vg_phase1.get('results', []):
            if 'Sonic' in result.get('title', '') and result.get('status') == 'success':
                print(f"\n--- Selected Game: {result['title']} ---")
                print(f"RecBole ID: {result['recbole_id']}")
                print(f"\nStage 1 Raw Knowledge Points (Exploratory):")
                for i, kp in enumerate(result.get('knowledge_points', []), 1):
                    print(f"  {i}. ({kp['relation']}, {kp['entity']})")
                break

    if vg_phase2_file.exists():
        vg_phase2 = load_json(vg_phase2_file)

        # Find same game in phase2
        for result in vg_phase2.get('results', []):
            if 'Sonic' in result.get('title', '') and 'Hedgehog 2' in result.get('title', ''):
                print(f"\nStage 2 Standardized Knowledge Points:")
                for i, kp in enumerate(result.get('knowledge_points', []), 1):
                    print(f"  {i}. ({kp['relation']}, {kp['entity']})")
                break

    # === Beauty ===
    print("\n" + "=" * 80)
    print("ALTERNATIVE EXAMPLE 2: Beauty Dataset")
    print("=" * 80)

    beauty_phase1_file = Path('results/beauty/phase1_results.json')
    beauty_phase2_file = Path('results/beauty/phase2_results.json')
    beauty_full_file = Path('results/beauty/full_extraction.json')

    if beauty_phase1_file.exists():
        beauty_phase1 = load_json(beauty_phase1_file)

        # Find a good cosmetics example
        print("\nLooking for a representative beauty product...")
        for result in beauty_phase1.get('results', []):
            if result.get('status') == 'success' and 'lipstick' in result.get('title', '').lower():
                print(f"\n--- Selected Product: {result['title'][:60]} ---")
                print(f"RecBole ID: {result['recbole_id']}")
                print(f"\nStage 1 Raw Knowledge Points:")
                for i, kp in enumerate(result.get('knowledge_points', []), 1):
                    print(f"  {i}. ({kp['relation']}, {kp['entity']})")
                if result.get('raw_response'):
                    print(f"\nRaw MLLM Response:")
                    print(f"  {result['raw_response'][:300]}...")
                break

    # === Table 11: Cross-Item Knowledge Sharing Statistics ===
    print("\n" + "=" * 80)
    print("TABLE 11: CROSS-ITEM KNOWLEDGE SHARING STATISTICS (ML-1M)")
    print("=" * 80)

    if ml1m_kg_file.exists():
        entity_items, _ = load_kg(ml1m_kg_file)
        item_mapping = load_item_mapping(ml1m_item_file)

        # Sort entities by number of items sharing them
        entity_counts = [(ent, len(items)) for ent, items in entity_items.items()]
        entity_counts.sort(key=lambda x: -x[1])

        # Group by relation type
        relation_entities = defaultdict(list)
        for ent, items in entity_items.items():
            if items:
                rel = items[0][1]  # Get relation from first item
                relation_entities[rel].append((ent, len(items)))

        print("\nTop Shared Entities by Relation Type:")
        print("-" * 70)

        # Select diverse entities from different relations
        selected_relations = ['color_palette', 'mood', 'visual_theme', 'lighting', 'character_type']

        for rel in selected_relations:
            if rel in relation_entities:
                # Sort by count and take top entity
                sorted_ents = sorted(relation_entities[rel], key=lambda x: -x[1])
                if sorted_ents:
                    top_ent, count = sorted_ents[0]
                    print(f"\nEntity: {top_ent} (relation: {rel})")
                    print(f"  Shared by: {count} items")
                    # Show example items
                    items_with_this = entity_items[top_ent][:5]
                    print(f"  Examples:")
                    for iid, r in items_with_this:
                        title = item_mapping.get(iid, f'ID:{iid}')
                        print(f"    - [{iid}] {title[:50]}")

        print("\n" + "-" * 70)
        print("\nOverall Statistics:")
        print(f"  Total unique entities: {len(entity_items)}")
        print(f"  Entities shared by 5+ items: {sum(1 for e, c in entity_counts if c >= 5)}")
        print(f"  Entities shared by 10+ items: {sum(1 for e, c in entity_counts if c >= 10)}")
        print(f"  Entities shared by 50+ items: {sum(1 for e, c in entity_counts if c >= 50)}")
        print(f"  Entities shared by 100+ items: {sum(1 for e, c in entity_counts if c >= 100)}")

        print("\n\nTop 10 Most Shared Entities Overall:")
        for i, (ent, count) in enumerate(entity_counts[:10], 1):
            items_with_this = entity_items[ent][:1]
            rel = items_with_this[0][1] if items_with_this else 'unknown'
            print(f"  {i}. {ent} ({rel}): {count} items")


if __name__ == '__main__':
    main()
