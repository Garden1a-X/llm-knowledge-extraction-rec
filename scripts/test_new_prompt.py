#!/usr/bin/env python3
"""
Test new prompt on a small sample to verify improvements.
"""

import json
import random
from pathlib import Path
from collections import Counter

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.poster_loader import PosterLoader
from src.extraction.mllm_interface import create_mllm
from src.extraction.prompts import PromptTemplates
from tqdm import tqdm


def main():
    # Load original results to get sample IDs
    original_file = Path('results/phase1_5percent_exploration.json')
    with open(original_file, 'r') as f:
        original_data = json.load(f)

    # Get successful extractions
    successful = [r for r in original_data['results'] if r['status'] == 'success']

    # Sample 10 random movies
    random.seed(42)
    test_sample = random.sample(successful, 10)
    test_ids = [r['recbole_id'] for r in test_sample]

    print(f"Testing new prompt on {len(test_ids)} movies:")
    print(f"RecBole IDs: {test_ids}")

    # Initialize
    poster_loader = PosterLoader(
        poster_dir='data/raw/ml-1m/posters',
        id_mapping_path='data/recbole/ml-1m/id_mappings.json'
    )

    # Get MLLM config from user
    print("\nMLLM Configuration:")
    backend = input("Backend (default: openai): ").strip() or "openai"
    model = input("Model (default: gpt-4o-mini): ").strip() or "gpt-4o-mini"
    base_url = input("Base URL (optional): ").strip() or None
    api_key = input("API Key (optional, will use env var if empty): ").strip() or None

    mllm_kwargs = {
        'backend': backend,
        'model_name': model
    }
    if api_key:
        mllm_kwargs['api_key'] = api_key
    if base_url:
        mllm_kwargs['base_url'] = base_url

    mllm = create_mllm(**mllm_kwargs)

    # Extract with new prompt
    print(f"\nExtracting with NEW prompt...")
    system_prompt = PromptTemplates.get_phase1_system_prompt()
    new_results = []

    for recbole_id in tqdm(test_ids, desc="Extracting"):
        try:
            original_id = poster_loader.get_original_movie_id(recbole_id)
            poster = poster_loader.load_poster(recbole_id, max_size=(1024, 1536))
            user_prompt = PromptTemplates.get_phase1_user_prompt()

            output = mllm.extract_from_image(
                image=poster,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
                max_tokens=1000
            )

            knowledge_points = PromptTemplates.parse_extraction_output(output)

            new_results.append({
                'recbole_id': recbole_id,
                'original_movie_id': original_id,
                'num_knowledge_points': len(knowledge_points),
                'knowledge_points': knowledge_points,
                'raw_output': output,
                'status': 'success'
            })

        except Exception as e:
            print(f"Error on ID {recbole_id}: {e}")
            new_results.append({
                'recbole_id': recbole_id,
                'status': 'error',
                'error': str(e)
            })

    # Save test results
    output_file = Path('results/prompt_test_new.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            'test_description': 'Testing new prompt on 10 random movies',
            'test_ids': test_ids,
            'config': mllm_kwargs,
            'results': new_results
        }, f, indent=2)

    print(f"\n✓ Test results saved to: {output_file}")

    # Quick analysis
    print("\n" + "="*60)
    print("QUICK ANALYSIS - NEW PROMPT")
    print("="*60)

    successful_new = [r for r in new_results if r['status'] == 'success']

    # Collect relations and entities
    all_relations = []
    all_entities = []
    for result in successful_new:
        for kp in result['knowledge_points']:
            all_relations.append(kp['relation'])
            all_entities.append(kp['entity'])

    relation_counts = Counter(all_relations)
    entity_counts = Counter(all_entities)

    # Check against old prompt examples
    old_prompt_relations = [
        'color_scheme', 'visual_style', 'main_element', 'composition',
        'mood', 'lighting', 'typography', 'texture'
    ]

    old_relation_count = sum(relation_counts[r] for r in old_prompt_relations if r in relation_counts)
    total_relations = len(all_relations)
    old_prompt_pct = (old_relation_count / total_relations * 100) if total_relations > 0 else 0

    print(f"\nRelations:")
    print(f"  Unique relations: {len(relation_counts)}")
    print(f"  Total relation instances: {total_relations}")
    print(f"  Old prompt relations %: {old_prompt_pct:.1f}%")
    print(f"\nTop 10 relations:")
    for rel, cnt in relation_counts.most_common(10):
        in_old = " [OLD]" if rel in old_prompt_relations else ""
        print(f"  {rel:25s}: {cnt:3d}{in_old}")

    print(f"\nEntities:")
    print(f"  Unique entities: {len(entity_counts)}")
    print(f"  Total entity instances: {len(all_entities)}")
    print(f"  Avg reuse per entity: {len(all_entities) / len(entity_counts):.2f}")
    print(f"\nTop 10 entities:")
    for ent, cnt in entity_counts.most_common(10):
        print(f"  {ent:30s}: {cnt:3d}")

    # Compare with old results
    print("\n" + "="*60)
    print("COMPARISON: OLD vs NEW")
    print("="*60)

    # Get old results for same IDs
    old_results_map = {r['recbole_id']: r for r in test_sample}

    old_relations = []
    old_entities = []
    for rid in test_ids:
        if rid in old_results_map:
            for kp in old_results_map[rid]['knowledge_points']:
                old_relations.append(kp['relation'])
                old_entities.append(kp['entity'])

    old_relation_counts = Counter(old_relations)
    old_entity_counts = Counter(old_entities)

    print(f"\nRelation Diversity:")
    print(f"  OLD: {len(old_relation_counts)} unique relations")
    print(f"  NEW: {len(relation_counts)} unique relations")
    print(f"  Improvement: {len(relation_counts) - len(old_relation_counts):+d}")

    print(f"\nEntity Reuse Rate:")
    old_reuse = len(old_entities) / len(old_entity_counts) if len(old_entity_counts) > 0 else 0
    new_reuse = len(all_entities) / len(entity_counts) if len(entity_counts) > 0 else 0
    print(f"  OLD: {old_reuse:.2f}x per entity")
    print(f"  NEW: {new_reuse:.2f}x per entity")
    print(f"  Improvement: {new_reuse - old_reuse:+.2f}x")

    print(f"\nOld Prompt Dependency:")
    old_in_old = sum(old_relation_counts[r] for r in old_prompt_relations if r in old_relation_counts)
    old_pct = (old_in_old / len(old_relations) * 100) if len(old_relations) > 0 else 0
    print(f"  OLD: {old_pct:.1f}%")
    print(f"  NEW: {old_prompt_pct:.1f}%")
    print(f"  Improvement: {old_pct - old_prompt_pct:+.1f}%")

    print("\n" + "="*60)
    print("\n✓ Review the results in results/prompt_test_new.json")
    print("✓ If satisfied, proceed with full re-extraction")


if __name__ == '__main__':
    main()
