#!/usr/bin/env python3
"""
Phase 3: Video Games Full Extraction with Compact Vocabulary

Extract knowledge from ALL items (14,969) using compact entity vocabulary.
Map extracted entities to canonical compact entities for collaborative filtering.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import re
from datetime import datetime
from typing import List, Dict, Set
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src.extraction.mllm_interface import create_mllm

# Import mapping functions from compact vocabulary script
sys.path.insert(0, str(Path(__file__).parent))
from create_compact_entity_vocabulary import map_to_compact_entity, normalize_entity


def parse_json_response(response: str) -> List[Dict]:
    """Parse JSON from MLLM response with robust error handling."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try markdown code block
    code_block_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
    match = re.search(code_block_pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding JSON array
    array_pattern = r'\[\s*\{.*?\}\s*\]'
    match = re.search(array_pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not parse JSON from response", response, 0)


def load_compact_vocabulary(vocab_file: Path) -> Dict:
    """Load compact entity vocabulary."""
    with open(vocab_file, 'r') as f:
        return json.load(f)


def load_relation_vocabulary(vocab_file: Path) -> Dict:
    """Load relation vocabulary."""
    with open(vocab_file, 'r') as f:
        return json.load(f)


def load_video_games_metadata(metadata_file: Path) -> Dict:
    """Load Video Games filtered metadata."""
    with open(metadata_file, 'r') as f:
        return json.load(f)


def load_item_mapping(mapping_file: Path) -> Dict:
    """Load item mapping (ASIN <-> RecBole ID)."""
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    return data['recbole_to_original'], data['original_to_recbole']


def load_existing_results(result_file: Path) -> tuple[Dict, Set[int]]:
    """Load existing extraction results to skip already processed items."""
    if not result_file.exists():
        return None, set()

    with open(result_file, 'r') as f:
        data = json.load(f)

    processed_ids = {
        r['recbole_id'] for r in data.get('results', [])
        if r.get('status') == 'success'
    }

    return data, processed_ids


def save_results(output_file: Path, config: Dict, results: List[Dict]):
    """Save extraction results to JSON file."""
    output_data = {
        'phase': 'phase3_full_extraction',
        'dataset': 'amazon-videogames',
        'config': config,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }

    # Write atomically
    temp_file = output_file.with_suffix('.tmp.json')
    with open(temp_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    temp_file.replace(output_file)


def create_extraction_prompt(title: str, categories: str, relations: List[Dict]) -> str:
    """Create extraction prompt with relation constraints."""

    # Format relation list
    relation_list = []
    for i, rel in enumerate(relations, 1):
        relation_list.append(
            f"{i}. **{rel['relation']}**: {rel['description']}"
        )
    relation_text = "\n".join(relation_list)

    prompt = f"""You are a video game expert analyzing game product images (cover art, screenshots, promotional images).

**Game Information:**
- Title: {title}
- Categories: {categories}

**Task:** Extract visual knowledge from the game product image using ONLY the predefined relation types below.

**Predefined Relation Types:**
{relation_text}

**Important Guidelines:**
- Only extract what you can SEE in the image
- Use ONLY the relation types listed above (no custom relations)
- Be specific but use GENERAL descriptive terms (e.g., "military_soldier" not specific character names)
- Focus on TYPE/CATEGORY rather than specific instances (e.g., "fantasy_warrior" not "Aragorn")
- Describe visual characteristics that are SHARED across multiple games
- Extract 3-8 knowledge points per image
- If a visual aspect doesn't fit any relation, use "has_additional_property"

**Output Format:**
Provide a JSON list of knowledge points, each with:
- relation: MUST be one of the 15 predefined relations above
- entity: General descriptive term (e.g., "realistic_3d", "military_soldier", "urban", "warm_colors")

Example:
[
  {{"relation": "has_visual_style", "entity": "realistic_3d"}},
  {{"relation": "features_character", "entity": "military_soldier"}},
  {{"relation": "set_in_environment", "entity": "urban"}},
  {{"relation": "has_color_palette", "entity": "dark_palette"}},
  {{"relation": "shows_perspective", "entity": "third_person"}}
]

**Output only the JSON array, nothing else.**"""

    return prompt


def filter_and_map_entities(knowledge_points: List[Dict],
                            compact_vocab: Dict,
                            allowed_relations: Set[str]) -> List[Dict]:
    """
    Filter knowledge points and map entities to compact vocabulary.

    Returns:
        Filtered and mapped knowledge points
    """
    filtered = []

    for kp in knowledge_points:
        relation = kp.get('relation', '')
        entity = kp.get('entity', '')

        # Fix typo
        if relation == 'show_platform':
            relation = 'shows_platform'

        # Check if relation is allowed
        if relation not in allowed_relations:
            continue

        # Map entity to compact vocabulary
        compact_entity = map_to_compact_entity(entity, relation)

        # Verify entity exists in compact vocabulary
        if relation in compact_vocab['relations']:
            valid_entities = {
                e['entity'] for e in compact_vocab['relations'][relation]['entities']
            }
            if compact_entity in valid_entities:
                filtered.append({
                    'relation': relation,
                    'entity': compact_entity
                })

    return filtered


def extract_game_knowledge(
    asin: str,
    recbole_id: int,
    title: str,
    categories: str,
    image_path: Path,
    relations: List[Dict],
    compact_vocab: Dict,
    allowed_relations: Set[str],
    mllm
) -> Dict:
    """Extract knowledge from a single game with entity mapping."""

    try:
        # Create prompt
        prompt = create_extraction_prompt(title, categories, relations)

        # Call MLLM
        response = mllm.extract_from_image(
            image=str(image_path),
            system_prompt="You are a video game expert analyzing game product images.",
            user_prompt=prompt,
            temperature=0.7,
            max_tokens=1000
        )

        # Parse response
        knowledge_points = parse_json_response(response)

        # Filter and map to compact vocabulary
        filtered_kps = filter_and_map_entities(
            knowledge_points,
            compact_vocab,
            allowed_relations
        )

        return {
            'asin': asin,
            'recbole_id': recbole_id,
            'title': title,
            'categories': categories,
            'knowledge_points': filtered_kps,
            'num_knowledge_points': len(filtered_kps),
            'num_raw_knowledge_points': len(knowledge_points),
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }

    except json.JSONDecodeError as e:
        return {
            'asin': asin,
            'recbole_id': recbole_id,
            'title': title,
            'categories': categories,
            'error': f'JSON parse error: {str(e)}',
            'raw_response': response[:500] if 'response' in locals() else None,
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'asin': asin,
            'recbole_id': recbole_id,
            'title': title,
            'categories': categories,
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }


def main():
    parser = argparse.ArgumentParser(
        description='Phase 3: Video Games Full Extraction with Compact Vocabulary'
    )
    parser.add_argument('--metadata', type=str, required=True,
                       help='Path to filtered metadata JSON')
    parser.add_argument('--mapping', type=str, required=True,
                       help='Path to item mapping JSON')
    parser.add_argument('--images_dir', type=str, required=True,
                       help='Directory containing product images')
    parser.add_argument('--relation_vocab', type=str, required=True,
                       help='Path to relation vocabulary JSON')
    parser.add_argument('--entity_vocab', type=str, required=True,
                       help='Path to compact entity vocabulary JSON')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file')
    parser.add_argument('--api_key', type=str, required=True,
                       help='OpenAI API key')
    parser.add_argument('--base_url', type=str, default=None,
                       help='OpenAI API base URL (optional)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='MLLM model to use (default: gpt-4o-mini)')
    parser.add_argument('--workers', type=int, default=20,
                       help='Number of concurrent workers (default: 20)')

    args = parser.parse_args()

    # Paths
    metadata_file = Path(args.metadata)
    mapping_file = Path(args.mapping)
    images_dir = Path(args.images_dir)
    relation_vocab_file = Path(args.relation_vocab)
    entity_vocab_file = Path(args.entity_vocab)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Phase 3: Video Games Full Extraction with Compact Vocabulary")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Metadata: {metadata_file}")
    print(f"  Mapping: {mapping_file}")
    print(f"  Images: {images_dir}")
    print(f"  Relation Vocabulary: {relation_vocab_file}")
    print(f"  Entity Vocabulary: {entity_vocab_file}")
    print(f"  Output: {output_file}")
    print(f"  Model: {args.model}")
    if args.base_url:
        print(f"  Base URL: {args.base_url}")
    print(f"  Workers: {args.workers}")
    print()

    # Load vocabularies
    print("Loading vocabularies...")
    relation_vocab = load_relation_vocabulary(relation_vocab_file)
    relations = relation_vocab['relations']
    allowed_relations = set(r['relation'] for r in relations)

    compact_vocab = load_compact_vocabulary(entity_vocab_file)
    total_compact_entities = compact_vocab['total_unique_entities']

    print(f"  ✓ Loaded {len(relations)} relations")
    print(f"  ✓ Loaded {total_compact_entities} compact entities")
    print()

    # Load data
    print("Loading data...")
    metadata = load_video_games_metadata(metadata_file)
    recbole_to_asin, asin_to_recbole = load_item_mapping(mapping_file)

    total_items = len(recbole_to_asin)
    print(f"  Total items: {total_items:,}")
    print()

    # All items
    all_ids = list(range(1, total_items + 1))

    # Load existing results
    existing_data, processed_ids = load_existing_results(output_file)

    if existing_data:
        print(f"✓ Found existing results: {len(processed_ids)} already extracted")
        results = existing_data['results']
    else:
        results = []

    # Filter out already processed
    items_to_extract = [id for id in all_ids if id not in processed_ids]

    if not items_to_extract:
        print("\n✓ All items already extracted!")
        return

    print(f"  Remaining to extract: {len(items_to_extract):,}")
    print()

    # Initialize MLLM
    print(f"Initializing {args.model}...")
    mllm_kwargs = {'api_key': args.api_key}
    if args.base_url:
        mllm_kwargs['base_url'] = args.base_url
        print(f"  Using custom base URL: {args.base_url}")
    mllm = create_mllm('openai', args.model, **mllm_kwargs)
    print("✓ MLLM ready")
    print()

    # Extract knowledge (parallel)
    print(f"Extracting knowledge from {len(items_to_extract):,} game products...")
    print(f"Using {args.workers} concurrent workers...")
    print()

    config = {
        'model': args.model,
        'total_items': total_items,
        'workers': args.workers,
        'relation_vocabulary_version': relation_vocab.get('version', '1.0'),
        'entity_vocabulary_version': compact_vocab.get('vocabulary_version', '4.0'),
        'num_relations': len(relations),
        'num_compact_entities': total_compact_entities,
        'source': 'full_dataset'
    }

    success_count = 0
    error_count = 0
    results_lock = threading.Lock()

    def extract_single_item(recbole_id):
        """Extract knowledge from a single item (thread-safe)."""
        asin = recbole_to_asin[str(recbole_id)]
        meta = metadata.get(asin, {})
        title = meta.get('title', f'Game_{asin}')
        categories = '|'.join(meta.get('category', ['Unknown']))
        image_path = images_dir / f"{asin}.jpg"

        if not image_path.exists():
            return {
                'asin': asin,
                'recbole_id': recbole_id,
                'title': title,
                'error': 'Image not found',
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }

        return extract_game_knowledge(
            asin=asin,
            recbole_id=recbole_id,
            title=title,
            categories=categories,
            image_path=image_path,
            relations=relations,
            compact_vocab=compact_vocab,
            allowed_relations=allowed_relations,
            mllm=mllm
        )

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(extract_single_item, recbole_id): recbole_id
            for recbole_id in items_to_extract
        }

        with tqdm(total=len(items_to_extract), desc="Extracting") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()

                    with results_lock:
                        results.append(result)

                        if result['status'] == 'success':
                            success_count += 1
                        else:
                            error_count += 1

                        # Save every 50 items
                        if len(results) % 50 == 0:
                            save_results(output_file, config, results)

                    pbar.update(1)

                except Exception as e:
                    recbole_id = futures[future]
                    pbar.write(f"Error processing ID {recbole_id}: {e}")
                    error_count += 1

    # Final save
    save_results(output_file, config, results)

    # Summary
    print()
    print("="*70)
    print("Extraction Complete")
    print("="*70)
    print(f"\nTotal processed: {len(results):,}")
    print(f"  Successful: {success_count:,}")
    print(f"  Errors: {error_count:,}")
    print(f"\nResults saved to: {output_file}")
    print("="*70)


if __name__ == '__main__':
    main()
