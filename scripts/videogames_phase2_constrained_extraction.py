#!/usr/bin/env python3
"""
Phase 2: Video Games Constrained Knowledge Extraction

Extract visual knowledge using the predefined 14+1 relation vocabulary.
This ensures consistency and enables direct comparison with baselines.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
import argparse
import re
from datetime import datetime
from typing import List, Dict, Set
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src.extraction.mllm_interface import create_mllm


def parse_json_response(response: str) -> List[Dict]:
    """
    Parse JSON from MLLM response, handling various formats.

    Tries multiple strategies:
    1. Direct JSON parse
    2. Extract JSON array from markdown code blocks
    3. Extract first JSON array found in text
    """
    # Try direct parse first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    code_block_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
    match = re.search(code_block_pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding JSON array in text
    array_pattern = r'\[\s*\{.*?\}\s*\]'
    match = re.search(array_pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # If all fails, raise original error
    raise json.JSONDecodeError("Could not parse JSON from response", response, 0)


def load_relation_vocabulary(vocab_file: Path) -> Dict:
    """Load the relation vocabulary."""
    with open(vocab_file, 'r') as f:
        return json.load(f)


def load_video_games_metadata(metadata_file: Path) -> Dict:
    """Load Video Games raw metadata (ASIN -> metadata)."""
    with open(metadata_file, 'r') as f:
        data = json.load(f)

    # Handle both raw format (list of dicts) and filtered format (dict)
    if isinstance(data, list):
        # Raw format: convert to dict with ASIN as key
        return {item['parent_asin']: item for item in data}
    else:
        # Already in dict format
        return data


def load_item_mapping(mapping_file: Path) -> Dict:
    """Load item mapping (ASIN <-> RecBole ID)."""
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    return data['recbole_to_original'], data['original_to_recbole']


def load_phase1_ids(phase1_file: Path) -> Set[int]:
    """Load RecBole IDs from Phase 1 results."""
    with open(phase1_file, 'r') as f:
        data = json.load(f)

    # Get all IDs from Phase 1 (both success and error, we want to re-extract all)
    phase1_ids = {r['recbole_id'] for r in data.get('results', [])}
    return phase1_ids


def load_existing_results(result_file: Path) -> tuple[Dict, Set[int]]:
    """Load existing extraction results to skip already processed items."""
    if not result_file.exists():
        return None, set()

    with open(result_file, 'r') as f:
        data = json.load(f)

    # Get IDs that were successfully extracted
    processed_ids = {
        r['recbole_id'] for r in data.get('results', [])
        if r.get('status') == 'success'
    }

    return data, processed_ids


def save_results(output_file: Path, config: Dict, results: List[Dict]):
    """Save extraction results to JSON file."""
    output_data = {
        'phase': 'phase2_constrained',
        'dataset': 'amazon-videogames',
        'percentage': config.get('percentage', 100.0),
        'config': config,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }

    # Write atomically
    temp_file = output_file.with_suffix('.tmp.json')
    with open(temp_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    temp_file.replace(output_file)


def create_constrained_extraction_prompt(
    title: str,
    categories: str,
    relations: List[Dict]
) -> str:
    """
    Create extraction prompt with predefined relation vocabulary.

    Args:
        title: Game title
        categories: Game categories
        relations: List of predefined relations from vocabulary
    """
    # Format relation list for prompt
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
- Be specific and detailed in the entity values
- Describe visual characteristics, not gameplay mechanics
- If multiple images show different aspects, describe each
- If a visual aspect doesn't fit any relation, use "has_additional_property"

**Output Format:**
Provide a JSON list of knowledge points, each with:
- relation: MUST be one of the 15 predefined relations above (e.g., "has_visual_style", "features_character")
- entity: The specific value (e.g., "pixel_art", "armored_knight", "medieval_castle")

Example:
[
  {{"relation": "has_visual_style", "entity": "realistic_3d_graphics"}},
  {{"relation": "features_character", "entity": "space_marine_in_powered_armor"}},
  {{"relation": "set_in_environment", "entity": "futuristic_space_station"}},
  {{"relation": "has_color_palette", "entity": "dark_blue_and_orange"}},
  {{"relation": "shows_perspective", "entity": "first_person_view"}}
]

**Output only the JSON array, nothing else.**"""

    return prompt


def extract_game_knowledge(
    asin: str,
    recbole_id: int,
    title: str,
    categories: str,
    image_path: Path,
    relations: List[Dict],
    mllm
) -> Dict:
    """Extract knowledge from a single game product image with constrained relations."""

    try:
        # Create prompt
        prompt = create_constrained_extraction_prompt(title, categories, relations)

        # Call MLLM
        response = mllm.extract_from_image(
            image=str(image_path),
            system_prompt="You are a video game expert analyzing game product images.",
            user_prompt=prompt,
            temperature=0.7,
            max_tokens=1000
        )

        # Parse response with robust JSON extraction
        knowledge_points = parse_json_response(response)

        return {
            'asin': asin,
            'recbole_id': recbole_id,
            'title': title,
            'categories': categories,
            'knowledge_points': knowledge_points,
            'num_knowledge_points': len(knowledge_points),
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
        description='Phase 2: Video Games Constrained Knowledge Extraction'
    )
    parser.add_argument('--metadata', type=str, required=True,
                       help='Path to raw metadata JSON (e.g., meta_Video_Games.json)')
    parser.add_argument('--mapping', type=str, required=True,
                       help='Path to item mapping JSON')
    parser.add_argument('--images_dir', type=str, required=True,
                       help='Directory containing product images')
    parser.add_argument('--vocabulary', type=str, required=True,
                       help='Path to relation vocabulary JSON')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file')
    parser.add_argument('--phase1_results', type=str, default=None,
                       help='Phase 1 results JSON (if provided, extract only Phase 1 items)')
    parser.add_argument('--percentage', type=float, default=100.0,
                       help='Percentage of items to extract (ignored if --phase1_results is set)')
    parser.add_argument('--api_key', type=str, required=True,
                       help='OpenAI API key')
    parser.add_argument('--base_url', type=str, default=None,
                       help='OpenAI API base URL (optional, for custom endpoints)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='MLLM model to use (default: gpt-4o-mini)')
    parser.add_argument('--workers', type=int, default=15,
                       help='Number of concurrent workers (default: 15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling (default: 42, ignored if --phase1_results is set)')

    args = parser.parse_args()

    # Paths
    metadata_file = Path(args.metadata)
    mapping_file = Path(args.mapping)
    images_dir = Path(args.images_dir)
    vocabulary_file = Path(args.vocabulary)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Phase 2: Video Games Constrained Knowledge Extraction")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Metadata: {metadata_file}")
    print(f"  Mapping: {mapping_file}")
    print(f"  Images: {images_dir}")
    print(f"  Vocabulary: {vocabulary_file}")
    print(f"  Output: {output_file}")
    if args.phase1_results:
        print(f"  Phase 1 Results: {args.phase1_results}")
    else:
        print(f"  Percentage: {args.percentage}%")
        print(f"  Random seed: {args.seed}")
    print(f"  Model: {args.model}")
    if args.base_url:
        print(f"  Base URL: {args.base_url}")
    print(f"  Workers: {args.workers}")
    print()

    # Load vocabulary
    print("Loading relation vocabulary...")
    vocabulary = load_relation_vocabulary(vocabulary_file)
    relations = vocabulary['relations']
    print(f"  ✓ Loaded {len(relations)} predefined relations")
    print()

    # Load data
    print("Loading data...")
    metadata = load_video_games_metadata(metadata_file)
    recbole_to_asin, asin_to_recbole = load_item_mapping(mapping_file)

    total_items = len(recbole_to_asin)

    # Determine which items to extract
    if args.phase1_results:
        # Load Phase 1 IDs
        phase1_file = Path(args.phase1_results)
        phase1_ids = load_phase1_ids(phase1_file)
        sampled_ids = sorted(list(phase1_ids))
        print(f"  Total items: {total_items:,}")
        print(f"  Phase 1 items: {len(sampled_ids):,}")
        print(f"✓ Re-extracting Phase 1 items with constrained relations")
    else:
        # Sample by percentage
        num_samples = int(total_items * args.percentage / 100)
        print(f"  Total items: {total_items:,}")
        print(f"  Sample size: {num_samples:,} ({args.percentage}%)")
        print()

        random.seed(args.seed)
        all_ids = list(range(1, total_items + 1))

        if args.percentage >= 100.0:
            sampled_ids = all_ids
            print(f"✓ Processing all {len(sampled_ids)} items")
        else:
            sampled_ids = sorted(random.sample(all_ids, num_samples))
            print(f"✓ Sampled {len(sampled_ids)} IDs")

    print()

    # Load existing results
    existing_data, processed_ids = load_existing_results(output_file)

    if existing_data:
        print(f"✓ Found existing results: {len(processed_ids)} already extracted")
        results = existing_data['results']
    else:
        results = []

    # Filter out already processed
    items_to_extract = [id for id in sampled_ids if id not in processed_ids]

    if not items_to_extract:
        print("\n✓ All sampled items already extracted!")
        return

    print(f"  Remaining to extract: {len(items_to_extract)}")
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
    print(f"Extracting knowledge from {len(items_to_extract)} game products...")
    print(f"Using {args.workers} concurrent workers...")
    print()

    config = {
        'model': args.model,
        'total_items': total_items,
        'sample_size': len(sampled_ids),
        'workers': args.workers,
        'vocabulary_version': vocabulary.get('version', '1.0'),
        'num_relations': len(relations)
    }

    if args.phase1_results:
        config['source'] = 'phase1'
        config['phase1_results_file'] = args.phase1_results
    else:
        config['source'] = 'sampling'
        config['percentage'] = args.percentage
        config['random_seed'] = args.seed

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

                        # Save every 10 items
                        if len(results) % 10 == 0:
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
    print(f"\nTotal processed: {len(results)}")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"\nResults saved to: {output_file}")
    print("="*70)


if __name__ == '__main__':
    main()
