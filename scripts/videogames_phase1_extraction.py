#!/usr/bin/env python3
"""
Phase 1: Video Games Knowledge Extraction (5% Exploration)

Extract visual knowledge from game product images (cover art, screenshots).
Similar to ML-1M Phase 1 but adapted for video game products.
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

    # Get IDs that were successfully extracted
    processed_ids = {
        r['recbole_id'] for r in data.get('results', [])
        if r.get('status') == 'success'
    }

    return data, processed_ids


def save_sampled_ids(ids: List[int], output_file: Path):
    """Save sampled IDs for reproducibility."""
    id_file = output_file.parent / f"{output_file.stem}_sampled_ids.json"
    with open(id_file, 'w') as f:
        json.dump({
            'sampled_ids': ids,
            'num_samples': len(ids),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"  ✓ Sampled IDs saved to: {id_file}")


def load_sampled_ids(output_file: Path) -> List[int]:
    """Load previously sampled IDs if exist."""
    id_file = output_file.parent / f"{output_file.stem}_sampled_ids.json"
    if id_file.exists():
        with open(id_file, 'r') as f:
            data = json.load(f)
        return data['sampled_ids']
    return None


def save_results(output_file: Path, config: Dict, results: List[Dict]):
    """Save extraction results to JSON file."""
    output_data = {
        'phase': 'phase1_exploration',
        'dataset': 'amazon-videogames',
        'percentage': config.get('percentage', 5.0),
        'config': config,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }

    # Write atomically
    temp_file = output_file.with_suffix('.tmp.json')
    with open(temp_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    temp_file.replace(output_file)


def create_game_extraction_prompt(title: str, categories: str) -> str:
    """
    Create extraction prompt for video game products.

    Focus on visual characteristics visible from product images.
    """
    prompt = f"""You are a video game expert analyzing game product images (cover art, screenshots, promotional images).

**Game Information:**
- Title: {title}
- Categories: {categories}

**Task:** Extract visual knowledge from the game product image.

Focus on aspects you can DIRECTLY observe in the image:
- **Visual Style**: Art style (realistic, cartoon, pixel art, anime, etc.), graphics quality, color palette
- **Game Type**: Genre indicators (first-person view, top-down, side-scrolling, 3D, 2D, etc.)
- **Characters**: Character designs, clothing, equipment, species/type
- **Environment**: Settings visible (urban, fantasy, sci-fi, nature, indoor/outdoor, time period)
- **UI Elements**: Interface style, HUD elements visible
- **Atmosphere**: Mood, tone (dark, colorful, gritty, whimsical, etc.)
- **Platform/Era**: Visual indicators of platform or generation (if apparent)

**Important Guidelines:**
- Only extract what you can SEE in the image
- Be specific and detailed
- Describe visual characteristics, not gameplay mechanics
- Use descriptive language for visual elements
- If multiple images show different aspects, describe each

**Output Format:**
Provide a JSON list of knowledge points, each with:
- relation: The aspect type (e.g., "has_visual_style", "features_character", "set_in_environment")
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
    mllm
) -> Dict:
    """Extract knowledge from a single game product image."""

    try:
        # Create prompt
        prompt = create_game_extraction_prompt(title, categories)

        # Call MLLM (using correct interface)
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
            'raw_response': response[:500] if 'response' in locals() else None,  # First 500 chars for debugging
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
        description='Phase 1: Video Games Knowledge Extraction (5% Exploration)'
    )
    parser.add_argument('--metadata', type=str, required=True,
                       help='Path to filtered metadata JSON')
    parser.add_argument('--mapping', type=str, required=True,
                       help='Path to item mapping JSON')
    parser.add_argument('--images_dir', type=str, required=True,
                       help='Directory containing product images')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file')
    parser.add_argument('--percentage', type=float, default=5.0,
                       help='Percentage of items to extract (default: 5.0)')
    parser.add_argument('--api_key', type=str, required=True,
                       help='OpenAI API key')
    parser.add_argument('--base_url', type=str, default=None,
                       help='OpenAI API base URL (optional, for custom endpoints)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='MLLM model to use (default: gpt-4o-mini)')
    parser.add_argument('--workers', type=int, default=15,
                       help='Number of concurrent workers (default: 15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling (default: 42)')

    args = parser.parse_args()

    # Paths
    metadata_file = Path(args.metadata)
    mapping_file = Path(args.mapping)
    images_dir = Path(args.images_dir)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Phase 1: Video Games Knowledge Extraction")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Metadata: {metadata_file}")
    print(f"  Mapping: {mapping_file}")
    print(f"  Images: {images_dir}")
    print(f"  Output: {output_file}")
    print(f"  Percentage: {args.percentage}%")
    print(f"  Model: {args.model}")
    if args.base_url:
        print(f"  Base URL: {args.base_url}")
    print(f"  Workers: {args.workers}")
    print(f"  Random seed: {args.seed}")
    print()

    # Load data
    print("Loading data...")
    metadata = load_video_games_metadata(metadata_file)
    recbole_to_asin, asin_to_recbole = load_item_mapping(mapping_file)

    total_items = len(recbole_to_asin)
    num_samples = int(total_items * args.percentage / 100)

    print(f"  Total items: {total_items:,}")
    print(f"  Sample size: {num_samples:,} ({args.percentage}%)")
    print()

    # Load or create sample
    sampled_ids = load_sampled_ids(output_file)

    if sampled_ids:
        print(f"✓ Loaded existing sample: {len(sampled_ids)} IDs")
    else:
        # Sample items
        random.seed(args.seed)
        all_ids = list(range(1, total_items + 1))
        sampled_ids = sorted(random.sample(all_ids, num_samples))
        save_sampled_ids(sampled_ids, output_file)
        print(f"✓ Created new sample: {len(sampled_ids)} IDs")

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
        'percentage': args.percentage,
        'total_items': total_items,
        'sample_size': num_samples,
        'random_seed': args.seed,
        'workers': args.workers
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
