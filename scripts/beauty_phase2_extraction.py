#!/usr/bin/env python3
"""
Phase 2: Beauty Constrained Knowledge Extraction

Extract visual knowledge using the predefined 9-relation vocabulary.
Entities are post-processed through mapping functions to standard vocabulary.

Usage:
    python scripts/beauty_phase2_extraction.py \
        --metadata data/recbole/amazon-beauty/filtered_metadata.json \
        --mapping data/recbole/amazon-beauty/id_mappings.json \
        --images_dir /path/to/beauty/images \
        --relation_vocab data/beauty_relation_vocabulary.json \
        --entity_vocab data/beauty_entity_vocabulary.json \
        --api_key YOUR_KEY \
        --base_url YOUR_URL \
        --output results/beauty/phase2_results.json \
        --percentage 20.0 \
        --workers 15
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

# Import entity mapping functions
from beauty_entity_mapping import map_to_standard_entity, normalize_entity


def parse_json_response(response: str) -> List[Dict]:
    """
    Parse JSON from MLLM response, handling various formats.
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


def load_entity_vocabulary(vocab_file: Path) -> Dict:
    """Load the entity vocabulary."""
    with open(vocab_file, 'r') as f:
        return json.load(f)


def load_beauty_metadata(metadata_file: Path) -> Dict:
    """Load Beauty filtered metadata."""
    with open(metadata_file, 'r') as f:
        return json.load(f)


def load_item_mapping(mapping_file: Path) -> Dict:
    """Load item mapping (ASIN <-> RecBole ID)."""
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    return data['recbole_to_original'], data['original_to_recbole']


def load_phase1_ids(phase1_file: Path) -> Set[int]:
    """Load RecBole IDs from Phase 1 results."""
    if not phase1_file.exists():
        return set()
    with open(phase1_file, 'r') as f:
        data = json.load(f)
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
        'dataset': 'amazon-beauty',
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
    """
    # Format relation list for prompt
    relation_list = []
    for i, rel in enumerate(relations, 1):
        relation_list.append(
            f"{i}. **{rel['relation']}**: {rel['description']}"
        )
    relation_text = "\n".join(relation_list)

    prompt = f"""You are a beauty product expert analyzing product images (cosmetics, skincare, haircare, etc.).

**Product Information:**
- Title: {title}
- Categories: {categories}

**Task:** Extract visual knowledge from the product image using ONLY the predefined relation types below.

**Predefined Relation Types:**
{relation_text}

**Important Guidelines:**
- Only extract what you can SEE in the image
- Use ONLY the relation types listed above (no custom relations)
- Use GENERAL descriptive terms for entities (e.g., "pink" not "coral_reef_sunset_pink")
- Focus on TYPE/CATEGORY rather than specific details (e.g., "lip_product" not "MAC_Ruby_Woo")
- Describe visual characteristics that could be shared across multiple products
- Extract 3-8 knowledge points per image
- If a visual aspect doesn't fit any relation, use "additional_property"

**Output Format:**
Provide a JSON list of knowledge points, each with:
- relation: MUST be one of the 9 predefined relations above
- entity: General descriptive term (e.g., "pink", "matte", "tube", "high_end")

Example:
[
  {{"relation": "product_type", "entity": "lipstick"}},
  {{"relation": "has_color", "entity": "red"}},
  {{"relation": "finish_type", "entity": "matte"}},
  {{"relation": "packaging", "entity": "tube"}},
  {{"relation": "brand_aesthetic", "entity": "luxury"}}
]

**Output only the JSON array, nothing else.**"""

    return prompt


def filter_and_map_entities(
    knowledge_points: List[Dict],
    entity_vocab: Dict,
    allowed_relations: Set[str]
) -> tuple[List[Dict], List[Dict]]:
    """
    Filter knowledge points and map entities to standard vocabulary.

    Returns:
        (filtered_kps, unmapped_kps): Filtered/mapped KPs and unmapped KPs
    """
    filtered = []
    unmapped = []

    for kp in knowledge_points:
        relation = kp.get('relation', '')
        entity = kp.get('entity', '')

        # Check if relation is allowed
        if relation not in allowed_relations:
            unmapped.append({
                'relation': relation,
                'entity': entity,
                'reason': 'invalid_relation'
            })
            continue

        # Map entity to standard vocabulary
        mapped_entity = map_to_standard_entity(entity, relation)

        # Verify entity exists in vocabulary
        if relation in entity_vocab['relations']:
            valid_entities = set(entity_vocab['relations'][relation]['standard_entities'])
            if mapped_entity in valid_entities:
                filtered.append({
                    'relation': relation,
                    'entity': mapped_entity,
                    'original_entity': entity if entity != mapped_entity else None
                })
            else:
                unmapped.append({
                    'relation': relation,
                    'entity': entity,
                    'mapped_entity': mapped_entity,
                    'reason': 'entity_not_in_vocab'
                })
        else:
            unmapped.append({
                'relation': relation,
                'entity': entity,
                'reason': 'relation_not_in_vocab'
            })

    # Clean up filtered results (remove None original_entity)
    for kp in filtered:
        if kp.get('original_entity') is None:
            del kp['original_entity']

    return filtered, unmapped


def extract_product_knowledge(
    asin: str,
    recbole_id: int,
    title: str,
    categories: str,
    image_path: Path,
    relations: List[Dict],
    entity_vocab: Dict,
    allowed_relations: Set[str],
    mllm
) -> Dict:
    """Extract knowledge from a single product image with entity mapping."""

    try:
        # Create prompt
        prompt = create_constrained_extraction_prompt(title, categories, relations)

        # Call MLLM
        response = mllm.extract_from_image(
            image=str(image_path),
            system_prompt="You are a beauty product expert analyzing product images.",
            user_prompt=prompt,
            temperature=0.7,
            max_tokens=1000
        )

        # Parse response
        raw_knowledge_points = parse_json_response(response)

        # Filter and map to standard vocabulary
        filtered_kps, unmapped_kps = filter_and_map_entities(
            raw_knowledge_points,
            entity_vocab,
            allowed_relations
        )

        return {
            'asin': asin,
            'recbole_id': recbole_id,
            'title': title,
            'categories': categories,
            'knowledge_points': filtered_kps,
            'num_knowledge_points': len(filtered_kps),
            'num_raw_knowledge_points': len(raw_knowledge_points),
            'unmapped_knowledge_points': unmapped_kps if unmapped_kps else None,
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
        description='Phase 2: Beauty Constrained Knowledge Extraction'
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
                       help='Path to entity vocabulary JSON')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file')
    parser.add_argument('--phase1_results', type=str, default=None,
                       help='Phase 1 results JSON (if provided, include Phase 1 items)')
    parser.add_argument('--api_key', type=str, required=True,
                       help='OpenAI API key')
    parser.add_argument('--base_url', type=str, default=None,
                       help='OpenAI API base URL (optional, for custom endpoints)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='MLLM model to use (default: gpt-4o-mini)')
    parser.add_argument('--workers', type=int, default=15,
                       help='Number of concurrent workers (default: 15)')
    parser.add_argument('--percentage', type=float, default=20.0,
                       help='Percentage of items to extract (default: 20.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')
    parser.add_argument('--save_interval', type=int, default=20,
                       help='Save checkpoint every N completions')

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
    print("Phase 2: Beauty Constrained Knowledge Extraction")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Metadata: {metadata_file}")
    print(f"  Mapping: {mapping_file}")
    print(f"  Images: {images_dir}")
    print(f"  Relation Vocabulary: {relation_vocab_file}")
    print(f"  Entity Vocabulary: {entity_vocab_file}")
    print(f"  Output: {output_file}")
    print(f"  Percentage: {args.percentage}%")
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

    entity_vocab = load_entity_vocabulary(entity_vocab_file)
    total_entities = entity_vocab.get('total_standard_entities', 0)

    print(f"  Relations: {len(relations)}")
    print(f"  Standard entities: {total_entities}")
    print()

    # Load data
    print("Loading data...")
    metadata = load_beauty_metadata(metadata_file)
    recbole_to_asin, asin_to_recbole = load_item_mapping(mapping_file)

    total_items = len(recbole_to_asin)
    print(f"  Total items: {total_items:,}")
    print()

    # Calculate sample size
    num_samples = int(total_items * args.percentage / 100)
    print(f"Target sample ({args.percentage}%): {num_samples}")

    # Load Phase 1 IDs if provided
    phase1_ids = set()
    if args.phase1_results:
        phase1_file = Path(args.phase1_results)
        phase1_ids = load_phase1_ids(phase1_file)
        print(f"  Phase 1 items: {len(phase1_ids)}")

    # Sample IDs (including Phase 1 IDs)
    random.seed(args.seed)
    all_ids = list(range(1, total_items + 1))

    if phase1_ids:
        # Include Phase 1 IDs and sample remaining
        remaining_needed = num_samples - len(phase1_ids)
        available_ids = [rid for rid in all_ids if rid not in phase1_ids]
        if remaining_needed > 0:
            new_samples = random.sample(available_ids, min(remaining_needed, len(available_ids)))
            sampled_ids = sorted(list(phase1_ids) + new_samples)
        else:
            sampled_ids = sorted(list(phase1_ids))
        print(f"  Phase 1 IDs: {len(phase1_ids)}")
        print(f"  New samples: {len(sampled_ids) - len(phase1_ids)}")
    else:
        sampled_ids = random.sample(all_ids, min(num_samples, len(all_ids)))

    print(f"  Total to extract: {len(sampled_ids)}")
    print()

    # Load existing results
    existing_data, processed_ids = load_existing_results(output_file)

    if existing_data:
        print(f"Found existing results: {len(processed_ids)} already extracted")
        results = existing_data['results']
    else:
        results = []

    # Filter out already processed
    items_to_extract = [id for id in sampled_ids if id not in processed_ids]

    if not items_to_extract:
        print("\nAll sampled items already extracted!")
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
    print("MLLM ready")
    print()

    # Config for saving
    config = {
        'model': args.model,
        'total_items': total_items,
        'sample_size': len(sampled_ids),
        'percentage': args.percentage,
        'seed': args.seed,
        'workers': args.workers,
        'relation_vocab_version': relation_vocab.get('version', '1.0'),
        'entity_vocab_version': entity_vocab.get('version', '1.0'),
        'num_relations': len(relations),
        'num_standard_entities': total_entities
    }

    # Extract knowledge (parallel)
    print(f"Extracting knowledge from {len(items_to_extract)} products...")
    print(f"Using {args.workers} concurrent workers...")
    print()

    success_count = 0
    error_count = 0
    results_lock = threading.Lock()
    completed_count = 0

    def extract_single_item(recbole_id):
        """Extract knowledge from a single item (thread-safe)."""
        asin = recbole_to_asin[str(recbole_id)]
        meta = metadata.get(asin, {})
        title = meta.get('title', f'Product_{asin}')
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

        return extract_product_knowledge(
            asin=asin,
            recbole_id=recbole_id,
            title=title,
            categories=categories,
            image_path=image_path,
            relations=relations,
            entity_vocab=entity_vocab,
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
                recbole_id = futures[future]

                try:
                    result = future.result()

                    with results_lock:
                        results.append(result)
                        completed_count += 1

                        if result['status'] == 'success':
                            success_count += 1
                            kps = result['num_knowledge_points']
                            raw_kps = result['num_raw_knowledge_points']
                            pbar.write(f"  ID {recbole_id}: {kps}/{raw_kps} KPs mapped")
                        else:
                            error_count += 1
                            pbar.write(f"  ID {recbole_id}: {result.get('error', 'Unknown error')}")

                        # Periodic save
                        if completed_count % args.save_interval == 0:
                            save_results(output_file, config, results)
                            pbar.write(f"  Checkpoint: {completed_count}/{len(items_to_extract)}")

                    pbar.update(1)

                except Exception as e:
                    pbar.write(f"Error processing ID {recbole_id}: {e}")
                    error_count += 1

    # Final save
    save_results(output_file, config, results)

    # Calculate statistics
    total_kps = sum(r.get('num_knowledge_points', 0) for r in results if r['status'] == 'success')
    total_raw_kps = sum(r.get('num_raw_knowledge_points', 0) for r in results if r['status'] == 'success')
    mapping_rate = total_kps / total_raw_kps * 100 if total_raw_kps > 0 else 0

    # Summary
    print()
    print("="*70)
    print("Extraction Complete")
    print("="*70)
    print(f"\nTotal processed: {len(results)}")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"\nKnowledge Points:")
    print(f"  Raw extracted: {total_raw_kps}")
    print(f"  Mapped to vocab: {total_kps}")
    print(f"  Mapping rate: {mapping_rate:.1f}%")
    print(f"\nResults saved to: {output_file}")
    print("="*70)


if __name__ == '__main__':
    main()
