#!/usr/bin/env python3
"""
Phase 2: Beauty Constrained Knowledge Extraction with NEW_ Mechanism

Extract visual knowledge using predefined vocabulary.
- Relations: MUST use one of the 9 predefined relations
- Entities: MUST use standard entities OR mark with NEW_ prefix

Usage:
    python scripts/beauty_phase2_extraction.py \
        --metadata data/recbole/amazon-beauty/filtered_metadata.json \
        --mapping data/recbole/amazon-beauty/id_mappings.json \
        --images_dir /path/to/beauty/images \
        --vocabulary data/beauty_entity_vocabulary.json \
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


def load_vocabulary(vocab_file: Path) -> Dict:
    """Load the entity vocabulary."""
    with open(vocab_file, 'r') as f:
        return json.load(f)


def load_beauty_metadata(metadata_file: Path) -> Dict:
    """Load Beauty filtered metadata."""
    with open(metadata_file, 'r') as f:
        return json.load(f)


def load_item_mapping(mapping_file: Path) -> tuple:
    """Load item mapping (ASIN <-> RecBole ID)."""
    with open(mapping_file, 'r') as f:
        data = json.load(f)

    # Format: id_mappings['item']['original_to_recbole']
    original_to_recbole = data['item']['original_to_recbole']

    # Build reverse mapping
    recbole_to_original = {v: k for k, v in original_to_recbole.items()}

    return recbole_to_original, original_to_recbole


def load_phase1_ids(phase1_file: Path) -> Set[int]:
    """Load RecBole IDs from Phase 1 results."""
    if not phase1_file.exists():
        return set()
    with open(phase1_file, 'r') as f:
        data = json.load(f)
    return {r['recbole_id'] for r in data.get('results', [])}


def load_existing_results(result_file: Path) -> tuple:
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


def save_results(output_file: Path, config: Dict, results: List[Dict], vocab_stats: Dict = None):
    """Save extraction results to JSON file."""
    output_data = {
        'phase': 'phase2_constrained',
        'dataset': 'amazon-beauty',
        'config': config,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }

    if vocab_stats:
        output_data['vocabulary_stats'] = vocab_stats

    temp_file = output_file.with_suffix('.tmp.json')
    with open(temp_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    temp_file.replace(output_file)


def create_system_prompt(vocabulary: Dict) -> str:
    """
    Create system prompt with full vocabulary listing and NEW_ mechanism.
    """
    relations = vocabulary['relations']
    total_entities = sum(len(r['standard_entities']) for r in relations.values())

    # Build vocabulary description
    vocab_str = "═══════════════════════════════════════════════════════════════\n"
    vocab_str += f"APPROVED VOCABULARY ({len(relations)} Relations, {total_entities} Standard Entities)\n"
    vocab_str += "═══════════════════════════════════════════════════════════════\n\n"

    for idx, (relation, info) in enumerate(relations.items(), 1):
        entities = info['standard_entities']
        vocab_str += f"【Relation {idx}: {relation}】\n"
        vocab_str += f"Description: {info['description']}\n"
        vocab_str += f"Standard Entities: {', '.join(entities)}\n\n"

    return f"""You are an expert in analyzing beauty product images and extracting visual knowledge for a recommendation system.

Your task is to analyze product images and extract visual knowledge points using ONLY the approved vocabulary below.

{vocab_str}═══════════════════════════════════════════════════════════════
CRITICAL RULES - READ VERY CAREFULLY!
═══════════════════════════════════════════════════════════════

1. **Relations**: MUST use one of the {len(relations)} relations listed above
   - NEVER create new relations

2. **Entities**: You have TWO options for each entity:

   OPTION A - Use a STANDARD entity (PREFERRED):
   - The entity MUST be an EXACT CHARACTER-BY-CHARACTER match from that relation's list
   - Example: If list has "glossy", you must write "glossy" (not "shiny", not "Glossy")
   - Check the description for synonyms (e.g., "shiny=glossy" means use "glossy")

   OPTION B - Use NEW_ prefix (ONLY when truly necessary):
   - ONLY use NEW_ when NO standard entity can describe the visual feature
   - Format: NEW_entity_name (e.g., NEW_holographic)
   - Before using NEW_, ask yourself: "Is there ANY standard entity that fits?"

3. **Common Mistakes to AVOID**:
   ❌ Writing "shiny" when "glossy" exists → Use "glossy"
   ❌ Writing "cream" for texture when "creamy" exists → Use "creamy"
   ❌ Writing "gray" when "grey" exists → Use "grey"
   ❌ Writing "NEW_pink" when "pink" exists → Use "pink"
   ❌ Using entity from wrong relation (e.g., "product_other" in packaging)

4. **Quantity**: Extract 5-10 knowledge points
5. **Focus**: ONLY what you can SEE in the image"""


def create_user_prompt(title: str, categories: str) -> str:
    """Create user prompt for extraction with Chain-of-Thought verification."""
    return f"""Analyze this beauty product image.

**Product Information:**
- Title: {title}
- Categories: {categories}

Follow these steps carefully:

## STEP 1: Initial Observation
List what you see in the image (colors, textures, packaging, style, etc.)

## STEP 2: Map to Vocabulary (CRITICAL)
For each observation, find the matching relation and entity:

Think through each one like this:
- "I see [observation]. Which relation does this belong to?"
- "For that relation, what are the standard entities?" (Look at the list above!)
- "Is my entity an EXACT match to one in the list?"
- "If yes → use it. If no → either find a similar one OR use NEW_"

## STEP 3: Verification Checklist
Before finalizing, verify EACH knowledge point:
□ Relation is one of the 9 valid relations
□ Entity is EXACTLY spelled as in the vocabulary (character-by-character)
□ If using NEW_, confirm no standard entity could work

## STEP 4: Final Output
Output ONLY the verified knowledge points below this line:
--- FINAL ---
<relation>: <entity>
(one per line, 5-10 knowledge points, no explanations)

Begin:"""


def parse_extraction_output(output: str) -> List[Dict[str, str]]:
    """Parse LLM output into structured knowledge points."""
    knowledge_points = []

    output = output.replace('```', '')

    # Extract only after "--- FINAL ---" if present
    if '--- FINAL ---' in output:
        parts = output.split('--- FINAL ---')
        if len(parts) > 1:
            output = parts[-1]

    for line in output.strip().split('\n'):
        line = line.strip()

        if not line or line.startswith('#') or line.startswith('//'):
            continue
        if line.startswith('##') or line.upper().startswith('STEP'):
            continue

        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                relation = parts[0].strip()
                entity = parts[1].strip()

                # Remove numbering
                relation = re.sub(r'^\d+[\.)]\s*', '', relation)
                entity = re.sub(r'^\d+[\.)]\s*', '', entity)

                if relation and entity:
                    knowledge_points.append({
                        'relation': relation,
                        'entity': entity
                    })

    return knowledge_points


def calculate_coverage_stats(results: List[Dict], vocabulary: Dict) -> Dict:
    """Calculate vocabulary coverage statistics."""
    total_kps = 0
    new_kps = 0
    invalid_kps = 0
    new_entities = []
    invalid_entities = []

    valid_relations = set(vocabulary['relations'].keys())

    for result in results:
        if result.get('status') != 'success':
            continue

        for kp in result.get('knowledge_points', []):
            total_kps += 1
            relation = kp['relation']
            entity = kp['entity']

            if entity.startswith('NEW_'):
                new_kps += 1
                new_entities.append({
                    'relation': relation,
                    'entity': entity,
                    'recbole_id': result['recbole_id']
                })
            else:
                # Validate
                if relation in valid_relations:
                    valid_entities = set(vocabulary['relations'][relation]['standard_entities'])
                    if entity not in valid_entities:
                        invalid_kps += 1
                        invalid_entities.append({
                            'relation': relation,
                            'entity': entity,
                            'recbole_id': result['recbole_id'],
                            'reason': f'Entity not in vocabulary'
                        })
                else:
                    invalid_kps += 1
                    invalid_entities.append({
                        'relation': relation,
                        'entity': entity,
                        'recbole_id': result['recbole_id'],
                        'reason': f'Invalid relation'
                    })

    valid_kps = total_kps - new_kps - invalid_kps
    coverage_rate = valid_kps / total_kps if total_kps > 0 else 0.0

    return {
        'total_knowledge_points': total_kps,
        'valid_entities': valid_kps,
        'new_entities_count': new_kps,
        'invalid_entities_count': invalid_kps,
        'coverage_rate': coverage_rate,
        'coverage_percentage': coverage_rate * 100,
        'new_entities': new_entities,
        'invalid_entities': invalid_entities[:50],  # Limit to first 50
        'target_coverage': 90.0,
        'meets_target': coverage_rate >= 0.90
    }


def extract_product_knowledge(
    asin: str,
    recbole_id: int,
    title: str,
    categories: str,
    image_path: Path,
    system_prompt: str,
    mllm
) -> Dict:
    """Extract knowledge from a single product image."""
    try:
        user_prompt = create_user_prompt(title, categories)

        response = mllm.extract_from_image(
            image=str(image_path),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=1000
        )

        knowledge_points = parse_extraction_output(response)

        return {
            'asin': asin,
            'recbole_id': recbole_id,
            'title': title,
            'categories': categories,
            'knowledge_points': knowledge_points,
            'num_knowledge_points': len(knowledge_points),
            'raw_output': response,
            'status': 'success',
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
    parser.add_argument('--vocabulary', type=str, required=True,
                       help='Path to entity vocabulary JSON')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file')
    parser.add_argument('--phase1_results', type=str, default=None,
                       help='Phase 1 results JSON (include Phase 1 items)')
    parser.add_argument('--api_key', type=str, required=True,
                       help='API key')
    parser.add_argument('--base_url', type=str, required=True,
                       help='API base URL')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='Model to use')
    parser.add_argument('--workers', type=int, default=15,
                       help='Number of concurrent workers')
    parser.add_argument('--percentage', type=float, default=20.0,
                       help='Percentage of items to extract')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_interval', type=int, default=20,
                       help='Save checkpoint every N completions')

    args = parser.parse_args()

    # Paths
    metadata_file = Path(args.metadata)
    mapping_file = Path(args.mapping)
    images_dir = Path(args.images_dir)
    vocabulary_file = Path(args.vocabulary)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Phase 2: Beauty Constrained Knowledge Extraction")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Vocabulary: {vocabulary_file}")
    print(f"  Output: {output_file}")
    print(f"  Percentage: {args.percentage}%")
    print(f"  Model: {args.model}")
    print(f"  Base URL: {args.base_url}")
    print(f"  Workers: {args.workers}")
    print()

    # Load vocabulary
    print("Loading vocabulary...")
    vocabulary = load_vocabulary(vocabulary_file)
    num_relations = len(vocabulary['relations'])
    num_entities = sum(len(r['standard_entities']) for r in vocabulary['relations'].values())
    print(f"  ✓ {num_relations} relations, {num_entities} standard entities")
    print()

    # Load data
    print("Loading data...")
    metadata = load_beauty_metadata(metadata_file)
    recbole_to_asin, asin_to_recbole = load_item_mapping(mapping_file)
    total_items = len(recbole_to_asin)
    print(f"  Total items: {total_items:,}")

    # Calculate sample size
    num_samples = int(total_items * args.percentage / 100)
    print(f"  Target ({args.percentage}%): {num_samples}")

    # Load Phase 1 IDs
    phase1_ids = set()
    if args.phase1_results:
        phase1_ids = load_phase1_ids(Path(args.phase1_results))
        print(f"  Phase 1 items: {len(phase1_ids)}")

    # Sample IDs
    random.seed(args.seed)
    all_ids = list(range(1, total_items + 1))

    if phase1_ids:
        remaining_needed = num_samples - len(phase1_ids)
        available_ids = [rid for rid in all_ids if rid not in phase1_ids]
        if remaining_needed > 0:
            new_samples = random.sample(available_ids, min(remaining_needed, len(available_ids)))
            sampled_ids = sorted(list(phase1_ids) + new_samples)
        else:
            sampled_ids = sorted(list(phase1_ids))
    else:
        sampled_ids = random.sample(all_ids, min(num_samples, len(all_ids)))

    print(f"  Total to extract: {len(sampled_ids)}")
    print()

    # Load existing results
    existing_data, processed_ids = load_existing_results(output_file)
    if existing_data:
        print(f"✓ Found existing: {len(processed_ids)} already extracted")
        results = existing_data['results']
    else:
        results = []

    items_to_extract = [id for id in sampled_ids if id not in processed_ids]

    if not items_to_extract:
        print("\n✓ All items already extracted!")
        return

    print(f"  Remaining: {len(items_to_extract)}")
    print()

    # Initialize MLLM
    print(f"Initializing {args.model}...")
    mllm = create_mllm('openai', args.model, api_key=args.api_key, base_url=args.base_url)
    print("✓ MLLM ready")
    print()

    # Create system prompt
    system_prompt = create_system_prompt(vocabulary)

    # Config
    config = {
        'model': args.model,
        'total_items': total_items,
        'sample_size': len(sampled_ids),
        'percentage': args.percentage,
        'seed': args.seed,
        'workers': args.workers,
        'vocabulary_version': vocabulary.get('version', '1.0'),
        'num_relations': num_relations,
        'num_standard_entities': num_entities
    }

    # Extract
    print(f"Extracting from {len(items_to_extract)} products...")
    print(f"Using {args.workers} workers...")
    print("-"*70)

    success_count = 0
    error_count = 0
    results_lock = threading.Lock()
    completed_count = 0

    def extract_single_item(recbole_id):
        asin = recbole_to_asin.get(recbole_id) or recbole_to_asin.get(str(recbole_id))
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
            system_prompt=system_prompt,
            mllm=mllm
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(extract_single_item, rid): rid
            for rid in items_to_extract
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
                            new_count = sum(1 for kp in result['knowledge_points']
                                          if kp['entity'].startswith('NEW_'))
                            pbar.write(f"  ✓ ID {recbole_id}: {kps} KPs ({new_count} NEW)")
                        else:
                            error_count += 1
                            pbar.write(f"  ✗ ID {recbole_id}: {result.get('error', 'Error')}")

                        if completed_count % args.save_interval == 0:
                            vocab_stats = calculate_coverage_stats(results, vocabulary)
                            save_results(output_file, config, results, vocab_stats)
                            pbar.write(f"  💾 Checkpoint: {completed_count}/{len(items_to_extract)} "
                                      f"(Coverage: {vocab_stats['coverage_percentage']:.1f}%)")

                    pbar.update(1)

                except Exception as e:
                    pbar.write(f"  ✗ ID {recbole_id}: {e}")
                    error_count += 1

    # Final save
    vocab_stats = calculate_coverage_stats(results, vocabulary)
    save_results(output_file, config, results, vocab_stats)

    # Summary
    print("-"*70)
    print("\nSUMMARY")
    print("="*70)
    print(f"Total processed: {len(results)}")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    print()
    print("VOCABULARY COVERAGE:")
    print(f"  Total KPs: {vocab_stats['total_knowledge_points']}")
    print(f"  Valid (in vocab): {vocab_stats['valid_entities']}")
    print(f"  NEW entities: {vocab_stats['new_entities_count']}")
    print(f"  Invalid: {vocab_stats['invalid_entities_count']}")
    print(f"  Coverage: {vocab_stats['coverage_percentage']:.1f}%")
    print(f"  Target: {vocab_stats['target_coverage']}%")
    print(f"  Meets target: {'✓ YES' if vocab_stats['meets_target'] else '✗ NO'}")
    print()
    print(f"Results saved to: {output_file}")
    print("="*70)


if __name__ == '__main__':
    main()
