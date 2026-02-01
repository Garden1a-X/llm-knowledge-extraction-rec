#!/usr/bin/env python3
"""
Extract user interests from Amazon Beauty rating history.

Uses the hybrid approach:
- 21-day buckets for short-term interests (statistical aggregation)
- LLM summarization every 84 days for long-term interests
- All entities must be in standard vocabulary

Usage:
    python scripts/extract_beauty_user_interests.py \
        --ratings data/recbole/amazon-beauty/amazon-beauty.inter \
        --product_kg results/beauty/full_extraction.json \
        --entity_vocab data/beauty_entity_vocabulary.json \
        --output results/beauty/user_interests.json \
        --model gpt-4o-mini \
        --base_url <your_base_url> \
        --api_key <your_api_key> \
        --workers 10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from typing import Dict, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.extraction.mllm_interface import create_mllm
from src.extraction.user_interest_extractor import (
    UserInterestExtractor,
    load_user_ratings
)


def load_existing_results(output_file: Path) -> Tuple[Dict, Set[int]]:
    """
    Load existing extraction results to resume from checkpoint.

    Returns:
        (existing_data, processed_user_ids)
    """
    if not output_file.exists():
        return None, set()

    try:
        with open(output_file, 'r') as f:
            data = json.load(f)

        processed_ids = {
            r['user_id'] for r in data.get('results', [])
        }

        return data, processed_ids
    except Exception as e:
        print(f"⚠️  Warning: Could not load existing results: {e}")
        return None, set()


def save_checkpoint(output_file: Path, results: List[Dict], config: Dict):
    """Save intermediate results as checkpoint."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'dataset': 'amazon-beauty',
        'config': config,
        'stats': {
            'total_users': len(results),
            'total_llm_calls': sum(r['num_llm_calls'] for r in results),
            'avg_llm_calls_per_user': sum(r['num_llm_calls'] for r in results) / len(results) if results else 0,
            'avg_interactions_per_user': sum(r['num_interactions'] for r in results) / len(results) if results else 0,
            'avg_active_days_per_user': sum(r['active_days'] for r in results) / len(results) if results else 0
        },
        'results': results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)


def load_entity_vocabulary(vocab_file: Path) -> Set[str]:
    """
    Load standardized entity vocabulary from Beauty vocabulary JSON.

    Args:
        vocab_file: Path to entity vocabulary JSON

    Returns:
        Set of valid entity strings
    """
    with open(vocab_file, 'r') as f:
        data = json.load(f)

    all_entities = set()

    # Beauty vocabulary format: {"relations": {"relation1": {"standard_entities": [...], ...}, ...}}
    if 'relations' in data:
        for relation, rel_data in data['relations'].items():
            if 'standard_entities' in rel_data:
                for entity in rel_data['standard_entities']:
                    all_entities.add(entity)

    return all_entities


def load_product_kg(
    extraction_file: Path,
    valid_entities: Set[str] = None
) -> Dict[int, List[Tuple[str, str]]]:
    """
    Load product knowledge graph from extraction results.

    Args:
        extraction_file: Full extraction JSON
        valid_entities: Set of valid entity strings (for filtering)

    Returns:
        Dict mapping item_id (RecBole) -> [(relation, entity), ...]
    """
    with open(extraction_file, 'r') as f:
        data = json.load(f)

    product_kg = {}

    results = data.get('results', [])

    for result in results:
        if result.get('status') != 'success':
            continue

        recbole_id = result.get('recbole_id')
        if recbole_id is None:
            continue

        kps = result.get('knowledge_points', [])

        # Filter by valid entities if vocabulary provided
        # Also include NEW_ entities
        if valid_entities:
            kps = [
                kp for kp in kps
                if kp.get('entity') in valid_entities or kp.get('entity', '').startswith('NEW_')
            ]

        product_kg[recbole_id] = [
            (kp['relation'], kp['entity'])
            for kp in kps
            if kp.get('relation') and kp.get('entity')
        ]

    return product_kg


def main():
    parser = argparse.ArgumentParser(
        description='Extract user interests from Beauty dataset (hybrid approach)'
    )

    parser.add_argument('--ratings', type=str,
                        default='data/recbole/amazon-beauty/amazon-beauty.inter',
                        help='Beauty ratings file (.inter)')
    parser.add_argument('--product_kg', type=str,
                        default='results/beauty/full_extraction.json',
                        help='Full extraction results')
    parser.add_argument('--entity_vocab', type=str,
                        default='data/beauty_entity_vocabulary.json',
                        help='Entity vocabulary file')
    parser.add_argument('--output', type=str,
                        default='results/beauty/user_interests.json',
                        help='Output JSON file')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='LLM model name')
    parser.add_argument('--base_url', type=str, default=None,
                        help='Custom API base URL')
    parser.add_argument('--api_key', type=str, required=True,
                        help='API key')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of concurrent workers')
    parser.add_argument('--test_users', type=int, default=None,
                        help='Test mode: process N random users only')
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                        help='Save checkpoint every N users')

    args = parser.parse_args()

    print("="*70)
    print("AMAZON BEAUTY USER INTEREST EXTRACTION (HYBRID)")
    print("="*70)
    print()

    # Load entity vocabulary
    print(f"Loading entity vocabulary: {args.entity_vocab}")
    valid_entities = load_entity_vocabulary(Path(args.entity_vocab))
    print(f"  ✓ {len(valid_entities)} valid entities")
    print()

    # Load product knowledge graph
    print(f"Loading product knowledge graph: {args.product_kg}")
    product_kg = load_product_kg(
        Path(args.product_kg),
        valid_entities=valid_entities
    )
    print(f"  ✓ {len(product_kg)} products with KG")

    # Stats
    total_kps = sum(len(kps) for kps in product_kg.values())
    avg_kps = total_kps / len(product_kg) if product_kg else 0
    print(f"  ✓ {total_kps} total knowledge points")
    print(f"  ✓ {avg_kps:.2f} KPs per product (avg)")
    print()

    # Load user ratings
    print(f"Loading user ratings: {args.ratings}")
    user_ratings_dict = load_user_ratings(Path(args.ratings))
    print(f"  ✓ {len(user_ratings_dict)} users")
    print()

    # Test mode
    if args.test_users:
        import random
        user_ids = random.sample(
            list(user_ratings_dict.keys()),
            min(args.test_users, len(user_ratings_dict))
        )
        print(f"⚠️  TEST MODE: Processing {len(user_ids)} users only")
        print()
    else:
        user_ids = list(user_ratings_dict.keys())

    # Initialize MLLM
    print(f"Initializing MLLM: {args.model}")
    mllm = create_mllm(
        backend='openai',
        model_name=args.model,
        base_url=args.base_url,
        api_key=args.api_key
    )
    print("  ✓ MLLM initialized")
    print()

    # Initialize extractor
    extractor = UserInterestExtractor(movie_kg=product_kg, mllm=mllm)

    # Load existing results for checkpoint/resume
    output_file = Path(args.output)
    existing_data, processed_user_ids = load_existing_results(output_file)

    if processed_user_ids:
        print(f"📂 Found existing checkpoint: {len(processed_user_ids)} users already processed")
        user_ids = [uid for uid in user_ids if uid not in processed_user_ids]
        print(f"  ✓ Resuming with {len(user_ids)} remaining users")
        print()

    # Start with existing results or empty list
    results = existing_data.get('results', []) if existing_data else []

    # Prepare config for checkpoints
    checkpoint_config = {
        'short_term_days': extractor.short_term_days,
        'long_term_buckets': extractor.long_term_buckets,
        'min_rating': extractor.min_rating,
        'short_term_top_k': extractor.short_term_top_k,
        'long_term_top_k': extractor.long_term_top_k,
        'model': args.model,
        'entity_vocab_file': args.entity_vocab,
        'num_valid_entities': len(valid_entities)
    }

    # Extract interests
    print(f"Extracting interests for {len(user_ids)} users...")
    print(f"  Strategy: 21-day buckets → 84-day LLM summarization")
    print(f"  Short-term: Top 10 entities per bucket (frequency > 1)")
    print(f"  Long-term: Top 5 entities (LLM summarized)")
    print(f"  Workers: {args.workers}")
    if len(results) > 0:
        print(f"  Starting from: {len(results)} already completed")
    print()

    def extract_for_user(user_id):
        result = extractor.extract_user_interests(
            user_ratings=user_ratings_dict[user_id],
            verbose=False
        )
        result['user_id'] = user_id
        return result

    if args.workers == 1:
        # Sequential
        for i, user_id in enumerate(user_ids, 1):
            result = extract_for_user(user_id)
            results.append(result)

            if i % args.checkpoint_interval == 0:
                print(f"  Processed {i}/{len(user_ids)} users...")
                save_checkpoint(output_file, results, checkpoint_config)
                print(f"  💾 Checkpoint saved ({len(results)} total users)")
    else:
        # Concurrent
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print("  (Install tqdm for progress bar: pip install tqdm)")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(extract_for_user, user_id): user_id
                for user_id in user_ids
            }

            completed_count = 0

            if use_tqdm:
                with tqdm(total=len(user_ids), desc="  Extracting") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        completed_count += 1
                        pbar.update(1)

                        if completed_count % args.checkpoint_interval == 0:
                            save_checkpoint(output_file, results, checkpoint_config)
                            pbar.write(f"  💾 Checkpoint saved ({len(results)} total users)")
            else:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    completed_count += 1

                    if completed_count % args.checkpoint_interval == 0:
                        print(f"  Processed {completed_count}/{len(user_ids)} users...")
                        save_checkpoint(output_file, results, checkpoint_config)
                        print(f"  💾 Checkpoint saved ({len(results)} total users)")

    print(f"  ✓ Completed: {len(results)} users")
    print()

    # Save final results
    if len(user_ids) > 0:
        save_checkpoint(output_file, results, checkpoint_config)
        print(f"💾 Final results saved: {output_file}")
    else:
        print(f"✓ All users already processed, no new extraction needed")
    print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total users: {len(results)}")
    if results:
        total_llm_calls = sum(r['num_llm_calls'] for r in results)
        avg_llm_calls = total_llm_calls / len(results)
        avg_interactions = sum(r['num_interactions'] for r in results) / len(results)
        avg_active_days = sum(r['active_days'] for r in results) / len(results)

        # Count interests
        total_short_term = sum(len(r.get('short_term_interests', [])) for r in results)
        total_long_term = sum(len(r.get('long_term_interests', [])) for r in results)

        print(f"Total LLM calls: {total_llm_calls}")
        print(f"Avg LLM calls/user: {avg_llm_calls:.2f}")
        print(f"Avg interactions/user: {avg_interactions:.1f}")
        print(f"Avg active days/user: {avg_active_days:.1f}")
        print(f"\nTotal short-term interests: {total_short_term}")
        print(f"Total long-term interests: {total_long_term}")
        print(f"Avg short-term/user: {total_short_term/len(results):.2f}")
        print(f"Avg long-term/user: {total_long_term/len(results):.2f}")

    print(f"\nOutput: {output_file}")
    print("="*70)

    print("\n📋 Next step: Convert to RecBole format")
    print("   python scripts/convert_user_interests_to_kg.py \\")
    print(f"       --input {output_file} \\")
    print("       --output_user data/recbole/amazon-beauty/amazon-beauty.user.kg \\")
    print("       --output_merged data/recbole/amazon-beauty/amazon-beauty.kg \\")
    print("       --item_kg data/recbole/amazon-beauty/amazon-beauty.item.kg")

    return 0


if __name__ == '__main__':
    exit(main())
