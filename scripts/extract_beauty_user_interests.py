#!/usr/bin/env python3
"""
Extract user interests from Beauty dataset rating history.

Uses the same hybrid approach as ML-1M:
- 21-day buckets for short-term interests
- LLM summarization every 84 days for long-term interests

Usage:
    python scripts/extract_beauty_user_interests.py \
        --ratings data/recbole/amazon-beauty/amazon-beauty.inter \
        --product_kg results/beauty_phase4_extraction.json \
        --item_mapping data/recbole/amazon-beauty/mappings/item_mapping.json \
        --output results/beauty_user_interests.json \
        --model gpt-4o-mini \
        --api_key YOUR_KEY \
        --workers 10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.extraction.mllm_interface import create_mllm
from src.extraction.user_interest_extractor import (
    UserInterestExtractor,
    load_user_ratings
)


def load_product_kg(
    phase4_file: Path,
    item_mapping_file: Path
) -> Dict[int, List[Tuple[str, str]]]:
    """
    Load product knowledge graph from Phase 4 extraction results.

    Args:
        phase4_file: Beauty Phase 4 extraction JSON (ASIN-based)
        item_mapping_file: ASIN to RecBole item_id mapping

    Returns:
        Dict mapping item_id (RecBole) -> [(relation, entity), ...]
    """
    # Load item mapping
    with open(item_mapping_file, 'r') as f:
        mapping_data = json.load(f)
    asin_to_recbole = mapping_data['original_to_recbole']

    # Load Phase 4 results
    with open(phase4_file, 'r') as f:
        data = json.load(f)

    product_kg = {}

    for result in data['results']:
        if result.get('status') != 'success':
            continue

        asin = result['asin']

        # Map ASIN to RecBole item_id
        if asin not in asin_to_recbole:
            continue

        item_id = asin_to_recbole[asin]
        kps = result.get('knowledge_points', [])

        product_kg[item_id] = [
            (kp['relation'], kp['entity'])
            for kp in kps
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
                        default='results/beauty_phase4_extraction.json',
                        help='Phase 4 extraction results')
    parser.add_argument('--item_mapping', type=str,
                        default='data/recbole/amazon-beauty/mappings/item_mapping.json',
                        help='Item mapping (ASIN to RecBole ID)')
    parser.add_argument('--output', type=str,
                        default='results/beauty_user_interests.json',
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

    args = parser.parse_args()

    print("="*70)
    print("BEAUTY USER INTEREST EXTRACTION (HYBRID)")
    print("="*70)
    print()

    # Load product knowledge graph
    print(f"Loading product knowledge graph...")
    print(f"  Phase 4: {args.product_kg}")
    print(f"  Mapping: {args.item_mapping}")

    product_kg = load_product_kg(
        Path(args.product_kg),
        Path(args.item_mapping)
    )

    print(f"  ✓ {len(product_kg)} products with KG")
    print()

    # Load user ratings
    print(f"Loading user ratings: {args.ratings}")
    user_ratings_dict = load_user_ratings(Path(args.ratings))
    print(f"  ✓ {len(user_ratings_dict)} users")
    print()

    # Test mode
    if args.test_users:
        import random
        user_ids = random.sample(list(user_ratings_dict.keys()),
                                 min(args.test_users, len(user_ratings_dict)))
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

    # Extract interests
    print(f"Extracting interests for {len(user_ids)} users...")
    print(f"  Strategy: 21-day buckets → 84-day LLM summarization")
    print(f"  Workers: {args.workers}")
    print()

    results = []

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

            if i % 100 == 0:
                print(f"  Processed {i}/{len(user_ids)} users...")
    else:
        # Concurrent
        from tqdm import tqdm
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(extract_for_user, user_id): user_id
                for user_id in user_ids
            }

            with tqdm(total=len(user_ids), desc="  Extracting") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)

    print(f"  ✓ Completed: {len(results)} users")
    print()

    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'dataset': 'amazon-beauty',
        'config': {
            'short_term_days': extractor.short_term_days,
            'long_term_buckets': extractor.long_term_buckets,
            'min_rating': extractor.min_rating,
            'short_term_top_k': extractor.short_term_top_k,
            'long_term_top_k': extractor.long_term_top_k,
            'model': args.model
        },
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

    print(f"Saved results: {output_file}")
    print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total users: {len(results)}")
    print(f"Total LLM calls: {output_data['stats']['total_llm_calls']}")
    print(f"Avg LLM calls/user: {output_data['stats']['avg_llm_calls_per_user']:.2f}")
    print(f"Avg interactions/user: {output_data['stats']['avg_interactions_per_user']:.1f}")
    print(f"Avg active days/user: {output_data['stats']['avg_active_days_per_user']:.1f}")
    print(f"\nOutput: {output_file}")
    print("="*70)

    return 0


if __name__ == '__main__':
    exit(main())
