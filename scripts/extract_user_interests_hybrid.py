#!/usr/bin/env python3
"""
Extract user interests from rating history using hybrid approach.

This is a CLI wrapper around src.extraction.user_interest_extractor.

Usage:
    python scripts/extract_user_interests_hybrid.py \
        --ratings data/recbole/ml-1m/ml-1m.inter \
        --movie_kg results/phase4_full_extraction_filtered.json \
        --output results/user_interests_hybrid.json \
        --model gpt-4o-mini \
        --workers 15
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.extraction.mllm_interface import create_mllm
from src.extraction.user_interest_extractor import (
    UserInterestExtractor,
    load_movie_kg,
    load_user_ratings
)


def main():
    parser = argparse.ArgumentParser(
        description='Extract user interests with hybrid statistical+LLM approach'
    )

    parser.add_argument('--ratings', type=str, required=True)
    parser.add_argument('--movie_kg', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--base_url', type=str, default=None)
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--test_users', type=int, default=None)

    args = parser.parse_args()

    print("="*70)
    print("USER INTEREST EXTRACTION (HYBRID)")
    print("="*70)
    print()

    # Load data
    print(f"Loading movie knowledge graph: {args.movie_kg}")
    movie_kg = load_movie_kg(Path(args.movie_kg))
    print(f"  ✓ {len(movie_kg)} movies with KG")
    print()

    print(f"Loading user ratings: {args.ratings}")
    user_ratings_dict = load_user_ratings(Path(args.ratings))
    print(f"  ✓ {len(user_ratings_dict)} users")
    print()

    # Test mode
    if args.test_users:
        import random
        user_ids = random.sample(list(user_ratings_dict.keys()), args.test_users)
        print(f"⚠️  TEST MODE: Processing {args.test_users} users only")
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
    extractor = UserInterestExtractor(movie_kg=movie_kg, mllm=mllm)

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
        for user_id in user_ids:
            result = extract_for_user(user_id)
            results.append(result)

            if len(results) % 100 == 0:
                print(f"  Processed {len(results)}/{len(user_ids)} users...")
    else:
        # Concurrent
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(extract_for_user, user_id): user_id
                for user_id in user_ids
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if len(results) % 100 == 0:
                    print(f"  Processed {len(results)}/{len(user_ids)} users...")

    print(f"  ✓ Completed: {len(results)} users")
    print()

    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
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
            'avg_llm_calls_per_user': sum(r['num_llm_calls'] for r in results) / len(results)
        },
        'results': results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved to: {output_file}")
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Users processed: {len(results)}")
    print(f"Total LLM calls: {output_data['stats']['total_llm_calls']}")
    print(f"Avg LLM calls/user: {output_data['stats']['avg_llm_calls_per_user']:.1f}")
    print("="*70)


if __name__ == '__main__':
    main()
