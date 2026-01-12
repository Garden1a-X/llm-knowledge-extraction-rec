#!/usr/bin/env python3
"""
Extract user interests from rating history using hybrid approach.

Strategy:
- Short-term interests: Statistical aggregation per 21-day bucket
- Long-term interests: LLM summarization every 4 buckets (84 days)

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
from typing import Dict, List, Tuple
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src.extraction.mllm_interface import create_mllm


# Configuration
SHORT_TERM_DAYS = 21
LONG_TERM_BUCKETS = 4
MIN_RATING = 4.0
SHORT_TERM_TOP_K = 10
LONG_TERM_TOP_K = 5


def load_movie_kg(kg_file: Path) -> Dict[int, List[Tuple[str, str]]]:
    """Load movie knowledge graph."""
    with open(kg_file, 'r') as f:
        data = json.load(f)

    movie_kg = {}
    for result in data['results']:
        if result.get('status') != 'success':
            continue

        recbole_id = result['recbole_id']
        kps = result.get('knowledge_points', [])

        movie_kg[recbole_id] = [
            (kp['relation'], kp['entity'])
            for kp in kps
        ]

    return movie_kg


def load_user_ratings(ratings_file: Path) -> Dict[int, List[Dict]]:
    """
    Load user ratings grouped by user.

    Returns:
        Dict mapping user_id -> [{'item_id', 'rating', 'timestamp', 'date'}, ...]
    """
    user_ratings = {}

    with open(ratings_file, 'r') as f:
        header = f.readline()  # Skip header

        for line in f:
            parts = line.strip().split('\t')
            user_id = int(parts[0])
            item_id = int(parts[1])
            rating = float(parts[2])
            timestamp = int(parts[3])

            if user_id not in user_ratings:
                user_ratings[user_id] = []

            user_ratings[user_id].append({
                'item_id': item_id,
                'rating': rating,
                'timestamp': timestamp,
                'date': datetime.fromtimestamp(timestamp).date()
            })

    # Sort by timestamp for each user
    for user_id in user_ratings:
        user_ratings[user_id].sort(key=lambda x: x['timestamp'])

    return user_ratings


def extract_short_term_interest(
    interactions: List[Dict],
    movie_kg: Dict[int, List[Tuple[str, str]]],
    min_rating: float = MIN_RATING,
    top_k: int = SHORT_TERM_TOP_K
) -> List[Dict]:
    """
    Extract short-term interests from interactions using statistical aggregation.

    Args:
        interactions: List of user interactions in this bucket
        movie_kg: Movie knowledge graph
        min_rating: Minimum rating threshold
        top_k: Number of top interests to return

    Returns:
        List of {"relation": str, "entity": str, "count": int}
    """
    # Count knowledge points from high-rating movies
    kp_counter = Counter()

    for interaction in interactions:
        if interaction['rating'] < min_rating:
            continue

        item_id = interaction['item_id']
        if item_id not in movie_kg:
            continue

        for relation, entity in movie_kg[item_id]:
            kp_counter[(relation, entity)] += 1

    # Filter: only keep KPs with count > 1
    filtered_kps = [(rel, ent, cnt) for (rel, ent), cnt in kp_counter.items() if cnt > 1]

    # Sort by count and take top-K
    top_interests = sorted(filtered_kps, key=lambda x: -x[2])[:top_k]

    return [
        {"relation": rel, "entity": ent, "count": cnt}
        for rel, ent, cnt in top_interests
    ]


def get_llm_summarization_prompt(
    short_term_history: List[List[Dict]],
    current_long_term: List[Dict],
    bucket_interval_days: int = SHORT_TERM_DAYS * LONG_TERM_BUCKETS
) -> str:
    """
    Create prompt for LLM to summarize long-term interests.

    Args:
        short_term_history: List of short-term interests from recent buckets
        current_long_term: Current long-term interests with ages
        bucket_interval_days: Days covered by this summarization

    Returns:
        Prompt string
    """
    prompt = f"""You are analyzing a user's movie preferences to extract their long-term interests.

## USER'S RECENT SHORT-TERM INTERESTS

The user's interests over the past {len(short_term_history)} periods ({bucket_interval_days} days total):

"""

    for i, short_term in enumerate(short_term_history, 1):
        prompt += f"\n### Period {i} ({SHORT_TERM_DAYS} days):\n"
        if not short_term:
            prompt += "  (No prominent interests this period)\n"
        else:
            for item in short_term[:5]:  # Show top 5
                prompt += f"  - {item['relation']}: {item['entity']} (appeared {item['count']} times)\n"

    prompt += "\n\n## USER'S CURRENT LONG-TERM INTERESTS\n\n"

    if not current_long_term:
        prompt += "(No long-term interests established yet)\n"
    else:
        for item in current_long_term:
            prompt += f"  - {item['relation']}: {item['entity']} (持续了 {item['age_days']} 天)\n"

    prompt += f"""

## TASK

Based on the user's short-term interests and current long-term interests:

1. **Update long-term interests**: Determine which interests are stable and should be kept
2. **Add new interests**: Identify emerging patterns from recent periods
3. **Remove outdated interests**: Remove interests that are no longer relevant
4. **Update ages**: For kept interests, increase age by {bucket_interval_days} days; for new interests, start at {bucket_interval_days} days

**Output exactly {LONG_TERM_TOP_K} long-term interests in JSON format:**

```json
[
  {{"relation": "mood", "entity": "romantic", "age_days": 168}},
  {{"relation": "genre", "entity": "drama", "age_days": 84}},
  ...
]
```

**Important:**
- Output ONLY the JSON array, no other text
- Exactly {LONG_TERM_TOP_K} interests
- Use ONLY relation-entity pairs that appeared in short-term interests
- Age should reflect how long the interest has been stable

Output:"""

    return prompt


def summarize_long_term_interests(
    short_term_history: List[List[Dict]],
    current_long_term: List[Dict],
    mllm,
    temperature: float = 0.0,
    max_tokens: int = 500
) -> List[Dict]:
    """
    Use LLM to summarize long-term interests.

    Returns:
        List of {"relation": str, "entity": str, "age_days": int}
    """
    prompt = get_llm_summarization_prompt(short_term_history, current_long_term)

    try:
        # Direct OpenAI API call for text-only generation
        messages = [
            {"role": "user", "content": prompt}
        ]

        response_obj = mllm.client.chat.completions.create(
            model=mllm.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        response = response_obj.choices[0].message.content

        # Parse JSON from response
        # Try to find JSON array in response
        import re
        json_match = re.search(r'\[[\s\S]*\]', response)
        if not json_match:
            print(f"  Warning: Could not parse LLM response, keeping current long-term interests")
            return current_long_term

        json_str = json_match.group(0)
        updated_long_term = json.loads(json_str)

        # Validate format
        if not isinstance(updated_long_term, list):
            print(f"  Warning: LLM response not a list, keeping current")
            return current_long_term

        # Ensure each item has required fields
        validated = []
        for item in updated_long_term[:LONG_TERM_TOP_K]:
            if 'relation' in item and 'entity' in item and 'age_days' in item:
                validated.append({
                    'relation': item['relation'],
                    'entity': item['entity'],
                    'age_days': int(item['age_days'])
                })

        return validated if validated else current_long_term

    except Exception as e:
        print(f"  Error in LLM summarization: {e}")
        return current_long_term


def extract_user_interests_single(
    user_id: int,
    user_ratings: List[Dict],
    movie_kg: Dict[int, List[Tuple[str, str]]],
    mllm,
    verbose: bool = False
) -> Dict:
    """
    Extract interests for a single user.

    Returns:
        {
            'user_id': int,
            'num_interactions': int,
            'active_days': int,
            'num_short_buckets': int,
            'num_llm_calls': int,
            'short_term_interests': [...],
            'long_term_interests': [...]
        }
    """
    if verbose:
        print(f"Processing user {user_id}...")

    # Calculate active days
    unique_dates = set(r['date'] for r in user_ratings)
    active_days = len(unique_dates)

    # Group interactions by buckets (every 21 active days)
    num_short_buckets = max(1, int((active_days + SHORT_TERM_DAYS - 1) // SHORT_TERM_DAYS))

    # Simple bucketing: sort dates and assign to buckets
    sorted_dates = sorted(unique_dates)
    date_to_bucket = {}
    for i, date in enumerate(sorted_dates):
        bucket_id = i // SHORT_TERM_DAYS
        date_to_bucket[date] = bucket_id

    # Group interactions by bucket
    buckets = {}
    for interaction in user_ratings:
        bucket_id = date_to_bucket[interaction['date']]
        if bucket_id not in buckets:
            buckets[bucket_id] = []
        buckets[bucket_id].append(interaction)

    # Extract short-term interests for each bucket
    short_term_per_bucket = []
    for bucket_id in sorted(buckets.keys()):
        short_term = extract_short_term_interest(buckets[bucket_id], movie_kg)
        short_term_per_bucket.append(short_term)

    # LLM summarization every 4 buckets
    long_term_interests = []
    num_llm_calls = 0
    short_term_buffer = []

    for i, short_term in enumerate(short_term_per_bucket):
        short_term_buffer.append(short_term)

        # Summarize every 4 buckets or at the end
        if (i + 1) % LONG_TERM_BUCKETS == 0 or i == len(short_term_per_bucket) - 1:
            long_term_interests = summarize_long_term_interests(
                short_term_history=short_term_buffer,
                current_long_term=long_term_interests,
                mllm=mllm
            )
            num_llm_calls += 1
            short_term_buffer = []  # Reset buffer

    # Final output: last bucket's short-term + final long-term
    final_short_term = short_term_per_bucket[-1] if short_term_per_bucket else []

    return {
        'user_id': user_id,
        'num_interactions': len(user_ratings),
        'active_days': active_days,
        'num_short_buckets': num_short_buckets,
        'num_llm_calls': num_llm_calls,
        'short_term_interests': final_short_term,
        'long_term_interests': long_term_interests
    }


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
    parser.add_argument('--workers', type=int, default=1, help='Concurrent workers (1 for sequential)')
    parser.add_argument('--test_users', type=int, default=None, help='Test on N users only')

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

    # Extract interests
    print(f"Extracting interests for {len(user_ids)} users...")
    print(f"  Strategy: 21-day buckets → 84-day LLM summarization")
    print(f"  Workers: {args.workers}")
    print()

    results = []

    if args.workers == 1:
        # Sequential
        for user_id in user_ids:
            result = extract_user_interests_single(
                user_id=user_id,
                user_ratings=user_ratings_dict[user_id],
                movie_kg=movie_kg,
                mllm=mllm,
                verbose=True
            )
            results.append(result)

            if len(results) % 100 == 0:
                print(f"  Processed {len(results)}/{len(user_ids)} users...")
    else:
        # Concurrent
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    extract_user_interests_single,
                    user_id=user_id,
                    user_ratings=user_ratings_dict[user_id],
                    movie_kg=movie_kg,
                    mllm=mllm,
                    verbose=False
                ): user_id
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
            'short_term_days': SHORT_TERM_DAYS,
            'long_term_buckets': LONG_TERM_BUCKETS,
            'min_rating': MIN_RATING,
            'short_term_top_k': SHORT_TERM_TOP_K,
            'long_term_top_k': LONG_TERM_TOP_K,
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
