"""
User interest extraction from rating history.

Strategy:
- Short-term: Statistical aggregation per time bucket
- Long-term: LLM summarization from short-term history
"""

import json
import re
from typing import Dict, List, Tuple
from collections import Counter
from datetime import datetime
from pathlib import Path


# Configuration constants
SHORT_TERM_DAYS = 21
LONG_TERM_BUCKETS = 4
MIN_RATING = 4.0
SHORT_TERM_TOP_K = 10
LONG_TERM_TOP_K = 5


class UserInterestExtractor:
    """Extract user interests using hybrid statistical+LLM approach."""

    def __init__(
        self,
        movie_kg: Dict[int, List[Tuple[str, str]]],
        mllm,
        short_term_days: int = SHORT_TERM_DAYS,
        long_term_buckets: int = LONG_TERM_BUCKETS,
        min_rating: float = MIN_RATING,
        short_term_top_k: int = SHORT_TERM_TOP_K,
        long_term_top_k: int = LONG_TERM_TOP_K
    ):
        """
        Initialize user interest extractor.

        Args:
            movie_kg: Dict mapping item_id -> [(relation, entity), ...]
            mllm: MLLM instance for long-term summarization
            short_term_days: Days per short-term bucket
            long_term_buckets: Number of short-term buckets per LLM call
            min_rating: Minimum rating for positive interaction
            short_term_top_k: Number of short-term interests to keep
            long_term_top_k: Number of long-term interests to keep
        """
        self.movie_kg = movie_kg
        self.mllm = mllm
        self.short_term_days = short_term_days
        self.long_term_buckets = long_term_buckets
        self.min_rating = min_rating
        self.short_term_top_k = short_term_top_k
        self.long_term_top_k = long_term_top_k

    def extract_short_term_interest(
        self,
        interactions: List[Dict]
    ) -> List[Dict]:
        """
        Extract short-term interests using statistical aggregation.

        Args:
            interactions: List of {'item_id', 'rating', 'timestamp', 'date'}

        Returns:
            List of {"relation": str, "entity": str, "count": int}
        """
        kp_counter = Counter()

        for interaction in interactions:
            if interaction['rating'] < self.min_rating:
                continue

            item_id = interaction['item_id']
            if item_id not in self.movie_kg:
                continue

            for relation, entity in self.movie_kg[item_id]:
                kp_counter[(relation, entity)] += 1

        # Filter: only keep KPs with count > 1
        filtered_kps = [
            (rel, ent, cnt)
            for (rel, ent), cnt in kp_counter.items()
            if cnt > 1
        ]

        # Sort by count and take top-K
        top_interests = sorted(filtered_kps, key=lambda x: -x[2])[:self.short_term_top_k]

        return [
            {"relation": rel, "entity": ent, "count": cnt}
            for rel, ent, cnt in top_interests
        ]

    def get_llm_summarization_prompt(
        self,
        short_term_history: List[List[Dict]],
        current_long_term: List[Dict]
    ) -> str:
        """Create prompt for LLM to summarize long-term interests."""
        bucket_interval_days = self.short_term_days * self.long_term_buckets

        prompt = f"""You are analyzing a user's movie preferences to extract their long-term interests.

## USER'S RECENT SHORT-TERM INTERESTS

The user's interests over the past {len(short_term_history)} periods ({bucket_interval_days} days total):

"""

        for i, short_term in enumerate(short_term_history, 1):
            prompt += f"\n### Period {i} ({self.short_term_days} days):\n"
            if not short_term:
                prompt += "  (No prominent interests this period)\n"
            else:
                for item in short_term[:5]:
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

**Output exactly {self.long_term_top_k} long-term interests in JSON format:**

```json
[
  {{"relation": "mood", "entity": "romantic", "age_days": 168}},
  {{"relation": "genre", "entity": "drama", "age_days": 84}},
  ...
]
```

**Important:**
- Output ONLY the JSON array, no other text
- Exactly {self.long_term_top_k} interests
- Use ONLY relation-entity pairs that appeared in short-term interests
- Age should reflect how long the interest has been stable

Output:"""

        return prompt

    def summarize_long_term_interests(
        self,
        short_term_history: List[List[Dict]],
        current_long_term: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 500
    ) -> List[Dict]:
        """
        Use LLM to summarize long-term interests.

        Returns:
            List of {"relation": str, "entity": str, "age_days": int}
        """
        prompt = self.get_llm_summarization_prompt(short_term_history, current_long_term)

        try:
            # Direct OpenAI API call for text-only generation
            messages = [{"role": "user", "content": prompt}]

            response_obj = self.mllm.client.chat.completions.create(
                model=self.mllm.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            response = response_obj.choices[0].message.content

            # Parse JSON from response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if not json_match:
                return current_long_term

            json_str = json_match.group(0)
            updated_long_term = json.loads(json_str)

            if not isinstance(updated_long_term, list):
                return current_long_term

            # Validate format
            validated = []
            for item in updated_long_term[:self.long_term_top_k]:
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

    def extract_user_interests(
        self,
        user_ratings: List[Dict],
        verbose: bool = False
    ) -> Dict:
        """
        Extract interests for a single user.

        Args:
            user_ratings: Sorted list of user interactions

        Returns:
            {
                'num_interactions': int,
                'active_days': int,
                'num_short_buckets': int,
                'num_llm_calls': int,
                'short_term_interests': [...],
                'long_term_interests': [...]
            }
        """
        # Calculate active days
        unique_dates = set(r['date'] for r in user_ratings)
        active_days = len(unique_dates)

        # Group by buckets
        num_short_buckets = max(1, int((active_days + self.short_term_days - 1) // self.short_term_days))

        sorted_dates = sorted(unique_dates)
        date_to_bucket = {}
        for i, date in enumerate(sorted_dates):
            bucket_id = i // self.short_term_days
            date_to_bucket[date] = bucket_id

        # Group interactions by bucket
        buckets = {}
        for interaction in user_ratings:
            bucket_id = date_to_bucket[interaction['date']]
            if bucket_id not in buckets:
                buckets[bucket_id] = []
            buckets[bucket_id].append(interaction)

        # Extract short-term interests per bucket
        short_term_per_bucket = []
        for bucket_id in sorted(buckets.keys()):
            short_term = self.extract_short_term_interest(buckets[bucket_id])
            short_term_per_bucket.append(short_term)

        # LLM summarization every N buckets
        long_term_interests = []
        num_llm_calls = 0
        short_term_buffer = []

        for i, short_term in enumerate(short_term_per_bucket):
            short_term_buffer.append(short_term)

            # Summarize every N buckets or at the end
            if (i + 1) % self.long_term_buckets == 0 or i == len(short_term_per_bucket) - 1:
                long_term_interests = self.summarize_long_term_interests(
                    short_term_history=short_term_buffer,
                    current_long_term=long_term_interests
                )
                num_llm_calls += 1
                short_term_buffer = []

        # Final output
        final_short_term = short_term_per_bucket[-1] if short_term_per_bucket else []

        return {
            'num_interactions': len(user_ratings),
            'active_days': active_days,
            'num_short_buckets': num_short_buckets,
            'num_llm_calls': num_llm_calls,
            'short_term_interests': final_short_term,
            'long_term_interests': long_term_interests
        }


def load_movie_kg(kg_file: Path) -> Dict[int, List[Tuple[str, str]]]:
    """Load movie knowledge graph from JSON."""
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
        header = f.readline()

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

    # Sort by timestamp
    for user_id in user_ratings:
        user_ratings[user_id].sort(key=lambda x: x['timestamp'])

    return user_ratings
