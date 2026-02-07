#!/usr/bin/env python3
"""
Prepare Knowledge Quality Evaluation Dataset

This script:
1. Samples 100 items from Phase 1 results for each dataset
2. Gets corresponding Stage 3 (standardized) knowledge
3. Outputs evaluation data for LLM and Human evaluation

Usage:
    python scripts/prepare_quality_evaluation.py --output_dir results/quality_eval
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def load_ml1m_phase1(path: str) -> List[dict]:
    """Load ML-1M Phase 1 results."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Filter successful extractions
    results = []
    for item in data['results']:
        if item.get('status') == 'success' and item.get('knowledge_points'):
            results.append({
                'item_id': item['recbole_id'],
                'original_id': item.get('original_movie_id', item['recbole_id']),
                'knowledge_points': item['knowledge_points'],
                'raw_output': item.get('raw_output', '')
            })
    return results


def load_ml1m_phase3(path: str) -> Dict[int, List[dict]]:
    """Load ML-1M Phase 3 results (KG format)."""
    kg = defaultdict(list)
    with open(path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                item_id = int(parts[0])
                relation = parts[1]
                entity = parts[2]
                kg[item_id].append({
                    'relation': relation,
                    'entity': entity
                })
    return kg


def load_videogames_phase1(path: str) -> List[dict]:
    """Load Video Games Phase 1 results."""
    with open(path, 'r') as f:
        data = json.load(f)

    results = []
    for item in data['results']:
        if item.get('status') == 'success' and item.get('knowledge_points'):
            results.append({
                'item_id': item['recbole_id'],
                'asin': item['asin'],
                'title': item.get('title', ''),
                'knowledge_points': item['knowledge_points']
            })
    return results


def load_videogames_phase3(path: str) -> Dict[int, List[dict]]:
    """Load Video Games Phase 3 results."""
    with open(path, 'r') as f:
        data = json.load(f)

    results = {}
    for item in data['results']:
        if item.get('status') == 'success' and item.get('knowledge_points'):
            results[item['recbole_id']] = item['knowledge_points']
    return results


def load_beauty_phase1(path: str) -> List[dict]:
    """Load Beauty Phase 1 results."""
    with open(path, 'r') as f:
        data = json.load(f)

    results = []
    for item in data['results']:
        if item.get('status') == 'success' and item.get('knowledge_points'):
            results.append({
                'item_id': item['recbole_id'],
                'asin': item['asin'],
                'title': item.get('title', ''),
                'knowledge_points': item['knowledge_points']
            })
    return results


def load_beauty_phase3(path: str) -> Dict[int, List[dict]]:
    """Load Beauty Phase 3 results."""
    with open(path, 'r') as f:
        data = json.load(f)

    results = {}
    for item in data['results']:
        if item.get('status') == 'success' and item.get('knowledge_points'):
            results[item['recbole_id']] = item['knowledge_points']
    return results


def load_ml1m_titles(path: str) -> Dict[int, str]:
    """Load ML-1M item titles."""
    titles = {}
    with open(path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                item_id = int(parts[0])
                title = parts[1]
                titles[item_id] = title
    return titles


def sample_items(items: List[dict], n: int = 100, seed: int = 42) -> List[dict]:
    """Randomly sample n items."""
    random.seed(seed)
    if len(items) <= n:
        return items
    return random.sample(items, n)


def prepare_evaluation_data(
    dataset_name: str,
    phase1_items: List[dict],
    phase3_data: Dict[int, List[dict]],
    titles: Optional[Dict[int, str]] = None,
    seed: int = 42
) -> List[dict]:
    """
    Prepare evaluation data combining Phase 1 and Phase 3 knowledge.

    For each item, we sample ONE knowledge point from Stage 1 and ONE from Stage 3
    to keep evaluation manageable (100 items x 2 stages = 200 evaluations per dataset).

    Returns list of evaluation items, each containing:
    - item_id, title
    - stage1_knowledge: ONE sampled (relation, entity) from Phase 1
    - stage3_knowledge: ONE sampled (relation, entity) from Phase 3
    """
    random.seed(seed)
    eval_data = []

    for item in phase1_items:
        item_id = item['item_id']

        # Get title
        if titles:
            title = titles.get(item_id, f"Item {item_id}")
        else:
            title = item.get('title', f"Item {item_id}")

        # Sample ONE knowledge point from Stage 1
        stage1_kps = item['knowledge_points']
        stage1_sampled = random.choice(stage1_kps) if stage1_kps else None

        # Get Phase 3 knowledge and sample ONE
        stage3_kps = phase3_data.get(item_id, [])
        stage3_sampled = random.choice(stage3_kps) if stage3_kps else None

        eval_item = {
            'dataset': dataset_name,
            'item_id': item_id,
            'title': title,
            'stage1_knowledge': stage1_sampled,
            'stage3_knowledge': stage3_sampled
        }

        # Add asin for Amazon datasets
        if 'asin' in item:
            eval_item['asin'] = item['asin']

        eval_data.append(eval_item)

    return eval_data


def generate_llm_eval_prompts(eval_data: List[dict], output_path: Path):
    """Generate prompts for LLM evaluation."""
    prompts = []

    for item in eval_data:
        # Stage 1 evaluation prompt (single knowledge point)
        kp = item['stage1_knowledge']
        if kp:
            prompts.append({
                'dataset': item['dataset'],
                'item_id': item['item_id'],
                'title': item['title'],
                'asin': item.get('asin', ''),
                'stage': 'stage1',
                'relation': kp['relation'],
                'entity': kp['entity'],
                'prompt': f"Looking at the image of \"{item['title']}\", does this item have the visual attribute: {kp['relation']} = {kp['entity']}? Answer Yes or No only."
            })

        # Stage 3 evaluation prompt (single knowledge point)
        kp = item['stage3_knowledge']
        if kp:
            prompts.append({
                'dataset': item['dataset'],
                'item_id': item['item_id'],
                'title': item['title'],
                'asin': item.get('asin', ''),
                'stage': 'stage3',
                'relation': kp['relation'],
                'entity': kp['entity'],
                'prompt': f"Looking at the image of \"{item['title']}\", does this item have the visual attribute: {kp['relation']} = {kp['entity']}? Answer Yes or No only."
            })

    with open(output_path, 'w') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(prompts)} LLM evaluation prompts -> {output_path}")
    return prompts


def generate_human_eval_csv(eval_data: List[dict], output_path: Path):
    """Generate CSV for human evaluation."""
    import csv

    rows = []
    for item in eval_data:
        # Stage 1 (single knowledge point)
        kp = item['stage1_knowledge']
        if kp:
            rows.append({
                'dataset': item['dataset'],
                'item_id': item['item_id'],
                'asin': item.get('asin', ''),
                'title': item['title'],
                'stage': 'stage1',
                'relation': kp['relation'],
                'entity': kp['entity'],
                'statement': f"This item has {kp['relation']}: {kp['entity']}",
                'correct': ''  # To be filled by human evaluator
            })

        # Stage 3 (single knowledge point)
        kp = item['stage3_knowledge']
        if kp:
            rows.append({
                'dataset': item['dataset'],
                'item_id': item['item_id'],
                'asin': item.get('asin', ''),
                'title': item['title'],
                'stage': 'stage3',
                'relation': kp['relation'],
                'entity': kp['entity'],
                'statement': f"This item has {kp['relation']}: {kp['entity']}",
                'correct': ''
            })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'item_id', 'asin', 'title', 'stage', 'relation', 'entity', 'statement', 'correct'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} human evaluation rows -> {output_path}")
    return rows


def print_statistics(eval_data: List[dict], dataset_name: str):
    """Print statistics for evaluation data."""
    stage1_count = sum(1 for item in eval_data if item['stage1_knowledge'])
    stage3_count = sum(1 for item in eval_data if item['stage3_knowledge'])

    # Count items with both stages
    both_stages = sum(1 for item in eval_data if item['stage1_knowledge'] and item['stage3_knowledge'])

    print(f"\n{dataset_name}:")
    print(f"  Sampled items: {len(eval_data)}")
    print(f"  Items with Stage 1 knowledge: {stage1_count}")
    print(f"  Items with Stage 3 knowledge: {stage3_count}")
    print(f"  Items with both stages: {both_stages}")

    # Relation distribution for Stage 1
    stage1_relations = defaultdict(int)
    for item in eval_data:
        kp = item['stage1_knowledge']
        if kp:
            stage1_relations[kp['relation']] += 1

    print(f"  Stage 1 relation distribution: {dict(sorted(stage1_relations.items(), key=lambda x: -x[1])[:5])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='results/quality_eval')
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(__file__).parent.parent

    all_eval_data = []

    # === ML-1M ===
    print("Loading ML-1M data...")
    ml1m_phase1 = load_ml1m_phase1(base_path / 'results/phase1_5percent_exploration.json')
    ml1m_phase3 = load_ml1m_phase3(base_path / 'data/recbole/ml-1m/ml-1m.item.kg')
    ml1m_titles = load_ml1m_titles(base_path / 'data/recbole/ml-1m/ml-1m.item')

    ml1m_sampled = sample_items(ml1m_phase1, args.sample_size, args.seed)
    ml1m_eval = prepare_evaluation_data('ML-1M', ml1m_sampled, ml1m_phase3, ml1m_titles)
    print_statistics(ml1m_eval, 'ML-1M')
    all_eval_data.extend(ml1m_eval)

    # === Video Games ===
    print("\nLoading Video Games data...")
    vg_phase1 = load_videogames_phase1(base_path / 'results/videogames/phase1_5percent_exploration.json')
    vg_phase3 = load_videogames_phase3(base_path / 'results/phase3_full_extraction.json')

    vg_sampled = sample_items(vg_phase1, args.sample_size, args.seed)
    vg_eval = prepare_evaluation_data('Video Games', vg_sampled, vg_phase3)
    print_statistics(vg_eval, 'Video Games')
    all_eval_data.extend(vg_eval)

    # === Beauty ===
    print("\nLoading Beauty data...")
    beauty_phase1 = load_beauty_phase1(base_path / 'results/beauty/phase1_results.json')
    beauty_phase3 = load_beauty_phase3(base_path / 'results/beauty/full_extraction.json')

    beauty_sampled = sample_items(beauty_phase1, args.sample_size, args.seed)
    beauty_eval = prepare_evaluation_data('Beauty', beauty_sampled, beauty_phase3)
    print_statistics(beauty_eval, 'Beauty')
    all_eval_data.extend(beauty_eval)

    # === Save combined data ===
    with open(output_dir / 'sampled_items.json', 'w') as f:
        json.dump(all_eval_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved sampled items -> {output_dir / 'sampled_items.json'}")

    # === Generate evaluation files ===
    generate_llm_eval_prompts(all_eval_data, output_dir / 'llm_eval_prompts.json')
    generate_human_eval_csv(all_eval_data, output_dir / 'human_eval.csv')

    # === Summary ===
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Total sampled items: {len(all_eval_data)}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
