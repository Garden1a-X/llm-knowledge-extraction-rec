#!/usr/bin/env python3
"""
Run LLM-based Knowledge Quality Evaluation using Claude

This script:
1. Loads evaluation prompts
2. For each prompt, sends the image + question to Claude
3. Parses Yes/No responses
4. Computes accuracy statistics

Usage:
    export ANTHROPIC_API_KEY='your-key'
    python scripts/run_llm_quality_evaluation.py --input results/quality_eval/llm_eval_prompts.json

Requirements:
    pip install anthropic
"""

import json
import base64
import argparse
import time
from pathlib import Path
from collections import defaultdict
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

try:
    import anthropic
except ImportError:
    print("Please install anthropic: pip install anthropic")
    exit(1)


# Image paths for each dataset
IMAGE_PATHS = {
    'ML-1M': '/data/xuao/llm-knowledge-extraction-rec/data/raw/ml-1m/posters',
    'Video Games': '/data/xuao/llm-knowledge-extraction-rec/data/recbole/amazon-videogames/images',
    'Beauty': '/data/xuao/llm-knowledge-extraction-rec/data/recbole/amazon-beauty/images'
}


def get_image_path(dataset: str, item_id: int, asin: str = '') -> Optional[Path]:
    """Get the image path for an item."""
    base_path = Path(IMAGE_PATHS.get(dataset, ''))

    if dataset == 'ML-1M':
        # ML-1M uses item_id.jpg
        for ext in ['.jpg', '.jpeg', '.png']:
            path = base_path / f"{item_id}{ext}"
            if path.exists():
                return path
    else:
        # Amazon datasets use asin.jpg
        if asin:
            for ext in ['.jpg', '.jpeg', '.png']:
                path = base_path / f"{asin}{ext}"
                if path.exists():
                    return path
    return None


def encode_image(image_path: Path) -> tuple:
    """Encode image to base64 and return with media type."""
    with open(image_path, 'rb') as f:
        data = base64.standard_b64encode(f.read()).decode('utf-8')

    suffix = image_path.suffix.lower()
    media_type = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
    }.get(suffix, 'image/jpeg')

    return data, media_type


def evaluate_single(client: anthropic.Anthropic, prompt_data: dict, model: str = "claude-sonnet-4-20250514") -> dict:
    """Evaluate a single knowledge point using Claude."""
    result = {
        'dataset': prompt_data['dataset'],
        'item_id': prompt_data['item_id'],
        'stage': prompt_data['stage'],
        'relation': prompt_data['relation'],
        'entity': prompt_data['entity'],
        'response': None,
        'judgment': None,
        'error': None
    }

    # Get image path
    image_path = get_image_path(
        prompt_data['dataset'],
        prompt_data['item_id'],
        prompt_data.get('asin', '')
    )

    if not image_path:
        result['error'] = 'Image not found'
        return result

    try:
        # Encode image
        image_data, media_type = encode_image(image_path)

        # Create prompt
        prompt = prompt_data['prompt']

        # Call Claude
        response = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        # Parse response
        response_text = response.content[0].text.strip().lower()
        result['response'] = response_text

        # Determine judgment
        if 'yes' in response_text:
            result['judgment'] = 'yes'
        elif 'no' in response_text:
            result['judgment'] = 'no'
        else:
            result['judgment'] = 'unclear'

    except Exception as e:
        result['error'] = str(e)

    return result


def run_evaluation(
    prompts: list,
    model: str = "claude-sonnet-4-20250514",
    max_workers: int = 5,
    delay: float = 0.5
) -> list:
    """Run evaluation on all prompts."""
    client = anthropic.Anthropic()
    results = []

    print(f"Running evaluation with model: {model}")
    print(f"Total prompts: {len(prompts)}")

    # Single-threaded for rate limiting
    for prompt in tqdm(prompts, desc="Evaluating"):
        result = evaluate_single(client, prompt, model)
        results.append(result)
        time.sleep(delay)  # Rate limiting

    return results


def compute_statistics(results: list) -> dict:
    """Compute accuracy statistics."""
    stats = {
        'by_dataset_stage': defaultdict(lambda: {'yes': 0, 'no': 0, 'unclear': 0, 'error': 0}),
        'overall': {'yes': 0, 'no': 0, 'unclear': 0, 'error': 0}
    }

    for r in results:
        key = (r['dataset'], r['stage'])

        if r['error']:
            stats['by_dataset_stage'][key]['error'] += 1
            stats['overall']['error'] += 1
        elif r['judgment'] == 'yes':
            stats['by_dataset_stage'][key]['yes'] += 1
            stats['overall']['yes'] += 1
        elif r['judgment'] == 'no':
            stats['by_dataset_stage'][key]['no'] += 1
            stats['overall']['no'] += 1
        else:
            stats['by_dataset_stage'][key]['unclear'] += 1
            stats['overall']['unclear'] += 1

    # Compute accuracy (% of 'yes' among valid responses)
    def calc_accuracy(counts):
        valid = counts['yes'] + counts['no']
        if valid == 0:
            return 0.0
        return counts['yes'] / valid * 100

    stats['accuracy'] = {}
    for key, counts in stats['by_dataset_stage'].items():
        dataset, stage = key
        stats['accuracy'][f"{dataset}_{stage}"] = calc_accuracy(counts)

    stats['accuracy']['overall'] = calc_accuracy(stats['overall'])

    return stats


def print_results_table(stats: dict):
    """Print results in table format for paper."""
    print("\n" + "=" * 80)
    print("Knowledge Quality Evaluation Results (LLM)")
    print("=" * 80)

    datasets = ['ML-1M', 'Video Games', 'Beauty']
    stages = ['stage1', 'stage3']

    # Header
    print(f"\n{'Stage':<10}", end='')
    for dataset in datasets:
        print(f"{dataset:>15}", end='')
    print(f"{'Average':>15}")

    print("-" * 70)

    # Rows
    for stage in stages:
        stage_label = 'Stage 1' if stage == 'stage1' else 'Stage 3'
        print(f"{stage_label:<10}", end='')

        accs = []
        for dataset in datasets:
            key = f"{dataset}_{stage}"
            acc = stats['accuracy'].get(key, 0)
            accs.append(acc)
            print(f"{acc:>14.1f}%", end='')

        avg = sum(accs) / len(accs) if accs else 0
        print(f"{avg:>14.1f}%")

    print("=" * 70)

    # LaTeX table format
    print("\n--- LaTeX Table Format ---")
    for stage in stages:
        stage_label = 'Stage 1' if stage == 'stage1' else 'Stage 3'
        row = [stage_label]
        for dataset in datasets:
            key = f"{dataset}_{stage}"
            acc = stats['accuracy'].get(key, 0)
            row.append(f"{acc:.1f}")
        avg = sum(float(x) for x in row[1:]) / len(datasets)
        row.append(f"{avg:.1f}")
        print(" & ".join(row) + " \\\\")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='results/quality_eval/llm_eval_prompts.json')
    parser.add_argument('--output', type=str, default='results/quality_eval/llm_eval_results.json')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-20250514')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between API calls (seconds)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of prompts (for testing)')
    args = parser.parse_args()

    # Check API key
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY not set")
        print("Please set: export ANTHROPIC_API_KEY='your-key'")
        return

    # Load prompts
    with open(args.input, 'r') as f:
        prompts = json.load(f)

    if args.limit:
        prompts = prompts[:args.limit]

    print(f"Loaded {len(prompts)} prompts from {args.input}")

    # Run evaluation
    results = run_evaluation(prompts, model=args.model, delay=args.delay)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Compute and print statistics
    stats = compute_statistics(results)
    print_results_table(stats)

    # Save statistics
    stats_path = output_path.with_suffix('.stats.json')
    # Convert defaultdict to regular dict for JSON serialization
    stats['by_dataset_stage'] = {str(k): v for k, v in stats['by_dataset_stage'].items()}
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")


if __name__ == '__main__':
    main()
