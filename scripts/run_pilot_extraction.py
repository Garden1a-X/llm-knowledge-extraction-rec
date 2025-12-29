#!/usr/bin/env python3
"""
Pilot script for Phase 1 knowledge extraction.

This script runs a small-scale test of the knowledge extraction pipeline:
1. Sample a subset of movies
2. Load posters using ID mapping
3. Extract visual knowledge using MLLM
4. Save results to JSON

Usage:
    python scripts/run_pilot_extraction.py \
        --poster_dir data/raw/ml-1m/posters \
        --id_mapping data/recbole/ml-1m/id_mappings.json \
        --backend openai \
        --model gpt-4o-mini \
        --num_samples 10 \
        --output results/pilot_extraction.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm

from src.extraction.poster_loader import PosterLoader
from src.extraction.mllm_interface import create_mllm
from src.extraction.prompts import PromptTemplates


def extract_knowledge_batch(
    poster_loader: PosterLoader,
    mllm,
    recbole_ids: List[int],
    movie_titles: Dict[int, str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    verbose: bool = True
) -> List[Dict]:
    """
    Extract knowledge from a batch of movies.

    Args:
        poster_loader: PosterLoader instance
        mllm: MLLM interface instance
        recbole_ids: List of RecBole item IDs to process
        movie_titles: Optional dict mapping recbole_id to movie title
        temperature: Sampling temperature
        max_tokens: Max tokens in response
        verbose: Show progress bar

    Returns:
        List of extraction results
    """
    results = []
    movie_titles = movie_titles or {}

    # Get prompts
    system_prompt = PromptTemplates.get_phase1_system_prompt()

    # Process each movie
    iterator = tqdm(recbole_ids, desc="Extracting") if verbose else recbole_ids

    for recbole_id in iterator:
        try:
            # Get movie info
            original_id = poster_loader.get_original_movie_id(recbole_id)
            movie_title = movie_titles.get(recbole_id, f"Movie {original_id}")

            # Load poster
            poster = poster_loader.load_poster(recbole_id, max_size=(1024, 1536))

            # Get user prompt
            user_prompt = PromptTemplates.get_phase1_user_prompt(movie_title)

            # Extract knowledge
            output = mllm.extract_from_image(
                image=poster,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Parse output
            knowledge_points = PromptTemplates.parse_extraction_output(output)

            # Store result
            result = {
                'recbole_id': recbole_id,
                'original_movie_id': original_id,
                'movie_title': movie_title,
                'num_knowledge_points': len(knowledge_points),
                'knowledge_points': knowledge_points,
                'raw_output': output,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }

            results.append(result)

            if verbose:
                print(f"  ✓ Movie {original_id}: {len(knowledge_points)} knowledge points")

        except Exception as e:
            # Log error but continue
            result = {
                'recbole_id': recbole_id,
                'original_movie_id': poster_loader.get_original_movie_id(recbole_id),
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)

            if verbose:
                print(f"  ✗ Movie {recbole_id}: Error - {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run pilot knowledge extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data paths
    parser.add_argument('--poster_dir', type=str, required=True,
                        help='Directory containing posters')
    parser.add_argument('--id_mapping', type=str, required=True,
                        help='Path to id_mappings.json')

    # MLLM settings
    parser.add_argument('--backend', type=str, default='openai',
                        choices=['openai', 'local'],
                        help='MLLM backend')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Model name')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key (if None, read from env)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=1000,
                        help='Max tokens in response')

    # Sampling
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of movies to sample')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    parser.add_argument('--sample_ids', type=str, default=None,
                        help='Comma-separated RecBole IDs (overrides sampling)')

    # Output
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Initialize poster loader
    print("="*60)
    print("PILOT KNOWLEDGE EXTRACTION")
    print("="*60)
    print(f"\nBackend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"Num samples: {args.num_samples}")
    print()

    poster_loader = PosterLoader(
        poster_dir=args.poster_dir,
        id_mapping_path=args.id_mapping
    )

    # Initialize MLLM
    print(f"\nInitializing {args.backend} MLLM...")
    mllm = create_mllm(
        backend=args.backend,
        model_name=args.model,
        api_key=args.api_key
    )

    # Sample movies
    if args.sample_ids:
        recbole_ids = [int(x.strip()) for x in args.sample_ids.split(',')]
        print(f"\nUsing specified RecBole IDs: {recbole_ids}")
    else:
        print(f"\nSampling {args.num_samples} random movies (seed={args.seed})...")
        recbole_ids = poster_loader.sample_items(args.num_samples, seed=args.seed)
        print(f"Sampled IDs: {recbole_ids}")

    # Extract knowledge
    print(f"\nStarting extraction...")
    print("-"*60)

    results = extract_knowledge_batch(
        poster_loader=poster_loader,
        mllm=mllm,
        recbole_ids=recbole_ids,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        verbose=not args.quiet
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'config': {
            'backend': args.backend,
            'model': args.model,
            'temperature': args.temperature,
            'max_tokens': args.max_tokens,
            'num_samples': len(recbole_ids),
            'seed': args.seed,
            'timestamp': datetime.now().isoformat()
        },
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Summary
    print("-"*60)
    print("\nSUMMARY")
    print("="*60)

    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    total_kps = sum(r.get('num_knowledge_points', 0) for r in results if r['status'] == 'success')
    avg_kps = total_kps / successful if successful > 0 else 0

    print(f"Total movies: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total knowledge points: {total_kps}")
    print(f"Average KPs per movie: {avg_kps:.1f}")
    print()
    print(f"Results saved to: {output_path}")
    print("="*60)

    # Show sample extraction
    if successful > 0:
        print("\nSample extraction (first successful result):")
        print("-"*60)
        sample = next(r for r in results if r['status'] == 'success')
        print(f"Movie: {sample.get('movie_title', sample['original_movie_id'])}")
        print(f"Knowledge points ({sample['num_knowledge_points']}):")
        for kp in sample['knowledge_points'][:10]:
            print(f"  - {kp['relation']}: {kp['entity']}")
        if sample['num_knowledge_points'] > 10:
            print(f"  ... and {sample['num_knowledge_points'] - 10} more")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit(main())
