#!/usr/bin/env python3
"""
Phase 1: 5% Exploration - Knowledge Extraction

This script extracts visual knowledge from 5% of movies (Phase 1).
Features:
- Saves sampled IDs for reproducibility
- Incremental extraction: skips already successfully extracted items
- Clear naming: phase1_5percent_exploration.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from datetime import datetime
from typing import List, Dict, Set
from tqdm import tqdm

from src.extraction.poster_loader import PosterLoader
from src.extraction.mllm_interface import create_mllm
from src.extraction.prompts import PromptTemplates


def load_existing_results(result_file: Path) -> tuple[Dict, Set[int]]:
    """
    Load existing extraction results.

    Args:
        result_file: Path to existing results JSON

    Returns:
        (existing_data, extracted_ids_set)
    """
    if not result_file.exists():
        return None, set()

    with open(result_file, 'r') as f:
        data = json.load(f)

    # Get IDs that were successfully extracted
    extracted_ids = {
        r['recbole_id'] for r in data.get('results', [])
        if r.get('status') == 'success'
    }

    return data, extracted_ids


def save_sampled_ids(ids: List[int], output_file: Path):
    """Save sampled IDs to file for reproducibility."""
    id_file = output_file.parent / f"{output_file.stem}_sampled_ids.json"
    with open(id_file, 'w') as f:
        json.dump({
            'sampled_ids': ids,
            'num_samples': len(ids),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"  Sampled IDs saved to: {id_file}")


def load_sampled_ids(output_file: Path) -> List[int]:
    """Load previously sampled IDs if exist."""
    id_file = output_file.parent / f"{output_file.stem}_sampled_ids.json"
    if id_file.exists():
        with open(id_file, 'r') as f:
            data = json.load(f)
        return data['sampled_ids']
    return None


def save_results(output_file: Path, config: Dict, results: List[Dict]):
    """
    Save extraction results to JSON file immediately.

    Args:
        output_file: Path to output JSON file
        config: Configuration dict
        results: List of extraction results
    """
    output_data = {
        'phase': 'phase1_exploration',
        'percentage': config.get('percentage', 5.0),
        'config': config,
        'results': results
    }

    # Write atomically: write to temp file first, then rename
    temp_file = output_file.with_suffix('.tmp.json')
    with open(temp_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Atomic rename (overwrites existing file)
    temp_file.replace(output_file)


def extract_knowledge_incremental(
    poster_loader: PosterLoader,
    mllm,
    recbole_ids: List[int],
    output_file: Path,
    config: Dict,
    existing_results: List[Dict] = None,
    already_extracted: Set[int] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    verbose: bool = True
) -> List[Dict]:
    """
    Extract knowledge incrementally with real-time saving, skipping already extracted items.

    Args:
        poster_loader: PosterLoader instance
        mllm: MLLM interface
        recbole_ids: List of RecBole IDs to process
        output_file: Path to output JSON file (saves after each extraction)
        config: Configuration dict to save with results
        existing_results: Previous results to preserve
        already_extracted: Set of already extracted IDs
        temperature: Sampling temperature
        max_tokens: Max tokens
        verbose: Show progress

    Returns:
        Combined list of all results
    """
    already_extracted = already_extracted or set()
    results = list(existing_results) if existing_results else []

    # Filter out already extracted
    to_extract = [rid for rid in recbole_ids if rid not in already_extracted]

    if verbose:
        print(f"\nExtraction status:")
        print(f"  Total items: {len(recbole_ids)}")
        print(f"  Already extracted: {len(already_extracted)}")
        print(f"  To extract: {len(to_extract)}")

    if not to_extract:
        print("  ✓ All items already extracted!")
        return results

    # Get prompts
    system_prompt = PromptTemplates.get_phase1_system_prompt()

    # Process remaining items
    iterator = tqdm(to_extract, desc="Extracting") if verbose else to_extract

    for recbole_id in iterator:
        try:
            # Get movie info
            original_id = poster_loader.get_original_movie_id(recbole_id)

            # Load poster
            poster = poster_loader.load_poster(recbole_id, max_size=(1024, 1536))

            # Get user prompt (no movie title to avoid bias)
            user_prompt = PromptTemplates.get_phase1_user_prompt()

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
                'num_knowledge_points': len(knowledge_points),
                'knowledge_points': knowledge_points,
                'raw_output': output,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }

            results.append(result)

            # Save immediately after each successful extraction
            save_results(output_file, config, results)

            if verbose:
                tqdm.write(f"  ✓ RecBole ID {recbole_id} (Movie {original_id}): "
                          f"{len(knowledge_points)} knowledge points")

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

            # Save immediately after each error too
            save_results(output_file, config, results)

            if verbose:
                tqdm.write(f"  ✗ RecBole ID {recbole_id}: Error - {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: 5% Exploration Knowledge Extraction',
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
                        help='API key')
    parser.add_argument('--base_url', type=str, default=None,
                        help='Base URL for API')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=1000,
                        help='Max tokens')

    # Sampling
    parser.add_argument('--percentage', type=float, default=5.0,
                        help='Percentage of movies to sample (default: 5.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resample', action='store_true',
                        help='Force resampling (ignore saved sample IDs)')

    # Output
    parser.add_argument('--output', type=str, default='results/phase1_5percent_exploration.json',
                        help='Output JSON file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Initialize
    print("="*60)
    print("PHASE 1: EXPLORATION EXTRACTION")
    print("="*60)
    print(f"\nBackend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"Percentage: {args.percentage}%")
    print()

    poster_loader = PosterLoader(
        poster_dir=args.poster_dir,
        id_mapping_path=args.id_mapping
    )

    # Calculate number of samples
    total_items = poster_loader.stats['num_items']
    num_samples = int(total_items * args.percentage / 100)
    print(f"Total items: {total_items}")
    print(f"Target samples ({args.percentage}%): {num_samples}")

    # Load or create sample IDs
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not args.resample:
        saved_ids = load_sampled_ids(output_file)
        if saved_ids:
            print(f"\n✓ Using previously sampled {len(saved_ids)} IDs")
            recbole_ids = saved_ids
        else:
            print(f"\nSampling {num_samples} random movies (seed={args.seed})...")
            recbole_ids = poster_loader.sample_items(num_samples, seed=args.seed)
            save_sampled_ids(recbole_ids, output_file)
    else:
        print(f"\nForce resampling {num_samples} movies (seed={args.seed})...")
        recbole_ids = poster_loader.sample_items(num_samples, seed=args.seed)
        save_sampled_ids(recbole_ids, output_file)

    # Load existing results
    existing_data, already_extracted = load_existing_results(output_file)

    # Initialize MLLM
    print(f"\nInitializing {args.backend} MLLM...")
    mllm_kwargs = {
        'backend': args.backend,
        'model_name': args.model,
        'api_key': args.api_key
    }
    if args.base_url:
        mllm_kwargs['base_url'] = args.base_url
        print(f"Using custom base URL: {args.base_url}")

    mllm = create_mllm(**mllm_kwargs)

    # Prepare config for saving
    config = {
        'backend': args.backend,
        'model': args.model,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'num_samples': len(recbole_ids),
        'seed': args.seed,
        'percentage': args.percentage,
        'timestamp': datetime.now().isoformat()
    }

    # Extract knowledge (saves automatically after each item)
    print(f"\nStarting extraction...")
    print("-"*60)

    results = extract_knowledge_incremental(
        poster_loader=poster_loader,
        mllm=mllm,
        recbole_ids=recbole_ids,
        output_file=output_file,
        config=config,
        existing_results=existing_data.get('results', []) if existing_data else None,
        already_extracted=already_extracted,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        verbose=not args.quiet
    )

    # Results already saved incrementally, no need to save again here

    # Summary
    print("-"*60)
    print("\nSUMMARY")
    print("="*60)

    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    total_kps = sum(r.get('num_knowledge_points', 0) for r in results if r['status'] == 'success')
    avg_kps = total_kps / successful if successful > 0 else 0

    print(f"Total sampled: {len(recbole_ids)}")
    print(f"Total results: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total knowledge points: {total_kps}")
    print(f"Average KPs per movie: {avg_kps:.1f}")
    print()
    print(f"Results saved to: {output_file}")
    print("="*60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit(main())
