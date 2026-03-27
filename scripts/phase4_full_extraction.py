#!/usr/bin/env python3
"""
Phase 4: Full Production Extraction (Multi-threaded)

Final extraction for recommendation system using validated vocabulary v2.
- Simplified prompt without NEW_ mechanism
- Direct extraction using vocabulary as reference
- Post-processing filters to vocabulary-only entities
- Multi-threaded for fast processing

Usage:
    python scripts/phase4_full_extraction.py \
        --poster_dir /path/to/posters \
        --id_mapping data/recbole/ml-1m/id_mappings.json \
        --vocabulary results/standard_entity_vocabulary_v2.json \
        --api_key YOUR_KEY \
        --base_url YOUR_URL \
        --percentage 100.0 \
        --workers 15 \
        --output results/phase4_full_extraction.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import threading
from datetime import datetime
from typing import List, Dict, Set
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.extraction.poster_loader import PosterLoader
from src.extraction.mllm_interface import create_mllm
from src.extraction.prompts import PromptTemplates


def load_vocabulary(vocab_file: Path) -> Dict[str, List[str]]:
    """Load standard vocabulary from JSON file."""
    with open(vocab_file, 'r') as f:
        data = json.load(f)

    vocabulary = {}
    for relation, info in data['vocabulary'].items():
        vocabulary[relation] = info['standard_entities']

    return vocabulary


def load_existing_results(result_file: Path) -> tuple[Dict, Set[int]]:
    """Load existing extraction results and return processed IDs."""
    if not result_file.exists():
        return None, set()

    with open(result_file, 'r') as f:
        data = json.load(f)

    # Get successfully processed IDs only
    processed_ids = {
        r['recbole_id'] for r in data.get('results', [])
        if r.get('status') == 'success'
    }

    return data, processed_ids


def save_results(output_file: Path, config: Dict, results: List[Dict]):
    """Save extraction results to JSON file (thread-safe)."""
    output_data = {
        'phase': 'phase4_full_extraction',
        'config': config,
        'results': results,
        'total_movies': len(results),
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'error')
    }

    # Atomic write
    temp_file = output_file.with_suffix('.tmp.json')
    with open(temp_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    temp_file.replace(output_file)


def extract_single_item(
    recbole_id: int,
    poster_loader: PosterLoader,
    mllm,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int
) -> Dict:
    """Extract knowledge from a single poster."""
    try:
        original_id = poster_loader.get_original_movie_id(recbole_id)
        poster = poster_loader.load_poster(recbole_id, max_size=(1024, 1536))

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

        return {
            'recbole_id': recbole_id,
            'original_movie_id': original_id,
            'num_knowledge_points': len(knowledge_points),
            'knowledge_points': knowledge_points,
            'raw_output': output,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }

    except Exception as e:
        return {
            'recbole_id': recbole_id,
            'original_movie_id': poster_loader.get_original_movie_id(recbole_id),
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def extract_concurrent(
    poster_loader: PosterLoader,
    mllm,
    recbole_ids: List[int],
    output_file: Path,
    config: Dict,
    vocabulary: Dict[str, List[str]],
    existing_results: List[Dict] = None,
    skip_processed: Set[int] = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
    max_workers: int = 15,
    save_interval: int = 20,
    verbose: bool = True
) -> List[Dict]:
    """Extract knowledge concurrently with multi-threading."""
    skip_processed = skip_processed or set()

    # Filter IDs to process
    to_extract = [rid for rid in recbole_ids if rid not in skip_processed]
    to_extract_set = set(to_extract)

    # Keep existing results not being reprocessed
    if existing_results:
        results = [r for r in existing_results if r['recbole_id'] not in to_extract_set]
    else:
        results = []

    if verbose:
        print(f"\nExtraction status:")
        print(f"  Total items: {len(recbole_ids)}")
        print(f"  Already processed: {len(skip_processed)}")
        print(f"  To extract: {len(to_extract)}")
        print(f"  Concurrent workers: {max_workers}")

    if not to_extract:
        print("  ✓ All items already extracted!")
        return results

    # Get prompts (Phase 4 - simplified)
    system_prompt = PromptTemplates.get_phase4_system_prompt(vocabulary)
    user_prompt = PromptTemplates.get_phase4_user_prompt()

    if verbose:
        vocab_size = sum(len(ents) for ents in vocabulary.values())
        print(f"\n  Using {len(vocabulary)} relations, {vocab_size} entities")
        print(f"  Prompt: Phase 4 (simplified, no NEW_ mechanism)")

    # Thread-safe results
    results_lock = threading.Lock()
    completed_count = 0

    if verbose:
        progress_bar = tqdm(total=len(to_extract), desc="Extracting")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_id = {
            executor.submit(
                extract_single_item,
                recbole_id,
                poster_loader,
                mllm,
                system_prompt,
                user_prompt,
                temperature,
                max_tokens
            ): recbole_id
            for recbole_id in to_extract
        }

        # Collect results
        for future in as_completed(future_to_id):
            recbole_id = future_to_id[future]

            try:
                result = future.result()

                with results_lock:
                    results.append(result)
                    completed_count += 1

                    if verbose:
                        if result['status'] == 'success':
                            kps = result['num_knowledge_points']
                            progress_bar.write(
                                f"  ✓ ID {recbole_id}: {kps} KPs"
                            )
                        else:
                            progress_bar.write(
                                f"  ✗ ID {recbole_id}: {result.get('error', 'Unknown error')}"
                            )
                        progress_bar.update(1)

                    # Periodic save
                    if completed_count % save_interval == 0:
                        save_results(output_file, config, results)
                        if verbose:
                            success_rate = sum(1 for r in results if r['status'] == 'success') / len(results) * 100
                            progress_bar.write(
                                f"  💾 Checkpoint: {completed_count}/{len(to_extract)} "
                                f"(Success: {success_rate:.1f}%)"
                            )

            except Exception as e:
                if verbose:
                    progress_bar.write(f"  ✗ ID {recbole_id}: Unexpected error - {e}")

    if verbose:
        progress_bar.close()

    # Final save
    save_results(output_file, config, results)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Phase 4: Full Production Extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data paths
    parser.add_argument('--poster_dir', type=str, required=True,
                        help='Directory containing posters')
    parser.add_argument('--id_mapping', type=str, required=True,
                        help='Path to id_mappings.json')
    parser.add_argument('--vocabulary', type=str, required=True,
                        help='Path to vocabulary v2 JSON')

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
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=800,
                        help='Max tokens')

    # Concurrency
    parser.add_argument('--workers', type=int, default=15,
                        help='Number of concurrent workers')
    parser.add_argument('--save-interval', type=int, default=20,
                        help='Save checkpoint every N completions')

    # Sampling
    parser.add_argument('--percentage', type=float, default=100.0,
                        help='Percentage of movies to extract (100.0 for full dataset)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Initialize
    print("="*70)
    print("PHASE 4: FULL PRODUCTION EXTRACTION")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"Dataset: {args.percentage}% of movies")
    print(f"Workers: {args.workers}")
    print()

    # Load vocabulary
    vocab_file = Path(args.vocabulary)
    if not vocab_file.exists():
        print(f"Error: Vocabulary file not found: {vocab_file}")
        return 1

    print(f"Loading vocabulary: {vocab_file}")
    vocabulary = load_vocabulary(vocab_file)
    vocab_size = sum(len(ents) for ents in vocabulary.values())
    print(f"  ✓ {len(vocabulary)} relations, {vocab_size} entities")
    print()

    poster_loader = PosterLoader(
        poster_dir=args.poster_dir,
        id_mapping_path=args.id_mapping
    )

    # Calculate samples
    total_items = poster_loader.stats['num_items']
    num_samples = int(total_items * args.percentage / 100)
    print(f"Total available: {total_items}")
    print(f"Target extraction: {num_samples} movies ({args.percentage}%)")

    # Sample IDs
    import random
    random.seed(args.seed)
    all_ids = poster_loader.get_all_item_ids()
    recbole_ids = random.sample(all_ids, min(num_samples, len(all_ids)))
    print(f"  Sampled {len(recbole_ids)} IDs")

    # Load existing results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    existing_data, skip_processed = load_existing_results(output_file)
    if skip_processed:
        print(f"\n  Found {len(skip_processed)} already processed IDs")

    # Initialize MLLM
    print(f"\nInitializing {args.backend} MLLM...")
    mllm_kwargs = {
        'backend': args.backend,
        'model_name': args.model,
        'api_key': args.api_key
    }
    if args.base_url:
        mllm_kwargs['base_url'] = args.base_url

    mllm = create_mllm(**mllm_kwargs)

    # Config
    config = {
        'backend': args.backend,
        'model': args.model,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'vocabulary_file': str(vocab_file),
        'num_relations': len(vocabulary),
        'num_entities': vocab_size,
        'concurrent_workers': args.workers,
        'save_interval': args.save_interval,
        'percentage': args.percentage,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat()
    }

    # Extract
    print(f"\nStarting extraction...")
    print("-"*70)

    results = extract_concurrent(
        poster_loader=poster_loader,
        mllm=mllm,
        recbole_ids=recbole_ids,
        output_file=output_file,
        config=config,
        vocabulary=vocabulary,
        existing_results=existing_data.get('results', []) if existing_data else None,
        skip_processed=skip_processed,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.workers,
        save_interval=args.save_interval,
        verbose=not args.quiet
    )

    # Summary
    print("-"*70)
    print("\nSUMMARY")
    print("="*70)

    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    total_kps = sum(r.get('num_knowledge_points', 0) for r in results if r['status'] == 'success')
    avg_kps = total_kps / successful if successful > 0 else 0

    print(f"Total extracted: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    print()
    print(f"Total knowledge points: {total_kps}")
    print(f"Average KPs/movie: {avg_kps:.1f}")
    print()
    print(f"✓ Results saved to: {output_file}")
    print()
    print("Next step: Run post-processing to filter vocabulary-only entities")
    print("="*70)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit(main())
