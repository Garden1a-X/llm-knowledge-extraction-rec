#!/usr/bin/env python3
"""
Phase 3: Validation & Expansion - Constrained Knowledge Extraction (Multi-threaded)

This script validates the standard vocabulary from Phase 2b by extracting knowledge
from 20% of movies (~800) using constrained extraction with concurrent API calls.

Features:
- Multi-threaded API calls for faster extraction
- Uses 165+ standard entities as vocabulary
- Allows NEW_ prefix for entities not in vocabulary
- Calculates coverage rate (target ≥90%)
- Includes Phase 1 movies for comprehensive coverage
- Incremental extraction with error recovery
- Thread-safe result saving
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import threading
from datetime import datetime
from typing import List, Dict, Set, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.extraction.poster_loader import PosterLoader
from src.extraction.mllm_interface import create_mllm
from src.extraction.prompts import PromptTemplates


def load_standard_vocabulary(vocab_file: Path) -> Dict[str, List[str]]:
    """
    Load standard vocabulary from Phase 2b results.

    Args:
        vocab_file: Path to standard_entity_vocabulary.json

    Returns:
        Dict mapping relation to list of standard entities
    """
    with open(vocab_file, 'r') as f:
        data = json.load(f)

    # Extract only standard entities (exclude noise entities)
    vocabulary = {}
    for relation, info in data['vocabulary'].items():
        vocabulary[relation] = info['standard_entities']

    return vocabulary


def load_phase1_ids(phase1_file: Path) -> Set[int]:
    """
    Load RecBole IDs from Phase 1 results.

    Args:
        phase1_file: Path to Phase 1 results JSON

    Returns:
        Set of RecBole IDs from Phase 1
    """
    if not phase1_file.exists():
        print(f"Warning: Phase 1 file not found: {phase1_file}")
        return set()

    with open(phase1_file, 'r') as f:
        data = json.load(f)

    # Get all RecBole IDs from Phase 1 results
    phase1_ids = {r['recbole_id'] for r in data.get('results', [])}
    print(f"  Found {len(phase1_ids)} movies from Phase 1")

    return phase1_ids


def load_existing_results(result_file: Path, skip_errors: bool = True) -> tuple[Dict, Set[int]]:
    """
    Load existing extraction results.

    Args:
        result_file: Path to existing results JSON
        skip_errors: If True, skip all processed IDs (success + error).
                     If False, only skip successfully extracted IDs (will retry errors).

    Returns:
        (existing_data, processed_ids_set)
    """
    if not result_file.exists():
        return None, set()

    with open(result_file, 'r') as f:
        data = json.load(f)

    # Get IDs that were already processed
    if skip_errors:
        # Skip ALL processed IDs (both success and error)
        processed_ids = {r['recbole_id'] for r in data.get('results', [])}
    else:
        # Only skip successfully extracted IDs (will retry errors)
        processed_ids = {
            r['recbole_id'] for r in data.get('results', [])
            if r.get('status') == 'success'
        }

    return data, processed_ids


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


def save_results(output_file: Path, config: Dict, results: List[Dict], vocabulary_stats: Dict = None):
    """
    Save extraction results to JSON file immediately (thread-safe).

    Args:
        output_file: Path to output JSON file
        config: Configuration dict
        results: List of extraction results
        vocabulary_stats: Optional vocabulary coverage statistics
    """
    output_data = {
        'phase': 'phase3_validation',
        'percentage': config.get('percentage', 20.0),
        'config': config,
        'results': results
    }

    if vocabulary_stats:
        output_data['vocabulary_stats'] = vocabulary_stats

    # Write atomically: write to temp file first, then rename
    temp_file = output_file.with_suffix('.tmp.json')
    with open(temp_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Atomic rename (overwrites existing file)
    temp_file.replace(output_file)


def calculate_coverage_stats(results: List[Dict], vocabulary: Dict[str, List[str]]) -> Dict:
    """
    Calculate vocabulary coverage statistics with validation.

    Args:
        results: List of extraction results
        vocabulary: Standard vocabulary dict

    Returns:
        Dict with coverage statistics including validation errors
    """
    total_kps = 0
    new_kps = 0
    invalid_kps = 0
    new_entities = []  # List of NEW_ entities
    invalid_entities = []  # List of entities not in vocabulary (validation errors)

    for result in results:
        if result.get('status') != 'success':
            continue

        for kp in result.get('knowledge_points', []):
            total_kps += 1
            relation = kp['relation']
            entity = kp['entity']

            if entity.startswith('NEW_'):
                new_kps += 1
                new_entities.append({
                    'relation': relation,
                    'entity': entity,
                    'recbole_id': result['recbole_id']
                })
            else:
                # Validate: check if entity is actually in the vocabulary for this relation
                if relation in vocabulary:
                    if entity not in vocabulary[relation]:
                        invalid_kps += 1
                        invalid_entities.append({
                            'relation': relation,
                            'entity': entity,
                            'recbole_id': result['recbole_id'],
                            'reason': f'Entity "{entity}" not in vocabulary for relation "{relation}"'
                        })
                else:
                    # Relation itself is invalid
                    invalid_kps += 1
                    invalid_entities.append({
                        'relation': relation,
                        'entity': entity,
                        'recbole_id': result['recbole_id'],
                        'reason': f'Invalid relation "{relation}"'
                    })

    valid_kps = total_kps - new_kps - invalid_kps
    coverage_rate = valid_kps / total_kps if total_kps > 0 else 0.0

    return {
        'total_knowledge_points': total_kps,
        'valid_entities': valid_kps,
        'new_entities_count': new_kps,
        'invalid_entities_count': invalid_kps,
        'coverage_rate': coverage_rate,
        'coverage_percentage': coverage_rate * 100,
        'new_entities': new_entities,
        'invalid_entities': invalid_entities,
        'target_coverage': 90.0,
        'meets_target': coverage_rate >= 0.90
    }


def extract_single_item(
    recbole_id: int,
    poster_loader: PosterLoader,
    mllm,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int
) -> Dict:
    """
    Extract knowledge from a single movie poster (to be called in parallel).

    Args:
        recbole_id: RecBole ID to process
        poster_loader: PosterLoader instance
        mllm: MLLM interface
        system_prompt: System prompt with vocabulary
        user_prompt: User prompt
        temperature: Sampling temperature
        max_tokens: Max tokens

    Returns:
        Result dict with extraction or error info
    """
    try:
        # Get movie info
        original_id = poster_loader.get_original_movie_id(recbole_id)

        # Load poster
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

        return result

    except Exception as e:
        # Log error
        result = {
            'recbole_id': recbole_id,
            'original_movie_id': poster_loader.get_original_movie_id(recbole_id),
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        return result


def extract_knowledge_concurrent(
    poster_loader: PosterLoader,
    mllm,
    recbole_ids: List[int],
    output_file: Path,
    config: Dict,
    vocabulary: Dict[str, List[str]],
    existing_results: List[Dict] = None,
    skip_processed: Set[int] = None,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    max_workers: int = 10,
    save_interval: int = 10,
    verbose: bool = True
) -> List[Dict]:
    """
    Extract knowledge concurrently with multi-threading for API calls.

    Args:
        poster_loader: PosterLoader instance
        mllm: MLLM interface
        recbole_ids: List of RecBole IDs to process
        output_file: Path to output JSON file (saves periodically)
        config: Configuration dict to save with results
        vocabulary: Standard vocabulary dict
        existing_results: Previous results to preserve
        skip_processed: Set of IDs to skip (already processed successfully or errors to skip)
        temperature: Sampling temperature (default 0.0 for consistency)
        max_tokens: Max tokens
        max_workers: Number of concurrent threads (default 10)
        save_interval: Save results every N completions (default 10)
        verbose: Show progress

    Returns:
        Combined list of all results
    """
    skip_processed = skip_processed or set()

    # Determine what to extract: IDs not in skip_processed set
    to_extract = [rid for rid in recbole_ids if rid not in skip_processed]
    to_extract_set = set(to_extract)

    # Keep only existing results that we're NOT going to reprocess
    if existing_results:
        results = [r for r in existing_results if r['recbole_id'] not in to_extract_set]
    else:
        results = []

    if verbose:
        print(f"\nExtraction status:")
        print(f"  Total items: {len(recbole_ids)}")
        print(f"  Already processed (kept): {len(skip_processed)}")
        print(f"  To extract/retry: {len(to_extract)}")
        print(f"  Concurrent workers: {max_workers}")

    if not to_extract:
        print("  ✓ All items already extracted!")
        return results

    # Get prompts with full vocabulary
    system_prompt = PromptTemplates.get_phase3_system_prompt(vocabulary)
    user_prompt = PromptTemplates.get_phase3_user_prompt()

    if verbose:
        vocab_size = sum(len(ents) for ents in vocabulary.values())
        print(f"\n  Using {len(vocabulary)} relations, {vocab_size} standard entities")

    # Thread-safe lock for result list
    results_lock = threading.Lock()
    completed_count = 0

    # Process items concurrently
    if verbose:
        progress_bar = tqdm(total=len(to_extract), desc="Extracting")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
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

        # Collect results as they complete
        for future in as_completed(future_to_id):
            recbole_id = future_to_id[future]

            try:
                result = future.result()

                # Thread-safe append
                with results_lock:
                    results.append(result)
                    completed_count += 1

                    # Log progress
                    if verbose:
                        if result['status'] == 'success':
                            kps = result['num_knowledge_points']
                            new_count = sum(1 for kp in result['knowledge_points']
                                          if kp['entity'].startswith('NEW_'))
                            progress_bar.write(
                                f"  ✓ RecBole ID {recbole_id} (Movie {result['original_movie_id']}): "
                                f"{kps} KPs ({new_count} NEW)"
                            )
                        else:
                            progress_bar.write(
                                f"  ✗ RecBole ID {recbole_id}: Error - {result.get('error', 'Unknown')}"
                            )
                        progress_bar.update(1)

                    # Periodic save
                    if completed_count % save_interval == 0:
                        coverage_stats = calculate_coverage_stats(results, vocabulary)
                        save_results(output_file, config, results, coverage_stats)
                        if verbose:
                            progress_bar.write(
                                f"  💾 Saved checkpoint: {completed_count}/{len(to_extract)} completed "
                                f"(Coverage: {coverage_stats['coverage_percentage']:.1f}%)"
                            )

            except Exception as e:
                # Should not happen as extract_single_item catches all exceptions
                if verbose:
                    progress_bar.write(f"  ✗ RecBole ID {recbole_id}: Unexpected error - {e}")

    if verbose:
        progress_bar.close()

    # Final save
    coverage_stats = calculate_coverage_stats(results, vocabulary)
    save_results(output_file, config, results, coverage_stats)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Phase 3: Validation & Expansion - Constrained Knowledge Extraction (Multi-threaded)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data paths
    parser.add_argument('--poster_dir', type=str, required=True,
                        help='Directory containing posters')
    parser.add_argument('--id_mapping', type=str, required=True,
                        help='Path to id_mappings.json')
    parser.add_argument('--vocabulary', type=str,
                        default='results/standard_entity_vocabulary.json',
                        help='Path to standard vocabulary JSON from Phase 2b')
    parser.add_argument('--phase1_results', type=str,
                        default='results/phase1_5percent_exploration.json',
                        help='Path to Phase 1 results (to include those IDs)')

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
                        help='Sampling temperature (0.0 for consistency)')
    parser.add_argument('--max_tokens', type=int, default=1000,
                        help='Max tokens')

    # Concurrency
    parser.add_argument('--workers', type=int, default=10,
                        help='Number of concurrent API workers (default: 10, recommended: 10-20)')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save results every N completions (default: 10)')

    # Sampling
    parser.add_argument('--percentage', type=float, default=20.0,
                        help='Percentage of movies to sample (default: 20.0 for ~800 movies)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resample', action='store_true',
                        help='Force resampling (ignore saved sample IDs)')
    parser.add_argument('--retry-errors', action='store_true',
                        help='Retry previously failed extractions (default: skip all processed IDs)')

    # Output
    parser.add_argument('--output', type=str, default='results/phase3_20percent_validation.json',
                        help='Output JSON file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Initialize
    print("="*60)
    print("PHASE 3: VALIDATION & EXPANSION (MULTI-THREADED)")
    print("="*60)
    print(f"\nBackend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature} (for consistency)")
    print(f"Percentage: {args.percentage}%")
    print(f"Concurrent workers: {args.workers}")
    print()

    # Load vocabulary
    vocab_file = Path(args.vocabulary)
    if not vocab_file.exists():
        print(f"Error: Vocabulary file not found: {vocab_file}")
        print("Please run Phase 2b first to generate standard vocabulary.")
        return 1

    print(f"Loading vocabulary from: {vocab_file}")
    vocabulary = load_standard_vocabulary(vocab_file)
    vocab_size = sum(len(ents) for ents in vocabulary.values())
    print(f"  ✓ Loaded {len(vocabulary)} relations, {vocab_size} standard entities")
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

    # Load Phase 1 IDs to ensure they're included
    phase1_file = Path(args.phase1_results)
    phase1_ids = load_phase1_ids(phase1_file)

    # Load or create sample IDs
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not args.resample:
        saved_ids = load_sampled_ids(output_file)
        if saved_ids:
            print(f"\n✓ Using previously sampled {len(saved_ids)} IDs")
            recbole_ids = saved_ids
        else:
            print(f"\nSampling {num_samples} movies (seed={args.seed})...")
            # Sample new IDs, excluding Phase 1 IDs
            new_samples_needed = num_samples - len(phase1_ids)
            if new_samples_needed > 0:
                # Get all available IDs and exclude Phase 1
                all_ids = poster_loader.get_all_item_ids()
                available_ids = [rid for rid in all_ids if rid not in phase1_ids]

                import random
                random.seed(args.seed)
                new_samples = random.sample(available_ids, min(new_samples_needed, len(available_ids)))

                # Combine Phase 1 IDs with new samples
                recbole_ids = list(phase1_ids) + new_samples
                print(f"  Phase 1 IDs: {len(phase1_ids)}")
                print(f"  New samples: {len(new_samples)}")
                print(f"  Total: {len(recbole_ids)}")
            else:
                recbole_ids = list(phase1_ids)
                print(f"  Using all {len(phase1_ids)} Phase 1 IDs (target reached)")

            save_sampled_ids(recbole_ids, output_file)
    else:
        print(f"\nForce resampling {num_samples} movies (seed={args.seed})...")
        new_samples_needed = num_samples - len(phase1_ids)
        if new_samples_needed > 0:
            all_ids = poster_loader.get_all_item_ids()
            available_ids = [rid for rid in all_ids if rid not in phase1_ids]

            import random
            random.seed(args.seed)
            new_samples = random.sample(available_ids, min(new_samples_needed, len(available_ids)))

            recbole_ids = list(phase1_ids) + new_samples
            print(f"  Phase 1 IDs: {len(phase1_ids)}")
            print(f"  New samples: {len(new_samples)}")
            print(f"  Total: {len(recbole_ids)}")
        else:
            recbole_ids = list(phase1_ids)

        save_sampled_ids(recbole_ids, output_file)

    # Load existing results
    skip_errors = not args.retry_errors
    existing_data, skip_processed = load_existing_results(output_file, skip_errors=skip_errors)

    if skip_errors and skip_processed:
        print(f"\n  Skipping all {len(skip_processed)} previously processed IDs (use --retry-errors to retry failed ones)")
    elif not skip_errors and skip_processed:
        print(f"\n  Skipping {len(skip_processed)} successfully extracted IDs, will retry errors")

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
        'vocabulary_file': str(vocab_file),
        'num_relations': len(vocabulary),
        'num_standard_entities': vocab_size,
        'concurrent_workers': args.workers,
        'save_interval': args.save_interval,
        'timestamp': datetime.now().isoformat()
    }

    # Extract knowledge (concurrent with periodic auto-save)
    print(f"\nStarting concurrent extraction...")
    print("-"*60)

    results = extract_knowledge_concurrent(
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

    # Calculate final coverage statistics
    coverage_stats = calculate_coverage_stats(results, vocabulary)

    # Results already saved incrementally, no need to save again here

    # Summary
    print("-"*60)
    print("\nSUMMARY")
    print("="*60)

    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    total_kps = coverage_stats['total_knowledge_points']
    avg_kps = total_kps / successful if successful > 0 else 0

    print(f"Total sampled: {len(recbole_ids)}")
    print(f"Total results: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()
    print("VOCABULARY COVERAGE:")
    print(f"  Total knowledge points: {total_kps}")
    print(f"  Valid matched entities: {coverage_stats['valid_entities']}")
    print(f"  NEW entities: {coverage_stats['new_entities_count']}")
    print(f"  Invalid entities: {coverage_stats['invalid_entities_count']}")
    print(f"  Coverage rate: {coverage_stats['coverage_percentage']:.2f}%")
    print(f"  Target: {coverage_stats['target_coverage']}%")
    print(f"  Meets target: {'✓ YES' if coverage_stats['meets_target'] else '✗ NO'}")
    print()

    if coverage_stats['invalid_entities_count'] > 0:
        print(f"⚠ WARNING: {coverage_stats['invalid_entities_count']} invalid entities detected!")
        print("  LLM is not strictly following the vocabulary.")
        print(f"  Invalid rate: {100 * coverage_stats['invalid_entities_count'] / total_kps:.2f}%")
        print()

    if not coverage_stats['meets_target']:
        print("⚠ Coverage below 90% - vocabulary expansion needed (v1 → v2)")
        print(f"  {coverage_stats['new_entities_count']} NEW entities need to be clustered and added")
    else:
        print("✓ Coverage meets 90% target - vocabulary v1 is sufficient!")
        print("  Ready to proceed to Phase 4 (full extraction)")

    print()
    print(f"Average KPs per movie: {avg_kps:.1f}")
    print(f"Results saved to: {output_file}")
    print("="*60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit(main())
