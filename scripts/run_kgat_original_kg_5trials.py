#!/usr/bin/env python3
"""
Run KGAT with original ML-1M KG (genres + year) for 5 trials.
Uses uni100 evaluation mode for fair comparison.
"""

import sys
import os
import argparse
from pathlib import Path

# Add RecBole to path if needed
try:
    from recbole.quick_start import run_recbole
except ImportError:
    print("Error: RecBole not found. Please install RecBole first.")
    sys.exit(1)


def run_kgat_trial(data_path: str, seed: int, output_dir: Path):
    """
    Run single KGAT trial with original ML-1M KG.

    Args:
        data_path: Path to dataset directory
        seed: Random seed
        output_dir: Output directory for this trial
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config_dict = {
        # Data
        'data_path': data_path,
        'dataset': 'ml-1m',
        'load_col': {
            'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
            'kg': ['head_id', 'relation_id', 'tail_id'],
            'link': ['item_id', 'entity_id']
        },

        # Data split (same as ours)
        'eval_args': {
            'split': {'RS': [0.7, 0.1, 0.2]},
            'order': 'TO',
            'group_by': 'user',
            'mode': 'uni100'  # Changed from 'full' to 'uni100'
        },

        # Evaluation metrics
        'metrics': ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': [5, 10, 20],
        'valid_metric': 'NDCG@10',

        # KGAT parameters
        'embedding_size': 64,
        'kg_embedding_size': 64,
        'reg_weight': 0.0001,

        # Training
        'epochs': 300,
        'train_batch_size': 2048,
        'learning_rate': 0.001,
        'stopping_step': 10,

        # Random seed
        'seed': seed,

        # Reproducibility
        'reproducibility': True,
        'state': 'INFO',

        # Checkpoint
        'checkpoint_dir': str(output_dir / 'checkpoints'),
        'show_progress': True,
    }

    print(f"\n{'='*70}")
    print(f"Running KGAT Trial (seed={seed})")
    print(f"{'='*70}")
    print(f"Data path: {data_path}")
    print(f"Evaluation mode: uni100")
    print(f"Output directory: {output_dir}")
    print()

    # Run RecBole
    result = run_recbole(
        model='KGAT',
        dataset='ml-1m',
        config_dict=config_dict,
        saved=True
    )

    print(f"\n{'='*70}")
    print(f"Trial (seed={seed}) completed!")
    print(f"{'='*70}\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run KGAT with original ML-1M KG for 5 trials'
    )
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset directory (e.g., /data/xuao/KG4RecEval/dataset/)')
    parser.add_argument('--output_base', type=str, required=True,
                       help='Base output directory (e.g., outputs/baselines/kgat_original_kg)')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 2023, 2024, 2025, 12345],
                       help='Random seeds for trials (default: 42 2023 2024 2025 12345)')

    args = parser.parse_args()

    data_path = args.data_path
    output_base = Path(args.output_base)
    seeds = args.seeds

    print("="*70)
    print("KGAT with Original ML-1M KG - 5 Trials")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data path: {data_path}")
    print(f"  Output base: {output_base}")
    print(f"  Seeds: {seeds}")
    print(f"  Evaluation mode: uni100")
    print(f"  KG source: Original ML-1M metadata (genres + year)")
    print()

    # Verify data path exists
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"Error: Data path not found: {data_dir}")
        return

    # Check required files
    dataset_dir = data_dir / 'ml-1m'
    required_files = ['ml-1m.inter', 'ml-1m.kg', 'ml-1m.link']
    missing_files = []

    for filename in required_files:
        filepath = dataset_dir / filename
        if not filepath.exists():
            missing_files.append(str(filepath))

    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure all required files exist:")
        print("  - ml-1m.inter (user-item interactions)")
        print("  - ml-1m.kg (knowledge graph)")
        print("  - ml-1m.link (item-entity mapping)")
        return

    print("✓ All required files found\n")

    # Run trials
    results = []

    for i, seed in enumerate(seeds, 1):
        print(f"\n{'#'*70}")
        print(f"# Trial {i}/{len(seeds)}: seed={seed}")
        print(f"{'#'*70}\n")

        # Create output directory for this trial
        trial_output = output_base / f"trial_{i}_seed_{seed}"

        try:
            result = run_kgat_trial(
                data_path=data_path,
                seed=seed,
                output_dir=trial_output
            )
            results.append({
                'trial': i,
                'seed': seed,
                'result': result,
                'status': 'success'
            })

        except Exception as e:
            print(f"\n✗ Trial {i} failed with error:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'trial': i,
                'seed': seed,
                'error': str(e),
                'status': 'failed'
            })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful

    print(f"\nTotal trials: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if successful > 0:
        print(f"\nResults saved to: {output_base}")
        print("\nNext step: Use summarize_baseline_trials.py to generate summary JSON")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
