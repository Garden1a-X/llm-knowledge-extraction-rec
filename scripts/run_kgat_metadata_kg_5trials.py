#!/usr/bin/env python3
"""
Run KGAT with metadata KG for 5 trials.
Supports both ML-1M and Amazon Video Games datasets.
Uses uni100 evaluation mode for fair comparison.
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add RecBole to path if needed
try:
    from recbole.quick_start import run_recbole
except ImportError:
    print("Error: RecBole not found. Please install RecBole first.")
    sys.exit(1)


def run_kgat_trial(dataset: str, data_path: str, seed: int, output_dir: Path):
    """
    Run single KGAT trial with metadata KG.

    Args:
        dataset: Dataset name (e.g., 'ml-1m', 'amazon-videogames')
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
        'dataset': dataset,
        'load_col': {
            'inter': ['user_id', 'item_id', 'timestamp'],
            'kg': ['head_id', 'relation_id', 'tail_id'],
            'link': ['item_id', 'entity_id']
        },

        # Data split (same as ours)
        'eval_args': {
            'split': {'RS': [0.7, 0.1, 0.2]},
            'order': 'TO',
            'group_by': 'user',
            'mode': 'uni100'  # uni100 mode for fair comparison
        },

        # Evaluation metrics
        'metrics': ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': [5, 10, 20],
        'valid_metric': 'NDCG@10',

        # KGAT parameters (aligned with our baselines)
        'embedding_size': 64,
        'kg_embedding_size': 64,
        'reg_weight': 0.00001,  # Same as BPR/LightGCN

        # Training
        'epochs': 300,
        'train_batch_size': 2048,
        'eval_batch_size': 100000,  # Large batch for fast evaluation
        'learning_rate': 0.001,
        'stopping_step': 10,
        'eval_step': 1,

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
    print(f"Dataset: {dataset}")
    print(f"Data path: {data_path}")
    print(f"Evaluation mode: uni100")
    print(f"Eval batch size: 100000")
    print(f"Output directory: {output_dir}")
    print()

    # Run RecBole
    result = run_recbole(
        model='KGAT',
        dataset=dataset,
        config_dict=config_dict,
        saved=True
    )

    # Save results to JSON
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'dataset': dataset,
            'model': 'KGAT',
            'seed': seed,
            'config': {
                'embedding_size': 64,
                'kg_embedding_size': 64,
                'reg_weight': 0.00001,
                'epochs': 300,
                'train_batch_size': 2048,
                'eval_batch_size': 100000,
                'learning_rate': 0.001,
                'eval_mode': 'uni100'
            },
            'best_valid_result': result['best_valid_result'],
            'test_result': result['test_result']
        }, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    print(f"\n{'='*70}")
    print(f"Trial (seed={seed}) completed!")
    print(f"Test NDCG@10: {result['test_result']['ndcg@10']:.4f}")
    print(f"Test Recall@10: {result['test_result']['recall@10']:.4f}")
    print(f"{'='*70}\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run KGAT with metadata KG for 5 trials'
    )
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., ml-1m, amazon-videogames)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to RecBole dataset directory (e.g., data/recbole)')
    parser.add_argument('--output_base', type=str, required=True,
                       help='Base output directory (e.g., outputs/baselines/kgat_videogames)')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 2023, 2024, 2025, 12345],
                       help='Random seeds for trials (default: 42 2023 2024 2025 12345)')

    args = parser.parse_args()

    dataset = args.dataset
    data_path = args.data_path
    output_base = Path(args.output_base)
    seeds = args.seeds

    print("="*70)
    print(f"KGAT with Metadata KG - {len(seeds)} Trials")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset}")
    print(f"  Data path: {data_path}")
    print(f"  Output base: {output_base}")
    print(f"  Seeds: {seeds}")
    print(f"  Evaluation mode: uni100")
    print(f"  Eval batch size: 100000")
    print(f"  KG source: Metadata (categories + price/year)")
    print()

    # Verify data path exists
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"Error: Data path not found: {data_dir}")
        return

    # Check required files
    dataset_dir = data_dir / dataset
    required_files = [f'{dataset}.inter', f'{dataset}.kg', f'{dataset}.link']
    missing_files = []

    for filename in required_files:
        filepath = dataset_dir / filename
        if not filepath.exists():
            missing_files.append(str(filepath))

    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print(f"\nPlease ensure all required files exist in {dataset_dir}")
        print("  - .inter (user-item interactions)")
        print("  - .kg (knowledge graph)")
        print("  - .link (item-entity mapping)")
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
                dataset=dataset,
                data_path=data_path,
                seed=seed,
                output_dir=trial_output
            )
            results.append({
                'trial': i,
                'seed': seed,
                'test_result': result['test_result'],
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

        # Print aggregated results
        ndcg_values = [r['test_result']['ndcg@10'] for r in results if r['status'] == 'success']
        recall_values = [r['test_result']['recall@10'] for r in results if r['status'] == 'success']

        import numpy as np
        print(f"\nTest NDCG@10: {np.mean(ndcg_values):.4f} ± {np.std(ndcg_values):.4f}")
        print(f"Test Recall@10: {np.mean(recall_values):.4f} ± {np.std(recall_values):.4f}")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
