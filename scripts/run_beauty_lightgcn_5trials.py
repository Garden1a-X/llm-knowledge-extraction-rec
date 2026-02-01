#!/usr/bin/env python3
"""
Run LightGCN baseline 5-trial experiments for Amazon Beauty dataset.

Usage:
    python scripts/run_beauty_lightgcn_5trials.py
    python scripts/run_beauty_lightgcn_5trials.py --seeds 42 123 456
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from recbole.quick_start import run_recbole


def run_trial(seed: int, trial_num: int, output_dir: Path):
    """Run single LightGCN trial."""

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        # Data
        'data_path': 'data/recbole',
        'dataset': 'amazon-beauty',
        'load_col': {
            'inter': ['user_id', 'item_id', 'timestamp'],
        },

        # Data split
        'eval_args': {
            'split': {'RS': [0.7, 0.1, 0.2]},
            'order': 'TO',
            'group_by': 'user',
            'mode': 'uni100'
        },

        # Evaluation metrics
        'metrics': ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': [5, 10, 20],
        'valid_metric': 'NDCG@10',

        # LightGCN parameters
        'embedding_size': 64,
        'n_layers': 2,
        'reg_weight': 0.0001,

        # Training
        'epochs': 300,
        'train_batch_size': 2048,
        'eval_batch_size': 100000,
        'learning_rate': 0.001,
        'stopping_step': 10,

        # Random seed
        'seed': seed,
        'reproducibility': True,
        'state': 'INFO',

        # Checkpoint
        'checkpoint_dir': str(checkpoint_dir),
        'show_progress': True,
        'save_dataset': False,
        'save_dataloaders': False,
    }

    print(f"\n{'='*60}")
    print(f"LightGCN Trial {trial_num} (seed={seed})")
    print(f"{'='*60}\n")

    result = run_recbole(
        model='LightGCN',
        dataset='amazon-beauty',
        config_dict=config_dict,
        saved=True
    )

    test_result = result.get('test_result', {})

    # Save trial results
    result_dict = {
        'trial': trial_num,
        'seed': seed,
        'model': 'LightGCN',
        'dataset': 'amazon-beauty',
        'timestamp': datetime.now().isoformat(),
        'test_metrics': {k: float(v) for k, v in test_result.items()},
    }

    with open(output_dir / 'trial_results.json', 'w') as f:
        json.dump(result_dict, f, indent=2)

    print(f"\n✓ Trial {trial_num} completed")
    print(f"  NDCG@10: {test_result.get('ndcg@10', 'N/A'):.4f}")
    print(f"  Recall@10: {test_result.get('recall@10', 'N/A'):.4f}")

    return result_dict


def main():
    parser = argparse.ArgumentParser(description='Run LightGCN 5-trial on Beauty')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[42, 123, 456, 789, 2024],
                        help='Random seeds for trials')
    args = parser.parse_args()

    print("=" * 60)
    print("Amazon Beauty - LightGCN Baseline 5-Trial")
    print("=" * 60)
    print(f"Seeds: {args.seeds}")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(f"outputs/beauty/lightgcn_{timestamp}")

    all_results = []

    for i, seed in enumerate(args.seeds):
        trial_num = i + 1
        trial_output = output_base / f"trial_{trial_num}_seed_{seed}"

        try:
            result = run_trial(seed, trial_num, trial_output)
            all_results.append(result)
        except Exception as e:
            print(f"\n✗ Trial {trial_num} failed: {e}")

    # Save summary
    if all_results:
        # Calculate mean and std
        metrics = {}
        for key in all_results[0]['test_metrics']:
            values = [r['test_metrics'][key] for r in all_results]
            metrics[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values
            }

        summary = {
            'model': 'LightGCN',
            'dataset': 'amazon-beauty',
            'num_trials': len(all_results),
            'seeds': args.seeds[:len(all_results)],
            'metrics': metrics,
            'trials': all_results
        }

        with open(output_base / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        for key in ['ndcg@10', 'recall@10', 'hit@10']:
            if key in metrics:
                print(f"  {key}: {metrics[key]['mean']:.4f} ± {metrics[key]['std']:.4f}")
        print(f"\nResults saved to: {output_base}")

    print("=" * 60)


if __name__ == '__main__':
    main()
