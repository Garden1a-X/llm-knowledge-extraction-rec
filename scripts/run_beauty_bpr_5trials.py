#!/usr/bin/env python3
"""
Run BPR baseline 5-trial experiments for Amazon Beauty dataset.

Usage:
    python scripts/run_beauty_bpr_5trials.py
    python scripts/run_beauty_bpr_5trials.py --seeds 42 123 456
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from recbole.quick_start import run_recbole


def main():
    parser = argparse.ArgumentParser(description='Run BPR 5-trial on Beauty')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[42, 123, 456, 789, 2024],
                        help='Random seeds for trials')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU ID')
    args = parser.parse_args()

    print("=" * 60)
    print("Amazon Beauty - BPR Baseline 5-Trial Experiments")
    print("=" * 60)
    print(f"Seeds: {args.seeds}")
    print(f"GPU: {args.gpu_id}")
    print()

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/beauty/bpr_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for i, seed in enumerate(args.seeds):
        trial_num = i + 1
        print()
        print("-" * 60)
        print(f"Trial {trial_num}/{len(args.seeds)} - Seed: {seed}")
        print("-" * 60)

        # Run RecBole
        result = run_recbole(
            model='BPR',
            dataset='amazon-beauty',
            config_dict={
                'data_path': 'data/recbole',
                'embedding_size': 64,
                'reg_weight': 0.00001,
                'epochs': 300,
                'train_batch_size': 2048,
                'eval_batch_size': 100000,
                'learning_rate': 0.001,
                'stopping_step': 10,
                'eval_step': 1,
                'metrics': ['Recall', 'NDCG', 'Precision', 'Hit'],
                'topk': [10, 20],
                'valid_metric': 'NDCG@10',
                'eval_args': {
                    'split': {'RS': [0.7, 0.1, 0.2]},
                    'order': 'TO',
                    'mode': 'uni100'
                },
                'load_col': {
                    'inter': ['user_id', 'item_id', 'timestamp']
                },
                'seed': seed,
                'gpu_id': args.gpu_id,
            }
        )

        # Extract test results
        test_result = result.get('test_result', {})
        trial_result = {
            'trial': trial_num,
            'seed': seed,
            'test_result': {k: float(v) for k, v in test_result.items()}
        }
        all_results.append(trial_result)

        print(f"✓ Trial {trial_num} completed")
        print(f"  NDCG@10: {test_result.get('ndcg@10', 'N/A')}")
        print(f"  Recall@10: {test_result.get('recall@10', 'N/A')}")

    # Save all results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'model': 'BPR',
            'dataset': 'amazon-beauty',
            'seeds': args.seeds,
            'trials': all_results
        }, f, indent=2)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    # Calculate mean and std
    ndcg_scores = [r['test_result'].get('ndcg@10', 0) for r in all_results]
    recall_scores = [r['test_result'].get('recall@10', 0) for r in all_results]

    import numpy as np
    print(f"NDCG@10:   {np.mean(ndcg_scores):.4f} ± {np.std(ndcg_scores):.4f}")
    print(f"Recall@10: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print()
    print(f"Results saved to: {results_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()
