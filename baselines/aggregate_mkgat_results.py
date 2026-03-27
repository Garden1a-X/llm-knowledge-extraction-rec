#!/usr/bin/env python3
"""
Aggregate results from multiple MKGAT trials.

Usage:
    python baselines/aggregate_mkgat_results.py outputs/mkgat_5trials/20260118_120000
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np


def aggregate_results(trial_dir):
    """Aggregate results from all trials."""
    trial_dir = Path(trial_dir)

    # Find all results.json files
    result_files = list(trial_dir.glob('seed_*/results.json'))

    if not result_files:
        print(f"No results found in {trial_dir}")
        return

    print("="*70)
    print(f"Aggregating {len(result_files)} trial results")
    print("="*70)
    print()

    # Collect metrics
    test_ndcg = []
    test_recall = []
    test_precision = []
    val_ndcg = []

    for result_file in sorted(result_files):
        with open(result_file, 'r') as f:
            results = json.load(f)

        test_metrics = results['test_metrics']
        val_metrics = results['best_val_metrics']

        test_ndcg.append(test_metrics['ndcg@10'])
        test_recall.append(test_metrics['recall@10'])
        test_precision.append(test_metrics['precision@10'])
        val_ndcg.append(val_metrics['ndcg@10'])

        seed = results['args']['seed']
        print(f"Seed {seed}:")
        print(f"  Val NDCG@10:  {val_metrics['ndcg@10']:.4f}")
        print(f"  Test NDCG@10: {test_metrics['ndcg@10']:.4f}")
        print(f"  Test Recall@10: {test_metrics['recall@10']:.4f}")
        print()

    # Compute statistics
    print("="*70)
    print("Aggregated Results (Mean ± Std)")
    print("="*70)
    print()
    print(f"Test NDCG@10:      {np.mean(test_ndcg):.4f} ± {np.std(test_ndcg):.4f}")
    print(f"Test Recall@10:    {np.mean(test_recall):.4f} ± {np.std(test_recall):.4f}")
    print(f"Test Precision@10: {np.mean(test_precision):.4f} ± {np.std(test_precision):.4f}")
    print()
    print(f"Best Val NDCG@10:  {np.mean(val_ndcg):.4f} ± {np.std(val_ndcg):.4f}")
    print("="*70)

    # Save aggregated results
    aggregated = {
        'n_trials': len(result_files),
        'test_metrics': {
            'ndcg@10': {
                'mean': float(np.mean(test_ndcg)),
                'std': float(np.std(test_ndcg)),
                'values': [float(x) for x in test_ndcg]
            },
            'recall@10': {
                'mean': float(np.mean(test_recall)),
                'std': float(np.std(test_recall)),
                'values': [float(x) for x in test_recall]
            },
            'precision@10': {
                'mean': float(np.mean(test_precision)),
                'std': float(np.std(test_precision)),
                'values': [float(x) for x in test_precision]
            }
        },
        'val_metrics': {
            'ndcg@10': {
                'mean': float(np.mean(val_ndcg)),
                'std': float(np.std(val_ndcg)),
                'values': [float(x) for x in val_ndcg]
            }
        }
    }

    output_file = trial_dir / 'aggregated_results.json'
    with open(output_file, 'w') as f:
        json.dump(aggregated, f, indent=2)

    print(f"\n✓ Aggregated results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate MKGAT trial results')
    parser.add_argument('trial_dir', type=str, help='Directory containing trial results')
    args = parser.parse_args()

    aggregate_results(args.trial_dir)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python aggregate_mkgat_results.py <trial_dir>")
        sys.exit(1)

    main()
