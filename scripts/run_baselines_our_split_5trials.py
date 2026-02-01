#!/usr/bin/env python3
"""
Run all baselines (BPR, LightGCN, KGAT) using our data split with 5 trials.

Usage:
    python scripts/run_baselines_our_split_5trials.py --method bpr
    python scripts/run_baselines_our_split_5trials.py --method lightgcn
    python scripts/run_baselines_our_split_5trials.py --method kgat
    python scripts/run_baselines_our_split_5trials.py --method all
"""

import subprocess
import sys
import numpy as np
from pathlib import Path

SEEDS = [42, 123, 456, 789, 2024]


def run_bpr_trials():
    """Run BPR 5 trials."""
    print("="*70)
    print("Running BPR with Our Split - 5 Trials")
    print("="*70)

    results = []
    for i, seed in enumerate(SEEDS):
        print(f"\n--- Trial {i+1}/5 (seed={seed}) ---")
        result = subprocess.run(
            ["python", "scripts/train_bpr_our_split.py", "--seed", str(seed)],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            continue

        # Parse results from output
        for line in result.stdout.split('\n'):
            if 'NDCG@10:' in line:
                ndcg = float(line.split(':')[1].strip())
            if 'Recall@10:' in line:
                recall = float(line.split(':')[1].strip())

        results.append({'ndcg': ndcg, 'recall': recall})
        print(f"Trial {i+1}: NDCG@10={ndcg:.4f}, Recall@10={recall:.4f}")

    return results


def run_lightgcn_trials():
    """Run LightGCN 5 trials."""
    print("="*70)
    print("Running LightGCN with Our Split - 5 Trials")
    print("="*70)

    results = []
    for i, seed in enumerate(SEEDS):
        print(f"\n--- Trial {i+1}/5 (seed={seed}) ---")
        result = subprocess.run(
            ["python", "scripts/train_lightgcn_our_split.py", "--seed", str(seed)],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            continue

        # Parse results from output
        for line in result.stdout.split('\n'):
            if 'NDCG@10:' in line:
                ndcg = float(line.split(':')[1].strip())
            if 'Recall@10:' in line:
                recall = float(line.split(':')[1].strip())

        results.append({'ndcg': ndcg, 'recall': recall})
        print(f"Trial {i+1}: NDCG@10={ndcg:.4f}, Recall@10={recall:.4f}")

    return results


def run_kgat_trials():
    """Run KGAT 5 trials."""
    print("="*70)
    print("Running KGAT with Our Split - 5 Trials")
    print("="*70)

    results = []
    for i, seed in enumerate(SEEDS):
        print(f"\n--- Trial {i+1}/5 (seed={seed}) ---")
        result = subprocess.run(
            ["python", "scripts/train_kgat_our_split.py", "--seed", str(seed)],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            continue

        # Parse results from output
        for line in result.stdout.split('\n'):
            if 'NDCG@10:' in line:
                ndcg = float(line.split(':')[1].strip())
            if 'Recall@10:' in line:
                recall = float(line.split(':')[1].strip())

        results.append({'ndcg': ndcg, 'recall': recall})
        print(f"Trial {i+1}: NDCG@10={ndcg:.4f}, Recall@10={recall:.4f}")

    return results


def print_summary(method, results):
    """Print summary statistics."""
    if not results:
        print(f"\n{method}: No results collected")
        return

    ndcgs = [r['ndcg'] for r in results]
    recalls = [r['recall'] for r in results]

    print(f"\n{'='*70}")
    print(f"{method} Summary (5 Trials)")
    print(f"{'='*70}")
    print(f"  NDCG@10:   {np.mean(ndcgs):.4f} ± {np.std(ndcgs):.4f}")
    print(f"  Recall@10: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"{'='*70}")

    # Print for easy copy to markdown
    print(f"\nMarkdown format:")
    print(f"| {method} | {np.mean(ndcgs):.4f} ± {np.std(ndcgs):.4f} | {np.mean(recalls):.4f} ± {np.std(recalls):.4f} |")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='all',
                        choices=['bpr', 'lightgcn', 'kgat', 'all'])
    args = parser.parse_args()

    all_results = {}

    if args.method in ['bpr', 'all']:
        results = run_bpr_trials()
        all_results['BPR'] = results
        print_summary('BPR', results)

    if args.method in ['lightgcn', 'all']:
        results = run_lightgcn_trials()
        all_results['LightGCN'] = results
        print_summary('LightGCN', results)

    if args.method in ['kgat', 'all']:
        results = run_kgat_trials()
        all_results['KGAT'] = results
        print_summary('KGAT', results)

    # Final summary
    if args.method == 'all':
        print("\n" + "="*70)
        print("All Baselines Summary (Our Split)")
        print("="*70)
        print("| Method | NDCG@10 | Recall@10 |")
        print("|--------|---------|-----------|")
        for method, results in all_results.items():
            if results:
                ndcgs = [r['ndcg'] for r in results]
                recalls = [r['recall'] for r in results]
                print(f"| {method} | {np.mean(ndcgs):.4f} ± {np.std(ndcgs):.4f} | {np.mean(recalls):.4f} ± {np.std(recalls):.4f} |")


if __name__ == '__main__':
    main()
