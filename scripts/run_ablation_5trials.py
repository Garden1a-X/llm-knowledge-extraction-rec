#!/usr/bin/env python3
"""
Run 5-trial ablation experiments on ML-1M dataset.

Ablation experiments:
1. ours_full - Full model (baseline for ablation)
2. ours_item_kg_only - Only Item KG (no User KG)
3. ours_user_kg_only - Only User KG (no Item KG)
4. ours_wo_long_term - User KG without long-term interests
5. ours_wo_short_term - User KG without short-term interests

Usage:
    # Run all ablation experiments
    python scripts/run_ablation_5trials.py --method all

    # Run specific ablation
    python scripts/run_ablation_5trials.py --method item_kg_only
    python scripts/run_ablation_5trials.py --method user_kg_only
    python scripts/run_ablation_5trials.py --method wo_long_term
    python scripts/run_ablation_5trials.py --method wo_short_term
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import subprocess
import time
import json
import numpy as np
from datetime import datetime


# Ablation configs
ABLATION_CONFIGS = {
    'full': 'configs/ours_full.yaml',
    'item_kg_only': 'configs/ours_item_kg_only.yaml',
    'user_kg_only': 'configs/ours_user_kg_only.yaml',
    'wo_long_term': 'configs/ours_wo_long_term.yaml',
    'wo_short_term': 'configs/ours_wo_short_term.yaml',
}

SEEDS = [42, 123, 456, 789, 2024]


def run_single_experiment(config_path, seed, verbose=True):
    """Run a single experiment with given config and seed."""
    # Modify config to use the seed
    # We need to create a temporary config or pass seed as argument
    # For now, we'll use a wrapper script approach

    cmd = [
        'python', 'scripts/train_model.py',
        '--config', config_path,
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    print(f"Seed: {seed}")

    # We need to modify the training script to accept seed override
    # For now, let's modify the config file temporarily
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    original_seed = config['train']['random_seed']
    config['train']['random_seed'] = seed

    # Create temp config
    temp_config_path = f'/tmp/ablation_config_seed{seed}.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    cmd = [
        'python', 'scripts/train_model.py',
        '--config', temp_config_path,
    ]

    start_time = time.time()

    if verbose:
        # Real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        output_lines = []
        for line in process.stdout:
            print(line, end='')
            output_lines.append(line)

        process.wait()
        returncode = process.returncode
        output = ''.join(output_lines)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        returncode = result.returncode
        output = result.stdout + result.stderr

    elapsed = time.time() - start_time

    if returncode != 0:
        print(f"ERROR: Experiment failed with return code {returncode}")
        print(output)
        return None

    # Parse results from output
    metrics = parse_metrics(output)
    metrics['elapsed_time'] = elapsed

    return metrics


def parse_metrics(output):
    """Parse metrics from training output."""
    metrics = {}

    # Look for test metrics in the output
    lines = output.split('\n')

    for i, line in enumerate(lines):
        # Look for patterns like "NDCG@10: 0.1234"
        if 'NDCG@10:' in line:
            try:
                value = float(line.split('NDCG@10:')[1].strip().split()[0])
                metrics['NDCG@10'] = value
            except (IndexError, ValueError):
                pass
        if 'Recall@10:' in line:
            try:
                value = float(line.split('Recall@10:')[1].strip().split()[0])
                metrics['Recall@10'] = value
            except (IndexError, ValueError):
                pass
        if 'NDCG@5:' in line:
            try:
                value = float(line.split('NDCG@5:')[1].strip().split()[0])
                metrics['NDCG@5'] = value
            except (IndexError, ValueError):
                pass
        if 'Recall@5:' in line:
            try:
                value = float(line.split('Recall@5:')[1].strip().split()[0])
                metrics['Recall@5'] = value
            except (IndexError, ValueError):
                pass
        if 'NDCG@20:' in line:
            try:
                value = float(line.split('NDCG@20:')[1].strip().split()[0])
                metrics['NDCG@20'] = value
            except (IndexError, ValueError):
                pass
        if 'Recall@20:' in line:
            try:
                value = float(line.split('Recall@20:')[1].strip().split()[0])
                metrics['Recall@20'] = value
            except (IndexError, ValueError):
                pass

    return metrics


def run_5_trials(method, config_path, verbose=True):
    """Run 5 trials for a given ablation method."""
    print("=" * 70)
    print(f"Running 5-trial experiments for: {method}")
    print(f"Config: {config_path}")
    print(f"Seeds: {SEEDS}")
    print("=" * 70)

    all_results = []

    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*70}")
        print(f"Trial {i+1}/5 - Seed: {seed}")
        print(f"{'='*70}")

        metrics = run_single_experiment(config_path, seed, verbose)

        if metrics:
            all_results.append(metrics)
            print(f"\nTrial {i+1} Results:")
            print(f"  NDCG@10: {metrics.get('NDCG@10', 'N/A')}")
            print(f"  Recall@10: {metrics.get('Recall@10', 'N/A')}")
        else:
            print(f"\nTrial {i+1} FAILED!")

    if not all_results:
        print("All trials failed!")
        return None

    # Compute statistics
    stats = compute_statistics(all_results)

    return stats


def compute_statistics(results):
    """Compute mean and std for all metrics."""
    stats = {}

    # Get all metric keys
    metric_keys = set()
    for r in results:
        metric_keys.update(r.keys())
    metric_keys.discard('elapsed_time')

    for key in metric_keys:
        values = [r.get(key) for r in results if r.get(key) is not None]
        if values:
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }

    # Add elapsed time
    elapsed_times = [r.get('elapsed_time', 0) for r in results]
    stats['elapsed_time'] = {
        'mean': np.mean(elapsed_times),
        'total': sum(elapsed_times)
    }

    return stats


def print_summary(method, stats):
    """Print summary of 5-trial results."""
    print("\n" + "=" * 70)
    print(f"5-Trial Summary: {method}")
    print("=" * 70)

    if 'NDCG@10' in stats:
        print(f"  NDCG@10: {stats['NDCG@10']['mean']:.4f} +/- {stats['NDCG@10']['std']:.4f}")
    if 'Recall@10' in stats:
        print(f"  Recall@10: {stats['Recall@10']['mean']:.4f} +/- {stats['Recall@10']['std']:.4f}")
    if 'NDCG@5' in stats:
        print(f"  NDCG@5: {stats['NDCG@5']['mean']:.4f} +/- {stats['NDCG@5']['std']:.4f}")
    if 'Recall@5' in stats:
        print(f"  Recall@5: {stats['Recall@5']['mean']:.4f} +/- {stats['Recall@5']['std']:.4f}")
    if 'NDCG@20' in stats:
        print(f"  NDCG@20: {stats['NDCG@20']['mean']:.4f} +/- {stats['NDCG@20']['std']:.4f}")
    if 'Recall@20' in stats:
        print(f"  Recall@20: {stats['Recall@20']['mean']:.4f} +/- {stats['Recall@20']['std']:.4f}")

    print(f"\n  Average time per trial: {stats['elapsed_time']['mean']/60:.1f} min")
    print(f"  Total time: {stats['elapsed_time']['total']/60:.1f} min")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Run 5-trial ablation experiments')
    parser.add_argument('--method', type=str, required=True,
                        choices=['all', 'full', 'item_kg_only', 'user_kg_only',
                                'wo_long_term', 'wo_short_term'],
                        help='Ablation method to run')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Show real-time output')
    parser.add_argument('--output', type=str, default='results/ml1m/ablation_results.json',
                        help='Output JSON file for results')

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if any
    all_stats = {}
    if Path(args.output).exists():
        with open(args.output, 'r') as f:
            all_stats = json.load(f)

    # Determine which methods to run
    if args.method == 'all':
        methods = list(ABLATION_CONFIGS.keys())
    else:
        methods = [args.method]

    print("\n" + "=" * 70)
    print("5-Trial Ablation Experiments (ML-1M)")
    print("=" * 70)
    print(f"Methods: {methods}")
    print(f"Seeds: {SEEDS}")
    print(f"Output: {args.output}")
    print("=" * 70)

    for method in methods:
        config_path = ABLATION_CONFIGS[method]

        if not Path(config_path).exists():
            print(f"ERROR: Config not found: {config_path}")
            continue

        stats = run_5_trials(method, config_path, args.verbose)

        if stats:
            print_summary(method, stats)

            # Save results
            all_stats[method] = {
                'NDCG@10': f"{stats['NDCG@10']['mean']:.4f} +/- {stats['NDCG@10']['std']:.4f}",
                'Recall@10': f"{stats['Recall@10']['mean']:.4f} +/- {stats['Recall@10']['std']:.4f}",
                'raw': {
                    k: {
                        'mean': v['mean'],
                        'std': v['std'],
                        'values': v.get('values', [])
                    }
                    for k, v in stats.items()
                    if k != 'elapsed_time'
                },
                'elapsed_time': stats['elapsed_time'],
                'timestamp': datetime.now().isoformat()
            }

            # Save after each method
            with open(args.output, 'w') as f:
                json.dump(all_stats, f, indent=2)

            print(f"\nResults saved to {args.output}")

    # Print final summary table
    print("\n" + "=" * 70)
    print("Final Summary Table")
    print("=" * 70)
    print(f"{'Method':<20} | {'NDCG@10':<22} | {'Recall@10':<22}")
    print("-" * 70)

    for method in methods:
        if method in all_stats:
            ndcg = all_stats[method]['NDCG@10']
            recall = all_stats[method]['Recall@10']
            print(f"{method:<20} | {ndcg:<22} | {recall:<22}")

    print("=" * 70)


if __name__ == '__main__':
    main()
