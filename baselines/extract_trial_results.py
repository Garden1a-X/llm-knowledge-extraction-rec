#!/usr/bin/env python3
"""
Extract trial results from RecBole log files.
Supports both OrderedDict and table formats.

Usage:
    python baselines/extract_trial_results.py outputs/baselines/multiple_trials/LightGCN_20260117_125214
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Optional
import numpy as np


def extract_metrics_orderdict(log_file: Path) -> Optional[Dict[str, float]]:
    """
    Extract test metrics from RecBole log (OrderedDict format).

    Format: INFO  test result: OrderedDict([('recall@10', 0.7665), ...])
    """
    with open(log_file, 'r') as f:
        content = f.read()

    test_pattern = r"test result:\s*OrderedDict\(\[(.*?)\]\)"
    match = re.search(test_pattern, content, re.DOTALL)

    if not match:
        return None

    pairs_str = match.group(1)
    pair_pattern = r"\('([^']+)',\s*([\d.]+)\)"
    metrics = {}

    for pair_match in re.finditer(pair_pattern, pairs_str):
        metric_name = pair_match.group(1).lower()
        metric_value = float(pair_match.group(2))
        metrics[metric_name] = metric_value

    return metrics if metrics else None


def extract_metrics_table(log_file: Path) -> Optional[Dict[str, float]]:
    """
    Extract test metrics from RecBole log (table format).

    Format:
    Test Results:
    ================================================================================
      hit@10              : 0.7470
      ndcg@10             : 0.2260
    """
    with open(log_file, 'r') as f:
        content = f.read()

    metrics = {}
    pattern = r'^\s*([a-z_]+@\d+)\s*:\s*([\d.]+)'

    for line in content.split('\n'):
        match = re.match(pattern, line)
        if match:
            metric_name = match.group(1).lower()
            metric_value = float(match.group(2))
            metrics[metric_name] = metric_value

    return metrics if metrics else None


def extract_metrics(log_file: Path) -> Dict[str, float]:
    """Extract test metrics from RecBole log file (try both formats)."""
    metrics = extract_metrics_orderdict(log_file)
    if not metrics:
        metrics = extract_metrics_table(log_file)
    if not metrics:
        raise ValueError(f"Could not extract metrics from {log_file}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Extract trial results from RecBole logs')
    parser.add_argument('trial_dir', type=str,
                       help='Path to trial directory containing trial_*_seed_*.log files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file (default: auto-generated)')
    args = parser.parse_args()

    trial_dir = Path(args.trial_dir)
    if not trial_dir.exists():
        print(f"Error: Directory not found: {trial_dir}")
        return

    # Find log files
    log_files = sorted(trial_dir.glob('trial_*_seed_*.log'))
    if not log_files:
        print(f"Error: No trial_*_seed_*.log files found in {trial_dir}")
        return

    # Auto-detect model name
    model_match = re.match(r'([A-Za-z]+)_\d{8}_\d{6}', trial_dir.name)
    model_name = model_match.group(1) if model_match else 'Unknown'

    print("="*70)
    print(f"Extracting {model_name} Trial Results")
    print("="*70)
    print(f"Directory: {trial_dir}")
    print(f"Log files found: {len(log_files)}")
    print()

    # Extract from each trial
    seeds = []
    individual_results = []

    for log_file in log_files:
        seed_match = re.search(r'seed_(\d+)', log_file.name)
        if seed_match:
            seed = int(seed_match.group(1))
            seeds.append(seed)
        else:
            seed = None

        try:
            metrics = extract_metrics(log_file)
            individual_results.append(metrics)
            print(f"✓ {log_file.name}: {len(metrics)} metrics")
        except Exception as e:
            print(f"✗ {log_file.name}: {e}")

    if not individual_results:
        print("\nError: No valid results extracted")
        return

    # Calculate statistics
    metric_names = list(individual_results[0].keys())
    means = {}
    stds = {}

    for metric_name in metric_names:
        values = [result[metric_name] for result in individual_results]
        means[metric_name] = float(np.mean(values))
        stds[metric_name] = float(np.std(values, ddof=1))

    # Create summary
    timestamp_match = re.search(r'(\d{8}_\d{6})', trial_dir.name)
    timestamp = timestamp_match.group(1) if timestamp_match else 'unknown'

    summary = {
        "model": model_name,
        "dataset": "ml-1m",
        "num_trials": len(individual_results),
        "seeds": seeds,
        "timestamp": timestamp,
        "individual_results": individual_results,
        "trial_output_dir": str(trial_dir),
        "statistics": {
            "means": means,
            "stds": stds
        }
    }

    # Save summary
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = trial_dir.parent / f"{model_name.lower()}_ml1m_summary_{len(individual_results)}trials.json"

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"✓ Saved summary to {output_file}")
    print()
    print("Results Summary:")
    print("-" * 70)

    # Print key metrics
    key_metrics = ['ndcg@10', 'recall@10', 'ndcg@20', 'recall@20']
    for metric in key_metrics:
        if metric in means:
            print(f"  {metric:15s}: {means[metric]:.4f} ± {stds[metric]:.4f}")

    print("-" * 70)
    print()
    print("All metrics:")
    for metric in sorted(means.keys()):
        print(f"  {metric:15s}: {means[metric]:.4f} ± {stds[metric]:.4f}")

    print("="*70)


if __name__ == '__main__':
    main()
