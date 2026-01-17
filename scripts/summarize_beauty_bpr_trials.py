#!/usr/bin/env python3
"""
Summarize BPR baseline results on Beauty dataset (5 seeds).
Extracts test metrics from RecBole log files and generates summary JSON.
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict
import numpy as np
from datetime import datetime


def extract_recbole_metrics_from_log(log_file: Path) -> Dict[str, float]:
    """
    Extract test metrics from RecBole BPR log file.

    RecBole format:
    INFO  test result: OrderedDict([('recall@10', 0.7665), ('recall@20', 0.8652), ...])

    Returns dict with lowercase metric names (e.g., 'ndcg@10', 'recall@20')
    """
    with open(log_file, 'r') as f:
        content = f.read()

    # Find test result line
    # Pattern: INFO  test result: OrderedDict([('recall@10', 0.7665), ...])
    test_pattern = r"test result:\s*OrderedDict\(\[(.*?)\]\)"
    match = re.search(test_pattern, content, re.DOTALL)

    if not match:
        raise ValueError(f"Could not find 'test result: OrderedDict' in {log_file}")

    # Extract the content inside OrderedDict([...])
    pairs_str = match.group(1)

    # Parse each (key, value) pair
    # Pattern: ('metric@k', value)
    pair_pattern = r"\('([^']+)',\s*([\d.]+)\)"
    metrics = {}

    for pair_match in re.finditer(pair_pattern, pairs_str):
        metric_name = pair_match.group(1).lower()  # e.g., 'recall@10' -> 'recall@10'
        metric_value = float(pair_match.group(2))
        metrics[metric_name] = metric_value

    if not metrics:
        raise ValueError(f"Could not parse metrics from {log_file}")

    return metrics


def process_bpr_trials(trial_dir: Path) -> Dict:
    """
    Process all 5 BPR trials and generate summary.

    Args:
        trial_dir: Directory containing trial logs (e.g., bpr_20260117_143419)
    """
    # Find all log files
    log_files = sorted(trial_dir.glob('trial_*_seed_*.log'))

    if len(log_files) != 5:
        print(f"Warning: Expected 5 log files in {trial_dir}, found {len(log_files)}")

    # Extract seed from filename (trial_1_seed_42.log -> 42)
    seeds = []
    individual_results = []

    for log_file in log_files:
        # Extract seed from filename
        seed_match = re.search(r'seed_(\d+)', log_file.name)
        if seed_match:
            seeds.append(int(seed_match.group(1)))

        # Extract metrics
        try:
            metrics = extract_recbole_metrics_from_log(log_file)
            individual_results.append(metrics)
        except Exception as e:
            print(f"Warning: Failed to extract metrics from {log_file}: {e}")
            continue

    # Calculate statistics (mean and std)
    if not individual_results:
        raise ValueError(f"No valid results found in {trial_dir}")

    # Get all metric names from first result
    metric_names = list(individual_results[0].keys())

    means = {}
    stds = {}

    for metric_name in metric_names:
        values = [result[metric_name] for result in individual_results]
        means[metric_name] = float(np.mean(values))
        stds[metric_name] = float(np.std(values, ddof=1))  # Sample std

    # Extract timestamp from directory name (bpr_20260117_143419)
    timestamp_match = re.search(r'(\d{8}_\d{6})', trial_dir.name)
    timestamp = timestamp_match.group(1) if timestamp_match else datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create summary
    summary = {
        "model": "BPR",
        "dataset": "amazon-beauty",
        "num_trials": len(individual_results),
        "seeds": seeds,
        "timestamp": timestamp,
        "individual_results": individual_results,
        "trial_output_dir": str(trial_dir.relative_to(trial_dir.parent.parent)),
        "statistics": {
            "means": means,
            "stds": stds
        }
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description='Summarize BPR baseline trial results on Beauty')
    parser.add_argument('trial_dir', type=str, nargs='?', default=None,
                       help='Path to specific trial directory (e.g., outputs/beauty_baselines/bpr_20260117_143419)')
    args = parser.parse_args()

    if args.trial_dir:
        # User specified a specific trial directory
        trial_dir = Path(args.trial_dir)
        if not trial_dir.exists():
            print(f"Error: Directory not found: {trial_dir}")
            return
    else:
        # Auto-detect: look for most recent BPR trial
        script_dir = Path(__file__).parent.parent
        base_dir = script_dir / 'outputs' / 'beauty_baselines'

        if not base_dir.exists():
            print(f"Error: Directory not found: {base_dir}")
            print("Please run ./scripts/run_beauty_bpr_5trials.sh first")
            return

        # Find most recent BPR trial
        trial_dirs = sorted(base_dir.glob('bpr_*'))
        if not trial_dirs:
            print(f"Error: No BPR trials found in {base_dir}")
            print("Please run ./scripts/run_beauty_bpr_5trials.sh first")
            return

        trial_dir = trial_dirs[-1]  # Most recent

    print("="*70)
    print("Summarizing BPR Baseline Results (Beauty)")
    print("="*70)
    print(f"Processing: {trial_dir}")
    print()

    try:
        summary = process_bpr_trials(trial_dir)

        # Save summary
        output_file = trial_dir.parent / f"bpr_amazon_beauty_summary_5trials.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Saved to {output_file}")
        print()
        print("Results:")
        print(f"  NDCG@10:   {summary['statistics']['means']['ndcg@10']:.4f} ± {summary['statistics']['stds']['ndcg@10']:.4f}")
        print(f"  Recall@10: {summary['statistics']['means']['recall@10']:.4f} ± {summary['statistics']['stds']['recall@10']:.4f}")
        print(f"  NDCG@20:   {summary['statistics']['means']['ndcg@20']:.4f} ± {summary['statistics']['stds']['ndcg@20']:.4f}")
        print(f"  Recall@20: {summary['statistics']['means']['recall@20']:.4f} ± {summary['statistics']['stds']['recall@20']:.4f}")
        print()

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print("="*70)
    print("Summary complete!")
    print("="*70)


if __name__ == '__main__':
    main()
