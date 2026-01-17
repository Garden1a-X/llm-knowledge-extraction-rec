#!/usr/bin/env python3
"""
Summarize results from Beauty dataset trials (5 seeds).
Extracts Final Test Metrics from log files and generates summary JSON.
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from datetime import datetime


def extract_metrics_from_log(log_file: Path) -> Dict[str, float]:
    """
    Extract Final Test Metrics from a log file.

    Returns dict with lowercase metric names (e.g., 'ndcg@10', 'recall@20')
    """
    with open(log_file, 'r') as f:
        content = f.read()

    # Find the Final Test Metrics section
    metrics_pattern = r'Final Test Metrics:.*?(?=\n\n|\Z)'
    match = re.search(metrics_pattern, content, re.DOTALL)

    if not match:
        raise ValueError(f"Could not find 'Final Test Metrics' in {log_file}")

    metrics_section = match.group(0)

    # Extract individual metrics
    # Format: "2026-01-17 15:59:21,992 - __main__ - INFO -   NDCG@5: 0.3756"
    metric_pattern = r'INFO\s+-\s+([^:]+):\s+([\d.]+)'

    metrics = {}
    for match in re.finditer(metric_pattern, metrics_section):
        metric_name = match.group(1).strip().lower().replace('@', '@')  # e.g., "NDCG@10" -> "ndcg@10"
        metric_value = float(match.group(2))
        metrics[metric_name] = metric_value

    return metrics


def process_method_trials(trial_dir: Path, method_name: str) -> Dict:
    """
    Process all 5 trials for a method and generate summary.

    Args:
        trial_dir: Directory containing trial logs (e.g., beauty_full_20260117_143419)
        method_name: Name of the method (e.g., "beauty_full")
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
        metrics = extract_metrics_from_log(log_file)
        individual_results.append(metrics)

    # Calculate statistics (mean and std)
    if not individual_results:
        raise ValueError(f"No results found in {trial_dir}")

    # Get all metric names from first result
    metric_names = list(individual_results[0].keys())

    means = {}
    stds = {}

    for metric_name in metric_names:
        values = [result[metric_name] for result in individual_results]
        means[metric_name] = float(np.mean(values))
        stds[metric_name] = float(np.std(values, ddof=1))  # Sample std

    # Extract timestamp from directory name (beauty_full_20260117_143419)
    timestamp_match = re.search(r'(\d{8}_\d{6})', trial_dir.name)
    timestamp = timestamp_match.group(1) if timestamp_match else datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create summary
    summary = {
        "model": method_name,
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
    parser = argparse.ArgumentParser(description='Summarize Beauty dataset trial results')
    parser.add_argument('trial_dir', type=str, nargs='?', default=None,
                       help='Path to specific trial directory (e.g., outputs/beauty_trials/beauty_full_20260117_143419)')
    args = parser.parse_args()

    if args.trial_dir:
        # User specified a specific trial directory
        trial_dir = Path(args.trial_dir)
        if not trial_dir.exists():
            print(f"Error: Directory not found: {trial_dir}")
            return

        # Extract method name from directory name
        method_name = trial_dir.name.split('_2026')[0]  # beauty_full_20260117_143419 -> beauty_full

        print("="*70)
        print("Summarizing Beauty Trial Results")
        print("="*70)
        print(f"Processing: {trial_dir}")
        print()

        try:
            summary = process_method_trials(trial_dir, method_name)

            # Save summary
            output_file = trial_dir.parent / f"{method_name}_amazon_beauty_summary_5trials.json"
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

    else:
        # Auto-detect: look for most recent beauty_full trial
        script_dir = Path(__file__).parent.parent
        base_dir = script_dir / 'outputs' / 'beauty_trials'

        if not base_dir.exists():
            print(f"Error: Directory not found: {base_dir}")
            print("Please run ./scripts/run_beauty_5_trials.sh first")
            return

        # Find most recent beauty_full trial
        trial_dirs = sorted(base_dir.glob('beauty_full_*'))
        if not trial_dirs:
            print(f"Error: No beauty_full trials found in {base_dir}")
            print("Please run ./scripts/run_beauty_5_trials.sh first")
            return

        trial_dir = trial_dirs[-1]  # Most recent
        method_name = "beauty_full"

        print("="*70)
        print("Summarizing Beauty Trial Results")
        print("="*70)
        print(f"Auto-detected: {trial_dir}")
        print()

        try:
            summary = process_method_trials(trial_dir, method_name)

            # Save summary
            output_file = base_dir / f"{method_name}_amazon_beauty_summary_5trials.json"
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
