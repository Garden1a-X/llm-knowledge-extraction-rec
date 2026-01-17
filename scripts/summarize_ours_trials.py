#!/usr/bin/env python3
"""
Summarize results from Ours method trials (5 seeds each).
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
        trial_dir: Directory containing trial logs (e.g., ours_full_20260117_143419)
        method_name: Name of the method (e.g., "ours_full")
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

    # Extract timestamp from directory name (ours_full_20260117_143419)
    timestamp_match = re.search(r'(\d{8}_\d{6})', trial_dir.name)
    timestamp = timestamp_match.group(1) if timestamp_match else datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create summary
    summary = {
        "model": method_name,
        "dataset": "ml-1m",
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
    parser = argparse.ArgumentParser(description='Summarize Ours method trial results')
    parser.add_argument('--trials_dir', type=str, default=None,
                       help='Path to ours_trials directory (default: auto-detect from script location)')
    args = parser.parse_args()

    # Base directory
    if args.trials_dir:
        base_dir = Path(args.trials_dir)
    else:
        # Auto-detect based on script location
        script_dir = Path(__file__).parent.parent  # Go up to project root
        base_dir = script_dir / 'outputs' / 'ours_trials'

    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        print("Please specify the correct path using --trials_dir")
        return

    # Find all method directories (ours_full_*, ours_kg_only_*, etc.)
    method_dirs = {}

    for pattern in ['ours_full_*', 'ours_kg_only_*', 'ours_wo_contrast_*', 'ours_wo_mask_*']:
        dirs = sorted(base_dir.glob(pattern))
        if dirs:
            # Take the most recent one (last in sorted list)
            method_name = pattern.replace('_*', '')
            method_dirs[method_name] = dirs[-1]

    print("="*70)
    print("Summarizing Ours Method Trial Results")
    print("="*70)
    print(f"\nFound {len(method_dirs)} methods:")
    for method_name, trial_dir in method_dirs.items():
        print(f"  {method_name}: {trial_dir}")
    print()

    # Process each method
    for method_name, trial_dir in method_dirs.items():
        print(f"Processing {method_name}...")

        try:
            summary = process_method_trials(trial_dir, method_name)

            # Save summary
            output_file = base_dir / f"{method_name}_ml_1m_summary_5trials.json"
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"  ✓ Saved to {output_file}")
            print(f"  Mean NDCG@10: {summary['statistics']['means']['ndcg@10']:.4f} ± {summary['statistics']['stds']['ndcg@10']:.4f}")
            print(f"  Mean Recall@10: {summary['statistics']['means']['recall@10']:.4f} ± {summary['statistics']['stds']['recall@10']:.4f}")
            print()

        except Exception as e:
            print(f"  ✗ Error: {e}")
            print()

    print("="*70)
    print("Summary complete!")
    print("="*70)


if __name__ == '__main__':
    main()
