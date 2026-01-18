#!/usr/bin/env python3
"""
Extract KGAT trial results from RecBole log files.

KGAT logs are typically in log/KGAT/ directory with format:
    INFO  test result: OrderedDict([('recall@5', 0.0885), ...])

Usage:
    python baselines/extract_kgat_results.py <log_dir>
    python baselines/extract_kgat_results.py /data/xuao/llm-knowledge-extraction-rec/log/KGAT
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np


def extract_metrics_from_kgat_log(log_file: Path) -> Dict[str, float]:
    """
    Extract test metrics from KGAT log file.

    Format: INFO  test result: OrderedDict([('recall@5', 0.0885), ...])
    """
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Find test result line
    test_pattern = r"test result:\s*OrderedDict\(\[(.*?)\]\)"
    match = re.search(test_pattern, content, re.DOTALL)

    if not match:
        raise ValueError(f"Could not find 'test result: OrderedDict' in {log_file}")

    pairs_str = match.group(1)
    pair_pattern = r"\('([^']+)',\s*([\d.]+)\)"
    metrics = {}

    for pair_match in re.finditer(pair_pattern, pairs_str):
        metric_name = pair_match.group(1).lower()
        metric_value = float(pair_match.group(2))
        metrics[metric_name] = metric_value

    if not metrics:
        raise ValueError(f"Could not parse metrics from {log_file}")

    return metrics


def extract_seed_from_filename(filename: str) -> Optional[int]:
    """Extract seed from KGAT log filename."""
    # Try common patterns
    # Pattern 1: KGAT-seed_42.log
    match = re.search(r'seed[_-](\d+)', filename)
    if match:
        return int(match.group(1))

    # Pattern 2: KGAT_42.log
    match = re.search(r'KGAT[_-](\d+)', filename)
    if match:
        return int(match.group(1))

    # Pattern 3: Just numbers in filename
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))

    return None


def main():
    parser = argparse.ArgumentParser(description='Extract KGAT trial results from log files')
    parser.add_argument('log_dir', type=str,
                       help='Path to directory containing KGAT log files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file (default: auto-generated)')
    parser.add_argument('--pattern', type=str, default='*.log',
                       help='Log file pattern (default: *.log)')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Directory not found: {log_dir}")
        return

    # Find log files
    log_files = sorted(log_dir.glob(args.pattern))
    if not log_files:
        print(f"Error: No log files matching '{args.pattern}' found in {log_dir}")
        return

    print("="*70)
    print("Extracting KGAT Trial Results")
    print("="*70)
    print(f"Directory: {log_dir}")
    print(f"Log files found: {len(log_files)}")
    print()

    # Extract from each trial
    seeds = []
    individual_results = []
    failed_files = []

    for log_file in log_files:
        seed = extract_seed_from_filename(log_file.name)
        if seed:
            seeds.append(seed)

        try:
            metrics = extract_metrics_from_kgat_log(log_file)
            individual_results.append(metrics)
            print(f"✓ {log_file.name}: {len(metrics)} metrics extracted")
        except Exception as e:
            print(f"✗ {log_file.name}: {e}")
            failed_files.append(log_file.name)

    if not individual_results:
        print("\nError: No valid results extracted")
        return

    print(f"\nSuccessfully extracted: {len(individual_results)}/{len(log_files)} files")
    if failed_files:
        print(f"Failed files: {', '.join(failed_files)}")

    # Calculate statistics
    metric_names = list(individual_results[0].keys())
    means = {}
    stds = {}

    for metric_name in metric_names:
        values = [result[metric_name] for result in individual_results]
        means[metric_name] = float(np.mean(values))
        stds[metric_name] = float(np.std(values, ddof=1))

    # Create summary
    summary = {
        "model": "KGAT",
        "dataset": "ml-1m",
        "kg_source": "original",  # KGAT uses original KG from metadata
        "num_trials": len(individual_results),
        "seeds": seeds if seeds else list(range(1, len(individual_results) + 1)),
        "log_directory": str(log_dir),
        "individual_results": individual_results,
        "statistics": {
            "means": means,
            "stds": stds
        }
    }

    # Save summary
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = log_dir.parent / f"kgat_ml1m_summary_{len(individual_results)}trials.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"✓ Saved summary to {output_file}")
    print()
    print("Results Summary:")
    print("-" * 70)

    # Print key metrics in order
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
