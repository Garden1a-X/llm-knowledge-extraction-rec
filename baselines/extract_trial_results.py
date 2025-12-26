#!/usr/bin/env python
"""
Extract test results from multiple trial logs and compute statistics.
Usage: python extract_trial_results.py <log_directory>
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime
import numpy as np


def parse_log_file(log_path):
    """Parse a single log file to extract test results."""
    with open(log_path, 'r') as f:
        content = f.read()

    # Look for test result line
    # Format: "INFO  test result: OrderedDict([('recall@10', 0.0578), ...])"
    pattern = r"test result:\s*OrderedDict\(\[(.*?)\]\)"
    match = re.search(pattern, content)

    if not match:
        return None

    # Parse the key-value pairs
    pairs_str = match.group(1)
    pair_pattern = r"\('([^']+)',\s*([0-9.]+)\)"
    pairs = re.findall(pair_pattern, pairs_str)

    results = {k: float(v) for k, v in pairs}
    return results


def extract_seed_from_filename(filename):
    """Extract seed from filename like 'trial_1_seed_42.log'."""
    match = re.search(r'seed_(\d+)', filename)
    return int(match.group(1)) if match else None


def compute_statistics(results_list):
    """Compute mean and std for each metric."""
    if not results_list:
        return None, None

    metrics = list(results_list[0].keys())
    means = {}
    stds = {}

    for metric in metrics:
        values = [r[metric] for r in results_list]
        means[metric] = np.mean(values)
        stds[metric] = np.std(values, ddof=1)  # Sample std with ddof=1

    return means, stds


def format_latex_table(means, stds, metrics_order=None):
    """Format results as LaTeX table for papers."""
    if means is None:
        return ""

    if metrics_order is None:
        metrics_order = ['recall@10', 'recall@20', 'ndcg@10', 'ndcg@20',
                        'precision@10', 'precision@20', 'hit@10', 'hit@20']

    # Filter to only available metrics
    metrics_order = [m for m in metrics_order if m in means]

    lines = []
    lines.append("% LaTeX table format")
    lines.append("% Metric & Value \\\\")
    lines.append("\\hline")

    for metric in metrics_order:
        mean_val = means[metric]
        std_val = stds[metric]
        # Format: metric & mean ± std
        lines.append(f"{metric.replace('@', '@')} & ${mean_val:.4f} \\pm {std_val:.4f}$ \\\\")

    lines.append("\\hline")
    return "\n".join(lines)


def format_markdown_table(means, stds, metrics_order=None):
    """Format results as Markdown table."""
    if means is None:
        return ""

    if metrics_order is None:
        metrics_order = ['recall@10', 'recall@20', 'ndcg@10', 'ndcg@20',
                        'precision@10', 'precision@20', 'hit@10', 'hit@20']

    # Filter to only available metrics
    metrics_order = [m for m in metrics_order if m in means]

    lines = []
    lines.append("| Metric | Mean | Std | Mean ± Std |")
    lines.append("|--------|------|-----|------------|")

    for metric in metrics_order:
        mean_val = means[metric]
        std_val = stds[metric]
        lines.append(f"| {metric} | {mean_val:.4f} | {std_val:.4f} | {mean_val:.4f} ± {std_val:.4f} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze results from multiple trial logs"
    )
    parser.add_argument("log_dir", type=str,
                        help="Directory containing trial log files")
    parser.add_argument("--output", type=str,
                        help="Output JSON file (optional)")

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"ERROR: Directory not found: {log_dir}")
        return 1

    # Find all log files
    log_files = sorted(log_dir.glob("trial_*.log"))
    if not log_files:
        print(f"ERROR: No log files found in {log_dir}")
        return 1

    print(f"Found {len(log_files)} log files")
    print("="*80)

    # Parse each log file
    all_results = []
    seeds = []

    for log_file in log_files:
        print(f"Processing: {log_file.name}")
        results = parse_log_file(log_file)

        if results:
            seed = extract_seed_from_filename(log_file.name)
            all_results.append(results)
            seeds.append(seed)
            print(f"  ✓ Extracted results (seed={seed})")
            # Print key metrics
            if 'ndcg@10' in results:
                print(f"    NDCG@10: {results['ndcg@10']:.4f}")
            if 'recall@10' in results:
                print(f"    Recall@10: {results['recall@10']:.4f}")
        else:
            print(f"  ✗ Could not parse results")

    print("="*80)

    if not all_results:
        print("ERROR: No valid results found!")
        return 1

    # Compute statistics
    means, stds = compute_statistics(all_results)

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(all_results)} successful trials")
    print(f"{'='*80}\n")

    print("Mean ± Std Results:")
    print("-"*80)
    for metric in sorted(means.keys()):
        mean_val = means[metric]
        std_val = stds[metric]
        print(f"{metric:<20}: {mean_val:.4f} ± {std_val:.4f}")

    # Print individual results
    print(f"\n{'='*80}")
    print("Individual Trial Results:")
    print(f"{'='*80}\n")

    for i, (seed, result) in enumerate(zip(seeds, all_results)):
        print(f"Trial {i+1} (seed={seed}):")
        for metric in sorted(result.keys()):
            print(f"  {metric:<18}: {result[metric]:.4f}")
        print()

    # Print formatted tables
    print("="*80)
    print("Markdown Table (for README):")
    print("="*80)
    print(format_markdown_table(means, stds))
    print()

    print("="*80)
    print("LaTeX Table (for paper):")
    print("="*80)
    print(format_latex_table(means, stds))
    print()

    # Save to JSON
    output_data = {
        "num_trials": len(all_results),
        "seeds": seeds,
        "timestamp": datetime.now().isoformat(),
        "individual_results": all_results,
        "statistics": {
            "means": means,
            "stds": stds
        }
    }

    # Save to output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = log_dir / "summary_statistics.json"

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
