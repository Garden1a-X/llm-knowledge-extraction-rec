#!/usr/bin/env python
"""
Extract test results from multiple trial outputs.
Reads results.json files from each trial's output directory.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np


def find_results_json(output_dir):
    """Find all results.json files in subdirectories."""
    results_files = []

    # Look for pattern: outputs/baselines/MODEL_DATASET_TIMESTAMP/results.json
    for subdir in sorted(output_dir.glob("LightGCN_*")):
        if subdir.is_dir():
            json_file = subdir / "results.json"
            if json_file.exists():
                results_files.append(json_file)

    return results_files


def load_results_json(json_file):
    """Load test results from a results.json file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract test_result field (which contains the actual test metrics)
        if 'test_result' in data:
            return data['test_result']
        else:
            print(f"Warning: No 'test_result' field in {json_file}")
            return None

    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None


def extract_seed_from_path(path):
    """Extract seed from config.json in the same directory."""
    config_file = path.parent / "config.json"
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config.get('seed', None)
    except:
        return None


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
        stds[metric] = np.std(values, ddof=1)  # Sample std

    return means, stds


def format_markdown_table(means, stds, metrics_order=None):
    """Format results as Markdown table."""
    if means is None:
        return ""

    if metrics_order is None:
        # Preferred metric order
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


def format_latex_table(means, stds, metrics_order=None):
    """Format results as LaTeX table."""
    if means is None:
        return ""

    if metrics_order is None:
        metrics_order = ['recall@10', 'recall@20', 'ndcg@10', 'ndcg@20',
                        'precision@10', 'precision@20', 'hit@10', 'hit@20']

    metrics_order = [m for m in metrics_order if m in means]

    lines = []
    lines.append("% LaTeX table format")
    lines.append("\\hline")

    for metric in metrics_order:
        mean_val = means[metric]
        std_val = stds[metric]
        # Clean metric name for LaTeX
        metric_name = metric.replace('_', '\\_')
        lines.append(f"{metric_name} & ${mean_val:.4f} \\pm {std_val:.4f}$ \\\\")

    lines.append("\\hline")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract results from multiple LightGCN trials"
    )
    parser.add_argument("output_dir", type=str, nargs='?',
                        default="outputs/baselines",
                        help="Directory containing trial outputs (default: outputs/baselines)")
    parser.add_argument("--output", type=str,
                        help="Output JSON file (optional)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"ERROR: Directory not found: {output_dir}")
        return 1

    print(f"Searching for results in: {output_dir}")
    print("="*80)

    # Find all results.json files
    results_files = find_results_json(output_dir)

    if not results_files:
        print(f"ERROR: No results.json files found in {output_dir}")
        print("\nTip: Make sure you're pointing to the directory containing")
        print("     the LightGCN_ml-1m_* subdirectories")
        return 1

    print(f"Found {len(results_files)} results files")
    print("="*80)

    # Load all results
    all_results = []
    seeds = []

    for i, json_file in enumerate(results_files):
        print(f"\n{i+1}. Loading: {json_file.parent.name}")
        results = load_results_json(json_file)

        if results:
            seed = extract_seed_from_path(json_file)
            all_results.append(results)
            seeds.append(seed)
            print(f"   ✓ Seed: {seed}")
            # Print key metrics
            for key in ['ndcg@10', 'recall@10', 'precision@10']:
                if key in results:
                    print(f"   {key}: {results[key]:.4f}")
        else:
            print(f"   ✗ Could not parse results")

    print("\n" + "="*80)

    if not all_results:
        print("ERROR: No valid results found!")
        return 1

    # Compute statistics
    means, stds = compute_statistics(all_results)

    # Print summary
    print(f"SUMMARY: {len(all_results)} successful trials")
    print("="*80)
    print("\nMean ± Std Results:")
    print("-"*80)

    # Sort metrics for display
    sorted_metrics = sorted(means.keys())
    for metric in sorted_metrics:
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
        # Infer model name from directory structure
        model_name = "unknown"
        if all_results and len(results_files) > 0:
            # Try to extract model name from first results file path
            # Path format: outputs/baselines/MODEL_DATASET_TIMESTAMP/results.json
            first_dir = results_files[0].parent.name  # e.g., "LightGCN_ml-1m_20251226_232706"
            model_name = first_dir.split('_')[0]  # Extract "LightGCN"

        output_file = output_dir / f"{model_name}_summary_{len(all_results)}trials.json"

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Results saved to: {output_file}")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
