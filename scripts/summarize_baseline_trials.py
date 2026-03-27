#!/usr/bin/env python3
"""
Summarize RecBole baseline results from multiple trials.
Extracts test_result from results.json and generates summary JSON.
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from datetime import datetime


def extract_test_metrics_from_json(results_file: Path) -> Dict[str, float]:
    """
    Extract test_result metrics from RecBole results.json.

    Returns dict with lowercase metric names (e.g., 'ndcg@10', 'recall@20')
    """
    with open(results_file, 'r') as f:
        data = json.load(f)

    if 'test_result' not in data:
        raise ValueError(f"No 'test_result' found in {results_file}")

    # RecBole format uses lowercase keys already
    return data['test_result']


def find_trial_directories(base_dir: Path, model_name: str, year: str = "2026") -> List[Path]:
    """
    Find all trial directories for a model.

    Args:
        base_dir: Base directory (e.g., outputs/baselines)
        model_name: Model name (e.g., "BPR", "LightGCN")
        year: Year prefix to filter (default: "2026" for uni100 results)

    Returns:
        List of trial directories sorted by timestamp
    """
    # Pattern: BPR_ml-1m_20260117_*
    pattern = f"{model_name}_ml-1m_{year}*"
    trial_dirs = sorted(base_dir.glob(pattern))

    # Filter out directories that don't have results.json
    valid_dirs = []
    for d in trial_dirs:
        results_file = d / "results.json"
        if results_file.exists():
            valid_dirs.append(d)

    return valid_dirs


def process_baseline_trials(base_dir: Path, model_name: str, dataset: str = "ml-1m") -> Dict:
    """
    Process all trials for a baseline model and generate summary.

    Args:
        base_dir: Base directory containing trial outputs
        model_name: Name of the model (e.g., "BPR")
        dataset: Dataset name (default: "ml-1m")
    """
    # Find trial directories
    trial_dirs = find_trial_directories(base_dir, model_name)

    if len(trial_dirs) == 0:
        raise ValueError(f"No trial directories found for {model_name}")

    if len(trial_dirs) != 5:
        print(f"Warning: Expected 5 trials for {model_name}, found {len(trial_dirs)}")

    # Extract metrics from each trial
    individual_results = []
    trial_output_dirs = []

    for trial_dir in trial_dirs:
        results_file = trial_dir / "results.json"
        metrics = extract_test_metrics_from_json(results_file)
        individual_results.append(metrics)
        trial_output_dirs.append(str(trial_dir.relative_to(base_dir.parent)))

    # Calculate statistics
    if not individual_results:
        raise ValueError(f"No results extracted for {model_name}")

    # Get all metric names from first result
    metric_names = list(individual_results[0].keys())

    means = {}
    stds = {}

    for metric_name in metric_names:
        values = [result[metric_name] for result in individual_results]
        means[metric_name] = float(np.mean(values))
        stds[metric_name] = float(np.std(values, ddof=1))  # Sample std

    # Get timestamp from first directory (format: BPR_ml-1m_20260117_142251)
    timestamp_match = re.search(r'(\d{8}_\d{6})', trial_dirs[0].name)
    timestamp = timestamp_match.group(1) if timestamp_match else datetime.now().strftime("%Y%m%d_%H%M%S")

    # Assume seeds are the standard 5
    seeds = [42, 2023, 2024, 2025, 12345]

    # Create summary
    summary = {
        "model": model_name,
        "dataset": dataset,
        "num_trials": len(individual_results),
        "seeds": seeds[:len(individual_results)],  # Use only as many as we have
        "timestamp": timestamp,
        "individual_results": individual_results,
        "trial_output_dirs": trial_output_dirs,
        "statistics": {
            "means": means,
            "stds": stds
        }
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description='Summarize RecBole baseline trial results')
    parser.add_argument('--baselines_dir', type=str, default=None,
                       help='Path to baselines directory (default: auto-detect from script location)')
    parser.add_argument('--models', type=str, nargs='+', default=['BPR', 'LightGCN', 'KGAT'],
                       help='Models to summarize (default: BPR LightGCN KGAT)')
    parser.add_argument('--dataset', type=str, default='ml-1m',
                       help='Dataset name (default: ml-1m)')

    args = parser.parse_args()

    # Base directory
    if args.baselines_dir:
        base_dir = Path(args.baselines_dir)
    else:
        # Auto-detect based on script location
        script_dir = Path(__file__).parent.parent  # Go up to project root
        base_dir = script_dir / 'outputs' / 'baselines'

    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        print("Please specify the correct path using --baselines_dir")
        return

    print("="*70)
    print("Summarizing RecBole Baseline Trial Results")
    print("="*70)
    print(f"\nBase directory: {base_dir}")
    print(f"Models: {', '.join(args.models)}")
    print()

    # Process each model
    for model_name in args.models:
        print(f"Processing {model_name}...")

        try:
            summary = process_baseline_trials(base_dir, model_name, args.dataset)

            # Save summary
            output_file = base_dir / "multiple_trials" / f"{model_name}_{args.dataset}_summary_5trials.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"  ✓ Saved to {output_file}")
            print(f"  Trials: {summary['num_trials']}")
            print(f"  Mean NDCG@10: {summary['statistics']['means']['ndcg@10']:.4f} ± {summary['statistics']['stds']['ndcg@10']:.4f}")
            print(f"  Mean Recall@10: {summary['statistics']['means']['recall@10']:.4f} ± {summary['statistics']['stds']['recall@10']:.4f}")
            print()

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("="*70)
    print("Summary complete!")
    print("="*70)


if __name__ == '__main__':
    main()
