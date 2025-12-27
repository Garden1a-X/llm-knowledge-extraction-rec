#!/usr/bin/env python
"""
Run multiple trials of a baseline model with different random seeds.
This is essential for getting stable results with mean and standard deviation.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import numpy as np


def run_single_trial(model, dataset, data_path, device, epochs, seed, trial_num):
    """Run a single trial with a specific seed."""
    print(f"\n{'='*80}")
    print(f"Trial {trial_num + 1} - Running {model} with seed {seed}")
    print(f"{'='*80}\n")

    cmd = [
        sys.executable,
        "baselines/run_baseline.py",
        "--model", model,
        "--dataset", dataset,
        "--data_path", data_path,
        "--device", device,
        "--epochs", str(epochs),
        "--seed", str(seed)
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )

        if result.returncode != 0:
            print(f"ERROR in trial {trial_num + 1}:")
            print(result.stderr)
            return None, None

        # Find the output directory from stdout
        output = result.stdout
        output_dir = None

        lines = output.split('\n')
        for line in lines:
            if 'Results saved to:' in line:
                # Extract directory path
                # Format: "Results saved to: outputs/baselines/MODEL_DATASET_TIMESTAMP/results.json"
                path_str = line.split('Results saved to:')[-1].strip()
                # Get the directory (remove /results.json if present)
                if path_str.endswith('results.json'):
                    output_dir = Path(path_str).parent
                else:
                    output_dir = Path(path_str)
                break

        if output_dir is None:
            print(f"Warning: Could not find output directory from trial {trial_num + 1}")
            print(f"Stdout snippet (last 30 lines):")
            print('\n'.join(lines[-30:]))
            print(f"\nStderr:")
            print(result.stderr[:2000] if result.stderr else "(empty)")
            return None, None

        # Read the results.json file
        results_file = output_dir / "results.json"
        if not results_file.exists():
            print(f"Warning: Results file not found: {results_file}")
            return None, None

        with open(results_file, 'r') as f:
            data = json.load(f)

        # Extract test_result field
        if 'test_result' not in data:
            print(f"Warning: No 'test_result' field in {results_file}")
            return None, None

        test_results = data['test_result']

        print(f"\n✓ Trial {trial_num + 1} completed successfully")
        print(f"  Output: {output_dir}")
        print(f"  NDCG@10: {test_results.get('ndcg@10', 'N/A'):.4f}")
        print(f"  Recall@10: {test_results.get('recall@10', 'N/A'):.4f}")

        return test_results, str(output_dir)

    except subprocess.TimeoutExpired:
        print(f"ERROR: Trial {trial_num + 1} timed out after 2 hours")
        return None, None
    except Exception as e:
        print(f"ERROR in trial {trial_num + 1}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def compute_statistics(results_list):
    """Compute mean and std for each metric."""
    if not results_list:
        return None, None

    # Get all metrics
    metrics = list(results_list[0].keys())

    means = {}
    stds = {}

    for metric in metrics:
        values = [r[metric] for r in results_list]
        means[metric] = np.mean(values)
        stds[metric] = np.std(values)

    return means, stds


def format_results_table(means, stds):
    """Format results as a nice table."""
    if means is None or stds is None:
        return "No valid results"

    table = "\n" + "="*80 + "\n"
    table += "FINAL RESULTS (Mean ± Std)\n"
    table += "="*80 + "\n"
    table += f"{'Metric':<20} {'Mean':>12} {'Std':>12} {'Format':>20}\n"
    table += "-"*80 + "\n"

    for metric in sorted(means.keys()):
        mean_val = means[metric]
        std_val = stds[metric]
        formatted = f"{mean_val:.4f} ± {std_val:.4f}"
        table += f"{metric:<20} {mean_val:>12.4f} {std_val:>12.4f} {formatted:>20}\n"

    table += "="*80 + "\n"
    return table


def save_results(model, dataset, num_trials, seeds, all_results, means, stds, output_dir, trial_output_dirs=None):
    """Save all results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model}_summary_{num_trials}trials.json"
    filepath = os.path.join(output_dir, filename)

    data = {
        "model": model,
        "dataset": dataset,
        "num_trials": num_trials,
        "seeds": seeds,
        "timestamp": timestamp,
        "individual_results": all_results,
        "trial_output_dirs": trial_output_dirs,
        "statistics": {
            "means": means,
            "stds": stds
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Results saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple trials of a baseline model with different random seeds"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., LightGCN, BPR, NGCF)")
    parser.add_argument("--dataset", type=str, default="ml-1m",
                        help="Dataset name (default: ml-1m)")
    parser.add_argument("--data_path", type=str, default="data/recbole",
                        help="Path to RecBole data (default: data/recbole)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of epochs (default: 300)")
    parser.add_argument("--num_trials", type=int, default=5,
                        help="Number of trials to run (default: 5)")
    parser.add_argument("--seeds", type=int, nargs='+',
                        help="Specific seeds to use (default: [42, 2023, 2024, 2025, 12345])")
    parser.add_argument("--output_dir", type=str, default="outputs/baselines/multiple_trials",
                        help="Directory to save results (default: outputs/baselines/multiple_trials)")

    args = parser.parse_args()

    # Set default seeds if not provided
    if args.seeds is None:
        args.seeds = [42, 2023, 2024, 2025, 12345][:args.num_trials]
    elif len(args.seeds) != args.num_trials:
        print(f"Warning: Number of seeds ({len(args.seeds)}) != num_trials ({args.num_trials})")
        print(f"Using first {args.num_trials} seeds")
        args.seeds = args.seeds[:args.num_trials]

    print("="*80)
    print(f"Running Multiple Trials for {args.model}")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs per trial: {args.epochs}")
    print(f"Device: {args.device}")
    print("="*80)

    # Run all trials
    all_results = []
    output_dirs = []

    for i, seed in enumerate(args.seeds):
        result, output_dir = run_single_trial(
            model=args.model,
            dataset=args.dataset,
            data_path=args.data_path,
            device=args.device,
            epochs=args.epochs,
            seed=seed,
            trial_num=i
        )

        if result is not None:
            all_results.append(result)
            output_dirs.append(output_dir)
        else:
            print(f"\n✗ Trial {i+1} failed")

    # Compute statistics
    print("\n" + "="*80)
    print(f"Completed {len(all_results)}/{args.num_trials} trials successfully")
    print("="*80)

    if len(all_results) == 0:
        print("ERROR: No successful trials!")
        return 1

    means, stds = compute_statistics(all_results)

    # Print results table
    print(format_results_table(means, stds))

    # Print individual trial results
    print("\nIndividual Trial Results:")
    print("-"*80)
    for i, (seed, result) in enumerate(zip(args.seeds[:len(all_results)], all_results)):
        print(f"\nTrial {i+1} (seed={seed}):")
        for metric, value in sorted(result.items()):
            print(f"  {metric:<20}: {value:.4f}")

    # Save results
    save_results(
        model=args.model,
        dataset=args.dataset,
        num_trials=args.num_trials,
        seeds=args.seeds,
        all_results=all_results,
        means=means,
        stds=stds,
        output_dir=args.output_dir,
        trial_output_dirs=output_dirs
    )

    print("\n" + "="*80)
    print("All trials completed!")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
