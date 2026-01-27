#!/usr/bin/env python3
"""
Run VBPR and MMGCN experiments on Amazon Video Games dataset.

Usage:
    python scripts/run_videogames_multimodal.py --method vbpr
    python scripts/run_videogames_multimodal.py --method mmgcn
    python scripts/run_videogames_multimodal.py --method all
"""

import subprocess
import sys
import json
import math
from pathlib import Path
from datetime import datetime


# Configuration
DATA_DIR = "/data/xuao/llm-knowledge-extraction-rec/data/recbole/amazon-videogames"
VISUAL_FEATURES = "/data/xuao/llm-knowledge-extraction-rec/data/recbole/amazon-videogames/visual_features.npy"
OUTPUT_BASE = Path(__file__).parent.parent / "outputs" / "videogames"

# 5 trials with different seeds (same as ML-1M experiments)
SEEDS = [42, 2023, 2024, 2025, 12345]

# Training parameters (same as ML-1M)
COMMON_ARGS = {
    "embedding_dim": 64,
    "epochs": 300,
    "batch_size": 1024,
    "lr": 0.001,
    "early_stop": 10,
    "reg_weight": 1e-5,
}

# MMGCN specific
MMGCN_ARGS = {
    "n_layers": 2,
}


def run_experiment(method: str, seed: int, trial_num: int):
    """Run a single experiment."""
    print(f"\n{'='*80}")
    print(f"Running {method.upper()} Trial {trial_num}/5 (seed={seed})")
    print(f"{'='*80}\n")

    # Output directory
    output_dir = OUTPUT_BASE / method / f"trial_{trial_num}_seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    script_path = Path(__file__).parent.parent / "baselines" / f"train_{method}.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--data_dir", DATA_DIR,
        "--visual_features", VISUAL_FEATURES,
        "--output_dir", str(output_dir),
        "--seed", str(seed),
        "--embedding_dim", str(COMMON_ARGS["embedding_dim"]),
        "--epochs", str(COMMON_ARGS["epochs"]),
        "--batch_size", str(COMMON_ARGS["batch_size"]),
        "--lr", str(COMMON_ARGS["lr"]),
        "--early_stop", str(COMMON_ARGS["early_stop"]),
        "--reg_weight", str(COMMON_ARGS["reg_weight"]),
    ]

    # Add method-specific args
    if method == "mmgcn":
        cmd.extend(["--n_layers", str(MMGCN_ARGS["n_layers"])])

    print(f"Command: {' '.join(cmd)}\n")

    # Run
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\n❌ Trial {trial_num} failed!")
        return None

    # Load results
    results_file = output_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)

    return None


def calculate_stats(results: list):
    """Calculate mean and std of results."""
    if not results:
        return None

    metrics = {}
    for key in results[0]["test_metrics"]:
        values = [r["test_metrics"][key] for r in results]
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance)
        metrics[key] = {"mean": mean, "std": std, "values": values}

    return metrics


def run_all_trials(method: str):
    """Run all 5 trials for a method."""
    print(f"\n{'#'*80}")
    print(f"# Running {method.upper()} on Amazon Video Games (5 trials)")
    print(f"{'#'*80}\n")

    results = []

    for i, seed in enumerate(SEEDS, 1):
        result = run_experiment(method, seed, i)
        if result:
            results.append(result)
            print(f"\n✅ Trial {i} completed: NDCG@10 = {result['test_metrics']['ndcg@10']:.4f}")
        else:
            print(f"\n⚠️ Trial {i} failed or no results")

    # Calculate statistics
    if results:
        stats = calculate_stats(results)

        print(f"\n{'='*80}")
        print(f"{method.upper()} Results Summary (Amazon Video Games)")
        print(f"{'='*80}")
        print(f"Successful trials: {len(results)}/5")
        print()

        for metric, data in stats.items():
            print(f"{metric}: {data['mean']:.4f} ± {data['std']:.4f}")
            print(f"  Values: {[f'{v:.4f}' for v in data['values']]}")

        # Save summary
        summary_file = OUTPUT_BASE / method / "summary.json"
        with open(summary_file, "w") as f:
            json.dump({
                "method": method,
                "dataset": "amazon-videogames",
                "n_trials": len(results),
                "seeds": SEEDS[:len(results)],
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

        print(f"\nSummary saved to: {summary_file}")

        return stats

    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run VBPR/MMGCN on Video Games")
    parser.add_argument("--method", type=str, choices=["vbpr", "mmgcn", "all"],
                        default="all", help="Method to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")

    args = parser.parse_args()

    # Check paths
    print("Checking paths...")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Visual features: {VISUAL_FEATURES}")

    if args.dry_run:
        print("\n🔍 DRY RUN - Commands to run:")
        methods = ["vbpr", "mmgcn"] if args.method == "all" else [args.method]
        for method in methods:
            for i, seed in enumerate(SEEDS, 1):
                print(f"\n  [{method.upper()}] Trial {i}: seed={seed}")
        return

    # Run experiments
    all_results = {}

    if args.method in ["vbpr", "all"]:
        all_results["vbpr"] = run_all_trials("vbpr")

    if args.method in ["mmgcn", "all"]:
        all_results["mmgcn"] = run_all_trials("mmgcn")

    # Final summary
    print(f"\n{'#'*80}")
    print("# Final Results Summary")
    print(f"{'#'*80}\n")

    for method, stats in all_results.items():
        if stats:
            print(f"{method.upper()}:")
            print(f"  NDCG@10:    {stats['ndcg@10']['mean']:.4f} ± {stats['ndcg@10']['std']:.4f}")
            print(f"  Recall@10:  {stats['recall@10']['mean']:.4f} ± {stats['recall@10']['std']:.4f}")
            print()


if __name__ == "__main__":
    main()
