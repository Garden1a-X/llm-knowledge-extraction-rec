#!/usr/bin/env python3
"""
Evaluate KGAT model from saved checkpoints.
Re-run evaluation for all 5 trials to get complete statistics.

Usage:
    python baselines/evaluate_kgat_from_checkpoints.py \
        --checkpoint_base /data/xuao/llm-knowledge-extraction-rec/outputs/baselines/KGAT_original_kg_ml-1m \
        --data_path /data/xuao/KG4RecEval/dataset/
"""

import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np

try:
    from recbole.quick_start import load_data_and_model
    from recbole.utils import get_trainer
except ImportError:
    print("Error: RecBole not found. Please install RecBole first.")
    sys.exit(1)


def find_checkpoint(trial_dir: Path) -> Path:
    """Find the checkpoint file in trial directory."""
    checkpoint_dir = trial_dir / 'checkpoints'
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Find .pth file
    checkpoints = list(checkpoint_dir.glob('KGAT-*.pth'))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    if len(checkpoints) > 1:
        print(f"  Warning: Multiple checkpoints found in {checkpoint_dir}, using first one")

    return checkpoints[0]


def evaluate_checkpoint(checkpoint_path: Path, data_path: str) -> dict:
    """
    Load checkpoint and evaluate.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        data_path: Path to dataset directory

    Returns:
        dict with test results
    """
    print(f"\n  Loading checkpoint: {checkpoint_path.name}")

    # Load data and model
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=str(checkpoint_path)
    )

    # Get trainer
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # Evaluate on test set
    print("  Running evaluation on test set...")
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)

    # Extract metrics
    metrics = {}
    for metric_name, metric_value in test_result.items():
        # Convert to lowercase for consistency
        metrics[metric_name.lower()] = float(metric_value)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate KGAT from checkpoints')
    parser.add_argument('--checkpoint_base', type=str, required=True,
                       help='Base directory containing trial directories')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 2023, 2024, 2025, 12345],
                       help='Seeds used in trials (default: 42 2023 2024 2025 12345)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file (default: auto-generated)')

    args = parser.parse_args()

    checkpoint_base = Path(args.checkpoint_base)
    data_path = args.data_path
    seeds = args.seeds

    print("="*70)
    print("Evaluating KGAT from Checkpoints")
    print("="*70)
    print(f"\nCheckpoint base: {checkpoint_base}")
    print(f"Data path: {data_path}")
    print(f"Seeds: {seeds}")
    print()

    # Evaluate each trial
    results = []
    successful_trials = []

    for i, seed in enumerate(seeds, 1):
        trial_dir = checkpoint_base / f"trial_{i}_seed_{seed}"

        print(f"\n{'='*70}")
        print(f"Trial {i}/5 (seed={seed})")
        print(f"{'='*70}")

        if not trial_dir.exists():
            print(f"  ✗ Trial directory not found: {trial_dir}")
            continue

        try:
            # Find checkpoint
            checkpoint_path = find_checkpoint(trial_dir)

            # Evaluate
            metrics = evaluate_checkpoint(checkpoint_path, data_path)

            results.append({
                'trial': i,
                'seed': seed,
                'metrics': metrics
            })
            successful_trials.append(i)

            # Print key metrics
            print(f"\n  ✓ Evaluation complete:")
            print(f"    NDCG@10:   {metrics.get('ndcg@10', 'N/A'):.4f}")
            print(f"    Recall@10: {metrics.get('recall@10', 'N/A'):.4f}")
            print(f"    NDCG@20:   {metrics.get('ndcg@20', 'N/A'):.4f}")
            print(f"    Recall@20: {metrics.get('recall@20', 'N/A'):.4f}")

        except Exception as e:
            print(f"  ✗ Error evaluating trial {i}: {e}")
            import traceback
            traceback.print_exc()

    # Calculate statistics
    if not results:
        print("\n✗ No successful evaluations")
        return

    print(f"\n{'='*70}")
    print("Computing Statistics")
    print(f"{'='*70}")
    print(f"\nSuccessful trials: {len(results)}/5")

    # Get all metric names
    metric_names = list(results[0]['metrics'].keys())

    # Calculate mean and std
    means = {}
    stds = {}

    for metric_name in metric_names:
        values = [r['metrics'][metric_name] for r in results]
        means[metric_name] = float(np.mean(values))
        stds[metric_name] = float(np.std(values, ddof=1))

    # Create summary
    summary = {
        "model": "KGAT",
        "dataset": "ml-1m",
        "kg_source": "original_metadata",
        "num_trials": len(results),
        "successful_trials": successful_trials,
        "seeds": [r['seed'] for r in results],
        "individual_results": [r['metrics'] for r in results],
        "checkpoint_base": str(checkpoint_base),
        "statistics": {
            "means": means,
            "stds": stds
        }
    }

    # Save summary
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = checkpoint_base.parent / f"kgat_ml1m_summary_{len(results)}trials.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Saved summary to {output_file}")

    # Print results
    print(f"\n{'='*70}")
    print("Results Summary")
    print(f"{'='*70}")
    print("\nKey Metrics:")
    print(f"  NDCG@10:   {means['ndcg@10']:.4f} ± {stds['ndcg@10']:.4f}")
    print(f"  Recall@10: {means['recall@10']:.4f} ± {stds['recall@10']:.4f}")
    print(f"  NDCG@20:   {means['ndcg@20']:.4f} ± {stds['ndcg@20']:.4f}")
    print(f"  Recall@20: {means['recall@20']:.4f} ± {stds['recall@20']:.4f}")

    print("\nAll Metrics:")
    for metric_name in sorted(means.keys()):
        print(f"  {metric_name:15s}: {means[metric_name]:.4f} ± {stds[metric_name]:.4f}")

    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
