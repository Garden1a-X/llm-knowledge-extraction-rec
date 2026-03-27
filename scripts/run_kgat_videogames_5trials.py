#!/usr/bin/env python3
"""
Run KGAT with standard Video Games KG (categories + price) for 5 trials.
Uses uni100 evaluation mode for fair comparison.

FIXED: Ensures checkpoints and results are properly saved for all trials.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add RecBole to path if needed
try:
    from recbole.quick_start import run_recbole
    from recbole.config import Config
except ImportError:
    print("Error: RecBole not found. Please install RecBole first.")
    sys.exit(1)


def save_trial_results(result_dict: dict, output_file: Path):
    """Save trial results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=2)

    print(f"  ✓ Results saved to: {output_file}")


def run_kgat_trial(data_path: str, seed: int, trial_num: int, output_dir: Path):
    """
    Run single KGAT trial with standard Video Games KG.

    Args:
        data_path: Path to dataset directory
        seed: Random seed
        trial_num: Trial number (1-5)
        output_dir: Output directory for this trial

    Returns:
        dict: Trial results
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoint directory for this trial
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config_dict = {
        # Data
        'data_path': data_path,
        'dataset': 'amazon-videogames',
        'load_col': {
            'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
            'kg': ['head_id', 'relation_id', 'tail_id'],
            'link': ['item_id', 'entity_id']
        },

        # Data split (same as ours)
        'eval_args': {
            'split': {'RS': [0.7, 0.1, 0.2]},
            'order': 'TO',
            'group_by': 'user',
            'mode': 'uni100'
        },

        # Evaluation metrics
        'metrics': ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': [5, 10, 20],
        'valid_metric': 'NDCG@10',

        # KGAT parameters (aligned with paper)
        'embedding_size': 64,
        'kg_embedding_size': 64,
        'reg_weight': 0.0001,

        # Training
        'epochs': 300,
        'train_batch_size': 2048,
        'eval_batch_size': 100000,  # Large batch to process all users at once (80GB GPU)
        'learning_rate': 0.001,
        'stopping_step': 10,

        # Random seed
        'seed': seed,

        # Reproducibility
        'reproducibility': True,
        'state': 'INFO',

        # Checkpoint (unique per trial)
        'checkpoint_dir': str(checkpoint_dir),
        'show_progress': True,

        # Save model
        'save_dataset': False,
        'save_dataloaders': False,
    }

    print(f"\n{'='*70}")
    print(f"Running KGAT Trial {trial_num}/5 (seed={seed})")
    print(f"{'='*70}")
    print(f"Dataset: amazon-videogames")
    print(f"Data path: {data_path}")
    print(f"Evaluation mode: uni100")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print()

    # Run RecBole
    try:
        result = run_recbole(
            model='KGAT',
            dataset='amazon-videogames',
            config_dict=config_dict,
            saved=True  # Save best model
        )

        # Extract metrics
        test_result = result.get('test_result', {})
        best_valid_result = result.get('best_valid_result', {})

        # Prepare result dictionary
        result_dict = {
            'trial': trial_num,
            'seed': seed,
            'dataset': 'amazon-videogames',
            'model': 'KGAT',
            'timestamp': datetime.now().isoformat(),
            'test_metrics': test_result,
            'best_valid_metrics': best_valid_result,
            'config': {
                'embedding_size': 64,
                'kg_embedding_size': 64,
                'reg_weight': 0.0001,
                'epochs': 300,
                'batch_size': 2048,
                'learning_rate': 0.001,
                'stopping_step': 10,
                'eval_mode': 'uni100',
            }
        }

        # Save trial results
        result_file = output_dir / 'trial_results.json'
        save_trial_results(result_dict, result_file)

        print(f"\n{'='*70}")
        print(f"Trial {trial_num}/5 (seed={seed}) completed!")
        print(f"{'='*70}")
        print(f"\nTest Results:")
        for metric, value in test_result.items():
            print(f"  {metric}: {value:.4f}")
        print()

        return result_dict

    except Exception as e:
        print(f"\n✗ Trial {trial_num}/5 failed with error:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()

        # Save error info
        error_dict = {
            'trial': trial_num,
            'seed': seed,
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
        }

        error_file = output_dir / 'trial_error.json'
        save_trial_results(error_dict, error_file)

        raise


def main():
    parser = argparse.ArgumentParser(
        description='Run KGAT with standard Video Games KG for 5 trials'
    )
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset directory (e.g., /data/xuao/KG4RecEval/dataset/)')
    parser.add_argument('--output_base', type=str, required=True,
                       help='Base output directory (e.g., outputs/baselines/kgat_videogames)')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 2023, 2024, 2025, 12345],
                       help='Random seeds for trials (default: 42 2023 2024 2025 12345)')

    args = parser.parse_args()

    data_path = args.data_path
    output_base = Path(args.output_base)
    seeds = args.seeds

    print("="*70)
    print("KGAT with Standard Video Games KG - 5 Trials")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dataset: amazon-videogames")
    print(f"  Data path: {data_path}")
    print(f"  Output base: {output_base}")
    print(f"  Seeds: {seeds}")
    print(f"  Evaluation mode: uni100")
    print(f"  KG source: Standard metadata (categories + price ranges)")
    print()

    # Verify data path exists
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"Error: Data path not found: {data_dir}")
        return

    # Check required files
    dataset_dir = data_dir / 'amazon-videogames'
    required_files = ['amazon-videogames.inter', 'amazon-videogames.kg', 'amazon-videogames.link']
    missing_files = []

    for filename in required_files:
        filepath = dataset_dir / filename
        if not filepath.exists():
            missing_files.append(str(filepath))

    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure all required files exist:")
        print("  - amazon-videogames.inter (user-item interactions)")
        print("  - amazon-videogames.kg (knowledge graph)")
        print("  - amazon-videogames.link (item-entity mapping)")
        print("\nRun these commands first:")
        print("  1. python scripts/prepare_videogames_dataset.py")
        print("  2. python scripts/generate_videogames_standard_kg.py \\")
        print(f"       --output_dir {dataset_dir}")
        return

    print("✓ All required files found\n")

    # Run trials
    results = []
    successful_results = []

    for i, seed in enumerate(seeds, 1):
        print(f"\n{'#'*70}")
        print(f"# Trial {i}/{len(seeds)}: seed={seed}")
        print(f"{'#'*70}\n")

        # Create output directory for this trial (unique name)
        trial_output = output_base / f"trial_{i}_seed_{seed}"

        try:
            result = run_kgat_trial(
                data_path=data_path,
                seed=seed,
                trial_num=i,
                output_dir=trial_output
            )

            results.append({
                'trial': i,
                'seed': seed,
                'output_dir': str(trial_output),
                'status': 'success',
                'metrics': result.get('test_metrics', {})
            })

            successful_results.append(result)

        except Exception as e:
            print(f"\n✗ Trial {i} failed")

            results.append({
                'trial': i,
                'seed': seed,
                'output_dir': str(trial_output),
                'error': str(e),
                'status': 'failed'
            })

    # Save overall summary
    summary_file = output_base / 'trials_summary.json'
    summary = {
        'dataset': 'amazon-videogames',
        'model': 'KGAT',
        'total_trials': len(results),
        'successful_trials': sum(1 for r in results if r['status'] == 'success'),
        'failed_trials': sum(1 for r in results if r['status'] == 'failed'),
        'timestamp': datetime.now().isoformat(),
        'trials': results
    }

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {summary_file}")

    # Compute average metrics if we have successful results
    if successful_results:
        print(f"\n{'='*70}")
        print("AVERAGE METRICS (Successful Trials)")
        print(f"{'='*70}")

        # Aggregate metrics
        all_metrics = {}
        for result in successful_results:
            for metric, value in result.get('test_metrics', {}).items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)

        # Compute mean and std
        import numpy as np
        for metric in sorted(all_metrics.keys()):
            values = all_metrics[metric]
            mean = np.mean(values)
            std = np.std(values)
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")

        # Save aggregated metrics
        aggregated_file = output_base / 'aggregated_metrics.json'
        aggregated = {
            'dataset': 'amazon-videogames',
            'model': 'KGAT',
            'num_trials': len(successful_results),
            'metrics': {}
        }

        for metric, values in all_metrics.items():
            aggregated['metrics'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': [float(v) for v in values]
            }

        with open(aggregated_file, 'w') as f:
            json.dump(aggregated, f, indent=2)

        print(f"\n✓ Aggregated metrics saved to: {aggregated_file}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful

    print(f"\nTotal trials: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if successful > 0:
        print(f"\nResults saved to: {output_base}")
        print(f"  - Individual trial results: trial_*/trial_results.json")
        print(f"  - Trials summary: trials_summary.json")
        print(f"  - Aggregated metrics: aggregated_metrics.json")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
