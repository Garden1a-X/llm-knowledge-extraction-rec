#!/usr/bin/env python3
"""
Run KGAT baseline 5-trial experiments for Amazon Beauty dataset.

Uses standard metadata KG (categories + price) for fair comparison.

Usage:
    # First generate standard KG:
    python scripts/generate_beauty_standard_kg.py \
        --output_dir /data/xuao/KG4RecEval/dataset/amazon-beauty

    # Then run KGAT:
    python scripts/run_beauty_kgat_5trials.py \
        --data_path /data/xuao/KG4RecEval/dataset
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from recbole.quick_start import run_recbole


def run_trial(data_path: str, seed: int, trial_num: int, output_dir: Path):
    """Run single KGAT trial."""

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        # Data - use external data path with standard metadata KG
        'data_path': data_path,
        'dataset': 'amazon-beauty',
        'load_col': {
            'inter': ['user_id', 'item_id', 'timestamp'],
            'kg': ['head_id', 'relation_id', 'tail_id'],
            'link': ['item_id', 'entity_id']
        },

        # Data split
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

        # KGAT parameters
        'embedding_size': 64,
        'kg_embedding_size': 64,
        'reg_weight': 0.0001,

        # Training
        'epochs': 300,
        'train_batch_size': 2048,
        'eval_batch_size': 100000,
        'learning_rate': 0.001,
        'stopping_step': 10,

        # Random seed
        'seed': seed,
        'reproducibility': True,
        'state': 'INFO',

        # Checkpoint
        'checkpoint_dir': str(checkpoint_dir),
        'show_progress': True,
        'save_dataset': False,
        'save_dataloaders': False,
    }

    print(f"\n{'='*60}")
    print(f"KGAT Trial {trial_num} (seed={seed})")
    print(f"{'='*60}")
    print(f"Data path: {data_path}")
    print(f"KG type: Standard metadata (categories + price)")
    print()

    result = run_recbole(
        model='KGAT',
        dataset='amazon-beauty',
        config_dict=config_dict,
        saved=True
    )

    test_result = result.get('test_result', {})

    # Save trial results
    result_dict = {
        'trial': trial_num,
        'seed': seed,
        'model': 'KGAT',
        'dataset': 'amazon-beauty',
        'kg_type': 'standard_metadata',
        'timestamp': datetime.now().isoformat(),
        'test_metrics': {k: float(v) for k, v in test_result.items()},
    }

    with open(output_dir / 'trial_results.json', 'w') as f:
        json.dump(result_dict, f, indent=2)

    print(f"\n✓ Trial {trial_num} completed")
    print(f"  NDCG@10: {test_result.get('ndcg@10', 'N/A'):.4f}")
    print(f"  Recall@10: {test_result.get('recall@10', 'N/A'):.4f}")

    return result_dict


def main():
    parser = argparse.ArgumentParser(description='Run KGAT 5-trial on Beauty')
    parser.add_argument('--data_path', type=str,
                        default='/data/xuao/KG4RecEval/dataset',
                        help='Path to dataset directory containing amazon-beauty/')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[42, 123, 456, 789, 2024],
                        help='Random seeds for trials')
    args = parser.parse_args()

    print("=" * 60)
    print("Amazon Beauty - KGAT Baseline 5-Trial")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Seeds: {args.seeds}")
    print(f"KG type: Standard metadata (categories + price)")
    print()

    # Check required files
    dataset_dir = Path(args.data_path) / 'amazon-beauty'
    required_files = ['amazon-beauty.inter', 'amazon-beauty.kg', 'amazon-beauty.link']

    missing = [f for f in required_files if not (dataset_dir / f).exists()]
    if missing:
        print(f"Error: Missing files in {dataset_dir}:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease run first:")
        print("  python scripts/generate_beauty_standard_kg.py \\")
        print(f"      --output_dir {dataset_dir}")
        return

    print("✓ All required files found\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(f"outputs/beauty/kgat_{timestamp}")

    all_results = []

    for i, seed in enumerate(args.seeds):
        trial_num = i + 1
        trial_output = output_base / f"trial_{trial_num}_seed_{seed}"

        try:
            result = run_trial(args.data_path, seed, trial_num, trial_output)
            all_results.append(result)
        except Exception as e:
            print(f"\n✗ Trial {trial_num} failed: {e}")
            import traceback
            traceback.print_exc()

    # Save summary
    if all_results:
        metrics = {}
        for key in all_results[0]['test_metrics']:
            values = [r['test_metrics'][key] for r in all_results]
            metrics[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values
            }

        summary = {
            'model': 'KGAT',
            'dataset': 'amazon-beauty',
            'kg_type': 'standard_metadata',
            'num_trials': len(all_results),
            'seeds': args.seeds[:len(all_results)],
            'metrics': metrics,
            'trials': all_results
        }

        with open(output_base / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        for key in ['ndcg@10', 'recall@10', 'hit@10']:
            if key in metrics:
                print(f"  {key}: {metrics[key]['mean']:.4f} ± {metrics[key]['std']:.4f}")
        print(f"\nResults saved to: {output_base}")

    print("=" * 60)


if __name__ == '__main__':
    main()
