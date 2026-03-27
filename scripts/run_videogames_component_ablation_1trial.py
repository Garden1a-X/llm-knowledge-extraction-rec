#!/usr/bin/env python3
"""
Run 1-trial component ablation experiments on Amazon Video Games dataset.

Component ablation experiments:
1. full - Full model (baseline)
2. wo_contrast - Without contrastive learning
3. wo_mask - Without entity masking
4. kg_only - Only KG view (without CF view)
5. cf_only - Only CF view (without KG view)

Usage:
    # Run all component ablation experiments
    python scripts/run_videogames_component_ablation_1trial.py --method all

    # Run specific ablation
    python scripts/run_videogames_component_ablation_1trial.py --method wo_contrast

    # Create configs only (don't run experiments)
    python scripts/run_videogames_component_ablation_1trial.py --create-configs-only
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import subprocess
import time
import json
import yaml
from datetime import datetime


# Base config for Video Games
BASE_CONFIG = {
    'name': 'ours_videogames_component_ablation',
    'description': 'Component ablation experiment on Amazon Video Games dataset',

    'data': {
        'item_kg_path': 'data/recbole/amazon-videogames/amazon-videogames.item.kg',
        'user_kg_path': 'data/recbole/amazon-videogames/amazon-videogames.user.kg',
        'inter_path': 'data/recbole/amazon-videogames/amazon-videogames.inter',
        'train_ratio': 0.7,
        'val_ratio': 0.1,
        'test_ratio': 0.2,
        'time_based_split': True,
        'per_user_split': True,
        'min_rating': 4.0,
    },

    'model': {
        'embedding_dim': 64,
        'num_gnn_layers': 2,
        'gat_heads': 4,
        'dropout': 0.2,
        'use_mask': True,
        'mask_min_freq': 5,
        'mask_max_freq': 1000,
    },

    'loss': {
        'alpha_contrast': 0.01,
        'beta_align': 0.01,
        'gamma_mask': 0.001,
        'temperature_rec': 0.2,
        'temperature_contrast': 0.1,
        'lambda_sparse': 1.0,
        'lambda_entropy': 0.1,
        'num_neg_align': 5,
    },

    'train': {
        'batch_size': 2048,
        'num_negatives': 1,
        'learning_rate': 0.001,
        'weight_decay': 0.00001,
        'num_epochs': 300,
        'early_stop_patience': 10,
        'eval_every': 1,
        'eval_mode': 'uni100',
        'eval_num_neg': 99,
        'log_every': 1,
        'save_every': 10,
        'num_workers': 4,
        'device': 'cuda',
        'random_seed': 42,
    },

    'ablation': {
        'use_contrast': True,
        'use_align': True,
        'use_mask': True,
        'use_cf_view': True,
        'use_kg_view': True,
    },

    'output_dir': 'outputs',
    'checkpoint_dir': 'checkpoints',
}

# Component ablation configurations
ABLATION_CONFIGS = {
    'full': {
        'name': 'ours_videogames_full',
        'description': 'Full model with all components',
        # Default settings
    },
    'wo_contrast': {
        'name': 'ours_videogames_wo_contrast',
        'description': 'Without multi-view contrastive learning',
        'loss': {
            'alpha_contrast': 0.0,  # Disable contrastive loss
        },
        'ablation': {
            'use_contrast': False,
        }
    },
    'wo_mask': {
        'name': 'ours_videogames_wo_mask',
        'description': 'Without entity masking',
        'model': {
            'use_mask': False,
        },
        'loss': {
            'gamma_mask': 0.0,  # Disable mask loss
        },
        'ablation': {
            'use_mask': False,
        }
    },
    'kg_only': {
        'name': 'ours_videogames_kg_only',
        'description': 'Only KG view (without CF view)',
        'ablation': {
            'use_cf_view': False,
            'use_kg_view': True,
        }
    },
    'cf_only': {
        'name': 'ours_videogames_cf_only',
        'description': 'Only CF view (without KG view)',
        'ablation': {
            'use_cf_view': True,
            'use_kg_view': False,
        }
    },
}


def deep_merge(base, override):
    """Deep merge two dictionaries."""
    import copy
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def create_ablation_configs():
    """Create all component ablation config files for Video Games dataset."""
    config_dir = Path('configs')
    config_dir.mkdir(exist_ok=True)

    created_configs = {}

    for method, overrides in ABLATION_CONFIGS.items():
        config = deep_merge(BASE_CONFIG, overrides)
        config_path = config_dir / f"ours_videogames_component_{method}.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        created_configs[method] = str(config_path)
        print(f"Created: {config_path}")

    return created_configs


def run_experiment(config_path, verbose=True):
    """Run a single experiment."""
    cmd = [
        'python', 'scripts/train_model.py',
        '--config', config_path,
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    start_time = time.time()

    if verbose:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        output_lines = []
        for line in process.stdout:
            print(line, end='')
            output_lines.append(line)

        process.wait()
        returncode = process.returncode
        output = ''.join(output_lines)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        returncode = result.returncode
        output = result.stdout + result.stderr

    elapsed = time.time() - start_time

    if returncode != 0:
        print(f"ERROR: Experiment failed with return code {returncode}")
        return None

    # Parse results
    metrics = parse_metrics(output)
    metrics['elapsed_time'] = elapsed

    return metrics


def parse_metrics(output):
    """Parse metrics from training output."""
    metrics = {}
    lines = output.split('\n')

    for line in lines:
        if 'NDCG@10:' in line:
            try:
                value = float(line.split('NDCG@10:')[1].strip().split()[0])
                metrics['NDCG@10'] = value
            except (IndexError, ValueError):
                pass
        if 'Recall@10:' in line:
            try:
                value = float(line.split('Recall@10:')[1].strip().split()[0])
                metrics['Recall@10'] = value
            except (IndexError, ValueError):
                pass

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run Video Games component ablation experiments (1 trial)')
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'full', 'wo_contrast', 'wo_mask', 'kg_only', 'cf_only'],
                        help='Ablation method to run')
    parser.add_argument('--create-configs-only', action='store_true',
                        help='Only create config files, do not run experiments')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Show real-time output')
    parser.add_argument('--output', type=str, default='results/videogames/component_ablation_results_1trial.json',
                        help='Output JSON file for results')

    args = parser.parse_args()

    # Create config files
    print("=" * 70)
    print("Creating component ablation config files...")
    print("=" * 70)

    config_paths = create_ablation_configs()

    if args.create_configs_only:
        print("\n✅ Config files created. Exiting without running experiments.")
        return

    # Determine which methods to run
    if args.method == 'all':
        methods = list(ABLATION_CONFIGS.keys())
    else:
        methods = [args.method]

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if any
    all_results = {}
    if Path(args.output).exists():
        with open(args.output, 'r') as f:
            all_results = json.load(f)

    print("\n" + "=" * 70)
    print("Video Games Component Ablation Experiments (1 Trial)")
    print("=" * 70)
    print(f"Methods: {methods}")
    print(f"Output: {args.output}")
    print("=" * 70)

    for method in methods:
        config_path = config_paths[method]

        print(f"\n{'='*70}")
        print(f"Running: {method}")
        print(f"Config: {config_path}")
        print(f"{'='*70}")

        metrics = run_experiment(config_path, args.verbose)

        if metrics:
            all_results[method] = {
                'NDCG@10': metrics.get('NDCG@10'),
                'Recall@10': metrics.get('Recall@10'),
                'elapsed_time': metrics.get('elapsed_time'),
                'timestamp': datetime.now().isoformat()
            }

            print(f"\n✅ {method} Results:")
            print(f"   NDCG@10: {metrics.get('NDCG@10', 'N/A')}")
            print(f"   Recall@10: {metrics.get('Recall@10', 'N/A')}")
            print(f"   Time: {metrics.get('elapsed_time', 0)/60:.1f} min")

            # Save after each method
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
        else:
            print(f"\n❌ {method} FAILED!")

    # Print summary
    print("\n" + "=" * 70)
    print("Component Ablation Summary Table")
    print("=" * 70)
    print(f"{'Method':<15} | {'NDCG@10':<20} | {'Recall@10':<12}")
    print("-" * 55)

    full_ndcg = all_results.get('full', {}).get('NDCG@10')

    for method in ['full', 'wo_contrast', 'wo_mask', 'kg_only', 'cf_only']:
        if method in all_results:
            ndcg = all_results[method].get('NDCG@10')
            recall = all_results[method].get('Recall@10')

            if ndcg and full_ndcg and method != 'full':
                drop = (ndcg - full_ndcg) / full_ndcg * 100
                ndcg_str = f"{ndcg:.4f} ({drop:+.1f}%)"
            else:
                ndcg_str = f"{ndcg:.4f}" if ndcg else "N/A"

            recall_str = f"{recall:.4f}" if recall else "N/A"
            print(f"{method:<15} | {ndcg_str:<20} | {recall_str:<12}")

    print("=" * 70)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
