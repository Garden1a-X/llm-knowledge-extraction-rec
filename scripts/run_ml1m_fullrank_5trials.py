#!/usr/bin/env python3
"""
Run ML-1M Full Rank experiments with 5 trials for paper appendix.

Usage:
    # Run all methods
    python scripts/run_ml1m_fullrank_5trials.py

    # Run specific method
    python scripts/run_ml1m_fullrank_5trials.py --method bpr
    python scripts/run_ml1m_fullrank_5trials.py --method lightgcn
    python scripts/run_ml1m_fullrank_5trials.py --method kgat
    python scripts/run_ml1m_fullrank_5trials.py --method ours

Note: KGAT uses original ML-1M metadata KG from /data/xuao/KG4RecEval/dataset/
"""

import argparse
import subprocess
import sys
import numpy as np
from pathlib import Path

SEEDS = [42, 123, 456, 789, 2024]

# KGAT uses KG4RecEval's original ML-1M KG (genres + year metadata)
KGAT_DATA_PATH = '/data/xuao/KG4RecEval/dataset/'


def run_kgat_5trials():
    """Run KGAT with original ML-1M KG (from KG4RecEval) for 5 trials."""
    print("=" * 70)
    print("Running KGAT (Full Rank) - 5 Trials")
    print(f"Using KG from: {KGAT_DATA_PATH}")
    print("=" * 70)

    from recbole.quick_start import run_recbole

    results = []
    for i, seed in enumerate(SEEDS):
        print(f"\n--- Trial {i+1}/5 (seed={seed}) ---")

        config_dict = {
            # Data - use KG4RecEval's original KG
            'data_path': KGAT_DATA_PATH,
            'dataset': 'ml-1m',
            'load_col': {
                'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
                'kg': ['head_id', 'relation_id', 'tail_id'],
                'link': ['item_id', 'entity_id']
            },

            # Data split (same as ours) - FULL RANK
            'eval_args': {
                'split': {'RS': [0.7, 0.1, 0.2]},
                'order': 'TO',
                'group_by': 'user',
                'mode': 'full'  # Full rank evaluation
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
            'learning_rate': 0.001,
            'stopping_step': 10,

            # Random seed
            'seed': seed,
            'reproducibility': True,
            'state': 'INFO',
            'show_progress': True,
        }

        try:
            result = run_recbole(
                model='KGAT',
                dataset='ml-1m',
                config_dict=config_dict,
                saved=False
            )

            # Extract metrics from result
            test_result = result['test_result']
            ndcg10 = test_result.get('ndcg@10', 0)
            recall10 = test_result.get('recall@10', 0)

            results.append({'ndcg': ndcg10, 'recall': recall10})
            print(f"Trial {i+1}: NDCG@10={ndcg10:.4f}, Recall@10={recall10:.4f}")

        except Exception as e:
            print(f"Error in trial {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def run_recbole_5trials(model_name, config_file):
    """Run RecBole model with 5 trials."""
    print("=" * 70)
    print(f"Running {model_name} (Full Rank) - 5 Trials")
    print("=" * 70)

    from recbole.quick_start import run_recbole
    import yaml

    results = []
    for i, seed in enumerate(SEEDS):
        print(f"\n--- Trial {i+1}/5 (seed={seed}) ---")

        # Load config and update seed
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        config['seed'] = seed

        # Write temp config
        temp_config = f'/tmp/{model_name}_seed{seed}.yaml'
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)

        try:
            result = run_recbole(
                model=model_name,
                dataset='ml-1m',
                config_file_list=[temp_config]
            )

            # Extract metrics from result
            test_result = result['test_result']
            ndcg10 = test_result.get('ndcg@10', 0)
            recall10 = test_result.get('recall@10', 0)

            results.append({'ndcg': ndcg10, 'recall': recall10})
            print(f"Trial {i+1}: NDCG@10={ndcg10:.4f}, Recall@10={recall10:.4f}")

        except Exception as e:
            print(f"Error in trial {i+1}: {e}")
            continue

    return results


def run_ours_5trials():
    """Run our method with 5 trials."""
    print("=" * 70)
    print("Running Ours (Full Rank) - 5 Trials")
    print("=" * 70)

    import yaml

    results = []
    for i, seed in enumerate(SEEDS):
        print(f"\n--- Trial {i+1}/5 (seed={seed}) ---")

        # Load config and update seed
        with open('configs/ours_full_ml1m_fullrank.yaml', 'r') as f:
            config = yaml.safe_load(f)
        config['train']['random_seed'] = seed

        # Write temp config
        temp_config = f'/tmp/ours_ml1m_fullrank_seed{seed}.yaml'
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)

        try:
            # Use Popen for real-time output
            process = subprocess.Popen(
                [sys.executable, '-u', 'scripts/train_model.py', '--config', temp_config],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            output_lines = []
            for line in process.stdout:
                print(line, end='')
                output_lines.append(line)

            process.wait()
            output = ''.join(output_lines)

            if process.returncode != 0:
                print(f"Error: Process exited with code {process.returncode}")
                continue

            # Parse results from output
            ndcg = None
            recall = None
            for line in output.split('\n'):
                if 'Test NDCG@10:' in line or 'ndcg@10:' in line.lower():
                    try:
                        ndcg = float(line.split(':')[-1].strip())
                    except:
                        pass
                if 'Test Recall@10:' in line or 'recall@10:' in line.lower():
                    try:
                        recall = float(line.split(':')[-1].strip())
                    except:
                        pass

            if ndcg is not None and recall is not None:
                results.append({'ndcg': ndcg, 'recall': recall})
                print(f"Trial {i+1}: NDCG@10={ndcg:.4f}, Recall@10={recall:.4f}")

        except subprocess.TimeoutExpired:
            print(f"Trial {i+1} timed out")
            continue
        except Exception as e:
            print(f"Error in trial {i+1}: {e}")
            continue

    return results


def print_summary(method, results):
    """Print summary statistics."""
    if not results:
        print(f"\n{method}: No results collected")
        return

    ndcgs = [r['ndcg'] for r in results]
    recalls = [r['recall'] for r in results]

    print(f"\n{'=' * 70}")
    print(f"{method} Summary (5 Trials, Full Rank)")
    print(f"{'=' * 70}")
    print(f"  NDCG@10:   {np.mean(ndcgs):.4f} ± {np.std(ndcgs):.4f}")
    print(f"  Recall@10: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"{'=' * 70}")

    # Print for easy copy to LaTeX
    print(f"\nLaTeX format:")
    print(f"{method} & {np.mean(ndcgs):.4f}$\\pm${np.std(ndcgs):.4f} & {np.mean(recalls):.4f}$\\pm${np.std(recalls):.4f} \\\\")


def main():
    parser = argparse.ArgumentParser(description='Run ML-1M Full Rank Experiments (5 Trials)')
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'bpr', 'lightgcn', 'kgat', 'ours'],
                        help='Which method to run (default: all)')
    args = parser.parse_args()

    all_results = {}

    if args.method in ['bpr', 'all']:
        results = run_recbole_5trials('BPR', 'configs/recbole_ml1m_bpr_full.yaml')
        all_results['BPR'] = results
        print_summary('BPR', results)

    if args.method in ['lightgcn', 'all']:
        results = run_recbole_5trials('LightGCN', 'configs/recbole_ml1m_lightgcn_full.yaml')
        all_results['LightGCN'] = results
        print_summary('LightGCN', results)

    if args.method in ['kgat', 'all']:
        results = run_kgat_5trials()
        all_results['KGAT'] = results
        print_summary('KGAT', results)

    if args.method in ['ours', 'all']:
        results = run_ours_5trials()
        all_results['Ours'] = results
        print_summary('Ours', results)

    # Final summary table
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("ML-1M Full Rank Results Summary (5 Trials)")
        print("=" * 70)
        print("| Method | NDCG@10 | Recall@10 |")
        print("|--------|---------|-----------|")
        for method, results in all_results.items():
            if results:
                ndcgs = [r['ndcg'] for r in results]
                recalls = [r['recall'] for r in results]
                print(f"| {method} | {np.mean(ndcgs):.4f} ± {np.std(ndcgs):.4f} | {np.mean(recalls):.4f} ± {np.std(recalls):.4f} |")

        print("\nLaTeX Table:")
        print("\\begin{tabular}{lcc}")
        print("\\toprule")
        print("Method & NDCG@10 & Recall@10 \\\\")
        print("\\midrule")
        for method, results in all_results.items():
            if results:
                ndcgs = [r['ndcg'] for r in results]
                recalls = [r['recall'] for r in results]
                print(f"{method} & {np.mean(ndcgs):.4f}$\\pm${np.std(ndcgs):.4f} & {np.mean(recalls):.4f}$\\pm${np.std(recalls):.4f} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == '__main__':
    main()
