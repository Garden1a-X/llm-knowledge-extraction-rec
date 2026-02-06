#!/usr/bin/env python3
"""
Run all ML-1M Full Rank experiments for paper appendix.

Usage:
    # Run all methods
    python scripts/run_ml1m_fullrank_all.py

    # Run specific method
    python scripts/run_ml1m_fullrank_all.py --method bpr
    python scripts/run_ml1m_fullrank_all.py --method lightgcn
    python scripts/run_ml1m_fullrank_all.py --method kgat
    python scripts/run_ml1m_fullrank_all.py --method ours
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_bpr():
    """Run BPR baseline with full rank."""
    print("=" * 60)
    print("Running BPR (Full Rank)")
    print("=" * 60)

    from recbole.quick_start import run_recbole
    run_recbole(
        model='BPR',
        dataset='ml-1m',
        config_file_list=['configs/recbole_ml1m_bpr_full.yaml']
    )


def run_lightgcn():
    """Run LightGCN baseline with full rank."""
    print("=" * 60)
    print("Running LightGCN (Full Rank)")
    print("=" * 60)

    from recbole.quick_start import run_recbole
    run_recbole(
        model='LightGCN',
        dataset='ml-1m',
        config_file_list=['configs/recbole_ml1m_lightgcn_full.yaml']
    )


def run_kgat():
    """Run KGAT baseline with full rank."""
    print("=" * 60)
    print("Running KGAT (Full Rank)")
    print("=" * 60)

    from recbole.quick_start import run_recbole
    run_recbole(
        model='KGAT',
        dataset='ml-1m',
        config_file_list=['configs/recbole_kgat.yaml']  # Already configured with mode: full
    )


def run_ours():
    """Run our method with full rank."""
    print("=" * 60)
    print("Running Ours (Full Rank)")
    print("=" * 60)

    subprocess.run([
        sys.executable, 'scripts/train_model.py',
        '--config', 'configs/ours_full_ml1m_fullrank.yaml'
    ], check=True)


def main():
    parser = argparse.ArgumentParser(description='Run ML-1M Full Rank Experiments')
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'bpr', 'lightgcn', 'kgat', 'ours'],
                        help='Which method to run (default: all)')
    args = parser.parse_args()

    methods = {
        'bpr': run_bpr,
        'lightgcn': run_lightgcn,
        'kgat': run_kgat,
        'ours': run_ours,
    }

    if args.method == 'all':
        for name, func in methods.items():
            try:
                func()
            except Exception as e:
                print(f"Error running {name}: {e}")
                continue
    else:
        methods[args.method]()

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
