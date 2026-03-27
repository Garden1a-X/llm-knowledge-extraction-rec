#!/usr/bin/env python3
"""
Run RecBole Baseline Experiments

Usage:
    python scripts/run_recbole_baselines.py --model KGAT
    python scripts/run_recbole_baselines.py --model LightGCN
"""

import argparse
from recbole.quick_start import run_recbole

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['KGAT', 'LightGCN', 'NGCF', 'BPR'],
                        help='Model name')
    parser.add_argument('--config', type=str, default=None,
                        help='Config file path (optional)')
    args = parser.parse_args()

    # 根据模型选择配置文件
    if args.config is None:
        if args.model == 'KGAT':
            config_file = 'configs/recbole_kgat.yaml'
        elif args.model == 'LightGCN':
            config_file = 'configs/recbole_lightgcn.yaml'
        else:
            config_file = None
    else:
        config_file = args.config

    print(f"Running {args.model} baseline...")
    print(f"Config: {config_file}")

    # 运行RecBole
    run_recbole(
        model=args.model,
        dataset='ml-1m',
        config_file_list=[config_file] if config_file else None
    )

if __name__ == '__main__':
    main()
