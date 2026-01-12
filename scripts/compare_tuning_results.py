#!/usr/bin/env python3
"""
对比4个超参数调优实验的结果

读取每个实验的history.json，对比最佳验证指标和最终测试指标。

Usage:
    python scripts/compare_tuning_results.py
"""

import json
from pathlib import Path
from tabulate import tabulate
import sys

# 配置名称映射
CONFIG_NAMES = {
    'tune1_loss_weights': 'Tune1: Loss Weights',
    'tune2_lr_embed': 'Tune2: LR & Embed',
    'tune3_depth_dropout': 'Tune3: Depth & Dropout',
    'tune4_hybrid': 'Tune4: Hybrid'
}

def find_latest_results():
    """找到最新的实验结果"""
    output_dir = Path('outputs/ours')

    if not output_dir.exists():
        print("❌ 未找到输出目录: outputs/ours")
        return []

    results = []

    for config_name in CONFIG_NAMES.keys():
        # 查找该配置的所有结果目录
        matching_dirs = sorted(
            output_dir.glob(f'{config_name}_*'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if matching_dirs:
            # 取最新的
            latest_dir = matching_dirs[0]
            history_file = latest_dir / 'history.json'

            if history_file.exists():
                results.append({
                    'config': config_name,
                    'path': history_file
                })

    return results


def load_and_compare():
    """加载并对比结果"""
    results = find_latest_results()

    if not results:
        print("❌ 未找到任何实验结果")
        print("\n请先运行实验:")
        print("  python scripts/run_parallel_tuning.py")
        return

    print("="*80)
    print("超参数调优结果对比")
    print("="*80)

    # 加载数据
    data = []
    for result in results:
        with open(result['path'], 'r') as f:
            history = json.load(f)

        config_name = result['config']
        display_name = CONFIG_NAMES.get(config_name, config_name)

        test_metrics = history.get('final_test_metrics', {})
        best_ndcg = history.get('best_ndcg', 0.0)
        best_epoch = history.get('best_epoch', 0)

        data.append({
            'Config': display_name,
            'Best Val NDCG@10': f"{best_ndcg:.4f}",
            'Best Epoch': best_epoch,
            'Test NDCG@10': f"{test_metrics.get('NDCG@10', 0.0):.4f}",
            'Test Recall@10': f"{test_metrics.get('Recall@10', 0.0):.4f}",
            'Test NDCG@20': f"{test_metrics.get('NDCG@20', 0.0):.4f}",
            'Test Recall@20': f"{test_metrics.get('Recall@20', 0.0):.4f}",
        })

    # 打印表格
    table = tabulate(data, headers='keys', tablefmt='grid')
    print("\n" + table)

    # 找到最佳配置
    best_idx = max(range(len(data)), key=lambda i: float(data[i]['Test NDCG@10']))
    print("\n" + "="*80)
    print(f"✓ 最佳配置: {data[best_idx]['Config']}")
    print(f"  Test NDCG@10: {data[best_idx]['Test NDCG@10']}")
    print("="*80)

    # 详细配置对比
    print("\n超参数设置对比:\n")

    config_files = {
        'tune1_loss_weights': 'configs/tune1_loss_weights.yaml',
        'tune2_lr_embed': 'configs/tune2_lr_embed.yaml',
        'tune3_depth_dropout': 'configs/tune3_depth_dropout.yaml',
        'tune4_hybrid': 'configs/tune4_hybrid.yaml'
    }

    import yaml

    # 提取关键超参数
    hyperparam_data = []
    for config_name in CONFIG_NAMES.keys():
        config_file = config_files[config_name]
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            hyperparam_data.append({
                'Config': CONFIG_NAMES[config_name],
                'Embed Dim': config['model']['embedding_dim'],
                'GNN Layers': config['model']['num_gnn_layers'],
                'GAT Heads': config['model']['gat_heads'],
                'Dropout': config['model']['dropout'],
                'LR': config['train']['learning_rate'],
                'alpha': config['loss']['alpha_contrast'],
                'beta': config['loss']['beta_align'],
                'gamma': config['loss']['gamma_mask']
            })

    hyperparam_table = tabulate(hyperparam_data, headers='keys', tablefmt='grid')
    print(hyperparam_table)

    print("\n" + "="*80)
    print("提示: 可以在 log_tensorboard/ 目录下使用 TensorBoard 查看训练曲线")
    print("  tensorboard --logdir log_tensorboard")
    print("="*80)


if __name__ == '__main__':
    try:
        import yaml
        from tabulate import tabulate
    except ImportError:
        print("❌ 缺少依赖包，请安装:")
        print("  pip install pyyaml tabulate")
        sys.exit(1)

    load_and_compare()
