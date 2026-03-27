#!/usr/bin/env python3
"""
Run baseline models using RecBole.

Supports: LightGCN, KGAT, RippleNet, and other RecBole models.

Usage:
    python baselines/run_baseline.py --model LightGCN --dataset ml-1m
    python baselines/run_baseline.py --model KGAT --dataset ml-1m --use_kg
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime

# Import RecBole
try:
    from recbole.quick_start import run_recbole
    from recbole.config import Config
except ImportError as e:
    print("ERROR: Failed to import RecBole!")
    print(f"Import error: {e}")
    print()
    print("RecBole may not be installed, or there's a dependency issue.")
    print("Please try: pip install recbole")
    print()
    print("If RecBole is installed, run this to diagnose:")
    print("  python baselines/test_recbole.py")
    import traceback
    traceback.print_exc()
    exit(1)


def get_model_config(model_name: str, use_kg: bool = False):
    """
    Get recommended configuration for each model.

    Args:
        model_name: Model name (e.g., 'LightGCN', 'KGAT')
        use_kg: Whether the model uses knowledge graph

    Returns:
        Dictionary of model-specific config
    """
    # Base config for all models
    base_config = {
        # Training
        'epochs': 300,
        'train_batch_size': 2048,
        'eval_batch_size': 4096,
        'learning_rate': 0.001,
        'stopping_step': 10,  # Early stopping patience
        'eval_step': 1,  # Evaluate every N epochs

        # Evaluation
        'metrics': ['Recall', 'NDCG', 'Precision', 'Hit'],
        'topk': [10, 20],
        'valid_metric': 'NDCG@10',

        # Data split
        'eval_args': {
            'split': {'RS': [0.7, 0.1, 0.2]},  # 70% train, 10% val, 20% test
            'order': 'TO',  # Time-based ordering (TO) or Random (RO)
            'mode': 'full',  # Full ranking
        },

        # Data loading - IMPORTANT: load timestamp for temporal split
        'load_col': {
            'inter': ['user_id', 'item_id', 'timestamp']
        },

        # Other
        'seed': 42,
        'gpu_id': '0',  # Will be overridden by --device
    }

    # Model-specific configs
    model_configs = {
        'LightGCN': {
            'embedding_size': 64,
            'n_layers': 3,
            'reg_weight': 1e-4,
        },

        'KGAT': {
            'embedding_size': 64,
            'kg_embedding_size': 64,
            'n_layers': 3,
            'reg_weight': 1e-5,
            'aggregator_type': 'bi-interaction',  # 'gcn', 'graphsage', 'bi-interaction'
        },

        'NGCF': {
            'embedding_size': 64,
            'hidden_size_list': [64, 64, 64],
            'node_dropout': [0.1, 0.1, 0.1],
            'message_dropout': [0.1, 0.1, 0.1],
            'reg_weight': 1e-5,
        },

        'BPR': {
            'embedding_size': 64,
            'reg_weight': 1e-5,
        },

        'NeuMF': {
            'mf_embedding_size': 64,
            'mlp_embedding_size': 64,
            'mlp_hidden_size': [128, 64, 32],
            'dropout_prob': 0.1,
        },
    }

    config = base_config.copy()
    if model_name in model_configs:
        config.update(model_configs[model_name])

    # Add KG loading config if use_kg is True
    if use_kg:
        config['load_col']['kg'] = ['head_id', 'relation_id', 'tail_id']

    return config


def run_baseline(
    model: str,
    dataset: str = 'ml-1m',
    data_path: str = 'data/recbole',
    output_dir: str = 'outputs/baselines',
    device: str = 'cuda',
    use_kg: bool = False,
    config_file: str = None,
    **kwargs
):
    """
    Run baseline model with RecBole.

    Args:
        model: Model name (e.g., 'LightGCN', 'KGAT')
        dataset: Dataset name
        data_path: Path to RecBole format data
        output_dir: Output directory
        device: Device ('cuda' or 'cpu')
        use_kg: Whether to use knowledge graph features
        config_file: Optional config file path
        **kwargs: Additional config parameters
    """
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_output_dir = Path(output_dir) / f'{model}_{dataset}_{timestamp}'
    model_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Running {model} on {dataset}")
    print(f"{'='*80}")

    # Get model config
    config_dict = get_model_config(model, use_kg)

    # Override with user params
    config_dict.update({
        'model': model,
        'dataset': dataset,
        'data_path': data_path,
        'checkpoint_dir': str(model_output_dir / 'checkpoints'),
        'gpu_id': '0' if device == 'cuda' else '-1',
    })

    # Add any additional kwargs
    config_dict.update(kwargs)

    # Print config
    print("\nConfiguration:")
    print("-" * 80)
    for key, value in sorted(config_dict.items()):
        print(f"  {key:25s}: {value}")
    print("-" * 80)

    # Save config
    config_path = model_output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"\nConfig saved to: {config_path}")

    # Run RecBole
    print("\nStarting training...\n")

    try:
        result = run_recbole(
            model=model,
            dataset=dataset,
            config_dict=config_dict,
            config_file_list=[config_file] if config_file else None,
            saved=True,  # Save model
        )

        # RecBole returns a tuple: (best_valid_score, test_result_dict)
        # Extract the test results
        if isinstance(result, tuple) and len(result) == 2:
            best_valid_score, test_result = result
            result_dict = {
                'best_valid_score': float(best_valid_score) if best_valid_score is not None else None,
                'test_result': test_result
            }
        elif isinstance(result, dict):
            # If it's already a dict, check if it has test_result
            if 'test_result' not in result:
                result_dict = {'test_result': result}
            else:
                result_dict = result
        else:
            print(f"Warning: Unexpected result type: {type(result)}")
            result_dict = {'test_result': result}

        # Save results
        results_path = model_output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

        print(f"\n{'='*80}")
        print("Training completed!")
        print(f"{'='*80}")
        print(f"Results saved to: {results_path}")
        print(f"Model saved to: {model_output_dir / 'checkpoints'}")

        # Print test results
        print(f"\n{'='*80}")
        print("Test Results:")
        print(f"{'='*80}")
        test_result = result_dict.get('test_result', {})
        if isinstance(test_result, dict):
            for metric, value in sorted(test_result.items()):
                if isinstance(value, (int, float)):
                    print(f"  {metric:20s}: {value:.4f}")
        print(f"{'='*80}\n")

        return result_dict

    except Exception as e:
        print(f"\nERROR: Training failed!")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Run baseline models with RecBole')

    # Required args
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (LightGCN, KGAT, NGCF, BPR, NeuMF, etc.)')

    # Data args
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        help='Dataset name')
    parser.add_argument('--data_path', type=str, default='data/recbole',
                        help='Path to RecBole format data')

    # Output args
    parser.add_argument('--output_dir', type=str, default='outputs/baselines',
                        help='Output directory')

    # Training args
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--embedding_size', type=int, default=None,
                        help='Embedding size')

    # Other args
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--use_kg', action='store_true',
                        help='Use knowledge graph features (for KGAT, etc.)')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Build kwargs from args
    kwargs = {}
    if args.epochs is not None:
        kwargs['epochs'] = args.epochs
    if args.batch_size is not None:
        kwargs['train_batch_size'] = args.batch_size
    if args.lr is not None:
        kwargs['learning_rate'] = args.lr
    if args.embedding_size is not None:
        kwargs['embedding_size'] = args.embedding_size
    if args.seed is not None:
        kwargs['seed'] = args.seed

    # Run baseline
    run_baseline(
        model=args.model,
        dataset=args.dataset,
        data_path=args.data_path,
        output_dir=args.output_dir,
        device=args.device,
        use_kg=args.use_kg,
        config_file=args.config_file,
        **kwargs
    )


if __name__ == '__main__':
    main()
