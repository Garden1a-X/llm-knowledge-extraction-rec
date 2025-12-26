#!/usr/bin/env python3
"""
Train LightGCN model on MovieLens 1M dataset.

This script:
1. Loads MovieLens 1M data
2. Splits into train/val/test
3. Builds user-item bipartite graph
4. Trains LightGCN model
5. Evaluates on test set
6. Saves model and results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
import argparse
import json
import numpy as np
from datetime import datetime

from src.data.loader import MovieLensLoader
from src.data.splitter import DataSplitter, create_user_item_mapping
from src.data.graph_builder import BipartiteGraphBuilder, create_train_dataloader
from src.model.lightgcn import LightGCN
from src.model.trainer import Trainer
from src.model.evaluator import evaluate_model_on_dataset, print_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train LightGCN model')

    # Data args
    parser.add_argument('--data_dir', type=str, default='data/raw/ml-1m',
                        help='Path to MovieLens data directory')
    parser.add_argument('--output_dir', type=str, default='outputs/lightgcn',
                        help='Output directory for models and results')

    # Split args
    parser.add_argument('--split_method', type=str, default='temporal',
                        choices=['temporal', 'random'],
                        help='Data split method')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test set ratio')

    # Model args
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of GCN layers')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')

    # Training args
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--reg_weight', type=float, default=1e-4,
                        help='L2 regularization weight')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')

    # Evaluation args
    parser.add_argument('--k_list', type=int, nargs='+', default=[10, 20],
                        help='K values for evaluation metrics')

    # Other args
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save model and results')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    # Parse arguments
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("LightGCN Training")
    print(f"{'='*80}")
    print(f"Device: {args.device}")
    print(f"Output dir: {output_dir}")
    print()

    # ========================================================================
    # Load data
    # ========================================================================
    print("Loading MovieLens 1M data...")
    loader = MovieLensLoader(args.data_dir)
    ratings = loader.load_ratings()
    print(f"  Loaded {len(ratings)} ratings")
    print(f"  Users: {ratings['user_id'].nunique()}")
    print(f"  Movies: {ratings['movie_id'].nunique()}")

    # ========================================================================
    # Split data
    # ========================================================================
    print(f"\nSplitting data ({args.split_method})...")
    splitter = DataSplitter(
        ratings,
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        split_method=args.split_method,
        random_seed=args.seed
    )
    split_data = splitter.split()
    train_df = split_data['train']
    val_df = split_data['val']
    test_df = split_data['test']

    # Print statistics
    stats = splitter.get_statistics()
    print(f"  Train: {stats['train_ratings']} ratings ({stats['train_ratio']:.1%})")
    print(f"  Val:   {stats['val_ratings']} ratings ({stats['val_ratio']:.1%})")
    print(f"  Test:  {stats['test_ratings']} ratings ({stats['test_ratio']:.1%})")

    # Save split data
    if not args.no_save:
        split_dir = output_dir / 'data_split'
        splitter.save(split_dir)

    # ========================================================================
    # Create mappings
    # ========================================================================
    print("\nCreating user-item mappings...")
    user2idx, idx2user, item2idx, idx2item = create_user_item_mapping(ratings)
    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"  Users: {n_users}")
    print(f"  Items: {n_items}")

    # Save mappings
    if not args.no_save:
        mapping_path = output_dir / 'mappings.json'
        with open(mapping_path, 'w') as f:
            json.dump({
                'user2idx': {int(k): int(v) for k, v in user2idx.items()},
                'item2idx': {int(k): int(v) for k, v in item2idx.items()}
            }, f)
        print(f"  Saved mappings to {mapping_path}")

    # ========================================================================
    # Build graph
    # ========================================================================
    print("\nBuilding user-item bipartite graph...")
    graph_builder = BipartiteGraphBuilder(
        train_df,
        user2idx,
        item2idx,
        n_users,
        n_items
    )
    graph = graph_builder.build_graph()
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.edge_index.size(1)}")

    # Get user_items dict for training
    user_items = graph_builder.user_items

    # ========================================================================
    # Initialize model
    # ========================================================================
    print("\nInitializing LightGCN model...")
    model = LightGCN(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Layers: {args.n_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ========================================================================
    # Initialize trainer
    # ========================================================================
    print("\nInitializing trainer...")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = output_dir / 'checkpoints' if not args.no_save else None
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=args.device,
        checkpoint_dir=checkpoint_dir,
        patience=args.patience,
        monitor_metric=f'NDCG@{args.k_list[0]}'
    )
    print(f"  Optimizer: Adam (lr={args.lr})")
    print(f"  Patience: {args.patience}")
    print(f"  Monitor metric: NDCG@{args.k_list[0]}")

    # ========================================================================
    # Train model
    # ========================================================================
    print("\nStarting training...")
    print(f"  Epochs: {args.n_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Reg weight: {args.reg_weight}")

    # Create training data loader (generator)
    def get_train_loader():
        return create_train_dataloader(
            train_df,
            user2idx,
            item2idx,
            n_users,
            n_items,
            batch_size=args.batch_size,
            n_negatives=1,
            shuffle=True,
            random_seed=args.seed
        )

    trainer.fit(
        edge_index=graph.edge_index,
        train_loader=get_train_loader(),
        val_df=val_df,
        train_df=train_df,
        user2idx=user2idx,
        item2idx=item2idx,
        n_epochs=args.n_epochs,
        reg_weight=args.reg_weight,
        k_list=args.k_list,
        val_every=1,
        verbose=True
    )

    # ========================================================================
    # Evaluate on test set
    # ========================================================================
    print("\n" + "="*80)
    print("Final Evaluation on Test Set")
    print("="*80)

    # Load best model
    if checkpoint_dir and (checkpoint_dir / 'best_model.pth').exists():
        print("\nLoading best model...")
        trainer.load_checkpoint('best_model.pth')

    # Evaluate
    test_results = evaluate_model_on_dataset(
        model=model,
        edge_index=graph.edge_index.to(args.device),
        test_df=test_df,
        train_df=train_df,
        user2idx=user2idx,
        item2idx=item2idx,
        k_list=args.k_list,
        batch_size=256,
        device=args.device
    )

    print_results(test_results)

    # ========================================================================
    # Save results
    # ========================================================================
    if not args.no_save:
        # Save test results
        results_path = output_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"Results saved to {results_path}")

        # Save config
        config_path = output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)
        print(f"Config saved to {config_path}")

        # Save training history
        history_path = output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'train': trainer.train_history,
                'val': trainer.val_history
            }, f, indent=2)
        print(f"Training history saved to {history_path}")

    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
