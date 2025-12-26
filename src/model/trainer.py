"""
Trainer for recommendation models.

Handles training loop, validation, checkpointing, and early stopping.
"""

import torch
import torch.optim as optim
from pathlib import Path
from typing import Dict, Optional, Callable
import time
import numpy as np
from .evaluator import evaluate_model_on_dataset, print_results


class Trainer:
    """Trainer for recommendation models."""

    def __init__(
        self,
        model,
        optimizer: optim.Optimizer,
        device: str = 'cpu',
        checkpoint_dir: Optional[Path] = None,
        patience: int = 10,
        monitor_metric: str = 'NDCG@10'
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            optimizer: Optimizer
            device: Device to use
            checkpoint_dir: Directory to save checkpoints
            patience: Patience for early stopping
            monitor_metric: Metric to monitor for best model
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.patience = patience
        self.monitor_metric = monitor_metric

        # Training state
        self.current_epoch = 0
        self.best_metric = -np.inf
        self.patience_counter = 0
        self.train_history = []
        self.val_history = []

        # Create checkpoint directory
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(
        self,
        edge_index: torch.Tensor,
        train_loader,
        reg_weight: float = 1e-4,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            edge_index: Graph edge indices
            train_loader: Training data loader (yields user, pos_item, neg_item batches)
            reg_weight: L2 regularization weight
            verbose: Whether to print progress

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        edge_index = edge_index.to(self.device)

        total_loss = 0.0
        total_bpr_loss = 0.0
        total_reg_loss = 0.0
        n_batches = 0

        start_time = time.time()

        for batch_idx, (users, pos_items, neg_items) in enumerate(train_loader):
            # Move to device
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            # Forward pass
            user_emb, pos_item_emb, neg_item_emb = self.model(
                edge_index,
                users=users,
                pos_items=pos_items,
                neg_items=neg_items
            )

            # Compute loss
            loss, bpr_loss, reg_loss = self.model.bpr_loss(
                user_emb,
                pos_item_emb,
                neg_item_emb,
                reg_weight=reg_weight
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_bpr_loss += bpr_loss.item()
            total_reg_loss += reg_loss.item()
            n_batches += 1

            # Print progress
            if verbose and (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / n_batches
                print(f"  Batch {batch_idx + 1}: loss={avg_loss:.4f}")

        elapsed = time.time() - start_time

        metrics = {
            'loss': total_loss / n_batches,
            'bpr_loss': total_bpr_loss / n_batches,
            'reg_loss': total_reg_loss / n_batches,
            'time': elapsed
        }

        return metrics

    def validate(
        self,
        edge_index: torch.Tensor,
        val_df,
        train_df,
        user2idx: Dict,
        item2idx: Dict,
        k_list: list = [10, 20],
        batch_size: int = 256
    ) -> Dict[str, float]:
        """
        Validate model.

        Args:
            edge_index: Graph edge indices
            val_df: Validation DataFrame
            train_df: Training DataFrame
            user2idx: User to index mapping
            item2idx: Item to index mapping
            k_list: List of K values for metrics
            batch_size: Batch size for evaluation

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        results = evaluate_model_on_dataset(
            model=self.model,
            edge_index=edge_index,
            test_df=val_df,
            train_df=train_df,
            user2idx=user2idx,
            item2idx=item2idx,
            k_list=k_list,
            batch_size=batch_size,
            device=self.device
        )

        return results

    def fit(
        self,
        edge_index: torch.Tensor,
        train_loader,
        val_df,
        train_df,
        user2idx: Dict,
        item2idx: Dict,
        n_epochs: int = 100,
        reg_weight: float = 1e-4,
        k_list: list = [10, 20],
        val_every: int = 1,
        verbose: bool = True
    ):
        """
        Train model for multiple epochs.

        Args:
            edge_index: Graph edge indices
            train_loader: Training data loader
            val_df: Validation DataFrame
            train_df: Training DataFrame
            user2idx: User to index mapping
            item2idx: Item to index mapping
            n_epochs: Number of epochs
            reg_weight: L2 regularization weight
            k_list: List of K values for metrics
            val_every: Validate every N epochs
            verbose: Whether to print progress
        """
        for epoch in range(n_epochs):
            self.current_epoch = epoch + 1

            if verbose:
                print(f"\n{'='*60}")
                print(f"Epoch {self.current_epoch}/{n_epochs}")
                print(f"{'='*60}")

            # Train
            train_metrics = self.train_epoch(
                edge_index,
                train_loader,
                reg_weight=reg_weight,
                verbose=verbose
            )
            self.train_history.append(train_metrics)

            if verbose:
                print(f"Train Loss: {train_metrics['loss']:.4f} "
                      f"(BPR: {train_metrics['bpr_loss']:.4f}, "
                      f"Reg: {train_metrics['reg_loss']:.4f}) "
                      f"[{train_metrics['time']:.1f}s]")

            # Validate
            if (self.current_epoch % val_every == 0) or (self.current_epoch == n_epochs):
                val_metrics = self.validate(
                    edge_index,
                    val_df,
                    train_df,
                    user2idx,
                    item2idx,
                    k_list=k_list
                )
                self.val_history.append(val_metrics)

                if verbose:
                    print(f"\nValidation:")
                    print_results(val_metrics)

                # Check for improvement
                current_metric = val_metrics.get(self.monitor_metric, -np.inf)

                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.patience_counter = 0

                    if verbose:
                        print(f"✓ New best {self.monitor_metric}: {current_metric:.4f}")

                    # Save best model
                    if self.checkpoint_dir:
                        self.save_checkpoint('best_model.pth')
                else:
                    self.patience_counter += 1

                    if verbose:
                        print(f"  No improvement ({self.patience_counter}/{self.patience})")

                    # Early stopping
                    if self.patience_counter >= self.patience:
                        if verbose:
                            print(f"\nEarly stopping at epoch {self.current_epoch}")
                        break

            # Save periodic checkpoint
            if self.checkpoint_dir and self.current_epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.current_epoch}.pth')

    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        if not self.checkpoint_dir:
            return

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'train_history': self.train_history,
            'val_history': self.val_history
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        if not self.checkpoint_dir:
            raise ValueError("checkpoint_dir not set")

        checkpoint_path = self.checkpoint_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']

        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Best metric: {self.best_metric:.4f}")
