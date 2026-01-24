#!/usr/bin/env python3
"""
Train VBPR (Visual Bayesian Personalized Ranking) model.

Usage:
    python baselines/train_vbpr.py \
        --data_dir data/recbole/ml-1m \
        --visual_features data/recbole/ml-1m/visual_features.npy \
        --output_dir outputs/vbpr \
        --seed 42
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import random
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from vbpr_model import VBPR


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class InteractionDataset(Dataset):
    """User-item interaction dataset with BPR sampling."""

    def __init__(self, interactions, n_items, neg_sampling=True):
        """
        Args:
            interactions: List of (user, item, rating) tuples
            n_items: Total number of items
            neg_sampling: Whether to sample negative items
        """
        self.interactions = interactions
        self.n_items = n_items
        self.neg_sampling = neg_sampling

        # Build user positive items
        self.user_pos_items = defaultdict(set)
        for u, i, r in interactions:
            self.user_pos_items[u].add(i)

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user, item, rating = self.interactions[idx]

        if self.neg_sampling:
            # Sample negative item
            while True:
                neg_item = random.randint(1, self.n_items)
                if neg_item not in self.user_pos_items[user]:
                    break
            return user, item, neg_item
        else:
            return user, item, item


def load_interactions(inter_file):
    """Load user-item interactions."""
    interactions = []

    with open(inter_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            user = int(parts[0])
            item = int(parts[1])
            rating = float(parts[2])
            timestamp = float(parts[3])

            interactions.append((user, item, rating, timestamp))

    return interactions


def split_data_temporal(interactions, train_ratio=0.7, val_ratio=0.1):
    """
    Split data temporally per user (70/10/20).

    Each user's interactions are split independently by timestamp.
    """
    user_interactions = defaultdict(list)
    for u, i, r, t in interactions:
        user_interactions[u].append((u, i, r, t))

    train = []
    val = []
    test = []

    for user, user_inters in user_interactions.items():
        user_inters.sort(key=lambda x: x[3])

        n = len(user_inters)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train.extend([(u, i, r) for u, i, r, t in user_inters[:train_end]])
        val.extend([(u, i, r) for u, i, r, t in user_inters[train_end:val_end]])
        test.extend([(u, i, r) for u, i, r, t in user_inters[val_end:]])

    return train, val, test


def evaluate(model, dataloader, device, k=10, n_items=None, train_data=None, val_data=None):
    """
    Evaluate model with uni100 mode (1 pos + 99 neg).

    Args:
        model: VBPR model
        dataloader: Evaluation dataloader
        device: torch device
        k: Top-K
        n_items: Total number of items
        train_data: Training data for negative sampling
        val_data: Validation data (optional, for test evaluation)

    Returns:
        Dictionary of metrics
    """
    model.eval()

    # Collect positive interactions
    pos_interactions = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                users, items, _ = batch
            else:
                users, items = batch[0], batch[1]

            for u, i in zip(users.numpy(), items.numpy()):
                pos_interactions.append((u, i))

    # Build user positive items
    user_pos_items = defaultdict(set)

    if train_data:
        for u, i, r in train_data:
            user_pos_items[u].add(i)

    if val_data:
        for u, i, r in val_data:
            user_pos_items[u].add(i)

    for u, i in pos_interactions:
        user_pos_items[u].add(i)

    # Pre-compute all item embeddings for fast scoring
    print("  Pre-computing item embeddings...")
    all_item_embeddings = []

    with torch.no_grad():
        batch_size_precompute = 512
        dummy_user = torch.LongTensor([1]).to(device)

        for i in tqdm(range(0, n_items, batch_size_precompute), desc="  Items", leave=False):
            batch_items = list(range(i+1, min(i+batch_size_precompute+1, n_items+1)))
            if not batch_items:
                continue

            batch_items_t = torch.LongTensor(batch_items).to(device)
            dummy_users = dummy_user.repeat(len(batch_items))

            # Get item embeddings (CF + visual)
            i_embed = model.item_embed(batch_items_t)
            visual_feat = model.visual_features[batch_items_t.cpu() - 1].to(device)
            visual_embed = model.visual_embed(visual_feat)

            # Combine CF and visual
            item_emb = i_embed + visual_embed  # Additive combination
            all_item_embeddings.append(item_emb.cpu())

        all_item_embeddings = torch.cat(all_item_embeddings, dim=0)

        # Pre-compute user embeddings
        all_user_embeddings = []
        for user_id in range(1, model.n_users + 1):
            user_emb = model.user_embed(torch.LongTensor([user_id]).to(device))
            all_user_embeddings.append(user_emb.cpu())

        all_user_embeddings = torch.cat(all_user_embeddings, dim=0)

    # Evaluate
    recalls = []
    ndcgs = []
    precisions = []

    print("  Evaluating...")
    for user, pos_item in tqdm(pos_interactions, desc="  Scoring", leave=False):
        # Sample 99 negative items
        neg_items = []
        while len(neg_items) < 99:
            neg_item = random.randint(1, n_items)
            if neg_item not in user_pos_items[user] and neg_item not in neg_items:
                neg_items.append(neg_item)

        # Candidate items: 1 pos + 99 neg
        candidate_items = [pos_item] + neg_items

        # Get embeddings
        user_emb = all_user_embeddings[user - 1]
        item_embs = all_item_embeddings[[i - 1 for i in candidate_items]]

        # Compute scores
        scores = (user_emb @ item_embs.T).numpy()

        # Rank
        ranked_indices = np.argsort(-scores)
        pos_rank = np.where(ranked_indices == 0)[0][0]

        # Metrics
        recall = 1.0 if pos_rank < k else 0.0
        recalls.append(recall)

        precision = 1.0 / k if pos_rank < k else 0.0
        precisions.append(precision)

        if pos_rank < k:
            ndcg = 1.0 / np.log2(pos_rank + 2)
        else:
            ndcg = 0.0
        ndcgs.append(ndcg)

    return {
        f'recall@{k}': np.mean(recalls),
        f'ndcg@{k}': np.mean(ndcgs),
        f'precision@{k}': np.mean(precisions)
    }


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_bpr_loss = 0
    total_reg_loss = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        users, pos_items, neg_items = batch

        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        optimizer.zero_grad()

        # Compute BPR loss
        loss, bpr_loss, reg_loss = model.bpr_loss(users, pos_items, neg_items)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bpr_loss += bpr_loss.item()
        total_reg_loss += reg_loss.item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'bpr_loss': total_bpr_loss / n_batches,
        'reg_loss': total_reg_loss / n_batches
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--visual_features', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--reg_weight', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    data_dir = Path(args.data_dir)
    inter_file = data_dir / f"{data_dir.name}.inter"

    all_interactions = load_interactions(inter_file)
    train_data, val_data, test_data = split_data_temporal(all_interactions)

    # Get counts
    n_users = max(u for u, i, r in train_data)
    n_items = max(i for u, i, r in train_data)

    print(f"Users: {n_users}")
    print(f"Items: {n_items}")
    print(f"Train: {len(train_data)}")
    print(f"Val: {len(val_data)}")
    print(f"Test: {len(test_data)}")

    # Load visual features
    print("Loading visual features...")
    visual_features = np.load(args.visual_features)
    print(f"Visual features shape: {visual_features.shape}")

    # Create datasets
    train_dataset = InteractionDataset(train_data, n_items, neg_sampling=True)
    val_dataset = InteractionDataset(val_data, n_items, neg_sampling=False)
    test_dataset = InteractionDataset(test_data, n_items, neg_sampling=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = VBPR(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=args.embedding_dim,
        visual_dim=visual_features.shape[1],
        reg_weight=args.reg_weight
    ).to(device)

    model.load_visual_features(visual_features)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("\nTraining...")
    best_val_ndcg = 0
    patience = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        print(f"  Train Loss: {train_metrics['loss']:.4f} (BPR: {train_metrics['bpr_loss']:.4f}, Reg: {train_metrics['reg_loss']:.4f})")

        # Validate
        print("  Validating...")
        val_metrics = evaluate(model, val_loader, device, k=10, n_items=n_items,
                              train_data=train_data, val_data=None)
        print(f"  Val NDCG@10: {val_metrics['ndcg@10']:.4f}, Recall@10: {val_metrics['recall@10']:.4f}")

        # Early stopping
        if val_metrics['ndcg@10'] > best_val_ndcg:
            best_val_ndcg = val_metrics['ndcg@10']
            best_epoch = epoch + 1
            patience = 0

            # Save best model
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
        else:
            patience += 1
            if patience >= args.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model and test
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))

    print("Testing...")
    test_metrics = evaluate(model, test_loader, device, k=10, n_items=n_items,
                           train_data=train_data, val_data=val_data)

    print("\n" + "="*50)
    print("Test Results:")
    print(f"  NDCG@10: {test_metrics['ndcg@10']:.4f}")
    print(f"  Recall@10: {test_metrics['recall@10']:.4f}")
    print(f"  Precision@10: {test_metrics['precision@10']:.4f}")
    print("="*50)

    # Save results
    results = {
        'args': vars(args),
        'best_epoch': best_epoch,
        'test_metrics': test_metrics
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == '__main__':
    main()
