#!/usr/bin/env python3
"""
Train BPR using our own data split (same as our model).
This is to verify if the data split is consistent.

Usage:
    python scripts/train_bpr_our_split.py --seed 42
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm

from src.data.dataset import split_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BPRModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, reg_weight=1e-5):
        super().__init__()
        self.user_embed = nn.Embedding(n_users + 1, embedding_dim)
        self.item_embed = nn.Embedding(n_items + 1, embedding_dim)
        self.reg_weight = reg_weight

        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

    def forward(self, users, pos_items, neg_items):
        user_e = self.user_embed(users)
        pos_e = self.item_embed(pos_items)
        neg_e = self.item_embed(neg_items)

        pos_scores = (user_e * pos_e).sum(dim=1)
        neg_scores = (user_e * neg_e).sum(dim=1)

        return pos_scores, neg_scores

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores, neg_scores = self.forward(users, pos_items, neg_items)

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        # Regularization
        reg_loss = self.reg_weight * (
            self.user_embed(users).norm(2).pow(2) +
            self.item_embed(pos_items).norm(2).pow(2) +
            self.item_embed(neg_items).norm(2).pow(2)
        ) / users.shape[0]

        return loss + reg_loss


class BPRDataset(Dataset):
    def __init__(self, interactions, n_items, user_pos_items):
        self.interactions = interactions
        self.n_items = n_items
        self.user_pos_items = user_pos_items

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user, item = self.interactions[idx]

        # Negative sampling
        while True:
            neg_item = random.randint(1, self.n_items)
            if neg_item not in self.user_pos_items[user]:
                break

        return user, item, neg_item


def evaluate_uni100(model, test_data, train_pos, val_pos, n_items, device, k=10):
    """Evaluate with uni100 mode (1 pos + 99 neg)."""
    model.eval()

    # Build all positive items per user
    user_pos_items = defaultdict(set)
    for u, items in train_pos.items():
        user_pos_items[u].update(items)
    for u, items in val_pos.items():
        user_pos_items[u].update(items)

    recalls = []
    ndcgs = []
    hits = []

    with torch.no_grad():
        for user, pos_item in tqdm(test_data, desc="Evaluating", leave=False):
            # Sample 99 negative items
            neg_items = []
            while len(neg_items) < 99:
                neg = random.randint(1, n_items)
                if neg not in user_pos_items[user] and neg != pos_item and neg not in neg_items:
                    neg_items.append(neg)

            # 1 pos + 99 neg
            candidates = [pos_item] + neg_items

            # Score
            user_t = torch.LongTensor([user]).to(device)
            items_t = torch.LongTensor(candidates).to(device)

            user_e = model.user_embed(user_t)
            items_e = model.item_embed(items_t)

            scores = (user_e * items_e).sum(dim=1).cpu().numpy()

            # Rank
            ranked = np.argsort(-scores)
            pos_rank = np.where(ranked == 0)[0][0]

            # Metrics
            hit = 1.0 if pos_rank < k else 0.0
            hits.append(hit)

            recall = 1.0 if pos_rank < k else 0.0
            recalls.append(recall)

            if pos_rank < k:
                ndcg = 1.0 / np.log2(pos_rank + 2)
            else:
                ndcg = 0.0
            ndcgs.append(ndcg)

    return {
        f'ndcg@{k}': np.mean(ndcgs),
        f'recall@{k}': np.mean(recalls),
        f'hit@{k}': np.mean(hits)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--early_stop', type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("BPR with Our Data Split")
    print("="*60)
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print()

    # Load and split data using OUR split function
    inter_path = 'data/recbole/amazon-beauty/amazon-beauty.inter'
    print("Splitting data using our split_data function...")
    train_df, val_df, test_df = split_data(
        inter_path,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        time_based=True,
        random_seed=args.seed,
        per_user_split=True
    )

    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    print()

    # Get n_users and n_items
    all_users = set(train_df['user_id:token'].unique())
    all_items = set(train_df['item_id:token'].unique())
    n_users = max(all_users)
    n_items = max(all_items)

    print(f"Users: {n_users}, Items: {n_items}")
    print()

    # Build user positive items
    train_pos = defaultdict(set)
    for _, row in train_df.iterrows():
        train_pos[row['user_id:token']].add(row['item_id:token'])

    val_pos = defaultdict(set)
    for _, row in val_df.iterrows():
        val_pos[row['user_id:token']].add(row['item_id:token'])

    # Create train dataset
    train_interactions = [(row['user_id:token'], row['item_id:token'])
                          for _, row in train_df.iterrows()]
    train_dataset = BPRDataset(train_interactions, n_items, train_pos)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Val and test data
    val_data = [(row['user_id:token'], row['item_id:token'])
                for _, row in val_df.iterrows()]
    test_data = [(row['user_id:token'], row['item_id:token'])
                 for _, row in test_df.iterrows()]

    # Model
    model = BPRModel(n_users, n_items, args.embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    best_val_ndcg = 0
    patience = 0
    best_epoch = 0

    print("Training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for users, pos_items, neg_items in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            optimizer.zero_grad()
            loss = model.bpr_loss(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validate
        val_metrics = evaluate_uni100(model, val_data, train_pos, {}, n_items, device)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val NDCG@10={val_metrics['ndcg@10']:.4f}, Val Recall@10={val_metrics['recall@10']:.4f}")

        if val_metrics['ndcg@10'] > best_val_ndcg:
            best_val_ndcg = val_metrics['ndcg@10']
            best_epoch = epoch + 1
            patience = 0
            torch.save(model.state_dict(), 'outputs/beauty/bpr_our_split_best.pth')
        else:
            patience += 1
            if patience >= args.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Test
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load('outputs/beauty/bpr_our_split_best.pth'))

    print("Testing...")
    test_metrics = evaluate_uni100(model, test_data, train_pos, val_pos, n_items, device)

    print()
    print("="*60)
    print("Test Results (BPR with Our Split)")
    print("="*60)
    print(f"  NDCG@10: {test_metrics['ndcg@10']:.4f}")
    print(f"  Recall@10: {test_metrics['recall@10']:.4f}")
    print(f"  Hit@10: {test_metrics['hit@10']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
