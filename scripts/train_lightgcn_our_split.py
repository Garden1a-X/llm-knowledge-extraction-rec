#!/usr/bin/env python3
"""
Train LightGCN using our own data split (same as our model).

Usage:
    python scripts/train_lightgcn_our_split.py --seed 42
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
import scipy.sparse as sp

from src.data.dataset import split_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3, reg_weight=1e-5):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.reg_weight = reg_weight

        self.user_embed = nn.Embedding(n_users + 1, embedding_dim)
        self.item_embed = nn.Embedding(n_items + 1, embedding_dim)

        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

        self.norm_adj = None

    def set_adj_matrix(self, adj_matrix):
        """Set normalized adjacency matrix for graph convolution."""
        self.norm_adj = adj_matrix

    def get_ego_embeddings(self):
        """Get initial embeddings."""
        user_embed = self.user_embed.weight
        item_embed = self.item_embed.weight
        return torch.cat([user_embed, item_embed], dim=0)

    def forward(self):
        """Graph convolution to get final embeddings."""
        all_embed = self.get_ego_embeddings()
        embeds_list = [all_embed]

        for _ in range(self.n_layers):
            all_embed = torch.sparse.mm(self.norm_adj, all_embed)
            embeds_list.append(all_embed)

        # Mean of all layers
        all_embed = torch.stack(embeds_list, dim=1).mean(dim=1)

        user_embed, item_embed = torch.split(all_embed, [self.n_users + 1, self.n_items + 1])
        return user_embed, item_embed

    def bpr_loss(self, users, pos_items, neg_items):
        user_embed, item_embed = self.forward()

        user_e = user_embed[users]
        pos_e = item_embed[pos_items]
        neg_e = item_embed[neg_items]

        pos_scores = (user_e * pos_e).sum(dim=1)
        neg_scores = (user_e * neg_e).sum(dim=1)

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        # Regularization on ego embeddings
        reg_loss = self.reg_weight * (
            self.user_embed(users).norm(2).pow(2) +
            self.item_embed(pos_items).norm(2).pow(2) +
            self.item_embed(neg_items).norm(2).pow(2)
        ) / users.shape[0]

        return loss + reg_loss

    def predict(self, users, items):
        user_embed, item_embed = self.forward()
        user_e = user_embed[users]
        items_e = item_embed[items]
        return (user_e * items_e).sum(dim=1)


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


def build_adj_matrix(n_users, n_items, train_interactions, device):
    """Build normalized adjacency matrix for LightGCN."""
    # Build user-item interaction matrix
    rows = []
    cols = []
    for user, item in train_interactions:
        rows.append(user)
        cols.append(item + n_users + 1)  # offset items
        rows.append(item + n_users + 1)
        cols.append(user)

    rows = np.array(rows)
    cols = np.array(cols)
    data = np.ones(len(rows))

    n_nodes = n_users + n_items + 2
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    # Normalize: D^{-1/2} A D^{-1/2}
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    norm_adj = norm_adj.tocoo()

    # Convert to torch sparse tensor
    indices = torch.LongTensor([norm_adj.row, norm_adj.col])
    values = torch.FloatTensor(norm_adj.data)
    shape = torch.Size(norm_adj.shape)

    return torch.sparse.FloatTensor(indices, values, shape).to(device)


def evaluate_uni100(model, test_data, train_pos, val_pos, n_items, device, k=10):
    """Evaluate with uni100 mode (1 pos + 99 neg)."""
    model.eval()

    # Build all positive items per user
    user_pos_items = defaultdict(set)
    for u, items in train_pos.items():
        user_pos_items[int(u)].update(int(i) for i in items)
    for u, items in val_pos.items():
        user_pos_items[int(u)].update(int(i) for i in items)

    recalls = []
    ndcgs = []
    hits = []

    with torch.no_grad():
        user_embed, item_embed = model.forward()

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
            user_e = user_embed[user]
            items_e = item_embed[candidates]

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
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--early_stop', type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("LightGCN with Our Data Split")
    print("="*60)
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Layers: {args.n_layers}")
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
    all_users = set(int(u) for u in train_df['user_id:token'].unique())
    all_items = set(int(i) for i in train_df['item_id:token'].unique())
    n_users = max(all_users)
    n_items = max(all_items)

    print(f"Users: {n_users}, Items: {n_items}")
    print()

    # Build user positive items
    train_pos = defaultdict(set)
    for _, row in train_df.iterrows():
        train_pos[int(row['user_id:token'])].add(int(row['item_id:token']))

    val_pos = defaultdict(set)
    for _, row in val_df.iterrows():
        val_pos[int(row['user_id:token'])].add(int(row['item_id:token']))

    # Create train dataset
    train_interactions = [(int(row['user_id:token']), int(row['item_id:token']))
                          for _, row in train_df.iterrows()]
    train_dataset = BPRDataset(train_interactions, n_items, train_pos)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Val and test data
    val_data = [(int(row['user_id:token']), int(row['item_id:token']))
                for _, row in val_df.iterrows()]
    test_data = [(int(row['user_id:token']), int(row['item_id:token']))
                 for _, row in test_df.iterrows()]

    # Model
    print("Building adjacency matrix...")
    model = LightGCN(n_users, n_items, args.embedding_dim, args.n_layers).to(device)
    adj_matrix = build_adj_matrix(n_users, n_items, train_interactions, device)
    model.set_adj_matrix(adj_matrix)
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
            torch.save(model.state_dict(), 'outputs/beauty/lightgcn_our_split_best.pth')
        else:
            patience += 1
            if patience >= args.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Test
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load('outputs/beauty/lightgcn_our_split_best.pth'))

    print("Testing...")
    test_metrics = evaluate_uni100(model, test_data, train_pos, val_pos, n_items, device)

    print()
    print("="*60)
    print("Test Results (LightGCN with Our Split)")
    print("="*60)
    print(f"  NDCG@10: {test_metrics['ndcg@10']:.4f}")
    print(f"  Recall@10: {test_metrics['recall@10']:.4f}")
    print(f"  Hit@10: {test_metrics['hit@10']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
