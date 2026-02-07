#!/usr/bin/env python3
"""
Run ML-1M multimodal baseline experiments with 5 trials.

Baselines: VBPR, MMGCN, MKGAT

Usage:
    python scripts/run_ml1m_multimodal_5trials.py --method vbpr
    python scripts/run_ml1m_multimodal_5trials.py --method mmgcn
    python scripts/run_ml1m_multimodal_5trials.py --method mkgat
    python scripts/run_ml1m_multimodal_5trials.py --method all
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "baselines"))

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

SEEDS = [42, 123, 456, 789, 2024]

# ML-1M paths
DATA_PATH = '/data/xuao/llm-knowledge-extraction-rec/data/recbole/ml-1m'
INTER_PATH = f'{DATA_PATH}/ml-1m.inter'
VISUAL_FEATURES_PATH = f'{DATA_PATH}/visual_features.npy'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BPRDataset(Dataset):
    def __init__(self, interactions, n_items, user_pos_items):
        self.interactions = interactions
        self.n_items = n_items
        self.user_pos_items = user_pos_items

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user, item = self.interactions[idx]
        while True:
            neg_item = random.randint(1, self.n_items)
            if neg_item not in self.user_pos_items[user]:
                break
        return user, item, neg_item


def evaluate_uni100(model, test_data, train_pos, val_pos, n_items, device, k=10, model_type='vbpr'):
    """Evaluate with uni100 mode (batched for speed)."""
    model.eval()

    user_pos_items = defaultdict(set)
    for u, items in train_pos.items():
        user_pos_items[int(u)].update(int(i) for i in items)
    for u, items in val_pos.items():
        user_pos_items[int(u)].update(int(i) for i in items)

    with torch.no_grad():
        if model_type == 'vbpr':
            # VBPR: compute embeddings manually
            all_item_embeddings = []
            batch_size = 512
            for i in range(0, n_items, batch_size):
                batch_items = list(range(i+1, min(i+batch_size+1, n_items+1)))
                if not batch_items:
                    continue
                batch_items_t = torch.LongTensor(batch_items).to(device)
                i_embed = model.item_embed(batch_items_t)
                visual_feat = model.visual_features[batch_items_t.cpu() - 1].to(device)
                visual_embed = model.visual_embed(visual_feat)
                item_emb = i_embed + visual_embed
                all_item_embeddings.append(item_emb.cpu())
            all_item_embeddings = torch.cat(all_item_embeddings, dim=0)
            all_user_embeddings = model.user_embed.weight[1:].cpu()
        else:
            # MMGCN/MKGAT: use get_all_embeddings
            all_user_embeddings, all_item_embeddings = model.get_all_embeddings(device)
            all_user_embeddings = all_user_embeddings.cpu()
            all_item_embeddings = all_item_embeddings.cpu()

    # Pre-sample negatives for all test samples (vectorized)
    print("Pre-sampling negatives...")
    all_items = set(range(1, n_items + 1))
    test_users = []
    test_pos_items = []
    test_candidates = []  # (n_test, 100) - pos at index 0

    for user, pos_item in test_data:
        excluded = user_pos_items[user] | {pos_item}
        available = list(all_items - excluded)
        if len(available) >= 99:
            neg_items = random.sample(available, 99)
        else:
            neg_items = available + [random.randint(1, n_items) for _ in range(99 - len(available))]
        test_users.append(user)
        test_pos_items.append(pos_item)
        test_candidates.append([pos_item] + neg_items)

    # Batch evaluation
    print("Batch evaluating...")
    batch_size = 16384  # A800 can handle large batches
    recalls = []
    ndcgs = []

    for i in tqdm(range(0, len(test_users), batch_size), desc="Evaluating", leave=False):
        batch_users = test_users[i:i+batch_size]
        batch_candidates = test_candidates[i:i+batch_size]

        # Get user embeddings: (batch, dim)
        user_embs = all_user_embeddings[[u - 1 for u in batch_users]]

        # Get candidate embeddings: (batch, 100, dim)
        candidate_embs = torch.stack([
            all_item_embeddings[[c - 1 for c in cands]]
            for cands in batch_candidates
        ])

        # Compute scores: (batch, 100)
        scores = torch.bmm(user_embs.unsqueeze(1), candidate_embs.transpose(1, 2)).squeeze(1)

        # Get rankings
        rankings = torch.argsort(scores, dim=1, descending=True)
        pos_ranks = (rankings == 0).nonzero(as_tuple=True)[1]  # Position of pos item (index 0)

        for rank in pos_ranks.numpy():
            recalls.append(1.0 if rank < k else 0.0)
            ndcgs.append(1.0 / np.log2(rank + 2) if rank < k else 0.0)

    return {'ndcg@10': np.mean(ndcgs), 'recall@10': np.mean(recalls)}


def run_vbpr_trial(seed, device):
    """Run single VBPR trial."""
    from baselines.vbpr_model import VBPR

    set_seed(seed)

    # Load data
    train_df, val_df, test_df = split_data(
        INTER_PATH, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
        time_based=True, random_seed=seed, per_user_split=True
    )

    n_users = train_df['user_id:token'].max()
    n_items = train_df['item_id:token'].max()

    # Load visual features
    visual_features = np.load(VISUAL_FEATURES_PATH)
    visual_dim = visual_features.shape[1]

    # Build datasets
    train_pos = defaultdict(set)
    for _, row in train_df.iterrows():
        train_pos[int(row['user_id:token'])].add(int(row['item_id:token']))

    val_pos = defaultdict(set)
    for _, row in val_df.iterrows():
        val_pos[int(row['user_id:token'])].add(int(row['item_id:token']))

    train_interactions = [(int(row['user_id:token']), int(row['item_id:token']))
                          for _, row in train_df.iterrows()]
    train_dataset = BPRDataset(train_interactions, n_items, train_pos)
    train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True)

    val_data = [(int(row['user_id:token']), int(row['item_id:token']))
                for _, row in val_df.iterrows()]
    test_data = [(int(row['user_id:token']), int(row['item_id:token']))
                 for _, row in test_df.iterrows()]

    # Model
    model = VBPR(n_users=n_users, n_items=n_items, embedding_dim=64,
                 visual_dim=visual_dim, reg_weight=1e-5).to(device)
    model.load_visual_features(visual_features)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    best_val_ndcg = 0
    patience = 0
    best_state = None

    for epoch in range(300):
        model.train()
        total_loss = 0
        for users, pos_items, neg_items in train_loader:
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            optimizer.zero_grad()
            loss, _, _ = model.bpr_loss(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate_uni100(model, val_data, train_pos, {}, n_items, device, model_type='vbpr')
        if val_metrics['ndcg@10'] > best_val_ndcg:
            best_val_ndcg = val_metrics['ndcg@10']
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                break

    # Test
    model.load_state_dict(best_state)
    test_metrics = evaluate_uni100(model, test_data, train_pos, val_pos, n_items, device, model_type='vbpr')
    return test_metrics


def run_mmgcn_trial(seed, device):
    """Run single MMGCN trial."""
    from baselines.mmgcn_model import MMGCN

    set_seed(seed)

    train_df, val_df, test_df = split_data(
        INTER_PATH, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
        time_based=True, random_seed=seed, per_user_split=True
    )

    n_users = train_df['user_id:token'].max()
    n_items = train_df['item_id:token'].max()

    visual_features = np.load(VISUAL_FEATURES_PATH)
    visual_dim = visual_features.shape[1]

    train_pos = defaultdict(set)
    for _, row in train_df.iterrows():
        train_pos[int(row['user_id:token'])].add(int(row['item_id:token']))

    val_pos = defaultdict(set)
    for _, row in val_df.iterrows():
        val_pos[int(row['user_id:token'])].add(int(row['item_id:token']))

    # train_interactions for BPRDataset (2-tuple)
    train_interactions = [(int(row['user_id:token']), int(row['item_id:token']))
                          for _, row in train_df.iterrows()]
    train_dataset = BPRDataset(train_interactions, n_items, train_pos)
    train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True)

    val_data = [(int(row['user_id:token']), int(row['item_id:token']))
                for _, row in val_df.iterrows()]
    test_data = [(int(row['user_id:token']), int(row['item_id:token']))
                 for _, row in test_df.iterrows()]

    # train_interactions for adjacency matrix (3-tuple: user, item, rating)
    train_interactions_3tuple = [(int(row['user_id:token']), int(row['item_id:token']), 1)
                                  for _, row in train_df.iterrows()]

    model = MMGCN(n_users=n_users, n_items=n_items, embedding_dim=64,
                  visual_dim=visual_dim, n_layers=2).to(device)
    model.load_visual_features(visual_features)
    model.adj_matrix = model.build_adjacency_matrix(train_interactions_3tuple, device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_ndcg = 0
    patience = 0
    best_state = None

    for epoch in range(300):
        model.train()
        for users, pos_items, neg_items in train_loader:
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            optimizer.zero_grad()
            loss, _, _ = model.bpr_loss(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_uni100(model, val_data, train_pos, {}, n_items, device, model_type='mmgcn')
        if val_metrics['ndcg@10'] > best_val_ndcg:
            best_val_ndcg = val_metrics['ndcg@10']
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate_uni100(model, test_data, train_pos, val_pos, n_items, device, model_type='mmgcn')
    return test_metrics


def run_mkgat_trial(seed, device):
    """Run single MKGAT trial."""
    from baselines.mkgat_model import MKGAT

    set_seed(seed)

    train_df, val_df, test_df = split_data(
        INTER_PATH, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
        time_based=True, random_seed=seed, per_user_split=True
    )

    n_users = train_df['user_id:token'].max()
    n_items = train_df['item_id:token'].max()

    visual_features = np.load(VISUAL_FEATURES_PATH)
    visual_dim = visual_features.shape[1]

    train_pos = defaultdict(set)
    for _, row in train_df.iterrows():
        train_pos[int(row['user_id:token'])].add(int(row['item_id:token']))

    val_pos = defaultdict(set)
    for _, row in val_df.iterrows():
        val_pos[int(row['user_id:token'])].add(int(row['item_id:token']))

    train_interactions = [(int(row['user_id:token']), int(row['item_id:token']))
                          for _, row in train_df.iterrows()]
    train_dataset = BPRDataset(train_interactions, n_items, train_pos)
    train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True)

    val_data = [(int(row['user_id:token']), int(row['item_id:token']))
                for _, row in val_df.iterrows()]
    test_data = [(int(row['user_id:token']), int(row['item_id:token']))
                 for _, row in test_df.iterrows()]

    # Build interaction matrix
    import scipy.sparse as sp
    rows, cols = [], []
    for u, items in train_pos.items():
        for i in items:
            rows.append(u - 1)
            cols.append(i - 1)
    interaction_matrix = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n_users, n_items)
    )

    # Load KG
    kg_path = f'{DATA_PATH}/ml-1m.item.kg'
    kg_triplets = []
    with open(kg_path, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                kg_triplets.append((int(parts[0]), parts[1], parts[2]))

    model = MKGAT(n_users=n_users, n_items=n_items, embedding_dim=64,
                  visual_dim=visual_dim, n_layers=2, n_heads=4).to(device)
    model.load_visual_features(visual_features)
    model.build_graph(interaction_matrix, kg_triplets, device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_ndcg = 0
    patience = 0
    best_state = None

    for epoch in range(300):
        model.train()
        for users, pos_items, neg_items in train_loader:
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            optimizer.zero_grad()
            loss, _, _ = model.bpr_loss(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_uni100(model, val_data, train_pos, {}, n_items, device, model_type='mkgat')
        if val_metrics['ndcg@10'] > best_val_ndcg:
            best_val_ndcg = val_metrics['ndcg@10']
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate_uni100(model, test_data, train_pos, val_pos, n_items, device, model_type='mkgat')
    return test_metrics


def run_5trials(method, device):
    """Run 5 trials for a method."""
    print("=" * 70)
    print(f"Running {method.upper()} - 5 Trials")
    print("=" * 70)

    trial_func = {
        'vbpr': run_vbpr_trial,
        'mmgcn': run_mmgcn_trial,
        'mkgat': run_mkgat_trial,
    }[method]

    results = []
    for i, seed in enumerate(SEEDS):
        print(f"\n--- Trial {i+1}/5 (seed={seed}) ---")
        try:
            metrics = trial_func(seed, device)
            results.append({'ndcg': metrics['ndcg@10'], 'recall': metrics['recall@10']})
            print(f"Trial {i+1}: NDCG@10={metrics['ndcg@10']:.4f}, Recall@10={metrics['recall@10']:.4f}")
        except Exception as e:
            print(f"Error in trial {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def print_summary(method, results):
    if not results:
        print(f"\n{method}: No results")
        return

    ndcgs = [r['ndcg'] for r in results]
    recalls = [r['recall'] for r in results]

    print(f"\n{'=' * 70}")
    print(f"{method} Summary (5 Trials)")
    print(f"{'=' * 70}")
    print(f"  NDCG@10:   {np.mean(ndcgs):.4f} ± {np.std(ndcgs):.4f}")
    print(f"  Recall@10: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'vbpr', 'mmgcn', 'mkgat'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    all_results = {}

    if args.method in ['vbpr', 'all']:
        results = run_5trials('vbpr', device)
        all_results['VBPR'] = results
        print_summary('VBPR', results)

    if args.method in ['mmgcn', 'all']:
        results = run_5trials('mmgcn', device)
        all_results['MMGCN'] = results
        print_summary('MMGCN', results)

    if args.method in ['mkgat', 'all']:
        results = run_5trials('mkgat', device)
        all_results['MKGAT'] = results
        print_summary('MKGAT', results)

    # Summary table
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("ML-1M Multimodal Baselines Summary")
        print("=" * 70)
        print("| Method | NDCG@10 | Recall@10 |")
        print("|--------|---------|-----------|")
        for method, results in all_results.items():
            if results:
                ndcgs = [r['ndcg'] for r in results]
                recalls = [r['recall'] for r in results]
                print(f"| {method} | {np.mean(ndcgs):.4f} ± {np.std(ndcgs):.4f} | {np.mean(recalls):.4f} ± {np.std(recalls):.4f} |")


if __name__ == '__main__':
    main()
